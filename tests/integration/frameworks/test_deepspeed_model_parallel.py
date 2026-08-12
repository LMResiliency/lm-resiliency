"""Validate SCOUT through real DeepSpeed model-parallel implementations."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency import ReplayHarnessConfig, ReplayWorkload, enable_resiliency

_WORLD_SIZE = 16
_FAULT_RANK = 15


class FaultBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.inject_replay_sdc = False

    def _inject(self, outputs: torch.Tensor) -> torch.Tensor:
        if self.inject_replay_sdc:
            outputs = outputs.clone()
            outputs.flatten()[-1] += 1
        return outputs


class AutoTPBlock(FaultBlock):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(32, 64, bias=False)
        self.down = nn.Linear(64, 32, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self._inject(inputs + self.down(torch.nn.functional.gelu(self.up(inputs))))


class LocalAttention(nn.Module):
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        return (query + key + value) / 3


class UlyssesBlock(FaultBlock):
    def __init__(self) -> None:
        super().__init__()
        self.qkv = nn.Linear(32, 3 * 32, bias=False)
        self.output = nn.Linear(32, 32, bias=False)
        self.attention: nn.Module | None = None

    def set_sequence_group(self, group: dist.ProcessGroup) -> None:
        from deepspeed.sequence.layer import DistributedAttention

        self.attention = DistributedAttention(
            LocalAttention(),
            group,
            scatter_idx=2,
            gather_idx=0,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.attention is None:
            raise RuntimeError("Ulysses sequence group was not configured")
        query, key, value = self.qkv(inputs).chunk(3, dim=-1)
        shape = (*query.shape[:-1], 4, 8)
        context = self.attention(
            query.reshape(shape),
            key.reshape(shape),
            value.reshape(shape),
            0,
        )
        outputs = inputs + self.output(context.flatten(-2))
        return self._inject(outputs)


class Expert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.up = nn.Linear(32, 64, bias=False)
        self.down = nn.Linear(64, 32, bias=False)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.down(torch.nn.functional.gelu(self.up(inputs)))


class MoEBlock(FaultBlock):
    def __init__(self) -> None:
        super().__init__()
        from deepspeed.moe.layer import MoE

        self.moe = MoE(
            hidden_size=32,
            expert=Expert(),
            num_experts=4,
            ep_size=2,
            k=1,
            min_capacity=1,
            drop_tokens=False,
            use_rts=False,
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs, _, _ = self.moe(inputs)
        return self._inject(outputs)


class Model(nn.Module):
    def __init__(self, block: FaultBlock) -> None:
        super().__init__()
        self.layers = nn.ModuleList([block])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def _config(case: str) -> dict[str, Any]:
    config: dict[str, Any] = {
        "train_micro_batch_size_per_gpu": 2,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "bf16": {"enabled": True},
        "steps_per_print": 1_000,
    }
    if case == "autotp":
        config["train_batch_size"] = 2 * _WORLD_SIZE
        config["tensor_parallel"] = {
            "autotp_size": 2,
            "partition_config": {
                "use_default_specs": False,
                "layer_specs": [
                    {
                        "patterns": [r".*\.up\.weight$"],
                        "partition_type": "column",
                    },
                    {
                        "patterns": [r".*\.down\.weight$"],
                        "partition_type": "row",
                    },
                ],
            },
        }
    elif case == "ulysses":
        config["train_batch_size"] = 2 * (_WORLD_SIZE // 2)
        config["data_parallel_size"] = _WORLD_SIZE // 2
        config["sequence_parallel_size"] = 2
    else:
        config["train_batch_size"] = 2 * _WORLD_SIZE
    return config


def _build(case: str) -> tuple[Model, torch.optim.Optimizer | None]:
    if case == "autotp":
        block: FaultBlock = AutoTPBlock()
    elif case == "ulysses":
        block = UlyssesBlock()
    else:
        block = MoEBlock()
    model = Model(block)
    if case != "moe":
        return model, None

    from deepspeed.moe.utils import (
        split_params_into_different_moe_groups_for_optimizer,
    )

    param_groups = split_params_into_different_moe_groups_for_optimizer(
        {"params": list(model.parameters()), "name": "parameters"}
    )
    return model, torch.optim.AdamW(param_groups, lr=1e-4)


def _input(case: str, device: torch.device) -> torch.Tensor:
    values = torch.linspace(-1.0, 1.0, 8 * 32, device=device, dtype=torch.bfloat16)
    return values.reshape(2, 4, 32)


def _step(engine, inputs: torch.Tensor, *, inject_fault: bool) -> None:
    loss = engine(inputs).float().square().mean()
    engine.backward(loss)
    target = engine.module.layers[0]
    if inject_fault and dist.get_rank() == _FAULT_RANK:
        target.inject_replay_sdc = True
    engine.step()
    target.inject_replay_sdc = False


def _assert_optimizer_replay_healthy(resiliency, case: str) -> None:
    failed = [
        type(replay._optimizer).__name__
        for replay in resiliency._optimizer_replays
        if replay._warning_emitted
    ]
    if failed:
        raise AssertionError(f"{case}: optimizer-transition replay failed for {failed}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("autotp", "ulysses", "moe"), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if int(os.environ["WORLD_SIZE"]) != _WORLD_SIZE:
        raise RuntimeError(f"this validation requires {_WORLD_SIZE} ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    torch.manual_seed(20260809)
    model, optimizer = _build(args.case)
    initialize_kwargs: dict[str, Any] = {
        "model": model,
        "model_parameters": model.parameters(),
        "config": _config(args.case),
    }
    if optimizer is not None:
        initialize_kwargs["optimizer"] = optimizer
    else:
        initialize_kwargs["config"]["optimizer"] = {
            "type": "AdamW",
            "params": {"lr": 1e-4},
        }
    engine, _, _, _ = deepspeed.initialize(**initialize_kwargs)

    if args.case == "ulysses":
        engine.module.layers[0].set_sequence_group(engine.seq_parallel_group)
    target = engine.module.layers[0]
    workload = (
        ReplayWorkload(replay_modules=(target,), peer_role="expert")
        if args.case == "moe"
        else ReplayWorkload.dense([target])
    )
    resiliency = enable_resiliency(
        engine,
        interval=1,
        enable_checkpoint=False,
        replay=ReplayHarnessConfig(
            check_interval=1,
            workload=workload,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
            compare_parameter_state=False,
            straggler_min_slowdown_ratio=100.0,
            straggler_min_slowdown_ms=10_000.0,
        ),
    )

    try:
        harness = resiliency._replay_harness
        if harness is None:
            raise AssertionError("DeepSpeed public API did not create a replay harness")
        inputs = _input(args.case, engine.device)

        _step(engine, inputs, inject_fault=False)
        _assert_optimizer_replay_healthy(resiliency, args.case)
        if harness.last_result is None or any(harness.last_result.sdc_bitmap):
            raise AssertionError(f"{args.case}: healthy replay was not clean")

        _step(engine, inputs, inject_fault=True)
        _assert_optimizer_replay_healthy(resiliency, args.case)
        fault_result = harness.last_result
        expected = [int(rank == _FAULT_RANK) for rank in fault_result.peer_ranks]
        if fault_result.sdc_bitmap != expected:
            raise AssertionError(
                f"{args.case}: localized {fault_result.sdc_bitmap} for "
                f"{fault_result.peer_ranks}, expected {expected}"
            )

        _step(engine, inputs, inject_fault=False)
        _assert_optimizer_replay_healthy(resiliency, args.case)
        if harness.last_result is None or any(harness.last_result.sdc_bitmap):
            raise AssertionError(f"{args.case}: post-fault replay was not clean")

        localized = [
            rank
            for rank, failed in zip(
                fault_result.peer_ranks,
                fault_result.sdc_bitmap,
                strict=True,
            )
            if failed
        ]
        local = {
            "rank": dist.get_rank(),
            "peer_ranks": fault_result.peer_ranks,
            "localized": localized,
        }
        observations: list[dict[str, Any] | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(observations, local)
        fault_observers = [
            item["rank"]
            for item in observations
            if item is not None and item["localized"] == [_FAULT_RANK]
        ]
        expected_observers = next(
            len(item["peer_ranks"])
            for item in observations
            if item is not None and _FAULT_RANK in item["peer_ranks"]
        )
        if len(fault_observers) != expected_observers:
            raise AssertionError(
                f"{args.case}: fault observers {fault_observers}, expected {expected_observers}"
            )

        metadata: dict[str, Any] = {}
        if args.case == "autotp":
            metadata = {
                "autotp_size": engine.autotp_size(),
                "tensor_parallel_group_size": dist.get_world_size(
                    engine.mpu.get_tensor_model_parallel_group()
                ),
            }
        elif args.case == "ulysses":
            metadata = {
                "sequence_parallel_size": engine.sequence_parallel_size,
                "ulysses_module": "deepspeed.sequence.layer.DistributedAttention",
            }
        else:
            metadata = {
                "expert_parallel_size": 2,
                "expert_data_parallel_groups": sorted(engine.expert_data_parallel_group),
                "moe_all_to_all": True,
            }

        if dist.get_rank() == 0:
            result = {
                "case": f"deepspeed-{args.case}",
                "framework": "DeepSpeed",
                "world_size": _WORLD_SIZE,
                "peer_group_size": len(fault_result.peer_ranks),
                "fault_rank": _FAULT_RANK,
                "fault_observers": fault_observers,
                "exact_fault_localization": True,
                "clean_control": True,
                "clean_post_fault_replay": True,
                "optimizer_transition_replay": "passed",
                **metadata,
            }
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            output = args.artifact_dir / f"deepspeed-{args.case}.json"
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"PASS deepspeed-{args.case}: {json.dumps(result)}", flush=True)
    finally:
        resiliency.close()
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
