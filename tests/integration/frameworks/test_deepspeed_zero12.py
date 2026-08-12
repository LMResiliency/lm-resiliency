"""Validate combined GEMINI and SCOUT through real DeepSpeed ZeRO-1/2 engines."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency import (
    InMemoryCkptConfig,
    OrchestrationHooks,
    ReplayHarnessConfig,
    enable_resiliency,
)
from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.integrations.deepspeed.adapter import DeepSpeedAdapter

_WORLD_SIZE = 16
_FAULT_RANK = 15


class ReplayBlock(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.inject_replay_sdc = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + torch.nn.functional.gelu(self.linear(inputs))
        if self.inject_replay_sdc:
            outputs = outputs.clone()
            outputs.flatten()[-1] += 1
        return outputs


class TinyTransformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ReplayBlock(), ReplayBlock()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def _snapshot(adapter: DeepSpeedAdapter) -> list[torch.Tensor]:
    return [tensor.detach().cpu().clone() for tensor in adapter.collect_checkpoint_tensors()]


def _matches(left: list[torch.Tensor], right: list[torch.Tensor]) -> bool:
    return len(left) == len(right) and all(
        a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b)
        for a, b in zip(left, right, strict=True)
    )


def _step(engine, inputs: torch.Tensor, *, inject_fault: bool) -> None:
    loss = engine(inputs).float().square().mean()
    engine.backward(loss)
    if inject_fault and dist.get_rank() == _FAULT_RANK:
        engine.module.layers[0].inject_replay_sdc = True
    engine.step()
    engine.module.layers[0].inject_replay_sdc = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-stage", type=int, choices=(1, 2), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if int(os.environ["WORLD_SIZE"]) != _WORLD_SIZE:
        raise RuntimeError(f"this validation requires {_WORLD_SIZE} ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 2 * _WORLD_SIZE,
            "zero_optimization": {"stage": args.zero_stage},
            "bf16": {"enabled": True},
            "steps_per_print": 1_000,
        },
    )
    recovery_decisions = []
    resiliency = enable_resiliency(
        engine,
        interval=1,
        checkpoint=InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(args.artifact_dir / f"gemini-zero{args.zero_stage}"),
            skip_replication_if_hsdp=True,
        ),
        replay=ReplayHarnessConfig(
            check_interval=1,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
            straggler_min_slowdown_ratio=100.0,
            straggler_min_slowdown_ms=10_000.0,
        ),
        orchestration=OrchestrationHooks(
            report_recovery=recovery_decisions.append,
        ),
    )
    adapter = DeepSpeedAdapter(engine)

    try:
        inputs = torch.linspace(
            -1.0,
            1.0,
            2 * 64,
            device=engine.device,
            dtype=torch.bfloat16,
        ).reshape(2, 64)
        harness = resiliency._replay_harness
        if harness is None:
            raise AssertionError("DeepSpeed public API did not create a replay harness")

        _step(engine, inputs, inject_fault=False)
        resiliency._ckpt_manager.maybe_wait()
        first_result = harness.last_result
        if first_result is None or any(first_result.sdc_bitmap):
            raise AssertionError("healthy ZeRO replay was not clean")
        optimizer_input = first_result.c3_results.get("optimizer.optimizer_replay_input")
        if (
            "optimizer" not in first_result.checked_recipe_ids
            or optimizer_input is None
            or optimizer_input.status is not C3Status.AGREE
        ):
            raise AssertionError("ZeRO optimizer source-broadcast replay was not verified")
        if resiliency._ckpt_manager.checkpoint_status.recovery_verified_step != 1:
            raise AssertionError("dense checkpoint 1 was not verified immediately")

        _step(engine, inputs, inject_fault=False)
        resiliency._ckpt_manager.maybe_wait()
        if harness.last_result is None or any(harness.last_result.sdc_bitmap):
            raise AssertionError("second healthy ZeRO replay was not clean")
        step_two_state = _snapshot(adapter)
        if resiliency._ckpt_manager.checkpoint_status.recovery_verified_step != 2:
            raise AssertionError("dense checkpoint 2 was not verified immediately")

        _step(engine, inputs, inject_fault=True)
        resiliency._ckpt_manager.maybe_wait()
        expected = [int(rank == _FAULT_RANK) for rank in range(_WORLD_SIZE)]
        fault_bitmap = list(harness.last_result.sdc_bitmap)
        if fault_bitmap != expected:
            raise AssertionError(
                f"ZeRO-{args.zero_stage} localized {fault_bitmap}, expected {expected}"
            )
        if (
            len(recovery_decisions) != 1
            or recovery_decisions[0]["failure_kind"] != "sdc"
            or recovery_decisions[0]["recovery_mode"] != "recovery_verified"
            or recovery_decisions[0]["checkpoint_step"] != 2
            or not recovery_decisions[0]["available"]
        ):
            raise AssertionError(
                f"ZeRO-{args.zero_stage} recovery handoff was invalid: {recovery_decisions}"
            )

        recovered_step = resiliency.try_recover()
        recovered_state = _snapshot(adapter)
        if recovered_step != 2:
            raise AssertionError(f"expected recovery from step 2, got {recovered_step}")
        if not _matches(step_two_state, recovered_state):
            raise AssertionError("DeepSpeed recovery did not restore step-2 tensor state")

        _step(engine, inputs, inject_fault=False)
        resiliency._ckpt_manager.maybe_wait()
        if harness.last_result is None or any(harness.last_result.sdc_bitmap):
            raise AssertionError("post-recovery ZeRO replay was not clean")
        if resiliency.step_count != 3:
            raise AssertionError(
                f"expected recovered step 2 plus one clean step, got {resiliency.step_count}"
            )

        if dist.get_rank() == 0:
            result = {
                "case": f"deepspeed-zero{args.zero_stage}",
                "framework": "DeepSpeed",
                "world_size": _WORLD_SIZE,
                "zero_stage": args.zero_stage,
                "fault_rank": _FAULT_RANK,
                "localized_sdc_bitmap": fault_bitmap,
                "optimizer_replay_input": "agree",
                "recovery_decision": "recovery_verified step 2",
                "contaminated_checkpoint": "not captured",
                "recovered_step": recovered_step,
                "recovered_tensor_state": "bitwise exact",
                "clean_post_recovery_step": True,
            }
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            output = args.artifact_dir / f"deepspeed-zero{args.zero_stage}.json"
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(
                f"PASS deepspeed-zero{args.zero_stage}: {json.dumps(result)}",
                flush=True,
            )
    finally:
        resiliency.close()
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
