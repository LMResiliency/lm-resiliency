"""Validate SCOUT layer replay with a real DeepSpeed ZeRO-3 engine.

Run on the two 8-GPU hosts:

    torchrun --nnodes=2 --nproc-per-node=8 \
      --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR \
      tests/integration/frameworks/test_deepspeed_zero3.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.replay_harness import ReplayHarnessConfig
from lm_resiliency.integrations.deepspeed import enable_resiliency


class ReplayBlock(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.inject_replay_sdc = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = inputs + self.activation(self.linear(inputs))
        if self.inject_replay_sdc:
            outputs = outputs.clone()
            outputs.flatten()[-1] += 1
        return outputs


class TinyTransformer(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.layers = nn.ModuleList([ReplayBlock(hidden_size) for _ in range(2)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


def main() -> None:
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise RuntimeError("DeepSpeed ZeRO-3 validation requires exactly 16 ranks")
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
            "train_batch_size": 2 * dist.get_world_size(),
            "zero_optimization": {
                "stage": 3,
                "stage3_param_persistence_threshold": 0,
            },
            "bf16": {"enabled": True},
            "steps_per_print": 1_000,
        },
    )
    resiliency = enable_resiliency(
        engine,
        interval=1,
        enable_checkpoint=False,
        detection_config=ReplayHarnessConfig(
            check_interval=1,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
            straggler_min_slowdown_ratio=100.0,
            straggler_min_slowdown_ms=10_000.0,
        ),
    )

    try:
        harness = resiliency._replay_harness
        if harness is None:
            raise AssertionError("ZeRO-3 engine did not create a SCOUT replay harness")
        inputs = torch.linspace(
            -1.0,
            1.0,
            2 * 64,
            device=engine.device,
            dtype=torch.bfloat16,
        ).reshape(2, 64)
        expected_sdc = [0] * dist.get_world_size()
        expected_sdc[-1] = 1
        detected_sdc = None
        for step in range(3):
            loss = engine(inputs).float().square().mean()
            engine.backward(loss)
            if step == 1 and dist.get_rank() == dist.get_world_size() - 1:
                engine.module.layers[0].inject_replay_sdc = True
            engine.step()
            if step == 1:
                detected_sdc = list(harness.last_result.sdc_bitmap)
                engine.module.layers[0].inject_replay_sdc = False

        if resiliency.step_count != 3:
            raise AssertionError(f"expected three ZeRO-3 steps, got {resiliency.step_count}")
        if harness is None or harness.last_result is None:
            raise AssertionError("ZeRO-3 training did not produce a SCOUT replay result")
        if detected_sdc != expected_sdc:
            raise AssertionError(f"ZeRO-3 replay localized {detected_sdc}, expected {expected_sdc}")
        if any(harness.last_result.sdc_bitmap):
            raise AssertionError(
                f"post-fault ZeRO-3 replay reported SDC: {harness.last_result.sdc_bitmap}"
            )
        if len(harness.last_result.peer_ranks) != dist.get_world_size():
            raise AssertionError(f"unexpected ZeRO-3 peers: {harness.last_result.peer_ranks}")
        for parameter in harness.target_layer.parameters():
            if parameter.ds_active_sub_modules:
                raise AssertionError(
                    f"ZeRO-3 replay left active module claims: {parameter.ds_active_sub_modules}"
                )
            if parameter.ds_status.name != "NOT_AVAILABLE":
                raise AssertionError(
                    f"ZeRO-3 replay left parameter status {parameter.ds_status.name}"
                )

        if dist.get_rank() == 0:
            artifact_dir = Path(
                os.environ.get(
                    "SCOUT_VALIDATION_ARTIFACT_DIR",
                    "/tmp/scout-all-parallelisms",
                )
            )
            artifact_dir.mkdir(parents=True, exist_ok=True)
            result = {
                "case": "deepspeed-zero3",
                "framework": "DeepSpeed",
                "world_size": dist.get_world_size(),
                "peer_group_size": len(harness.last_result.peer_ranks),
                "zero_stage": 3,
                "steps": resiliency.step_count,
                "deepspeed_hook_parameter_materialization": True,
                "parameters_repartitioned": True,
                "injected_sdc_rank": dist.get_world_size() - 1,
                "localized_sdc_bitmap": detected_sdc,
                "exact_fault_localization": True,
                "clean_replay": True,
            }
            (artifact_dir / "deepspeed-zero3.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(f"PASS deepspeed-zero3: {json.dumps(result)}", flush=True)
    finally:
        resiliency.close()
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
