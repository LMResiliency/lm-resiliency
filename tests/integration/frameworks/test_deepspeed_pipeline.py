"""Validate SCOUT through a real two-stage DeepSpeed PipelineEngine.

Run on the two 8-GPU hosts:

    torchrun --nnodes=2 --nproc-per-node=8 \
      --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR \
      tests/integration/frameworks/test_deepspeed_pipeline.py
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn
from deepspeed.pipe import PipelineModule

from lm_resiliency.detection.replay_harness import ReplayHarnessConfig
from lm_resiliency.integrations.deepspeed import enable_resiliency


class ReplayBlock(nn.Module):
    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.inject_replay_sdc = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.activation(self.linear(inputs))
        if self.inject_replay_sdc:
            outputs = outputs.clone()
            outputs.flatten()[-1] += 1
        return outputs


def _batches(hidden_size: int):
    inputs = torch.linspace(
        -1.0,
        1.0,
        2 * hidden_size,
        dtype=torch.bfloat16,
    ).reshape(2, hidden_size)
    labels = torch.zeros_like(inputs)
    while True:
        yield inputs, labels


def main() -> None:
    if int(os.environ["WORLD_SIZE"]) != 16:
        raise RuntimeError("DeepSpeed pipeline validation requires exactly 16 ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")

    hidden_size = 64
    pipeline_stages = 2
    data_parallel_size = dist.get_world_size() // pipeline_stages
    model = PipelineModule(
        layers=[ReplayBlock(hidden_size) for _ in range(4)],
        num_stages=pipeline_stages,
        partition_method="uniform",
        loss_fn=nn.MSELoss(),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 2 * data_parallel_size,
            "zero_optimization": {"stage": 0},
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
        iterator = _batches(hidden_size)
        harness = resiliency._replay_harness
        if harness is None:
            raise AssertionError("pipeline engine did not create a SCOUT replay harness")

        engine.train_batch(data_iter=iterator)
        target = harness.target_layer

        def arm_replay_fault(module, grad_input, grad_output):
            del grad_input, grad_output
            if dist.get_rank() == dist.get_world_size() - 1:
                module.inject_replay_sdc = True

        hook = target.register_full_backward_hook(arm_replay_fault)
        engine.train_batch(data_iter=iterator)
        hook.remove()
        fault_result = harness.last_result
        target.inject_replay_sdc = False
        engine.train_batch(data_iter=iterator)

        if resiliency._step_attribute != "_exec_optimizer_step":
            raise AssertionError("SCOUT did not hook PipelineEngine's optimizer boundary")
        if resiliency.step_count != 3:
            raise AssertionError(f"expected three pipeline steps, got {resiliency.step_count}")
        if fault_result is None or harness.last_result is None:
            raise AssertionError("pipeline training did not produce a SCOUT replay result")
        expected_sdc = [int(rank == dist.get_world_size() - 1) for rank in fault_result.peer_ranks]
        if fault_result.sdc_bitmap != expected_sdc:
            raise AssertionError(
                f"pipeline replay localized {fault_result.sdc_bitmap}, "
                f"expected {expected_sdc} for peers {fault_result.peer_ranks}"
            )
        if any(harness.last_result.sdc_bitmap):
            raise AssertionError(
                f"post-fault pipeline replay reported SDC: {harness.last_result.sdc_bitmap}"
            )

        peer_ranks = harness.last_result.peer_ranks
        if len(peer_ranks) != data_parallel_size:
            raise AssertionError(f"expected {data_parallel_size} DP peers, got {peer_ranks}")
        localized = [
            rank
            for rank, failed in zip(
                fault_result.peer_ranks,
                fault_result.sdc_bitmap,
                strict=True,
            )
            if failed
        ]
        gathered_localizations: list[list[int] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_localizations, localized)
        fault_observers = sum(
            localization == [dist.get_world_size() - 1] for localization in gathered_localizations
        )
        if fault_observers != data_parallel_size:
            raise AssertionError(
                f"expected {data_parallel_size} pipeline peers to localize rank 15, "
                f"got {fault_observers}"
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
                "case": "deepspeed-pipeline-engine",
                "framework": "DeepSpeed",
                "world_size": dist.get_world_size(),
                "pipeline_stages": pipeline_stages,
                "data_parallel_size": data_parallel_size,
                "peer_group_size": len(peer_ranks),
                "optimizer_boundary_hook": resiliency._step_attribute,
                "steps": resiliency.step_count,
                "injected_sdc_rank": dist.get_world_size() - 1,
                "fault_observers": fault_observers,
                "exact_fault_localization": True,
                "clean_replay": True,
            }
            (artifact_dir / "deepspeed-pipeline-engine.json").write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n"
            )
            print(f"PASS deepspeed-pipeline-engine: {json.dumps(result)}", flush=True)
    finally:
        resiliency.close()
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
