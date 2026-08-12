"""Distributed source-broadcast optimizer replay integration test.

Run:
    torchrun --standalone --nproc_per_node=4 \
        tests/integration/core/test_optimizer_replay_broadcast.py
"""

from __future__ import annotations

import os
from types import MethodType

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.detection.layer_replay import LayerReplayDetector
from lm_resiliency.detection.optimizer_step import (
    OPTIMIZER_REPLAY_INPUT,
    OPTIMIZER_UPDATED_WEIGHT,
    OptimizerStepReplay,
    collect_optimizer_replays,
)


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size >= 3

    parameter = nn.Parameter(
        torch.linspace(
            rank * 10.0,
            rank * 10.0 + 1.0,
            32,
            device=device,
        )
    )
    optimizer = torch.optim.AdamW([parameter], lr=0.01, weight_decay=0.1)
    parameter.grad = torch.linspace(rank + 0.1, rank + 0.4, 32, device=device)
    optimizer.step()

    replay = OptimizerStepReplay(optimizer, slice_numel=16)
    parameter.grad = torch.linspace(rank + 0.5, rank + 0.9, 32, device=device)
    replay.arm()
    optimizer.step()
    batch = collect_optimizer_replays([replay])
    assert batch is not None

    detector = LayerReplayDetector(
        group=dist.group.WORLD,
        device=device,
        deterministic=False,
    )
    clean = detector.replay_optimizer_batch(batch)
    assert clean[OPTIMIZER_REPLAY_INPUT].status is C3Status.AGREE, clean
    assert clean[OPTIMIZER_UPDATED_WEIGHT].status is C3Status.AGREE, clean

    victim = 2
    recipe = batch.recipes[0]
    original_replay = recipe.replay
    if rank == victim:

        def corrupted_replay(self, payload):
            del self
            output = original_replay(payload)
            output[0].add_(1.0)
            return output

        recipe.replay = MethodType(corrupted_replay, recipe)

    faulty = detector.replay_optimizer_batch(batch)
    expected = [0] * world_size
    expected[victim] = 1
    assert faulty[OPTIMIZER_UPDATED_WEIGHT].bitmap == expected, faulty

    replay.remove()
    dist.barrier()
    if rank == 0:
        print(
            "SCOUT OPTIMIZER REPLAY OK: different local transitions converged "
            "under one CUDA source recipe and rank 2 was localized."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
