"""Minimal DDP smoke worker activated entirely through torchrun flags.

This module deliberately does not import ``lm_resiliency``. The
``lm_resiliency`` rendezvous backend installs import monitoring before this
module starts and infers the native PyTorch adapter from this module's imports.
This is a bootstrap validation fixture, not a production-loop example.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=2)
    args = parser.parse_args()
    if args.steps < 1:
        raise ValueError("steps must be positive")

    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    dist.init_process_group("gloo")
    rank = dist.get_rank()

    torch.manual_seed(123)
    model = DistributedDataParallel(torch.nn.Linear(4, 2))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    losses: list[float] = []
    for step in range(args.steps):
        inputs = torch.full((2, 4), float(rank + step + 1))
        optimizer.zero_grad(set_to_none=True)
        loss = model(inputs).square().mean()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    artifact = args.artifact_dir / f"rank-{rank}.json"
    artifact.write_text(
        json.dumps(
            {
                "local_rank": int(os.environ["LOCAL_RANK"]),
                "resiliency_adapter_attached": (
                    os.environ.get("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED") == "1"
                ),
                "losses": losses,
                "rank": rank,
                "world_size": dist.get_world_size(),
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
