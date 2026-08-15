"""Two-rank regression for asymmetric scheduled replay capture.

Run:
    torchrun --standalone --nproc-per-node=2 \
        tests/integration/core/test_replay_readiness.py
"""

from __future__ import annotations

from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.detection.replay_harness import ModelReplayHarness, ReplayHarnessConfig


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.linear(value)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([_Block(), _Block()])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


def main() -> None:
    dist.init_process_group("gloo")
    group = dist.new_group(backend="gloo")
    torch.manual_seed(7)
    model = _Model()
    # OOB uses POSIX shared memory and is orthogonal to this replay-only test.
    with patch.object(ModelReplayHarness, "_start_oob_hang_detection"):
        harness = ModelReplayHarness(
            model,
            group=group,
            nccl_group=group,
            device=torch.device("cpu"),
            config=ReplayHarnessConfig(
                check_interval=1,
                embedding_check_interval=0,
                hidden_check_interval=1,
                output_check_interval=0,
                optimizer_check_interval=0,
                rotate_layers=False,
                all_to_all_policy=None,
            ),
            layers=model.layers,
        )

    if dist.get_rank() == 0:
        model(torch.ones(2, 4))

    result = harness.step()
    readiness = result.c3_results["replay_readiness"] if result is not None else None
    local_ok = (
        result is not None
        and result.replay_mode == "readiness_incomplete"
        and readiness is not None
        and readiness.status is C3Status.INCONCLUSIVE
        and not result.completed_scheduled_cycle
        and harness._detector is not None
    )
    agreed = torch.tensor([int(local_ok)], dtype=torch.int32)
    dist.all_reduce(agreed, op=dist.ReduceOp.MIN)
    assert agreed.item() == 1

    harness.remove_hooks()
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
