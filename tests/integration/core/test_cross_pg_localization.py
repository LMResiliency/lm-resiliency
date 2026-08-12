"""Four-process integration test for Cross-PG endpoint localization.

Run:
    torchrun --standalone --nproc_per_node=4 \
        tests/integration/core/test_cross_pg_localization.py
"""

from __future__ import annotations

import os
import socket
import time

import torch
import torch.distributed as dist

from lm_resiliency.detection.cross_pg import (
    CollectiveTimingSample,
    CrossPGCoordinator,
)


def _timed_sample(
    *,
    group: dist.ProcessGroup,
    group_ranks: tuple[int, ...],
    role: str,
    device: torch.device,
) -> CollectiveTimingSample:
    value = torch.ones(1024, device=device)
    torch.cuda.synchronize(device)
    started = time.perf_counter()
    if dist.get_rank() == 0:
        time.sleep(0.2)
    dist.all_reduce(value, group=group)
    torch.cuda.synchronize(device)
    latency_ms = (time.perf_counter() - started) * 1000.0
    return CollectiveTimingSample(
        collective="all_reduce",
        group_ranks=group_ranks,
        message_bytes=value.numel() * value.element_size(),
        sequence=0,
        latency_ms=latency_ms,
        slow=latency_ms >= 100.0,
        topology_role=role,
    )


def main() -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    assert dist.get_world_size() == 4
    memberships = (
        ("fsdp", (0, 1)),
        ("model_parallel", (0, 2)),
    )
    local_samples = []
    for role, members in memberships:
        group = dist.new_group(list(members), backend="nccl")
        if rank in members:
            local_samples.append(
                _timed_sample(
                    group=group,
                    group_ranks=members,
                    role=role,
                    device=device,
                )
            )

    hostnames: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(hostnames, socket.gethostname())
    result = CrossPGCoordinator().localize(local_samples)
    assert result.confirmed, result
    assert result.failed_rank == 0, result
    assert result.failed_node == hostnames[0], result
    expected_failed_ranks = tuple(
        index for index, hostname in enumerate(hostnames) if hostname == hostnames[0]
    )
    assert result.failed_ranks == expected_failed_ranks, result
    assert len(result.supporting_groups) == 2, result

    if rank == 0:
        print(
            "SCOUT CROSS-PG OK: two measured slow CUDA groups intersected at "
            f"rank 0 and selected {result.failed_node}."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
