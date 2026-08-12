"""Automatic cross-process-group communication localization."""

from __future__ import annotations

import socket
from dataclasses import dataclass
from typing import Any, Callable

import torch.distributed as dist

from lm_resiliency.detection.communication_localizer import (
    CollectiveObservation,
    CommunicationLocalizer,
)


@dataclass(frozen=True)
class CollectiveTimingSample:
    """One rank's classified timing for a replay-visible collective."""

    collective: str
    group_ranks: tuple[int, ...]
    message_bytes: int
    sequence: int
    latency_ms: float
    slow: bool
    topology_role: str = "model_parallel"

    def __post_init__(self) -> None:
        if not self.collective or not self.group_ranks:
            raise ValueError("collective and group_ranks must be non-empty")
        if self.message_bytes < 0 or self.sequence < 0 or self.latency_ms < 0:
            raise ValueError("collective timing values must be non-negative")

    @property
    def group_id(self) -> str:
        ranks = ",".join(str(rank) for rank in self.group_ranks)
        return (
            f"{self.topology_role}:{self.collective}:{self.sequence}:"
            f"{self.message_bytes}B:[{ranks}]"
        )


@dataclass(frozen=True)
class CrossPGResult:
    """Confirmed or unresolved cross-PG endpoint evidence."""

    status: str
    failed_rank: int | None = None
    failed_node: str | None = None
    failed_ranks: tuple[int, ...] = ()
    supporting_groups: tuple[str, ...] = ()
    slow_groups: tuple[str, ...] = ()
    healthy_groups: tuple[str, ...] = ()
    confidence: float = 0.0
    detail: str = ""

    @property
    def confirmed(self) -> bool:
        return self.status == "localized" and self.failed_rank is not None


GatherPayload = Callable[
    [list[CollectiveTimingSample]],
    list[tuple[int, str, list[CollectiveTimingSample]]],
]


class CrossPGCoordinator:
    """Gather replay timings and intersect independently slow process groups."""

    def __init__(
        self,
        *,
        gather: GatherPayload | None = None,
        rank_to_node: dict[int, str] | None = None,
    ) -> None:
        self._gather = gather or _gather_world_samples
        self._rank_to_node = dict(rank_to_node or {})

    def localize(
        self,
        local_samples: list[CollectiveTimingSample],
    ) -> CrossPGResult:
        """Return a machine-level diagnosis only for a confirmed rank intersection."""
        payloads = self._gather(local_samples)
        samples_by_group: dict[
            tuple[str, tuple[int, ...], int, int, str],
            list[tuple[int, CollectiveTimingSample]],
        ] = {}
        for reporter_rank, node, samples in payloads:
            self._rank_to_node.setdefault(int(reporter_rank), str(node))
            for sample in samples:
                key = (
                    sample.collective,
                    sample.group_ranks,
                    sample.message_bytes,
                    sample.sequence,
                    sample.topology_role,
                )
                samples_by_group.setdefault(key, []).append((int(reporter_rank), sample))

        observations: list[CollectiveObservation] = []
        for members in samples_by_group.values():
            sample = members[0][1]
            reporters = {rank for rank, _ in members}
            expected = set(sample.group_ranks)
            if reporters != expected:
                continue
            slow_votes = sum(int(item.slow) for _, item in members)
            is_slow = slow_votes > len(members) // 2
            observations.append(
                CollectiveObservation(
                    group_id=sample.group_id,
                    members=tuple(str(rank) for rank in sample.group_ranks),
                    collective=sample.collective,
                    message_bytes=sample.message_bytes,
                    topology_role=sample.topology_role,
                    resource_class="machine_endpoint",
                    latency_s=2.0 if is_slow else 1.0,
                    baseline_latency_s=1.0,
                    resource_kind="rank",
                )
            )

        verdict = CommunicationLocalizer(
            slow_factor=1.5,
            healthy_factor=1.15,
            persistence=1,
        ).observe(observations)
        suspect = verdict.suspect
        if not verdict.confirmed or suspect is None:
            return CrossPGResult(
                status=verdict.status,
                supporting_groups=(suspect.supporting_groups if suspect is not None else ()),
                slow_groups=verdict.slow_groups,
                healthy_groups=verdict.healthy_groups,
                confidence=verdict.confidence,
                detail=verdict.detail,
            )

        failed_rank = int(suspect.resource_id)
        failed_node = self._rank_to_node.get(failed_rank)
        failed_ranks = tuple(
            sorted(
                rank
                for rank, node in self._rank_to_node.items()
                if failed_node is not None and node == failed_node
            )
        )
        if not failed_ranks:
            failed_ranks = (failed_rank,)
        return CrossPGResult(
            status="localized",
            failed_rank=failed_rank,
            failed_node=failed_node,
            failed_ranks=failed_ranks,
            supporting_groups=suspect.supporting_groups,
            slow_groups=verdict.slow_groups,
            healthy_groups=verdict.healthy_groups,
            confidence=verdict.confidence,
            detail=verdict.detail,
        )


def _gather_world_samples(
    local_samples: list[CollectiveTimingSample],
) -> list[tuple[int, str, list[CollectiveTimingSample]]]:
    rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
    payload = (rank, socket.gethostname(), local_samples)
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() == 1:
        return [payload]
    gathered: list[Any] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, payload)
    return [(int(item[0]), str(item[1]), list(item[2])) for item in gathered if item is not None]


__all__ = [
    "CollectiveTimingSample",
    "CrossPGCoordinator",
    "CrossPGResult",
]
