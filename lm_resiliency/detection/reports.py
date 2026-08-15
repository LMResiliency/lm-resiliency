# mypy: ignore-errors
"""Stable SCOUT fault-report payloads for external orchestration."""

from __future__ import annotations

from typing import Callable, TypedDict

from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.detection.layer_replay import ReplayResult, replay_result_has_sdc


class SCOUTFaultReport(TypedDict, total=False):
    """JSON-ready fault report emitted at the worker/orchestrator boundary."""

    failed_ranks: list[int]
    kind: str
    scope: str
    layer_id: int
    sources: list[str]
    straggler_type: str
    confirmations: int
    op_ids: list[int]
    steps: list[int]
    collective_metadata: list[int]
    mismatch_kind: str | None
    stall_duration_s: float
    dataloader_active: bool
    dataloader_key: int | None
    dataloader_sequence: int | None
    dataloader_bitmap: list[int]
    dataloader_latencies_ms: list[float]
    dataloader_culprit_ranks: list[int]
    dataloader_confirmations: int
    stage_active: bool
    stage_kind: str | None
    stage_key: int | None
    stage_sequence: int | None
    stage_bitmap: list[int]
    stage_latencies_ms: list[float]
    stage_culprit_ranks: list[int]
    stage_confirmations: int
    endpoint_kind: str
    endpoint_id: str
    endpoint_rank: int
    supporting_groups: list[str]
    slow_groups: list[str]
    healthy_groups: list[str]
    confidence: float


SCOUTFaultCallback = Callable[[SCOUTFaultReport], None]


def replay_fault_reports(result: ReplayResult) -> list[SCOUTFaultReport]:
    """Translate one replay result into separate SDC and straggler reports."""
    peer_ranks = result.peer_ranks

    def global_rank(local_index: int) -> int:
        if peer_ranks and local_index < len(peer_ranks):
            return peer_ranks[local_index]
        return local_index

    reports: list[SCOUTFaultReport] = []
    sdc_ranks = sorted(
        {global_rank(index) for index, value in enumerate(result.sdc_bitmap) if value}
    )
    if replay_result_has_sdc(result):
        inconclusive_sources = [
            source
            for source, c3_result in result.c3_results.items()
            if c3_result.status is C3Status.INCONCLUSIVE
        ]
        if not sdc_ranks:
            sdc_ranks = list(peer_ranks or range(len(result.sdc_bitmap)))
        reports.append(
            {
                "failed_ranks": sdc_ranks,
                "layer_id": result.layer_id,
                "kind": "sdc",
                "scope": "rank" if any(result.sdc_bitmap) else "peer_group",
                "sources": sorted(set(result.sdc_sources) | set(inconclusive_sources)),
            }
        )

    straggler_ranks = sorted(
        {global_rank(index) for index, value in enumerate(result.straggler_bitmap) if value}
    )
    cross_pg = result.cross_pg_result
    if cross_pg is not None and cross_pg.confirmed:
        straggler_ranks = list(cross_pg.failed_ranks)
    if result.temporal_group_slowdown and not straggler_ranks:
        straggler_ranks = list(peer_ranks or range(len(result.straggler_bitmap)))
    if straggler_ranks:
        report: SCOUTFaultReport = {
            "failed_ranks": straggler_ranks,
            "layer_id": result.layer_id,
            "kind": "straggler",
            "straggler_type": (
                "communication"
                if cross_pg is not None and cross_pg.confirmed
                else (
                    result.straggler_detail.straggler_type
                    if result.straggler_detail is not None
                    else "unknown"
                )
            ),
            "scope": (
                "node"
                if cross_pg is not None and cross_pg.confirmed
                else ("peer_group" if result.temporal_group_slowdown else "rank")
            ),
            "confirmations": result.straggler_confirmations,
        }
        if cross_pg is not None and cross_pg.confirmed:
            report.update(
                {
                    "endpoint_kind": "node",
                    "endpoint_id": cross_pg.failed_node or str(cross_pg.failed_rank),
                    "endpoint_rank": int(cross_pg.failed_rank),
                    "supporting_groups": list(cross_pg.supporting_groups),
                    "slow_groups": list(cross_pg.slow_groups),
                    "healthy_groups": list(cross_pg.healthy_groups),
                    "confidence": cross_pg.confidence,
                }
            )
        reports.append(report)
    return reports


def dispatch_replay_faults(
    result: ReplayResult,
    callback: SCOUTFaultCallback,
) -> None:
    """Send every normalized report from ``result`` to ``callback``."""
    for report in replay_fault_reports(result):
        callback(report)


__all__ = [
    "SCOUTFaultCallback",
    "SCOUTFaultReport",
    "dispatch_replay_faults",
    "replay_fault_reports",
]
