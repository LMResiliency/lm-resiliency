"""Tests for automatic cross-process-group endpoint localization."""

from __future__ import annotations

from lm_resiliency.detection.cross_pg import (
    CollectiveTimingSample,
    CrossPGCoordinator,
    CrossPGResult,
)
from lm_resiliency.detection.layer_replay import ReplayResult, replay_result_has_fault
from lm_resiliency.detection.reports import replay_fault_reports


def _sample(
    collective: str,
    ranks: tuple[int, ...],
    *,
    slow: bool,
    role: str,
) -> CollectiveTimingSample:
    return CollectiveTimingSample(
        collective=collective,
        group_ranks=ranks,
        message_bytes=4096,
        sequence=0,
        latency_ms=5.0 if slow else 1.0,
        slow=slow,
        topology_role=role,
    )


def _payloads(
    samples: dict[int, list[CollectiveTimingSample]],
) -> list[tuple[int, str, list[CollectiveTimingSample]]]:
    nodes = {
        4: "machine-a",
        12: "machine-a",
        5: "machine-b",
        9: "machine-b",
    }
    ranks = sorted(set(samples) | set(nodes))
    return [(rank, nodes.get(rank, f"machine-{rank}"), samples.get(rank, [])) for rank in ranks]


def test_slow_fsdp_and_tp_groups_localize_their_shared_rank_and_machine():
    fsdp = _sample(
        "fsdp_parameter_all_gather",
        (4, 5),
        slow=True,
        role="fsdp",
    )
    tp = _sample("all_reduce", (4, 12), slow=True, role="model_parallel")
    payloads = _payloads({4: [fsdp, tp], 5: [fsdp], 12: [tp]})
    coordinator = CrossPGCoordinator(gather=lambda _: payloads)

    result = coordinator.localize([])

    assert result.confirmed
    assert result.failed_rank == 4
    assert result.failed_node == "machine-a"
    assert result.failed_ranks == (4, 12)
    assert len(result.supporting_groups) == 2


def test_incomplete_group_observations_are_not_used_for_attribution():
    fsdp = _sample(
        "fsdp_parameter_all_gather",
        (4, 5),
        slow=True,
        role="fsdp",
    )
    tp = _sample("all_reduce", (4, 12), slow=True, role="model_parallel")
    payloads = _payloads({4: [fsdp, tp], 12: [tp]})
    coordinator = CrossPGCoordinator(gather=lambda _: payloads)

    result = coordinator.localize([])

    assert not result.confirmed
    assert result.status == "inconclusive"


def test_healthy_group_using_the_shared_rank_blocks_attribution():
    fsdp = _sample(
        "fsdp_parameter_all_gather",
        (4, 5),
        slow=True,
        role="fsdp",
    )
    tp = _sample("all_reduce", (4, 12), slow=True, role="model_parallel")
    healthy = _sample("all_gather", (4, 9), slow=False, role="context_parallel")
    payloads = _payloads(
        {
            4: [fsdp, tp, healthy],
            5: [fsdp],
            9: [healthy],
            12: [tp],
        }
    )
    coordinator = CrossPGCoordinator(gather=lambda _: payloads)

    result = coordinator.localize([])

    assert not result.confirmed
    assert result.status == "inconclusive"


def test_confirmed_cross_pg_result_enriches_manager_fault_report():
    result = ReplayResult(
        sdc_bitmap=[0, 0],
        straggler_bitmap=[0, 0],
        replay_time_ms=1.0,
        layer_id=3,
        peer_ranks=[4, 8],
        cross_pg_result=CrossPGResult(
            status="localized",
            failed_rank=4,
            failed_node="machine-a",
            failed_ranks=(4, 12),
            supporting_groups=("fsdp", "tp"),
            slow_groups=("fsdp", "tp"),
            confidence=1.0,
        ),
    )

    assert replay_result_has_fault(result)
    assert replay_fault_reports(result) == [
        {
            "failed_ranks": [4, 12],
            "layer_id": 3,
            "kind": "straggler",
            "straggler_type": "communication",
            "scope": "node",
            "confirmations": 0,
            "endpoint_kind": "node",
            "endpoint_id": "machine-a",
            "endpoint_rank": 4,
            "supporting_groups": ["fsdp", "tp"],
            "slow_groups": ["fsdp", "tp"],
            "healthy_groups": [],
            "confidence": 1.0,
        }
    ]
