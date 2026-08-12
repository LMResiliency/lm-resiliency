"""Focused tests for replay-round result aggregation."""

from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.replay_analysis import (
    merge_replay_rounds,
    merge_sdc_source_bitmaps,
)
from lm_resiliency.detection.temporal import TemporalAssessment


def _result(*, sdc: list[int], straggler: list[int], times: list[float]) -> ReplayResult:
    return ReplayResult(
        sdc_bitmap=sdc,
        straggler_bitmap=straggler,
        replay_time_ms=max(times),
        layer_id=3,
        peer_ranks=list(range(len(sdc))),
        replay_times_ms=times,
        sdc_source_bitmaps={"output": sdc},
        spatial_straggler_bitmap=straggler,
    )


def test_merge_replay_rounds_confirms_timing_and_unions_sdc_sources():
    rounds = [
        _result(sdc=[0, 1], straggler=[1, 0], times=[20.0, 10.0]),
        _result(sdc=[0, 0], straggler=[1, 0], times=[22.0, 12.0]),
    ]
    assessments = [
        TemporalAssessment([0, 1], False),
        TemporalAssessment([0, 1], False),
    ]

    result = merge_replay_rounds(rounds, assessments, required=2)

    assert result.sdc_bitmap == [0, 1]
    assert result.spatial_straggler_bitmap == [1, 0]
    assert result.temporal_straggler_bitmap == [0, 1]
    assert result.straggler_bitmap == [1, 1]
    assert result.replay_times_ms == [21.0, 11.0]


def test_merge_sdc_source_bitmaps_updates_combined_bitmap():
    result = _result(sdc=[0, 0], straggler=[0, 0], times=[1.0, 1.0])

    merge_sdc_source_bitmaps(result, {"optimizer_updated_weight": [1, 0]})

    assert result.sdc_bitmap == [1, 0]
    assert result.sdc_sources == ["optimizer_updated_weight"]
