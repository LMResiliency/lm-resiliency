"""Pure result aggregation helpers for repeated replay rounds."""

from __future__ import annotations

import statistics

from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.temporal import TemporalAssessment


def merge_sdc_source_bitmaps(
    result: ReplayResult,
    source_bitmaps: dict[str, list[int]],
) -> None:
    """Merge additional C3 sources into a replay result in place."""
    if not source_bitmaps:
        return
    width = len(result.sdc_bitmap)
    if any(len(bitmap) != width for bitmap in source_bitmaps.values()):
        raise RuntimeError("optimizer-step C3 bitmap does not match the replay peer group")
    result.sdc_source_bitmaps.update(source_bitmaps)
    result.sdc_bitmap = [
        int(any(bitmap[index] for bitmap in result.sdc_source_bitmaps.values()))
        for index in range(width)
    ]
    result.sdc_sources = [
        source for source, bitmap in result.sdc_source_bitmaps.items() if any(bitmap)
    ]


def has_timing_candidate(result: ReplayResult, assessment: TemporalAssessment) -> bool:
    """Return whether spatial or temporal timing needs confirmation."""
    return (
        any(result.spatial_straggler_bitmap or result.straggler_bitmap)
        or any(assessment.rank_bitmap)
        or assessment.group_slowdown
    )


def merge_replay_rounds(
    rounds: list[ReplayResult],
    assessments: list[TemporalAssessment],
    *,
    required: int,
) -> ReplayResult:
    """Merge confirmation rounds into the first result object."""
    result = rounds[0]
    width = len(result.straggler_bitmap)
    spatial_counts = [0] * width
    temporal_counts = [0] * width
    for replay, assessment in zip(rounds, assessments):
        for index, flagged in enumerate(replay.spatial_straggler_bitmap or replay.straggler_bitmap):
            spatial_counts[index] += int(bool(flagged))
        for index, flagged in enumerate(assessment.rank_bitmap):
            temporal_counts[index] += int(bool(flagged))

    result.spatial_straggler_bitmap = [int(count >= required) for count in spatial_counts]
    result.temporal_straggler_bitmap = [int(count >= required) for count in temporal_counts]
    result.straggler_bitmap = [
        int(spatial or temporal)
        for spatial, temporal in zip(
            result.spatial_straggler_bitmap, result.temporal_straggler_bitmap
        )
    ]
    group_count = sum(int(assessment.group_slowdown) for assessment in assessments)
    result.temporal_group_slowdown = group_count >= required
    result.straggler_confirmations = max([group_count, *spatial_counts, *temporal_counts])

    source_names = {name for replay in rounds for name in replay.sdc_source_bitmaps}
    result.sdc_source_bitmaps = {
        name: [
            int(any(replay.sdc_source_bitmaps.get(name, [0] * width)[index] for replay in rounds))
            for index in range(width)
        ]
        for name in source_names
    }
    result.sdc_bitmap = [
        int(any(bitmap[index] for bitmap in result.sdc_source_bitmaps.values()))
        for index in range(width)
    ]
    result.sdc_sources = [
        source for source, bitmap in result.sdc_source_bitmaps.items() if any(bitmap)
    ]

    timing_vectors = [
        replay.replay_times_ms for replay in rounds if len(replay.replay_times_ms) == width
    ]
    if timing_vectors:
        result.replay_times_ms = [
            float(statistics.median(vector[index] for vector in timing_vectors))
            for index in range(width)
        ]
    result.replay_time_ms = float(statistics.median(replay.replay_time_ms for replay in rounds))
    return result
