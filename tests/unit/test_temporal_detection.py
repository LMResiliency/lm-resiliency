"""Temporal baseline tests for hierarchical replay detection."""

from lm_resiliency.detection.temporal import TemporalBaselineStore


def test_temporal_detects_one_rank_and_group_wide_slowdown():
    store = TemporalBaselineStore(
        window_size=8,
        min_samples=3,
        slowdown_ratio=1.2,
        threshold_sigma=4.0,
    )
    for _ in range(3):
        store.observe_clean("layer", [10.0, 10.0, 10.0, 10.0])

    one_rank = store.assess("layer", [10.0, 16.0, 10.0, 10.0])
    assert one_rank.rank_bitmap == [0, 1, 0, 0]
    assert one_rank.group_slowdown is False

    shared = store.assess("layer", [15.0, 15.0, 15.0, 15.0])
    assert shared.rank_bitmap == [1, 1, 1, 1]
    assert shared.group_slowdown is True


def test_temporal_state_round_trip_is_bounded():
    original = TemporalBaselineStore(window_size=3, min_samples=2)
    for value in range(8):
        original.observe_clean("key", [float(value + 1), float(value + 1)])

    state = original.state_dict()
    assert len(state["baselines"]["key"]["group"]) == 3

    restored = TemporalBaselineStore(window_size=3, min_samples=2)
    restored.load_state_dict(state)
    assessment = restored.assess("key", [20.0, 20.0])
    assert assessment.group_slowdown is True
