from lm_resiliency.cadence import ResiliencyCadence


def test_checkpoint_only_uses_component_interval():
    cadence = ResiliencyCadence(interval=3, checkpoint_enabled=True)
    assert [step for step in range(1, 7) if cadence.checkpoint_due(step)] == [3, 6]


def test_detection_only_never_checkpoints():
    cadence = ResiliencyCadence(interval=4, detection_enabled=True)
    assert [step for step in range(1, 9) if cadence.detection_due(step)] == [4, 8]
    assert not any(cadence.checkpoint_due(step) for step in range(1, 9))


def test_detection_and_checkpoint_use_the_same_even_interval():
    cadence = ResiliencyCadence(
        interval=4,
        checkpoint_enabled=True,
        detection_enabled=True,
    )
    assert [step for step in range(1, 9) if cadence.checkpoint_due(step)] == [4, 8]
    assert [step for step in range(1, 9) if cadence.detection_due(step)] == [4, 8]


def test_detection_and_checkpoint_use_the_same_odd_interval():
    cadence = ResiliencyCadence(
        interval=3,
        checkpoint_enabled=True,
        detection_enabled=True,
    )
    assert [step for step in range(1, 7) if cadence.checkpoint_due(step)] == [3, 6]
    assert [step for step in range(1, 7) if cadence.detection_due(step)] == [3, 6]


def test_detection_every_step_checkpoints_every_step():
    cadence = ResiliencyCadence(
        interval=1,
        checkpoint_enabled=True,
        detection_enabled=True,
    )
    assert all(cadence.checkpoint_due(step) for step in range(1, 4))


def test_component_intervals_must_be_coupled():
    try:
        ResiliencyCadence.from_component_intervals(
            checkpoint_interval=4,
            detection_interval=8,
        )
    except ValueError as exc:
        assert "one interval" in str(exc)
    else:
        raise AssertionError("independent GEMINI and SCOUT intervals should be rejected")
