"""DataLoader timing instrumentation."""

from __future__ import annotations

import threading

import pytest

from lm_resiliency.detection.op_tracker import DiagnosticStage, OpTracker
from lm_resiliency.detection.stage_instrumentation import (
    DiagnosticStageMonitor,
    checkpoint_io,
    instrument_dataloader,
)


def test_dataloader_measurement_respects_detection_interval():
    tracker = OpTracker(rank=207)
    try:
        monitor = DiagnosticStageMonitor(tracker, threading.Lock())
        dataloader = instrument_dataloader(
            [10, 20],
            monitor,
            detection_interval=2,
        )
        iterator = iter(dataloader)

        assert next(iterator) == 10
        assert tracker.op_id == 0
        assert tracker._completed_stage is DiagnosticStage.NONE

        tracker.step_boundary()
        assert next(iterator) == 20
        assert tracker.op_id == 2
        assert tracker._completed_stage is DiagnosticStage.DATA_LOADING
    finally:
        tracker.close()


def test_zero_detection_interval_disables_dataloader_measurement():
    tracker = OpTracker(rank=208)
    try:
        monitor = DiagnosticStageMonitor(tracker, threading.Lock())
        dataloader = instrument_dataloader([10], monitor, detection_interval=0)

        assert next(iter(dataloader)) == 10
        assert tracker.op_id == 0
        assert tracker._completed_stage is DiagnosticStage.NONE
    finally:
        tracker.close()


def test_checkpoint_read_and_write_publish_explicit_stages():
    tracker = OpTracker(rank=209)
    try:
        monitor = DiagnosticStageMonitor(tracker, threading.Lock())

        with checkpoint_io(monitor, "write", name="step-10"):
            assert tracker._active_stage is DiagnosticStage.CHECKPOINT_WRITE
        assert tracker._completed_stage is DiagnosticStage.CHECKPOINT_WRITE

        with checkpoint_io(monitor, "read", name="recovery"):
            assert tracker._active_stage is DiagnosticStage.CHECKPOINT_READ
        assert tracker._completed_stage is DiagnosticStage.CHECKPOINT_READ
    finally:
        tracker.close()


def test_checkpoint_io_without_monitor_is_a_noop():
    with checkpoint_io(None, "write"):
        pass


def test_checkpoint_io_rejects_unknown_operation():
    with pytest.raises(ValueError, match="read.*write"):
        with checkpoint_io(None, "delete"):  # type: ignore[arg-type]
            pass
