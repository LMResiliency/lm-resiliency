"""Unit tests for OpTracker and OpTrackerReader (shared memory interface)."""

import time
import unittest
from unittest.mock import MagicMock, patch

from lm_resiliency.detection.op_tracker import (
    DiagnosticStage,
    OpTracker,
    OpTrackerReader,
    _shm_name,
)


class TestOpTracker(unittest.TestCase):
    def setUp(self):
        self.tracker = OpTracker(rank=99)

    def tearDown(self):
        self.tracker.close()

    def test_initial_state(self):
        self.assertEqual(self.tracker.op_id, 0)
        self.assertEqual(self.tracker.step, 0)

    def test_advance(self):
        self.tracker.advance(metadata_fingerprint=123)
        self.assertEqual(self.tracker.op_id, 1)
        self.assertEqual(self.tracker.metadata_fingerprint, 123)
        self.tracker.advance()
        self.assertEqual(self.tracker.op_id, 2)
        self.assertEqual(self.tracker.metadata_fingerprint, 0)

    def test_step_boundary_resets_op_id(self):
        self.tracker.advance()
        self.tracker.advance()
        self.assertEqual(self.tracker.op_id, 2)
        self.tracker.step_boundary()
        self.assertEqual(self.tracker.op_id, 0)
        self.assertEqual(self.tracker.step, 1)

    def test_diagnostic_stage_publishes_active_and_completed_timing(self):
        token = self.tracker.stage_started(DiagnosticStage.DATA_LOADING, key=123)
        active = self.tracker._active_stage

        self.assertEqual(active, DiagnosticStage.DATA_LOADING)
        self.assertEqual(self.tracker.op_id, 1)
        duration_ns = self.tracker.stage_finished(token)

        self.assertGreaterEqual(duration_ns, 0)
        self.assertEqual(self.tracker._active_stage, DiagnosticStage.NONE)
        self.assertEqual(self.tracker._completed_stage, DiagnosticStage.DATA_LOADING)
        self.assertEqual(self.tracker._completed_stage_key, 123)
        self.assertEqual(self.tracker.op_id, 2)

    def test_regular_progress_notifications_are_rate_limited(self):
        progress_event = MagicMock()
        with patch(
            "lm_resiliency.detection.op_tracker.time.monotonic_ns",
            side_effect=[1_000_000_000, 1_100_000_000, 1_600_000_001],
        ):
            tracker = OpTracker(rank=100, progress_event=progress_event)
            try:
                progress_event.reset_mock()
                tracker.advance()
                progress_event.set.assert_not_called()

                tracker.advance()
                progress_event.set.assert_called_once()
            finally:
                tracker.close()

    def test_boundaries_force_progress_notifications(self):
        progress_event = MagicMock()
        tracker = OpTracker(rank=101, progress_event=progress_event)
        try:
            progress_event.reset_mock()
            tracker.step_boundary()
            progress_event.set.assert_called_once()

            progress_event.reset_mock()
            token = tracker.stage_started(DiagnosticStage.DATA_LOADING, key=123)
            tracker.stage_finished(token)
            self.assertEqual(progress_event.set.call_count, 2)

            progress_event.reset_mock()
            tracker.advance(force_signal=True)
            progress_event.set.assert_called_once()
        finally:
            tracker.close()

    def test_shm_name(self):
        self.assertEqual(_shm_name(99), "scout_op_tracker_rank_99")
        self.assertEqual(_shm_name(99, "opaque-channel"), "opaque-channel")

    def test_live_owner_collision_does_not_attach_or_unlink(self):
        name = "scout_test_owner_collision"
        owner = OpTracker(rank=102, shm_name=name, owner_token=b"a" * 16)
        try:
            with self.assertRaisesRegex(FileExistsError, "owned by live process"):
                OpTracker(rank=102, shm_name=name, owner_token=b"b" * 16)

            wrong_reader = OpTrackerReader(
                rank=102,
                shm_name=name,
                owner_token=b"b" * 16,
            )
            with self.assertRaisesRegex(RuntimeError, "ownership token mismatch"):
                wrong_reader.attach()
        finally:
            owner.close()


class TestOpTrackerReader(unittest.TestCase):
    def setUp(self):
        self.tracker = OpTracker(rank=98)
        self.reader = OpTrackerReader(rank=98)

    def tearDown(self):
        self.reader.close()
        self.tracker.close()

    def test_attach_succeeds(self):
        self.assertTrue(self.reader.attach())

    def test_read_initial_state(self):
        self.reader.attach()
        op_id, heartbeat, step = self.reader.read()
        self.assertEqual(op_id, 0)
        self.assertEqual(step, 0)
        self.assertGreater(heartbeat, 0)

    def test_read_after_advance(self):
        self.reader.attach()
        self.tracker.advance()
        self.tracker.advance()
        self.tracker.advance()
        op_id, _, step = self.reader.read()
        self.assertEqual(op_id, 3)
        self.assertEqual(step, 0)

    def test_read_after_step_boundary(self):
        self.reader.attach()
        self.tracker.advance()
        self.tracker.step_boundary()
        self.tracker.advance()
        op_id, _, step = self.reader.read()
        self.assertEqual(op_id, 1)
        self.assertEqual(step, 1)

    def test_read_collective_metadata(self):
        self.reader.attach()
        self.tracker.advance(metadata_fingerprint=-12345)
        op_id, _, step, metadata = self.reader.read_with_metadata()
        self.assertEqual((op_id, step, metadata), (1, 0, -12345))

    def test_read_diagnostic_stage_snapshot(self):
        self.reader.attach()
        token = self.tracker.stage_started(DiagnosticStage.DATA_LOADING, key=-45)

        active = self.reader.read_snapshot()
        self.assertEqual(active.active_stage, DiagnosticStage.DATA_LOADING)
        self.assertEqual(active.active_stage_key, -45)
        self.assertEqual(active.active_stage_sequence, 1)
        self.assertGreater(active.active_stage_started_ns, 0)

        self.tracker.stage_finished(token)
        completed = self.reader.read_snapshot()
        self.assertEqual(completed.active_stage, DiagnosticStage.NONE)
        self.assertEqual(completed.completed_stage, DiagnosticStage.DATA_LOADING)
        self.assertEqual(completed.completed_stage_key, -45)
        self.assertEqual(completed.completed_stage_sequence, 1)
        self.assertGreaterEqual(completed.completed_stage_duration_ns, 0)

    def test_read_without_attach_returns_sentinel(self):
        op_id, heartbeat, step = self.reader.read()
        self.assertEqual(op_id, -1)
        self.assertEqual(step, -1)

    def test_heartbeat_updates(self):
        self.reader.attach()
        _, t1, _ = self.reader.read()
        time.sleep(0.01)
        self.tracker.advance()
        _, t2, _ = self.reader.read()
        self.assertGreater(t2, t1)


class TestOpTrackerReaderNotFound(unittest.TestCase):
    def test_attach_fails_when_no_tracker(self):
        reader = OpTrackerReader(rank=9999)
        self.assertFalse(reader.attach())
        reader.close()


if __name__ == "__main__":
    unittest.main()
