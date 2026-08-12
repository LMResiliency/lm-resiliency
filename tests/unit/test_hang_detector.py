"""Unit tests for HangDetectionDaemon with mocked distributed operations.

Tests the daemon logic without requiring torchrun or multiple processes.
All dist.* calls are mocked so tests run in a single process.
"""

import time
import unittest
from unittest.mock import MagicMock, patch

from lm_resiliency.detection.c3 import C3, C3Mode
from lm_resiliency.detection.hang_detector import HangDetectionDaemon, HangLocalizationResult
from lm_resiliency.detection.op_tracker import DiagnosticStage, OpTracker, OpTrackerReader


def _c3_result(values, mode=C3Mode.EXACT):
    return C3.classify_evidence(values, mode)


def _mock_c3(op_ids):
    """Create a mock C3 that simulates AllGather with the given op_ids."""
    mock = MagicMock(spec=C3)
    mock._world_size = len(op_ids)

    def fake_run_scalar(value, mode=None, threshold_sigma=3.0):
        del value
        return C3.classify_evidence(
            op_ids,
            mode or C3Mode.EXACT,
            threshold_sigma,
        )

    mock.run_scalar.side_effect = fake_run_scalar
    return mock


class TestHangDetectorLocalization(unittest.TestCase):
    """Tests C³ localization logic with mocked distributed ops."""

    def setUp(self):
        self.tracker = OpTracker(rank=50)

    def tearDown(self):
        self.tracker.close()

    def _make_daemon(self, op_ids, local_op_id=None):
        """Create a daemon with mocked C3 that will report the given op_ids."""
        world_size = len(op_ids)
        mock_group = MagicMock()

        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = _mock_c3(op_ids)
            daemon = HangDetectionDaemon(
                rank=0,
                world_size=world_size,
                group=mock_group,
                stall_threshold_s=0.05,
                confirmation_interval_s=0.01,
            )

        # Point reader at our tracker's shm
        daemon._reader.close()
        daemon._reader = OpTrackerReader(rank=50)
        daemon._reader._shm = self.tracker._shm

        # Set the tracker to the local op_id
        if local_op_id is not None:
            self.tracker._op_id = local_op_id
            self.tracker._write()

        daemon._last_op_id = self.tracker.op_id
        daemon._last_step = self.tracker.step
        daemon._last_change_time = time.time()

        return daemon

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_lagging_rank_detected(self, mock_all_gather):
        """Case 1: one rank lags — C³ identifies it as outlier."""
        op_ids = [10, 10, 5, 10]
        self.tracker._op_id = 10
        self.tracker._write()

        daemon = self._make_daemon(op_ids, local_op_id=10)

        def fake_all_gather(gathered, local_tensor, group=None):
            for i, t in enumerate(gathered):
                t.fill_(op_ids[i])

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertEqual(result.culprit_rank, 2)
        self.assertEqual(result.bitmap, [0, 0, 1, 0])
        self.assertEqual(result.op_ids, [10, 10, 5, 10])
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_diverged_rank_detected(self, mock_all_gather):
        """Case 2: one rank diverged to a different op — C³ identifies it."""
        op_ids = [10, 23, 10, 10]
        self.tracker._op_id = 10
        self.tracker._write()

        daemon = self._make_daemon(op_ids, local_op_id=10)

        def fake_all_gather(gathered, local_tensor, group=None):
            for i, t in enumerate(gathered):
                t.fill_(op_ids[i])

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertEqual(result.culprit_rank, 1)
        self.assertEqual(result.bitmap, [0, 1, 0, 0])
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_no_outlier_all_same(self, mock_all_gather):
        """Case 3: all ranks at same op_id — no culprit (possible network hang)."""
        op_ids = [17, 17, 17, 17]
        self.tracker._op_id = 17
        self.tracker._write()

        daemon = self._make_daemon(op_ids, local_op_id=17)

        def fake_all_gather(gathered, local_tensor, group=None):
            for t in gathered:
                t.fill_(17)

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertIsNone(result.culprit_rank)
        self.assertEqual(result.bitmap, [0, 0, 0, 0])
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_collective_metadata_mismatch_localizes_rank(self, mock_all_gather):
        op_ids = [17, 17, 17, 17]
        metadata = [123, 123, 999, 123]
        self.tracker._op_id = 17
        self.tracker._metadata_fingerprint = 123
        self.tracker._write()
        daemon = self._make_daemon(op_ids, local_op_id=17)

        def fake_all_gather(gathered, local_tensor, group=None):
            for index, tensor in enumerate(gathered):
                tensor[0] = op_ids[index]
                tensor[1] = 0
                tensor[2] = metadata[index]

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertEqual(result.culprit_rank, 2)
        self.assertEqual(result.culprit_ranks, [2])
        self.assertEqual(result.bitmap, [0, 0, 1, 0])
        self.assertEqual(result.mismatch_kind, "collective_metadata")
        self.assertEqual(result.metadata_fingerprints, metadata)
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_collective_metadata_without_majority_stays_group_scoped(self, mock_all_gather):
        op_ids = [17, 17, 17, 17]
        metadata = [123, 123, 999, 999]
        daemon = self._make_daemon(op_ids, local_op_id=17)

        def fake_all_gather(gathered, local_tensor, group=None):
            for index, tensor in enumerate(gathered):
                tensor[0] = op_ids[index]
                tensor[1] = 0
                tensor[2] = metadata[index]

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertIsNone(result.culprit_rank)
        self.assertEqual(result.culprit_ranks, [])
        self.assertEqual(result.bitmap, [0, 0, 0, 0])
        self.assertIsNone(result.mismatch_kind)
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_multiple_outliers_with_majority(self, mock_all_gather):
        """Two ranks lag and three agree — both lagging ranks are flagged."""
        op_ids = [10, 3, 10, 7, 10]
        self.tracker._op_id = 10
        self.tracker._write()

        daemon = self._make_daemon(op_ids, local_op_id=10)

        def fake_all_gather(gathered, local_tensor, group=None):
            for i, t in enumerate(gathered):
                t.fill_(op_ids[i])

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertEqual(result.bitmap, [0, 1, 0, 1, 0])
        self.assertEqual(result.culprit_rank, 1)
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_progress_without_majority_stays_group_scoped(self, mock_all_gather):
        op_ids = [10, 10, 7, 7]
        daemon = self._make_daemon(op_ids, local_op_id=10)

        def fake_all_gather(gathered, local_tensor, group=None):
            for index, tensor in enumerate(gathered):
                tensor[0] = op_ids[index]
                tensor[1] = 0
                tensor[2] = 0

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertIsNone(result.culprit_rank)
        self.assertEqual(result.culprit_ranks, [])
        self.assertEqual(result.bitmap, [0, 0, 0, 0])
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_is_hang_flag_when_stalled(self, mock_all_gather):
        """is_hang is True when stall exceeds threshold."""
        op_ids = [10, 10, 5, 10]
        self.tracker._op_id = 10
        self.tracker._write()

        daemon = self._make_daemon(op_ids, local_op_id=10)
        # Simulate stall by backdating the last change time
        daemon._last_change_time = time.time() - 1.0

        def fake_all_gather(gathered, local_tensor, group=None):
            for i, t in enumerate(gathered):
                t.fill_(op_ids[i])

        mock_all_gather.side_effect = fake_all_gather
        result = daemon.check_once()

        self.assertTrue(result.is_hang)
        self.assertGreater(result.stall_duration_s, 0.9)
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_data_loading_latency_localizes_rank(self, mock_all_gather):
        daemon = self._make_daemon([10, 10, 10, 10], local_op_id=10)
        key = 123
        sequence = 2
        rows = [[10, 0, 0, 0, 0, 0, 0, 1, key, sequence, 10_000_000] for _ in range(4)]
        rows[2] = [
            10,
            0,
            0,
            int(DiagnosticStage.DATA_LOADING),
            key,
            sequence,
            6_000_000_000,
            0,
            0,
            0,
            0,
        ]

        def fake_all_gather(gathered, local_tensor, group=None):
            for tensor, values in zip(gathered, rows):
                tensor.copy_(local_tensor.new_tensor(values))

        mock_all_gather.side_effect = fake_all_gather
        daemon._c3.run_scalar.side_effect = [
            _c3_result([10, 10, 10, 10]),
            _c3_result([10.0, 10.0, 6000.0, 10.0], C3Mode.STATISTICAL),
        ]

        result = daemon.check_once()

        self.assertTrue(result.is_dataloader_straggler)
        self.assertTrue(result.dataloader_active)
        self.assertEqual(result.culprit_rank, 2)
        self.assertEqual(result.mismatch_kind, "dataloader_latency")
        self.assertEqual(result.dataloader_culprit_ranks, [2])
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_checkpoint_write_latency_localizes_rank(self, mock_all_gather):
        daemon = self._make_daemon([10, 10, 10, 10], local_op_id=10)
        daemon._checkpoint_io_latency_threshold_ms = 5_000.0
        key = 456
        sequence = 3
        rows = [
            [
                10,
                0,
                0,
                0,
                0,
                0,
                0,
                int(DiagnosticStage.CHECKPOINT_WRITE),
                key,
                sequence,
                10_000_000,
            ]
            for _ in range(4)
        ]
        rows[1] = [
            10,
            0,
            0,
            int(DiagnosticStage.CHECKPOINT_WRITE),
            key,
            sequence,
            6_000_000_000,
            0,
            0,
            0,
            0,
        ]

        def fake_all_gather(gathered, local_tensor, group=None):
            for tensor, values in zip(gathered, rows):
                tensor.copy_(local_tensor.new_tensor(values))

        mock_all_gather.side_effect = fake_all_gather
        daemon._c3.run_scalar.side_effect = [
            _c3_result([10, 10, 10, 10]),
            _c3_result([10.0, 6000.0, 10.0, 10.0], C3Mode.STATISTICAL),
        ]

        result = daemon.check_once()

        self.assertTrue(result.is_stage_straggler)
        self.assertTrue(result.stage_active)
        self.assertEqual(result.stage_kind, "checkpoint_write")
        self.assertEqual(result.culprit_rank, 1)
        self.assertEqual(result.mismatch_kind, "checkpoint_write_latency")
        self.assertEqual(result.stage_culprit_ranks, [1])
        self.assertFalse(result.dataloader_active)
        daemon.close()

    @patch("lm_resiliency.detection.hang_detector.dist.all_gather")
    def test_other_rank_dataloader_does_not_label_progress_culprit(self, mock_all_gather):
        daemon = self._make_daemon([10, 10, 5, 10], local_op_id=10)
        rows = [[10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(4)]
        rows[1][3:7] = [int(DiagnosticStage.DATA_LOADING), 99, 1, 6_000_000_000]
        rows[2][0] = 5

        def fake_all_gather(gathered, local_tensor, group=None):
            for tensor, values in zip(gathered, rows):
                tensor.copy_(local_tensor.new_tensor(values))

        mock_all_gather.side_effect = fake_all_gather
        daemon._c3.run_scalar.side_effect = [
            _c3_result([10, 10, 5, 10]),
            _c3_result([0.0, 6000.0, 0.0, 0.0], C3Mode.STATISTICAL),
        ]

        result = daemon.check_once()

        self.assertEqual(result.culprit_rank, 2)
        self.assertEqual(result.mismatch_kind, "progress")
        self.assertFalse(result.dataloader_active)
        daemon.close()


class TestHangDetectorProgressObservation(unittest.TestCase):
    """Tests progress state read after an event or timeout."""

    def setUp(self):
        self.tracker = OpTracker(rank=51)

    def tearDown(self):
        self.tracker.close()

    def _make_daemon(self):
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=51,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=0.05,
                confirmation_interval_s=0.01,
            )

        daemon._reader.close()
        daemon._reader = OpTrackerReader(rank=51)
        daemon._reader._shm = self.tracker._shm
        daemon._last_op_id = self.tracker.op_id
        daemon._last_step = self.tracker.step
        daemon._last_change_time = time.time()
        return daemon

    def test_observation_has_no_stall_when_progress_advanced(self):
        daemon = self._make_daemon()

        self.tracker.advance()
        stall = daemon._observe_stall()
        self.assertEqual(stall, 0.0)

        self.tracker.advance()
        stall = daemon._observe_stall()
        self.assertEqual(stall, 0.0)

        daemon.close()

    def test_observation_reports_idle_duration(self):
        daemon = self._make_daemon()

        time.sleep(0.06)
        stall = daemon._observe_stall()
        self.assertGreaterEqual(stall, 0.05)

        daemon.close()

    def test_observation_resets_after_progress(self):
        daemon = self._make_daemon()

        time.sleep(0.06)
        stall = daemon._observe_stall()
        self.assertGreater(stall, 0.0)

        self.tracker.advance()
        stall = daemon._observe_stall()
        self.assertEqual(stall, 0.0)

        daemon.close()


class TestHangDetectorRunLoop(unittest.TestCase):
    """Tests the daemon's run() loop behavior."""

    def setUp(self):
        self.tracker = OpTracker(rank=52)

    def tearDown(self):
        self.tracker.close()

    def test_run_triggers_check_on_stall(self):
        """run() calls check_once when stall exceeds threshold."""
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=52,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=0.03,
                confirmation_interval_s=0.01,
            )

        daemon._reader.close()
        daemon._reader = OpTrackerReader(rank=52)
        daemon._reader._shm = self.tracker._shm
        daemon._last_op_id = self.tracker.op_id
        daemon._last_step = self.tracker.step
        daemon._last_change_time = time.time()
        daemon._running = True

        with patch.object(daemon, "start", return_value=True):
            with patch.object(daemon, "check_once") as mock_check:
                mock_check.return_value = HangLocalizationResult(
                    is_hang=True,
                    culprit_rank=None,
                    bitmap=[0, 0, 0, 0],
                    op_ids=[0, 0, 0, 0],
                    local_op_id=0,
                    stall_duration_s=0.05,
                )
                daemon.run(max_rounds=1)

        mock_check.assert_called_once()
        daemon.close()

    def test_progress_event_resets_timeout_without_c3(self):
        """Healthy local progress does not produce periodic C3 traffic."""
        from threading import Event, Thread

        progress_event = Event()
        self.tracker._progress_event = progress_event
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=52,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=1.0,
                confirmation_interval_s=0.01,
                progress_event=progress_event,
            )

        def publish_progress():
            for _ in range(3):
                time.sleep(0.01)
                self.tracker.advance(force_signal=True)
            daemon.stop()

        publisher = Thread(target=publish_progress)
        publisher.start()
        with (
            patch.object(daemon, "start", return_value=True),
            patch.object(daemon, "check_once") as mock_check,
        ):
            daemon.run()
        publisher.join()

        mock_check.assert_not_called()
        daemon.close()

    def test_dataloader_threshold_triggers_consensus_before_hard_hang(self):
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=52,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=30.0,
                dataloader_latency_threshold_s=5.0,
            )

        self.assertEqual(daemon._consensus_trigger_threshold_s, 5.0)
        daemon.close()

    def test_run_respects_max_rounds(self):
        """run() exits after max_rounds detection rounds."""
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=52,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=0.02,
                confirmation_interval_s=0.005,
            )

        daemon._reader.close()
        daemon._reader = OpTrackerReader(rank=52)
        daemon._reader._shm = self.tracker._shm
        daemon._last_op_id = self.tracker.op_id
        daemon._last_step = self.tracker.step
        daemon._last_change_time = time.time()

        call_count = 0

        def fake_check():
            nonlocal call_count
            call_count += 1
            daemon._last_change_time = time.time()
            return HangLocalizationResult(
                is_hang=True,
                culprit_rank=None,
                bitmap=[0, 0, 0, 0],
                op_ids=[0, 0, 0, 0],
                local_op_id=0,
                stall_duration_s=0.03,
            )

        with patch.object(daemon, "start", return_value=True):
            daemon.check_once = fake_check
            daemon.run(max_rounds=3)

        self.assertEqual(call_count, 3)
        daemon.close()

    def test_stop_terminates_loop(self):
        """stop() causes run() to exit."""
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=52,
                world_size=4,
                group=MagicMock(),
                stall_threshold_s=100,  # Very high — won't trigger naturally
                confirmation_interval_s=0.01,
            )

        daemon._reader.close()
        daemon._reader = OpTrackerReader(rank=52)
        daemon._reader._shm = self.tracker._shm
        daemon._last_op_id = self.tracker.op_id
        daemon._last_step = self.tracker.step
        daemon._last_change_time = time.time()

        import threading

        def stop_after_delay():
            time.sleep(0.05)
            daemon.stop()

        t = threading.Thread(target=stop_after_delay)
        t.start()

        with patch.object(daemon, "start", return_value=True):
            daemon.run()  # Should exit when stop() is called

        t.join()
        daemon.close()


class TestHangDetectorStart(unittest.TestCase):
    """Tests daemon attach/start behavior."""

    def test_start_returns_false_when_no_shm(self):
        with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
            MockC3Class.return_value = MagicMock()
            daemon = HangDetectionDaemon(
                rank=9999,
                world_size=4,
                group=MagicMock(),
            )

        self.assertFalse(daemon.start())
        daemon.close()

    def test_start_returns_true_when_tracker_exists(self):
        tracker = OpTracker(rank=53)
        try:
            with patch("lm_resiliency.detection.hang_detector.C3") as MockC3Class:
                MockC3Class.return_value = MagicMock()
                daemon = HangDetectionDaemon(
                    rank=53,
                    world_size=4,
                    group=MagicMock(),
                )

            self.assertTrue(daemon.start())
            daemon.close()
        finally:
            tracker.close()


if __name__ == "__main__":
    unittest.main()
