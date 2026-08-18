"""Tests for the GEMINI checkpoint engine with mocked distributed state."""

import json
import pickle
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.disk import (
    CheckpointStatus,
    CheckpointStatusStore,
    shard_checksums,
)
from lm_resiliency.checkpointing.manager import (
    InMemoryCheckpointManager,
    RecoveryMode,
    _dumps_meta,
    _loads_meta,
)
from lm_resiliency.checkpointing.state_dict import TensorEntry


@pytest.fixture
def tmp_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


def _make_state_dict():
    return {
        "model": {"weight": torch.randn(4, 4), "bias": torch.randn(4)},
        "optimizer": {"exp_avg": torch.zeros(4, 4)},
        "step": 10,
    }


def test_checkpoint_status_store_is_rank_scoped(tmp_dir):
    rank_zero = CheckpointStatusStore(tmp_dir, rank=0)
    rank_one = CheckpointStatusStore(tmp_dir, rank=1)
    zero_status = CheckpointStatus(candidate_step=10)
    one_status = CheckpointStatus(candidate_step=20, recovery_verified_step=10)

    rank_zero.write(zero_status)
    rank_one.write(one_status)

    assert rank_zero.read() == zero_status
    assert rank_one.read() == one_status
    assert rank_zero.path.name == "GEMINI_CHECKPOINT_STATUS.rank-0"
    assert rank_one.path.name == "GEMINI_CHECKPOINT_STATUS.rank-1"


def test_checkpoint_status_round_trips_verified_history(tmp_dir):
    store = CheckpointStatusStore(tmp_dir, rank=0)
    status = CheckpointStatus(
        candidate_step=30,
        recovery_verified_step=20,
        recovery_verified_steps=(10,),
    )

    store.write(status)

    assert store.read() == CheckpointStatus(
        candidate_step=30,
        recovery_verified_step=20,
        recovery_verified_steps=(10, 20),
    )
    assert store.read().is_recovery_verified(10)
    legacy = status.to_dict()
    legacy.pop("recovery_verified_steps")
    assert CheckpointStatus.from_dict(legacy) == CheckpointStatus(
        candidate_step=30,
        recovery_verified_step=20,
    )


def test_checkpoint_status_retries_transient_partial_json(tmp_dir):
    store = CheckpointStatusStore(tmp_dir, rank=0, run_id="run-a", topology_id="topology-a")
    status = CheckpointStatus(candidate_step=10, recovery_verified_step=5)
    store.write(status)
    complete = store.path.read_bytes()

    with (
        patch.object(
            Path,
            "read_bytes",
            autospec=True,
            side_effect=[b'{"schema_version":2,"run_id":"run-a', complete],
        ),
        patch("lm_resiliency.checkpointing.disk.time.sleep"),
    ):
        assert store.read() == status


def test_checkpoint_status_persistently_malformed_json_fails_closed(tmp_dir, monkeypatch):
    store = CheckpointStatusStore(tmp_dir, rank=0)
    store.path.write_text('{"schema_version":', encoding="utf-8")
    monkeypatch.setattr(
        "lm_resiliency.checkpointing.disk._STATUS_READ_RETRY_SECONDS",
        0.0,
    )

    with pytest.raises(json.JSONDecodeError):
        store.read()


def test_checkpoint_status_rejects_another_run_or_topology(tmp_dir):
    writer = CheckpointStatusStore(tmp_dir, rank=0, run_id="run-a", topology_id="topology-a")
    writer.write(CheckpointStatus(candidate_step=10))

    assert (
        CheckpointStatusStore(tmp_dir, rank=0, run_id="run-b", topology_id="topology-a").read()
        == CheckpointStatus()
    )
    assert (
        CheckpointStatusStore(tmp_dir, rank=0, run_id="run-a", topology_id="topology-b").read()
        == CheckpointStatus()
    )


def test_rank_scoped_status_store_reads_legacy_status(tmp_dir):
    legacy = CheckpointStatusStore(tmp_dir)
    status = CheckpointStatus(candidate_step=10, recovery_verified_step=5)
    legacy.write(status)

    assert CheckpointStatusStore(tmp_dir, rank=7).read() == status


def test_missing_rank_status_uses_conservative_peer_verified_step(tmp_dir):
    CheckpointStatusStore(tmp_dir, rank=0).write(
        CheckpointStatus(candidate_step=20, recovery_verified_step=10)
    )
    CheckpointStatusStore(tmp_dir, rank=1).write(
        CheckpointStatus(candidate_step=15, recovery_verified_step=5)
    )

    recovered = CheckpointStatusStore(tmp_dir, rank=7).read()

    assert recovered.candidate_step == -1
    assert recovered.recovery_verified_step == 5
    assert recovered.recovery_mode == RecoveryMode.RECOVERY_VERIFIED.value


def test_replication_metadata_preserves_peer_tensor_entries():
    entries = [
        TensorEntry(
            key_path=("optimizer", "state"),
            shape=torch.Size([7]),
            dtype=torch.float32,
            device=torch.device("cuda", 0),
        )
    ]

    non_tensor, loaded_entries = _loads_meta(_dumps_meta({"cursor": 3}, entries))

    assert non_tensor == {"cursor": 3}
    assert loaded_entries == entries
    assert _loads_meta(pickle.dumps({"legacy": True})) == ({"legacy": True}, None)


@patch("torch.distributed.is_initialized", return_value=False)
def test_disabled(mock_dist):
    config = InMemoryCkptConfig(enable=False)
    mgr = InMemoryCheckpointManager(config)
    mgr.save(_make_state_dict(), step=1)
    mgr.maybe_wait()
    assert mgr.find_latest() == -1
    assert mgr.load() is None
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_save_and_load_single_rank(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state = _make_state_dict()
    mgr.save(state, step=10)
    mgr.maybe_wait()

    # Should find step 10 in local buffer
    latest = mgr.find_latest()
    assert latest == 10

    result = mgr.load()
    assert result is not None
    loaded_state, step = result
    assert step == 10
    assert torch.equal(loaded_state["model"]["weight"], state["model"]["weight"])
    assert loaded_state["step"] == 10

    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_multiple_saves_rotation(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state1 = {"w": torch.randn(2, 2), "s": 1}
    state2 = {"w": torch.randn(2, 2), "s": 2}

    mgr.save(state1, step=10)
    mgr.maybe_wait()

    mgr.save(state2, step=20)
    mgr.maybe_wait()

    latest = mgr.find_latest()
    assert latest == 20

    result = mgr.load()
    assert result is not None
    assert result[1] == 20
    assert torch.equal(result[0]["w"], state2["w"])

    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_save_tensors_loads_from_memory(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    tensor = torch.randn(3, 3)
    mgr.save_tensors([tensor], step=3, extra={"cursor": 11})
    mgr.maybe_wait()

    result = mgr.load_tensors()
    assert result is not None
    tensors, step, extra = result
    assert step == 3
    assert torch.equal(tensors[0], tensor)
    assert extra == {"cursor": 11}

    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_cycle_boundaries_track_candidate_and_recovery_verified(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    mgr.save({"w": torch.full((1,), 10.0)}, step=10)
    mgr.persist_cycle_boundary(10)
    first = mgr.checkpoint_status
    assert first.candidate_step == 10
    assert first.recovery_verified_step == -1

    mgr.save({"w": torch.full((1,), 20.0)}, step=20)
    mgr.persist_cycle_boundary(20)
    second = mgr.checkpoint_status
    assert second.candidate_step == 20
    assert second.recovery_verified_step == 10

    mgr.save({"w": torch.full((1,), 23.0)}, step=23)
    latest = mgr.load(mode=RecoveryMode.LATEST_GEMINI)
    verified = mgr.load(mode=RecoveryMode.RECOVERY_VERIFIED)

    assert latest is not None and latest[1] == 23
    assert verified is not None and verified[1] == 10
    assert torch.equal(verified[0]["w"], torch.full((1,), 10.0))
    assert mgr.checkpoint_status.candidate_step == -1
    assert mgr.checkpoint_status.recovery_mode == RecoveryMode.RECOVERY_VERIFIED.value
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_verified_boundary_immediately_marks_dense_checkpoint_recoverable(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((1,), 10.0)}, step=10)

    status = mgr.persist_verified_boundary(10)

    assert status.candidate_step == -1
    assert status.recovery_verified_step == 10
    assert status.recovery_mode == RecoveryMode.LATEST_GEMINI.value
    with patch.object(
        mgr,
        "_collective_min_step",
        side_effect=AssertionError("local lookup must not use collectives"),
    ):
        assert mgr.local_recovery_step(RecoveryMode.RECOVERY_VERIFIED) == 10
    recovered = mgr.load(mode=RecoveryMode.RECOVERY_VERIFIED)
    assert recovered is not None and recovered[1] == 10
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_exact_recovery_keeps_older_verified_generation_eligible(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((1,), 10.0)}, step=10)
    mgr.persist_verified_boundary(10)
    mgr.save({"w": torch.full((1,), 20.0)}, step=20)
    status = mgr.persist_verified_boundary(20)

    assert status.recovery_verified_step == 20
    assert status.recovery_verified_steps == (10, 20)
    recovered = mgr.load(mode=RecoveryMode.RECOVERY_VERIFIED, step=10)

    assert recovered is not None
    assert recovered[1] == 10
    assert torch.equal(recovered[0]["w"], torch.full((1,), 10.0))
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_sdc_rejection_persists_verified_recovery_mode(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((1,), 10.0)}, step=10)
    mgr.persist_cycle_boundary(10)
    mgr.save({"w": torch.full((1,), 20.0)}, step=20)
    mgr.persist_cycle_boundary(20)
    mgr.save({"w": torch.full((1,), 23.0)}, step=23)

    mgr.reject_candidate()

    assert mgr.checkpoint_status.candidate_step == -1
    assert mgr.checkpoint_status.recovery_verified_step == 10
    assert mgr.checkpoint_status.recovery_mode == RecoveryMode.RECOVERY_VERIFIED.value
    recovered = mgr.load()
    assert recovered is not None and recovered[1] == 10
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_copy_to_writes_status_for_local_and_peer_shards(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((1,), 10.0)}, step=10)
    status = mgr.persist_cycle_boundary(10)
    mgr._replicator = MagicMock(enabled=False, peer_rank=7)

    with tempfile.TemporaryDirectory() as destination:
        mgr.copy_to(destination)

        namespaced = mgr._disk.namespace_folder(destination, mgr._run_id, mgr._topology_id)

        assert (
            CheckpointStatusStore(
                namespaced, rank=0, run_id=mgr._run_id, topology_id=mgr._topology_id
            ).read()
            == status
        )
        assert (
            CheckpointStatusStore(
                namespaced, rank=7, run_id=mgr._run_id, topology_id=mgr._topology_id
            ).read()
            == status
        )

    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_replication_starts_immediately_after_host_copy(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=4,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    replicator = MagicMock()
    replicator.enabled = True
    replicator.in_flight = False
    replicator.peer_rank = 1
    replicator.wait.return_value = (-1, b"")
    replication_started = threading.Event()
    replicator.start_replication.side_effect = lambda **kwargs: replication_started.set()
    mgr._replicator = replicator

    mgr.save({"w": torch.full((2,), 2.0)}, step=2)

    assert replication_started.wait(timeout=1.0)
    assert replicator.start_replication.call_args.kwargs["send_step"] == 2
    assert mgr.find_latest() == 2
    assert mgr.load()[1] == 2
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_completed_replication_is_committed_before_receive_slot_rotation(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=2,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    replicator = MagicMock()
    replicator.enabled = True
    replicator.in_flight = False
    replicator.peer_rank = 1
    replicator.wait.side_effect = [(2, b""), (4, b""), (6, b"")]
    mgr._replicator = replicator

    mgr.save({"w": torch.full((2,), 2.0)}, step=2)
    mgr.maybe_wait()
    mgr.save({"w": torch.full((2,), 4.0)}, step=4)
    mgr.maybe_wait()
    mgr.save({"w": torch.full((2,), 6.0)}, step=6)
    mgr.maybe_wait()

    assert mgr._buffer_pool.peer_previous.step == 4
    assert mgr._buffer_pool.peer_previous.state.name == "READY"
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_failed_replication_preserves_previous_aligned_recovery_pair(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=2,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    replicator = MagicMock()
    replicator.enabled = True
    replicator.in_flight = False
    replicator.peer_rank = 1
    replicator.wait.side_effect = [(2, b""), (-1, b""), (6, b"")]
    mgr._replicator = replicator

    mgr.save({"w": torch.full((2,), 2.0)}, step=2)
    mgr.maybe_wait()
    mgr.save({"w": torch.full((2,), 4.0)}, step=4)
    mgr.maybe_wait()
    mgr.save({"w": torch.full((2,), 6.0)}, step=6)
    mgr.maybe_wait()

    assert mgr._buffer_pool.own_previous.step == 2
    assert mgr._buffer_pool.peer_previous.step == 2
    assert mgr._buffer_pool.own_current.step == 6
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_disk_flush_and_load(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=10,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state = {"w": torch.randn(3, 3), "info": "test"}

    # Save at step 5 (won't trigger disk flush)
    mgr.save(state, step=5)
    mgr.maybe_wait()

    # Save at step 10 (triggers disk flush of own_previous = step 5)
    mgr.save(state, step=10)
    mgr.maybe_wait()

    # Wait for disk flush
    mgr._disk.wait()

    # Verify disk has a checkpoint
    disk_step = mgr._disk.find_latest_on_disk()
    assert disk_step == 5

    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_periodic_disk_flush_uses_previous_slot_metadata(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=10,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state5 = {"w": torch.full((2,), 5.0), "step": 5, "extra": {"cursor": 5}}
    state10 = {"w": torch.full((2,), 10.0), "step": 10, "extra": {"cursor": 10}}

    mgr.save(state5, step=5)
    mgr.maybe_wait()
    mgr.save(state10, step=10)
    mgr.maybe_wait()
    mgr._disk.wait()

    fresh_mgr = InMemoryCheckpointManager(config)
    recovered = fresh_mgr.load()
    assert recovered is not None
    loaded, step = recovered
    assert step == 5
    assert torch.equal(loaded["w"], state5["w"])
    assert loaded["step"] == 5
    assert loaded["extra"]["cursor"] == 5

    fresh_mgr.close()
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_explicit_disk_flush_uses_requested_slot_metadata(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state7 = {"w": torch.full((2,), 7.0), "step": 7}
    state9 = {"w": torch.full((2,), 9.0), "step": 9}

    mgr.save(state7, step=7)
    mgr.maybe_wait()
    mgr.save(state9, step=9)
    mgr.maybe_wait()

    mgr.flush_to_disk(step=7)
    mgr._disk.wait()

    fresh_mgr = InMemoryCheckpointManager(config)
    recovered = fresh_mgr.load()
    assert recovered is not None
    loaded, step = recovered
    assert step == 7
    assert torch.equal(loaded["w"], state7["w"])
    assert loaded["step"] == 7

    fresh_mgr.close()
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_hsdp_skips_replication(mock_dist, tmp_dir):
    class FakeParallelDims:
        dp_replicate = 2
        dp_shard = 4

    config = InMemoryCkptConfig(
        enable=True,
        skip_replication_if_hsdp=True,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config, parallel_dims=FakeParallelDims())
    assert mgr._skip_replication is True
    assert mgr._replicator.enabled is False
    # Buffer pool should be 2-slot mode
    assert mgr._buffer_pool._num_slots == 2
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_flush_to_disk_explicit(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,  # auto-flush disabled
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)

    state = {"w": torch.randn(2, 2)}
    mgr.save(state, step=7)
    mgr.maybe_wait()

    mgr.flush_to_disk(step=7)
    mgr._disk.wait()

    assert mgr._disk.find_latest_on_disk() == 7
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_uncaught_exit_flushes_latest_checkpoint(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((2,), 7.0)}, step=7)

    assert mgr._exit_flush_registered
    mgr._flush_on_exit()

    assert not mgr._exit_flush_registered
    assert mgr._disk.find_latest_on_disk() == 7
    mgr.close()


@patch("torch.distributed.is_initialized", return_value=False)
def test_clean_close_disarms_uncaught_exit_flush(mock_dist, tmp_dir):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=tmp_dir,
    )
    mgr = InMemoryCheckpointManager(config)
    mgr.save({"w": torch.full((2,), 9.0)}, step=9)

    assert mgr._exit_flush_registered
    mgr.close()
    assert not mgr._exit_flush_registered
    assert mgr._disk.find_latest_on_disk() == -1


@patch(
    "lm_resiliency.checkpointing.disk.ThreadPoolExecutor",
    side_effect=RuntimeError("cannot schedule new futures after interpreter shutdown"),
)
def test_checksum_falls_back_to_serial_during_interpreter_shutdown(mock_executor):
    tensor = torch.arange(16, dtype=torch.int64)

    expected = shard_checksums([tensor], chunk_size=16, workers=1)
    observed = shard_checksums([tensor], chunk_size=16, workers=2)

    assert observed == expected
