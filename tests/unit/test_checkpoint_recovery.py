"""Core GEMINI recovery and integrity tests."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from lm_resiliency.api import enable_resiliency
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.disk import (
    CheckpointFormatError,
    ChecksumMismatch,
    DiskSerializer,
    atomic_copy_file,
    shard_checksums,
)
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager
from lm_resiliency.checkpointing.state_dict import FlatStateDictMetadata, flatten, unflatten
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import replay_result_has_sdc


def test_flush_for_restart_own_shard_round_trip(tmp_path):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(tmp_path),
    )
    manager = InMemoryCheckpointManager(config)
    state = {"model": {"w": torch.randn(4, 4)}, "optimizer": {"m": torch.randn(4, 4)}}
    manager.save(state, step=7)
    manager.maybe_wait()

    assert manager.flush_for_restart() == 7
    manager.close()

    reloaded_manager = InMemoryCheckpointManager(config)
    reloaded = reloaded_manager.load()
    assert reloaded is not None
    recovered, step = reloaded
    assert step == 7
    assert torch.allclose(recovered["model"]["w"], state["model"]["w"])
    reloaded_manager.close()


def test_flush_for_restart_disarms_exit_flush_until_next_save(tmp_path):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(tmp_path),
    )
    manager = InMemoryCheckpointManager(config)
    manager.save({"model": {"w": torch.ones(4)}}, step=1)
    manager.maybe_wait()

    assert manager._exit_flush_registered
    assert manager.flush_for_restart() == 1
    assert not manager._exit_flush_registered

    manager.save({"model": {"w": torch.full((4,), 2.0)}}, step=2)
    assert manager._exit_flush_registered
    manager.close()


def test_gemini_is_single_tier(tmp_path):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(tmp_path),
    )
    manager = InMemoryCheckpointManager(config)
    assert manager.load() is None
    assert manager.find_latest() == -1
    assert not hasattr(manager, "_global_disk")
    assert not hasattr(config, "global_disk_folder")
    manager.close()


def test_shard_checksums_detect_corruption():
    a = torch.randn(4096, dtype=torch.bfloat16)
    b = torch.randn(2048, dtype=torch.float32)
    baseline = shard_checksums([a, b])
    assert shard_checksums([a, b]) == baseline
    a[7] += 1.0
    assert shard_checksums([a, b]) != baseline


def test_disk_integrity_round_trip_and_corruption(tmp_path):
    metadata, tensors = flatten(
        {"w": torch.randn(500, dtype=torch.bfloat16), "m": torch.randn(500)}
    )
    serializer = DiskSerializer(str(tmp_path), rank=0, integrity=True)
    path = serializer.save_sync(metadata, tensors, step=3)
    serializer.load(3)

    raw = torch.load(path, weights_only=True)
    raw["tensors"][0][0] += 7.0
    torch.save(raw, path)
    with pytest.raises(ChecksumMismatch):
        serializer.load(3)


def test_disk_save_does_not_publish_partial_replacement(tmp_path):
    serializer = DiskSerializer(str(tmp_path), rank=0)
    metadata, tensors = flatten({"w": torch.ones(2)})
    path = serializer.save_sync(metadata, tensors, step=3)

    def write_partial_then_fail(_payload, handle):
        handle.write(b"partial checkpoint")
        raise OSError("disk full")

    replacement_metadata, replacement_tensors = flatten({"w": torch.full((2,), 9.0)})
    with (
        patch("lm_resiliency.checkpointing.disk.torch.save", side_effect=write_partial_then_fail),
        pytest.raises(OSError, match="disk full"),
    ):
        serializer.save_sync(replacement_metadata, replacement_tensors, step=3)

    loaded_metadata, loaded_tensors = serializer.load(3)
    assert torch.equal(unflatten(loaded_metadata, loaded_tensors)["w"], torch.ones(2))
    assert not list(path.parent.glob("*.tmp"))
    assert not list(path.parent.glob(".*.tmp"))


def test_disk_save_cleans_temp_file_from_dead_writer(tmp_path):
    step_dir = tmp_path / "step-3"
    step_dir.mkdir()
    stale = step_dir / ".rank-0.pt.pid-999999999.stale.tmp"
    stale.write_bytes(b"partial checkpoint")
    metadata, tensors = flatten({"w": torch.ones(2)})

    DiskSerializer(str(tmp_path), rank=0).save_sync(metadata, tensors, step=3)

    assert not stale.exists()


def test_atomic_copy_does_not_publish_partial_replacement(tmp_path):
    metadata, tensors = flatten({"w": torch.ones(2)})
    source = DiskSerializer(str(tmp_path / "source"), rank=0).save_sync(metadata, tensors, step=3)
    destination_serializer = DiskSerializer(str(tmp_path / "destination"), rank=0)
    destination = destination_serializer.save_sync(*flatten({"w": torch.full((2,), 7.0)}), step=3)

    def copy_partial_then_fail(_source, temporary):
        Path(temporary).write_bytes(b"partial checkpoint")
        raise OSError("copy interrupted")

    with (
        patch(
            "lm_resiliency.checkpointing.disk.shutil.copy2",
            side_effect=copy_partial_then_fail,
        ),
        pytest.raises(OSError, match="copy interrupted"),
    ):
        atomic_copy_file(source, destination)

    loaded_metadata, loaded_tensors = destination_serializer.load(3)
    assert torch.equal(unflatten(loaded_metadata, loaded_tensors)["w"], torch.full((2,), 7.0))
    assert not list(destination.parent.glob(".*.tmp"))


def test_disk_integrity_rejects_checkpoint_without_checksums(tmp_path):
    metadata, tensors = flatten({"w": torch.arange(4, dtype=torch.float32)})
    DiskSerializer(str(tmp_path), rank=0, integrity=False).save_sync(metadata, tensors, step=3)

    with pytest.raises(ChecksumMismatch, match="does not contain checksums"):
        DiskSerializer(str(tmp_path), rank=0, integrity=True).load(3)


def test_disk_without_integrity_accepts_checkpoint_without_checksums(tmp_path):
    metadata, tensors = flatten({"w": torch.arange(4, dtype=torch.float32)})
    serializer = DiskSerializer(str(tmp_path), rank=0, integrity=False)
    serializer.save_sync(metadata, tensors, step=3)

    loaded_metadata, loaded_tensors = serializer.load(3)

    assert loaded_metadata == metadata
    assert torch.equal(loaded_tensors[0], tensors[0])


def test_gemini_falls_back_when_integrity_is_enabled_for_unsigned_checkpoint(tmp_path):
    metadata, tensors = flatten({"w": torch.arange(4, dtype=torch.float32)})
    DiskSerializer(str(tmp_path), rank=0, integrity=False).save_sync(metadata, tensors, step=3)
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            disk_folder=str(tmp_path),
            disk_flush_interval=0,
            verify_integrity=True,
        )
    )

    assert manager.find_latest() == -1
    assert manager.load() is None
    manager.close()


def test_gemini_load_returns_none_on_corruption(tmp_path):
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(tmp_path),
        verify_integrity=True,
    )
    manager = InMemoryCheckpointManager(config)
    manager.save({"model": {"w": torch.randn(64)}}, step=4)
    manager.maybe_wait()
    manager.flush_for_restart()
    assert manager.load() is not None
    manager.close()

    rank_file = next(tmp_path.glob("step-*/rank-0.pt"))
    raw = torch.load(rank_file, weights_only=True)
    raw["tensors"][0][0] += 9.0
    torch.save(raw, rank_file)

    corrupted_manager = InMemoryCheckpointManager(config)
    assert corrupted_manager.load() is None
    corrupted_manager.close()


def test_disk_safe_format_round_trips_schema_constrained_metadata(tmp_path):
    state = {
        "tensor": torch.arange(4, dtype=torch.float32),
        "python_rng": (3, (1, 2, 3), None),
        "numpy_rng": np.arange(8, dtype=np.uint32),
        "types": [torch.Size([2, 3]), torch.float16, torch.device("cpu")],
        "bytes": b"checkpoint",
    }
    metadata, tensors = flatten(state)
    serializer = DiskSerializer(str(tmp_path), rank=0)

    path = serializer.save_sync(metadata, tensors, step=6)
    raw = torch.load(path, weights_only=True)
    assert raw["format"] == "lm-resiliency.gemini.node-local"
    assert raw["version"] == 2
    assert isinstance(raw["metadata_json"], str)

    loaded_metadata, loaded_tensors = serializer.load(6)
    restored = unflatten(loaded_metadata, loaded_tensors)
    assert torch.equal(restored["tensor"], state["tensor"])
    assert restored["python_rng"] == state["python_rng"]
    assert np.array_equal(restored["numpy_rng"], state["numpy_rng"])
    assert restored["types"] == state["types"]
    assert restored["bytes"] == state["bytes"]


def test_disk_safe_format_round_trips_tensor_valued_extra(tmp_path):
    rng_state = torch.get_rng_state()
    metadata = FlatStateDictMetadata(non_tensor_data={"extra": {"rng": rng_state}})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    serializer.save_sync(metadata, [], step=7)
    loaded_metadata, loaded_tensors = serializer.load(7)

    assert loaded_tensors == []
    assert torch.equal(loaded_metadata.non_tensor_data["extra"]["rng"], rng_state)


def test_disk_load_rejects_pickle_payload_without_executing_it(tmp_path):
    marker = tmp_path / "pickle-executed"
    path = tmp_path / "step-8" / "rank-0.pt"
    path.parent.mkdir(parents=True)

    class Malicious:
        def __reduce__(self):
            return os.system, (f"touch {marker}",)

    torch.save({"metadata": Malicious()}, path)
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(CheckpointFormatError, match="unsafe, malformed, or legacy"):
        serializer.load(8)
    assert not marker.exists()


def test_disk_load_rejects_malformed_schema_before_recovery(tmp_path):
    metadata, tensors = flatten({"w": torch.ones(2)})
    serializer = DiskSerializer(str(tmp_path), rank=0)
    path = serializer.save_sync(metadata, tensors, step=9)
    raw = torch.load(path, weights_only=True)
    document = json.loads(raw["metadata_json"])
    document["tensor_entries"][0]["dtype"] = "not_a_dtype"
    raw["metadata_json"] = json.dumps(document)
    torch.save(raw, path)

    with pytest.raises(CheckpointFormatError, match="dtype is unsupported"):
        serializer.load(9)


@pytest.mark.parametrize("key_path", [[], ["missing"]])
def test_disk_load_rejects_tensor_paths_that_do_not_match_skeleton(tmp_path, key_path):
    metadata, tensors = flatten({"w": torch.ones(2)})
    serializer = DiskSerializer(str(tmp_path), rank=0)
    path = serializer.save_sync(metadata, tensors, step=10)
    raw = torch.load(path, weights_only=True)
    document = json.loads(raw["metadata_json"])
    document["tensor_entries"][0]["key_path"] = key_path
    raw["metadata_json"] = json.dumps(document)
    torch.save(raw, path)

    with pytest.raises(CheckpointFormatError, match="key_path"):
        serializer.load(10)


def test_disk_save_rejects_unsupported_metadata_instead_of_using_pickle(tmp_path):
    metadata, tensors = flatten({"w": torch.ones(2), "unsafe": object()})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="unsupported checkpoint metadata type: object"):
        serializer.save_sync(metadata, tensors, step=10)


def test_disk_save_rejects_structured_numpy_dtype(tmp_path):
    metadata = FlatStateDictMetadata(
        non_tensor_data={
            "structured": np.array([(1, 2.0)], dtype=[("index", "<i4"), ("value", "<f4")])
        }
    )
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="structured and subarray NumPy dtypes"):
        serializer.save_sync(metadata, [], step=11)


def test_disk_save_rejects_numpy_array_subclass(tmp_path):
    masked = np.ma.array([1.0, 2.0], mask=[False, True])
    metadata = FlatStateDictMetadata(non_tensor_data={"masked": masked})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="unsupported checkpoint metadata type: MaskedArray"):
        serializer.save_sync(metadata, [], step=12)


def test_disk_save_rejects_dictionary_subclass(tmp_path):
    metadata = FlatStateDictMetadata(non_tensor_data={"extra": defaultdict(int, attempts=3)})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="unsupported checkpoint metadata type: defaultdict"):
        serializer.save_sync(metadata, [], step=12)


def test_manager_treats_unsafe_disk_checkpoint_as_unavailable(tmp_path):
    step_dir = tmp_path / "step-11"
    step_dir.mkdir()
    torch.save({"legacy": "payload"}, step_dir / "rank-0.pt")
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )

    assert manager.load() is None
    manager.close()


def test_manager_collectively_rejects_valid_shard_when_peer_is_invalid(tmp_path):
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )
    metadata, tensors = flatten({"w": torch.ones(2)})
    manager._disk.save_sync(metadata, tensors, step=13)

    with (
        patch.object(manager, "_recovery_steps", return_value=(-1, 13)),
        patch.object(manager, "_collective_min_step", return_value=0) as collective,
    ):
        assert manager.load() is None

    assert [entry.args[0] for entry in collective.call_args_list] == [1, -1]
    manager.close()


def test_load_tensors_collectively_rejects_invalid_local_shard(tmp_path):
    step_dir = tmp_path / "step-14"
    step_dir.mkdir()
    torch.save({"legacy": "payload"}, step_dir / "rank-0.pt")
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )

    with (
        patch.object(manager, "_recovery_steps", return_value=(-1, 14)),
        patch.object(manager, "_collective_min_step", return_value=0) as collective,
    ):
        assert manager.load_tensors() is None

    assert [entry.args[0] for entry in collective.call_args_list] == [0, -1]
    manager.close()


def test_manager_votes_invalid_after_unexpected_local_load_error(tmp_path):
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )

    with (
        patch.object(manager, "_recovery_steps", return_value=(-1, 15)),
        patch.object(manager._disk, "load", side_effect=RuntimeError("decoder failed")),
        patch.object(manager, "_collective_min_step", return_value=0) as collective,
    ):
        assert manager.load() is None

    assert [entry.args[0] for entry in collective.call_args_list] == [0, -1]
    manager.close()


def test_manager_recovers_older_generation_when_newest_is_malformed(tmp_path):
    metadata, tensors = flatten({"w": torch.full((2,), 3.0)})
    DiskSerializer(str(tmp_path), rank=0).save_sync(metadata, tensors, step=3)
    newest = tmp_path / "step-4" / "rank-0.pt"
    newest.parent.mkdir()
    newest.write_bytes(b"torn checkpoint")

    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )

    assert manager.find_latest() == 3
    recovered = manager.load()
    assert recovered is not None
    state, step = recovered
    assert step == 3
    assert torch.equal(state["w"], torch.full((2,), 3.0))
    manager.close()


def test_consistent_disk_step_descends_to_exact_common_generation(tmp_path):
    manager = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        )
    )
    metadata, tensors = flatten({"w": torch.ones(2)})
    manager._disk.save_sync(metadata, tensors, step=8)
    manager._disk.save_sync(metadata, tensors, step=10)

    with patch.object(manager, "_collective_min_step", side_effect=[9, 8, 8]) as collective:
        assert manager._consistent_step(manager._disk) == 8

    assert [entry.args[0] for entry in collective.call_args_list] == [10, 8, 8]
    manager.close()


def test_restart_destination_mirrors_flushed_checkpoint(tmp_path):
    local = tmp_path / "local"
    mirror = tmp_path / "mirror"
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(local),
        verify_integrity=True,
    )
    checkpoint = InMemoryCheckpointManager(config)
    checkpoint.save({"model": {"w": torch.full((4,), 3.0)}}, step=5)
    checkpoint.maybe_wait()
    checkpoint.set_restart_destination(lambda: mirror)

    assert checkpoint._flush_and_mirror_for_restart() == 5
    checkpoint.close()

    recovered = InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(mirror),
            verify_integrity=True,
        )
    )
    result = recovered.load()
    assert result is not None
    state, step = result
    assert step == 5
    assert torch.equal(state["model"]["w"], torch.full((4,), 3.0))
    recovered.close()


def test_extra_state_round_trips_through_gemini(tmp_path):
    model = nn.Linear(8, 8)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    saved = {"rng": torch.get_rng_state(), "dl_cursor": 12345, "consumed": [1, 2, 3]}
    restored: dict = {}
    config = InMemoryCkptConfig(
        enable=True,
        interval=1,
        disk_flush_interval=0,
        disk_folder=str(tmp_path),
    )

    state = enable_resiliency(
        model,
        optimizer,
        interval=1,
        enable_detection=False,
        checkpoint=config,
        extra_state_fn=lambda: saved,
    )
    model(torch.randn(2, 8)).sum().backward()
    optimizer.step()
    state.flush_for_restart()
    state.close()

    recovered_model = nn.Linear(8, 8)
    recovered_optimizer = torch.optim.SGD(recovered_model.parameters(), lr=0.1)
    recovered_state = enable_resiliency(
        recovered_model,
        recovered_optimizer,
        interval=1,
        enable_detection=False,
        checkpoint=config,
        load_extra_state_fn=lambda extra: restored.update(extra),
    )
    assert restored["dl_cursor"] == 12345
    assert restored["consumed"] == [1, 2, 3]
    assert torch.equal(restored["rng"], saved["rng"])
    recovered_state.close()


def test_sdc_blocks_checkpoint_predicate():
    assert replay_result_has_sdc(None) is False
    assert (
        replay_result_has_sdc(SimpleNamespace(sdc_bitmap=[0, 0], straggler_bitmap=[0, 0])) is False
    )
    assert (
        replay_result_has_sdc(SimpleNamespace(sdc_bitmap=[0, 1], straggler_bitmap=[0, 0])) is True
    )
    assert (
        replay_result_has_sdc(SimpleNamespace(sdc_bitmap=[0, 0], straggler_bitmap=[1, 0])) is False
    )
    assert (
        replay_result_has_sdc(
            SimpleNamespace(
                sdc_bitmap=[0, 0],
                c3_results={
                    "replay_input": C3Result(
                        C3Status.INCONCLUSIVE,
                        [0, 0],
                        [11, 12],
                    )
                },
            )
        )
        is True
    )
