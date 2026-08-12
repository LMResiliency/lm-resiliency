"""Core GEMINI recovery and integrity tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from lm_resiliency.api import enable_resiliency
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.disk import ChecksumMismatch, DiskSerializer, shard_checksums
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager
from lm_resiliency.checkpointing.state_dict import flatten
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

    raw = torch.load(path, weights_only=False)
    raw["tensors"][0][0] += 7.0
    torch.save(raw, path)
    with pytest.raises(ChecksumMismatch):
        serializer.load(3)


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
    raw = torch.load(rank_file, weights_only=False)
    raw["tensors"][0][0] += 9.0
    torch.save(raw, rank_file)

    corrupted_manager = InMemoryCheckpointManager(config)
    assert corrupted_manager.load() is None
    corrupted_manager.close()


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
