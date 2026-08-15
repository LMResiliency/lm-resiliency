"""CPU-only regressions for safe GEMINI checkpoint recovery."""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from lm_resiliency.checkpointing._disk_format import CheckpointFormatError
from lm_resiliency.checkpointing.buffer import SlotState
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.disk import DiskSerializer
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager
from lm_resiliency.checkpointing.state_dict import FlatStateDictMetadata, flatten


class _TaggedFloat(np.float32):
    pass


def test_disk_save_rejects_tensor_subclass_metadata(tmp_path):
    metadata = FlatStateDictMetadata(
        non_tensor_data={"parameter": torch.nn.Parameter(torch.ones(2))}
    )
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="unsupported checkpoint metadata type: Parameter"):
        serializer.save_sync(metadata, [], step=20)


def test_disk_save_rejects_requires_grad_tensor_metadata(tmp_path):
    metadata = FlatStateDictMetadata(
        non_tensor_data={"activation": torch.ones(2, requires_grad=True)}
    )
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="tensors requiring gradients"):
        serializer.save_sync(metadata, [], step=20)


def test_disk_save_rejects_negative_torch_size_metadata(tmp_path):
    metadata = FlatStateDictMetadata(non_tensor_data={"shape": torch.Size([-1, 768])})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="cannot contain negative dimensions"):
        serializer.save_sync(metadata, [], step=20)


def test_disk_save_rejects_numpy_scalar_subclass(tmp_path):
    metadata = FlatStateDictMetadata(non_tensor_data={"tagged": _TaggedFloat(1.5)})
    serializer = DiskSerializer(str(tmp_path), rank=0)

    with pytest.raises(TypeError, match="unsupported checkpoint metadata type: _TaggedFloat"):
        serializer.save_sync(metadata, [], step=21)


def test_disk_load_rejects_sparse_payload_tensor(tmp_path):
    metadata, tensors = flatten({"weight": torch.ones(2)})
    serializer = DiskSerializer(str(tmp_path), rank=0)
    path = serializer.save_sync(metadata, tensors, step=22)
    payload = torch.load(path, weights_only=True)
    payload["tensors"][0] = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([1.0]),
        size=(2,),
    )
    torch.save(payload, path)

    with pytest.raises(CheckpointFormatError, match="dense, strided, and non-quantized"):
        serializer.load(22)


def test_disk_load_rejects_meta_payload_tensor(tmp_path):
    metadata, tensors = flatten({"weight": torch.ones(2)})
    serializer = DiskSerializer(str(tmp_path), rank=0)
    path = serializer.save_sync(metadata, tensors, step=22)
    payload = torch.load(path, weights_only=True)
    payload["tensors"][0] = torch.empty(2, device="meta")
    torch.save(payload, path)

    with pytest.raises(CheckpointFormatError, match="must be materialized on CPU"):
        serializer.load(22)


def test_memory_lookup_never_uses_peer_replica_as_local_state():
    manager = object.__new__(InMemoryCheckpointManager)
    manager._metadata = FlatStateDictMetadata()
    own_slot = SimpleNamespace(step=24, state=SlotState.READY)
    peer_slot = SimpleNamespace(step=23, state=SlotState.READY)
    manager._buffer_pool = SimpleNamespace(
        allocated=True,
        own_slots=(own_slot,),
        get_slot_by_step=lambda step: peer_slot if step == 23 else None,
    )

    assert manager._memory_slot_by_step(23) is None
    assert manager._memory_slot_by_step(24) is own_slot


def _gloo_recovery_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(seconds=30),
    )
    try:
        manager = object.__new__(InMemoryCheckpointManager)
        manager._world_size = world_size
        manager._process_group = None

        # MIN(latest) selects step 10, but rank 1 has already rotated that exact
        # slot out while retaining a newer step. Both ranks must reject memory.
        local_latest = 10 if rank == 0 else 20
        local_slot = object() if rank == 0 else None
        with (
            patch.object(manager, "_latest_memory_step", return_value=local_latest),
            patch.object(manager, "_memory_slot_by_step", return_value=local_slot),
        ):
            memory_step = manager._consistent_memory_step()

        serializer = DiskSerializer(checkpoint_dir, rank=rank)
        metadata, tensors = flatten({"weight": torch.ones(2) * rank})
        path = serializer.save_sync(metadata, tensors, step=23)
        dist.barrier()

        if rank == 0:
            payload = torch.load(path, weights_only=True)
            document = json.loads(payload["metadata_json"])
            document["tensor_entries"][0]["key_path"] = ["missing"]
            payload["metadata_json"] = json.dumps(document)
            torch.save(payload, path)
        dist.barrier()

        manager._disk = serializer
        disk_result = manager._load_collectively_validated_disk_shard(23)

        Path(result_dir, f"rank-{rank}.json").write_text(
            json.dumps(
                {
                    "memory_step": memory_step,
                    "disk_rejected": disk_result is None,
                }
            )
        )
    finally:
        dist.destroy_process_group()


def _gloo_identity_worker(
    rank: int,
    world_size: int,
    rendezvous: str,
    checkpoint_dir: str,
    result_dir: str,
) -> None:
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=rank,
        world_size=world_size,
        timeout=dt.timedelta(seconds=30),
    )
    try:
        rejected = False
        try:
            InMemoryCheckpointManager(
                InMemoryCkptConfig(
                    disk_folder=checkpoint_dir,
                    run_id=f"different-run-{rank}",
                ),
                parallelism_info=SimpleNamespace(has_natural_replicas=True),
            )
        except RuntimeError as error:
            rejected = "disagree on run_id or topology" in str(error)
        Path(result_dir, f"identity-rank-{rank}.json").write_text(
            json.dumps({"rejected": rejected})
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo backend is unavailable")
def test_cpu_gloo_recovery_consensus(tmp_path):
    world_size = 2
    rendezvous = tmp_path / "gloo-rendezvous"
    checkpoint_dir = tmp_path / "checkpoints"
    result_dir = tmp_path / "results"
    result_dir.mkdir()

    mp.spawn(
        _gloo_recovery_worker,
        args=(world_size, str(rendezvous), str(checkpoint_dir), str(result_dir)),
        nprocs=world_size,
        join=True,
    )

    results = [
        json.loads((result_dir / f"rank-{rank}.json").read_text()) for rank in range(world_size)
    ]
    assert results == [
        {"memory_step": -1, "disk_rejected": True},
        {"memory_step": -1, "disk_rejected": True},
    ]


@pytest.mark.skipif(not dist.is_gloo_available(), reason="PyTorch Gloo backend is unavailable")
def test_cpu_gloo_requires_exact_run_identity_agreement(tmp_path):
    world_size = 2
    rendezvous = tmp_path / "identity-rendezvous"
    checkpoint_dir = tmp_path / "identity-checkpoints"
    result_dir = tmp_path / "identity-results"
    result_dir.mkdir()

    mp.spawn(
        _gloo_identity_worker,
        args=(world_size, str(rendezvous), str(checkpoint_dir), str(result_dir)),
        nprocs=world_size,
        join=True,
    )

    results = [
        json.loads((result_dir / f"identity-rank-{rank}.json").read_text())
        for rank in range(world_size)
    ]
    assert results == [{"rejected": True}, {"rejected": True}]
