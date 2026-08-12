"""Tests for BufferPool slot management."""

import pytest
import torch

from lm_resiliency.checkpointing.buffer import BufferPool, SlotState
from lm_resiliency.checkpointing.state_dict import TensorEntry


def _make_entries(shapes: list[tuple[int, ...]]) -> list[TensorEntry]:
    return [
        TensorEntry(
            key_path=(f"t{i}",),
            shape=torch.Size(s),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )
        for i, s in enumerate(shapes)
    ]


def test_allocation():
    pool = BufferPool(num_slots=4, pin_memory=False)
    assert not pool.allocated

    entries = _make_entries([(3, 4), (5,)])
    pool.allocate(entries)
    assert pool.allocated

    # Each slot should have 2 tensors
    assert len(pool.own_current.tensors) == 2
    assert len(pool.own_previous.tensors) == 2
    assert len(pool.peer_current.tensors) == 2
    assert len(pool.peer_previous.tensors) == 2

    # Shapes match
    assert pool.own_current.tensors[0].shape == torch.Size([3, 4])
    assert pool.own_current.tensors[1].shape == torch.Size([5])


def test_allocation_idempotent():
    pool = BufferPool(num_slots=2, pin_memory=False)
    entries = _make_entries([(2, 2)])
    pool.allocate(entries)
    # Second call should be a no-op
    pool.allocate(_make_entries([(10, 10)]))
    assert pool.own_current.tensors[0].shape == torch.Size([2, 2])


def test_rotation():
    pool = BufferPool(num_slots=4, pin_memory=False)
    entries = _make_entries([(2, 2)])
    pool.allocate(entries)

    # Set steps
    pool.own_current.step = 10
    pool.own_current.state = SlotState.READY
    pool.own_previous.step = 5

    slot_a_id = id(pool.own_current)
    slot_b_id = id(pool.own_previous)

    pool.rotate()

    # After rotation, slots swap
    assert id(pool.own_current) == slot_b_id
    assert id(pool.own_previous) == slot_a_id
    assert pool.own_previous.step == 10


def test_two_slot_mode():
    pool = BufferPool(num_slots=2, pin_memory=False)
    entries = _make_entries([(4,)])
    pool.allocate(entries)

    # Peer slots should raise
    with pytest.raises(RuntimeError, match="Peer slots not available"):
        _ = pool.peer_current
    with pytest.raises(RuntimeError, match="Peer slots not available"):
        _ = pool.peer_previous


def test_get_slot_by_step():
    pool = BufferPool(num_slots=4, pin_memory=False)
    entries = _make_entries([(3,)])
    pool.allocate(entries)

    pool.own_current.step = 20
    pool.own_current.state = SlotState.READY
    pool.own_previous.step = 10
    pool.own_previous.state = SlotState.READY

    assert pool.get_slot_by_step(20) is pool.own_current
    assert pool.get_slot_by_step(10) is pool.own_previous
    assert pool.get_slot_by_step(99) is None


def test_get_slot_not_ready():
    pool = BufferPool(num_slots=4, pin_memory=False)
    entries = _make_entries([(3,)])
    pool.allocate(entries)

    pool.own_current.step = 20
    pool.own_current.state = SlotState.COPYING  # not ready
    assert pool.get_slot_by_step(20) is None


def test_slot_is_recoverable_while_replication_reads_it():
    pool = BufferPool(num_slots=2, pin_memory=False)
    pool.allocate(_make_entries([(3,)]))
    pool.own_previous.step = 20
    pool.own_previous.state = SlotState.REPLICATING

    assert pool.get_slot_by_step(20) is pool.own_previous


def test_total_memory_bytes():
    pool = BufferPool(num_slots=2, pin_memory=False)
    entries = _make_entries([(10,)])  # 10 float32 = 40 bytes per slot
    pool.allocate(entries)
    assert pool.total_memory_bytes() == 2 * 10 * 4  # 2 slots * 10 elements * 4 bytes


def test_replicated_rotation_retains_matching_previous_generation():
    pool = BufferPool(num_slots=4, pin_memory=False)
    pool.allocate(_make_entries([(1,)]))
    pool.own_current.step = 22
    pool.own_previous.step = 20
    pool.peer_current.step = 22
    pool.peer_previous.step = 20

    pool.rotate()

    assert pool.own_current.step == 20
    assert pool.own_previous.step == 22
    assert pool.peer_current.step == 20
    assert pool.peer_previous.step == 22


def test_invalid_slot_count_is_rejected():
    with pytest.raises(ValueError, match="two local slots or four replicated slots"):
        BufferPool(num_slots=5, pin_memory=False)


def test_get_latest_steps():
    pool = BufferPool(num_slots=4, pin_memory=False)
    entries = _make_entries([(2,)])
    pool.allocate(entries)

    pool.own_current.step = 30
    pool.own_previous.step = 20
    pool.peer_current.step = 25
    pool.peer_previous.step = 15

    assert pool.get_latest_own_step() == 30
    assert pool.get_latest_peer_step() == 25
