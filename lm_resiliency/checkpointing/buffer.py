"""Buffer pool for in-memory checkpoint storage with pinned CPU memory."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field

import torch

from lm_resiliency.checkpointing.state_dict import TensorEntry


class SlotState(enum.Enum):
    EMPTY = "empty"
    COPYING = "copying"
    READY = "ready"
    REPLICATING = "replicating"


@dataclass(slots=True)
class BufferSlot:
    """A single buffer slot holding a checkpoint's tensor data on CPU.

    non_tensor_data is this slot's own copy of the checkpoint's non-tensor
    skeleton (dataloader/RNG/step values, etc.), captured at the save that filled
    the slot — or received from the peer for a replica slot. Stored per slot (not
    once on the manager) so an older slot flushes with *its* step's values, and so
    a replicated peer slot carries the *peer's* values, not the local rank's.
    """

    tensors: list[torch.Tensor] = field(default_factory=list)
    tensor_entries: list[TensorEntry] | None = None
    step: int = -1
    state: SlotState = SlotState.EMPTY
    non_tensor_data: object | None = None


class BufferPool:
    """Manages pinned CPU buffers for in-memory checkpointing.

    In 4-slot replicated mode:
      - own_current: receiving GPU→CPU copy for the current step
      - own_previous: prior completed local recovery copy
      - peer_current: receiving data from peer (resized for its shard layout)
      - peer_previous: prior completed peer recovery copy

    The current local slot is replicated as soon as its GPU-to-CPU copy completes.
    The previous local and peer slots remain aligned while that exchange is in
    flight, so a failed exchange cannot remove the prior recoverable generation.

    In 2-slot mode (HSDP, no replication needed):
      - own_current and own_previous only
    """

    def __init__(self, num_slots: int = 4, pin_memory: bool = True) -> None:
        if num_slots not in (2, 4):
            raise ValueError("BufferPool supports two local slots or four replicated slots")
        self._num_slots = num_slots
        self._pin_memory = pin_memory
        self._slots: list[BufferSlot] = [BufferSlot() for _ in range(num_slots)]
        self._allocated = False
        self._tensor_entries: list[TensorEntry] | None = None

    @property
    def allocated(self) -> bool:
        return self._allocated

    @property
    def own_current(self) -> BufferSlot:
        return self._slots[0]

    @property
    def own_previous(self) -> BufferSlot:
        return self._slots[1]

    @property
    def own_slots(self) -> tuple[BufferSlot, ...]:
        return tuple(self._slots[:2])

    @property
    def _peer_offset(self) -> int:
        return 2

    @property
    def peer_current(self) -> BufferSlot:
        if self._num_slots < 4:
            raise RuntimeError("Peer slots not available in 2-slot mode")
        return self._slots[self._peer_offset]

    @property
    def peer_previous(self) -> BufferSlot:
        if self._num_slots < 4:
            raise RuntimeError("Peer slots not available in 2-slot mode")
        return self._slots[self._peer_offset + 1]

    def allocate(self, tensor_entries: list[TensorEntry]) -> None:
        """Allocate all buffer slots based on the tensor metadata from the first checkpoint."""
        if self._allocated:
            return
        self._tensor_entries = tensor_entries
        for slot in self._slots:
            slot.tensors = self._allocate_tensors(tensor_entries)
            slot.tensor_entries = list(tensor_entries)
        self._allocated = True

    def _allocate_tensors(self, entries: list[TensorEntry]) -> list[torch.Tensor]:
        """Allocate a list of CPU tensors matching the given entries."""
        tensors = []
        for entry in entries:
            t = torch.empty(entry.shape, dtype=entry.dtype, device="cpu")
            if self._pin_memory and torch.cuda.is_available():
                t = t.pin_memory()
            tensors.append(t)
        return tensors

    def rotate(self) -> None:
        """Rotate buffer slots after a checkpoint interval.

        own_current (completed local copy) → own_previous
        own_previous → own_current (freed for next copy)
        peer_current (completed peer copy) → peer_previous
        peer_previous → peer_current (freed for next receive)
        """
        self._slots[0], self._slots[1] = self._slots[1], self._slots[0]
        if self._num_slots >= 4:
            current = self._peer_offset
            previous = current + 1
            self._slots[current], self._slots[previous] = (
                self._slots[previous],
                self._slots[current],
            )

    def get_latest_own_step(self) -> int:
        """Return the most recent step stored in own slots."""
        return max(slot.step for slot in self.own_slots)

    def get_latest_peer_step(self) -> int:
        """Return the most recent step stored in peer slots."""
        if self._num_slots < 4:
            return -1
        return max(self.peer_current.step, self.peer_previous.step)

    def get_slot_by_step(
        self,
        step: int,
    ) -> BufferSlot | None:
        """Find a complete slot containing the requested step."""
        for slot in self._slots:
            if slot.step == step and slot.state in (SlotState.READY, SlotState.REPLICATING):
                return slot
        return None

    def total_memory_bytes(self) -> int:
        """Total CPU memory used by all buffer slots."""
        if not self._allocated:
            return 0
        total = 0
        for slot in self._slots:
            for t in slot.tensors:
                total += t.nelement() * t.element_size()
        return total
