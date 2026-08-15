"""Fail-closed reads of immutable torchrun generation state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class GenerationStateError(RuntimeError):
    """Base error for generation-state operations."""


class GenerationStateCorrupt(GenerationStateError):
    """Raised when persisted generation state is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class StoredGenerationSnapshot:
    """One decoded immutable snapshot and its store metadata."""

    record: GenerationSnapshotRecord
    revision: int
    committed_at_unix_ms: int


@dataclass(frozen=True, slots=True)
class CurrentGeneration:
    """The generation head and the immutable snapshot it references."""

    snapshot: StoredGenerationSnapshot
    head_revision: int


class GenerationStateReader:
    """Read and verify one run's immutable generation history."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._coordinator_lease_key = f"{self._run_prefix}/coordinator-lease"
        self._head_key = f"{self._run_prefix}/generation-head"

    @property
    def coordinator_lease_key(self) -> str:
        return self._coordinator_lease_key

    @property
    def head_key(self) -> str:
        return self._head_key

    def snapshot_key(self, generation: int) -> str:
        normalized_generation = _nonnegative_integer(generation, "generation")
        return f"{self._run_prefix}/generations/{normalized_generation}"

    def current(self) -> CurrentGeneration | None:
        generation_zero_key = self.snapshot_key(0)
        for _ in range(_MAX_READ_ATTEMPTS):
            result = self._read_current_history()
            if result is not None:
                current, _ = result
                return current
            if self._store.get(generation_zero_key) is None:
                return None
            if self._store.get(self._head_key) is not None:
                continue
            raise GenerationStateCorrupt("generation snapshots exist without a generation head")
        raise GenerationStateError("generation head changed repeatedly during read")

    def get(self, generation: int) -> StoredGenerationSnapshot | None:
        normalized_generation = _nonnegative_integer(generation, "generation")
        snapshot_key = self.snapshot_key(normalized_generation)
        for _ in range(_MAX_READ_ATTEMPTS):
            result = self._read_current_history()
            if result is None:
                generation_zero = self._store.get(self.snapshot_key(0))
                requested_snapshot = self._store.get(snapshot_key)
                if generation_zero is None and requested_snapshot is None:
                    return None
                if self._store.get(self._head_key) is not None:
                    continue
                raise GenerationStateCorrupt("generation snapshot exists without a generation head")
            current, history = result
            current_generation = current.snapshot.record.assignment.generation
            if normalized_generation <= current_generation:
                return history[normalized_generation]
            if self._store.get(snapshot_key) is None:
                return None
            observed_head = self._store.get(self._head_key)
            if observed_head is not None and observed_head.revision != current.head_revision:
                continue
            if observed_head is None:
                raise GenerationStateCorrupt("generation head disappeared while reading a snapshot")
            if observed_head.revision == current.head_revision:
                raise GenerationStateCorrupt(
                    "generation snapshot is newer than the generation head"
                )
        raise GenerationStateError("generation head changed repeatedly during read")

    def _read_current_history(
        self,
    ) -> tuple[CurrentGeneration, dict[int, StoredGenerationSnapshot]] | None:
        head_entry = self._store.get(self._head_key)
        if head_entry is None:
            return None
        head = self._decode_head(head_entry)
        snapshot_entry = self._store.get(self.snapshot_key(head.generation))
        if snapshot_entry is None:
            raise GenerationStateCorrupt("generation head references a missing snapshot")
        snapshot = self._decode_snapshot(
            snapshot_entry,
            expected_generation=head.generation,
        )
        history = self._read_snapshot_history(snapshot)
        self._validate_head_link(head, head_entry, snapshot)
        return (
            CurrentGeneration(
                snapshot=snapshot,
                head_revision=head_entry.revision,
            ),
            history,
        )

    def _decode_head(self, entry: ControlStoreEntry) -> GenerationHeadRecord:
        try:
            head = GenerationHeadRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise GenerationStateCorrupt("generation head is malformed") from error
        if head.run_id != self._run_id:
            raise GenerationStateCorrupt("generation head belongs to another run")
        if entry.committed_at_unix_ms is None:
            raise GenerationStateCorrupt("generation head has no authoritative commit time")
        return head

    def _decode_snapshot(
        self,
        entry: ControlStoreEntry,
        *,
        expected_generation: int,
    ) -> StoredGenerationSnapshot:
        try:
            record = GenerationSnapshotRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise GenerationStateCorrupt("generation snapshot is malformed") from error
        assignment = record.assignment
        if assignment.run_id != self._run_id:
            raise GenerationStateCorrupt("generation snapshot belongs to another run")
        if assignment.generation != expected_generation:
            raise GenerationStateCorrupt("generation snapshot key and payload disagree")
        if entry.committed_at_unix_ms is None:
            raise GenerationStateCorrupt("generation snapshot has no authoritative commit time")
        self._validate_guard_provenance(entry, record)
        return StoredGenerationSnapshot(
            record=record,
            revision=entry.revision,
            committed_at_unix_ms=entry.committed_at_unix_ms,
        )

    def _validate_head_link(
        self,
        head: GenerationHeadRecord,
        head_entry: ControlStoreEntry,
        snapshot: StoredGenerationSnapshot,
    ) -> None:
        if head.snapshot_digest != snapshot.record.digest:
            raise GenerationStateCorrupt("generation head snapshot digest does not match")
        if head_entry.committed_at_unix_ms != snapshot.committed_at_unix_ms:
            raise GenerationStateCorrupt(
                "generation head and snapshot commit timestamps do not match"
            )
        self._validate_guard_provenance(head_entry, snapshot.record)

    def _read_snapshot_history(
        self,
        snapshot: StoredGenerationSnapshot,
    ) -> dict[int, StoredGenerationSnapshot]:
        successor = snapshot
        history = {successor.record.assignment.generation: successor}
        seen_lease_ids = {successor.record.lease_id}
        while successor.record.assignment.generation > 0:
            predecessor_generation = successor.record.assignment.generation - 1
            predecessor_entry = self._store.get(self.snapshot_key(predecessor_generation))
            if predecessor_entry is None:
                raise GenerationStateCorrupt("generation snapshot references a missing predecessor")
            predecessor = self._decode_snapshot(
                predecessor_entry,
                expected_generation=predecessor_generation,
            )
            if successor.record.previous_snapshot_digest != predecessor.record.digest:
                raise GenerationStateCorrupt(
                    "generation snapshot predecessor digest does not match"
                )
            if predecessor.committed_at_unix_ms > successor.committed_at_unix_ms:
                raise GenerationStateCorrupt("generation snapshot commit timestamps move backward")
            if (
                predecessor.record.coordinator_fencing_token
                > successor.record.coordinator_fencing_token
            ):
                raise GenerationStateCorrupt("generation snapshot fencing tokens move backward")
            if (
                predecessor.record.coordinator_fencing_token
                == successor.record.coordinator_fencing_token
                and (
                    predecessor.record.coordinator_id != successor.record.coordinator_id
                    or predecessor.record.lease_id != successor.record.lease_id
                )
            ):
                raise GenerationStateCorrupt("generation snapshots disagree on one lease identity")
            if (
                predecessor.record.lease_id == successor.record.lease_id
                and predecessor.record.coordinator_id != successor.record.coordinator_id
            ):
                raise GenerationStateCorrupt("one generation lease changes coordinator identity")
            if predecessor.record.lease_id != successor.record.lease_id:
                if predecessor.record.lease_id in seen_lease_ids:
                    raise GenerationStateCorrupt(
                        "generation snapshot lease identity reappears after replacement"
                    )
                seen_lease_ids.add(predecessor.record.lease_id)
            history[predecessor_generation] = predecessor
            successor = predecessor
        return history

    def _validate_guard_provenance(
        self,
        entry: ControlStoreEntry,
        snapshot: GenerationSnapshotRecord,
    ) -> None:
        if entry.guard_key != self._coordinator_lease_key:
            raise GenerationStateCorrupt(
                "generation state has no matching coordinator-lease guard key"
            )
        if entry.guard_revision != snapshot.coordinator_fencing_token:
            raise GenerationStateCorrupt(
                "generation state guard revision does not match its fencing token"
            )
        if entry.guard_value_digest != snapshot.coordinator_lease_digest:
            raise GenerationStateCorrupt(
                "generation state guard digest does not match its lease identity"
            )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


__all__ = [
    "CurrentGeneration",
    "GenerationStateCorrupt",
    "GenerationStateError",
    "GenerationStateReader",
    "StoredGenerationSnapshot",
]
