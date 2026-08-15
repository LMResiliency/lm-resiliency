"""Contract tests for fail-closed torchrun generation-state reads."""

from __future__ import annotations

import hashlib
import threading

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateCorrupt,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import RankAssignment, SlotAssignment

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms

    def set(self, now_unix_ms: int) -> None:
        with self._lock:
            self.now_unix_ms = now_unix_ms


class StaticControlStore:
    def __init__(self) -> None:
        self.entries: dict[str, ControlStoreEntry] = {}

    def get(self, key: str) -> ControlStoreEntry | None:
        return self.entries.get(key)


def _assignment(generation: int) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=2,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-b",
                first_global_rank=2,
                local_world_size=2,
            ),
        ),
        topology_digest="topology-v1",
    )


def _state():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    lease = lease_manager.acquire()
    reader = GenerationStateReader(store, run_id=RUN_ID)
    return clock, store, lease, reader


def _commit(
    store: InMemoryControlStore,
    reader: GenerationStateReader,
    lease: HeldCoordinatorLease,
    *,
    generation: int,
    previous_snapshot_digest: str | None,
    expected_head_revision: int | None,
):
    snapshot = GenerationSnapshotRecord(
        assignment=_assignment(generation),
        previous_snapshot_digest=previous_snapshot_digest,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=generation,
        snapshot_digest=snapshot.digest,
    )
    return store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=expected_head_revision,
                value=head.to_json(),
            ),
            reader.snapshot_key(generation): ControlStoreWrite(
                expected_revision=None,
                value=snapshot.to_json(),
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )


def _snapshot(
    generation: int,
    *,
    previous_snapshot_digest: str | None,
    coordinator_id: str,
    lease_id: str,
    fencing_token: int,
    lease_duration_ms: int = 100,
) -> GenerationSnapshotRecord:
    return GenerationSnapshotRecord(
        assignment=_assignment(generation),
        previous_snapshot_digest=previous_snapshot_digest,
        coordinator_id=coordinator_id,
        lease_id=lease_id,
        coordinator_lease_duration_ms=lease_duration_ms,
        coordinator_fencing_token=fencing_token,
    )


def _reader_from_history(
    records: tuple[GenerationSnapshotRecord, ...],
    *,
    head_committed_at_offset_ms: int = 0,
) -> GenerationStateReader:
    store = StaticControlStore()
    reader = GenerationStateReader(store, run_id=RUN_ID)
    for record in records:
        generation = record.assignment.generation
        store.entries[reader.snapshot_key(generation)] = ControlStoreEntry(
            value=record.to_json(),
            revision=1,
            committed_at_unix_ms=1_000 + generation,
            guard_key=reader.coordinator_lease_key,
            guard_revision=record.coordinator_fencing_token,
            guard_value_digest=record.coordinator_lease_digest,
            guard_committed_at_unix_ms=1_000 + generation,
        )
    latest = records[-1]
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=latest.assignment.generation,
        snapshot_digest=latest.digest,
    )
    store.entries[reader.head_key] = ControlStoreEntry(
        value=head.to_json(),
        revision=len(records),
        committed_at_unix_ms=(1_000 + latest.assignment.generation + head_committed_at_offset_ms),
        mutation_sequence=len(records),
        guard_key=reader.coordinator_lease_key,
        guard_revision=latest.coordinator_fencing_token,
        guard_value_digest=latest.coordinator_lease_digest,
        guard_committed_at_unix_ms=1_000 + latest.assignment.generation,
    )
    return reader


def _history(
    *provenance: tuple[str, str, int],
) -> tuple[GenerationSnapshotRecord, ...]:
    records: list[GenerationSnapshotRecord] = []
    for generation, (coordinator_id, lease_id, fencing_token) in enumerate(provenance):
        records.append(
            _snapshot(
                generation,
                previous_snapshot_digest=None if not records else records[-1].digest,
                coordinator_id=coordinator_id,
                lease_id=lease_id,
                fencing_token=fencing_token,
            )
        )
    return tuple(records)


def _duration_change_history() -> tuple[GenerationSnapshotRecord, ...]:
    generation_zero = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        fencing_token=1,
        lease_duration_ms=100,
    )
    generation_one = _snapshot(
        1,
        previous_snapshot_digest=generation_zero.digest,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        fencing_token=2,
        lease_duration_ms=200,
    )
    return generation_zero, generation_one


def test_generation_reader_returns_none_before_initialization():
    _, _, _, reader = _state()

    assert reader.current() is None
    assert reader.get(0) is None


def test_generation_reader_retries_concurrent_initialization(monkeypatch):
    _, store, lease, reader = _state()
    original_get = store.get
    triggered = False

    def racing_get(key):
        nonlocal triggered
        if key == reader.snapshot_key(0) and not triggered:
            triggered = True
            _commit(
                store,
                reader,
                lease,
                generation=0,
                previous_snapshot_digest=None,
                expected_head_revision=None,
            )
        return original_get(key)

    monkeypatch.setattr(store, "get", racing_get)

    current = reader.current()

    assert current is not None
    assert current.snapshot.record.assignment.generation == 0
    assert reader.get(0) == current.snapshot


def test_generation_reader_reads_current_and_historical_snapshots():
    clock, store, lease, reader = _state()
    generation_zero = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    current_zero = reader.current()
    assert current_zero is not None
    clock.set(1_010)
    _commit(
        store,
        reader,
        lease,
        generation=1,
        previous_snapshot_digest=current_zero.snapshot.record.digest,
        expected_head_revision=generation_zero[reader.head_key].revision,
    )

    current = reader.current()

    assert current is not None
    assert current.snapshot.record.assignment.generation == 1
    assert reader.get(0) == current_zero.snapshot
    assert reader.get(1) == current.snapshot


def test_generation_reader_uses_distinct_run_namespaces():
    _, _, _, reader = _state()
    other = GenerationStateReader(InMemoryControlStore(), run_id="other-run")

    assert reader.head_key != other.head_key
    assert reader.coordinator_lease_key != other.coordinator_lease_key
    assert reader.snapshot_key(0) != other.snapshot_key(0)


def test_generation_reader_rejects_snapshots_outside_committed_head():
    _, store, lease, reader = _state()
    orphan_zero = GenerationSnapshotRecord(
        assignment=_assignment(0),
        previous_snapshot_digest=None,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    store.compare_set_in_window(
        reader.snapshot_key(0),
        expected_revision=None,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=orphan_zero.to_json(),
    )
    with pytest.raises(GenerationStateCorrupt, match="without a generation head"):
        reader.current()
    with pytest.raises(GenerationStateCorrupt, match="without a generation head"):
        reader.get(0)

    clock, store, lease, reader = _state()
    committed = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    clock.set(1_010)
    orphan_one = GenerationSnapshotRecord(
        assignment=_assignment(1),
        previous_snapshot_digest=GenerationSnapshotRecord.from_json(
            committed[reader.snapshot_key(0)].value
        ).digest,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    store.compare_set_in_window(
        reader.snapshot_key(1),
        expected_revision=None,
        not_before_unix_ms=1_010,
        deadline_unix_ms=None,
        value=orphan_one.to_json(),
    )
    with pytest.raises(GenerationStateCorrupt, match="newer than"):
        reader.get(1)


def test_generation_reader_rejects_missing_or_recreated_snapshot():
    _, store, lease, reader = _state()
    committed = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    snapshot_entry = committed[reader.snapshot_key(0)]
    store.compare_delete(
        reader.snapshot_key(0),
        expected_revision=snapshot_entry.revision,
    )

    with pytest.raises(GenerationStateCorrupt, match="missing snapshot"):
        reader.current()

    store.compare_set(
        reader.snapshot_key(0),
        expected_revision=None,
        value=snapshot_entry.value,
    )
    with pytest.raises(GenerationStateCorrupt, match="replaced or recreated"):
        reader.current()


def test_generation_reader_rejects_untimed_or_expired_lease_guards():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_record = CoordinatorLeaseRecord(
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        lease_duration_ms=100,
    )
    untimed_guard = store.compare_set(
        "lm_resiliency/torchrun/v1/runs/"
        + hashlib.sha256(RUN_ID.encode()).hexdigest()
        + "/coordinator-lease",
        expected_revision=None,
        value=lease_record.to_json(),
    )
    reader = GenerationStateReader(store, run_id=RUN_ID)
    snapshot = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id=lease_record.coordinator_id,
        lease_id=lease_record.lease_id,
        fencing_token=untimed_guard.revision,
    )
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=snapshot.digest,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=None,
                value=head.to_json(),
            ),
            reader.snapshot_key(0): ControlStoreWrite(
                expected_revision=None,
                value=snapshot.to_json(),
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=untimed_guard.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_200,
    )
    with pytest.raises(GenerationStateCorrupt, match="grant time"):
        reader.current()

    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    reader = GenerationStateReader(store, run_id=RUN_ID)
    timed_guard = store.compare_set_in_window(
        reader.coordinator_lease_key,
        expected_revision=None,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=lease_record.to_json(),
    )
    clock.set(1_100)
    expired_snapshot = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id=lease_record.coordinator_id,
        lease_id=lease_record.lease_id,
        fencing_token=timed_guard.revision,
    )
    expired_head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=expired_snapshot.digest,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=None,
                value=expired_head.to_json(),
            ),
            reader.snapshot_key(0): ControlStoreWrite(
                expected_revision=None,
                value=expired_snapshot.to_json(),
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=timed_guard.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_200,
    )
    with pytest.raises(GenerationStateCorrupt, match="lease expired"):
        reader.current()


def test_generation_reader_rejects_replaced_snapshot_key():
    clock, store, lease, reader = _state()
    committed = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    snapshot_entry = committed[reader.snapshot_key(0)]
    head_entry = committed[reader.head_key]
    clock.set(1_010)
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=head_entry.revision,
                value=head_entry.value,
            ),
            reader.snapshot_key(0): ControlStoreWrite(
                expected_revision=snapshot_entry.revision,
                value=snapshot_entry.value,
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=1_010,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )

    with pytest.raises(GenerationStateCorrupt, match="replaced or recreated"):
        reader.current()


def test_generation_reader_rejects_head_digest_or_timestamp_substitution():
    _, store, lease, reader = _state()
    committed = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    head_entry = committed[reader.head_key]
    substituted = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest="f" * 64,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=head_entry.revision,
                value=substituted.to_json(),
            )
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=1_000,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )
    with pytest.raises(GenerationStateCorrupt, match="digest"):
        reader.current()

    snapshot_zero = GenerationSnapshotRecord.from_json(committed[reader.snapshot_key(0)].value)
    timestamp_reader = _reader_from_history(
        (snapshot_zero,),
        head_committed_at_offset_ms=1,
    )
    with pytest.raises(GenerationStateCorrupt, match="timestamps"):
        timestamp_reader.current()


def test_generation_reader_rejects_same_timestamp_head_rollback():
    _, store, lease, reader = _state()
    generation_zero = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    snapshot_zero = GenerationSnapshotRecord.from_json(
        generation_zero[reader.snapshot_key(0)].value
    )
    generation_one = _commit(
        store,
        reader,
        lease,
        generation=1,
        previous_snapshot_digest=snapshot_zero.digest,
        expected_head_revision=generation_zero[reader.head_key].revision,
    )
    rollback = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=snapshot_zero.digest,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=generation_one[reader.head_key].revision,
                value=rollback.to_json(),
            )
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )

    with pytest.raises(GenerationStateCorrupt, match="mutation sequence"):
        reader.current()


def test_generation_reader_rejects_missing_or_substituted_predecessor():
    clock, store, lease, reader = _state()
    generation_zero = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
    )
    snapshot_zero = GenerationSnapshotRecord.from_json(
        generation_zero[reader.snapshot_key(0)].value
    )
    clock.set(1_010)
    _commit(
        store,
        reader,
        lease,
        generation=1,
        previous_snapshot_digest=snapshot_zero.digest,
        expected_head_revision=generation_zero[reader.head_key].revision,
    )
    predecessor_entry = store.get(reader.snapshot_key(0))
    assert predecessor_entry is not None
    store.compare_delete(
        reader.snapshot_key(0),
        expected_revision=predecessor_entry.revision,
    )

    with pytest.raises(GenerationStateCorrupt, match="missing predecessor"):
        reader.current()

    original_zero = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        fencing_token=1,
    )
    substituted_zero = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id="coordinator-a",
        lease_id="substituted-lease",
        fencing_token=1,
    )
    generation_one = _snapshot(
        1,
        previous_snapshot_digest=original_zero.digest,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        fencing_token=1,
    )
    substituted_reader = _reader_from_history((substituted_zero, generation_one))
    with pytest.raises(GenerationStateCorrupt, match="predecessor digest"):
        substituted_reader.current()


def test_generation_reader_anchors_snapshot_to_store_guard():
    _, store, lease, reader = _state()
    forged = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id="forged-coordinator",
        lease_id="forged-lease",
        fencing_token=lease.fencing_token,
    )
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=forged.digest,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=None,
                value=head.to_json(),
            ),
            reader.snapshot_key(0): ControlStoreWrite(
                expected_revision=None,
                value=forged.to_json(),
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )

    with pytest.raises(GenerationStateCorrupt, match="guard digest"):
        reader.current()


def test_generation_reader_anchors_fencing_token_to_store_guard():
    _, store, lease, reader = _state()
    forged = _snapshot(
        0,
        previous_snapshot_digest=None,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        fencing_token=lease.fencing_token + 1,
    )
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=forged.digest,
    )
    store.compare_set_many_guarded(
        {
            reader.head_key: ControlStoreWrite(
                expected_revision=None,
                value=head.to_json(),
            ),
            reader.snapshot_key(0): ControlStoreWrite(
                expected_revision=None,
                value=forged.to_json(),
            ),
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )

    with pytest.raises(GenerationStateCorrupt, match="guard revision"):
        reader.current()


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (
            _history(
                ("coordinator-a", "lease-a", 5),
                ("coordinator-b", "lease-b", 4),
            ),
            "fencing tokens",
        ),
        (
            _history(
                ("coordinator-a", "lease-a", 4),
                ("coordinator-b", "lease-b", 4),
            ),
            "lease identity",
        ),
        (
            _history(
                ("coordinator-a", "lease-a", 3),
                ("coordinator-b", "lease-a", 4),
            ),
            "changes coordinator",
        ),
        (
            _duration_change_history(),
            "changes its duration",
        ),
        (
            _history(
                ("coordinator-a", "lease-a", 1),
                ("coordinator-b", "lease-b", 3),
                ("coordinator-a", "lease-a", 5),
            ),
            "reappears",
        ),
    ],
)
def test_generation_reader_rejects_invalid_lease_history(history, message):
    reader = _reader_from_history(history)

    with pytest.raises(GenerationStateCorrupt, match=message):
        reader.current()
