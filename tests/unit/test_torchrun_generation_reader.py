"""Contract tests for fail-closed torchrun generation-state reads."""

from __future__ import annotations

import threading

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
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
    coordinator_id: str = "coordinator-a",
    lease_id: str | None = None,
    fencing_token: int | None = None,
):
    snapshot = GenerationSnapshotRecord(
        assignment=_assignment(generation),
        previous_snapshot_digest=previous_snapshot_digest,
        coordinator_id=coordinator_id,
        lease_id=lease.record.lease_id if lease_id is None else lease_id,
        coordinator_fencing_token=(lease.fencing_token if fencing_token is None else fencing_token),
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

    snapshot = reader.get(0)

    assert snapshot is not None
    assert snapshot.record.assignment.generation == 0


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


def test_generation_reader_rejects_missing_or_unstamped_snapshot():
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
    with pytest.raises(GenerationStateCorrupt, match="commit time"):
        reader.current()


def test_generation_reader_rejects_head_digest_or_timestamp_substitution():
    clock, store, lease, reader = _state()
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
    store.compare_set_in_window(
        reader.head_key,
        expected_revision=head_entry.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=substituted.to_json(),
    )
    with pytest.raises(GenerationStateCorrupt, match="digest"):
        reader.current()

    current_head = store.get(reader.head_key)
    assert current_head is not None
    clock.set(1_010)
    original = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=0,
        snapshot_digest=GenerationSnapshotRecord.from_json(
            committed[reader.snapshot_key(0)].value
        ).digest,
    )
    store.compare_set_in_window(
        reader.head_key,
        expected_revision=current_head.revision,
        not_before_unix_ms=1_010,
        deadline_unix_ms=None,
        value=original.to_json(),
    )
    with pytest.raises(GenerationStateCorrupt, match="timestamps"):
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

    substituted = GenerationSnapshotRecord(
        assignment=_assignment(0),
        previous_snapshot_digest=None,
        coordinator_id="coordinator-a",
        lease_id="substituted-lease",
        coordinator_fencing_token=lease.fencing_token,
    )
    store.compare_set_in_window(
        reader.snapshot_key(0),
        expected_revision=None,
        not_before_unix_ms=1_010,
        deadline_unix_ms=None,
        value=substituted.to_json(),
    )
    with pytest.raises(GenerationStateCorrupt, match="predecessor digest"):
        reader.current()


@pytest.mark.parametrize(
    ("first_fencing_token", "first_lease_id", "second_lease_id", "message"),
    [
        (5, "lease-a", "lease-b", "fencing tokens"),
        (4, "lease-a", "lease-b", "lease identity"),
        (3, "lease-a", "lease-a", "changes coordinator"),
    ],
)
def test_generation_reader_rejects_invalid_lease_provenance(
    first_fencing_token,
    first_lease_id,
    second_lease_id,
    message,
):
    clock, store, lease, reader = _state()
    generation_zero = _commit(
        store,
        reader,
        lease,
        generation=0,
        previous_snapshot_digest=None,
        expected_head_revision=None,
        coordinator_id="coordinator-a",
        lease_id=first_lease_id,
        fencing_token=first_fencing_token,
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
        coordinator_id="coordinator-b",
        lease_id=second_lease_id,
        fencing_token=4,
    )

    with pytest.raises(GenerationStateCorrupt, match=message):
        reader.current()
