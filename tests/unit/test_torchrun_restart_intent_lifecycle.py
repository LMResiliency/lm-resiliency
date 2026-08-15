"""Contract tests for restart-intent lifecycle observation."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_reader import CurrentGeneration
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle import (
    RestartIntentLifecycleConflict,
    RestartIntentLifecycleCorrupt,
    RestartIntentLifecycleReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)

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


class CallbackStore(InMemoryControlStore):
    def __init__(self, *, clock: ManualClock) -> None:
        super().__init__(clock=clock)
        self.callback_key: str | None = None
        self.callback: Callable[[], None] | None = None

    def get(self, key: str) -> ControlStoreEntry | None:
        entry = super().get(key)
        if key == self.callback_key and self.callback is not None:
            callback = self.callback
            self.callback = None
            callback()
        return entry


class EntryOverrideStore(InMemoryControlStore):
    def __init__(self, *, clock: ManualClock) -> None:
        super().__init__(clock=clock)
        self.overrides: dict[str, ControlStoreEntry] = {}

    def get(self, key: str) -> ControlStoreEntry | None:
        return self.overrides.get(key, super().get(key))


def _assignment(
    generation: int = 0,
    *,
    replacement_node_id: str = "node-b",
) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, replacement_node_id, 2, 2),
        ),
        topology_digest="topology-v1",
    )


def _intent(
    *,
    intent_id: str = "intent-a",
    generation: int = 0,
) -> RestartIntent:
    return RestartIntent(
        intent_id=intent_id,
        run_id=RUN_ID,
        generation=generation,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=2_000,
    )


def _state(
    *,
    clock: ManualClock | None = None,
    store: InMemoryControlStore | None = None,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    GenerationStateManager,
    RestartIntentLifecycleReader,
    HeldCoordinatorLease,
    CurrentGeneration,
]:
    clock = ManualClock() if clock is None else clock
    store = InMemoryControlStore(clock=clock) if store is None else store
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    reader = RestartIntentLifecycleReader(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    current = generation_manager.initialize(lease, _assignment())
    return clock, store, lease_manager, generation_manager, reader, lease, current


def _records(
    current: CurrentGeneration,
    opening_lease: HeldCoordinatorLease,
    closing_lease: HeldCoordinatorLease,
    *,
    intent_id: str = "intent-a",
    generation_snapshot_digest: str | None = None,
) -> tuple[
    RestartIntentRecord,
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
]:
    record = RestartIntentRecord(
        intent=_intent(intent_id=intent_id),
        generation_snapshot_digest=(
            current.snapshot.record.digest
            if generation_snapshot_digest is None
            else generation_snapshot_digest
        ),
        coordinator_id=opening_lease.record.coordinator_id,
        lease_id=opening_lease.record.lease_id,
        coordinator_lease_duration_ms=opening_lease.record.lease_duration_ms,
        coordinator_fencing_token=opening_lease.fencing_token,
    )
    head = RestartIntentHeadRecord(
        run_id=RUN_ID,
        generation=0,
        intent_id=intent_id,
        intent_digest=record.digest,
    )
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=head,
        coordinator_id=closing_lease.record.coordinator_id,
        lease_id=closing_lease.record.lease_id,
        coordinator_lease_duration_ms=closing_lease.record.lease_duration_ms,
        coordinator_fencing_token=closing_lease.fencing_token,
    )
    return record, head, lifecycle


def _commit_intent(
    store: InMemoryControlStore,
    generation_manager: GenerationStateManager,
    reader: RestartIntentLifecycleReader,
    current: CurrentGeneration,
    lease: HeldCoordinatorLease,
    record: RestartIntentRecord,
) -> None:
    store.compare_set_many_guarded(
        {
            reader.intent_key(record.intent.intent_id): ControlStoreWrite(
                expected_revision=None,
                value=record.to_json(),
                require_never_created=True,
            )
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
        conditions={
            generation_manager.head_key: current.head_revision,
            generation_manager.snapshot_key(0): current.snapshot.revision,
        },
    )


def _commit_lifecycle(
    store: InMemoryControlStore,
    reader: RestartIntentLifecycleReader,
    lease: HeldCoordinatorLease,
    lifecycle: RestartIntentLifecycleRecord,
    *,
    expected_revision: int | None = None,
):
    return store.compare_set_many_guarded(
        {
            reader.lifecycle_key: ControlStoreWrite(
                expected_revision=expected_revision,
                value=lifecycle.to_json(),
            )
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )[reader.lifecycle_key]


def test_current_returns_none_for_never_created_lifecycle():
    _, _, _, _, reader, _, current = _state()

    assert reader.current(current) is None


def test_current_rechecks_generation_before_returning_no_lifecycle():
    clock = ManualClock()
    store = CallbackStore(clock=clock)
    (
        _,
        _,
        _,
        generation_manager,
        reader,
        lease,
        current,
    ) = _state(clock=clock, store=store)
    clock.set(1_010)
    store.callback_key = reader.lifecycle_key
    store.callback = lambda: generation_manager.commit_successor(
        lease,
        current,
        _assignment(1, replacement_node_id="node-c"),
    )

    with pytest.raises(RestartIntentLifecycleConflict, match="generation changed"):
        reader.current(current)


def test_current_verifies_closed_intent_and_closing_lease():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    committed = _commit_lifecycle(store, reader, lease, lifecycle)

    observed = reader.current(current)

    assert observed is not None
    assert observed.record == lifecycle
    assert observed.revision == committed.revision
    assert observed.committed_at_unix_ms == 1_000
    assert observed.transaction_sequence == committed.transaction_sequence


def test_current_accepts_closure_under_renewed_lease():
    clock, store, lease_manager, generation_manager, reader, lease, current = _state()
    record, _, _ = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    _, _, lifecycle = _records(current, lease, renewed)
    committed = _commit_lifecycle(store, reader, renewed, lifecycle)

    observed = reader.current(current)

    assert observed is not None
    assert observed.record == lifecycle
    assert observed.revision == committed.revision


def test_current_accepts_later_closure_under_the_same_lease():
    clock, store, _, generation_manager, reader, lease, current = _state()
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    clock.set(1_010)
    committed = _commit_lifecycle(store, reader, lease, lifecycle)

    observed = reader.current(current)

    assert observed is not None
    assert observed.record == lifecycle
    assert observed.revision == committed.revision
    assert observed.committed_at_unix_ms == 1_010


def test_current_rejects_stale_generation():
    clock, _, _, generation_manager, reader, lease, current = _state()
    clock.set(1_010)
    generation_manager.commit_successor(
        lease,
        current,
        _assignment(1, replacement_node_id="node-c"),
    )

    with pytest.raises(RestartIntentLifecycleConflict, match="generation"):
        reader.current(current)


def test_current_rejects_deleted_lifecycle():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    committed = _commit_lifecycle(store, reader, lease, lifecycle)
    store.compare_delete(
        reader.lifecycle_key,
        expected_revision=committed.revision,
    )

    with pytest.raises(RestartIntentLifecycleCorrupt, match="deleted"):
        reader.current(current)


def test_current_rejects_malformed_or_future_lifecycle():
    _, store, _, _, reader, lease, current = _state()
    store.compare_set_many_guarded(
        {
            reader.lifecycle_key: ControlStoreWrite(
                expected_revision=None,
                value=b"{}",
            )
        },
        guard_key=reader.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )
    with pytest.raises(RestartIntentLifecycleCorrupt, match="malformed"):
        reader.current(current)

    _, store, _, _, reader, lease, current = _state()
    future_head = RestartIntentHeadRecord(
        run_id=RUN_ID,
        generation=1,
        intent_id="future-intent",
        intent_digest="a" * 64,
    )
    future = RestartIntentLifecycleRecord(
        closed_intent=future_head,
        coordinator_id="coordinator-a",
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=100,
        coordinator_fencing_token=lease.fencing_token,
    )
    _commit_lifecycle(store, reader, lease, future)
    with pytest.raises(RestartIntentLifecycleCorrupt, match="current generation"):
        reader.current(current)


def test_current_rejects_missing_or_mismatched_closed_intent():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, head, lifecycle = _records(current, lease, lease)
    _commit_lifecycle(store, reader, lease, lifecycle)
    with pytest.raises(RestartIntentLifecycleCorrupt, match="missing intent"):
        reader.current(current)

    store = InMemoryControlStore(clock=lambda: 1_000)
    reader = RestartIntentLifecycleReader(store, run_id=RUN_ID)
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=lambda: 1_000,
    )
    lease = lease_manager.acquire()
    current = generation_manager.initialize(lease, _assignment())
    record, head, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    lifecycle = replace(
        lifecycle,
        closed_intent=replace(head, intent_digest="b" * 64),
    )
    _commit_lifecycle(store, reader, lease, lifecycle)
    with pytest.raises(RestartIntentLifecycleCorrupt, match="does not identify"):
        reader.current(current)


def test_current_rejects_unguarded_intent_or_lifecycle():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, _, lifecycle = _records(current, lease, lease)
    store.compare_set(
        reader.intent_key(record.intent.intent_id),
        expected_revision=None,
        value=record.to_json(),
    )
    _commit_lifecycle(store, reader, lease, lifecycle)
    with pytest.raises(RestartIntentLifecycleCorrupt, match="guard provenance"):
        reader.current(current)

    store = InMemoryControlStore(clock=lambda: 1_000)
    reader = RestartIntentLifecycleReader(store, run_id=RUN_ID)
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=lambda: 1_000,
    )
    lease = lease_manager.acquire()
    current = generation_manager.initialize(lease, _assignment())
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    store.compare_set(
        reader.lifecycle_key,
        expected_revision=None,
        value=lifecycle.to_json(),
    )
    with pytest.raises(RestartIntentLifecycleCorrupt, match="guard provenance"):
        reader.current(current)


def test_current_rejects_wrong_generation_snapshot_digest():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, head, lifecycle = _records(
        current,
        lease,
        lease,
        generation_snapshot_digest="b" * 64,
    )
    lifecycle = replace(lifecycle, closed_intent=replace(head, intent_digest=record.digest))
    _commit_intent(store, generation_manager, reader, current, lease, record)
    _commit_lifecycle(store, reader, lease, lifecycle)

    with pytest.raises(RestartIntentLifecycleCorrupt, match="generation snapshot"):
        reader.current(current)


def test_current_rejects_intent_committed_before_its_snapshot():
    clock = ManualClock()
    store = EntryOverrideStore(clock=clock)
    (
        _,
        _,
        _,
        generation_manager,
        reader,
        lease,
        current,
    ) = _state(clock=clock, store=store)
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    _commit_lifecycle(store, reader, lease, lifecycle)
    snapshot_key = generation_manager.snapshot_key(0)
    snapshot_entry = store.get(snapshot_key)
    head_entry = store.get(generation_manager.head_key)
    assert snapshot_entry is not None
    assert head_entry is not None
    store.overrides[snapshot_key] = replace(
        snapshot_entry,
        committed_at_unix_ms=1_010,
    )
    store.overrides[generation_manager.head_key] = replace(
        head_entry,
        committed_at_unix_ms=1_010,
    )
    current = generation_manager.current()
    assert current is not None

    with pytest.raises(RestartIntentLifecycleCorrupt, match="predates"):
        reader.current(current)


def test_current_rejects_intent_transaction_before_snapshot_at_same_time():
    clock = ManualClock()
    store = EntryOverrideStore(clock=clock)
    (
        _,
        _,
        _,
        generation_manager,
        reader,
        lease,
        current,
    ) = _state(clock=clock, store=store)
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    _commit_lifecycle(store, reader, lease, lifecycle)
    intent_entry = store.get(reader.intent_key(record.intent.intent_id))
    snapshot_key = generation_manager.snapshot_key(0)
    snapshot_entry = store.get(snapshot_key)
    head_entry = store.get(generation_manager.head_key)
    assert intent_entry is not None
    assert snapshot_entry is not None
    assert head_entry is not None
    later_sequence = intent_entry.transaction_sequence + 1
    store.overrides[snapshot_key] = replace(
        snapshot_entry,
        transaction_sequence=later_sequence,
    )
    store.overrides[generation_manager.head_key] = replace(
        head_entry,
        transaction_sequence=later_sequence,
    )
    current = generation_manager.current()
    assert current is not None

    with pytest.raises(RestartIntentLifecycleCorrupt, match="predates"):
        reader.current(current)


def test_current_rejects_closure_transaction_before_intent_at_same_time():
    clock = ManualClock()
    store = EntryOverrideStore(clock=clock)
    (
        _,
        _,
        _,
        generation_manager,
        reader,
        lease,
        current,
    ) = _state(clock=clock, store=store)
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    _commit_lifecycle(store, reader, lease, lifecycle)
    intent_entry = store.get(reader.intent_key(record.intent.intent_id))
    lifecycle_entry = store.get(reader.lifecycle_key)
    assert intent_entry is not None
    assert lifecycle_entry is not None
    store.overrides[reader.lifecycle_key] = replace(
        lifecycle_entry,
        transaction_sequence=intent_entry.transaction_sequence,
    )

    with pytest.raises(RestartIntentLifecycleCorrupt, match="predates intent opening"):
        reader.current(current)


def test_current_rejects_closure_guarded_by_an_older_lease_mutation():
    clock = ManualClock()
    store = EntryOverrideStore(clock=clock)
    (
        _,
        _,
        lease_manager,
        generation_manager,
        reader,
        lease,
        current,
    ) = _state(clock=clock, store=store)
    renewed = lease_manager.renew(lease)
    record, _, lifecycle = _records(current, renewed, renewed)
    _commit_intent(
        store,
        generation_manager,
        reader,
        current,
        renewed,
        record,
    )
    _commit_lifecycle(store, reader, renewed, lifecycle)
    lifecycle_entry = store.get(reader.lifecycle_key)
    assert lifecycle_entry is not None
    older_lifecycle = replace(
        lifecycle,
        coordinator_fencing_token=lease.fencing_token,
    )
    store.overrides[reader.lifecycle_key] = replace(
        lifecycle_entry,
        value=older_lifecycle.to_json(),
        guard_revision=lease.fencing_token,
        guard_mutation_sequence=1,
        guard_value_sequence=1,
        guard_lifetime_sequence=1,
        guard_committed_at_unix_ms=lease.granted_at_unix_ms,
    )

    with pytest.raises(RestartIntentLifecycleCorrupt, match="predates intent opening"):
        reader.current(current)


def test_current_rejects_recreated_or_rewritten_lifecycle():
    _, store, _, generation_manager, reader, lease, current = _state()
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    first = _commit_lifecycle(store, reader, lease, lifecycle)
    store.compare_delete(reader.lifecycle_key, expected_revision=first.revision)
    _commit_lifecycle(store, reader, lease, lifecycle)
    with pytest.raises(RestartIntentLifecycleCorrupt, match="store provenance"):
        reader.current(current)

    store = InMemoryControlStore(clock=lambda: 1_000)
    reader = RestartIntentLifecycleReader(store, run_id=RUN_ID)
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=lambda: 1_000,
    )
    lease = lease_manager.acquire()
    current = generation_manager.initialize(lease, _assignment())
    record, _, lifecycle = _records(current, lease, lease)
    _commit_intent(store, generation_manager, reader, current, lease, record)
    first = _commit_lifecycle(store, reader, lease, lifecycle)
    _commit_lifecycle(
        store,
        reader,
        lease,
        lifecycle,
        expected_revision=first.revision,
    )
    with pytest.raises(RestartIntentLifecycleCorrupt, match="store provenance"):
        reader.current(current)


def test_keys_hide_plaintext_identities():
    store = InMemoryControlStore()
    reader_a = RestartIntentLifecycleReader(store, run_id="run-a")
    reader_b = RestartIntentLifecycleReader(store, run_id="run-b")

    first = reader_a.intent_key("intent-a")
    second = reader_a.intent_key("intent-b")
    third = reader_b.intent_key("intent-a")

    assert len({first, second, third}) == 3
    assert "run-a" not in first
    assert "intent-a" not in first
