"""Contract tests for persisted initial restart-intent closure reads."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseHistoryError,
)
from lm_resiliency.integrations.torchrun._generation_reader import GenerationStateError
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_reader import (
    InitialRestartIntentLifecycleReader,
    RestartIntentLifecycleReadCorrupt,
    RestartIntentLifecycleReadError,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
    RestartIntentOpenExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle_reader import (
    RestartPlanPublicationLifecycleConflict,
    RestartPlanPublicationLifecycleCorrupt,
    RestartPlanPublicationLifecycleReader,
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


class FailingLifecycleReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def read(self):
        raise self._error


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    coordinator_id: str,
) -> CoordinatorLeaseManager:
    return CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id=coordinator_id,
        lease_duration_ms=100,
        clock=clock,
    )


def _assignment(
    generation: int = 0,
    *,
    node_id: str = "node-b",
) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, node_id, 2, 2),
        ),
        topology_digest="topology-v1",
    )


def _intent() -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=1_200,
    )


def _open_state() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    GenerationStateManager,
    HeldCoordinatorLease,
    CommittedInitialRestartIntentOpen,
    InitialRestartIntentLifecycleReader,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = _manager(store, clock, "coordinator-a")
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    prepared = RestartIntentOpenPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_open(lease, current, _intent())
    opened = RestartIntentOpenExecutor(
        store,
        run_id=RUN_ID,
    ).execute_initial_open(prepared)
    return (
        clock,
        store,
        lease_manager,
        generation_manager,
        lease,
        opened,
        InitialRestartIntentLifecycleReader(store, run_id=RUN_ID),
    )


def _records(
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
    *,
    closure_index: int = 1,
) -> InitialRestartIntentClosureRecords:
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=opened.prepared.head,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    lifecycle_head = RestartIntentLifecycleHeadRecord(
        run_id=RUN_ID,
        closure_index=closure_index,
        generation=opened.prepared.record.intent.generation,
        intent_id=opened.prepared.record.intent.intent_id,
        lifecycle_digest=lifecycle.digest,
    )
    run_prefix = opened.prepared.intent_head_key.rsplit("/", 1)[0]
    return InitialRestartIntentClosureRecords(
        opened=opened,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=RestartIntentClosedHeadRecord(
            run_id=RUN_ID,
            closure_index=closure_index,
            generation=lifecycle_head.generation,
            intent_id=lifecycle_head.intent_id,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=opened.prepared.intent_key,
        intent_head_key=opened.prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/{closure_index}",
        lifecycle_head_key=opened.prepared.lifecycle_head_key,
    )


def _commit_closure(
    clock: ManualClock,
    store: InMemoryControlStore,
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
) -> InitialRestartIntentClosureRecords:
    records = _records(opened, lease)
    clock.set(max(clock.now_unix_ms, lease.granted_at_unix_ms, 1_010))
    store.compare_set_many_guarded(
        records.writes,
        guard_key=opened.prepared.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
        conditions=records.conditions,
    )
    return records


def _replace_entry(
    store: InMemoryControlStore,
    key: str,
    **changes,
) -> None:
    updated = replace(store._entries[key], **changes)
    store._entries[key] = updated
    store._histories[key][-1] = updated


def test_lifecycle_reader_returns_none_before_closure():
    store = InMemoryControlStore(clock=ManualClock())
    reader = InitialRestartIntentLifecycleReader(store, run_id=RUN_ID)

    assert reader.read() is None

    _, _, _, _, _, _, open_reader = _open_state()
    assert open_reader.read() is None


def test_publication_lifecycle_reader_returns_exact_fence():
    clock, store, _, _, lease, opened, _ = _open_state()
    records = _commit_closure(clock, store, opened, lease)

    fence = RestartPlanPublicationLifecycleReader(
        store,
        run_id=RUN_ID,
    ).read()

    intent_entry = store.get(records.intent_key)
    intent_head_entry = store.get(records.intent_head_key)
    closure_entry = store.get(records.closure_key)
    lifecycle_head_entry = store.get(records.lifecycle_head_key)
    assert intent_entry is not None
    assert intent_head_entry is not None
    assert closure_entry is not None
    assert lifecycle_head_entry is not None
    assert fence.conditions == {
        records.intent_key: intent_entry.revision,
        records.intent_head_key: intent_head_entry.revision,
        records.closure_key: closure_entry.revision,
        records.lifecycle_head_key: lifecycle_head_entry.revision,
    }


def test_publication_lifecycle_reader_rejects_missing_or_open_intent():
    store = InMemoryControlStore(clock=ManualClock())
    reader = RestartPlanPublicationLifecycleReader(store, run_id=RUN_ID)

    with pytest.raises(RestartPlanPublicationLifecycleConflict, match="not closed"):
        reader.read()

    _, store, _, _, _, _, _ = _open_state()
    with pytest.raises(RestartPlanPublicationLifecycleConflict, match="not closed"):
        RestartPlanPublicationLifecycleReader(store, run_id=RUN_ID).read()


def test_publication_lifecycle_reader_rejects_existing_successor():
    clock, store, _, generation_manager, lease, opened, _ = _open_state()
    _commit_closure(clock, store, opened, lease)
    current = generation_manager.current()
    assert current is not None
    clock.set(1_020)
    generation_manager.commit_successor(
        lease,
        current,
        _assignment(generation=1, node_id="node-c"),
    )

    with pytest.raises(RestartPlanPublicationLifecycleConflict, match="already committed"):
        RestartPlanPublicationLifecycleReader(store, run_id=RUN_ID).read()


def test_publication_lifecycle_reader_translates_corruption():
    clock, store, _, _, lease, opened, _ = _open_state()
    records = _commit_closure(clock, store, opened, lease)
    closure_entry = store.get(records.closure_key)
    assert closure_entry is not None
    store.compare_set(
        records.closure_key,
        expected_revision=closure_entry.revision,
        value=b"{}",
    )

    with pytest.raises(RestartPlanPublicationLifecycleCorrupt, match="lifecycle is corrupt"):
        RestartPlanPublicationLifecycleReader(store, run_id=RUN_ID).read()


def test_publication_lifecycle_reader_translates_contention():
    reader = RestartPlanPublicationLifecycleReader(
        InMemoryControlStore(clock=ManualClock()),
        run_id=RUN_ID,
    )
    cast(Any, reader)._lifecycle_reader = FailingLifecycleReader(
        RestartIntentLifecycleReadError("changed repeatedly")
    )

    with pytest.raises(RestartPlanPublicationLifecycleConflict, match="changed repeatedly"):
        reader.read()


@pytest.mark.parametrize(
    "dependency_error",
    [
        CoordinatorLeaseHistoryError("lease history changed"),
        GenerationStateError("generation changed"),
    ],
)
def test_publication_lifecycle_reader_translates_dependency_contention(
    dependency_error: RuntimeError,
):
    reader = RestartPlanPublicationLifecycleReader(
        InMemoryControlStore(clock=ManualClock()),
        run_id=RUN_ID,
    )
    cast(Any, reader)._lifecycle_reader = FailingLifecycleReader(dependency_error)

    with pytest.raises(
        RestartPlanPublicationLifecycleConflict,
        match="dependencies changed repeatedly",
    ):
        reader.read()


def test_lifecycle_reader_rejects_closed_head_without_lifecycle():
    _, store, _, _, lease, opened, reader = _open_state()
    records = _records(opened, lease)
    store.compare_set(
        records.intent_head_key,
        expected_revision=opened.head_entry.revision,
        value=records.closed_head.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="no lifecycle"):
        reader.read()


def test_lifecycle_reader_rejects_rewritten_open_head_without_lifecycle():
    _, store, _, _, _, opened, reader = _open_state()
    store.compare_set(
        opened.prepared.intent_head_key,
        expected_revision=opened.head_entry.revision,
        value=opened.prepared.head.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="initial creation"):
        reader.read()


def test_lifecycle_reader_rejects_orphaned_closure_without_lifecycle_head():
    _, store, _, _, lease, opened, reader = _open_state()
    records = _records(opened, lease)
    store.compare_set(
        records.closure_key,
        expected_revision=None,
        value=records.lifecycle.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="without a lifecycle"):
        reader.read()


def test_lifecycle_reader_rejects_missing_open_intent_without_lifecycle():
    _, store, _, _, _, opened, reader = _open_state()
    intent_entry = store.get(opened.prepared.intent_key)
    assert intent_entry is not None
    store.compare_delete(
        opened.prepared.intent_key,
        expected_revision=intent_entry.revision,
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="intent is missing"):
        reader.read()


def test_lifecycle_reader_rejects_rewritten_open_intent_without_lifecycle():
    _, store, _, _, _, opened, reader = _open_state()
    intent_entry = store.get(opened.prepared.intent_key)
    assert intent_entry is not None
    store.compare_set(
        opened.prepared.intent_key,
        expected_revision=intent_entry.revision,
        value=opened.prepared.record.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="immutable retained"):
        reader.read()


def test_lifecycle_reader_rejects_split_open_transaction_without_lifecycle():
    _, store, _, _, _, opened, reader = _open_state()
    intent_entry = store.get(opened.prepared.intent_key)
    assert intent_entry is not None
    _replace_entry(
        store,
        opened.prepared.intent_key,
        transaction_sequence=intent_entry.transaction_sequence + 1,
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="guarded transaction"):
        reader.read()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("guard_key", "other-guard"),
        ("guard_revision", 10_000),
        ("guard_value_digest", "0" * 64),
    ],
)
def test_lifecycle_reader_rejects_invalid_open_guard_provenance(
    field: str,
    replacement: object,
):
    _, store, _, _, _, opened, reader = _open_state()
    changes = {field: replacement}
    _replace_entry(store, opened.prepared.intent_head_key, **changes)
    _replace_entry(store, opened.prepared.intent_key, **changes)

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="lease provenance"):
        reader.read()


def test_lifecycle_reader_rejects_open_authority_absent_from_lease_history():
    _, store, _, _, _, opened, reader = _open_state()
    intent_entry = store.get(opened.prepared.intent_key)
    assert intent_entry is not None
    assert intent_entry.guard_mutation_sequence is not None
    changes = {
        "guard_mutation_sequence": intent_entry.guard_mutation_sequence + 1,
    }
    _replace_entry(store, opened.prepared.intent_head_key, **changes)
    _replace_entry(store, opened.prepared.intent_key, **changes)

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="durable lease history"):
        reader.read()


def test_lifecycle_reader_bounds_open_commit_by_next_lease_mutation():
    clock, store, lease_manager, _, lease, opened, reader = _open_state()
    clock.set(1_010)
    lease_manager.renew(lease)
    next_authority = reader._lease_history_reader.read()[-1]
    changes = {
        "committed_at_unix_ms": next_authority.lease.granted_at_unix_ms,
        "transaction_sequence": next_authority.transaction_sequence,
    }
    _replace_entry(store, opened.prepared.intent_head_key, **changes)
    _replace_entry(store, opened.prepared.intent_key, **changes)

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="lease.*window"):
        reader.read()


def test_lifecycle_reader_authenticates_generation_lease_history():
    _, store, _, generation_manager, _, _, reader = _open_state()
    snapshot_key = generation_manager.snapshot_key(0)
    snapshot_entry = store.get(snapshot_key)
    head_entry = store.get(generation_manager.head_key)
    assert snapshot_entry is not None
    assert head_entry is not None
    assert snapshot_entry.guard_mutation_sequence is not None
    changes = {
        "guard_mutation_sequence": snapshot_entry.guard_mutation_sequence + 1,
    }
    _replace_entry(store, snapshot_key, **changes)
    _replace_entry(store, generation_manager.head_key, **changes)

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="generation coordinator lease"):
        reader.read()


def test_lifecycle_reader_rejects_open_entries_without_commit_times():
    _, store, _, _, _, opened, reader = _open_state()
    _replace_entry(
        store,
        opened.prepared.intent_head_key,
        committed_at_unix_ms=None,
    )
    _replace_entry(
        store,
        opened.prepared.intent_key,
        committed_at_unix_ms=None,
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="commit time"):
        reader.read()


def test_lifecycle_reader_rejects_invalid_open_generation_binding():
    _, store, _, _, _, opened, reader = _open_state()
    record = replace(
        opened.prepared.record,
        generation_snapshot_digest="0" * 64,
    )
    head = RestartIntentHeadRecord(
        run_id=RUN_ID,
        generation=record.intent.generation,
        intent_id=record.intent.intent_id,
        intent_digest=record.digest,
    )
    _replace_entry(
        store,
        opened.prepared.intent_head_key,
        value=head.to_json(),
    )
    _replace_entry(
        store,
        opened.prepared.intent_key,
        value=record.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="generation"):
        reader.read()


def test_lifecycle_reader_rejects_open_suspects_outside_generation():
    _, store, _, _, _, opened, reader = _open_state()
    intent = replace(
        opened.prepared.record.intent,
        suspected_node_ids=("node-z",),
    )
    record = replace(opened.prepared.record, intent=intent)
    head = replace(opened.prepared.head, intent_digest=record.digest)
    _replace_entry(
        store,
        opened.prepared.intent_head_key,
        value=head.to_json(),
    )
    _replace_entry(
        store,
        opened.prepared.intent_key,
        value=record.to_json(),
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="suspects nodes outside"):
        reader.read()


def test_lifecycle_reader_reconstructs_verified_closure():
    clock, store, _, _, lease, opened, reader = _open_state()
    records = _commit_closure(clock, store, opened, lease)

    observed = reader.read()

    assert observed is not None
    assert observed.intent == opened.prepared.record
    assert observed.open_head == opened.prepared.head
    assert observed.closed_head == records.closed_head
    assert observed.lifecycle == records.lifecycle
    assert observed.lifecycle_head == records.lifecycle_head
    assert observed.opening_authority.lease == opened.prepared.lease
    assert observed.closing_authority.lease == lease
    assert observed.closed_at_unix_ms == clock.now_unix_ms


def test_lifecycle_reader_accepts_renewed_or_replaced_closing_lease():
    clock, store, lease_manager, _, lease, opened, reader = _open_state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    _commit_closure(clock, store, opened, renewed)

    observed = reader.read()
    assert observed is not None
    assert observed.closing_authority.lease == renewed

    clock, store, _, _, lease, opened, reader = _open_state()
    clock.set(lease.expires_at_unix_ms)
    replacement = _manager(store, clock, "coordinator-b").acquire()
    _commit_closure(clock, store, opened, replacement)

    observed = reader.read()
    assert observed is not None
    assert observed.closing_authority.lease == replacement


def test_lifecycle_reader_accepts_closure_after_generation_advance():
    clock, store, _, generation_manager, lease, opened, reader = _open_state()
    _commit_closure(clock, store, opened, lease)
    current = generation_manager.current()
    assert current is not None
    clock.set(1_020)
    generation_manager.commit_successor(
        lease,
        current,
        _assignment(generation=1, node_id="node-c"),
    )

    observed = reader.read()

    assert observed is not None
    assert observed.generation_snapshot.record.assignment.generation == 0


def test_lifecycle_reader_rejects_mixed_lease_and_generation_histories(
    monkeypatch,
):
    clock, store, lease_manager, generation_manager, lease, opened, reader = _open_state()
    _commit_closure(clock, store, opened, lease)
    original_current_with_history = reader._generation_reader.current_with_history
    triggered = False

    def racing_current_with_history():
        nonlocal triggered
        if not triggered:
            triggered = True
            current = generation_manager.current()
            assert current is not None
            lease_manager.release(lease)
            replacement = _manager(store, clock, "coordinator-b").acquire()
            generation_manager.commit_successor(
                replacement,
                current,
                _assignment(generation=1, node_id="node-c"),
            )
        return original_current_with_history()

    monkeypatch.setattr(
        reader._generation_reader,
        "current_with_history",
        racing_current_with_history,
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="contradictory"):
        reader.read()


def test_lifecycle_reader_retries_atomic_closure_during_read(monkeypatch):
    clock, store, _, _, lease, opened, reader = _open_state()
    records = _records(opened, lease)
    original_get = store.get
    triggered = False

    def racing_get(key: str):
        nonlocal triggered
        if key == reader.lifecycle_head_key and not triggered:
            triggered = True
            clock.set(1_010)
            store.compare_set_many_guarded(
                records.writes,
                guard_key=opened.prepared.coordinator_lease_key,
                expected_guard_revision=lease.fencing_token,
                not_before_unix_ms=1_010,
                deadline_unix_ms=lease.expires_at_unix_ms,
                conditions=records.conditions,
            )
        return original_get(key)

    monkeypatch.setattr(store, "get", racing_get)

    observed = reader.read()
    assert observed is not None
    assert observed.lifecycle == records.lifecycle


def test_lifecycle_reader_translates_invalid_persisted_closure():
    clock, store, _, _, lease, opened, reader = _open_state()
    records = _commit_closure(clock, store, opened, lease)
    _replace_entry(store, records.closure_key, value=b"invalid")

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="contradictory"):
        reader.read()


def test_lifecycle_reader_rejects_missing_or_rewritten_records():
    clock, store, _, _, lease, opened, reader = _open_state()
    records = _commit_closure(clock, store, opened, lease)
    closure_entry = store.get(records.closure_key)
    assert closure_entry is not None
    store.compare_delete(
        records.closure_key,
        expected_revision=closure_entry.revision,
    )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="closure is missing"):
        reader.read()

    clock, store, _, _, lease, opened, reader = _open_state()
    records = _commit_closure(clock, store, opened, lease)
    lifecycle_entry = store.get(records.lifecycle_head_key)
    assert lifecycle_entry is not None
    store.compare_set(
        records.lifecycle_head_key,
        expected_revision=lifecycle_entry.revision,
        value=records.lifecycle_head.to_json(),
    )
    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="immutable"):
        reader.read()


def test_lifecycle_reader_rejects_closure_outside_causal_window():
    clock, store, _, _, lease, opened, reader = _open_state()
    records = _commit_closure(clock, store, opened, lease)
    for key in (
        records.intent_head_key,
        records.lifecycle_head_key,
        records.closure_key,
    ):
        _replace_entry(
            store,
            key,
            committed_at_unix_ms=lease.expires_at_unix_ms,
        )

    with pytest.raises(RestartIntentLifecycleReadCorrupt, match="contradictory"):
        reader.read()


@pytest.mark.parametrize(
    "dependency_error",
    [
        CoordinatorLeaseHistoryError("lease history changed"),
        GenerationStateError("generation changed"),
    ],
)
def test_lifecycle_reader_preserves_retryable_dependency_errors(
    dependency_error: RuntimeError,
    monkeypatch,
):
    clock, store, _, _, lease, opened, reader = _open_state()
    _commit_closure(clock, store, opened, lease)

    def fail(*args):
        raise dependency_error

    if isinstance(dependency_error, CoordinatorLeaseHistoryError):
        monkeypatch.setattr(reader._lease_history_reader, "read", fail)
    else:
        monkeypatch.setattr(reader._generation_reader, "current_with_history", fail)

    with pytest.raises(type(dependency_error), match="changed"):
        reader.read()


def test_lifecycle_reader_is_run_scoped():
    _, store, _, _, _, _, _ = _open_state()

    assert InitialRestartIntentLifecycleReader(store, run_id="other-run").read() is None
    with pytest.raises(ValueError, match="non-empty"):
        InitialRestartIntentLifecycleReader(store, run_id="")
