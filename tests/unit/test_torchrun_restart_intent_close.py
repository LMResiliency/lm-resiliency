"""Contract tests for initial restart-intent closure authority."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close import (
    PreparedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
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
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
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


def _assignment() -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=0,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
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
        prepare_deadline_unix_ms=1_050,
    )


def _records(
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
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
        closure_index=1,
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
            closure_index=1,
            generation=lifecycle_head.generation,
            intent_id=lifecycle_head.intent_id,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=opened.prepared.intent_key,
        intent_head_key=opened.prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/1",
        lifecycle_head_key=opened.prepared.lifecycle_head_key,
    )


def _state() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    CommittedInitialRestartIntentOpen,
    HeldCoordinatorLease,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    open_preparer = RestartIntentOpenPreparer(store, run_id=RUN_ID, clock=clock)
    open_executor = RestartIntentOpenExecutor(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    opened = open_executor.execute_initial_open(
        open_preparer.prepare_initial_open(lease, current, _intent())
    )
    return clock, store, lease_manager, opened, lease


def _prepared(
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
    lease_entry: ControlStoreEntry,
) -> PreparedInitialRestartIntentClosure:
    return PreparedInitialRestartIntentClosure(
        records=_records(opened, lease),
        lease=lease,
        coordinator_lease_transaction_sequence=lease_entry.transaction_sequence,
        coordinator_lease_mutation_sequence=lease_entry.mutation_sequence,
        coordinator_lease_value_sequence=lease_entry.value_sequence,
        coordinator_lease_lifetime_sequence=lease_entry.lifetime_sequence,
        not_before_unix_ms=max(
            opened.committed_at_unix_ms,
            lease.granted_at_unix_ms,
        ),
        deadline_unix_ms=lease.expires_at_unix_ms,
    )


def _initial_prepared() -> PreparedInitialRestartIntentClosure:
    _, store, _, opened, lease = _state()
    lease_entry = store.get(opened.prepared.coordinator_lease_key)
    assert lease_entry is not None
    return _prepared(opened, lease, lease_entry)


def _with_lease(
    prepared: PreparedInitialRestartIntentClosure,
    lease: HeldCoordinatorLease,
    *,
    transaction_sequence: int,
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
) -> PreparedInitialRestartIntentClosure:
    return PreparedInitialRestartIntentClosure(
        records=_records(prepared.records.opened, lease),
        lease=lease,
        coordinator_lease_transaction_sequence=transaction_sequence,
        coordinator_lease_mutation_sequence=mutation_sequence,
        coordinator_lease_value_sequence=value_sequence,
        coordinator_lease_lifetime_sequence=lifetime_sequence,
        not_before_unix_ms=max(
            prepared.records.opened.committed_at_unix_ms,
            lease.granted_at_unix_ms,
        ),
        deadline_unix_ms=lease.expires_at_unix_ms,
    )


def test_prepared_initial_closure_delegates_immutable_transaction_inputs():
    prepared = _initial_prepared()

    assert prepared.coordinator_lease_key == prepared.records.opened.prepared.coordinator_lease_key
    assert prepared.expected_guard_revision == prepared.lease.fencing_token
    assert prepared.writes == prepared.records.writes
    assert prepared.conditions == prepared.records.conditions
    with pytest.raises(TypeError):
        cast(Any, prepared.writes)["other"] = next(iter(prepared.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, prepared.conditions)["other"] = 1
    with pytest.raises(AttributeError):
        prepared.deadline_unix_ms = 1


def test_prepared_initial_closure_requires_expected_types():
    prepared = _initial_prepared()

    with pytest.raises(TypeError, match="InitialRestartIntentClosureRecords"):
        replace(prepared, records={})
    with pytest.raises(TypeError, match="HeldCoordinatorLease"):
        replace(prepared, lease={})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("coordinator_lease_transaction_sequence", 0, "positive integer"),
        ("coordinator_lease_mutation_sequence", True, "positive integer"),
        ("coordinator_lease_value_sequence", 0, "positive integer"),
        ("coordinator_lease_lifetime_sequence", 0, "positive integer"),
        ("not_before_unix_ms", 999, "cannot precede its open"),
        ("deadline_unix_ms", 1_101, "exceeds its coordinator lease"),
    ],
)
def test_prepared_initial_closure_rejects_invalid_authority_fields(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(_initial_prepared(), **{field: value})


def test_prepared_initial_closure_rejects_lifecycle_lease_mismatch():
    prepared = _initial_prepared()
    mismatched = replace(
        prepared.records.lifecycle,
        coordinator_fencing_token=prepared.lease.fencing_token + 1,
    )
    lifecycle_head = replace(
        prepared.records.lifecycle_head,
        lifecycle_digest=mismatched.digest,
    )
    records = replace(
        prepared.records,
        lifecycle=mismatched,
        lifecycle_head=lifecycle_head,
        closed_head=replace(
            prepared.records.closed_head,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
    )

    with pytest.raises(ValueError, match="does not authorize"):
        replace(prepared, records=records)


def test_prepared_initial_closure_rejects_changed_same_token_authority():
    prepared = _initial_prepared()

    with pytest.raises(ValueError, match="changes one fencing token"):
        replace(
            prepared,
            coordinator_lease_transaction_sequence=(
                prepared.coordinator_lease_transaction_sequence + 1
            ),
        )


def test_prepared_initial_closure_accepts_nonexpired_renewal():
    clock, store, lease_manager, opened, lease = _state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    lease_entry = store.get(opened.prepared.coordinator_lease_key)
    assert lease_entry is not None

    prepared = _prepared(opened, renewed, lease_entry)

    assert prepared.lease == renewed
    assert prepared.coordinator_lease_mutation_sequence == 2
    assert prepared.coordinator_lease_value_sequence == 1
    assert prepared.coordinator_lease_lifetime_sequence == 1


def test_prepared_initial_closure_rejects_expired_renewal():
    prepared = _initial_prepared()
    expired_renewal = replace(
        prepared.lease,
        fencing_token=prepared.lease.fencing_token + 1,
        granted_at_unix_ms=prepared.lease.expires_at_unix_ms,
    )

    with pytest.raises(ValueError, match="expired"):
        _with_lease(
            prepared,
            expired_renewal,
            transaction_sequence=prepared.records.opened.transaction_sequence + 1,
            mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
            value_sequence=prepared.coordinator_lease_value_sequence,
            lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
        )


def _replacement(
    prepared: PreparedInitialRestartIntentClosure,
    *,
    granted_at_unix_ms: int,
) -> HeldCoordinatorLease:
    return HeldCoordinatorLease(
        record=CoordinatorLeaseRecord(
            run_id=RUN_ID,
            coordinator_id="coordinator-b",
            lease_id="lease-b",
            lease_duration_ms=100,
        ),
        fencing_token=prepared.lease.fencing_token + 1,
        granted_at_unix_ms=granted_at_unix_ms,
    )


def test_prepared_initial_closure_rejects_overlapping_replacement():
    prepared = _initial_prepared()

    with pytest.raises(ValueError, match="overlap"):
        _with_lease(
            prepared,
            _replacement(prepared, granted_at_unix_ms=1_010),
            transaction_sequence=prepared.records.opened.transaction_sequence + 1,
            mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
            value_sequence=prepared.coordinator_lease_value_sequence + 1,
            lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
        )


def test_prepared_initial_closure_accepts_nonoverlapping_replacement():
    prepared = _initial_prepared()
    replacement = _replacement(
        prepared,
        granted_at_unix_ms=prepared.lease.expires_at_unix_ms,
    )

    replaced = _with_lease(
        prepared,
        replacement,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
        value_sequence=prepared.coordinator_lease_value_sequence + 1,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    assert replaced.lease == replacement


def test_prepared_initial_closure_rejects_impossible_mutation_gap():
    prepared = _initial_prepared()
    renewed = replace(
        prepared.lease,
        fencing_token=prepared.lease.fencing_token + 1,
        granted_at_unix_ms=1_001,
    )

    with pytest.raises(ValueError, match="sequence deltas"):
        _with_lease(
            prepared,
            renewed,
            transaction_sequence=prepared.records.opened.transaction_sequence + 1,
            mutation_sequence=prepared.coordinator_lease_mutation_sequence + 2,
            value_sequence=prepared.coordinator_lease_value_sequence,
            lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
        )
