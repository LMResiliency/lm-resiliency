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
    CoordinatorLeaseAuthority,
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
    CommittedInitialRestartIntentOpen,
    HeldCoordinatorLease,
    CoordinatorLeaseAuthority,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    opening_manager = _manager(store, clock, "coordinator-a")
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    open_preparer = RestartIntentOpenPreparer(store, run_id=RUN_ID, clock=clock)
    open_executor = RestartIntentOpenExecutor(store, run_id=RUN_ID)
    lease = opening_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    opened = open_executor.execute_initial_open(
        open_preparer.prepare_initial_open(lease, current, _intent())
    )
    lease_entry = store.get(opened.prepared.coordinator_lease_key)
    assert lease_entry is not None
    return clock, store, opened, lease, _authority(lease, lease_entry)


def _authority(
    lease: HeldCoordinatorLease,
    entry: ControlStoreEntry,
) -> CoordinatorLeaseAuthority:
    return CoordinatorLeaseAuthority(
        lease=lease,
        transaction_sequence=entry.transaction_sequence,
        mutation_sequence=entry.mutation_sequence,
        value_sequence=entry.value_sequence,
        lifetime_sequence=entry.lifetime_sequence,
    )


def _authority_with_sequences(
    lease: HeldCoordinatorLease,
    *,
    transaction_sequence: int,
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
) -> CoordinatorLeaseAuthority:
    return CoordinatorLeaseAuthority(
        lease=lease,
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
    )


def _prepared(
    opened: CommittedInitialRestartIntentOpen,
    chain: tuple[CoordinatorLeaseAuthority, ...],
) -> PreparedInitialRestartIntentClosure:
    lease = chain[-1].lease
    return PreparedInitialRestartIntentClosure(
        records=_records(opened, lease),
        lease_authority_chain=chain,
        not_before_unix_ms=max(
            opened.committed_at_unix_ms,
            lease.granted_at_unix_ms,
        ),
        deadline_unix_ms=lease.expires_at_unix_ms,
    )


def _initial_prepared() -> PreparedInitialRestartIntentClosure:
    _, _, opened, _, opening = _state()
    return _prepared(opened, (opening,))


def _replacement_lease(
    previous: HeldCoordinatorLease,
    *,
    coordinator_id: str,
    lease_id: str,
    granted_at_unix_ms: int,
) -> HeldCoordinatorLease:
    return HeldCoordinatorLease(
        record=CoordinatorLeaseRecord(
            run_id=RUN_ID,
            coordinator_id=coordinator_id,
            lease_id=lease_id,
            lease_duration_ms=100,
        ),
        fencing_token=previous.fencing_token + 1,
        granted_at_unix_ms=granted_at_unix_ms,
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
    with pytest.raises(TypeError, match="must be tuple"):
        replace(prepared, lease_authority_chain=[])
    with pytest.raises(TypeError, match="CoordinatorLeaseAuthority"):
        replace(prepared, lease_authority_chain=({},))
    with pytest.raises(ValueError, match="must not be empty"):
        replace(prepared, lease_authority_chain=())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("transaction_sequence", 0),
        ("mutation_sequence", True),
        ("value_sequence", 0),
        ("lifetime_sequence", 0),
    ],
)
def test_coordinator_lease_authority_requires_positive_sequences(field, value):
    authority = _initial_prepared().lease_authority

    with pytest.raises(ValueError, match="positive integer"):
        replace(authority, **{field: value})


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("not_before_unix_ms", 999, "cannot precede its open"),
        ("deadline_unix_ms", 1_101, "exceeds its coordinator lease"),
    ],
)
def test_prepared_initial_closure_rejects_invalid_time_window(field, value, message):
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


def test_prepared_initial_closure_requires_exact_opening_authority():
    prepared = _initial_prepared()
    changed = replace(
        prepared.lease_authority,
        transaction_sequence=prepared.lease_authority.transaction_sequence + 1,
    )

    with pytest.raises(ValueError, match="does not begin"):
        replace(prepared, lease_authority_chain=(changed,))


def test_prepared_initial_closure_accepts_nonexpired_renewal():
    clock, store, opened, lease, opening = _state()
    clock.set(1_010)
    renewed = _manager(store, clock, "coordinator-a").renew(lease)
    entry = store.get(opened.prepared.coordinator_lease_key)
    assert entry is not None

    prepared = _prepared(opened, (opening, _authority(renewed, entry)))

    assert prepared.lease == renewed
    assert prepared.coordinator_lease_mutation_sequence == 2
    assert prepared.coordinator_lease_value_sequence == 1


def test_prepared_initial_closure_rejects_expired_renewal():
    prepared = _initial_prepared()
    expired = replace(
        prepared.lease,
        fencing_token=prepared.lease.fencing_token + 1,
        granted_at_unix_ms=prepared.lease.expires_at_unix_ms,
    )
    renewed = _authority_with_sequences(
        expired,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
        value_sequence=prepared.coordinator_lease_value_sequence,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    with pytest.raises(ValueError, match="expired"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, renewed),
        )


def test_prepared_initial_closure_accepts_replacement_after_post_open_renewal():
    clock, store, opened, lease, opening = _state()
    clock.set(1_010)
    renewed = _manager(store, clock, "coordinator-a").renew(lease)
    renewed_entry = store.get(opened.prepared.coordinator_lease_key)
    assert renewed_entry is not None
    clock.set(renewed.expires_at_unix_ms)
    replacement = _manager(store, clock, "coordinator-b").acquire()
    replacement_entry = store.get(opened.prepared.coordinator_lease_key)
    assert replacement_entry is not None

    prepared = _prepared(
        opened,
        (
            opening,
            _authority(renewed, renewed_entry),
            _authority(replacement, replacement_entry),
        ),
    )

    assert prepared.lease == replacement


def test_prepared_initial_closure_accepts_multiple_coordinator_replacements():
    clock, store, opened, lease, opening = _state()
    clock.set(lease.expires_at_unix_ms)
    replacement_b = _manager(store, clock, "coordinator-b").acquire()
    entry_b = store.get(opened.prepared.coordinator_lease_key)
    assert entry_b is not None
    clock.set(replacement_b.expires_at_unix_ms)
    replacement_c = _manager(store, clock, "coordinator-c").acquire()
    entry_c = store.get(opened.prepared.coordinator_lease_key)
    assert entry_c is not None

    prepared = _prepared(
        opened,
        (
            opening,
            _authority(replacement_b, entry_b),
            _authority(replacement_c, entry_c),
        ),
    )

    assert prepared.lease == replacement_c


def test_prepared_initial_closure_rejects_overlapping_replacement():
    prepared = _initial_prepared()
    replacement = _replacement_lease(
        prepared.lease,
        coordinator_id="coordinator-b",
        lease_id="lease-b",
        granted_at_unix_ms=1_010,
    )
    authority = _authority_with_sequences(
        replacement,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
        value_sequence=prepared.coordinator_lease_value_sequence + 1,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    with pytest.raises(ValueError, match="overlap"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, authority),
        )


def test_prepared_initial_closure_rejects_replacement_fencing_token_reuse():
    prepared = _initial_prepared()
    replacement = _replacement_lease(
        prepared.lease,
        coordinator_id="coordinator-b",
        lease_id="lease-b",
        granted_at_unix_ms=prepared.lease.expires_at_unix_ms,
    )
    replacement = replace(
        replacement,
        fencing_token=prepared.lease.fencing_token,
    )
    authority = _authority_with_sequences(
        replacement,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
        value_sequence=prepared.coordinator_lease_value_sequence + 1,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    with pytest.raises(ValueError, match="reuses its fencing token"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, authority),
        )


def test_prepared_initial_closure_rejects_omitted_replacement_authority():
    prepared = _initial_prepared()
    replacement_c = _replacement_lease(
        prepared.lease,
        coordinator_id="coordinator-c",
        lease_id="lease-c",
        granted_at_unix_ms=1_200,
    )
    authority = _authority_with_sequences(
        replacement_c,
        transaction_sequence=prepared.records.opened.transaction_sequence + 2,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 2,
        value_sequence=prepared.coordinator_lease_value_sequence + 2,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    with pytest.raises(ValueError, match="not one lease-key mutation"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, authority),
        )


def test_prepared_initial_closure_rejects_lease_identity_reappearing():
    prepared = _initial_prepared()
    replacement_b = _replacement_lease(
        prepared.lease,
        coordinator_id="coordinator-b",
        lease_id="lease-b",
        granted_at_unix_ms=prepared.lease.expires_at_unix_ms,
    )
    authority_b = _authority_with_sequences(
        replacement_b,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 1,
        value_sequence=prepared.coordinator_lease_value_sequence + 1,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )
    replayed_a = HeldCoordinatorLease(
        record=prepared.lease.record,
        fencing_token=replacement_b.fencing_token + 1,
        granted_at_unix_ms=replacement_b.expires_at_unix_ms,
    )
    authority_a = _authority_with_sequences(
        replayed_a,
        transaction_sequence=authority_b.transaction_sequence + 1,
        mutation_sequence=authority_b.mutation_sequence + 1,
        value_sequence=authority_b.value_sequence + 1,
        lifetime_sequence=authority_b.lifetime_sequence,
    )

    with pytest.raises(ValueError, match="lease identity reappears"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, authority_b, authority_a),
        )


def test_prepared_initial_closure_rejects_impossible_transaction_gap():
    prepared = _initial_prepared()
    renewed = replace(
        prepared.lease,
        fencing_token=prepared.lease.fencing_token + 1,
        granted_at_unix_ms=1_001,
    )
    authority = _authority_with_sequences(
        renewed,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence + 4,
        value_sequence=prepared.coordinator_lease_value_sequence,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )

    with pytest.raises(ValueError, match="transaction ordering"):
        _prepared(
            prepared.records.opened,
            (*prepared.lease_authority_chain, authority),
        )
