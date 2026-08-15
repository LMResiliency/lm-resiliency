"""Contract tests for initial restart-intent closure authority."""

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
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryReader,
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


def _intent(*, deadline_unix_ms: int = 1_050) -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=deadline_unix_ms,
    )


def _open_state(
    *,
    replace_before_open: bool = False,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    CommittedInitialRestartIntentOpen,
    HeldCoordinatorLease,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first_manager = _manager(store, clock, "coordinator-a")
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease = first_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    if replace_before_open:
        clock.set(lease.expires_at_unix_ms)
        lease = _manager(store, clock, "coordinator-b").acquire()
    current = generation_manager.current()
    assert current is not None
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(store, run_id=RUN_ID, clock=clock).prepare_initial_open(
            lease,
            current,
            _intent(deadline_unix_ms=clock.now_unix_ms + 200),
        )
    )
    return clock, store, opened, lease


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


def _opening_prepared() -> PreparedInitialRestartIntentClosure:
    _, store, opened, _ = _open_state()
    return _prepared(
        opened,
        CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read(),
    )


def test_prepared_initial_closure_delegates_immutable_transaction_inputs():
    prepared = _opening_prepared()

    assert prepared.expected_guard_revision == prepared.lease.fencing_token
    assert prepared.writes == prepared.records.writes
    assert prepared.conditions == prepared.records.conditions
    with pytest.raises(TypeError):
        cast(Any, prepared.writes)["other"] = next(iter(prepared.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, prepared.conditions)["other"] = 1
    with pytest.raises(AttributeError):
        prepared.deadline_unix_ms = 1


def test_prepared_initial_closure_requires_expected_types_and_opening_authority():
    prepared = _opening_prepared()

    with pytest.raises(TypeError, match="InitialRestartIntentClosureRecords"):
        replace(prepared, records={})
    with pytest.raises(TypeError, match="must be tuple"):
        replace(prepared, lease_authority_chain=[])
    with pytest.raises(TypeError, match="CoordinatorLeaseAuthority"):
        replace(prepared, lease_authority_chain=({},))
    with pytest.raises(ValueError, match="must not be empty"):
        replace(prepared, lease_authority_chain=())
    changed = replace(
        prepared.lease_authority,
        transaction_sequence=prepared.lease_authority.transaction_sequence + 1,
    )
    with pytest.raises(ValueError, match="does not begin"):
        replace(prepared, lease_authority_chain=(changed,))


def test_prepared_initial_closure_rejects_invalid_window_or_lifecycle_authority():
    prepared = _opening_prepared()

    with pytest.raises(ValueError, match="cannot precede its open"):
        replace(
            prepared,
            not_before_unix_ms=prepared.records.opened.committed_at_unix_ms - 1,
        )
    with pytest.raises(ValueError, match="exceeds its coordinator lease"):
        replace(prepared, deadline_unix_ms=prepared.lease.expires_at_unix_ms + 1)
    lifecycle = replace(
        prepared.records.lifecycle,
        coordinator_fencing_token=prepared.lease.fencing_token + 1,
    )
    lifecycle_head = replace(
        prepared.records.lifecycle_head,
        lifecycle_digest=lifecycle.digest,
    )
    records = replace(
        prepared.records,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=replace(
            prepared.records.closed_head,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
    )
    with pytest.raises(ValueError, match="does not authorize"):
        replace(prepared, records=records)


def test_prepared_initial_closure_accepts_renewal_and_multiple_replacements():
    clock, store, opened, lease = _open_state()
    clock.set(1_010)
    renewed = _manager(store, clock, "coordinator-a").renew(lease)
    clock.set(renewed.expires_at_unix_ms)
    replacement_b = _manager(store, clock, "coordinator-b").acquire()
    clock.set(replacement_b.expires_at_unix_ms)
    replacement_c = _manager(store, clock, "coordinator-c").acquire()

    prepared = _prepared(
        opened,
        CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read(),
    )

    assert prepared.lease == replacement_c
    assert len(prepared.lease_authority_chain) == 4


def test_prepared_initial_closure_accepts_delete_and_recreate():
    clock, store, opened, lease = _open_state()
    clock.set(1_010)
    _manager(store, clock, "coordinator-a").release(lease)
    replacement = _manager(store, clock, "coordinator-b").acquire()

    prepared = _prepared(
        opened,
        CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read(),
    )

    assert prepared.lease == replacement
    assert prepared.coordinator_lease_lifetime_sequence == 2


def test_prepared_initial_closure_rejects_expired_renewal_and_overlap():
    prepared = _opening_prepared()
    opening = prepared.lease_authority
    expired_lease = replace(
        opening.lease,
        fencing_token=opening.lease.fencing_token + 1,
        granted_at_unix_ms=opening.lease.expires_at_unix_ms,
    )
    expired = CoordinatorLeaseAuthority(
        lease=expired_lease,
        transaction_sequence=prepared.records.opened.transaction_sequence + 1,
        mutation_sequence=opening.mutation_sequence + 1,
        value_sequence=opening.value_sequence,
        lifetime_sequence=opening.lifetime_sequence,
    )
    with pytest.raises(ValueError, match="expired"):
        _prepared(prepared.records.opened, (opening, expired))

    _, store, opened, lease = _open_state()
    clock_entry = CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read()[0]
    overlapping_lease = HeldCoordinatorLease(
        record=replace(
            lease.record,
            coordinator_id="coordinator-b",
            lease_id="lease-b",
        ),
        fencing_token=lease.fencing_token + 1,
        granted_at_unix_ms=lease.granted_at_unix_ms + 10,
    )
    overlapping = CoordinatorLeaseAuthority(
        lease=overlapping_lease,
        transaction_sequence=opened.transaction_sequence + 1,
        mutation_sequence=clock_entry.mutation_sequence + 1,
        value_sequence=clock_entry.value_sequence + 1,
        lifetime_sequence=clock_entry.lifetime_sequence,
    )
    with pytest.raises(ValueError, match="overlap"):
        _prepared(opened, (clock_entry, overlapping))


def test_prepared_initial_closure_rejects_opening_lease_replay():
    clock, store, opened, opening_lease = _open_state(replace_before_open=True)
    history = CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read()
    opening = history[-1]
    assert opening.lease == opening_lease
    clock.set(opening_lease.expires_at_unix_ms)
    replacement = _manager(store, clock, "coordinator-c").acquire()
    history = CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read()
    replacement_authority = history[-1]
    replayed_lease = HeldCoordinatorLease(
        record=opening_lease.record,
        fencing_token=replacement.fencing_token + 1,
        granted_at_unix_ms=replacement.expires_at_unix_ms,
    )
    replayed = CoordinatorLeaseAuthority(
        lease=replayed_lease,
        transaction_sequence=replacement_authority.transaction_sequence + 1,
        mutation_sequence=replacement_authority.mutation_sequence + 1,
        value_sequence=replacement_authority.value_sequence + 1,
        lifetime_sequence=replacement_authority.lifetime_sequence,
    )

    with pytest.raises(ValueError, match="lease identity reappears"):
        _prepared(
            opened,
            (opening, replacement_authority, replayed),
        )


def test_prepared_initial_closure_rejects_omitted_transition():
    prepared = _opening_prepared()
    opening = prepared.lease_authority
    skipped_lease = HeldCoordinatorLease(
        record=replace(
            opening.lease.record,
            coordinator_id="coordinator-c",
            lease_id="lease-c",
        ),
        fencing_token=opening.lease.fencing_token + 2,
        granted_at_unix_ms=opening.lease.expires_at_unix_ms + 100,
    )
    skipped = CoordinatorLeaseAuthority(
        lease=skipped_lease,
        transaction_sequence=prepared.records.opened.transaction_sequence + 2,
        mutation_sequence=opening.mutation_sequence + 2,
        value_sequence=opening.value_sequence + 2,
        lifetime_sequence=opening.lifetime_sequence,
    )

    with pytest.raises(ValueError, match="not one lease-key mutation"):
        _prepared(prepared.records.opened, (opening, skipped))
