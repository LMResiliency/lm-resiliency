"""Contract tests for persisted restart-intent closure authentication."""

from __future__ import annotations

import threading
from dataclasses import dataclass, replace
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
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateReader,
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_auth import (
    AuthenticatedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_state import (
    PersistedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
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


@dataclass(frozen=True, slots=True)
class ClosureFixture:
    clock: ManualClock
    store: InMemoryControlStore
    generation_manager: GenerationStateManager
    lease_manager: CoordinatorLeaseManager
    opening_lease: HeldCoordinatorLease
    state: PersistedInitialRestartIntentClosure
    generation: StoredGenerationSnapshot
    successor: StoredGenerationSnapshot | None
    lease_history: tuple[CoordinatorLeaseAuthority, ...]


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


def _assignment(generation: int = 0, *, node_id: str = "node-b") -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, node_id, 2, 2),
        ),
        topology_digest="topology-v1",
    )


def _fixture(*, closing_mode: str = "opening") -> ClosureFixture:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = _manager(store, clock, "coordinator-a")
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    opening_lease = lease_manager.acquire()
    generation_manager.initialize(opening_lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    intent = RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=1_200,
    )
    prepared = RestartIntentOpenPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_open(opening_lease, current, intent)
    opened = RestartIntentOpenExecutor(
        store,
        run_id=RUN_ID,
    ).execute_initial_open(prepared)
    closing_lease = opening_lease
    if closing_mode == "renewed":
        clock.set(1_010)
        closing_lease = lease_manager.renew(opening_lease)
    elif closing_mode == "replacement":
        clock.set(opening_lease.expires_at_unix_ms)
        closing_lease = _manager(store, clock, "coordinator-b").acquire()
    elif closing_mode != "opening":
        raise AssertionError(f"unsupported closing mode: {closing_mode}")
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=prepared.head,
        coordinator_id=closing_lease.record.coordinator_id,
        lease_id=closing_lease.record.lease_id,
        coordinator_lease_duration_ms=closing_lease.record.lease_duration_ms,
        coordinator_fencing_token=closing_lease.fencing_token,
    )
    lifecycle_head = RestartIntentLifecycleHeadRecord(
        run_id=RUN_ID,
        closure_index=1,
        generation=0,
        intent_id=intent.intent_id,
        lifecycle_digest=lifecycle.digest,
    )
    run_prefix = prepared.intent_head_key.rsplit("/", 1)[0]
    records = InitialRestartIntentClosureRecords(
        opened=opened,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=RestartIntentClosedHeadRecord(
            run_id=RUN_ID,
            closure_index=1,
            generation=0,
            intent_id=intent.intent_id,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=prepared.intent_key,
        intent_head_key=prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/1",
        lifecycle_head_key=prepared.lifecycle_head_key,
    )
    clock.set(max(clock.now_unix_ms, closing_lease.granted_at_unix_ms, 1_010))
    committed = store.compare_set_many_guarded(
        records.writes,
        guard_key=prepared.coordinator_lease_key,
        expected_guard_revision=closing_lease.fencing_token,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=closing_lease.expires_at_unix_ms,
        conditions=records.conditions,
    )
    head_history = store.get_history(records.intent_head_key)
    assert len(head_history) == 2
    state = PersistedInitialRestartIntentClosure.from_entries(
        run_id=RUN_ID,
        intent_entry=opened.intent_entry,
        open_head_entry=head_history[0],
        closed_head_entry=committed[records.intent_head_key],
        lifecycle_entry=committed[records.closure_key],
        lifecycle_head_entry=committed[records.lifecycle_head_key],
    )
    generation_reader = GenerationStateReader(store, run_id=RUN_ID)
    generation = generation_reader.get(0)
    assert generation is not None
    return ClosureFixture(
        clock=clock,
        store=store,
        generation_manager=generation_manager,
        lease_manager=lease_manager,
        opening_lease=opening_lease,
        state=state,
        generation=generation,
        successor=generation_reader.get(1),
        lease_history=CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).read(),
    )


def _authenticate(
    fixture: ClosureFixture,
) -> AuthenticatedInitialRestartIntentClosure:
    return AuthenticatedInitialRestartIntentClosure(
        state=fixture.state,
        generation_snapshot=fixture.generation,
        immediate_successor=fixture.successor,
        lease_history=fixture.lease_history,
    )


@pytest.mark.parametrize("closing_mode", ["opening", "renewed", "replacement"])
def test_authenticated_closure_resolves_exact_authorities(closing_mode: str):
    fixture = _fixture(closing_mode=closing_mode)

    authenticated = _authenticate(fixture)

    assert authenticated.intent == fixture.state.intent
    assert authenticated.open_head == fixture.state.open_head
    assert authenticated.closed_head == fixture.state.closed_head
    assert authenticated.lifecycle == fixture.state.lifecycle
    assert authenticated.lifecycle_head == fixture.state.lifecycle_head
    assert authenticated.generation_authority == fixture.lease_history[0]
    assert authenticated.opening_authority == fixture.lease_history[0]
    assert authenticated.closing_authority == fixture.lease_history[-1]
    assert authenticated.closed_at_unix_ms == fixture.state.closed_at_unix_ms
    assert authenticated.transaction_sequence == fixture.state.closing_transaction_sequence


def test_authenticated_closure_accepts_immediate_generation_successor():
    fixture = _fixture()
    current = fixture.generation_manager.current()
    assert current is not None
    fixture.clock.set(1_020)
    fixture.generation_manager.commit_successor(
        fixture.opening_lease,
        current,
        _assignment(generation=1, node_id="node-c"),
    )
    successor = GenerationStateReader(fixture.store, run_id=RUN_ID).get(1)
    assert successor is not None

    authenticated = AuthenticatedInitialRestartIntentClosure(
        state=fixture.state,
        generation_snapshot=fixture.generation,
        immediate_successor=successor,
        lease_history=CoordinatorLeaseHistoryReader(
            fixture.store,
            run_id=RUN_ID,
        ).read(),
    )

    assert authenticated.immediate_successor == successor
    assert authenticated.successor_authority == authenticated.generation_authority

    with pytest.raises(ValueError, match="opening is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=replace(
                successor,
                transaction_sequence=fixture.state.opening_transaction_sequence,
            ),
            lease_history=CoordinatorLeaseHistoryReader(
                fixture.store,
                run_id=RUN_ID,
            ).read(),
        )
    with pytest.raises(ValueError, match="successor is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=replace(
                successor,
                committed_at_unix_ms=fixture.state.opened_at_unix_ms - 1,
            ),
            lease_history=CoordinatorLeaseHistoryReader(
                fixture.store,
                run_id=RUN_ID,
            ).read(),
        )


def test_authenticated_closure_authenticates_successor_lease_authority():
    fixture = _fixture()
    current = fixture.generation_manager.current()
    assert current is not None
    fixture.clock.set(1_020)
    fixture.generation_manager.commit_successor(
        fixture.opening_lease,
        current,
        _assignment(generation=1, node_id="node-c"),
    )
    successor = GenerationStateReader(fixture.store, run_id=RUN_ID).get(1)
    assert successor is not None
    lease_history = CoordinatorLeaseHistoryReader(
        fixture.store,
        run_id=RUN_ID,
    ).read()

    with pytest.raises(ValueError, match="successor authority is absent"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=replace(
                successor,
                guard_mutation_sequence=successor.guard_mutation_sequence + 1,
            ),
            lease_history=lease_history,
        )
    with pytest.raises(ValueError, match="successor is outside its lease window"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=replace(
                successor,
                transaction_sequence=lease_history[0].transaction_sequence,
            ),
            lease_history=lease_history,
        )


def test_authenticated_closure_rejects_successor_that_retains_suspect():
    fixture = _fixture()
    current = fixture.generation_manager.current()
    assert current is not None
    fixture.clock.set(1_020)
    fixture.generation_manager.commit_successor(
        fixture.opening_lease,
        current,
        _assignment(generation=1),
    )
    successor = GenerationStateReader(fixture.store, run_id=RUN_ID).get(1)
    assert successor is not None

    with pytest.raises(ValueError, match="successor retains suspected nodes"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=successor,
            lease_history=CoordinatorLeaseHistoryReader(
                fixture.store,
                run_id=RUN_ID,
            ).read(),
        )


def test_authenticated_closure_rejects_wrong_generation_or_successor():
    fixture = _fixture()

    with pytest.raises(ValueError, match="wrong generation"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=replace(
                fixture.generation,
                record=replace(
                    fixture.generation.record,
                    assignment=_assignment(generation=1, node_id="node-c"),
                    previous_snapshot_digest="0" * 64,
                ),
            ),
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )

    with pytest.raises(ValueError, match="immediate successor"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=fixture.generation,
            lease_history=fixture.lease_history,
        )


def test_authenticated_closure_rejects_suspects_outside_generation():
    fixture = _fixture()
    intent = replace(
        fixture.state.intent.intent,
        suspected_node_ids=("node-z",),
    )
    record = replace(fixture.state.intent, intent=intent)
    open_head = replace(fixture.state.open_head, intent_digest=record.digest)
    lifecycle = replace(fixture.state.lifecycle, closed_intent=open_head)
    lifecycle_head = replace(
        fixture.state.lifecycle_head,
        lifecycle_digest=lifecycle.digest,
    )
    closed_head = replace(
        fixture.state.closed_head,
        lifecycle_head_digest=lifecycle_head.digest,
    )
    state = PersistedInitialRestartIntentClosure(
        intent=record,
        open_head=open_head,
        closed_head=closed_head,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        intent_entry=replace(
            fixture.state.intent_entry,
            value=record.to_json(),
        ),
        open_head_entry=replace(
            fixture.state.open_head_entry,
            value=open_head.to_json(),
        ),
        closed_head_entry=replace(
            fixture.state.closed_head_entry,
            value=closed_head.to_json(),
        ),
        lifecycle_entry=replace(
            fixture.state.lifecycle_entry,
            value=lifecycle.to_json(),
        ),
        lifecycle_head_entry=replace(
            fixture.state.lifecycle_head_entry,
            value=lifecycle_head.to_json(),
        ),
    )

    with pytest.raises(ValueError, match="suspects nodes outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )


def test_authenticated_closure_requires_exact_ordered_authorities():
    fixture = _fixture(closing_mode="replacement")

    with pytest.raises(ValueError, match="generation authority is absent"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=(),
        )

    with pytest.raises(ValueError, match="out of order"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=tuple(reversed(fixture.lease_history)),
        )


def test_authenticated_closure_rejects_generation_outside_lease_window():
    fixture = _fixture()
    authority = fixture.lease_history[0]

    with pytest.raises(ValueError, match="generation is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=replace(
                fixture.generation,
                transaction_sequence=authority.transaction_sequence,
            ),
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )


def test_authenticated_closure_bounds_old_authority_by_next_mutation():
    fixture = _fixture(closing_mode="renewed")
    next_authority = fixture.lease_history[1]

    with pytest.raises(ValueError, match="generation is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=replace(
                fixture.generation,
                transaction_sequence=next_authority.transaction_sequence,
            ),
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )

    state = replace(
        fixture.state,
        intent_entry=replace(
            fixture.state.intent_entry,
            transaction_sequence=next_authority.transaction_sequence,
        ),
        open_head_entry=replace(
            fixture.state.open_head_entry,
            transaction_sequence=next_authority.transaction_sequence,
        ),
    )
    with pytest.raises(ValueError, match="opening is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )

    fixture = _fixture()
    fixture.clock.set(1_020)
    fixture.lease_manager.renew(fixture.opening_lease)
    lease_history = CoordinatorLeaseHistoryReader(
        fixture.store,
        run_id=RUN_ID,
    ).read()
    state = replace(
        fixture.state,
        closed_head_entry=replace(
            fixture.state.closed_head_entry,
            transaction_sequence=lease_history[-1].transaction_sequence,
        ),
        lifecycle_entry=replace(
            fixture.state.lifecycle_entry,
            transaction_sequence=lease_history[-1].transaction_sequence,
        ),
        lifecycle_head_entry=replace(
            fixture.state.lifecycle_head_entry,
            transaction_sequence=lease_history[-1].transaction_sequence,
        ),
    )
    with pytest.raises(ValueError, match="closure is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=lease_history,
        )


def test_authenticated_closure_rejects_unorderable_delete_recreate_history():
    fixture = _fixture()
    fixture.clock.set(1_020)
    fixture.lease_manager.release(fixture.opening_lease)
    _manager(fixture.store, fixture.clock, "coordinator-b").acquire()
    lease_history = CoordinatorLeaseHistoryReader(
        fixture.store,
        run_id=RUN_ID,
    ).read()

    with pytest.raises(ValueError, match="generation is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=lease_history,
        )


def test_authenticated_closure_rejects_opening_outside_causal_window():
    fixture = _fixture()
    transaction_sequence = fixture.generation.transaction_sequence
    state = replace(
        fixture.state,
        intent_entry=replace(
            fixture.state.intent_entry,
            transaction_sequence=transaction_sequence,
        ),
        open_head_entry=replace(
            fixture.state.open_head_entry,
            transaction_sequence=transaction_sequence,
        ),
    )

    with pytest.raises(ValueError, match="opening is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )


def test_authenticated_closure_rejects_closure_outside_lease_window():
    fixture = _fixture()
    expiry = fixture.lease_history[-1].lease.expires_at_unix_ms
    state = replace(
        fixture.state,
        closed_head_entry=replace(
            fixture.state.closed_head_entry,
            committed_at_unix_ms=expiry,
        ),
        lifecycle_entry=replace(
            fixture.state.lifecycle_entry,
            committed_at_unix_ms=expiry,
        ),
        lifecycle_head_entry=replace(
            fixture.state.lifecycle_head_entry,
            committed_at_unix_ms=expiry,
        ),
    )

    with pytest.raises(ValueError, match="closure is outside"):
        AuthenticatedInitialRestartIntentClosure(
            state=state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )


def test_authenticated_closure_is_immutable_and_strictly_typed():
    fixture = _fixture()
    authenticated = _authenticate(fixture)

    with pytest.raises(AttributeError):
        authenticated.closing_authority = authenticated.opening_authority
    with pytest.raises(TypeError, match="state"):
        AuthenticatedInitialRestartIntentClosure(
            state=cast(Any, {}),
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=fixture.lease_history,
        )
    with pytest.raises(TypeError, match="lease_history"):
        AuthenticatedInitialRestartIntentClosure(
            state=fixture.state,
            generation_snapshot=fixture.generation,
            immediate_successor=None,
            lease_history=cast(Any, []),
        )
