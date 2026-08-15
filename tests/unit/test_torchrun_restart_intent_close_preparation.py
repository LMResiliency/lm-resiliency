"""Contract tests for restart-intent closure preparation."""

from __future__ import annotations

import threading

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
from lm_resiliency.integrations.torchrun._restart_intent_close_preparation import (
    RestartIntentClosurePreparationClockError,
    RestartIntentClosurePreparationConflict,
    RestartIntentClosurePreparationCorrupt,
    RestartIntentClosurePreparationLeaseLost,
    RestartIntentClosurePreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
    RestartIntentOpenExecutor,
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


class FailingOpenReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def read(self):
        raise self._error


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    coordinator_id: str,
    *,
    run_id: str = RUN_ID,
) -> CoordinatorLeaseManager:
    return CoordinatorLeaseManager(
        store,
        run_id=run_id,
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


def _open_state() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
    CommittedInitialRestartIntentOpen,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = _manager(store, clock, "coordinator-a")
    lease = lease_manager.acquire()
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    current = generation_manager.initialize(lease, _assignment())
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
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_open(lease, current, intent)
    )
    return clock, store, lease_manager, lease, opened


def test_closure_preparer_returns_nonmutating_opening_authority_inputs():
    clock, store, _, lease, opened = _open_state()
    histories_before = {
        key: store.get_history(key)
        for key in (
            opened.prepared.intent_key,
            opened.prepared.intent_head_key,
            opened.prepared.lifecycle_head_key,
        )
    }

    prepared = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_closure(lease)

    assert prepared.records.opened == opened
    assert prepared.lease == lease
    assert prepared.not_before_unix_ms == clock.now_unix_ms
    assert prepared.deadline_unix_ms == lease.expires_at_unix_ms
    assert len(prepared.lease_authority_chain) == 1
    assert {key: store.get_history(key) for key in histories_before} == histories_before


def test_closure_preparer_accepts_renewed_and_replacement_leases():
    clock, store, lease_manager, lease, _ = _open_state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)

    renewed_prepared = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_closure(renewed)

    assert renewed_prepared.lease == renewed
    assert len(renewed_prepared.lease_authority_chain) == 2

    clock.set(renewed.expires_at_unix_ms)
    replacement = _manager(store, clock, "coordinator-b").acquire()
    replacement_prepared = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_closure(replacement)

    assert replacement_prepared.lease == replacement
    assert len(replacement_prepared.lease_authority_chain) == 3


def test_closure_preparer_rejects_missing_or_closed_intent():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = _manager(store, clock, "coordinator-a").acquire()
    preparer = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )

    with pytest.raises(RestartIntentClosurePreparationConflict, match="no current"):
        preparer.prepare_initial_closure(lease)

    clock, store, _, lease, _ = _open_state()
    preparer = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )
    prepared = preparer.prepare_initial_closure(lease)
    store.compare_set_many_guarded(
        prepared.writes,
        guard_key=prepared.coordinator_lease_key,
        expected_guard_revision=prepared.expected_guard_revision,
        not_before_unix_ms=prepared.not_before_unix_ms,
        deadline_unix_ms=prepared.deadline_unix_ms,
        conditions=prepared.conditions,
    )

    with pytest.raises(RestartIntentClosurePreparationConflict, match="already closed"):
        preparer.prepare_initial_closure(lease)


def test_closure_preparer_rejects_stale_foreign_or_expired_lease():
    clock, store, lease_manager, lease, _ = _open_state()
    clock.set(1_010)
    lease_manager.renew(lease)
    preparer = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )

    with pytest.raises(RestartIntentClosurePreparationLeaseLost, match="live durable"):
        preparer.prepare_initial_closure(lease)
    foreign = _manager(
        InMemoryControlStore(clock=clock),
        clock,
        "coordinator-x",
        run_id="other-run",
    ).acquire()
    with pytest.raises(ValueError, match="another run"):
        preparer.prepare_initial_closure(foreign)

    clock, store, _, lease, _ = _open_state()
    clock.set(lease.expires_at_unix_ms)
    with pytest.raises(RestartIntentClosurePreparationLeaseLost, match="expired"):
        RestartIntentClosurePreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_closure(lease)


def test_closure_preparer_rejects_invalid_or_regressing_clock():
    store_clock, store, _, lease, _ = _open_state()
    preparation_clock = ManualClock(lease.granted_at_unix_ms - 1)
    preparer = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=preparation_clock,
    )

    with pytest.raises(RestartIntentClosurePreparationClockError, match="precedes"):
        preparer.prepare_initial_closure(lease)

    preparation_clock.set(store_clock.now_unix_ms + 10)
    preparer.prepare_initial_closure(lease)
    preparation_clock.set(store_clock.now_unix_ms + 5)
    with pytest.raises(RestartIntentClosurePreparationClockError, match="moved backward"):
        preparer.prepare_initial_closure(lease)


def test_closure_preparer_translates_corrupt_opening():
    clock, store, _, lease, opened = _open_state()
    intent_entry = store.get(opened.prepared.intent_key)
    assert intent_entry is not None
    store.compare_set(
        opened.prepared.intent_key,
        expected_revision=intent_entry.revision,
        value=b"{}",
    )

    with pytest.raises(RestartIntentClosurePreparationCorrupt, match="opening is corrupt"):
        RestartIntentClosurePreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_closure(lease)


@pytest.mark.parametrize(
    "error",
    [
        GenerationStateError("generation changed repeatedly"),
        CoordinatorLeaseHistoryError("lease history changed repeatedly"),
    ],
)
def test_closure_preparer_translates_dependency_contention(error: Exception):
    clock, store, _, lease, _ = _open_state()
    preparer = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )
    preparer._open_reader = FailingOpenReader(error)

    with pytest.raises(
        RestartIntentClosurePreparationConflict,
        match="dependencies changed repeatedly",
    ):
        preparer.prepare_initial_closure(lease)
