"""Contract tests for guarded restart-intent closure execution."""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_execution import (
    RestartIntentClosureExecutionClockError,
    RestartIntentClosureExecutionConflict,
    RestartIntentClosureExecutionCorrupt,
    RestartIntentClosureExecutionLeaseLost,
    RestartIntentClosureExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_preparation import (
    RestartIntentClosurePreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_reader import (
    InitialRestartIntentLifecycleReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
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


class TamperedClosureResultStore(InMemoryControlStore):
    def __init__(self, *, clock: ManualClock, tamper: str) -> None:
        super().__init__(clock=clock)
        self._tamper = tamper

    def compare_set_many_guarded(
        self,
        writes: Mapping[str, ControlStoreWrite],
        *,
        guard_key: str,
        expected_guard_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        conditions: Mapping[str, int | None] | None = None,
        never_created_conditions: Collection[str] | None = None,
    ) -> Mapping[str, ControlStoreEntry]:
        committed = dict(
            super().compare_set_many_guarded(
                writes,
                guard_key=guard_key,
                expected_guard_revision=expected_guard_revision,
                not_before_unix_ms=not_before_unix_ms,
                deadline_unix_ms=deadline_unix_ms,
                conditions=conditions,
                never_created_conditions=never_created_conditions,
            )
        )
        if not any(key.endswith("/restart-intent-lifecycle-head") for key in committed):
            return committed
        head_key = next(key for key in committed if key.endswith("/restart-intent-head"))
        closure_key = next(key for key in committed if "/restart-intent-closures/" in key)
        lifecycle_key = next(
            key for key in committed if key.endswith("/restart-intent-lifecycle-head")
        )
        if self._tamper == "missing_lifecycle":
            return {head_key: committed[head_key], closure_key: committed[closure_key]}
        if self._tamper == "value":
            committed[closure_key] = replace(committed[closure_key], value=b"{}")
        elif self._tamper == "transaction":
            committed[lifecycle_key] = replace(
                committed[lifecycle_key],
                transaction_sequence=committed[lifecycle_key].transaction_sequence + 1,
            )
        elif self._tamper == "guard":
            committed[head_key] = replace(
                committed[head_key],
                guard_value_digest="0" * 64,
            )
        elif self._tamper == "head_lineage":
            committed[head_key] = replace(
                committed[head_key],
                mutation_sequence=committed[head_key].mutation_sequence + 1,
            )
        elif self._tamper == "time":
            committed_at = committed[lifecycle_key].committed_at_unix_ms
            assert committed_at is not None
            committed[lifecycle_key] = replace(
                committed[lifecycle_key],
                committed_at_unix_ms=committed_at + 1,
            )
        elif self._tamper == "order":
            for key in committed:
                committed[key] = replace(committed[key], transaction_sequence=3)
        else:
            raise AssertionError(f"unsupported tamper {self._tamper!r}")
        return committed


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


def _state(*, tamper: str | None = None, preparation_clock: ManualClock | None = None):
    store_clock = ManualClock()
    store = (
        InMemoryControlStore(clock=store_clock)
        if tamper is None
        else TamperedClosureResultStore(clock=store_clock, tamper=tamper)
    )
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=store_clock,
    )
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
    RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=store_clock,
        ).prepare_initial_open(lease, current, intent)
    )
    prepared = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=preparation_clock or store_clock,
    ).prepare_initial_closure(lease)
    return store_clock, store, lease_manager, lease, prepared


def test_execute_initial_closure_commits_and_verifies_all_records():
    _, store, _, _, prepared = _state()

    committed = RestartIntentClosureExecutor(
        store,
        run_id=RUN_ID,
    ).execute_initial_closure(prepared)

    assert committed.prepared == prepared
    assert committed.closed_head_entry == store.get(prepared.records.intent_head_key)
    assert committed.lifecycle_entry == store.get(prepared.records.closure_key)
    assert committed.lifecycle_head_entry == store.get(prepared.records.lifecycle_head_key)
    assert committed.committed_at_unix_ms == 1_000
    assert committed.transaction_sequence > prepared.records.opened.transaction_sequence
    assert InitialRestartIntentLifecycleReader(store, run_id=RUN_ID).read() is not None


def test_execute_initial_closure_rejects_changed_state_or_duplicate_execution():
    _, store, _, _, prepared = _state()
    intent_entry = store.get(prepared.records.intent_key)
    assert intent_entry is not None
    store.compare_set(
        prepared.records.intent_key,
        expected_revision=intent_entry.revision,
        value=intent_entry.value,
    )
    executor = RestartIntentClosureExecutor(store, run_id=RUN_ID)

    with pytest.raises(RestartIntentClosureExecutionConflict, match="state changed"):
        executor.execute_initial_closure(prepared)

    _, store, _, _, prepared = _state()
    executor = RestartIntentClosureExecutor(store, run_id=RUN_ID)
    executor.execute_initial_closure(prepared)
    with pytest.raises(RestartIntentClosureExecutionConflict):
        executor.execute_initial_closure(prepared)


def test_execute_initial_closure_rejects_stale_or_expired_lease():
    clock, store, lease_manager, lease, prepared = _state()
    clock.set(1_010)
    lease_manager.renew(lease)
    executor = RestartIntentClosureExecutor(store, run_id=RUN_ID)

    with pytest.raises(RestartIntentClosureExecutionLeaseLost, match="changed"):
        executor.execute_initial_closure(prepared)

    clock, store, _, lease, prepared = _state()
    clock.set(lease.expires_at_unix_ms)
    with pytest.raises(RestartIntentClosureExecutionLeaseLost, match="expired"):
        RestartIntentClosureExecutor(
            store,
            run_id=RUN_ID,
        ).execute_initial_closure(prepared)


def test_execute_initial_closure_rejects_store_time_before_preparation():
    preparation_clock = ManualClock(1_010)
    _, store, _, _, prepared = _state(preparation_clock=preparation_clock)

    with pytest.raises(RestartIntentClosureExecutionClockError, match="contradicts"):
        RestartIntentClosureExecutor(
            store,
            run_id=RUN_ID,
        ).execute_initial_closure(prepared)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_lifecycle",
        "value",
        "transaction",
        "guard",
        "head_lineage",
        "time",
        "order",
    ],
)
def test_execute_initial_closure_rejects_tampered_results(tamper: str):
    _, store, _, _, prepared = _state(tamper=tamper)

    with pytest.raises(RestartIntentClosureExecutionCorrupt):
        RestartIntentClosureExecutor(
            store,
            run_id=RUN_ID,
        ).execute_initial_closure(prepared)
