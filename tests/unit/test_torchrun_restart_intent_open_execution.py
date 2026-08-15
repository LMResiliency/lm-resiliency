"""Contract tests for guarded initial restart-intent execution."""

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
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    RestartIntentOpenExecutionClockError,
    RestartIntentOpenExecutionConflict,
    RestartIntentOpenExecutionCorrupt,
    RestartIntentOpenExecutionDeadlineElapsed,
    RestartIntentOpenExecutionLeaseLost,
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


class TamperedTransactionResultStore(InMemoryControlStore):
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
        if not any("/restart-intents/" in key for key in committed):
            return committed
        intent_key = next(key for key in committed if "/restart-intents/" in key)
        head_key = next(key for key in committed if key.endswith("/restart-intent-head"))
        if self._tamper == "missing_head":
            return {intent_key: committed[intent_key]}
        if self._tamper == "intent_value":
            committed[intent_key] = replace(committed[intent_key], value=b"{}")
        elif self._tamper == "transaction_sequence":
            committed[head_key] = replace(
                committed[head_key],
                transaction_sequence=committed[head_key].transaction_sequence + 1,
            )
        elif self._tamper == "guard_digest":
            committed[intent_key] = replace(
                committed[intent_key],
                guard_value_digest="0" * 64,
            )
        elif self._tamper == "commit_time":
            committed_at_unix_ms = committed[head_key].committed_at_unix_ms
            assert committed_at_unix_ms is not None
            committed[head_key] = replace(
                committed[head_key],
                committed_at_unix_ms=committed_at_unix_ms + 1,
            )
        elif self._tamper == "generation_order":
            committed[intent_key] = replace(
                committed[intent_key],
                transaction_sequence=2,
            )
            committed[head_key] = replace(
                committed[head_key],
                transaction_sequence=2,
            )
        elif self._tamper == "guard_lineage":
            for key in (intent_key, head_key):
                committed[key] = replace(
                    committed[key],
                    guard_mutation_sequence=1,
                    guard_value_sequence=1,
                    guard_lifetime_sequence=1,
                )
        else:
            raise AssertionError(f"unsupported tamper {self._tamper!r}")
        return committed


def _assignment(generation: int = 0) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )


def _intent(*, prepare_deadline_unix_ms: int = 1_050) -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=prepare_deadline_unix_ms,
    )


def _state(
    *,
    store_tamper: str | None = None,
    preparation_clock: ManualClock | None = None,
    prepare_deadline_unix_ms: int = 1_050,
    renew_before_prepare: bool = False,
):
    store_clock = ManualClock()
    if store_tamper is None:
        store = InMemoryControlStore(clock=store_clock)
    else:
        store = TamperedTransactionResultStore(
            clock=store_clock,
            tamper=store_tamper,
        )
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=store_clock,
    )
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    preparer = RestartIntentOpenPreparer(
        store,
        run_id=RUN_ID,
        clock=preparation_clock or store_clock,
    )
    executor = RestartIntentOpenExecutor(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    if renew_before_prepare:
        store_clock.set(1_010)
        lease = lease_manager.renew(lease)
    prepared = preparer.prepare_initial_open(
        lease,
        current,
        _intent(prepare_deadline_unix_ms=prepare_deadline_unix_ms),
    )
    return (
        store_clock,
        store,
        lease_manager,
        generation_manager,
        executor,
        lease,
        current,
        prepared,
    )


def test_execute_initial_open_commits_and_verifies_both_records():
    _, store, _, _, executor, _, current, prepared = _state()

    committed = executor.execute_initial_open(prepared)

    assert committed.prepared == prepared
    assert committed.intent_entry == store.get(prepared.intent_key)
    assert committed.head_entry == store.get(prepared.intent_head_key)
    assert committed.committed_at_unix_ms == 1_000
    assert committed.transaction_sequence > current.snapshot.transaction_sequence
    assert committed.intent_entry.transaction_sequence == committed.head_entry.transaction_sequence


def test_execute_initial_open_accepts_lease_renewed_after_generation_commit():
    _, _, _, _, executor, _, current, prepared = _state(renew_before_prepare=True)

    committed = executor.execute_initial_open(prepared)

    assert committed.intent_entry.guard_mutation_sequence > current.snapshot.guard_mutation_sequence
    assert committed.intent_entry.guard_value_sequence == current.snapshot.guard_value_sequence
    assert (
        committed.intent_entry.guard_lifetime_sequence == current.snapshot.guard_lifetime_sequence
    )


def test_execute_initial_open_rejects_changed_generation():
    clock, store, _, generation_manager, executor, lease, current, prepared = _state()
    clock.set(1_010)
    generation_manager.commit_successor(
        lease,
        current,
        _assignment(generation=1),
    )

    with pytest.raises(RestartIntentOpenExecutionConflict, match="generation-head"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None
    assert store.get(prepared.intent_head_key) is None


def test_execute_initial_open_rejects_stale_lease():
    clock, store, lease_manager, _, executor, lease, _, prepared = _state()
    clock.set(1_010)
    lease_manager.renew(lease)

    with pytest.raises(RestartIntentOpenExecutionLeaseLost, match="changed"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None


def test_execute_initial_open_rejects_lifecycle_history_created_after_preparation():
    _, store, _, _, executor, _, _, prepared = _state()
    lifecycle = store.compare_set(
        prepared.lifecycle_head_key,
        expected_revision=None,
        value=b"closed",
    )
    store.compare_delete(
        prepared.lifecycle_head_key,
        expected_revision=lifecycle.revision,
    )

    with pytest.raises(RestartIntentOpenExecutionConflict, match="prior history"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None
    assert store.get(prepared.intent_head_key) is None


def test_execute_initial_open_rejects_duplicate_execution():
    _, _, _, _, executor, _, _, prepared = _state()
    executor.execute_initial_open(prepared)

    with pytest.raises(RestartIntentOpenExecutionConflict):
        executor.execute_initial_open(prepared)


def test_execute_initial_open_reports_intent_deadline():
    clock, store, _, _, executor, _, _, prepared = _state()
    clock.set(1_050)

    with pytest.raises(RestartIntentOpenExecutionDeadlineElapsed, match="deadline"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None


def test_execute_initial_open_reports_expired_lease():
    clock, store, _, _, executor, _, _, prepared = _state(prepare_deadline_unix_ms=2_000)
    clock.set(1_100)

    with pytest.raises(RestartIntentOpenExecutionLeaseLost, match="expired"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None


def test_execute_initial_open_rejects_store_time_before_preparation():
    preparation_clock = ManualClock(1_010)
    _, store, _, _, executor, _, _, prepared = _state(preparation_clock=preparation_clock)

    with pytest.raises(RestartIntentOpenExecutionClockError, match="time"):
        executor.execute_initial_open(prepared)

    assert store.get(prepared.intent_key) is None


@pytest.mark.parametrize(
    "tamper",
    [
        "missing_head",
        "intent_value",
        "transaction_sequence",
        "guard_digest",
        "commit_time",
        "generation_order",
        "guard_lineage",
    ],
)
def test_execute_initial_open_rejects_tampered_transaction_results(tamper):
    _, _, _, _, executor, _, _, prepared = _state(
        store_tamper=tamper,
        renew_before_prepare=tamper == "guard_lineage",
    )

    with pytest.raises(RestartIntentOpenExecutionCorrupt, match="transaction"):
        executor.execute_initial_open(prepared)


def test_execute_initial_open_is_run_scoped_and_type_checked():
    _, store, _, _, _, _, _, prepared = _state()
    executor = RestartIntentOpenExecutor(store, run_id="other-run")

    with pytest.raises(ValueError, match="another run"):
        executor.execute_initial_open(prepared)
    with pytest.raises(TypeError, match="PreparedInitialRestartIntentOpen"):
        executor.execute_initial_open({})
