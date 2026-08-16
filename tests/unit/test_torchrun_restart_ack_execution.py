"""Contract tests for guarded restart-acknowledgement execution."""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
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
    AgentIdentity,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_execution import (
    RestartAckExecutionClockError,
    RestartAckExecutionConflict,
    RestartAckExecutionCorrupt,
    RestartAckExecutionDeadlineElapsed,
    RestartAckExecutionLeaseLost,
    RestartAckExecutionRegistrationLost,
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
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


class TamperedAckResultStore(InMemoryControlStore):
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
        acknowledgement_keys = [key for key in committed if "/acknowledgements/" in key]
        if not acknowledgement_keys:
            return committed
        acknowledgement_key = acknowledgement_keys[0]
        if self._tamper == "missing":
            return {}
        if self._tamper == "unexpected":
            committed[f"{acknowledgement_key}/extra"] = committed[acknowledgement_key]
        elif self._tamper == "value":
            committed[acknowledgement_key] = replace(
                committed[acknowledgement_key],
                value=b"{}",
            )
        elif self._tamper == "lineage":
            committed[acknowledgement_key] = replace(
                committed[acknowledgement_key],
                mutation_sequence=2,
                value_sequence=2,
            )
        elif self._tamper == "guard":
            committed[acknowledgement_key] = replace(
                committed[acknowledgement_key],
                guard_value_digest="0" * 64,
            )
        elif self._tamper == "time":
            committed[acknowledgement_key] = replace(
                committed[acknowledgement_key],
                committed_at_unix_ms=None,
            )
        elif self._tamper == "order":
            committed[acknowledgement_key] = replace(
                committed[acknowledgement_key],
                transaction_sequence=1,
            )
        else:
            raise AssertionError(f"unsupported tamper {self._tamper!r}")
        return committed


def _state(
    *,
    tamper: str | None = None,
    coordinator_lease_duration_ms: int = 1_000,
    registration_lease_duration_ms: int = 400,
    intent_deadline_unix_ms: int = 1_500,
):
    clock = ManualClock()
    store = (
        InMemoryControlStore(clock=clock)
        if tamper is None
        else TamperedAckResultStore(clock=clock, tamper=tamper)
    )
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=coordinator_lease_duration_ms,
        clock=clock,
    )
    lease = lease_manager.acquire()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=0,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )
    current = GenerationStateManager(store, run_id=RUN_ID).initialize(
        lease,
        assignment,
    )
    intent = RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=intent_deadline_unix_ms,
    )
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_open(lease, current, intent)
    )
    registration_manager = AgentRegistrationManager(
        store,
        agent_identity=AgentIdentity(
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            hostname="host-node-a",
            local_world_size=2,
            resource_ids=("gpu-node-a-0", "gpu-node-a-1"),
            environment_digest="environment-v1",
        ),
        lease_duration_ms=registration_lease_duration_ms,
        clock=clock,
    )
    registration = registration_manager.register()
    receipt = RestartAckReceiptRecord(
        acknowledgement=RestartAck(
            intent_id=intent.intent_id,
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            generation=0,
            flushed_step=40,
            inventory_event_digests={"inventory-a": "b" * 64},
            transferred_owner_ranks=(0, 1),
            transferred_peer_ranks=(2, 3),
            success=True,
            reason="prepared",
        ),
        intent_record=opened.prepared.record,
        agent_registration=registration.record,
        registration_fencing_token=registration.fencing_token,
        registration_granted_at_unix_ms=registration.granted_at_unix_ms,
        received_at_unix_ms=clock.now_unix_ms,
    )
    prepared = RestartAckPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare(receipt, lease)
    return (
        clock,
        store,
        lease_manager,
        lease,
        registration_manager,
        registration,
        prepared,
    )


def test_restart_ack_executor_commits_and_verifies_receipt():
    _, store, _, _, _, _, prepared = _state()

    committed = RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)

    assert committed.prepared == prepared
    assert committed.receipt_entry == store.get(prepared.records.acknowledgement_key)
    assert committed.committed_at_unix_ms == 1_000
    assert committed.transaction_sequence > max(
        prepared.records.opened.transaction_sequence,
        prepared.registration_authority.transaction_sequence,
        prepared.coordinator_authority.transaction_sequence,
    )


def test_restart_ack_executor_rejects_changed_intent_and_duplicate_execution():
    _, store, _, _, _, _, prepared = _state()
    intent_entry = store.get(prepared.records.opened.prepared.intent_key)
    assert intent_entry is not None
    store.compare_set(
        prepared.records.opened.prepared.intent_key,
        expected_revision=intent_entry.revision,
        value=intent_entry.value,
    )
    executor = RestartAckExecutor(store, run_id=RUN_ID)

    with pytest.raises(RestartAckExecutionConflict, match="state changed"):
        executor.execute(prepared)

    _, store, _, _, _, _, prepared = _state()
    executor = RestartAckExecutor(store, run_id=RUN_ID)
    executor.execute(prepared)
    with pytest.raises(RestartAckExecutionConflict):
        executor.execute(prepared)


def test_restart_ack_executor_rejects_changed_registration():
    clock, store, _, _, registration_manager, registration, prepared = _state()
    clock.set(1_010)
    registration_manager.renew(registration)

    with pytest.raises(RestartAckExecutionRegistrationLost, match="changed"):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


def test_restart_ack_executor_rejects_changed_lease():
    clock, store, lease_manager, lease, _, _, prepared = _state()
    clock.set(1_010)
    lease_manager.renew(lease)

    with pytest.raises(RestartAckExecutionLeaseLost, match="changed"):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


@pytest.mark.parametrize(
    ("state_kwargs", "expected_error"),
    [
        (
            {"registration_lease_duration_ms": 100},
            RestartAckExecutionRegistrationLost,
        ),
        (
            {
                "registration_lease_duration_ms": 1_000,
                "coordinator_lease_duration_ms": 100,
            },
            RestartAckExecutionLeaseLost,
        ),
        (
            {
                "registration_lease_duration_ms": 1_000,
                "coordinator_lease_duration_ms": 1_000,
                "intent_deadline_unix_ms": 1_100,
            },
            RestartAckExecutionDeadlineElapsed,
        ),
    ],
)
def test_restart_ack_executor_classifies_elapsed_deadline(
    state_kwargs,
    expected_error,
):
    clock, store, _, _, _, _, prepared = _state(**state_kwargs)
    clock.set(prepared.deadline_unix_ms)

    with pytest.raises(expected_error):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


def test_restart_ack_executor_reports_defensively_shortened_deadline():
    clock, store, _, _, _, _, prepared = _state()
    prepared = replace(prepared, deadline_unix_ms=1_050)
    clock.set(prepared.deadline_unix_ms)

    with pytest.raises(RestartAckExecutionDeadlineElapsed):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


def test_restart_ack_executor_rejects_store_time_before_preparation():
    _, store, _, _, _, _, prepared = _state()
    prepared = replace(
        prepared,
        not_before_unix_ms=1_010,
    )

    with pytest.raises(RestartAckExecutionClockError, match="contradicts"):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


@pytest.mark.parametrize(
    "tamper",
    ("missing", "unexpected", "value", "lineage", "guard", "time", "order"),
)
def test_restart_ack_executor_rejects_tampered_results(tamper: str):
    _, store, _, _, _, _, prepared = _state(tamper=tamper)

    with pytest.raises(RestartAckExecutionCorrupt):
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)


def test_restart_ack_executor_validates_run_and_input_type():
    _, store, _, _, _, _, prepared = _state()

    with pytest.raises(ValueError, match="another run"):
        RestartAckExecutor(store, run_id="other-run").execute(prepared)
    with pytest.raises(TypeError, match="prepared"):
        RestartAckExecutor(store, run_id=RUN_ID).execute({})
