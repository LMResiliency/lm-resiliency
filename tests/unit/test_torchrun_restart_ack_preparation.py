"""Contract tests for restart-acknowledgement preparation."""

from __future__ import annotations

import threading
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparationClockError,
    RestartAckPreparationConflict,
    RestartAckPreparationCorrupt,
    RestartAckPreparationDeadlineElapsed,
    RestartAckPreparationLeaseLost,
    RestartAckPreparationRegistrationLost,
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_ack_state_reader import (
    RestartAckStateConflict,
    RestartAckStateCorrupt,
    RestartAckStateLeaseLost,
    RestartAckStateRegistrationLost,
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


class FailingStateReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def read(self, receipt, lease):
        raise self._error


def _state(
    *,
    coordinator_lease_duration_ms: int = 1_000,
    registration_lease_duration_ms: int = 400,
    intent_deadline_unix_ms: int = 1_500,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
    AgentRegistrationManager,
    RestartAckReceiptRecord,
    CommittedInitialRestartIntentOpen,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
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
    return (
        clock,
        store,
        lease_manager,
        lease,
        registration_manager,
        receipt,
        opened,
    )


def test_restart_ack_preparer_adds_window_without_mutating_store():
    clock, store, lease_manager, lease, registration_manager, receipt, opened = _state()
    watched_keys = (
        opened.prepared.intent_key,
        opened.prepared.intent_head_key,
        registration_manager.registration_key,
        lease_manager.lease_key,
    )
    histories_before = {key: store.get_history(key) for key in watched_keys}

    prepared = RestartAckPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare(receipt, lease)

    assert prepared.records.receipt == receipt
    assert prepared.records.opened == opened
    assert prepared.registration == receipt.authenticated_registration
    assert prepared.registration_authority.transaction_sequence > 0
    assert prepared.lease == lease
    assert prepared.not_before_unix_ms == 1_000
    assert prepared.deadline_unix_ms == 1_400
    assert {key: store.get_history(key) for key in watched_keys} == histories_before


def test_restart_ack_preparer_accepts_live_renewed_coordinator_lease():
    clock, store, lease_manager, lease, _, receipt, _ = _state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)

    prepared = RestartAckPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare(receipt, renewed)

    assert prepared.lease == renewed
    assert prepared.coordinator_authority.mutation_sequence == 2
    assert prepared.not_before_unix_ms == 1_010


@pytest.mark.parametrize(
    ("state_kwargs", "now_unix_ms", "error", "message"),
    [
        (
            {"registration_lease_duration_ms": 100},
            1_100,
            RestartAckPreparationRegistrationLost,
            "registration expired",
        ),
        (
            {"coordinator_lease_duration_ms": 100},
            1_100,
            RestartAckPreparationLeaseLost,
            "coordinator lease expired",
        ),
        (
            {
                "coordinator_lease_duration_ms": 1_000,
                "registration_lease_duration_ms": 1_000,
                "intent_deadline_unix_ms": 1_100,
            },
            1_100,
            RestartAckPreparationDeadlineElapsed,
            "deadline elapsed",
        ),
    ],
)
def test_restart_ack_preparer_classifies_elapsed_windows(
    state_kwargs,
    now_unix_ms,
    error,
    message,
):
    clock, store, _, lease, _, receipt, _ = _state(**state_kwargs)
    clock.set(now_unix_ms)

    with pytest.raises(error, match=message):
        RestartAckPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(receipt, lease)


def test_restart_ack_preparer_rejects_clock_before_state_and_rollback():
    clock, store, _, lease, _, receipt, _ = _state()
    clock.set(999)
    preparer = RestartAckPreparer(store, run_id=RUN_ID, clock=clock)

    with pytest.raises(RestartAckPreparationClockError, match="precedes"):
        preparer.prepare(receipt, lease)

    clock.set(1_000)
    preparer.prepare(receipt, lease)
    clock.set(999)

    with pytest.raises(RestartAckPreparationClockError, match="moved backward"):
        preparer.prepare(receipt, lease)


@pytest.mark.parametrize(
    ("state_error", "preparation_error"),
    [
        (RestartAckStateConflict("intent changed"), RestartAckPreparationConflict),
        (
            RestartAckStateRegistrationLost("registration changed"),
            RestartAckPreparationRegistrationLost,
        ),
        (RestartAckStateLeaseLost("lease changed"), RestartAckPreparationLeaseLost),
        (RestartAckStateCorrupt("state corrupt"), RestartAckPreparationCorrupt),
    ],
)
def test_restart_ack_preparer_translates_state_errors(
    state_error,
    preparation_error,
):
    clock, store, _, lease, _, receipt, _ = _state()
    preparer = RestartAckPreparer(store, run_id=RUN_ID, clock=clock)
    preparer._state_reader = cast(Any, FailingStateReader(state_error))

    with pytest.raises(preparation_error):
        preparer.prepare(receipt, lease)


@pytest.mark.parametrize("clock_value", (0, True, "invalid"))
def test_restart_ack_preparer_rejects_invalid_clock(clock_value):
    _, store, _, lease, _, receipt, _ = _state()

    with pytest.raises(RestartAckPreparationClockError, match="clock is invalid"):
        RestartAckPreparer(
            store,
            run_id=RUN_ID,
            clock=lambda: clock_value,
        ).prepare(receipt, lease)


def test_restart_ack_preparer_validates_constructor_and_input_types():
    clock, store, _, lease, _, receipt, _ = _state()

    with pytest.raises(ValueError, match="run_id"):
        RestartAckPreparer(store, run_id="", clock=clock)
    with pytest.raises(TypeError, match="clock"):
        RestartAckPreparer(store, run_id=RUN_ID, clock=None)

    preparer = RestartAckPreparer(store, run_id=RUN_ID, clock=clock)
    with pytest.raises(TypeError, match="receipt"):
        preparer.prepare({}, lease)
    with pytest.raises(TypeError, match="lease"):
        preparer.prepare(receipt, {})
