"""Contract tests for stable restart-acknowledgement state reads."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
)
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
    AgentIdentity,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_ack_state_reader import (
    RestartAckStateConflict,
    RestartAckStateCorrupt,
    RestartAckStateLeaseLost,
    RestartAckStateReader,
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


class FailingReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def read(self):
        raise self._error


class MissingRegistrationHistoryStore:
    def __init__(self, delegate: InMemoryControlStore, registration_key: str) -> None:
        self._delegate = delegate
        self._registration_key = registration_key

    def get_history(self, key: str):
        if key == self._registration_key:
            return ()
        return self._delegate.get_history(key)

    def __getattr__(self, name: str):
        return getattr(self._delegate, name)


def _identity(
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-a",
) -> AgentIdentity:
    return AgentIdentity(
        run_id=RUN_ID,
        node_id=node_id,
        agent_id=agent_id,
        hostname=f"host-{node_id}",
        local_world_size=2,
        resource_ids=(f"gpu-{node_id}-0", f"gpu-{node_id}-1"),
        environment_digest="environment-v1",
    )


def _state() -> tuple[
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
        lease_duration_ms=1_000,
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
        prepare_deadline_unix_ms=1_500,
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
        agent_identity=_identity(),
        lease_duration_ms=400,
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


def test_restart_ack_state_reader_authenticates_without_mutating_store():
    _, store, lease_manager, lease, _, receipt, opened = _state()
    registration_key = agent_registration_key(RUN_ID, "node-a")
    watched_keys = (
        opened.prepared.intent_key,
        opened.prepared.intent_head_key,
        registration_key,
        lease_manager.lease_key,
    )
    histories_before = {key: store.get_history(key) for key in watched_keys}

    state = RestartAckStateReader(store, run_id=RUN_ID).read(receipt, lease)

    assert state.receipt == receipt
    assert state.opened == opened
    assert state.registration == receipt.authenticated_registration
    assert state.coordinator_authority.lease == lease
    assert {key: store.get_history(key) for key in watched_keys} == histories_before


def test_restart_ack_state_reader_accepts_live_renewed_coordinator_lease():
    clock, store, lease_manager, lease, _, receipt, _ = _state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)

    state = RestartAckStateReader(store, run_id=RUN_ID).read(receipt, renewed)

    assert state.coordinator_authority.lease == renewed
    assert state.coordinator_authority.mutation_sequence == 2


def test_restart_ack_state_reader_rejects_changed_registration_and_lease():
    clock, store, lease_manager, lease, registration_manager, receipt, _ = _state()
    clock.set(1_010)
    renewed_registration = registration_manager.renew(receipt.authenticated_registration)

    with pytest.raises(RestartAckStateRegistrationLost, match="no longer current"):
        RestartAckStateReader(store, run_id=RUN_ID).read(receipt, lease)

    current_receipt = replace(
        receipt,
        agent_registration=renewed_registration.record,
        registration_fencing_token=renewed_registration.fencing_token,
        registration_granted_at_unix_ms=renewed_registration.granted_at_unix_ms,
        received_at_unix_ms=clock.now_unix_ms,
    )
    lease_manager.renew(lease)

    with pytest.raises(RestartAckStateLeaseLost, match="live durable"):
        RestartAckStateReader(store, run_id=RUN_ID).read(current_receipt, lease)


def test_restart_ack_state_reader_rejects_missing_or_different_open_intent():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=1_000,
        clock=clock,
    ).acquire()
    _, _, _, _, _, receipt, _ = _state()

    with pytest.raises(RestartAckStateConflict, match="no current"):
        RestartAckStateReader(store, run_id=RUN_ID).read(receipt, lease)

    _, store, _, lease, _, receipt, _ = _state()
    different_receipt = replace(
        receipt,
        acknowledgement=replace(
            receipt.acknowledgement,
            intent_id="intent-b",
        ),
        intent_record=replace(
            receipt.intent_record,
            intent=replace(
                receipt.intent_record.intent,
                intent_id="intent-b",
            ),
        ),
    )

    with pytest.raises(RestartAckStateConflict, match="current restart intent"):
        RestartAckStateReader(store, run_id=RUN_ID).read(different_receipt, lease)


def test_restart_ack_state_reader_rejects_corrupt_registration():
    clock, store, _, lease, _, receipt, _ = _state()
    registration_key = agent_registration_key(RUN_ID, "node-a")
    current = store.get(registration_key)
    assert current is not None
    corrupt = AgentRegistrationRecord(
        agent_identity=_identity(node_id="node-b", agent_id="agent-b"),
        registration_id="registration-b",
        lease_duration_ms=400,
    )
    store.compare_set_in_window(
        registration_key,
        expected_revision=current.revision,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=None,
        value=corrupt.to_json(),
    )

    with pytest.raises(RestartAckStateCorrupt, match="another run or node"):
        RestartAckStateReader(store, run_id=RUN_ID).read(receipt, lease)


def test_restart_ack_state_reader_rejects_registration_missing_from_history():
    _, source, _, lease, _, receipt, _ = _state()
    store = MissingRegistrationHistoryStore(
        source,
        agent_registration_key(RUN_ID, "node-a"),
    )

    with pytest.raises(RestartAckStateCorrupt, match="absent from its durable history"):
        RestartAckStateReader(store, run_id=RUN_ID).read(receipt, lease)


@pytest.mark.parametrize(
    "dependency_error",
    [
        CoordinatorLeaseHistoryError("lease history changed"),
        GenerationStateError("generation changed"),
    ],
)
def test_restart_ack_state_reader_preserves_retryable_dependency_contention(
    dependency_error,
):
    _, store, _, lease, _, receipt, _ = _state()
    reader = RestartAckStateReader(store, run_id=RUN_ID)
    reader._open_reader = FailingReader(dependency_error)

    with pytest.raises(RestartAckStateConflict, match="dependencies changed"):
        reader.read(receipt, lease)


def test_restart_ack_state_reader_validates_input_types_and_runs():
    _, store, _, lease, _, receipt, _ = _state()
    reader = RestartAckStateReader(store, run_id=RUN_ID)

    with pytest.raises(TypeError, match="receipt"):
        reader.read({}, lease)
    with pytest.raises(TypeError, match="lease"):
        reader.read(receipt, {})
    with pytest.raises(ValueError, match="another run"):
        RestartAckStateReader(store, run_id="other-run").read(receipt, lease)
