"""Contract tests for authenticated restart-acknowledgement state."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    AuthenticatedRestartAckState,
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


def _state(
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-a",
) -> tuple[InMemoryControlStore, AuthenticatedRestartAckState]:
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
        agent_identity=AgentIdentity(
            run_id=RUN_ID,
            node_id=node_id,
            agent_id=agent_id,
            hostname=f"host-{node_id}",
            local_world_size=2,
            resource_ids=(f"gpu-{node_id}-0", f"gpu-{node_id}-1"),
            environment_digest="environment-v1",
        ),
        lease_duration_ms=400,
        clock=clock,
    )
    registration = registration_manager.register()
    receipt = RestartAckReceiptRecord(
        acknowledgement=RestartAck(
            intent_id=intent.intent_id,
            run_id=RUN_ID,
            node_id=node_id,
            agent_id=agent_id,
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
    lease_entry = store.get(lease_manager.lease_key)
    registration_entry = store.get(registration_manager.registration_key)
    assert lease_entry is not None
    assert registration_entry is not None
    return store, AuthenticatedRestartAckState(
        receipt=receipt,
        opened=opened,
        registration_authority=AgentRegistrationAuthority.from_entry(
            registration_entry,
            run_id=RUN_ID,
            node_id=node_id,
        ),
        coordinator_authority=CoordinatorLeaseAuthority.from_entry(
            lease_entry,
            run_id=RUN_ID,
        ),
    )


def test_authenticated_restart_ack_records_binds_inputs_without_mutation():
    store, state = _state()
    histories_before = {
        key: store.get_history(key)
        for key in (
            state.opened.prepared.intent_key,
            state.opened.prepared.intent_head_key,
        )
    }

    assert state.receipt.intent_record == state.opened.prepared.record
    assert state.registration == state.receipt.authenticated_registration
    assert state.registration_authority.registration == state.registration
    assert state.coordinator_authority.lease.record.run_id == RUN_ID
    assert {key: store.get_history(key) for key in histories_before} == histories_before

    with pytest.raises(AttributeError):
        state.registration_authority = state.registration_authority


def test_authenticated_restart_ack_records_rejects_different_intent():
    _, state = _state()
    changed_receipt = replace(
        state.receipt,
        acknowledgement=replace(
            state.receipt.acknowledgement,
            intent_id="intent-b",
        ),
        intent_record=replace(
            state.receipt.intent_record,
            intent=replace(
                state.receipt.intent_record.intent,
                intent_id="intent-b",
            ),
        ),
    )

    with pytest.raises(ValueError, match="current intent"):
        replace(state, receipt=changed_receipt)


def test_authenticated_restart_ack_records_rejects_receipt_before_intent():
    _, state = _state()
    backdated_receipt = replace(
        state.receipt,
        registration_granted_at_unix_ms=state.receipt.registration_granted_at_unix_ms - 1,
        received_at_unix_ms=state.opened.committed_at_unix_ms - 1,
    )

    with pytest.raises(ValueError, match="predates"):
        replace(state, receipt=backdated_receipt)


def test_authenticated_restart_ack_records_rejects_different_registration():
    _, state = _state()

    with pytest.raises(ValueError, match="current registration"):
        replace(
            state,
            registration_authority=replace(
                state.registration_authority,
                registration=replace(
                    state.registration,
                    fencing_token=state.registration.fencing_token + 1,
                ),
            ),
        )


def test_authenticated_restart_ack_records_rejects_cross_run_authority():
    _, state = _state()
    authority = replace(
        state.coordinator_authority,
        lease=replace(
            state.coordinator_authority.lease,
            record=replace(
                state.coordinator_authority.lease.record,
                run_id="other-run",
            ),
        ),
    )

    with pytest.raises(ValueError, match="another run"):
        replace(state, coordinator_authority=authority)


def test_authenticated_restart_ack_records_rejects_inactive_sender():
    with pytest.raises(ValueError, match="not active"):
        _state(node_id="node-c", agent_id="agent-c")


def test_authenticated_restart_ack_records_requires_expected_types():
    _, state = _state()

    with pytest.raises(TypeError, match="receipt"):
        replace(state, receipt={})
    with pytest.raises(TypeError, match="opened"):
        replace(state, opened={})
    with pytest.raises(TypeError, match="registration"):
        replace(state, registration_authority={})
    with pytest.raises(TypeError, match="coordinator_authority"):
        replace(state, coordinator_authority={})
