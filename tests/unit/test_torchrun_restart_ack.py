"""Contract tests for coordinator-authorized restart acknowledgement writes."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
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
from lm_resiliency.integrations.torchrun._restart_ack import PreparedRestartAckWrite
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_ack_writes import (
    RestartAckWriteRecords,
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


def _prepared() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    PreparedRestartAckWrite,
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
        agent_identity=AgentIdentity(
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            hostname="host-a",
            local_world_size=2,
            resource_ids=("gpu-a0", "gpu-a1"),
            environment_digest="environment-v1",
        ),
        lease_duration_ms=400,
        clock=clock,
    )
    registration = registration_manager.register()
    receipt = RestartAckReceiptRecord(
        acknowledgement=RestartAck(
            intent_id="intent-a",
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
    node_digest = hashlib.sha256(b"node-a").hexdigest()
    records = RestartAckWriteRecords(
        receipt=receipt,
        opened=opened,
        acknowledgement_key=(f"{opened.prepared.intent_key}/acknowledgements/{node_digest}"),
        agent_registration_key=agent_registration_key(RUN_ID, "node-a"),
    )
    lease_entry = store.get(lease_manager.lease_key)
    registration_entry = store.get(registration_manager.registration_key)
    assert lease_entry is not None
    assert registration_entry is not None
    authority = CoordinatorLeaseAuthority.from_entry(
        lease_entry,
        run_id=RUN_ID,
    )
    return (
        clock,
        store,
        lease_manager,
        PreparedRestartAckWrite(
            records=records,
            registration_authority=AgentRegistrationAuthority.from_entry(
                registration_entry,
                run_id=RUN_ID,
                node_id="node-a",
            ),
            coordinator_authority=authority,
            not_before_unix_ms=1_000,
            deadline_unix_ms=1_400,
        ),
    )


def test_prepared_restart_ack_delegates_immutable_transaction_inputs():
    _, _, _, prepared = _prepared()

    assert prepared.lease == prepared.coordinator_authority.lease
    assert prepared.registration == prepared.records.receipt.authenticated_registration
    assert (
        prepared.conditions[prepared.records.agent_registration_key]
        == prepared.registration_authority.registration.fencing_token
    )
    assert prepared.coordinator_lease_key == prepared.records.opened.prepared.coordinator_lease_key
    assert prepared.expected_guard_revision == prepared.lease.fencing_token
    assert prepared.writes == prepared.records.writes
    assert prepared.conditions == prepared.records.conditions


def test_prepared_restart_ack_is_immutable():
    _, _, _, prepared = _prepared()

    with pytest.raises(AttributeError):
        prepared.deadline_unix_ms = 1_499


def test_prepared_restart_ack_accepts_renewed_coordinator_authority():
    clock, store, lease_manager, prepared = _prepared()
    clock.set(1_010)
    renewed = lease_manager.renew(prepared.lease)
    renewed_entry = store.get(lease_manager.lease_key)
    assert renewed_entry is not None
    renewed_authority = CoordinatorLeaseAuthority.from_entry(
        renewed_entry,
        run_id=RUN_ID,
    )

    renewed_prepared = replace(
        prepared,
        coordinator_authority=renewed_authority,
        not_before_unix_ms=1_010,
        deadline_unix_ms=1_400,
    )

    assert renewed_prepared.lease == renewed


def test_prepared_restart_ack_rejects_cross_run_authority():
    _, _, _, prepared = _prepared()
    authority = replace(
        prepared.coordinator_authority,
        lease=HeldCoordinatorLease(
            record=CoordinatorLeaseRecord(
                run_id="other-run",
                coordinator_id="coordinator-a",
                lease_id="lease-a",
                lease_duration_ms=1_000,
            ),
            fencing_token=prepared.lease.fencing_token,
            granted_at_unix_ms=prepared.lease.granted_at_unix_ms,
        ),
    )

    with pytest.raises(ValueError, match="another run"):
        replace(prepared, coordinator_authority=authority)


def test_prepared_restart_ack_rejects_different_registration_authority():
    _, _, _, prepared = _prepared()
    authority = replace(
        prepared.registration_authority,
        registration=replace(
            prepared.registration,
            fencing_token=prepared.registration.fencing_token + 1,
        ),
    )

    with pytest.raises(ValueError, match="registration authority"):
        replace(prepared, registration_authority=authority)


@pytest.mark.parametrize(
    ("not_before_unix_ms", "message"),
    [
        (999, "acknowledgement receipt"),
    ],
)
def test_prepared_restart_ack_rejects_early_commit_lower_bound(
    not_before_unix_ms,
    message,
):
    _, _, _, prepared = _prepared()

    with pytest.raises(ValueError, match=message):
        replace(prepared, not_before_unix_ms=not_before_unix_ms)


def test_prepared_restart_ack_rejects_lower_bound_before_lease_grant():
    _, _, _, prepared = _prepared()
    authority = replace(
        prepared.coordinator_authority,
        lease=replace(
            prepared.lease,
            granted_at_unix_ms=1_001,
        ),
    )

    with pytest.raises(ValueError, match="lease grant"):
        replace(
            prepared,
            coordinator_authority=authority,
            not_before_unix_ms=1_000,
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"not_before_unix_ms": 0}, "not_before_unix_ms"),
        ({"not_before_unix_ms": True}, "not_before_unix_ms"),
        ({"deadline_unix_ms": 0}, "deadline_unix_ms"),
        ({"deadline_unix_ms": True}, "deadline_unix_ms"),
        (
            {"not_before_unix_ms": 1_500, "deadline_unix_ms": 1_500},
            "must precede",
        ),
        ({"deadline_unix_ms": 1_501}, "restart intent"),
    ],
)
def test_prepared_restart_ack_validates_time_window(changes, message):
    _, _, _, prepared = _prepared()

    with pytest.raises(ValueError, match=message):
        replace(prepared, **changes)


def test_prepared_restart_ack_rejects_deadline_after_lease_expiry():
    _, _, _, prepared = _prepared()
    authority = replace(
        prepared.coordinator_authority,
        lease=HeldCoordinatorLease(
            record=replace(
                prepared.lease.record,
                lease_duration_ms=400,
            ),
            fencing_token=prepared.lease.fencing_token,
            granted_at_unix_ms=prepared.lease.granted_at_unix_ms,
        ),
    )

    with pytest.raises(ValueError, match="coordinator lease"):
        replace(
            prepared,
            coordinator_authority=authority,
            deadline_unix_ms=1_500,
        )


def test_prepared_restart_ack_rejects_deadline_after_registration_expiry():
    _, _, _, prepared = _prepared()

    with pytest.raises(ValueError, match="agent registration"):
        replace(prepared, deadline_unix_ms=1_450)


def test_prepared_restart_ack_requires_expected_types():
    _, _, _, prepared = _prepared()

    with pytest.raises(TypeError, match="RestartAckWriteRecords"):
        replace(prepared, records={})
    with pytest.raises(TypeError, match="AgentRegistrationAuthority"):
        replace(prepared, registration_authority={})
    with pytest.raises(TypeError, match="CoordinatorLeaseAuthority"):
        replace(prepared, coordinator_authority={})
