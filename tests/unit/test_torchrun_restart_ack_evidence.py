"""Contract tests for latest-checkpoint restart-acknowledgement evidence."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    CheckpointInventoryEvent,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
    WorkerIdentity,
    checkpoint_inventory_digest,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckCollection,
)
from lm_resiliency.integrations.torchrun._restart_ack_evidence import (
    RestartAckEvidence,
)
from lm_resiliency.integrations.torchrun._restart_ack_execution import (
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_reader import RestartAckReader
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


def _event(
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-node-a",
    logical_node_slot: int = 0,
    global_rank: int = 0,
    local_rank: int = 0,
    generation: int = 0,
    step: int = 40,
    trust: str = "latest",
    event_id: str = "inventory-a",
) -> CheckpointInventoryEvent:
    return CheckpointInventoryEvent(
        event_id=event_id,
        run_id=RUN_ID,
        generation=generation,
        reporter=WorkerIdentity(
            run_id=RUN_ID,
            generation=generation,
            node_id=node_id,
            agent_id=agent_id,
            logical_node_slot=logical_node_slot,
            global_rank=global_rank,
            local_rank=local_rank,
            local_world_size=2,
            hostname=f"host-{node_id}",
            gpu_uuid=f"gpu-{node_id}-{local_rank}",
            topology_digest="topology-v1",
        ),
        step=step,
        trust=trust,
        topology_digest="topology-v1",
        copies=(),
    )


def _evidence(
    *,
    acknowledgement_success: bool | None = True,
) -> tuple[RestartAckEvidence, CheckpointInventoryEvent]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=1_000,
        clock=clock,
    ).acquire()
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
    event = _event()
    persisted = None
    if acknowledgement_success is not None:
        registration = AgentRegistrationManager(
            store,
            agent_identity=AgentIdentity(
                run_id=RUN_ID,
                node_id="node-a",
                agent_id="agent-node-a",
                hostname="host-node-a",
                local_world_size=2,
                resource_ids=("gpu-node-a-0", "gpu-node-a-1"),
                environment_digest="environment-v1",
            ),
            lease_duration_ms=400,
            clock=clock,
        ).register()
        receipt = RestartAckReceiptRecord(
            acknowledgement=RestartAck(
                intent_id=intent.intent_id,
                run_id=RUN_ID,
                node_id="node-a",
                agent_id="agent-node-a",
                generation=0,
                flushed_step=40 if acknowledgement_success else -1,
                inventory_event_digests={event.event_id: checkpoint_inventory_digest(event)},
                transferred_owner_ranks=(0, 1) if acknowledgement_success else (),
                transferred_peer_ranks=(2, 3) if acknowledgement_success else (),
                success=acknowledgement_success,
                reason="prepared" if acknowledgement_success else "preparation failed",
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
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)
        persisted = RestartAckReader(
            store,
            run_id=RUN_ID,
            node_id="node-a",
        ).read()
        assert persisted is not None
    collection = RestartAckCollection(
        opened=opened,
        receipts_by_node_id={
            "node-a": persisted,
            "node-b": None,
        },
    )
    return RestartAckEvidence(collection), event


def test_restart_ack_evidence_authorizes_exact_latest_inventory():
    evidence, event = _evidence()

    assert evidence.authorizes_latest_inventory(event)


@pytest.mark.parametrize(
    "event",
    [
        _event(agent_id="agent-other"),
        _event(step=41),
        _event(event_id="inventory-other"),
        _event(logical_node_slot=1, global_rank=2),
        _event(trust="candidate"),
        _event(trust="recovery_verified"),
    ],
)
def test_restart_ack_evidence_rejects_mismatched_or_nonlatest_inventory(event):
    evidence, _ = _evidence()

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_evidence_rejects_reused_event_id_with_different_bytes():
    evidence, event = _evidence()
    changed = replace(event, reporter=replace(event.reporter, hostname="host-changed"))

    assert changed.event_id == event.event_id
    assert not evidence.authorizes_latest_inventory(changed)


@pytest.mark.parametrize("acknowledgement_success", [None, False])
def test_restart_ack_evidence_rejects_missing_or_failed_preparation(
    acknowledgement_success,
):
    evidence, event = _evidence(
        acknowledgement_success=acknowledgement_success,
    )

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_evidence_rejects_event_from_missing_node():
    evidence, _ = _evidence()
    event = _event(
        node_id="node-b",
        agent_id="agent-node-b",
        logical_node_slot=1,
        global_rank=2,
    )

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_evidence_validates_types():
    evidence, _ = _evidence()

    with pytest.raises(TypeError, match="CheckpointInventoryEvent"):
        evidence.authorizes_latest_inventory(cast(Any, object()))
    with pytest.raises(TypeError, match="RestartAckCollection"):
        RestartAckEvidence(cast(Any, object()))
