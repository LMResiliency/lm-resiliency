"""Contract tests for immutable restart-acknowledgement transaction records."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
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
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
    RestartAckWriteRecords,
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
    CommittedInitialRestartIntentOpen,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=1_000,
        clock=clock,
    ).acquire()
    current = GenerationStateManager(store, run_id=RUN_ID).initialize(
        lease,
        _assignment(),
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
    return clock, store, opened


def _receipt(
    clock: ManualClock,
    store: InMemoryControlStore,
    opened: CommittedInitialRestartIntentOpen,
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-a",
) -> RestartAckReceiptRecord:
    registration = AgentRegistrationManager(
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
    ).register()
    return RestartAckReceiptRecord(
        acknowledgement=RestartAck(
            intent_id=opened.prepared.record.intent.intent_id,
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


def _records(
    *,
    node_id: str = "node-a",
) -> tuple[InMemoryControlStore, RestartAckWriteRecords]:
    clock, store, opened = _open_state()
    receipt = _receipt(clock, store, opened, node_id=node_id)
    node_digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
    return store, RestartAckWriteRecords(
        receipt=receipt,
        opened=opened,
        acknowledgement_key=(f"{opened.prepared.intent_key}/acknowledgements/{node_digest}"),
        agent_registration_key=agent_registration_key(RUN_ID, node_id),
    )


def test_restart_ack_write_records_build_create_once_transaction_inputs():
    store, records = _records()
    histories_before = {
        key: store.get_history(key)
        for key in (
            records.acknowledgement_key,
            records.agent_registration_key,
            records.opened.prepared.intent_key,
            records.opened.prepared.intent_head_key,
        )
    }

    assert set(records.writes) == {records.acknowledgement_key}
    write = records.writes[records.acknowledgement_key]
    assert write.expected_revision is None
    assert write.value == records.receipt.to_json()
    assert write.require_never_created
    assert records.conditions == {
        records.opened.prepared.intent_key: records.opened.intent_entry.revision,
        records.opened.prepared.intent_head_key: records.opened.head_entry.revision,
        records.agent_registration_key: records.receipt.registration_fencing_token,
    }
    assert {key: store.get_history(key) for key in histories_before} == histories_before


def test_restart_ack_write_records_expose_immutable_mappings():
    _, records = _records()

    with pytest.raises(TypeError):
        cast(Any, records.writes)["other"] = next(iter(records.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, records.conditions)["other"] = 1


def test_restart_ack_write_records_are_immutable():
    _, records = _records()

    with pytest.raises(AttributeError):
        records.acknowledgement_key = "other"


def test_restart_ack_write_records_require_exact_committed_intent():
    _, records = _records()
    changed_intent_record = replace(
        records.receipt.intent_record,
        coordinator_id="other-coordinator",
    )
    changed_receipt = replace(
        records.receipt,
        intent_record=changed_intent_record,
    )

    with pytest.raises(ValueError, match="committed restart intent"):
        replace(records, receipt=changed_receipt)


def test_restart_ack_write_records_reject_receipt_before_intent_opened():
    _, records = _records()
    backdated_receipt = replace(
        records.receipt,
        registration_granted_at_unix_ms=records.receipt.registration_granted_at_unix_ms - 1,
        received_at_unix_ms=records.opened.committed_at_unix_ms - 1,
    )

    with pytest.raises(ValueError, match="predates"):
        replace(records, receipt=backdated_receipt)


def test_restart_ack_write_records_reject_inactive_sender_node():
    clock, store, opened = _open_state()
    receipt = _receipt(
        clock,
        store,
        opened,
        node_id="node-c",
        agent_id="agent-c",
    )
    node_digest = hashlib.sha256(b"node-c").hexdigest()

    with pytest.raises(ValueError, match="not active"):
        RestartAckWriteRecords(
            receipt=receipt,
            opened=opened,
            acknowledgement_key=(f"{opened.prepared.intent_key}/acknowledgements/{node_digest}"),
            agent_registration_key=agent_registration_key(RUN_ID, "node-c"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("acknowledgement_key", "other", "acknowledgement_key"),
        ("agent_registration_key", "other", "agent_registration_key"),
    ],
)
def test_restart_ack_write_records_require_canonical_keys(field, value, message):
    _, records = _records()

    with pytest.raises(ValueError, match=message):
        replace(records, **{field: value})


def test_restart_ack_write_records_require_record_types():
    _, records = _records()

    with pytest.raises(TypeError, match="RestartAckReceiptRecord"):
        replace(records, receipt={})
    with pytest.raises(TypeError, match="CommittedInitialRestartIntentOpen"):
        replace(records, opened={})
