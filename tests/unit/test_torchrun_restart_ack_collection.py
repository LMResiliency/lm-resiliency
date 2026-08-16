"""Contract tests for immutable restart-acknowledgement collections."""

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
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckCollection,
)
from lm_resiliency.integrations.torchrun._restart_ack_execution import (
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
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


def _state() -> tuple[
    CommittedInitialRestartIntentOpen,
    dict[str, PersistedRestartAck],
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
    prepared_open = RestartIntentOpenPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_open(lease, current, intent)
    clock.set(1_010)
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(prepared_open)
    for node_id, success in (("node-a", True), ("node-b", False)):
        agent_id = f"agent-{node_id}"
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
        receipt = RestartAckReceiptRecord(
            acknowledgement=RestartAck(
                intent_id=intent.intent_id,
                run_id=RUN_ID,
                node_id=node_id,
                agent_id=agent_id,
                generation=0,
                flushed_step=40 if success else -1,
                inventory_event_digests=({f"inventory-{node_id}": "b" * 64} if success else {}),
                transferred_owner_ranks=(0, 1) if success else (),
                transferred_peer_ranks=(2, 3) if success else (),
                success=success,
                reason="prepared" if success else "preparation failed",
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
    receipts: dict[str, PersistedRestartAck] = {}
    for node_id in ("node-a", "node-b"):
        persisted = RestartAckReader(store, run_id=RUN_ID, node_id=node_id).read()
        assert persisted is not None
        receipts[node_id] = persisted
    return opened, receipts


def test_restart_ack_collection_classifies_exact_active_node_receipts():
    opened, receipts = _state()
    assert receipts["node-a"].opened != opened
    assert receipts["node-a"].opened.intent_entry == opened.intent_entry
    assert receipts["node-a"].opened.head_entry == opened.head_entry

    collection = RestartAckCollection(
        opened=opened,
        receipts_by_node_id={
            "node-b": receipts["node-b"],
            "node-a": receipts["node-a"],
        },
    )

    assert collection.active_node_ids == ("node-a", "node-b")
    assert tuple(collection.receipts_by_node_id) == collection.active_node_ids
    assert collection.received_node_ids == ("node-a", "node-b")
    assert collection.missing_node_ids == ()
    assert collection.successful_node_ids == ("node-a",)
    assert collection.failed_node_ids == ("node-b",)
    with pytest.raises(TypeError):
        collection.receipts_by_node_id["node-a"] = None


def test_restart_ack_collection_preserves_missing_separately_from_failed():
    opened, receipts = _state()

    collection = RestartAckCollection(
        opened=opened,
        receipts_by_node_id={
            "node-a": receipts["node-a"],
            "node-b": None,
        },
    )

    assert collection.received_node_ids == ("node-a",)
    assert collection.missing_node_ids == ("node-b",)
    assert collection.successful_node_ids == ("node-a",)
    assert collection.failed_node_ids == ()


@pytest.mark.parametrize(
    "receipts_by_node_id",
    [
        {"node-a": None},
        {"node-a": None, "node-b": None, "node-c": None},
    ],
)
def test_restart_ack_collection_requires_exact_active_node_keys(
    receipts_by_node_id,
):
    opened, _ = _state()

    with pytest.raises(ValueError, match="exactly match"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id=receipts_by_node_id,
        )


def test_restart_ack_collection_rejects_receipt_under_another_node():
    opened, receipts = _state()

    with pytest.raises(ValueError, match="another node identity"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id={
                "node-a": receipts["node-b"],
                "node-b": receipts["node-a"],
            },
        )


def test_restart_ack_collection_rejects_receipt_from_another_opening():
    opened, receipts = _state()
    other_opened = replace(
        opened,
        head_entry=replace(opened.head_entry, revision=opened.head_entry.revision + 100),
    )
    other_receipt = replace(receipts["node-a"], opened=other_opened)

    with pytest.raises(ValueError, match="another restart intent opening"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id={
                "node-a": other_receipt,
                "node-b": receipts["node-b"],
            },
        )


def test_restart_ack_collection_validates_opened_type():
    with pytest.raises(TypeError, match="opened"):
        RestartAckCollection(
            opened=cast(Any, object()),
            receipts_by_node_id={},
        )


def test_restart_ack_collection_validates_mapping_type():
    opened, _ = _state()

    with pytest.raises(TypeError, match="mapping"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id=cast(Any, []),
        )


def test_restart_ack_collection_validates_node_and_receipt_types():
    opened, _ = _state()

    with pytest.raises(ValueError, match="non-empty string"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id={
                "": None,
                "node-b": None,
            },
        )
    with pytest.raises(TypeError, match="PersistedRestartAck or None"):
        RestartAckCollection(
            opened=opened,
            receipts_by_node_id={
                "node-a": cast(Any, object()),
                "node-b": None,
            },
        )
