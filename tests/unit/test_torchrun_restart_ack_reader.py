"""Contract tests for stable persisted restart-acknowledgement reads."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
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
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_reader import (
    RestartAckReadConflict,
    RestartAckReadCorrupt,
    RestartAckReader,
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


class CorruptibleStore(InMemoryControlStore):
    def replace_current(self, key: str, entry: ControlStoreEntry) -> None:
        with self._lock:
            self._entries[key] = entry
            self._histories[key][-1] = entry

    def drop_history(self, key: str) -> None:
        with self._lock:
            self._histories[key] = []


def _state(*, commit_receipt: bool):
    clock = ManualClock()
    store = CorruptibleStore(clock=clock)
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
            hostname="host-node-a",
            local_world_size=2,
            resource_ids=("gpu-node-a-0", "gpu-node-a-1"),
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
    acknowledgement_key = None
    if commit_receipt:
        prepared = RestartAckPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(receipt, lease)
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)
        acknowledgement_key = prepared.records.acknowledgement_key
    return (
        clock,
        store,
        lease_manager,
        lease,
        registration_manager,
        registration,
        acknowledgement_key,
    )


def test_restart_ack_reader_returns_none_for_never_created_receipt():
    _, store, _, _, _, _, _ = _state(commit_receipt=False)

    assert RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read() is None


def test_restart_ack_reader_reconstructs_committed_receipt():
    _, store, _, _, _, _, acknowledgement_key = _state(commit_receipt=True)

    persisted = RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()

    assert persisted is not None
    assert persisted.receipt_entry == store.get(acknowledgement_key)
    assert persisted.receipt.acknowledgement.node_id == "node-a"


def test_restart_ack_reader_accepts_historical_authorities_after_renewal():
    clock, store, lease_manager, lease, registration_manager, registration, _ = _state(
        commit_receipt=True
    )
    current_registration = registration
    for now_unix_ms in (1_010, 1_020, 1_030, 1_040):
        clock.set(now_unix_ms)
        current_registration = registration_manager.renew(current_registration)
    lease_manager.renew(lease)

    persisted = RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()
    registration_history = store.get_history(registration_manager.registration_key)

    assert persisted is not None
    assert persisted.registration_authority.registration == registration
    assert persisted.coordinator_authority.lease == lease
    assert registration_history[0].revision == registration.fencing_token
    assert registration_history[-1].revision == current_registration.fencing_token
    assert len(registration_history) == 2


def test_restart_ack_reader_is_node_scoped():
    _, store, _, _, _, _, _ = _state(commit_receipt=True)

    assert RestartAckReader(store, run_id=RUN_ID, node_id="node-b").read() is None


def test_restart_ack_reader_rejects_deleted_or_rewritten_receipt():
    _, store, _, _, _, _, acknowledgement_key = _state(commit_receipt=True)
    assert acknowledgement_key is not None
    entry = store.get(acknowledgement_key)
    assert entry is not None
    store.compare_delete(acknowledgement_key, expected_revision=entry.revision)

    with pytest.raises(RestartAckReadCorrupt, match="deleted"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()

    _, store, _, _, _, _, acknowledgement_key = _state(commit_receipt=True)
    assert acknowledgement_key is not None
    entry = store.get(acknowledgement_key)
    assert entry is not None
    store.compare_set(
        acknowledgement_key,
        expected_revision=entry.revision,
        value=entry.value,
    )

    with pytest.raises(RestartAckReadCorrupt, match="immutable"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()


def test_restart_ack_reader_rejects_malformed_receipt():
    _, store, _, _, _, _, acknowledgement_key = _state(commit_receipt=True)
    assert acknowledgement_key is not None
    entry = store.get(acknowledgement_key)
    assert entry is not None
    store.replace_current(
        acknowledgement_key,
        replace(entry, value=b"{}"),
    )

    with pytest.raises(RestartAckReadCorrupt, match="malformed"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()


def test_restart_ack_reader_rejects_missing_registration_history():
    _, store, _, _, registration_manager, _, _ = _state(commit_receipt=True)
    store.drop_history(registration_manager.registration_key)

    with pytest.raises(RestartAckReadCorrupt, match="registration history"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()


def test_restart_ack_reader_rejects_missing_coordinator_authority():
    _, store, lease_manager, _, _, _, acknowledgement_key = _state(commit_receipt=True)
    assert acknowledgement_key is not None
    entry = store.get(acknowledgement_key)
    assert entry is not None
    store.replace_current(
        acknowledgement_key,
        replace(entry, guard_value_digest="0" * 64),
    )

    with pytest.raises(RestartAckReadCorrupt, match="coordinator-lease authority"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()
    assert store.get(lease_manager.lease_key) is not None


def test_restart_ack_reader_rejects_missing_current_intent():
    store = InMemoryControlStore(clock=ManualClock())

    with pytest.raises(RestartAckReadConflict, match="no current"):
        RestartAckReader(store, run_id=RUN_ID, node_id="node-a").read()


def test_restart_ack_reader_validates_constructor_inputs():
    store = InMemoryControlStore(clock=ManualClock())

    with pytest.raises(ValueError, match="run_id"):
        RestartAckReader(store, run_id="", node_id="node-a")
    with pytest.raises(ValueError, match="node_id"):
        RestartAckReader(store, run_id=RUN_ID, node_id="")
