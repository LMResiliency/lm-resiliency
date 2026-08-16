"""Contract tests for stable multi-node restart-acknowledgement collection."""

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
from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_reader import (
    RestartAckCollectionReadConflict,
    RestartAckCollectionReadCorrupt,
    RestartAckCollector,
    RestartAckReadConflict,
    RestartAckReadCorrupt,
    RestartAckReader,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_execution import (
    RestartIntentClosureExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_preparation import (
    RestartIntentClosurePreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_reader import (
    InitialRestartIntentLifecycleReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    PersistedInitialRestartIntentOpen,
    RestartIntentOpenExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_reader import (
    RestartIntentOpenStateClosed,
    RestartIntentOpenStateCorrupt,
    RestartIntentOpenStateError,
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
    committed_node_ids: tuple[str, ...],
) -> tuple[InMemoryControlStore, dict[str, PersistedRestartAck]]:
    store, persisted, _, _ = _state_details(committed_node_ids=committed_node_ids)
    return store, persisted


def _state_details(
    *,
    committed_node_ids: tuple[str, ...],
):
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
    for node_id in ("node-a", "node-b"):
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
        if node_id not in committed_node_ids:
            continue
        receipt = RestartAckReceiptRecord(
            acknowledgement=RestartAck(
                intent_id=intent.intent_id,
                run_id=RUN_ID,
                node_id=node_id,
                agent_id=agent_id,
                generation=0,
                flushed_step=40,
                inventory_event_digests={f"inventory-{node_id}": "b" * 64},
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
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)
    persisted: dict[str, PersistedRestartAck] = {}
    for node_id in committed_node_ids:
        persisted_receipt = RestartAckReader(
            store,
            run_id=RUN_ID,
            node_id=node_id,
        ).read()
        assert persisted_receipt is not None
        persisted[node_id] = persisted_receipt
    return store, persisted, clock, lease


def test_restart_ack_reader_returns_stable_complete_snapshot():
    store, _ = _state(committed_node_ids=("node-a", "node-b"))

    collection = RestartAckCollector(store, run_id=RUN_ID).collect()

    assert collection.active_node_ids == ("node-a", "node-b")
    assert collection.received_node_ids == ("node-a", "node-b")
    assert collection.missing_node_ids == ()


def test_restart_ack_reader_preserves_stable_absence():
    store, _ = _state(committed_node_ids=("node-a",))

    collection = RestartAckCollector(store, run_id=RUN_ID).collect()

    assert collection.received_node_ids == ("node-a",)
    assert collection.missing_node_ids == ("node-b",)


def test_restart_ack_collector_reconstructs_historical_receipts_after_closure():
    store, _, clock, lease = _state_details(
        committed_node_ids=("node-a", "node-b"),
    )
    prepared = RestartIntentClosurePreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_closure(lease)
    RestartIntentClosureExecutor(
        store,
        run_id=RUN_ID,
    ).execute_initial_closure(prepared)
    closure = InitialRestartIntentLifecycleReader(store, run_id=RUN_ID).read()
    assert closure is not None

    with pytest.raises(RestartAckCollectionReadConflict, match="closed"):
        RestartAckCollector(store, run_id=RUN_ID).collect()

    collection = RestartAckCollector(store, run_id=RUN_ID).collect_for_closure(closure)

    assert collection.opened.record == closure.intent
    assert collection.opened.head == closure.open_head
    assert collection.opened.intent_entry == closure.state.intent_entry
    assert collection.opened.head_entry == closure.state.open_head_entry
    assert collection.received_node_ids == ("node-a", "node-b")


class SequencedCollector(RestartAckCollector):
    def __init__(
        self,
        store: InMemoryControlStore,
        *,
        received: dict[str, PersistedRestartAck],
    ) -> None:
        super().__init__(store, run_id=RUN_ID)
        self.received = received
        self.read_count = 0

    def _read_receipts(
        self,
        node_ids: tuple[str, ...],
        *,
        opened: PersistedInitialRestartIntentOpen | None = None,
    ) -> dict[str, PersistedRestartAck | None]:
        assert opened is None
        self.read_count += 1
        node_b = None if self.read_count == 1 else self.received["node-b"]
        return {
            "node-a": self.received["node-a"],
            "node-b": node_b,
        }


def test_restart_ack_reader_retries_receipt_committed_between_scans():
    store, received = _state(committed_node_ids=("node-a", "node-b"))
    collector = SequencedCollector(store, received=received)

    collection = collector.collect()

    assert collector.read_count == 4
    assert collection.received_node_ids == ("node-a", "node-b")


class FailingAckReader:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def read(self):
        raise self.error


class FailingOpenReader:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def read(self):
        raise self.error


@pytest.mark.parametrize(
    ("error", "expected", "message"),
    [
        (
            RestartAckReadConflict("moving"),
            RestartAckCollectionReadConflict,
            "changed repeatedly",
        ),
        (
            RestartAckReadCorrupt("broken"),
            RestartAckCollectionReadCorrupt,
            "is corrupt",
        ),
    ],
)
def test_restart_ack_reader_translates_per_node_reader_errors(
    error,
    expected,
    message,
):
    store, _ = _state(committed_node_ids=())
    collector = RestartAckCollector(store, run_id=RUN_ID)
    collector._receipt_readers["node-a"] = FailingAckReader(error)

    with pytest.raises(expected, match=message):
        collector.collect()


@pytest.mark.parametrize(
    ("error", "expected", "message"),
    [
        (
            RestartIntentOpenStateClosed("closed"),
            RestartAckCollectionReadConflict,
            "closed",
        ),
        (
            RestartIntentOpenStateError("moving"),
            RestartAckCollectionReadConflict,
            "changed repeatedly",
        ),
        (
            RestartIntentOpenStateCorrupt("broken"),
            RestartAckCollectionReadCorrupt,
            "is corrupt",
        ),
    ],
)
def test_restart_ack_reader_translates_open_reader_errors(
    error,
    expected,
    message,
):
    store = InMemoryControlStore(clock=ManualClock())
    collector = RestartAckCollector(store, run_id=RUN_ID)
    collector._open_reader = cast(Any, FailingOpenReader(error))

    with pytest.raises(expected, match=message):
        collector.collect()


def test_restart_ack_reader_requires_current_open_intent():
    store = InMemoryControlStore(clock=ManualClock())

    with pytest.raises(RestartAckCollectionReadConflict, match="no current"):
        RestartAckCollector(store, run_id=RUN_ID).collect()


@pytest.mark.parametrize("run_id", ["", " "])
def test_restart_ack_reader_validates_run_id(run_id):
    store = InMemoryControlStore(clock=ManualClock())

    with pytest.raises(ValueError, match="non-empty"):
        RestartAckCollector(store, run_id=run_id)
