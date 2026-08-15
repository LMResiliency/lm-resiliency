"""Contract tests for persisted initial restart-intent opening reads."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
    RestartIntentOpenExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_reader import (
    RestartIntentOpenStateCorrupt,
    RestartIntentOpenStateReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
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


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    coordinator_id: str,
) -> CoordinatorLeaseManager:
    return CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id=coordinator_id,
        lease_duration_ms=100,
        clock=clock,
    )


def _assignment(generation: int = 0) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )


def _intent() -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=1_200,
    )


def _state() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    GenerationStateManager,
    HeldCoordinatorLease,
    CommittedInitialRestartIntentOpen,
    RestartIntentOpenStateReader,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = _manager(store, clock, "coordinator-a")
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    prepared = RestartIntentOpenPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare_initial_open(lease, current, _intent())
    opened = RestartIntentOpenExecutor(
        store,
        run_id=RUN_ID,
    ).execute_initial_open(prepared)
    return (
        clock,
        store,
        lease_manager,
        generation_manager,
        lease,
        opened,
        RestartIntentOpenStateReader(store, run_id=RUN_ID),
    )


def _closure_records(
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
) -> InitialRestartIntentClosureRecords:
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=opened.prepared.head,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    lifecycle_head = RestartIntentLifecycleHeadRecord(
        run_id=RUN_ID,
        closure_index=1,
        generation=opened.prepared.record.intent.generation,
        intent_id=opened.prepared.record.intent.intent_id,
        lifecycle_digest=lifecycle.digest,
    )
    run_prefix = opened.prepared.intent_head_key.rsplit("/", 1)[0]
    return InitialRestartIntentClosureRecords(
        opened=opened,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=RestartIntentClosedHeadRecord(
            run_id=RUN_ID,
            closure_index=1,
            generation=lifecycle_head.generation,
            intent_id=lifecycle_head.intent_id,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=opened.prepared.intent_key,
        intent_head_key=opened.prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/1",
        lifecycle_head_key=opened.prepared.lifecycle_head_key,
    )


def test_open_reader_returns_none_before_initial_open():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)

    assert RestartIntentOpenStateReader(store, run_id=RUN_ID).read() is None


def test_open_reader_reconstructs_committed_open_from_persisted_state():
    _, _, _, _, _, opened, reader = _state()

    observed = reader.read()

    assert observed == opened


def test_open_reader_survives_lease_renewal_and_replacement_after_open():
    clock, store, lease_manager, _, lease, opened, reader = _state()
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    clock.set(renewed.expires_at_unix_ms)
    replacement = _manager(store, clock, "coordinator-b").acquire()

    observed = reader.read()

    assert observed == opened
    assert replacement.fencing_token != opened.prepared.lease.fencing_token


def test_open_reader_rejects_generation_advance_while_intent_is_open():
    clock, _, _, generation_manager, lease, _, reader = _state()
    current = generation_manager.current()
    assert current is not None
    clock.set(1_010)
    generation_manager.commit_successor(
        lease,
        current,
        _assignment(generation=1),
    )

    with pytest.raises(
        RestartIntentOpenStateCorrupt,
        match="current generation",
    ):
        reader.read()


def test_open_reader_rejects_missing_or_malformed_intent_state():
    _, store, _, _, _, opened, reader = _state()
    del store._entries[opened.prepared.intent_key]

    with pytest.raises(RestartIntentOpenStateCorrupt, match="missing intent"):
        reader.read()

    store._entries[opened.prepared.intent_key] = replace(
        opened.intent_entry,
        value=b"invalid",
    )
    with pytest.raises(RestartIntentOpenStateCorrupt, match="malformed"):
        reader.read()


def test_open_reader_rejects_head_or_intent_link_substitution():
    _, store, _, _, _, opened, reader = _state()
    store._entries[opened.prepared.intent_head_key] = replace(
        opened.head_entry,
        value=replace(
            opened.prepared.head,
            intent_digest="0" * 64,
        ).to_json(),
    )

    with pytest.raises(RestartIntentOpenStateCorrupt, match="does not identify"):
        reader.read()


def test_open_reader_rejects_open_head_with_lifecycle_state():
    _, store, _, _, _, _, reader = _state()
    store.compare_set(
        reader.lifecycle_head_key,
        expected_revision=None,
        value=b"lifecycle",
    )

    with pytest.raises(RestartIntentOpenStateCorrupt, match="coexists"):
        reader.read()


def test_open_reader_rejects_deleted_current_head():
    _, store, _, _, _, opened, reader = _state()
    store.compare_delete(
        opened.prepared.intent_head_key,
        expected_revision=opened.head_entry.revision,
    )

    with pytest.raises(RestartIntentOpenStateCorrupt, match="disappeared"):
        reader.read()


def test_open_reader_rejects_live_head_without_durable_history():
    _, store, _, _, _, opened, reader = _state()
    del store._last_revisions[opened.prepared.intent_head_key]

    with pytest.raises(RestartIntentOpenStateCorrupt, match="no durable history"):
        reader.read()


def test_open_reader_rejects_closed_head_without_lifecycle_state():
    _, store, _, _, _, opened, reader = _state()
    records = _closure_records(opened, opened.prepared.lease)
    store.compare_set(
        opened.prepared.intent_head_key,
        expected_revision=opened.head_entry.revision,
        value=records.closed_head.to_json(),
    )

    with pytest.raises(RestartIntentOpenStateCorrupt, match="no lifecycle state"):
        reader.read()


def test_open_reader_returns_none_after_atomic_initial_closure():
    clock, store, _, _, lease, opened, reader = _state()
    records = _closure_records(opened, lease)
    clock.set(1_010)
    store.compare_set_many_guarded(
        records.writes,
        guard_key=opened.prepared.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=1_010,
        deadline_unix_ms=lease.expires_at_unix_ms,
        conditions=records.conditions,
    )

    assert reader.read() is None


def test_open_reader_rejects_opening_authority_missing_from_history():
    clock, store, lease_manager, _, lease, _, reader = _state()
    clock.set(lease.expires_at_unix_ms)
    lease_manager = _manager(store, clock, "coordinator-b")
    lease_manager.acquire()
    store._histories[lease_manager.lease_key] = store._histories[lease_manager.lease_key][1:]

    with pytest.raises(RestartIntentOpenStateCorrupt, match="dependencies"):
        reader.read()


def test_open_reader_is_run_scoped():
    _, store, _, _, _, _, _ = _state()

    assert RestartIntentOpenStateReader(store, run_id="other-run").read() is None
    with pytest.raises(ValueError, match="non-empty"):
        RestartIntentOpenStateReader(store, run_id="")
