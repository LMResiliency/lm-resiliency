"""Contract tests for authenticated restart-intent transaction writes."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreConflict,
    ControlStoreWrite,
    InMemoryControlStore,
)
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
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_writes import (
    PreparedRestartIntentOpen,
    RestartIntentPreparationConflict,
    RestartIntentPreparationCorrupt,
    RestartIntentPreparationDeadlineElapsed,
    RestartIntentPreparationLeaseLost,
    RestartIntentWriteRepository,
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


def _intent(
    *,
    intent_id: str = "intent-a",
    run_id: str = RUN_ID,
    generation: int = 0,
    suspected_node_ids: tuple[str, ...] = ("node-b",),
    prepare_deadline_unix_ms: int = 1_050,
) -> RestartIntent:
    return RestartIntent(
        intent_id=intent_id,
        run_id=run_id,
        generation=generation,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=suspected_node_ids,
        prepare_deadline_unix_ms=prepare_deadline_unix_ms,
    )


def _lifecycle(
    prepared: PreparedRestartIntentOpen,
    lease: HeldCoordinatorLease,
) -> RestartIntentLifecycleRecord:
    return RestartIntentLifecycleRecord(
        closed_intent=prepared.head,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )


def _state() -> tuple[
    ManualClock,
    InMemoryControlStore,
    CoordinatorLeaseManager,
    GenerationStateManager,
    RestartIntentWriteRepository,
    HeldCoordinatorLease,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    repository = RestartIntentWriteRepository(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    return clock, store, lease_manager, generation_manager, repository, lease


def test_prepare_open_builds_create_once_writes_and_generation_conditions():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None

    prepared = repository.prepare_open(lease, current, _intent())

    assert prepared.record == RestartIntentRecord(
        intent=_intent(),
        generation_snapshot_digest=current.snapshot.record.digest,
        coordinator_id="coordinator-a",
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=100,
        coordinator_fencing_token=lease.fencing_token,
    )
    assert prepared.head == RestartIntentHeadRecord(
        run_id=RUN_ID,
        generation=0,
        intent_id="intent-a",
        intent_digest=prepared.record.digest,
    )
    assert set(prepared.writes) == {prepared.intent_head_key, prepared.intent_key}
    assert all(write.expected_revision is None for write in prepared.writes.values())
    assert not prepared.writes[prepared.intent_head_key].require_never_created
    assert prepared.writes[prepared.intent_key].require_never_created
    assert prepared.conditions == {
        prepared.generation_head_key: current.head_revision,
        prepared.generation_snapshot_key: current.snapshot.revision,
        prepared.intent_lifecycle_key: None,
    }
    assert prepared.coordinator_lease_key == repository.coordinator_lease_key
    assert prepared.expected_guard_revision == lease.fencing_token
    assert prepared.coordinator_lease_granted_at_unix_ms == 1_000
    assert prepared.not_before_unix_ms == 1_000
    assert prepared.deadline_unix_ms == 1_050
    assert store.get(prepared.intent_head_key) is None
    assert store.get(prepared.intent_key) is None
    with pytest.raises(TypeError):
        cast(Any, prepared.writes)["other"] = next(iter(prepared.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, prepared.conditions)["other"] = 1


def test_stale_prepared_open_is_fenced_by_lifecycle_closure():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    prepared = repository.prepare_open(lease, current, _intent())
    committed = store.compare_set_many_guarded(
        prepared.writes,
        guard_key=prepared.coordinator_lease_key,
        expected_guard_revision=prepared.expected_guard_revision,
        not_before_unix_ms=prepared.not_before_unix_ms,
        deadline_unix_ms=prepared.deadline_unix_ms,
        conditions=prepared.conditions,
    )
    first_head = committed[prepared.intent_head_key]
    store.compare_delete(
        prepared.intent_head_key,
        expected_revision=first_head.revision,
    )
    store.compare_set_many_guarded(
        {
            prepared.intent_lifecycle_key: ControlStoreWrite(
                expected_revision=None,
                value=_lifecycle(prepared, lease).to_json(),
            )
        },
        guard_key=prepared.coordinator_lease_key,
        expected_guard_revision=prepared.expected_guard_revision,
        not_before_unix_ms=prepared.not_before_unix_ms,
        deadline_unix_ms=prepared.deadline_unix_ms,
    )

    with pytest.raises(ControlStoreConflict):
        store.compare_set_many_guarded(
            prepared.writes,
            guard_key=prepared.coordinator_lease_key,
            expected_guard_revision=prepared.expected_guard_revision,
            not_before_unix_ms=prepared.not_before_unix_ms,
            deadline_unix_ms=prepared.deadline_unix_ms,
            conditions=prepared.conditions,
        )


def test_prepare_open_reuses_head_after_observing_lifecycle_closure():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    first = repository.prepare_open(lease, current, _intent())
    committed = store.compare_set_many_guarded(
        first.writes,
        guard_key=first.coordinator_lease_key,
        expected_guard_revision=first.expected_guard_revision,
        not_before_unix_ms=first.not_before_unix_ms,
        deadline_unix_ms=first.deadline_unix_ms,
        conditions=first.conditions,
    )
    store.compare_delete(
        first.intent_head_key,
        expected_revision=committed[first.intent_head_key].revision,
    )
    closed = store.compare_set_many_guarded(
        {
            first.intent_lifecycle_key: ControlStoreWrite(
                expected_revision=None,
                value=_lifecycle(first, lease).to_json(),
            )
        },
        guard_key=first.coordinator_lease_key,
        expected_guard_revision=first.expected_guard_revision,
        not_before_unix_ms=first.not_before_unix_ms,
        deadline_unix_ms=first.deadline_unix_ms,
    )

    second = repository.prepare_open(
        lease,
        current,
        _intent(intent_id="intent-b"),
    )

    assert (
        second.conditions[second.intent_lifecycle_key]
        == closed[second.intent_lifecycle_key].revision
    )
    reopened = store.compare_set_many_guarded(
        second.writes,
        guard_key=second.coordinator_lease_key,
        expected_guard_revision=second.expected_guard_revision,
        not_before_unix_ms=second.not_before_unix_ms,
        deadline_unix_ms=second.deadline_unix_ms,
        conditions=second.conditions,
    )
    assert set(reopened) == {second.intent_head_key, second.intent_key}


def test_prepare_open_accepts_lifecycle_closed_under_renewed_lease():
    clock, store, lease_manager, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    first = repository.prepare_open(lease, current, _intent())
    committed = store.compare_set_many_guarded(
        first.writes,
        guard_key=first.coordinator_lease_key,
        expected_guard_revision=first.expected_guard_revision,
        not_before_unix_ms=first.not_before_unix_ms,
        deadline_unix_ms=first.deadline_unix_ms,
        conditions=first.conditions,
    )
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    store.compare_delete(
        first.intent_head_key,
        expected_revision=committed[first.intent_head_key].revision,
    )
    closed = store.compare_set_many_guarded(
        {
            first.intent_lifecycle_key: ControlStoreWrite(
                expected_revision=None,
                value=_lifecycle(first, renewed).to_json(),
            )
        },
        guard_key=first.coordinator_lease_key,
        expected_guard_revision=renewed.fencing_token,
        not_before_unix_ms=renewed.granted_at_unix_ms,
        deadline_unix_ms=renewed.expires_at_unix_ms,
    )

    second = repository.prepare_open(
        renewed,
        current,
        _intent(intent_id="intent-b"),
    )

    assert (
        second.conditions[second.intent_lifecycle_key]
        == closed[second.intent_lifecycle_key].revision
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"expected_guard_revision": 999},
        {"intent_key": "same", "intent_head_key": "same"},
        {"coordinator_lease_granted_at_unix_ms": 1_001},
        {"deadline_unix_ms": 1_101},
    ],
)
def test_prepared_open_rejects_contradictory_authority(changes):
    _, _, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    prepared = repository.prepare_open(lease, current, _intent())

    with pytest.raises(ValueError):
        replace(prepared, **changes)


def test_prepare_open_uses_earlier_lease_deadline():
    _, _, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None

    prepared = repository.prepare_open(
        lease,
        current,
        _intent(prepare_deadline_unix_ms=2_000),
    )

    assert prepared.deadline_unix_ms == lease.expires_at_unix_ms


def test_prepare_open_rejects_stale_current_generation():
    clock, store, _, generation_manager, repository, lease = _state()
    stale = generation_manager.current()
    assert stale is not None
    clock.set(1_010)
    generation_manager.commit_successor(
        lease,
        stale,
        RankAssignment.from_assignments(
            run_id=RUN_ID,
            generation=1,
            assignments=(
                SlotAssignment(0, "node-a", 0, 2),
                SlotAssignment(1, "node-c", 2, 2),
            ),
            topology_digest="topology-v1",
        ),
    )

    with pytest.raises(RestartIntentPreparationConflict, match="current generation"):
        repository.prepare_open(lease, stale, _intent())

    assert store.get(repository.intent_head_key) is None


@pytest.mark.parametrize(
    ("intent", "message"),
    [
        (_intent(run_id="other-run"), "another run"),
        (_intent(generation=1), "current generation"),
        (_intent(suspected_node_ids=("node-c",)), "outside"),
    ],
)
def test_prepare_open_rejects_invalid_intent_scope(intent, message):
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None

    with pytest.raises(ValueError, match=message):
        repository.prepare_open(lease, current, intent)

    assert store.get(repository.intent_head_key) is None


def test_prepare_open_rejects_stale_or_fabricated_lease():
    _, store, lease_manager, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    renewed = lease_manager.renew(lease)

    with pytest.raises(RestartIntentPreparationLeaseLost, match="changed"):
        repository.prepare_open(lease, current, _intent())

    fabricated = replace(
        renewed,
        record=replace(renewed.record, lease_id="forged"),
    )
    with pytest.raises(RestartIntentPreparationLeaseLost, match="persisted ownership"):
        repository.prepare_open(fabricated, current, _intent())

    assert store.get(repository.intent_head_key) is None


def test_prepare_open_rejects_an_active_intent():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    prepared = repository.prepare_open(lease, current, _intent())
    store.compare_set(
        prepared.intent_head_key,
        expected_revision=None,
        value=prepared.head.to_json(),
    )

    with pytest.raises(RestartIntentPreparationConflict, match="already current"):
        repository.prepare_open(
            lease,
            current,
            _intent(intent_id="intent-b"),
        )


def test_prepare_open_rejects_deleted_lifecycle_state():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    lifecycle = store.compare_set(
        repository.intent_lifecycle_key,
        expected_revision=None,
        value=b"closed",
    )
    store.compare_delete(
        repository.intent_lifecycle_key,
        expected_revision=lifecycle.revision,
    )

    with pytest.raises(RestartIntentPreparationCorrupt, match="deleted"):
        repository.prepare_open(lease, current, _intent())


def test_prepare_open_rejects_unguarded_lifecycle_state():
    _, store, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None
    prepared = repository.prepare_open(lease, current, _intent())
    committed = store.compare_set_many_guarded(
        prepared.writes,
        guard_key=prepared.coordinator_lease_key,
        expected_guard_revision=prepared.expected_guard_revision,
        not_before_unix_ms=prepared.not_before_unix_ms,
        deadline_unix_ms=prepared.deadline_unix_ms,
        conditions=prepared.conditions,
    )
    store.compare_delete(
        prepared.intent_head_key,
        expected_revision=committed[prepared.intent_head_key].revision,
    )
    store.compare_set(
        repository.intent_lifecycle_key,
        expected_revision=None,
        value=_lifecycle(prepared, lease).to_json(),
    )

    with pytest.raises(RestartIntentPreparationCorrupt, match="provenance"):
        repository.prepare_open(
            lease,
            current,
            _intent(intent_id="intent-b"),
        )


def test_prepare_open_requires_remaining_lease_and_prepare_windows():
    _, _, _, generation_manager, repository, lease = _state()
    current = generation_manager.current()
    assert current is not None

    with pytest.raises(RestartIntentPreparationDeadlineElapsed, match="elapsed"):
        repository.prepare_open(
            lease,
            current,
            _intent(prepare_deadline_unix_ms=1_000),
        )


def test_restart_intent_keys_hide_plaintext_identity():
    store = InMemoryControlStore()
    repository_a = RestartIntentWriteRepository(store, run_id="run-a")
    repository_b = RestartIntentWriteRepository(store, run_id="run-b")

    first = repository_a.intent_key("intent-a")
    second = repository_a.intent_key("intent-b")
    third = repository_b.intent_key("intent-a")

    assert len({first, second, third}) == 3
    assert "run-a" not in first
    assert "intent-a" not in first
