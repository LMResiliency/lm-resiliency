"""Contract tests for authenticated restart-intent transaction writes."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

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
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_writes import (
    RestartIntentPreparationConflict,
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
    assert all(write.require_never_created for write in prepared.writes.values())
    assert prepared.conditions == {
        prepared.generation_head_key: current.head_revision,
        prepared.generation_snapshot_key: current.snapshot.revision,
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
