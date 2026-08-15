"""Contract tests for initial restart-intent opening preparation."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreConflict,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    PreparedInitialRestartIntentOpen,
    RestartIntentOpenPreparationConflict,
    RestartIntentOpenPreparationCorrupt,
    RestartIntentOpenPreparationDeadlineElapsed,
    RestartIntentOpenPreparationLeaseLost,
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
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


def _state():
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
    preparer = RestartIntentOpenPreparer(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(lease, _assignment())
    current = generation_manager.current()
    assert current is not None
    return clock, store, lease_manager, generation_manager, preparer, lease, current


def test_prepare_initial_open_builds_create_once_writes_and_generation_conditions():
    _, store, _, _, preparer, lease, current = _state()

    prepared = preparer.prepare_initial_open(lease, current, _intent())

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
    assert prepared.coordinator_lease_key == preparer.coordinator_lease_key
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


def test_prepared_initial_open_is_self_validating():
    _, _, _, _, preparer, lease, current = _state()
    prepared = preparer.prepare_initial_open(lease, current, _intent())

    invalid_changes = (
        {"head": replace(prepared.head, intent_id="other")},
        {"expected_guard_revision": 999},
        {"intent_key": "other"},
        {"intent_head_key": "other"},
        {"coordinator_lease_key": "other"},
        {"generation_head_key": "other"},
        {"generation_snapshot_key": "other"},
        {"coordinator_lease_granted_at_unix_ms": 1_001},
        {"not_before_unix_ms": 999},
        {"deadline_unix_ms": 1_101},
    )
    for changes in invalid_changes:
        with pytest.raises(ValueError):
            replace(prepared, **changes)

    with pytest.raises(TypeError):
        replace(prepared, record={})
    with pytest.raises(TypeError):
        replace(prepared, head={})
    with pytest.raises(TypeError):
        replace(prepared, current={})
    with pytest.raises(ValueError, match="generation"):
        replace(
            prepared,
            current=replace(
                prepared.current,
                snapshot=replace(
                    prepared.current.snapshot,
                    record=replace(
                        prepared.current.snapshot.record,
                        coordinator_id="other-coordinator",
                    ),
                ),
            ),
        )


def test_prepare_initial_open_uses_earlier_lease_deadline():
    _, _, _, _, preparer, lease, current = _state()

    prepared = preparer.prepare_initial_open(
        lease,
        current,
        _intent(prepare_deadline_unix_ms=2_000),
    )

    assert prepared.deadline_unix_ms == lease.expires_at_unix_ms


def test_prepared_initial_open_fences_generation_change_at_execution():
    clock, store, _, generation_manager, preparer, lease, current = _state()
    prepared = preparer.prepare_initial_open(lease, current, _intent())
    clock.set(1_010)
    generation_manager.commit_successor(
        lease,
        current,
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

    with pytest.raises(ControlStoreConflict, match="generation-head"):
        store.compare_set_many_guarded(
            prepared.writes,
            guard_key=prepared.coordinator_lease_key,
            expected_guard_revision=prepared.expected_guard_revision,
            not_before_unix_ms=prepared.not_before_unix_ms,
            deadline_unix_ms=prepared.deadline_unix_ms,
            conditions=prepared.conditions,
        )

    assert store.get(prepared.intent_head_key) is None
    assert store.get(prepared.intent_key) is None


def test_prepare_initial_open_rejects_stale_current_generation():
    clock, store, _, generation_manager, preparer, lease, stale = _state()
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

    with pytest.raises(
        RestartIntentOpenPreparationConflict,
        match="current generation",
    ):
        preparer.prepare_initial_open(lease, stale, _intent())

    assert store.get(preparer.intent_head_key) is None


@pytest.mark.parametrize(
    ("intent", "message"),
    [
        (_intent(run_id="other-run"), "another run"),
        (_intent(generation=1), "current generation"),
        (_intent(suspected_node_ids=("node-c",)), "outside"),
    ],
)
def test_prepare_initial_open_rejects_invalid_intent_scope(intent, message):
    _, store, _, _, preparer, lease, current = _state()

    with pytest.raises(ValueError, match=message):
        preparer.prepare_initial_open(lease, current, intent)

    assert store.get(preparer.intent_head_key) is None


def test_prepare_initial_open_requires_expected_types():
    _, _, _, _, preparer, lease, current = _state()

    with pytest.raises(TypeError, match="HeldCoordinatorLease"):
        preparer.prepare_initial_open({}, current, _intent())
    with pytest.raises(TypeError, match="CurrentGeneration"):
        preparer.prepare_initial_open(lease, {}, _intent())
    with pytest.raises(TypeError, match="RestartIntent"):
        preparer.prepare_initial_open(lease, current, {})


def test_prepare_initial_open_rejects_stale_or_fabricated_lease():
    _, store, lease_manager, _, preparer, lease, current = _state()
    renewed = lease_manager.renew(lease)

    with pytest.raises(RestartIntentOpenPreparationLeaseLost, match="changed"):
        preparer.prepare_initial_open(lease, current, _intent())

    fabricated = replace(
        renewed,
        record=replace(renewed.record, lease_id="forged"),
    )
    with pytest.raises(
        RestartIntentOpenPreparationLeaseLost,
        match="persisted ownership",
    ):
        preparer.prepare_initial_open(fabricated, current, _intent())

    assert store.get(preparer.intent_head_key) is None


def test_prepare_initial_open_rejects_malformed_or_untimed_lease():
    _, store, _, _, preparer, lease, current = _state()
    entry = store.get(preparer.coordinator_lease_key)
    assert entry is not None
    store._entries[preparer.coordinator_lease_key] = replace(entry, value=b"invalid")

    with pytest.raises(RestartIntentOpenPreparationCorrupt, match="malformed"):
        preparer.prepare_initial_open(lease, current, _intent())

    store._entries[preparer.coordinator_lease_key] = replace(
        entry,
        committed_at_unix_ms=None,
    )
    with pytest.raises(RestartIntentOpenPreparationCorrupt, match="grant time"):
        preparer.prepare_initial_open(lease, current, _intent())


def test_prepare_initial_open_rejects_current_or_prior_head_history():
    _, store, _, _, preparer, lease, current = _state()
    current_head = store.compare_set(
        preparer.intent_head_key,
        expected_revision=None,
        value=b"open",
    )

    with pytest.raises(RestartIntentOpenPreparationConflict, match="already"):
        preparer.prepare_initial_open(lease, current, _intent())

    store.compare_delete(
        preparer.intent_head_key,
        expected_revision=current_head.revision,
    )
    with pytest.raises(RestartIntentOpenPreparationConflict, match="lifecycle"):
        preparer.prepare_initial_open(lease, current, _intent())


def test_prepare_initial_open_rejects_orphan_lifecycle_head():
    _, store, _, _, preparer, lease, current = _state()
    lifecycle_head = store.compare_set(
        preparer.lifecycle_head_key,
        expected_revision=None,
        value=b"closed",
    )

    with pytest.raises(RestartIntentOpenPreparationCorrupt, match="without"):
        preparer.prepare_initial_open(lease, current, _intent())

    store.compare_delete(
        preparer.lifecycle_head_key,
        expected_revision=lifecycle_head.revision,
    )
    with pytest.raises(RestartIntentOpenPreparationCorrupt, match="without"):
        preparer.prepare_initial_open(lease, current, _intent())


def test_prepare_initial_open_requires_remaining_prepare_window():
    _, _, _, _, preparer, lease, current = _state()

    with pytest.raises(
        RestartIntentOpenPreparationDeadlineElapsed,
        match="elapsed",
    ):
        preparer.prepare_initial_open(
            lease,
            current,
            _intent(prepare_deadline_unix_ms=1_000),
        )


def test_restart_intent_keys_hide_plaintext_identity():
    store = InMemoryControlStore()
    preparer_a = RestartIntentOpenPreparer(store, run_id="run-a")
    preparer_b = RestartIntentOpenPreparer(store, run_id="run-b")

    first = preparer_a.intent_key("intent-a")
    second = preparer_a.intent_key("intent-b")
    third = preparer_b.intent_key("intent-a")

    assert len({first, second, third}) == 3
    assert "run-a" not in first
    assert "intent-a" not in first


def test_prepared_initial_open_rejects_invalid_constructor_values():
    _, _, _, _, preparer, lease, current = _state()
    prepared = preparer.prepare_initial_open(lease, current, _intent())

    with pytest.raises(ValueError, match="non-empty"):
        RestartIntentOpenPreparer(InMemoryControlStore(), run_id="")
    with pytest.raises(ValueError, match="positive integer"):
        replace(
            prepared,
            current=replace(prepared.current, head_revision=False),
        )
    with pytest.raises(ValueError, match="precede"):
        replace(
            prepared,
            not_before_unix_ms=prepared.deadline_unix_ms,
        )


def test_prepared_initial_open_has_expected_type():
    _, _, _, _, preparer, lease, current = _state()

    prepared = preparer.prepare_initial_open(lease, current, _intent())

    assert isinstance(prepared, PreparedInitialRestartIntentOpen)
