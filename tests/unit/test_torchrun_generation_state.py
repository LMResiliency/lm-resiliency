"""Contract tests for lease-fenced torchrun generation state."""

from __future__ import annotations

import threading
from collections.abc import Collection, Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import CoordinatorLeaseManager
from lm_resiliency.integrations.torchrun._generation_reader import GenerationStateCorrupt
from lm_resiliency.integrations.torchrun._generation_records import GenerationHeadRecord
from lm_resiliency.integrations.torchrun._generation_state import (
    GenerationStateClockError,
    GenerationStateConflict,
    GenerationStateLeaseLost,
    GenerationStateManager,
)
from lm_resiliency.integrations.torchrun._protocol import RankAssignment, SlotAssignment

RUN_ID = "training-run"
TOPOLOGY_DIGEST = "topology-v1"


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


class TamperedTransactionResultStore(InMemoryControlStore):
    def __init__(self, *, clock: ManualClock, tamper: str) -> None:
        super().__init__(clock=clock)
        self._tamper = tamper

    def compare_set_many_guarded(
        self,
        writes: Mapping[str, ControlStoreWrite],
        *,
        guard_key: str,
        expected_guard_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        conditions: Mapping[str, int | None] | None = None,
        never_created_conditions: Collection[str] | None = None,
    ) -> Mapping[str, ControlStoreEntry]:
        committed = dict(
            super().compare_set_many_guarded(
                writes,
                guard_key=guard_key,
                expected_guard_revision=expected_guard_revision,
                not_before_unix_ms=not_before_unix_ms,
                deadline_unix_ms=deadline_unix_ms,
                conditions=conditions,
                never_created_conditions=never_created_conditions,
            )
        )
        head_key = next(key for key in committed if key.endswith("/generation-head"))
        if self._tamper == "missing_snapshot":
            return {head_key: committed[head_key]}
        committed[head_key] = replace(
            committed[head_key],
            value=GenerationHeadRecord(
                run_id=RUN_ID,
                generation=0,
                snapshot_digest="0" * 64,
            ).to_json(),
        )
        return committed


def _assignment(
    generation: int,
    *,
    node_ids: tuple[str, ...] = ("node-a", "node-b"),
    topology_digest: str = TOPOLOGY_DIGEST,
    local_world_size: int = 2,
) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=tuple(
            SlotAssignment(
                logical_node_slot=slot,
                node_id=node_id,
                first_global_rank=slot * local_world_size,
                local_world_size=local_world_size,
            )
            for slot, node_id in enumerate(node_ids)
        ),
        topology_digest=topology_digest,
    )


def _managers():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    generation_manager = GenerationStateManager(
        store,
        run_id=RUN_ID,
    )
    return clock, store, lease_manager, generation_manager


def test_generation_state_initializes_and_reads_immutable_snapshot():
    _, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()

    current = manager.initialize(lease, _assignment(0))

    assert current.snapshot.record.assignment == _assignment(0)
    assert current.snapshot.record.previous_snapshot_digest is None
    assert current.snapshot.record.coordinator_fencing_token == lease.fencing_token
    assert current.snapshot.committed_at_unix_ms == lease.granted_at_unix_ms
    assert manager.current() == current
    assert manager.get(0) == current.snapshot
    assert store.get(manager.snapshot_key(1)) is None


def test_generation_state_rejects_duplicate_initialization_without_overwrite():
    _, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))

    with pytest.raises(GenerationStateConflict, match="already initialized"):
        manager.initialize(
            lease,
            _assignment(0, node_ids=("node-x", "node-y")),
        )

    assert manager.current() == current


def test_generation_state_rejects_reinitialization_after_head_deletion():
    _, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))
    store.compare_delete(
        manager.head_key,
        expected_revision=current.head_revision,
    )

    with pytest.raises(GenerationStateCorrupt, match="deleted after initialization"):
        manager.initialize(lease, _assignment(0))

    assert store.get(manager.head_key) is None
    assert store.get(manager.snapshot_key(0)) is not None


def test_generation_state_commits_successor_and_preserves_history():
    clock, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    generation_zero = manager.initialize(lease, _assignment(0))
    clock.set(1_010)

    generation_one = manager.commit_successor(
        lease,
        generation_zero,
        _assignment(1, node_ids=("node-a", "node-spare")),
    )

    assert generation_one.snapshot.record.assignment.generation == 1
    assert (
        generation_one.snapshot.record.previous_snapshot_digest
        == generation_zero.snapshot.record.digest
    )
    assert generation_one.head_revision > generation_zero.head_revision
    assert manager.current() == generation_one
    assert manager.get(0) == generation_zero.snapshot
    assert manager.get(1) == generation_one.snapshot


def test_generation_state_commits_successor_after_lease_renewal():
    clock, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    generation_zero = manager.initialize(lease, _assignment(0))
    clock.set(1_010)
    renewed = lease_manager.renew(lease)
    clock.set(1_020)

    generation_one = manager.commit_successor(
        renewed,
        generation_zero,
        _assignment(1, node_ids=("node-a", "node-spare")),
    )

    assert generation_one.snapshot.record.lease_id == lease.record.lease_id
    assert generation_one.snapshot.record.coordinator_fencing_token == renewed.fencing_token
    assert manager.current() == generation_one


def test_generation_state_commits_successor_after_expired_lease_takeover():
    clock, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    generation_zero = manager.initialize(lease, _assignment(0))
    clock.set(lease.expires_at_unix_ms)
    takeover_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-b",
        lease_duration_ms=100,
        clock=clock,
    )
    takeover = takeover_manager.acquire()

    generation_one = manager.commit_successor(
        takeover,
        generation_zero,
        _assignment(1, node_ids=("node-a", "node-spare")),
    )

    assert generation_one.snapshot.record.coordinator_id == "coordinator-b"
    assert generation_one.snapshot.record.lease_id == takeover.record.lease_id
    assert manager.current() == generation_one


def test_generation_state_rejects_stale_or_expired_lease():
    clock, store, lease_manager, manager = _managers()
    stale = lease_manager.acquire()
    current_lease = lease_manager.renew(stale)

    with pytest.raises(GenerationStateLeaseLost, match="changed"):
        manager.initialize(stale, _assignment(0))
    assert store.get(manager.head_key) is None
    assert store.get(manager.snapshot_key(0)) is None

    clock.set(current_lease.expires_at_unix_ms)
    with pytest.raises(GenerationStateLeaseLost, match="expired"):
        manager.initialize(current_lease, _assignment(0))
    assert store.get(manager.head_key) is None
    assert store.get(manager.snapshot_key(0)) is None


def test_generation_state_rejects_backward_store_clock_without_writes():
    clock, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    clock.set(lease.granted_at_unix_ms - 1)

    with pytest.raises(GenerationStateClockError, match="contradicts"):
        manager.initialize(lease, _assignment(0))

    assert store.get(manager.head_key) is None
    assert store.get(manager.snapshot_key(0)) is None


@pytest.mark.parametrize(
    ("tamper", "message"),
    [
        ("missing_snapshot", "unexpected committed key set"),
        ("substituted_head", "unexpected head record"),
    ],
)
def test_generation_state_rejects_tampered_transaction_results(tamper, message):
    clock = ManualClock()
    store = TamperedTransactionResultStore(clock=clock, tamper=tamper)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    manager = GenerationStateManager(store, run_id=RUN_ID)
    lease = lease_manager.acquire()

    with pytest.raises(GenerationStateCorrupt, match=message):
        manager.initialize(lease, _assignment(0))


def test_generation_state_rejects_fabricated_lease_identity():
    _, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    fabricated = replace(
        lease,
        record=replace(lease.record, lease_id="forged-lease"),
    )

    with pytest.raises(GenerationStateLeaseLost, match="persisted ownership"):
        manager.initialize(fabricated, _assignment(0))

    assert store.get(manager.head_key) is None
    assert store.get(manager.snapshot_key(0)) is None


def test_generation_state_allows_only_one_concurrent_initializer():
    _, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    workers = 8
    barrier = threading.Barrier(workers)

    def initialize():
        barrier.wait()
        try:
            return manager.initialize(lease, _assignment(0))
        except GenerationStateConflict:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: initialize(), range(workers)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert manager.current() == winners[0]


def test_generation_state_allows_only_one_concurrent_successor():
    clock, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))
    clock.set(1_010)
    workers = 8
    barrier = threading.Barrier(workers)

    def commit_successor(index: int):
        barrier.wait()
        try:
            return manager.commit_successor(
                lease,
                current,
                _assignment(
                    1,
                    node_ids=("node-a", f"node-spare-{index}"),
                ),
            )
        except GenerationStateConflict:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(commit_successor, range(workers)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert manager.current() == winners[0]


def test_generation_state_rejects_stale_successor_without_overwrite():
    clock, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    generation_zero = manager.initialize(lease, _assignment(0))
    clock.set(1_010)
    generation_one = manager.commit_successor(
        lease,
        generation_zero,
        _assignment(1, node_ids=("node-a", "node-spare")),
    )

    with pytest.raises(GenerationStateConflict):
        manager.commit_successor(
            lease,
            generation_zero,
            _assignment(1, node_ids=("node-replacement", "node-b")),
        )

    assert manager.current() == generation_one
    assert manager.get(1) == generation_one.snapshot


def test_generation_state_rejects_fabricated_current_generation():
    _, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))
    fabricated = replace(
        current,
        snapshot=replace(
            current.snapshot,
            record=replace(
                current.snapshot.record,
                assignment=_assignment(0, node_ids=("node-x", "node-y")),
            ),
        ),
    )

    with pytest.raises(GenerationStateConflict, match="committed generation head"):
        manager.commit_successor(lease, fabricated, _assignment(1))

    assert manager.current() == current
    assert manager.get(1) is None


@pytest.mark.parametrize(
    ("assignment", "message"),
    [
        (_assignment(2), "exactly one"),
        (
            _assignment(1, node_ids=("node-a",)),
            "active node count",
        ),
        (
            _assignment(1, local_world_size=1),
            "local world size",
        ),
        (
            _assignment(1, topology_digest="topology-v2"),
            "topology digest",
        ),
        (
            _assignment(1, node_ids=("node-b", "node-a")),
            "surviving node slots",
        ),
    ],
)
def test_generation_state_rejects_incompatible_successors(assignment, message):
    _, _, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))

    with pytest.raises(ValueError, match=message):
        manager.commit_successor(lease, current, assignment)


def test_generation_state_rejects_wrong_run_assignment_and_lease():
    _, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    wrong_assignment = replace(_assignment(0), run_id="other-run")

    with pytest.raises(ValueError, match="another run"):
        manager.initialize(lease, wrong_assignment)

    other_lease_manager = CoordinatorLeaseManager(
        store,
        run_id="other-run",
        coordinator_id="coordinator-b",
        lease_duration_ms=100,
        clock=ManualClock(),
    )
    other_lease = other_lease_manager.acquire()
    with pytest.raises(ValueError, match="another run"):
        manager.initialize(other_lease, _assignment(0))


def test_generation_state_rejects_orphan_successor_without_advancing_head():
    clock, store, lease_manager, manager = _managers()
    lease = lease_manager.acquire()
    current = manager.initialize(lease, _assignment(0))
    clock.set(1_010)
    orphan = replace(
        current.snapshot.record,
        assignment=_assignment(1),
        previous_snapshot_digest=current.snapshot.record.digest,
    )
    store.compare_set_in_window(
        manager.snapshot_key(1),
        expected_revision=None,
        not_before_unix_ms=1_010,
        deadline_unix_ms=None,
        value=orphan.to_json(),
    )

    with pytest.raises(GenerationStateCorrupt, match="newer than"):
        manager.commit_successor(lease, current, _assignment(1))

    with pytest.raises(GenerationStateCorrupt, match="newer than"):
        manager.current()
