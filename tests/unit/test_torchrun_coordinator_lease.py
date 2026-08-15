"""Contract tests for the internal torchrun coordinator lease."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseClockError,
    CoordinatorLeaseCorrupt,
    CoordinatorLeaseLost,
    CoordinatorLeaseManager,
    CoordinatorLeaseRecord,
    CoordinatorLeaseUnavailable,
)


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

    def advance(self, duration_ms: int) -> None:
        with self._lock:
            self.now_unix_ms += duration_ms


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    coordinator_id: str,
    *,
    run_id: str = "training-run",
) -> CoordinatorLeaseManager:
    return CoordinatorLeaseManager(
        store,
        run_id=run_id,
        coordinator_id=coordinator_id,
        lease_duration_ms=100,
        clock=clock,
    )


def test_coordinator_lease_record_round_trips_strict_json():
    record = CoordinatorLeaseRecord(
        run_id="training-run",
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        acquired_at_unix_ms=1_000,
        expires_at_unix_ms=1_100,
    )

    assert CoordinatorLeaseRecord.from_json(record.to_json()) == record
    assert json.loads(record.to_json()) == record.to_dict()

    value = record.to_dict()
    value["unknown"] = True
    with pytest.raises(ValueError, match="unknown fields"):
        CoordinatorLeaseRecord.from_json(json.dumps(value).encode("utf-8"))

    for invalid_schema in (2, 1.0, True):
        value = record.to_dict()
        value["schema_version"] = invalid_schema
        with pytest.raises(ValueError, match="unsupported"):
            CoordinatorLeaseRecord.from_json(json.dumps(value).encode("utf-8"))


def test_coordinator_lease_acquire_is_exclusive_and_idempotent():
    store = InMemoryControlStore()
    clock = ManualClock()
    first = _manager(store, clock, "coordinator-a")
    second = _manager(store, clock, "coordinator-b")

    lease = first.acquire()

    assert first.acquire() == lease
    assert first.current() == lease
    assert lease.record.coordinator_id == "coordinator-a"
    assert lease.record.expires_at_unix_ms == 1_100
    with pytest.raises(CoordinatorLeaseUnavailable, match="coordinator-a"):
        second.acquire()


def test_coordinator_lease_renewal_advances_fencing_token():
    store = InMemoryControlStore()
    clock = ManualClock()
    manager = _manager(store, clock, "coordinator-a")
    original = manager.acquire()

    clock.advance(50)
    renewed = manager.renew(original)

    assert renewed.record.lease_id == original.record.lease_id
    assert renewed.record.acquired_at_unix_ms == original.record.acquired_at_unix_ms
    assert renewed.record.expires_at_unix_ms == 1_150
    assert renewed.fencing_token > original.fencing_token
    assert manager.current() == renewed

    with pytest.raises(CoordinatorLeaseLost, match="changed before release"):
        manager.release(original)


def test_early_renewal_still_checks_and_advances_fencing_token():
    store = InMemoryControlStore()
    clock = ManualClock()
    manager = _manager(store, clock, "coordinator-a")
    original = manager.acquire()

    renewed = manager.renew(original)

    assert renewed.record == original.record
    assert renewed.fencing_token > original.fencing_token

    manager.release(renewed)
    with pytest.raises(CoordinatorLeaseLost, match="changed before renewal"):
        manager.renew(renewed)


def test_expired_lease_takeover_fences_old_coordinator():
    store = InMemoryControlStore()
    clock = ManualClock()
    first = _manager(store, clock, "coordinator-a")
    second = _manager(store, clock, "coordinator-b")
    stale = first.acquire()

    clock.set(stale.record.expires_at_unix_ms)
    replacement = second.acquire()

    assert replacement.record.coordinator_id == "coordinator-b"
    assert replacement.record.lease_id != stale.record.lease_id
    assert replacement.fencing_token > stale.fencing_token
    with pytest.raises(CoordinatorLeaseLost, match="expired"):
        first.renew(stale)
    with pytest.raises(CoordinatorLeaseLost, match="changed before release"):
        first.release(stale)


def test_release_and_reacquire_never_reuses_fencing_token():
    store = InMemoryControlStore()
    clock = ManualClock()
    manager = _manager(store, clock, "coordinator-a")
    original = manager.acquire()

    tombstone_revision = manager.release(original)
    replacement = manager.acquire()

    assert original.fencing_token < tombstone_revision < replacement.fencing_token
    assert replacement.record.lease_id != original.record.lease_id


def test_concurrent_coordinator_acquisition_has_one_winner():
    store = InMemoryControlStore()
    clock = ManualClock()
    workers = 8
    barrier = threading.Barrier(workers)
    managers = [_manager(store, clock, f"coordinator-{index}") for index in range(workers)]

    def acquire(manager: CoordinatorLeaseManager):
        barrier.wait()
        try:
            return manager.acquire()
        except CoordinatorLeaseUnavailable:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(acquire, managers))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert managers[0].current() == winners[0]


def test_coordinator_lease_fails_closed_on_corrupt_record():
    store = InMemoryControlStore()
    clock = ManualClock()
    manager = _manager(store, clock, "coordinator-a")
    store.compare_set(
        manager.lease_key,
        expected_revision=None,
        value=b"{}",
    )

    with pytest.raises(CoordinatorLeaseCorrupt, match="malformed"):
        manager.current()
    with pytest.raises(CoordinatorLeaseCorrupt, match="malformed"):
        manager.acquire()


def test_coordinator_lease_clock_cannot_move_backward():
    store = InMemoryControlStore()
    clock = ManualClock()
    manager = _manager(store, clock, "coordinator-a")
    lease = manager.acquire()

    clock.set(999)
    with pytest.raises(CoordinatorLeaseClockError, match="backward"):
        manager.renew(lease)


def test_coordinator_lease_rejects_handle_from_another_manager():
    store = InMemoryControlStore()
    clock = ManualClock()
    first = _manager(store, clock, "coordinator-a")
    second = _manager(store, clock, "coordinator-b")
    lease = first.acquire()

    with pytest.raises(CoordinatorLeaseLost, match="another manager"):
        second.renew(lease)
    with pytest.raises(CoordinatorLeaseLost, match="another manager"):
        second.release(lease)


def test_distinct_runs_use_distinct_lease_keys():
    store = InMemoryControlStore()
    clock = ManualClock()
    first = _manager(store, clock, "coordinator-a", run_id="run-a")
    second = _manager(store, clock, "coordinator-b", run_id="run-b")

    assert first.lease_key != second.lease_key
    assert first.acquire().fencing_token == 1
    assert second.acquire().fencing_token == 1


@pytest.mark.parametrize(
    "record",
    [
        CoordinatorLeaseRecord(
            run_id="training-run",
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            acquired_at_unix_ms=1_000,
            expires_at_unix_ms=1_100,
        ),
    ],
)
def test_coordinator_lease_record_rejects_invalid_expiry(record):
    with pytest.raises(ValueError, match="after acquisition"):
        replace(record, expires_at_unix_ms=record.acquired_at_unix_ms)
