"""Contract tests for the internal torchrun control store."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    InMemoryControlStore,
)


class ManualClock:
    def __init__(self, now_unix_ms: int) -> None:
        self.now_unix_ms = now_unix_ms

    def __call__(self) -> int:
        return self.now_unix_ms


def test_control_store_create_update_delete_and_recreate():
    store = InMemoryControlStore()

    created = store.compare_set("run/control", expected_revision=None, value=b"one")
    updated = store.compare_set(
        "run/control",
        expected_revision=created.revision,
        value=b"two",
    )
    tombstone_revision = store.compare_delete(
        "run/control",
        expected_revision=updated.revision,
    )
    recreated = store.compare_set("run/control", expected_revision=None, value=b"three")

    assert created.value == b"one"
    assert updated.value == b"two"
    assert store.get("run/control") == recreated
    assert created.revision < updated.revision < tombstone_revision < recreated.revision


def test_control_store_rejects_stale_update_and_delete():
    store = InMemoryControlStore()
    created = store.compare_set("run/control", expected_revision=None, value=b"one")
    updated = store.compare_set(
        "run/control",
        expected_revision=created.revision,
        value=b"two",
    )

    with pytest.raises(ControlStoreConflict) as update_error:
        store.compare_set(
            "run/control",
            expected_revision=created.revision,
            value=b"stale",
        )
    with pytest.raises(ControlStoreConflict) as delete_error:
        store.compare_delete(
            "run/control",
            expected_revision=created.revision,
        )

    assert update_error.value.expected_revision == created.revision
    assert update_error.value.actual_revision == updated.revision
    assert delete_error.value.actual_revision == updated.revision
    assert store.get("run/control") == updated


def test_control_store_revision_prevents_delete_recreate_aba():
    store = InMemoryControlStore()
    original = store.compare_set("run/control", expected_revision=None, value=b"same")
    store.compare_delete("run/control", expected_revision=original.revision)
    recreated = store.compare_set("run/control", expected_revision=None, value=b"same")

    with pytest.raises(ControlStoreConflict) as error:
        store.compare_set(
            "run/control",
            expected_revision=original.revision,
            value=b"stale",
        )

    assert error.value.actual_revision == recreated.revision
    assert recreated.revision != original.revision


def test_control_store_allows_only_one_concurrent_creator():
    store = InMemoryControlStore()
    workers = 8
    barrier = threading.Barrier(workers)

    def create(index: int):
        barrier.wait()
        try:
            return store.compare_set(
                "run/control",
                expected_revision=None,
                value=str(index).encode("ascii"),
            )
        except ControlStoreConflict:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(create, range(workers)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert store.get("run/control") == winners[0]


def test_control_store_serializes_concurrent_compare_set_updates():
    store = InMemoryControlStore()
    initial = store.compare_set("run/counter", expected_revision=None, value=b"0")
    assert initial.revision == 1

    workers = 8
    updates_per_worker = 25

    def increment() -> None:
        for _ in range(updates_per_worker):
            while True:
                current = store.get("run/counter")
                assert current is not None
                try:
                    store.compare_set(
                        "run/counter",
                        expected_revision=current.revision,
                        value=str(int(current.value) + 1).encode("ascii"),
                    )
                except ControlStoreConflict:
                    continue
                break

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(lambda _: increment(), range(workers)))

    result = store.get("run/counter")
    assert result is not None
    assert result.value == str(workers * updates_per_worker).encode("ascii")
    assert result.revision == 1 + workers * updates_per_worker


@pytest.mark.parametrize("key", ["", " run/control", "run/control ", "run/\x00control"])
def test_control_store_rejects_invalid_keys(key):
    store = InMemoryControlStore()

    with pytest.raises(ValueError, match="key"):
        store.get(key)


@pytest.mark.parametrize("revision", [False, 0, -1, "1"])
def test_control_store_rejects_invalid_expected_revisions(revision):
    store = InMemoryControlStore()

    with pytest.raises(ValueError, match="expected_revision"):
        store.compare_set("run/control", expected_revision=revision, value=b"value")


def test_control_store_rejects_mutable_values():
    store = InMemoryControlStore()

    with pytest.raises(TypeError, match="bytes"):
        store.compare_set(
            "run/control",
            expected_revision=None,
            value=bytearray(b"value"),  # type: ignore[arg-type]
        )


def test_control_store_deadline_guard_is_checked_atomically():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set("run/control", expected_revision=None, value=b"one")

    updated = store.compare_set_before(
        "run/control",
        expected_revision=created.revision,
        deadline_unix_ms=101,
        value=b"two",
    )
    clock.now_unix_ms = 101

    with pytest.raises(ControlStoreDeadlineExceeded) as error:
        store.compare_set_before(
            "run/control",
            expected_revision=updated.revision,
            deadline_unix_ms=101,
            value=b"expired",
        )

    assert error.value.observed_unix_ms == 101
    assert store.get("run/control") == updated


def test_control_store_deadline_guard_rejects_backward_clock():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set("run/control", expected_revision=None, value=b"one")
    updated = store.compare_set_before(
        "run/control",
        expected_revision=created.revision,
        deadline_unix_ms=200,
        value=b"two",
    )
    clock.now_unix_ms = 99

    with pytest.raises(ControlStoreClockError, match="backward"):
        store.compare_set_before(
            "run/control",
            expected_revision=updated.revision,
            deadline_unix_ms=200,
            value=b"stale-clock",
        )

    assert store.get("run/control") == updated


@pytest.mark.parametrize(
    ("value", "revision", "error"),
    [
        (bytearray(b"value"), 1, TypeError),
        (b"value", 0, ValueError),
    ],
)
def test_control_store_entry_rejects_invalid_backend_values(value, revision, error):
    with pytest.raises(error):
        ControlStoreEntry(value=value, revision=revision)
