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
    ControlStoreTooEarly,
    InMemoryControlStore,
)


class ManualClock:
    def __init__(self, now_unix_ms: int) -> None:
        self.now_unix_ms = now_unix_ms

    def __call__(self) -> int:
        return self.now_unix_ms


def test_control_store_create_update_delete_and_recreate():
    store = InMemoryControlStore()
    assert not store.has_history("run/control")

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
    assert created.transaction_sequence == 1
    assert updated.transaction_sequence == 2
    assert recreated.transaction_sequence == 4
    assert created.mutation_sequence == 1
    assert updated.mutation_sequence == 2
    assert recreated.mutation_sequence == 4
    assert created.value_sequence == 1
    assert updated.value_sequence == 2
    assert recreated.value_sequence == 3
    assert created.lifetime_sequence == 1
    assert updated.lifetime_sequence == 1
    assert recreated.lifetime_sequence == 2
    assert store.has_history("run/control")
    assert store.get_history("run/control") == (created, updated, recreated)
    assert created.revision < updated.revision < tombstone_revision < recreated.revision


def test_control_store_transaction_sequence_orders_keys_and_consumes_deletes():
    store = InMemoryControlStore()
    first = store.compare_set("run/first", expected_revision=None, value=b"one")

    with pytest.raises(ControlStoreConflict):
        store.compare_set("run/first", expected_revision=None, value=b"conflict")

    second = store.compare_set("run/second", expected_revision=None, value=b"two")
    store.compare_delete("run/first", expected_revision=first.revision)
    third = store.compare_set("run/third", expected_revision=None, value=b"three")

    assert first.transaction_sequence == 1
    assert second.transaction_sequence == 2
    assert third.transaction_sequence == 4


def test_control_store_retains_key_history_after_delete():
    store = InMemoryControlStore()
    created = store.compare_set("run/control", expected_revision=None, value=b"one")

    store.compare_delete("run/control", expected_revision=created.revision)

    assert store.get("run/control") is None
    assert store.has_history("run/control")
    assert store.get_history("run/control") == (created,)


def test_control_store_history_is_immutable_and_empty_for_unknown_key():
    store = InMemoryControlStore()
    assert store.get_history("run/unknown") == ()
    created = store.compare_set("run/control", expected_revision=None, value=b"one")
    history = store.get_history("run/control")

    assert history == (created,)
    with pytest.raises(TypeError):
        history[0] = created
    updated = store.compare_set(
        "run/control",
        expected_revision=created.revision,
        value=b"two",
    )
    assert history == (created,)
    assert store.get_history("run/control") == (created, updated)


def test_control_store_value_sequence_changes_only_with_value_or_lifetime():
    store = InMemoryControlStore()
    created = store.compare_set("run/control", expected_revision=None, value=b"one")
    repeated = store.compare_set(
        "run/control",
        expected_revision=created.revision,
        value=b"one",
    )
    changed = store.compare_set(
        "run/control",
        expected_revision=repeated.revision,
        value=b"two",
    )
    store.compare_delete("run/control", expected_revision=changed.revision)
    recreated = store.compare_set("run/control", expected_revision=None, value=b"two")

    assert created.value_sequence == repeated.value_sequence == 1
    assert changed.value_sequence == 2
    assert recreated.value_sequence == 3


def test_control_store_compacts_equivalent_unpinned_refreshes():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set_in_window(
        "run/registration",
        expected_revision=None,
        not_before_unix_ms=100,
        deadline_unix_ms=None,
        value=b"same",
    )
    current = created
    for now_unix_ms in (101, 102, 103, 104):
        clock.now_unix_ms = now_unix_ms
        current = store.compare_refresh_in_window(
            "run/registration",
            expected_revision=current.revision,
            not_before_unix_ms=now_unix_ms,
            deadline_unix_ms=200,
            value=b"same",
        )

    assert store.get_history("run/registration") == (created, current)
    assert current.mutation_sequence == 5
    assert current.value_sequence == 1
    assert current.lifetime_sequence == 1


def test_control_store_refresh_rejects_changed_value():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set_in_window(
        "run/registration",
        expected_revision=None,
        not_before_unix_ms=100,
        deadline_unix_ms=None,
        value=b"same",
    )

    with pytest.raises(ValueError, match="equal"):
        store.compare_refresh_in_window(
            "run/registration",
            expected_revision=created.revision,
            not_before_unix_ms=100,
            deadline_unix_ms=200,
            value=b"different",
        )

    assert store.get("run/registration") == created


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
    assert store.get_history("run/control") == (created, updated)


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
    history = store.get_history("run/counter")
    assert len(history) == 1 + workers * updates_per_worker
    assert tuple(entry.revision for entry in history) == tuple(
        range(1, 2 + workers * updates_per_worker)
    )


@pytest.mark.parametrize("key", ["", " run/control", "run/control ", "run/\x00control"])
def test_control_store_rejects_invalid_keys(key):
    store = InMemoryControlStore()

    with pytest.raises(ValueError, match="key"):
        store.get(key)
    with pytest.raises(ValueError, match="key"):
        store.get_history(key)


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
            value=bytearray(b"value"),
        )


def test_control_store_time_window_uses_one_authoritative_sample():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set_in_window(
        "run/control",
        expected_revision=None,
        not_before_unix_ms=100,
        deadline_unix_ms=None,
        value=b"one",
    )
    assert created.committed_at_unix_ms == 100

    with pytest.raises(ControlStoreTooEarly) as early_error:
        store.compare_set_in_window(
            "run/control",
            expected_revision=created.revision,
            not_before_unix_ms=101,
            deadline_unix_ms=103,
            value=b"early",
        )
    assert early_error.value.observed_unix_ms == 100
    assert store.get("run/control") == created

    clock.now_unix_ms = 101
    updated = store.compare_set_in_window(
        "run/control",
        expected_revision=created.revision,
        not_before_unix_ms=101,
        deadline_unix_ms=103,
        value=b"two",
    )
    assert updated.committed_at_unix_ms == 101
    clock.now_unix_ms = 103

    with pytest.raises(ControlStoreDeadlineExceeded):
        store.compare_set_in_window(
            "run/control",
            expected_revision=updated.revision,
            not_before_unix_ms=101,
            deadline_unix_ms=103,
            value=b"expired",
        )

    assert store.get("run/control") == updated


def test_control_store_time_window_rejects_backward_clock():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set_in_window(
        "run/control",
        expected_revision=None,
        not_before_unix_ms=100,
        deadline_unix_ms=None,
        value=b"one",
    )
    clock.now_unix_ms = 99

    with pytest.raises(ControlStoreClockError, match="backward"):
        store.compare_set_in_window(
            "run/control",
            expected_revision=created.revision,
            not_before_unix_ms=99,
            deadline_unix_ms=None,
            value=b"stale-clock",
        )

    assert store.get("run/control") == created


def test_control_store_guarded_delete_uses_same_time_window():
    clock = ManualClock(100)
    store = InMemoryControlStore(clock=clock)
    created = store.compare_set("run/control", expected_revision=None, value=b"one")

    with pytest.raises(ControlStoreTooEarly):
        store.compare_delete_in_window(
            "run/control",
            expected_revision=created.revision,
            not_before_unix_ms=101,
            deadline_unix_ms=103,
        )
    clock.now_unix_ms = 102
    tombstone_revision = store.compare_delete_in_window(
        "run/control",
        expected_revision=created.revision,
        not_before_unix_ms=101,
        deadline_unix_ms=103,
    )

    recreated = store.compare_set("run/control", expected_revision=None, value=b"two")
    clock.now_unix_ms = 103
    with pytest.raises(ControlStoreDeadlineExceeded):
        store.compare_delete_in_window(
            "run/control",
            expected_revision=recreated.revision,
            not_before_unix_ms=101,
            deadline_unix_ms=103,
        )

    assert tombstone_revision > created.revision
    assert store.get("run/control") == recreated


@pytest.mark.parametrize(
    ("value", "revision", "committed_at_unix_ms", "error"),
    [
        (bytearray(b"value"), 1, None, TypeError),
        (b"value", 0, None, ValueError),
        (b"value", 1, 0, ValueError),
    ],
)
def test_control_store_entry_rejects_invalid_backend_values(
    value,
    revision,
    committed_at_unix_ms,
    error,
):
    with pytest.raises(error):
        ControlStoreEntry(
            value=value,
            revision=revision,
            committed_at_unix_ms=committed_at_unix_ms,
        )


def test_control_store_entry_rejects_invalid_transaction_sequence():
    with pytest.raises(ValueError, match="transaction_sequence"):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            transaction_sequence=0,
        )


@pytest.mark.parametrize(
    ("mutation_sequence", "value_sequence", "lifetime_sequence"),
    [
        (2, 1, 2),
        (3, 1, 2),
        (1, 2, 1),
    ],
)
def test_control_store_entry_rejects_impossible_sequence_lineage(
    mutation_sequence,
    value_sequence,
    lifetime_sequence,
):
    with pytest.raises(ValueError, match="control-store entry"):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            mutation_sequence=mutation_sequence,
            value_sequence=value_sequence,
            lifetime_sequence=lifetime_sequence,
        )


@pytest.mark.parametrize(
    ("mutation_sequence", "value_sequence", "lifetime_sequence"),
    [
        (2, 1, 2),
        (3, 1, 2),
        (1, 2, 1),
    ],
)
def test_control_store_entry_rejects_impossible_guard_sequence_lineage(
    mutation_sequence,
    value_sequence,
    lifetime_sequence,
):
    with pytest.raises(ValueError, match="control-store guard provenance"):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key="run/lease",
            guard_revision=1,
            guard_value_digest="a" * 64,
            guard_mutation_sequence=mutation_sequence,
            guard_value_sequence=value_sequence,
            guard_lifetime_sequence=lifetime_sequence,
        )


@pytest.mark.parametrize(
    ("guard_key", "guard_revision", "guard_value_digest"),
    [
        ("run/lease", None, None),
        (None, 1, None),
        (None, None, "a" * 64),
        ("run/lease", 1, "invalid"),
    ],
)
def test_control_store_entry_rejects_invalid_guard_provenance(
    guard_key,
    guard_revision,
    guard_value_digest,
):
    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key=guard_key,
            guard_revision=guard_revision,
            guard_value_digest=guard_value_digest,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key="run/lease",
            guard_revision=1,
            guard_value_digest="a" * 64,
            guard_mutation_sequence=1,
            guard_value_sequence=1,
            guard_lifetime_sequence=1,
            guard_committed_at_unix_ms=0,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            mutation_sequence=0,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            value_sequence=0,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            lifetime_sequence=0,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key="run/lease",
            guard_revision=1,
            guard_value_digest="a" * 64,
            guard_mutation_sequence=0,
            guard_value_sequence=1,
            guard_lifetime_sequence=1,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key="run/lease",
            guard_revision=1,
            guard_value_digest="a" * 64,
            guard_mutation_sequence=1,
            guard_value_sequence=0,
            guard_lifetime_sequence=1,
        )

    with pytest.raises(ValueError):
        ControlStoreEntry(
            value=b"value",
            revision=1,
            guard_key="run/lease",
            guard_revision=1,
            guard_value_digest="a" * 64,
        )
