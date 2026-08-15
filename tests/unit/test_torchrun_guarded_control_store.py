"""Atomic guarded-transaction tests for the internal torchrun control store."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from types import MappingProxyType

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreTooEarly,
    ControlStoreWrite,
    InMemoryControlStore,
)


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms

    def __call__(self) -> int:
        return self.now_unix_ms


def _guard(store: InMemoryControlStore):
    return store.compare_set(
        "run/coordinator-lease",
        expected_revision=None,
        value=b"lease-a",
    )


def _generation_writes(
    *,
    head_revision: int | None = None,
) -> dict[str, ControlStoreWrite]:
    return {
        "run/generation-head": ControlStoreWrite(
            expected_revision=head_revision,
            value=b"generation-0",
        ),
        "run/generations/0": ControlStoreWrite(
            expected_revision=None,
            value=b"snapshot-0",
        ),
    }


def test_guarded_transaction_commits_all_writes_at_one_store_time():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    guard = _guard(store)

    committed = store.compare_set_many_guarded(
        _generation_writes(),
        guard_key="run/coordinator-lease",
        expected_guard_revision=guard.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_100,
    )

    assert isinstance(committed, MappingProxyType)
    assert set(committed) == {"run/generation-head", "run/generations/0"}
    assert {entry.committed_at_unix_ms for entry in committed.values()} == {1_000}
    assert store.get("run/generation-head") == committed["run/generation-head"]
    assert store.get("run/generations/0") == committed["run/generations/0"]
    with pytest.raises(TypeError):
        committed["run/other"] = committed["run/generation-head"]  # type: ignore[index]


def test_guarded_transaction_atomically_updates_head_and_creates_successor():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    guard = _guard(store)
    initial = store.compare_set_many_guarded(
        _generation_writes(),
        guard_key="run/coordinator-lease",
        expected_guard_revision=guard.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_100,
    )
    clock.now_unix_ms = 1_001

    successor = store.compare_set_many_guarded(
        {
            "run/generation-head": ControlStoreWrite(
                expected_revision=initial["run/generation-head"].revision,
                value=b"generation-1",
            ),
            "run/generations/1": ControlStoreWrite(
                expected_revision=None,
                value=b"snapshot-1",
            ),
        },
        guard_key="run/coordinator-lease",
        expected_guard_revision=guard.revision,
        not_before_unix_ms=1_001,
        deadline_unix_ms=1_100,
    )

    assert store.get("run/generations/0") == initial["run/generations/0"]
    assert store.get("run/generation-head") == successor["run/generation-head"]
    assert store.get("run/generations/1") == successor["run/generations/1"]
    assert {entry.committed_at_unix_ms for entry in successor.values()} == {1_001}


def test_guarded_transaction_rejects_stale_guard_without_partial_writes():
    store = InMemoryControlStore()
    stale_guard = _guard(store)
    current_guard = store.compare_set(
        "run/coordinator-lease",
        expected_revision=stale_guard.revision,
        value=b"lease-b",
    )

    with pytest.raises(ControlStoreConflict) as error:
        store.compare_set_many_guarded(
            _generation_writes(),
            guard_key="run/coordinator-lease",
            expected_guard_revision=stale_guard.revision,
            not_before_unix_ms=1,
            deadline_unix_ms=2,
        )

    assert error.value.key == "run/coordinator-lease"
    assert error.value.actual_revision == current_guard.revision
    assert store.get("run/generation-head") is None
    assert store.get("run/generations/0") is None


def test_guarded_transaction_rejects_target_conflict_without_partial_writes():
    store = InMemoryControlStore()
    guard = _guard(store)
    current_head = store.compare_set(
        "run/generation-head",
        expected_revision=None,
        value=b"generation-0",
    )

    with pytest.raises(ControlStoreConflict) as error:
        store.compare_set_many_guarded(
            _generation_writes(),
            guard_key="run/coordinator-lease",
            expected_guard_revision=guard.revision,
            not_before_unix_ms=1,
            deadline_unix_ms=2,
        )

    assert error.value.key == "run/generation-head"
    assert store.get("run/generation-head") == current_head
    assert store.get("run/generations/0") is None


@pytest.mark.parametrize(
    ("now_unix_ms", "error"),
    [
        (999, ControlStoreTooEarly),
        (1_100, ControlStoreDeadlineExceeded),
    ],
)
def test_guarded_transaction_enforces_store_time_without_partial_writes(
    now_unix_ms,
    error,
):
    clock = ManualClock(now_unix_ms)
    store = InMemoryControlStore(clock=clock)
    guard = _guard(store)

    with pytest.raises(error):
        store.compare_set_many_guarded(
            _generation_writes(),
            guard_key="run/coordinator-lease",
            expected_guard_revision=guard.revision,
            not_before_unix_ms=1_000,
            deadline_unix_ms=1_100,
        )

    assert store.get("run/generation-head") is None
    assert store.get("run/generations/0") is None


def test_guarded_transaction_allows_only_one_concurrent_commit():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    guard = _guard(store)
    workers = 8
    barrier = threading.Barrier(workers)

    def commit():
        barrier.wait()
        try:
            return store.compare_set_many_guarded(
                _generation_writes(),
                guard_key="run/coordinator-lease",
                expected_guard_revision=guard.revision,
                not_before_unix_ms=1_000,
                deadline_unix_ms=1_100,
            )
        except ControlStoreConflict:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(lambda _: commit(), range(workers)))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert store.get("run/generation-head") == winners[0]["run/generation-head"]
    assert store.get("run/generations/0") == winners[0]["run/generations/0"]


def test_guarded_transaction_rejects_invalid_write_sets():
    store = InMemoryControlStore()
    guard = _guard(store)

    with pytest.raises(ValueError, match="non-empty"):
        store.compare_set_many_guarded(
            {},
            guard_key="run/coordinator-lease",
            expected_guard_revision=guard.revision,
            not_before_unix_ms=1,
            deadline_unix_ms=2,
        )
    with pytest.raises(ValueError, match="must not also"):
        store.compare_set_many_guarded(
            {
                "run/coordinator-lease": ControlStoreWrite(
                    expected_revision=guard.revision,
                    value=b"replacement",
                )
            },
            guard_key="run/coordinator-lease",
            expected_guard_revision=guard.revision,
            not_before_unix_ms=1,
            deadline_unix_ms=2,
        )


@pytest.mark.parametrize(
    ("expected_revision", "value", "error"),
    [
        (False, b"value", ValueError),
        (None, bytearray(b"value"), TypeError),
    ],
)
def test_control_store_write_rejects_invalid_values(expected_revision, value, error):
    with pytest.raises(error):
        ControlStoreWrite(
            expected_revision=expected_revision,
            value=value,
        )
