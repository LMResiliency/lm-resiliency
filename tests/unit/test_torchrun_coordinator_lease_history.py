"""Contract tests for torchrun coordinator lease authority values."""

from __future__ import annotations

import dataclasses
import threading

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseAuthorityCorrupt,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
    CoordinatorLeaseHistoryReader,
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


class StaticHistoryStore(InMemoryControlStore):
    def __init__(
        self,
        history: tuple[ControlStoreEntry, ...],
        *,
        current: ControlStoreEntry | None = None,
        has_history: bool | None = None,
    ) -> None:
        self._history = history
        self._current = history[-1] if current is None and history else current
        self._has_history = bool(history) if has_history is None else has_history

    def get(self, key: str) -> ControlStoreEntry | None:
        return self._current

    def get_history(self, key: str) -> tuple[ControlStoreEntry, ...]:
        return self._history

    def has_history(self, key: str) -> bool:
        return self._has_history


class UnstableHistoryStore(StaticHistoryStore):
    def __init__(self, history: tuple[ControlStoreEntry, ...]) -> None:
        super().__init__(history)
        self._reads = 0

    def get_history(self, key: str) -> tuple[ControlStoreEntry, ...]:
        self._reads += 1
        return self._history[:-1] if self._reads % 2 else self._history


def _record(
    *,
    run_id: str = "training-run",
    coordinator_id: str = "coordinator-a",
    lease_id: str = "lease-a",
) -> CoordinatorLeaseRecord:
    return CoordinatorLeaseRecord(
        run_id=run_id,
        coordinator_id=coordinator_id,
        lease_id=lease_id,
        lease_duration_ms=100,
    )


def _entry(
    *,
    record: CoordinatorLeaseRecord | None = None,
    revision: int = 7,
    committed_at_unix_ms: int | None = 1_000,
    transaction_sequence: int = 11,
    mutation_sequence: int = 1,
    value_sequence: int = 1,
    lifetime_sequence: int = 1,
) -> ControlStoreEntry:
    lease_record = record or _record()
    return ControlStoreEntry(
        value=lease_record.to_json(),
        revision=revision,
        committed_at_unix_ms=committed_at_unix_ms,
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
    )


def test_coordinator_lease_authority_decodes_canonical_entry():
    entry = _entry()

    authority = CoordinatorLeaseAuthority.from_entry(
        entry,
        run_id="training-run",
    )

    assert authority == CoordinatorLeaseAuthority(
        lease=HeldCoordinatorLease(
            record=_record(),
            fencing_token=7,
            granted_at_unix_ms=1_000,
        ),
        transaction_sequence=11,
        mutation_sequence=1,
        value_sequence=1,
        lifetime_sequence=1,
    )


def test_coordinator_lease_authority_is_immutable():
    authority = CoordinatorLeaseAuthority.from_entry(
        _entry(),
        run_id="training-run",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        authority.mutation_sequence = 2


@pytest.mark.parametrize(
    ("mutation_sequence", "value_sequence", "lifetime_sequence", "message"),
    [
        (2, 1, 2, "mutation_sequence is too small"),
        (3, 1, 2, "value_sequence is too small"),
        (3, 3, 2, "value_sequence is too large"),
    ],
)
def test_coordinator_lease_authority_rejects_impossible_sequences(
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        CoordinatorLeaseAuthority.from_entry(
            _entry(
                mutation_sequence=mutation_sequence,
                value_sequence=value_sequence,
                lifetime_sequence=lifetime_sequence,
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_transaction_before_mutation():
    with pytest.raises(ValueError, match="transaction_sequence is too small"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(
                _entry(mutation_sequence=2, value_sequence=1),
                transaction_sequence=1,
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_malformed_or_wrong_run():
    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="malformed"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(_entry(), value=b"not-json"),
            run_id="training-run",
        )

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="another run"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(record=_record(run_id="other-run")),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_noncanonical_bytes():
    entry = _entry()

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="noncanonical"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(
                entry,
                value=entry.value.replace(b",", b", "),
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_requires_authoritative_commit_time():
    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="commit time"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(committed_at_unix_ms=None),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_guarded_entry():
    entry = dataclasses.replace(
        _entry(),
        guard_key="guard",
        guard_revision=1,
        guard_value_digest="a" * 64,
        guard_mutation_sequence=1,
        guard_value_sequence=1,
        guard_lifetime_sequence=1,
        guard_committed_at_unix_ms=1_000,
    )

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="guard provenance"):
        CoordinatorLeaseAuthority.from_entry(
            entry,
            run_id="training-run",
        )


@pytest.mark.parametrize("run_id", ("", " ", None, 1))
def test_coordinator_lease_authority_rejects_invalid_expected_run_id(run_id: object):
    with pytest.raises(ValueError, match="run_id"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(),
            run_id=run_id,  # type: ignore[arg-type]
        )


def test_coordinator_lease_authority_rejects_wrong_entry_type():
    with pytest.raises(TypeError, match="ControlStoreEntry"):
        CoordinatorLeaseAuthority.from_entry(
            object(),
            run_id="training-run",
        )


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    coordinator_id: str,
) -> CoordinatorLeaseManager:
    return CoordinatorLeaseManager(
        store,
        run_id="training-run",
        coordinator_id=coordinator_id,
        lease_duration_ms=100,
        clock=clock,
    )


def test_coordinator_lease_history_reads_renewal_and_replacement():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "coordinator-a")
    second = _manager(store, clock, "coordinator-b")

    opened = first.acquire()
    clock.set(1_050)
    renewed = first.renew(opened)
    clock.set(renewed.expires_at_unix_ms)
    replacement = second.acquire()

    history = CoordinatorLeaseHistoryReader(store, run_id="training-run").read()

    assert tuple(authority.lease for authority in history) == (
        opened,
        renewed,
        replacement,
    )
    assert tuple(authority.mutation_sequence for authority in history) == (1, 2, 3)
    assert tuple(authority.value_sequence for authority in history) == (1, 1, 2)
    assert tuple(authority.lifetime_sequence for authority in history) == (1, 1, 1)


def test_coordinator_lease_history_preserves_delete_and_recreate():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "coordinator-a")
    second = _manager(store, clock, "coordinator-b")

    opened = first.acquire()
    clock.set(1_010)
    first.release(opened)
    replacement = second.acquire()

    history = CoordinatorLeaseHistoryReader(store, run_id="training-run").read()

    assert tuple(authority.lease for authority in history) == (opened, replacement)
    assert tuple(authority.mutation_sequence for authority in history) == (1, 3)
    assert tuple(authority.lifetime_sequence for authority in history) == (1, 2)


def test_coordinator_lease_history_rejects_deleted_current_lease():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "coordinator-a")
    lease = manager.acquire()
    manager.release(lease)

    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="deleted"):
        CoordinatorLeaseHistoryReader(store, run_id="training-run").read()


def test_coordinator_lease_history_reads_empty_store():
    assert (
        CoordinatorLeaseHistoryReader(
            InMemoryControlStore(),
            run_id="training-run",
        ).read()
        == ()
    )


@pytest.mark.parametrize(
    (
        "transaction_sequence",
        "mutation_sequence",
        "value_sequence",
        "lifetime_sequence",
        "committed_at_unix_ms",
        "message",
    ),
    [
        (1, 1, 1, 1, 1_100, "transaction sequences"),
        (3, 3, 2, 1, 1_100, "omits a key mutation"),
        (2, 2, 1, 1, 1_100, "value sequence"),
        (5, 5, 3, 3, 1_100, "lifetime"),
        (2, 2, 2, 1, 900, "grant times"),
    ],
)
def test_coordinator_lease_history_rejects_impossible_transition(
    transaction_sequence: int,
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    committed_at_unix_ms: int,
    message: str,
):
    first = _entry(transaction_sequence=1, revision=10)
    second = _entry(
        record=_record(coordinator_id="coordinator-b", lease_id="lease-b"),
        revision=20,
        committed_at_unix_ms=committed_at_unix_ms,
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
    )

    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match=message):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((first, second)),
            run_id="training-run",
        ).read()


def test_coordinator_lease_history_rejects_expired_renewal_and_overlap():
    first = _entry(transaction_sequence=1, revision=10)
    expired = _entry(
        revision=20,
        committed_at_unix_ms=1_100,
        transaction_sequence=2,
        mutation_sequence=2,
    )
    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="expired lease"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((first, expired)),
            run_id="training-run",
        ).read()

    overlapping = dataclasses.replace(
        expired,
        value=_record(
            coordinator_id="coordinator-b",
            lease_id="lease-b",
        ).to_json(),
        committed_at_unix_ms=1_099,
        value_sequence=2,
    )
    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="overlap"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((first, overlapping)),
            run_id="training-run",
        ).read()


def test_coordinator_lease_history_rejects_identity_and_token_replay():
    records = (
        _record(),
        _record(coordinator_id="coordinator-b", lease_id="lease-b"),
        _record(coordinator_id="coordinator-c", lease_id="lease-a"),
    )
    history = tuple(
        _entry(
            record=record,
            revision=revision,
            committed_at_unix_ms=committed_at,
            transaction_sequence=index,
            mutation_sequence=index,
            value_sequence=index,
        )
        for index, (record, revision, committed_at) in enumerate(
            zip(records, (10, 20, 30), (1_000, 1_100, 1_200), strict=True),
            start=1,
        )
    )
    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="identity reappears"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore(history),
            run_id="training-run",
        ).read()

    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="fencing token reappears"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((*history[:2], dataclasses.replace(history[2], revision=10))),
            run_id="training-run",
        ).read()


def test_coordinator_lease_history_rejects_noninitial_or_incomplete_history():
    noninitial = _entry(
        transaction_sequence=3,
        mutation_sequence=3,
        value_sequence=2,
        lifetime_sequence=2,
    )
    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="does not begin"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((noninitial,)),
            run_id="training-run",
        ).read()

    retained = _entry(transaction_sequence=1, revision=1)
    different_current = _entry(
        record=_record(coordinator_id="coordinator-b", lease_id="lease-b"),
        transaction_sequence=2,
        mutation_sequence=2,
        value_sequence=2,
        revision=2,
        committed_at_unix_ms=1_100,
    )
    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="absent from"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((retained,), current=different_current),
            run_id="training-run",
        ).read()

    with pytest.raises(CoordinatorLeaseHistoryCorrupt, match="durable history marker"):
        CoordinatorLeaseHistoryReader(
            StaticHistoryStore((), has_history=True),
            run_id="training-run",
        ).read()


def test_coordinator_lease_history_fails_after_repeatedly_unstable_reads():
    with pytest.raises(CoordinatorLeaseHistoryError, match="changed repeatedly"):
        CoordinatorLeaseHistoryReader(
            UnstableHistoryStore((_entry(),)),
            run_id="training-run",
        ).read()
