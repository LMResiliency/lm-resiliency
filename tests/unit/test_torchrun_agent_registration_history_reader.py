"""Contract tests for stable agent-registration history reads."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistory,
    AgentRegistrationHistoryCorrupt,
    AgentRegistrationHistoryError,
    AgentRegistrationHistoryReader,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._protocol import AgentIdentity

RUN_ID = "training-run"
NODE_ID = "node-a"


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
        *,
        key: str,
        history: tuple[ControlStoreEntry, ...],
        current: ControlStoreEntry | None,
        has_history: bool,
    ) -> None:
        self._key = key
        self._history = history
        self._current = current
        self._has_history = has_history

    def get_history(self, key: str) -> tuple[ControlStoreEntry, ...]:
        return self._history if key == self._key else ()

    def get(self, key: str) -> ControlStoreEntry | None:
        return self._current if key == self._key else None

    def has_history(self, key: str) -> bool:
        return self._has_history if key == self._key else False


class UnstableHistoryStore(StaticHistoryStore):
    def __init__(self, *, key: str, history: tuple[ControlStoreEntry, ...]) -> None:
        super().__init__(
            key=key,
            history=history,
            current=history[-1],
            has_history=True,
        )
        self._history_reads = 0

    def get_history(self, key: str) -> tuple[ControlStoreEntry, ...]:
        self._history_reads += 1
        history = super().get_history(key)
        return history[:-1] if self._history_reads % 2 else history


def _identity(agent_id: str) -> AgentIdentity:
    return AgentIdentity(
        run_id=RUN_ID,
        node_id=NODE_ID,
        agent_id=agent_id,
        hostname=f"host-{agent_id}",
        local_world_size=2,
        resource_ids=(f"{agent_id}-gpu-0", f"{agent_id}-gpu-1"),
        environment_digest="environment-v1",
    )


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    agent_id: str,
) -> AgentRegistrationManager:
    return AgentRegistrationManager(
        store,
        agent_identity=_identity(agent_id),
        lease_duration_ms=100,
        clock=clock,
    )


def test_registration_history_reader_returns_empty_state_before_registration():
    reader = AgentRegistrationHistoryReader(
        InMemoryControlStore(),
        run_id=RUN_ID,
        node_id=NODE_ID,
    )

    history = reader.read()

    assert history.authorities == ()
    assert history.current is None
    assert reader.registration_key == agent_registration_key(RUN_ID, NODE_ID)


def test_registration_history_reader_verifies_renewal_and_replacement():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    initial = first.register()
    clock.set(1_010)
    renewed = first.renew(initial)
    clock.set(renewed.expires_at_unix_ms)
    replacement = _manager(store, clock, "agent-b").register()

    history = AgentRegistrationHistoryReader(
        store,
        run_id=RUN_ID,
        node_id=NODE_ID,
    ).read()

    assert tuple(authority.registration for authority in history.authorities) == (
        initial,
        renewed,
        replacement,
    )
    assert history.current == replacement
    assert tuple(authority.mutation_sequence for authority in history.authorities) == (
        1,
        2,
        3,
    )
    assert tuple(authority.value_sequence for authority in history.authorities) == (
        1,
        1,
        2,
    )


def test_registration_history_reader_preserves_release_and_recreate():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    initial = first.register()
    clock.set(1_010)
    first.release(initial)
    replacement = _manager(store, clock, "agent-b").register()

    history = AgentRegistrationHistoryReader(
        store,
        run_id=RUN_ID,
        node_id=NODE_ID,
    ).read()

    assert tuple(authority.registration for authority in history.authorities) == (
        initial,
        replacement,
    )
    assert history.current == replacement
    assert history.authorities[-1].mutation_sequence == 3
    assert history.authorities[-1].lifetime_sequence == 2


def test_registration_history_reader_preserves_released_state():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    initial = manager.register()
    clock.set(1_010)
    manager.release(initial)

    history = AgentRegistrationHistoryReader(
        store,
        run_id=RUN_ID,
        node_id=NODE_ID,
    ).read()

    assert tuple(authority.registration for authority in history.authorities) == (initial,)
    assert history.current is None


def test_registration_history_reader_rejects_truncated_history():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    initial = manager.register()
    clock.set(1_010)
    manager.renew(initial)
    retained = store.get_history(manager.registration_key)

    with pytest.raises(AgentRegistrationHistoryCorrupt, match="initial store sequences"):
        AgentRegistrationHistoryReader(
            StaticHistoryStore(
                key=manager.registration_key,
                history=retained[1:],
                current=retained[-1],
                has_history=True,
            ),
            run_id=RUN_ID,
            node_id=NODE_ID,
        ).read()


@pytest.mark.parametrize(
    ("history", "current", "has_history", "message"),
    [
        ((), None, True, "durable marker"),
        (
            (),
            ControlStoreEntry(value=b"value", revision=1),
            False,
            "absent from its value history",
        ),
    ],
)
def test_registration_history_reader_rejects_store_contradictions(
    history: tuple[ControlStoreEntry, ...],
    current: ControlStoreEntry | None,
    has_history: bool,
    message: str,
):
    key = agent_registration_key(RUN_ID, NODE_ID)

    with pytest.raises(AgentRegistrationHistoryCorrupt, match=message):
        AgentRegistrationHistoryReader(
            StaticHistoryStore(
                key=key,
                history=history,
                current=current,
                has_history=has_history,
            ),
            run_id=RUN_ID,
            node_id=NODE_ID,
        ).read()


def test_registration_history_reader_rejects_current_missing_from_history():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    manager.register()
    retained = store.get_history(manager.registration_key)
    changed = replace(retained[-1], revision=retained[-1].revision + 1)

    with pytest.raises(AgentRegistrationHistoryCorrupt, match="absent from its value history"):
        AgentRegistrationHistoryReader(
            StaticHistoryStore(
                key=manager.registration_key,
                history=retained,
                current=changed,
                has_history=True,
            ),
            run_id=RUN_ID,
            node_id=NODE_ID,
        ).read()


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
        (3, 2, 1, 1, 1_010, "transaction sequences"),
        (4, 3, 1, 1, 1_010, "omits a key mutation"),
        (3, 2, 2, 1, 1_010, "value sequence"),
        (5, 5, 3, 3, 1_010, "key lifetime"),
        (3, 2, 1, 1, 999, "grant times"),
    ],
)
def test_registration_history_rejects_impossible_transition(
    transaction_sequence: int,
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    committed_at_unix_ms: int,
    message: str,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    initial = manager.register()
    clock.set(1_010)
    manager.renew(initial)
    retained = store.get_history(manager.registration_key)
    first = (
        replace(retained[0], transaction_sequence=3)
        if message == "transaction sequences"
        else retained[0]
    )
    changed = replace(
        retained[-1],
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
        committed_at_unix_ms=committed_at_unix_ms,
    )

    with pytest.raises(AgentRegistrationHistoryCorrupt, match=message):
        AgentRegistrationHistory(
            authorities=tuple(
                AgentRegistrationAuthority.from_entry(
                    entry,
                    run_id=RUN_ID,
                    node_id=NODE_ID,
                )
                for entry in (first, changed)
            ),
            current=None,
        )


def test_registration_history_rejects_expired_renewal_and_overlap():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    initial = first.register()
    first_entry = store.get(first.registration_key)
    assert first_entry is not None
    expired_renewal = replace(
        first_entry,
        revision=first_entry.revision + 1,
        committed_at_unix_ms=initial.expires_at_unix_ms,
        transaction_sequence=first_entry.transaction_sequence + 1,
        mutation_sequence=2,
    )
    replacement_record = AgentRegistrationRecord(
        agent_identity=_identity("agent-b"),
        registration_id="registration-b",
        lease_duration_ms=100,
    )
    overlapping = replace(
        expired_renewal,
        value=replacement_record.to_json(),
        committed_at_unix_ms=initial.expires_at_unix_ms - 1,
        value_sequence=2,
    )

    with pytest.raises(AgentRegistrationHistoryCorrupt, match="expired registration"):
        AgentRegistrationHistory(
            authorities=tuple(
                AgentRegistrationAuthority.from_entry(
                    entry,
                    run_id=RUN_ID,
                    node_id=NODE_ID,
                )
                for entry in (first_entry, expired_renewal)
            ),
            current=None,
        )
    with pytest.raises(AgentRegistrationHistoryCorrupt, match="overlap"):
        AgentRegistrationHistory(
            authorities=tuple(
                AgentRegistrationAuthority.from_entry(
                    entry,
                    run_id=RUN_ID,
                    node_id=NODE_ID,
                )
                for entry in (first_entry, overlapping)
            ),
            current=None,
        )


def test_registration_history_rejects_registration_and_token_replay():
    records = (
        AgentRegistrationRecord(_identity("agent-a"), "registration-a", 100),
        AgentRegistrationRecord(_identity("agent-b"), "registration-b", 100),
        AgentRegistrationRecord(_identity("agent-c"), "registration-a", 100),
    )
    entries = tuple(
        ControlStoreEntry(
            value=record.to_json(),
            revision=revision,
            committed_at_unix_ms=committed_at_unix_ms,
            transaction_sequence=index,
            mutation_sequence=index,
            value_sequence=index,
        )
        for index, (record, revision, committed_at_unix_ms) in enumerate(
            zip(records, (10, 20, 30), (1_000, 1_100, 1_200), strict=True),
            start=1,
        )
    )
    authorities = tuple(
        AgentRegistrationAuthority.from_entry(
            entry,
            run_id=RUN_ID,
            node_id=NODE_ID,
        )
        for entry in entries
    )

    with pytest.raises(AgentRegistrationHistoryCorrupt, match="identity reappears"):
        AgentRegistrationHistory(authorities=authorities, current=None)

    repeated_token = replace(entries[-1], value_sequence=3, revision=10)
    with pytest.raises(AgentRegistrationHistoryCorrupt, match="fencing token reappears"):
        AgentRegistrationHistory(
            authorities=(
                authorities[0],
                authorities[1],
                AgentRegistrationAuthority.from_entry(
                    repeated_token,
                    run_id=RUN_ID,
                    node_id=NODE_ID,
                ),
            ),
            current=None,
        )


def test_registration_history_reader_reports_repeated_contention():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    manager.register()
    retained = store.get_history(manager.registration_key)

    with pytest.raises(AgentRegistrationHistoryError, match="changed repeatedly"):
        AgentRegistrationHistoryReader(
            UnstableHistoryStore(
                key=manager.registration_key,
                history=retained,
            ),
            run_id=RUN_ID,
            node_id=NODE_ID,
        ).read()


def test_registration_history_value_rejects_invalid_types_and_current_tail():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    registration = manager.register()
    entry = store.get(manager.registration_key)
    assert entry is not None
    authority = AgentRegistrationAuthority.from_entry(
        entry,
        run_id=RUN_ID,
        node_id=NODE_ID,
    )

    with pytest.raises(TypeError, match="authorities"):
        AgentRegistrationHistory(authorities=[], current=None)
    with pytest.raises(TypeError, match="current"):
        AgentRegistrationHistory(authorities=(authority,), current={})
    with pytest.raises(ValueError, match="history tail"):
        AgentRegistrationHistory(
            authorities=(authority,),
            current=replace(
                registration,
                fencing_token=registration.fencing_token + 1,
            ),
        )
