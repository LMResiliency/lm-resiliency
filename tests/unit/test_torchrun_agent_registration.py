"""Contract tests for lease-backed torchrun agent registration."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationClockError,
    AgentRegistrationCorrupt,
    AgentRegistrationLost,
    AgentRegistrationManager,
    AgentRegistrationUnavailable,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._protocol import AgentIdentity


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


class ExpireDuringMutationStore(InMemoryControlStore):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock=clock)
        self._manual_clock = clock
        self.expire_next_mutation = False

    def compare_set_in_window(
        self,
        key: str,
        *,
        expected_revision: int | None,
        not_before_unix_ms: int,
        deadline_unix_ms: int | None,
        value: bytes,
    ):
        if self.expire_next_mutation:
            assert deadline_unix_ms is not None
            self._manual_clock.set(deadline_unix_ms)
            self.expire_next_mutation = False
        return super().compare_set_in_window(
            key,
            expected_revision=expected_revision,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
            value=value,
        )

    def compare_refresh_in_window(
        self,
        key: str,
        *,
        expected_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        value: bytes,
    ):
        if self.expire_next_mutation:
            self._manual_clock.set(deadline_unix_ms)
            self.expire_next_mutation = False
        return super().compare_refresh_in_window(
            key,
            expected_revision=expected_revision,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
            value=value,
        )

    def compare_delete_in_window(
        self,
        key: str,
        *,
        expected_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
    ):
        if self.expire_next_mutation:
            self._manual_clock.set(deadline_unix_ms)
            self.expire_next_mutation = False
        return super().compare_delete_in_window(
            key,
            expected_revision=expected_revision,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
        )


class DelayResponseStore(InMemoryControlStore):
    def __init__(self, clock: ManualClock) -> None:
        super().__init__(clock=clock)
        self._manual_clock = clock
        self.delay_next_response_ms = 0

    def compare_set_in_window(
        self,
        key: str,
        *,
        expected_revision: int | None,
        not_before_unix_ms: int,
        deadline_unix_ms: int | None,
        value: bytes,
    ):
        entry = super().compare_set_in_window(
            key,
            expected_revision=expected_revision,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
            value=value,
        )
        if self.delay_next_response_ms:
            self._manual_clock.advance(self.delay_next_response_ms)
            self.delay_next_response_ms = 0
        return entry

    def compare_refresh_in_window(
        self,
        key: str,
        *,
        expected_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        value: bytes,
    ):
        entry = super().compare_refresh_in_window(
            key,
            expected_revision=expected_revision,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=deadline_unix_ms,
            value=value,
        )
        if self.delay_next_response_ms:
            self._manual_clock.advance(self.delay_next_response_ms)
            self.delay_next_response_ms = 0
        return entry


def _identity(
    agent_id: str,
    *,
    run_id: str = "training-run",
    node_id: str = "node-a",
    environment_digest: str = "environment-v1",
) -> AgentIdentity:
    return AgentIdentity(
        run_id=run_id,
        node_id=node_id,
        agent_id=agent_id,
        hostname=f"host-{node_id}",
        local_world_size=2,
        resource_ids=(f"{node_id}-gpu-0", f"{node_id}-gpu-1"),
        environment_digest=environment_digest,
    )


def _manager(
    store: InMemoryControlStore,
    clock: ManualClock,
    agent_id: str,
    *,
    run_id: str = "training-run",
    node_id: str = "node-a",
    lease_duration_ms: int = 100,
    environment_digest: str = "environment-v1",
) -> AgentRegistrationManager:
    return AgentRegistrationManager(
        store,
        agent_identity=_identity(
            agent_id,
            run_id=run_id,
            node_id=node_id,
            environment_digest=environment_digest,
        ),
        lease_duration_ms=lease_duration_ms,
        clock=clock,
    )


def test_agent_registration_is_exclusive_and_same_agent_retry_is_idempotent():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    contender = _manager(store, clock, "agent-b")

    registration = first.register()
    retry = first.register()

    assert retry.record == registration.record
    assert retry.fencing_token > registration.fencing_token
    assert retry.granted_at_unix_ms == 1_000
    assert retry.expires_at_unix_ms == 1_100
    assert first.current() == retry
    with pytest.raises(AgentRegistrationUnavailable, match="agent-a"):
        contender.register()


def test_agent_registration_retry_rejects_configuration_drift():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    original = _manager(store, clock, "agent-a")
    duration_drift = _manager(
        store,
        clock,
        "agent-a",
        lease_duration_ms=200,
    )
    environment_drift = _manager(
        store,
        clock,
        "agent-a",
        environment_digest="environment-v2",
    )
    registration = original.register()

    with pytest.raises(AgentRegistrationUnavailable, match="lease duration"):
        duration_drift.register()
    with pytest.raises(AgentRegistrationUnavailable, match="agent-a"):
        environment_drift.register()

    assert original.current() == registration


def test_agent_registration_renewal_advances_fencing_token():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    original = manager.register()
    clock.advance(50)

    renewed = manager.renew(original)

    assert renewed.record.registration_id == original.record.registration_id
    assert renewed.granted_at_unix_ms == 1_050
    assert renewed.expires_at_unix_ms == 1_150
    assert renewed.fencing_token > original.fencing_token
    assert manager.current() == renewed
    with pytest.raises(AgentRegistrationLost, match="persisted ownership"):
        manager.release(original)


def test_agent_registration_expired_takeover_fences_old_agent():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    second = _manager(store, clock, "agent-b")
    stale = first.register()
    clock.set(stale.expires_at_unix_ms)

    replacement = second.register()

    assert replacement.record.agent_identity.agent_id == "agent-b"
    assert replacement.record.registration_id != stale.record.registration_id
    assert replacement.fencing_token > stale.fencing_token
    with pytest.raises(AgentRegistrationLost, match="persisted ownership"):
        first.renew(stale)


def test_agent_registration_release_and_reacquire_never_reuses_token():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    original = manager.register()

    tombstone_revision = manager.release(original)
    replacement = manager.register()

    assert original.fencing_token < tombstone_revision < replacement.fencing_token
    assert replacement.record.registration_id != original.record.registration_id


def test_distinct_nodes_and_runs_use_distinct_registration_keys():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a", run_id="run-a", node_id="node-a")
    second = _manager(store, clock, "agent-b", run_id="run-a", node_id="node-b")
    third = _manager(store, clock, "agent-c", run_id="run-b", node_id="node-a")

    assert (
        len(
            {
                first.registration_key,
                second.registration_key,
                third.registration_key,
            }
        )
        == 3
    )
    assert first.register().fencing_token == 1
    assert second.register().fencing_token == 1
    assert third.register().fencing_token == 1


def test_concurrent_agent_registration_has_one_winner_per_node():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    workers = 8
    barrier = threading.Barrier(workers)
    managers = [_manager(store, clock, f"agent-{index}") for index in range(workers)]

    def register(manager: AgentRegistrationManager):
        barrier.wait()
        try:
            return manager.register()
        except AgentRegistrationUnavailable:
            return None

    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(register, managers))

    winners = [result for result in results if result is not None]
    assert len(winners) == 1
    assert managers[0].current() == winners[0]


def test_agent_registration_rejects_fabricated_or_foreign_handle():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _manager(store, clock, "agent-a")
    second = _manager(store, clock, "agent-b")
    registration = first.register()
    fabricated = replace(
        registration,
        record=replace(
            registration.record,
            registration_id="forged-registration",
        ),
    )

    with pytest.raises(AgentRegistrationLost, match="persisted ownership"):
        first.renew(fabricated)
    with pytest.raises(AgentRegistrationLost, match="another manager"):
        second.release(registration)

    assert first.current() == registration


def test_agent_registration_fails_closed_on_corrupt_record():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    store.compare_set(
        manager.registration_key,
        expected_revision=None,
        value=b"{}",
    )

    with pytest.raises(AgentRegistrationCorrupt, match="malformed"):
        manager.current()
    with pytest.raises(AgentRegistrationCorrupt, match="malformed"):
        manager.register()


def test_agent_registration_requires_authoritative_commit_time():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    record = AgentRegistrationRecord(
        agent_identity=_identity("agent-a"),
        registration_id="registration-a",
        lease_duration_ms=100,
    )
    store.compare_set(
        manager.registration_key,
        expected_revision=None,
        value=record.to_json(),
    )

    with pytest.raises(AgentRegistrationCorrupt, match="commit time"):
        manager.current()


def test_agent_registration_rejects_record_for_another_node():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    record = AgentRegistrationRecord(
        agent_identity=_identity("agent-b", node_id="node-b"),
        registration_id="registration-b",
        lease_duration_ms=100,
    )
    store.compare_set_in_window(
        manager.registration_key,
        expected_revision=None,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=record.to_json(),
    )

    with pytest.raises(AgentRegistrationCorrupt, match="another run or node"):
        manager.current()


def test_registration_rejects_response_after_committed_expiry():
    clock = ManualClock()
    store = DelayResponseStore(clock)
    manager = _manager(store, clock, "agent-a")
    store.delay_next_response_ms = 100

    with pytest.raises(AgentRegistrationUnavailable, match="response arrived"):
        manager.register()

    expired = manager.current()
    assert expired is not None
    assert expired.expires_at_unix_ms == 1_100


def test_renewal_rejects_response_after_committed_expiry():
    clock = ManualClock()
    store = DelayResponseStore(clock)
    manager = _manager(store, clock, "agent-a")
    registration = manager.register()
    clock.set(1_050)
    store.delay_next_response_ms = 100

    with pytest.raises(AgentRegistrationLost, match="response arrived"):
        manager.renew(registration)

    expired = manager.current()
    assert expired is not None
    assert expired.expires_at_unix_ms == 1_150


def test_renewal_cannot_commit_after_store_observes_expiry():
    clock = ManualClock()
    store = ExpireDuringMutationStore(clock)
    manager = _manager(store, clock, "agent-a")
    registration = manager.register()
    clock.set(1_050)
    store.expire_next_mutation = True

    with pytest.raises(AgentRegistrationLost, match="expired at the control store"):
        manager.renew(registration)

    assert manager.current() == registration


def test_release_cannot_commit_after_store_observes_expiry():
    clock = ManualClock()
    store = ExpireDuringMutationStore(clock)
    manager = _manager(store, clock, "agent-a")
    registration = manager.register()
    clock.set(1_050)
    store.expire_next_mutation = True

    with pytest.raises(AgentRegistrationLost, match="expired at the control store"):
        manager.release(registration)

    assert manager.current() == registration


def test_agent_registration_clock_cannot_move_backward():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = _manager(store, clock, "agent-a")
    registration = manager.register()
    clock.set(999)

    with pytest.raises(AgentRegistrationClockError, match="backward"):
        manager.renew(registration)


def test_agent_registration_validates_constructor_inputs():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)

    with pytest.raises(TypeError, match="AgentIdentity"):
        AgentRegistrationManager(
            store,
            agent_identity={},
            lease_duration_ms=100,
            clock=clock,
        )
    with pytest.raises(ValueError, match="lease_duration_ms"):
        AgentRegistrationManager(
            store,
            agent_identity=_identity("agent-a"),
            lease_duration_ms=0,
            clock=clock,
        )
    with pytest.raises(TypeError, match="clock"):
        AgentRegistrationManager(
            store,
            agent_identity=_identity("agent-a"),
            lease_duration_ms=100,
            clock=None,
        )
