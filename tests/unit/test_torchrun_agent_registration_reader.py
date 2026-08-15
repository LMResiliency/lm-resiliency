"""Contract tests for trusted torchrun registration observation."""

from __future__ import annotations

import threading

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationClockError,
    AgentRegistrationCorrupt,
    AgentRegistrationManager,
    AgentRegistrationObservation,
    AgentRegistrationReader,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
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


def _identity(node_id: str, agent_id: str | None = None) -> AgentIdentity:
    return AgentIdentity(
        run_id="training-run",
        node_id=node_id,
        agent_id=agent_id or f"agent-{node_id}",
        hostname=f"host-{node_id}",
        local_world_size=2,
        resource_ids=(f"{node_id}-gpu-0", f"{node_id}-gpu-1"),
        environment_digest="environment-v1",
    )


def _register(
    store: InMemoryControlStore,
    clock: ManualClock,
    node_id: str,
) -> HeldAgentRegistration:
    return AgentRegistrationManager(
        store,
        agent_identity=_identity(node_id),
        lease_duration_ms=100,
        clock=clock,
    ).register()


def test_registration_reader_uses_same_hashed_key_as_manager():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    identity = _identity("node-a")
    manager = AgentRegistrationManager(
        store,
        agent_identity=identity,
        lease_duration_ms=100,
        clock=clock,
    )

    assert manager.registration_key == agent_registration_key(
        identity.run_id,
        identity.node_id,
    )
    assert "training-run" not in manager.registration_key
    assert "node-a" not in manager.registration_key


def test_registration_reader_gets_only_requested_trusted_node():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    node_a = _register(store, clock, "node-a")
    _register(store, clock, "node-b")
    reader = AgentRegistrationReader(
        store,
        run_id="training-run",
        clock=clock,
    )

    assert reader.get("node-a") == node_a
    assert reader.get("node-c") is None


def test_registration_reader_classifies_live_expired_and_missing_nodes():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    expired = _register(store, clock, "node-a")
    clock.set(1_050)
    live = _register(store, clock, "node-b")
    clock.set(1_100)
    reader = AgentRegistrationReader(
        store,
        run_id="training-run",
        clock=clock,
    )

    observation = reader.observe(("node-a", "node-b", "node-c"))

    assert observation.observed_at_unix_ms == 1_100
    assert dict(observation.live) == {"node-b": live}
    assert dict(observation.expired) == {"node-a": expired}
    assert observation.missing_node_ids == ("node-c",)
    with pytest.raises(TypeError):
        observation.live["node-c"] = live


def test_registration_reader_rejects_future_authoritative_grant():
    clock = ManualClock(1_100)
    store = InMemoryControlStore(clock=clock)
    _register(store, clock, "node-a")
    clock.set(1_000)
    reader = AgentRegistrationReader(
        store,
        run_id="training-run",
        clock=clock,
    )

    with pytest.raises(AgentRegistrationClockError, match="precedes"):
        reader.observe(("node-a",))


def test_registration_reader_clock_cannot_move_backward():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    reader = AgentRegistrationReader(
        store,
        run_id="training-run",
        clock=clock,
    )
    reader.observe(("node-a",))
    clock.set(999)

    with pytest.raises(AgentRegistrationClockError, match="backward"):
        reader.observe(("node-a",))


def test_registration_reader_fails_closed_on_wrong_node_record():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    record = AgentRegistrationRecord(
        agent_identity=_identity("node-b"),
        registration_id="registration-b",
        lease_duration_ms=100,
    )
    store.compare_set_in_window(
        agent_registration_key("training-run", "node-a"),
        expected_revision=None,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=record.to_json(),
    )
    reader = AgentRegistrationReader(
        store,
        run_id="training-run",
        clock=clock,
    )

    with pytest.raises(AgentRegistrationCorrupt, match="another run or node"):
        reader.get("node-a")


@pytest.mark.parametrize(
    "node_ids",
    [
        (),
        ("node-a", "node-a"),
        "node-a",
        ("",),
    ],
)
def test_registration_reader_rejects_invalid_trusted_node_sets(node_ids):
    reader = AgentRegistrationReader(
        InMemoryControlStore(),
        run_id="training-run",
        clock=ManualClock(),
    )

    with pytest.raises((TypeError, ValueError)):
        reader.observe(node_ids)


def test_registration_observation_validates_disjoint_classification():
    registration = HeldAgentRegistration(
        record=AgentRegistrationRecord(
            agent_identity=_identity("node-a"),
            registration_id="registration-a",
            lease_duration_ms=100,
        ),
        fencing_token=1,
        granted_at_unix_ms=1_000,
    )

    with pytest.raises(ValueError, match="disjoint"):
        AgentRegistrationObservation(
            observed_at_unix_ms=1_050,
            live={"node-a": registration},
            expired={},
            missing_node_ids=("node-a",),
        )


def test_registration_observation_requires_matching_node_key():
    registration = HeldAgentRegistration(
        record=AgentRegistrationRecord(
            agent_identity=_identity("node-a"),
            registration_id="registration-a",
            lease_duration_ms=100,
        ),
        fencing_token=1,
        granted_at_unix_ms=1_000,
    )

    with pytest.raises(ValueError, match="does not match key"):
        AgentRegistrationObservation(
            observed_at_unix_ms=1_050,
            live={"node-b": registration},
            expired={},
            missing_node_ids=(),
        )
