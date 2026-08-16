"""Contract tests for torchrun agent-registration authority values."""

from __future__ import annotations

import dataclasses
import threading
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
    AgentRegistrationAuthorityCorrupt,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._protocol import AgentIdentity


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms

    def advance(self, duration_ms: int) -> None:
        with self._lock:
            self.now_unix_ms += duration_ms


def _identity(
    *,
    run_id: str = "training-run",
    node_id: str = "node-a",
    agent_id: str = "agent-a",
) -> AgentIdentity:
    return AgentIdentity(
        run_id=run_id,
        node_id=node_id,
        agent_id=agent_id,
        hostname=f"host-{node_id}",
        local_world_size=2,
        resource_ids=(f"{node_id}-gpu-0", f"{node_id}-gpu-1"),
        environment_digest="environment-v1",
    )


def _record(
    *,
    run_id: str = "training-run",
    node_id: str = "node-a",
) -> AgentRegistrationRecord:
    return AgentRegistrationRecord(
        agent_identity=_identity(run_id=run_id, node_id=node_id),
        registration_id="registration-a",
        lease_duration_ms=100,
    )


def _entry(
    *,
    record: AgentRegistrationRecord | None = None,
    revision: int = 7,
    committed_at_unix_ms: int | None = 1_000,
    transaction_sequence: int = 11,
    mutation_sequence: int = 1,
    value_sequence: int = 1,
    lifetime_sequence: int = 1,
) -> ControlStoreEntry:
    return ControlStoreEntry(
        value=(record or _record()).to_json(),
        revision=revision,
        committed_at_unix_ms=committed_at_unix_ms,
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
    )


def test_agent_registration_authority_decodes_real_registration_and_renewal():
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager = AgentRegistrationManager(
        store,
        agent_identity=_identity(),
        lease_duration_ms=100,
        clock=clock,
    )
    opened = manager.register()
    clock.advance(50)
    renewed = manager.renew(opened)
    key = agent_registration_key("training-run", "node-a")

    authorities = tuple(
        AgentRegistrationAuthority.from_entry(
            entry,
            run_id="training-run",
            node_id="node-a",
        )
        for entry in store.get_history(key)
    )

    assert tuple(authority.registration for authority in authorities) == (
        opened,
        renewed,
    )
    assert tuple(authority.mutation_sequence for authority in authorities) == (1, 2)
    assert tuple(authority.value_sequence for authority in authorities) == (1, 1)
    assert tuple(authority.lifetime_sequence for authority in authorities) == (1, 1)


def test_agent_registration_authority_is_immutable():
    authority = AgentRegistrationAuthority.from_entry(
        _entry(),
        run_id="training-run",
        node_id="node-a",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        authority.mutation_sequence = 2


def test_agent_registration_authority_decodes_canonical_entry():
    authority = AgentRegistrationAuthority.from_entry(
        _entry(),
        run_id="training-run",
        node_id="node-a",
    )

    assert authority == AgentRegistrationAuthority(
        registration=HeldAgentRegistration(
            record=_record(),
            fencing_token=7,
            granted_at_unix_ms=1_000,
        ),
        transaction_sequence=11,
        mutation_sequence=1,
        value_sequence=1,
        lifetime_sequence=1,
    )


@pytest.mark.parametrize(
    ("mutation_sequence", "value_sequence", "lifetime_sequence", "message"),
    [
        (2, 1, 2, "mutation_sequence is too small"),
        (3, 1, 2, "value_sequence is too small"),
        (3, 3, 2, "value_sequence is too large"),
    ],
)
def test_agent_registration_authority_rejects_impossible_sequences(
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        AgentRegistrationAuthority.from_entry(
            _entry(
                mutation_sequence=mutation_sequence,
                value_sequence=value_sequence,
                lifetime_sequence=lifetime_sequence,
            ),
            run_id="training-run",
            node_id="node-a",
        )


def test_agent_registration_authority_rejects_transaction_before_mutation():
    with pytest.raises(ValueError, match="transaction_sequence is too small"):
        AgentRegistrationAuthority.from_entry(
            dataclasses.replace(
                _entry(mutation_sequence=2, value_sequence=1),
                transaction_sequence=1,
            ),
            run_id="training-run",
            node_id="node-a",
        )


def test_agent_registration_authority_rejects_malformed_or_wrong_identity():
    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="malformed"):
        AgentRegistrationAuthority.from_entry(
            dataclasses.replace(_entry(), value=b"not-json"),
            run_id="training-run",
            node_id="node-a",
        )

    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="another run or node"):
        AgentRegistrationAuthority.from_entry(
            _entry(record=_record(run_id="other-run")),
            run_id="training-run",
            node_id="node-a",
        )

    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="another run or node"):
        AgentRegistrationAuthority.from_entry(
            _entry(record=_record(node_id="node-b")),
            run_id="training-run",
            node_id="node-a",
        )


def test_agent_registration_authority_rejects_noncanonical_bytes():
    entry = _entry()

    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="noncanonical"):
        AgentRegistrationAuthority.from_entry(
            dataclasses.replace(entry, value=entry.value.replace(b",", b", ")),
            run_id="training-run",
            node_id="node-a",
        )


def test_agent_registration_authority_requires_authoritative_commit_time():
    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="commit time"):
        AgentRegistrationAuthority.from_entry(
            _entry(committed_at_unix_ms=None),
            run_id="training-run",
            node_id="node-a",
        )


def test_agent_registration_authority_rejects_guarded_entry():
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

    with pytest.raises(AgentRegistrationAuthorityCorrupt, match="guard provenance"):
        AgentRegistrationAuthority.from_entry(
            entry,
            run_id="training-run",
            node_id="node-a",
        )


@pytest.mark.parametrize("run_id", ("", " ", None, 1))
def test_agent_registration_authority_rejects_invalid_run_id(run_id: object):
    with pytest.raises(ValueError, match="run_id"):
        AgentRegistrationAuthority.from_entry(
            _entry(),
            run_id=cast(Any, run_id),
            node_id="node-a",
        )


@pytest.mark.parametrize("node_id", ("", " ", None, 1))
def test_agent_registration_authority_rejects_invalid_node_id(node_id: object):
    with pytest.raises(ValueError, match="node_id"):
        AgentRegistrationAuthority.from_entry(
            _entry(),
            run_id="training-run",
            node_id=cast(Any, node_id),
        )


def test_agent_registration_authority_rejects_wrong_entry_type():
    with pytest.raises(TypeError, match="ControlStoreEntry"):
        AgentRegistrationAuthority.from_entry(
            object(),
            run_id="training-run",
            node_id="node-a",
        )
