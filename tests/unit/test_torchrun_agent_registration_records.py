"""Contract tests for persisted torchrun agent registration records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._protocol import AgentIdentity


def _identity() -> AgentIdentity:
    return AgentIdentity(
        run_id="training-run",
        node_id="node-a",
        agent_id="agent-a",
        hostname="host-a",
        local_world_size=2,
        resource_ids=("gpu-a0", "gpu-a1", "hca-a"),
        environment_digest="environment-v1",
    )


def _record() -> AgentRegistrationRecord:
    return AgentRegistrationRecord(
        agent_identity=_identity(),
        registration_id="registration-a",
        lease_duration_ms=30_000,
    )


def test_agent_registration_record_round_trips_strict_json():
    record = _record()

    encoded = record.to_json()
    decoded = AgentRegistrationRecord.from_json(encoded)

    assert decoded == record
    assert encoded == (
        b'{"agent_identity":{"agent_id":"agent-a",'
        b'"environment_digest":"environment-v1","hostname":"host-a",'
        b'"local_world_size":2,"node_id":"node-a",'
        b'"resource_ids":["gpu-a0","gpu-a1","hca-a"],'
        b'"run_id":"training-run","schema_version":1},'
        b'"lease_duration_ms":30000,"registration_id":"registration-a",'
        b'"schema_version":1}'
    )


def test_agent_registration_record_is_immutable():
    record = _record()

    with pytest.raises(AttributeError):
        record.registration_id = "other"
    with pytest.raises(AttributeError):
        record.agent_identity.resource_ids = ()


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.pop("agent_identity"),
            "missing fields",
        ),
        (
            lambda value: value.update({"unknown": True}),
            "unknown fields",
        ),
        (
            lambda value: value.update({"schema_version": 2}),
            "unsupported value",
        ),
        (
            lambda value: value.update({"schema_version": True}),
            "unsupported value",
        ),
        (
            lambda value: value.update({"agent_identity": []}),
            "agent_identity must be an object",
        ),
        (
            lambda value: value["agent_identity"].update({"run_id": ""}),
            "agent_identity is invalid",
        ),
        (
            lambda value: value.update({"registration_id": ""}),
            "registration_id",
        ),
        (
            lambda value: value.update({"lease_duration_ms": 0}),
            "lease_duration_ms",
        ),
        (
            lambda value: value.update({"lease_duration_ms": True}),
            "lease_duration_ms",
        ),
    ],
)
def test_agent_registration_record_rejects_invalid_wire_values(mutate, message):
    value = json.loads(_record().to_json())
    mutate(value)

    with pytest.raises(ValueError, match=message):
        AgentRegistrationRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"not-json",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        (
            b'{"agent_identity":{"schema_version":1,"run_id":"training-run",'
            b'"run_id":"other-run","node_id":"node-a","agent_id":"agent-a",'
            b'"hostname":"host-a","local_world_size":2,"resource_ids":[],'
            b'"environment_digest":"environment-v1"},"registration_id":"registration-a",'
            b'"lease_duration_ms":30000,"schema_version":1}'
        ),
    ],
)
def test_agent_registration_record_rejects_malformed_or_duplicate_json(encoded):
    with pytest.raises(ValueError):
        AgentRegistrationRecord.from_json(encoded)


def test_agent_registration_record_requires_encoded_bytes():
    with pytest.raises(ValueError, match="encoded bytes"):
        AgentRegistrationRecord.from_json(_record().to_json().decode("utf-8"))


def test_agent_registration_record_requires_agent_identity():
    with pytest.raises(TypeError, match="AgentIdentity"):
        AgentRegistrationRecord(
            agent_identity={},
            registration_id="registration-a",
            lease_duration_ms=30_000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registration_id", ""),
        ("lease_duration_ms", 0),
        ("lease_duration_ms", -1),
        ("lease_duration_ms", True),
    ],
)
def test_agent_registration_record_validates_constructor_fields(field, value):
    with pytest.raises(ValueError):
        replace(_record(), **{field: value})


def test_held_agent_registration_derives_expiry():
    held = HeldAgentRegistration(
        record=_record(),
        fencing_token=7,
        granted_at_unix_ms=1_000,
    )

    assert held.expires_at_unix_ms == 31_000


def test_held_agent_registration_is_immutable():
    held = HeldAgentRegistration(
        record=_record(),
        fencing_token=7,
        granted_at_unix_ms=1_000,
    )

    with pytest.raises(AttributeError):
        held.fencing_token = 8


def test_held_agent_registration_requires_registration_record():
    with pytest.raises(TypeError, match="AgentRegistrationRecord"):
        HeldAgentRegistration(
            record={},
            fencing_token=7,
            granted_at_unix_ms=1_000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("fencing_token", 0),
        ("fencing_token", True),
        ("granted_at_unix_ms", 0),
        ("granted_at_unix_ms", True),
    ],
)
def test_held_agent_registration_validates_positive_metadata(field, value):
    held = HeldAgentRegistration(
        record=_record(),
        fencing_token=7,
        granted_at_unix_ms=1_000,
    )

    with pytest.raises(ValueError):
        replace(held, **{field: value})
