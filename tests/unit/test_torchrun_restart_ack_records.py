"""Contract tests for persisted torchrun restart acknowledgement receipts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    RestartAck,
    RestartIntent,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentRecord,
)


def _intent() -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id="training-run",
        generation=4,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=50_000,
    )


def _intent_record() -> RestartIntentRecord:
    return RestartIntentRecord(
        intent=_intent(),
        generation_snapshot_digest="a" * 64,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=30_000,
        coordinator_fencing_token=7,
    )


def _registration() -> AgentRegistrationRecord:
    return AgentRegistrationRecord(
        agent_identity=AgentIdentity(
            run_id="training-run",
            node_id="node-a",
            agent_id="agent-a",
            hostname="host-a",
            local_world_size=2,
            resource_ids=("gpu-a0", "gpu-a1"),
            environment_digest="environment-v1",
        ),
        registration_id="registration-a",
        lease_duration_ms=30_000,
    )


def _acknowledgement() -> RestartAck:
    return RestartAck(
        intent_id="intent-a",
        run_id="training-run",
        node_id="node-a",
        agent_id="agent-a",
        generation=4,
        flushed_step=40,
        inventory_event_digests={"inventory-a": "b" * 64},
        transferred_owner_ranks=(0, 1),
        transferred_peer_ranks=(2, 3),
        success=True,
        reason="prepared",
    )


def _record(
    *,
    acknowledgement: RestartAck | None = None,
    intent_record: RestartIntentRecord | None = None,
    agent_registration: AgentRegistrationRecord | None = None,
    registration_fencing_token: int = 9,
    registration_granted_at_unix_ms: int = 1_000,
    received_at_unix_ms: int = 20_000,
) -> RestartAckReceiptRecord:
    return RestartAckReceiptRecord(
        acknowledgement=acknowledgement or _acknowledgement(),
        intent_record=intent_record or _intent_record(),
        agent_registration=agent_registration or _registration(),
        registration_fencing_token=registration_fencing_token,
        registration_granted_at_unix_ms=registration_granted_at_unix_ms,
        received_at_unix_ms=received_at_unix_ms,
    )


def test_restart_ack_receipt_round_trips_canonical_json():
    record = _record()

    encoded = record.to_json()
    decoded = RestartAckReceiptRecord.from_json(encoded)

    assert decoded == record
    assert encoded == json.dumps(
        record.to_dict(),
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    assert record.digest == hashlib.sha256(encoded).hexdigest()


def test_restart_ack_receipt_reconstructs_authenticated_registration():
    record = _record()

    authenticated = record.authenticated_registration

    assert authenticated.record == record.agent_registration
    assert authenticated.fencing_token == record.registration_fencing_token
    assert authenticated.granted_at_unix_ms == record.registration_granted_at_unix_ms
    assert authenticated.expires_at_unix_ms == record.registration_expires_at_unix_ms


def test_restart_ack_receipt_is_immutable():
    record = _record()

    with pytest.raises(AttributeError):
        record.received_at_unix_ms = 20_001
    with pytest.raises(AttributeError):
        record.acknowledgement.inventory_event_digests = {}


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("intent_id", "intent-b", "restart intent"),
        ("run_id", "other-run", "restart intent"),
        ("generation", 5, "restart intent"),
    ],
)
def test_restart_ack_receipt_requires_matching_intent(field, value, message):
    acknowledgement = replace(_acknowledgement(), **{field: value})

    with pytest.raises(ValueError, match=message):
        _record(acknowledgement=acknowledgement)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("run_id", "other-run"),
        ("node_id", "node-b"),
        ("agent_id", "agent-b"),
    ],
)
def test_restart_ack_receipt_requires_matching_authenticated_identity(field, value):
    identity = replace(_registration().agent_identity, **{field: value})
    registration = replace(_registration(), agent_identity=identity)

    with pytest.raises(ValueError, match="authenticated agent registration"):
        _record(agent_registration=registration)


def test_restart_ack_receipt_accepts_registration_grant_boundary():
    record = _record(received_at_unix_ms=1_000)

    assert record.received_at_unix_ms == record.registration_granted_at_unix_ms


@pytest.mark.parametrize(
    ("received_at_unix_ms", "message"),
    [
        (999, "precedes"),
        (31_000, "registration lifetime"),
        (31_001, "registration lifetime"),
    ],
)
def test_restart_ack_receipt_rejects_receipt_outside_registration_lifetime(
    received_at_unix_ms,
    message,
):
    with pytest.raises(ValueError, match=message):
        _record(received_at_unix_ms=received_at_unix_ms)


@pytest.mark.parametrize("received_at_unix_ms", [50_000, 50_001])
def test_restart_ack_receipt_rejects_receipt_at_or_after_intent_deadline(
    received_at_unix_ms,
):
    with pytest.raises(ValueError, match="intent deadline"):
        _record(
            registration_granted_at_unix_ms=25_000,
            received_at_unix_ms=received_at_unix_ms,
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("acknowledgement", {}, "RestartAck"),
        ("intent_record", {}, "RestartIntentRecord"),
        ("agent_registration", {}, "AgentRegistrationRecord"),
    ],
)
def test_restart_ack_receipt_requires_record_types(field, value, message):
    kwargs = {
        "acknowledgement": _acknowledgement(),
        "intent_record": _intent_record(),
        "agent_registration": _registration(),
    }
    kwargs[field] = value

    with pytest.raises(TypeError, match=message):
        RestartAckReceiptRecord(
            **kwargs,
            registration_fencing_token=9,
            registration_granted_at_unix_ms=1_000,
            received_at_unix_ms=20_000,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("registration_fencing_token", 0),
        ("registration_fencing_token", True),
        ("registration_granted_at_unix_ms", 0),
        ("registration_granted_at_unix_ms", True),
        ("received_at_unix_ms", 0),
        ("received_at_unix_ms", True),
    ],
)
def test_restart_ack_receipt_validates_positive_metadata(field, value):
    with pytest.raises(ValueError, match=field):
        replace(_record(), **{field: value})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda value: value.pop("acknowledgement"), "missing fields"),
        (lambda value: value.update({"unknown": True}), "unknown fields"),
        (lambda value: value.update({"schema_version": 2}), "unsupported value"),
        (lambda value: value.update({"schema_version": True}), "unsupported value"),
        (lambda value: value.update({"schema_version": 1.0}), "unsupported value"),
        (
            lambda value: value.update({"registration_fencing_token": 0}),
            "registration_fencing_token",
        ),
        (
            lambda value: value.update({"registration_granted_at_unix_ms": 0}),
            "registration_granted_at_unix_ms",
        ),
        (
            lambda value: value.update({"received_at_unix_ms": 0}),
            "received_at_unix_ms",
        ),
    ],
)
def test_restart_ack_receipt_rejects_invalid_outer_wire_values(mutate, message):
    value = json.loads(_record().to_json())
    mutate(value)

    with pytest.raises(ValueError, match=message):
        RestartAckReceiptRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.parametrize(
    ("field", "mutate", "message"),
    [
        (
            "acknowledgement",
            lambda value: value.update({"schema_version": 1.0}),
            "schema_version",
        ),
        (
            "acknowledgement",
            lambda value: value.update({"agent_id": ""}),
            "acknowledgement is invalid",
        ),
        (
            "intent_record",
            lambda value: value.update({"schema_version": 1.0}),
            "intent_record is invalid",
        ),
        (
            "agent_registration",
            lambda value: value.update({"schema_version": 1.0}),
            "agent_registration is invalid",
        ),
        (
            "agent_registration",
            lambda value: value["agent_identity"].update({"schema_version": 1.0}),
            "agent_identity.schema_version",
        ),
    ],
)
def test_restart_ack_receipt_rejects_invalid_nested_wire_values(field, mutate, message):
    value = json.loads(_record().to_json())
    mutate(value[field])

    with pytest.raises(ValueError, match=message):
        RestartAckReceiptRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"not-json",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        (
            b'{"acknowledgement":{"schema_version":1,"schema_version":1},'
            b'"agent_registration":{},"intent_record":{},'
            b'"received_at_unix_ms":20000,"registration_fencing_token":9,'
            b'"registration_granted_at_unix_ms":1000,"schema_version":1}'
        ),
    ],
)
def test_restart_ack_receipt_rejects_malformed_or_duplicate_json(encoded):
    with pytest.raises(ValueError):
        RestartAckReceiptRecord.from_json(encoded)


def test_restart_ack_receipt_requires_encoded_bytes():
    with pytest.raises(ValueError, match="encoded bytes"):
        RestartAckReceiptRecord.from_json(_record().to_json().decode("utf-8"))
