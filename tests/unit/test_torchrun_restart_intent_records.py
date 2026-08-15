"""Contract tests for persisted torchrun restart-intent records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._protocol import RestartIntent
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentRecord,
)


def _intent() -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id="training-run",
        generation=4,
        incident_ids=("incident-a", "incident-b"),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=50_000,
    )


def _record() -> RestartIntentRecord:
    return RestartIntentRecord(
        intent=_intent(),
        generation_snapshot_digest="a" * 64,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=30_000,
        coordinator_fencing_token=7,
    )


def test_restart_intent_record_round_trips_canonical_json():
    record = _record()

    encoded = record.to_json()
    decoded = RestartIntentRecord.from_json(encoded)

    assert decoded == record
    assert encoded == (
        b'{"coordinator_fencing_token":7,"coordinator_id":"coordinator-a",'
        b'"coordinator_lease_duration_ms":30000,'
        b'"generation_snapshot_digest":"'
        + b"a"
        * 64
        + b'","intent":{"generation":4,"incident_ids":["incident-a","incident-b"],'
        b'"intent_id":"intent-a","minimum_recovery_mode":"recovery_verified",'
        b'"prepare_deadline_unix_ms":50000,"reason_code":"attributed_sdc",'
        b'"run_id":"training-run","schema_version":1,'
        b'"suspected_node_ids":["node-b"]},"lease_id":"lease-a","schema_version":1}'
    )
    assert record.digest == hashlib.sha256(encoded).hexdigest()
    lease_record = CoordinatorLeaseRecord(
        run_id=record.intent.run_id,
        coordinator_id=record.coordinator_id,
        lease_id=record.lease_id,
        lease_duration_ms=record.coordinator_lease_duration_ms,
    )
    assert record.coordinator_lease_digest == hashlib.sha256(lease_record.to_json()).hexdigest()


def test_restart_intent_record_is_immutable():
    record = _record()

    with pytest.raises(AttributeError):
        record.lease_id = "other"
    with pytest.raises(AttributeError):
        record.intent.incident_ids = ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("generation_snapshot_digest", "", "generation_snapshot_digest"),
        ("generation_snapshot_digest", "A" * 64, "generation_snapshot_digest"),
        ("coordinator_id", "", "coordinator_id"),
        ("lease_id", "", "lease_id"),
        ("coordinator_lease_duration_ms", 0, "coordinator_lease_duration_ms"),
        ("coordinator_lease_duration_ms", True, "coordinator_lease_duration_ms"),
        ("coordinator_fencing_token", 0, "coordinator_fencing_token"),
        ("coordinator_fencing_token", True, "coordinator_fencing_token"),
    ],
)
def test_restart_intent_record_validates_constructor_fields(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(_record(), **{field: value})


def test_restart_intent_record_requires_restart_intent():
    with pytest.raises(TypeError, match="RestartIntent"):
        RestartIntentRecord(
            intent={},
            generation_snapshot_digest="a" * 64,
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            coordinator_lease_duration_ms=30_000,
            coordinator_fencing_token=7,
        )


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.pop("intent"),
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
            lambda value: value.update({"intent": []}),
            "intent must be an object",
        ),
        (
            lambda value: value["intent"].update({"generation": -1}),
            "intent is invalid",
        ),
        (
            lambda value: value["intent"].update({"schema_version": 2}),
            "intent is invalid",
        ),
        (
            lambda value: value["intent"].update({"unknown": True}),
            "intent is invalid",
        ),
        (
            lambda value: value.update({"generation_snapshot_digest": "A" * 64}),
            "generation_snapshot_digest",
        ),
        (
            lambda value: value.update({"coordinator_id": ""}),
            "coordinator_id",
        ),
        (
            lambda value: value.update({"lease_id": ""}),
            "lease_id",
        ),
        (
            lambda value: value.update({"coordinator_lease_duration_ms": 0}),
            "coordinator_lease_duration_ms",
        ),
        (
            lambda value: value.update({"coordinator_fencing_token": 0}),
            "coordinator_fencing_token",
        ),
    ],
)
def test_restart_intent_record_rejects_invalid_wire_values(mutate, message):
    value = json.loads(_record().to_json())
    mutate(value)

    with pytest.raises(ValueError, match=message):
        RestartIntentRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"not-json",
        b"[]",
        b'{"schema_version":1,"schema_version":1}',
        (
            b'{"coordinator_fencing_token":7,"coordinator_id":"coordinator-a",'
            b'"coordinator_lease_duration_ms":30000,'
            b'"generation_snapshot_digest":"'
            + b"a"
            * 64
            + b'","intent":{"generation":4,"generation":5,'
            b'"incident_ids":["incident-a"],"intent_id":"intent-a",'
            b'"minimum_recovery_mode":"latest","prepare_deadline_unix_ms":50000,'
            b'"reason_code":"failure","run_id":"training-run","schema_version":1,'
            b'"suspected_node_ids":[]},"lease_id":"lease-a","schema_version":1}'
        ),
    ],
)
def test_restart_intent_record_rejects_malformed_or_duplicate_json(encoded):
    with pytest.raises(ValueError):
        RestartIntentRecord.from_json(encoded)


def test_restart_intent_record_requires_encoded_bytes():
    with pytest.raises(ValueError, match="encoded bytes"):
        RestartIntentRecord.from_json(_record().to_json().decode("utf-8"))
