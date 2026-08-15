"""Contract tests for persisted torchrun node quarantine records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
)


def _record() -> NodeQuarantineRecord:
    return NodeQuarantineRecord(
        run_id="training-run",
        node_id="node-b",
        plan_id="plan-a",
        intent_id="intent-a",
        from_generation=0,
        effective_generation=1,
        incident_ids=("incident-a", "incident-b"),
        reason_code="attributed_sdc",
        resource_ids=("gpu-b0",),
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=30_000,
        coordinator_fencing_token=7,
    )


def test_node_quarantine_record_round_trips_canonical_json():
    record = _record()

    encoded = record.to_json()
    decoded = NodeQuarantineRecord.from_json(encoded)

    assert decoded == record
    assert encoded == (
        b'{"coordinator_fencing_token":7,"coordinator_id":"coordinator-a",'
        b'"coordinator_lease_duration_ms":30000,"effective_generation":1,'
        b'"from_generation":0,'
        b'"incident_ids":["incident-a","incident-b"],"intent_id":"intent-a",'
        b'"lease_id":"lease-a","node_id":"node-b","plan_id":"plan-a",'
        b'"reason_code":"attributed_sdc","resource_ids":["gpu-b0"],'
        b'"run_id":"training-run","schema_version":2}'
    )
    assert record.digest == hashlib.sha256(encoded).hexdigest()


def test_node_quarantine_record_is_immutable():
    record = _record()

    with pytest.raises(AttributeError):
        record.node_id = "node-c"
    with pytest.raises(AttributeError):
        record.incident_ids = ()


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "", "run_id"),
        ("node_id", "", "node_id"),
        ("plan_id", "", "plan_id"),
        ("intent_id", "", "intent_id"),
        ("from_generation", -1, "from_generation"),
        ("from_generation", True, "from_generation"),
        ("effective_generation", 0, "effective_generation"),
        ("effective_generation", 2, "successor generation"),
        ("incident_ids", (), "at least one"),
        ("incident_ids", ("incident-a", "incident-a"), "unique"),
        ("reason_code", "", "reason_code"),
        ("resource_ids", ("gpu-b0", "gpu-b0"), "unique"),
        ("coordinator_id", "", "coordinator_id"),
        ("lease_id", "", "lease_id"),
        ("coordinator_lease_duration_ms", 0, "coordinator_lease_duration_ms"),
        ("coordinator_fencing_token", 0, "coordinator_fencing_token"),
    ],
)
def test_node_quarantine_record_validates_constructor_fields(field, value, message):
    with pytest.raises(ValueError, match=message):
        replace(_record(), **{field: value})


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda value: value.pop("node_id"),
            "missing fields",
        ),
        (
            lambda value: value.update({"unknown": True}),
            "unknown fields",
        ),
        (
            lambda value: value.update({"schema_version": 1}),
            "unsupported value",
        ),
        (
            lambda value: value.update({"schema_version": True}),
            "unsupported value",
        ),
        (
            lambda value: value.update({"effective_generation": 4}),
            "successor generation",
        ),
        (
            lambda value: value.update({"incident_ids": []}),
            "at least one",
        ),
        (
            lambda value: value.update({"resource_ids": "gpu-b0"}),
            "array",
        ),
    ],
)
def test_node_quarantine_record_rejects_invalid_wire_values(mutate, message):
    value = json.loads(_record().to_json())
    mutate(value)

    with pytest.raises(ValueError, match=message):
        NodeQuarantineRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"not-json",
        b"[]",
        b'{"schema_version":2,"schema_version":2}',
        (
            b'{"coordinator_fencing_token":7,"coordinator_id":"coordinator-a",'
            b'"coordinator_lease_duration_ms":30000,"effective_generation":1,'
            b'"from_generation":0,'
            b'"incident_ids":["incident-a"],"intent_id":"intent-a",'
            b'"lease_id":"lease-a","node_id":"node-b","node_id":"node-c",'
            b'"plan_id":"plan-a","reason_code":"sdc","resource_ids":[],'
            b'"run_id":"training-run","schema_version":2}'
        ),
    ],
)
def test_node_quarantine_record_rejects_malformed_or_duplicate_json(encoded):
    with pytest.raises(ValueError):
        NodeQuarantineRecord.from_json(encoded)


def test_node_quarantine_record_requires_encoded_bytes():
    with pytest.raises(ValueError, match="encoded bytes"):
        NodeQuarantineRecord.from_json(_record().to_json().decode("utf-8"))


def test_node_quarantine_record_allows_empty_resource_evidence():
    record = replace(_record(), resource_ids=())

    assert NodeQuarantineRecord.from_json(record.to_json()) == record
