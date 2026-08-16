"""Contract tests for persisted torchrun restart-plan records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointCopy,
    RankCheckpointCopies,
    RecoveryManifest,
    RestartPlan,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanRecord,
)

RUN_ID = "training-run"
SOURCE_SNAPSHOT_DIGEST = "a" * 64
MANIFEST_RECORD_DIGEST = "b" * 64
LIFECYCLE_RECORD_DIGEST = "c" * 64
FROM_SNAPSHOT_DIGEST = "d" * 64
TO_SNAPSHOT_DIGEST = "e" * 64
QUARANTINE_RECORD_DIGEST = "f" * 64


def _copy(rank: int) -> CheckpointCopy:
    return CheckpointCopy(
        owner_global_rank=rank,
        checkpoint_step=40,
        inventory_event_id=f"inventory-{rank}",
        checkpoint_id=None,
        holder_node_id="node-a" if rank < 2 else "node-b",
        holder_kind="owner",
        storage_kind="node_local",
        location_token=f"copy-{rank}",
        complete=True,
        checksums_available=True,
    )


def _manifest() -> RecoveryManifest:
    return RecoveryManifest(
        manifest_id="manifest-40",
        run_id=RUN_ID,
        source_generation=4,
        step=40,
        trust="latest",
        topology_digest="topology-v1",
        rank_copies=tuple(
            RankCheckpointCopies(
                owner_global_rank=rank,
                copies=(_copy(rank),),
            )
            for rank in range(4)
        ),
    )


def _record() -> RecoveryManifestRecord:
    return RecoveryManifestRecord(
        manifest=_manifest(),
        source_generation_snapshot_digest=SOURCE_SNAPSHOT_DIGEST,
    )


def _plan() -> RestartPlan:
    return RestartPlan(
        plan_id="plan-5",
        intent_id="intent-4",
        run_id=RUN_ID,
        from_generation=4,
        to_generation=5,
        incident_ids=("incident-1",),
        reason_code="confirmed_straggler",
        recovery_mode="latest",
        checkpoint_source="gemini",
        checkpoint_step=40,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-40",
        slot_assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=2,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-c",
                first_global_rank=2,
                local_world_size=2,
            ),
        ),
        quarantined_node_ids=("node-b",),
        expected_world_size=4,
        topology_digest="topology-v1",
        restart_deadline_unix_ms=2_000,
    )


def _plan_record() -> RestartPlanRecord:
    return RestartPlanRecord(
        plan=_plan(),
        recovery_manifest_record_digest=MANIFEST_RECORD_DIGEST,
        intent_lifecycle_record_digest=LIFECYCLE_RECORD_DIGEST,
        from_generation_snapshot_digest=FROM_SNAPSHOT_DIGEST,
        to_generation_snapshot_digest=TO_SNAPSHOT_DIGEST,
        quarantine_record_digests={"node-b": QUARANTINE_RECORD_DIGEST},
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=9,
    )


def test_recovery_manifest_record_round_trips_as_canonical_json():
    record = _record()

    assert RecoveryManifestRecord.from_json(record.to_json()) == record
    assert json.loads(record.to_json()) == record.to_dict()
    assert record.digest == hashlib.sha256(record.to_json()).hexdigest()
    assert record.to_json() == record.to_json()


def test_recovery_manifest_record_is_immutable():
    record = _record()

    with pytest.raises(AttributeError):
        record.source_generation_snapshot_digest = "b" * 64


def test_recovery_manifest_record_requires_exact_types():
    with pytest.raises(TypeError, match="manifest must be RecoveryManifest"):
        RecoveryManifestRecord(
            manifest=_manifest().to_dict(),
            source_generation_snapshot_digest=SOURCE_SNAPSHOT_DIGEST,
        )

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(
            _record(),
            source_generation_snapshot_digest="A" * 64,
        )


def test_recovery_manifest_record_rejects_duplicate_fields_at_any_depth():
    outer_duplicate = (
        b'{"manifest":'
        + _record().manifest.to_json().encode()
        + b',"schema_version":1,"schema_version":1,'
        + b'"source_generation_snapshot_digest":"'
        + SOURCE_SNAPSHOT_DIGEST.encode()
        + b'"}'
    )
    with pytest.raises(ValueError, match="duplicate field 'schema_version'"):
        RecoveryManifestRecord.from_json(outer_duplicate)

    nested = (
        _record()
        .to_json()
        .decode()
        .replace(
            '"manifest_id":"manifest-40"',
            '"manifest_id":"manifest-40","manifest_id":"other"',
        )
    )
    with pytest.raises(ValueError, match="duplicate field 'manifest_id'"):
        RecoveryManifestRecord.from_json(nested.encode())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("manifest"),
        lambda value: value.pop("source_generation_snapshot_digest"),
        lambda value: value.update({"unknown": "field"}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"manifest": []}),
        lambda value: value.update({"source_generation_snapshot_digest": "not-a-digest"}),
    ],
)
def test_recovery_manifest_record_rejects_invalid_wire_shapes(mutation):
    value = _record().to_dict()
    mutation(value)

    with pytest.raises(ValueError):
        RecoveryManifestRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize("schema_version", [1.0, "1", None, False])
def test_recovery_manifest_record_requires_exact_nested_manifest_schema(
    schema_version,
):
    value = _record().to_dict()
    manifest = dict(value["manifest"])
    manifest["schema_version"] = schema_version
    value["manifest"] = manifest

    with pytest.raises(ValueError, match="manifest.schema_version"):
        RecoveryManifestRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


def test_recovery_manifest_record_rejects_invalid_nested_manifest():
    value = _record().to_dict()
    manifest = dict(value["manifest"])
    manifest["step"] = 0
    value["manifest"] = manifest

    with pytest.raises(ValueError, match="manifest is invalid"):
        RecoveryManifestRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize("encoded", [b"[]", b"null", b"not-json"])
def test_recovery_manifest_record_requires_json_object(encoded):
    with pytest.raises(ValueError):
        RecoveryManifestRecord.from_json(encoded)


def test_recovery_manifest_record_requires_encoded_bytes():
    with pytest.raises(ValueError, match="expected encoded bytes"):
        RecoveryManifestRecord.from_json(_record().to_json().decode())


def test_restart_plan_record_round_trips_as_canonical_json():
    record = _plan_record()

    assert RestartPlanRecord.from_json(record.to_json()) == record
    assert json.loads(record.to_json()) == record.to_dict()
    assert record.digest == hashlib.sha256(record.to_json()).hexdigest()
    assert len(record.coordinator_lease_digest) == 64
    assert record.to_json() == record.to_json()


def test_restart_plan_record_freezes_quarantine_digests():
    quarantine_digests = {"node-b": QUARANTINE_RECORD_DIGEST}
    record = replace(
        _plan_record(),
        quarantine_record_digests=quarantine_digests,
    )

    quarantine_digests["node-b"] = "0" * 64

    assert record.quarantine_record_digests == {
        "node-b": QUARANTINE_RECORD_DIGEST,
    }
    with pytest.raises(TypeError):
        record.quarantine_record_digests["node-b"] = "0" * 64


def test_restart_plan_record_requires_exact_quarantine_digest_coverage():
    with pytest.raises(ValueError, match="exactly match"):
        replace(_plan_record(), quarantine_record_digests={})

    with pytest.raises(ValueError, match="exactly match"):
        replace(
            _plan_record(),
            quarantine_record_digests={
                "node-b": QUARANTINE_RECORD_DIGEST,
                "node-c": "0" * 64,
            },
        )


def test_restart_plan_record_allows_no_quarantine_records_when_plan_has_none():
    plan = replace(_plan(), quarantined_node_ids=())

    record = replace(
        _plan_record(),
        plan=plan,
        quarantine_record_digests={},
    )

    assert record.quarantine_record_digests == {}


def test_restart_plan_record_requires_exact_types():
    with pytest.raises(TypeError, match="plan must be RestartPlan"):
        replace(_plan_record(), plan=_plan().to_dict())

    with pytest.raises(ValueError, match="positive integer"):
        replace(_plan_record(), coordinator_fencing_token=0)

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_plan_record(), recovery_manifest_record_digest="B" * 64)


def test_restart_plan_record_rejects_duplicate_fields_at_any_depth():
    outer_duplicate = (
        _plan_record()
        .to_json()
        .decode()
        .replace(
            '"schema_version":1',
            '"schema_version":1,"schema_version":1',
            1,
        )
    )
    with pytest.raises(ValueError, match="duplicate field 'schema_version'"):
        RestartPlanRecord.from_json(outer_duplicate.encode())

    nested_duplicate = (
        _plan_record()
        .to_json()
        .decode()
        .replace(
            '"plan_id":"plan-5"',
            '"plan_id":"plan-5","plan_id":"other"',
        )
    )
    with pytest.raises(ValueError, match="duplicate field 'plan_id'"):
        RestartPlanRecord.from_json(nested_duplicate.encode())

    mapping_duplicate = (
        _plan_record()
        .to_json()
        .decode()
        .replace(
            f'"node-b":"{QUARANTINE_RECORD_DIGEST}"',
            f'"node-b":"{QUARANTINE_RECORD_DIGEST}","node-b":"{"0" * 64}"',
        )
    )
    with pytest.raises(ValueError, match="duplicate field 'node-b'"):
        RestartPlanRecord.from_json(mapping_duplicate.encode())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("plan"),
        lambda value: value.pop("recovery_manifest_record_digest"),
        lambda value: value.pop("intent_lifecycle_record_digest"),
        lambda value: value.pop("from_generation_snapshot_digest"),
        lambda value: value.pop("to_generation_snapshot_digest"),
        lambda value: value.pop("quarantine_record_digests"),
        lambda value: value.pop("coordinator_id"),
        lambda value: value.pop("lease_id"),
        lambda value: value.pop("coordinator_lease_duration_ms"),
        lambda value: value.pop("coordinator_fencing_token"),
        lambda value: value.update({"unknown": "field"}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"plan": []}),
        lambda value: value.update({"quarantine_record_digests": []}),
    ],
)
def test_restart_plan_record_rejects_invalid_wire_shapes(mutation):
    value = _plan_record().to_dict()
    mutation(value)

    with pytest.raises(ValueError):
        RestartPlanRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize("schema_version", [1.0, "1", None, False])
def test_restart_plan_record_requires_exact_nested_plan_schema(schema_version):
    value = _plan_record().to_dict()
    plan = dict(value["plan"])
    plan["schema_version"] = schema_version
    value["plan"] = plan

    with pytest.raises(ValueError, match="plan.schema_version"):
        RestartPlanRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


def test_restart_plan_record_rejects_invalid_nested_plan():
    value = _plan_record().to_dict()
    plan = dict(value["plan"])
    plan["to_generation"] = 7
    value["plan"] = plan

    with pytest.raises(ValueError, match="plan is invalid"):
        RestartPlanRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


def test_restart_plan_record_requires_encoded_bytes():
    with pytest.raises(ValueError, match="expected encoded bytes"):
        RestartPlanRecord.from_json(_plan_record().to_json().decode())
