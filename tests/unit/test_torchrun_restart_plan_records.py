"""Contract tests for persisted torchrun restart-plan records."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointCertification,
    CheckpointCopy,
    CheckpointInventoryEvent,
    RankCheckpointCopies,
    RecoveryManifest,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
    checkpoint_inventory_digest,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanEvidenceRecord,
    RestartPlanRecord,
)

RUN_ID = "training-run"
SOURCE_SNAPSHOT_DIGEST = "a" * 64
MANIFEST_RECORD_DIGEST = "b" * 64
EVIDENCE_RECORD_DIGEST = "1" * 64
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


def _inventory_events() -> dict[str, CheckpointInventoryEvent]:
    events: dict[str, CheckpointInventoryEvent] = {}
    for rank in range(4):
        event_id = f"inventory-{rank}"
        node_id = "node-a" if rank < 2 else "node-b"
        local_rank = rank % 2
        events[event_id] = CheckpointInventoryEvent(
            event_id=event_id,
            run_id=RUN_ID,
            generation=4,
            reporter=WorkerIdentity(
                run_id=RUN_ID,
                generation=4,
                node_id=node_id,
                agent_id=f"agent-{node_id}",
                logical_node_slot=rank // 2,
                global_rank=rank,
                local_rank=local_rank,
                local_world_size=2,
                hostname=f"host-{node_id}",
                gpu_uuid=f"gpu-{node_id}-{local_rank}",
                topology_digest="topology-v1",
            ),
            step=40,
            trust="recovery_verified",
            topology_digest="topology-v1",
            copies=(_copy(rank),),
        )
    return events


def _certification() -> CheckpointCertification:
    events = _inventory_events()
    return CheckpointCertification(
        certification_id="certification-40",
        run_id=RUN_ID,
        source_generation=4,
        step=40,
        topology_digest="topology-v1",
        checkpoint_source="gemini",
        checkpoint_id=None,
        expected_world_size=4,
        certification_kind="dense_consensus",
        inventory_event_digests={
            event_id: checkpoint_inventory_digest(event) for event_id, event in events.items()
        },
    )


def _evidence_record() -> RestartPlanEvidenceRecord:
    return RestartPlanEvidenceRecord(
        plan_id="plan-5",
        run_id=RUN_ID,
        manifest_id="manifest-40",
        inventory_events=_inventory_events(),
        certifications=(_certification(),),
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
        recovery_evidence_record_digest=EVIDENCE_RECORD_DIGEST,
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


def test_restart_plan_evidence_record_round_trips_as_canonical_json():
    record = _evidence_record()

    assert RestartPlanEvidenceRecord.from_json(record.to_json()) == record
    assert json.loads(record.to_json()) == record.to_dict()
    assert record.digest == hashlib.sha256(record.to_json()).hexdigest()
    assert tuple(record.inventory_events) == tuple(sorted(record.inventory_events))


def test_restart_plan_evidence_record_freezes_and_sorts_evidence():
    events = dict(reversed(tuple(_inventory_events().items())))
    certification = _certification()
    record = RestartPlanEvidenceRecord(
        plan_id="plan-5",
        run_id=RUN_ID,
        manifest_id="manifest-40",
        inventory_events=events,
        certifications=(certification,),
    )
    events.clear()

    assert len(record.inventory_events) == 4
    with pytest.raises(TypeError):
        record.inventory_events["other"] = next(iter(record.inventory_events.values()))
    with pytest.raises(AttributeError):
        record.certifications = ()


def test_restart_plan_evidence_record_requires_bound_event_and_certification_identity():
    events = _inventory_events()
    event = events.pop("inventory-0")
    events["other"] = event
    with pytest.raises(ValueError, match="identity"):
        replace(_evidence_record(), inventory_events=events)

    foreign = replace(_certification(), run_id="other-run")
    with pytest.raises(ValueError, match="another run"):
        replace(_evidence_record(), certifications=(foreign,))

    with pytest.raises(ValueError, match="unique"):
        replace(
            _evidence_record(),
            certifications=(_certification(), _certification()),
        )


def test_restart_plan_evidence_record_rejects_duplicate_fields_at_any_depth():
    duplicate = (
        _evidence_record()
        .to_json()
        .decode()
        .replace(
            '"plan_id":"plan-5"',
            '"plan_id":"plan-5","plan_id":"other"',
            1,
        )
    )

    with pytest.raises(ValueError, match="duplicate field 'plan_id'"):
        RestartPlanEvidenceRecord.from_json(duplicate.encode())


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("plan_id"),
        lambda value: value.pop("inventory_events"),
        lambda value: value.pop("certifications"),
        lambda value: value.update({"unknown": "field"}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"inventory_events": []}),
        lambda value: value.update({"certifications": {}}),
    ],
)
def test_restart_plan_evidence_record_rejects_invalid_wire_shapes(mutation):
    value = _evidence_record().to_dict()
    mutation(value)

    with pytest.raises(ValueError):
        RestartPlanEvidenceRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize("path", ["inventory", "certification"])
@pytest.mark.parametrize("schema_version", [1.0, "1", None, False])
def test_restart_plan_evidence_record_requires_exact_nested_schema(
    path: str,
    schema_version,
):
    value = _evidence_record().to_dict()
    if path == "inventory":
        events = dict(value["inventory_events"])
        event = dict(events["inventory-0"])
        event["schema_version"] = schema_version
        events["inventory-0"] = event
        value["inventory_events"] = events
    else:
        certifications = list(value["certifications"])
        certification = dict(certifications[0])
        certification["schema_version"] = schema_version
        certifications[0] = certification
        value["certifications"] = certifications

    with pytest.raises(ValueError, match="schema_version"):
        RestartPlanEvidenceRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


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

    with pytest.raises(ValueError, match="lowercase SHA-256"):
        replace(_plan_record(), recovery_evidence_record_digest="not-a-digest")


def test_restart_plan_record_rejects_duplicate_fields_at_any_depth():
    outer_duplicate = (
        _plan_record()
        .to_json()
        .decode()
        .replace(
            '"schema_version":2',
            '"schema_version":2,"schema_version":2',
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
        lambda value: value.pop("recovery_evidence_record_digest"),
        lambda value: value.pop("intent_lifecycle_record_digest"),
        lambda value: value.pop("from_generation_snapshot_digest"),
        lambda value: value.pop("to_generation_snapshot_digest"),
        lambda value: value.pop("quarantine_record_digests"),
        lambda value: value.pop("coordinator_id"),
        lambda value: value.pop("lease_id"),
        lambda value: value.pop("coordinator_lease_duration_ms"),
        lambda value: value.pop("coordinator_fencing_token"),
        lambda value: value.update({"unknown": "field"}),
        lambda value: value.update({"schema_version": 3}),
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
