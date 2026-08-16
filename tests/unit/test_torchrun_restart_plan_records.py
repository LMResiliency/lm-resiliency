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
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
)

RUN_ID = "training-run"
SOURCE_SNAPSHOT_DIGEST = "a" * 64


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
