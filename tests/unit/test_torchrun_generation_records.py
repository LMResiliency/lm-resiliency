"""Contract tests for persisted torchrun generation records."""

from __future__ import annotations

import hashlib
import json

import pytest

from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import RankAssignment, SlotAssignment

RUN_ID = "training-run"


def _assignment(generation: int) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=2,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-b",
                first_global_rank=2,
                local_world_size=2,
            ),
        ),
        topology_digest="topology-v1",
    )


def _snapshot(generation: int = 1) -> GenerationSnapshotRecord:
    return GenerationSnapshotRecord(
        assignment=_assignment(generation),
        previous_snapshot_digest=None if generation == 0 else "a" * 64,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_fencing_token=4,
    )


def test_generation_records_round_trip_as_canonical_json():
    snapshot = _snapshot()
    head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=1,
        snapshot_digest=snapshot.digest,
    )

    assert GenerationSnapshotRecord.from_json(snapshot.to_json()) == snapshot
    assert GenerationHeadRecord.from_json(head.to_json()) == head
    assert json.loads(snapshot.to_json()) == snapshot.to_dict()
    assert json.loads(head.to_json()) == head.to_dict()
    assert snapshot.digest == hashlib.sha256(snapshot.to_json()).hexdigest()
    assert snapshot.to_json() == snapshot.to_json()


def test_generation_records_reject_duplicate_fields_at_any_depth():
    duplicate_head_generation = (
        b'{"schema_version":1,"run_id":"training-run",'
        b'"generation":0,"generation":1,"snapshot_digest":"' + b"a" * 64 + b'"}'
    )
    with pytest.raises(ValueError, match="duplicate field 'generation'"):
        GenerationHeadRecord.from_json(duplicate_head_generation)

    value = _snapshot().to_dict()
    assignment = json.dumps(value["assignment"], separators=(",", ":"), sort_keys=True)
    assignment = assignment.replace(
        '"generation":1',
        '"generation":1,"generation":2',
    )
    encoded = (
        '{"assignment":'
        + assignment
        + ',"coordinator_fencing_token":4,"coordinator_id":"coordinator-a",'
        '"lease_id":"lease-a","previous_snapshot_digest":"' + "a" * 64 + '","schema_version":1}'
    ).encode()
    with pytest.raises(ValueError, match="duplicate field 'generation'"):
        GenerationSnapshotRecord.from_json(encoded)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.pop("lease_id"),
        lambda value: value.update({"unknown": "field"}),
        lambda value: value.update({"schema_version": 2}),
        lambda value: value.update({"schema_version": True}),
        lambda value: value.update({"assignment": []}),
    ],
)
def test_generation_snapshot_rejects_invalid_wire_shapes(mutation):
    value = _snapshot().to_dict()
    mutation(value)

    with pytest.raises(ValueError):
        GenerationSnapshotRecord.from_json(
            json.dumps(value, separators=(",", ":"), sort_keys=True).encode()
        )


@pytest.mark.parametrize(
    "encoded",
    [
        b"[]",
        b"null",
        b"not-json",
    ],
)
def test_generation_records_require_json_objects(encoded):
    with pytest.raises(ValueError):
        GenerationHeadRecord.from_json(encoded)


def test_generation_snapshot_requires_exact_predecessor_shape():
    with pytest.raises(ValueError, match="must not name"):
        GenerationSnapshotRecord(
            assignment=_assignment(0),
            previous_snapshot_digest="a" * 64,
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            coordinator_fencing_token=1,
        )

    with pytest.raises(ValueError, match="SHA-256"):
        GenerationSnapshotRecord(
            assignment=_assignment(1),
            previous_snapshot_digest=None,
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            coordinator_fencing_token=1,
        )

    with pytest.raises(ValueError, match="SHA-256"):
        GenerationSnapshotRecord(
            assignment=_assignment(1),
            previous_snapshot_digest="A" * 64,
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            coordinator_fencing_token=1,
        )


@pytest.mark.parametrize("fencing_token", [0, -1, True])
def test_generation_snapshot_requires_positive_integer_fencing_token(fencing_token):
    with pytest.raises(ValueError, match="positive integer"):
        GenerationSnapshotRecord(
            assignment=_assignment(1),
            previous_snapshot_digest="a" * 64,
            coordinator_id="coordinator-a",
            lease_id="lease-a",
            coordinator_fencing_token=fencing_token,
        )
