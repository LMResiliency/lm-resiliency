"""Contract tests for resolved torchrun restart-plan state."""

from __future__ import annotations

from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointCopy,
    RankAssignment,
    RankCheckpointCopies,
    RecoveryManifest,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    ResolvedRecoveryManifest,
)

RUN_ID = "training-run"


def _assignment(
    *,
    run_id: str = RUN_ID,
    generation: int = 4,
    topology_digest: str = "topology-v1",
) -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=run_id,
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
        topology_digest=topology_digest,
    )


def _snapshot(
    *,
    assignment: RankAssignment | None = None,
) -> StoredGenerationSnapshot:
    record = GenerationSnapshotRecord(
        assignment=assignment or _assignment(),
        previous_snapshot_digest="a" * 64,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=4,
    )
    return StoredGenerationSnapshot(
        record=record,
        revision=8,
        committed_at_unix_ms=1_000,
        transaction_sequence=8,
        guard_mutation_sequence=4,
        guard_value_sequence=2,
        guard_lifetime_sequence=1,
        guard_committed_at_unix_ms=900,
    )


def _manifest(
    *,
    run_id: str = RUN_ID,
    source_generation: int = 4,
    topology_digest: str = "topology-v1",
) -> RecoveryManifest:
    copies = tuple(
        RankCheckpointCopies(
            owner_global_rank=rank,
            copies=(
                CheckpointCopy(
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
                ),
            ),
        )
        for rank in range(4)
    )
    return RecoveryManifest(
        manifest_id="manifest-40",
        run_id=run_id,
        source_generation=source_generation,
        step=40,
        trust="latest",
        topology_digest=topology_digest,
        rank_copies=copies,
    )


def _resolved() -> ResolvedRecoveryManifest:
    snapshot = _snapshot()
    return ResolvedRecoveryManifest(
        record=RecoveryManifestRecord(
            manifest=_manifest(),
            source_generation_snapshot_digest=snapshot.record.digest,
        ),
        source_snapshot=snapshot,
    )


def test_resolved_recovery_manifest_exposes_exact_manifest_and_assignment():
    resolved = _resolved()

    assert resolved.manifest == resolved.record.manifest
    assert resolved.source_assignment == resolved.source_snapshot.record.assignment


def test_resolved_recovery_manifest_is_immutable():
    resolved = _resolved()

    with pytest.raises(AttributeError):
        resolved.record = resolved.record


def test_resolved_recovery_manifest_requires_exact_types():
    resolved = _resolved()

    with pytest.raises(TypeError, match="record must be RecoveryManifestRecord"):
        ResolvedRecoveryManifest(
            record=resolved.record.to_dict(),
            source_snapshot=resolved.source_snapshot,
        )

    with pytest.raises(TypeError, match="source_snapshot must be StoredGenerationSnapshot"):
        ResolvedRecoveryManifest(
            record=resolved.record,
            source_snapshot=resolved.source_snapshot.record,
        )


def test_resolved_recovery_manifest_rejects_wrong_snapshot_digest():
    resolved = _resolved()

    with pytest.raises(ValueError, match="source snapshot digest"):
        replace(
            resolved,
            record=replace(
                resolved.record,
                source_generation_snapshot_digest="f" * 64,
            ),
        )


@pytest.mark.parametrize(
    ("manifest", "assignment"),
    [
        (_manifest(run_id="other-run"), _assignment()),
        (_manifest(source_generation=3), _assignment()),
        (_manifest(topology_digest="topology-v2"), _assignment()),
    ],
)
def test_resolved_recovery_manifest_rejects_source_identity_mismatch(
    manifest,
    assignment,
):
    snapshot = _snapshot(assignment=assignment)
    record = RecoveryManifestRecord(
        manifest=manifest,
        source_generation_snapshot_digest=snapshot.record.digest,
    )

    with pytest.raises(ValueError, match="does not match its source generation"):
        ResolvedRecoveryManifest(
            record=record,
            source_snapshot=snapshot,
        )


def test_resolved_recovery_manifest_does_not_claim_completeness():
    snapshot = _snapshot()
    incomplete_manifest = replace(
        _manifest(),
        rank_copies=_manifest().rank_copies[:-1],
    )
    record = RecoveryManifestRecord(
        manifest=incomplete_manifest,
        source_generation_snapshot_digest=snapshot.record.digest,
    )

    resolved = ResolvedRecoveryManifest(
        record=record,
        source_snapshot=snapshot,
    )

    assert len(resolved.manifest.rank_copies) == 3
