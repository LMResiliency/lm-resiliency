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
    RestartIntent,
    RestartPlan,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    ResolvedRecoveryManifest,
    RestartPlanGenerationState,
    RestartPlanManifestState,
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


def _intent_record() -> RestartIntentRecord:
    return RestartIntentRecord(
        intent=RestartIntent(
            intent_id="intent-4",
            run_id=RUN_ID,
            generation=4,
            incident_ids=("incident-1",),
            reason_code="confirmed_straggler",
            minimum_recovery_mode="latest",
            suspected_node_ids=("node-b",),
            prepare_deadline_unix_ms=1_500,
        ),
        generation_snapshot_digest=_generation_record(4).digest,
        coordinator_id="coordinator-a",
        lease_id="lease-open",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=4,
    )


def _lifecycle_record() -> RestartIntentLifecycleRecord:
    intent_record = _intent_record()
    return RestartIntentLifecycleRecord(
        closed_intent=RestartIntentHeadRecord(
            run_id=RUN_ID,
            generation=4,
            intent_id="intent-4",
            intent_digest=intent_record.digest,
        ),
        coordinator_id="coordinator-a",
        lease_id="lease-close",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=6,
    )


def _generation_record(
    generation: int,
    *,
    assignment: RankAssignment | None = None,
    previous_snapshot_digest: str | None = None,
) -> GenerationSnapshotRecord:
    if assignment is None:
        assignment = _assignment(generation=generation)
    return GenerationSnapshotRecord(
        assignment=assignment,
        previous_snapshot_digest=(
            previous_snapshot_digest if previous_snapshot_digest is not None else "a" * 64
        ),
        coordinator_id="coordinator-plan",
        lease_id="lease-plan",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=9,
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
        quarantined_node_ids=(),
        expected_world_size=4,
        topology_digest="topology-v1",
        restart_deadline_unix_ms=2_000,
    )


def _generation_state() -> RestartPlanGenerationState:
    from_snapshot = _generation_record(4)
    to_assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=5,
        assignments=_plan().slot_assignments,
        topology_digest="topology-v1",
    )
    to_snapshot = _generation_record(
        5,
        assignment=to_assignment,
        previous_snapshot_digest=from_snapshot.digest,
    )
    lifecycle = _lifecycle_record()
    record = RestartPlanRecord(
        plan=_plan(),
        recovery_manifest_record_digest="b" * 64,
        intent_lifecycle_record_digest=lifecycle.digest,
        from_generation_snapshot_digest=from_snapshot.digest,
        to_generation_snapshot_digest=to_snapshot.digest,
        quarantine_record_digests={},
        coordinator_id=to_snapshot.coordinator_id,
        lease_id=to_snapshot.lease_id,
        coordinator_lease_duration_ms=to_snapshot.coordinator_lease_duration_ms,
        coordinator_fencing_token=to_snapshot.coordinator_fencing_token,
    )
    return RestartPlanGenerationState(
        record=record,
        intent_record=_intent_record(),
        lifecycle_record=lifecycle,
        from_snapshot=from_snapshot,
        to_snapshot=to_snapshot,
    )


def test_restart_plan_generation_state_binds_exact_records():
    state = _generation_state()

    assert state.plan == state.record.plan
    assert state.from_assignment == state.from_snapshot.assignment
    assert state.to_assignment == state.to_snapshot.assignment


def test_restart_plan_generation_state_is_immutable():
    state = _generation_state()

    with pytest.raises(AttributeError):
        state.record = state.record


def test_restart_plan_generation_state_requires_exact_types():
    state = _generation_state()

    with pytest.raises(TypeError, match="record must be RestartPlanRecord"):
        replace(state, record=state.record.to_dict())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("intent_lifecycle_record_digest", "0" * 64),
        ("from_generation_snapshot_digest", "1" * 64),
        ("to_generation_snapshot_digest", "2" * 64),
    ],
)
def test_restart_plan_generation_state_rejects_envelope_digest_mismatch(
    field,
    value,
):
    state = _generation_state()

    with pytest.raises(ValueError, match="envelope digests"):
        replace(state, record=replace(state.record, **{field: value}))


def test_restart_plan_generation_state_rejects_wrong_closed_intent():
    state = _generation_state()
    wrong_lifecycle = replace(
        state.lifecycle_record,
        closed_intent=replace(
            state.lifecycle_record.closed_intent,
            intent_digest="0" * 64,
        ),
    )

    with pytest.raises(ValueError, match="does not close"):
        replace(
            state,
            record=replace(
                state.record,
                intent_lifecycle_record_digest=wrong_lifecycle.digest,
            ),
            lifecycle_record=wrong_lifecycle,
        )


def test_restart_plan_generation_state_rejects_plan_intent_mismatch():
    state = _generation_state()

    with pytest.raises(ValueError, match="does not match its restart intent"):
        replace(
            state,
            record=replace(
                state.record,
                plan=replace(state.plan, reason_code="other"),
            ),
        )


def test_restart_plan_generation_state_rejects_weaker_recovery_mode():
    state = _generation_state()
    verified_intent = replace(
        state.intent_record.intent,
        minimum_recovery_mode="recovery_verified",
    )
    intent_record = replace(state.intent_record, intent=verified_intent)
    lifecycle = replace(
        state.lifecycle_record,
        closed_intent=replace(
            state.lifecycle_record.closed_intent,
            intent_digest=intent_record.digest,
        ),
    )

    with pytest.raises(ValueError, match="recovery mode is weaker"):
        replace(
            state,
            intent_record=intent_record,
            lifecycle_record=lifecycle,
            record=replace(
                state.record,
                intent_lifecycle_record_digest=lifecycle.digest,
            ),
        )


def test_restart_plan_generation_state_rejects_wrong_successor_link():
    state = _generation_state()

    with pytest.raises(ValueError, match="does not reference"):
        replace(
            state,
            to_snapshot=replace(
                state.to_snapshot,
                previous_snapshot_digest="0" * 64,
            ),
            record=replace(
                state.record,
                to_generation_snapshot_digest=replace(
                    state.to_snapshot,
                    previous_snapshot_digest="0" * 64,
                ).digest,
            ),
        )


def test_restart_plan_generation_state_rejects_wrong_successor_assignment():
    state = _generation_state()
    wrong_assignment = _assignment(generation=5)
    wrong_snapshot = replace(state.to_snapshot, assignment=wrong_assignment)

    with pytest.raises(ValueError, match="successor assignment"):
        replace(
            state,
            to_snapshot=wrong_snapshot,
            record=replace(
                state.record,
                to_generation_snapshot_digest=wrong_snapshot.digest,
            ),
        )


def test_restart_plan_generation_state_rejects_different_publication_authority():
    state = _generation_state()

    with pytest.raises(ValueError, match="different publication authority"):
        replace(
            state,
            record=replace(state.record, lease_id="other-lease"),
        )


def _manifest_state(
    *,
    manifest: RecoveryManifest | None = None,
    plan: RestartPlan | None = None,
) -> RestartPlanManifestState:
    selected_manifest = manifest or _manifest()
    generation_state = _generation_state()
    if plan is not None:
        generation_state = replace(
            generation_state,
            record=replace(generation_state.record, plan=plan),
        )
    source_snapshot = _snapshot(
        assignment=_assignment(
            run_id=selected_manifest.run_id,
            generation=selected_manifest.source_generation,
            topology_digest=selected_manifest.topology_digest,
        )
    )
    manifest_record = RecoveryManifestRecord(
        manifest=selected_manifest,
        source_generation_snapshot_digest=source_snapshot.record.digest,
    )
    generation_state = replace(
        generation_state,
        record=replace(
            generation_state.record,
            recovery_manifest_record_digest=manifest_record.digest,
        ),
    )
    return RestartPlanManifestState(
        generation_state=generation_state,
        resolved_manifest=ResolvedRecoveryManifest(
            record=manifest_record,
            source_snapshot=source_snapshot,
        ),
    )


def test_restart_plan_manifest_state_binds_exact_plan_and_manifest():
    state = _manifest_state()

    assert state.plan == state.generation_state.plan
    assert state.manifest == state.resolved_manifest.manifest


def test_restart_plan_manifest_state_requires_exact_types():
    state = _manifest_state()

    with pytest.raises(TypeError, match="generation_state must be"):
        replace(state, generation_state=state.generation_state.record)

    with pytest.raises(TypeError, match="resolved_manifest must be"):
        replace(state, resolved_manifest=state.resolved_manifest.record)


def test_restart_plan_manifest_state_rejects_wrong_manifest_digest():
    state = _manifest_state()

    with pytest.raises(ValueError, match="manifest record digest"):
        replace(
            state,
            generation_state=replace(
                state.generation_state,
                record=replace(
                    state.generation_state.record,
                    recovery_manifest_record_digest="0" * 64,
                ),
            ),
        )


@pytest.mark.parametrize(
    "manifest",
    [
        _manifest(run_id="other-run"),
        _manifest(source_generation=5),
        replace(_manifest(), manifest_id="other-manifest"),
        _manifest(topology_digest="topology-v2"),
    ],
)
def test_restart_plan_manifest_state_rejects_metadata_mismatch(manifest):
    with pytest.raises(ValueError, match="manifest metadata"):
        _manifest_state(manifest=manifest)


def test_restart_plan_manifest_state_rejects_source_world_size_mismatch():
    state = _manifest_state()
    smaller_assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=4,
        assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=2,
            ),
        ),
        topology_digest="topology-v1",
    )
    source_snapshot = _snapshot(assignment=smaller_assignment)
    manifest_record = replace(
        state.resolved_manifest.record,
        source_generation_snapshot_digest=source_snapshot.record.digest,
    )

    with pytest.raises(ValueError, match="source world size"):
        replace(
            state,
            generation_state=replace(
                state.generation_state,
                record=replace(
                    state.generation_state.record,
                    recovery_manifest_record_digest=manifest_record.digest,
                ),
            ),
            resolved_manifest=ResolvedRecoveryManifest(
                record=manifest_record,
                source_snapshot=source_snapshot,
            ),
        )


def test_restart_plan_manifest_state_rejects_conflicting_current_source_assignment():
    state = _manifest_state()
    conflicting_assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=4,
        assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-b",
                first_global_rank=0,
                local_world_size=2,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-a",
                first_global_rank=2,
                local_world_size=2,
            ),
        ),
        topology_digest="topology-v1",
    )
    source_snapshot = _snapshot(assignment=conflicting_assignment)
    manifest_record = replace(
        state.resolved_manifest.record,
        source_generation_snapshot_digest=source_snapshot.record.digest,
    )

    with pytest.raises(ValueError, match="source assignment conflicts"):
        replace(
            state,
            generation_state=replace(
                state.generation_state,
                record=replace(
                    state.generation_state.record,
                    recovery_manifest_record_digest=manifest_record.digest,
                ),
            ),
            resolved_manifest=ResolvedRecoveryManifest(
                record=manifest_record,
                source_snapshot=source_snapshot,
            ),
        )


def test_restart_plan_manifest_state_requires_verified_manifest_for_verified_mode():
    plan = replace(_plan(), recovery_mode="recovery_verified")

    with pytest.raises(ValueError, match="verified recovery"):
        _manifest_state(plan=plan)


def test_restart_plan_manifest_state_requires_verified_manifest_for_durable_source():
    plan = replace(
        _plan(),
        checkpoint_source="durable",
        checkpoint_id="durable-40",
    )

    with pytest.raises(ValueError, match="durable recovery"):
        _manifest_state(plan=plan)


def test_restart_plan_manifest_state_accepts_verified_manifest():
    plan = replace(
        _plan(),
        recovery_mode="recovery_verified",
        checkpoint_source="durable",
        checkpoint_id="durable-40",
    )

    state = _manifest_state(
        manifest=replace(_manifest(), trust="recovery_verified"),
        plan=plan,
    )

    assert state.manifest.trust == "recovery_verified"


def test_restart_plan_manifest_state_does_not_claim_copy_completeness():
    incomplete = replace(_manifest(), rank_copies=_manifest().rank_copies[:-1])

    state = _manifest_state(manifest=incomplete)

    assert len(state.manifest.rank_copies) == 3
