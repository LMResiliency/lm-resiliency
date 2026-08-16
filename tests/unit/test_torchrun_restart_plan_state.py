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
    CheckpointInventoryEvent,
    RankAssignment,
    RankCheckpointCopies,
    RecoveryManifest,
    RestartIntent,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
)
from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
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
    RestartPlanInventoryState,
    RestartPlanManifestState,
    RestartPlanQuarantineState,
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


def _quarantine_state(
    *,
    manifest_state: RestartPlanManifestState | None = None,
) -> RestartPlanQuarantineState:
    manifest_state = manifest_state or _manifest_state()
    plan = replace(
        manifest_state.plan,
        quarantined_node_ids=("node-b",),
    )
    plan_record = manifest_state.generation_state.record
    quarantine_record = NodeQuarantineRecord(
        run_id=plan.run_id,
        node_id="node-b",
        plan_id=plan.plan_id,
        intent_id=plan.intent_id,
        from_generation=plan.from_generation,
        effective_generation=plan.to_generation,
        incident_ids=plan.incident_ids,
        reason_code=plan.reason_code,
        resource_ids=("gpu-b0",),
        coordinator_id=plan_record.coordinator_id,
        lease_id=plan_record.lease_id,
        coordinator_lease_duration_ms=plan_record.coordinator_lease_duration_ms,
        coordinator_fencing_token=plan_record.coordinator_fencing_token,
    )
    generation_state = replace(
        manifest_state.generation_state,
        record=replace(
            plan_record,
            plan=plan,
            quarantine_record_digests={"node-b": quarantine_record.digest},
        ),
    )
    return RestartPlanQuarantineState(
        manifest_state=replace(
            manifest_state,
            generation_state=generation_state,
        ),
        quarantine_records={"node-b": quarantine_record},
    )


def test_restart_plan_quarantine_state_binds_exact_records():
    state = _quarantine_state()

    assert state.plan == state.manifest_state.plan
    assert state.quarantine_records["node-b"].node_id == "node-b"

    with pytest.raises(TypeError):
        state.quarantine_records["node-b"] = state.quarantine_records["node-b"]


def test_restart_plan_quarantine_state_allows_no_quarantine_records():
    state = RestartPlanQuarantineState(
        manifest_state=_manifest_state(),
        quarantine_records={},
    )

    assert not state.quarantine_records


def test_restart_plan_quarantine_state_requires_exact_types():
    state = _quarantine_state()

    with pytest.raises(TypeError, match="manifest_state must be"):
        replace(state, manifest_state=state.manifest_state.generation_state)

    with pytest.raises(TypeError, match="must be a mapping"):
        replace(state, quarantine_records=())

    with pytest.raises(TypeError, match="must be NodeQuarantineRecord"):
        replace(state, quarantine_records={"node-b": state.quarantine_records["node-b"].to_dict()})


@pytest.mark.parametrize(
    "quarantine_records",
    [
        {},
        {
            "node-b": _quarantine_state().quarantine_records["node-b"],
            "node-c": replace(
                _quarantine_state().quarantine_records["node-b"],
                node_id="node-c",
            ),
        },
    ],
)
def test_restart_plan_quarantine_state_requires_exact_node_coverage(quarantine_records):
    with pytest.raises(ValueError, match="exactly cover"):
        replace(
            _quarantine_state(),
            quarantine_records=quarantine_records,
        )


def test_restart_plan_quarantine_state_rejects_record_node_key_mismatch():
    state = _quarantine_state()

    with pytest.raises(ValueError, match="node does not match"):
        replace(
            state,
            quarantine_records={
                "node-b": replace(
                    state.quarantine_records["node-b"],
                    node_id="node-c",
                )
            },
        )


def test_restart_plan_quarantine_state_rejects_wrong_record_digest():
    state = _quarantine_state()

    with pytest.raises(ValueError, match="record digest"):
        replace(
            state,
            quarantine_records={
                "node-b": replace(
                    state.quarantine_records["node-b"],
                    resource_ids=("gpu-b1",),
                )
            },
        )


@pytest.mark.parametrize(
    "record",
    [
        replace(_quarantine_state().quarantine_records["node-b"], run_id="other-run"),
        replace(_quarantine_state().quarantine_records["node-b"], plan_id="other-plan"),
        replace(_quarantine_state().quarantine_records["node-b"], intent_id="other-intent"),
        replace(
            _quarantine_state().quarantine_records["node-b"],
            from_generation=3,
            effective_generation=4,
        ),
        replace(
            _quarantine_state().quarantine_records["node-b"],
            incident_ids=("other-incident",),
        ),
        replace(_quarantine_state().quarantine_records["node-b"], reason_code="other-reason"),
    ],
)
def test_restart_plan_quarantine_state_rejects_plan_metadata_mismatch(record):
    state = _quarantine_state()
    plan_record = replace(
        state.manifest_state.generation_state.record,
        quarantine_record_digests={"node-b": record.digest},
    )

    with pytest.raises(ValueError, match="does not match its plan"):
        replace(
            state,
            manifest_state=replace(
                state.manifest_state,
                generation_state=replace(
                    state.manifest_state.generation_state,
                    record=plan_record,
                ),
            ),
            quarantine_records={"node-b": record},
        )


@pytest.mark.parametrize(
    "record",
    [
        replace(
            _quarantine_state().quarantine_records["node-b"],
            coordinator_id="other-coordinator",
        ),
        replace(_quarantine_state().quarantine_records["node-b"], lease_id="other-lease"),
        replace(
            _quarantine_state().quarantine_records["node-b"],
            coordinator_lease_duration_ms=1_000,
        ),
        replace(
            _quarantine_state().quarantine_records["node-b"],
            coordinator_fencing_token=10,
        ),
    ],
)
def test_restart_plan_quarantine_state_rejects_publication_authority_mismatch(record):
    state = _quarantine_state()
    plan_record = replace(
        state.manifest_state.generation_state.record,
        quarantine_record_digests={"node-b": record.digest},
    )

    with pytest.raises(ValueError, match="publication authority"):
        replace(
            state,
            manifest_state=replace(
                state.manifest_state,
                generation_state=replace(
                    state.manifest_state.generation_state,
                    record=plan_record,
                ),
            ),
            quarantine_records={"node-b": record},
        )


def _inventory_event(
    state: RestartPlanQuarantineState,
    rank: int,
    *,
    trust: str | None = None,
) -> CheckpointInventoryEvent:
    manifest = state.manifest_state.manifest
    rank_copies = next(entry for entry in manifest.rank_copies if entry.owner_global_rank == rank)
    copy = rank_copies.copies[0]
    logical_node_slot = rank // 2
    node_id = "node-a" if logical_node_slot == 0 else "node-b"
    return CheckpointInventoryEvent(
        event_id=copy.inventory_event_id,
        run_id=manifest.run_id,
        generation=manifest.source_generation,
        reporter=WorkerIdentity(
            run_id=manifest.run_id,
            generation=manifest.source_generation,
            node_id=node_id,
            agent_id=f"agent-{node_id}",
            logical_node_slot=logical_node_slot,
            global_rank=rank,
            local_rank=rank % 2,
            local_world_size=2,
            hostname=f"host-{node_id}",
            gpu_uuid=f"gpu-{node_id}-{rank % 2}",
            topology_digest=manifest.topology_digest,
        ),
        step=manifest.step,
        trust=trust or manifest.trust,
        topology_digest=manifest.topology_digest,
        copies=(copy,),
    )


def _inventory_state(
    *,
    quarantine_state: RestartPlanQuarantineState | None = None,
    inventory_events: dict[str, CheckpointInventoryEvent] | None = None,
) -> RestartPlanInventoryState:
    selected_state = quarantine_state or _quarantine_state()
    events = inventory_events or {
        event.event_id: event
        for event in (
            _inventory_event(selected_state, rank)
            for rank in range(selected_state.plan.expected_world_size)
        )
    }
    return RestartPlanInventoryState(
        quarantine_state=selected_state,
        inventory_events=events,
    )


def test_restart_plan_inventory_state_binds_exact_events():
    state = _inventory_state()

    assert state.plan == state.quarantine_state.plan
    assert state.manifest == state.quarantine_state.manifest_state.manifest
    assert tuple(state.inventory_events) == (
        "inventory-0",
        "inventory-1",
        "inventory-2",
        "inventory-3",
    )

    with pytest.raises(TypeError):
        state.inventory_events["other"] = state.inventory_events["inventory-0"]


def test_restart_plan_inventory_state_requires_exact_types():
    state = _inventory_state()

    with pytest.raises(TypeError, match="quarantine_state must be"):
        replace(state, quarantine_state=state.quarantine_state.manifest_state)

    with pytest.raises(TypeError, match="must be a mapping"):
        replace(state, inventory_events=())

    with pytest.raises(TypeError, match="must be CheckpointInventoryEvent"):
        replace(
            state,
            inventory_events={
                event_id: event.to_dict() for event_id, event in state.inventory_events.items()
            },
        )


def test_restart_plan_inventory_state_requires_exact_event_coverage():
    state = _inventory_state()
    missing = dict(state.inventory_events)
    missing.pop("inventory-3")

    with pytest.raises(ValueError, match="exactly cover"):
        replace(state, inventory_events=missing)

    with pytest.raises(ValueError, match="exactly cover"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-extra": replace(
                    state.inventory_events["inventory-0"],
                    event_id="inventory-extra",
                    copies=(),
                ),
            },
        )


def test_restart_plan_inventory_state_rejects_event_key_mismatch():
    state = _inventory_state()

    with pytest.raises(ValueError, match="event ID does not match"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": replace(
                    state.inventory_events["inventory-0"],
                    event_id="inventory-other",
                    copies=(),
                ),
            },
        )


@pytest.mark.parametrize(
    "event",
    [
        replace(
            _inventory_state().inventory_events["inventory-0"],
            run_id="other-run",
            reporter=replace(
                _inventory_state().inventory_events["inventory-0"].reporter,
                run_id="other-run",
            ),
        ),
        replace(
            _inventory_state().inventory_events["inventory-0"],
            generation=3,
            reporter=replace(
                _inventory_state().inventory_events["inventory-0"].reporter,
                generation=3,
            ),
        ),
        replace(
            _inventory_state().inventory_events["inventory-0"],
            topology_digest="topology-v2",
            reporter=replace(
                _inventory_state().inventory_events["inventory-0"].reporter,
                topology_digest="topology-v2",
            ),
        ),
        replace(
            _inventory_state().inventory_events["inventory-0"],
            step=39,
            copies=(
                replace(
                    _inventory_state().inventory_events["inventory-0"].copies[0],
                    checkpoint_step=39,
                ),
            ),
        ),
    ],
)
def test_restart_plan_inventory_state_rejects_manifest_metadata_mismatch(event):
    state = _inventory_state()

    with pytest.raises(ValueError, match="does not match its manifest"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": event,
            },
        )


def test_restart_plan_inventory_state_rejects_reporter_outside_source_assignment():
    state = _inventory_state()
    event = state.inventory_events["inventory-0"]
    conflicting_reporter = replace(
        event.reporter,
        node_id="node-b",
        agent_id="agent-node-b",
        hostname="host-node-b",
        gpu_uuid="gpu-node-b-0",
    )

    with pytest.raises(ValueError, match="reporter does not match"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": replace(event, reporter=conflicting_reporter),
            },
        )


def test_restart_plan_inventory_state_rejects_copy_missing_from_event():
    state = _inventory_state()

    with pytest.raises(ValueError, match="copy does not match"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": replace(
                    state.inventory_events["inventory-0"],
                    copies=(),
                ),
            },
        )


def test_restart_plan_inventory_state_rejects_local_copy_reported_by_another_node():
    state = _inventory_state()
    event = state.inventory_events["inventory-0"]
    reporter = state.inventory_events["inventory-2"].reporter

    with pytest.raises(ValueError, match="not reported by its holder"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": replace(
                    event,
                    reporter=reporter,
                ),
            },
        )


def test_restart_plan_inventory_state_rejects_candidate_inventory():
    state = _inventory_state()

    with pytest.raises(ValueError, match="trust is incompatible"):
        replace(
            state,
            inventory_events={
                **state.inventory_events,
                "inventory-0": replace(
                    state.inventory_events["inventory-0"],
                    trust="candidate",
                ),
            },
        )


def test_restart_plan_inventory_state_requires_verified_events_for_verified_manifest():
    verified_manifest = replace(_manifest(), trust="recovery_verified")
    verified_plan = replace(
        _plan(),
        recovery_mode="recovery_verified",
        checkpoint_source="durable",
        checkpoint_id="durable-40",
    )
    quarantine_state = _quarantine_state(
        manifest_state=_manifest_state(
            manifest=verified_manifest,
            plan=verified_plan,
        )
    )
    events = {
        event.event_id: event
        for event in (
            _inventory_event(quarantine_state, rank, trust="latest")
            for rank in range(quarantine_state.plan.expected_world_size)
        )
    }

    with pytest.raises(ValueError, match="trust is incompatible"):
        _inventory_state(
            quarantine_state=quarantine_state,
            inventory_events=events,
        )


def test_restart_plan_inventory_state_does_not_claim_copy_eligibility():
    manifest = _manifest()
    incomplete_manifest = replace(
        manifest,
        rank_copies=(
            replace(
                manifest.rank_copies[0],
                copies=(
                    replace(
                        manifest.rank_copies[0].copies[0],
                        complete=False,
                    ),
                ),
            ),
            *manifest.rank_copies[1:],
        ),
    )
    quarantine_state = _quarantine_state(
        manifest_state=_manifest_state(manifest=incomplete_manifest)
    )

    state = _inventory_state(quarantine_state=quarantine_state)

    assert not state.manifest.rank_copies[0].copies[0].complete
