"""Contract tests for internal torchrun standby-replacement records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    CheckpointCopy,
    CheckpointInventoryEvent,
    FaultEvent,
    HardwareFaultReport,
    ProtocolValidationError,
    RankAssignment,
    RankCheckpointCopies,
    RecoveryManifest,
    RecoveryProposalEvent,
    RestartAck,
    RestartContext,
    RestartIntent,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
    validate_event_reporter,
    validate_restart_plan,
    validate_worker_identity,
)

RUN_ID = "training-run"
TOPOLOGY_DIGEST = "topology-sha256"


def _assignments() -> tuple[SlotAssignment, ...]:
    return (
        SlotAssignment(
            logical_node_slot=0,
            node_id="node-a",
            first_global_rank=0,
            local_world_size=2,
        ),
        SlotAssignment(
            logical_node_slot=1,
            node_id="node-spare",
            first_global_rank=2,
            local_world_size=2,
        ),
    )


def _worker() -> WorkerIdentity:
    return WorkerIdentity(
        run_id=RUN_ID,
        generation=4,
        node_id="node-a",
        agent_id="agent-a",
        logical_node_slot=0,
        global_rank=0,
        local_rank=0,
        local_world_size=2,
        hostname="host-a",
        gpu_uuid="GPU-0",
        topology_digest=TOPOLOGY_DIGEST,
    )


def _intent(*, minimum_recovery_mode: str = "latest") -> RestartIntent:
    return RestartIntent(
        intent_id="intent-4",
        run_id=RUN_ID,
        generation=4,
        incident_ids=("incident-a",),
        reason_code="replace_straggler",
        minimum_recovery_mode=minimum_recovery_mode,
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=2_000_000_000_000,
    )


def _current_assignment() -> RankAssignment:
    return RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=4,
        assignments=(
            _assignments()[0],
            replace(_assignments()[1], node_id="node-b"),
        ),
        topology_digest=TOPOLOGY_DIGEST,
    )


def _plan(
    *,
    recovery_mode: str = "latest",
    checkpoint_source: str = "gemini",
) -> RestartPlan:
    return RestartPlan(
        plan_id="plan-5",
        intent_id="intent-4",
        run_id=RUN_ID,
        from_generation=4,
        to_generation=5,
        incident_ids=("incident-a",),
        reason_code="replace_straggler",
        recovery_mode=recovery_mode,
        checkpoint_source=checkpoint_source,
        checkpoint_step=40,
        checkpoint_id="durable-40" if checkpoint_source == "durable" else None,
        checkpoint_manifest_id="manifest-40",
        slot_assignments=_assignments(),
        quarantined_node_ids=("node-b",),
        expected_world_size=4,
        topology_digest=TOPOLOGY_DIGEST,
        restart_deadline_unix_ms=2_000_000_000_000,
    )


def _copy(
    rank: int,
    *,
    holder_node_id: str,
    holder_kind: str = "owner",
    checkpoint_step: int = 40,
    inventory_event_id: str = "inventory-a",
    storage_kind: str | None = None,
    complete: bool = True,
) -> CheckpointCopy:
    return CheckpointCopy(
        owner_global_rank=rank,
        checkpoint_step=checkpoint_step,
        inventory_event_id=inventory_event_id,
        holder_node_id=holder_node_id,
        holder_kind=holder_kind,
        storage_kind=storage_kind or ("remote" if holder_kind == "durable" else "memory"),
        location_token=f"copy-{rank}-{holder_node_id}",
        complete=complete,
        checksums_available=True,
    )


def _manifest(
    *,
    trust: str = "latest",
    holder_kind: str = "owner",
    checkpoint_step: int = 40,
    ranks: tuple[int, ...] = (0, 1, 2, 3),
    incomplete_rank: int | None = None,
    holder_overrides: dict[int, str] | None = None,
) -> RecoveryManifest:
    holder_overrides = holder_overrides or {}
    return RecoveryManifest(
        manifest_id="manifest-40",
        run_id=RUN_ID,
        source_generation=4,
        step=40,
        trust=trust,
        topology_digest=TOPOLOGY_DIGEST,
        rank_copies=tuple(
            RankCheckpointCopies(
                owner_global_rank=rank,
                copies=(
                    _copy(
                        rank,
                        holder_node_id=holder_overrides.get(
                            rank,
                            "node-a" if rank < 2 else "node-spare",
                        ),
                        holder_kind=holder_kind,
                        checkpoint_step=checkpoint_step,
                        complete=rank != incomplete_rank,
                    ),
                ),
            )
            for rank in ranks
        ),
    )


def _inventory_event(
    manifest: RecoveryManifest,
    *,
    trust: str | None = None,
) -> CheckpointInventoryEvent:
    return CheckpointInventoryEvent(
        event_id="inventory-a",
        run_id=manifest.run_id,
        generation=manifest.source_generation,
        reporter=replace(
            _worker(),
            generation=manifest.source_generation,
        ),
        step=manifest.step,
        trust=trust or manifest.trust,
        topology_digest=manifest.topology_digest,
        copies=tuple(copy for rank_copies in manifest.rank_copies for copy in rank_copies.copies),
    )


def _validate(
    plan: RestartPlan | None = None,
    manifest: RecoveryManifest | None = None,
    *,
    intent: RestartIntent | None = None,
    current_assignment: RankAssignment | None = None,
    inventory_events: tuple[CheckpointInventoryEvent, ...] | None = None,
    now_unix_ms: int = 1_900_000_000_000,
    eligible_node_ids: tuple[str, ...] = ("node-a", "node-spare"),
) -> None:
    selected_manifest = manifest or _manifest()
    validate_restart_plan(
        plan or _plan(),
        intent or _intent(),
        selected_manifest,
        inventory_events=(
            (_inventory_event(selected_manifest),) if inventory_events is None else inventory_events
        ),
        current_assignment=current_assignment or _current_assignment(),
        now_unix_ms=now_unix_ms,
        eligible_node_ids=eligible_node_ids,
    )


def _records():
    worker = _worker()
    plan = _plan()
    assignments = _assignments()
    hardware_report = HardwareFaultReport(
        kind="hardware",
        resource_kind="gpu",
        resource_id="GPU-0",
        metric="uncorrectable_ecc",
        value=1.0,
        severity="fatal",
        message="fatal ECC",
    )
    return (
        AgentIdentity(
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            hostname="host-a",
            local_world_size=2,
            resource_ids=("GPU-0", "GPU-1"),
            environment_digest="environment-sha256",
        ),
        worker,
        RankAssignment.from_assignments(
            run_id=RUN_ID,
            generation=4,
            assignments=assignments,
            topology_digest=TOPOLOGY_DIGEST,
        ),
        FaultEvent(
            event_id="fault-a",
            incident_id="incident-a",
            run_id=RUN_ID,
            generation=4,
            reporter=worker,
            optimizer_step=41,
            report=hardware_report,
        ),
        RecoveryProposalEvent(
            event_id="recovery-a",
            incident_id="incident-a",
            run_id=RUN_ID,
            generation=4,
            reporter=worker,
            decision={
                "failure_kind": "straggler",
                "recovery_mode": "latest",
                "checkpoint_source": "gemini",
                "checkpoint_step": 40,
                "checkpoint_id": None,
                "all_ranks_accessible": True,
                "available": True,
                "reason": "accessible_straggler",
            },
        ),
        CheckpointInventoryEvent(
            event_id="inventory-a",
            run_id=RUN_ID,
            generation=4,
            reporter=worker,
            step=40,
            trust="latest",
            topology_digest=TOPOLOGY_DIGEST,
            copies=(_copy(0, holder_node_id="node-a"),),
        ),
        _intent(),
        RestartAck(
            intent_id="intent-4",
            node_id="node-a",
            generation=4,
            flushed_step=40,
            inventory_event_ids=("inventory-a",),
            transferred_owner_ranks=(0, 1),
            transferred_peer_ranks=(2, 3),
            success=True,
            reason="prepared",
        ),
        plan,
        RestartContext.from_plan(plan, "node-spare"),
        _manifest(),
    )


@pytest.mark.parametrize("record", _records(), ids=lambda value: type(value).__name__)
def test_wire_records_round_trip_json(record):
    restored = type(record).from_json(record.to_json())

    assert restored == record
    assert json.loads(record.to_json()) == record.to_dict()


def test_wire_records_reject_unknown_schema_and_fields():
    value = _plan().to_dict()
    value["schema_version"] = 99

    with pytest.raises(ProtocolValidationError, match="unsupported value"):
        RestartPlan.from_dict(value)

    value = _plan().to_dict()
    value["unexpected"] = True

    with pytest.raises(ProtocolValidationError, match="unknown fields"):
        RestartPlan.from_dict(value)


def test_event_payloads_are_deeply_immutable_copies():
    report = {
        "kind": "straggler",
        "failed_ranks": [0],
        "evidence": {"groups": ["dp"]},
    }
    event = FaultEvent(
        event_id="fault-a",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report=report,
    )

    report["failed_ranks"].append(1)
    report["evidence"]["groups"].append("tp")

    assert event.to_dict()["report"] == {
        "kind": "straggler",
        "failed_ranks": [0],
        "evidence": {"groups": ["dp"]},
    }


def test_rank_assignment_requires_dense_slots_and_stable_rank_ranges():
    with pytest.raises(ProtocolValidationError, match="dense from zero"):
        RankAssignment.from_assignments(
            run_id=RUN_ID,
            generation=4,
            assignments=(
                SlotAssignment(
                    logical_node_slot=1,
                    node_id="node-a",
                    first_global_rank=2,
                    local_world_size=2,
                ),
            ),
            topology_digest=TOPOLOGY_DIGEST,
        )

    with pytest.raises(ProtocolValidationError, match="must start at rank 2"):
        replace(
            _plan(),
            slot_assignments=(
                _assignments()[0],
                SlotAssignment(
                    logical_node_slot=1,
                    node_id="node-spare",
                    first_global_rank=4,
                    local_world_size=2,
                ),
            ),
        )


def test_restart_plan_and_complete_manifest_validate():
    _validate()


@pytest.mark.parametrize(
    ("manifest", "message"),
    [
        (_manifest(ranks=(0, 1, 2)), "rank coverage mismatch"),
        (_manifest(incomplete_rank=2), "no complete eligible copy"),
        (
            _manifest(holder_overrides={2: "node-b"}),
            "no complete eligible copy",
        ),
    ],
)
def test_restart_plan_rejects_incomplete_or_quarantined_manifest(manifest, message):
    with pytest.raises(ProtocolValidationError, match=message):
        _validate(
            manifest=manifest,
            eligible_node_ids=("node-a", "node-spare", "node-b"),
        )


def test_recovery_manifest_rejects_checkpoint_copies_from_another_step():
    with pytest.raises(ProtocolValidationError, match="copy steps do not match"):
        _manifest(checkpoint_step=39)


def test_restart_plan_rejects_uncertified_manifest_trust():
    manifest = _manifest(trust="recovery_verified")

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            plan=_plan(recovery_mode="recovery_verified"),
            intent=_intent(minimum_recovery_mode="recovery_verified"),
            manifest=manifest,
            inventory_events=(
                _inventory_event(
                    manifest,
                    trust="candidate",
                ),
            ),
        )


def test_restart_plan_requires_inventory_provenance_for_every_copy():
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(inventory_events=())


def test_restart_plan_requires_exact_inventory_copy_match():
    manifest = _manifest()
    inventory = _inventory_event(manifest)
    altered_copies = tuple(
        replace(copy, location_token="altered-location") if copy.owner_global_rank == 0 else copy
        for copy in inventory.copies
    )

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            manifest=manifest,
            inventory_events=(replace(inventory, copies=altered_copies),),
        )


def test_checkpoint_inventory_rejects_mismatched_provenance_id():
    with pytest.raises(ProtocolValidationError, match="provenance does not match"):
        CheckpointInventoryEvent(
            event_id="inventory-b",
            run_id=RUN_ID,
            generation=4,
            reporter=_worker(),
            step=40,
            trust="latest",
            topology_digest=TOPOLOGY_DIGEST,
            copies=(
                _copy(
                    0,
                    holder_node_id="node-a",
                    inventory_event_id="inventory-a",
                ),
            ),
        )


@pytest.mark.parametrize(
    ("plan", "manifest"),
    [
        (_plan(checkpoint_source="durable"), _manifest(holder_kind="owner")),
        (_plan(checkpoint_source="gemini"), _manifest(holder_kind="durable")),
    ],
)
def test_restart_plan_rejects_copies_from_the_wrong_source(plan, manifest):
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(plan=plan, manifest=manifest)


def test_restart_plan_accepts_remote_durable_copies():
    _validate(
        plan=_plan(
            recovery_mode="recovery_verified",
            checkpoint_source="durable",
        ),
        manifest=_manifest(
            trust="recovery_verified",
            holder_kind="durable",
            holder_overrides={
                0: "durable-store",
                1: "durable-store",
                2: "durable-store",
                3: "durable-store",
            },
        ),
    )


def test_restart_plan_requires_complete_remote_or_shared_durable_copies():
    with pytest.raises(ProtocolValidationError, match="shared or remote"):
        _copy(
            0,
            holder_node_id="node-a",
            holder_kind="durable",
            storage_kind="node_local",
        )


def test_restart_plan_enforces_intent_fencing_and_minimum_recovery_mode():
    intent = _intent(minimum_recovery_mode="recovery_verified")

    with pytest.raises(ProtocolValidationError, match="weaker than restart intent"):
        _validate(intent=intent)


@pytest.mark.parametrize(
    ("plan", "message"),
    [
        (replace(_plan(), intent_id="other-intent"), "intent_id"),
        (replace(_plan(), run_id="other-run"), "run_id"),
        (
            replace(_plan(), from_generation=3, to_generation=4),
            "from_generation",
        ),
        (
            replace(_plan(), incident_ids=("other-incident",)),
            "incident_ids",
        ),
        (replace(_plan(), reason_code="other-reason"), "reason_code"),
    ],
)
def test_restart_plan_rejects_intent_fence_mismatch(plan, message):
    with pytest.raises(ProtocolValidationError, match=message):
        _validate(plan=plan)


def test_restart_plan_rejects_expired_deadline():
    with pytest.raises(ProtocolValidationError, match="deadline has elapsed"):
        _validate(now_unix_ms=2_000_000_000_000)


def test_restart_plan_requires_a_replacement_node():
    unchanged_plan = replace(
        _plan(),
        slot_assignments=(
            _assignments()[0],
            replace(_assignments()[1], node_id="node-b"),
        ),
        quarantined_node_ids=(),
    )

    with pytest.raises(ProtocolValidationError, match="replacement node"):
        _validate(
            plan=unchanged_plan,
            eligible_node_ids=("node-a", "node-b"),
        )


def test_restart_plan_preserves_committed_world_size():
    smaller_plan = replace(
        _plan(),
        slot_assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=1,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-spare",
                first_global_rank=1,
                local_world_size=1,
            ),
        ),
        expected_world_size=2,
    )

    with pytest.raises(ProtocolValidationError, match="local world size"):
        _validate(
            plan=smaller_plan,
            manifest=_manifest(ranks=(0, 1)),
        )


def test_worker_identity_rejects_inconsistent_global_rank():
    with pytest.raises(ProtocolValidationError, match="does not match logical slot"):
        replace(_worker(), global_rank=3)


def test_event_reporter_must_match_committed_rank_assignment():
    assignment = _current_assignment()
    contradictory_worker = WorkerIdentity(
        run_id=RUN_ID,
        generation=4,
        node_id="node-b",
        agent_id="agent-b",
        logical_node_slot=1,
        global_rank=1,
        local_rank=0,
        local_world_size=1,
        hostname="host-b",
        gpu_uuid="GPU-2",
        topology_digest=TOPOLOGY_DIGEST,
    )
    event = FaultEvent(
        event_id="fault-contradictory",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=contradictory_worker,
        optimizer_step=41,
        report={"kind": "straggler", "failed_ranks": [1]},
    )

    with pytest.raises(ProtocolValidationError, match="local_world_size"):
        validate_worker_identity(contradictory_worker, assignment)

    with pytest.raises(ProtocolValidationError, match="local_world_size"):
        validate_event_reporter(event, assignment)


@pytest.mark.parametrize(
    ("failure_kind", "all_ranks_accessible"),
    [("sdc", True), ("machine_unavailable", True), ("hang", False)],
)
def test_recovery_proposal_rejects_latest_for_unsafe_failures(
    failure_kind,
    all_ranks_accessible,
):
    with pytest.raises(ProtocolValidationError, match="require recovery_verified"):
        RecoveryProposalEvent(
            event_id="recovery-unsafe",
            incident_id="incident-a",
            run_id=RUN_ID,
            generation=4,
            reporter=_worker(),
            decision={
                "failure_kind": failure_kind,
                "recovery_mode": "latest",
                "checkpoint_source": "gemini",
                "checkpoint_step": 40,
                "checkpoint_id": None,
                "all_ranks_accessible": all_ranks_accessible,
                "available": True,
                "reason": "unsafe",
            },
        )


def test_checkpoint_records_reject_step_zero():
    with pytest.raises(ProtocolValidationError, match="value >= 1"):
        replace(_plan(), checkpoint_step=0)

    with pytest.raises(ProtocolValidationError, match="value >= 1"):
        replace(_manifest(), step=0)

    with pytest.raises(ProtocolValidationError, match="value >= 1"):
        _copy(0, holder_node_id="node-a", checkpoint_step=0)


def test_checkpoint_inventory_rejects_copy_from_another_step():
    with pytest.raises(ProtocolValidationError, match="copy steps do not match"):
        CheckpointInventoryEvent(
            event_id="inventory-mixed",
            run_id=RUN_ID,
            generation=4,
            reporter=_worker(),
            step=40,
            trust="latest",
            topology_digest=TOPOLOGY_DIGEST,
            copies=(
                _copy(
                    0,
                    holder_node_id="node-a",
                    checkpoint_step=39,
                ),
            ),
        )


def test_verified_recovery_rejects_latest_manifest():
    with pytest.raises(ProtocolValidationError, match="verified manifest"):
        _validate(
            plan=_plan(recovery_mode="recovery_verified"),
            manifest=_manifest(trust="latest"),
        )


def test_candidate_cannot_become_recovery_manifest():
    with pytest.raises(ProtocolValidationError, match="unsupported value"):
        _manifest(trust="candidate")


def test_restart_context_validates_worker_environment():
    context = RestartContext.from_plan(_plan(), "node-spare")
    environment = {
        "RANK": "3",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "4",
        "TORCHELASTIC_RUN_ID": RUN_ID,
    }

    context.validate_worker_environment(environment)

    with pytest.raises(ProtocolValidationError, match="RANK=2"):
        context.validate_worker_environment({**environment, "RANK": "2"})

    with pytest.raises(ProtocolValidationError, match="TORCHELASTIC_RUN_ID"):
        context.validate_worker_environment({**environment, "TORCHELASTIC_RUN_ID": "stale-run"})


def test_restart_context_rejects_partial_local_worker_slot():
    context = RestartContext.from_plan(_plan(), "node-a")

    with pytest.raises(ProtocolValidationError, match="complete local worker slots"):
        replace(context, expected_world_size=3)


def test_restart_context_rejects_unassigned_node():
    with pytest.raises(ProtocolValidationError, match="is not assigned"):
        RestartContext.from_plan(_plan(), "node-unknown")
