"""Contract tests for internal torchrun standby-replacement records."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    CheckpointCertification,
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
    checkpoint_inventory_digest,
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


def _agent() -> AgentIdentity:
    return AgentIdentity(
        run_id=RUN_ID,
        node_id="node-a",
        agent_id="agent-a",
        hostname="host-a",
        local_world_size=2,
        resource_ids=("GPU-0", "GPU-1"),
        environment_digest="environment-sha256",
    )


def _intent(
    *,
    minimum_recovery_mode: str = "latest",
    suspected_node_ids: tuple[str, ...] = ("node-b",),
) -> RestartIntent:
    return RestartIntent(
        intent_id="intent-4",
        run_id=RUN_ID,
        generation=4,
        incident_ids=("incident-a",),
        reason_code="replace_straggler",
        minimum_recovery_mode=minimum_recovery_mode,
        suspected_node_ids=suspected_node_ids,
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
    checkpoint_id: str | None = None,
    complete: bool = True,
) -> CheckpointCopy:
    return CheckpointCopy(
        owner_global_rank=rank,
        checkpoint_step=checkpoint_step,
        inventory_event_id=inventory_event_id,
        checkpoint_id=(
            checkpoint_id
            if checkpoint_id is not None
            else ("durable-40" if holder_kind == "durable" else None)
        ),
        holder_node_id=holder_node_id,
        holder_kind=holder_kind,
        storage_kind=storage_kind or ("remote" if holder_kind == "durable" else "memory"),
        location_token=f"copy-{rank}-{holder_node_id}",
        complete=complete,
        checksums_available=True,
    )


def _ack(
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-a",
    flushed_step: int = 40,
    inventory_event_digests: dict[str, str] | None = None,
    success: bool = True,
) -> RestartAck:
    if inventory_event_digests is None:
        event = _inventory_event(_manifest())
        inventory_event_digests = {
            event.event_id: checkpoint_inventory_digest(event),
        }
    return RestartAck(
        intent_id="intent-4",
        run_id=RUN_ID,
        node_id=node_id,
        agent_id=agent_id,
        generation=4,
        flushed_step=flushed_step if success else -1,
        inventory_event_digests=inventory_event_digests,
        transferred_owner_ranks=(0, 1),
        transferred_peer_ranks=(2, 3),
        success=success,
        reason="prepared" if success else "preparation failed",
    )


def _manifest(
    *,
    trust: str = "latest",
    holder_kind: str | None = None,
    checkpoint_step: int = 40,
    ranks: tuple[int, ...] = (0, 1, 2, 3),
    incomplete_rank: int | None = None,
    holder_overrides: dict[int, str] | None = None,
    holder_kind_overrides: dict[int, str] | None = None,
    storage_kind: str | None = None,
    durable_checkpoint_id: str = "durable-40",
) -> RecoveryManifest:
    holder_overrides = holder_overrides or {}
    holder_kind_overrides = holder_kind_overrides or {}

    def rank_holder_kind(rank: int) -> str:
        return holder_kind_overrides.get(
            rank,
            holder_kind or ("owner" if rank < 2 else "peer"),
        )

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
                            "node-a",
                        ),
                        holder_kind=rank_holder_kind(rank),
                        checkpoint_step=checkpoint_step,
                        storage_kind=(
                            storage_kind
                            or ("remote" if rank_holder_kind(rank) == "durable" else "node_local")
                        ),
                        checkpoint_id=(
                            durable_checkpoint_id if rank_holder_kind(rank) == "durable" else None
                        ),
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
    reporter: WorkerIdentity | None = None,
) -> CheckpointInventoryEvent:
    return CheckpointInventoryEvent(
        event_id="inventory-a",
        run_id=manifest.run_id,
        generation=manifest.source_generation,
        reporter=replace(
            reporter or _worker(),
            generation=manifest.source_generation,
        ),
        step=manifest.step,
        trust=trust or manifest.trust,
        topology_digest=manifest.topology_digest,
        copies=tuple(copy for rank_copies in manifest.rank_copies for copy in rank_copies.copies),
    )


def _certification(
    plan: RestartPlan,
    manifest: RecoveryManifest,
    inventory_events: tuple[CheckpointInventoryEvent, ...],
) -> CheckpointCertification:
    return CheckpointCertification(
        certification_id=f"certification-{manifest.step}",
        run_id=manifest.run_id,
        source_generation=manifest.source_generation,
        step=manifest.step,
        topology_digest=manifest.topology_digest,
        checkpoint_source=plan.checkpoint_source,
        checkpoint_id=plan.checkpoint_id,
        expected_world_size=plan.expected_world_size,
        certification_kind="dense_consensus",
        inventory_event_digests={
            event.event_id: checkpoint_inventory_digest(event) for event in inventory_events
        },
    )


def _validate(
    plan: RestartPlan | None = None,
    manifest: RecoveryManifest | None = None,
    *,
    intent: RestartIntent | None = None,
    current_assignment: RankAssignment | None = None,
    inventory_events: tuple[CheckpointInventoryEvent, ...] | None = None,
    trusted_certifications: tuple[CheckpointCertification, ...] | None = None,
    restart_acks: tuple[RestartAck, ...] | None = None,
    authenticated_ack_agent_ids: dict[str, str] | None = None,
    authenticated_ack_received_unix_ms: dict[str, int] | None = None,
    source_assignment: RankAssignment | None = None,
    now_unix_ms: int = 1_900_000_000_000,
    eligible_node_ids: tuple[str, ...] = ("node-a", "node-spare"),
) -> None:
    selected_plan = plan or _plan()
    selected_intent = intent or _intent()
    selected_manifest = manifest or _manifest()
    selected_acks = (_ack(),) if restart_acks is None else restart_acks
    selected_inventory_events = (
        (_inventory_event(selected_manifest),) if inventory_events is None else inventory_events
    )
    validate_restart_plan(
        selected_plan,
        selected_intent,
        selected_manifest,
        inventory_events=selected_inventory_events,
        trusted_certifications=(
            (
                _certification(
                    selected_plan,
                    selected_manifest,
                    selected_inventory_events,
                ),
            )
            if trusted_certifications is None and selected_manifest.trust == "recovery_verified"
            else (() if trusted_certifications is None else trusted_certifications)
        ),
        restart_acks=selected_acks,
        authenticated_ack_agent_ids=(
            {ack.node_id: ack.agent_id for ack in selected_acks}
            if authenticated_ack_agent_ids is None
            else authenticated_ack_agent_ids
        ),
        authenticated_ack_received_unix_ms=(
            {ack.node_id: selected_intent.prepare_deadline_unix_ms - 1 for ack in selected_acks}
            if authenticated_ack_received_unix_ms is None
            else authenticated_ack_received_unix_ms
        ),
        source_assignment=source_assignment or _current_assignment(),
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
        _agent(),
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
        _certification(
            _plan(recovery_mode="recovery_verified"),
            _manifest(trust="recovery_verified"),
            (_inventory_event(_manifest(trust="recovery_verified")),),
        ),
        _intent(),
        _ack(),
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


def test_verified_inventory_requires_trusted_catalog_certification():
    manifest = _manifest(trust="recovery_verified")

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            plan=_plan(recovery_mode="recovery_verified"),
            intent=_intent(minimum_recovery_mode="recovery_verified"),
            manifest=manifest,
            trusted_certifications=(),
        )


def test_certification_binds_immutable_inventory_contents():
    plan = _plan(recovery_mode="recovery_verified")
    original_manifest = _manifest(trust="recovery_verified")
    original_event = _inventory_event(original_manifest)
    certification = _certification(
        plan,
        original_manifest,
        (original_event,),
    )
    altered_manifest = replace(
        original_manifest,
        rank_copies=tuple(
            replace(
                rank_copies,
                copies=tuple(
                    replace(copy, location_token="substituted-location")
                    if copy.owner_global_rank == 0
                    else copy
                    for copy in rank_copies.copies
                ),
            )
            for rank_copies in original_manifest.rank_copies
        ),
    )

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            plan=plan,
            intent=_intent(minimum_recovery_mode="recovery_verified"),
            manifest=altered_manifest,
            inventory_events=(_inventory_event(altered_manifest),),
            trusted_certifications=(certification,),
        )


def test_restart_ack_binds_immutable_inventory_contents():
    original_manifest = _manifest()
    original_event = _inventory_event(original_manifest)
    altered_manifest = replace(
        original_manifest,
        rank_copies=tuple(
            replace(
                rank_copies,
                copies=tuple(
                    replace(copy, location_token="substituted-location")
                    if copy.owner_global_rank == 0
                    else copy
                    for copy in rank_copies.copies
                ),
            )
            for rank_copies in original_manifest.rank_copies
        ),
    )

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            manifest=altered_manifest,
            inventory_events=(_inventory_event(altered_manifest),),
            restart_acks=(
                _ack(
                    inventory_event_digests={
                        original_event.event_id: checkpoint_inventory_digest(original_event),
                    }
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


def test_restart_plan_rejects_ineligible_copy_alongside_eligible_copy():
    base_manifest = _manifest()
    inventory = _inventory_event(base_manifest)
    manifest = replace(
        base_manifest,
        rank_copies=tuple(
            replace(
                entry,
                copies=entry.copies
                + (
                    replace(
                        entry.copies[0],
                        inventory_event_id="inventory-unproven",
                        location_token="unproven-copy",
                    ),
                ),
            )
            if entry.owner_global_rank == 0
            else entry
            for entry in base_manifest.rank_copies
        ),
    )

    with pytest.raises(ProtocolValidationError, match="contains an ineligible copy"):
        _validate(
            manifest=manifest,
            inventory_events=(inventory,),
            restart_acks=(
                _ack(
                    inventory_event_digests={
                        inventory.event_id: checkpoint_inventory_digest(inventory),
                    }
                ),
            ),
        )


def test_restart_plan_rejects_local_copy_reported_by_another_node():
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            manifest=_manifest(
                holder_overrides={0: "node-spare"},
            ),
        )


def test_restart_plan_rejects_local_copy_on_departing_holder():
    departing_worker = replace(
        _worker(),
        node_id="node-b",
        agent_id="agent-b",
        logical_node_slot=1,
        global_rank=2,
        hostname="host-b",
        gpu_uuid="GPU-2",
    )
    manifest = _manifest(
        holder_overrides={
            0: "node-b",
            1: "node-b",
            2: "node-b",
            3: "node-b",
        },
        holder_kind_overrides={
            0: "peer",
            1: "peer",
            2: "owner",
            3: "owner",
        },
    )
    departing_event = _inventory_event(
        manifest,
        reporter=departing_worker,
    )

    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            plan=replace(_plan(), quarantined_node_ids=()),
            manifest=manifest,
            inventory_events=(departing_event,),
            restart_acks=(
                _ack(
                    node_id="node-b",
                    agent_id="agent-b",
                    inventory_event_digests={
                        departing_event.event_id: checkpoint_inventory_digest(departing_event),
                    },
                ),
            ),
            authenticated_ack_agent_ids={"node-b": "agent-b"},
            eligible_node_ids=("node-a", "node-b", "node-spare"),
        )


def test_restart_plan_rejects_process_memory_copies():
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            manifest=_manifest(storage_kind="memory"),
        )


def test_restart_plan_rejects_owner_copy_held_by_another_node():
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            manifest=_manifest(holder_kind="owner"),
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
        (
            _plan(checkpoint_source="durable"),
            _manifest(trust="recovery_verified", holder_kind="owner"),
        ),
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


def test_restart_plan_binds_durable_copies_to_checkpoint_id():
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(
            plan=_plan(
                recovery_mode="recovery_verified",
                checkpoint_source="durable",
            ),
            manifest=_manifest(
                trust="recovery_verified",
                holder_kind="durable",
                durable_checkpoint_id="durable-other",
                holder_overrides={
                    0: "durable-store",
                    1: "durable-store",
                    2: "durable-store",
                    3: "durable-store",
                },
            ),
        )


def test_durable_checkpoint_copy_requires_checkpoint_id():
    with pytest.raises(ProtocolValidationError, match="require a checkpoint ID"):
        replace(
            _copy(
                0,
                holder_node_id="durable-store",
                holder_kind="durable",
            ),
            checkpoint_id=None,
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
            intent=_intent(suspected_node_ids=()),
            eligible_node_ids=("node-a", "node-b"),
        )


def test_restart_plan_preserves_surviving_nodes_logical_slots():
    shifted_plan = replace(
        _plan(),
        slot_assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-spare",
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
    )

    with pytest.raises(ProtocolValidationError, match="changed logical slots"):
        _validate(plan=shifted_plan)


def test_restart_plan_removes_every_node_suspected_by_intent():
    wrong_replacement = replace(
        _plan(),
        slot_assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-spare",
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
        quarantined_node_ids=(),
    )

    with pytest.raises(ProtocolValidationError, match="suspected nodes remain assigned"):
        _validate(
            plan=wrong_replacement,
            eligible_node_ids=("node-b", "node-spare"),
        )


def test_restart_plan_quarantines_only_policy_approved_removed_nodes():
    with pytest.raises(ProtocolValidationError, match="not in the intent's suspected scope"):
        _validate(
            plan=replace(
                _plan(),
                quarantined_node_ids=("node-b", "node-c"),
            ),
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


def test_source_assignment_cannot_conflict_with_current_generation():
    conflicting_source = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=4,
        assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-spare",
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
        topology_digest=TOPOLOGY_DIGEST,
    )

    with pytest.raises(ProtocolValidationError, match="conflicts with the committed assignment"):
        _validate(source_assignment=conflicting_source)


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
        validate_event_reporter(
            event,
            assignment,
            agent_identity=_agent(),
            resource_to_node_id={"GPU-0": "node-a"},
            resource_to_kind={"GPU-0": "gpu"},
            resource_to_global_rank={"GPU-0": 0},
        )


def test_hardware_report_resource_must_match_registered_owner():
    event = FaultEvent(
        event_id="fault-resource",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report=HardwareFaultReport(
            kind="hardware",
            resource_kind="gpu",
            resource_id="GPU-1",
            metric="uncorrectable_ecc",
            value=1.0,
            severity="fatal",
            message="fatal ECC",
        ),
    )

    with pytest.raises(ProtocolValidationError, match="trusted resource owner"):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={
                "GPU-0": "node-a",
                "GPU-1": "node-b",
            },
            resource_to_kind={
                "GPU-0": "gpu",
                "GPU-1": "gpu",
            },
            resource_to_global_rank={
                "GPU-0": 0,
                "GPU-1": 1,
            },
        )


def test_hardware_report_accepts_registered_resource_on_reporter_node():
    event = FaultEvent(
        event_id="fault-resource",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report=HardwareFaultReport(
            kind="hardware",
            resource_kind="gpu",
            resource_id="GPU-0",
            metric="uncorrectable_ecc",
            value=1.0,
            severity="fatal",
            message="fatal ECC",
        ),
    )

    validate_event_reporter(
        event,
        _current_assignment(),
        agent_identity=_agent(),
        resource_to_node_id={
            "GPU-0": "node-a",
            "GPU-1": "node-a",
        },
        resource_to_kind={
            "GPU-0": "gpu",
            "GPU-1": "gpu",
        },
        resource_to_global_rank={
            "GPU-0": 0,
            "GPU-1": 1,
        },
    )


def test_hardware_report_resource_kind_must_match_trusted_inventory():
    event = FaultEvent(
        event_id="fault-resource-kind",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report=HardwareFaultReport(
            kind="hardware",
            resource_kind="nic",
            resource_id="GPU-1",
            metric="link_down",
            value=1.0,
            severity="fatal",
            message="reported with the wrong resource kind",
        ),
    )

    with pytest.raises(ProtocolValidationError, match="resource kind"):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={
                "GPU-0": "node-a",
                "GPU-1": "node-a",
            },
            resource_to_kind={
                "GPU-0": "gpu",
                "GPU-1": "gpu",
            },
            resource_to_global_rank={
                "GPU-0": 0,
                "GPU-1": 1,
            },
        )


def test_hardware_report_rejects_oversized_numeric_metric():
    event = FaultEvent(
        event_id="fault-oversized-metric",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report=HardwareFaultReport(
            kind="hardware",
            resource_kind="gpu",
            resource_id="GPU-0",
            metric="counter",
            value=1.0,
            severity="fatal",
            message="oversized metric",
        ),
    )
    payload = event.to_dict()
    payload["report"]["value"] = 10**1000

    with pytest.raises(ProtocolValidationError, match="finite float"):
        FaultEvent.from_json(json.dumps(payload))


def test_fault_report_rejects_rank_outside_committed_assignment():
    event = FaultEvent(
        event_id="fault-rank",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report={"kind": "straggler", "failed_ranks": [999]},
    )

    with pytest.raises(ProtocolValidationError, match="not active"):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={"GPU-0": "node-a"},
            resource_to_kind={"GPU-0": "gpu"},
            resource_to_global_rank={"GPU-0": 0},
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataloader_culprit_ranks", [3], "not present in failed_ranks"),
        ("stage_culprit_ranks", [999], "not active"),
    ],
)
def test_fault_report_validates_all_rank_attribution_fields(field, value, message):
    event = FaultEvent(
        event_id="fault-attribution",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report={
            "kind": "data_stall",
            "failed_ranks": [0],
            field: value,
        },
    )

    with pytest.raises(ProtocolValidationError, match=message):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={"GPU-0": "node-a"},
            resource_to_kind={"GPU-0": "gpu"},
            resource_to_global_rank={"GPU-0": 0},
        )


def test_fault_report_endpoint_must_match_failed_rank_and_node():
    event = FaultEvent(
        event_id="fault-endpoint",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report={
            "kind": "straggler",
            "failed_ranks": [0],
            "endpoint_kind": "node",
            "endpoint_id": "node-b",
            "endpoint_rank": 0,
        },
    )

    with pytest.raises(ProtocolValidationError, match="endpoint rank's node"):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={"GPU-0": "node-a"},
            resource_to_kind={"GPU-0": "gpu"},
            resource_to_global_rank={"GPU-0": 0},
        )


def test_fault_report_resource_endpoint_must_match_specific_rank():
    event = FaultEvent(
        event_id="fault-resource-endpoint",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=4,
        reporter=_worker(),
        optimizer_step=41,
        report={
            "kind": "straggler",
            "failed_ranks": [0],
            "endpoint_kind": "gpu",
            "endpoint_id": "GPU-1",
            "endpoint_rank": 0,
        },
    )

    with pytest.raises(ProtocolValidationError, match="resource rank"):
        validate_event_reporter(
            event,
            _current_assignment(),
            agent_identity=_agent(),
            resource_to_node_id={
                "GPU-0": "node-a",
                "GPU-1": "node-a",
            },
            resource_to_kind={
                "GPU-0": "gpu",
                "GPU-1": "gpu",
            },
            resource_to_global_rank={
                "GPU-0": 0,
                "GPU-1": 1,
            },
        )


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


def test_recovery_proposal_rejects_unknown_failure_kind():
    with pytest.raises(ProtocolValidationError, match="unsupported value"):
        RecoveryProposalEvent(
            event_id="recovery-unknown",
            incident_id="incident-a",
            run_id=RUN_ID,
            generation=4,
            reporter=_worker(),
            decision={
                "failure_kind": "stragler",
                "recovery_mode": "latest",
                "checkpoint_source": "gemini",
                "checkpoint_step": 40,
                "checkpoint_id": None,
                "all_ranks_accessible": True,
                "available": True,
                "reason": "unknown failure kind",
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


def test_durable_recovery_rejects_latest_manifest_even_in_latest_mode():
    with pytest.raises(ProtocolValidationError, match="durable recovery requires"):
        _validate(
            plan=_plan(checkpoint_source="durable"),
            manifest=_manifest(
                trust="latest",
                holder_kind="durable",
                holder_overrides={
                    0: "durable-store",
                    1: "durable-store",
                    2: "durable-store",
                    3: "durable-store",
                },
            ),
        )


@pytest.mark.parametrize(
    "restart_acks",
    [
        (),
        (_ack(success=False),),
        (_ack(flushed_step=39),),
        (_ack(inventory_event_digests={"inventory-other": "0" * 64}),),
    ],
)
def test_latest_recovery_requires_successful_preparation_ack(restart_acks):
    with pytest.raises(ProtocolValidationError, match="no complete eligible copy"):
        _validate(restart_acks=restart_acks)


def test_restart_ack_must_match_authenticated_agent():
    with pytest.raises(ProtocolValidationError, match="authenticated transport sender"):
        _validate(
            authenticated_ack_agent_ids={"node-a": "agent-other"},
        )


def test_restart_ack_must_arrive_by_prepare_deadline():
    with pytest.raises(ProtocolValidationError, match="received after"):
        _validate(
            authenticated_ack_received_unix_ms={"node-a": 2_000_000_000_001},
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

    context.validate_worker_environment(
        environment,
        committed_plan=_plan(),
        now_unix_ms=1_900_000_000_000,
    )

    with pytest.raises(ProtocolValidationError, match="RANK=2"):
        context.validate_worker_environment(
            {**environment, "RANK": "2"},
            committed_plan=_plan(),
            now_unix_ms=1_900_000_000_000,
        )

    with pytest.raises(ProtocolValidationError, match="TORCHELASTIC_RUN_ID"):
        context.validate_worker_environment(
            {**environment, "TORCHELASTIC_RUN_ID": "stale-run"},
            committed_plan=_plan(),
            now_unix_ms=1_900_000_000_000,
        )


def test_restart_context_rejects_expired_committed_plan():
    plan = _plan()
    context = RestartContext.from_plan(plan, "node-spare")
    environment = {
        "RANK": "3",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "4",
        "TORCHELASTIC_RUN_ID": RUN_ID,
    }

    with pytest.raises(ProtocolValidationError, match="deadline has elapsed"):
        context.validate_worker_environment(
            environment,
            committed_plan=plan,
            now_unix_ms=plan.restart_deadline_unix_ms,
        )


def test_restart_context_must_match_current_committed_plan():
    stale_context = RestartContext.from_plan(_plan(), "node-spare")
    current_plan = replace(
        _plan(),
        plan_id="plan-6",
        from_generation=5,
        to_generation=6,
        recovery_mode="recovery_verified",
    )
    environment = {
        "RANK": "3",
        "LOCAL_RANK": "1",
        "LOCAL_WORLD_SIZE": "2",
        "WORLD_SIZE": "4",
        "TORCHELASTIC_RUN_ID": RUN_ID,
    }

    with pytest.raises(ProtocolValidationError, match="currently committed"):
        stale_context.validate_worker_environment(
            environment,
            committed_plan=current_plan,
            now_unix_ms=1_900_000_000_000,
        )


def test_restart_context_rejects_partial_local_worker_slot():
    context = RestartContext.from_plan(_plan(), "node-a")

    with pytest.raises(ProtocolValidationError, match="complete local worker slots"):
        replace(context, expected_world_size=3)


def test_restart_context_rejects_unassigned_node():
    with pytest.raises(ProtocolValidationError, match="is not assigned"):
        RestartContext.from_plan(_plan(), "node-unknown")
