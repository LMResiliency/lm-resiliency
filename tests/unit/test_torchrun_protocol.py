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
    validate_restart_plan,
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


def _plan(*, recovery_mode: str = "latest") -> RestartPlan:
    return RestartPlan(
        plan_id="plan-5",
        intent_id="intent-4",
        run_id=RUN_ID,
        from_generation=4,
        to_generation=5,
        incident_ids=("incident-a",),
        reason_code="replace_straggler",
        recovery_mode=recovery_mode,
        checkpoint_source="gemini",
        checkpoint_step=40,
        checkpoint_id=None,
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
    complete: bool = True,
) -> CheckpointCopy:
    return CheckpointCopy(
        owner_global_rank=rank,
        holder_node_id=holder_node_id,
        holder_kind=holder_kind,
        storage_kind="remote" if holder_kind == "durable" else "memory",
        location_token=f"copy-{rank}-{holder_node_id}",
        complete=complete,
        checksums_available=True,
    )


def _manifest(
    *,
    trust: str = "latest",
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
                        complete=rank != incomplete_rank,
                    ),
                ),
            )
            for rank in ranks
        ),
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
        RestartIntent(
            intent_id="intent-4",
            run_id=RUN_ID,
            generation=4,
            incident_ids=("incident-a",),
            reason_code="replace_straggler",
            minimum_recovery_mode="latest",
            suspected_node_ids=("node-b",),
            prepare_deadline_unix_ms=2_000_000_000_000,
        ),
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
    validate_restart_plan(
        _plan(),
        _manifest(),
        current_generation=4,
        expected_active_nodes=2,
        expected_topology_digest=TOPOLOGY_DIGEST,
        eligible_node_ids=("node-a", "node-spare"),
    )


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
        validate_restart_plan(
            _plan(),
            manifest,
            current_generation=4,
            expected_active_nodes=2,
            expected_topology_digest=TOPOLOGY_DIGEST,
            eligible_node_ids=("node-a", "node-spare", "node-b"),
        )


def test_verified_recovery_rejects_latest_manifest():
    with pytest.raises(ProtocolValidationError, match="verified manifest"):
        validate_restart_plan(
            _plan(recovery_mode="recovery_verified"),
            _manifest(trust="latest"),
            current_generation=4,
            expected_active_nodes=2,
            expected_topology_digest=TOPOLOGY_DIGEST,
            eligible_node_ids=("node-a", "node-spare"),
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


def test_restart_context_rejects_unassigned_node():
    with pytest.raises(ProtocolValidationError, match="is not assigned"):
        RestartContext.from_plan(_plan(), "node-unknown")
