"""Contract tests for resolved torchrun restart-plan state."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistory,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    CheckpointCertification,
    CheckpointCopy,
    CheckpointInventoryEvent,
    RankAssignment,
    RankCheckpointCopies,
    RecoveryManifest,
    RestartIntent,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
    checkpoint_inventory_digest,
)
from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication import (
    RestartPlanPublicationAuthorityPreparer,
    RestartPlanPublicationPreparationClockError,
    RestartPlanPublicationPreparationConflict,
    RestartPlanPublicationPreparationCorrupt,
    RestartPlanPublicationPreparationLeaseLost,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_records import (
    RestartPlanPublicationAuthority,
    RestartPlanPublicationRecords,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    PersistedRestartPlanPublication,
    ResolvedRecoveryManifest,
    RestartPlanCandidateState,
    RestartPlanCertificationState,
    RestartPlanCopyEligibilityState,
    RestartPlanGenerationState,
    RestartPlanInventoryState,
    RestartPlanManifestState,
    RestartPlanPlacementState,
    RestartPlanQuarantineState,
    RestartPlanRecoveryEvidenceState,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms

    def set(self, now_unix_ms: int) -> None:
        with self._lock:
            self.now_unix_ms = now_unix_ms


class FailingLeaseHistoryReader:
    def __init__(self, error: Exception) -> None:
        self._error = error

    def read(self):
        raise self._error


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


def _registration_history(
    node_id: str,
    *,
    local_world_size: int = 2,
    environment_digest: str = "environment-v1",
    granted_at_unix_ms: int = 1_100,
    lease_duration_ms: int = 500,
    current: bool = True,
) -> AgentRegistrationHistory:
    record = AgentRegistrationRecord(
        agent_identity=AgentIdentity(
            run_id=RUN_ID,
            node_id=node_id,
            agent_id=f"agent-{node_id}",
            hostname=f"host-{node_id}",
            local_world_size=local_world_size,
            resource_ids=(f"{node_id}-gpu-0", f"{node_id}-gpu-1"),
            environment_digest=environment_digest,
        ),
        registration_id=f"registration-{node_id}",
        lease_duration_ms=lease_duration_ms,
    )
    held = HeldAgentRegistration(
        record=record,
        fencing_token=11,
        granted_at_unix_ms=granted_at_unix_ms,
    )
    authority = AgentRegistrationAuthority(
        registration=held,
        transaction_sequence=11,
        mutation_sequence=1,
        value_sequence=1,
        lifetime_sequence=1,
    )
    return AgentRegistrationHistory(
        authorities=(authority,),
        current=held if current else None,
    )


def _placement_state(
    *,
    generation_state: RestartPlanGenerationState | None = None,
    registration_histories: dict[str, AgentRegistrationHistory] | None = None,
    observed_at_unix_ms: int = 1_200,
    environment_digest: str = "environment-v1",
) -> RestartPlanPlacementState:
    return RestartPlanPlacementState(
        generation_state=generation_state or _generation_state(),
        registration_histories=registration_histories
        or {
            "node-a": _registration_history("node-a"),
            "node-c": _registration_history("node-c"),
        },
        observed_at_unix_ms=observed_at_unix_ms,
        environment_digest=environment_digest,
    )


def _replace_generation_intent(
    state: RestartPlanGenerationState,
    intent: RestartIntent,
) -> RestartPlanGenerationState:
    intent_record = replace(state.intent_record, intent=intent)
    lifecycle = replace(
        state.lifecycle_record,
        closed_intent=replace(
            state.lifecycle_record.closed_intent,
            intent_digest=intent_record.digest,
        ),
    )
    return replace(
        state,
        intent_record=intent_record,
        lifecycle_record=lifecycle,
        record=replace(
            state.record,
            intent_lifecycle_record_digest=lifecycle.digest,
        ),
    )


def _replace_generation_plan(
    state: RestartPlanGenerationState,
    plan: RestartPlan,
) -> RestartPlanGenerationState:
    assignment = RankAssignment.from_assignments(
        run_id=plan.run_id,
        generation=plan.to_generation,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    to_snapshot = replace(state.to_snapshot, assignment=assignment)
    return replace(
        state,
        to_snapshot=to_snapshot,
        record=replace(
            state.record,
            plan=plan,
            to_generation_snapshot_digest=to_snapshot.digest,
            quarantine_record_digests={node_id: "0" * 64 for node_id in plan.quarantined_node_ids},
        ),
    )


def test_restart_plan_placement_state_binds_live_compatible_registrations():
    state = _placement_state()

    assert state.plan == state.generation_state.plan
    assert tuple(state.registration_histories) == ("node-a", "node-c")
    assert state.registration_histories["node-c"].current is not None


def test_restart_plan_placement_state_requires_exact_types():
    state = _placement_state()

    with pytest.raises(TypeError, match="generation_state must be RestartPlanGenerationState"):
        replace(state, generation_state=state.generation_state.record)
    with pytest.raises(TypeError, match="registration_histories must be a mapping"):
        replace(state, registration_histories=())
    with pytest.raises(TypeError, match="values must be AgentRegistrationHistory"):
        replace(
            state,
            registration_histories={
                "node-a": _registration_history("node-a"),
                "node-c": _registration_history("node-c").current,
            },
        )


@pytest.mark.parametrize("observed_at_unix_ms", [0, True])
def test_restart_plan_placement_state_rejects_invalid_observation_time(
    observed_at_unix_ms,
):
    with pytest.raises(ValueError, match="observed_at_unix_ms"):
        _placement_state(observed_at_unix_ms=observed_at_unix_ms)


def test_restart_plan_placement_state_rejects_invalid_environment_digest():
    with pytest.raises(ValueError, match="environment_digest"):
        _placement_state(environment_digest=" ")


@pytest.mark.parametrize(
    ("suspected_node_ids", "message"),
    [
        (("node-z",), "not active"),
        (("node-a",), "remain assigned"),
    ],
)
def test_restart_plan_placement_state_rejects_invalid_suspect_scope(
    suspected_node_ids,
    message,
):
    state = _generation_state()
    changed = _replace_generation_intent(
        state,
        replace(
            state.intent_record.intent,
            suspected_node_ids=suspected_node_ids,
        ),
    )

    with pytest.raises(ValueError, match=message):
        _placement_state(generation_state=changed)


def test_restart_plan_placement_state_rejects_moved_survivor():
    state = _generation_state()
    changed = _replace_generation_plan(
        state,
        replace(
            state.plan,
            slot_assignments=(
                SlotAssignment(
                    logical_node_slot=0,
                    node_id="node-c",
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
        ),
    )

    with pytest.raises(ValueError, match="changed logical slots"):
        _placement_state(generation_state=changed)


def test_restart_plan_placement_state_requires_replacement():
    state = _replace_generation_intent(
        _generation_state(),
        replace(_intent_record().intent, suspected_node_ids=()),
    )
    changed = _replace_generation_plan(
        state,
        replace(
            state.plan,
            slot_assignments=tuple(
                SlotAssignment(
                    logical_node_slot=slot,
                    node_id=node_id,
                    first_global_rank=slot * 2,
                    local_world_size=2,
                )
                for slot, node_id in state.from_assignment.slot_to_node_id.items()
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires at least one replacement"):
        _placement_state(
            generation_state=changed,
            registration_histories={
                "node-a": _registration_history("node-a"),
                "node-b": _registration_history("node-b"),
            },
        )


def test_restart_plan_placement_state_rejects_shrink_without_replacement():
    state = _generation_state()
    changed = _replace_generation_plan(
        state,
        replace(
            state.plan,
            slot_assignments=(
                SlotAssignment(
                    logical_node_slot=0,
                    node_id="node-a",
                    first_global_rank=0,
                    local_world_size=4,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="requires at least one replacement"):
        _placement_state(
            generation_state=changed,
            registration_histories={
                "node-a": _registration_history("node-a", local_world_size=4),
            },
        )


def test_restart_plan_placement_state_preserves_active_node_count():
    state = _generation_state()
    changed = _replace_generation_plan(
        state,
        replace(
            state.plan,
            slot_assignments=(
                SlotAssignment(
                    logical_node_slot=0,
                    node_id="node-c",
                    first_global_rank=0,
                    local_world_size=4,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="active node count"):
        _placement_state(
            generation_state=changed,
            registration_histories={
                "node-c": _registration_history("node-c", local_world_size=4),
            },
        )


def test_restart_plan_placement_state_rejects_quarantine_outside_intent_scope():
    state = _generation_state()
    changed = _replace_generation_plan(
        state,
        replace(state.plan, quarantined_node_ids=("node-z",)),
    )

    with pytest.raises(ValueError, match="outside the intent scope"):
        _placement_state(generation_state=changed)


@pytest.mark.parametrize(
    "registration_histories",
    [
        {"node-a": _registration_history("node-a")},
        {
            "node-a": _registration_history("node-a"),
            "node-c": _registration_history("node-c"),
            "node-d": _registration_history("node-d"),
        },
    ],
)
def test_restart_plan_placement_state_requires_exact_registration_coverage(
    registration_histories,
):
    with pytest.raises(ValueError, match="exactly cover"):
        _placement_state(registration_histories=registration_histories)


@pytest.mark.parametrize(
    ("history", "message"),
    [
        (_registration_history("node-c", current=False), "no current registration"),
        (
            _registration_history("node-c", granted_at_unix_ms=1_201),
            "granted after",
        ),
        (
            _registration_history(
                "node-c",
                granted_at_unix_ms=1_100,
                lease_duration_ms=100,
            ),
            "not live",
        ),
        (
            _registration_history("node-c", local_world_size=1),
            "local world size",
        ),
        (
            _registration_history("node-c", environment_digest="environment-v2"),
            "environment is incompatible",
        ),
        (_registration_history("node-z"), "wrong identity"),
    ],
)
def test_restart_plan_placement_state_rejects_ineligible_registration(
    history,
    message,
):
    with pytest.raises(ValueError, match=message):
        _placement_state(
            registration_histories={
                "node-a": _registration_history("node-a"),
                "node-c": history,
            },
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
        copies=rank_copies.copies,
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


def _shared_manifest() -> RecoveryManifest:
    manifest = _manifest()
    return replace(
        manifest,
        rank_copies=tuple(
            replace(
                entry,
                copies=tuple(
                    replace(
                        copy,
                        storage_kind="shared",
                    )
                    for copy in entry.copies
                ),
            )
            for entry in manifest.rank_copies
        ),
    )


def _inventory_state_for_manifest(
    manifest: RecoveryManifest,
    *,
    plan: RestartPlan | None = None,
) -> RestartPlanInventoryState:
    return _inventory_state(
        quarantine_state=_quarantine_state(
            manifest_state=_manifest_state(
                manifest=manifest,
                plan=plan,
            )
        )
    )


def _copy_eligibility_state() -> RestartPlanCopyEligibilityState:
    return RestartPlanCopyEligibilityState(
        inventory_state=_inventory_state_for_manifest(_shared_manifest())
    )


def test_restart_plan_copy_eligibility_state_accepts_shared_gemini_copies():
    state = _copy_eligibility_state()

    assert state.plan == state.inventory_state.plan
    assert state.manifest == state.inventory_state.manifest


def test_restart_plan_copy_eligibility_state_requires_exact_type():
    state = _copy_eligibility_state()

    with pytest.raises(TypeError, match="inventory_state must be"):
        replace(
            state,
            inventory_state=state.inventory_state.quarantine_state,
        )


def test_restart_plan_copy_eligibility_state_requires_exact_rank_coverage():
    manifest = replace(
        _shared_manifest(),
        rank_copies=_shared_manifest().rank_copies[:-1],
    )
    quarantine_state = _quarantine_state(manifest_state=_manifest_state(manifest=manifest))
    inventory_state = _inventory_state(
        quarantine_state=quarantine_state,
        inventory_events={
            event.event_id: event
            for event in (
                _inventory_event(quarantine_state, rank)
                for rank in range(manifest.rank_copies[-1].owner_global_rank + 1)
            )
        },
    )

    with pytest.raises(ValueError, match="rank coverage mismatch"):
        RestartPlanCopyEligibilityState(inventory_state)


def test_restart_plan_copy_eligibility_state_requires_one_copy_per_rank():
    manifest = _shared_manifest()
    manifest = replace(
        manifest,
        rank_copies=(
            replace(manifest.rank_copies[0], copies=()),
            *manifest.rank_copies[1:],
        ),
    )
    quarantine_state = _quarantine_state(manifest_state=_manifest_state(manifest=manifest))
    inventory_state = _inventory_state(
        quarantine_state=quarantine_state,
        inventory_events={
            event.event_id: event
            for event in (
                _inventory_event(quarantine_state, rank)
                for rank in range(1, manifest.rank_copies[-1].owner_global_rank + 1)
            )
        },
    )

    with pytest.raises(ValueError, match="has no eligible copy"):
        RestartPlanCopyEligibilityState(inventory_state)


@pytest.mark.parametrize(
    "copy",
    [
        replace(
            _shared_manifest().rank_copies[0].copies[0],
            complete=False,
        ),
        replace(
            _shared_manifest().rank_copies[0].copies[0],
            storage_kind="memory",
        ),
        replace(
            _shared_manifest().rank_copies[0].copies[0],
            holder_node_id="node-b",
        ),
        replace(
            _shared_manifest().rank_copies[0].copies[0],
            holder_kind="peer",
        ),
    ],
)
def test_restart_plan_copy_eligibility_state_rejects_ineligible_gemini_copy(copy):
    manifest = _shared_manifest()
    manifest = replace(
        manifest,
        rank_copies=(
            replace(manifest.rank_copies[0], copies=(copy,)),
            *manifest.rank_copies[1:],
        ),
    )

    with pytest.raises(ValueError, match="contains an ineligible copy"):
        RestartPlanCopyEligibilityState(_inventory_state_for_manifest(manifest))


def test_restart_plan_copy_eligibility_state_rejects_departing_node_local_holder():
    with pytest.raises(ValueError, match="contains an ineligible copy"):
        RestartPlanCopyEligibilityState(_inventory_state())


def test_restart_plan_copy_eligibility_state_rejects_ineligible_alternative_copy():
    manifest = _shared_manifest()
    eligible_copy = manifest.rank_copies[0].copies[0]
    ineligible_copy = replace(
        eligible_copy,
        storage_kind="memory",
        location_token="memory-copy-0",
    )
    manifest = replace(
        manifest,
        rank_copies=(
            replace(
                manifest.rank_copies[0],
                copies=(eligible_copy, ineligible_copy),
            ),
            *manifest.rank_copies[1:],
        ),
    )

    with pytest.raises(ValueError, match="contains an ineligible copy"):
        RestartPlanCopyEligibilityState(_inventory_state_for_manifest(manifest))


def _durable_manifest(
    *,
    checkpoint_id: str = "durable-40",
) -> RecoveryManifest:
    manifest = replace(_manifest(), trust="recovery_verified")
    return replace(
        manifest,
        rank_copies=tuple(
            replace(
                entry,
                copies=tuple(
                    replace(
                        copy,
                        checkpoint_id=checkpoint_id,
                        holder_node_id="durable-store",
                        holder_kind="durable",
                        storage_kind="remote",
                    )
                    for copy in entry.copies
                ),
            )
            for entry in manifest.rank_copies
        ),
    )


def _durable_plan() -> RestartPlan:
    return replace(
        _plan(),
        recovery_mode="recovery_verified",
        checkpoint_source="durable",
        checkpoint_id="durable-40",
    )


def test_restart_plan_copy_eligibility_state_accepts_remote_durable_copies():
    state = RestartPlanCopyEligibilityState(
        _inventory_state_for_manifest(
            _durable_manifest(),
            plan=_durable_plan(),
        )
    )

    assert state.plan.checkpoint_source == "durable"


def test_restart_plan_copy_eligibility_state_rejects_wrong_source_kind():
    verified_manifest = replace(_shared_manifest(), trust="recovery_verified")

    with pytest.raises(ValueError, match="contains an ineligible copy"):
        RestartPlanCopyEligibilityState(
            _inventory_state_for_manifest(
                verified_manifest,
                plan=_durable_plan(),
            )
        )


def test_restart_plan_copy_eligibility_state_rejects_wrong_checkpoint_id():
    with pytest.raises(ValueError, match="contains an ineligible copy"):
        RestartPlanCopyEligibilityState(
            _inventory_state_for_manifest(
                _durable_manifest(checkpoint_id="durable-other"),
                plan=_durable_plan(),
            )
        )


def test_restart_plan_recovery_evidence_state_accepts_verified_certification():
    inventory_state = _verified_inventory_state(
        manifest=replace(_shared_manifest(), trust="recovery_verified")
    )
    certification_state = RestartPlanCertificationState(
        inventory_state=inventory_state,
        certifications=(_certification(inventory_state),),
    )

    state = RestartPlanRecoveryEvidenceState(
        copy_state=RestartPlanCopyEligibilityState(inventory_state),
        trust_state=certification_state,
    )

    assert state.plan == inventory_state.plan
    assert state.manifest == inventory_state.manifest


def test_restart_plan_recovery_evidence_state_requires_exact_types():
    copy_state = _copy_eligibility_state()
    certification_state = _certification_state()

    with pytest.raises(TypeError, match="copy_state must be"):
        RestartPlanRecoveryEvidenceState(
            copy_state=copy_state.inventory_state,
            trust_state=certification_state,
        )
    with pytest.raises(TypeError, match="trust_state must be"):
        RestartPlanRecoveryEvidenceState(
            copy_state=copy_state,
            trust_state=copy_state,
        )


def test_restart_plan_recovery_evidence_state_rejects_cross_inventory_evidence():
    with pytest.raises(ValueError, match="same inventory state"):
        RestartPlanRecoveryEvidenceState(
            copy_state=_copy_eligibility_state(),
            trust_state=_certification_state(),
        )


def _candidate_state() -> RestartPlanCandidateState:
    inventory_state = _verified_inventory_state(
        manifest=replace(_shared_manifest(), trust="recovery_verified")
    )
    recovery_state = RestartPlanRecoveryEvidenceState(
        copy_state=RestartPlanCopyEligibilityState(inventory_state),
        trust_state=RestartPlanCertificationState(
            inventory_state=inventory_state,
            certifications=(_certification(inventory_state),),
        ),
    )
    generation_state = inventory_state.quarantine_state.manifest_state.generation_state
    return RestartPlanCandidateState(
        recovery_state=recovery_state,
        placement_state=_placement_state(generation_state=generation_state),
    )


def test_restart_plan_candidate_state_composes_matching_evidence_and_placement():
    state = _candidate_state()

    assert state.plan == state.recovery_state.plan
    assert state.manifest == state.recovery_state.manifest


def test_restart_plan_candidate_state_requires_exact_types():
    state = _candidate_state()

    with pytest.raises(TypeError, match="recovery_state must be"):
        replace(state, recovery_state=state.recovery_state.copy_state)
    with pytest.raises(TypeError, match="placement_state must be"):
        replace(state, placement_state=state.placement_state.generation_state)


def test_restart_plan_candidate_state_rejects_cross_plan_composition():
    state = _candidate_state()

    with pytest.raises(ValueError, match="same plan generation"):
        replace(state, placement_state=_placement_state())


def test_restart_plan_candidate_state_rejects_elapsed_restart_deadline():
    state = _candidate_state()
    placement = _placement_state(
        generation_state=state.placement_state.generation_state,
        registration_histories={
            "node-a": _registration_history("node-a", lease_duration_ms=2_000),
            "node-c": _registration_history("node-c", lease_duration_ms=2_000),
        },
        observed_at_unix_ms=state.plan.restart_deadline_unix_ms,
    )

    with pytest.raises(ValueError, match="deadline has elapsed"):
        replace(state, placement_state=placement)


def _publication_records() -> RestartPlanPublicationRecords:
    candidate = _candidate_state()
    inventory_state = candidate.recovery_state.copy_state.inventory_state
    quarantine_state = inventory_state.quarantine_state
    manifest_state = quarantine_state.manifest_state
    source_snapshot = replace(
        manifest_state.resolved_manifest.source_snapshot,
        record=manifest_state.generation_state.from_snapshot,
    )
    manifest_record = replace(
        manifest_state.resolved_manifest.record,
        source_generation_snapshot_digest=source_snapshot.record.digest,
    )
    generation_state = replace(
        manifest_state.generation_state,
        record=replace(
            manifest_state.generation_state.record,
            recovery_manifest_record_digest=manifest_record.digest,
        ),
    )
    manifest_state = RestartPlanManifestState(
        generation_state=generation_state,
        resolved_manifest=ResolvedRecoveryManifest(
            record=manifest_record,
            source_snapshot=source_snapshot,
        ),
    )
    quarantine_state = RestartPlanQuarantineState(
        manifest_state=manifest_state,
        quarantine_records=quarantine_state.quarantine_records,
    )
    inventory_state = RestartPlanInventoryState(
        quarantine_state=quarantine_state,
        inventory_events=inventory_state.inventory_events,
    )
    copy_state = RestartPlanCopyEligibilityState(inventory_state)
    trust_state = candidate.recovery_state.trust_state
    assert isinstance(trust_state, RestartPlanCertificationState)
    candidate = RestartPlanCandidateState(
        recovery_state=RestartPlanRecoveryEvidenceState(
            copy_state=copy_state,
            trust_state=RestartPlanCertificationState(
                inventory_state=inventory_state,
                certifications=trust_state.certifications,
            ),
        ),
        placement_state=replace(
            candidate.placement_state,
            generation_state=generation_state,
        ),
    )
    return RestartPlanPublicationRecords(
        candidate=candidate,
        current=CurrentGeneration(
            snapshot=StoredGenerationSnapshot(
                record=generation_state.from_snapshot,
                revision=8,
                committed_at_unix_ms=1_000,
                transaction_sequence=17,
                guard_mutation_sequence=9,
                guard_value_sequence=5,
                guard_lifetime_sequence=1,
                guard_committed_at_unix_ms=900,
            ),
            head_revision=18,
        ),
    )


def _persisted_publication() -> PersistedRestartPlanPublication:
    publication = _publication_records()
    generation_state = publication.candidate.placement_state.generation_state
    quarantine_state = (
        publication.candidate.recovery_state.copy_state.inventory_state.quarantine_state
    )
    manifest_record = quarantine_state.manifest_state.resolved_manifest.record
    plan_record = generation_state.record
    run_digest = hashlib.sha256(RUN_ID.encode()).hexdigest()
    guard_key = f"lm_resiliency/torchrun/v1/runs/{run_digest}/coordinator-lease"

    def immutable_entry(value: bytes, revision: int) -> ControlStoreEntry:
        return ControlStoreEntry(
            value=value,
            revision=revision,
            committed_at_unix_ms=1_200,
            transaction_sequence=30,
            guard_key=guard_key,
            guard_revision=plan_record.coordinator_fencing_token,
            guard_value_digest=plan_record.coordinator_lease_digest,
            guard_mutation_sequence=9,
            guard_value_sequence=5,
            guard_lifetime_sequence=1,
            guard_committed_at_unix_ms=900,
        )

    quarantine_entries = {
        node_id: immutable_entry(record.to_json(), 24 + index)
        for index, (node_id, record) in enumerate(quarantine_state.quarantine_records.items())
    }
    return PersistedRestartPlanPublication.from_entries(
        run_id=RUN_ID,
        plan_entry=immutable_entry(plan_record.to_json(), 21),
        manifest_entry=immutable_entry(manifest_record.to_json(), 22),
        successor_snapshot_entry=immutable_entry(
            generation_state.to_snapshot.to_json(),
            23,
        ),
        generation_head_entry=replace(
            immutable_entry(publication.generation_head.to_json(), 29),
            mutation_sequence=publication.candidate.plan.to_generation + 1,
            value_sequence=publication.candidate.plan.to_generation + 1,
        ),
        quarantine_entries=quarantine_entries,
    )


def test_persisted_restart_plan_publication_decodes_atomic_records():
    state = _persisted_publication()

    assert state.record.plan == state.plan
    assert state.generation_head.snapshot_digest == state.successor_snapshot.digest
    assert state.committed_at_unix_ms == 1_200
    assert state.transaction_sequence == 30
    assert set(state.quarantine_records) == set(state.plan.quarantined_node_ids)
    with pytest.raises(TypeError):
        state.quarantine_entries["other"] = state.plan_entry


def test_persisted_restart_plan_publication_rejects_malformed_records():
    state = _persisted_publication()

    with pytest.raises(ValueError, match="malformed records"):
        PersistedRestartPlanPublication.from_entries(
            run_id=RUN_ID,
            plan_entry=replace(state.plan_entry, value=b"{}"),
            manifest_entry=state.manifest_entry,
            successor_snapshot_entry=state.successor_snapshot_entry,
            generation_head_entry=state.generation_head_entry,
            quarantine_entries=state.quarantine_entries,
        )


def test_persisted_restart_plan_publication_binds_the_requested_run():
    state = _persisted_publication()

    with pytest.raises(ValueError, match="another run"):
        PersistedRestartPlanPublication.from_entries(
            run_id="other-run",
            plan_entry=state.plan_entry,
            manifest_entry=state.manifest_entry,
            successor_snapshot_entry=state.successor_snapshot_entry,
            generation_head_entry=state.generation_head_entry,
            quarantine_entries=state.quarantine_entries,
        )
    with pytest.raises(ValueError, match="non-empty"):
        PersistedRestartPlanPublication.from_entries(
            run_id="",
            plan_entry=state.plan_entry,
            manifest_entry=state.manifest_entry,
            successor_snapshot_entry=state.successor_snapshot_entry,
            generation_head_entry=state.generation_head_entry,
            quarantine_entries=state.quarantine_entries,
        )


def test_persisted_restart_plan_publication_rejects_digest_substitution():
    state = _persisted_publication()

    with pytest.raises(ValueError, match="manifest digest"):
        replace(
            state,
            manifest_record=replace(
                state.manifest_record,
                source_generation_snapshot_digest="f" * 64,
            ),
        )
    with pytest.raises(ValueError, match="successor digest"):
        replace(
            state,
            successor_snapshot=replace(
                state.successor_snapshot,
                coordinator_fencing_token=10,
            ),
        )
    with pytest.raises(ValueError, match="quarantine records"):
        replace(state, quarantine_records={})


def test_persisted_restart_plan_publication_rejects_manifest_metadata_substitution():
    state = _persisted_publication()
    manifest_record = replace(
        state.manifest_record,
        manifest=replace(
            state.manifest_record.manifest,
            manifest_id="other-manifest",
        ),
    )
    plan_record = replace(
        state.record,
        recovery_manifest_record_digest=manifest_record.digest,
    )

    with pytest.raises(ValueError, match="manifest metadata"):
        replace(
            state,
            record=plan_record,
            manifest_record=manifest_record,
            plan_entry=replace(state.plan_entry, value=plan_record.to_json()),
            manifest_entry=replace(
                state.manifest_entry,
                value=manifest_record.to_json(),
            ),
        )


def test_persisted_restart_plan_publication_rejects_weaker_manifest_trust():
    state = _persisted_publication()
    manifest_record = replace(
        state.manifest_record,
        manifest=replace(state.manifest_record.manifest, trust="latest"),
    )
    plan_record = replace(
        state.record,
        plan=replace(state.plan, recovery_mode="recovery_verified"),
        recovery_manifest_record_digest=manifest_record.digest,
    )

    with pytest.raises(ValueError, match="verified recovery"):
        replace(
            state,
            record=plan_record,
            manifest_record=manifest_record,
            plan_entry=replace(state.plan_entry, value=plan_record.to_json()),
            manifest_entry=replace(
                state.manifest_entry,
                value=manifest_record.to_json(),
            ),
        )


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda record: replace(record, plan_id="other-plan"), "does not match its plan"),
        (
            lambda record: replace(record, lease_id="other-lease"),
            "publication authority",
        ),
    ],
)
def test_persisted_restart_plan_publication_rejects_quarantine_substitution(
    change: Callable[[NodeQuarantineRecord], NodeQuarantineRecord],
    message: str,
) -> None:
    state = _persisted_publication()
    node_id, quarantine_record = next(iter(state.quarantine_records.items()))
    changed_record = change(quarantine_record)
    plan_record = replace(
        state.record,
        quarantine_record_digests={node_id: changed_record.digest},
    )

    with pytest.raises(ValueError, match=message):
        replace(
            state,
            record=plan_record,
            quarantine_records={node_id: changed_record},
            plan_entry=replace(state.plan_entry, value=plan_record.to_json()),
            quarantine_entries={
                node_id: replace(
                    state.quarantine_entries[node_id],
                    value=changed_record.to_json(),
                )
            },
        )


def test_persisted_restart_plan_publication_requires_the_planned_successor():
    state = _persisted_publication()
    slot_assignments = state.plan.slot_assignments
    reversed_nodes = tuple(assignment.node_id for assignment in reversed(slot_assignments))
    changed_assignment = RankAssignment.from_assignments(
        run_id=state.plan.run_id,
        generation=state.plan.to_generation,
        assignments=tuple(
            replace(assignment, node_id=node_id)
            for assignment, node_id in zip(
                slot_assignments,
                reversed_nodes,
                strict=True,
            )
        ),
        topology_digest=state.plan.topology_digest,
    )
    assert changed_assignment != state.successor_snapshot.assignment

    for successor in (
        replace(state.successor_snapshot, assignment=changed_assignment),
        replace(
            state.successor_snapshot,
            previous_snapshot_digest="e" * 64,
        ),
    ):
        record = replace(
            state.record,
            to_generation_snapshot_digest=successor.digest,
        )
        with pytest.raises(ValueError, match="successor does not implement"):
            replace(
                state,
                record=record,
                successor_snapshot=successor,
                plan_entry=replace(state.plan_entry, value=record.to_json()),
                successor_snapshot_entry=replace(
                    state.successor_snapshot_entry,
                    value=successor.to_json(),
                ),
            )


@pytest.mark.parametrize(
    ("entry_name", "change", "message"),
    [
        (
            "plan_entry",
            lambda entry: replace(entry, transaction_sequence=31),
            "share one transaction",
        ),
        (
            "manifest_entry",
            lambda entry: replace(entry, committed_at_unix_ms=1_201),
            "share one commit time",
        ),
        (
            "successor_snapshot_entry",
            lambda entry: replace(entry, guard_value_digest="0" * 64),
            "share guard provenance",
        ),
        (
            "plan_entry",
            lambda entry: replace(
                entry,
                mutation_sequence=2,
                value_sequence=2,
            ),
            "is not immutable",
        ),
        (
            "generation_head_entry",
            lambda entry: replace(
                entry,
                mutation_sequence=7,
                value_sequence=7,
            ),
            "generation head has invalid lineage",
        ),
    ],
)
def test_persisted_restart_plan_publication_rejects_entry_substitution(
    entry_name: str,
    change: Callable[[ControlStoreEntry], ControlStoreEntry],
    message: str,
) -> None:
    state = _persisted_publication()

    with pytest.raises(ValueError, match=message):
        replace(
            state,
            **cast(Any, {entry_name: change(getattr(state, entry_name))}),
        )


def test_persisted_restart_plan_publication_rejects_guard_or_time_substitution():
    state = _persisted_publication()
    changed_guard = {
        node_id: replace(entry, guard_revision=10)
        for node_id, entry in state.quarantine_entries.items()
    }

    with pytest.raises(ValueError, match="share guard provenance"):
        replace(state, quarantine_entries=changed_guard)
    wrong_guard_key = "lm_resiliency/torchrun/v1/runs/other/coordinator-lease"
    with pytest.raises(ValueError, match="invalid coordinator guard provenance"):
        replace(
            state,
            plan_entry=replace(state.plan_entry, guard_key=wrong_guard_key),
            manifest_entry=replace(state.manifest_entry, guard_key=wrong_guard_key),
            successor_snapshot_entry=replace(
                state.successor_snapshot_entry,
                guard_key=wrong_guard_key,
            ),
            generation_head_entry=replace(
                state.generation_head_entry,
                guard_key=wrong_guard_key,
            ),
            quarantine_entries={
                node_id: replace(entry, guard_key=wrong_guard_key)
                for node_id, entry in state.quarantine_entries.items()
            },
        )
    with pytest.raises(ValueError, match="authority window"):
        replace(
            state,
            plan_entry=replace(state.plan_entry, committed_at_unix_ms=1_400),
            manifest_entry=replace(state.manifest_entry, committed_at_unix_ms=1_400),
            successor_snapshot_entry=replace(
                state.successor_snapshot_entry,
                committed_at_unix_ms=1_400,
            ),
            generation_head_entry=replace(
                state.generation_head_entry,
                committed_at_unix_ms=1_400,
            ),
            quarantine_entries={
                node_id: replace(entry, committed_at_unix_ms=1_400)
                for node_id, entry in state.quarantine_entries.items()
            },
        )


def test_persisted_restart_plan_publication_requires_exact_types():
    state = _persisted_publication()

    with pytest.raises(TypeError, match="record must be"):
        replace(state, record=state.record.plan)
    with pytest.raises(TypeError, match="quarantine_entries must be a mapping"):
        replace(state, quarantine_entries=())


def test_restart_plan_publication_records_build_canonical_atomic_inputs():
    records = _publication_records()
    generation_state = records.candidate.placement_state.generation_state
    quarantine_state = records.candidate.recovery_state.copy_state.inventory_state.quarantine_state
    manifest_record = quarantine_state.manifest_state.resolved_manifest.record

    assert records.generation_head == GenerationHeadRecord(
        run_id=RUN_ID,
        generation=records.candidate.plan.to_generation,
        snapshot_digest=generation_state.to_snapshot.digest,
    )
    assert records.writes[records.generation_head_key] == ControlStoreWrite(
        expected_revision=18,
        value=records.generation_head.to_json(),
    )
    assert records.writes[records.successor_generation_snapshot_key] == ControlStoreWrite(
        expected_revision=None,
        value=generation_state.to_snapshot.to_json(),
        require_never_created=True,
    )
    assert records.writes[records.recovery_manifest_key] == ControlStoreWrite(
        expected_revision=None,
        value=manifest_record.to_json(),
        require_never_created=True,
    )
    assert records.writes[records.plan_key] == ControlStoreWrite(
        expected_revision=None,
        value=generation_state.record.to_json(),
        require_never_created=True,
    )
    quarantine_record = quarantine_state.quarantine_records["node-b"]
    assert records.writes[records.quarantine_keys["node-b"]] == ControlStoreWrite(
        expected_revision=None,
        value=quarantine_record.to_json(),
        require_never_created=True,
    )
    assert records.conditions == {
        records.source_generation_snapshot_key: 8,
        records.registration_keys["node-a"]: 11,
        records.registration_keys["node-c"]: 11,
    }
    assert records.deadline_unix_ms == 1_600


def test_restart_plan_publication_records_derive_run_scoped_keys():
    records = _publication_records()
    expected_run_digest = hashlib.sha256(RUN_ID.encode()).hexdigest()

    assert records.run_prefix == f"lm_resiliency/torchrun/v1/runs/{expected_run_digest}"
    assert records.plan_key.endswith("/restart-plans/5")
    assert records.recovery_manifest_key.endswith("/restart-plans/5/recovery-manifest")
    assert records.generation_head_key.endswith("/generation-head")
    assert records.source_generation_snapshot_key.endswith("/generations/4")
    assert records.manifest_source_generation_snapshot_key.endswith("/generations/4")
    assert records.successor_generation_snapshot_key.endswith("/generations/5")
    assert records.registration_keys == {
        "node-a": agent_registration_key(RUN_ID, "node-a"),
        "node-c": agent_registration_key(RUN_ID, "node-c"),
    }
    assert RUN_ID not in records.plan_key


def test_restart_plan_publication_records_freeze_mappings():
    records = _publication_records()

    with pytest.raises(TypeError):
        records.writes["other"] = next(iter(records.writes.values()))
    with pytest.raises(TypeError):
        records.conditions["other"] = 1
    with pytest.raises(TypeError):
        records.quarantine_keys["node-b"] = "other"
    with pytest.raises(TypeError):
        records.registration_keys["node-a"] = "other"


def test_restart_plan_publication_records_require_exact_types():
    records = _publication_records()

    with pytest.raises(TypeError, match="candidate must be"):
        replace(records, candidate=records.candidate.recovery_state)
    with pytest.raises(TypeError, match="current must be"):
        replace(records, current=records.current.snapshot)


def test_restart_plan_publication_records_require_exact_current_generation():
    records = _publication_records()
    changed_snapshot = replace(
        records.current.snapshot,
        record=replace(
            records.current.snapshot.record,
            coordinator_fencing_token=10,
        ),
    )

    with pytest.raises(ValueError, match="current generation does not match"):
        replace(
            records,
            current=replace(records.current, snapshot=changed_snapshot),
        )


def test_restart_plan_publication_records_require_matching_shared_source_revision():
    records = _publication_records()

    with pytest.raises(ValueError, match="source snapshots disagree"):
        replace(
            records,
            current=replace(
                records.current,
                snapshot=replace(records.current.snapshot, revision=9),
            ),
        )


def test_restart_plan_publication_records_reject_divergent_shared_source_record():
    candidate = _candidate_state()

    with pytest.raises(ValueError, match="source snapshots disagree"):
        RestartPlanPublicationRecords(
            candidate=candidate,
            current=CurrentGeneration(
                snapshot=replace(
                    _publication_records().current.snapshot,
                    record=candidate.placement_state.generation_state.from_snapshot,
                ),
                head_revision=18,
            ),
        )


def test_restart_plan_publication_records_use_earliest_registration_expiry():
    records = _publication_records()
    generation_state = records.candidate.placement_state.generation_state
    placement = _placement_state(
        generation_state=generation_state,
        registration_histories={
            "node-a": _registration_history(
                "node-a",
                lease_duration_ms=200,
            ),
            "node-c": _registration_history("node-c"),
        },
    )

    updated = replace(
        records,
        candidate=replace(records.candidate, placement_state=placement),
    )

    assert updated.deadline_unix_ms == 1_300


def _publication_authority() -> RestartPlanPublicationAuthority:
    records = _publication_records()
    plan_record = records.candidate.placement_state.generation_state.record
    return RestartPlanPublicationAuthority(
        records=records,
        coordinator_authority=CoordinatorLeaseAuthority(
            lease=HeldCoordinatorLease(
                record=CoordinatorLeaseRecord(
                    run_id=RUN_ID,
                    coordinator_id=plan_record.coordinator_id,
                    lease_id=plan_record.lease_id,
                    lease_duration_ms=plan_record.coordinator_lease_duration_ms,
                ),
                fencing_token=plan_record.coordinator_fencing_token,
                granted_at_unix_ms=900,
            ),
            transaction_sequence=9,
            mutation_sequence=9,
            value_sequence=5,
            lifetime_sequence=1,
        ),
        observed_at_unix_ms=1_200,
    )


def test_restart_plan_publication_records_binds_exact_lease_window():
    authority = _publication_authority()

    assert authority.not_before_unix_ms == 1_200
    assert authority.deadline_unix_ms == 1_400


def test_restart_plan_publication_records_is_immutable():
    authority = _publication_authority()

    with pytest.raises(AttributeError):
        authority.observed_at_unix_ms = 1_201


def test_restart_plan_publication_records_requires_exact_types():
    authority = _publication_authority()

    with pytest.raises(TypeError, match="records must be"):
        replace(authority, records=authority.records.candidate)
    with pytest.raises(TypeError, match="coordinator_authority must be"):
        replace(
            authority,
            coordinator_authority=authority.coordinator_authority.lease,
        )


@pytest.mark.parametrize("observed_at_unix_ms", [0, True])
def test_restart_plan_publication_records_requires_positive_observation(
    observed_at_unix_ms,
):
    authority = _publication_authority()

    with pytest.raises(ValueError, match="observed_at_unix_ms"):
        replace(authority, observed_at_unix_ms=observed_at_unix_ms)


@pytest.mark.parametrize(
    "lease",
    [
        lambda authority: replace(
            authority.coordinator_authority.lease,
            record=replace(
                authority.coordinator_authority.lease.record,
                run_id="other-run",
            ),
        ),
        lambda authority: replace(
            authority.coordinator_authority.lease,
            record=replace(
                authority.coordinator_authority.lease.record,
                coordinator_id="other-coordinator",
            ),
        ),
        lambda authority: replace(
            authority.coordinator_authority.lease,
            record=replace(
                authority.coordinator_authority.lease.record,
                lease_id="other-lease",
            ),
        ),
        lambda authority: replace(
            authority.coordinator_authority.lease,
            record=replace(
                authority.coordinator_authority.lease.record,
                lease_duration_ms=501,
            ),
        ),
        lambda authority: replace(
            authority.coordinator_authority.lease,
            fencing_token=10,
        ),
    ],
)
def test_restart_plan_publication_records_rejects_wrong_lease(lease):
    authority = _publication_authority()

    with pytest.raises(ValueError, match="does not authorize"):
        replace(
            authority,
            coordinator_authority=replace(
                authority.coordinator_authority,
                lease=lease(authority),
            ),
        )


@pytest.mark.parametrize("observed_at_unix_ms", [899, 999, 1_199])
def test_restart_plan_publication_records_rejects_early_observation(
    observed_at_unix_ms,
):
    authority = _publication_authority()

    with pytest.raises(ValueError, match="precedes one of its inputs"):
        replace(authority, observed_at_unix_ms=observed_at_unix_ms)


def test_restart_plan_publication_records_rejects_elapsed_window():
    authority = _publication_authority()

    with pytest.raises(ValueError, match="window has elapsed"):
        replace(authority, observed_at_unix_ms=authority.deadline_unix_ms)


def _publication_preparation_state(
    *,
    lease_writes: int = 9,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    RestartPlanPublicationRecords,
]:
    records = _publication_records()
    plan_record = records.candidate.placement_state.generation_state.record
    clock = ManualClock(900)
    store = InMemoryControlStore(clock=clock)
    lease_key = CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).lease_key
    lease_record = CoordinatorLeaseRecord(
        run_id=RUN_ID,
        coordinator_id=plan_record.coordinator_id,
        lease_id=plan_record.lease_id,
        lease_duration_ms=plan_record.coordinator_lease_duration_ms,
    )
    expected_revision = None
    for _ in range(lease_writes):
        entry = store.compare_set_in_window(
            lease_key,
            expected_revision=expected_revision,
            not_before_unix_ms=900,
            deadline_unix_ms=1_400,
            value=lease_record.to_json(),
        )
        expected_revision = entry.revision
    clock.set(1_200)
    return clock, store, records


def test_restart_plan_publication_preparer_authenticates_without_mutation():
    clock, store, records = _publication_preparation_state()
    lease_key = CoordinatorLeaseHistoryReader(store, run_id=RUN_ID).lease_key
    history_before = store.get_history(lease_key)

    authority = RestartPlanPublicationAuthorityPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare(records)

    assert authority.records == records
    assert authority.coordinator_authority.lease.fencing_token == 9
    assert authority.observed_at_unix_ms == 1_200
    assert authority.deadline_unix_ms == 1_400
    assert store.get_history(lease_key) == history_before


def test_restart_plan_publication_preparer_rejects_missing_or_stale_lease():
    clock = ManualClock(1_200)
    store = InMemoryControlStore(clock=clock)
    records = _publication_records()

    with pytest.raises(RestartPlanPublicationPreparationLeaseLost, match="no live"):
        RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(records)

    clock, store, records = _publication_preparation_state(lease_writes=8)
    with pytest.raises(RestartPlanPublicationPreparationLeaseLost, match="not authorized"):
        RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(records)


def test_restart_plan_publication_preparer_rejects_elapsed_or_unsafe_clock():
    clock, store, records = _publication_preparation_state()
    clock.set(1_400)
    with pytest.raises(RestartPlanPublicationPreparationLeaseLost, match="window elapsed"):
        RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(records)

    clock, store, records = _publication_preparation_state()
    preparation_clock = ManualClock(1_199)
    preparer = RestartPlanPublicationAuthorityPreparer(
        store,
        run_id=RUN_ID,
        clock=preparation_clock,
    )
    with pytest.raises(RestartPlanPublicationPreparationClockError, match="precedes"):
        preparer.prepare(records)

    preparation_clock.set(1_200)
    preparer.prepare(records)
    preparation_clock.set(1_199)
    with pytest.raises(RestartPlanPublicationPreparationClockError, match="moved backward"):
        preparer.prepare(records)


def test_restart_plan_publication_preparer_translates_history_failures():
    clock, store, records = _publication_preparation_state()
    preparer = RestartPlanPublicationAuthorityPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )
    cast(Any, preparer)._lease_history_reader = FailingLeaseHistoryReader(
        CoordinatorLeaseHistoryError("changed repeatedly")
    )
    with pytest.raises(RestartPlanPublicationPreparationConflict, match="changed repeatedly"):
        preparer.prepare(records)

    cast(Any, preparer)._lease_history_reader = FailingLeaseHistoryReader(
        CoordinatorLeaseHistoryCorrupt("malformed")
    )
    with pytest.raises(RestartPlanPublicationPreparationCorrupt, match="history is corrupt"):
        preparer.prepare(records)


def test_restart_plan_publication_preparer_requires_exact_inputs():
    clock, store, records = _publication_preparation_state()
    preparer = RestartPlanPublicationAuthorityPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    )

    with pytest.raises(TypeError, match="records must be"):
        preparer.prepare(records.candidate)
    with pytest.raises(ValueError, match="another run"):
        RestartPlanPublicationAuthorityPreparer(
            store,
            run_id="other-run",
            clock=clock,
        ).prepare(records)
    with pytest.raises(TypeError, match="clock must be callable"):
        RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=RUN_ID,
            clock=cast(Any, None),
        )


@pytest.mark.parametrize(
    "current",
    [
        lambda records: replace(records.current, head_revision=0),
        lambda records: replace(
            records.current,
            snapshot=replace(records.current.snapshot, revision=0),
        ),
    ],
)
def test_restart_plan_publication_records_require_positive_revisions(current):
    records = _publication_records()

    with pytest.raises(ValueError, match="positive integer"):
        replace(records, current=current(records))


def _verified_inventory_state(
    *,
    manifest: RecoveryManifest | None = None,
) -> RestartPlanInventoryState:
    selected_manifest = manifest or replace(_manifest(), trust="recovery_verified")
    plan = replace(_plan(), recovery_mode="recovery_verified")
    quarantine_state = _quarantine_state(
        manifest_state=_manifest_state(
            manifest=selected_manifest,
            plan=plan,
        )
    )
    return _inventory_state(quarantine_state=quarantine_state)


def _certification(
    state: RestartPlanInventoryState,
    *,
    certification_id: str = "certification-40",
    inventory_event_digests: dict[str, str] | None = None,
) -> CheckpointCertification:
    return CheckpointCertification(
        certification_id=certification_id,
        run_id=state.manifest.run_id,
        source_generation=state.manifest.source_generation,
        step=state.manifest.step,
        topology_digest=state.manifest.topology_digest,
        checkpoint_source=state.plan.checkpoint_source,
        checkpoint_id=state.plan.checkpoint_id,
        expected_world_size=state.plan.expected_world_size,
        certification_kind="dense_consensus",
        inventory_event_digests=inventory_event_digests
        or {
            event_id: checkpoint_inventory_digest(event)
            for event_id, event in state.inventory_events.items()
        },
    )


def _certification_state() -> RestartPlanCertificationState:
    inventory_state = _verified_inventory_state()
    return RestartPlanCertificationState(
        inventory_state=inventory_state,
        certifications=(_certification(inventory_state),),
    )


def test_restart_plan_certification_state_binds_exact_certifications():
    state = _certification_state()

    assert state.plan == state.inventory_state.plan
    assert state.manifest == state.inventory_state.manifest
    assert tuple(cert.certification_id for cert in state.certifications) == ("certification-40",)


def test_restart_plan_certification_state_requires_exact_types():
    state = _certification_state()

    with pytest.raises(TypeError, match="inventory_state must be"):
        replace(state, inventory_state=state.inventory_state.quarantine_state)

    with pytest.raises(TypeError, match="must be a tuple"):
        replace(state, certifications=list(state.certifications))

    with pytest.raises(TypeError, match="must be CheckpointCertification"):
        replace(state, certifications=(state.certifications[0].to_dict(),))


def test_restart_plan_certification_state_requires_verified_manifest():
    state = _inventory_state()

    with pytest.raises(ValueError, match="requires a recovery-verified manifest"):
        RestartPlanCertificationState(
            inventory_state=state,
            certifications=(),
        )


def test_restart_plan_certification_state_sorts_and_rejects_duplicate_ids():
    inventory_state = _verified_inventory_state()
    first = _certification(
        inventory_state,
        certification_id="certification-b",
    )
    second = _certification(
        inventory_state,
        certification_id="certification-a",
    )
    state = RestartPlanCertificationState(
        inventory_state=inventory_state,
        certifications=(first, second),
    )

    assert tuple(cert.certification_id for cert in state.certifications) == (
        "certification-a",
        "certification-b",
    )

    with pytest.raises(ValueError, match="IDs must be unique"):
        replace(state, certifications=(first, first))


@pytest.mark.parametrize(
    "certification",
    [
        replace(_certification(_verified_inventory_state()), run_id="other-run"),
        replace(
            _certification(_verified_inventory_state()),
            source_generation=3,
        ),
        replace(_certification(_verified_inventory_state()), step=39),
        replace(
            _certification(_verified_inventory_state()),
            topology_digest="topology-v2",
        ),
        replace(
            _certification(_verified_inventory_state()),
            checkpoint_source="durable",
            checkpoint_id="durable-40",
        ),
        replace(
            _certification(_verified_inventory_state()),
            expected_world_size=8,
        ),
    ],
)
def test_restart_plan_certification_state_rejects_metadata_mismatch(certification):
    with pytest.raises(ValueError, match="does not match its plan"):
        RestartPlanCertificationState(
            inventory_state=_verified_inventory_state(),
            certifications=(certification,),
        )


def test_restart_plan_certification_state_requires_every_event_digest():
    inventory_state = _verified_inventory_state()
    digests = {
        event_id: checkpoint_inventory_digest(event)
        for event_id, event in inventory_state.inventory_events.items()
    }
    digests.pop("inventory-3")

    with pytest.raises(ValueError, match="not exactly certified"):
        RestartPlanCertificationState(
            inventory_state=inventory_state,
            certifications=(
                _certification(
                    inventory_state,
                    inventory_event_digests=digests,
                ),
            ),
        )


def test_restart_plan_certification_state_rejects_wrong_event_digest():
    inventory_state = _verified_inventory_state()
    digests = {
        event_id: checkpoint_inventory_digest(event)
        for event_id, event in inventory_state.inventory_events.items()
    }
    digests["inventory-0"] = "0" * 64

    with pytest.raises(ValueError, match="not exactly certified"):
        RestartPlanCertificationState(
            inventory_state=inventory_state,
            certifications=(
                _certification(
                    inventory_state,
                    inventory_event_digests=digests,
                ),
            ),
        )


def test_restart_plan_certification_state_rejects_conflicting_event_digests():
    inventory_state = _verified_inventory_state()
    first = _certification(
        inventory_state,
        certification_id="certification-a",
    )
    second_digests = dict(first.inventory_event_digests)
    second_digests["inventory-0"] = "0" * 64
    second = _certification(
        inventory_state,
        certification_id="certification-b",
        inventory_event_digests=second_digests,
    )

    with pytest.raises(ValueError, match="conflicting inventory digests"):
        RestartPlanCertificationState(
            inventory_state=inventory_state,
            certifications=(first, second),
        )


def test_restart_plan_certification_state_allows_certification_across_records():
    inventory_state = _verified_inventory_state()
    event_ids = tuple(inventory_state.inventory_events)
    first = _certification(
        inventory_state,
        certification_id="certification-a",
        inventory_event_digests={
            event_id: checkpoint_inventory_digest(inventory_state.inventory_events[event_id])
            for event_id in event_ids[:2]
        },
    )
    second = _certification(
        inventory_state,
        certification_id="certification-b",
        inventory_event_digests={
            event_id: checkpoint_inventory_digest(inventory_state.inventory_events[event_id])
            for event_id in event_ids[2:]
        },
    )

    state = RestartPlanCertificationState(
        inventory_state=inventory_state,
        certifications=(second, first),
    )

    assert len(state.certifications) == 2


def test_restart_plan_certification_state_does_not_claim_copy_eligibility():
    manifest = replace(_manifest(), trust="recovery_verified")
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
    inventory_state = _verified_inventory_state(manifest=incomplete_manifest)

    state = RestartPlanCertificationState(
        inventory_state=inventory_state,
        certifications=(_certification(inventory_state),),
    )

    assert not state.manifest.rank_copies[0].copies[0].complete
