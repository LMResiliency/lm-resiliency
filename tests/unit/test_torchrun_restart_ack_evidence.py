"""Contract tests for latest-checkpoint restart-acknowledgement evidence."""

from __future__ import annotations

import hashlib
import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    CheckpointCopy,
    CheckpointInventoryEvent,
    RankAssignment,
    RankCheckpointCopies,
    RecoveryManifest,
    RestartAck,
    RestartIntent,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
    checkpoint_inventory_digest,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckCollection,
    RestartAckEvidence,
)
from lm_resiliency.integrations.torchrun._restart_ack_execution import (
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_reader import RestartAckReader
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    RestartIntentOpenExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentLifecycleRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanEvidenceRecord,
    RestartPlanRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    PersistedRestartPlanPublication,
    ResolvedRecoveryManifest,
    RestartPlanCopyEligibilityState,
    RestartPlanGenerationState,
    RestartPlanInventoryState,
    RestartPlanLatestEvidenceState,
    RestartPlanManifestState,
    RestartPlanPersistedRecoveryState,
    RestartPlanQuarantineState,
    RestartPlanRecoveryEvidenceState,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms


def _event(
    *,
    node_id: str = "node-a",
    agent_id: str = "agent-node-a",
    logical_node_slot: int = 0,
    global_rank: int = 0,
    local_rank: int = 0,
    generation: int = 0,
    step: int = 40,
    trust: str = "latest",
    event_id: str = "inventory-a",
    copies: tuple[CheckpointCopy, ...] = (),
) -> CheckpointInventoryEvent:
    return CheckpointInventoryEvent(
        event_id=event_id,
        run_id=RUN_ID,
        generation=generation,
        reporter=WorkerIdentity(
            run_id=RUN_ID,
            generation=generation,
            node_id=node_id,
            agent_id=agent_id,
            logical_node_slot=logical_node_slot,
            global_rank=global_rank,
            local_rank=local_rank,
            local_world_size=2,
            hostname=f"host-{node_id}",
            gpu_uuid=f"gpu-{node_id}-{local_rank}",
            topology_digest="topology-v1",
        ),
        step=step,
        trust=trust,
        topology_digest="topology-v1",
        copies=copies,
    )


def _evidence(
    *,
    acknowledgement_success: bool | None = True,
    event: CheckpointInventoryEvent | None = None,
    intent_id: str = "intent-a",
    generation: int = 0,
) -> tuple[RestartAckEvidence, CheckpointInventoryEvent]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=1_000,
        clock=clock,
    ).acquire()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=generation,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )
    current = GenerationStateManager(store, run_id=RUN_ID).initialize(
        lease,
        assignment,
    )
    intent = RestartIntent(
        intent_id=intent_id,
        run_id=RUN_ID,
        generation=generation,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="latest",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=1_500,
    )
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_open(lease, current, intent)
    )
    event = event or _event(generation=generation)
    persisted = None
    if acknowledgement_success is not None:
        registration = AgentRegistrationManager(
            store,
            agent_identity=AgentIdentity(
                run_id=RUN_ID,
                node_id="node-a",
                agent_id="agent-node-a",
                hostname="host-node-a",
                local_world_size=2,
                resource_ids=("gpu-node-a-0", "gpu-node-a-1"),
                environment_digest="environment-v1",
            ),
            lease_duration_ms=400,
            clock=clock,
        ).register()
        receipt = RestartAckReceiptRecord(
            acknowledgement=RestartAck(
                intent_id=intent.intent_id,
                run_id=RUN_ID,
                node_id="node-a",
                agent_id="agent-node-a",
                generation=generation,
                flushed_step=40 if acknowledgement_success else -1,
                inventory_event_digests={event.event_id: checkpoint_inventory_digest(event)},
                transferred_owner_ranks=(0, 1) if acknowledgement_success else (),
                transferred_peer_ranks=(2, 3) if acknowledgement_success else (),
                success=acknowledgement_success,
                reason="prepared" if acknowledgement_success else "preparation failed",
            ),
            intent_record=opened.prepared.record,
            agent_registration=registration.record,
            registration_fencing_token=registration.fencing_token,
            registration_granted_at_unix_ms=registration.granted_at_unix_ms,
            received_at_unix_ms=clock.now_unix_ms,
        )
        prepared = RestartAckPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare(receipt, lease)
        RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)
        persisted = RestartAckReader(
            store,
            run_id=RUN_ID,
            node_id="node-a",
        ).read()
        assert persisted is not None
    collection = RestartAckCollection(
        opened=opened,
        receipts_by_node_id={
            "node-a": persisted,
            "node-b": None,
        },
    )
    return RestartAckEvidence(collection), event


def _latest_event(*, complete: bool = True) -> CheckpointInventoryEvent:
    copies = tuple(
        CheckpointCopy(
            owner_global_rank=rank,
            checkpoint_step=40,
            inventory_event_id="inventory-a",
            checkpoint_id=None,
            holder_node_id="node-a" if rank < 2 else "node-b",
            holder_kind="owner",
            storage_kind="shared",
            location_token=f"shared-copy-{rank}",
            complete=complete,
            checksums_available=True,
        )
        for rank in range(4)
    )
    return _event(copies=copies)


def _latest_inventory_state(
    evidence: RestartAckEvidence,
    event: CheckpointInventoryEvent,
) -> RestartPlanInventoryState:
    opened = evidence.collection.opened
    intent_record = opened.prepared.record
    intent = intent_record.intent
    from_snapshot = opened.prepared.current.snapshot.record
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=RestartIntentHeadRecord(
            run_id=intent.run_id,
            generation=intent.generation,
            intent_id=intent.intent_id,
            intent_digest=intent_record.digest,
        ),
        coordinator_id="coordinator-close",
        lease_id="lease-close",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=3,
    )
    slot_assignments = (
        SlotAssignment(0, "node-a", 0, 2),
        SlotAssignment(1, "node-c", 2, 2),
    )
    plan = RestartPlan(
        plan_id="plan-1",
        intent_id=intent.intent_id,
        run_id=intent.run_id,
        from_generation=intent.generation,
        to_generation=intent.generation + 1,
        incident_ids=intent.incident_ids,
        reason_code=intent.reason_code,
        recovery_mode="latest",
        checkpoint_source="gemini",
        checkpoint_step=event.step,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-40",
        slot_assignments=slot_assignments,
        quarantined_node_ids=(),
        expected_world_size=4,
        topology_digest=event.topology_digest,
        restart_deadline_unix_ms=2_000,
    )
    manifest = RecoveryManifest(
        manifest_id=plan.checkpoint_manifest_id,
        run_id=plan.run_id,
        source_generation=plan.from_generation,
        step=plan.checkpoint_step,
        trust="latest",
        topology_digest=plan.topology_digest,
        rank_copies=tuple(
            RankCheckpointCopies(
                owner_global_rank=rank,
                copies=tuple(copy for copy in event.copies if copy.owner_global_rank == rank),
            )
            for rank in range(plan.expected_world_size)
        ),
    )
    manifest_record = RecoveryManifestRecord(
        manifest=manifest,
        source_generation_snapshot_digest=from_snapshot.digest,
    )
    to_assignment = RankAssignment.from_assignments(
        run_id=plan.run_id,
        generation=plan.to_generation,
        assignments=slot_assignments,
        topology_digest=plan.topology_digest,
    )
    to_snapshot = GenerationSnapshotRecord(
        assignment=to_assignment,
        previous_snapshot_digest=from_snapshot.digest,
        coordinator_id="coordinator-plan",
        lease_id="lease-plan",
        coordinator_lease_duration_ms=500,
        coordinator_fencing_token=9,
    )
    plan_record = RestartPlanRecord(
        plan=plan,
        recovery_manifest_record_digest=manifest_record.digest,
        recovery_evidence_record_digest="a" * 64,
        intent_lifecycle_record_digest=lifecycle.digest,
        from_generation_snapshot_digest=from_snapshot.digest,
        to_generation_snapshot_digest=to_snapshot.digest,
        quarantine_record_digests={},
        coordinator_id=to_snapshot.coordinator_id,
        lease_id=to_snapshot.lease_id,
        coordinator_lease_duration_ms=to_snapshot.coordinator_lease_duration_ms,
        coordinator_fencing_token=to_snapshot.coordinator_fencing_token,
    )
    generation_state = RestartPlanGenerationState(
        record=plan_record,
        intent_record=intent_record,
        lifecycle_record=lifecycle,
        from_snapshot=from_snapshot,
        to_snapshot=to_snapshot,
    )
    manifest_state = RestartPlanManifestState(
        generation_state=generation_state,
        resolved_manifest=ResolvedRecoveryManifest(
            record=manifest_record,
            source_snapshot=opened.prepared.current.snapshot,
        ),
    )
    return RestartPlanInventoryState(
        quarantine_state=RestartPlanQuarantineState(
            manifest_state=manifest_state,
            quarantine_records={},
        ),
        inventory_events={event.event_id: event},
    )


def _persisted_latest_recovery_inputs(
    evidence: RestartAckEvidence,
    event: CheckpointInventoryEvent,
) -> tuple[
    PersistedRestartPlanPublication,
    RestartPlanGenerationState,
    StoredGenerationSnapshot,
]:
    inventory_state = _latest_inventory_state(evidence, event)
    manifest_state = inventory_state.quarantine_state.manifest_state
    evidence_record = RestartPlanEvidenceRecord(
        plan_id=manifest_state.plan.plan_id,
        run_id=manifest_state.plan.run_id,
        manifest_id=manifest_state.manifest.manifest_id,
        inventory_events=inventory_state.inventory_events,
        certifications=(),
    )
    generation_state = replace(
        manifest_state.generation_state,
        record=replace(
            manifest_state.generation_state.record,
            recovery_evidence_record_digest=evidence_record.digest,
        ),
    )
    plan_record = generation_state.record
    manifest_record = manifest_state.resolved_manifest.record
    generation_head = GenerationHeadRecord(
        run_id=RUN_ID,
        generation=generation_state.plan.to_generation,
        snapshot_digest=generation_state.to_snapshot.digest,
    )
    guard_key = (
        "lm_resiliency/torchrun/v1/runs/"
        f"{hashlib.sha256(RUN_ID.encode('utf-8')).hexdigest()}/coordinator-lease"
    )

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

    publication = PersistedRestartPlanPublication.from_entries(
        run_id=RUN_ID,
        to_generation=generation_state.plan.to_generation,
        plan_entry=immutable_entry(plan_record.to_json(), 21),
        manifest_entry=immutable_entry(manifest_record.to_json(), 22),
        evidence_entry=immutable_entry(evidence_record.to_json(), 23),
        successor_snapshot_entry=immutable_entry(
            generation_state.to_snapshot.to_json(),
            24,
        ),
        generation_head_entry=replace(
            immutable_entry(generation_head.to_json(), 25),
            mutation_sequence=generation_state.plan.to_generation + 1,
            value_sequence=generation_state.plan.to_generation + 1,
        ),
        quarantine_entries={},
    )
    return (
        publication,
        generation_state,
        manifest_state.resolved_manifest.source_snapshot,
    )


def test_persisted_latest_recovery_accepts_matching_acknowledgements():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    publication, generation_state, source_snapshot = _persisted_latest_recovery_inputs(
        evidence,
        event,
    )

    state = RestartPlanPersistedRecoveryState(
        publication=publication,
        generation_state=generation_state,
        manifest_source_snapshot=source_snapshot,
        acknowledgement_evidence=evidence,
    )

    assert isinstance(state.recovery_state.trust_state, RestartPlanLatestEvidenceState)
    assert state.recovery_state.trust_state.acknowledgement_evidence == evidence


@pytest.mark.parametrize("acknowledgement_success", [None, False])
def test_persisted_latest_recovery_rejects_missing_or_failed_acknowledgements(
    acknowledgement_success,
):
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    publication, generation_state, source_snapshot = _persisted_latest_recovery_inputs(
        evidence,
        event,
    )
    persisted = evidence.collection.receipts_by_node_id["node-a"]
    assert persisted is not None
    if acknowledgement_success is None:
        unavailable_receipt = None
    else:
        failed_acknowledgement = replace(
            persisted.receipt.acknowledgement,
            flushed_step=-1,
            inventory_event_digests={},
            transferred_owner_ranks=(),
            transferred_peer_ranks=(),
            success=False,
            reason="preparation failed",
        )
        failed_receipt = replace(
            persisted.receipt,
            acknowledgement=failed_acknowledgement,
        )
        unavailable_receipt = replace(
            persisted,
            receipt=failed_receipt,
            receipt_entry=replace(
                persisted.receipt_entry,
                value=failed_receipt.to_json(),
            ),
        )
    unavailable_evidence = RestartAckEvidence(
        RestartAckCollection(
            opened=evidence.collection.opened,
            receipts_by_node_id={
                "node-a": unavailable_receipt,
                "node-b": None,
            },
        )
    )

    with pytest.raises(ValueError, match="not authorized"):
        RestartPlanPersistedRecoveryState(
            publication=publication,
            generation_state=generation_state,
            manifest_source_snapshot=source_snapshot,
            acknowledgement_evidence=unavailable_evidence,
        )


def test_persisted_latest_recovery_rejects_acknowledgements_for_another_intent():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    publication, generation_state, source_snapshot = _persisted_latest_recovery_inputs(
        evidence,
        event,
    )
    other_evidence, _ = _evidence(
        event=event,
        intent_id="intent-other",
    )

    with pytest.raises(ValueError, match="another restart intent"):
        RestartPlanPersistedRecoveryState(
            publication=publication,
            generation_state=generation_state,
            manifest_source_snapshot=source_snapshot,
            acknowledgement_evidence=other_evidence,
        )


def test_restart_ack_collection_authorizes_exact_latest_inventory():
    evidence, event = _evidence()

    assert evidence.authorizes_latest_inventory(event)


@pytest.mark.parametrize(
    "event",
    [
        _event(agent_id="agent-other"),
        _event(step=41),
        _event(event_id="inventory-other"),
        _event(logical_node_slot=1, global_rank=2),
        _event(trust="candidate"),
        _event(trust="recovery_verified"),
    ],
)
def test_restart_ack_collection_rejects_mismatched_or_nonlatest_inventory(event):
    evidence, _ = _evidence()

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_collection_rejects_reused_event_id_with_different_bytes():
    evidence, event = _evidence()
    changed = replace(event, reporter=replace(event.reporter, hostname="host-changed"))

    assert changed.event_id == event.event_id
    assert not evidence.authorizes_latest_inventory(changed)


@pytest.mark.parametrize("acknowledgement_success", [None, False])
def test_restart_ack_collection_rejects_missing_or_failed_preparation(
    acknowledgement_success,
):
    evidence, event = _evidence(
        acknowledgement_success=acknowledgement_success,
    )

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_collection_rejects_event_from_missing_node():
    evidence, _ = _evidence()
    event = _event(
        node_id="node-b",
        agent_id="agent-node-b",
        logical_node_slot=1,
        global_rank=2,
    )

    assert not evidence.authorizes_latest_inventory(event)


def test_restart_ack_collection_validates_types():
    evidence, _ = _evidence()

    with pytest.raises(TypeError, match="CheckpointInventoryEvent"):
        evidence.authorizes_latest_inventory(cast(Any, object()))
    with pytest.raises(TypeError, match="RestartAckCollection"):
        RestartAckEvidence(cast(Any, object()))


def _inventory_state_with_trust(
    state: RestartPlanInventoryState,
    *,
    trust: str,
) -> RestartPlanInventoryState:
    manifest_state = state.quarantine_state.manifest_state
    manifest = replace(manifest_state.manifest, trust=trust)
    manifest_record = replace(
        manifest_state.resolved_manifest.record,
        manifest=manifest,
    )
    plan = replace(
        manifest_state.plan,
        recovery_mode="recovery_verified" if trust == "recovery_verified" else "latest",
    )
    generation_state = replace(
        manifest_state.generation_state,
        record=replace(
            manifest_state.generation_state.record,
            plan=plan,
            recovery_manifest_record_digest=manifest_record.digest,
        ),
    )
    updated_manifest_state = RestartPlanManifestState(
        generation_state=generation_state,
        resolved_manifest=ResolvedRecoveryManifest(
            record=manifest_record,
            source_snapshot=manifest_state.resolved_manifest.source_snapshot,
        ),
    )
    event = next(iter(state.inventory_events.values()))
    return RestartPlanInventoryState(
        quarantine_state=RestartPlanQuarantineState(
            manifest_state=updated_manifest_state,
            quarantine_records={},
        ),
        inventory_events={
            event.event_id: replace(
                event,
                trust=trust,
            )
        },
    )


def test_restart_plan_latest_evidence_state_binds_exact_acknowledgements():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    inventory_state = _latest_inventory_state(evidence, event)

    state = RestartPlanLatestEvidenceState(
        inventory_state=inventory_state,
        acknowledgement_evidence=evidence,
    )

    assert state.plan == inventory_state.plan
    assert state.manifest == inventory_state.manifest
    assert state.acknowledgement_evidence.authorizes_latest_inventory(event)


def test_restart_plan_latest_evidence_state_requires_exact_types():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    state = RestartPlanLatestEvidenceState(
        inventory_state=_latest_inventory_state(evidence, event),
        acknowledgement_evidence=evidence,
    )

    with pytest.raises(TypeError, match="inventory_state must be"):
        replace(
            state,
            inventory_state=state.inventory_state.quarantine_state,
        )
    with pytest.raises(TypeError, match="acknowledgement_evidence must be"):
        replace(
            state,
            acknowledgement_evidence=state.acknowledgement_evidence.collection,
        )


def test_restart_plan_latest_evidence_state_requires_latest_manifest():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    inventory_state = _inventory_state_with_trust(
        _latest_inventory_state(evidence, event),
        trust="recovery_verified",
    )

    with pytest.raises(ValueError, match="requires a latest manifest"):
        RestartPlanLatestEvidenceState(
            inventory_state=inventory_state,
            acknowledgement_evidence=evidence,
        )


def test_restart_plan_latest_evidence_state_rejects_another_intent():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    other_evidence, _ = _evidence(
        event=event,
        intent_id="intent-other",
    )

    with pytest.raises(ValueError, match="another restart intent"):
        RestartPlanLatestEvidenceState(
            inventory_state=_latest_inventory_state(evidence, event),
            acknowledgement_evidence=other_evidence,
        )


def test_restart_plan_latest_evidence_state_requires_exact_source_snapshot():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    inventory_state = _latest_inventory_state(evidence, event)
    manifest_state = inventory_state.quarantine_state.manifest_state
    alternate_source_snapshot = replace(
        manifest_state.resolved_manifest.source_snapshot,
        record=replace(
            manifest_state.resolved_manifest.source_snapshot.record,
            coordinator_id="coordinator-other",
        ),
    )
    manifest_record = replace(
        manifest_state.resolved_manifest.record,
        source_generation_snapshot_digest=alternate_source_snapshot.record.digest,
    )
    alternate_manifest_state = RestartPlanManifestState(
        generation_state=replace(
            manifest_state.generation_state,
            record=replace(
                manifest_state.generation_state.record,
                recovery_manifest_record_digest=manifest_record.digest,
            ),
        ),
        resolved_manifest=ResolvedRecoveryManifest(
            record=manifest_record,
            source_snapshot=alternate_source_snapshot,
        ),
    )
    alternate_inventory_state = RestartPlanInventoryState(
        quarantine_state=RestartPlanQuarantineState(
            manifest_state=alternate_manifest_state,
            quarantine_records={},
        ),
        inventory_events=inventory_state.inventory_events,
    )

    with pytest.raises(ValueError, match="exact current generation snapshot"):
        RestartPlanLatestEvidenceState(
            inventory_state=alternate_inventory_state,
            acknowledgement_evidence=evidence,
        )


@pytest.mark.parametrize("acknowledgement_success", [None, False])
def test_restart_plan_latest_evidence_state_rejects_missing_or_failed_preparation(
    acknowledgement_success,
):
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    persisted = evidence.collection.receipts_by_node_id["node-a"]
    assert persisted is not None
    if acknowledgement_success is None:
        unavailable_receipt = None
    else:
        failed_acknowledgement = replace(
            persisted.receipt.acknowledgement,
            flushed_step=-1,
            inventory_event_digests={},
            transferred_owner_ranks=(),
            transferred_peer_ranks=(),
            success=False,
            reason="preparation failed",
        )
        failed_receipt = replace(
            persisted.receipt,
            acknowledgement=failed_acknowledgement,
        )
        unavailable_receipt = replace(
            persisted,
            receipt=failed_receipt,
            receipt_entry=replace(
                persisted.receipt_entry,
                value=failed_receipt.to_json(),
            ),
        )
    unavailable_evidence = RestartAckEvidence(
        RestartAckCollection(
            opened=evidence.collection.opened,
            receipts_by_node_id={
                "node-a": unavailable_receipt,
                "node-b": None,
            },
        )
    )

    with pytest.raises(ValueError, match="not authorized"):
        RestartPlanLatestEvidenceState(
            inventory_state=_latest_inventory_state(evidence, event),
            acknowledgement_evidence=unavailable_evidence,
        )


def test_restart_plan_latest_evidence_state_rejects_reused_event_id():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    inventory_state = _latest_inventory_state(evidence, event)
    substituted_event = replace(
        event,
        reporter=replace(
            event.reporter,
            hostname="host-substituted",
        ),
    )
    substituted_inventory_state = replace(
        inventory_state,
        inventory_events={substituted_event.event_id: substituted_event},
    )

    with pytest.raises(ValueError, match="not authorized"):
        RestartPlanLatestEvidenceState(
            inventory_state=substituted_inventory_state,
            acknowledgement_evidence=evidence,
        )


def test_restart_plan_latest_evidence_state_does_not_claim_copy_eligibility():
    event = _latest_event(complete=False)
    evidence, _ = _evidence(event=event)

    state = RestartPlanLatestEvidenceState(
        inventory_state=_latest_inventory_state(evidence, event),
        acknowledgement_evidence=evidence,
    )

    assert not state.manifest.rank_copies[0].copies[0].complete


def test_restart_plan_recovery_evidence_state_accepts_latest_acknowledgements():
    event = _latest_event()
    evidence, _ = _evidence(event=event)
    inventory_state = _latest_inventory_state(evidence, event)

    state = RestartPlanRecoveryEvidenceState(
        copy_state=RestartPlanCopyEligibilityState(inventory_state),
        trust_state=RestartPlanLatestEvidenceState(
            inventory_state=inventory_state,
            acknowledgement_evidence=evidence,
        ),
    )

    assert state.plan == inventory_state.plan
    assert state.manifest == inventory_state.manifest
