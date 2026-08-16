"""Pure resolved state for torchrun restart plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistory,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointCertification,
    CheckpointInventoryEvent,
    ProtocolValidationError,
    RankAssignment,
    RecoveryManifest,
    RestartPlan,
    checkpoint_inventory_digest,
    validate_worker_identity,
)
from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckEvidence,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
    RestartPlanRecord,
)


@dataclass(frozen=True, slots=True)
class ResolvedRecoveryManifest:
    """One manifest record bound to its exact immutable source generation."""

    record: RecoveryManifestRecord
    source_snapshot: StoredGenerationSnapshot

    def __post_init__(self) -> None:
        if not isinstance(self.record, RecoveryManifestRecord):
            raise TypeError("ResolvedRecoveryManifest.record must be RecoveryManifestRecord")
        if not isinstance(self.source_snapshot, StoredGenerationSnapshot):
            raise TypeError(
                "ResolvedRecoveryManifest.source_snapshot must be StoredGenerationSnapshot"
            )
        snapshot_record = self.source_snapshot.record
        if snapshot_record.digest != self.record.source_generation_snapshot_digest:
            raise ValueError(
                "ResolvedRecoveryManifest source snapshot digest does not match its record"
            )
        manifest = self.record.manifest
        assignment = snapshot_record.assignment
        if (
            manifest.run_id != assignment.run_id
            or manifest.source_generation != assignment.generation
            or manifest.topology_digest != assignment.topology_digest
        ):
            raise ValueError(
                "ResolvedRecoveryManifest manifest does not match its source generation"
            )

    @property
    def manifest(self) -> RecoveryManifest:
        return self.record.manifest

    @property
    def source_assignment(self) -> RankAssignment:
        return self.source_snapshot.record.assignment


@dataclass(frozen=True, slots=True)
class RestartPlanGenerationState:
    """One plan envelope bound to its intent and generation records."""

    record: RestartPlanRecord
    intent_record: RestartIntentRecord
    lifecycle_record: RestartIntentLifecycleRecord
    from_snapshot: GenerationSnapshotRecord
    to_snapshot: GenerationSnapshotRecord

    def __post_init__(self) -> None:
        expected_types = (
            ("record", self.record, RestartPlanRecord),
            ("intent_record", self.intent_record, RestartIntentRecord),
            (
                "lifecycle_record",
                self.lifecycle_record,
                RestartIntentLifecycleRecord,
            ),
            ("from_snapshot", self.from_snapshot, GenerationSnapshotRecord),
            ("to_snapshot", self.to_snapshot, GenerationSnapshotRecord),
        )
        for path, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"RestartPlanGenerationState.{path} must be {expected_type.__name__}"
                )
        self._validate_digests()
        self._validate_intent()
        self._validate_generations()
        self._validate_publication_authority()

    @property
    def plan(self) -> RestartPlan:
        return self.record.plan

    @property
    def from_assignment(self) -> RankAssignment:
        return self.from_snapshot.assignment

    @property
    def to_assignment(self) -> RankAssignment:
        return self.to_snapshot.assignment

    def _validate_digests(self) -> None:
        if (
            self.record.intent_lifecycle_record_digest != self.lifecycle_record.digest
            or self.record.from_generation_snapshot_digest != self.from_snapshot.digest
            or self.record.to_generation_snapshot_digest != self.to_snapshot.digest
            or self.intent_record.generation_snapshot_digest != self.from_snapshot.digest
        ):
            raise ValueError(
                "RestartPlanGenerationState records do not match their envelope digests"
            )
        closed_intent = self.lifecycle_record.closed_intent
        intent = self.intent_record.intent
        if (
            closed_intent.run_id != intent.run_id
            or closed_intent.generation != intent.generation
            or closed_intent.intent_id != intent.intent_id
            or closed_intent.intent_digest != self.intent_record.digest
        ):
            raise ValueError(
                "RestartPlanGenerationState lifecycle does not close its intent record"
            )

    def _validate_intent(self) -> None:
        plan = self.record.plan
        intent = self.intent_record.intent
        if (
            plan.intent_id != intent.intent_id
            or plan.run_id != intent.run_id
            or plan.from_generation != intent.generation
            or plan.incident_ids != intent.incident_ids
            or plan.reason_code != intent.reason_code
        ):
            raise ValueError("RestartPlanGenerationState plan does not match its restart intent")
        if (
            intent.minimum_recovery_mode == "recovery_verified"
            and plan.recovery_mode != "recovery_verified"
        ):
            raise ValueError(
                "RestartPlanGenerationState plan recovery mode is weaker than its intent"
            )

    def _validate_generations(self) -> None:
        plan = self.record.plan
        from_assignment = self.from_snapshot.assignment
        if (
            from_assignment.run_id != plan.run_id
            or from_assignment.generation != plan.from_generation
            or from_assignment.topology_digest != plan.topology_digest
            or from_assignment.active_nodes * from_assignment.local_world_size
            != plan.expected_world_size
        ):
            raise ValueError("RestartPlanGenerationState source generation does not match its plan")
        if self.to_snapshot.previous_snapshot_digest != self.from_snapshot.digest:
            raise ValueError(
                "RestartPlanGenerationState successor does not reference its source generation"
            )
        expected_assignment = RankAssignment.from_assignments(
            run_id=plan.run_id,
            generation=plan.to_generation,
            assignments=plan.slot_assignments,
            topology_digest=plan.topology_digest,
        )
        if self.to_snapshot.assignment != expected_assignment:
            raise ValueError(
                "RestartPlanGenerationState successor assignment does not match its plan"
            )

    def _validate_publication_authority(self) -> None:
        if (
            self.record.coordinator_id != self.to_snapshot.coordinator_id
            or self.record.lease_id != self.to_snapshot.lease_id
            or self.record.coordinator_lease_duration_ms
            != self.to_snapshot.coordinator_lease_duration_ms
            or self.record.coordinator_fencing_token != self.to_snapshot.coordinator_fencing_token
        ):
            raise ValueError(
                "RestartPlanGenerationState plan and successor use different publication authority"
            )


@dataclass(frozen=True, slots=True)
class RestartPlanManifestState:
    """One generation-bound plan linked to its resolved recovery manifest."""

    generation_state: RestartPlanGenerationState
    resolved_manifest: ResolvedRecoveryManifest

    def __post_init__(self) -> None:
        if not isinstance(self.generation_state, RestartPlanGenerationState):
            raise TypeError(
                "RestartPlanManifestState.generation_state must be RestartPlanGenerationState"
            )
        if not isinstance(self.resolved_manifest, ResolvedRecoveryManifest):
            raise TypeError(
                "RestartPlanManifestState.resolved_manifest must be ResolvedRecoveryManifest"
            )
        plan_record = self.generation_state.record
        plan = plan_record.plan
        manifest_record = self.resolved_manifest.record
        manifest = manifest_record.manifest
        if plan_record.recovery_manifest_record_digest != manifest_record.digest:
            raise ValueError(
                "RestartPlanManifestState manifest record digest does not match its plan"
            )
        if (
            manifest.manifest_id != plan.checkpoint_manifest_id
            or manifest.run_id != plan.run_id
            or manifest.source_generation > plan.from_generation
            or manifest.step != plan.checkpoint_step
            or manifest.topology_digest != plan.topology_digest
        ):
            raise ValueError("RestartPlanManifestState manifest metadata does not match its plan")
        source_assignment = self.resolved_manifest.source_assignment
        source_world_size = source_assignment.active_nodes * source_assignment.local_world_size
        if source_world_size != plan.expected_world_size:
            raise ValueError("RestartPlanManifestState source world size does not match its plan")
        if (
            source_assignment.generation == plan.from_generation
            and source_assignment != self.generation_state.from_assignment
        ):
            raise ValueError(
                "RestartPlanManifestState source assignment conflicts with its current generation"
            )
        if plan.recovery_mode == "recovery_verified" and manifest.trust != "recovery_verified":
            raise ValueError(
                "RestartPlanManifestState verified recovery requires a verified manifest"
            )
        if plan.checkpoint_source == "durable" and manifest.trust != "recovery_verified":
            raise ValueError(
                "RestartPlanManifestState durable recovery requires a verified manifest"
            )

    @property
    def plan(self) -> RestartPlan:
        return self.generation_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.resolved_manifest.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanQuarantineState:
    """One manifest-bound plan linked to its exact quarantine records."""

    manifest_state: RestartPlanManifestState
    quarantine_records: Mapping[str, NodeQuarantineRecord]

    def __post_init__(self) -> None:
        if not isinstance(self.manifest_state, RestartPlanManifestState):
            raise TypeError(
                "RestartPlanQuarantineState.manifest_state must be RestartPlanManifestState"
            )
        if not isinstance(self.quarantine_records, Mapping):
            raise TypeError("RestartPlanQuarantineState.quarantine_records must be a mapping")
        records: dict[str, NodeQuarantineRecord] = {}
        for node_id, record in self.quarantine_records.items():
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(
                    "RestartPlanQuarantineState.quarantine_records keys must be non-empty node IDs"
                )
            if not isinstance(record, NodeQuarantineRecord):
                raise TypeError(
                    "RestartPlanQuarantineState.quarantine_records values "
                    "must be NodeQuarantineRecord"
                )
            if record.node_id != node_id:
                raise ValueError(
                    "RestartPlanQuarantineState quarantine record node does not match its key"
                )
            records[node_id] = record
        records = dict(sorted(records.items()))
        plan_record = self.manifest_state.generation_state.record
        expected_digests = plan_record.quarantine_record_digests
        if set(records) != set(expected_digests):
            raise ValueError(
                "RestartPlanQuarantineState records must exactly cover the plan's quarantined nodes"
            )
        plan = plan_record.plan
        for node_id, record in records.items():
            if record.digest != expected_digests[node_id]:
                raise ValueError(
                    "RestartPlanQuarantineState quarantine record digest does not match its plan"
                )
            if (
                record.run_id != plan.run_id
                or record.plan_id != plan.plan_id
                or record.intent_id != plan.intent_id
                or record.from_generation != plan.from_generation
                or record.effective_generation != plan.to_generation
                or record.incident_ids != plan.incident_ids
                or record.reason_code != plan.reason_code
            ):
                raise ValueError(
                    "RestartPlanQuarantineState quarantine record does not match its plan"
                )
            if (
                record.coordinator_id != plan_record.coordinator_id
                or record.lease_id != plan_record.lease_id
                or record.coordinator_lease_duration_ms != plan_record.coordinator_lease_duration_ms
                or record.coordinator_fencing_token != plan_record.coordinator_fencing_token
            ):
                raise ValueError(
                    "RestartPlanQuarantineState quarantine record uses different "
                    "publication authority"
                )
        object.__setattr__(
            self,
            "quarantine_records",
            MappingProxyType(records),
        )

    @property
    def plan(self) -> RestartPlan:
        return self.manifest_state.plan


@dataclass(frozen=True, slots=True)
class RestartPlanInventoryState:
    """One quarantine-bound plan linked to exact checkpoint inventory events."""

    quarantine_state: RestartPlanQuarantineState
    inventory_events: Mapping[str, CheckpointInventoryEvent]

    def __post_init__(self) -> None:
        if not isinstance(self.quarantine_state, RestartPlanQuarantineState):
            raise TypeError(
                "RestartPlanInventoryState.quarantine_state must be RestartPlanQuarantineState"
            )
        if not isinstance(self.inventory_events, Mapping):
            raise TypeError("RestartPlanInventoryState.inventory_events must be a mapping")
        events: dict[str, CheckpointInventoryEvent] = {}
        for event_id, event in self.inventory_events.items():
            if not isinstance(event_id, str) or not event_id.strip():
                raise ValueError(
                    "RestartPlanInventoryState.inventory_events keys must be non-empty event IDs"
                )
            if not isinstance(event, CheckpointInventoryEvent):
                raise TypeError(
                    "RestartPlanInventoryState.inventory_events values "
                    "must be CheckpointInventoryEvent"
                )
            if event.event_id != event_id:
                raise ValueError(
                    "RestartPlanInventoryState inventory event ID does not match its key"
                )
            events[event_id] = event
        events = dict(sorted(events.items()))
        manifest = self.quarantine_state.manifest_state.manifest
        referenced_event_ids = {
            copy.inventory_event_id
            for rank_copies in manifest.rank_copies
            for copy in rank_copies.copies
        }
        if set(events) != referenced_event_ids:
            raise ValueError(
                "RestartPlanInventoryState events must exactly cover the manifest's references"
            )
        source_assignment = self.quarantine_state.manifest_state.resolved_manifest.source_assignment
        for event in events.values():
            if (
                event.run_id != manifest.run_id
                or event.generation != manifest.source_generation
                or event.step != manifest.step
                or event.topology_digest != manifest.topology_digest
            ):
                raise ValueError(
                    "RestartPlanInventoryState inventory event does not match its manifest"
                )
            try:
                validate_worker_identity(event.reporter, source_assignment)
            except ProtocolValidationError as error:
                raise ValueError(
                    "RestartPlanInventoryState inventory reporter does not match "
                    "the source assignment"
                ) from error
            if event.trust == "candidate" or (
                manifest.trust == "recovery_verified" and event.trust != "recovery_verified"
            ):
                raise ValueError(
                    "RestartPlanInventoryState inventory trust is incompatible with its manifest"
                )
        for rank_copies in manifest.rank_copies:
            for copy in rank_copies.copies:
                event = events[copy.inventory_event_id]
                if copy not in event.copies:
                    raise ValueError(
                        "RestartPlanInventoryState manifest copy does not match its inventory event"
                    )
                if (
                    copy.storage_kind in {"memory", "node_local"}
                    and copy.holder_node_id != event.reporter.node_id
                ):
                    raise ValueError(
                        "RestartPlanInventoryState local copy was not reported by its holder"
                    )
        object.__setattr__(
            self,
            "inventory_events",
            MappingProxyType(events),
        )

    @property
    def plan(self) -> RestartPlan:
        return self.quarantine_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.quarantine_state.manifest_state.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanCertificationState:
    """One inventory-bound plan authorized by trusted checkpoint certification."""

    inventory_state: RestartPlanInventoryState
    certifications: tuple[CheckpointCertification, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.inventory_state, RestartPlanInventoryState):
            raise TypeError(
                "RestartPlanCertificationState.inventory_state must be RestartPlanInventoryState"
            )
        if not isinstance(self.certifications, tuple):
            raise TypeError("RestartPlanCertificationState.certifications must be a tuple")
        manifest = self.inventory_state.manifest
        if manifest.trust != "recovery_verified":
            raise ValueError("RestartPlanCertificationState requires a recovery-verified manifest")
        plan = self.inventory_state.plan
        certification_ids: set[str] = set()
        certified_event_digests: dict[str, str] = {}
        for certification in self.certifications:
            if not isinstance(certification, CheckpointCertification):
                raise TypeError(
                    "RestartPlanCertificationState.certifications values "
                    "must be CheckpointCertification"
                )
            if certification.certification_id in certification_ids:
                raise ValueError("RestartPlanCertificationState certification IDs must be unique")
            certification_ids.add(certification.certification_id)
            if (
                certification.run_id != manifest.run_id
                or certification.source_generation != manifest.source_generation
                or certification.step != manifest.step
                or certification.topology_digest != manifest.topology_digest
                or certification.checkpoint_source != plan.checkpoint_source
                or certification.checkpoint_id != plan.checkpoint_id
                or certification.expected_world_size != plan.expected_world_size
            ):
                raise ValueError(
                    "RestartPlanCertificationState certification does not match its plan"
                )
            for event_id, digest in certification.inventory_event_digests.items():
                previous_digest = certified_event_digests.get(event_id)
                if previous_digest is not None and previous_digest != digest:
                    raise ValueError(
                        "RestartPlanCertificationState certifications contain "
                        "conflicting inventory digests"
                    )
                certified_event_digests[event_id] = digest
        for event_id, event in self.inventory_state.inventory_events.items():
            if certified_event_digests.get(event_id) != checkpoint_inventory_digest(event):
                raise ValueError(
                    "RestartPlanCertificationState inventory event is not exactly certified"
                )
        object.__setattr__(
            self,
            "certifications",
            tuple(sorted(self.certifications, key=lambda value: value.certification_id)),
        )

    @property
    def plan(self) -> RestartPlan:
        return self.inventory_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.inventory_state.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanLatestEvidenceState:
    """One inventory-bound latest plan authorized by restart acknowledgements."""

    inventory_state: RestartPlanInventoryState
    acknowledgement_evidence: RestartAckEvidence

    def __post_init__(self) -> None:
        if not isinstance(self.inventory_state, RestartPlanInventoryState):
            raise TypeError(
                "RestartPlanLatestEvidenceState.inventory_state must be RestartPlanInventoryState"
            )
        if not isinstance(self.acknowledgement_evidence, RestartAckEvidence):
            raise TypeError(
                "RestartPlanLatestEvidenceState.acknowledgement_evidence must be RestartAckEvidence"
            )
        manifest_state = self.inventory_state.quarantine_state.manifest_state
        generation_state = manifest_state.generation_state
        manifest = manifest_state.manifest
        if manifest.trust != "latest":
            raise ValueError("RestartPlanLatestEvidenceState requires a latest manifest")
        if manifest.source_generation != generation_state.plan.from_generation:
            raise ValueError(
                "RestartPlanLatestEvidenceState latest manifest is not from the current generation"
            )
        if (
            manifest_state.resolved_manifest.source_snapshot.record
            != generation_state.from_snapshot
        ):
            raise ValueError(
                "RestartPlanLatestEvidenceState latest manifest does not use "
                "the exact current generation snapshot"
            )
        opened = self.acknowledgement_evidence.collection.opened
        if opened.prepared.record != generation_state.intent_record:
            raise ValueError(
                "RestartPlanLatestEvidenceState acknowledgements answer another restart intent"
            )
        if opened.prepared.current.snapshot.record != generation_state.from_snapshot:
            raise ValueError(
                "RestartPlanLatestEvidenceState acknowledgements belong to another generation"
            )
        for event in self.inventory_state.inventory_events.values():
            if not self.acknowledgement_evidence.authorizes_latest_inventory(event):
                raise ValueError(
                    "RestartPlanLatestEvidenceState inventory event is not authorized "
                    "by restart acknowledgement evidence"
                )

    @property
    def plan(self) -> RestartPlan:
        return self.inventory_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.inventory_state.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanCopyEligibilityState:
    """One inventory-bound plan whose advertised copies are all usable."""

    inventory_state: RestartPlanInventoryState

    def __post_init__(self) -> None:
        if not isinstance(self.inventory_state, RestartPlanInventoryState):
            raise TypeError(
                "RestartPlanCopyEligibilityState.inventory_state must be RestartPlanInventoryState"
            )
        plan = self.inventory_state.plan
        manifest = self.inventory_state.manifest
        entries = {entry.owner_global_rank: entry for entry in manifest.rank_copies}
        required_ranks = set(range(plan.expected_world_size))
        if set(entries) != required_ranks:
            missing = sorted(required_ranks - set(entries))
            extra = sorted(set(entries) - required_ranks)
            raise ValueError(
                "RestartPlanCopyEligibilityState rank coverage mismatch; "
                f"missing={missing!r}, extra={extra!r}"
            )
        source_assignment = (
            self.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_assignment
        )
        source_owner_by_rank = {
            rank: source_assignment.slot_to_node_id[slot]
            for slot, (first_rank, last_rank) in source_assignment.slot_to_rank_range.items()
            for rank in range(first_rank, last_rank)
        }
        assigned_nodes = {assignment.node_id for assignment in plan.slot_assignments}
        compatible_holder_kinds = (
            {"durable"} if plan.checkpoint_source == "durable" else {"owner", "peer"}
        )
        for rank, entry in entries.items():
            if not entry.copies:
                raise ValueError(
                    f"RestartPlanCopyEligibilityState rank {rank} has no eligible copy"
                )
            for copy in entry.copies:
                if not (
                    copy.complete
                    and copy.checkpoint_step == manifest.step
                    and copy.holder_kind in compatible_holder_kinds
                    and copy.storage_kind in {"node_local", "shared", "remote"}
                    and copy.checkpoint_id == plan.checkpoint_id
                    and (
                        copy.holder_kind == "durable"
                        or (
                            copy.holder_kind == "owner"
                            and copy.holder_node_id == source_owner_by_rank[rank]
                        )
                        or (
                            copy.holder_kind == "peer"
                            and copy.holder_node_id != source_owner_by_rank[rank]
                        )
                    )
                    and (
                        copy.storage_kind in {"shared", "remote"}
                        or copy.holder_node_id in assigned_nodes
                    )
                ):
                    raise ValueError(
                        f"RestartPlanCopyEligibilityState rank {rank} contains an ineligible copy"
                    )

    @property
    def plan(self) -> RestartPlan:
        return self.inventory_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.inventory_state.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanRecoveryEvidenceState:
    """One copy-eligible plan authorized by one exact recovery trust path."""

    copy_state: RestartPlanCopyEligibilityState
    trust_state: RestartPlanLatestEvidenceState | RestartPlanCertificationState

    def __post_init__(self) -> None:
        if not isinstance(self.copy_state, RestartPlanCopyEligibilityState):
            raise TypeError(
                "RestartPlanRecoveryEvidenceState.copy_state "
                "must be RestartPlanCopyEligibilityState"
            )
        if not isinstance(
            self.trust_state,
            (RestartPlanLatestEvidenceState, RestartPlanCertificationState),
        ):
            raise TypeError(
                "RestartPlanRecoveryEvidenceState.trust_state must be "
                "RestartPlanLatestEvidenceState or RestartPlanCertificationState"
            )
        if self.copy_state.inventory_state != self.trust_state.inventory_state:
            raise ValueError(
                "RestartPlanRecoveryEvidenceState copy and trust evidence "
                "do not describe the same inventory state"
            )

    @property
    def plan(self) -> RestartPlan:
        return self.copy_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.copy_state.manifest


@dataclass(frozen=True, slots=True)
class RestartPlanPlacementState:
    """One successor placement backed by exact live agent registrations."""

    generation_state: RestartPlanGenerationState
    registration_histories: Mapping[str, AgentRegistrationHistory]
    observed_at_unix_ms: int
    environment_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.generation_state, RestartPlanGenerationState):
            raise TypeError(
                "RestartPlanPlacementState.generation_state must be RestartPlanGenerationState"
            )
        if not isinstance(self.registration_histories, Mapping):
            raise TypeError("RestartPlanPlacementState.registration_histories must be a mapping")
        if (
            isinstance(self.observed_at_unix_ms, bool)
            or not isinstance(self.observed_at_unix_ms, int)
            or self.observed_at_unix_ms < 1
        ):
            raise ValueError(
                "RestartPlanPlacementState.observed_at_unix_ms must be a positive integer"
            )
        if not isinstance(self.environment_digest, str) or not self.environment_digest.strip():
            raise ValueError(
                "RestartPlanPlacementState.environment_digest must be a non-empty string"
            )
        self._validate_topology()
        histories = self._validate_registrations()
        object.__setattr__(
            self,
            "registration_histories",
            MappingProxyType(histories),
        )

    @property
    def plan(self) -> RestartPlan:
        return self.generation_state.plan

    def _validate_topology(self) -> None:
        plan = self.plan
        intent = self.generation_state.intent_record.intent
        current_slots = {
            node_id: slot
            for slot, node_id in self.generation_state.from_assignment.slot_to_node_id.items()
        }
        planned_slots = {
            assignment.node_id: assignment.logical_node_slot for assignment in plan.slot_assignments
        }
        current_nodes = set(current_slots)
        planned_nodes = set(planned_slots)
        suspected_nodes = set(intent.suspected_node_ids)
        unknown_suspects = sorted(suspected_nodes - current_nodes)
        if unknown_suspects:
            raise ValueError(
                "RestartPlanPlacementState suspected nodes are not active in "
                f"the source generation: {unknown_suspects!r}"
            )
        retained_suspects = sorted(suspected_nodes & planned_nodes)
        if retained_suspects:
            raise ValueError(
                f"RestartPlanPlacementState suspected nodes remain assigned: {retained_suspects!r}"
            )
        moved_survivors = sorted(
            node_id
            for node_id in current_nodes & planned_nodes
            if current_slots[node_id] != planned_slots[node_id]
        )
        if moved_survivors:
            raise ValueError(
                "RestartPlanPlacementState surviving nodes changed logical slots: "
                f"{moved_survivors!r}"
            )
        if not planned_nodes - current_nodes:
            raise ValueError(
                "RestartPlanPlacementState version 1 requires at least one replacement node"
            )
        if len(plan.slot_assignments) != self.generation_state.from_assignment.active_nodes:
            raise ValueError(
                "RestartPlanPlacementState successor active node count "
                "does not match the source generation"
            )
        removed_nodes = current_nodes - planned_nodes
        quarantined_nodes = set(plan.quarantined_node_ids)
        unsupported_quarantine = sorted(quarantined_nodes - suspected_nodes)
        if unsupported_quarantine:
            raise ValueError(
                "RestartPlanPlacementState quarantines nodes outside the intent scope: "
                f"{unsupported_quarantine!r}"
            )
        unremoved_quarantine = sorted(quarantined_nodes - removed_nodes)
        if unremoved_quarantine:
            raise ValueError(
                "RestartPlanPlacementState quarantined nodes remain in the successor "
                f"assignment: {unremoved_quarantine!r}"
            )

    def _validate_registrations(self) -> dict[str, AgentRegistrationHistory]:
        plan = self.plan
        assignments = {assignment.node_id: assignment for assignment in plan.slot_assignments}
        histories: dict[str, AgentRegistrationHistory] = {}
        for node_id, history in self.registration_histories.items():
            if not isinstance(node_id, str) or not node_id.strip():
                raise ValueError(
                    "RestartPlanPlacementState.registration_histories keys "
                    "must be non-empty node IDs"
                )
            if not isinstance(history, AgentRegistrationHistory):
                raise TypeError(
                    "RestartPlanPlacementState.registration_histories values "
                    "must be AgentRegistrationHistory"
                )
            histories[node_id] = history
        if set(histories) != set(assignments):
            raise ValueError(
                "RestartPlanPlacementState registration histories must exactly "
                "cover the successor assignment"
            )
        for node_id, assignment in assignments.items():
            registration = histories[node_id].current
            if registration is None:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} has no current registration"
                )
            identity = registration.record.agent_identity
            if identity.run_id != plan.run_id or identity.node_id != node_id:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} registration "
                    "has the wrong identity"
                )
            if registration.granted_at_unix_ms > self.observed_at_unix_ms:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} registration "
                    "was granted after the observation"
                )
            if registration.expires_at_unix_ms <= self.observed_at_unix_ms:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} registration is not live"
                )
            if identity.local_world_size != assignment.local_world_size:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} local world size "
                    "does not match its slot"
                )
            if identity.environment_digest != self.environment_digest:
                raise ValueError(
                    f"RestartPlanPlacementState node {node_id!r} environment is incompatible"
                )
        return dict(sorted(histories.items()))


@dataclass(frozen=True, slots=True)
class RestartPlanCandidateState:
    """One recovery- and placement-admitted plan candidate."""

    recovery_state: RestartPlanRecoveryEvidenceState
    placement_state: RestartPlanPlacementState

    def __post_init__(self) -> None:
        if not isinstance(self.recovery_state, RestartPlanRecoveryEvidenceState):
            raise TypeError(
                "RestartPlanCandidateState.recovery_state must be RestartPlanRecoveryEvidenceState"
            )
        if not isinstance(self.placement_state, RestartPlanPlacementState):
            raise TypeError(
                "RestartPlanCandidateState.placement_state must be RestartPlanPlacementState"
            )
        recovery_generation_state = self.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.generation_state
        if recovery_generation_state != self.placement_state.generation_state:
            raise ValueError(
                "RestartPlanCandidateState recovery and placement states "
                "do not describe the same plan generation"
            )
        if self.placement_state.observed_at_unix_ms >= self.plan.restart_deadline_unix_ms:
            raise ValueError("RestartPlanCandidateState restart deadline has elapsed")

    @property
    def plan(self) -> RestartPlan:
        return self.recovery_state.plan

    @property
    def manifest(self) -> RecoveryManifest:
        return self.recovery_state.manifest


__all__ = [
    "ResolvedRecoveryManifest",
    "RestartPlanCandidateState",
    "RestartPlanCertificationState",
    "RestartPlanCopyEligibilityState",
    "RestartPlanGenerationState",
    "RestartPlanInventoryState",
    "RestartPlanLatestEvidenceState",
    "RestartPlanManifestState",
    "RestartPlanPlacementState",
    "RestartPlanQuarantineState",
    "RestartPlanRecoveryEvidenceState",
]
