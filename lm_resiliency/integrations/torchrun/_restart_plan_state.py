"""Pure resolved state for torchrun restart plans."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

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
from lm_resiliency.integrations.torchrun._restart_ack_evidence import (
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


__all__ = [
    "ResolvedRecoveryManifest",
    "RestartPlanCertificationState",
    "RestartPlanGenerationState",
    "RestartPlanInventoryState",
    "RestartPlanLatestEvidenceState",
    "RestartPlanManifestState",
    "RestartPlanQuarantineState",
]
