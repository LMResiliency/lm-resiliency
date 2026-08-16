"""Pure resolved state for torchrun restart plans."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RecoveryManifest,
    RestartPlan,
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


__all__ = [
    "ResolvedRecoveryManifest",
    "RestartPlanGenerationState",
]
