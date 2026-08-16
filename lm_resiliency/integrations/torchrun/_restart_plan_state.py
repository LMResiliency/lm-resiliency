"""Pure resolved state for torchrun restart plans."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RecoveryManifest,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import (
    RecoveryManifestRecord,
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


__all__ = ["ResolvedRecoveryManifest"]
