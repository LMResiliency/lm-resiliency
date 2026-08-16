"""Coordinator authority for one admitted restart-plan publication."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_records import (
    RestartPlanPublicationRecords,
)


@dataclass(frozen=True, slots=True)
class RestartPlanPublicationAuthority:
    """One publication bundle bound to its coordinator lease and time window."""

    records: RestartPlanPublicationRecords
    coordinator_authority: CoordinatorLeaseAuthority
    observed_at_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, RestartPlanPublicationRecords):
            raise TypeError(
                "RestartPlanPublicationAuthority.records must be RestartPlanPublicationRecords"
            )
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "RestartPlanPublicationAuthority.coordinator_authority must be "
                "CoordinatorLeaseAuthority"
            )
        _positive_integer(
            self.observed_at_unix_ms,
            "RestartPlanPublicationAuthority.observed_at_unix_ms",
        )
        self._validate_authority()
        required_observation = max(
            self.coordinator_authority.lease.granted_at_unix_ms,
            self.records.current.snapshot.committed_at_unix_ms,
            self.records.candidate.placement_state.observed_at_unix_ms,
            self.records.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot.committed_at_unix_ms,
        )
        if self.observed_at_unix_ms < required_observation:
            raise ValueError(
                "RestartPlanPublicationAuthority observation precedes one of its inputs"
            )
        if self.observed_at_unix_ms >= self.deadline_unix_ms:
            raise ValueError("RestartPlanPublicationAuthority publication window has elapsed")

    def _validate_authority(self) -> None:
        authority = self.coordinator_authority.lease
        plan_record = self.records.candidate.placement_state.generation_state.record
        if (
            authority.record.run_id != self.records.candidate.plan.run_id
            or authority.record.coordinator_id != plan_record.coordinator_id
            or authority.record.lease_id != plan_record.lease_id
            or authority.record.lease_duration_ms != plan_record.coordinator_lease_duration_ms
            or authority.fencing_token != plan_record.coordinator_fencing_token
        ):
            raise ValueError(
                "RestartPlanPublicationAuthority coordinator lease does not "
                "authorize its plan records"
            )

    @property
    def not_before_unix_ms(self) -> int:
        return self.observed_at_unix_ms

    @property
    def deadline_unix_ms(self) -> int:
        return min(
            self.records.deadline_unix_ms,
            self.coordinator_authority.lease.expires_at_unix_ms,
        )


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = ["RestartPlanPublicationAuthority"]
