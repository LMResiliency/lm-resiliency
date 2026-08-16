"""Immutable transaction records for publishing one torchrun restart plan."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._generation_reader import CurrentGeneration
from lm_resiliency.integrations.torchrun._generation_records import GenerationHeadRecord
from lm_resiliency.integrations.torchrun._quarantine_store import node_quarantine_key
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import RestartPlanCandidateState

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class RestartPlanPublicationRecords:
    """Canonical records and store inputs for one restart-plan publication."""

    candidate: RestartPlanCandidateState
    current: CurrentGeneration

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RestartPlanCandidateState):
            raise TypeError(
                "RestartPlanPublicationRecords.candidate must be RestartPlanCandidateState"
            )
        if not isinstance(self.current, CurrentGeneration):
            raise TypeError("RestartPlanPublicationRecords.current must be CurrentGeneration")
        generation_state = self.candidate.placement_state.generation_state
        if self.current.snapshot.record != generation_state.from_snapshot:
            raise ValueError(
                "RestartPlanPublicationRecords current generation does not match its candidate"
            )
        _positive_integer(
            self.current.head_revision,
            "RestartPlanPublicationRecords.current.head_revision",
        )
        _positive_integer(
            self.current.snapshot.revision,
            "RestartPlanPublicationRecords.current.snapshot.revision",
        )
        manifest_source = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot
        _positive_integer(
            manifest_source.revision,
            "RestartPlanPublicationRecords manifest source revision",
        )
        if self.manifest_source_generation_snapshot_key == self.source_generation_snapshot_key and (
            manifest_source.revision != self.current.snapshot.revision
            or manifest_source.record != self.current.snapshot.record
        ):
            raise ValueError(
                "RestartPlanPublicationRecords current and manifest source snapshots disagree"
            )

    @property
    def run_prefix(self) -> str:
        run_digest = hashlib.sha256(self.candidate.plan.run_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}"

    @property
    def plan_key(self) -> str:
        return f"{self.run_prefix}/restart-plans/{self.candidate.plan.to_generation}"

    @property
    def recovery_manifest_key(self) -> str:
        return f"{self.plan_key}/recovery-manifest"

    @property
    def generation_head_key(self) -> str:
        return f"{self.run_prefix}/generation-head"

    @property
    def source_generation_snapshot_key(self) -> str:
        return f"{self.run_prefix}/generations/{self.candidate.plan.from_generation}"

    @property
    def successor_generation_snapshot_key(self) -> str:
        return f"{self.run_prefix}/generations/{self.candidate.plan.to_generation}"

    @property
    def manifest_source_generation_snapshot_key(self) -> str:
        source_generation = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.manifest.source_generation
        return f"{self.run_prefix}/generations/{source_generation}"

    @property
    def quarantine_keys(self) -> Mapping[str, str]:
        records = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state
        return MappingProxyType(
            {
                node_id: node_quarantine_key(self.candidate.plan.run_id, node_id)
                for node_id in records.quarantine_records
            }
        )

    @property
    def registration_keys(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                node_id: agent_registration_key(self.candidate.plan.run_id, node_id)
                for node_id in self.candidate.placement_state.registration_histories
            }
        )

    @property
    def generation_head(self) -> GenerationHeadRecord:
        generation_state = self.candidate.placement_state.generation_state
        return GenerationHeadRecord(
            run_id=self.candidate.plan.run_id,
            generation=self.candidate.plan.to_generation,
            snapshot_digest=generation_state.to_snapshot.digest,
        )

    @property
    def deadline_unix_ms(self) -> int:
        registration_expiries = []
        for history in self.candidate.placement_state.registration_histories.values():
            registration = history.current
            if registration is None:
                raise AssertionError("validated placement lost its current registration")
            registration_expiries.append(registration.expires_at_unix_ms)
        return min(
            self.candidate.plan.restart_deadline_unix_ms,
            *registration_expiries,
        )

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        generation_state = self.candidate.placement_state.generation_state
        manifest_state = (
            self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state
        )
        quarantine_state = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state
        writes = {
            self.generation_head_key: ControlStoreWrite(
                expected_revision=self.current.head_revision,
                value=self.generation_head.to_json(),
            ),
            self.successor_generation_snapshot_key: ControlStoreWrite(
                expected_revision=None,
                value=generation_state.to_snapshot.to_json(),
                require_never_created=True,
            ),
            self.recovery_manifest_key: ControlStoreWrite(
                expected_revision=None,
                value=manifest_state.resolved_manifest.record.to_json(),
                require_never_created=True,
            ),
            self.plan_key: ControlStoreWrite(
                expected_revision=None,
                value=generation_state.record.to_json(),
                require_never_created=True,
            ),
        }
        writes.update(
            {
                self.quarantine_keys[node_id]: ControlStoreWrite(
                    expected_revision=None,
                    value=record.to_json(),
                    require_never_created=True,
                )
                for node_id, record in quarantine_state.quarantine_records.items()
            }
        )
        return MappingProxyType(writes)

    @property
    def conditions(self) -> Mapping[str, int]:
        manifest_source = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot
        conditions = {
            self.source_generation_snapshot_key: self.current.snapshot.revision,
            self.manifest_source_generation_snapshot_key: manifest_source.revision,
        }
        for node_id, history in self.candidate.placement_state.registration_histories.items():
            registration = history.current
            if registration is None:
                raise AssertionError("validated placement lost its current registration")
            conditions[self.registration_keys[node_id]] = registration.fencing_token
        return MappingProxyType(conditions)


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


@dataclass(frozen=True, slots=True)
class PreparedRestartPlanPublication:
    """One admitted publication bundle bound to its closed lifecycle."""

    authority: RestartPlanPublicationAuthority
    lifecycle_fence: RestartPlanPublicationLifecycleFence

    def __post_init__(self) -> None:
        if not isinstance(self.authority, RestartPlanPublicationAuthority):
            raise TypeError(
                "PreparedRestartPlanPublication.authority must be RestartPlanPublicationAuthority"
            )
        if not isinstance(self.lifecycle_fence, RestartPlanPublicationLifecycleFence):
            raise TypeError(
                "PreparedRestartPlanPublication.lifecycle_fence must be "
                "RestartPlanPublicationLifecycleFence"
            )
        self._validate_records()
        self._validate_authority_history()
        self._validate_conditions()

    def _validate_records(self) -> None:
        records = self.authority.records
        generation_state = records.candidate.placement_state.generation_state
        closure = self.lifecycle_fence.closure
        if (
            generation_state.intent_record != closure.intent
            or generation_state.lifecycle_record != closure.lifecycle
            or generation_state.from_snapshot != closure.generation_snapshot.record
            or records.current.snapshot != closure.generation_snapshot
        ):
            raise ValueError(
                "PreparedRestartPlanPublication candidate does not match its closed lifecycle"
            )
        if self.authority.observed_at_unix_ms < closure.closed_at_unix_ms:
            raise ValueError(
                "PreparedRestartPlanPublication authority observation precedes closure"
            )

    def _validate_authority_history(self) -> None:
        closure = self.lifecycle_fence.closure
        history = closure.lease_history
        closing_matches = tuple(
            index
            for index, authority in enumerate(history)
            if authority == closure.closing_authority
        )
        publication_matches = tuple(
            index
            for index, authority in enumerate(history)
            if authority == self.authority.coordinator_authority
        )
        if len(closing_matches) != 1 or len(publication_matches) != 1:
            raise ValueError(
                "PreparedRestartPlanPublication authorities are absent from durable lease history"
            )
        if publication_matches[0] < closing_matches[0]:
            raise ValueError("PreparedRestartPlanPublication authority predates closure authority")

    def _validate_conditions(self) -> None:
        publication_conditions = self.authority.records.conditions
        lifecycle_conditions = self.lifecycle_fence.conditions
        conflicting = sorted(
            key
            for key in set(publication_conditions) & set(lifecycle_conditions)
            if publication_conditions[key] != lifecycle_conditions[key]
        )
        if conflicting:
            raise ValueError(
                "PreparedRestartPlanPublication condition revisions disagree for "
                f"keys: {conflicting!r}"
            )
        overlapping_targets = sorted(
            set(self.authority.records.writes)
            & (set(publication_conditions) | set(lifecycle_conditions))
        )
        if overlapping_targets:
            raise ValueError(
                "PreparedRestartPlanPublication conditions must not also be "
                f"transaction targets: {overlapping_targets!r}"
            )
        if self.guard_key in publication_conditions or self.guard_key in lifecycle_conditions:
            raise ValueError("PreparedRestartPlanPublication guard key must not be a condition")
        if self.guard_key in self.authority.records.writes:
            raise ValueError(
                "PreparedRestartPlanPublication guard key must not be a transaction target"
            )

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return self.authority.records.writes

    @property
    def conditions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                **self.authority.records.conditions,
                **self.lifecycle_fence.conditions,
            }
        )

    @property
    def guard_key(self) -> str:
        return f"{self.authority.records.run_prefix}/coordinator-lease"

    @property
    def expected_guard_revision(self) -> int:
        return self.authority.coordinator_authority.lease.fencing_token

    @property
    def not_before_unix_ms(self) -> int:
        return self.authority.not_before_unix_ms

    @property
    def deadline_unix_ms(self) -> int:
        return self.authority.deadline_unix_ms


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "PreparedRestartPlanPublication",
    "RestartPlanPublicationAuthority",
    "RestartPlanPublicationRecords",
]
