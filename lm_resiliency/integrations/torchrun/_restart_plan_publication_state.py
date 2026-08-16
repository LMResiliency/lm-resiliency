"""Pure prepared state for one restart-plan publication transaction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._restart_plan_publication_authority import (
    RestartPlanPublicationAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
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


__all__ = ["PreparedRestartPlanPublication"]
