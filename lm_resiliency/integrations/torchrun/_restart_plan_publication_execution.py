"""Guarded execution of prepared restart-plan publications."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    ControlStoreHistoryConflict,
    ControlStoreTooEarly,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_state import (
    PreparedRestartPlanPublication,
)


class RestartPlanPublicationExecutionError(RuntimeError):
    """Base error for committing one prepared restart-plan publication."""


class RestartPlanPublicationExecutionConflict(RestartPlanPublicationExecutionError):
    """Raised when lifecycle or generation state changes before publication."""


class RestartPlanPublicationExecutionRegistrationLost(RestartPlanPublicationExecutionError):
    """Raised when a selected successor registration changes or expires."""


class RestartPlanPublicationExecutionLeaseLost(RestartPlanPublicationExecutionError):
    """Raised when the coordinator lease changes or expires."""


class RestartPlanPublicationExecutionDeadlineElapsed(RestartPlanPublicationExecutionError):
    """Raised when the prepared restart deadline elapses."""


class RestartPlanPublicationExecutionClockError(RestartPlanPublicationExecutionError):
    """Raised when authoritative store time contradicts preparation."""


class RestartPlanPublicationExecutionCorrupt(RestartPlanPublicationExecutionError):
    """Raised when the transaction returns contradictory committed state."""


@dataclass(frozen=True, slots=True)
class CommittedRestartPlanPublication:
    """One verified committed restart-plan publication transaction."""

    prepared: PreparedRestartPlanPublication
    entries: Mapping[str, ControlStoreEntry]

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedRestartPlanPublication):
            raise TypeError(
                "CommittedRestartPlanPublication.prepared must be PreparedRestartPlanPublication"
            )
        if not isinstance(self.entries, Mapping):
            raise TypeError("CommittedRestartPlanPublication.entries must be a mapping")
        object.__setattr__(self, "entries", MappingProxyType(dict(self.entries)))
        self._validate_entries()

    def _validate_entries(self) -> None:
        writes = self.prepared.writes
        if set(self.entries) != set(writes):
            raise ValueError(
                "CommittedRestartPlanPublication entries do not match the prepared key set"
            )
        commit_times: set[int | None] = set()
        transaction_sequences: set[int] = set()
        for key, write in writes.items():
            entry = self.entries[key]
            if not isinstance(entry, ControlStoreEntry):
                raise TypeError(
                    "CommittedRestartPlanPublication entries must contain ControlStoreEntry"
                )
            if entry.value != write.value:
                raise ValueError(
                    f"CommittedRestartPlanPublication entry {key!r} does not match preparation"
                )
            self._validate_lineage(key, entry)
            self._validate_guard(key, entry)
            commit_times.add(entry.committed_at_unix_ms)
            transaction_sequences.add(entry.transaction_sequence)
        if len(commit_times) != 1 or None in commit_times or len(transaction_sequences) != 1:
            raise ValueError("CommittedRestartPlanPublication entries do not share one transaction")
        committed_at_unix_ms = next(iter(commit_times))
        if committed_at_unix_ms is None:
            raise AssertionError("validated publication lost its commit time")
        if (
            committed_at_unix_ms < self.prepared.not_before_unix_ms
            or committed_at_unix_ms >= self.prepared.deadline_unix_ms
        ):
            raise ValueError(
                "CommittedRestartPlanPublication commit is outside its prepared time window"
            )
        if self.transaction_sequence <= self._latest_input_transaction_sequence():
            raise ValueError(
                "CommittedRestartPlanPublication does not follow all authenticated inputs"
            )

    def _validate_lineage(self, key: str, entry: ControlStoreEntry) -> None:
        records = self.prepared.authority.records
        if key == records.generation_head_key:
            generation = records.candidate.plan.to_generation
            if (
                entry.revision == records.current.head_revision
                or entry.mutation_sequence != generation + 1
                or entry.value_sequence != generation + 1
                or entry.lifetime_sequence != 1
            ):
                raise ValueError(
                    "CommittedRestartPlanPublication generation head has invalid lineage"
                )
            return
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
        ):
            raise ValueError(
                f"CommittedRestartPlanPublication entry {key!r} is not an immutable creation"
            )

    def _validate_guard(self, key: str, entry: ControlStoreEntry) -> None:
        authority = self.prepared.authority.coordinator_authority
        expected_digest = hashlib.sha256(authority.lease.record.to_json()).hexdigest()
        if (
            entry.guard_key != self.prepared.guard_key
            or entry.guard_revision != self.prepared.expected_guard_revision
            or entry.guard_value_digest != expected_digest
            or entry.guard_committed_at_unix_ms != authority.lease.granted_at_unix_ms
            or entry.guard_mutation_sequence != authority.mutation_sequence
            or entry.guard_value_sequence != authority.value_sequence
            or entry.guard_lifetime_sequence != authority.lifetime_sequence
        ):
            raise ValueError(
                f"CommittedRestartPlanPublication entry {key!r} has invalid lease provenance"
            )

    def _latest_input_transaction_sequence(self) -> int:
        records = self.prepared.authority.records
        manifest_source = records.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot
        transaction_sequences = [
            self.prepared.authority.coordinator_authority.transaction_sequence,
            self.prepared.lifecycle_fence.transaction_sequence,
            records.current.snapshot.transaction_sequence,
            manifest_source.transaction_sequence,
        ]
        for history in records.candidate.placement_state.registration_histories.values():
            registration = history.current
            if registration is None or not history.authorities:
                raise ValueError(
                    "CommittedRestartPlanPublication placement lost a selected registration"
                )
            transaction_sequences.append(history.authorities[-1].transaction_sequence)
        return max(transaction_sequences)

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = next(iter(self.entries.values())).committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated publication lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return next(iter(self.entries.values())).transaction_sequence

    @property
    def generation_head_entry(self) -> ControlStoreEntry:
        return self.entries[self.prepared.authority.records.generation_head_key]

    @property
    def successor_snapshot_entry(self) -> ControlStoreEntry:
        return self.entries[self.prepared.authority.records.successor_generation_snapshot_key]


class RestartPlanPublicationExecutor:
    """Commit and verify one prepared restart-plan publication."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")

    def execute(
        self,
        prepared: PreparedRestartPlanPublication,
    ) -> CommittedRestartPlanPublication:
        """Commit and verify one restart-plan publication."""

        if not isinstance(prepared, PreparedRestartPlanPublication):
            raise TypeError("prepared must be PreparedRestartPlanPublication")
        if prepared.authority.records.candidate.plan.run_id != self._run_id:
            raise ValueError("prepared restart-plan publication belongs to another run")
        try:
            committed = self._store.compare_set_many_guarded(
                prepared.writes,
                guard_key=prepared.guard_key,
                expected_guard_revision=prepared.expected_guard_revision,
                not_before_unix_ms=prepared.not_before_unix_ms,
                deadline_unix_ms=prepared.deadline_unix_ms,
                conditions=prepared.conditions,
            )
        except ControlStoreConflict as error:
            if error.key == prepared.guard_key:
                raise RestartPlanPublicationExecutionLeaseLost(
                    "coordinator lease changed before restart-plan publication"
                ) from error
            registration_keys = set(prepared.authority.records.registration_keys.values())
            if error.key in registration_keys:
                raise RestartPlanPublicationExecutionRegistrationLost(
                    f"selected registration changed at {error.key!r}"
                ) from error
            raise RestartPlanPublicationExecutionConflict(
                f"restart-plan publication state changed at {error.key!r}"
            ) from error
        except ControlStoreHistoryConflict as error:
            raise RestartPlanPublicationExecutionConflict(
                f"restart-plan publication key {error.key!r} has prior history"
            ) from error
        except ControlStoreDeadlineExceeded as error:
            if prepared.deadline_unix_ms == (
                prepared.authority.coordinator_authority.lease.expires_at_unix_ms
            ):
                raise RestartPlanPublicationExecutionLeaseLost(
                    "coordinator lease expired before restart-plan publication"
                ) from error
            if prepared.deadline_unix_ms in _registration_expiries(prepared):
                raise RestartPlanPublicationExecutionRegistrationLost(
                    "selected registration expired before restart-plan publication"
                ) from error
            raise RestartPlanPublicationExecutionDeadlineElapsed(
                "prepared restart-plan publication deadline elapsed before commit"
            ) from error
        except (ControlStoreTooEarly, ControlStoreClockError) as error:
            raise RestartPlanPublicationExecutionClockError(
                "control-store time contradicts restart-plan publication preparation"
            ) from error
        if set(committed) != set(prepared.writes):
            raise RestartPlanPublicationExecutionCorrupt(
                "restart-plan transaction returned an unexpected committed key set"
            )
        try:
            return CommittedRestartPlanPublication(
                prepared=prepared,
                entries=dict(committed),
            )
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationExecutionCorrupt(
                "restart-plan transaction returned contradictory committed state"
            ) from error


def _registration_expiries(prepared: PreparedRestartPlanPublication) -> set[int]:
    expiries: set[int] = set()
    histories = prepared.authority.records.candidate.placement_state.registration_histories
    for history in histories.values():
        registration = history.current
        if registration is None:
            continue
        expiries.add(registration.expires_at_unix_ms)
    return expiries


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "CommittedRestartPlanPublication",
    "RestartPlanPublicationExecutionClockError",
    "RestartPlanPublicationExecutionConflict",
    "RestartPlanPublicationExecutionCorrupt",
    "RestartPlanPublicationExecutionDeadlineElapsed",
    "RestartPlanPublicationExecutionError",
    "RestartPlanPublicationExecutionLeaseLost",
    "RestartPlanPublicationExecutionRegistrationLost",
    "RestartPlanPublicationExecutor",
]
