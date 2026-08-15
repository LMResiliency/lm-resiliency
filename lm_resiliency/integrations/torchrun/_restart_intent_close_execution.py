"""Guarded execution of prepared initial restart-intent closures."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    ControlStoreHistoryConflict,
    ControlStoreTooEarly,
)
from lm_resiliency.integrations.torchrun._restart_intent_close import (
    PreparedInitialRestartIntentClosure,
)


class RestartIntentClosureExecutionError(RuntimeError):
    """Base error for committing a prepared restart-intent closure."""


class RestartIntentClosureExecutionConflict(RestartIntentClosureExecutionError):
    """Raised when opening or lifecycle state changes before commit."""


class RestartIntentClosureExecutionLeaseLost(RestartIntentClosureExecutionError):
    """Raised when the closing coordinator lease changes or expires."""


class RestartIntentClosureExecutionClockError(RestartIntentClosureExecutionError):
    """Raised when authoritative store time contradicts preparation."""


class RestartIntentClosureExecutionCorrupt(RestartIntentClosureExecutionError):
    """Raised when the transaction returns contradictory committed state."""


@dataclass(frozen=True, slots=True)
class CommittedInitialRestartIntentClosure:
    """Verified committed entries for one initial restart-intent closure."""

    prepared: PreparedInitialRestartIntentClosure
    closed_head_entry: ControlStoreEntry
    lifecycle_entry: ControlStoreEntry
    lifecycle_head_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedInitialRestartIntentClosure):
            raise TypeError(
                "CommittedInitialRestartIntentClosure.prepared must be "
                "PreparedInitialRestartIntentClosure"
            )
        entries = (
            ("closed_head_entry", self.closed_head_entry),
            ("lifecycle_entry", self.lifecycle_entry),
            ("lifecycle_head_entry", self.lifecycle_head_entry),
        )
        for path, entry in entries:
            if not isinstance(entry, ControlStoreEntry):
                raise TypeError(
                    f"CommittedInitialRestartIntentClosure.{path} must be ControlStoreEntry"
                )
        records = self.prepared.records
        expected_values = {
            "closed_head_entry": records.closed_head.to_json(),
            "lifecycle_entry": records.lifecycle.to_json(),
            "lifecycle_head_entry": records.lifecycle_head.to_json(),
        }
        for path, entry in entries:
            if entry.value != expected_values[path]:
                raise ValueError(
                    f"CommittedInitialRestartIntentClosure.{path} does not match its preparation"
                )
            self._validate_guard(path, entry)
        self._validate_entry_lineage()
        self._validate_transaction()

    def _validate_guard(self, path: str, entry: ControlStoreEntry) -> None:
        authority = self.prepared.lease_authority
        if (
            entry.guard_key != self.prepared.coordinator_lease_key
            or entry.guard_revision != self.prepared.expected_guard_revision
            or entry.guard_value_digest != self.prepared.records.lifecycle.coordinator_lease_digest
            or entry.guard_committed_at_unix_ms != authority.lease.granted_at_unix_ms
            or entry.guard_mutation_sequence != authority.mutation_sequence
            or entry.guard_value_sequence != authority.value_sequence
            or entry.guard_lifetime_sequence != authority.lifetime_sequence
        ):
            raise ValueError(
                f"CommittedInitialRestartIntentClosure.{path} has invalid lease provenance"
            )
        if entry.committed_at_unix_ms is None:
            raise ValueError(
                f"CommittedInitialRestartIntentClosure.{path} has no authoritative commit time"
            )

    def _validate_entry_lineage(self) -> None:
        opened_head = self.prepared.records.opened.head_entry
        closed_head = self.closed_head_entry
        if (
            closed_head.revision == opened_head.revision
            or closed_head.mutation_sequence != opened_head.mutation_sequence + 1
            or closed_head.value_sequence != opened_head.value_sequence + 1
            or closed_head.lifetime_sequence != opened_head.lifetime_sequence
        ):
            raise ValueError(
                "CommittedInitialRestartIntentClosure closed head is not one in-place update"
            )
        for path, entry in (
            ("lifecycle_entry", self.lifecycle_entry),
            ("lifecycle_head_entry", self.lifecycle_head_entry),
        ):
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
            ):
                raise ValueError(
                    f"CommittedInitialRestartIntentClosure.{path} is not an immutable creation"
                )

    def _validate_transaction(self) -> None:
        entries = (
            self.closed_head_entry,
            self.lifecycle_entry,
            self.lifecycle_head_entry,
        )
        commit_times = {entry.committed_at_unix_ms for entry in entries}
        transaction_sequences = {entry.transaction_sequence for entry in entries}
        if len(commit_times) != 1 or None in commit_times or len(transaction_sequences) != 1:
            raise ValueError(
                "CommittedInitialRestartIntentClosure entries do not share one transaction"
            )
        committed_at_unix_ms = self.closed_head_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated closure lost its commit time")
        if (
            committed_at_unix_ms < self.prepared.not_before_unix_ms
            or committed_at_unix_ms >= self.prepared.deadline_unix_ms
        ):
            raise ValueError(
                "CommittedInitialRestartIntentClosure commit is outside its prepared window"
            )
        if self.closed_head_entry.transaction_sequence <= max(
            self.prepared.records.opened.transaction_sequence,
            self.prepared.coordinator_lease_transaction_sequence,
        ):
            raise ValueError(
                "CommittedInitialRestartIntentClosure does not follow its opening and lease"
            )

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.closed_head_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated closure lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.closed_head_entry.transaction_sequence


class RestartIntentClosureExecutor:
    """Commit and verify one prepared initial restart-intent closure."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")

    def execute_initial_closure(
        self,
        prepared: PreparedInitialRestartIntentClosure,
    ) -> CommittedInitialRestartIntentClosure:
        if not isinstance(prepared, PreparedInitialRestartIntentClosure):
            raise TypeError("prepared must be PreparedInitialRestartIntentClosure")
        if prepared.records.opened.prepared.record.intent.run_id != self._run_id:
            raise ValueError("prepared restart-intent closure belongs to another run")
        try:
            committed = self._store.compare_set_many_guarded(
                prepared.writes,
                guard_key=prepared.coordinator_lease_key,
                expected_guard_revision=prepared.expected_guard_revision,
                not_before_unix_ms=prepared.not_before_unix_ms,
                deadline_unix_ms=prepared.deadline_unix_ms,
                conditions=prepared.conditions,
            )
        except ControlStoreConflict as error:
            if error.key == prepared.coordinator_lease_key:
                raise RestartIntentClosureExecutionLeaseLost(
                    "coordinator lease changed before restart-intent closure"
                ) from error
            raise RestartIntentClosureExecutionConflict(
                f"restart-intent closure state changed at {error.key!r}"
            ) from error
        except ControlStoreHistoryConflict as error:
            raise RestartIntentClosureExecutionConflict(
                f"restart-intent closure key {error.key!r} has prior history"
            ) from error
        except ControlStoreDeadlineExceeded as error:
            raise RestartIntentClosureExecutionLeaseLost(
                "coordinator lease expired before restart-intent closure"
            ) from error
        except (ControlStoreTooEarly, ControlStoreClockError) as error:
            raise RestartIntentClosureExecutionClockError(
                "control-store time contradicts restart-intent closure preparation"
            ) from error
        expected_keys = {
            prepared.records.intent_head_key,
            prepared.records.closure_key,
            prepared.records.lifecycle_head_key,
        }
        if set(committed) != expected_keys:
            raise RestartIntentClosureExecutionCorrupt(
                "restart-intent closure returned an unexpected committed key set"
            )
        try:
            return CommittedInitialRestartIntentClosure(
                prepared=prepared,
                closed_head_entry=committed[prepared.records.intent_head_key],
                lifecycle_entry=committed[prepared.records.closure_key],
                lifecycle_head_entry=committed[prepared.records.lifecycle_head_key],
            )
        except (TypeError, ValueError) as error:
            raise RestartIntentClosureExecutionCorrupt(
                "restart-intent closure returned contradictory committed state"
            ) from error


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "CommittedInitialRestartIntentClosure",
    "RestartIntentClosureExecutionClockError",
    "RestartIntentClosureExecutionConflict",
    "RestartIntentClosureExecutionCorrupt",
    "RestartIntentClosureExecutionError",
    "RestartIntentClosureExecutionLeaseLost",
    "RestartIntentClosureExecutor",
]
