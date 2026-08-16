"""Guarded execution of prepared initial restart-intent openings."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TypeAlias

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    ControlStoreHistoryConflict,
    ControlStoreTooEarly,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_records import (
    PreparedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


class RestartIntentOpenExecutionError(RuntimeError):
    """Base error for committing a prepared initial restart intent."""


class RestartIntentOpenExecutionConflict(RestartIntentOpenExecutionError):
    """Raised when generation or lifecycle state changed before commit."""


class RestartIntentOpenExecutionLeaseLost(RestartIntentOpenExecutionError):
    """Raised when the coordinator lease changed or expired before commit."""


class RestartIntentOpenExecutionDeadlineElapsed(RestartIntentOpenExecutionError):
    """Raised when the restart-intent preparation deadline elapsed."""


class RestartIntentOpenExecutionClockError(RestartIntentOpenExecutionError):
    """Raised when authoritative store time contradicts preparation time."""


class RestartIntentOpenExecutionCorrupt(RestartIntentOpenExecutionError):
    """Raised when the transaction returns contradictory committed state."""


@dataclass(frozen=True, slots=True)
class PersistedInitialRestartIntentOpen:
    """One durable initial opening reconstructed without preparation inputs."""

    record: RestartIntentRecord
    head: RestartIntentHeadRecord
    generation_snapshot: StoredGenerationSnapshot
    intent_entry: ControlStoreEntry
    head_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        if not isinstance(self.record, RestartIntentRecord):
            raise TypeError("PersistedInitialRestartIntentOpen.record must be RestartIntentRecord")
        if not isinstance(self.head, RestartIntentHeadRecord):
            raise TypeError(
                "PersistedInitialRestartIntentOpen.head must be RestartIntentHeadRecord"
            )
        if not isinstance(self.generation_snapshot, StoredGenerationSnapshot):
            raise TypeError(
                "PersistedInitialRestartIntentOpen.generation_snapshot must be "
                "StoredGenerationSnapshot"
            )
        if not isinstance(self.intent_entry, ControlStoreEntry):
            raise TypeError(
                "PersistedInitialRestartIntentOpen.intent_entry must be ControlStoreEntry"
            )
        if not isinstance(self.head_entry, ControlStoreEntry):
            raise TypeError(
                "PersistedInitialRestartIntentOpen.head_entry must be ControlStoreEntry"
            )
        if (
            self.head.run_id != self.record.intent.run_id
            or self.head.generation != self.record.intent.generation
            or self.head.intent_id != self.record.intent.intent_id
            or self.head.intent_digest != self.record.digest
        ):
            raise ValueError("PersistedInitialRestartIntentOpen head does not identify its intent")
        snapshot_record = self.generation_snapshot.record
        if (
            snapshot_record.assignment.run_id != self.record.intent.run_id
            or snapshot_record.assignment.generation != self.record.intent.generation
            or snapshot_record.digest != self.record.generation_snapshot_digest
        ):
            raise ValueError(
                "PersistedInitialRestartIntentOpen source generation does not identify its intent"
            )
        if self.intent_entry.value != self.record.to_json():
            raise ValueError("PersistedInitialRestartIntentOpen intent entry is noncanonical")
        if self.head_entry.value != self.head.to_json():
            raise ValueError("PersistedInitialRestartIntentOpen head entry is noncanonical")
        for path, entry in (
            ("intent_entry", self.intent_entry),
            ("head_entry", self.head_entry),
        ):
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
            ):
                raise ValueError(
                    f"PersistedInitialRestartIntentOpen.{path} is not an immutable creation"
                )
            if entry.committed_at_unix_ms is None:
                raise ValueError(
                    f"PersistedInitialRestartIntentOpen.{path} has no authoritative commit time"
                )
            if (
                entry.guard_key != self.coordinator_lease_key
                or entry.guard_revision != self.record.coordinator_fencing_token
                or entry.guard_value_digest != self.record.coordinator_lease_digest
                or entry.guard_committed_at_unix_ms is None
                or entry.guard_mutation_sequence is None
                or entry.guard_value_sequence is None
                or entry.guard_lifetime_sequence is None
            ):
                raise ValueError(
                    f"PersistedInitialRestartIntentOpen.{path} has invalid lease provenance"
                )
        if (
            self.intent_entry.transaction_sequence != self.head_entry.transaction_sequence
            or self.intent_entry.committed_at_unix_ms != self.head_entry.committed_at_unix_ms
            or self.intent_entry.guard_committed_at_unix_ms
            != self.head_entry.guard_committed_at_unix_ms
            or self.intent_entry.guard_mutation_sequence != self.head_entry.guard_mutation_sequence
            or self.intent_entry.guard_value_sequence != self.head_entry.guard_value_sequence
            or self.intent_entry.guard_lifetime_sequence != self.head_entry.guard_lifetime_sequence
        ):
            raise ValueError(
                "PersistedInitialRestartIntentOpen entries do not share one transaction"
            )
        if self.transaction_sequence <= self.generation_snapshot.transaction_sequence:
            raise ValueError(
                "PersistedInitialRestartIntentOpen does not follow its source generation"
            )
        lease_granted_at_unix_ms = self.intent_entry.guard_committed_at_unix_ms
        if lease_granted_at_unix_ms is None:
            raise AssertionError("validated opening lost its lease grant time")
        if (
            self.committed_at_unix_ms < self.generation_snapshot.committed_at_unix_ms
            or self.committed_at_unix_ms < lease_granted_at_unix_ms
            or self.committed_at_unix_ms
            >= min(
                lease_granted_at_unix_ms + self.record.coordinator_lease_duration_ms,
                self.record.intent.prepare_deadline_unix_ms,
            )
        ):
            raise ValueError(
                "PersistedInitialRestartIntentOpen commit is outside its authority window"
            )

    @classmethod
    def from_committed(
        cls,
        opened: CommittedInitialRestartIntentOpen,
    ) -> PersistedInitialRestartIntentOpen:
        """Project one executor result into its durable identity."""

        if not isinstance(opened, CommittedInitialRestartIntentOpen):
            raise TypeError("opened must be CommittedInitialRestartIntentOpen")
        return cls(
            record=opened.prepared.record,
            head=opened.prepared.head,
            generation_snapshot=opened.prepared.current.snapshot,
            intent_entry=opened.intent_entry,
            head_entry=opened.head_entry,
        )

    @property
    def run_id(self) -> str:
        return self.record.intent.run_id

    @property
    def intent_key(self) -> str:
        run_digest = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()
        intent_digest = hashlib.sha256(self.record.intent.intent_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}/restart-intents/{intent_digest}"

    @property
    def intent_head_key(self) -> str:
        run_digest = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}/restart-intent-head"

    @property
    def coordinator_lease_key(self) -> str:
        run_digest = hashlib.sha256(self.run_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}/coordinator-lease"

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.intent_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated opening lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.intent_entry.transaction_sequence


@dataclass(frozen=True, slots=True)
class CommittedInitialRestartIntentOpen:
    """Verified committed entries for one initial restart-intent opening."""

    prepared: PreparedInitialRestartIntentOpen
    intent_entry: ControlStoreEntry
    head_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedInitialRestartIntentOpen):
            raise TypeError(
                "CommittedInitialRestartIntentOpen.prepared must be "
                "PreparedInitialRestartIntentOpen"
            )
        if not isinstance(self.intent_entry, ControlStoreEntry):
            raise TypeError(
                "CommittedInitialRestartIntentOpen.intent_entry must be ControlStoreEntry"
            )
        if not isinstance(self.head_entry, ControlStoreEntry):
            raise TypeError(
                "CommittedInitialRestartIntentOpen.head_entry must be ControlStoreEntry"
            )
        if self.intent_entry.value != self.prepared.record.to_json():
            raise ValueError(
                "CommittedInitialRestartIntentOpen intent entry does not match its preparation"
            )
        if self.head_entry.value != self.prepared.head.to_json():
            raise ValueError(
                "CommittedInitialRestartIntentOpen head entry does not match its preparation"
            )
        for path, entry in (
            ("intent_entry", self.intent_entry),
            ("head_entry", self.head_entry),
        ):
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
            ):
                raise ValueError(
                    f"CommittedInitialRestartIntentOpen.{path} is not an immutable creation"
                )
            if entry.committed_at_unix_ms is None:
                raise ValueError(
                    f"CommittedInitialRestartIntentOpen.{path} has no authoritative commit time"
                )
            if (
                entry.guard_key != self.prepared.coordinator_lease_key
                or entry.guard_revision != self.prepared.expected_guard_revision
                or entry.guard_value_digest != self.prepared.record.coordinator_lease_digest
                or entry.guard_committed_at_unix_ms != self.prepared.lease.granted_at_unix_ms
                or entry.guard_mutation_sequence
                != self.prepared.coordinator_lease_mutation_sequence
                or entry.guard_value_sequence != self.prepared.coordinator_lease_value_sequence
                or entry.guard_lifetime_sequence
                != self.prepared.coordinator_lease_lifetime_sequence
            ):
                raise ValueError(
                    f"CommittedInitialRestartIntentOpen.{path} has invalid lease provenance"
                )
        intent_committed_at_unix_ms = self.intent_entry.committed_at_unix_ms
        head_committed_at_unix_ms = self.head_entry.committed_at_unix_ms
        if intent_committed_at_unix_ms is None or head_committed_at_unix_ms is None:
            raise ValueError(
                "CommittedInitialRestartIntentOpen entries have no authoritative commit time"
            )
        if (
            intent_committed_at_unix_ms != head_committed_at_unix_ms
            or self.intent_entry.transaction_sequence != self.head_entry.transaction_sequence
            or self.intent_entry.guard_mutation_sequence != self.head_entry.guard_mutation_sequence
            or self.intent_entry.guard_value_sequence != self.head_entry.guard_value_sequence
            or self.intent_entry.guard_lifetime_sequence != self.head_entry.guard_lifetime_sequence
        ):
            raise ValueError(
                "CommittedInitialRestartIntentOpen entries do not share one transaction"
            )
        if (
            intent_committed_at_unix_ms < self.prepared.not_before_unix_ms
            or intent_committed_at_unix_ms >= self.prepared.deadline_unix_ms
        ):
            raise ValueError(
                "CommittedInitialRestartIntentOpen commit is outside its prepared time window"
            )
        if self.intent_entry.transaction_sequence <= max(
            self.prepared.current.snapshot.transaction_sequence,
            self.prepared.coordinator_lease_transaction_sequence,
        ):
            raise ValueError(
                "CommittedInitialRestartIntentOpen does not follow its generation and lease"
            )

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.intent_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated committed result lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.intent_entry.transaction_sequence

    @property
    def record(self) -> RestartIntentRecord:
        return self.prepared.record

    @property
    def head(self) -> RestartIntentHeadRecord:
        return self.prepared.head

    @property
    def generation_snapshot(self) -> StoredGenerationSnapshot:
        return self.prepared.current.snapshot

    @property
    def intent_key(self) -> str:
        return self.prepared.intent_key

    @property
    def intent_head_key(self) -> str:
        return self.prepared.intent_head_key

    @property
    def coordinator_lease_key(self) -> str:
        return self.prepared.coordinator_lease_key


RestartIntentOpening: TypeAlias = (
    CommittedInitialRestartIntentOpen | PersistedInitialRestartIntentOpen
)


class RestartIntentOpenExecutor:
    """Commit and verify one prepared initial restart-intent opening."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")

    def execute_initial_open(
        self,
        prepared: PreparedInitialRestartIntentOpen,
    ) -> CommittedInitialRestartIntentOpen:
        if not isinstance(prepared, PreparedInitialRestartIntentOpen):
            raise TypeError("prepared must be PreparedInitialRestartIntentOpen")
        if prepared.record.intent.run_id != self._run_id:
            raise ValueError("prepared restart intent belongs to another run")
        try:
            committed = self._store.compare_set_many_guarded(
                prepared.writes,
                guard_key=prepared.coordinator_lease_key,
                expected_guard_revision=prepared.expected_guard_revision,
                not_before_unix_ms=prepared.not_before_unix_ms,
                deadline_unix_ms=prepared.deadline_unix_ms,
                conditions=prepared.conditions,
                never_created_conditions=prepared.never_created_conditions,
            )
        except ControlStoreConflict as error:
            if error.key == prepared.coordinator_lease_key:
                raise RestartIntentOpenExecutionLeaseLost(
                    "coordinator lease changed before restart-intent commit"
                ) from error
            raise RestartIntentOpenExecutionConflict(
                f"restart-intent transaction state changed at {error.key!r}"
            ) from error
        except ControlStoreHistoryConflict as error:
            raise RestartIntentOpenExecutionConflict(
                f"restart-intent transaction key {error.key!r} has prior history"
            ) from error
        except ControlStoreDeadlineExceeded as error:
            if prepared.lease.expires_at_unix_ms <= prepared.record.intent.prepare_deadline_unix_ms:
                raise RestartIntentOpenExecutionLeaseLost(
                    "coordinator lease expired before restart-intent commit"
                ) from error
            raise RestartIntentOpenExecutionDeadlineElapsed(
                "restart-intent preparation deadline elapsed before commit"
            ) from error
        except (ControlStoreTooEarly, ControlStoreClockError) as error:
            raise RestartIntentOpenExecutionClockError(
                "control-store time contradicts restart-intent preparation"
            ) from error
        expected_keys = {prepared.intent_key, prepared.intent_head_key}
        if set(committed) != expected_keys:
            raise RestartIntentOpenExecutionCorrupt(
                "restart-intent transaction returned an unexpected committed key set"
            )
        try:
            return CommittedInitialRestartIntentOpen(
                prepared=prepared,
                intent_entry=committed[prepared.intent_key],
                head_entry=committed[prepared.intent_head_key],
            )
        except (TypeError, ValueError) as error:
            raise RestartIntentOpenExecutionCorrupt(
                "restart-intent transaction returned contradictory committed state"
            ) from error


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "CommittedInitialRestartIntentOpen",
    "PersistedInitialRestartIntentOpen",
    "RestartIntentOpening",
    "RestartIntentOpenExecutionClockError",
    "RestartIntentOpenExecutionConflict",
    "RestartIntentOpenExecutionCorrupt",
    "RestartIntentOpenExecutionDeadlineElapsed",
    "RestartIntentOpenExecutionError",
    "RestartIntentOpenExecutionLeaseLost",
    "RestartIntentOpenExecutor",
]
