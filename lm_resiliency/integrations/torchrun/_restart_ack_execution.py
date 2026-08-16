"""Guarded execution of prepared restart acknowledgements."""

from __future__ import annotations

import hashlib
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
from lm_resiliency.integrations.torchrun._restart_ack_records import PreparedRestartAckWrite


class RestartAckExecutionError(RuntimeError):
    """Base error for committing a prepared restart acknowledgement."""


class RestartAckExecutionConflict(RestartAckExecutionError):
    """Raised when restart-intent state changed before commit."""


class RestartAckExecutionRegistrationLost(RestartAckExecutionError):
    """Raised when the authenticated agent registration changed or expired."""


class RestartAckExecutionLeaseLost(RestartAckExecutionError):
    """Raised when the coordinator lease changed or expired."""


class RestartAckExecutionDeadlineElapsed(RestartAckExecutionError):
    """Raised when the prepared acknowledgement deadline elapsed."""


class RestartAckExecutionClockError(RestartAckExecutionError):
    """Raised when authoritative store time contradicts preparation."""


class RestartAckExecutionCorrupt(RestartAckExecutionError):
    """Raised when the transaction returns contradictory committed state."""


@dataclass(frozen=True, slots=True)
class CommittedRestartAck:
    """One verified committed restart-acknowledgement receipt."""

    prepared: PreparedRestartAckWrite
    receipt_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        if not isinstance(self.prepared, PreparedRestartAckWrite):
            raise TypeError("CommittedRestartAck.prepared must be PreparedRestartAckWrite")
        if not isinstance(self.receipt_entry, ControlStoreEntry):
            raise TypeError("CommittedRestartAck.receipt_entry must be ControlStoreEntry")
        entry = self.receipt_entry
        if entry.value != self.prepared.records.receipt.to_json():
            raise ValueError("CommittedRestartAck receipt does not match its preparation")
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
        ):
            raise ValueError("CommittedRestartAck receipt is not an immutable creation")
        committed_at_unix_ms = entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise ValueError("CommittedRestartAck receipt has no authoritative commit time")
        authority = self.prepared.coordinator_authority
        expected_guard_digest = hashlib.sha256(authority.lease.record.to_json()).hexdigest()
        if (
            entry.guard_key != self.prepared.coordinator_lease_key
            or entry.guard_revision != self.prepared.expected_guard_revision
            or entry.guard_value_digest != expected_guard_digest
            or entry.guard_committed_at_unix_ms != authority.lease.granted_at_unix_ms
            or entry.guard_mutation_sequence != authority.mutation_sequence
            or entry.guard_value_sequence != authority.value_sequence
            or entry.guard_lifetime_sequence != authority.lifetime_sequence
        ):
            raise ValueError("CommittedRestartAck receipt has invalid lease provenance")
        if (
            committed_at_unix_ms < self.prepared.not_before_unix_ms
            or committed_at_unix_ms >= self.prepared.deadline_unix_ms
        ):
            raise ValueError("CommittedRestartAck commit is outside its prepared time window")
        if entry.transaction_sequence <= max(
            self.prepared.records.opened.transaction_sequence,
            self.prepared.registration_authority.transaction_sequence,
            authority.transaction_sequence,
        ):
            raise ValueError(
                "CommittedRestartAck does not follow its intent, registration, and lease"
            )

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.receipt_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated restart acknowledgement lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.receipt_entry.transaction_sequence


class RestartAckExecutor:
    """Commit and verify one prepared restart acknowledgement."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")

    def execute(self, prepared: PreparedRestartAckWrite) -> CommittedRestartAck:
        """Commit and verify one acknowledgement receipt."""

        if not isinstance(prepared, PreparedRestartAckWrite):
            raise TypeError("prepared must be PreparedRestartAckWrite")
        if prepared.records.receipt.acknowledgement.run_id != self._run_id:
            raise ValueError("prepared restart acknowledgement belongs to another run")
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
                raise RestartAckExecutionLeaseLost(
                    "coordinator lease changed before restart acknowledgement"
                ) from error
            if error.key == prepared.records.agent_registration_key:
                raise RestartAckExecutionRegistrationLost(
                    "agent registration changed before restart acknowledgement"
                ) from error
            raise RestartAckExecutionConflict(
                f"restart-acknowledgement state changed at {error.key!r}"
            ) from error
        except ControlStoreHistoryConflict as error:
            raise RestartAckExecutionConflict(
                f"restart-acknowledgement key {error.key!r} has prior history"
            ) from error
        except ControlStoreDeadlineExceeded as error:
            registration_expiry = prepared.registration.expires_at_unix_ms
            lease_expiry = prepared.lease.expires_at_unix_ms
            if prepared.deadline_unix_ms == registration_expiry:
                raise RestartAckExecutionRegistrationLost(
                    "agent registration expired before restart acknowledgement"
                ) from error
            if prepared.deadline_unix_ms == lease_expiry:
                raise RestartAckExecutionLeaseLost(
                    "coordinator lease expired before restart acknowledgement"
                ) from error
            raise RestartAckExecutionDeadlineElapsed(
                "prepared restart-acknowledgement deadline elapsed before commit"
            ) from error
        except (ControlStoreTooEarly, ControlStoreClockError) as error:
            raise RestartAckExecutionClockError(
                "control-store time contradicts restart-acknowledgement preparation"
            ) from error
        acknowledgement_key = prepared.records.acknowledgement_key
        if set(committed) != {acknowledgement_key}:
            raise RestartAckExecutionCorrupt(
                "restart-acknowledgement transaction returned an unexpected committed key set"
            )
        try:
            return CommittedRestartAck(
                prepared=prepared,
                receipt_entry=committed[acknowledgement_key],
            )
        except (TypeError, ValueError) as error:
            raise RestartAckExecutionCorrupt(
                "restart-acknowledgement transaction returned contradictory committed state"
            ) from error


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "CommittedRestartAck",
    "RestartAckExecutionClockError",
    "RestartAckExecutionConflict",
    "RestartAckExecutionCorrupt",
    "RestartAckExecutionDeadlineElapsed",
    "RestartAckExecutionError",
    "RestartAckExecutionLeaseLost",
    "RestartAckExecutionRegistrationLost",
    "RestartAckExecutor",
]
