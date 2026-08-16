"""Canonical persisted restart-acknowledgement state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreEntry
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
    PersistedInitialRestartIntentOpen,
    RestartIntentOpening,
)


@dataclass(frozen=True, slots=True)
class PersistedRestartAck:
    """One committed receipt authenticated against durable dependencies."""

    receipt: RestartAckReceiptRecord
    receipt_entry: ControlStoreEntry
    opened: RestartIntentOpening
    registration_authority: AgentRegistrationAuthority
    coordinator_authority: CoordinatorLeaseAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RestartAckReceiptRecord):
            raise TypeError("PersistedRestartAck.receipt must be RestartAckReceiptRecord")
        if not isinstance(self.receipt_entry, ControlStoreEntry):
            raise TypeError("PersistedRestartAck.receipt_entry must be ControlStoreEntry")
        if not isinstance(
            self.opened,
            (CommittedInitialRestartIntentOpen, PersistedInitialRestartIntentOpen),
        ):
            raise TypeError("PersistedRestartAck.opened must be a restart-intent opening")
        if not isinstance(self.registration_authority, AgentRegistrationAuthority):
            raise TypeError(
                "PersistedRestartAck.registration_authority must be AgentRegistrationAuthority"
            )
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "PersistedRestartAck.coordinator_authority must be CoordinatorLeaseAuthority"
            )
        self._validate_records()
        self._validate_entry()

    @classmethod
    def from_entry(
        cls,
        *,
        run_id: str,
        receipt_entry: ControlStoreEntry,
        opened: RestartIntentOpening,
        registration_authority: AgentRegistrationAuthority,
        coordinator_authority: CoordinatorLeaseAuthority,
    ) -> PersistedRestartAck:
        """Decode and authenticate one persisted acknowledgement receipt."""

        normalized_run_id = _nonempty_string(run_id, "run_id")
        if not isinstance(receipt_entry, ControlStoreEntry):
            raise TypeError("receipt_entry must be ControlStoreEntry")
        try:
            receipt = RestartAckReceiptRecord.from_json(receipt_entry.value)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted restart acknowledgement is malformed") from error
        persisted = cls(
            receipt=receipt,
            receipt_entry=receipt_entry,
            opened=opened,
            registration_authority=registration_authority,
            coordinator_authority=coordinator_authority,
        )
        if persisted.receipt.acknowledgement.run_id != normalized_run_id:
            raise ValueError("persisted restart acknowledgement belongs to another run")
        return persisted

    @property
    def committed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.receipt_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated restart acknowledgement lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.receipt_entry.transaction_sequence

    def _validate_records(self) -> None:
        if self.receipt.intent_record != self.opened.record:
            raise ValueError("PersistedRestartAck does not answer its committed restart intent")
        if self.receipt.received_at_unix_ms < self.opened.committed_at_unix_ms:
            raise ValueError("PersistedRestartAck receipt predates its committed restart intent")
        if self.receipt.authenticated_registration != self.registration_authority.registration:
            raise ValueError("PersistedRestartAck does not match its agent-registration authority")
        run_id = self.receipt.acknowledgement.run_id
        if self.coordinator_authority.lease.record.run_id != run_id:
            raise ValueError("PersistedRestartAck coordinator authority belongs to another run")
        active_node_ids = set(
            self.opened.generation_snapshot.record.assignment.slot_to_node_id.values()
        )
        if self.receipt.acknowledgement.node_id not in active_node_ids:
            raise ValueError("PersistedRestartAck acknowledgement node is not active")

    def _validate_entry(self) -> None:
        entry = self.receipt_entry
        if entry.value != self.receipt.to_json():
            raise ValueError("PersistedRestartAck receipt entry is noncanonical")
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
        ):
            raise ValueError("PersistedRestartAck receipt entry is not an immutable creation")
        if entry.committed_at_unix_ms is None:
            raise ValueError("PersistedRestartAck receipt entry has no authoritative commit time")
        authority = self.coordinator_authority
        expected_guard_digest = hashlib.sha256(authority.lease.record.to_json()).hexdigest()
        if (
            entry.guard_key != self.opened.coordinator_lease_key
            or entry.guard_revision != authority.lease.fencing_token
            or entry.guard_value_digest != expected_guard_digest
            or entry.guard_committed_at_unix_ms != authority.lease.granted_at_unix_ms
            or entry.guard_mutation_sequence != authority.mutation_sequence
            or entry.guard_value_sequence != authority.value_sequence
            or entry.guard_lifetime_sequence != authority.lifetime_sequence
        ):
            raise ValueError("PersistedRestartAck receipt entry has invalid lease provenance")
        if self.transaction_sequence <= max(
            self.opened.transaction_sequence,
            self.registration_authority.transaction_sequence,
            authority.transaction_sequence,
        ):
            raise ValueError(
                "PersistedRestartAck does not follow its intent, registration, and lease"
            )
        lower_bound = max(
            self.receipt.received_at_unix_ms,
            self.opened.committed_at_unix_ms,
            self.registration_authority.registration.granted_at_unix_ms,
            authority.lease.granted_at_unix_ms,
        )
        upper_bound = min(
            self.receipt.intent_record.intent.prepare_deadline_unix_ms,
            self.registration_authority.registration.expires_at_unix_ms,
            authority.lease.expires_at_unix_ms,
        )
        if self.committed_at_unix_ms < lower_bound or self.committed_at_unix_ms >= upper_bound:
            raise ValueError("PersistedRestartAck commit is outside its authority window")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = ["PersistedRestartAck"]
