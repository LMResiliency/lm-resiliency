"""Store-stamped agent-registration authority values."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._agent_registration_records import (
    AgentRegistrationRecord,
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreEntry


class AgentRegistrationAuthorityCorrupt(RuntimeError):
    """Raised when one persisted registration authority is invalid."""


@dataclass(frozen=True, slots=True)
class AgentRegistrationAuthority:
    """One canonical, store-stamped agent-registration value."""

    registration: HeldAgentRegistration
    transaction_sequence: int
    mutation_sequence: int
    value_sequence: int
    lifetime_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.registration, HeldAgentRegistration):
            raise TypeError("AgentRegistrationAuthority.registration must be HeldAgentRegistration")
        for path, value in (
            ("transaction_sequence", self.transaction_sequence),
            ("mutation_sequence", self.mutation_sequence),
            ("value_sequence", self.value_sequence),
            ("lifetime_sequence", self.lifetime_sequence),
        ):
            _positive_integer(value, f"AgentRegistrationAuthority.{path}")
        _validate_sequence_lineage(
            self.mutation_sequence,
            self.value_sequence,
            self.lifetime_sequence,
        )
        if self.transaction_sequence < self.mutation_sequence:
            raise ValueError(
                "AgentRegistrationAuthority.transaction_sequence is too small for mutation_sequence"
            )

    @classmethod
    def from_entry(
        cls,
        entry: ControlStoreEntry,
        *,
        run_id: str,
        node_id: str,
    ) -> AgentRegistrationAuthority:
        """Decode one authoritative agent-registration store entry."""

        if not isinstance(entry, ControlStoreEntry):
            raise TypeError("entry must be ControlStoreEntry")
        normalized_run_id = _nonempty_string(run_id, "run_id")
        normalized_node_id = _nonempty_string(node_id, "node_id")
        try:
            record = AgentRegistrationRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise AgentRegistrationAuthorityCorrupt(
                "agent-registration authority contains a malformed record"
            ) from error
        identity = record.agent_identity
        if identity.run_id != normalized_run_id or identity.node_id != normalized_node_id:
            raise AgentRegistrationAuthorityCorrupt(
                "agent-registration authority belongs to another run or node"
            )
        if entry.value != record.to_json():
            raise AgentRegistrationAuthorityCorrupt(
                "agent-registration authority contains noncanonical record bytes"
            )
        if entry.committed_at_unix_ms is None:
            raise AgentRegistrationAuthorityCorrupt(
                "agent-registration authority has no authoritative commit time"
            )
        if entry.guard_key is not None:
            raise AgentRegistrationAuthorityCorrupt(
                "agent-registration authority unexpectedly carries guard provenance"
            )
        return cls(
            registration=HeldAgentRegistration(
                record=record,
                fencing_token=entry.revision,
                granted_at_unix_ms=entry.committed_at_unix_ms,
            ),
            transaction_sequence=entry.transaction_sequence,
            mutation_sequence=entry.mutation_sequence,
            value_sequence=entry.value_sequence,
            lifetime_sequence=entry.lifetime_sequence,
        )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _validate_sequence_lineage(
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
) -> None:
    if mutation_sequence < 2 * lifetime_sequence - 1:
        raise ValueError(
            "AgentRegistrationAuthority.mutation_sequence is too small for lifetime_sequence"
        )
    if value_sequence < lifetime_sequence:
        raise ValueError(
            "AgentRegistrationAuthority.value_sequence is too small for lifetime_sequence"
        )
    if value_sequence > mutation_sequence - lifetime_sequence + 1:
        raise ValueError(
            "AgentRegistrationAuthority.value_sequence is too large for mutation_sequence"
        )


__all__ = [
    "AgentRegistrationAuthority",
    "AgentRegistrationAuthorityCorrupt",
]
