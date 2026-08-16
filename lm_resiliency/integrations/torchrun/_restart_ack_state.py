"""Authenticated restart-acknowledgement dependency values."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedRestartAckState:
    """One receipt bound to current intent, registration, and lease state."""

    receipt: RestartAckReceiptRecord
    opened: CommittedInitialRestartIntentOpen
    registration_authority: AgentRegistrationAuthority
    coordinator_authority: CoordinatorLeaseAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RestartAckReceiptRecord):
            raise TypeError("AuthenticatedRestartAckState.receipt must be RestartAckReceiptRecord")
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError(
                "AuthenticatedRestartAckState.opened must be CommittedInitialRestartIntentOpen"
            )
        if not isinstance(self.registration_authority, AgentRegistrationAuthority):
            raise TypeError(
                "AuthenticatedRestartAckState.registration_authority must be "
                "AgentRegistrationAuthority"
            )
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "AuthenticatedRestartAckState.coordinator_authority must be "
                "CoordinatorLeaseAuthority"
            )
        if self.receipt.intent_record != self.opened.prepared.record:
            raise ValueError(
                "AuthenticatedRestartAckState receipt does not answer its current intent"
            )
        if self.receipt.received_at_unix_ms < self.opened.committed_at_unix_ms:
            raise ValueError("AuthenticatedRestartAckState receipt predates its current intent")
        if self.registration != self.receipt.authenticated_registration:
            raise ValueError(
                "AuthenticatedRestartAckState receipt does not match its current registration"
            )
        run_id = self.receipt.acknowledgement.run_id
        if self.coordinator_authority.lease.record.run_id != run_id:
            raise ValueError(
                "AuthenticatedRestartAckState coordinator lease belongs to another run"
            )
        active_node_ids = set(
            self.opened.prepared.current.snapshot.record.assignment.slot_to_node_id.values()
        )
        if self.receipt.acknowledgement.node_id not in active_node_ids:
            raise ValueError("AuthenticatedRestartAckState acknowledgement node is not active")

    @property
    def registration(self) -> HeldAgentRegistration:
        """Return the authenticated held registration."""

        return self.registration_authority.registration


__all__ = ["AuthenticatedRestartAckState"]
