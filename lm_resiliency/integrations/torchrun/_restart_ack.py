"""Coordinator-authorized restart acknowledgement transaction inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_ack_writes import (
    RestartAckWriteRecords,
)


@dataclass(frozen=True, slots=True)
class PreparedRestartAckWrite:
    """One immutable acknowledgement write with coordinator authority."""

    records: RestartAckWriteRecords
    coordinator_authority: CoordinatorLeaseAuthority
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, RestartAckWriteRecords):
            raise TypeError("PreparedRestartAckWrite.records must be RestartAckWriteRecords")
        if not isinstance(self.coordinator_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "PreparedRestartAckWrite.coordinator_authority must be CoordinatorLeaseAuthority"
            )
        _positive_integer(
            self.not_before_unix_ms,
            "PreparedRestartAckWrite.not_before_unix_ms",
        )
        _positive_integer(
            self.deadline_unix_ms,
            "PreparedRestartAckWrite.deadline_unix_ms",
        )
        receipt = self.records.receipt
        lease = self.coordinator_authority.lease
        if lease.record.run_id != receipt.acknowledgement.run_id:
            raise ValueError("PreparedRestartAckWrite coordinator lease belongs to another run")
        if self.not_before_unix_ms < receipt.received_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite cannot precede acknowledgement receipt")
        if self.not_before_unix_ms < lease.granted_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite cannot precede coordinator lease grant")
        if self.not_before_unix_ms >= self.deadline_unix_ms:
            raise ValueError("PreparedRestartAckWrite.not_before_unix_ms must precede its deadline")
        if self.deadline_unix_ms > lease.expires_at_unix_ms:
            raise ValueError("PreparedRestartAckWrite deadline exceeds coordinator lease")
        if self.deadline_unix_ms > receipt.intent_record.intent.prepare_deadline_unix_ms:
            raise ValueError("PreparedRestartAckWrite deadline exceeds restart intent")

    @property
    def lease(self) -> HeldCoordinatorLease:
        return self.coordinator_authority.lease

    @property
    def coordinator_lease_key(self) -> str:
        return self.records.opened.prepared.coordinator_lease_key

    @property
    def expected_guard_revision(self) -> int:
        return self.lease.fencing_token

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return self.records.writes

    @property
    def conditions(self) -> Mapping[str, int]:
        return self.records.conditions


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = ["PreparedRestartAckWrite"]
