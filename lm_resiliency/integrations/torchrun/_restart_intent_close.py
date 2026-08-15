"""Lease authority for closing the first restart intent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)


@dataclass(frozen=True, slots=True)
class PreparedInitialRestartIntentClosure:
    """Lease-fenced inputs for the first restart-intent closure transaction."""

    records: InitialRestartIntentClosureRecords
    lease_authority_chain: tuple[CoordinatorLeaseAuthority, ...]
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, InitialRestartIntentClosureRecords):
            raise TypeError(
                "PreparedInitialRestartIntentClosure.records must be "
                "InitialRestartIntentClosureRecords"
            )
        if not isinstance(self.lease_authority_chain, tuple):
            raise TypeError(
                "PreparedInitialRestartIntentClosure.lease_authority_chain must be tuple"
            )
        if not self.lease_authority_chain:
            raise ValueError(
                "PreparedInitialRestartIntentClosure.lease_authority_chain must not be empty"
            )
        if not all(
            isinstance(authority, CoordinatorLeaseAuthority)
            for authority in self.lease_authority_chain
        ):
            raise TypeError(
                "PreparedInitialRestartIntentClosure.lease_authority_chain must contain "
                "CoordinatorLeaseAuthority values"
            )
        _positive_integer(
            self.not_before_unix_ms,
            "PreparedInitialRestartIntentClosure.not_before_unix_ms",
        )
        _positive_integer(
            self.deadline_unix_ms,
            "PreparedInitialRestartIntentClosure.deadline_unix_ms",
        )
        self._validate_lease_lineage()
        lifecycle = self.records.lifecycle
        if (
            self.lease.record.run_id != lifecycle.closed_intent.run_id
            or lifecycle.coordinator_id != self.lease.record.coordinator_id
            or lifecycle.lease_id != self.lease.record.lease_id
            or lifecycle.coordinator_lease_duration_ms != self.lease.record.lease_duration_ms
            or lifecycle.coordinator_fencing_token != self.lease.fencing_token
        ):
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease does not authorize its lifecycle"
            )
        if self.not_before_unix_ms < self.records.opened.committed_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentClosure cannot precede its open transaction"
            )
        if self.not_before_unix_ms < self.lease.granted_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentClosure cannot precede its coordinator lease grant"
            )
        if self.not_before_unix_ms >= self.deadline_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentClosure.not_before_unix_ms must precede its deadline"
            )
        if self.deadline_unix_ms > self.lease.expires_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentClosure deadline exceeds its coordinator lease"
            )

    def _validate_lease_lineage(self) -> None:
        opening = self.records.opened.prepared
        expected_opening = CoordinatorLeaseAuthority(
            lease=opening.lease,
            transaction_sequence=opening.coordinator_lease_transaction_sequence,
            mutation_sequence=opening.coordinator_lease_mutation_sequence,
            value_sequence=opening.coordinator_lease_value_sequence,
            lifetime_sequence=opening.coordinator_lease_lifetime_sequence,
        )
        if self.lease_authority_chain[0] != expected_opening:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease chain does not begin "
                "with its opening authority"
            )
        lease_ids = list(opening.generation_lease_id_history)
        fencing_tokens = list(opening.generation_fencing_token_history)
        for authority in self.lease_authority_chain:
            lease_ids.append(authority.lease.record.lease_id)
            fencing_tokens.append(authority.lease.fencing_token)
        _reject_noncontiguous_recurrence(
            tuple(lease_ids),
            "lease identity",
        )
        _reject_noncontiguous_recurrence(
            tuple(fencing_tokens),
            "fencing token",
        )
        for previous, current in zip(
            self.lease_authority_chain,
            self.lease_authority_chain[1:],
            strict=False,
        ):
            self._validate_lease_transition(previous, current)

    def _validate_lease_transition(
        self,
        previous: CoordinatorLeaseAuthority,
        current: CoordinatorLeaseAuthority,
    ) -> None:
        opened = self.records.opened
        if (
            current.transaction_sequence <= opened.transaction_sequence
            or current.transaction_sequence <= previous.transaction_sequence
        ):
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease mutation does not follow "
                "its open and predecessor"
            )
        mutation_delta = current.mutation_sequence - previous.mutation_sequence
        transaction_delta = current.transaction_sequence - previous.transaction_sequence
        value_delta = current.value_sequence - previous.value_sequence
        lifetime_delta = current.lifetime_sequence - previous.lifetime_sequence
        if transaction_delta < mutation_delta:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease transition has impossible "
                "transaction ordering"
            )
        if current.lease.granted_at_unix_ms < previous.lease.granted_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease transition moves backward in time"
            )
        if current.lease.fencing_token == previous.lease.fencing_token:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease transition reuses its fencing token"
            )
        if current.lease.record == previous.lease.record:
            if mutation_delta != 1 or value_delta != 0 or lifetime_delta != 0:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure renewal is not one lease-key mutation"
                )
            if current.lease.granted_at_unix_ms >= previous.lease.expires_at_unix_ms:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure renews an expired coordinator lease"
                )
            return
        if current.lease.record.lease_id == previous.lease.record.lease_id:
            raise ValueError(
                "PreparedInitialRestartIntentClosure changes one lease identity's record"
            )
        if lifetime_delta == 0:
            if mutation_delta != 1 or value_delta != 1:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure in-place replacement is not "
                    "one lease-key mutation"
                )
            if current.lease.granted_at_unix_ms < previous.lease.expires_at_unix_ms:
                raise ValueError("PreparedInitialRestartIntentClosure coordinator leases overlap")
            return
        if lifetime_delta == 1:
            if mutation_delta != 2 or value_delta != 1:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure recreated lease has invalid lineage"
                )
            return
        raise ValueError(
            "PreparedInitialRestartIntentClosure lease chain omits one or more "
            "replacement authorities"
        )

    @property
    def lease_authority(self) -> CoordinatorLeaseAuthority:
        return self.lease_authority_chain[-1]

    @property
    def lease(self) -> HeldCoordinatorLease:
        return self.lease_authority.lease

    @property
    def coordinator_lease_transaction_sequence(self) -> int:
        return self.lease_authority.transaction_sequence

    @property
    def coordinator_lease_mutation_sequence(self) -> int:
        return self.lease_authority.mutation_sequence

    @property
    def coordinator_lease_value_sequence(self) -> int:
        return self.lease_authority.value_sequence

    @property
    def coordinator_lease_lifetime_sequence(self) -> int:
        return self.lease_authority.lifetime_sequence

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


def _reject_noncontiguous_recurrence(values: tuple[object, ...], label: str) -> None:
    seen: set[object] = set()
    previous: object | None = None
    for value in values:
        if value != previous:
            if value in seen:
                raise ValueError(
                    f"PreparedInitialRestartIntentClosure {label} reappears after replacement"
                )
            seen.add(value)
        previous = value


__all__ = ["PreparedInitialRestartIntentClosure"]
