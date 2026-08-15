"""Lease authority for closing the first restart intent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)


@dataclass(frozen=True, slots=True)
class PreparedInitialRestartIntentClosure:
    """Lease-fenced inputs for the first restart-intent closure transaction."""

    records: InitialRestartIntentClosureRecords
    lease: HeldCoordinatorLease
    coordinator_lease_transaction_sequence: int
    coordinator_lease_mutation_sequence: int
    coordinator_lease_value_sequence: int
    coordinator_lease_lifetime_sequence: int
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.records, InitialRestartIntentClosureRecords):
            raise TypeError(
                "PreparedInitialRestartIntentClosure.records must be "
                "InitialRestartIntentClosureRecords"
            )
        if not isinstance(self.lease, HeldCoordinatorLease):
            raise TypeError(
                "PreparedInitialRestartIntentClosure.lease must be HeldCoordinatorLease"
            )
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
        for path, value in (
            (
                "coordinator_lease_transaction_sequence",
                self.coordinator_lease_transaction_sequence,
            ),
            (
                "coordinator_lease_mutation_sequence",
                self.coordinator_lease_mutation_sequence,
            ),
            (
                "coordinator_lease_value_sequence",
                self.coordinator_lease_value_sequence,
            ),
            (
                "coordinator_lease_lifetime_sequence",
                self.coordinator_lease_lifetime_sequence,
            ),
            ("not_before_unix_ms", self.not_before_unix_ms),
            ("deadline_unix_ms", self.deadline_unix_ms),
        ):
            _positive_integer(value, f"PreparedInitialRestartIntentClosure.{path}")
        self._validate_lease_lineage()
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
        opened = self.records.opened
        opening = opened.prepared
        if self.lease.fencing_token == opening.lease.fencing_token:
            if (
                self.lease != opening.lease
                or self.coordinator_lease_transaction_sequence
                != opening.coordinator_lease_transaction_sequence
                or self.coordinator_lease_mutation_sequence
                != opening.coordinator_lease_mutation_sequence
                or self.coordinator_lease_value_sequence != opening.coordinator_lease_value_sequence
                or self.coordinator_lease_lifetime_sequence
                != opening.coordinator_lease_lifetime_sequence
            ):
                raise ValueError(
                    "PreparedInitialRestartIntentClosure changes one fencing token's authority"
                )
            return
        if self.coordinator_lease_transaction_sequence <= opened.transaction_sequence:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease mutation does not follow its open"
            )
        if (
            self.coordinator_lease_mutation_sequence <= opening.coordinator_lease_mutation_sequence
            or self.coordinator_lease_value_sequence < opening.coordinator_lease_value_sequence
            or self.coordinator_lease_lifetime_sequence
            < opening.coordinator_lease_lifetime_sequence
            or self.lease.granted_at_unix_ms < opened.committed_at_unix_ms
        ):
            raise ValueError("PreparedInitialRestartIntentClosure lease lineage predates its open")
        mutation_delta = (
            self.coordinator_lease_mutation_sequence - opening.coordinator_lease_mutation_sequence
        )
        transaction_delta = (
            self.coordinator_lease_transaction_sequence - opened.transaction_sequence
        )
        value_delta = (
            self.coordinator_lease_value_sequence - opening.coordinator_lease_value_sequence
        )
        lifetime_delta = (
            self.coordinator_lease_lifetime_sequence - opening.coordinator_lease_lifetime_sequence
        )
        if (
            transaction_delta < mutation_delta
            or mutation_delta < 2 * lifetime_delta
            or value_delta < lifetime_delta
            or value_delta > mutation_delta - lifetime_delta
        ):
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease lineage has impossible sequence deltas"
            )
        if (
            self.lease.record.lease_id != opening.lease.record.lease_id
            and self.lease.record.lease_id in opening.generation_lease_id_history
        ):
            raise ValueError("PreparedInitialRestartIntentClosure reuses an older lease identity")
        if (
            self.lease.fencing_token != opening.lease.fencing_token
            and self.lease.fencing_token in opening.generation_fencing_token_history
        ):
            raise ValueError("PreparedInitialRestartIntentClosure reuses an older fencing token")
        same_lease_id = self.lease.record.lease_id == opening.lease.record.lease_id
        if same_lease_id:
            if self.lease.record.coordinator_id != opening.lease.record.coordinator_id:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure changes one lease's coordinator"
                )
            if self.lease.record.lease_duration_ms != opening.lease.record.lease_duration_ms:
                raise ValueError("PreparedInitialRestartIntentClosure changes one lease's duration")
            if lifetime_delta != 0 or value_delta != 0:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure changes one lease's store identity"
                )
            latest_valid_grant = opening.lease.granted_at_unix_ms + mutation_delta * (
                opening.lease.record.lease_duration_ms - 1
            )
            if self.lease.granted_at_unix_ms > latest_valid_grant:
                raise ValueError(
                    "PreparedInitialRestartIntentClosure renews an expired coordinator lease"
                )
            return
        if value_delta == 0:
            raise ValueError(
                "PreparedInitialRestartIntentClosure changes lease identity without a new value"
            )
        if lifetime_delta == 0 and mutation_delta != 1:
            raise ValueError(
                "PreparedInitialRestartIntentClosure lease replacement has ambiguous mutations"
            )
        if lifetime_delta == 0 and self.lease.granted_at_unix_ms < (
            opening.lease.granted_at_unix_ms + opening.lease.record.lease_duration_ms
        ):
            raise ValueError("PreparedInitialRestartIntentClosure coordinator leases overlap")

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


__all__ = ["PreparedInitialRestartIntentClosure"]
