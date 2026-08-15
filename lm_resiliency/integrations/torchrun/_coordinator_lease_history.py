"""Coordinator lease authority values reconstructed from control-store history."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class CoordinatorLeaseAuthorityCorrupt(RuntimeError):
    """Raised when one persisted coordinator lease authority is invalid."""


class CoordinatorLeaseHistoryError(RuntimeError):
    """Base error for coordinator lease history reads."""


class CoordinatorLeaseHistoryCorrupt(CoordinatorLeaseHistoryError):
    """Raised when persisted coordinator lease history is contradictory."""


@dataclass(frozen=True, slots=True)
class CoordinatorLeaseAuthority:
    """One canonical, store-stamped coordinator lease value."""

    lease: HeldCoordinatorLease
    transaction_sequence: int
    mutation_sequence: int
    value_sequence: int
    lifetime_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.lease, HeldCoordinatorLease):
            raise TypeError("CoordinatorLeaseAuthority.lease must be HeldCoordinatorLease")
        for path, value in (
            ("transaction_sequence", self.transaction_sequence),
            ("mutation_sequence", self.mutation_sequence),
            ("value_sequence", self.value_sequence),
            ("lifetime_sequence", self.lifetime_sequence),
        ):
            _positive_integer(value, f"CoordinatorLeaseAuthority.{path}")
        _validate_sequence_lineage(
            self.mutation_sequence,
            self.value_sequence,
            self.lifetime_sequence,
        )
        if self.transaction_sequence < self.mutation_sequence:
            raise ValueError(
                "CoordinatorLeaseAuthority.transaction_sequence is too small for mutation_sequence"
            )

    @classmethod
    def from_entry(
        cls,
        entry: ControlStoreEntry,
        *,
        run_id: str,
    ) -> CoordinatorLeaseAuthority:
        """Decode one authoritative coordinator lease store entry."""

        if not isinstance(entry, ControlStoreEntry):
            raise TypeError("entry must be ControlStoreEntry")
        normalized_run_id = _nonempty_string(run_id, "run_id")
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise CoordinatorLeaseAuthorityCorrupt(
                "coordinator lease authority contains a malformed record"
            ) from error
        if record.run_id != normalized_run_id:
            raise CoordinatorLeaseAuthorityCorrupt(
                "coordinator lease authority belongs to another run"
            )
        if entry.value != record.to_json():
            raise CoordinatorLeaseAuthorityCorrupt(
                "coordinator lease authority contains noncanonical record bytes"
            )
        if entry.committed_at_unix_ms is None:
            raise CoordinatorLeaseAuthorityCorrupt(
                "coordinator lease authority has no authoritative commit time"
            )
        if entry.guard_key is not None:
            raise CoordinatorLeaseAuthorityCorrupt(
                "coordinator lease authority unexpectedly carries guard provenance"
            )
        return cls(
            lease=HeldCoordinatorLease(
                record=record,
                fencing_token=entry.revision,
                granted_at_unix_ms=entry.committed_at_unix_ms,
            ),
            transaction_sequence=entry.transaction_sequence,
            mutation_sequence=entry.mutation_sequence,
            value_sequence=entry.value_sequence,
            lifetime_sequence=entry.lifetime_sequence,
        )


class CoordinatorLeaseHistoryReader:
    """Read and verify one run's complete coordinator lease value history."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._lease_key = f"{_CONTROL_PREFIX}/runs/{run_digest}/coordinator-lease"

    @property
    def lease_key(self) -> str:
        return self._lease_key

    def read(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        """Return one stable, verified snapshot of all committed lease values."""

        for _ in range(_MAX_READ_ATTEMPTS):
            history = self._store.get_history(self._lease_key)
            current = self._store.get(self._lease_key)
            has_history = self._store.has_history(self._lease_key)
            confirmed_history = self._store.get_history(self._lease_key)
            confirmed_current = self._store.get(self._lease_key)
            confirmed_has_history = self._store.has_history(self._lease_key)
            if (
                history != confirmed_history
                or current != confirmed_current
                or has_history != confirmed_has_history
            ):
                continue
            if bool(history) != has_history:
                raise CoordinatorLeaseHistoryCorrupt(
                    "coordinator lease value history contradicts its durable history marker"
                )
            if current is not None and (not history or history[-1] != current):
                raise CoordinatorLeaseHistoryCorrupt(
                    "current coordinator lease is absent from its value history"
                )
            try:
                authorities = tuple(
                    CoordinatorLeaseAuthority.from_entry(
                        entry,
                        run_id=self._run_id,
                    )
                    for entry in history
                )
            except (CoordinatorLeaseAuthorityCorrupt, TypeError, ValueError) as error:
                raise CoordinatorLeaseHistoryCorrupt(
                    "coordinator lease history contains an invalid authority"
                ) from error
            self._validate_history(authorities)
            return authorities
        raise CoordinatorLeaseHistoryError(
            "coordinator lease history changed repeatedly during read"
        )

    def _validate_history(
        self,
        authorities: tuple[CoordinatorLeaseAuthority, ...],
    ) -> None:
        if not authorities:
            return
        first = authorities[0]
        if (
            first.mutation_sequence != 1
            or first.value_sequence != 1
            or first.lifetime_sequence != 1
        ):
            raise CoordinatorLeaseHistoryCorrupt(
                "coordinator lease history does not begin at initial store sequences"
            )
        seen_lease_ids = {first.lease.record.lease_id}
        seen_fencing_tokens = {first.lease.fencing_token}
        for previous, current in zip(authorities, authorities[1:], strict=False):
            self._validate_transition(previous, current)
            if current.lease.fencing_token in seen_fencing_tokens:
                raise CoordinatorLeaseHistoryCorrupt(
                    "coordinator lease fencing token reappears in history"
                )
            seen_fencing_tokens.add(current.lease.fencing_token)
            if current.lease.record.lease_id != previous.lease.record.lease_id:
                if current.lease.record.lease_id in seen_lease_ids:
                    raise CoordinatorLeaseHistoryCorrupt(
                        "coordinator lease identity reappears after replacement"
                    )
                seen_lease_ids.add(current.lease.record.lease_id)

    def _validate_transition(
        self,
        previous: CoordinatorLeaseAuthority,
        current: CoordinatorLeaseAuthority,
    ) -> None:
        if current.transaction_sequence <= previous.transaction_sequence:
            raise CoordinatorLeaseHistoryCorrupt(
                "coordinator lease transaction sequences do not advance"
            )
        if current.lease.granted_at_unix_ms < previous.lease.granted_at_unix_ms:
            raise CoordinatorLeaseHistoryCorrupt("coordinator lease grant times move backward")
        mutation_delta = current.mutation_sequence - previous.mutation_sequence
        value_delta = current.value_sequence - previous.value_sequence
        lifetime_delta = current.lifetime_sequence - previous.lifetime_sequence
        transaction_delta = current.transaction_sequence - previous.transaction_sequence
        if transaction_delta < mutation_delta:
            raise CoordinatorLeaseHistoryCorrupt(
                "coordinator lease mutation count exceeds transaction ordering"
            )
        if lifetime_delta not in (0, 1):
            raise CoordinatorLeaseHistoryCorrupt("coordinator lease history omits a key lifetime")
        expected_mutation_delta = 1 if lifetime_delta == 0 else 2
        if mutation_delta != expected_mutation_delta:
            raise CoordinatorLeaseHistoryCorrupt("coordinator lease history omits a key mutation")
        same_record = current.lease.record == previous.lease.record
        expected_value_delta = 0 if lifetime_delta == 0 and same_record else 1
        if value_delta != expected_value_delta:
            raise CoordinatorLeaseHistoryCorrupt(
                "coordinator lease value sequence contradicts its records"
            )
        if same_record:
            if lifetime_delta != 0:
                raise CoordinatorLeaseHistoryCorrupt(
                    "one coordinator lease crosses a recreated key lifetime"
                )
            if current.lease.granted_at_unix_ms >= previous.lease.expires_at_unix_ms:
                raise CoordinatorLeaseHistoryCorrupt(
                    "coordinator lease history renews an expired lease"
                )
            return
        if current.lease.record.lease_id == previous.lease.record.lease_id:
            raise CoordinatorLeaseHistoryCorrupt(
                "one coordinator lease identity changes its persisted record"
            )
        if (
            lifetime_delta == 0
            and current.lease.granted_at_unix_ms < previous.lease.expires_at_unix_ms
        ):
            raise CoordinatorLeaseHistoryCorrupt("coordinator lease replacements overlap")


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
            "CoordinatorLeaseAuthority.mutation_sequence is too small for lifetime_sequence"
        )
    if value_sequence < lifetime_sequence:
        raise ValueError(
            "CoordinatorLeaseAuthority.value_sequence is too small for lifetime_sequence"
        )
    if value_sequence > mutation_sequence - lifetime_sequence + 1:
        raise ValueError(
            "CoordinatorLeaseAuthority.value_sequence is too large for mutation_sequence"
        )


__all__ = [
    "CoordinatorLeaseAuthority",
    "CoordinatorLeaseAuthorityCorrupt",
    "CoordinatorLeaseHistoryCorrupt",
    "CoordinatorLeaseHistoryError",
    "CoordinatorLeaseHistoryReader",
]
