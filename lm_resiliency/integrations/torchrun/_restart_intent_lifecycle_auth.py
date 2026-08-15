"""Pure authentication of one persisted initial restart-intent closure."""

from __future__ import annotations

from dataclasses import dataclass, field

from lm_resiliency.integrations.torchrun._control_store import ControlStoreEntry
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_state import (
    PersistedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)


@dataclass(frozen=True, slots=True)
class AuthenticatedInitialRestartIntentClosure:
    """One persisted closure bound to generation and lease history."""

    state: PersistedInitialRestartIntentClosure
    generation_snapshot: StoredGenerationSnapshot
    immediate_successor: StoredGenerationSnapshot | None
    lease_history: tuple[CoordinatorLeaseAuthority, ...]
    generation_authority: CoordinatorLeaseAuthority = field(init=False)
    opening_authority: CoordinatorLeaseAuthority = field(init=False)
    closing_authority: CoordinatorLeaseAuthority = field(init=False)

    def __post_init__(self) -> None:
        if not isinstance(self.state, PersistedInitialRestartIntentClosure):
            raise TypeError(
                "AuthenticatedInitialRestartIntentClosure.state must be "
                "PersistedInitialRestartIntentClosure"
            )
        if not isinstance(self.generation_snapshot, StoredGenerationSnapshot):
            raise TypeError(
                "AuthenticatedInitialRestartIntentClosure.generation_snapshot must "
                "be StoredGenerationSnapshot"
            )
        if self.immediate_successor is not None and not isinstance(
            self.immediate_successor,
            StoredGenerationSnapshot,
        ):
            raise TypeError(
                "AuthenticatedInitialRestartIntentClosure.immediate_successor must "
                "be StoredGenerationSnapshot or None"
            )
        if not isinstance(self.lease_history, tuple) or any(
            not isinstance(authority, CoordinatorLeaseAuthority) for authority in self.lease_history
        ):
            raise TypeError(
                "AuthenticatedInitialRestartIntentClosure.lease_history must be a "
                "tuple of CoordinatorLeaseAuthority values"
            )
        self._validate_generation()
        generation_authority = self._generation_authority()
        opening_authority = self._entry_authority(
            self.state.intent_entry,
            CoordinatorLeaseRecord(
                run_id=self.state.intent.intent.run_id,
                coordinator_id=self.state.intent.coordinator_id,
                lease_id=self.state.intent.lease_id,
                lease_duration_ms=self.state.intent.coordinator_lease_duration_ms,
            ),
            self.state.intent.coordinator_fencing_token,
            self.state.intent.coordinator_lease_digest,
            "opening",
        )
        closing_authority = self._entry_authority(
            self.state.closed_head_entry,
            CoordinatorLeaseRecord(
                run_id=self.state.intent.intent.run_id,
                coordinator_id=self.state.lifecycle.coordinator_id,
                lease_id=self.state.lifecycle.lease_id,
                lease_duration_ms=self.state.lifecycle.coordinator_lease_duration_ms,
            ),
            self.state.lifecycle.coordinator_fencing_token,
            self.state.lifecycle.coordinator_lease_digest,
            "closing",
        )
        self._validate_authority_order(
            generation_authority,
            opening_authority,
            closing_authority,
        )
        self._validate_causal_windows(
            generation_authority,
            opening_authority,
            closing_authority,
        )
        object.__setattr__(self, "generation_authority", generation_authority)
        object.__setattr__(self, "opening_authority", opening_authority)
        object.__setattr__(self, "closing_authority", closing_authority)

    @property
    def intent(self) -> RestartIntentRecord:
        return self.state.intent

    @property
    def open_head(self) -> RestartIntentHeadRecord:
        return self.state.open_head

    @property
    def closed_head(self) -> RestartIntentClosedHeadRecord:
        return self.state.closed_head

    @property
    def lifecycle(self) -> RestartIntentLifecycleRecord:
        return self.state.lifecycle

    @property
    def lifecycle_head(self) -> RestartIntentLifecycleHeadRecord:
        return self.state.lifecycle_head

    @property
    def closed_at_unix_ms(self) -> int:
        return self.state.closed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.state.closing_transaction_sequence

    def _validate_generation(self) -> None:
        intent = self.state.intent.intent
        generation_record = self.generation_snapshot.record
        if (
            generation_record.assignment.run_id != intent.run_id
            or generation_record.assignment.generation != intent.generation
            or generation_record.digest != self.state.intent.generation_snapshot_digest
        ):
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure references the wrong generation"
            )
        successor = self.immediate_successor
        if successor is not None and (
            successor.record.assignment.run_id != intent.run_id
            or successor.record.assignment.generation != intent.generation + 1
            or successor.record.previous_snapshot_digest != generation_record.digest
        ):
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure immediate successor is invalid"
            )

    def _generation_authority(self) -> CoordinatorLeaseAuthority:
        snapshot = self.generation_snapshot
        record = CoordinatorLeaseRecord(
            run_id=snapshot.record.assignment.run_id,
            coordinator_id=snapshot.record.coordinator_id,
            lease_id=snapshot.record.lease_id,
            lease_duration_ms=snapshot.record.coordinator_lease_duration_ms,
        )
        return self._unique_authority(
            record=record,
            fencing_token=snapshot.record.coordinator_fencing_token,
            granted_at_unix_ms=snapshot.guard_committed_at_unix_ms,
            mutation_sequence=snapshot.guard_mutation_sequence,
            value_sequence=snapshot.guard_value_sequence,
            lifetime_sequence=snapshot.guard_lifetime_sequence,
            label="generation",
        )

    def _entry_authority(
        self,
        entry: ControlStoreEntry,
        record: CoordinatorLeaseRecord,
        fencing_token: int,
        lease_digest: str,
        label: str,
    ) -> CoordinatorLeaseAuthority:
        if (
            entry.guard_revision != fencing_token
            or entry.guard_value_digest != lease_digest
            or entry.guard_committed_at_unix_ms is None
            or entry.guard_mutation_sequence is None
            or entry.guard_value_sequence is None
            or entry.guard_lifetime_sequence is None
        ):
            raise ValueError(
                f"AuthenticatedInitialRestartIntentClosure {label} lease provenance is incomplete"
            )
        return self._unique_authority(
            record=record,
            fencing_token=fencing_token,
            granted_at_unix_ms=entry.guard_committed_at_unix_ms,
            mutation_sequence=entry.guard_mutation_sequence,
            value_sequence=entry.guard_value_sequence,
            lifetime_sequence=entry.guard_lifetime_sequence,
            label=label,
        )

    def _unique_authority(
        self,
        *,
        record: CoordinatorLeaseRecord,
        fencing_token: int,
        granted_at_unix_ms: int,
        mutation_sequence: int,
        value_sequence: int,
        lifetime_sequence: int,
        label: str,
    ) -> CoordinatorLeaseAuthority:
        matches = tuple(
            authority
            for authority in self.lease_history
            if (
                authority.lease.record == record
                and authority.lease.fencing_token == fencing_token
                and authority.lease.granted_at_unix_ms == granted_at_unix_ms
                and authority.mutation_sequence == mutation_sequence
                and authority.value_sequence == value_sequence
                and authority.lifetime_sequence == lifetime_sequence
            )
        )
        if len(matches) != 1:
            raise ValueError(
                f"AuthenticatedInitialRestartIntentClosure {label} authority is "
                "absent from lease history"
            )
        return matches[0]

    def _validate_authority_order(
        self,
        generation: CoordinatorLeaseAuthority,
        opening: CoordinatorLeaseAuthority,
        closing: CoordinatorLeaseAuthority,
    ) -> None:
        generation_index = self.lease_history.index(generation)
        opening_index = self.lease_history.index(opening)
        closing_index = self.lease_history.index(closing)
        if generation_index > opening_index or opening_index > closing_index:
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure lease authorities are out of order"
            )

    def _validate_causal_windows(
        self,
        generation: CoordinatorLeaseAuthority,
        opening: CoordinatorLeaseAuthority,
        closing: CoordinatorLeaseAuthority,
    ) -> None:
        snapshot = self.generation_snapshot
        state = self.state
        if (
            snapshot.transaction_sequence <= generation.transaction_sequence
            or snapshot.committed_at_unix_ms < generation.lease.granted_at_unix_ms
            or snapshot.committed_at_unix_ms >= generation.lease.expires_at_unix_ms
            or not self._commit_precedes_next_authority(
                generation,
                transaction_sequence=snapshot.transaction_sequence,
                committed_at_unix_ms=snapshot.committed_at_unix_ms,
            )
        ):
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure generation is outside its lease window"
            )
        if (
            state.opening_transaction_sequence
            <= max(snapshot.transaction_sequence, opening.transaction_sequence)
            or state.opened_at_unix_ms < snapshot.committed_at_unix_ms
            or state.opened_at_unix_ms < opening.lease.granted_at_unix_ms
            or state.opened_at_unix_ms
            >= min(
                opening.lease.expires_at_unix_ms,
                state.intent.intent.prepare_deadline_unix_ms,
            )
            or (
                self.immediate_successor is not None
                and (
                    state.opening_transaction_sequence
                    >= self.immediate_successor.transaction_sequence
                    or state.opened_at_unix_ms > self.immediate_successor.committed_at_unix_ms
                )
            )
            or not self._commit_precedes_next_authority(
                opening,
                transaction_sequence=state.opening_transaction_sequence,
                committed_at_unix_ms=state.opened_at_unix_ms,
            )
        ):
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure opening is outside its causal window"
            )
        if (
            state.closing_transaction_sequence
            <= max(state.opening_transaction_sequence, closing.transaction_sequence)
            or state.closed_at_unix_ms < closing.lease.granted_at_unix_ms
            or state.closed_at_unix_ms >= closing.lease.expires_at_unix_ms
            or not self._commit_precedes_next_authority(
                closing,
                transaction_sequence=state.closing_transaction_sequence,
                committed_at_unix_ms=state.closed_at_unix_ms,
            )
        ):
            raise ValueError(
                "AuthenticatedInitialRestartIntentClosure closure is outside its "
                "causal lease window"
            )

    def _commit_precedes_next_authority(
        self,
        authority: CoordinatorLeaseAuthority,
        *,
        transaction_sequence: int,
        committed_at_unix_ms: int,
    ) -> bool:
        authority_index = self.lease_history.index(authority)
        if authority_index + 1 == len(self.lease_history):
            return True
        successor = self.lease_history[authority_index + 1]
        if successor.lifetime_sequence != authority.lifetime_sequence:
            return False
        return (
            transaction_sequence < successor.transaction_sequence
            and committed_at_unix_ms <= successor.lease.granted_at_unix_ms
        )


__all__ = ["AuthenticatedInitialRestartIntentClosure"]
