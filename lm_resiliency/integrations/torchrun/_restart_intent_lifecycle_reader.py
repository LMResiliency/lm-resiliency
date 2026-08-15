"""Fail-closed reads of the first committed restart-intent closure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateCorrupt,
    GenerationStateReader,
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

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class RestartIntentLifecycleReadError(RuntimeError):
    """Base error for persisted restart-intent lifecycle reads."""


class RestartIntentLifecycleReadCorrupt(RestartIntentLifecycleReadError):
    """Raised when persisted restart-intent lifecycle state is contradictory."""


@dataclass(frozen=True, slots=True)
class StoredInitialRestartIntentClosure:
    """One closure authenticated against generation and lease history."""

    state: PersistedInitialRestartIntentClosure
    generation_snapshot: StoredGenerationSnapshot
    opening_authority: CoordinatorLeaseAuthority
    closing_authority: CoordinatorLeaseAuthority

    def __post_init__(self) -> None:
        if not isinstance(self.state, PersistedInitialRestartIntentClosure):
            raise TypeError(
                "StoredInitialRestartIntentClosure.state must be "
                "PersistedInitialRestartIntentClosure"
            )
        if not isinstance(self.generation_snapshot, StoredGenerationSnapshot):
            raise TypeError(
                "StoredInitialRestartIntentClosure.generation_snapshot must be "
                "StoredGenerationSnapshot"
            )
        if not isinstance(self.opening_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "StoredInitialRestartIntentClosure.opening_authority must be "
                "CoordinatorLeaseAuthority"
            )
        if not isinstance(self.closing_authority, CoordinatorLeaseAuthority):
            raise TypeError(
                "StoredInitialRestartIntentClosure.closing_authority must be "
                "CoordinatorLeaseAuthority"
            )

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


@dataclass(frozen=True, slots=True)
class _ObservedKey:
    entry: ControlStoreEntry | None
    history: tuple[ControlStoreEntry, ...]
    has_history: bool


class InitialRestartIntentLifecycleReader:
    """Read and verify the first committed restart-intent closure."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._intent_head_key = f"{self._run_prefix}/restart-intent-head"
        self._lifecycle_head_key = f"{self._run_prefix}/restart-intent-lifecycle-head"
        self._generation_reader = GenerationStateReader(store, run_id=self._run_id)
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    @property
    def intent_head_key(self) -> str:
        return self._intent_head_key

    @property
    def lifecycle_head_key(self) -> str:
        return self._lifecycle_head_key

    def intent_key(self, intent_id: str) -> str:
        normalized_intent_id = _nonempty_string(intent_id, "intent_id")
        intent_digest = hashlib.sha256(normalized_intent_id.encode("utf-8")).hexdigest()
        return f"{self._run_prefix}/restart-intents/{intent_digest}"

    def closure_key(self, closure_index: int) -> str:
        normalized_index = _positive_integer(closure_index, "closure_index")
        return f"{self._run_prefix}/restart-intent-closures/{normalized_index}"

    def read(self) -> StoredInitialRestartIntentClosure | None:
        """Return one stable verified closure, or ``None`` before closure."""

        for _ in range(_MAX_READ_ATTEMPTS):
            head_observation = self._observe(self._intent_head_key)
            lifecycle_observation = self._observe(self._lifecycle_head_key)
            if not self._stable(
                self._intent_head_key,
                head_observation,
                self._lifecycle_head_key,
                lifecycle_observation,
            ):
                continue
            if lifecycle_observation.entry is None:
                self._validate_absent_lifecycle(
                    head_observation,
                    lifecycle_observation,
                )
                return None
            lifecycle_head = self._decode_lifecycle_head(lifecycle_observation)
            closure_key = self.closure_key(lifecycle_head.closure_index)
            intent_key = self.intent_key(lifecycle_head.intent_id)
            closure_observation = self._observe(closure_key)
            intent_observation = self._observe(intent_key)
            try:
                lease_history = self._lease_history_reader.read()
                generation_snapshot = self._generation_reader.get(lifecycle_head.generation)
                successor = self._generation_reader.get(lifecycle_head.generation + 1)
            except (
                CoordinatorLeaseHistoryCorrupt,
                GenerationStateCorrupt,
            ) as error:
                raise RestartIntentLifecycleReadCorrupt(
                    "restart-intent lifecycle dependencies are corrupt"
                ) from error
            if not self._stable(
                self._intent_head_key,
                head_observation,
                self._lifecycle_head_key,
                lifecycle_observation,
                closure_key,
                closure_observation,
                intent_key,
                intent_observation,
            ):
                continue
            return self._validate_closure(
                head_observation,
                lifecycle_observation,
                closure_observation,
                intent_observation,
                generation_snapshot,
                successor,
                lease_history,
            )
        raise RestartIntentLifecycleReadError(
            "restart-intent lifecycle changed repeatedly during read"
        )

    def _validate_absent_lifecycle(
        self,
        head: _ObservedKey,
        lifecycle: _ObservedKey,
    ) -> None:
        if lifecycle.has_history or lifecycle.history:
            raise RestartIntentLifecycleReadCorrupt("restart-intent lifecycle head was deleted")
        if head.entry is None:
            if head.has_history or head.history:
                raise RestartIntentLifecycleReadCorrupt("current restart-intent head was deleted")
            return
        if not head.has_history or not head.history:
            raise RestartIntentLifecycleReadCorrupt(
                "live restart-intent head has no durable history"
            )
        try:
            open_head = RestartIntentHeadRecord.from_json(head.entry.value)
        except (TypeError, ValueError) as open_error:
            try:
                RestartIntentClosedHeadRecord.from_json(head.entry.value)
            except (TypeError, ValueError) as closed_error:
                raise RestartIntentLifecycleReadCorrupt(
                    "current restart-intent head is malformed"
                ) from closed_error
            raise RestartIntentLifecycleReadCorrupt(
                "closed restart-intent head has no lifecycle state"
            ) from open_error
        if open_head.run_id != self._run_id or head.entry.value != open_head.to_json():
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head is noncanonical or belongs to another run"
            )

    def _validate_closure(
        self,
        head_observation: _ObservedKey,
        lifecycle_observation: _ObservedKey,
        closure_observation: _ObservedKey,
        intent_observation: _ObservedKey,
        generation_snapshot: StoredGenerationSnapshot | None,
        successor: StoredGenerationSnapshot | None,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> StoredInitialRestartIntentClosure:
        head_entry = _required_entry(head_observation, "current restart-intent head")
        lifecycle_head_entry = _immutable_entry(
            lifecycle_observation,
            "restart-intent lifecycle head",
        )
        closure_entry = _immutable_entry(
            closure_observation,
            "immutable restart-intent closure",
        )
        intent_entry = _immutable_entry(
            intent_observation,
            "immutable restart intent",
        )
        if len(head_observation.history) != 2:
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head does not retain one open predecessor"
            )
        open_head_entry, retained_closed_entry = head_observation.history
        if retained_closed_entry != head_entry:
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head is absent from its history"
            )
        try:
            state = PersistedInitialRestartIntentClosure.from_entries(
                run_id=self._run_id,
                intent_entry=intent_entry,
                open_head_entry=open_head_entry,
                closed_head_entry=head_entry,
                lifecycle_entry=closure_entry,
                lifecycle_head_entry=lifecycle_head_entry,
            )
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleReadCorrupt(
                "persisted restart-intent closure is contradictory"
            ) from error
        if generation_snapshot is None:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure references a missing generation"
            )
        if generation_snapshot.record.digest != state.intent.generation_snapshot_digest:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure references the wrong generation snapshot"
            )
        generation_authority = self._generation_authority(
            generation_snapshot,
            lease_history,
        )
        opening_authority = self._entry_authority(
            state.intent_entry,
            CoordinatorLeaseRecord(
                run_id=self._run_id,
                coordinator_id=state.intent.coordinator_id,
                lease_id=state.intent.lease_id,
                lease_duration_ms=state.intent.coordinator_lease_duration_ms,
            ),
            state.intent.coordinator_fencing_token,
            state.intent.coordinator_lease_digest,
            lease_history,
            "opening",
        )
        closing_authority = self._entry_authority(
            state.closed_head_entry,
            CoordinatorLeaseRecord(
                run_id=self._run_id,
                coordinator_id=state.lifecycle.coordinator_id,
                lease_id=state.lifecycle.lease_id,
                lease_duration_ms=state.lifecycle.coordinator_lease_duration_ms,
            ),
            state.lifecycle.coordinator_fencing_token,
            state.lifecycle.coordinator_lease_digest,
            lease_history,
            "closing",
        )
        generation_index = lease_history.index(generation_authority)
        opening_index = lease_history.index(opening_authority)
        closing_index = lease_history.index(closing_authority)
        if generation_index > opening_index or opening_index > closing_index:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle lease authorities are out of order"
            )
        if (
            state.opening_transaction_sequence
            <= max(
                generation_snapshot.transaction_sequence,
                opening_authority.transaction_sequence,
            )
            or state.opened_at_unix_ms < generation_snapshot.committed_at_unix_ms
            or state.opened_at_unix_ms < opening_authority.lease.granted_at_unix_ms
            or state.opened_at_unix_ms
            >= min(
                opening_authority.lease.expires_at_unix_ms,
                state.intent.intent.prepare_deadline_unix_ms,
            )
            or (
                successor is not None
                and state.opening_transaction_sequence >= successor.transaction_sequence
            )
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent opening is outside its causal window"
            )
        if (
            state.closing_transaction_sequence
            <= max(
                state.opening_transaction_sequence,
                closing_authority.transaction_sequence,
            )
            or state.closed_at_unix_ms < closing_authority.lease.granted_at_unix_ms
            or state.closed_at_unix_ms >= closing_authority.lease.expires_at_unix_ms
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure is outside its causal lease window"
            )
        return StoredInitialRestartIntentClosure(
            state=state,
            generation_snapshot=generation_snapshot,
            opening_authority=opening_authority,
            closing_authority=closing_authority,
        )

    def _decode_lifecycle_head(
        self,
        observation: _ObservedKey,
    ) -> RestartIntentLifecycleHeadRecord:
        entry = _immutable_entry(observation, "restart-intent lifecycle head")
        try:
            lifecycle_head = RestartIntentLifecycleHeadRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle head is malformed"
            ) from error
        if lifecycle_head.run_id != self._run_id or entry.value != lifecycle_head.to_json():
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle head is noncanonical or belongs to another run"
            )
        return lifecycle_head

    def _generation_authority(
        self,
        snapshot: StoredGenerationSnapshot,
        history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> CoordinatorLeaseAuthority:
        record = CoordinatorLeaseRecord(
            run_id=self._run_id,
            coordinator_id=snapshot.record.coordinator_id,
            lease_id=snapshot.record.lease_id,
            lease_duration_ms=snapshot.record.coordinator_lease_duration_ms,
        )
        authority = _unique_authority(
            history,
            record=record,
            fencing_token=snapshot.record.coordinator_fencing_token,
            granted_at_unix_ms=snapshot.guard_committed_at_unix_ms,
            mutation_sequence=snapshot.guard_mutation_sequence,
            value_sequence=snapshot.guard_value_sequence,
            lifetime_sequence=snapshot.guard_lifetime_sequence,
            label="generation",
        )
        if (
            snapshot.transaction_sequence <= authority.transaction_sequence
            or snapshot.committed_at_unix_ms < authority.lease.granted_at_unix_ms
            or snapshot.committed_at_unix_ms >= authority.lease.expires_at_unix_ms
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent generation is outside its causal lease window"
            )
        return authority

    def _entry_authority(
        self,
        entry: ControlStoreEntry,
        record: CoordinatorLeaseRecord,
        fencing_token: int,
        lease_digest: str,
        history: tuple[CoordinatorLeaseAuthority, ...],
        label: str,
    ) -> CoordinatorLeaseAuthority:
        provenance = (
            entry.guard_key,
            entry.guard_revision,
            entry.guard_value_digest,
            entry.guard_mutation_sequence,
            entry.guard_value_sequence,
            entry.guard_lifetime_sequence,
            entry.guard_committed_at_unix_ms,
        )
        if (
            entry.guard_key != self._generation_reader.coordinator_lease_key
            or entry.guard_revision != fencing_token
            or entry.guard_value_digest != lease_digest
            or any(value is None for value in provenance)
        ):
            raise RestartIntentLifecycleReadCorrupt(
                f"restart-intent {label} has invalid lease provenance"
            )
        return _unique_authority(
            history,
            record=record,
            fencing_token=fencing_token,
            granted_at_unix_ms=entry.guard_committed_at_unix_ms,
            mutation_sequence=entry.guard_mutation_sequence,
            value_sequence=entry.guard_value_sequence,
            lifetime_sequence=entry.guard_lifetime_sequence,
            label=label,
        )

    def _observe(self, key: str) -> _ObservedKey:
        return _ObservedKey(
            entry=self._store.get(key),
            history=self._store.get_history(key),
            has_history=self._store.has_history(key),
        )

    def _stable(self, *items: object) -> bool:
        for index in range(0, len(items), 2):
            key = items[index]
            observation = items[index + 1]
            if not isinstance(key, str) or not isinstance(observation, _ObservedKey):
                raise AssertionError("invalid lifecycle stability input")
            if self._observe(key) != observation:
                return False
        return True


def _required_entry(observation: _ObservedKey, path: str) -> ControlStoreEntry:
    if observation.entry is None:
        raise RestartIntentLifecycleReadCorrupt(f"{path} is missing")
    if not observation.has_history or not observation.history:
        raise RestartIntentLifecycleReadCorrupt(f"{path} has no durable history")
    return observation.entry


def _immutable_entry(observation: _ObservedKey, path: str) -> ControlStoreEntry:
    entry = _required_entry(observation, path)
    if observation.history != (entry,):
        raise RestartIntentLifecycleReadCorrupt(f"{path} is not an immutable retained value")
    return entry


def _unique_authority(
    history: tuple[CoordinatorLeaseAuthority, ...],
    *,
    record: CoordinatorLeaseRecord,
    fencing_token: int,
    granted_at_unix_ms: object,
    mutation_sequence: object,
    value_sequence: object,
    lifetime_sequence: object,
    label: str,
) -> CoordinatorLeaseAuthority:
    matches = tuple(
        authority
        for authority in history
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
        raise RestartIntentLifecycleReadCorrupt(
            f"restart-intent {label} authority is absent from lease history"
        )
    return matches[0]


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "InitialRestartIntentLifecycleReader",
    "RestartIntentLifecycleReadCorrupt",
    "RestartIntentLifecycleReadError",
    "StoredInitialRestartIntentClosure",
]
