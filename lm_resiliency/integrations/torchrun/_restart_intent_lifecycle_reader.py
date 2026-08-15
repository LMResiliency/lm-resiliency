"""Fail-closed reads of the first committed restart-intent closure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol, Self, TypeVar

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
    """One verified first restart-intent closure and its authoritative entries."""

    intent: RestartIntentRecord
    open_head: RestartIntentHeadRecord
    closed_head: RestartIntentClosedHeadRecord
    lifecycle: RestartIntentLifecycleRecord
    lifecycle_head: RestartIntentLifecycleHeadRecord
    generation_snapshot: StoredGenerationSnapshot
    opening_authority: CoordinatorLeaseAuthority
    closing_authority: CoordinatorLeaseAuthority
    intent_entry: ControlStoreEntry
    open_head_entry: ControlStoreEntry
    closed_head_entry: ControlStoreEntry
    lifecycle_entry: ControlStoreEntry
    lifecycle_head_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        expected_types = (
            ("intent", self.intent, RestartIntentRecord),
            ("open_head", self.open_head, RestartIntentHeadRecord),
            ("closed_head", self.closed_head, RestartIntentClosedHeadRecord),
            ("lifecycle", self.lifecycle, RestartIntentLifecycleRecord),
            (
                "lifecycle_head",
                self.lifecycle_head,
                RestartIntentLifecycleHeadRecord,
            ),
            (
                "generation_snapshot",
                self.generation_snapshot,
                StoredGenerationSnapshot,
            ),
            (
                "opening_authority",
                self.opening_authority,
                CoordinatorLeaseAuthority,
            ),
            (
                "closing_authority",
                self.closing_authority,
                CoordinatorLeaseAuthority,
            ),
        )
        for path, value, expected_type in expected_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"StoredInitialRestartIntentClosure.{path} must be {expected_type.__name__}"
                )
        for path in (
            "intent_entry",
            "open_head_entry",
            "closed_head_entry",
            "lifecycle_entry",
            "lifecycle_head_entry",
        ):
            if not isinstance(getattr(self, path), ControlStoreEntry):
                raise TypeError(
                    f"StoredInitialRestartIntentClosure.{path} must be ControlStoreEntry"
                )
        if (
            self.lifecycle.closed_intent != self.open_head
            or self.lifecycle_head.lifecycle_digest != self.lifecycle.digest
            or self.closed_head.lifecycle_head_digest != self.lifecycle_head.digest
        ):
            raise ValueError("StoredInitialRestartIntentClosure records do not form one closure")

    @property
    def closed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.closed_head_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated closure lost its commit time")
        return committed_at_unix_ms

    @property
    def transaction_sequence(self) -> int:
        return self.closed_head_entry.transaction_sequence


@dataclass(frozen=True, slots=True)
class _ObservedKey:
    entry: ControlStoreEntry | None
    history: tuple[ControlStoreEntry, ...]
    has_history: bool


class _CanonicalRecord(Protocol):
    @classmethod
    def from_json(cls, encoded: bytes) -> Self: ...

    def to_json(self) -> bytes: ...


_CanonicalRecordT = TypeVar("_CanonicalRecordT", bound=_CanonicalRecord)


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
            if lifecycle_head.closure_index != 1:
                raise RestartIntentLifecycleReadCorrupt(
                    "initial restart-intent closure index is not one"
                )
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
                lifecycle_head,
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
            RestartIntentHeadRecord.from_json(head.entry.value)
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

    def _validate_closure(
        self,
        head_observation: _ObservedKey,
        lifecycle_observation: _ObservedKey,
        closure_observation: _ObservedKey,
        intent_observation: _ObservedKey,
        lifecycle_head: RestartIntentLifecycleHeadRecord,
        generation_snapshot: StoredGenerationSnapshot | None,
        successor: StoredGenerationSnapshot | None,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> StoredInitialRestartIntentClosure:
        head_entry = _required_entry(head_observation, "current restart-intent head")
        lifecycle_head_entry = _required_entry(
            lifecycle_observation,
            "restart-intent lifecycle head",
        )
        closure_entry = _required_entry(
            closure_observation,
            "immutable restart-intent closure",
        )
        intent_entry = _required_entry(
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
        _require_immutable_history(
            lifecycle_observation,
            "restart-intent lifecycle head",
        )
        _require_immutable_history(
            closure_observation,
            "immutable restart-intent closure",
        )
        _require_immutable_history(
            intent_observation,
            "immutable restart intent",
        )
        open_head = _decode_canonical(
            open_head_entry,
            RestartIntentHeadRecord,
            "predecessor restart-intent head",
        )
        closed_head = _decode_canonical(
            head_entry,
            RestartIntentClosedHeadRecord,
            "closed restart-intent head",
        )
        lifecycle = _decode_canonical(
            closure_entry,
            RestartIntentLifecycleRecord,
            "immutable restart-intent closure",
        )
        intent = _decode_canonical(
            intent_entry,
            RestartIntentRecord,
            "immutable restart intent",
        )
        if (
            closed_head.run_id != self._run_id
            or lifecycle_head.run_id != self._run_id
            or lifecycle.closed_intent != open_head
            or open_head.run_id != intent.intent.run_id
            or open_head.generation != intent.intent.generation
            or open_head.intent_id != intent.intent.intent_id
            or open_head.intent_digest != intent.digest
            or lifecycle_head.closure_index != closed_head.closure_index
            or lifecycle_head.generation != closed_head.generation
            or lifecycle_head.intent_id != closed_head.intent_id
            or lifecycle_head.digest != closed_head.lifecycle_head_digest
            or lifecycle_head.lifecycle_digest != lifecycle.digest
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure records do not identify one intent"
            )
        if generation_snapshot is None:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure references a missing generation"
            )
        if generation_snapshot.record.digest != intent.generation_snapshot_digest:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure references the wrong generation snapshot"
            )
        self._validate_store_sequences(
            intent_entry,
            open_head_entry,
            head_entry,
            lifecycle_head_entry,
            closure_entry,
        )
        opening_authority = self._authority(
            intent_entry,
            CoordinatorLeaseRecord(
                run_id=self._run_id,
                coordinator_id=intent.coordinator_id,
                lease_id=intent.lease_id,
                lease_duration_ms=intent.coordinator_lease_duration_ms,
            ),
            intent.coordinator_fencing_token,
            intent.coordinator_lease_digest,
            lease_history,
            "opening",
        )
        closing_authority = self._authority(
            head_entry,
            CoordinatorLeaseRecord(
                run_id=self._run_id,
                coordinator_id=lifecycle.coordinator_id,
                lease_id=lifecycle.lease_id,
                lease_duration_ms=lifecycle.coordinator_lease_duration_ms,
            ),
            lifecycle.coordinator_fencing_token,
            lifecycle.coordinator_lease_digest,
            lease_history,
            "closing",
        )
        self._validate_transactions(
            intent,
            intent_entry,
            open_head_entry,
            head_entry,
            lifecycle_head_entry,
            closure_entry,
            generation_snapshot,
            successor,
            opening_authority,
            closing_authority,
        )
        return StoredInitialRestartIntentClosure(
            intent=intent,
            open_head=open_head,
            closed_head=closed_head,
            lifecycle=lifecycle,
            lifecycle_head=lifecycle_head,
            generation_snapshot=generation_snapshot,
            opening_authority=opening_authority,
            closing_authority=closing_authority,
            intent_entry=intent_entry,
            open_head_entry=open_head_entry,
            closed_head_entry=head_entry,
            lifecycle_entry=closure_entry,
            lifecycle_head_entry=lifecycle_head_entry,
        )

    def _decode_lifecycle_head(
        self,
        observation: _ObservedKey,
    ) -> RestartIntentLifecycleHeadRecord:
        entry = _required_entry(observation, "restart-intent lifecycle head")
        _require_immutable_history(observation, "restart-intent lifecycle head")
        lifecycle_head = _decode_canonical(
            entry,
            RestartIntentLifecycleHeadRecord,
            "restart-intent lifecycle head",
        )
        if lifecycle_head.run_id != self._run_id:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle head belongs to another run"
            )
        return lifecycle_head

    def _validate_store_sequences(
        self,
        intent_entry: ControlStoreEntry,
        open_head_entry: ControlStoreEntry,
        closed_head_entry: ControlStoreEntry,
        lifecycle_head_entry: ControlStoreEntry,
        closure_entry: ControlStoreEntry,
    ) -> None:
        for path, entry in (
            ("immutable restart intent", intent_entry),
            ("predecessor restart-intent head", open_head_entry),
            ("restart-intent lifecycle head", lifecycle_head_entry),
            ("immutable restart-intent closure", closure_entry),
        ):
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
            ):
                raise RestartIntentLifecycleReadCorrupt(
                    f"{path} is not an immutable initial creation"
                )
        if (
            closed_head_entry.mutation_sequence != 2
            or closed_head_entry.value_sequence != 2
            or closed_head_entry.lifetime_sequence != 1
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "closed restart-intent head has invalid store sequences"
            )

    def _validate_transactions(
        self,
        intent: RestartIntentRecord,
        intent_entry: ControlStoreEntry,
        open_head_entry: ControlStoreEntry,
        closed_head_entry: ControlStoreEntry,
        lifecycle_head_entry: ControlStoreEntry,
        closure_entry: ControlStoreEntry,
        generation_snapshot: StoredGenerationSnapshot,
        successor: StoredGenerationSnapshot | None,
        opening_authority: CoordinatorLeaseAuthority,
        closing_authority: CoordinatorLeaseAuthority,
    ) -> None:
        if (
            intent_entry.committed_at_unix_ms is None
            or open_head_entry.committed_at_unix_ms is None
            or intent_entry.committed_at_unix_ms != open_head_entry.committed_at_unix_ms
            or intent_entry.transaction_sequence != open_head_entry.transaction_sequence
            or _guard_provenance(intent_entry) != _guard_provenance(open_head_entry)
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent opening records do not share one transaction"
            )
        opened_at = intent_entry.committed_at_unix_ms
        if (
            intent_entry.transaction_sequence
            <= max(
                generation_snapshot.transaction_sequence,
                opening_authority.transaction_sequence,
            )
            or opened_at < generation_snapshot.committed_at_unix_ms
            or opened_at < opening_authority.lease.granted_at_unix_ms
            or opened_at
            >= min(
                opening_authority.lease.expires_at_unix_ms,
                intent.intent.prepare_deadline_unix_ms,
            )
            or (
                successor is not None
                and intent_entry.transaction_sequence >= successor.transaction_sequence
            )
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent opening is outside its causal window"
            )
        closure_entries = (
            closed_head_entry,
            lifecycle_head_entry,
            closure_entry,
        )
        closed_at = closed_head_entry.committed_at_unix_ms
        closure_transaction = closed_head_entry.transaction_sequence
        closure_guard = _guard_provenance(closed_head_entry)
        if closed_at is None or any(
            entry.committed_at_unix_ms != closed_at
            or entry.transaction_sequence != closure_transaction
            or _guard_provenance(entry) != closure_guard
            for entry in closure_entries[1:]
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure records do not share one transaction"
            )
        if (
            closure_transaction
            <= max(
                intent_entry.transaction_sequence,
                closing_authority.transaction_sequence,
            )
            or closed_at < opened_at
            or closed_at < closing_authority.lease.granted_at_unix_ms
            or closed_at >= closing_authority.lease.expires_at_unix_ms
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure is outside its causal lease window"
            )

    def _authority(
        self,
        entry: ControlStoreEntry,
        record: CoordinatorLeaseRecord,
        fencing_token: int,
        lease_digest: str,
        history: tuple[CoordinatorLeaseAuthority, ...],
        label: str,
    ) -> CoordinatorLeaseAuthority:
        provenance = _guard_provenance(entry)
        (
            guard_key,
            guard_revision,
            guard_value_digest,
            guard_mutation_sequence,
            guard_value_sequence,
            guard_lifetime_sequence,
            guard_committed_at_unix_ms,
        ) = provenance
        if (
            guard_key != self._generation_reader.coordinator_lease_key
            or guard_revision != fencing_token
            or guard_value_digest != lease_digest
        ):
            raise RestartIntentLifecycleReadCorrupt(
                f"restart-intent {label} has invalid lease provenance"
            )
        matches = tuple(
            authority
            for authority in history
            if (
                authority.lease.record == record
                and authority.lease.fencing_token == guard_revision
                and authority.lease.granted_at_unix_ms == guard_committed_at_unix_ms
                and authority.mutation_sequence == guard_mutation_sequence
                and authority.value_sequence == guard_value_sequence
                and authority.lifetime_sequence == guard_lifetime_sequence
            )
        )
        if len(matches) != 1:
            raise RestartIntentLifecycleReadCorrupt(
                f"restart-intent {label} authority is absent from lease history"
            )
        return matches[0]

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


def _require_immutable_history(observation: _ObservedKey, path: str) -> None:
    entry = _required_entry(observation, path)
    if observation.history != (entry,):
        raise RestartIntentLifecycleReadCorrupt(f"{path} is not an immutable retained value")


def _decode_canonical(
    entry: ControlStoreEntry,
    record_type: type[_CanonicalRecordT],
    path: str,
) -> _CanonicalRecordT:
    try:
        record = record_type.from_json(entry.value)
    except (TypeError, ValueError) as error:
        raise RestartIntentLifecycleReadCorrupt(f"{path} is malformed") from error
    if entry.value != record.to_json():
        raise RestartIntentLifecycleReadCorrupt(f"{path} is noncanonical")
    return record


def _guard_provenance(entry: ControlStoreEntry) -> tuple[object, ...]:
    provenance = (
        entry.guard_key,
        entry.guard_revision,
        entry.guard_value_digest,
        entry.guard_mutation_sequence,
        entry.guard_value_sequence,
        entry.guard_lifetime_sequence,
        entry.guard_committed_at_unix_ms,
    )
    if any(value is None for value in provenance):
        raise RestartIntentLifecycleReadCorrupt(
            "restart-intent lifecycle has incomplete guard provenance"
        )
    return provenance


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
