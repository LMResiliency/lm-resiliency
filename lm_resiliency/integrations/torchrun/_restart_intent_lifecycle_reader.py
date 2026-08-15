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
    CurrentGeneration,
    GenerationStateCorrupt,
    GenerationStateReader,
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_auth import (
    AuthenticatedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_state import (
    PersistedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class RestartIntentLifecycleReadError(RuntimeError):
    """Base error for persisted restart-intent lifecycle reads."""


class RestartIntentLifecycleReadCorrupt(RestartIntentLifecycleReadError):
    """Raised when persisted restart-intent lifecycle state is contradictory."""


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
        self._initial_closure_key = f"{self._run_prefix}/restart-intent-closures/1"
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

    def read(self) -> AuthenticatedInitialRestartIntentClosure | None:
        """Return one stable authenticated closure, or ``None`` before closure."""

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
                closure_observation = self._observe(self._initial_closure_key)
                if not self._stable(
                    self._intent_head_key,
                    head_observation,
                    self._lifecycle_head_key,
                    lifecycle_observation,
                    self._initial_closure_key,
                    closure_observation,
                ):
                    continue
                open_head = self._validate_absent_lifecycle(
                    head_observation,
                    lifecycle_observation,
                    closure_observation,
                )
                if open_head is None:
                    return None
                intent_key = self.intent_key(open_head.intent_id)
                intent_observation = self._observe(intent_key)
                lease_history, generation_result = self._read_dependencies()
                if not self._stable(
                    self._intent_head_key,
                    head_observation,
                    self._lifecycle_head_key,
                    lifecycle_observation,
                    self._initial_closure_key,
                    closure_observation,
                    intent_key,
                    intent_observation,
                ):
                    continue
                confirmed_lease_history, confirmed_generation_result = self._read_dependencies()
                if (
                    lease_history != confirmed_lease_history
                    or generation_result != confirmed_generation_result
                ):
                    continue
                intent = self._validate_open_intent(
                    open_head,
                    head_observation,
                    intent_observation,
                )
                opening_authority = self._opening_authority(
                    intent,
                    intent_observation,
                    lease_history,
                )
                self._validate_open_generation(
                    intent,
                    intent_observation,
                    generation_result,
                    opening_authority,
                    lease_history,
                )
                return None
            lifecycle_head = self._decode_lifecycle_head(lifecycle_observation)
            closure_key = self.closure_key(lifecycle_head.closure_index)
            intent_key = self.intent_key(lifecycle_head.intent_id)
            closure_observation = self._observe(closure_key)
            intent_observation = self._observe(intent_key)
            lease_history, generation_result = self._read_dependencies()
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
            confirmed_lease_history, confirmed_generation_result = self._read_dependencies()
            if (
                lease_history != confirmed_lease_history
                or generation_result != confirmed_generation_result
            ):
                continue
            generation_snapshot = None
            successor = None
            if generation_result is not None:
                _, generation_history = generation_result
                if lifecycle_head.generation < len(generation_history):
                    generation_snapshot = generation_history[lifecycle_head.generation]
                    successor_index = lifecycle_head.generation + 1
                    if successor_index < len(generation_history):
                        successor = generation_history[successor_index]
            return self._authenticate_closure(
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
        closure: _ObservedKey,
    ) -> RestartIntentHeadRecord | None:
        if lifecycle.has_history or lifecycle.history:
            raise RestartIntentLifecycleReadCorrupt("restart-intent lifecycle head was deleted")
        if closure.entry is not None or closure.has_history or closure.history:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure exists without a lifecycle head"
            )
        if head.entry is None:
            if head.has_history or head.history:
                raise RestartIntentLifecycleReadCorrupt("current restart-intent head was deleted")
            return None
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
        if (
            head.history != (head.entry,)
            or head.entry.mutation_sequence != 1
            or head.entry.value_sequence != 1
            or head.entry.lifetime_sequence != 1
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "live restart-intent head is not an immutable initial creation"
            )
        if open_head.run_id != self._run_id or head.entry.value != open_head.to_json():
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head is noncanonical or belongs to another run"
            )
        return open_head

    def _validate_open_intent(
        self,
        head: RestartIntentHeadRecord,
        head_observation: _ObservedKey,
        intent_observation: _ObservedKey,
    ) -> RestartIntentRecord:
        head_entry = _required_entry(head_observation, "current restart-intent head")
        intent_entry = _immutable_entry(intent_observation, "immutable restart intent")
        if (
            intent_entry.mutation_sequence != 1
            or intent_entry.value_sequence != 1
            or intent_entry.lifetime_sequence != 1
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "immutable restart intent is not an initial creation"
            )
        try:
            intent = RestartIntentRecord.from_json(intent_entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleReadCorrupt(
                "immutable restart intent is malformed"
            ) from error
        if intent_entry.value != intent.to_json():
            raise RestartIntentLifecycleReadCorrupt("immutable restart intent is noncanonical")
        if (
            intent.intent.run_id != self._run_id
            or head.generation != intent.intent.generation
            or head.intent_id != intent.intent.intent_id
            or head.intent_digest != intent.digest
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head does not identify its immutable record"
            )
        if head_entry.committed_at_unix_ms is None or intent_entry.committed_at_unix_ms is None:
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent opening has no authoritative commit time"
            )
        if _transaction_provenance(head_entry) != _transaction_provenance(intent_entry):
            raise RestartIntentLifecycleReadCorrupt(
                "current restart-intent head and immutable intent "
                "do not share one guarded transaction"
            )
        guard_provenance = (
            intent_entry.guard_key,
            intent_entry.guard_revision,
            intent_entry.guard_value_digest,
            intent_entry.guard_committed_at_unix_ms,
            intent_entry.guard_mutation_sequence,
            intent_entry.guard_value_sequence,
            intent_entry.guard_lifetime_sequence,
        )
        if any(value is None for value in guard_provenance):
            raise RestartIntentLifecycleReadCorrupt(
                "immutable restart intent has incomplete coordinator lease provenance"
            )
        if (
            intent_entry.guard_key != self._generation_reader.coordinator_lease_key
            or intent_entry.guard_revision != intent.coordinator_fencing_token
            or intent_entry.guard_value_digest != intent.coordinator_lease_digest
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "immutable restart intent has invalid coordinator lease provenance"
            )
        return intent

    def _validate_open_generation(
        self,
        intent: RestartIntentRecord,
        intent_observation: _ObservedKey,
        generation_result: (tuple[CurrentGeneration, tuple[StoredGenerationSnapshot, ...]] | None),
        opening_authority: CoordinatorLeaseAuthority,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> None:
        if generation_result is None:
            raise RestartIntentLifecycleReadCorrupt(
                "open restart intent exists without committed generation state"
            )
        current, _ = generation_result
        snapshot = current.snapshot
        if (
            snapshot.record.assignment.generation != intent.intent.generation
            or snapshot.record.digest != intent.generation_snapshot_digest
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "current generation does not match the open restart intent"
            )
        active_nodes = set(snapshot.record.assignment.slot_to_node_id.values())
        unknown_nodes = sorted(set(intent.intent.suspected_node_ids) - active_nodes)
        if unknown_nodes:
            raise RestartIntentLifecycleReadCorrupt(
                f"open restart intent suspects nodes outside its generation: {unknown_nodes!r}"
            )
        generation_authority = self._generation_authority(
            snapshot,
            lease_history,
        )
        if lease_history.index(generation_authority) > lease_history.index(opening_authority):
            raise RestartIntentLifecycleReadCorrupt(
                "open restart-intent lease predates its generation authority"
            )
        if (
            snapshot.transaction_sequence <= generation_authority.transaction_sequence
            or snapshot.committed_at_unix_ms < generation_authority.lease.granted_at_unix_ms
            or snapshot.committed_at_unix_ms >= generation_authority.lease.expires_at_unix_ms
            or not _commit_precedes_next_authority(
                lease_history,
                generation_authority,
                transaction_sequence=snapshot.transaction_sequence,
                committed_at_unix_ms=snapshot.committed_at_unix_ms,
            )
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "current generation is outside its coordinator lease window"
            )
        intent_entry = _required_entry(intent_observation, "immutable restart intent")
        committed_at_unix_ms = intent_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated restart intent lost its commit time")
        if (
            intent_entry.transaction_sequence
            <= max(
                snapshot.transaction_sequence,
                opening_authority.transaction_sequence,
            )
            or committed_at_unix_ms
            < max(
                snapshot.committed_at_unix_ms,
                opening_authority.lease.granted_at_unix_ms,
            )
            or committed_at_unix_ms
            >= min(
                opening_authority.lease.expires_at_unix_ms,
                intent.intent.prepare_deadline_unix_ms,
            )
            or not _commit_precedes_next_authority(
                lease_history,
                opening_authority,
                transaction_sequence=intent_entry.transaction_sequence,
                committed_at_unix_ms=committed_at_unix_ms,
            )
        ):
            raise RestartIntentLifecycleReadCorrupt(
                "open restart intent is outside its generation, lease, or deadline window"
            )

    def _generation_authority(
        self,
        snapshot: StoredGenerationSnapshot,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> CoordinatorLeaseAuthority:
        record = CoordinatorLeaseRecord(
            run_id=snapshot.record.assignment.run_id,
            coordinator_id=snapshot.record.coordinator_id,
            lease_id=snapshot.record.lease_id,
            lease_duration_ms=snapshot.record.coordinator_lease_duration_ms,
        )
        matches = tuple(
            authority
            for authority in lease_history
            if (
                authority.lease.record == record
                and authority.lease.fencing_token == snapshot.record.coordinator_fencing_token
                and authority.lease.granted_at_unix_ms == snapshot.guard_committed_at_unix_ms
                and authority.mutation_sequence == snapshot.guard_mutation_sequence
                and authority.value_sequence == snapshot.guard_value_sequence
                and authority.lifetime_sequence == snapshot.guard_lifetime_sequence
            )
        )
        if len(matches) != 1:
            raise RestartIntentLifecycleReadCorrupt(
                "generation coordinator lease is absent from durable lease history"
            )
        return matches[0]

    def _opening_authority(
        self,
        intent: RestartIntentRecord,
        intent_observation: _ObservedKey,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> CoordinatorLeaseAuthority:
        intent_entry = _required_entry(intent_observation, "immutable restart intent")
        provenance = (
            intent_entry.guard_revision,
            intent_entry.guard_committed_at_unix_ms,
            intent_entry.guard_mutation_sequence,
            intent_entry.guard_value_sequence,
            intent_entry.guard_lifetime_sequence,
        )
        if any(value is None for value in provenance):
            raise RestartIntentLifecycleReadCorrupt(
                "immutable restart intent has incomplete coordinator lease provenance"
            )
        (
            guard_revision,
            guard_committed_at_unix_ms,
            guard_mutation_sequence,
            guard_value_sequence,
            guard_lifetime_sequence,
        ) = provenance
        lease_record = CoordinatorLeaseRecord(
            run_id=intent.intent.run_id,
            coordinator_id=intent.coordinator_id,
            lease_id=intent.lease_id,
            lease_duration_ms=intent.coordinator_lease_duration_ms,
        )
        matches = tuple(
            authority
            for authority in lease_history
            if (
                authority.lease.record == lease_record
                and authority.lease.fencing_token == guard_revision
                and authority.lease.granted_at_unix_ms == guard_committed_at_unix_ms
                and authority.mutation_sequence == guard_mutation_sequence
                and authority.value_sequence == guard_value_sequence
                and authority.lifetime_sequence == guard_lifetime_sequence
            )
        )
        if len(matches) != 1:
            raise RestartIntentLifecycleReadCorrupt(
                "opening coordinator lease is absent from durable lease history"
            )
        return matches[0]

    def _read_dependencies(
        self,
    ) -> tuple[
        tuple[CoordinatorLeaseAuthority, ...],
        tuple[CurrentGeneration, tuple[StoredGenerationSnapshot, ...]] | None,
    ]:
        try:
            lease_history = self._lease_history_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle dependencies are corrupt"
            ) from error
        return lease_history, self._read_generation()

    def _read_generation(
        self,
    ) -> tuple[CurrentGeneration, tuple[StoredGenerationSnapshot, ...]] | None:
        try:
            return self._generation_reader.current_with_history()
        except GenerationStateCorrupt as error:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent lifecycle dependencies are corrupt"
            ) from error

    def _authenticate_closure(
        self,
        head_observation: _ObservedKey,
        lifecycle_observation: _ObservedKey,
        closure_observation: _ObservedKey,
        intent_observation: _ObservedKey,
        generation_snapshot: StoredGenerationSnapshot | None,
        successor: StoredGenerationSnapshot | None,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> AuthenticatedInitialRestartIntentClosure:
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
        if generation_snapshot is None:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure references a missing generation"
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
            return AuthenticatedInitialRestartIntentClosure(
                state=state,
                generation_snapshot=generation_snapshot,
                immediate_successor=successor,
                lease_history=lease_history,
            )
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleReadCorrupt(
                "persisted restart-intent closure is contradictory"
            ) from error

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


def _transaction_provenance(entry: ControlStoreEntry) -> tuple[object, ...]:
    return (
        entry.committed_at_unix_ms,
        entry.transaction_sequence,
        entry.guard_key,
        entry.guard_revision,
        entry.guard_value_digest,
        entry.guard_committed_at_unix_ms,
        entry.guard_mutation_sequence,
        entry.guard_value_sequence,
        entry.guard_lifetime_sequence,
    )


def _commit_precedes_next_authority(
    lease_history: tuple[CoordinatorLeaseAuthority, ...],
    authority: CoordinatorLeaseAuthority,
    *,
    transaction_sequence: int,
    committed_at_unix_ms: int,
) -> bool:
    authority_index = lease_history.index(authority)
    if authority_index + 1 == len(lease_history):
        return True
    successor = lease_history[authority_index + 1]
    if successor.lifetime_sequence != authority.lifetime_sequence:
        return False
    return (
        transaction_sequence < successor.transaction_sequence
        and committed_at_unix_ms <= successor.lease.granted_at_unix_ms
    )


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
]
