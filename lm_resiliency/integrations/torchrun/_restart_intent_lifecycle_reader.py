"""Fail-closed reads of the first committed restart-intent closure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
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
                self._validate_absent_lifecycle(
                    head_observation,
                    lifecycle_observation,
                    closure_observation,
                )
                return None
            lifecycle_head = self._decode_lifecycle_head(lifecycle_observation)
            closure_key = self.closure_key(lifecycle_head.closure_index)
            intent_key = self.intent_key(lifecycle_head.intent_id)
            closure_observation = self._observe(closure_key)
            intent_observation = self._observe(intent_key)
            try:
                lease_history = self._lease_history_reader.read()
                generation_result = self._generation_reader.current_with_history()
            except (
                CoordinatorLeaseHistoryCorrupt,
                GenerationStateCorrupt,
            ) as error:
                raise RestartIntentLifecycleReadCorrupt(
                    "restart-intent lifecycle dependencies are corrupt"
                ) from error
            generation_snapshot = None
            successor = None
            if generation_result is not None:
                _, generation_history = generation_result
                if lifecycle_head.generation < len(generation_history):
                    generation_snapshot = generation_history[lifecycle_head.generation]
                    successor_index = lifecycle_head.generation + 1
                    if successor_index < len(generation_history):
                        successor = generation_history[successor_index]
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
    ) -> None:
        if lifecycle.has_history or lifecycle.history:
            raise RestartIntentLifecycleReadCorrupt("restart-intent lifecycle head was deleted")
        if closure.entry is not None or closure.has_history or closure.history:
            raise RestartIntentLifecycleReadCorrupt(
                "restart-intent closure exists without a lifecycle head"
            )
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
