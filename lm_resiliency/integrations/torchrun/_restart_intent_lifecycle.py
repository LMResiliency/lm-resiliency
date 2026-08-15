"""Fail-closed observation of restart-intent lifecycle state."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class RestartIntentLifecycleError(RuntimeError):
    """Base error for restart-intent lifecycle observation."""


class RestartIntentLifecycleConflict(RestartIntentLifecycleError):
    """Raised when the supplied generation is no longer current."""


class RestartIntentLifecycleCorrupt(RestartIntentLifecycleError):
    """Raised when persisted lifecycle state is malformed or contradictory."""


@dataclass(frozen=True, slots=True)
class StoredRestartIntentLifecycle:
    """One verified lifecycle record and its exact store revision."""

    record: RestartIntentLifecycleRecord
    revision: int
    committed_at_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, RestartIntentLifecycleRecord):
            raise TypeError(
                "StoredRestartIntentLifecycle.record must be RestartIntentLifecycleRecord"
            )
        _positive_integer(
            self.revision,
            "StoredRestartIntentLifecycle.revision",
        )
        _positive_integer(
            self.committed_at_unix_ms,
            "StoredRestartIntentLifecycle.committed_at_unix_ms",
        )


class RestartIntentLifecycleReader:
    """Read the permanent last-closed-intent fence for one run."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._generation_reader = GenerationStateReader(store, run_id=self._run_id)
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._lifecycle_key = f"{self._run_prefix}/restart-intent-lifecycle"

    @property
    def coordinator_lease_key(self) -> str:
        return self._generation_reader.coordinator_lease_key

    @property
    def lifecycle_key(self) -> str:
        return self._lifecycle_key

    def intent_key(self, intent_id: str) -> str:
        normalized_intent_id = _nonempty_string(intent_id, "intent_id")
        intent_digest = hashlib.sha256(normalized_intent_id.encode("utf-8")).hexdigest()
        return f"{self._run_prefix}/restart-intents/{intent_digest}"

    def current(
        self,
        generation: CurrentGeneration,
    ) -> StoredRestartIntentLifecycle | None:
        if not isinstance(generation, CurrentGeneration):
            raise TypeError("generation must be CurrentGeneration")
        for _ in range(_MAX_READ_ATTEMPTS):
            if self._generation_reader.current() != generation:
                raise RestartIntentLifecycleConflict(
                    "generation does not match the committed generation head"
                )
            entry = self._store.get(self._lifecycle_key)
            if entry is None:
                has_history = self._store.has_history(self._lifecycle_key)
                if self._store.get(self._lifecycle_key) is not None:
                    continue
                if has_history:
                    raise RestartIntentLifecycleCorrupt(
                        "restart-intent lifecycle record was deleted"
                    )
                return None
            stored = self._decode_entry(entry, generation)
            observed = self._store.get(self._lifecycle_key)
            if observed is None or observed.revision != entry.revision:
                continue
            if self._generation_reader.current() != generation:
                raise RestartIntentLifecycleConflict(
                    "generation changed during restart-intent lifecycle observation"
                )
            return stored
        raise RestartIntentLifecycleError(
            "restart-intent lifecycle record changed repeatedly during read"
        )

    def _decode_entry(
        self,
        entry: ControlStoreEntry,
        generation: CurrentGeneration,
    ) -> StoredRestartIntentLifecycle:
        try:
            record = RestartIntentLifecycleRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleCorrupt(
                "restart-intent lifecycle record is malformed"
            ) from error
        current_generation = generation.snapshot.record.assignment.generation
        if (
            record.closed_intent.run_id != self._run_id
            or record.closed_intent.generation > current_generation
        ):
            raise RestartIntentLifecycleCorrupt(
                "restart-intent lifecycle record contradicts the current generation"
            )
        closed_intent_entry = self._store.get(self.intent_key(record.closed_intent.intent_id))
        if closed_intent_entry is None:
            raise RestartIntentLifecycleCorrupt(
                "restart-intent lifecycle record references a missing intent"
            )
        closed_intent, closed_intent_committed_at_unix_ms = self._decode_closed_intent(
            closed_intent_entry,
            record,
        )
        snapshot = self._generation_reader.get(closed_intent.intent.generation)
        if snapshot is None or snapshot.record.digest != closed_intent.generation_snapshot_digest:
            raise RestartIntentLifecycleCorrupt(
                "closed restart intent does not identify its generation snapshot"
            )
        committed_at_unix_ms = _guarded_commit_time(
            entry,
            guard_key=self.coordinator_lease_key,
            guard_revision=record.coordinator_fencing_token,
            guard_value_digest=record.coordinator_lease_digest,
            lease_duration_ms=record.coordinator_lease_duration_ms,
            path="restart-intent lifecycle record",
        )
        if (
            entry.lifetime_sequence != 1
            or entry.value_sequence != entry.mutation_sequence
            or committed_at_unix_ms < closed_intent_committed_at_unix_ms
        ):
            raise RestartIntentLifecycleCorrupt(
                "restart-intent lifecycle record has invalid store provenance"
            )
        return StoredRestartIntentLifecycle(
            record=record,
            revision=entry.revision,
            committed_at_unix_ms=committed_at_unix_ms,
        )

    def _decode_closed_intent(
        self,
        entry: ControlStoreEntry,
        lifecycle: RestartIntentLifecycleRecord,
    ) -> tuple[RestartIntentRecord, int]:
        try:
            record = RestartIntentRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentLifecycleCorrupt(
                "closed restart-intent record is malformed"
            ) from error
        head = lifecycle.closed_intent
        if (
            record.intent.run_id != head.run_id
            or record.intent.generation != head.generation
            or record.intent.intent_id != head.intent_id
            or record.digest != head.intent_digest
        ):
            raise RestartIntentLifecycleCorrupt(
                "restart-intent lifecycle record does not identify its closed intent"
            )
        committed_at_unix_ms = _guarded_commit_time(
            entry,
            guard_key=self.coordinator_lease_key,
            guard_revision=record.coordinator_fencing_token,
            guard_value_digest=record.coordinator_lease_digest,
            lease_duration_ms=record.coordinator_lease_duration_ms,
            path="closed restart-intent record",
        )
        if (
            entry.lifetime_sequence != 1
            or entry.mutation_sequence != 1
            or entry.value_sequence != 1
        ):
            raise RestartIntentLifecycleCorrupt(
                "closed restart-intent record has invalid store provenance"
            )
        return record, committed_at_unix_ms


def _guarded_commit_time(
    entry: ControlStoreEntry,
    *,
    guard_key: str,
    guard_revision: int,
    guard_value_digest: str,
    lease_duration_ms: int,
    path: str,
) -> int:
    guard_committed_at_unix_ms = entry.guard_committed_at_unix_ms
    committed_at_unix_ms = entry.committed_at_unix_ms
    if (
        entry.guard_key != guard_key
        or entry.guard_revision != guard_revision
        or entry.guard_value_digest != guard_value_digest
        or guard_committed_at_unix_ms is None
        or committed_at_unix_ms is None
        or committed_at_unix_ms < guard_committed_at_unix_ms
        or committed_at_unix_ms >= guard_committed_at_unix_ms + lease_duration_ms
    ):
        raise RestartIntentLifecycleCorrupt(f"{path} has invalid guard provenance")
    return committed_at_unix_ms


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "RestartIntentLifecycleConflict",
    "RestartIntentLifecycleCorrupt",
    "RestartIntentLifecycleError",
    "RestartIntentLifecycleReader",
    "StoredRestartIntentLifecycle",
]
