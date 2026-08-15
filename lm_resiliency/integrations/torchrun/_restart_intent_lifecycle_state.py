"""Canonical persisted state for the first restart-intent closure."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import ControlStoreEntry
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class PersistedInitialRestartIntentClosure:
    """One canonical first closure reconstructed from persisted store entries."""

    intent: RestartIntentRecord
    open_head: RestartIntentHeadRecord
    closed_head: RestartIntentClosedHeadRecord
    lifecycle: RestartIntentLifecycleRecord
    lifecycle_head: RestartIntentLifecycleHeadRecord
    intent_entry: ControlStoreEntry
    open_head_entry: ControlStoreEntry
    closed_head_entry: ControlStoreEntry
    lifecycle_entry: ControlStoreEntry
    lifecycle_head_entry: ControlStoreEntry

    def __post_init__(self) -> None:
        record_types = (
            ("intent", self.intent, RestartIntentRecord),
            ("open_head", self.open_head, RestartIntentHeadRecord),
            ("closed_head", self.closed_head, RestartIntentClosedHeadRecord),
            ("lifecycle", self.lifecycle, RestartIntentLifecycleRecord),
            (
                "lifecycle_head",
                self.lifecycle_head,
                RestartIntentLifecycleHeadRecord,
            ),
        )
        for path, value, expected_type in record_types:
            if not isinstance(value, expected_type):
                raise TypeError(
                    f"PersistedInitialRestartIntentClosure.{path} must be {expected_type.__name__}"
                )
        entries = (
            ("intent_entry", self.intent_entry),
            ("open_head_entry", self.open_head_entry),
            ("closed_head_entry", self.closed_head_entry),
            ("lifecycle_entry", self.lifecycle_entry),
            ("lifecycle_head_entry", self.lifecycle_head_entry),
        )
        for path, entry in entries:
            if not isinstance(entry, ControlStoreEntry):
                raise TypeError(
                    f"PersistedInitialRestartIntentClosure.{path} must be ControlStoreEntry"
                )
        self._validate_records()
        self._validate_entries()

    @classmethod
    def from_entries(
        cls,
        *,
        run_id: str,
        intent_entry: ControlStoreEntry,
        open_head_entry: ControlStoreEntry,
        closed_head_entry: ControlStoreEntry,
        lifecycle_entry: ControlStoreEntry,
        lifecycle_head_entry: ControlStoreEntry,
    ) -> PersistedInitialRestartIntentClosure:
        """Decode and validate one first-closure entry bundle."""

        normalized_run_id = _nonempty_string(run_id, "run_id")
        entries = {
            "intent": intent_entry,
            "open_head": open_head_entry,
            "closed_head": closed_head_entry,
            "lifecycle": lifecycle_entry,
            "lifecycle_head": lifecycle_head_entry,
        }
        for path, entry in entries.items():
            if not isinstance(entry, ControlStoreEntry):
                raise TypeError(f"{path}_entry must be ControlStoreEntry")
        try:
            intent = RestartIntentRecord.from_json(intent_entry.value)
            open_head = RestartIntentHeadRecord.from_json(open_head_entry.value)
            closed_head = RestartIntentClosedHeadRecord.from_json(closed_head_entry.value)
            lifecycle = RestartIntentLifecycleRecord.from_json(lifecycle_entry.value)
            lifecycle_head = RestartIntentLifecycleHeadRecord.from_json(lifecycle_head_entry.value)
        except (TypeError, ValueError) as error:
            raise ValueError("persisted initial restart-intent closure is malformed") from error
        decoded = cls(
            intent=intent,
            open_head=open_head,
            closed_head=closed_head,
            lifecycle=lifecycle,
            lifecycle_head=lifecycle_head,
            intent_entry=intent_entry,
            open_head_entry=open_head_entry,
            closed_head_entry=closed_head_entry,
            lifecycle_entry=lifecycle_entry,
            lifecycle_head_entry=lifecycle_head_entry,
        )
        if decoded.intent.intent.run_id != normalized_run_id:
            raise ValueError("persisted initial restart-intent closure belongs to another run")
        return decoded

    @property
    def opened_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.intent_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated opening lost its commit time")
        return committed_at_unix_ms

    @property
    def closed_at_unix_ms(self) -> int:
        committed_at_unix_ms = self.closed_head_entry.committed_at_unix_ms
        if committed_at_unix_ms is None:
            raise AssertionError("validated closure lost its commit time")
        return committed_at_unix_ms

    @property
    def opening_transaction_sequence(self) -> int:
        return self.intent_entry.transaction_sequence

    @property
    def closing_transaction_sequence(self) -> int:
        return self.closed_head_entry.transaction_sequence

    def _validate_records(self) -> None:
        if (
            self.intent.intent.run_id != self.open_head.run_id
            or self.intent.intent.generation != self.open_head.generation
            or self.intent.intent.intent_id != self.open_head.intent_id
            or self.intent.digest != self.open_head.intent_digest
            or self.lifecycle.closed_intent != self.open_head
            or self.lifecycle_head.run_id != self.open_head.run_id
            or self.lifecycle_head.closure_index != 1
            or self.lifecycle_head.generation != self.open_head.generation
            or self.lifecycle_head.intent_id != self.open_head.intent_id
            or self.lifecycle_head.lifecycle_digest != self.lifecycle.digest
            or self.closed_head.run_id != self.lifecycle_head.run_id
            or self.closed_head.closure_index != self.lifecycle_head.closure_index
            or self.closed_head.generation != self.lifecycle_head.generation
            or self.closed_head.intent_id != self.lifecycle_head.intent_id
            or self.closed_head.lifecycle_head_digest != self.lifecycle_head.digest
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure records do not form one initial closure"
            )

    def _validate_entries(self) -> None:
        record_entries = (
            (self.intent, self.intent_entry, "intent_entry"),
            (self.open_head, self.open_head_entry, "open_head_entry"),
            (self.closed_head, self.closed_head_entry, "closed_head_entry"),
            (self.lifecycle, self.lifecycle_entry, "lifecycle_entry"),
            (
                self.lifecycle_head,
                self.lifecycle_head_entry,
                "lifecycle_head_entry",
            ),
        )
        for record, entry, path in record_entries:
            if entry.value != record.to_json():
                raise ValueError(
                    f"PersistedInitialRestartIntentClosure.{path} is noncanonical "
                    "or does not match its record"
                )
            if entry.committed_at_unix_ms is None:
                raise ValueError(f"PersistedInitialRestartIntentClosure.{path} has no commit time")
        for entry, path in (
            (self.intent_entry, "intent_entry"),
            (self.open_head_entry, "open_head_entry"),
            (self.lifecycle_entry, "lifecycle_entry"),
            (self.lifecycle_head_entry, "lifecycle_head_entry"),
        ):
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
            ):
                raise ValueError(
                    f"PersistedInitialRestartIntentClosure.{path} is not an "
                    "immutable initial creation"
                )
        if (
            self.closed_head_entry.mutation_sequence != 2
            or self.closed_head_entry.value_sequence != 2
            or self.closed_head_entry.lifetime_sequence != 1
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure.closed_head_entry does not "
                "replace exactly one open head"
            )
        canonical_guard_key = _coordinator_lease_key(self.intent.intent.run_id)
        opening_guard = _guard_provenance(self.intent_entry, canonical_guard_key)
        if _guard_provenance(self.open_head_entry, canonical_guard_key) != opening_guard:
            raise ValueError(
                "PersistedInitialRestartIntentClosure opening entries do not share one guard"
            )
        if (
            self.intent_entry.committed_at_unix_ms != self.open_head_entry.committed_at_unix_ms
            or self.intent_entry.transaction_sequence != self.open_head_entry.transaction_sequence
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure opening entries do not share one transaction"
            )
        if (
            self.intent.coordinator_fencing_token != self.intent_entry.guard_revision
            or self.intent.coordinator_lease_digest != self.intent_entry.guard_value_digest
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure opening authority does not match its guard"
            )
        closure_entries = (
            self.closed_head_entry,
            self.lifecycle_entry,
            self.lifecycle_head_entry,
        )
        closing_guard = _guard_provenance(
            self.closed_head_entry,
            canonical_guard_key,
        )
        if any(
            _guard_provenance(entry, canonical_guard_key) != closing_guard
            for entry in closure_entries[1:]
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure closure entries do not share one guard"
            )
        if any(
            entry.committed_at_unix_ms != self.closed_head_entry.committed_at_unix_ms
            or entry.transaction_sequence != self.closed_head_entry.transaction_sequence
            for entry in closure_entries[1:]
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure closure entries do not share one transaction"
            )
        if (
            self.lifecycle.coordinator_fencing_token != self.closed_head_entry.guard_revision
            or self.lifecycle.coordinator_lease_digest != self.closed_head_entry.guard_value_digest
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure closing authority does not match its guard"
            )
        if (
            self.closing_transaction_sequence <= self.opening_transaction_sequence
            or self.closed_at_unix_ms < self.opened_at_unix_ms
        ):
            raise ValueError(
                "PersistedInitialRestartIntentClosure closure does not follow its opening"
            )


def _guard_provenance(
    entry: ControlStoreEntry,
    expected_guard_key: str,
) -> tuple[object, ...]:
    provenance = (
        entry.guard_key,
        entry.guard_revision,
        entry.guard_value_digest,
        entry.guard_mutation_sequence,
        entry.guard_value_sequence,
        entry.guard_lifetime_sequence,
        entry.guard_committed_at_unix_ms,
    )
    if entry.guard_key != expected_guard_key or any(value is None for value in provenance):
        raise ValueError("PersistedInitialRestartIntentClosure entry has invalid guard provenance")
    return provenance


def _coordinator_lease_key(run_id: str) -> str:
    run_digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
    return f"{_CONTROL_PREFIX}/runs/{run_digest}/coordinator-lease"


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = ["PersistedInitialRestartIntentClosure"]
