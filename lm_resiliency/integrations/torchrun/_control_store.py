"""Atomic control-store primitives for the internal torchrun integration."""

from __future__ import annotations

import hashlib
import threading
import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol


class ControlStoreError(RuntimeError):
    """Base error for torchrun control-store operations."""


class ControlStoreConflict(ControlStoreError):
    """Raised when a conditional mutation observes an unexpected revision."""

    def __init__(
        self,
        key: str,
        expected_revision: int | None,
        actual_revision: int | None,
    ) -> None:
        self.key = key
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        super().__init__(
            f"control-store conflict for {key!r}: expected revision "
            f"{expected_revision!r}, found {actual_revision!r}"
        )


class ControlStoreHistoryConflict(ControlStoreError):
    """Raised when a create-once mutation targets a previously used key."""

    def __init__(self, key: str) -> None:
        self.key = key
        super().__init__(f"control-store key {key!r} has prior committed history")


class ControlStoreDeadlineExceeded(ControlStoreError):
    """Raised when a conditional mutation reaches its store-side deadline."""

    def __init__(self, key: str, deadline_unix_ms: int, observed_unix_ms: int) -> None:
        self.key = key
        self.deadline_unix_ms = deadline_unix_ms
        self.observed_unix_ms = observed_unix_ms
        super().__init__(
            f"control-store deadline for {key!r} elapsed at {observed_unix_ms}; "
            f"deadline was {deadline_unix_ms}"
        )


class ControlStoreTooEarly(ControlStoreError):
    """Raised when a conditional mutation precedes its store-side start time."""

    def __init__(self, key: str, not_before_unix_ms: int, observed_unix_ms: int) -> None:
        self.key = key
        self.not_before_unix_ms = not_before_unix_ms
        self.observed_unix_ms = observed_unix_ms
        super().__init__(
            f"control-store time for {key!r} was {observed_unix_ms}; "
            f"mutation is not valid before {not_before_unix_ms}"
        )


class ControlStoreClockError(ControlStoreError):
    """Raised when the authoritative store clock moves backward."""


@dataclass(frozen=True, slots=True)
class ControlStoreEntry:
    """One immutable control-store value and its opaque per-key revision."""

    value: bytes
    revision: int
    committed_at_unix_ms: int | None = None
    transaction_sequence: int = 1
    mutation_sequence: int = 1
    value_sequence: int = 1
    lifetime_sequence: int = 1
    guard_key: str | None = None
    guard_revision: int | None = None
    guard_value_digest: str | None = None
    guard_mutation_sequence: int | None = None
    guard_value_sequence: int | None = None
    guard_lifetime_sequence: int | None = None
    guard_committed_at_unix_ms: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _control_value(self.value))
        object.__setattr__(self, "revision", _required_revision(self.revision))
        object.__setattr__(
            self,
            "transaction_sequence",
            _positive_integer(self.transaction_sequence, "transaction_sequence"),
        )
        object.__setattr__(
            self,
            "mutation_sequence",
            _positive_integer(self.mutation_sequence, "mutation_sequence"),
        )
        object.__setattr__(
            self,
            "value_sequence",
            _positive_integer(self.value_sequence, "value_sequence"),
        )
        object.__setattr__(
            self,
            "lifetime_sequence",
            _positive_integer(self.lifetime_sequence, "lifetime_sequence"),
        )
        _validate_sequence_lineage(
            self.mutation_sequence,
            self.value_sequence,
            self.lifetime_sequence,
            path="control-store entry",
        )
        if self.committed_at_unix_ms is not None:
            object.__setattr__(
                self,
                "committed_at_unix_ms",
                _positive_integer(
                    self.committed_at_unix_ms,
                    "committed_at_unix_ms",
                ),
            )
        guard_values = (
            self.guard_key,
            self.guard_revision,
            self.guard_value_digest,
            self.guard_mutation_sequence,
            self.guard_value_sequence,
            self.guard_lifetime_sequence,
        )
        if any(value is not None for value in guard_values):
            if not all(value is not None for value in guard_values):
                raise ValueError("control-store guard provenance must be complete")
            object.__setattr__(self, "guard_key", _control_key(self.guard_key))
            object.__setattr__(
                self,
                "guard_revision",
                _required_revision(self.guard_revision),
            )
            object.__setattr__(
                self,
                "guard_value_digest",
                _sha256_digest(
                    self.guard_value_digest,
                    "guard_value_digest",
                ),
            )
            guard_mutation_sequence = _positive_integer(
                self.guard_mutation_sequence,
                "guard_mutation_sequence",
            )
            guard_value_sequence = _positive_integer(
                self.guard_value_sequence,
                "guard_value_sequence",
            )
            guard_lifetime_sequence = _positive_integer(
                self.guard_lifetime_sequence,
                "guard_lifetime_sequence",
            )
            object.__setattr__(
                self,
                "guard_mutation_sequence",
                guard_mutation_sequence,
            )
            object.__setattr__(
                self,
                "guard_value_sequence",
                guard_value_sequence,
            )
            object.__setattr__(
                self,
                "guard_lifetime_sequence",
                guard_lifetime_sequence,
            )
            _validate_sequence_lineage(
                guard_mutation_sequence,
                guard_value_sequence,
                guard_lifetime_sequence,
                path="control-store guard provenance",
            )
            if self.guard_committed_at_unix_ms is not None:
                object.__setattr__(
                    self,
                    "guard_committed_at_unix_ms",
                    _positive_integer(
                        self.guard_committed_at_unix_ms,
                        "guard_committed_at_unix_ms",
                    ),
                )
        elif self.guard_committed_at_unix_ms is not None:
            raise ValueError("control-store guard grant time requires guard provenance")


@dataclass(frozen=True, slots=True)
class ControlStoreWrite:
    """One expected revision and immutable value in an atomic store transaction."""

    expected_revision: int | None
    value: bytes
    require_never_created: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "expected_revision",
            _expected_revision(self.expected_revision, allow_absent=True),
        )
        object.__setattr__(self, "value", _control_value(self.value))
        if not isinstance(self.require_never_created, bool):
            raise TypeError("require_never_created must be bool")
        if self.require_never_created and self.expected_revision is not None:
            raise ValueError("require_never_created is valid only for create-if-absent writes")


class ControlStore(Protocol):
    """Strongly consistent per-key compare-and-set storage."""

    def get(self, key: str) -> ControlStoreEntry | None:
        """Return the current value, or ``None`` when the key is absent."""
        ...

    def has_history(self, key: str) -> bool:
        """Return whether ``key`` has ever held a committed value.

        History remains true after deletion so callers can distinguish a
        never-created key from a deleted authoritative record.
        """
        ...

    def compare_set(
        self,
        key: str,
        *,
        expected_revision: int | None,
        value: bytes,
    ) -> ControlStoreEntry:
        """Create or replace ``key`` when its revision matches.

        ``expected_revision=None`` means create only when the key is absent.
        """
        ...

    def compare_delete(self, key: str, *, expected_revision: int) -> int:
        """Delete ``key`` when its revision matches and return the tombstone revision."""
        ...

    def compare_set_in_window(
        self,
        key: str,
        *,
        expected_revision: int | None,
        not_before_unix_ms: int,
        deadline_unix_ms: int | None,
        value: bytes,
    ) -> ControlStoreEntry:
        """Create or replace ``key`` in a store-time window.

        The start is inclusive. ``deadline_unix_ms=None`` leaves the end
        unbounded. The returned entry carries the authoritative commit time.
        """
        ...

    def compare_delete_in_window(
        self,
        key: str,
        *,
        expected_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
    ) -> int:
        """Delete ``key`` within an inclusive-start, exclusive-end window."""
        ...

    def compare_set_many_guarded(
        self,
        writes: Mapping[str, ControlStoreWrite],
        *,
        guard_key: str,
        expected_guard_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        conditions: Mapping[str, int | None] | None = None,
        never_created_conditions: Collection[str] | None = None,
    ) -> Mapping[str, ControlStoreEntry]:
        """Atomically publish writes while a guard and read conditions hold.

        Every returned target entry carries store-stamped guard key, revision,
        value digest, ordered mutation and value sequences, key-lifetime
        sequence, and authoritative guard commit time from the same
        linearization point.
        """
        ...


class InMemoryControlStore:
    """Thread-safe in-memory implementation used by coordinator contract tests."""

    def __init__(self, *, clock: Callable[[], int] | None = None) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ControlStoreEntry] = {}
        self._last_revisions: dict[str, int] = {}
        self._mutation_sequences: dict[str, int] = {}
        self._value_sequences: dict[str, int] = {}
        self._lifetime_sequences: dict[str, int] = {}
        self._transaction_sequence = 0
        self._clock = clock or _system_unix_ms
        self._last_now_unix_ms = 0

    def get(self, key: str) -> ControlStoreEntry | None:
        normalized_key = _control_key(key)
        with self._lock:
            return self._entries.get(normalized_key)

    def has_history(self, key: str) -> bool:
        normalized_key = _control_key(key)
        with self._lock:
            return normalized_key in self._last_revisions

    def compare_set(
        self,
        key: str,
        *,
        expected_revision: int | None,
        value: bytes,
    ) -> ControlStoreEntry:
        normalized_key = _control_key(key)
        normalized_revision = _expected_revision(expected_revision, allow_absent=True)
        normalized_value = _control_value(value)
        with self._lock:
            self._require_revision(normalized_key, normalized_revision)
            return self._set_entry(
                normalized_key,
                normalized_value,
                transaction_sequence=self._next_transaction_sequence(),
            )

    def compare_delete(self, key: str, *, expected_revision: int) -> int:
        normalized_key = _control_key(key)
        normalized_revision = _required_revision(expected_revision)
        with self._lock:
            self._require_revision(normalized_key, normalized_revision)
            self._next_transaction_sequence()
            del self._entries[normalized_key]
            return self._next_revision(normalized_key)

    def compare_set_in_window(
        self,
        key: str,
        *,
        expected_revision: int | None,
        not_before_unix_ms: int,
        deadline_unix_ms: int | None,
        value: bytes,
    ) -> ControlStoreEntry:
        normalized_key = _control_key(key)
        normalized_revision = _expected_revision(expected_revision, allow_absent=True)
        normalized_not_before = _positive_integer(
            not_before_unix_ms,
            "not_before_unix_ms",
        )
        normalized_deadline = (
            None
            if deadline_unix_ms is None
            else _positive_integer(deadline_unix_ms, "deadline_unix_ms")
        )
        if normalized_deadline is not None and normalized_not_before >= normalized_deadline:
            raise ValueError("not_before_unix_ms must be before deadline_unix_ms")
        normalized_value = _control_value(value)
        with self._lock:
            self._require_revision(normalized_key, normalized_revision)
            now_unix_ms = self._store_now_unix_ms()
            if now_unix_ms < normalized_not_before:
                raise ControlStoreTooEarly(
                    normalized_key,
                    normalized_not_before,
                    now_unix_ms,
                )
            if normalized_deadline is not None and now_unix_ms >= normalized_deadline:
                raise ControlStoreDeadlineExceeded(
                    normalized_key,
                    normalized_deadline,
                    now_unix_ms,
                )
            return self._set_entry(
                normalized_key,
                normalized_value,
                committed_at_unix_ms=now_unix_ms,
                transaction_sequence=self._next_transaction_sequence(),
            )

    def compare_delete_in_window(
        self,
        key: str,
        *,
        expected_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
    ) -> int:
        normalized_key = _control_key(key)
        normalized_revision = _required_revision(expected_revision)
        normalized_not_before = _positive_integer(
            not_before_unix_ms,
            "not_before_unix_ms",
        )
        normalized_deadline = _positive_integer(
            deadline_unix_ms,
            "deadline_unix_ms",
        )
        if normalized_not_before >= normalized_deadline:
            raise ValueError("not_before_unix_ms must be before deadline_unix_ms")
        with self._lock:
            self._require_revision(normalized_key, normalized_revision)
            now_unix_ms = self._store_now_unix_ms()
            if now_unix_ms < normalized_not_before:
                raise ControlStoreTooEarly(
                    normalized_key,
                    normalized_not_before,
                    now_unix_ms,
                )
            if now_unix_ms >= normalized_deadline:
                raise ControlStoreDeadlineExceeded(
                    normalized_key,
                    normalized_deadline,
                    now_unix_ms,
                )
            self._next_transaction_sequence()
            del self._entries[normalized_key]
            return self._next_revision(normalized_key)

    def compare_set_many_guarded(
        self,
        writes: Mapping[str, ControlStoreWrite],
        *,
        guard_key: str,
        expected_guard_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        conditions: Mapping[str, int | None] | None = None,
        never_created_conditions: Collection[str] | None = None,
    ) -> Mapping[str, ControlStoreEntry]:
        normalized_writes = _control_writes(writes)
        normalized_conditions = _control_conditions(conditions)
        normalized_never_created_conditions = _control_key_collection(
            never_created_conditions,
            "never_created_conditions",
        )
        normalized_guard_key = _control_key(guard_key)
        if normalized_guard_key in normalized_writes:
            raise ValueError("guard_key must not also be a transaction target")
        if normalized_guard_key in normalized_conditions:
            raise ValueError("guard_key must not also be a transaction condition")
        if normalized_guard_key in normalized_never_created_conditions:
            raise ValueError("guard_key must not also be a never-created condition")
        overlapping_keys = sorted(set(normalized_writes) & set(normalized_conditions))
        if overlapping_keys:
            raise ValueError(
                f"transaction conditions must not also be targets: {overlapping_keys!r}"
            )
        overlapping_never_created_targets = sorted(
            set(normalized_writes) & normalized_never_created_conditions
        )
        if overlapping_never_created_targets:
            raise ValueError(
                "never-created conditions must not also be targets: "
                f"{overlapping_never_created_targets!r}"
            )
        overlapping_conditions = sorted(
            set(normalized_conditions) & normalized_never_created_conditions
        )
        if overlapping_conditions:
            raise ValueError(
                "never-created conditions must not also be revision conditions: "
                f"{overlapping_conditions!r}"
            )
        normalized_guard_revision = _required_revision(expected_guard_revision)
        normalized_not_before = _positive_integer(
            not_before_unix_ms,
            "not_before_unix_ms",
        )
        normalized_deadline = _positive_integer(
            deadline_unix_ms,
            "deadline_unix_ms",
        )
        if normalized_not_before >= normalized_deadline:
            raise ValueError("not_before_unix_ms must be before deadline_unix_ms")
        with self._lock:
            self._require_revision(normalized_guard_key, normalized_guard_revision)
            guard_entry = self._entries[normalized_guard_key]
            guard_value_digest = hashlib.sha256(guard_entry.value).hexdigest()
            for key, expected_revision in normalized_conditions.items():
                self._require_revision(key, expected_revision)
            for key in sorted(normalized_never_created_conditions):
                self._require_revision(key, None)
                if key in self._last_revisions:
                    raise ControlStoreHistoryConflict(key)
            for key, write in normalized_writes.items():
                self._require_revision(key, write.expected_revision)
                if write.require_never_created and key in self._last_revisions:
                    raise ControlStoreHistoryConflict(key)
            now_unix_ms = self._store_now_unix_ms()
            if now_unix_ms < normalized_not_before:
                raise ControlStoreTooEarly(
                    normalized_guard_key,
                    normalized_not_before,
                    now_unix_ms,
                )
            if now_unix_ms >= normalized_deadline:
                raise ControlStoreDeadlineExceeded(
                    normalized_guard_key,
                    normalized_deadline,
                    now_unix_ms,
                )
            transaction_sequence = self._next_transaction_sequence()
            committed = {
                key: self._set_entry(
                    key,
                    write.value,
                    committed_at_unix_ms=now_unix_ms,
                    transaction_sequence=transaction_sequence,
                    guard_key=normalized_guard_key,
                    guard_revision=normalized_guard_revision,
                    guard_value_digest=guard_value_digest,
                    guard_mutation_sequence=guard_entry.mutation_sequence,
                    guard_value_sequence=guard_entry.value_sequence,
                    guard_lifetime_sequence=guard_entry.lifetime_sequence,
                    guard_committed_at_unix_ms=guard_entry.committed_at_unix_ms,
                )
                for key, write in normalized_writes.items()
            }
            return MappingProxyType(committed)

    def _require_revision(self, key: str, expected_revision: int | None) -> None:
        entry = self._entries.get(key)
        actual_revision = None if entry is None else entry.revision
        if actual_revision != expected_revision:
            raise ControlStoreConflict(key, expected_revision, actual_revision)

    def _set_entry(
        self,
        key: str,
        value: bytes,
        *,
        committed_at_unix_ms: int | None = None,
        transaction_sequence: int,
        guard_key: str | None = None,
        guard_revision: int | None = None,
        guard_value_digest: str | None = None,
        guard_mutation_sequence: int | None = None,
        guard_value_sequence: int | None = None,
        guard_lifetime_sequence: int | None = None,
        guard_committed_at_unix_ms: int | None = None,
    ) -> ControlStoreEntry:
        current_entry = self._entries.get(key)
        if current_entry is None:
            self._lifetime_sequences[key] = self._lifetime_sequences.get(key, 0) + 1
            self._value_sequences[key] = self._value_sequences.get(key, 0) + 1
        elif current_entry.value != value:
            self._value_sequences[key] += 1
        revision = self._next_revision(key)
        entry = ControlStoreEntry(
            value=value,
            revision=revision,
            committed_at_unix_ms=committed_at_unix_ms,
            transaction_sequence=transaction_sequence,
            mutation_sequence=self._mutation_sequences[key],
            value_sequence=self._value_sequences[key],
            lifetime_sequence=self._lifetime_sequences[key],
            guard_key=guard_key,
            guard_revision=guard_revision,
            guard_value_digest=guard_value_digest,
            guard_mutation_sequence=guard_mutation_sequence,
            guard_value_sequence=guard_value_sequence,
            guard_lifetime_sequence=guard_lifetime_sequence,
            guard_committed_at_unix_ms=guard_committed_at_unix_ms,
        )
        self._entries[key] = entry
        return entry

    def _next_revision(self, key: str) -> int:
        revision = self._last_revisions.get(key, 0) + 1
        self._last_revisions[key] = revision
        self._mutation_sequences[key] = self._mutation_sequences.get(key, 0) + 1
        return revision

    def _next_transaction_sequence(self) -> int:
        self._transaction_sequence += 1
        return self._transaction_sequence

    def _store_now_unix_ms(self) -> int:
        now_unix_ms = _positive_integer(self._clock(), "control-store clock")
        if now_unix_ms < self._last_now_unix_ms:
            raise ControlStoreClockError("authoritative control-store clock moved backward")
        self._last_now_unix_ms = now_unix_ms
        return now_unix_ms


def _control_key(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError("control-store key must be a non-empty normalized string")
    return value


def _control_value(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("control-store value must be bytes")
    return bytes(value)


def _sha256_digest(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


def _control_writes(value: object) -> dict[str, ControlStoreWrite]:
    if not isinstance(value, Mapping) or not value:
        raise ValueError("writes must be a non-empty mapping")
    result: dict[str, ControlStoreWrite] = {}
    for key, write in value.items():
        normalized_key = _control_key(key)
        if not isinstance(write, ControlStoreWrite):
            raise TypeError("writes values must be ControlStoreWrite")
        result[normalized_key] = write
    return dict(sorted(result.items()))


def _control_conditions(
    value: object,
) -> dict[str, int | None]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("conditions must be a mapping")
    result: dict[str, int | None] = {}
    for key, expected_revision in value.items():
        normalized_key = _control_key(key)
        result[normalized_key] = _expected_revision(
            expected_revision,
            allow_absent=True,
        )
    if len(result) != len(value):
        raise ValueError("condition keys must be unique")
    return dict(sorted(result.items()))


def _control_key_collection(value: object, path: str) -> frozenset[str]:
    if value is None:
        return frozenset()
    if isinstance(value, (str, bytes)) or not isinstance(value, Collection):
        raise TypeError(f"{path} must be a collection of control-store keys")
    normalized_keys = tuple(_control_key(key) for key in value)
    if len(set(normalized_keys)) != len(normalized_keys):
        raise ValueError(f"{path} keys must be unique")
    return frozenset(normalized_keys)


def _expected_revision(value: object, *, allow_absent: bool) -> int | None:
    if allow_absent and value is None:
        return None
    return _required_revision(value)


def _required_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("expected_revision must be a positive integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _validate_sequence_lineage(
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    *,
    path: str,
) -> None:
    minimum_mutation_sequence = 2 * lifetime_sequence - 1
    maximum_value_sequence = mutation_sequence - lifetime_sequence + 1
    if mutation_sequence < minimum_mutation_sequence:
        raise ValueError(f"{path} mutation_sequence is too small for lifetime_sequence")
    if value_sequence < lifetime_sequence:
        raise ValueError(f"{path} value_sequence is too small for lifetime_sequence")
    if value_sequence > maximum_value_sequence:
        raise ValueError(f"{path} value_sequence is too large for mutation_sequence")


def _system_unix_ms() -> int:
    return time.time_ns() // 1_000_000


__all__ = [
    "ControlStore",
    "ControlStoreClockError",
    "ControlStoreConflict",
    "ControlStoreDeadlineExceeded",
    "ControlStoreEntry",
    "ControlStoreError",
    "ControlStoreHistoryConflict",
    "ControlStoreTooEarly",
    "ControlStoreWrite",
    "InMemoryControlStore",
]
