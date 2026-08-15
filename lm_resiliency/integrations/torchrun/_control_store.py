"""Atomic control-store primitives for the internal torchrun integration."""

from __future__ import annotations

import threading
from dataclasses import dataclass
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


@dataclass(frozen=True, slots=True)
class ControlStoreEntry:
    """One immutable control-store value and its opaque per-key revision."""

    value: bytes
    revision: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _control_value(self.value))
        object.__setattr__(self, "revision", _required_revision(self.revision))


class ControlStore(Protocol):
    """Strongly consistent per-key compare-and-set storage."""

    def get(self, key: str) -> ControlStoreEntry | None:
        """Return the current value, or ``None`` when the key is absent."""
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


class InMemoryControlStore:
    """Thread-safe in-memory implementation used by coordinator contract tests."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._entries: dict[str, ControlStoreEntry] = {}
        self._last_revisions: dict[str, int] = {}

    def get(self, key: str) -> ControlStoreEntry | None:
        normalized_key = _control_key(key)
        with self._lock:
            return self._entries.get(normalized_key)

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
            revision = self._next_revision(normalized_key)
            entry = ControlStoreEntry(value=normalized_value, revision=revision)
            self._entries[normalized_key] = entry
            return entry

    def compare_delete(self, key: str, *, expected_revision: int) -> int:
        normalized_key = _control_key(key)
        normalized_revision = _required_revision(expected_revision)
        with self._lock:
            self._require_revision(normalized_key, normalized_revision)
            del self._entries[normalized_key]
            return self._next_revision(normalized_key)

    def _require_revision(self, key: str, expected_revision: int | None) -> None:
        entry = self._entries.get(key)
        actual_revision = None if entry is None else entry.revision
        if actual_revision != expected_revision:
            raise ControlStoreConflict(key, expected_revision, actual_revision)

    def _next_revision(self, key: str) -> int:
        revision = self._last_revisions.get(key, 0) + 1
        self._last_revisions[key] = revision
        return revision


def _control_key(value: object) -> str:
    if not isinstance(value, str) or not value or value != value.strip() or "\x00" in value:
        raise ValueError("control-store key must be a non-empty normalized string")
    return value


def _control_value(value: object) -> bytes:
    if not isinstance(value, bytes):
        raise TypeError("control-store value must be bytes")
    return bytes(value)


def _expected_revision(value: object, *, allow_absent: bool) -> int | None:
    if allow_absent and value is None:
        return None
    return _required_revision(value)


def _required_revision(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("expected_revision must be a positive integer")
    return value


__all__ = [
    "ControlStore",
    "ControlStoreConflict",
    "ControlStoreEntry",
    "ControlStoreError",
    "InMemoryControlStore",
]
