"""Persistent campaign occurrence journal."""

from __future__ import annotations

import fcntl
import json
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Protocol

_JOURNAL_KEYS = frozenset({"campaign", "manifest_identity", "attempts"})


@dataclass(slots=True)
class CampaignJournal:
    """Attempt counts keyed by incident candidate iteration."""

    campaign: str
    manifest_identity: str | None = None
    attempts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.campaign, str) or not self.campaign:
            raise ValueError("campaign state campaign must be a non-empty string")
        if self.manifest_identity is not None and (
            not isinstance(self.manifest_identity, str) or not self.manifest_identity
        ):
            raise ValueError("campaign state manifest_identity must be a non-empty string")
        validated: dict[str, int] = {}
        for key, count in self.attempts.items():
            _validate_candidate_key(key)
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("campaign state attempt counts must be integers")
            if count <= 0:
                raise ValueError("campaign state attempt counts must be positive")
            validated[key] = count
        self.attempts = validated

    def bind_manifest(self, manifest_identity: str) -> None:
        """Bind persisted attempts to one exact executable campaign manifest."""
        if not manifest_identity:
            raise ValueError("campaign manifest identity must be non-empty")
        if self.manifest_identity is None:
            if self.attempts:
                raise ValueError(
                    "campaign state contains attempts without a manifest identity; "
                    "start with a new state file"
                )
            self.manifest_identity = manifest_identity
            return
        if self.manifest_identity != manifest_identity:
            raise ValueError("campaign state manifest identity does not match the current campaign")

    def attempt_count(self, incident_id: str, iteration: int) -> int:
        return self.attempts.get(_candidate_key(incident_id, iteration), 0)

    def record_attempt(self, incident_id: str, iteration: int) -> int:
        key = _candidate_key(incident_id, iteration)
        _validate_candidate_key(key)
        attempt = self.attempts.get(key, 0) + 1
        self.attempts[key] = attempt
        return attempt

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "manifest_identity": self.manifest_identity,
            "attempts": dict(sorted(self.attempts.items())),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CampaignJournal":
        if not isinstance(value, dict):
            raise TypeError("campaign state must be an object")
        unknown = sorted(set(value) - _JOURNAL_KEYS)
        if unknown:
            raise ValueError(f"campaign state contains unknown fields: {unknown}")
        missing = sorted(_JOURNAL_KEYS - set(value))
        if missing:
            raise ValueError(f"campaign state is missing required fields: {missing}")
        if not isinstance(value.get("campaign"), str):
            raise TypeError("campaign state campaign must be a string")
        manifest_identity = value.get("manifest_identity")
        if manifest_identity is not None and not isinstance(manifest_identity, str):
            raise TypeError("campaign state manifest_identity must be a string")
        attempts = value["attempts"]
        if not isinstance(attempts, dict):
            raise TypeError("campaign state attempts must be an object")
        return cls(
            campaign=value["campaign"],
            manifest_identity=manifest_identity,
            attempts=dict(attempts),
        )


class CampaignStateStore(Protocol):
    """Storage contract for restart-stable campaign progress."""

    def load(self, campaign: str) -> CampaignJournal: ...

    def save(self, journal: CampaignJournal) -> None: ...

    def compare_and_swap(
        self,
        expected: CampaignJournal,
        updated: CampaignJournal,
    ) -> bool: ...


class MemoryCampaignStateStore:
    """Process-local state store for campaigns that do not cross restart."""

    def __init__(self) -> None:
        self._journals: dict[str, CampaignJournal] = {}
        self._lock = threading.RLock()

    def load(self, campaign: str) -> CampaignJournal:
        with self._lock:
            journal = self._journals.get(campaign)
            if journal is None:
                return CampaignJournal(campaign)
            return CampaignJournal.from_dict(journal.to_dict())

    def save(self, journal: CampaignJournal) -> None:
        with self._lock:
            self._journals[journal.campaign] = CampaignJournal.from_dict(journal.to_dict())

    def compare_and_swap(
        self,
        expected: CampaignJournal,
        updated: CampaignJournal,
    ) -> bool:
        with self._lock:
            current = self._journals.get(expected.campaign)
            current_value = (
                CampaignJournal(expected.campaign).to_dict()
                if current is None
                else current.to_dict()
            )
            if current_value != expected.to_dict():
                return False
            self.save(updated)
            return True


class JsonCampaignStateStore:
    """Atomic JSON state store suitable for manager-preserved local storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_name(f".{self.path.name}.lock")
        self._lock = threading.RLock()

    def load(self, campaign: str) -> CampaignJournal:
        with self._lock:
            return self._load_unlocked(campaign)

    def save(self, journal: CampaignJournal) -> None:
        with self._lock:
            with self._file_lock():
                self._save_unlocked(journal)

    def compare_and_swap(
        self,
        expected: CampaignJournal,
        updated: CampaignJournal,
    ) -> bool:
        if expected.campaign != updated.campaign:
            raise ValueError("campaign state compare-and-swap requires one campaign")
        with self._lock:
            with self._file_lock():
                current = self._load_unlocked(expected.campaign)
                if current.to_dict() != expected.to_dict():
                    return False
                self._save_unlocked(updated)
                return True

    def _load_unlocked(self, campaign: str) -> CampaignJournal:
        if not self.path.exists():
            return CampaignJournal(campaign)
        with self.path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("campaign state JSON must contain an object")
        journal = CampaignJournal.from_dict(value)
        if journal.campaign != campaign:
            raise ValueError(f"campaign state belongs to {journal.campaign!r}, not {campaign!r}")
        return journal

    def _save_unlocked(self, journal: CampaignJournal) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            dir=self.path.parent,
            text=True,
        )
        try:
            with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
                json.dump(journal.to_dict(), stream, indent=2, sort_keys=True)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    @contextmanager
    def _file_lock(self) -> Iterator[None]:
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _candidate_key(incident_id: str, iteration: int) -> str:
    return f"{incident_id}@{iteration}"


def _validate_candidate_key(key: Any) -> None:
    if not isinstance(key, str):
        raise TypeError("campaign state attempt keys must be strings")
    incident_id, separator, iteration = key.rpartition("@")
    if (
        not separator
        or not incident_id
        or not iteration.isdigit()
        or int(iteration) <= 0
        or iteration != str(int(iteration))
    ):
        raise ValueError(
            "campaign state attempt keys must use '<incident_id>@<positive_iteration>'"
        )


__all__ = [
    "CampaignJournal",
    "CampaignStateStore",
    "JsonCampaignStateStore",
    "MemoryCampaignStateStore",
]
