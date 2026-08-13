"""Persistent campaign occurrence journal."""

from __future__ import annotations

import json
import os
import tempfile
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol


@dataclass(slots=True)
class CampaignJournal:
    """Attempt counts keyed by incident candidate iteration."""

    campaign: str
    attempts: dict[str, int] = field(default_factory=dict)

    def attempt_count(self, incident_id: str, iteration: int) -> int:
        return int(self.attempts.get(_candidate_key(incident_id, iteration), 0))

    def record_attempt(self, incident_id: str, iteration: int) -> int:
        key = _candidate_key(incident_id, iteration)
        attempt = int(self.attempts.get(key, 0)) + 1
        self.attempts[key] = attempt
        return attempt

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "attempts": dict(sorted(self.attempts.items())),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "CampaignJournal":
        return cls(
            campaign=str(value["campaign"]),
            attempts={
                str(key): int(count) for key, count in dict(value.get("attempts", {})).items()
            },
        )


class CampaignStateStore(Protocol):
    """Storage contract for restart-stable campaign progress."""

    def load(self, campaign: str) -> CampaignJournal: ...

    def save(self, journal: CampaignJournal) -> None: ...


class MemoryCampaignStateStore:
    """Process-local state store for campaigns that do not cross restart."""

    def __init__(self) -> None:
        self._journals: dict[str, CampaignJournal] = {}
        self._lock = threading.RLock()

    def load(self, campaign: str) -> CampaignJournal:
        with self._lock:
            journal = self._journals.get(campaign)
            if journal is None:
                journal = CampaignJournal(campaign)
                self._journals[campaign] = journal
            return CampaignJournal.from_dict(journal.to_dict())

    def save(self, journal: CampaignJournal) -> None:
        with self._lock:
            self._journals[journal.campaign] = CampaignJournal.from_dict(journal.to_dict())


class JsonCampaignStateStore:
    """Atomic JSON state store suitable for manager-preserved local storage."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()

    def load(self, campaign: str) -> CampaignJournal:
        with self._lock:
            if not self.path.exists():
                return CampaignJournal(campaign)
            with self.path.open(encoding="utf-8") as stream:
                value = json.load(stream)
            if not isinstance(value, dict):
                raise ValueError("campaign state JSON must contain an object")
            journal = CampaignJournal.from_dict(value)
            if journal.campaign != campaign:
                raise ValueError(
                    f"campaign state belongs to {journal.campaign!r}, not {campaign!r}"
                )
            return journal

    def save(self, journal: CampaignJournal) -> None:
        with self._lock:
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


def _candidate_key(incident_id: str, iteration: int) -> str:
    return f"{incident_id}@{iteration}"


__all__ = [
    "CampaignJournal",
    "CampaignStateStore",
    "JsonCampaignStateStore",
    "MemoryCampaignStateStore",
]
