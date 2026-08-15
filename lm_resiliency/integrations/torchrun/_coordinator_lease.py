"""Coordinator lease and fencing records for torchrun resiliency."""

from __future__ import annotations

import hashlib
import json
import threading
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import ClassVar

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreEntry,
    ControlStoreTooEarly,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_CAS_ATTEMPTS = 16


class CoordinatorLeaseError(RuntimeError):
    """Base error for coordinator lease operations."""


class CoordinatorLeaseUnavailable(CoordinatorLeaseError):
    """Raised when another coordinator owns the active lease."""


class CoordinatorLeaseLost(CoordinatorLeaseError):
    """Raised when a held lease expired or its fencing token became stale."""


class CoordinatorLeaseCorrupt(CoordinatorLeaseError):
    """Raised when the persisted lease record is malformed or contradictory."""


class CoordinatorLeaseClockError(CoordinatorLeaseError):
    """Raised when the authoritative lease clock moves backward."""


@dataclass(frozen=True, slots=True)
class CoordinatorLeaseRecord:
    """Persisted ownership record; the store revision is its fencing token."""

    SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    coordinator_id: str
    lease_id: str
    lease_duration_ms: int

    def __post_init__(self) -> None:
        _nonempty_string(self.run_id, "CoordinatorLeaseRecord.run_id")
        _nonempty_string(
            self.coordinator_id,
            "CoordinatorLeaseRecord.coordinator_id",
        )
        _nonempty_string(self.lease_id, "CoordinatorLeaseRecord.lease_id")
        _positive_integer(
            self.lease_duration_ms,
            "CoordinatorLeaseRecord.lease_duration_ms",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "lease_duration_ms": self.lease_duration_ms,
        }

    def to_json(self) -> bytes:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    @classmethod
    def from_json(cls, encoded: bytes) -> CoordinatorLeaseRecord:
        if not isinstance(encoded, bytes):
            raise ValueError("CoordinatorLeaseRecord: expected encoded bytes")
        try:
            value = json.loads(
                encoded,
                object_pairs_hook=_reject_duplicate_object_fields,
            )
        except _DuplicateLeaseField as error:
            raise ValueError(f"CoordinatorLeaseRecord: {error}") from error
        except (TypeError, ValueError) as error:
            raise ValueError("CoordinatorLeaseRecord: invalid JSON") from error
        if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
            raise ValueError("CoordinatorLeaseRecord: expected a JSON object")
        expected = {
            "schema_version",
            "run_id",
            "coordinator_id",
            "lease_id",
            "lease_duration_ms",
        }
        missing = expected - set(value)
        unknown = set(value) - expected
        if missing:
            raise ValueError(f"CoordinatorLeaseRecord: missing fields {sorted(missing)!r}")
        if unknown:
            raise ValueError(f"CoordinatorLeaseRecord: unknown fields {sorted(unknown)!r}")
        if (
            isinstance(value["schema_version"], bool)
            or not isinstance(value["schema_version"], int)
            or value["schema_version"] != cls.SCHEMA_VERSION
        ):
            raise ValueError(
                "CoordinatorLeaseRecord.schema_version: unsupported value "
                f"{value['schema_version']!r}"
            )
        return cls(
            run_id=_nonempty_string(
                value["run_id"],
                "CoordinatorLeaseRecord.run_id",
            ),
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "CoordinatorLeaseRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "CoordinatorLeaseRecord.lease_id",
            ),
            lease_duration_ms=_positive_integer(
                value["lease_duration_ms"],
                "CoordinatorLeaseRecord.lease_duration_ms",
            ),
        )


@dataclass(frozen=True, slots=True)
class HeldCoordinatorLease:
    """One observed lease and the revision that fences its mutations."""

    record: CoordinatorLeaseRecord
    fencing_token: int
    granted_at_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, CoordinatorLeaseRecord):
            raise TypeError("HeldCoordinatorLease.record must be CoordinatorLeaseRecord")
        _positive_integer(
            self.fencing_token,
            "HeldCoordinatorLease.fencing_token",
        )
        _positive_integer(
            self.granted_at_unix_ms,
            "HeldCoordinatorLease.granted_at_unix_ms",
        )

    @property
    def expires_at_unix_ms(self) -> int:
        return self.granted_at_unix_ms + self.record.lease_duration_ms


class CoordinatorLeaseManager:
    """Acquire and maintain one run-scoped coordinator lease."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        coordinator_id: str,
        lease_duration_ms: int,
        clock: Callable[[], int],
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._coordinator_id = _nonempty_string(
            coordinator_id,
            "coordinator_id",
        )
        self._lease_duration_ms = _positive_integer(
            lease_duration_ms,
            "lease_duration_ms",
        )
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._clock_lock = threading.Lock()
        self._last_now_unix_ms = 0
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._lease_key = f"{_CONTROL_PREFIX}/runs/{run_digest}/coordinator-lease"

    @property
    def lease_key(self) -> str:
        return self._lease_key

    def current(self) -> HeldCoordinatorLease | None:
        entry = self._store.get(self._lease_key)
        if entry is None:
            return None
        return self._decode_entry(entry)

    def acquire(self) -> HeldCoordinatorLease:
        now_unix_ms = self._now_unix_ms()
        for _ in range(_MAX_CAS_ATTEMPTS):
            current = self.current()
            if current is not None and current.expires_at_unix_ms > now_unix_ms:
                if current.record.coordinator_id == self._coordinator_id:
                    try:
                        entry = self._store.compare_set_in_window(
                            self._lease_key,
                            expected_revision=current.fencing_token,
                            not_before_unix_ms=now_unix_ms,
                            deadline_unix_ms=current.expires_at_unix_ms,
                            value=current.record.to_json(),
                        )
                    except ControlStoreDeadlineExceeded as error:
                        raise CoordinatorLeaseUnavailable(
                            "existing coordinator lease expired at the control store "
                            "during acquisition"
                        ) from error
                    except ControlStoreTooEarly as error:
                        raise CoordinatorLeaseClockError(
                            "control-store time precedes the coordinator acquisition observation"
                        ) from error
                    except ControlStoreClockError as error:
                        raise CoordinatorLeaseClockError(
                            "authoritative control-store clock moved backward"
                        ) from error
                    except ControlStoreConflict:
                        continue
                    return self._require_live_response(self._decode_entry(entry))
                raise CoordinatorLeaseUnavailable(
                    "coordinator lease is held by "
                    f"{current.record.coordinator_id!r} until "
                    f"{current.expires_at_unix_ms}"
                )
            record = CoordinatorLeaseRecord(
                run_id=self._run_id,
                coordinator_id=self._coordinator_id,
                lease_id=uuid.uuid4().hex,
                lease_duration_ms=self._lease_duration_ms,
            )
            try:
                entry = self._store.compare_set_in_window(
                    self._lease_key,
                    expected_revision=None if current is None else current.fencing_token,
                    not_before_unix_ms=max(
                        now_unix_ms,
                        0 if current is None else current.expires_at_unix_ms,
                    ),
                    deadline_unix_ms=None,
                    value=record.to_json(),
                )
            except ControlStoreTooEarly as error:
                raise CoordinatorLeaseClockError(
                    "control-store time precedes the coordinator acquisition observation "
                    "or existing lease expiry"
                ) from error
            except ControlStoreClockError as error:
                raise CoordinatorLeaseClockError(
                    "authoritative control-store clock moved backward"
                ) from error
            except ControlStoreConflict:
                continue
            return self._require_live_response(self._decode_entry(entry))
        raise CoordinatorLeaseUnavailable("coordinator lease changed repeatedly during acquisition")

    def renew(self, lease: HeldCoordinatorLease) -> HeldCoordinatorLease:
        self._validate_owned_handle(lease)
        now_unix_ms = self._now_unix_ms()
        if lease.expires_at_unix_ms <= now_unix_ms:
            raise CoordinatorLeaseLost("coordinator lease expired before renewal")
        try:
            entry = self._store.compare_set_in_window(
                self._lease_key,
                expected_revision=lease.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=lease.expires_at_unix_ms,
                value=lease.record.to_json(),
            )
        except ControlStoreDeadlineExceeded as error:
            raise CoordinatorLeaseLost(
                "coordinator lease expired at the control store before renewal"
            ) from error
        except ControlStoreTooEarly as error:
            raise CoordinatorLeaseClockError(
                "control-store time precedes the coordinator renewal observation"
            ) from error
        except ControlStoreClockError as error:
            raise CoordinatorLeaseClockError(
                "authoritative control-store clock moved backward"
            ) from error
        except ControlStoreConflict as error:
            raise CoordinatorLeaseLost("coordinator lease changed before renewal") from error
        try:
            return self._require_live_response(self._decode_entry(entry))
        except CoordinatorLeaseUnavailable as error:
            raise CoordinatorLeaseLost(
                "renewed coordinator lease expired before the response arrived"
            ) from error

    def release(self, lease: HeldCoordinatorLease) -> int:
        self._validate_owned_handle(lease)
        now_unix_ms = self._now_unix_ms()
        if lease.expires_at_unix_ms <= now_unix_ms:
            raise CoordinatorLeaseLost("coordinator lease expired before release")
        try:
            return self._store.compare_delete_in_window(
                self._lease_key,
                expected_revision=lease.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=lease.expires_at_unix_ms,
            )
        except ControlStoreDeadlineExceeded as error:
            raise CoordinatorLeaseLost(
                "coordinator lease expired at the control store before release"
            ) from error
        except ControlStoreTooEarly as error:
            raise CoordinatorLeaseClockError(
                "control-store time precedes the coordinator release observation"
            ) from error
        except ControlStoreClockError as error:
            raise CoordinatorLeaseClockError(
                "authoritative control-store clock moved backward"
            ) from error
        except ControlStoreConflict as error:
            raise CoordinatorLeaseLost("coordinator lease changed before release") from error

    def _decode_entry(self, entry: ControlStoreEntry) -> HeldCoordinatorLease:
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise CoordinatorLeaseCorrupt("persisted coordinator lease is malformed") from error
        if record.run_id != self._run_id:
            raise CoordinatorLeaseCorrupt("persisted coordinator lease belongs to another run")
        if entry.committed_at_unix_ms is None:
            raise CoordinatorLeaseCorrupt(
                "persisted coordinator lease has no authoritative commit time"
            )
        return HeldCoordinatorLease(
            record=record,
            fencing_token=entry.revision,
            granted_at_unix_ms=entry.committed_at_unix_ms,
        )

    def _require_live_response(self, lease: HeldCoordinatorLease) -> HeldCoordinatorLease:
        now_unix_ms = self._now_unix_ms()
        if now_unix_ms < lease.granted_at_unix_ms:
            raise CoordinatorLeaseClockError(
                "coordinator clock precedes the authoritative lease commit time"
            )
        if lease.expires_at_unix_ms <= now_unix_ms:
            raise CoordinatorLeaseUnavailable(
                "coordinator lease expired before the store response arrived"
            )
        return lease

    def _validate_owned_handle(self, lease: HeldCoordinatorLease) -> None:
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if (
            lease.record.run_id != self._run_id
            or lease.record.coordinator_id != self._coordinator_id
        ):
            raise CoordinatorLeaseLost("coordinator lease handle belongs to another manager")

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            now_unix_ms = _positive_integer(self._clock(), "clock")
            if now_unix_ms < self._last_now_unix_ms:
                raise CoordinatorLeaseClockError("coordinator lease clock moved backward")
            self._last_now_unix_ms = now_unix_ms
        return now_unix_ms


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


class _DuplicateLeaseField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateLeaseField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = [
    "CoordinatorLeaseClockError",
    "CoordinatorLeaseCorrupt",
    "CoordinatorLeaseError",
    "CoordinatorLeaseLost",
    "CoordinatorLeaseManager",
    "CoordinatorLeaseRecord",
    "CoordinatorLeaseUnavailable",
    "HeldCoordinatorLease",
]
