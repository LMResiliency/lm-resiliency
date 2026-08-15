"""Strict persisted restart-intent records for torchrun resiliency."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    ProtocolValidationError,
    RestartIntent,
)


@dataclass(frozen=True, slots=True)
class RestartIntentRecord:
    """One restart intent and the generation and lease that authorized it."""

    SCHEMA_VERSION: ClassVar[int] = 1

    intent: RestartIntent
    generation_snapshot_digest: str
    coordinator_id: str
    lease_id: str
    coordinator_lease_duration_ms: int
    coordinator_fencing_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.intent, RestartIntent):
            raise TypeError("RestartIntentRecord.intent must be RestartIntent")
        _digest(
            self.generation_snapshot_digest,
            "RestartIntentRecord.generation_snapshot_digest",
        )
        _nonempty_string(
            self.coordinator_id,
            "RestartIntentRecord.coordinator_id",
        )
        _nonempty_string(self.lease_id, "RestartIntentRecord.lease_id")
        _positive_integer(
            self.coordinator_lease_duration_ms,
            "RestartIntentRecord.coordinator_lease_duration_ms",
        )
        _positive_integer(
            self.coordinator_fencing_token,
            "RestartIntentRecord.coordinator_fencing_token",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    @property
    def coordinator_lease_digest(self) -> str:
        record = CoordinatorLeaseRecord(
            run_id=self.intent.run_id,
            coordinator_id=self.coordinator_id,
            lease_id=self.lease_id,
            lease_duration_ms=self.coordinator_lease_duration_ms,
        )
        return hashlib.sha256(record.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "intent": self.intent.to_dict(),
            "generation_snapshot_digest": self.generation_snapshot_digest,
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "coordinator_lease_duration_ms": self.coordinator_lease_duration_ms,
            "coordinator_fencing_token": self.coordinator_fencing_token,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartIntentRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "intent",
                "generation_snapshot_digest",
                "coordinator_id",
                "lease_id",
                "coordinator_lease_duration_ms",
                "coordinator_fencing_token",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        intent_value = value["intent"]
        if not isinstance(intent_value, Mapping):
            raise ValueError("RestartIntentRecord.intent must be an object")
        nested_schema_version = intent_value.get("schema_version")
        if (
            isinstance(nested_schema_version, bool)
            or not isinstance(nested_schema_version, int)
            or nested_schema_version != RestartIntent.SCHEMA_VERSION
        ):
            raise ValueError("RestartIntentRecord.intent is invalid")
        try:
            intent = RestartIntent.from_dict(intent_value)
        except ProtocolValidationError as error:
            raise ValueError("RestartIntentRecord.intent is invalid") from error
        return cls(
            intent=intent,
            generation_snapshot_digest=_digest(
                value["generation_snapshot_digest"],
                "RestartIntentRecord.generation_snapshot_digest",
            ),
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "RestartIntentRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "RestartIntentRecord.lease_id",
            ),
            coordinator_lease_duration_ms=_positive_integer(
                value["coordinator_lease_duration_ms"],
                "RestartIntentRecord.coordinator_lease_duration_ms",
            ),
            coordinator_fencing_token=_positive_integer(
                value["coordinator_fencing_token"],
                "RestartIntentRecord.coordinator_fencing_token",
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartIntentHeadRecord:
    """Pointer from the current-intent head to one immutable intent record."""

    SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    generation: int
    intent_id: str
    intent_digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.run_id, "RestartIntentHeadRecord.run_id")
        _nonnegative_integer(
            self.generation,
            "RestartIntentHeadRecord.generation",
        )
        _nonempty_string(self.intent_id, "RestartIntentHeadRecord.intent_id")
        _digest(self.intent_digest, "RestartIntentHeadRecord.intent_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": self.generation,
            "intent_id": self.intent_id,
            "intent_digest": self.intent_digest,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartIntentHeadRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "run_id",
                "generation",
                "intent_id",
                "intent_digest",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        return cls(
            run_id=_nonempty_string(
                value["run_id"],
                "RestartIntentHeadRecord.run_id",
            ),
            generation=_nonnegative_integer(
                value["generation"],
                "RestartIntentHeadRecord.generation",
            ),
            intent_id=_nonempty_string(
                value["intent_id"],
                "RestartIntentHeadRecord.intent_id",
            ),
            intent_digest=_digest(
                value["intent_digest"],
                "RestartIntentHeadRecord.intent_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartIntentLifecycleRecord:
    """Last closed intent and the coordinator lease that closed it."""

    SCHEMA_VERSION: ClassVar[int] = 1

    closed_intent: RestartIntentHeadRecord
    coordinator_id: str
    lease_id: str
    coordinator_lease_duration_ms: int
    coordinator_fencing_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.closed_intent, RestartIntentHeadRecord):
            raise TypeError(
                "RestartIntentLifecycleRecord.closed_intent must be RestartIntentHeadRecord"
            )
        _nonempty_string(
            self.coordinator_id,
            "RestartIntentLifecycleRecord.coordinator_id",
        )
        _nonempty_string(
            self.lease_id,
            "RestartIntentLifecycleRecord.lease_id",
        )
        _positive_integer(
            self.coordinator_lease_duration_ms,
            "RestartIntentLifecycleRecord.coordinator_lease_duration_ms",
        )
        _positive_integer(
            self.coordinator_fencing_token,
            "RestartIntentLifecycleRecord.coordinator_fencing_token",
        )

    @property
    def coordinator_lease_digest(self) -> str:
        record = CoordinatorLeaseRecord(
            run_id=self.closed_intent.run_id,
            coordinator_id=self.coordinator_id,
            lease_id=self.lease_id,
            lease_duration_ms=self.coordinator_lease_duration_ms,
        )
        return hashlib.sha256(record.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "closed_intent": self.closed_intent.to_dict(),
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "coordinator_lease_duration_ms": self.coordinator_lease_duration_ms,
            "coordinator_fencing_token": self.coordinator_fencing_token,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartIntentLifecycleRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "closed_intent",
                "coordinator_id",
                "lease_id",
                "coordinator_lease_duration_ms",
                "coordinator_fencing_token",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        closed_intent_value = value["closed_intent"]
        if not isinstance(closed_intent_value, Mapping):
            raise ValueError("RestartIntentLifecycleRecord.closed_intent must be an object")
        nested_schema_version = closed_intent_value.get("schema_version")
        if (
            isinstance(nested_schema_version, bool)
            or not isinstance(nested_schema_version, int)
            or nested_schema_version != RestartIntentHeadRecord.SCHEMA_VERSION
        ):
            raise ValueError("RestartIntentLifecycleRecord.closed_intent is invalid")
        try:
            closed_intent = RestartIntentHeadRecord.from_json(_canonical_json(closed_intent_value))
        except ValueError as error:
            raise ValueError("RestartIntentLifecycleRecord.closed_intent is invalid") from error
        return cls(
            closed_intent=closed_intent,
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "RestartIntentLifecycleRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "RestartIntentLifecycleRecord.lease_id",
            ),
            coordinator_lease_duration_ms=_positive_integer(
                value["coordinator_lease_duration_ms"],
                "RestartIntentLifecycleRecord.coordinator_lease_duration_ms",
            ),
            coordinator_fencing_token=_positive_integer(
                value["coordinator_fencing_token"],
                "RestartIntentLifecycleRecord.coordinator_fencing_token",
            ),
        )


def _canonical_json(value: Mapping[str, object]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _json_object(encoded: bytes, path: str) -> Mapping[str, object]:
    if not isinstance(encoded, bytes):
        raise ValueError(f"{path}: expected encoded bytes")
    try:
        value = json.loads(
            encoded,
            object_pairs_hook=_reject_duplicate_object_fields,
        )
    except _DuplicateRestartIntentField as error:
        raise ValueError(f"{path}: {error}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def _fields(
    value: Mapping[str, object],
    *,
    path: str,
    required: set[str],
) -> None:
    missing = required - set(value)
    unknown = set(value) - required
    if missing:
        raise ValueError(f"{path}: missing fields {sorted(missing)!r}")
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown)!r}")


def _schema_version(value: object, path: str, *, expected: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value != expected:
        raise ValueError(f"{path}.schema_version: unsupported value {value!r}")
    return value


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _digest(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


class _DuplicateRestartIntentField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateRestartIntentField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = [
    "RestartIntentHeadRecord",
    "RestartIntentLifecycleRecord",
    "RestartIntentRecord",
]
