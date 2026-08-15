"""Strict persisted quarantine records for torchrun replacement."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, slots=True)
class NodeQuarantineRecord:
    """One permanent node exclusion authorized by a committed restart plan."""

    SCHEMA_VERSION: ClassVar[int] = 2

    run_id: str
    node_id: str
    plan_id: str
    intent_id: str
    from_generation: int
    effective_generation: int
    incident_ids: tuple[str, ...]
    reason_code: str
    resource_ids: tuple[str, ...]
    coordinator_id: str
    lease_id: str
    coordinator_lease_duration_ms: int
    coordinator_fencing_token: int

    def __post_init__(self) -> None:
        _nonempty_string(self.run_id, "NodeQuarantineRecord.run_id")
        _nonempty_string(self.node_id, "NodeQuarantineRecord.node_id")
        _nonempty_string(self.plan_id, "NodeQuarantineRecord.plan_id")
        _nonempty_string(self.intent_id, "NodeQuarantineRecord.intent_id")
        _nonnegative_integer(
            self.from_generation,
            "NodeQuarantineRecord.from_generation",
        )
        _positive_integer(
            self.effective_generation,
            "NodeQuarantineRecord.effective_generation",
        )
        if self.effective_generation != self.from_generation + 1:
            raise ValueError(
                "NodeQuarantineRecord.effective_generation must be the successor generation"
            )
        incident_ids = _strings(
            self.incident_ids,
            "NodeQuarantineRecord.incident_ids",
            require_nonempty=True,
        )
        resource_ids = _strings(
            self.resource_ids,
            "NodeQuarantineRecord.resource_ids",
            require_nonempty=False,
        )
        object.__setattr__(self, "incident_ids", incident_ids)
        object.__setattr__(self, "resource_ids", resource_ids)
        _nonempty_string(
            self.reason_code,
            "NodeQuarantineRecord.reason_code",
        )
        _nonempty_string(
            self.coordinator_id,
            "NodeQuarantineRecord.coordinator_id",
        )
        _nonempty_string(
            self.lease_id,
            "NodeQuarantineRecord.lease_id",
        )
        _positive_integer(
            self.coordinator_lease_duration_ms,
            "NodeQuarantineRecord.coordinator_lease_duration_ms",
        )
        _positive_integer(
            self.coordinator_fencing_token,
            "NodeQuarantineRecord.coordinator_fencing_token",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "plan_id": self.plan_id,
            "intent_id": self.intent_id,
            "from_generation": self.from_generation,
            "effective_generation": self.effective_generation,
            "incident_ids": list(self.incident_ids),
            "reason_code": self.reason_code,
            "resource_ids": list(self.resource_ids),
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "coordinator_lease_duration_ms": self.coordinator_lease_duration_ms,
            "coordinator_fencing_token": self.coordinator_fencing_token,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> NodeQuarantineRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "run_id",
                "node_id",
                "plan_id",
                "intent_id",
                "from_generation",
                "effective_generation",
                "incident_ids",
                "reason_code",
                "resource_ids",
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
        return cls(
            run_id=_nonempty_string(
                value["run_id"],
                "NodeQuarantineRecord.run_id",
            ),
            node_id=_nonempty_string(
                value["node_id"],
                "NodeQuarantineRecord.node_id",
            ),
            plan_id=_nonempty_string(
                value["plan_id"],
                "NodeQuarantineRecord.plan_id",
            ),
            intent_id=_nonempty_string(
                value["intent_id"],
                "NodeQuarantineRecord.intent_id",
            ),
            from_generation=_nonnegative_integer(
                value["from_generation"],
                "NodeQuarantineRecord.from_generation",
            ),
            effective_generation=_positive_integer(
                value["effective_generation"],
                "NodeQuarantineRecord.effective_generation",
            ),
            incident_ids=_strings(
                value["incident_ids"],
                "NodeQuarantineRecord.incident_ids",
                require_nonempty=True,
            ),
            reason_code=_nonempty_string(
                value["reason_code"],
                "NodeQuarantineRecord.reason_code",
            ),
            resource_ids=_strings(
                value["resource_ids"],
                "NodeQuarantineRecord.resource_ids",
                require_nonempty=False,
            ),
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "NodeQuarantineRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "NodeQuarantineRecord.lease_id",
            ),
            coordinator_lease_duration_ms=_positive_integer(
                value["coordinator_lease_duration_ms"],
                "NodeQuarantineRecord.coordinator_lease_duration_ms",
            ),
            coordinator_fencing_token=_positive_integer(
                value["coordinator_fencing_token"],
                "NodeQuarantineRecord.coordinator_fencing_token",
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
    except _DuplicateQuarantineField as error:
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


def _strings(
    value: object,
    path: str,
    *,
    require_nonempty: bool,
) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be an array")
    result = tuple(_nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if require_nonempty and not result:
        raise ValueError(f"{path} must contain at least one value")
    if len(result) != len(set(result)):
        raise ValueError(f"{path} values must be unique")
    return result


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


class _DuplicateQuarantineField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateQuarantineField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = ["NodeQuarantineRecord"]
