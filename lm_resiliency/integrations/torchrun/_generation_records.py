"""Strict persisted records for torchrun resiliency generations."""

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
    RankAssignment,
)


@dataclass(frozen=True, slots=True)
class GenerationSnapshotRecord:
    """One immutable rank assignment and its coordinator fencing provenance."""

    SCHEMA_VERSION: ClassVar[int] = 2

    assignment: RankAssignment
    previous_snapshot_digest: str | None
    coordinator_id: str
    lease_id: str
    coordinator_lease_duration_ms: int
    coordinator_fencing_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.assignment, RankAssignment):
            raise TypeError("GenerationSnapshotRecord.assignment must be RankAssignment")
        if self.assignment.generation == 0:
            if self.previous_snapshot_digest is not None:
                raise ValueError("generation zero must not name a previous snapshot")
        else:
            _digest(
                self.previous_snapshot_digest,
                "GenerationSnapshotRecord.previous_snapshot_digest",
            )
        _nonempty_string(
            self.coordinator_id,
            "GenerationSnapshotRecord.coordinator_id",
        )
        _nonempty_string(self.lease_id, "GenerationSnapshotRecord.lease_id")
        _positive_integer(
            self.coordinator_lease_duration_ms,
            "GenerationSnapshotRecord.coordinator_lease_duration_ms",
        )
        _positive_integer(
            self.coordinator_fencing_token,
            "GenerationSnapshotRecord.coordinator_fencing_token",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    @property
    def coordinator_lease_digest(self) -> str:
        record = CoordinatorLeaseRecord(
            run_id=self.assignment.run_id,
            coordinator_id=self.coordinator_id,
            lease_id=self.lease_id,
            lease_duration_ms=self.coordinator_lease_duration_ms,
        )
        return hashlib.sha256(record.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "assignment": self.assignment.to_dict(),
            "previous_snapshot_digest": self.previous_snapshot_digest,
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "coordinator_lease_duration_ms": self.coordinator_lease_duration_ms,
            "coordinator_fencing_token": self.coordinator_fencing_token,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> GenerationSnapshotRecord:
        value = _json_object(encoded, cls.__name__)
        if "schema_version" not in value:
            raise ValueError(f"{cls.__name__}: missing fields ['schema_version']")
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "assignment",
                "previous_snapshot_digest",
                "coordinator_id",
                "lease_id",
                "coordinator_lease_duration_ms",
                "coordinator_fencing_token",
            },
        )
        assignment_value = value["assignment"]
        if not isinstance(assignment_value, Mapping):
            raise ValueError("GenerationSnapshotRecord.assignment must be an object")
        try:
            assignment = RankAssignment.from_dict(assignment_value)
        except ProtocolValidationError as error:
            raise ValueError("GenerationSnapshotRecord.assignment is invalid") from error
        previous_digest = value["previous_snapshot_digest"]
        if previous_digest is not None:
            previous_digest = _digest(
                previous_digest,
                "GenerationSnapshotRecord.previous_snapshot_digest",
            )
        return cls(
            assignment=assignment,
            previous_snapshot_digest=previous_digest,
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "GenerationSnapshotRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "GenerationSnapshotRecord.lease_id",
            ),
            coordinator_lease_duration_ms=_positive_integer(
                value["coordinator_lease_duration_ms"],
                "GenerationSnapshotRecord.coordinator_lease_duration_ms",
            ),
            coordinator_fencing_token=_positive_integer(
                value["coordinator_fencing_token"],
                "GenerationSnapshotRecord.coordinator_fencing_token",
            ),
        )


@dataclass(frozen=True, slots=True)
class GenerationHeadRecord:
    """Pointer from the mutable generation head to an immutable snapshot."""

    SCHEMA_VERSION: ClassVar[int] = 1

    run_id: str
    generation: int
    snapshot_digest: str

    def __post_init__(self) -> None:
        _nonempty_string(self.run_id, "GenerationHeadRecord.run_id")
        _nonnegative_integer(self.generation, "GenerationHeadRecord.generation")
        _digest(self.snapshot_digest, "GenerationHeadRecord.snapshot_digest")

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": self.generation,
            "snapshot_digest": self.snapshot_digest,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> GenerationHeadRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "run_id",
                "generation",
                "snapshot_digest",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        return cls(
            run_id=_nonempty_string(value["run_id"], "GenerationHeadRecord.run_id"),
            generation=_nonnegative_integer(
                value["generation"],
                "GenerationHeadRecord.generation",
            ),
            snapshot_digest=_digest(
                value["snapshot_digest"],
                "GenerationHeadRecord.snapshot_digest",
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
    except _DuplicateGenerationField as error:
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


def _nonnegative_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


def _digest(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


class _DuplicateGenerationField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateGenerationField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = [
    "GenerationHeadRecord",
    "GenerationSnapshotRecord",
]
