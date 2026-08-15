"""Strict persisted records for torchrun agent registration."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    ProtocolValidationError,
)


@dataclass(frozen=True, slots=True)
class AgentRegistrationRecord:
    """One agent identity and its registration-lease lifetime."""

    SCHEMA_VERSION: ClassVar[int] = 1

    agent_identity: AgentIdentity
    registration_id: str
    lease_duration_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.agent_identity, AgentIdentity):
            raise TypeError("AgentRegistrationRecord.agent_identity must be AgentIdentity")
        _nonempty_string(
            self.registration_id,
            "AgentRegistrationRecord.registration_id",
        )
        _positive_integer(
            self.lease_duration_ms,
            "AgentRegistrationRecord.lease_duration_ms",
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "agent_identity": self.agent_identity.to_dict(),
            "registration_id": self.registration_id,
            "lease_duration_ms": self.lease_duration_ms,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> AgentRegistrationRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "agent_identity",
                "registration_id",
                "lease_duration_ms",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        identity_value = value["agent_identity"]
        if not isinstance(identity_value, Mapping):
            raise ValueError("AgentRegistrationRecord.agent_identity must be an object")
        try:
            identity = AgentIdentity.from_dict(identity_value)
        except ProtocolValidationError as error:
            raise ValueError("AgentRegistrationRecord.agent_identity is invalid") from error
        return cls(
            agent_identity=identity,
            registration_id=_nonempty_string(
                value["registration_id"],
                "AgentRegistrationRecord.registration_id",
            ),
            lease_duration_ms=_positive_integer(
                value["lease_duration_ms"],
                "AgentRegistrationRecord.lease_duration_ms",
            ),
        )


@dataclass(frozen=True, slots=True)
class HeldAgentRegistration:
    """One observed registration and the revision fencing its mutations."""

    record: AgentRegistrationRecord
    fencing_token: int
    granted_at_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, AgentRegistrationRecord):
            raise TypeError("HeldAgentRegistration.record must be AgentRegistrationRecord")
        _positive_integer(
            self.fencing_token,
            "HeldAgentRegistration.fencing_token",
        )
        _positive_integer(
            self.granted_at_unix_ms,
            "HeldAgentRegistration.granted_at_unix_ms",
        )

    @property
    def expires_at_unix_ms(self) -> int:
        return self.granted_at_unix_ms + self.record.lease_duration_ms


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
    except _DuplicateRegistrationField as error:
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


class _DuplicateRegistrationField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateRegistrationField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = [
    "AgentRegistrationRecord",
    "HeldAgentRegistration",
]
