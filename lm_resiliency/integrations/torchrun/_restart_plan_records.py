"""Strict persisted records for torchrun restart plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar

from lm_resiliency.integrations.torchrun._protocol import (
    ProtocolValidationError,
    RecoveryManifest,
)


@dataclass(frozen=True, slots=True)
class RecoveryManifestRecord:
    """One immutable recovery manifest bound to its source generation."""

    SCHEMA_VERSION: ClassVar[int] = 1

    manifest: RecoveryManifest
    source_generation_snapshot_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.manifest, RecoveryManifest):
            raise TypeError("RecoveryManifestRecord.manifest must be RecoveryManifest")
        _digest(
            self.source_generation_snapshot_digest,
            "RecoveryManifestRecord.source_generation_snapshot_digest",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest": self.manifest.to_dict(),
            "source_generation_snapshot_digest": self.source_generation_snapshot_digest,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RecoveryManifestRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "manifest",
                "source_generation_snapshot_digest",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        manifest_value = _mapping(
            value["manifest"],
            "RecoveryManifestRecord.manifest",
        )
        _schema_version(
            manifest_value.get("schema_version"),
            "RecoveryManifestRecord.manifest",
            expected=RecoveryManifest.SCHEMA_VERSION,
        )
        try:
            manifest = RecoveryManifest.from_dict(manifest_value)
        except ProtocolValidationError as error:
            raise ValueError("RecoveryManifestRecord.manifest is invalid") from error
        return cls(
            manifest=manifest,
            source_generation_snapshot_digest=_digest(
                value["source_generation_snapshot_digest"],
                "RecoveryManifestRecord.source_generation_snapshot_digest",
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
    except _DuplicateRestartPlanField as error:
        raise ValueError(f"{path}: {error}") from error
    except (TypeError, ValueError) as error:
        raise ValueError(f"{path}: invalid JSON") from error
    return _mapping(value, path)


def _mapping(value: object, path: str) -> Mapping[str, object]:
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


def _digest(value: object, path: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{path} must be a lowercase SHA-256 digest")
    return value


class _DuplicateRestartPlanField(ValueError):
    pass


def _reject_duplicate_object_fields(
    pairs: list[tuple[str, object]],
) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateRestartPlanField(f"duplicate field {key!r}")
        value[key] = item
    return value


__all__ = ["RecoveryManifestRecord"]
