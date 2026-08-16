"""Strict persisted records for torchrun restart plans."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointCertification,
    CheckpointInventoryEvent,
    ProtocolValidationError,
    RecoveryManifest,
    RestartPlan,
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


@dataclass(frozen=True, slots=True)
class RestartPlanEvidenceRecord:
    """Immutable checkpoint evidence retained with one restart plan."""

    SCHEMA_VERSION: ClassVar[int] = 1

    plan_id: str
    run_id: str
    manifest_id: str
    inventory_events: Mapping[str, CheckpointInventoryEvent]
    certifications: tuple[CheckpointCertification, ...]

    def __post_init__(self) -> None:
        _nonempty_string(self.plan_id, "RestartPlanEvidenceRecord.plan_id")
        _nonempty_string(self.run_id, "RestartPlanEvidenceRecord.run_id")
        _nonempty_string(self.manifest_id, "RestartPlanEvidenceRecord.manifest_id")
        if not isinstance(self.inventory_events, Mapping):
            raise TypeError("RestartPlanEvidenceRecord.inventory_events must be a mapping")
        events: dict[str, CheckpointInventoryEvent] = {}
        for event_id, event in self.inventory_events.items():
            normalized_event_id = _nonempty_string(
                event_id,
                "RestartPlanEvidenceRecord.inventory_events key",
            )
            if not isinstance(event, CheckpointInventoryEvent):
                raise TypeError(
                    "RestartPlanEvidenceRecord.inventory_events values "
                    "must be CheckpointInventoryEvent"
                )
            if event.event_id != normalized_event_id or event.run_id != self.run_id:
                raise ValueError(
                    "RestartPlanEvidenceRecord inventory event identity does not match its key/run"
                )
            events[normalized_event_id] = event
        if not events:
            raise ValueError("RestartPlanEvidenceRecord requires at least one inventory event")
        if not isinstance(self.certifications, tuple):
            raise TypeError("RestartPlanEvidenceRecord.certifications must be a tuple")
        certification_ids: set[str] = set()
        certifications: list[CheckpointCertification] = []
        for certification in self.certifications:
            if not isinstance(certification, CheckpointCertification):
                raise TypeError(
                    "RestartPlanEvidenceRecord.certifications values "
                    "must be CheckpointCertification"
                )
            if certification.run_id != self.run_id:
                raise ValueError("RestartPlanEvidenceRecord certification belongs to another run")
            if certification.certification_id in certification_ids:
                raise ValueError("RestartPlanEvidenceRecord certification IDs must be unique")
            certification_ids.add(certification.certification_id)
            certifications.append(certification)
        object.__setattr__(
            self,
            "inventory_events",
            MappingProxyType(dict(sorted(events.items()))),
        )
        object.__setattr__(
            self,
            "certifications",
            tuple(sorted(certifications, key=lambda value: value.certification_id)),
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "manifest_id": self.manifest_id,
            "inventory_events": {
                event_id: event.to_dict() for event_id, event in self.inventory_events.items()
            },
            "certifications": [certification.to_dict() for certification in self.certifications],
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartPlanEvidenceRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "plan_id",
                "run_id",
                "manifest_id",
                "inventory_events",
                "certifications",
            },
        )
        _schema_version(
            value["schema_version"],
            cls.__name__,
            expected=cls.SCHEMA_VERSION,
        )
        inventory_values = _mapping(
            value["inventory_events"],
            "RestartPlanEvidenceRecord.inventory_events",
        )
        inventory_events: dict[str, CheckpointInventoryEvent] = {}
        for event_id, event_value in inventory_values.items():
            normalized_event_id = _nonempty_string(
                event_id,
                "RestartPlanEvidenceRecord.inventory_events key",
            )
            event_mapping = _mapping(
                event_value,
                f"RestartPlanEvidenceRecord.inventory_events[{normalized_event_id!r}]",
            )
            _schema_version(
                event_mapping.get("schema_version"),
                f"RestartPlanEvidenceRecord.inventory_events[{normalized_event_id!r}]",
                expected=CheckpointInventoryEvent.SCHEMA_VERSION,
            )
            try:
                inventory_events[normalized_event_id] = CheckpointInventoryEvent.from_dict(
                    event_mapping
                )
            except ProtocolValidationError as error:
                raise ValueError(
                    "RestartPlanEvidenceRecord contains an invalid inventory event"
                ) from error
        certification_values = _array(
            value["certifications"],
            "RestartPlanEvidenceRecord.certifications",
        )
        certifications: list[CheckpointCertification] = []
        for index, certification_value in enumerate(certification_values):
            certification_mapping = _mapping(
                certification_value,
                f"RestartPlanEvidenceRecord.certifications[{index}]",
            )
            _schema_version(
                certification_mapping.get("schema_version"),
                f"RestartPlanEvidenceRecord.certifications[{index}]",
                expected=CheckpointCertification.SCHEMA_VERSION,
            )
            try:
                certifications.append(CheckpointCertification.from_dict(certification_mapping))
            except ProtocolValidationError as error:
                raise ValueError(
                    "RestartPlanEvidenceRecord contains an invalid certification"
                ) from error
        return cls(
            plan_id=_nonempty_string(
                value["plan_id"],
                "RestartPlanEvidenceRecord.plan_id",
            ),
            run_id=_nonempty_string(
                value["run_id"],
                "RestartPlanEvidenceRecord.run_id",
            ),
            manifest_id=_nonempty_string(
                value["manifest_id"],
                "RestartPlanEvidenceRecord.manifest_id",
            ),
            inventory_events=inventory_events,
            certifications=tuple(certifications),
        )


@dataclass(frozen=True, slots=True)
class RestartPlanRecord:
    """One immutable plan envelope for atomic restart publication."""

    SCHEMA_VERSION: ClassVar[int] = 2

    plan: RestartPlan
    recovery_manifest_record_digest: str
    recovery_evidence_record_digest: str
    intent_lifecycle_record_digest: str
    from_generation_snapshot_digest: str
    to_generation_snapshot_digest: str
    quarantine_record_digests: Mapping[str, str]
    coordinator_id: str
    lease_id: str
    coordinator_lease_duration_ms: int
    coordinator_fencing_token: int

    def __post_init__(self) -> None:
        if not isinstance(self.plan, RestartPlan):
            raise TypeError("RestartPlanRecord.plan must be RestartPlan")
        for value, path in (
            (
                self.recovery_manifest_record_digest,
                "RestartPlanRecord.recovery_manifest_record_digest",
            ),
            (
                self.recovery_evidence_record_digest,
                "RestartPlanRecord.recovery_evidence_record_digest",
            ),
            (
                self.intent_lifecycle_record_digest,
                "RestartPlanRecord.intent_lifecycle_record_digest",
            ),
            (
                self.from_generation_snapshot_digest,
                "RestartPlanRecord.from_generation_snapshot_digest",
            ),
            (
                self.to_generation_snapshot_digest,
                "RestartPlanRecord.to_generation_snapshot_digest",
            ),
        ):
            _digest(value, path)
        quarantine_digests = _digest_mapping(
            self.quarantine_record_digests,
            "RestartPlanRecord.quarantine_record_digests",
        )
        expected_quarantine_nodes = set(self.plan.quarantined_node_ids)
        if set(quarantine_digests) != expected_quarantine_nodes:
            raise ValueError(
                "RestartPlanRecord.quarantine_record_digests keys must exactly "
                "match the plan's quarantined nodes"
            )
        object.__setattr__(
            self,
            "quarantine_record_digests",
            MappingProxyType(quarantine_digests),
        )
        _nonempty_string(self.coordinator_id, "RestartPlanRecord.coordinator_id")
        _nonempty_string(self.lease_id, "RestartPlanRecord.lease_id")
        _positive_integer(
            self.coordinator_lease_duration_ms,
            "RestartPlanRecord.coordinator_lease_duration_ms",
        )
        _positive_integer(
            self.coordinator_fencing_token,
            "RestartPlanRecord.coordinator_fencing_token",
        )

    @property
    def digest(self) -> str:
        return hashlib.sha256(self.to_json()).hexdigest()

    @property
    def coordinator_lease_digest(self) -> str:
        record = CoordinatorLeaseRecord(
            run_id=self.plan.run_id,
            coordinator_id=self.coordinator_id,
            lease_id=self.lease_id,
            lease_duration_ms=self.coordinator_lease_duration_ms,
        )
        return hashlib.sha256(record.to_json()).hexdigest()

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan": self.plan.to_dict(),
            "recovery_manifest_record_digest": self.recovery_manifest_record_digest,
            "recovery_evidence_record_digest": self.recovery_evidence_record_digest,
            "intent_lifecycle_record_digest": self.intent_lifecycle_record_digest,
            "from_generation_snapshot_digest": self.from_generation_snapshot_digest,
            "to_generation_snapshot_digest": self.to_generation_snapshot_digest,
            "quarantine_record_digests": dict(self.quarantine_record_digests),
            "coordinator_id": self.coordinator_id,
            "lease_id": self.lease_id,
            "coordinator_lease_duration_ms": self.coordinator_lease_duration_ms,
            "coordinator_fencing_token": self.coordinator_fencing_token,
        }

    def to_json(self) -> bytes:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, encoded: bytes) -> RestartPlanRecord:
        value = _json_object(encoded, cls.__name__)
        _fields(
            value,
            path=cls.__name__,
            required={
                "schema_version",
                "plan",
                "recovery_manifest_record_digest",
                "recovery_evidence_record_digest",
                "intent_lifecycle_record_digest",
                "from_generation_snapshot_digest",
                "to_generation_snapshot_digest",
                "quarantine_record_digests",
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
        plan_value = _mapping(value["plan"], "RestartPlanRecord.plan")
        _schema_version(
            plan_value.get("schema_version"),
            "RestartPlanRecord.plan",
            expected=RestartPlan.SCHEMA_VERSION,
        )
        try:
            plan = RestartPlan.from_dict(plan_value)
        except ProtocolValidationError as error:
            raise ValueError("RestartPlanRecord.plan is invalid") from error
        return cls(
            plan=plan,
            recovery_manifest_record_digest=_digest(
                value["recovery_manifest_record_digest"],
                "RestartPlanRecord.recovery_manifest_record_digest",
            ),
            recovery_evidence_record_digest=_digest(
                value["recovery_evidence_record_digest"],
                "RestartPlanRecord.recovery_evidence_record_digest",
            ),
            intent_lifecycle_record_digest=_digest(
                value["intent_lifecycle_record_digest"],
                "RestartPlanRecord.intent_lifecycle_record_digest",
            ),
            from_generation_snapshot_digest=_digest(
                value["from_generation_snapshot_digest"],
                "RestartPlanRecord.from_generation_snapshot_digest",
            ),
            to_generation_snapshot_digest=_digest(
                value["to_generation_snapshot_digest"],
                "RestartPlanRecord.to_generation_snapshot_digest",
            ),
            quarantine_record_digests=_digest_mapping(
                value["quarantine_record_digests"],
                "RestartPlanRecord.quarantine_record_digests",
            ),
            coordinator_id=_nonempty_string(
                value["coordinator_id"],
                "RestartPlanRecord.coordinator_id",
            ),
            lease_id=_nonempty_string(
                value["lease_id"],
                "RestartPlanRecord.lease_id",
            ),
            coordinator_lease_duration_ms=_positive_integer(
                value["coordinator_lease_duration_ms"],
                "RestartPlanRecord.coordinator_lease_duration_ms",
            ),
            coordinator_fencing_token=_positive_integer(
                value["coordinator_fencing_token"],
                "RestartPlanRecord.coordinator_fencing_token",
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


def _array(value: object, path: str) -> list[object]:
    if not isinstance(value, list):
        raise ValueError(f"{path}: expected a JSON array")
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


def _digest_mapping(value: object, path: str) -> dict[str, str]:
    mapping = _mapping(value, path)
    result: dict[str, str] = {}
    for key, digest in mapping.items():
        normalized_key = _nonempty_string(key, f"{path}.key")
        result[normalized_key] = _digest(digest, f"{path}[{normalized_key!r}]")
    return dict(sorted(result.items()))


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
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


__all__ = [
    "RecoveryManifestRecord",
    "RestartPlanEvidenceRecord",
    "RestartPlanRecord",
]
