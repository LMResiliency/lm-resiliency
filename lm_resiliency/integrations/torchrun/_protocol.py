"""Canonical manager-to-torchrun recovery records."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, TypeVar

_T = TypeVar("_T", bound="_WireRecord")


class ProtocolValidationError(ValueError):
    """Raised when an untrusted torchrun recovery record is invalid."""


class _WireRecord:
    SCHEMA_VERSION: ClassVar[int] = 1

    def to_dict(self) -> dict[str, Any]:
        raise NotImplementedError

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )

    @classmethod
    def from_json(cls: type[_T], encoded: str | bytes | bytearray) -> _T:
        try:
            value = json.loads(encoded, object_pairs_hook=_reject_duplicate_fields)
        except ProtocolValidationError:
            raise
        except (TypeError, ValueError, UnicodeError) as error:
            raise ProtocolValidationError(f"{cls.__name__}: invalid JSON") from error
        return cls.from_dict(_mapping(value, cls.__name__))  # type: ignore[attr-defined]


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ProtocolValidationError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


def _mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{path}: expected an object")
    if not all(isinstance(key, str) for key in value):
        raise ProtocolValidationError(f"{path}: object keys must be strings")
    return value


def _record_fields(
    value: Mapping[str, Any],
    *,
    path: str,
    required: set[str],
) -> None:
    schema_version = value.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != 1
    ):
        raise ProtocolValidationError(
            f"{path}.schema_version: unsupported value {schema_version!r}; expected 1"
        )
    expected = required | {"schema_version"}
    missing = required - set(value)
    unknown = set(value) - expected
    if missing:
        raise ProtocolValidationError(f"{path}: missing fields {sorted(missing)!r}")
    if unknown:
        raise ProtocolValidationError(f"{path}: unknown fields {sorted(unknown)!r}")


def _string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProtocolValidationError(f"{path}: expected a non-empty string")
    return value


def _optional_string(value: object, path: str) -> str | None:
    return None if value is None else _string(value, path)


def _integer(value: object, path: str, *, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{path}: expected an integer")
    if value < minimum:
        raise ProtocolValidationError(f"{path}: expected a value >= {minimum}")
    return value


def _choice(value: object, path: str, choices: set[str]) -> str:
    selected = _string(value, path)
    if selected not in choices:
        raise ProtocolValidationError(
            f"{path}: unsupported value {selected!r}; expected one of {sorted(choices)!r}"
        )
    return selected


def _sequence(value: object, path: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ProtocolValidationError(f"{path}: expected an array")
    return value


def _strings(value: object, path: str, *, unique: bool) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    )
    if unique and len(result) != len(set(result)):
        raise ProtocolValidationError(f"{path}: values must be unique")
    return result


@dataclass(frozen=True, slots=True)
class SlotAssignment:
    """One stable logical node slot in a recovery plan."""

    logical_node_slot: int
    node_id: str
    first_global_rank: int
    local_world_size: int

    def __post_init__(self) -> None:
        _integer(self.logical_node_slot, "SlotAssignment.logical_node_slot", minimum=0)
        _string(self.node_id, "SlotAssignment.node_id")
        _integer(self.first_global_rank, "SlotAssignment.first_global_rank", minimum=0)
        _integer(self.local_world_size, "SlotAssignment.local_world_size", minimum=1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "logical_node_slot": self.logical_node_slot,
            "node_id": self.node_id,
            "first_global_rank": self.first_global_rank,
            "local_world_size": self.local_world_size,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> SlotAssignment:
        required = {
            "logical_node_slot",
            "node_id",
            "first_global_rank",
            "local_world_size",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ProtocolValidationError(f"SlotAssignment: missing fields {sorted(missing)!r}")
        if unknown:
            raise ProtocolValidationError(f"SlotAssignment: unknown fields {sorted(unknown)!r}")
        return cls(
            logical_node_slot=_integer(
                value["logical_node_slot"],
                "SlotAssignment.logical_node_slot",
                minimum=0,
            ),
            node_id=_string(value["node_id"], "SlotAssignment.node_id"),
            first_global_rank=_integer(
                value["first_global_rank"],
                "SlotAssignment.first_global_rank",
                minimum=0,
            ),
            local_world_size=_integer(
                value["local_world_size"],
                "SlotAssignment.local_world_size",
                minimum=1,
            ),
        )


def _assignments(value: object, path: str) -> tuple[SlotAssignment, ...]:
    assignments = tuple(
        item
        if isinstance(item, SlotAssignment)
        else SlotAssignment.from_dict(_mapping(item, f"{path}[{index}]"))
        for index, item in enumerate(_sequence(value, path))
    )
    if not assignments:
        raise ProtocolValidationError(f"{path}: at least one assignment is required")
    slots = [assignment.logical_node_slot for assignment in assignments]
    if len(slots) != len(set(slots)):
        raise ProtocolValidationError(f"{path}: logical slots must be unique")
    if set(slots) != set(range(len(slots))):
        raise ProtocolValidationError(f"{path}: logical slots must be dense from zero")
    node_ids = [assignment.node_id for assignment in assignments]
    if len(node_ids) != len(set(node_ids)):
        raise ProtocolValidationError(f"{path}: assigned node IDs must be unique")
    local_sizes = {assignment.local_world_size for assignment in assignments}
    if len(local_sizes) != 1:
        raise ProtocolValidationError(f"{path}: all assignments must use one local world size")
    local_world_size = assignments[0].local_world_size
    for assignment in assignments:
        expected = assignment.logical_node_slot * local_world_size
        if assignment.first_global_rank != expected:
            raise ProtocolValidationError(
                f"{path}: slot {assignment.logical_node_slot} must start at rank {expected}"
            )
    return tuple(sorted(assignments, key=lambda assignment: assignment.logical_node_slot))


@dataclass(frozen=True, slots=True)
class RestartPlan(_WireRecord):
    """One manager-owned replacement and checkpoint decision."""

    plan_id: str
    intent_id: str
    run_id: str
    from_generation: int
    to_generation: int
    incident_ids: tuple[str, ...]
    reason_code: str
    recovery_mode: str
    checkpoint_source: str
    checkpoint_step: int
    checkpoint_id: str | None
    checkpoint_manifest_id: str
    slot_assignments: tuple[SlotAssignment, ...]
    quarantined_node_ids: tuple[str, ...]
    expected_world_size: int
    topology_digest: str
    restart_deadline_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.plan_id, "RestartPlan.plan_id")
        _string(self.intent_id, "RestartPlan.intent_id")
        _string(self.run_id, "RestartPlan.run_id")
        _integer(self.from_generation, "RestartPlan.from_generation", minimum=0)
        _integer(self.to_generation, "RestartPlan.to_generation", minimum=1)
        if self.to_generation != self.from_generation + 1:
            raise ProtocolValidationError(
                "RestartPlan.to_generation: must be the successor generation"
            )
        incident_ids = _strings(
            self.incident_ids,
            "RestartPlan.incident_ids",
            unique=True,
        )
        if not incident_ids:
            raise ProtocolValidationError(
                "RestartPlan.incident_ids: at least one incident is required"
            )
        object.__setattr__(self, "incident_ids", incident_ids)
        _string(self.reason_code, "RestartPlan.reason_code")
        _choice(
            self.recovery_mode,
            "RestartPlan.recovery_mode",
            {"latest", "recovery_verified"},
        )
        _choice(
            self.checkpoint_source,
            "RestartPlan.checkpoint_source",
            {"gemini", "durable"},
        )
        _integer(self.checkpoint_step, "RestartPlan.checkpoint_step", minimum=1)
        _optional_string(self.checkpoint_id, "RestartPlan.checkpoint_id")
        if self.checkpoint_source == "durable" and self.checkpoint_id is None:
            raise ProtocolValidationError(
                "RestartPlan.checkpoint_id: durable recovery requires an ID"
            )
        if self.checkpoint_source == "gemini" and self.checkpoint_id is not None:
            raise ProtocolValidationError(
                "RestartPlan.checkpoint_id: GEMINI recovery must not set an ID"
            )
        _string(self.checkpoint_manifest_id, "RestartPlan.checkpoint_manifest_id")
        assignments = _assignments(self.slot_assignments, "RestartPlan.slot_assignments")
        object.__setattr__(self, "slot_assignments", assignments)
        quarantined = _strings(
            self.quarantined_node_ids,
            "RestartPlan.quarantined_node_ids",
            unique=True,
        )
        object.__setattr__(self, "quarantined_node_ids", quarantined)
        overlap = {assignment.node_id for assignment in assignments} & set(quarantined)
        if overlap:
            raise ProtocolValidationError(
                f"RestartPlan: quarantined nodes cannot be assigned: {sorted(overlap)!r}"
            )
        _integer(self.expected_world_size, "RestartPlan.expected_world_size", minimum=1)
        calculated_world_size = sum(assignment.local_world_size for assignment in assignments)
        if self.expected_world_size != calculated_world_size:
            raise ProtocolValidationError(
                "RestartPlan.expected_world_size: does not match slot assignments"
            )
        _string(self.topology_digest, "RestartPlan.topology_digest")
        _integer(
            self.restart_deadline_unix_ms,
            "RestartPlan.restart_deadline_unix_ms",
            minimum=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "intent_id": self.intent_id,
            "run_id": self.run_id,
            "from_generation": self.from_generation,
            "to_generation": self.to_generation,
            "incident_ids": list(self.incident_ids),
            "reason_code": self.reason_code,
            "recovery_mode": self.recovery_mode,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_manifest_id": self.checkpoint_manifest_id,
            "slot_assignments": [assignment.to_dict() for assignment in self.slot_assignments],
            "quarantined_node_ids": list(self.quarantined_node_ids),
            "expected_world_size": self.expected_world_size,
            "topology_digest": self.topology_digest,
            "restart_deadline_unix_ms": self.restart_deadline_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RestartPlan:
        required = {
            "plan_id",
            "intent_id",
            "run_id",
            "from_generation",
            "to_generation",
            "incident_ids",
            "reason_code",
            "recovery_mode",
            "checkpoint_source",
            "checkpoint_step",
            "checkpoint_id",
            "checkpoint_manifest_id",
            "slot_assignments",
            "quarantined_node_ids",
            "expected_world_size",
            "topology_digest",
            "restart_deadline_unix_ms",
        }
        _record_fields(value, path=cls.__name__, required=required)
        return cls(
            plan_id=_string(value["plan_id"], "RestartPlan.plan_id"),
            intent_id=_string(value["intent_id"], "RestartPlan.intent_id"),
            run_id=_string(value["run_id"], "RestartPlan.run_id"),
            from_generation=_integer(
                value["from_generation"],
                "RestartPlan.from_generation",
                minimum=0,
            ),
            to_generation=_integer(
                value["to_generation"],
                "RestartPlan.to_generation",
                minimum=1,
            ),
            incident_ids=_strings(
                value["incident_ids"],
                "RestartPlan.incident_ids",
                unique=True,
            ),
            reason_code=_string(value["reason_code"], "RestartPlan.reason_code"),
            recovery_mode=_choice(
                value["recovery_mode"],
                "RestartPlan.recovery_mode",
                {"latest", "recovery_verified"},
            ),
            checkpoint_source=_choice(
                value["checkpoint_source"],
                "RestartPlan.checkpoint_source",
                {"gemini", "durable"},
            ),
            checkpoint_step=_integer(
                value["checkpoint_step"],
                "RestartPlan.checkpoint_step",
                minimum=1,
            ),
            checkpoint_id=_optional_string(
                value["checkpoint_id"],
                "RestartPlan.checkpoint_id",
            ),
            checkpoint_manifest_id=_string(
                value["checkpoint_manifest_id"],
                "RestartPlan.checkpoint_manifest_id",
            ),
            slot_assignments=_assignments(
                value["slot_assignments"],
                "RestartPlan.slot_assignments",
            ),
            quarantined_node_ids=_strings(
                value["quarantined_node_ids"],
                "RestartPlan.quarantined_node_ids",
                unique=True,
            ),
            expected_world_size=_integer(
                value["expected_world_size"],
                "RestartPlan.expected_world_size",
                minimum=1,
            ),
            topology_digest=_string(
                value["topology_digest"],
                "RestartPlan.topology_digest",
            ),
            restart_deadline_unix_ms=_integer(
                value["restart_deadline_unix_ms"],
                "RestartPlan.restart_deadline_unix_ms",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartContext(_WireRecord):
    """Node-local handoff derived exactly from one committed plan."""

    plan_id: str
    run_id: str
    generation: int
    node_id: str
    logical_node_slot: int
    first_global_rank: int
    local_world_size: int
    expected_world_size: int
    topology_digest: str
    recovery_mode: str
    checkpoint_source: str
    checkpoint_step: int
    checkpoint_id: str | None
    checkpoint_manifest_id: str
    reason_code: str

    def __post_init__(self) -> None:
        _string(self.plan_id, "RestartContext.plan_id")
        _string(self.run_id, "RestartContext.run_id")
        _integer(self.generation, "RestartContext.generation", minimum=0)
        _string(self.node_id, "RestartContext.node_id")
        _integer(
            self.logical_node_slot,
            "RestartContext.logical_node_slot",
            minimum=0,
        )
        _integer(self.first_global_rank, "RestartContext.first_global_rank", minimum=0)
        _integer(self.local_world_size, "RestartContext.local_world_size", minimum=1)
        expected_first_rank = self.logical_node_slot * self.local_world_size
        if self.first_global_rank != expected_first_rank:
            raise ProtocolValidationError(
                "RestartContext.first_global_rank: does not match logical slot"
            )
        _integer(
            self.expected_world_size,
            "RestartContext.expected_world_size",
            minimum=1,
        )
        if self.expected_world_size % self.local_world_size:
            raise ProtocolValidationError(
                "RestartContext.expected_world_size: must contain complete local worker slots"
            )
        if self.first_global_rank + self.local_world_size > self.expected_world_size:
            raise ProtocolValidationError(
                "RestartContext: local rank range exceeds expected world size"
            )
        _string(self.topology_digest, "RestartContext.topology_digest")
        _choice(
            self.recovery_mode,
            "RestartContext.recovery_mode",
            {"latest", "recovery_verified"},
        )
        _choice(
            self.checkpoint_source,
            "RestartContext.checkpoint_source",
            {"gemini", "durable"},
        )
        _integer(self.checkpoint_step, "RestartContext.checkpoint_step", minimum=1)
        _optional_string(self.checkpoint_id, "RestartContext.checkpoint_id")
        if self.checkpoint_source == "durable" and self.checkpoint_id is None:
            raise ProtocolValidationError(
                "RestartContext.checkpoint_id: durable recovery requires an ID"
            )
        if self.checkpoint_source == "gemini" and self.checkpoint_id is not None:
            raise ProtocolValidationError(
                "RestartContext.checkpoint_id: GEMINI recovery must not set an ID"
            )
        _string(self.checkpoint_manifest_id, "RestartContext.checkpoint_manifest_id")
        _string(self.reason_code, "RestartContext.reason_code")

    @classmethod
    def from_plan(cls, plan: RestartPlan, node_id: str) -> RestartContext:
        assignment = next(
            (assignment for assignment in plan.slot_assignments if assignment.node_id == node_id),
            None,
        )
        if assignment is None:
            raise ProtocolValidationError(
                f"RestartContext: node {node_id!r} is not assigned by the plan"
            )
        return cls(
            plan_id=plan.plan_id,
            run_id=plan.run_id,
            generation=plan.to_generation,
            node_id=assignment.node_id,
            logical_node_slot=assignment.logical_node_slot,
            first_global_rank=assignment.first_global_rank,
            local_world_size=assignment.local_world_size,
            expected_world_size=plan.expected_world_size,
            topology_digest=plan.topology_digest,
            recovery_mode=plan.recovery_mode,
            checkpoint_source=plan.checkpoint_source,
            checkpoint_step=plan.checkpoint_step,
            checkpoint_id=plan.checkpoint_id,
            checkpoint_manifest_id=plan.checkpoint_manifest_id,
            reason_code=plan.reason_code,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "plan_id": self.plan_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "node_id": self.node_id,
            "logical_node_slot": self.logical_node_slot,
            "first_global_rank": self.first_global_rank,
            "local_world_size": self.local_world_size,
            "expected_world_size": self.expected_world_size,
            "topology_digest": self.topology_digest,
            "recovery_mode": self.recovery_mode,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_step": self.checkpoint_step,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_manifest_id": self.checkpoint_manifest_id,
            "reason_code": self.reason_code,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RestartContext:
        required = {
            "plan_id",
            "run_id",
            "generation",
            "node_id",
            "logical_node_slot",
            "first_global_rank",
            "local_world_size",
            "expected_world_size",
            "topology_digest",
            "recovery_mode",
            "checkpoint_source",
            "checkpoint_step",
            "checkpoint_id",
            "checkpoint_manifest_id",
            "reason_code",
        }
        _record_fields(value, path=cls.__name__, required=required)
        return cls(
            plan_id=_string(value["plan_id"], "RestartContext.plan_id"),
            run_id=_string(value["run_id"], "RestartContext.run_id"),
            generation=_integer(
                value["generation"],
                "RestartContext.generation",
                minimum=0,
            ),
            node_id=_string(value["node_id"], "RestartContext.node_id"),
            logical_node_slot=_integer(
                value["logical_node_slot"],
                "RestartContext.logical_node_slot",
                minimum=0,
            ),
            first_global_rank=_integer(
                value["first_global_rank"],
                "RestartContext.first_global_rank",
                minimum=0,
            ),
            local_world_size=_integer(
                value["local_world_size"],
                "RestartContext.local_world_size",
                minimum=1,
            ),
            expected_world_size=_integer(
                value["expected_world_size"],
                "RestartContext.expected_world_size",
                minimum=1,
            ),
            topology_digest=_string(
                value["topology_digest"],
                "RestartContext.topology_digest",
            ),
            recovery_mode=_choice(
                value["recovery_mode"],
                "RestartContext.recovery_mode",
                {"latest", "recovery_verified"},
            ),
            checkpoint_source=_choice(
                value["checkpoint_source"],
                "RestartContext.checkpoint_source",
                {"gemini", "durable"},
            ),
            checkpoint_step=_integer(
                value["checkpoint_step"],
                "RestartContext.checkpoint_step",
                minimum=1,
            ),
            checkpoint_id=_optional_string(
                value["checkpoint_id"],
                "RestartContext.checkpoint_id",
            ),
            checkpoint_manifest_id=_string(
                value["checkpoint_manifest_id"],
                "RestartContext.checkpoint_manifest_id",
            ),
            reason_code=_string(value["reason_code"], "RestartContext.reason_code"),
        )


__all__ = [
    "ProtocolValidationError",
    "RestartContext",
    "RestartPlan",
    "SlotAssignment",
]
