"""Strict internal wire records for torchrun standby replacement."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, ClassVar, Literal, TypeVar

RecoveryModeValue = Literal["latest", "recovery_verified"]
CheckpointSource = Literal["gemini", "durable"]
CheckpointTrust = Literal["latest", "candidate", "recovery_verified"]
RecoveryManifestTrust = Literal["latest", "recovery_verified"]

_T = TypeVar("_T", bound="_WireRecord")


class ProtocolValidationError(ValueError):
    """Raised when an untrusted torchrun protocol record is invalid."""


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
            value = json.loads(encoded)
        except (TypeError, ValueError) as error:
            raise ProtocolValidationError(f"{cls.__name__}: invalid JSON") from error
        return cls.from_dict(_require_mapping(value, cls.__name__))  # type: ignore[attr-defined]


def _require_mapping(value: object, path: str) -> Mapping[str, Any]:
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
    optional: set[str] | None = None,
    schema_version: int = 1,
) -> None:
    optional = optional or set()
    actual_schema = value.get("schema_version")
    if isinstance(actual_schema, bool) or actual_schema != schema_version:
        raise ProtocolValidationError(
            f"{path}.schema_version: unsupported value {actual_schema!r}; expected {schema_version}"
        )
    expected = required | optional | {"schema_version"}
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
    if value is None:
        return None
    return _string(value, path)


def _integer(value: object, path: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ProtocolValidationError(f"{path}: expected an integer")
    if minimum is not None and value < minimum:
        raise ProtocolValidationError(f"{path}: expected a value >= {minimum}")
    return value


def _boolean(value: object, path: str) -> bool:
    if not isinstance(value, bool):
        raise ProtocolValidationError(f"{path}: expected a boolean")
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


def _strings(value: object, path: str, *, unique: bool = False) -> tuple[str, ...]:
    result = tuple(
        _string(item, f"{path}[{index}]") for index, item in enumerate(_sequence(value, path))
    )
    if unique and len(result) != len(set(result)):
        raise ProtocolValidationError(f"{path}: values must be unique")
    return result


def _integers(
    value: object,
    path: str,
    *,
    minimum: int | None = None,
    unique: bool = False,
) -> tuple[int, ...]:
    result = tuple(
        _integer(item, f"{path}[{index}]", minimum=minimum)
        for index, item in enumerate(_sequence(value, path))
    )
    if unique and len(result) != len(set(result)):
        raise ProtocolValidationError(f"{path}: values must be unique")
    return result


def _freeze_json(value: object, path: str) -> object:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ProtocolValidationError(f"{path}: floating-point values must be finite")
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ProtocolValidationError(f"{path}: object keys must be strings")
        return MappingProxyType(
            {key: _freeze_json(item, f"{path}.{key}") for key, item in value.items()}
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, f"{path}[{index}]") for index, item in enumerate(value))
    raise ProtocolValidationError(f"{path}: value is not JSON-serializable")


def _thaw_json(value: object) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AgentIdentity(_WireRecord):
    run_id: str
    node_id: str
    agent_id: str
    hostname: str
    local_world_size: int
    resource_ids: tuple[str, ...]
    environment_digest: str

    def __post_init__(self) -> None:
        _string(self.run_id, "AgentIdentity.run_id")
        _string(self.node_id, "AgentIdentity.node_id")
        _string(self.agent_id, "AgentIdentity.agent_id")
        _string(self.hostname, "AgentIdentity.hostname")
        _integer(self.local_world_size, "AgentIdentity.local_world_size", minimum=1)
        object.__setattr__(
            self,
            "resource_ids",
            _strings(self.resource_ids, "AgentIdentity.resource_ids", unique=True),
        )
        _string(self.environment_digest, "AgentIdentity.environment_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "hostname": self.hostname,
            "local_world_size": self.local_world_size,
            "resource_ids": list(self.resource_ids),
            "environment_digest": self.environment_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> AgentIdentity:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "run_id",
                "node_id",
                "agent_id",
                "hostname",
                "local_world_size",
                "resource_ids",
                "environment_digest",
            },
        )
        return cls(
            run_id=_string(value["run_id"], "AgentIdentity.run_id"),
            node_id=_string(value["node_id"], "AgentIdentity.node_id"),
            agent_id=_string(value["agent_id"], "AgentIdentity.agent_id"),
            hostname=_string(value["hostname"], "AgentIdentity.hostname"),
            local_world_size=_integer(
                value["local_world_size"],
                "AgentIdentity.local_world_size",
                minimum=1,
            ),
            resource_ids=_strings(
                value["resource_ids"],
                "AgentIdentity.resource_ids",
                unique=True,
            ),
            environment_digest=_string(
                value["environment_digest"],
                "AgentIdentity.environment_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class WorkerIdentity(_WireRecord):
    run_id: str
    generation: int
    node_id: str
    agent_id: str
    logical_node_slot: int
    global_rank: int
    local_rank: int
    local_world_size: int
    hostname: str
    gpu_uuid: str | None
    topology_digest: str

    def __post_init__(self) -> None:
        _string(self.run_id, "WorkerIdentity.run_id")
        _integer(self.generation, "WorkerIdentity.generation", minimum=0)
        _string(self.node_id, "WorkerIdentity.node_id")
        _string(self.agent_id, "WorkerIdentity.agent_id")
        _integer(self.logical_node_slot, "WorkerIdentity.logical_node_slot", minimum=0)
        _integer(self.global_rank, "WorkerIdentity.global_rank", minimum=0)
        _integer(self.local_rank, "WorkerIdentity.local_rank", minimum=0)
        _integer(self.local_world_size, "WorkerIdentity.local_world_size", minimum=1)
        if self.local_rank >= self.local_world_size:
            raise ProtocolValidationError(
                "WorkerIdentity.local_rank: must be smaller than local_world_size"
            )
        expected_global_rank = self.logical_node_slot * self.local_world_size + self.local_rank
        if self.global_rank != expected_global_rank:
            raise ProtocolValidationError(
                "WorkerIdentity.global_rank: does not match logical slot and local rank"
            )
        _string(self.hostname, "WorkerIdentity.hostname")
        _optional_string(self.gpu_uuid, "WorkerIdentity.gpu_uuid")
        _string(self.topology_digest, "WorkerIdentity.topology_digest")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": self.generation,
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "logical_node_slot": self.logical_node_slot,
            "global_rank": self.global_rank,
            "local_rank": self.local_rank,
            "local_world_size": self.local_world_size,
            "hostname": self.hostname,
            "gpu_uuid": self.gpu_uuid,
            "topology_digest": self.topology_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> WorkerIdentity:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "run_id",
                "generation",
                "node_id",
                "agent_id",
                "logical_node_slot",
                "global_rank",
                "local_rank",
                "local_world_size",
                "hostname",
                "gpu_uuid",
                "topology_digest",
            },
        )
        return cls(
            run_id=_string(value["run_id"], "WorkerIdentity.run_id"),
            generation=_integer(
                value["generation"],
                "WorkerIdentity.generation",
                minimum=0,
            ),
            node_id=_string(value["node_id"], "WorkerIdentity.node_id"),
            agent_id=_string(value["agent_id"], "WorkerIdentity.agent_id"),
            logical_node_slot=_integer(
                value["logical_node_slot"],
                "WorkerIdentity.logical_node_slot",
                minimum=0,
            ),
            global_rank=_integer(
                value["global_rank"],
                "WorkerIdentity.global_rank",
                minimum=0,
            ),
            local_rank=_integer(
                value["local_rank"],
                "WorkerIdentity.local_rank",
                minimum=0,
            ),
            local_world_size=_integer(
                value["local_world_size"],
                "WorkerIdentity.local_world_size",
                minimum=1,
            ),
            hostname=_string(value["hostname"], "WorkerIdentity.hostname"),
            gpu_uuid=_optional_string(value["gpu_uuid"], "WorkerIdentity.gpu_uuid"),
            topology_digest=_string(
                value["topology_digest"],
                "WorkerIdentity.topology_digest",
            ),
        )


@dataclass(frozen=True, slots=True)
class SlotAssignment:
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


def _assignments(
    value: object,
    path: str,
    *,
    require_nonempty: bool = True,
) -> tuple[SlotAssignment, ...]:
    assignments = tuple(
        item
        if isinstance(item, SlotAssignment)
        else SlotAssignment.from_dict(_require_mapping(item, f"{path}[{index}]"))
        for index, item in enumerate(_sequence(value, path))
    )
    if require_nonempty and not assignments:
        raise ProtocolValidationError(f"{path}: at least one assignment is required")
    slots = [assignment.logical_node_slot for assignment in assignments]
    if len(slots) != len(set(slots)):
        raise ProtocolValidationError(f"{path}: logical slots must be unique")
    if slots and set(slots) != set(range(len(slots))):
        raise ProtocolValidationError(f"{path}: logical slots must be dense from zero")
    node_ids = [assignment.node_id for assignment in assignments]
    if len(node_ids) != len(set(node_ids)):
        raise ProtocolValidationError(f"{path}: assigned node IDs must be unique")
    local_sizes = {assignment.local_world_size for assignment in assignments}
    if len(local_sizes) > 1:
        raise ProtocolValidationError(f"{path}: all assignments must use one local world size")
    if assignments:
        local_world_size = assignments[0].local_world_size
        for assignment in assignments:
            expected = assignment.logical_node_slot * local_world_size
            if assignment.first_global_rank != expected:
                raise ProtocolValidationError(
                    f"{path}: slot {assignment.logical_node_slot} must start at rank {expected}"
                )
    return tuple(sorted(assignments, key=lambda item: item.logical_node_slot))


@dataclass(frozen=True, slots=True)
class RankAssignment(_WireRecord):
    run_id: str
    generation: int
    active_nodes: int
    local_world_size: int
    slot_to_node_id: Mapping[int, str]
    slot_to_rank_range: Mapping[int, tuple[int, int]]
    topology_digest: str

    def __post_init__(self) -> None:
        _string(self.run_id, "RankAssignment.run_id")
        _integer(self.generation, "RankAssignment.generation", minimum=0)
        _integer(self.active_nodes, "RankAssignment.active_nodes", minimum=1)
        _integer(self.local_world_size, "RankAssignment.local_world_size", minimum=1)
        node_map = _normalize_slot_node_map(
            self.slot_to_node_id,
            "RankAssignment.slot_to_node_id",
        )
        range_map = _normalize_slot_range_map(
            self.slot_to_rank_range,
            "RankAssignment.slot_to_rank_range",
        )
        expected_slots = set(range(self.active_nodes))
        if set(node_map) != expected_slots or set(range_map) != expected_slots:
            raise ProtocolValidationError(
                "RankAssignment: node and rank maps must contain every active slot"
            )
        if len(set(node_map.values())) != len(node_map):
            raise ProtocolValidationError("RankAssignment.slot_to_node_id: node IDs must be unique")
        for slot, rank_range in range_map.items():
            expected = (
                slot * self.local_world_size,
                (slot + 1) * self.local_world_size,
            )
            if rank_range != expected:
                raise ProtocolValidationError(
                    f"RankAssignment.slot_to_rank_range[{slot}]: expected {expected!r}"
                )
        object.__setattr__(self, "slot_to_node_id", MappingProxyType(node_map))
        object.__setattr__(self, "slot_to_rank_range", MappingProxyType(range_map))
        _string(self.topology_digest, "RankAssignment.topology_digest")

    @classmethod
    def from_assignments(
        cls,
        *,
        run_id: str,
        generation: int,
        assignments: Sequence[SlotAssignment],
        topology_digest: str,
    ) -> RankAssignment:
        normalized = _assignments(assignments, "RankAssignment.assignments")
        local_world_size = normalized[0].local_world_size
        return cls(
            run_id=run_id,
            generation=generation,
            active_nodes=len(normalized),
            local_world_size=local_world_size,
            slot_to_node_id={
                assignment.logical_node_slot: assignment.node_id for assignment in normalized
            },
            slot_to_rank_range={
                assignment.logical_node_slot: (
                    assignment.first_global_rank,
                    assignment.first_global_rank + assignment.local_world_size,
                )
                for assignment in normalized
            },
            topology_digest=topology_digest,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "run_id": self.run_id,
            "generation": self.generation,
            "active_nodes": self.active_nodes,
            "local_world_size": self.local_world_size,
            "slot_to_node_id": {
                str(slot): node_id for slot, node_id in self.slot_to_node_id.items()
            },
            "slot_to_rank_range": {
                str(slot): list(rank_range) for slot, rank_range in self.slot_to_rank_range.items()
            },
            "topology_digest": self.topology_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankAssignment:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "run_id",
                "generation",
                "active_nodes",
                "local_world_size",
                "slot_to_node_id",
                "slot_to_rank_range",
                "topology_digest",
            },
        )
        return cls(
            run_id=_string(value["run_id"], "RankAssignment.run_id"),
            generation=_integer(
                value["generation"],
                "RankAssignment.generation",
                minimum=0,
            ),
            active_nodes=_integer(
                value["active_nodes"],
                "RankAssignment.active_nodes",
                minimum=1,
            ),
            local_world_size=_integer(
                value["local_world_size"],
                "RankAssignment.local_world_size",
                minimum=1,
            ),
            slot_to_node_id=_parse_slot_node_map(
                value["slot_to_node_id"],
                "RankAssignment.slot_to_node_id",
            ),
            slot_to_rank_range=_parse_slot_range_map(
                value["slot_to_rank_range"],
                "RankAssignment.slot_to_rank_range",
            ),
            topology_digest=_string(
                value["topology_digest"],
                "RankAssignment.topology_digest",
            ),
        )


def _slot_key(value: object, path: str) -> int:
    if isinstance(value, int) and not isinstance(value, bool):
        return _integer(value, path, minimum=0)
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ProtocolValidationError(f"{path}: slot keys must be non-negative integers")


def _normalize_slot_node_map(
    value: object,
    path: str,
) -> dict[int, str]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{path}: expected an object")
    mapping = value
    result: dict[int, str] = {}
    for raw_slot, node_id in mapping.items():
        slot = _slot_key(raw_slot, f"{path}.key")
        if slot in result:
            raise ProtocolValidationError(f"{path}: duplicate slot {slot}")
        result[slot] = _string(node_id, f"{path}[{slot}]")
    return dict(sorted(result.items()))


def _parse_slot_node_map(value: object, path: str) -> dict[int, str]:
    return _normalize_slot_node_map(value, path)


def _normalize_slot_range_map(
    value: object,
    path: str,
) -> dict[int, tuple[int, int]]:
    if not isinstance(value, Mapping):
        raise ProtocolValidationError(f"{path}: expected an object")
    mapping = value
    result: dict[int, tuple[int, int]] = {}
    for raw_slot, raw_range in mapping.items():
        slot = _slot_key(raw_slot, f"{path}.key")
        if slot in result:
            raise ProtocolValidationError(f"{path}: duplicate slot {slot}")
        values = _integers(raw_range, f"{path}[{slot}]", minimum=0)
        if len(values) != 2 or values[1] <= values[0]:
            raise ProtocolValidationError(
                f"{path}[{slot}]: expected a non-empty half-open rank range"
            )
        result[slot] = (values[0], values[1])
    return dict(sorted(result.items()))


def _parse_slot_range_map(value: object, path: str) -> dict[int, tuple[int, int]]:
    return _normalize_slot_range_map(value, path)


@dataclass(frozen=True, slots=True)
class HardwareFaultReport:
    kind: Literal["hardware"]
    resource_kind: str
    resource_id: str
    metric: str
    value: float
    severity: Literal["fatal"]
    message: str

    def __post_init__(self) -> None:
        if self.kind != "hardware":
            raise ProtocolValidationError("HardwareFaultReport.kind: expected 'hardware'")
        _choice(
            self.resource_kind,
            "HardwareFaultReport.resource_kind",
            {"gpu", "node", "nic", "hca", "link"},
        )
        _string(self.resource_id, "HardwareFaultReport.resource_id")
        _string(self.metric, "HardwareFaultReport.metric")
        if isinstance(self.value, bool) or not isinstance(self.value, (int, float)):
            raise ProtocolValidationError("HardwareFaultReport.value: expected a number")
        if not math.isfinite(float(self.value)):
            raise ProtocolValidationError("HardwareFaultReport.value: must be finite")
        if self.severity != "fatal":
            raise ProtocolValidationError("HardwareFaultReport.severity: expected 'fatal'")
        _string(self.message, "HardwareFaultReport.message")

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "resource_kind": self.resource_kind,
            "resource_id": self.resource_id,
            "metric": self.metric,
            "value": self.value,
            "severity": self.severity,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> HardwareFaultReport:
        required = {
            "kind",
            "resource_kind",
            "resource_id",
            "metric",
            "value",
            "severity",
            "message",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ProtocolValidationError(
                f"HardwareFaultReport: missing fields {sorted(missing)!r}"
            )
        if unknown:
            raise ProtocolValidationError(
                f"HardwareFaultReport: unknown fields {sorted(unknown)!r}"
            )
        raw_value = value["value"]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ProtocolValidationError("HardwareFaultReport.value: expected a number")
        return cls(
            kind=_choice(value["kind"], "HardwareFaultReport.kind", {"hardware"}),  # type: ignore[arg-type]
            resource_kind=_choice(
                value["resource_kind"],
                "HardwareFaultReport.resource_kind",
                {"gpu", "node", "nic", "hca", "link"},
            ),
            resource_id=_string(
                value["resource_id"],
                "HardwareFaultReport.resource_id",
            ),
            metric=_string(value["metric"], "HardwareFaultReport.metric"),
            value=float(raw_value),
            severity=_choice(  # type: ignore[arg-type]
                value["severity"],
                "HardwareFaultReport.severity",
                {"fatal"},
            ),
            message=_string(value["message"], "HardwareFaultReport.message"),
        )


def _fault_report(value: object, path: str) -> Mapping[str, Any]:
    if isinstance(value, HardwareFaultReport):
        normalized: Mapping[str, Any] = value.to_dict()
    else:
        normalized = _require_mapping(value, path)
        kind = _string(normalized.get("kind"), f"{path}.kind")
        if kind == "hardware":
            normalized = HardwareFaultReport.from_dict(normalized).to_dict()
        elif "failed_ranks" in normalized:
            _integers(
                normalized["failed_ranks"],
                f"{path}.failed_ranks",
                minimum=0,
                unique=True,
            )
    frozen = _freeze_json(normalized, path)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class FaultEvent(_WireRecord):
    event_id: str
    incident_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    optimizer_step: int
    report: Mapping[str, Any] | HardwareFaultReport

    def __post_init__(self) -> None:
        _string(self.event_id, "FaultEvent.event_id")
        _string(self.incident_id, "FaultEvent.incident_id")
        _string(self.run_id, "FaultEvent.run_id")
        _integer(self.generation, "FaultEvent.generation", minimum=0)
        if not isinstance(self.reporter, WorkerIdentity):
            raise ProtocolValidationError("FaultEvent.reporter: expected WorkerIdentity")
        if self.reporter.run_id != self.run_id:
            raise ProtocolValidationError("FaultEvent: reporter run_id does not match event")
        if self.reporter.generation != self.generation:
            raise ProtocolValidationError("FaultEvent: reporter generation does not match event")
        _integer(self.optimizer_step, "FaultEvent.optimizer_step", minimum=0)
        object.__setattr__(self, "report", _fault_report(self.report, "FaultEvent.report"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "event_id": self.event_id,
            "incident_id": self.incident_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "reporter": self.reporter.to_dict(),
            "optimizer_step": self.optimizer_step,
            "report": _thaw_json(self.report),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> FaultEvent:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "event_id",
                "incident_id",
                "run_id",
                "generation",
                "reporter",
                "optimizer_step",
                "report",
            },
        )
        return cls(
            event_id=_string(value["event_id"], "FaultEvent.event_id"),
            incident_id=_string(value["incident_id"], "FaultEvent.incident_id"),
            run_id=_string(value["run_id"], "FaultEvent.run_id"),
            generation=_integer(
                value["generation"],
                "FaultEvent.generation",
                minimum=0,
            ),
            reporter=WorkerIdentity.from_dict(
                _require_mapping(value["reporter"], "FaultEvent.reporter")
            ),
            optimizer_step=_integer(
                value["optimizer_step"],
                "FaultEvent.optimizer_step",
                minimum=0,
            ),
            report=_require_mapping(value["report"], "FaultEvent.report"),
        )


def _recovery_decision(value: object, path: str) -> Mapping[str, Any]:
    mapping = _require_mapping(value, path)
    fields = {
        "failure_kind",
        "recovery_mode",
        "checkpoint_source",
        "checkpoint_step",
        "checkpoint_id",
        "all_ranks_accessible",
        "available",
        "reason",
    }
    missing = fields - set(mapping)
    unknown = set(mapping) - fields
    if missing:
        raise ProtocolValidationError(f"{path}: missing fields {sorted(missing)!r}")
    if unknown:
        raise ProtocolValidationError(f"{path}: unknown fields {sorted(unknown)!r}")
    normalized = {
        "failure_kind": _string(mapping["failure_kind"], f"{path}.failure_kind"),
        "recovery_mode": _choice(
            mapping["recovery_mode"],
            f"{path}.recovery_mode",
            {"latest", "recovery_verified"},
        ),
        "checkpoint_source": _choice(
            mapping["checkpoint_source"],
            f"{path}.checkpoint_source",
            {"gemini", "durable", "none"},
        ),
        "checkpoint_step": _integer(
            mapping["checkpoint_step"],
            f"{path}.checkpoint_step",
            minimum=-1,
        ),
        "checkpoint_id": _optional_string(
            mapping["checkpoint_id"],
            f"{path}.checkpoint_id",
        ),
        "all_ranks_accessible": _boolean(
            mapping["all_ranks_accessible"],
            f"{path}.all_ranks_accessible",
        ),
        "available": _boolean(mapping["available"], f"{path}.available"),
        "reason": _string(mapping["reason"], f"{path}.reason"),
    }
    available = normalized["available"]
    checkpoint_step = normalized["checkpoint_step"]
    checkpoint_source = normalized["checkpoint_source"]
    checkpoint_id = normalized["checkpoint_id"]
    failure_kind = normalized["failure_kind"]
    recovery_mode = normalized["recovery_mode"]
    all_ranks_accessible = normalized["all_ranks_accessible"]
    if (
        not all_ranks_accessible or failure_kind in {"sdc", "machine_unavailable"}
    ) and recovery_mode != "recovery_verified":
        raise ProtocolValidationError(
            f"{path}.recovery_mode: unsafe failures require recovery_verified"
        )
    if available:
        if checkpoint_step <= 0 or checkpoint_source == "none":
            raise ProtocolValidationError(
                f"{path}: available decisions require a positive step and source"
            )
    elif checkpoint_source != "none" or checkpoint_step > 0 or checkpoint_id is not None:
        raise ProtocolValidationError(f"{path}: unavailable decisions must not select a checkpoint")
    if checkpoint_source == "durable" and checkpoint_id is None:
        raise ProtocolValidationError(f"{path}: durable decisions require checkpoint_id")
    if checkpoint_source != "durable" and checkpoint_id is not None:
        raise ProtocolValidationError(f"{path}: checkpoint_id is only valid for durable decisions")
    frozen = _freeze_json(normalized, path)
    assert isinstance(frozen, Mapping)
    return frozen


@dataclass(frozen=True, slots=True)
class RecoveryProposalEvent(_WireRecord):
    event_id: str
    incident_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    decision: Mapping[str, Any]

    def __post_init__(self) -> None:
        _string(self.event_id, "RecoveryProposalEvent.event_id")
        _string(self.incident_id, "RecoveryProposalEvent.incident_id")
        _string(self.run_id, "RecoveryProposalEvent.run_id")
        _integer(self.generation, "RecoveryProposalEvent.generation", minimum=0)
        if not isinstance(self.reporter, WorkerIdentity):
            raise ProtocolValidationError("RecoveryProposalEvent.reporter: expected WorkerIdentity")
        if self.reporter.run_id != self.run_id:
            raise ProtocolValidationError(
                "RecoveryProposalEvent: reporter run_id does not match event"
            )
        if self.reporter.generation != self.generation:
            raise ProtocolValidationError(
                "RecoveryProposalEvent: reporter generation does not match event"
            )
        object.__setattr__(
            self,
            "decision",
            _recovery_decision(self.decision, "RecoveryProposalEvent.decision"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "event_id": self.event_id,
            "incident_id": self.incident_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "reporter": self.reporter.to_dict(),
            "decision": _thaw_json(self.decision),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecoveryProposalEvent:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "event_id",
                "incident_id",
                "run_id",
                "generation",
                "reporter",
                "decision",
            },
        )
        return cls(
            event_id=_string(
                value["event_id"],
                "RecoveryProposalEvent.event_id",
            ),
            incident_id=_string(
                value["incident_id"],
                "RecoveryProposalEvent.incident_id",
            ),
            run_id=_string(value["run_id"], "RecoveryProposalEvent.run_id"),
            generation=_integer(
                value["generation"],
                "RecoveryProposalEvent.generation",
                minimum=0,
            ),
            reporter=WorkerIdentity.from_dict(
                _require_mapping(
                    value["reporter"],
                    "RecoveryProposalEvent.reporter",
                )
            ),
            decision=_require_mapping(
                value["decision"],
                "RecoveryProposalEvent.decision",
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointCopy:
    owner_global_rank: int
    checkpoint_step: int
    inventory_event_id: str
    checkpoint_id: str | None
    holder_node_id: str
    holder_kind: str
    storage_kind: str
    location_token: str
    complete: bool
    checksums_available: bool

    def __post_init__(self) -> None:
        _integer(self.owner_global_rank, "CheckpointCopy.owner_global_rank", minimum=0)
        _integer(self.checkpoint_step, "CheckpointCopy.checkpoint_step", minimum=1)
        _string(self.inventory_event_id, "CheckpointCopy.inventory_event_id")
        _optional_string(self.checkpoint_id, "CheckpointCopy.checkpoint_id")
        _string(self.holder_node_id, "CheckpointCopy.holder_node_id")
        _choice(
            self.holder_kind,
            "CheckpointCopy.holder_kind",
            {"owner", "peer", "durable"},
        )
        _choice(
            self.storage_kind,
            "CheckpointCopy.storage_kind",
            {"memory", "node_local", "shared", "remote"},
        )
        if self.holder_kind == "durable" and self.storage_kind not in {
            "shared",
            "remote",
        }:
            raise ProtocolValidationError(
                "CheckpointCopy.storage_kind: durable copies must use shared or remote storage"
            )
        if self.holder_kind == "durable" and self.checkpoint_id is None:
            raise ProtocolValidationError(
                "CheckpointCopy.checkpoint_id: durable copies require a checkpoint ID"
            )
        if self.holder_kind != "durable" and self.checkpoint_id is not None:
            raise ProtocolValidationError(
                "CheckpointCopy.checkpoint_id: only durable copies may carry a checkpoint ID"
            )
        _string(self.location_token, "CheckpointCopy.location_token")
        _boolean(self.complete, "CheckpointCopy.complete")
        _boolean(self.checksums_available, "CheckpointCopy.checksums_available")

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_global_rank": self.owner_global_rank,
            "checkpoint_step": self.checkpoint_step,
            "inventory_event_id": self.inventory_event_id,
            "checkpoint_id": self.checkpoint_id,
            "holder_node_id": self.holder_node_id,
            "holder_kind": self.holder_kind,
            "storage_kind": self.storage_kind,
            "location_token": self.location_token,
            "complete": self.complete,
            "checksums_available": self.checksums_available,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointCopy:
        required = {
            "owner_global_rank",
            "checkpoint_step",
            "inventory_event_id",
            "checkpoint_id",
            "holder_node_id",
            "holder_kind",
            "storage_kind",
            "location_token",
            "complete",
            "checksums_available",
        }
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ProtocolValidationError(f"CheckpointCopy: missing fields {sorted(missing)!r}")
        if unknown:
            raise ProtocolValidationError(f"CheckpointCopy: unknown fields {sorted(unknown)!r}")
        return cls(
            owner_global_rank=_integer(
                value["owner_global_rank"],
                "CheckpointCopy.owner_global_rank",
                minimum=0,
            ),
            checkpoint_step=_integer(
                value["checkpoint_step"],
                "CheckpointCopy.checkpoint_step",
                minimum=1,
            ),
            inventory_event_id=_string(
                value["inventory_event_id"],
                "CheckpointCopy.inventory_event_id",
            ),
            checkpoint_id=_optional_string(
                value["checkpoint_id"],
                "CheckpointCopy.checkpoint_id",
            ),
            holder_node_id=_string(
                value["holder_node_id"],
                "CheckpointCopy.holder_node_id",
            ),
            holder_kind=_choice(
                value["holder_kind"],
                "CheckpointCopy.holder_kind",
                {"owner", "peer", "durable"},
            ),
            storage_kind=_choice(
                value["storage_kind"],
                "CheckpointCopy.storage_kind",
                {"memory", "node_local", "shared", "remote"},
            ),
            location_token=_string(
                value["location_token"],
                "CheckpointCopy.location_token",
            ),
            complete=_boolean(value["complete"], "CheckpointCopy.complete"),
            checksums_available=_boolean(
                value["checksums_available"],
                "CheckpointCopy.checksums_available",
            ),
        )


def _checkpoint_copies(value: object, path: str) -> tuple[CheckpointCopy, ...]:
    copies = tuple(
        item
        if isinstance(item, CheckpointCopy)
        else CheckpointCopy.from_dict(_require_mapping(item, f"{path}[{index}]"))
        for index, item in enumerate(_sequence(value, path))
    )
    identities = [
        (
            copy.owner_global_rank,
            copy.checkpoint_step,
            copy.inventory_event_id,
            copy.checkpoint_id,
            copy.holder_node_id,
            copy.holder_kind,
            copy.storage_kind,
            copy.location_token,
        )
        for copy in copies
    ]
    if len(identities) != len(set(identities)):
        raise ProtocolValidationError(f"{path}: duplicate checkpoint copies")
    return copies


@dataclass(frozen=True, slots=True)
class CheckpointInventoryEvent(_WireRecord):
    event_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    step: int
    trust: str
    topology_digest: str
    copies: tuple[CheckpointCopy, ...]

    def __post_init__(self) -> None:
        _string(self.event_id, "CheckpointInventoryEvent.event_id")
        _string(self.run_id, "CheckpointInventoryEvent.run_id")
        _integer(self.generation, "CheckpointInventoryEvent.generation", minimum=0)
        if not isinstance(self.reporter, WorkerIdentity):
            raise ProtocolValidationError(
                "CheckpointInventoryEvent.reporter: expected WorkerIdentity"
            )
        if self.reporter.run_id != self.run_id:
            raise ProtocolValidationError(
                "CheckpointInventoryEvent: reporter run_id does not match event"
            )
        if self.reporter.generation != self.generation:
            raise ProtocolValidationError(
                "CheckpointInventoryEvent: reporter generation does not match event"
            )
        _integer(self.step, "CheckpointInventoryEvent.step", minimum=1)
        _choice(
            self.trust,
            "CheckpointInventoryEvent.trust",
            {"latest", "candidate", "recovery_verified"},
        )
        _string(self.topology_digest, "CheckpointInventoryEvent.topology_digest")
        if self.topology_digest != self.reporter.topology_digest:
            raise ProtocolValidationError(
                "CheckpointInventoryEvent: reporter topology digest does not match event"
            )
        copies = _checkpoint_copies(
            self.copies,
            "CheckpointInventoryEvent.copies",
        )
        mismatched_steps = sorted(
            {copy.checkpoint_step for copy in copies if copy.checkpoint_step != self.step}
        )
        if mismatched_steps:
            raise ProtocolValidationError(
                "CheckpointInventoryEvent.copies: copy steps do not match event step"
            )
        mismatched_event_ids = sorted(
            {copy.inventory_event_id for copy in copies if copy.inventory_event_id != self.event_id}
        )
        if mismatched_event_ids:
            raise ProtocolValidationError(
                "CheckpointInventoryEvent.copies: provenance does not match event ID"
            )
        object.__setattr__(self, "copies", copies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "event_id": self.event_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "reporter": self.reporter.to_dict(),
            "step": self.step,
            "trust": self.trust,
            "topology_digest": self.topology_digest,
            "copies": [copy.to_dict() for copy in self.copies],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointInventoryEvent:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "event_id",
                "run_id",
                "generation",
                "reporter",
                "step",
                "trust",
                "topology_digest",
                "copies",
            },
        )
        return cls(
            event_id=_string(
                value["event_id"],
                "CheckpointInventoryEvent.event_id",
            ),
            run_id=_string(value["run_id"], "CheckpointInventoryEvent.run_id"),
            generation=_integer(
                value["generation"],
                "CheckpointInventoryEvent.generation",
                minimum=0,
            ),
            reporter=WorkerIdentity.from_dict(
                _require_mapping(
                    value["reporter"],
                    "CheckpointInventoryEvent.reporter",
                )
            ),
            step=_integer(
                value["step"],
                "CheckpointInventoryEvent.step",
                minimum=1,
            ),
            trust=_choice(
                value["trust"],
                "CheckpointInventoryEvent.trust",
                {"latest", "candidate", "recovery_verified"},
            ),
            topology_digest=_string(
                value["topology_digest"],
                "CheckpointInventoryEvent.topology_digest",
            ),
            copies=_checkpoint_copies(
                value["copies"],
                "CheckpointInventoryEvent.copies",
            ),
        )


@dataclass(frozen=True, slots=True)
class CheckpointCertification(_WireRecord):
    certification_id: str
    run_id: str
    source_generation: int
    step: int
    topology_digest: str
    checkpoint_source: str
    checkpoint_id: str | None
    expected_world_size: int
    certification_kind: str
    inventory_event_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _string(self.certification_id, "CheckpointCertification.certification_id")
        _string(self.run_id, "CheckpointCertification.run_id")
        _integer(
            self.source_generation,
            "CheckpointCertification.source_generation",
            minimum=0,
        )
        _integer(self.step, "CheckpointCertification.step", minimum=1)
        _string(self.topology_digest, "CheckpointCertification.topology_digest")
        _choice(
            self.checkpoint_source,
            "CheckpointCertification.checkpoint_source",
            {"gemini", "durable"},
        )
        _optional_string(self.checkpoint_id, "CheckpointCertification.checkpoint_id")
        if self.checkpoint_source == "durable" and self.checkpoint_id is None:
            raise ProtocolValidationError(
                "CheckpointCertification.checkpoint_id: durable certification requires an ID"
            )
        if self.checkpoint_source == "gemini" and self.checkpoint_id is not None:
            raise ProtocolValidationError(
                "CheckpointCertification.checkpoint_id: GEMINI certification must not set an ID"
            )
        _integer(
            self.expected_world_size,
            "CheckpointCertification.expected_world_size",
            minimum=1,
        )
        _choice(
            self.certification_kind,
            "CheckpointCertification.certification_kind",
            {"dense_consensus", "dynamic_candidate_promotion"},
        )
        event_ids = _strings(
            self.inventory_event_ids,
            "CheckpointCertification.inventory_event_ids",
            unique=True,
        )
        if not event_ids:
            raise ProtocolValidationError(
                "CheckpointCertification.inventory_event_ids: at least one event is required"
            )
        object.__setattr__(self, "inventory_event_ids", event_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "certification_id": self.certification_id,
            "run_id": self.run_id,
            "source_generation": self.source_generation,
            "step": self.step,
            "topology_digest": self.topology_digest,
            "checkpoint_source": self.checkpoint_source,
            "checkpoint_id": self.checkpoint_id,
            "expected_world_size": self.expected_world_size,
            "certification_kind": self.certification_kind,
            "inventory_event_ids": list(self.inventory_event_ids),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CheckpointCertification:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "certification_id",
                "run_id",
                "source_generation",
                "step",
                "topology_digest",
                "checkpoint_source",
                "checkpoint_id",
                "expected_world_size",
                "certification_kind",
                "inventory_event_ids",
            },
        )
        return cls(
            certification_id=_string(
                value["certification_id"],
                "CheckpointCertification.certification_id",
            ),
            run_id=_string(value["run_id"], "CheckpointCertification.run_id"),
            source_generation=_integer(
                value["source_generation"],
                "CheckpointCertification.source_generation",
                minimum=0,
            ),
            step=_integer(
                value["step"],
                "CheckpointCertification.step",
                minimum=1,
            ),
            topology_digest=_string(
                value["topology_digest"],
                "CheckpointCertification.topology_digest",
            ),
            checkpoint_source=_choice(
                value["checkpoint_source"],
                "CheckpointCertification.checkpoint_source",
                {"gemini", "durable"},
            ),
            checkpoint_id=_optional_string(
                value["checkpoint_id"],
                "CheckpointCertification.checkpoint_id",
            ),
            expected_world_size=_integer(
                value["expected_world_size"],
                "CheckpointCertification.expected_world_size",
                minimum=1,
            ),
            certification_kind=_choice(
                value["certification_kind"],
                "CheckpointCertification.certification_kind",
                {"dense_consensus", "dynamic_candidate_promotion"},
            ),
            inventory_event_ids=_strings(
                value["inventory_event_ids"],
                "CheckpointCertification.inventory_event_ids",
                unique=True,
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartIntent(_WireRecord):
    intent_id: str
    run_id: str
    generation: int
    incident_ids: tuple[str, ...]
    reason_code: str
    minimum_recovery_mode: str
    suspected_node_ids: tuple[str, ...]
    prepare_deadline_unix_ms: int

    def __post_init__(self) -> None:
        _string(self.intent_id, "RestartIntent.intent_id")
        _string(self.run_id, "RestartIntent.run_id")
        _integer(self.generation, "RestartIntent.generation", minimum=0)
        object.__setattr__(
            self,
            "incident_ids",
            _strings(self.incident_ids, "RestartIntent.incident_ids", unique=True),
        )
        if not self.incident_ids:
            raise ProtocolValidationError(
                "RestartIntent.incident_ids: at least one incident is required"
            )
        _string(self.reason_code, "RestartIntent.reason_code")
        _choice(
            self.minimum_recovery_mode,
            "RestartIntent.minimum_recovery_mode",
            {"latest", "recovery_verified"},
        )
        object.__setattr__(
            self,
            "suspected_node_ids",
            _strings(
                self.suspected_node_ids,
                "RestartIntent.suspected_node_ids",
                unique=True,
            ),
        )
        _integer(
            self.prepare_deadline_unix_ms,
            "RestartIntent.prepare_deadline_unix_ms",
            minimum=1,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "run_id": self.run_id,
            "generation": self.generation,
            "incident_ids": list(self.incident_ids),
            "reason_code": self.reason_code,
            "minimum_recovery_mode": self.minimum_recovery_mode,
            "suspected_node_ids": list(self.suspected_node_ids),
            "prepare_deadline_unix_ms": self.prepare_deadline_unix_ms,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RestartIntent:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "intent_id",
                "run_id",
                "generation",
                "incident_ids",
                "reason_code",
                "minimum_recovery_mode",
                "suspected_node_ids",
                "prepare_deadline_unix_ms",
            },
        )
        return cls(
            intent_id=_string(value["intent_id"], "RestartIntent.intent_id"),
            run_id=_string(value["run_id"], "RestartIntent.run_id"),
            generation=_integer(
                value["generation"],
                "RestartIntent.generation",
                minimum=0,
            ),
            incident_ids=_strings(
                value["incident_ids"],
                "RestartIntent.incident_ids",
                unique=True,
            ),
            reason_code=_string(
                value["reason_code"],
                "RestartIntent.reason_code",
            ),
            minimum_recovery_mode=_choice(
                value["minimum_recovery_mode"],
                "RestartIntent.minimum_recovery_mode",
                {"latest", "recovery_verified"},
            ),
            suspected_node_ids=_strings(
                value["suspected_node_ids"],
                "RestartIntent.suspected_node_ids",
                unique=True,
            ),
            prepare_deadline_unix_ms=_integer(
                value["prepare_deadline_unix_ms"],
                "RestartIntent.prepare_deadline_unix_ms",
                minimum=1,
            ),
        )


@dataclass(frozen=True, slots=True)
class RestartAck(_WireRecord):
    intent_id: str
    run_id: str
    node_id: str
    agent_id: str
    generation: int
    flushed_step: int
    inventory_event_ids: tuple[str, ...]
    transferred_owner_ranks: tuple[int, ...]
    transferred_peer_ranks: tuple[int, ...]
    success: bool
    reason: str

    def __post_init__(self) -> None:
        _string(self.intent_id, "RestartAck.intent_id")
        _string(self.run_id, "RestartAck.run_id")
        _string(self.node_id, "RestartAck.node_id")
        _string(self.agent_id, "RestartAck.agent_id")
        _integer(self.generation, "RestartAck.generation", minimum=0)
        _integer(self.flushed_step, "RestartAck.flushed_step", minimum=-1)
        object.__setattr__(
            self,
            "inventory_event_ids",
            _strings(
                self.inventory_event_ids,
                "RestartAck.inventory_event_ids",
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "transferred_owner_ranks",
            _integers(
                self.transferred_owner_ranks,
                "RestartAck.transferred_owner_ranks",
                minimum=0,
                unique=True,
            ),
        )
        object.__setattr__(
            self,
            "transferred_peer_ranks",
            _integers(
                self.transferred_peer_ranks,
                "RestartAck.transferred_peer_ranks",
                minimum=0,
                unique=True,
            ),
        )
        _boolean(self.success, "RestartAck.success")
        _string(self.reason, "RestartAck.reason")
        if self.success and self.flushed_step < 0:
            raise ProtocolValidationError(
                "RestartAck.flushed_step: successful acknowledgements require a step"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "intent_id": self.intent_id,
            "run_id": self.run_id,
            "node_id": self.node_id,
            "agent_id": self.agent_id,
            "generation": self.generation,
            "flushed_step": self.flushed_step,
            "inventory_event_ids": list(self.inventory_event_ids),
            "transferred_owner_ranks": list(self.transferred_owner_ranks),
            "transferred_peer_ranks": list(self.transferred_peer_ranks),
            "success": self.success,
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RestartAck:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "intent_id",
                "run_id",
                "node_id",
                "agent_id",
                "generation",
                "flushed_step",
                "inventory_event_ids",
                "transferred_owner_ranks",
                "transferred_peer_ranks",
                "success",
                "reason",
            },
        )
        return cls(
            intent_id=_string(value["intent_id"], "RestartAck.intent_id"),
            run_id=_string(value["run_id"], "RestartAck.run_id"),
            node_id=_string(value["node_id"], "RestartAck.node_id"),
            agent_id=_string(value["agent_id"], "RestartAck.agent_id"),
            generation=_integer(
                value["generation"],
                "RestartAck.generation",
                minimum=0,
            ),
            flushed_step=_integer(
                value["flushed_step"],
                "RestartAck.flushed_step",
                minimum=-1,
            ),
            inventory_event_ids=_strings(
                value["inventory_event_ids"],
                "RestartAck.inventory_event_ids",
                unique=True,
            ),
            transferred_owner_ranks=_integers(
                value["transferred_owner_ranks"],
                "RestartAck.transferred_owner_ranks",
                minimum=0,
                unique=True,
            ),
            transferred_peer_ranks=_integers(
                value["transferred_peer_ranks"],
                "RestartAck.transferred_peer_ranks",
                minimum=0,
                unique=True,
            ),
            success=_boolean(value["success"], "RestartAck.success"),
            reason=_string(value["reason"], "RestartAck.reason"),
        )


@dataclass(frozen=True, slots=True)
class RestartPlan(_WireRecord):
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
        object.__setattr__(
            self,
            "incident_ids",
            _strings(self.incident_ids, "RestartPlan.incident_ids", unique=True),
        )
        if not self.incident_ids:
            raise ProtocolValidationError(
                "RestartPlan.incident_ids: at least one incident is required"
            )
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
        _string(
            self.checkpoint_manifest_id,
            "RestartPlan.checkpoint_manifest_id",
        )
        assignments = _assignments(
            self.slot_assignments,
            "RestartPlan.slot_assignments",
        )
        object.__setattr__(self, "slot_assignments", assignments)
        object.__setattr__(
            self,
            "quarantined_node_ids",
            _strings(
                self.quarantined_node_ids,
                "RestartPlan.quarantined_node_ids",
                unique=True,
            ),
        )
        assigned_nodes = {assignment.node_id for assignment in assignments}
        overlap = assigned_nodes & set(self.quarantined_node_ids)
        if overlap:
            raise ProtocolValidationError(
                f"RestartPlan: quarantined nodes cannot be assigned: {sorted(overlap)!r}"
            )
        _integer(
            self.expected_world_size,
            "RestartPlan.expected_world_size",
            minimum=1,
        )
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
        _record_fields(
            value,
            path=cls.__name__,
            required={
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
            },
        )
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
        _integer(
            self.first_global_rank,
            "RestartContext.first_global_rank",
            minimum=0,
        )
        _integer(
            self.local_world_size,
            "RestartContext.local_world_size",
            minimum=1,
        )
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
        if self.expected_world_size % self.local_world_size != 0:
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
        _string(
            self.checkpoint_manifest_id,
            "RestartContext.checkpoint_manifest_id",
        )
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

    def validate_worker_environment(
        self,
        environment: Mapping[str, str],
        *,
        committed_plan: RestartPlan,
    ) -> None:
        if not isinstance(committed_plan, RestartPlan):
            raise ProtocolValidationError("committed_plan: expected RestartPlan")
        expected_context = RestartContext.from_plan(committed_plan, self.node_id)
        if self != expected_context:
            raise ProtocolValidationError(
                "RestartContext: does not match the currently committed restart plan"
            )
        required = {
            "RANK",
            "LOCAL_RANK",
            "LOCAL_WORLD_SIZE",
            "WORLD_SIZE",
            "TORCHELASTIC_RUN_ID",
        }
        missing = required - set(environment)
        if missing:
            raise ProtocolValidationError(
                f"RestartContext.environment: missing fields {sorted(missing)!r}"
            )
        local_rank = _environment_integer(
            environment["LOCAL_RANK"],
            "LOCAL_RANK",
            minimum=0,
        )
        if local_rank >= self.local_world_size:
            raise ProtocolValidationError(
                "RestartContext.environment: LOCAL_RANK exceeds local world size"
            )
        expected = {
            "RANK": self.first_global_rank + local_rank,
            "LOCAL_WORLD_SIZE": self.local_world_size,
            "WORLD_SIZE": self.expected_world_size,
        }
        for key, expected_value in expected.items():
            actual = _environment_integer(environment[key], key, minimum=0)
            if actual != expected_value:
                raise ProtocolValidationError(
                    f"RestartContext.environment: {key}={actual} does not match "
                    f"expected {expected_value}"
                )
        if environment["TORCHELASTIC_RUN_ID"] != self.run_id:
            raise ProtocolValidationError(
                "RestartContext.environment: TORCHELASTIC_RUN_ID does not match run_id"
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
        _record_fields(
            value,
            path=cls.__name__,
            required={
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
            },
        )
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
            reason_code=_string(
                value["reason_code"],
                "RestartContext.reason_code",
            ),
        )


def _environment_integer(value: object, name: str, *, minimum: int) -> int:
    if not isinstance(value, str):
        raise ProtocolValidationError(f"RestartContext.environment: {name} must be a string")
    try:
        parsed = int(value)
    except ValueError as error:
        raise ProtocolValidationError(
            f"RestartContext.environment: {name} must be an integer"
        ) from error
    return _integer(parsed, f"RestartContext.environment.{name}", minimum=minimum)


@dataclass(frozen=True, slots=True)
class RankCheckpointCopies:
    owner_global_rank: int
    copies: tuple[CheckpointCopy, ...]

    def __post_init__(self) -> None:
        _integer(
            self.owner_global_rank,
            "RankCheckpointCopies.owner_global_rank",
            minimum=0,
        )
        copies = _checkpoint_copies(self.copies, "RankCheckpointCopies.copies")
        for copy in copies:
            if copy.owner_global_rank != self.owner_global_rank:
                raise ProtocolValidationError(
                    "RankCheckpointCopies: copy owner does not match entry owner"
                )
        object.__setattr__(self, "copies", copies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "owner_global_rank": self.owner_global_rank,
            "copies": [copy.to_dict() for copy in self.copies],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RankCheckpointCopies:
        required = {"owner_global_rank", "copies"}
        missing = required - set(value)
        unknown = set(value) - required
        if missing:
            raise ProtocolValidationError(
                f"RankCheckpointCopies: missing fields {sorted(missing)!r}"
            )
        if unknown:
            raise ProtocolValidationError(
                f"RankCheckpointCopies: unknown fields {sorted(unknown)!r}"
            )
        return cls(
            owner_global_rank=_integer(
                value["owner_global_rank"],
                "RankCheckpointCopies.owner_global_rank",
                minimum=0,
            ),
            copies=_checkpoint_copies(
                value["copies"],
                "RankCheckpointCopies.copies",
            ),
        )


def _rank_checkpoint_copies(
    value: object,
    path: str,
) -> tuple[RankCheckpointCopies, ...]:
    entries = tuple(
        item
        if isinstance(item, RankCheckpointCopies)
        else RankCheckpointCopies.from_dict(_require_mapping(item, f"{path}[{index}]"))
        for index, item in enumerate(_sequence(value, path))
    )
    owner_ranks = [entry.owner_global_rank for entry in entries]
    if len(owner_ranks) != len(set(owner_ranks)):
        raise ProtocolValidationError(f"{path}: owner ranks must be unique")
    return tuple(sorted(entries, key=lambda item: item.owner_global_rank))


@dataclass(frozen=True, slots=True)
class RecoveryManifest(_WireRecord):
    manifest_id: str
    run_id: str
    source_generation: int
    step: int
    trust: str
    topology_digest: str
    rank_copies: tuple[RankCheckpointCopies, ...]

    def __post_init__(self) -> None:
        _string(self.manifest_id, "RecoveryManifest.manifest_id")
        _string(self.run_id, "RecoveryManifest.run_id")
        _integer(
            self.source_generation,
            "RecoveryManifest.source_generation",
            minimum=0,
        )
        _integer(self.step, "RecoveryManifest.step", minimum=1)
        _choice(
            self.trust,
            "RecoveryManifest.trust",
            {"latest", "recovery_verified"},
        )
        _string(self.topology_digest, "RecoveryManifest.topology_digest")
        rank_copies = _rank_checkpoint_copies(
            self.rank_copies,
            "RecoveryManifest.rank_copies",
        )
        mismatched_steps = sorted(
            {
                copy.checkpoint_step
                for entry in rank_copies
                for copy in entry.copies
                if copy.checkpoint_step != self.step
            }
        )
        if mismatched_steps:
            raise ProtocolValidationError(
                "RecoveryManifest.rank_copies: copy steps do not match manifest step"
            )
        object.__setattr__(self, "rank_copies", rank_copies)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "run_id": self.run_id,
            "source_generation": self.source_generation,
            "step": self.step,
            "trust": self.trust,
            "topology_digest": self.topology_digest,
            "rank_copies": [entry.to_dict() for entry in self.rank_copies],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> RecoveryManifest:
        _record_fields(
            value,
            path=cls.__name__,
            required={
                "manifest_id",
                "run_id",
                "source_generation",
                "step",
                "trust",
                "topology_digest",
                "rank_copies",
            },
        )
        return cls(
            manifest_id=_string(
                value["manifest_id"],
                "RecoveryManifest.manifest_id",
            ),
            run_id=_string(value["run_id"], "RecoveryManifest.run_id"),
            source_generation=_integer(
                value["source_generation"],
                "RecoveryManifest.source_generation",
                minimum=0,
            ),
            step=_integer(value["step"], "RecoveryManifest.step", minimum=1),
            trust=_choice(
                value["trust"],
                "RecoveryManifest.trust",
                {"latest", "recovery_verified"},
            ),
            topology_digest=_string(
                value["topology_digest"],
                "RecoveryManifest.topology_digest",
            ),
            rank_copies=_rank_checkpoint_copies(
                value["rank_copies"],
                "RecoveryManifest.rank_copies",
            ),
        )


def validate_worker_identity(
    identity: WorkerIdentity,
    assignment: RankAssignment,
) -> None:
    """Bind a worker-supplied identity to one committed rank assignment."""
    if identity.run_id != assignment.run_id:
        raise ProtocolValidationError("WorkerIdentity.run_id: does not match committed assignment")
    if identity.generation != assignment.generation:
        raise ProtocolValidationError(
            "WorkerIdentity.generation: does not match committed assignment"
        )
    if identity.topology_digest != assignment.topology_digest:
        raise ProtocolValidationError(
            "WorkerIdentity.topology_digest: does not match committed assignment"
        )
    expected_node_id = assignment.slot_to_node_id.get(identity.logical_node_slot)
    if expected_node_id is None:
        raise ProtocolValidationError(
            "WorkerIdentity.logical_node_slot: is not active in committed assignment"
        )
    if identity.node_id != expected_node_id:
        raise ProtocolValidationError("WorkerIdentity.node_id: does not match committed assignment")
    if identity.local_world_size != assignment.local_world_size:
        raise ProtocolValidationError(
            "WorkerIdentity.local_world_size: does not match committed assignment"
        )
    first_rank, last_rank = assignment.slot_to_rank_range[identity.logical_node_slot]
    expected_global_rank = first_rank + identity.local_rank
    if identity.global_rank != expected_global_rank or identity.global_rank >= last_rank:
        raise ProtocolValidationError(
            "WorkerIdentity.global_rank: does not match committed rank range"
        )


def validate_event_reporter(
    event: FaultEvent | RecoveryProposalEvent | CheckpointInventoryEvent,
    assignment: RankAssignment,
    *,
    agent_identity: AgentIdentity,
    resource_to_node_id: Mapping[str, str],
) -> None:
    """Validate a report against committed rank and infrastructure identities."""
    if not isinstance(
        event,
        (FaultEvent, RecoveryProposalEvent, CheckpointInventoryEvent),
    ):
        raise ProtocolValidationError(
            "event: expected a fault, recovery, or checkpoint inventory event"
        )
    validate_worker_identity(event.reporter, assignment)
    if not isinstance(agent_identity, AgentIdentity):
        raise ProtocolValidationError("agent_identity: expected AgentIdentity")
    reporter = event.reporter
    if agent_identity.run_id != reporter.run_id:
        raise ProtocolValidationError("AgentIdentity.run_id: does not match event reporter")
    if agent_identity.node_id != reporter.node_id:
        raise ProtocolValidationError("AgentIdentity.node_id: does not match event reporter")
    if agent_identity.agent_id != reporter.agent_id:
        raise ProtocolValidationError("AgentIdentity.agent_id: does not match event reporter")
    if agent_identity.hostname != reporter.hostname:
        raise ProtocolValidationError("AgentIdentity.hostname: does not match event reporter")
    if agent_identity.local_world_size != reporter.local_world_size:
        raise ProtocolValidationError(
            "AgentIdentity.local_world_size: does not match event reporter"
        )
    if not isinstance(resource_to_node_id, Mapping):
        raise ProtocolValidationError("resource_to_node_id: expected an object")
    resource_owners = {
        _string(resource_id, "resource_to_node_id.key"): _string(
            node_id,
            f"resource_to_node_id[{resource_id!r}]",
        )
        for resource_id, node_id in resource_to_node_id.items()
    }
    if reporter.gpu_uuid is not None:
        if reporter.gpu_uuid not in agent_identity.resource_ids:
            raise ProtocolValidationError(
                "WorkerIdentity.gpu_uuid: is not registered to the reporting agent"
            )
        if resource_owners.get(reporter.gpu_uuid) != reporter.node_id:
            raise ProtocolValidationError(
                "WorkerIdentity.gpu_uuid: trusted resource owner does not match reporter"
            )
    if isinstance(event, FaultEvent) and "failed_ranks" in event.report:
        reported_ranks = _integers(
            event.report["failed_ranks"],
            "FaultEvent.report.failed_ranks",
            minimum=0,
            unique=True,
        )
        assigned_ranks = {
            rank
            for first_rank, last_rank in assignment.slot_to_rank_range.values()
            for rank in range(first_rank, last_rank)
        }
        unknown_ranks = sorted(set(reported_ranks) - assigned_ranks)
        if unknown_ranks:
            raise ProtocolValidationError(
                "FaultEvent.report.failed_ranks: ranks are not active in the committed "
                f"assignment: {unknown_ranks!r}"
            )
    if not isinstance(event, FaultEvent) or event.report.get("kind") != "hardware":
        return
    resource_kind = _choice(
        event.report.get("resource_kind"),
        "FaultEvent.report.resource_kind",
        {"gpu", "node", "nic", "hca", "link"},
    )
    resource_id = _string(
        event.report.get("resource_id"),
        "FaultEvent.report.resource_id",
    )
    if resource_kind == "node":
        if resource_id != reporter.node_id:
            raise ProtocolValidationError(
                "FaultEvent.report.resource_id: node fault does not identify the reporter's node"
            )
    elif resource_id not in agent_identity.resource_ids:
        raise ProtocolValidationError(
            "FaultEvent.report.resource_id: resource is not registered to the reporting agent"
        )
    if resource_owners.get(resource_id) != reporter.node_id:
        raise ProtocolValidationError(
            "FaultEvent.report.resource_id: trusted resource owner does not match reporter"
        )


def validate_restart_plan(
    plan: RestartPlan,
    intent: RestartIntent,
    manifest: RecoveryManifest,
    *,
    inventory_events: Sequence[CheckpointInventoryEvent],
    trusted_certifications: Sequence[CheckpointCertification],
    restart_acks: Sequence[RestartAck],
    authenticated_ack_agent_ids: Mapping[str, str],
    current_assignment: RankAssignment,
    now_unix_ms: int,
    eligible_node_ids: Sequence[str],
    quarantined_node_ids: Sequence[str] = (),
) -> None:
    """Validate that one immutable plan is safe to expose through rendezvous."""
    _integer(now_unix_ms, "now_unix_ms", minimum=1)
    if plan.restart_deadline_unix_ms <= now_unix_ms:
        raise ProtocolValidationError("RestartPlan.restart_deadline_unix_ms: deadline has elapsed")
    if plan.intent_id != intent.intent_id:
        raise ProtocolValidationError("RestartPlan.intent_id: does not match restart intent")
    if plan.run_id != intent.run_id or plan.run_id != current_assignment.run_id:
        raise ProtocolValidationError(
            "RestartPlan.run_id: does not match intent and committed assignment"
        )
    if (
        plan.from_generation != intent.generation
        or plan.from_generation != current_assignment.generation
    ):
        raise ProtocolValidationError(
            "RestartPlan.from_generation: does not match intent and committed assignment"
        )
    if plan.incident_ids != intent.incident_ids:
        raise ProtocolValidationError("RestartPlan.incident_ids: do not match restart intent")
    if plan.reason_code != intent.reason_code:
        raise ProtocolValidationError("RestartPlan.reason_code: does not match restart intent")
    if (
        intent.minimum_recovery_mode == "recovery_verified"
        and plan.recovery_mode != "recovery_verified"
    ):
        raise ProtocolValidationError(
            "RestartPlan.recovery_mode: weaker than restart intent minimum"
        )
    if len(plan.slot_assignments) != current_assignment.active_nodes:
        raise ProtocolValidationError(
            "RestartPlan.slot_assignments: active node count does not match committed assignment"
        )
    plan_local_world_size = plan.slot_assignments[0].local_world_size
    if plan_local_world_size != current_assignment.local_world_size:
        raise ProtocolValidationError(
            "RestartPlan.slot_assignments: local world size does not match committed assignment"
        )
    committed_world_size = current_assignment.active_nodes * current_assignment.local_world_size
    if plan.expected_world_size != committed_world_size:
        raise ProtocolValidationError(
            "RestartPlan.expected_world_size: does not match committed assignment"
        )
    if plan.topology_digest != current_assignment.topology_digest:
        raise ProtocolValidationError(
            "RestartPlan.topology_digest: does not match the committed topology"
        )
    current_nodes = set(current_assignment.slot_to_node_id.values())
    assigned_nodes = {assignment.node_id for assignment in plan.slot_assignments}
    suspected_nodes = set(intent.suspected_node_ids)
    unknown_suspects = sorted(suspected_nodes - current_nodes)
    if unknown_suspects:
        raise ProtocolValidationError(
            "RestartIntent.suspected_node_ids: nodes are not active in the committed "
            f"generation: {unknown_suspects!r}"
        )
    retained_suspects = sorted(suspected_nodes & assigned_nodes)
    if retained_suspects:
        raise ProtocolValidationError(
            f"RestartPlan.slot_assignments: suspected nodes remain assigned: {retained_suspects!r}"
        )
    current_slots_by_node = {
        node_id: slot for slot, node_id in current_assignment.slot_to_node_id.items()
    }
    planned_slots_by_node = {
        assignment.node_id: assignment.logical_node_slot for assignment in plan.slot_assignments
    }
    moved_survivors = sorted(
        node_id
        for node_id in current_nodes & assigned_nodes
        if current_slots_by_node[node_id] != planned_slots_by_node[node_id]
    )
    if moved_survivors:
        raise ProtocolValidationError(
            "RestartPlan.slot_assignments: surviving nodes changed logical slots: "
            f"{moved_survivors!r}"
        )
    if assigned_nodes == current_nodes:
        raise ProtocolValidationError(
            "RestartPlan.slot_assignments: version 1 requires at least one replacement node"
        )
    eligible = set(_strings(eligible_node_ids, "eligible_node_ids", unique=True))
    quarantined = set(_strings(quarantined_node_ids, "quarantined_node_ids", unique=True)) | set(
        plan.quarantined_node_ids
    )
    unavailable = assigned_nodes - eligible
    if unavailable:
        raise ProtocolValidationError(
            f"RestartPlan: assigned nodes are not eligible: {sorted(unavailable)!r}"
        )
    unsafe = assigned_nodes & quarantined
    if unsafe:
        raise ProtocolValidationError(
            f"RestartPlan: assigned nodes are quarantined: {sorted(unsafe)!r}"
        )
    if manifest.manifest_id != plan.checkpoint_manifest_id:
        raise ProtocolValidationError("RecoveryManifest.manifest_id: does not match restart plan")
    if manifest.run_id != plan.run_id:
        raise ProtocolValidationError("RecoveryManifest.run_id: does not match restart plan")
    if manifest.source_generation > plan.from_generation:
        raise ProtocolValidationError(
            "RecoveryManifest.source_generation: cannot be newer than the failed generation"
        )
    if manifest.step != plan.checkpoint_step:
        raise ProtocolValidationError("RecoveryManifest.step: does not match restart plan")
    if manifest.topology_digest != plan.topology_digest:
        raise ProtocolValidationError(
            "RecoveryManifest.topology_digest: does not match restart plan"
        )
    if plan.recovery_mode == "recovery_verified" and manifest.trust != "recovery_verified":
        raise ProtocolValidationError(
            "RecoveryManifest.trust: verified recovery requires a verified manifest"
        )
    if plan.checkpoint_source == "durable" and manifest.trust != "recovery_verified":
        raise ProtocolValidationError(
            "RecoveryManifest.trust: durable recovery requires a verified manifest"
        )
    inventory_by_id: dict[str, CheckpointInventoryEvent] = {}
    for index, event in enumerate(_sequence(inventory_events, "inventory_events")):
        if not isinstance(event, CheckpointInventoryEvent):
            raise ProtocolValidationError(
                f"inventory_events[{index}]: expected CheckpointInventoryEvent"
            )
        if event.event_id in inventory_by_id:
            raise ProtocolValidationError(
                f"inventory_events: duplicate event ID {event.event_id!r}"
            )
        inventory_by_id[event.event_id] = event
    certification_ids: set[str] = set()
    certified_inventory_event_ids: set[str] = set()
    for index, certification in enumerate(
        _sequence(trusted_certifications, "trusted_certifications")
    ):
        if not isinstance(certification, CheckpointCertification):
            raise ProtocolValidationError(
                f"trusted_certifications[{index}]: expected CheckpointCertification"
            )
        if certification.certification_id in certification_ids:
            raise ProtocolValidationError(
                "trusted_certifications: duplicate certification ID "
                f"{certification.certification_id!r}"
            )
        certification_ids.add(certification.certification_id)
        if (
            certification.run_id == manifest.run_id
            and certification.source_generation == manifest.source_generation
            and certification.step == manifest.step
            and certification.topology_digest == manifest.topology_digest
            and certification.checkpoint_source == plan.checkpoint_source
            and certification.checkpoint_id == plan.checkpoint_id
            and certification.expected_world_size == plan.expected_world_size
        ):
            certified_inventory_event_ids.update(certification.inventory_event_ids)
    if not isinstance(authenticated_ack_agent_ids, Mapping):
        raise ProtocolValidationError("authenticated_ack_agent_ids: expected an object")
    authenticated_ack_agents = {
        _string(node_id, "authenticated_ack_agent_ids.key"): _string(
            agent_id,
            f"authenticated_ack_agent_ids[{node_id!r}]",
        )
        for node_id, agent_id in authenticated_ack_agent_ids.items()
    }
    ack_by_node: dict[str, RestartAck] = {}
    for index, ack in enumerate(_sequence(restart_acks, "restart_acks")):
        if not isinstance(ack, RestartAck):
            raise ProtocolValidationError(f"restart_acks[{index}]: expected RestartAck")
        if ack.node_id in ack_by_node:
            raise ProtocolValidationError(f"restart_acks: duplicate node ID {ack.node_id!r}")
        if ack.intent_id != intent.intent_id:
            raise ProtocolValidationError(
                f"restart_acks[{index}].intent_id: does not match restart intent"
            )
        if ack.run_id != intent.run_id:
            raise ProtocolValidationError(
                f"restart_acks[{index}].run_id: does not match restart intent"
            )
        if ack.generation != intent.generation:
            raise ProtocolValidationError(
                f"restart_acks[{index}].generation: does not match restart intent"
            )
        if ack.node_id not in current_nodes:
            raise ProtocolValidationError(
                f"restart_acks[{index}].node_id: is not active in the committed generation"
            )
        if authenticated_ack_agents.get(ack.node_id) != ack.agent_id:
            raise ProtocolValidationError(
                f"restart_acks[{index}].agent_id: does not match authenticated transport sender"
            )
        ack_by_node[ack.node_id] = ack
    if set(authenticated_ack_agents) != set(ack_by_node):
        raise ProtocolValidationError(
            "authenticated_ack_agent_ids: bindings must exactly match submitted acknowledgements"
        )
    entries = {entry.owner_global_rank: entry for entry in manifest.rank_copies}
    required_ranks = set(range(plan.expected_world_size))
    if set(entries) != required_ranks:
        missing = sorted(required_ranks - set(entries))
        extra = sorted(set(entries) - required_ranks)
        raise ProtocolValidationError(
            f"RecoveryManifest.rank_copies: rank coverage mismatch; "
            f"missing={missing!r}, extra={extra!r}"
        )
    for rank, entry in entries.items():
        compatible_holder_kinds = (
            {"durable"} if plan.checkpoint_source == "durable" else {"owner", "peer"}
        )
        eligible_copies = [
            copy
            for copy in entry.copies
            if copy.complete
            and copy.checkpoint_step == manifest.step
            and _copy_has_trusted_inventory_provenance(
                copy,
                manifest,
                inventory_by_id,
                ack_by_node,
                certified_inventory_event_ids,
            )
            and copy.holder_kind in compatible_holder_kinds
            and copy.checkpoint_id == plan.checkpoint_id
            and (copy.storage_kind in {"shared", "remote"} or copy.holder_node_id in assigned_nodes)
        ]
        if not eligible_copies:
            raise ProtocolValidationError(
                f"RecoveryManifest.rank_copies[{rank}]: no complete eligible copy"
            )


def _copy_has_trusted_inventory_provenance(
    copy: CheckpointCopy,
    manifest: RecoveryManifest,
    inventory_by_id: Mapping[str, CheckpointInventoryEvent],
    ack_by_node: Mapping[str, RestartAck],
    certified_inventory_event_ids: set[str],
) -> bool:
    event = inventory_by_id.get(copy.inventory_event_id)
    if event is None:
        return False
    if (
        event.run_id != manifest.run_id
        or event.generation != manifest.source_generation
        or event.step != manifest.step
        or event.topology_digest != manifest.topology_digest
        or event.trust == "candidate"
        or (manifest.trust == "recovery_verified" and event.trust != "recovery_verified")
    ):
        return False
    if copy not in event.copies:
        return False
    if (
        copy.storage_kind in {"memory", "node_local"}
        and copy.holder_node_id != event.reporter.node_id
    ):
        return False
    if manifest.trust != "latest":
        return event.event_id in certified_inventory_event_ids
    ack = ack_by_node.get(event.reporter.node_id)
    return (
        ack is not None
        and ack.success
        and ack.agent_id == event.reporter.agent_id
        and ack.flushed_step == manifest.step
        and event.event_id in ack.inventory_event_ids
    )


__all__ = [
    "AgentIdentity",
    "CheckpointCertification",
    "CheckpointCopy",
    "CheckpointInventoryEvent",
    "FaultEvent",
    "HardwareFaultReport",
    "ProtocolValidationError",
    "RankAssignment",
    "RankCheckpointCopies",
    "RecoveryManifest",
    "RecoveryProposalEvent",
    "RestartAck",
    "RestartContext",
    "RestartIntent",
    "RestartPlan",
    "SlotAssignment",
    "WorkerIdentity",
    "validate_restart_plan",
    "validate_event_reporter",
    "validate_worker_identity",
]
