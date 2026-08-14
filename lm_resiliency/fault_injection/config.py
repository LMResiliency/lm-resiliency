"""Versioned incident-oriented fault campaign configuration."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

from lm_resiliency.fault_injection._json import freeze_json_mapping, thaw_json

SCHEMA_VERSION = 1

_CAMPAIGN_KEYS = {
    "schema_version",
    "name",
    "seed",
    "clock",
    "incidents",
    "metadata",
}
_CLOCK_KEYS = {"type", "origin"}
_RANGE_KEYS = {"start", "end", "every"}
_TRIGGER_KEYS = {"at", "range", "probability"}
_LIFETIME_KEYS = {"matching_calls", "iterations", "until"}
_TARGET_KEYS = {
    "surface",
    "rank",
    "model_part",
    "component",
    "index",
    "module_path",
    "operation",
    "resource",
    "path",
    "metadata",
}
_FAULT_KEYS = {"fault_id", "id", "type", "target", "parameters"}
_INCIDENT_KEYS = {
    "incident_id",
    "id",
    "trigger",
    "lifetime",
    "faults",
    "retrigger",
    "max_occurrences",
}


class ClockType(str, Enum):
    """Progress clock used to schedule incidents."""

    TRAINING_ITERATION = "training_iteration"


class ClockOrigin(str, Enum):
    """Origin for training-iteration numbers."""

    TRAINING_RUN = "training_run"
    CAMPAIGN_START = "campaign_start"


class FailureType(str, Enum):
    """Canonical observable failure effects."""

    TENSOR_CORRUPTION = "tensor_corruption"
    STALE_STATE = "stale_state"
    DROP = "drop"
    DUPLICATE = "duplicate"
    REORDER = "reorder"
    DELAY = "delay"
    HANG = "hang"
    TIMEOUT = "timeout"
    EXCEPTION = "exception"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    PROCESS_TERMINATION = "process_termination"
    RESOURCE_UNAVAILABLE = "resource_unavailable"
    CHECKPOINT_CORRUPTION = "checkpoint_corruption"
    CHECKPOINT_TRUNCATION = "checkpoint_truncation"
    CHECKPOINT_MISSING = "checkpoint_missing"
    IO_ERROR = "io_error"
    PAYLOAD_CORRUPTION = "payload_corruption"
    COLLECTIVE_DESYNC = "collective_desync"
    MESSAGE_DROP = "message_drop"
    NETWORK_PARTITION = "network_partition"
    CONFIG_DRIFT = "config_drift"


class CorruptionOperation(str, Enum):
    """Numerical operation used by tensor and payload corruption."""

    SINGLE_BITFLIP = "single_bitflip"
    MULTI_BITFLIP = "multi_bitflip"
    SET_VALUE = "set_value"
    SCALE = "scale"
    NOISE = "noise"
    SIGN_FLIP = "sign_flip"


class FaultMagnitude(str, Enum):
    """Relative numerical corruption severity."""

    CATASTROPHIC = "catastrophic"
    LARGE = "large"
    MEDIUM = "medium"
    SUBTLE = "subtle"
    NEAR_INVISIBLE = "near_invisible"


class FaultScope(str, Enum):
    """Tensor elements affected by one operation."""

    SINGLE = "single"
    ROW = "row"
    PERCENT_1 = "1%"
    PERCENT_10 = "10%"
    FULL = "100%"


class FaultSurface(str, Enum):
    """Logical training surface affected by a fault."""

    INPUT = "input"
    OUTPUT = "output"
    WEIGHT = "weight"
    BIAS = "bias"
    GRADIENT = "gradient"
    OPTIMIZER_STATE = "optimizer_state"
    RNG_STATE = "rng_state"
    SAMPLER_STATE = "sampler_state"
    DATA = "data"
    CHECKPOINT = "checkpoint"
    COMPUTE = "compute"
    COLLECTIVE = "collective"
    PROCESS = "process"
    RESOURCE = "resource"
    CONFIG = "config"


class RetriggerPolicy(str, Enum):
    """Whether an occurrence may fire again after rollback."""

    ONCE = "once"
    EVERY_ATTEMPT = "every_attempt"
    MAX_OCCURRENCES = "max_occurrences"


class SafetyClass(str, Enum):
    """Minimum isolation required to execute a fault."""

    SAFE_IN_PROCESS = "safe_in_process"
    ISOLATED_DESTRUCTIVE = "isolated_destructive"
    CLUSTER_DESTRUCTIVE = "cluster_destructive"


_ISOLATED_DESTRUCTIVE = {
    FailureType.HANG,
    FailureType.TIMEOUT,
    FailureType.EXCEPTION,
    FailureType.RESOURCE_EXHAUSTION,
    FailureType.PROCESS_TERMINATION,
    FailureType.CHECKPOINT_CORRUPTION,
    FailureType.CHECKPOINT_TRUNCATION,
    FailureType.CHECKPOINT_MISSING,
    FailureType.IO_ERROR,
}
_CLUSTER_DESTRUCTIVE = {
    FailureType.RESOURCE_UNAVAILABLE,
    FailureType.COLLECTIVE_DESYNC,
    FailureType.MESSAGE_DROP,
    FailureType.NETWORK_PARTITION,
}
_STRAGGLER_FAILURES = {FailureType.DELAY, FailureType.TIMEOUT}
_HANG_FAILURES = {
    FailureType.HANG,
    FailureType.MESSAGE_DROP,
    FailureType.NETWORK_PARTITION,
}
_PROCESS_FAILURES = {
    FailureType.EXCEPTION,
    FailureType.RESOURCE_EXHAUSTION,
    FailureType.PROCESS_TERMINATION,
    FailureType.RESOURCE_UNAVAILABLE,
}


def minimum_safety(failure_type: FailureType) -> SafetyClass:
    """Return the minimum isolation required for a failure type."""
    if failure_type in _CLUSTER_DESTRUCTIVE:
        return SafetyClass.CLUSTER_DESTRUCTIVE
    if failure_type in _ISOLATED_DESTRUCTIVE:
        return SafetyClass.ISOLATED_DESTRUCTIVE
    return SafetyClass.SAFE_IN_PROCESS


def expected_failure_kind(failure_type: FailureType) -> str:
    """Map an injected effect to a neutral localization category."""
    if failure_type in _STRAGGLER_FAILURES:
        return "straggler"
    if failure_type in _HANG_FAILURES:
        return "hang"
    if failure_type in _PROCESS_FAILURES:
        return "process_failure"
    return "sdc"


@dataclass(frozen=True, slots=True)
class ClockSpec:
    """Campaign progress clock."""

    type: ClockType = ClockType.TRAINING_ITERATION
    origin: ClockOrigin = ClockOrigin.TRAINING_RUN

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", ClockType(self.type))
        object.__setattr__(self, "origin", ClockOrigin(self.origin))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ClockSpec":
        _reject_unknown_keys(value, _CLOCK_KEYS, "clock")
        return cls(
            type=ClockType(value.get("type", ClockType.TRAINING_ITERATION.value)),
            origin=ClockOrigin(value.get("origin", ClockOrigin.TRAINING_RUN.value)),
        )

    def to_dict(self) -> dict[str, str]:
        return {"type": self.type.value, "origin": self.origin.value}


@dataclass(frozen=True, slots=True)
class IterationRange:
    """Inclusive periodic training-iteration range."""

    start: int
    end: int
    every: int = 1

    def __post_init__(self) -> None:
        object.__setattr__(self, "start", _strict_int(self.start, "trigger range start"))
        object.__setattr__(self, "end", _strict_int(self.end, "trigger range end"))
        object.__setattr__(self, "every", _strict_int(self.every, "trigger range every"))
        if self.start <= 0:
            raise ValueError("trigger range start must be positive")
        if self.end < self.start:
            raise ValueError("trigger range end must be at least start")
        if self.every <= 0:
            raise ValueError("trigger range every must be positive")

    def matches(self, iteration: int) -> bool:
        return self.start <= iteration <= self.end and (iteration - self.start) % self.every == 0

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IterationRange":
        _reject_unknown_keys(value, _RANGE_KEYS, "trigger range")
        return cls(
            start=value["start"],
            end=value["end"],
            every=value.get("every", 1),
        )

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "end": self.end, "every": self.every}


@dataclass(frozen=True, slots=True)
class IncidentTrigger:
    """Exact or periodic candidate iterations for an incident."""

    at: tuple[int, ...] = ()
    range: IterationRange | None = None
    probability: float = 1.0

    def __post_init__(self) -> None:
        normalized_at = tuple(_strict_int(iteration, "trigger iteration") for iteration in self.at)
        object.__setattr__(self, "at", normalized_at)
        if isinstance(self.range, Mapping):
            object.__setattr__(self, "range", IterationRange.from_dict(self.range))
        if bool(self.at) == bool(self.range):
            raise ValueError("incident trigger requires exactly one of at or range")
        if any(iteration <= 0 for iteration in self.at):
            raise ValueError("trigger iterations must be positive")
        if len(set(self.at)) != len(self.at):
            raise ValueError("trigger iterations must not contain duplicates")
        if tuple(sorted(self.at)) != self.at:
            raise ValueError("trigger iterations must be sorted")
        if isinstance(self.probability, bool) or not isinstance(self.probability, (int, float)):
            raise TypeError("trigger probability must be a number")
        probability = float(self.probability)
        object.__setattr__(self, "probability", probability)
        if not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
            raise ValueError("trigger probability must be between 0 and 1")

    @property
    def has_multiple_candidates(self) -> bool:
        if self.range is not None:
            return 1 + (self.range.end - self.range.start) // self.range.every > 1
        return len(self.at) > 1

    def matches(self, iteration: int) -> bool:
        if self.range is not None:
            return self.range.matches(iteration)
        return iteration in self.at

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IncidentTrigger":
        _reject_unknown_keys(value, _TRIGGER_KEYS, "incident trigger")
        return cls(
            at=tuple(value.get("at", ())),
            range=(
                None if value.get("range") is None else IterationRange.from_dict(value["range"])
            ),
            probability=value.get("probability", 1.0),
        )

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {"probability": self.probability}
        if self.range is not None:
            value["range"] = self.range.to_dict()
        else:
            value["at"] = list(self.at)
        return value


@dataclass(frozen=True, slots=True)
class IncidentLifetime:
    """Duration of one incident occurrence."""

    matching_calls: int | None = None
    iterations: int | None = None
    until: str | None = None

    def __post_init__(self) -> None:
        if self.matching_calls is not None:
            object.__setattr__(
                self,
                "matching_calls",
                _strict_int(self.matching_calls, "lifetime matching_calls"),
            )
        if self.iterations is not None:
            object.__setattr__(
                self,
                "iterations",
                _strict_int(self.iterations, "lifetime iterations"),
            )
        selected = sum(
            value is not None for value in (self.matching_calls, self.iterations, self.until)
        )
        if selected != 1:
            raise ValueError(
                "incident lifetime requires exactly one of matching_calls, iterations, or until"
            )
        if self.matching_calls is not None and self.matching_calls <= 0:
            raise ValueError("lifetime matching_calls must be positive")
        if self.iterations is not None and self.iterations <= 0:
            raise ValueError("lifetime iterations must be positive")
        if self.until not in {None, "recovery", "replacement", "campaign_end"}:
            raise ValueError("lifetime until must be recovery, replacement, or campaign_end")

    @property
    def permanent(self) -> bool:
        return self.until is not None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IncidentLifetime":
        _reject_unknown_keys(value, _LIFETIME_KEYS, "incident lifetime")
        return cls(
            matching_calls=(
                None if value.get("matching_calls") is None else value["matching_calls"]
            ),
            iterations=(None if value.get("iterations") is None else value["iterations"]),
            until=value.get("until"),
        )

    def to_dict(self) -> dict[str, Any]:
        if self.matching_calls is not None:
            return {"matching_calls": self.matching_calls}
        if self.iterations is not None:
            return {"iterations": self.iterations}
        return {"until": self.until}


@dataclass(frozen=True, slots=True)
class FaultTarget:
    """Framework-neutral logical or explicit failure target."""

    surface: FaultSurface
    rank: int | None = None
    model_part: int = 0
    component: str | None = None
    index: int | None = None
    module_path: str | None = None
    operation: str | None = None
    resource: str | None = None
    path: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "surface", FaultSurface(self.surface))
        if self.rank is not None:
            object.__setattr__(self, "rank", _strict_int(self.rank, "fault target rank"))
        object.__setattr__(
            self,
            "model_part",
            _strict_int(self.model_part, "fault target model_part"),
        )
        if self.index is not None:
            object.__setattr__(self, "index", _strict_int(self.index, "fault target index"))
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, "fault target metadata"),
        )
        if self.rank is not None and self.rank < 0:
            raise ValueError("fault target rank must be non-negative")
        if self.model_part < 0:
            raise ValueError("fault target model_part must be non-negative")
        if self.index is not None and self.index < 0:
            raise ValueError("fault target index must be non-negative")
        if self.module_path is not None and not self.module_path:
            raise ValueError("fault target module_path must be non-empty")
        module_surfaces = {
            FaultSurface.INPUT,
            FaultSurface.OUTPUT,
            FaultSurface.WEIGHT,
            FaultSurface.BIAS,
            FaultSurface.GRADIENT,
            FaultSurface.OPTIMIZER_STATE,
            FaultSurface.COMPUTE,
        }
        if self.surface in module_surfaces and self.module_path is None and self.component is None:
            raise ValueError("module fault targets require module_path or a logical component")

    @property
    def execution_rank(self) -> int:
        """Rank responsible for applying the fault."""
        return 0 if self.rank is None else self.rank

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultTarget":
        _reject_unknown_keys(value, _TARGET_KEYS, "fault target")
        return cls(
            surface=FaultSurface(value["surface"]),
            rank=None if value.get("rank") is None else value["rank"],
            model_part=value.get("model_part", 0),
            component=value.get("component"),
            index=None if value.get("index") is None else value["index"],
            module_path=value.get("module_path"),
            operation=value.get("operation"),
            resource=value.get("resource"),
            path=value.get("path"),
            metadata=value.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            key: value
            for key, value in {
                "surface": self.surface.value,
                "rank": self.rank,
                "model_part": self.model_part,
                "component": self.component,
                "index": self.index,
                "module_path": self.module_path,
                "operation": self.operation,
                "resource": self.resource,
                "path": self.path,
                "metadata": thaw_json(self.metadata),
            }.items()
            if value is not None and value != {}
        }


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One failure effect within an incident."""

    fault_id: str
    type: FailureType
    target: FaultTarget
    parameters: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "type", FailureType(self.type))
        if isinstance(self.target, Mapping):
            object.__setattr__(self, "target", FaultTarget.from_dict(self.target))
        if not isinstance(self.target, FaultTarget):
            raise TypeError("fault target must be a FaultTarget or mapping")
        object.__setattr__(
            self,
            "parameters",
            freeze_json_mapping(self.parameters, "fault parameters"),
        )
        if not self.fault_id or not self.fault_id.strip():
            raise ValueError("fault_id must be non-empty")
        self._validate_parameters()

    @property
    def safety(self) -> SafetyClass:
        return minimum_safety(self.type)

    @property
    def expected_kind(self) -> str:
        return expected_failure_kind(self.type)

    def _validate_parameters(self) -> None:
        if self.type is FailureType.TENSOR_CORRUPTION:
            operation = self.parameters.get("operation")
            if operation is None:
                raise ValueError("tensor_corruption requires parameters.operation")
            CorruptionOperation(operation)
            if "scope" in self.parameters:
                FaultScope(self.parameters["scope"])
            if "magnitude" in self.parameters:
                FaultMagnitude(self.parameters["magnitude"])
        if self.type is FailureType.DELAY:
            delay_ms = float(self.parameters.get("delay_ms", 0.0))
            if delay_ms <= 0:
                raise ValueError("delay requires parameters.delay_ms greater than zero")
        if (
            self.type is FailureType.TENSOR_CORRUPTION
            and self.parameters.get("operation") == CorruptionOperation.SET_VALUE.value
        ):
            if "value" not in self.parameters:
                raise ValueError("set_value corruption requires parameters.value")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultSpec":
        _reject_unknown_keys(value, _FAULT_KEYS, "fault")
        _reject_alias_pair(value, "fault_id", "id", "fault")
        return cls(
            fault_id=str(value.get("fault_id", value.get("id", ""))),
            type=FailureType(value["type"]),
            target=FaultTarget.from_dict(value["target"]),
            parameters=value.get("parameters", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fault_id": self.fault_id,
            "type": self.type.value,
            "target": self.target.to_dict(),
            "parameters": thaw_json(self.parameters),
        }


@dataclass(frozen=True, slots=True)
class FaultIncident:
    """One scheduled incident containing correlated failure effects."""

    incident_id: str
    trigger: IncidentTrigger
    lifetime: IncidentLifetime
    faults: tuple[FaultSpec, ...]
    retrigger: RetriggerPolicy = RetriggerPolicy.ONCE
    max_occurrences: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.trigger, Mapping):
            object.__setattr__(self, "trigger", IncidentTrigger.from_dict(self.trigger))
        if isinstance(self.lifetime, Mapping):
            object.__setattr__(self, "lifetime", IncidentLifetime.from_dict(self.lifetime))
        normalized_faults = tuple(
            FaultSpec.from_dict(fault) if isinstance(fault, Mapping) else fault
            for fault in self.faults
        )
        object.__setattr__(self, "faults", normalized_faults)
        object.__setattr__(self, "retrigger", RetriggerPolicy(self.retrigger))
        if self.max_occurrences is not None:
            object.__setattr__(
                self,
                "max_occurrences",
                _strict_int(self.max_occurrences, "max_occurrences"),
            )
        if not self.incident_id or not self.incident_id.strip():
            raise ValueError("incident_id must be non-empty")
        if not self.faults:
            raise ValueError("incident must contain at least one fault")
        if not all(isinstance(fault, FaultSpec) for fault in self.faults):
            raise TypeError("incident faults must be FaultSpec instances or mappings")
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("fault_id values must be unique within an incident")
        if self.retrigger is RetriggerPolicy.MAX_OCCURRENCES:
            if self.max_occurrences is None or self.max_occurrences <= 0:
                raise ValueError("max_occurrences retrigger requires a positive max_occurrences")
        elif self.max_occurrences is not None:
            raise ValueError(
                "max_occurrences is only valid with the max_occurrences retrigger policy"
            )
        if self.lifetime.permanent and self.trigger.has_multiple_candidates:
            raise ValueError("permanent incidents require a single trigger candidate")
        if self.lifetime.matching_calls is not None and any(
            fault.target.surface
            in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.OPTIMIZER_STATE,
            }
            for fault in self.faults
        ):
            raise ValueError(
                "weight, bias, and optimizer_state faults do not support matching_calls; "
                "use an iterations or until lifetime so the mutation remains active "
                "through backward and the optimizer boundary"
            )

    @property
    def temporal_behavior(self) -> str:
        if self.lifetime.permanent:
            return "permanent"
        if self.trigger.has_multiple_candidates or self.trigger.probability < 1.0:
            return "intermittent"
        return "transient"

    @property
    def safety(self) -> SafetyClass:
        order = {
            SafetyClass.SAFE_IN_PROCESS: 0,
            SafetyClass.ISOLATED_DESTRUCTIVE: 1,
            SafetyClass.CLUSTER_DESTRUCTIVE: 2,
        }
        return max((fault.safety for fault in self.faults), key=order.__getitem__)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultIncident":
        _reject_unknown_keys(value, _INCIDENT_KEYS, "incident")
        _reject_alias_pair(value, "incident_id", "id", "incident")
        return cls(
            incident_id=str(value.get("incident_id", value.get("id", ""))),
            trigger=IncidentTrigger.from_dict(value["trigger"]),
            lifetime=IncidentLifetime.from_dict(value["lifetime"]),
            faults=tuple(FaultSpec.from_dict(fault) for fault in value["faults"]),
            retrigger=RetriggerPolicy(value.get("retrigger", RetriggerPolicy.ONCE.value)),
            max_occurrences=(
                None if value.get("max_occurrences") is None else value["max_occurrences"]
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = {
            "incident_id": self.incident_id,
            "trigger": self.trigger.to_dict(),
            "lifetime": self.lifetime.to_dict(),
            "retrigger": self.retrigger.value,
            "faults": [fault.to_dict() for fault in self.faults],
        }
        if self.max_occurrences is not None:
            value["max_occurrences"] = self.max_occurrences
        return value


@dataclass(frozen=True, slots=True)
class FaultCampaign:
    """A reproducible sequence of framework-neutral failure incidents."""

    name: str
    incidents: tuple[FaultIncident, ...]
    schema_version: int = SCHEMA_VERSION
    seed: int = 0
    clock: ClockSpec = field(default_factory=ClockSpec)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_incidents = tuple(
            FaultIncident.from_dict(incident) if isinstance(incident, Mapping) else incident
            for incident in self.incidents
        )
        object.__setattr__(self, "incidents", normalized_incidents)
        if isinstance(self.clock, Mapping):
            object.__setattr__(self, "clock", ClockSpec.from_dict(self.clock))
        object.__setattr__(
            self,
            "schema_version",
            _strict_int(self.schema_version, "campaign schema_version"),
        )
        object.__setattr__(self, "seed", _strict_int(self.seed, "campaign seed"))
        object.__setattr__(
            self,
            "metadata",
            freeze_json_mapping(self.metadata, "campaign metadata"),
        )
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"unsupported campaign schema_version {self.schema_version}; "
                f"expected {SCHEMA_VERSION}"
            )
        if not self.name or not self.name.strip():
            raise ValueError("campaign name must be non-empty")
        if not -(2**127) <= self.seed < 2**127:
            raise ValueError("campaign seed must fit in a signed 128-bit integer")
        if not self.incidents:
            raise ValueError("campaign must contain at least one incident")
        if not all(isinstance(incident, FaultIncident) for incident in self.incidents):
            raise TypeError("campaign incidents must be FaultIncident instances or mappings")
        incident_ids = [incident.incident_id for incident in self.incidents]
        if len(set(incident_ids)) != len(incident_ids):
            raise ValueError("campaign incident_id values must be unique")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultCampaign":
        _reject_unknown_keys(value, _CAMPAIGN_KEYS, "campaign")
        return cls(
            schema_version=value.get("schema_version", SCHEMA_VERSION),
            name=str(value["name"]),
            seed=value.get("seed", 0),
            clock=ClockSpec.from_dict(value.get("clock", {})),
            incidents=tuple(FaultIncident.from_dict(incident) for incident in value["incidents"]),
            metadata=value.get("metadata", {}),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultCampaign":
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("fault campaign JSON must contain an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "name": self.name,
            "seed": self.seed,
            "clock": self.clock.to_dict(),
            "incidents": [incident.to_dict() for incident in self.incidents],
            "metadata": thaw_json(self.metadata),
        }

    @property
    def manifest_identity(self) -> str:
        """Return a stable identity for the complete executable manifest."""
        manifest = json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(manifest.encode("utf-8")).hexdigest()

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")


def _reject_unknown_keys(
    value: Mapping[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(f"{label} contains unknown fields: {unknown}")


def _reject_alias_pair(
    value: Mapping[str, Any],
    canonical: str,
    alias: str,
    label: str,
) -> None:
    if canonical in value and alias in value:
        raise ValueError(f"{label} cannot contain both {canonical!r} and {alias!r}")


def _strict_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


__all__ = [
    "SCHEMA_VERSION",
    "ClockOrigin",
    "ClockSpec",
    "ClockType",
    "CorruptionOperation",
    "FailureType",
    "FaultCampaign",
    "FaultIncident",
    "FaultMagnitude",
    "FaultScope",
    "FaultSpec",
    "FaultSurface",
    "FaultTarget",
    "IncidentLifetime",
    "IncidentTrigger",
    "IterationRange",
    "RetriggerPolicy",
    "SafetyClass",
    "expected_failure_kind",
    "minimum_safety",
]
