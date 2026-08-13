"""Declarative fault campaign configuration."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

_FRAMEWORKS = {"auto", "pytorch", "torchtitan", "megatron", "deepspeed"}


class FaultType(str, Enum):
    """Fault operation applied to a parameter, output, or invocation."""

    SINGLE_BITFLIP = "single_bitflip"
    MULTI_BITFLIP = "multi_bitflip"
    STUCK_AT_ZERO = "stuck_at_zero"
    STUCK_AT_ONE = "stuck_at_one"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    GAUSSIAN_NOISE = "gaussian_noise"
    SIGN_FLIP = "sign_flip"
    SET_NAN = "set_nan"
    SET_INF = "set_inf"
    DELAY = "delay"


class FaultMagnitude(str, Enum):
    """Relative severity used by scalable numerical faults."""

    CATASTROPHIC = "catastrophic"
    LARGE = "large"
    MEDIUM = "medium"
    SUBTLE = "subtle"
    NEAR_INVISIBLE = "near_invisible"


class FaultScope(str, Enum):
    """Tensor elements affected by one injection."""

    SINGLE = "single"
    ROW = "row"
    PERCENT_1 = "1%"
    PERCENT_10 = "10%"
    FULL = "100%"


class FaultLocation(str, Enum):
    """Framework surface modified by an injection."""

    WEIGHT = "weight"
    BIAS = "bias"
    OUTPUT = "output"


class FaultPersistence(str, Enum):
    """How long an injected fault remains active."""

    TRANSIENT = "transient"
    PERSISTENT = "persistent"


@dataclass(frozen=True, slots=True)
class FaultTarget:
    """Ground-truth location for one fault specification."""

    rank: int
    module: str = ""
    location: FaultLocation = FaultLocation.WEIGHT
    model_part: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "location", FaultLocation(self.location))
        if self.rank < 0:
            raise ValueError("fault target rank must be non-negative")
        if self.model_part < 0:
            raise ValueError("fault target model_part must be non-negative")
        if not isinstance(self.module, str):
            raise TypeError("fault target module must be a string")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultTarget":
        """Build a target from a JSON-ready mapping."""
        return cls(
            rank=int(value["rank"]),
            module=str(value.get("module", "")),
            location=FaultLocation(value.get("location", FaultLocation.WEIGHT.value)),
            model_part=int(value.get("model_part", 0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready target."""
        return {
            "rank": self.rank,
            "module": self.module,
            "location": self.location.value,
            "model_part": self.model_part,
        }


@dataclass(frozen=True, slots=True)
class FaultSpec:
    """One scheduled fault in a campaign."""

    fault_id: str
    fault_type: FaultType
    target: FaultTarget
    steps: tuple[int, ...] = (1,)
    magnitude: FaultMagnitude = FaultMagnitude.MEDIUM
    scope: FaultScope = FaultScope.SINGLE
    persistence: FaultPersistence = FaultPersistence.TRANSIENT
    probability: float = 1.0
    seed: int = 0
    call_index: int = 1
    delay_ms: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "fault_type", FaultType(self.fault_type))
        object.__setattr__(self, "magnitude", FaultMagnitude(self.magnitude))
        object.__setattr__(self, "scope", FaultScope(self.scope))
        object.__setattr__(self, "persistence", FaultPersistence(self.persistence))
        if isinstance(self.target, Mapping):
            object.__setattr__(self, "target", FaultTarget.from_dict(self.target))
        if not isinstance(self.target, FaultTarget):
            raise TypeError("fault target must be a FaultTarget or mapping")
        if isinstance(self.steps, int):
            normalized_steps = (self.steps,)
        else:
            normalized_steps = tuple(int(step) for step in self.steps)
        object.__setattr__(self, "steps", normalized_steps)

        if not self.fault_id or not self.fault_id.strip():
            raise ValueError("fault_id must be non-empty")
        if not self.steps or any(step <= 0 for step in self.steps):
            raise ValueError("fault steps must contain positive integers")
        if len(set(self.steps)) != len(self.steps):
            raise ValueError("fault steps must not contain duplicates")
        if not 0.0 <= self.probability <= 1.0:
            raise ValueError("fault probability must be between 0 and 1")
        if self.call_index <= 0:
            raise ValueError("fault call_index must be positive")
        if self.delay_ms < 0:
            raise ValueError("fault delay_ms must be non-negative")
        if self.fault_type is FaultType.DELAY:
            if self.target.location is not FaultLocation.OUTPUT:
                raise ValueError("delay faults must target module output")
            if self.persistence is not FaultPersistence.TRANSIENT:
                raise ValueError("delay faults must be transient")
            if self.delay_ms <= 0:
                raise ValueError("delay faults require delay_ms greater than zero")
        elif self.delay_ms != 0:
            raise ValueError("delay_ms is only valid for delay faults")
        if (
            self.target.location is FaultLocation.OUTPUT
            and self.persistence is FaultPersistence.PERSISTENT
        ):
            raise ValueError("output faults cannot be persistent")
        if self.persistence is FaultPersistence.PERSISTENT and len(self.steps) != 1:
            raise ValueError("persistent faults support exactly one trigger step")
        if self.persistence is FaultPersistence.PERSISTENT and self.call_index != 1:
            raise ValueError("call_index is only configurable for transient faults")

    @property
    def expected_kind(self) -> str:
        """Expected high-level localization kind."""
        return "straggler" if self.fault_type is FaultType.DELAY else "sdc"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultSpec":
        """Build a specification from a JSON-ready mapping."""
        raw_steps = value.get("steps", value.get("step", (1,)))
        steps = (raw_steps,) if isinstance(raw_steps, int) else tuple(raw_steps)
        return cls(
            fault_id=str(value["fault_id"]),
            fault_type=FaultType(value["fault_type"]),
            target=FaultTarget.from_dict(value["target"]),
            steps=steps,
            magnitude=FaultMagnitude(value.get("magnitude", FaultMagnitude.MEDIUM.value)),
            scope=FaultScope(value.get("scope", FaultScope.SINGLE.value)),
            persistence=FaultPersistence(
                value.get("persistence", FaultPersistence.TRANSIENT.value)
            ),
            probability=float(value.get("probability", 1.0)),
            seed=int(value.get("seed", 0)),
            call_index=int(value.get("call_index", 1)),
            delay_ms=float(value.get("delay_ms", 0.0)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready specification."""
        return {
            "fault_id": self.fault_id,
            "fault_type": self.fault_type.value,
            "target": self.target.to_dict(),
            "steps": list(self.steps),
            "magnitude": self.magnitude.value,
            "scope": self.scope.value,
            "persistence": self.persistence.value,
            "probability": self.probability,
            "seed": self.seed,
            "call_index": self.call_index,
            "delay_ms": self.delay_ms,
        }


@dataclass(frozen=True, slots=True)
class FaultCampaign:
    """A reproducible collection of framework-level fault specifications."""

    name: str
    faults: tuple[FaultSpec, ...]
    framework: str = "auto"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("campaign name must be non-empty")
        normalized_faults = tuple(
            FaultSpec.from_dict(fault) if isinstance(fault, Mapping) else fault
            for fault in self.faults
        )
        object.__setattr__(self, "faults", normalized_faults)
        if not self.faults:
            raise ValueError("campaign must contain at least one fault")
        if not all(isinstance(fault, FaultSpec) for fault in self.faults):
            raise TypeError("campaign faults must be FaultSpec instances or mappings")
        fault_ids = [fault.fault_id for fault in self.faults]
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("campaign fault_id values must be unique")
        occurrences: dict[tuple[Any, ...], str] = {}
        for fault in self.faults:
            for step in fault.steps:
                key = (
                    fault.target.rank,
                    fault.target.model_part,
                    fault.target.module,
                    fault.target.location,
                    step,
                    fault.call_index,
                )
                previous = occurrences.setdefault(key, fault.fault_id)
                if previous != fault.fault_id:
                    raise ValueError(
                        f"faults {previous!r} and {fault.fault_id!r} overlap at "
                        f"the same target, step, and call_index"
                    )
        if self.framework not in _FRAMEWORKS:
            raise ValueError(f"unsupported framework: {self.framework!r}")
        object.__setattr__(self, "metadata", dict(self.metadata))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "FaultCampaign":
        """Build a campaign from a JSON-ready mapping."""
        return cls(
            name=str(value["name"]),
            faults=tuple(FaultSpec.from_dict(fault) for fault in value["faults"]),
            framework=str(value.get("framework", "auto")),
            metadata=dict(value.get("metadata", {})),
        )

    @classmethod
    def from_json(cls, path: str | Path) -> "FaultCampaign":
        """Load a campaign manifest from JSON."""
        with Path(path).open(encoding="utf-8") as stream:
            value = json.load(stream)
        if not isinstance(value, dict):
            raise ValueError("fault campaign JSON must contain an object")
        return cls.from_dict(value)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready campaign manifest."""
        return {
            "name": self.name,
            "framework": self.framework,
            "metadata": dict(self.metadata),
            "faults": [fault.to_dict() for fault in self.faults],
        }

    def to_json(self, path: str | Path) -> None:
        """Write a campaign manifest as stable formatted JSON."""
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")

    def bind(
        self,
        target: Any,
        *,
        framework: str | None = None,
        rank: int | None = None,
    ) -> Any:
        """Bind this campaign to initialized framework training objects."""
        from lm_resiliency.fault_injection.injector import FaultInjectionSession

        return FaultInjectionSession(
            target,
            self,
            framework=framework or self.framework,
            rank=rank,
        )


__all__ = [
    "FaultCampaign",
    "FaultLocation",
    "FaultMagnitude",
    "FaultPersistence",
    "FaultScope",
    "FaultSpec",
    "FaultTarget",
    "FaultType",
]
