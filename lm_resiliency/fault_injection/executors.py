"""Pluggable executors for environment-specific destructive failures."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol

from lm_resiliency.fault_injection.config import (
    FailureType,
    FaultSpec,
    IncidentLifetime,
    SafetyClass,
)

_SAFETY_ORDER = {
    SafetyClass.SAFE_IN_PROCESS: 0,
    SafetyClass.ISOLATED_DESTRUCTIVE: 1,
    SafetyClass.CLUSTER_DESTRUCTIVE: 2,
}


@dataclass(frozen=True, slots=True)
class FaultExecutionRequest:
    """Fully scheduled failure request passed to an executor."""

    campaign: str
    seed: int
    occurrence_id: str
    incident_id: str
    iteration: int
    attempt: int
    temporal_behavior: str
    lifetime: IncidentLifetime
    fault: FaultSpec

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "seed": self.seed,
            "occurrence_id": self.occurrence_id,
            "incident_id": self.incident_id,
            "iteration": self.iteration,
            "attempt": self.attempt,
            "temporal_behavior": self.temporal_behavior,
            "lifetime": self.lifetime.to_dict(),
            "fault": self.fault.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class FaultExecutionResult:
    """Verified executor outcome and optional deactivation token."""

    verified: bool
    active: bool = True
    token: Any = None
    evidence: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.verified, bool):
            raise TypeError("fault execution verified must be a boolean")
        if not isinstance(self.active, bool):
            raise TypeError("fault execution active must be a boolean")
        object.__setattr__(self, "evidence", dict(self.evidence))


class FaultExecutor(Protocol):
    """Environment-specific failure executor."""

    @property
    def name(self) -> str: ...

    def supports(self, fault: FaultSpec) -> bool: ...

    def activate(self, request: FaultExecutionRequest) -> FaultExecutionResult: ...

    def deactivate(
        self,
        request: FaultExecutionRequest,
        result: FaultExecutionResult,
    ) -> Mapping[str, Any] | None: ...


class CallbackFaultExecutor:
    """Callback adapter for process, storage, network, and manager backends."""

    def __init__(
        self,
        *,
        name: str,
        supported_types: set[FailureType] | frozenset[FailureType],
        activate: Callable[[FaultExecutionRequest], FaultExecutionResult],
        validate: Callable[[FaultExecutionRequest], None] | None = None,
        deactivate: Callable[
            [FaultExecutionRequest, FaultExecutionResult],
            Mapping[str, Any] | None,
        ]
        | None = None,
        max_safety: SafetyClass = SafetyClass.SAFE_IN_PROCESS,
    ) -> None:
        if not name or not name.strip():
            raise ValueError("fault executor name must be non-empty")
        self._name = name
        self._supported_types = frozenset(
            FailureType(failure_type) for failure_type in supported_types
        )
        self._activate = activate
        self._validate = validate
        self._deactivate = deactivate
        self._max_safety = SafetyClass(max_safety)

    @property
    def name(self) -> str:
        return self._name

    def supports(self, fault: FaultSpec) -> bool:
        return (
            fault.type in self._supported_types
            and _SAFETY_ORDER[fault.safety] <= _SAFETY_ORDER[self._max_safety]
        )

    @property
    def can_deactivate(self) -> bool:
        return self._deactivate is not None

    def validate(self, request: FaultExecutionRequest) -> None:
        if self._validate is not None:
            self._validate(request)

    def activate(self, request: FaultExecutionRequest) -> FaultExecutionResult:
        result = self._activate(request)
        if not isinstance(result, FaultExecutionResult):
            raise TypeError("fault executor activate callback must return FaultExecutionResult")
        return result

    def deactivate(
        self,
        request: FaultExecutionRequest,
        result: FaultExecutionResult,
    ) -> Mapping[str, Any] | None:
        if self._deactivate is None:
            return None
        evidence = self._deactivate(request, result)
        return None if evidence is None else dict(evidence)


def supported_failure_types(executors: tuple[FaultExecutor, ...]) -> frozenset[FailureType]:
    """Return the canonical types accepted by at least one executor."""
    supported: set[FailureType] = set()
    for failure_type in FailureType:
        probe = _probe_fault(failure_type)
        if any(executor.supports(probe) for executor in executors):
            supported.add(failure_type)
    return frozenset(supported)


def _probe_fault(failure_type: FailureType) -> FaultSpec:
    from lm_resiliency.fault_injection.config import FaultSurface, FaultTarget

    parameters: dict[str, Any] = {}
    if failure_type is FailureType.TENSOR_CORRUPTION:
        parameters = {"operation": "sign_flip"}
    elif failure_type is FailureType.DELAY:
        parameters = {"delay_ms": 1.0}
    return FaultSpec(
        fault_id="capability-probe",
        type=failure_type,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.RESOURCE,
            resource="capability-probe",
        ),
        parameters=parameters,
    )


__all__ = [
    "CallbackFaultExecutor",
    "FaultExecutionRequest",
    "FaultExecutionResult",
    "FaultExecutor",
    "supported_failure_types",
]
