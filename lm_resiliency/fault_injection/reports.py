"""Neutral ground truth and localization evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class InjectionStatus(str, Enum):
    """Lifecycle state for one fault action occurrence."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    SKIPPED_PROBABILITY = "skipped_probability"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class FaultInjectionRecord:
    """Verified ground truth for one fault action occurrence."""

    injection_id: str
    occurrence_id: str
    incident_id: str
    fault_id: str
    iteration: int
    attempt: int
    temporal_behavior: str
    failure_type: str
    expected_kind: str
    safety: str
    framework: str
    executor: str
    execution_rank: int
    target: Mapping[str, Any]
    parameters: Mapping[str, Any]
    status: InjectionStatus = InjectionStatus.PENDING
    verified: bool = False
    activated_at_ns: int | None = None
    completed_at_ns: int | None = None
    evidence: Mapping[str, Any] = field(default_factory=dict)
    error: str | None = None

    @property
    def injection_succeeded(self) -> bool:
        return self.verified and self.status in {
            InjectionStatus.ACTIVE,
            InjectionStatus.COMPLETED,
        }

    @property
    def expected_rank(self) -> int | None:
        rank = self.target.get("rank")
        return self.execution_rank if rank is None else int(rank)

    @property
    def expected_resource(self) -> str | None:
        resource = self.target.get("resource")
        return None if resource is None else str(resource)

    @property
    def expected_component(self) -> str | None:
        component = self.target.get("component")
        module_path = self.target.get("module_path")
        value = component if component is not None else module_path
        return None if value is None else str(value)

    def to_dict(self) -> dict[str, Any]:
        return {
            "injection_id": self.injection_id,
            "occurrence_id": self.occurrence_id,
            "incident_id": self.incident_id,
            "fault_id": self.fault_id,
            "iteration": self.iteration,
            "attempt": self.attempt,
            "temporal_behavior": self.temporal_behavior,
            "failure_type": self.failure_type,
            "expected_kind": self.expected_kind,
            "safety": self.safety,
            "framework": self.framework,
            "executor": self.executor,
            "execution_rank": self.execution_rank,
            "target": dict(self.target),
            "parameters": dict(self.parameters),
            "status": self.status.value,
            "verified": self.verified,
            "injection_succeeded": self.injection_succeeded,
            "activated_at_ns": self.activated_at_ns,
            "completed_at_ns": self.completed_at_ns,
            "evidence": dict(self.evidence),
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Neutral result submitted by the resiliency system under evaluation."""

    occurrence_id: str
    detected: bool
    failed_ranks: tuple[int, ...] = ()
    failed_resources: tuple[str, ...] = ()
    kind: str | None = None
    components: tuple[str, ...] = ()
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failed_ranks",
            tuple(sorted({int(rank) for rank in self.failed_ranks})),
        )
        object.__setattr__(
            self,
            "failed_resources",
            tuple(sorted({str(resource) for resource in self.failed_resources})),
        )
        object.__setattr__(
            self,
            "components",
            tuple(sorted({str(component) for component in self.components})),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.occurrence_id:
            raise ValueError("localization occurrence_id must be non-empty")
        if not isinstance(self.detected, bool):
            raise TypeError("localization detected must be a boolean")
        if any(rank < 0 for rank in self.failed_ranks):
            raise ValueError("localization failed_ranks must be non-negative")
        if not self.detected and (self.failed_ranks or self.failed_resources or self.components):
            raise ValueError("undetected localization results cannot report failed targets")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("localization latency_ms must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalizationResult":
        return cls(
            occurrence_id=str(value.get("occurrence_id", value.get("injection_id", ""))),
            detected=value["detected"],
            failed_ranks=tuple(value.get("failed_ranks", ())),
            failed_resources=tuple(value.get("failed_resources", ())),
            kind=value.get("kind"),
            components=tuple(value.get("components", ())),
            latency_ms=(None if value.get("latency_ms") is None else float(value["latency_ms"])),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "detected": self.detected,
            "failed_ranks": list(self.failed_ranks),
            "failed_resources": list(self.failed_resources),
            "kind": self.kind,
            "components": list(self.components),
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FaultEvaluation:
    """Comparison of one localization result with one incident occurrence."""

    occurrence_id: str
    injection_succeeded: bool
    detected: bool
    localized: bool
    expected_ranks: tuple[int, ...]
    reported_ranks: tuple[int, ...]
    unexpected_ranks: tuple[int, ...]
    expected_resources: tuple[str, ...]
    reported_resources: tuple[str, ...]
    unexpected_resources: tuple[str, ...]
    kind_matches: bool | None
    component_matches: bool | None
    latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "occurrence_id": self.occurrence_id,
            "injection_succeeded": self.injection_succeeded,
            "detected": self.detected,
            "localized": self.localized,
            "expected_ranks": list(self.expected_ranks),
            "reported_ranks": list(self.reported_ranks),
            "unexpected_ranks": list(self.unexpected_ranks),
            "expected_resources": list(self.expected_resources),
            "reported_resources": list(self.reported_resources),
            "unexpected_resources": list(self.unexpected_resources),
            "kind_matches": self.kind_matches,
            "component_matches": self.component_matches,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Machine-readable result for one rank-local campaign session."""

    campaign: str
    manifest_identity: str
    manifest: Mapping[str, Any]
    framework: str
    rank: int
    completed_iterations: int
    injections: tuple[FaultInjectionRecord, ...]
    localizations: tuple[LocalizationResult, ...]
    evaluations: tuple[FaultEvaluation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "campaign": self.campaign,
            "manifest_identity": self.manifest_identity,
            "manifest": dict(self.manifest),
            "framework": self.framework,
            "rank": self.rank,
            "completed_iterations": self.completed_iterations,
            "metadata": dict(self.metadata),
            "injections": [record.to_dict() for record in self.injections],
            "localizations": [result.to_dict() for result in self.localizations],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }

    def to_json(self, path: str | Path) -> None:
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, allow_nan=False, indent=2, sort_keys=True)
            stream.write("\n")


__all__ = [
    "CampaignReport",
    "FaultEvaluation",
    "FaultInjectionRecord",
    "InjectionStatus",
    "LocalizationResult",
]
