"""Neutral injection ground truth and localization evaluation reports."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Mapping


class InjectionStatus(str, Enum):
    """Lifecycle state for one scheduled injection."""

    PENDING = "pending"
    INJECTED = "injected"
    SKIPPED_PROBABILITY = "skipped_probability"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class FaultInjectionRecord:
    """Verified ground truth for one fault occurrence."""

    injection_id: str
    fault_id: str
    step: int
    framework: str
    rank: int
    expected_rank: int
    model_part: int
    module: str
    location: str
    fault_type: str
    expected_kind: str
    magnitude: str
    scope: str
    persistence: str
    probability: float
    seed: int
    call_index: int
    delay_ms: float
    status: InjectionStatus = InjectionStatus.PENDING
    affected_elements: int = 0
    triggered_at_ns: int | None = None
    injected_at_ns: int | None = None
    error: str | None = None

    @property
    def injection_succeeded(self) -> bool:
        """Whether the fault effect was verified."""
        return self.status is InjectionStatus.INJECTED

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-ready injection ground truth."""
        return {
            "injection_id": self.injection_id,
            "fault_id": self.fault_id,
            "step": self.step,
            "framework": self.framework,
            "rank": self.rank,
            "expected_rank": self.expected_rank,
            "model_part": self.model_part,
            "module": self.module,
            "location": self.location,
            "fault_type": self.fault_type,
            "expected_kind": self.expected_kind,
            "magnitude": self.magnitude,
            "scope": self.scope,
            "persistence": self.persistence,
            "probability": self.probability,
            "seed": self.seed,
            "call_index": self.call_index,
            "delay_ms": self.delay_ms,
            "status": self.status.value,
            "injection_succeeded": self.injection_succeeded,
            "affected_elements": self.affected_elements,
            "triggered_at_ns": self.triggered_at_ns,
            "injected_at_ns": self.injected_at_ns,
            "error": self.error,
        }


@dataclass(frozen=True, slots=True)
class LocalizationResult:
    """Neutral result submitted by the resiliency system under evaluation."""

    injection_id: str
    detected: bool
    failed_ranks: tuple[int, ...] = ()
    kind: str | None = None
    scope: str | None = None
    component: str | None = None
    latency_ms: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "failed_ranks",
            tuple(sorted({int(rank) for rank in self.failed_ranks})),
        )
        object.__setattr__(self, "metadata", dict(self.metadata))
        if not self.injection_id:
            raise ValueError("localization injection_id must be non-empty")
        if any(rank < 0 for rank in self.failed_ranks):
            raise ValueError("localization failed_ranks must be non-negative")
        if not self.detected and self.failed_ranks:
            raise ValueError("undetected localization results cannot report failed ranks")
        if self.latency_ms is not None and self.latency_ms < 0:
            raise ValueError("localization latency_ms must be non-negative")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "LocalizationResult":
        """Build a localization result from a JSON-ready mapping."""
        return cls(
            injection_id=str(value["injection_id"]),
            detected=bool(value["detected"]),
            failed_ranks=tuple(value.get("failed_ranks", ())),
            kind=value.get("kind"),
            scope=value.get("scope"),
            component=value.get("component"),
            latency_ms=(None if value.get("latency_ms") is None else float(value["latency_ms"])),
            metadata=dict(value.get("metadata", {})),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready localization result."""
        return {
            "injection_id": self.injection_id,
            "detected": self.detected,
            "failed_ranks": list(self.failed_ranks),
            "kind": self.kind,
            "scope": self.scope,
            "component": self.component,
            "latency_ms": self.latency_ms,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class FaultEvaluation:
    """Comparison of one submitted result with injection ground truth."""

    injection_id: str
    injection_succeeded: bool
    detected: bool
    localized: bool
    expected_rank: int
    reported_ranks: tuple[int, ...]
    unexpected_ranks: tuple[int, ...]
    kind_matches: bool | None
    component_matches: bool | None
    latency_ms: float | None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready evaluation."""
        return {
            "injection_id": self.injection_id,
            "injection_succeeded": self.injection_succeeded,
            "detected": self.detected,
            "localized": self.localized,
            "expected_rank": self.expected_rank,
            "reported_ranks": list(self.reported_ranks),
            "unexpected_ranks": list(self.unexpected_ranks),
            "kind_matches": self.kind_matches,
            "component_matches": self.component_matches,
            "latency_ms": self.latency_ms,
        }


@dataclass(frozen=True, slots=True)
class CampaignReport:
    """Machine-readable result for one rank-local campaign session."""

    campaign: str
    manifest: Mapping[str, Any]
    framework: str
    rank: int
    injections: tuple[FaultInjectionRecord, ...]
    localizations: tuple[LocalizationResult, ...]
    evaluations: tuple[FaultEvaluation, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready campaign report."""
        return {
            "campaign": self.campaign,
            "manifest": dict(self.manifest),
            "framework": self.framework,
            "rank": self.rank,
            "metadata": dict(self.metadata),
            "injections": [record.to_dict() for record in self.injections],
            "localizations": [result.to_dict() for result in self.localizations],
            "evaluations": [evaluation.to_dict() for evaluation in self.evaluations],
        }

    def to_json(self, path: str | Path) -> None:
        """Write stable formatted JSON."""
        with Path(path).open("w", encoding="utf-8") as stream:
            json.dump(self.to_dict(), stream, indent=2, sort_keys=True)
            stream.write("\n")


__all__ = [
    "CampaignReport",
    "FaultEvaluation",
    "FaultInjectionRecord",
    "InjectionStatus",
    "LocalizationResult",
]
