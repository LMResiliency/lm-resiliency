"""Framework-aware fault injection and localization evaluation."""

from lm_resiliency.fault_injection.config import (
    FaultCampaign,
    FaultLocation,
    FaultMagnitude,
    FaultPersistence,
    FaultScope,
    FaultSpec,
    FaultTarget,
    FaultType,
)
from lm_resiliency.fault_injection.injector import (
    FaultInjectionSession,
    enable_fault_injection,
)
from lm_resiliency.fault_injection.reports import (
    CampaignReport,
    FaultEvaluation,
    FaultInjectionRecord,
    InjectionStatus,
    LocalizationResult,
)

__all__ = [
    "CampaignReport",
    "FaultCampaign",
    "FaultEvaluation",
    "FaultInjectionRecord",
    "FaultInjectionSession",
    "FaultLocation",
    "FaultMagnitude",
    "FaultPersistence",
    "FaultScope",
    "FaultSpec",
    "FaultTarget",
    "FaultType",
    "InjectionStatus",
    "LocalizationResult",
    "enable_fault_injection",
]
