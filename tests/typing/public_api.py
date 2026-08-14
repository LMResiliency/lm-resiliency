from collections.abc import Callable

from lm_resiliency import (
    InMemoryCkptConfig,
    OrchestrationHooks,
    RecoveryDecision,
    RecoveryDecisionCallback,
    ResiliencyHandle,
    SCOUTFaultReport,
    enable_resiliency,
)


def accept_config(config: InMemoryCkptConfig) -> InMemoryCkptConfig:
    return config


def accept_handle(handle: ResiliencyHandle) -> ResiliencyHandle:
    return handle


def accept_hooks(hooks: OrchestrationHooks) -> OrchestrationHooks:
    return hooks


def accept_recovery(decision: RecoveryDecision) -> RecoveryDecision:
    return decision


def accept_report(report: SCOUTFaultReport) -> SCOUTFaultReport:
    return report


recovery_callback: RecoveryDecisionCallback
resiliency_entrypoint: Callable[..., ResiliencyHandle] = enable_resiliency
