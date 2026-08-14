from typing_extensions import assert_type

from lm_resiliency import (
    InMemoryCkptConfig,
    OrchestrationHooks,
    RecoveryDecision,
    RecoveryDecisionCallback,
    ResiliencyHandle,
    SCOUTFaultReport,
    enable_resiliency,
)
from lm_resiliency.manager_api import (
    OrchestrationHooks as ManagerOrchestrationHooks,
)
from lm_resiliency.manager_api import (
    RecoveryDecision as ManagerRecoveryDecision,
)


def check_entrypoint(model: object) -> None:
    assert_type(enable_resiliency(model), ResiliencyHandle)


def check_config(config: InMemoryCkptConfig) -> None:
    assert_type(config.interval, int)
    assert_type(config.disk_folder, str)


def check_hooks(hooks: OrchestrationHooks) -> None:
    assert_type(hooks, OrchestrationHooks)
    assert_type(hooks, ManagerOrchestrationHooks)


def check_recovery(
    decision: RecoveryDecision,
    callback: RecoveryDecisionCallback,
) -> None:
    assert_type(decision, RecoveryDecision)
    assert_type(decision, ManagerRecoveryDecision)
    assert_type(decision["checkpoint_step"], int)
    callback(decision)


def check_report(report: SCOUTFaultReport) -> None:
    assert_type(report["failed_ranks"], list[int])
    assert_type(report["confidence"], float)
