"""Platform-neutral hooks for external training orchestration."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Protocol

from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.layer_replay import (
    ReplayResult,
    replay_result_has_fault,
    replay_result_has_sdc,
)
from lm_resiliency.detection.reports import (
    SCOUTFaultCallback,
    SCOUTFaultReport,
    dispatch_replay_faults,
)
from lm_resiliency.recovery import RecoveryDecision, RecoveryDecisionCallback

RestartDestinationResolver = Callable[[], str | Path | None]
logger = logging.getLogger(__name__)


class _OrchestrationHandle(Protocol):
    def set_restart_destination(
        self,
        resolver: RestartDestinationResolver | None,
    ) -> None: ...

    def set_recovery_decision_callback(
        self,
        callback: RecoveryDecisionCallback | None,
    ) -> None: ...

    def describe_recovery(
        self,
        failure_kind: str,
        recovery_mode: RecoveryMode | str,
        *,
        all_ranks_accessible: bool,
        reason: str,
        allow_collective: bool = False,
    ) -> RecoveryDecision: ...

    def _publish_recovery_decision(
        self,
        failure_kind: str,
        recovery_mode: RecoveryMode,
        *,
        all_ranks_accessible: bool,
        reason: str | None = None,
    ) -> RecoveryDecision: ...


@dataclass
class OrchestrationHooks:
    """Policies supplied by a launcher or cluster manager.

    ``report_fault`` receives the same JSON-ready SCOUT report for replay,
    hang, and DataLoader-stall detections. ``report_recovery`` receives the
    selected checkpoint trust and identity. ``restart_destination`` is resolved
    only when GEMINI needs an optional checkpoint mirror during restart.
    """

    report_fault: SCOUTFaultCallback | None = None
    report_recovery: RecoveryDecisionCallback | None = None
    restart_destination: RestartDestinationResolver | None = None
    _handle: _OrchestrationHandle | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )

    def replay_fault_callback(self, result: ReplayResult) -> None:
        """Translate an in-process replay result to normalized fault reports."""
        if (
            self.report_recovery is not None
            and self._handle is not None
            and replay_result_has_fault(result)
        ):
            has_sdc = replay_result_has_sdc(result)
            try:
                self._handle._publish_recovery_decision(
                    "sdc" if has_sdc else "straggler",
                    (RecoveryMode.RECOVERY_VERIFIED if has_sdc else RecoveryMode.LATEST_GEMINI),
                    all_ranks_accessible=True,
                    reason=(
                        "replay_sdc_requires_verified_recovery"
                        if has_sdc
                        else "replay_straggler_allows_latest_recovery"
                    ),
                )
            except Exception:
                logger.exception("recovery decision callback failed")
        if self.report_fault is not None:
            dispatch_replay_faults(result, self.report_fault)

    def oob_fault_callback(self, report: SCOUTFaultReport) -> None:
        """Report OOB evidence and a conservative hang recovery decision."""
        kind = report.get("kind")
        if (
            self.report_recovery is not None
            and self._handle is not None
            and kind in {"hang", "data_stall", "checkpoint_stall"}
        ):
            is_hang = kind == "hang"
            try:
                self._handle._publish_recovery_decision(
                    str(kind),
                    (RecoveryMode.RECOVERY_VERIFIED if is_hang else RecoveryMode.LATEST_GEMINI),
                    all_ranks_accessible=not is_hang,
                    reason=(
                        "oob_hang_requires_conservative_recovery"
                        if is_hang
                        else (
                            "checkpoint_io_stall_allows_latest_recovery"
                            if kind == "checkpoint_stall"
                            else "dataloader_stall_allows_latest_recovery"
                        )
                    ),
                )
            except Exception:
                logger.exception("recovery decision callback failed")
        if self.report_fault is not None:
            self.report_fault(report)

    def bind(self, handle: _OrchestrationHandle) -> None:
        """Apply checkpoint restart policy to a public resiliency handle."""
        self._handle = handle
        if self.report_recovery is not None:
            handle.set_recovery_decision_callback(self.report_recovery)
        if self.restart_destination is not None:
            handle.set_restart_destination(self.restart_destination)


def _resolve_orchestration_callbacks(
    orchestration: OrchestrationHooks | None,
    fault_callback: Callable[[ReplayResult], None] | None,
    oob_fault_callback: SCOUTFaultCallback | None,
) -> tuple[Callable[[ReplayResult], None] | None, SCOUTFaultCallback | None]:
    if orchestration is None:
        return fault_callback, oob_fault_callback
    if orchestration.report_fault is not None and (
        fault_callback is not None or oob_fault_callback is not None
    ):
        raise ValueError(
            "orchestration.report_fault cannot be combined with fault_callback "
            "or oob_fault_callback"
        )
    if orchestration.report_recovery is None:
        if orchestration.report_fault is None:
            return fault_callback, oob_fault_callback
        return orchestration.replay_fault_callback, orchestration.report_fault

    def replay_callback(result: ReplayResult) -> None:
        orchestration.replay_fault_callback(result)
        if orchestration.report_fault is None and fault_callback is not None:
            fault_callback(result)

    def report_callback(report: SCOUTFaultReport) -> None:
        orchestration.oob_fault_callback(report)
        if orchestration.report_fault is None and oob_fault_callback is not None:
            oob_fault_callback(report)

    return replay_callback, report_callback


def _bind_orchestration(
    orchestration: OrchestrationHooks | None,
    handle: _OrchestrationHandle,
) -> None:
    if orchestration is not None:
        orchestration.bind(handle)


__all__ = [
    "OrchestrationHooks",
    "RestartDestinationResolver",
]
