"""Manager-facing checkpoint recovery decisions."""

from __future__ import annotations

from typing import Callable, Literal, TypedDict

from lm_resiliency.checkpointing.manager import RecoveryMode

RecoveryCheckpointSource = Literal["gemini", "durable", "none"]


class RecoveryDecision(TypedDict):
    """JSON-ready checkpoint selection emitted to an external manager."""

    failure_kind: str
    recovery_mode: str
    checkpoint_source: RecoveryCheckpointSource
    checkpoint_step: int
    checkpoint_id: str | None
    all_ranks_accessible: bool
    available: bool
    reason: str


RecoveryDecisionCallback = Callable[[RecoveryDecision], None]


def build_recovery_decision(
    *,
    failure_kind: str,
    recovery_mode: RecoveryMode | str,
    all_ranks_accessible: bool,
    reason: str,
    checkpoint_manager: object | None,
    durable_checkpoint: object | None,
    allow_collective: bool,
) -> RecoveryDecision:
    """Resolve a selected trust mode to a concrete recoverable checkpoint."""
    mode = RecoveryMode(recovery_mode)
    checkpoint_step = -1
    checkpoint_id: str | None = None
    source: RecoveryCheckpointSource = "none"

    if checkpoint_manager is not None:
        if allow_collective:
            checkpoint_step = int(checkpoint_manager.find_latest(mode))  # type: ignore[attr-defined]
        elif hasattr(checkpoint_manager, "local_recovery_step"):
            checkpoint_step = int(
                checkpoint_manager.local_recovery_step(mode)
            )
        elif mode is RecoveryMode.RECOVERY_VERIFIED:
            status = checkpoint_manager.checkpoint_status  # type: ignore[attr-defined]
            checkpoint_step = int(status.recovery_verified_step)
        if checkpoint_step > 0:
            source = "gemini"

    if checkpoint_step <= 0 and durable_checkpoint is not None:
        record = durable_checkpoint.latest_validated  # type: ignore[attr-defined]
        if record is not None:
            checkpoint_step = int(record.step)
            checkpoint_id = str(record.checkpoint_id)
            source = "durable"

    return RecoveryDecision(
        failure_kind=str(failure_kind),
        recovery_mode=mode.value,
        checkpoint_source=source,
        checkpoint_step=checkpoint_step,
        checkpoint_id=checkpoint_id,
        all_ranks_accessible=bool(all_ranks_accessible),
        available=checkpoint_step > 0,
        reason=reason,
    )


def recovery_decision_reason(
    failure_kind: str,
    recovery_mode: RecoveryMode | str,
    *,
    all_ranks_accessible: bool,
) -> str:
    """Return a stable manager-facing explanation for one selection."""
    mode = RecoveryMode(recovery_mode)
    if not all_ranks_accessible or failure_kind == "machine_unavailable":
        return "required_machine_unavailable"
    if failure_kind == "sdc":
        return "sdc_detected"
    if failure_kind == "straggler":
        return "accessible_straggler"
    if mode is RecoveryMode.LATEST_GEMINI:
        return "full_catalog_replay_clean"
    return "full_catalog_replay_incomplete_or_sdc"


__all__ = [
    "RecoveryCheckpointSource",
    "RecoveryDecision",
    "RecoveryDecisionCallback",
    "build_recovery_decision",
    "recovery_decision_reason",
]
