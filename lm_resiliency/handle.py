"""Public lifecycle handle returned by :func:`enable_resiliency`."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from lm_resiliency.checkpointing.durable import DurableCheckpointCoordinator
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager, RecoveryMode
from lm_resiliency.detection.replay_harness import ModelReplayHarness
from lm_resiliency.detection.stage_instrumentation import (
    CheckpointOperation,
    checkpoint_io,
)
from lm_resiliency.lifecycle import unregister_automatic_cleanup
from lm_resiliency.recovery import (
    RecoveryDecision,
    RecoveryDecisionCallback,
    build_recovery_decision,
    recovery_decision_reason,
)


class ResiliencyHandle:
    """Own enabled resiliency features and their training-hook lifecycle."""

    def __init__(self) -> None:
        self.ckpt_manager: InMemoryCheckpointManager | None = None
        self.durable_checkpoint: DurableCheckpointCoordinator | None = None
        self.replay_harness: ModelReplayHarness | None = None
        self._recovered_step = -1
        self._step_count = 0
        self._hooks: list[Any] = []
        self._close_callbacks: list[Callable[[], None]] = []
        self._prepare_recovery_callback: Callable[[str, bool, int], RecoveryMode] | None = None
        self._recovery_decision_callback: RecoveryDecisionCallback | None = None
        self._last_recovery_decision: RecoveryDecision | None = None
        self._closed = False

    @property
    def step_count(self) -> int:
        """Steps completed since initialization, including any recovered steps."""
        return self._step_count

    @property
    def recovered_step(self) -> int:
        """Recovered checkpoint step, or ``-1`` when training started fresh."""
        return self._recovered_step

    @property
    def closed(self) -> bool:
        """Whether :meth:`close` has released this handle."""
        return self._closed

    def add_close_callback(self, callback: Callable[[], None]) -> None:
        """Register a callback to run once when the handle closes."""
        if self._closed:
            raise RuntimeError("cannot register a callback on a closed handle")
        self._close_callbacks.append(callback)

    def flush_for_restart(self) -> int:
        """Flush the latest recoverable checkpoint before a worker restart."""
        if self.ckpt_manager is None:
            return -1
        return self.ckpt_manager.flush_for_restart()

    def set_restart_destination(
        self,
        resolver: Callable[[], str | Path | None] | None,
    ) -> None:
        """Set GEMINI's optional signal-triggered checkpoint mirror destination."""
        if self.ckpt_manager is not None:
            self.ckpt_manager.set_restart_destination(resolver)

    def copy_checkpoint_to(self, destination: str | Path) -> int:
        """Copy GEMINI's already-flushed own and peer shards to ``destination``."""
        if self.ckpt_manager is None:
            return -1
        return self.ckpt_manager.copy_to(destination)

    def instrument_dataloader(self, dataloader: Any, *, name: str = "train") -> Any:
        """Return a DataLoader proxy sampled at SCOUT's detection interval."""
        if self.replay_harness is None:
            return dataloader
        return self.replay_harness.instrument_dataloader(dataloader, name=name)

    def checkpoint_io(
        self,
        operation: CheckpointOperation,
        *,
        name: str = "framework",
    ):
        """Return a context manager that publishes checkpoint I/O progress."""
        if self.replay_harness is None:
            return checkpoint_io(None, operation, name=name)
        return self.replay_harness.checkpoint_io(operation, name=name)

    def prepare_recovery(
        self,
        failure_kind: str,
        *,
        all_ranks_accessible: bool = True,
    ) -> RecoveryMode:
        """Run any required SCOUT check and select GEMINI recovery trust."""
        if self._prepare_recovery_callback is None:
            mode = (
                RecoveryMode.LATEST_GEMINI
                if failure_kind == "straggler" and all_ranks_accessible
                else RecoveryMode.RECOVERY_VERIFIED
            )
            if self.ckpt_manager is not None:
                self.ckpt_manager.set_recovery_mode(mode)
        else:
            mode = self._prepare_recovery_callback(
                failure_kind,
                all_ranks_accessible,
                self._step_count,
            )
        self._publish_recovery_decision(
            failure_kind,
            mode,
            all_ranks_accessible=all_ranks_accessible,
        )
        return mode

    @property
    def last_recovery_decision(self) -> RecoveryDecision | None:
        """Most recent checkpoint selection reported to orchestration."""
        return self._last_recovery_decision

    def set_recovery_decision_callback(
        self,
        callback: RecoveryDecisionCallback | None,
    ) -> None:
        """Set the external manager callback for checkpoint selections."""
        self._recovery_decision_callback = callback

    def describe_recovery(
        self,
        failure_kind: str,
        recovery_mode: RecoveryMode | str,
        *,
        all_ranks_accessible: bool,
        reason: str,
        allow_collective: bool = False,
    ) -> RecoveryDecision:
        """Describe a checkpoint selection without changing recovery state."""
        return build_recovery_decision(
            failure_kind=failure_kind,
            recovery_mode=recovery_mode,
            all_ranks_accessible=all_ranks_accessible,
            reason=reason,
            checkpoint_manager=self.ckpt_manager,
            durable_checkpoint=self.durable_checkpoint,
            allow_collective=allow_collective,
        )

    def close(self) -> None:
        """Remove hooks and release all enabled resiliency resources."""
        if self._closed:
            return
        self._closed = True
        unregister_automatic_cleanup(self)
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        if self.ckpt_manager is not None:
            self.ckpt_manager.close()
        if self.replay_harness is not None:
            self.replay_harness.remove_hooks()
        for callback in self._close_callbacks:
            callback()
        self._close_callbacks.clear()

    def _register_hook(self, hook: Any) -> None:
        self._hooks.append(hook)

    def _set_prepare_recovery_callback(
        self,
        callback: Callable[[str, bool, int], RecoveryMode],
    ) -> None:
        self._prepare_recovery_callback = callback

    def _publish_recovery_decision(
        self,
        failure_kind: str,
        recovery_mode: RecoveryMode,
        *,
        all_ranks_accessible: bool,
        reason: str | None = None,
    ) -> RecoveryDecision:
        decision = self.describe_recovery(
            failure_kind,
            recovery_mode,
            all_ranks_accessible=all_ranks_accessible,
            reason=(
                reason
                or recovery_decision_reason(
                    failure_kind,
                    recovery_mode,
                    all_ranks_accessible=all_ranks_accessible,
                )
            ),
            allow_collective=False,
        )
        self._last_recovery_decision = decision
        if self._recovery_decision_callback is not None:
            self._recovery_decision_callback(decision)
        return decision

    def _advance_step(self) -> int:
        self._step_count += 1
        return self._step_count

    def _restore_step(self, step: int) -> None:
        self._recovered_step = step
        self._step_count = step
