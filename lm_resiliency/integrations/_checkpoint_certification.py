"""Shared checkpoint certification workflow for framework integrations."""

from __future__ import annotations

import enum
import logging
from dataclasses import dataclass
from typing import Callable

import torch
import torch.distributed as dist

from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointCoordinator,
    DurableCheckpointEvent,
)
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager, RecoveryMode
from lm_resiliency.detection.layer_replay import (
    ReplayResult,
    replay_result_has_sdc,
)
from lm_resiliency.detection.optimizer_step import OptimizerStepEvidence
from lm_resiliency.detection.replay_harness import ModelReplayHarness
from lm_resiliency.integrations._common import report_replay_result


class FailureKind(str, enum.Enum):
    """Failure classification used to select checkpoint trust."""

    HANG = "hang"
    SDC = "sdc"
    STRAGGLER = "straggler"
    UNCERTAIN = "uncertain"
    MACHINE_UNAVAILABLE = "machine_unavailable"


@dataclass(frozen=True, slots=True)
class _ReplayConsensus:
    """Job-wide facts needed before changing checkpoint trust."""

    all_results_available: bool
    any_sdc: bool
    scheduled_cycles_agree: bool
    scheduled_cycle_complete: bool
    all_shape_cycles_complete: bool


class CheckpointCertificationCoordinator:
    """Apply rotating SCOUT evidence to GEMINI and durable checkpoint roles."""

    def __init__(
        self,
        *,
        checkpoint_manager: InMemoryCheckpointManager | None,
        replay_harness: ModelReplayHarness | None,
        durable_checkpoint: DurableCheckpointCoordinator | None,
        cadence: ResiliencyCadence,
        checkpoint_tensors: Callable[[], list[torch.Tensor]],
        checkpoint_extra: Callable[[], dict],
        fault_callback: Callable[[ReplayResult], None] | None,
        logger: logging.Logger,
        checkpoint_save: Callable[[int], None] | None = None,
    ) -> None:
        self.checkpoint_manager = checkpoint_manager
        self.replay_harness = replay_harness
        self.durable_checkpoint = durable_checkpoint
        self.cadence = cadence
        self._checkpoint_tensors = checkpoint_tensors
        self._checkpoint_extra = checkpoint_extra
        self._fault_callback = fault_callback
        self._logger = logger
        self._checkpoint_save = checkpoint_save
        candidate_step = (
            checkpoint_manager.checkpoint_status.candidate_step
            if checkpoint_manager is not None
            else -1
        )
        if (
            durable_checkpoint is not None
            and checkpoint_manager is not None
            and not durable_checkpoint.has_pending
            and isinstance(candidate_step, int)
            and candidate_step > 0
        ):
            # Durable startup rejects an interrupted validation window. Keep
            # GEMINI's recovery choice, but do not later promote an unmatched
            # candidate from that discarded window.
            checkpoint_manager.clear_candidate()

    def post_step(
        self,
        step: int,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        optimizer_step_tensors: OptimizerStepEvidence | None = None,
    ) -> ReplayResult | None:
        """Run scheduled replay and checkpoint transitions after a training step."""
        detection_due = self.cadence.detection_due(step)

        replay_result = self._run_replay(
            optimizer=optimizer,
            optimizer_step_tensors=optimizer_step_tensors,
        )

        if not self.cadence.detection_enabled:
            if self.cadence.checkpoint_due(step) and self.checkpoint_manager is not None:
                self._save_checkpoint(step)
            return replay_result

        if detection_due or replay_result is not None:
            self.apply_result(step, replay_result)
        return replay_result

    def apply_result(
        self,
        step: int,
        replay_result: ReplayResult | None,
        *,
        force_checkpoint: bool = False,
        require_full_shape_cycle: bool = False,
    ) -> None:
        """Apply one scheduled SCOUT result to checkpoint state transitions."""
        if self.replay_harness is not None:
            self.replay_harness.finalize_communication_localization(replay_result)
        report_replay_result(
            replay_result,
            self._fault_callback,
            step=step,
            log=self._logger,
        )
        consensus = _global_replay_consensus(replay_result)
        if not consensus.all_results_available:
            self._logger.warning(
                "step %s: SCOUT replay did not return on every rank; checkpoint skipped",
                step,
            )
            return

        if consensus.any_sdc:
            self._reject_candidate("SCOUT detected numerical divergence")
            self._log_skipped_checkpoint(step, durable=self.durable_checkpoint is not None)
            return

        if require_full_shape_cycle and not consensus.all_shape_cycles_complete:
            self._logger.warning(
                "step %s: SCOUT full recipe sweep was incomplete; checkpoint skipped",
                step,
            )
            return

        if not consensus.scheduled_cycles_agree:
            raise RuntimeError(
                "SCOUT recipe schedulers disagree on the cycle boundary across ranks"
            )

        assert replay_result is not None
        dense_replay = replay_result.dense_replay
        should_save = force_checkpoint or self.cadence.checkpoint_due(step)
        if should_save and self.checkpoint_manager is not None:
            self._save_checkpoint(step)

        durable_event = DurableCheckpointEvent.NONE
        if self.durable_checkpoint is not None and self.durable_checkpoint.has_pending:
            durable_event = self.durable_checkpoint.observe(replay_result, step=step)
            if durable_event is DurableCheckpointEvent.REJECTED:
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.reject_candidate()
                return

        if not consensus.scheduled_cycle_complete:
            return

        if (
            self.durable_checkpoint is not None
            and self.durable_checkpoint.has_pending
            and durable_event is not DurableCheckpointEvent.COMMITTED
        ):
            self.durable_checkpoint.reject(
                "SCOUT scheduled cycle ended without complete candidate evidence"
            )
            if self.checkpoint_manager is not None:
                self.checkpoint_manager.reject_candidate()
            return

        if should_save and self.checkpoint_manager is not None:
            if dense_replay:
                self.checkpoint_manager.persist_verified_boundary(step)
            else:
                self.checkpoint_manager.persist_cycle_boundary(step)

        should_start_durable = self.durable_checkpoint is not None and (
            force_checkpoint
            or not self.cadence.checkpoint_enabled
            or self.cadence.checkpoint_due(step)
        )
        if should_start_durable:
            harness = self.replay_harness
            if harness is None:
                raise RuntimeError("durable checkpointing requires a SCOUT replay harness")
            try:
                self.durable_checkpoint.begin_candidate(
                    step=step,
                    first_shape_id=harness.current_replay_shape.shape_id,
                )
                if dense_replay:
                    event = self.durable_checkpoint.observe(replay_result, step=step)
                    if event is not DurableCheckpointEvent.COMMITTED:
                        raise RuntimeError("a clean dense checkpoint did not commit immediately")
            except Exception:
                if self.checkpoint_manager is not None:
                    self.checkpoint_manager.clear_candidate()
                raise

    def save_now(
        self,
        step: int,
        *,
        check_now: Callable[[], ReplayResult | None],
    ) -> None:
        """Capture and certify an immediate checkpoint."""
        if self.checkpoint_manager is None and self.durable_checkpoint is None:
            return

        replay_result = check_now()
        if replay_result is not None and replay_result.completed_shape_cycle:
            replay_result.completed_scheduled_cycle = True
        self.apply_result(
            step,
            replay_result,
            force_checkpoint=True,
            require_full_shape_cycle=True,
        )

    def prepare_recovery(
        self,
        *,
        failure_kind: FailureKind | str,
        all_ranks_accessible: bool,
        check_all_recipes: Callable[[], ReplayResult | None],
        step: int,
    ) -> RecoveryMode:
        """Select and persist the recovery source for one classified failure."""
        failure = FailureKind(failure_kind)
        if not all_ranks_accessible or failure is FailureKind.MACHINE_UNAVAILABLE:
            mode = RecoveryMode.RECOVERY_VERIFIED
        elif failure is FailureKind.SDC:
            mode = RecoveryMode.RECOVERY_VERIFIED
        elif failure is FailureKind.STRAGGLER:
            mode = RecoveryMode.LATEST_GEMINI
        else:
            result = check_all_recipes()
            report_replay_result(
                result,
                self._fault_callback,
                step=step,
                log=self._logger,
            )
            consensus = _global_replay_consensus(result)
            mode = (
                RecoveryMode.RECOVERY_VERIFIED
                if not consensus.all_results_available
                or not consensus.all_shape_cycles_complete
                or consensus.any_sdc
                else RecoveryMode.LATEST_GEMINI
            )

        if mode is RecoveryMode.RECOVERY_VERIFIED:
            self._reject_candidate(f"{failure.value} recovery invalidated the current candidate")
        elif self.checkpoint_manager is not None:
            self.checkpoint_manager.set_recovery_mode(mode)
        return mode

    def _run_replay(
        self,
        *,
        optimizer: torch.optim.Optimizer | None,
        optimizer_step_tensors: OptimizerStepEvidence | None,
    ) -> ReplayResult | None:
        harness = self.replay_harness
        if harness is None:
            return None
        replay_kwargs = {
            "optimizer": optimizer,
            "optimizer_step_tensors": optimizer_step_tensors,
        }
        return harness.step(**replay_kwargs)

    def _log_skipped_checkpoint(self, step: int, *, durable: bool) -> None:
        self._logger.warning(
            "step %s: SDC detected - %s checkpoint capture skipped; "
            "recovery requires RECOVERY_VERIFIED",
            step,
            "durable and in-memory" if durable else "in-memory",
        )

    def _reject_candidate(self, reason: str) -> None:
        if self.durable_checkpoint is not None:
            self.durable_checkpoint.reject(reason)
        if self.checkpoint_manager is not None:
            self.checkpoint_manager.reject_candidate()

    def _save_checkpoint(self, step: int) -> None:
        if self._checkpoint_save is not None:
            self._checkpoint_save(step)
            return
        assert self.checkpoint_manager is not None
        self.checkpoint_manager.save_tensors(
            self._checkpoint_tensors(),
            step,
            extra=self._checkpoint_extra(),
        )


def _global_replay_consensus(replay_result: ReplayResult | None) -> _ReplayConsensus:
    """Combine group-local SCOUT evidence before a job-wide checkpoint transition."""
    available = replay_result is not None
    flags = torch.tensor(
        [
            int(not available),
            int(available and replay_result_has_sdc(replay_result)),
            int(available and replay_result.completed_scheduled_cycle),
            int(available and not replay_result.completed_scheduled_cycle),
            int(available and replay_result.completed_shape_cycle),
            int(available and not replay_result.completed_shape_cycle),
        ],
        dtype=torch.int32,
        device=_consensus_device(),
    )
    if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1:
        dist.all_reduce(flags, op=dist.ReduceOp.MAX)

    (
        any_missing,
        any_sdc,
        any_scheduled_complete,
        any_scheduled_incomplete,
        any_shape_complete,
        any_shape_incomplete,
    ) = (bool(value) for value in flags.tolist())
    return _ReplayConsensus(
        all_results_available=not any_missing,
        any_sdc=any_sdc,
        scheduled_cycles_agree=not (any_scheduled_complete and any_scheduled_incomplete),
        scheduled_cycle_complete=any_scheduled_complete and not any_scheduled_incomplete,
        all_shape_cycles_complete=any_shape_complete and not any_shape_incomplete,
    )


def _consensus_device() -> torch.device:
    if not dist.is_available() or not dist.is_initialized():
        return torch.device("cpu")
    try:
        backend = str(dist.get_backend()).lower()
    except Exception:  # noqa: BLE001 - backend probing is best effort
        backend = ""
    if backend == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")
