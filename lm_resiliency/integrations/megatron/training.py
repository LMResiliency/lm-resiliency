"""Megatron-core training integration for GEMINI + SCOUT.

Provides hook-based integration into Megatron's training loop without
requiring modifications to megatron-core source. Two usage modes:

1. Hook-based (recommended): Wrap Megatron's optimizer to inject post-step
   callbacks for in-memory checkpointing and fault detection.

2. Explicit call: Call save/check manually from your training script after
   each optimizer.step().

Example (hook-based):
    from lm_resiliency.integrations.megatron import enable_megatron_resiliency

    # After Megatron initializes model/optimizer:
    resiliency = enable_megatron_resiliency(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        interval=10,
    )

    # Before training loop: try fast recovery
    recovered_step = resiliency.try_recover()
    if recovered_step > 0:
        iteration = recovered_step

    # Training loop runs unchanged
    for iteration in range(...):
        train_step(...)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lm_resiliency._feature_wiring import _normalize_feature_configs
from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager, RecoveryMode
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.optimizer_step import (
    OptimizerStepCheckUnsupported,
    OptimizerStepEvidence,
    OptimizerStepReplay,
    collect_optimizer_replays,
)
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    ReplayHarnessConfig,
)
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.detection.stage_instrumentation import (
    CheckpointOperation,
    checkpoint_io,
)
from lm_resiliency.detection.topology import ReplayPeerGroup, ReplayPeerRole
from lm_resiliency.integrations._checkpoint_certification import (
    CheckpointCertificationCoordinator,
)
from lm_resiliency.integrations._common import (
    build_checkpoint_manager,
    build_durable_checkpoint,
    checkpoint_extra,
    create_gloo_peer_group,
    prepare_checkpoint_tensor_load,
    recover_with_fallback,
    restore_checkpoint_extra,
)
from lm_resiliency.integrations.megatron.adapter import MegatronAdapter
from lm_resiliency.lifecycle import (
    register_automatic_cleanup,
    unregister_automatic_cleanup,
)
from lm_resiliency.orchestration import (
    OrchestrationHooks,
    _bind_orchestration,
    _resolve_orchestration_callbacks,
)
from lm_resiliency.recovery import (
    RecoveryDecision,
    RecoveryDecisionCallback,
    build_recovery_decision,
    recovery_decision_reason,
)

logger = logging.getLogger(__name__)


class MegatronResiliency:
    """Manages GEMINI checkpointing and SCOUT detection for megatron-core.

    Wraps the optimizer's step() to inject post-step hooks for:
    - Non-blocking GPU→CPU in-memory checkpoint at configurable intervals
    - Layer replay fault detection (SDC + straggler) at configurable intervals

    Uses Megatron's data-parallel group as the natural peer group for both
    replication and detection.

    Args:
        model: List of model chunks (from Megatron's build_model).
        optimizer: Megatron's optimizer (DistributedOptimizer or similar).
        opt_param_scheduler: LR scheduler.
        ckpt_config: In-memory checkpoint configuration. None disables GEMINI.
        detection_config: Replay detection configuration. None disables SCOUT.
        device: CUDA device. Defaults to current device.
        fault_callback: Called with ReplayResult when a fault is detected.
        extra_state_fn: Callable returning extra state to include in checkpoints
            (e.g., iteration, RNG state). Called at each save.
    """

    def __init__(
        self,
        model: list[Any],
        optimizer: Any,
        opt_param_scheduler: Any = None,
        ckpt_config: InMemoryCkptConfig | None = None,
        detection_config: ReplayHarnessConfig | None = None,
        device: torch.device | None = None,
        fault_callback: Callable[[ReplayResult], None] | None = None,
        oob_fault_callback: SCOUTFaultCallback | None = None,
        extra_state_fn: Callable[[], dict[str, Any]] | None = None,
        load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
        durable_checkpoint: DurableCheckpointConfig | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._opt_param_scheduler = opt_param_scheduler
        self._device = device or torch.device("cuda")
        self._extra_state_fn = extra_state_fn
        self._load_extra_state_fn = load_extra_state_fn
        self._step_count = 0
        self._closed = False
        self._recovery_decision_callback: RecoveryDecisionCallback | None = None
        self._last_recovery_decision: RecoveryDecision | None = None

        self._fault_callback = fault_callback

        self._adapter = MegatronAdapter(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
        )

        # Checkpoint shards follow the optimizer topology. Replay peers follow
        # the selected dense or expert model-state role.
        self._checkpoint_group = self._get_checkpoint_group()
        self._checkpoint_gloo_group = create_gloo_peer_group(self._checkpoint_group)
        self._replay_peer_group = ReplayPeerGroup(ReplayPeerRole.DENSE, None, None)
        if detection_config is not None:
            workload = detection_config.workload
            replay_role = workload.peer_role if workload is not None else ReplayPeerRole.DENSE
            self._replay_peer_group = self._adapter.get_replay_peer_group(replay_role)

        # GEMINI: in-memory checkpointing
        self._ckpt_tensors: list[torch.Tensor] | None = None
        self._ckpt_manager, self._ckpt_interval = build_checkpoint_manager(
            ckpt_config,
            manager_factory=InMemoryCheckpointManager,
            parallelism_info_fn=self._adapter.get_parallelism_info,
            process_group=self._checkpoint_gloo_group,
        )
        # SCOUT: layer replay detection
        self._replay_harness: ModelReplayHarness | None = None
        if detection_config is not None:
            layers = self._adapter.get_repeated_layers()
            if layers is not None:
                self._replay_harness = ModelReplayHarness(
                    model=model[0],
                    optimizer=None,
                    device=self._device,
                    config=detection_config,
                    layers=layers,
                    callback=self._fault_callback,
                    oob_fault_callback=oob_fault_callback,
                    peer_group=self._replay_peer_group,
                )
            else:
                logger.warning(
                    "SCOUT: Could not find repeated transformer layers. Detection disabled."
                )

        self._cadence = ResiliencyCadence.from_component_intervals(
            checkpoint_interval=self._ckpt_interval,
            detection_interval=(
                detection_config.check_interval
                if self._replay_harness is not None and detection_config is not None
                else 0
            ),
        )
        self._durable_checkpoint = build_durable_checkpoint(
            durable_checkpoint,
            self._replay_harness,
        )
        self._optimizer_replays = self._build_optimizer_replays()
        self._certification = CheckpointCertificationCoordinator(
            checkpoint_manager=self._ckpt_manager,
            replay_harness=self._replay_harness,
            durable_checkpoint=self._durable_checkpoint,
            cadence=self._cadence,
            checkpoint_tensors=self._checkpoint_tensor_list,
            checkpoint_extra=self._checkpoint_extra,
            fault_callback=self._fault_callback,
            logger=logger,
        )

        # Wrap optimizer step
        self._original_step = optimizer.step
        optimizer.step = self._wrapped_step

    def _wrapped_step(self, *args, **kwargs) -> Any:
        """Wraps optimizer.step() to inject post-step hooks."""
        if self._optimizer_replay_due():
            for replay in self._optimizer_replays:
                replay.arm()
        try:
            result = self._original_step(*args, **kwargs)
        except Exception:
            for replay in self._optimizer_replays:
                replay.cancel()
            raise
        optimizer_step_tensors = collect_optimizer_replays(self._optimizer_replays)
        self._post_step(optimizer_step_tensors)
        return result

    def _post_step(
        self,
        optimizer_step_tensors: OptimizerStepEvidence | None = None,
    ) -> None:
        """Post-optimizer-step hook: checkpoint save + fault detection."""
        self._step_count += 1
        self._certification.post_step(
            self._step_count,
            optimizer_step_tensors=optimizer_step_tensors,
        )

    def try_recover(self, mode: RecoveryMode | str | None = None) -> int:
        """Attempt fast recovery from in-memory checkpoint.

        All ranks must call this collectively. Returns the recovered step
        number, or -1 if no checkpoint found.

        Uses load_tensors() to copy directly into live model/optimizer tensors,
        matching the save_tensors() fast path.
        """
        if self._ckpt_manager is None:
            return -1

        result = self._ckpt_manager.load_tensors(mode=mode)
        if result is None:
            return -1

        saved_tensors, step, extra = result
        prepare_checkpoint_tensor_load(self._adapter, saved_tensors)
        self._adapter.load_checkpoint_tensors(saved_tensors)
        self._step_count = step
        restore_checkpoint_extra(
            extra,
            self._replay_harness,
            self._load_extra_state_fn,
        )
        logger.info(f"GEMINI: recovered from in-memory checkpoint at step {step}")
        return step

    def save_now(self, step: int | None = None) -> None:
        """Force an immediate in-memory checkpoint save."""
        target_step = step or self._step_count
        self._certification.save_now(target_step, check_now=self.check_now)

    def _checkpoint_tensor_list(self) -> list[torch.Tensor]:
        if self._ckpt_tensors is None:
            self._ckpt_tensors = self._adapter.collect_checkpoint_tensors()
        return self._ckpt_tensors

    def _checkpoint_extra(self) -> dict[str, Any]:
        return checkpoint_extra(self._replay_harness, self._extra_state_fn)

    def check_now(self) -> ReplayResult | None:
        """Force an immediate SCOUT detection check."""
        if self._replay_harness is None or not self._replay_harness.has_replay_capture:
            return None
        if self._ckpt_manager is not None or self._durable_checkpoint is not None:
            return self._replay_harness.check_shape_cycle()
        return self._replay_harness.check()

    def prepare_recovery(
        self,
        failure_kind: str,
        *,
        all_ranks_accessible: bool = True,
    ) -> RecoveryMode:
        """Select latest GEMINI or RECOVERY_VERIFIED for restart."""

        def check_all_recipes() -> ReplayResult | None:
            if self._replay_harness is None or not self._replay_harness.has_replay_capture:
                return None
            return self._replay_harness.check_shape_cycle(
                preserve_scheduler=True,
            )

        mode = self._certification.prepare_recovery(
            failure_kind=failure_kind,
            all_ranks_accessible=all_ranks_accessible,
            check_all_recipes=check_all_recipes,
            step=self._step_count,
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
            checkpoint_manager=self._ckpt_manager,
            durable_checkpoint=self._durable_checkpoint,
            allow_collective=allow_collective,
        )

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

    def instrument_dataloader(self, dataloader: Any, *, name: str = "train") -> Any:
        """Return a DataLoader proxy sampled at SCOUT's detection interval."""
        if self._replay_harness is None:
            return dataloader
        return self._replay_harness.instrument_dataloader(dataloader, name=name)

    def checkpoint_io(
        self,
        operation: CheckpointOperation,
        *,
        name: str = "framework",
    ):
        """Return a context manager that publishes checkpoint I/O progress."""
        if self._replay_harness is None:
            return checkpoint_io(None, operation, name=name)
        return self._replay_harness.checkpoint_io(operation, name=name)

    def flush_for_restart(self) -> int:
        """Flush GEMINI's latest recoverable checkpoint to node-local storage."""
        if self._ckpt_manager is None:
            return -1
        return self._ckpt_manager.flush_for_restart()

    def set_restart_destination(
        self,
        resolver: Callable[[], str | Path | None] | None,
    ) -> None:
        """Set GEMINI's optional signal-triggered checkpoint mirror destination."""
        if self._ckpt_manager is not None:
            self._ckpt_manager.set_restart_destination(resolver)

    def copy_checkpoint_to(self, destination: str | Path) -> int:
        """Copy GEMINI's already-flushed own and peer shards to ``destination``."""
        if self._ckpt_manager is None:
            return -1
        return self._ckpt_manager.copy_to(destination)

    @property
    def step_count(self) -> int:
        return self._step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self._step_count = value

    def close(self) -> None:
        """Cleanup: restore original optimizer step and release resources."""
        if self._closed:
            return
        self._closed = True
        unregister_automatic_cleanup(self)
        self._optimizer.step = self._original_step
        for replay in self._optimizer_replays:
            replay.remove()
        if self._ckpt_manager is not None:
            self._ckpt_manager.close()
        if self._replay_harness is not None:
            self._replay_harness.remove_hooks()

    def _optimizer_replay_due(self) -> bool:
        return bool(
            self._optimizer_replays
            and self._replay_harness is not None
            and self._replay_harness.has_capture
            and self._replay_harness.optimizer_replay_due(self._step_count + 1)
        )

    def _build_optimizer_replays(self) -> list[OptimizerStepReplay]:
        if self._replay_harness is None:
            return []
        replays: list[OptimizerStepReplay] = []
        try:
            for optimizer in self._adapter.get_base_optimizers():
                replays.append(OptimizerStepReplay(optimizer))
            return replays
        except (AttributeError, OptimizerStepCheckUnsupported) as exc:
            for replay in replays:
                replay.remove()
            logger.warning(
                "SCOUT Megatron optimizer-step replay unavailable: %s. "
                "Layer replay remains enabled.",
                exc,
            )
            return []

    def _get_checkpoint_group(self) -> dist.ProcessGroup | None:
        """Get the narrowest group whose ranks own corresponding optimizer state."""
        optimizer_replica_group = self._adapter.get_optimizer_replica_group()
        if optimizer_replica_group is not None:
            logger.info(
                "SCOUT using Megatron's inter-optimizer-instance group for "
                "corresponding optimizer-shard comparison"
            )
            return optimizer_replica_group
        try:
            from megatron.core import mpu

            return mpu.get_data_parallel_group()
        except (ImportError, AssertionError):
            return None


def enable_resiliency(
    model: list[Any],
    optimizer: Any,
    opt_param_scheduler: Any = None,
    *,
    interval: int = 10,
    enable_checkpoint: bool = True,
    enable_detection: bool = True,
    ckpt_config: InMemoryCkptConfig | None = None,
    detection_config: ReplayHarnessConfig | None = None,
    device: torch.device | None = None,
    fault_callback: Callable[[ReplayResult], None] | None = None,
    oob_fault_callback: SCOUTFaultCallback | None = None,
    orchestration: OrchestrationHooks | None = None,
    extra_state_fn: Callable[[], dict[str, Any]] | None = None,
    load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
    load_fallback: Callable[[], int | None] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
) -> MegatronResiliency:
    """Enable GEMINI + SCOUT for a megatron-core training job. One call.

    Wraps the optimizer to inject post-step hooks. Handles recovery
    automatically — tries in-memory checkpoint first, falls back to the
    user-provided loader if nothing found.

    Example:
        resiliency = enable_megatron_resiliency(
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            interval=10,
            load_fallback=lambda: load_checkpoint_from_disk(),
        )

        # Training loop unchanged — recovery handled internally
        for iteration in range(resiliency.step_count, end):
            train_step(...)

    Args:
        model: List of model chunks from Megatron's build_model.
        optimizer: Megatron's optimizer (DistributedOptimizer).
        opt_param_scheduler: Megatron's LR scheduler.
        interval: SCOUT replay cadence and checkpoint-certification interval.
        enable_checkpoint: Enable GEMINI.
        enable_detection: Enable SCOUT.
        ckpt_config: Advanced checkpoint settings; cadence is set by interval.
        detection_config: Advanced detection settings; cadence is set by interval.
        device: CUDA device.
        fault_callback: Called with ReplayResult on fault detection.
        oob_fault_callback: Called with JSON-ready hang and DataLoader-stall reports.
        orchestration: Platform-neutral manager fault and restart hooks.
        extra_state_fn: Returns extra state to include in each checkpoint.
        load_extra_state_fn: Restores caller-owned iteration, sampler, or dataset
            state after GEMINI recovery.
        load_fallback: Called when no in-memory checkpoint is found. Should load
            from disk and return the recovered step number (or None for fresh start).
        durable_checkpoint: SCOUT-gated framework checkpoint callbacks and manifest.

    Returns:
        MegatronResiliency handle. Use resiliency.step_count for the resume point.
    """
    ckpt_config, detection_config = _normalize_feature_configs(
        interval,
        enable_checkpoint=enable_checkpoint,
        enable_detection=enable_detection,
        checkpoint=ckpt_config,
        replay=detection_config,
    )
    fault_callback, oob_fault_callback = _resolve_orchestration_callbacks(
        orchestration,
        fault_callback,
        oob_fault_callback,
    )

    resiliency = MegatronResiliency(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        ckpt_config=ckpt_config,
        detection_config=detection_config,
        device=device,
        fault_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        extra_state_fn=extra_state_fn,
        load_extra_state_fn=load_extra_state_fn,
        durable_checkpoint=durable_checkpoint,
    )
    _bind_orchestration(orchestration, resiliency)

    recover_with_fallback(resiliency, load_fallback, recovery_mode)

    register_automatic_cleanup(resiliency)
    return resiliency


# Backwards-compatible alias
enable_megatron_resiliency = enable_resiliency
