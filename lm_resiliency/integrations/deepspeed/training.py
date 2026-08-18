"""DeepSpeed training integration for GEMINI + SCOUT.

Provides hook-based integration into DeepSpeed's training loop without
requiring modifications to the DeepSpeed source. Wraps engine.step() to
inject post-step callbacks.

Example:
    from lm_resiliency.integrations.deepspeed import enable_deepspeed_resiliency

    # After deepspeed.initialize():
    model_engine, optimizer, _, _ = deepspeed.initialize(...)

    resiliency = enable_deepspeed_resiliency(
        engine=model_engine,
        interval=10,
    )

    # Fast recovery before training loop
    recovered_step = resiliency.try_recover()
    if recovered_step > 0:
        global_step = recovered_step

    # Training loop runs unchanged — engine.step() is wrapped internally
    for step in range(start, end):
        loss = model_engine(batch)
        model_engine.backward(loss)
        model_engine.step()
"""

from __future__ import annotations

import copy
import logging
from dataclasses import replace
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
from lm_resiliency.integrations.deepspeed.adapter import DeepSpeedAdapter
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
_UNSET = object()
_DEEPSPEED_LR_SCHEDULER_KEY = "deepspeed_lr_scheduler"


class DeepSpeedResiliency:
    """Manages GEMINI checkpointing and SCOUT detection for DeepSpeed.

    Wraps the engine's step() to inject post-step hooks for:
    - Non-blocking GPU→CPU in-memory checkpoint at configurable intervals
    - Layer replay fault detection (SDC + straggler) at configurable intervals

    Uses DeepSpeed's data-parallel group as the natural peer group for both
    replication and detection.

    Args:
        engine: DeepSpeed engine instance (from deepspeed.initialize()).
        ckpt_config: In-memory checkpoint configuration. None disables GEMINI.
        detection_config: Replay detection configuration. None disables SCOUT.
        device: CUDA device. Defaults to current device.
        fault_callback: Called with ReplayResult when a fault is detected.
    """

    def __init__(
        self,
        engine: Any,
        ckpt_config: InMemoryCkptConfig | None = None,
        detection_config: ReplayHarnessConfig | None = None,
        device: torch.device | None = None,
        fault_callback: Callable[[ReplayResult], None] | None = None,
        oob_fault_callback: SCOUTFaultCallback | None = None,
        durable_checkpoint: DurableCheckpointConfig | None = None,
        expected_topology_id: str | None = None,
    ) -> None:
        self._engine = engine
        self._device = device or torch.device("cuda")
        self._step_count = 0
        self._closed = False
        self._recovery_decision_callback: RecoveryDecisionCallback | None = None
        self._last_recovery_decision: RecoveryDecision | None = None

        self._fault_callback = fault_callback

        self._adapter = DeepSpeedAdapter(engine)

        # Checkpoint state follows DeepSpeed's DP/ZeRO group. Replay selects
        # dense-DP or expert-DP peers while fixing model-parallel coordinates.
        self._dp_group = self._adapter.get_dp_group()
        self._dp_gloo_group = self._create_gloo_group()
        self._replay_peer_group = ReplayPeerGroup(ReplayPeerRole.DENSE, None, None)
        if detection_config is not None:
            workload = detection_config.workload
            replay_role = workload.peer_role if workload is not None else ReplayPeerRole.DENSE
            replay_modules = workload.replay_modules if workload is not None else ()
            self._replay_peer_group = self._adapter.get_replay_peer_group(
                replay_role,
                replay_modules,
            )

        # GEMINI: in-memory checkpointing
        self._ckpt_tensors: list[torch.Tensor] | None = None
        self._ckpt_manager, self._ckpt_interval = build_checkpoint_manager(
            ckpt_config,
            manager_factory=InMemoryCheckpointManager,
            parallelism_info_fn=self._adapter.get_parallelism_info,
            process_group=self._dp_gloo_group,
            expected_topology_id=expected_topology_id,
        )
        # SCOUT: layer replay detection
        self._replay_harness: ModelReplayHarness | None = None
        if detection_config is not None:
            layers = self._adapter.get_repeated_layers()
            if layers is not None:
                parallelism = self._adapter.get_parallelism_info()
                replay_config = (
                    replace(detection_config, compare_parameter_state=False)
                    if parallelism.dp_shard > 1
                    else detection_config
                )
                self._replay_harness = ModelReplayHarness(
                    model=engine.module,
                    optimizer=None,
                    device=self._device,
                    config=replay_config,
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

        # PipelineEngine performs its optimizer boundary inside train_batch() and
        # intentionally disables step(). Hook that boundary directly.
        self._is_pipeline_engine = callable(getattr(engine, "_exec_optimizer_step", None)) and any(
            cls.__name__ == "PipelineEngine" for cls in type(engine).__mro__
        )
        self._step_attribute = "_exec_optimizer_step" if self._is_pipeline_engine else "step"
        self._original_step = getattr(engine, self._step_attribute)
        setattr(engine, self._step_attribute, self._wrapped_step)
        self._original_instruction_map: Any = _UNSET
        if self._is_pipeline_engine:
            self._install_pipeline_optimizer_instruction()

    def _wrapped_step(self, *args, **kwargs) -> Any:
        """Wrap the framework optimizer boundary to inject post-step hooks."""
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
        """Post-step hook: checkpoint save + fault detection."""
        self._step_count += 1
        try:
            self._certification.post_step(
                self._step_count,
                optimizer_step_tensors=optimizer_step_tensors,
            )
        finally:
            self._release_zero3_replay_parameters()

    def try_recover(
        self,
        mode: RecoveryMode | str | None = None,
        *,
        step: int | None = None,
    ) -> int:
        """Attempt fast recovery from in-memory checkpoint.

        All ranks must call this collectively. Returns the recovered step
        number, or -1 if no checkpoint found.

        Uses load_tensors() to copy directly into live engine tensors,
        matching the save_tensors() fast path.
        """
        if self._ckpt_manager is None:
            return -1

        result = self._ckpt_manager.load_tensors(mode=mode, step=step)
        if result is None:
            return -1

        saved_tensors, step, extra = result
        prepare_checkpoint_tensor_load(self._adapter, saved_tensors)
        self._adapter.load_checkpoint_tensors(saved_tensors)
        self._step_count = step
        self._engine.global_steps = step
        restore_checkpoint_extra(
            extra,
            self._replay_harness,
            self._restore_framework_extra_state,
        )
        logger.info(f"GEMINI: recovered from in-memory checkpoint at step {step}")
        return step

    def save_now(self, step: int | None = None) -> None:
        """Force an immediate in-memory checkpoint save."""
        target_step = step or self._step_count
        try:
            self._certification.save_now(target_step, check_now=self.check_now)
        finally:
            self._release_zero3_replay_parameters()

    def _checkpoint_tensor_list(self) -> list[torch.Tensor]:
        if self._ckpt_tensors is None:
            self._ckpt_tensors = self._adapter.collect_checkpoint_tensors()
        return self._ckpt_tensors

    def _checkpoint_extra(self) -> dict[str, Any]:
        return checkpoint_extra(
            self._replay_harness,
            self._capture_framework_extra_state,
        )

    def _capture_framework_extra_state(self) -> dict[str, Any]:
        scheduler = getattr(self._engine, "lr_scheduler", None)
        if scheduler is None:
            return {}
        state_dict = getattr(scheduler, "state_dict", None)
        if not callable(state_dict):
            raise RuntimeError("DeepSpeed lr_scheduler does not provide state_dict()")
        state = state_dict()
        if not isinstance(state, dict):
            raise RuntimeError("DeepSpeed lr_scheduler.state_dict() must return a dictionary")
        return {_DEEPSPEED_LR_SCHEDULER_KEY: copy.deepcopy(state)}

    def _restore_framework_extra_state(self, state: dict[str, Any]) -> None:
        scheduler = getattr(self._engine, "lr_scheduler", None)
        saved = state.get(_DEEPSPEED_LR_SCHEDULER_KEY, _UNSET)
        if scheduler is None:
            if saved is not _UNSET:
                raise RuntimeError(
                    "checkpoint contains DeepSpeed scheduler state but the engine has no scheduler"
                )
            return
        if saved is _UNSET:
            raise RuntimeError("checkpoint is missing DeepSpeed lr_scheduler state")
        if not isinstance(saved, dict):
            raise RuntimeError("checkpoint DeepSpeed lr_scheduler state must be a dictionary")
        load_state_dict = getattr(scheduler, "load_state_dict", None)
        if not callable(load_state_dict):
            raise RuntimeError("DeepSpeed lr_scheduler does not provide load_state_dict()")
        load_state_dict(copy.deepcopy(saved))

    def check_now(self) -> ReplayResult | None:
        """Force an immediate SCOUT detection check."""
        if self._replay_harness is None or not self._replay_harness.has_replay_capture:
            return None
        try:
            if self._ckpt_manager is not None or self._durable_checkpoint is not None:
                return self._replay_harness.check_shape_cycle()
            return self._replay_harness.check()
        finally:
            self._release_zero3_replay_parameters()

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
            try:
                return self._replay_harness.check_shape_cycle(
                    preserve_scheduler=True,
                )
            finally:
                self._release_zero3_replay_parameters()

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
        """Cleanup: restore original engine step and release resources."""
        if self._closed:
            return
        self._closed = True
        unregister_automatic_cleanup(self)
        setattr(self._engine, self._step_attribute, self._original_step)
        if self._original_instruction_map is _UNSET:
            self._engine.__dict__.pop("_INSTRUCTION_MAP", None)
        else:
            self._engine._INSTRUCTION_MAP = self._original_instruction_map
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
                "SCOUT DeepSpeed optimizer-step replay unavailable: %s. "
                "Layer replay remains enabled.",
                exc,
            )
            return []

    def _create_gloo_group(self) -> dist.ProcessGroup | None:
        """Create a Gloo-backend group matching the DP group for C³ scalar ops."""
        return create_gloo_peer_group(self._dp_group)

    def _install_pipeline_optimizer_instruction(self) -> None:
        """Replace PipelineEngine's instruction-table entry for optimizer steps."""
        instruction_map = self._engine._INSTRUCTION_MAP
        optimizer_instruction = next(
            (
                instruction
                for instruction in instruction_map
                if instruction.__name__ == "OptimizerStep"
            ),
            None,
        )
        if optimizer_instruction is None:
            raise RuntimeError("DeepSpeed PipelineEngine has no OptimizerStep instruction")
        self._original_instruction_map = self._engine.__dict__.get(
            "_INSTRUCTION_MAP",
            _UNSET,
        )
        local_map = dict(instruction_map)

        def wrapped_optimizer_instruction(engine, *args, **kwargs):
            del engine
            return self._wrapped_step(*args, **kwargs)

        local_map[optimizer_instruction] = wrapped_optimizer_instruction
        self._engine._INSTRUCTION_MAP = local_map

    def _release_zero3_replay_parameters(self) -> None:
        """Release module claims left by direct replay outside ``engine.backward``."""
        if self._adapter._zero_stage() < 3 or self._replay_harness is None:
            return
        parameter_offload = getattr(self._engine.optimizer, "parameter_offload", None)
        release = getattr(parameter_offload, "release_backward_leftovers", None)
        if not callable(release):
            raise RuntimeError(
                "DeepSpeed ZeRO-3 optimizer does not expose "
                "parameter_offload.release_backward_leftovers()"
            )
        release()


def enable_resiliency(
    engine: Any,
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
    load_fallback: Callable[[], int | None] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
    _recovery_step: int | None = None,
    _expected_topology_id: str | None = None,
) -> DeepSpeedResiliency:
    """Enable GEMINI + SCOUT for a DeepSpeed training job. One call.

    Wraps engine.step() to inject post-step hooks. Handles recovery
    automatically — tries in-memory checkpoint first, falls back to the
    user-provided loader if nothing found.

    Example:
        model_engine, optimizer, _, _ = deepspeed.initialize(...)

        resiliency = enable_deepspeed_resiliency(
            engine=model_engine,
            interval=10,
            load_fallback=lambda: load_from_disk(),
        )

        # Training loop unchanged — recovery handled internally
        for step in range(resiliency.step_count, end):
            loss = model_engine(batch)
            model_engine.backward(loss)
            model_engine.step()

    Args:
        engine: DeepSpeed engine (from deepspeed.initialize()).
        interval: SCOUT replay cadence and checkpoint-certification interval.
        enable_checkpoint: Enable GEMINI.
        enable_detection: Enable SCOUT.
        ckpt_config: Advanced checkpoint settings; cadence is set by interval.
        detection_config: Advanced detection settings; cadence is set by interval.
        device: CUDA device.
        fault_callback: Called with ReplayResult on fault detection.
        oob_fault_callback: Called with JSON-ready hang and DataLoader-stall reports.
        orchestration: Platform-neutral manager fault and restart hooks.
        load_fallback: Called when no in-memory checkpoint is found. Should load
            from disk and return the recovered step number (or None for fresh start).
        durable_checkpoint: SCOUT-gated framework checkpoint callbacks and manifest.

    Returns:
        DeepSpeedResiliency handle. Use resiliency.step_count for the resume point.
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

    resiliency = DeepSpeedResiliency(
        engine=engine,
        ckpt_config=ckpt_config,
        detection_config=detection_config,
        device=device,
        fault_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        durable_checkpoint=durable_checkpoint,
        expected_topology_id=_expected_topology_id,
    )
    _bind_orchestration(orchestration, resiliency)

    recover_with_fallback(resiliency, load_fallback, recovery_mode, _recovery_step)

    register_automatic_cleanup(resiliency)
    return resiliency


# Backwards-compatible alias
enable_deepspeed_resiliency = enable_resiliency
