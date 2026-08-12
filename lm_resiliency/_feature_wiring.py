"""Internal feature configuration and framework-agnostic hook wiring."""

from __future__ import annotations

import logging
from dataclasses import replace
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointConfig,
    DurableCheckpointCoordinator,
)
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager, RecoveryMode
from lm_resiliency.checkpointing.rng import RNG_KEY, capture_rng_state, restore_rng_state
from lm_resiliency.detection.peer_group import (
    form_detection_groups,
    parallelism_device_mesh,
)
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    ReplayHarnessConfig,
    ReplayResult,
)
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.detection.temporal import SCOUT_TEMPORAL_KEY
from lm_resiliency.handle import ResiliencyHandle
from lm_resiliency.integrations._checkpoint_certification import (
    CheckpointCertificationCoordinator,
)

logger = logging.getLogger(__name__)


def _normalize_feature_configs(
    interval: int,
    *,
    enable_checkpoint: bool,
    enable_detection: bool,
    checkpoint: InMemoryCkptConfig | None,
    replay: ReplayHarnessConfig | None,
) -> tuple[InMemoryCkptConfig | None, ReplayHarnessConfig | None]:
    """Build component configs from one public SCOUT-cycle interval."""
    if interval <= 0:
        raise ValueError("interval must be greater than zero")

    if enable_checkpoint:
        checkpoint = replace(
            checkpoint or InMemoryCkptConfig(),
            enable=True,
            interval=interval,
        )
    else:
        checkpoint = None

    if enable_detection:
        replay = replace(
            replay or ReplayHarnessConfig(),
            check_interval=interval,
        )
    else:
        replay = None

    return checkpoint, replay


def _wire_features(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    checkpoint: InMemoryCkptConfig | None = None,
    replay: ReplayHarnessConfig | None = None,
    group: dist.ProcessGroup | None = None,
    nccl_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    parallelism_info: Any | None = None,
    replay_callback: Callable[[ReplayResult], None] | None = None,
    oob_fault_callback: SCOUTFaultCallback | None = None,
    extra_state_fn: Callable[[], dict] | None = None,
    load_extra_state_fn: Callable[[dict], None] | None = None,
    load_fallback: Callable[[], None] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
) -> ResiliencyHandle:
    """Wire GEMINI checkpointing and/or SCOUT detection onto the optimizer.

    See :func:`enable_resiliency` for the public argument reference.

    Fire-and-forget: attach once, then run your training loop unchanged.
    Handles recovery automatically — tries in-memory checkpoint first,
    falls back to the user-provided loader if nothing found.

        enable_resiliency(
            model, optimizer,
            interval=10,
            checkpoint=InMemoryCkptConfig(replication_jump=8),
            replay=ReplayHarnessConfig(rotate_layers=True),
            group=dp_gloo_pg, nccl_group=dp_nccl_pg,
            load_fallback=lambda: checkpointer.load(step=cfg.checkpoint.load_step),
        )

    Args:
        model: The training model.
        optimizer: Training optimizer (hooks attach here).
        checkpoint: Advanced in-memory checkpoint settings. Its interval and enable
            fields are controlled by the unified API.
        replay: Advanced layer replay settings. Its check interval is controlled by
            the unified API.
        group: Gloo process group (for C3 scalar ops and coordination).
        nccl_group: NCCL process group (for GPU tensor broadcast/allreduce).
        device: CUDA device. Defaults to current device.
        parallelism_info: Object with dp_replicate/dp_shard attrs (for HSDP skip).
        replay_callback: Called with ReplayResult when a fault is detected.
        oob_fault_callback: Called with JSON-ready hang and DataLoader-stall reports.
        extra_state_fn: Returns a dict of *extra* training state to checkpoint
            alongside model+optimizer — dataloader/sampler state_dict, RNG state,
            LR scheduler, step/token counts. Called synchronously at each
            checkpoint step; its accessors (e.g. torch.get_rng_state,
            dataloader.state_dict) return owned snapshots, so it is cheap and
            captures a consistent step-boundary view. Tensor *and* non-tensor
            values are handled: both are versioned per checkpoint slot and
            replicated to the peer, so they survive both a local reload and peer
            recovery. Keep model/optimizer OUT of it — they stay
            on GEMINI's reference path (no full state_dict gather).
        load_extra_state_fn: Called on recovery with the restored extra-state dict
            so the caller can apply it (dataloader.load_state_dict,
            torch.set_rng_state, scheduler.load_state_dict). Makes a GEMINI
            recovery equivalent to the framework's own resume.
        load_fallback: Called when no in-memory checkpoint is found. Typically
            your existing checkpoint loader (e.g., torchtitan's checkpointer.load).
            If None and no in-memory checkpoint exists, training starts fresh.
            Ignored when durable_checkpoint is configured because an unconstrained
            latest-checkpoint loader could select an unvalidated candidate.
        durable_checkpoint: Framework checkpoint callbacks and shared manifest
            storage for SCOUT-gated durable candidate/commit.

    Returns:
        Public lifecycle handle for the enabled features.
    """
    state = ResiliencyHandle()
    checkpoint_interval = 0
    restored_temporal_state = None

    # Auto-discover peer group if not provided
    if group is None and nccl_group is None:
        if dist.is_initialized():
            topology_model = model
            if replay is not None and replay.workload is not None:
                replay_modules = replay.workload.replay_modules
                if replay_modules:
                    topology_model = replay_modules[0]
            expert = bool(
                replay is not None
                and replay.workload is not None
                and replay.workload.peer_role.value == "expert"
            )
            group, nccl_group = form_detection_groups(
                topology_model,
                device_mesh=parallelism_device_mesh(
                    parallelism_info,
                    expert=expert,
                ),
            )
            logger.info("Auto-discovered detection peer groups from model topology")

    if checkpoint is not None and checkpoint.enable and checkpoint.interval > 0:
        state.ckpt_manager = InMemoryCheckpointManager(
            config=checkpoint,
            parallelism_info=parallelism_info,
            process_group=group,
        )
        checkpoint_interval = checkpoint.interval

    if replay is not None:
        state.replay_harness = ModelReplayHarness(
            model,
            optimizer=None,
            group=group,
            nccl_group=nccl_group,
            device=device,
            config=replay,
            callback=replay_callback,
            oob_fault_callback=oob_fault_callback,
        )

    if durable_checkpoint is not None:
        if state.replay_harness is None:
            raise ValueError("durable checkpoint certification requires SCOUT replay detection")
        state.durable_checkpoint = DurableCheckpointCoordinator(
            durable_checkpoint,
            shape_plan_id=state.replay_harness.replay_shape_plan_id,
            shape_ids=[shape.shape_id for shape in state.replay_harness.replay_shapes],
            checkpoint_io=lambda operation, name: state.replay_harness.checkpoint_io(
                operation,
                name=name,
            ),
        )

    recovered = False
    if state.ckpt_manager is not None:
        result = state.ckpt_manager.load(mode=recovery_mode)
        if result is not None:
            sd, step = result
            model.load_state_dict(sd["model"])
            optimizer.load_state_dict(sd["optimizer"])
            extra = sd.get("extra") or {}
            restored_temporal_state = extra.get(SCOUT_TEMPORAL_KEY)
            if load_extra_state_fn is not None:
                load_extra_state_fn(
                    {
                        key: value
                        for key, value in extra.items()
                        if key not in (RNG_KEY, SCOUT_TEMPORAL_KEY)
                    }
                )
            # Restore RNG last so the first resumed stochastic forward matches.
            restore_rng_state(extra.get(RNG_KEY))
            state._restore_step(step)
            recovered = True
            logger.info(f"Recovered from in-memory checkpoint at step {step}")

    if not recovered and state.durable_checkpoint is not None:
        step = state.durable_checkpoint.load_latest_validated()
        if step is not None:
            state._restore_step(step)
            recovered = True
            logger.info("Recovered SCOUT-validated durable checkpoint at step %s", step)

    if not recovered and state.durable_checkpoint is None and load_fallback is not None:
        with state.checkpoint_io("read", name="fallback"):
            load_fallback()

    if state.replay_harness is not None:
        state.replay_harness.load_temporal_state_dict(restored_temporal_state)

    cadence = ResiliencyCadence.from_component_intervals(
        checkpoint_interval=checkpoint_interval,
        detection_interval=(
            replay.check_interval if state.replay_harness is not None and replay is not None else 0
        ),
    )

    def _save_in_memory(step: int) -> None:
        if state.ckpt_manager is None:
            return
        sd = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
        extra = dict(extra_state_fn()) if extra_state_fn is not None else {}
        extra[RNG_KEY] = capture_rng_state()
        if state.replay_harness is not None:
            extra[SCOUT_TEMPORAL_KEY] = state.replay_harness.temporal_state_dict()
        sd["extra"] = extra
        state.ckpt_manager.save(sd, step)

    certification = CheckpointCertificationCoordinator(
        checkpoint_manager=state.ckpt_manager,
        replay_harness=state.replay_harness,
        durable_checkpoint=state.durable_checkpoint,
        cadence=cadence,
        checkpoint_tensors=lambda: [],
        checkpoint_extra=lambda: {},
        checkpoint_save=_save_in_memory,
        fault_callback=replay_callback,
        logger=logger,
    )
    state._set_prepare_recovery_callback(
        lambda failure_kind, all_ranks_accessible, step: certification.prepare_recovery(
            failure_kind=failure_kind,
            all_ranks_accessible=all_ranks_accessible,
            check_all_recipes=(
                lambda: (
                    state.replay_harness.check_shape_cycle(
                        optimizer=optimizer,
                        preserve_scheduler=True,
                    )
                    if state.replay_harness is not None and state.replay_harness.has_replay_capture
                    else None
                )
            ),
            step=step,
        )
    )

    def _unified_post_step_hook(opt, args, kwargs) -> None:
        del args, kwargs
        step = state._advance_step()
        certification.post_step(step, optimizer=opt)

    hook = optimizer.register_step_post_hook(_unified_post_step_hook)
    state._register_hook(hook)

    return state
