# mypy: ignore-errors
"""Native PyTorch integration for DDP, FSDP2, and HSDP."""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency._feature_wiring import (
    _normalize_feature_configs,
    _wire_features,
)
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig, ReplayResult
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.handle import ResiliencyHandle
from lm_resiliency.integrations.pytorch.fsdp import (
    PyTorchFSDPResiliency,
    enable_fsdp2_resiliency,
    has_dtensor_params,
    has_fsdp_modules,
)
from lm_resiliency.lifecycle import register_automatic_cleanup
from lm_resiliency.orchestration import (
    OrchestrationHooks,
    _bind_orchestration,
    _resolve_orchestration_callbacks,
)


def enable_resiliency(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    interval: int = 10,
    enable_checkpoint: bool = True,
    enable_detection: bool = True,
    checkpoint: InMemoryCkptConfig | None = None,
    replay: ReplayHarnessConfig | None = None,
    group: dist.ProcessGroup | None = None,
    nccl_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    parallelism_info: Any | None = None,
    fault_callback: Callable[[ReplayResult], None] | None = None,
    oob_fault_callback: SCOUTFaultCallback | None = None,
    orchestration: OrchestrationHooks | None = None,
    extra_state_fn: Callable[[], dict[str, Any]] | None = None,
    load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
    load_fallback: Callable[[], Any] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
    _step_hook_registrar: Callable[[Callable[[Any, Any, Any], None]], Any] | None = None,
) -> ResiliencyHandle:
    """Enable resiliency for replicated, DDP, FSDP2, or HSDP training."""
    checkpoint, replay = _normalize_feature_configs(
        interval,
        enable_checkpoint=enable_checkpoint,
        enable_detection=enable_detection,
        checkpoint=checkpoint,
        replay=replay,
    )
    if replay is not None and (group is None) != (nccl_group is None):
        raise ValueError("group and nccl_group must be supplied together")

    if has_dtensor_params(model):
        handle = enable_fsdp2_resiliency(
            model,
            optimizer,
            ckpt_config=checkpoint,
            detection_config=replay,
            device=device,
            fault_callback=fault_callback,
            oob_fault_callback=oob_fault_callback,
            orchestration=orchestration,
            group=group,
            nccl_group=nccl_group,
            load_fallback=load_fallback,
            parallelism_info=parallelism_info,
            extra_state_fn=extra_state_fn,
            load_extra_state_fn=load_extra_state_fn,
            durable_checkpoint=durable_checkpoint,
            recovery_mode=recovery_mode,
            step_hook_registrar=_step_hook_registrar,
        )
        register_automatic_cleanup(handle)
        return handle

    fault_callback, oob_fault_callback = _resolve_orchestration_callbacks(
        orchestration,
        fault_callback,
        oob_fault_callback,
    )
    handle = _wire_features(
        model,
        optimizer,
        checkpoint=checkpoint,
        replay=replay,
        group=group,
        nccl_group=nccl_group,
        device=device,
        parallelism_info=parallelism_info,
        replay_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        extra_state_fn=extra_state_fn,
        load_extra_state_fn=load_extra_state_fn,
        load_fallback=load_fallback,
        durable_checkpoint=durable_checkpoint,
        recovery_mode=recovery_mode,
        step_hook_registrar=_step_hook_registrar,
    )
    _bind_orchestration(orchestration, handle)
    register_automatic_cleanup(handle)
    return handle


__all__ = [
    "PyTorchFSDPResiliency",
    "enable_resiliency",
    "has_fsdp_modules",
]
