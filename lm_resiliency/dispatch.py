# mypy: ignore-errors
"""Automatic framework dispatch for the package-root public API."""

from __future__ import annotations

from typing import Any, Callable, Literal

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig, ReplayResult
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.handle import ResiliencySession
from lm_resiliency.orchestration import OrchestrationHooks

FrameworkName = Literal["auto", "pytorch", "torchtitan", "megatron", "deepspeed"]


def enable_resiliency(
    model: Any,
    optimizer: Any | None = None,
    *,
    opt_param_scheduler: Any | None = None,
    framework: FrameworkName = "auto",
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
) -> ResiliencySession:
    """Enable resiliency through the matching framework integration.

    Framework selection is automatic for DeepSpeed engines and Megatron model-chunk
    lists. Native PyTorch modules, including DDP, FSDP2, and HSDP, use the PyTorch
    integration. Set ``framework="torchtitan"`` only when a TorchTitan-specific
    wrapper requires that compatibility entry point.
    """
    selected = _select_framework(model, framework)

    common = {
        "interval": interval,
        "enable_checkpoint": enable_checkpoint,
        "enable_detection": enable_detection,
        "device": device,
        "fault_callback": fault_callback,
        "oob_fault_callback": oob_fault_callback,
        "orchestration": orchestration,
        "load_fallback": load_fallback,
        "durable_checkpoint": durable_checkpoint,
        "recovery_mode": recovery_mode,
    }

    if selected == "deepspeed":
        _reject_options(
            selected,
            optimizer=optimizer,
            opt_param_scheduler=opt_param_scheduler,
            group=group,
            nccl_group=nccl_group,
            parallelism_info=parallelism_info,
            extra_state_fn=extra_state_fn,
            load_extra_state_fn=load_extra_state_fn,
        )
        from lm_resiliency.integrations.deepspeed import (
            enable_resiliency as enable_deepspeed,
        )

        return enable_deepspeed(
            model,
            ckpt_config=checkpoint,
            detection_config=replay,
            **common,
        )

    if optimizer is None and selected != "torchtitan":
        raise TypeError(f"{selected} resiliency requires an optimizer")

    if selected == "megatron":
        _reject_options(
            selected,
            group=group,
            nccl_group=nccl_group,
            parallelism_info=parallelism_info,
        )
        from lm_resiliency.integrations.megatron import (
            enable_resiliency as enable_megatron,
        )

        model_chunks = list(model) if isinstance(model, tuple) else model
        return enable_megatron(
            model_chunks,
            optimizer,
            opt_param_scheduler=opt_param_scheduler,
            ckpt_config=checkpoint,
            detection_config=replay,
            extra_state_fn=extra_state_fn,
            load_extra_state_fn=load_extra_state_fn,
            **common,
        )

    if selected == "torchtitan":
        _reject_options(
            selected,
            opt_param_scheduler=opt_param_scheduler,
            group=group,
            nccl_group=nccl_group,
            extra_state_fn=extra_state_fn,
            load_extra_state_fn=load_extra_state_fn,
        )
        from lm_resiliency.integrations.torchtitan import (
            enable_resiliency as enable_torchtitan,
        )

        return enable_torchtitan(
            model,
            optimizer,
            ckpt_config=checkpoint,
            detection_config=replay,
            parallelism_info=parallelism_info,
            **common,
        )

    _reject_options(
        selected,
        opt_param_scheduler=opt_param_scheduler,
    )
    from lm_resiliency.api import enable_resiliency as enable_pytorch

    return enable_pytorch(
        model,
        optimizer,
        checkpoint=checkpoint,
        replay=replay,
        group=group,
        nccl_group=nccl_group,
        parallelism_info=parallelism_info,
        extra_state_fn=extra_state_fn,
        load_extra_state_fn=load_extra_state_fn,
        **common,
    )


def _select_framework(target: Any, framework: FrameworkName) -> str:
    if framework not in {"auto", "pytorch", "torchtitan", "megatron", "deepspeed"}:
        raise ValueError(f"unsupported framework: {framework!r}")
    if framework != "auto":
        return framework
    if _is_deepspeed_engine(target):
        return "deepspeed"
    if _is_torchtitan_trainer(target):
        return "torchtitan"
    if isinstance(target, (list, tuple)):
        return "megatron"
    if isinstance(target, nn.Module):
        return "pytorch"
    raise TypeError(
        "could not infer the training framework; pass framework= with a supported value"
    )


def _is_torchtitan_trainer(target: Any) -> bool:
    return all(
        hasattr(target, attribute)
        for attribute in (
            "model_parts",
            "optimizers",
            "lr_schedulers",
            "parallel_dims",
            "checkpointer",
            "train",
        )
    )


def _is_deepspeed_engine(target: Any) -> bool:
    module_name = type(target).__module__.split(".", 1)[0]
    if module_name == "deepspeed":
        return True
    return (
        hasattr(target, "module")
        and hasattr(target, "optimizer")
        and callable(getattr(target, "step", None))
        and callable(getattr(target, "zero_optimization_stage", None))
    )


def _reject_options(framework: str, **options: Any) -> None:
    unsupported = sorted(name for name, value in options.items() if value is not None)
    if unsupported:
        joined = ", ".join(unsupported)
        raise TypeError(f"{framework} integration does not accept: {joined}")


__all__ = ["FrameworkName", "enable_resiliency"]
