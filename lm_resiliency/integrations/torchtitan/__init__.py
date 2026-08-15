# mypy: ignore-errors
"""torchtitan integration for lm_resiliency.

Provides GEMINI (in-memory checkpointing) and SCOUT (fault detection)
for torchtitan distributed training.

Usage:
    from lm_resiliency.integrations.torchtitan import enable_resiliency

    enable_resiliency(
        model, optimizer,
        interval=10,
        load_fallback=lambda: checkpointer.load(),
    )
"""

from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn
import torch.optim

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig, ReplayResult
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.integrations.pytorch import (
    enable_resiliency as _enable_pytorch_resiliency,
)
from lm_resiliency.integrations.torchtitan.adapter import TorchTitanAdapter
from lm_resiliency.orchestration import OrchestrationHooks


def enable_resiliency(
    model: nn.Module | Any,
    optimizer: torch.optim.Optimizer | None = None,
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
    parallelism_info: Any | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
) -> Any:
    """Enable GEMINI + SCOUT for a torchtitan training job. One call.

    Uses optimizer post-step hooks — no changes to the training loop.
    Recovery is automatic: tries in-memory checkpoint first, falls back
    to the user-provided loader if nothing found.

    Example:
        from lm_resiliency.integrations.torchtitan import enable_resiliency

        enable_resiliency(
            model, optimizer,
            interval=10,
            load_fallback=lambda: checkpointer.load(),
        )

        # Training loop unchanged
        for step in range(start, end):
            train_step(...)

    Args:
        model: The training model.
        optimizer: Training optimizer (hooks attach here).
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
        parallelism_info: Object with dp_replicate/dp_shard attrs for HSDP
            auto-detection. If omitted, SCOUT inspects the parameter DeviceMesh.
        durable_checkpoint: SCOUT-gated framework checkpoint callbacks and manifest.

    Returns:
        Resiliency state handle.
    """
    trainer = model if _is_torchtitan_trainer(model) else None
    adapter = TorchTitanAdapter(trainer) if trainer is not None else None
    if adapter is not None:
        if optimizer is not None:
            raise TypeError(
                "do not pass optimizer when enabling resiliency on a TorchTitan Trainer"
            )
        model = adapter.model
        optimizer = adapter.optimizer
        if parallelism_info is None:
            parallelism_info = trainer.parallel_dims
    elif optimizer is None:
        raise TypeError("TorchTitan resiliency requires a Trainer or model and optimizer")

    handle = _enable_pytorch_resiliency(
        model,
        optimizer,
        interval=interval,
        enable_checkpoint=enable_checkpoint,
        enable_detection=enable_detection,
        checkpoint=ckpt_config,
        replay=detection_config,
        device=device,
        parallelism_info=parallelism_info,
        fault_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        orchestration=orchestration,
        load_fallback=load_fallback,
        extra_state_fn=adapter.get_extra_state_dict if adapter is not None else None,
        load_extra_state_fn=adapter.load_extra_state_dict if adapter is not None else None,
        durable_checkpoint=durable_checkpoint,
        recovery_mode=recovery_mode,
    )
    if trainer is not None:
        _bind_trainer_checkpoint_load(trainer, handle)
    return handle


def _is_torchtitan_trainer(target: Any) -> bool:
    """Recognize the stable Trainer surface without importing TorchTitan eagerly."""
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


def _bind_trainer_checkpoint_load(trainer: Any, handle: Any) -> None:
    """Coordinate TorchTitan's durable load with an earlier GEMINI recovery."""
    original_load = trainer.checkpointer.load

    def load(*args: Any, **kwargs: Any) -> Any:
        if handle.recovered_step >= 0:
            return True
        result = original_load(*args, **kwargs)
        handle._restore_step(int(trainer.step))
        return result

    trainer.checkpointer.load = load

    def restore_load() -> None:
        trainer.checkpointer.load = original_load

    handle.add_close_callback(restore_load)


__all__ = [
    "TorchTitanAdapter",
    "enable_resiliency",
]
