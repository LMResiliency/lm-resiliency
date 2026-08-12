"""Unified public API for enabling lm_resiliency features.

Example:
    from lm_resiliency import enable_resiliency, InMemoryCkptConfig, ReplayHarnessConfig

    handle = enable_resiliency(
        model,
        optimizer,
        interval=10,
        checkpoint=InMemoryCkptConfig(replication_jump=8),
        replay=ReplayHarnessConfig(rotate_layers=True),
        group=dp_gloo,
        nccl_group=dp_nccl,
    )
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig, ReplayResult
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.handle import ResiliencyHandle
from lm_resiliency.integrations.pytorch import enable_resiliency as _enable_pytorch
from lm_resiliency.orchestration import OrchestrationHooks


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
    extra_state_fn: Callable[[], dict] | None = None,
    load_extra_state_fn: Callable[[dict], None] | None = None,
    load_fallback: Callable[[], None] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
) -> ResiliencyHandle:
    """Enable GEMINI checkpointing and SCOUT detection with an unchanged loop.

    The returned handle owns progress, local recovery, instrumentation, and
    teardown. Cluster lifecycle and restart orchestration remain the launcher's
    responsibility.

    ``fault_callback`` receives in-process replay results. ``oob_fault_callback``
    receives JSON-ready hang and DataLoader-stall reports. External managers can
    pass ``orchestration`` instead of wiring both callback forms and restart
    mirroring separately.
    """
    return _enable_pytorch(
        model,
        optimizer,
        interval=interval,
        enable_checkpoint=enable_checkpoint,
        enable_detection=enable_detection,
        checkpoint=checkpoint,
        replay=replay,
        group=group,
        nccl_group=nccl_group,
        device=device,
        parallelism_info=parallelism_info,
        fault_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        orchestration=orchestration,
        extra_state_fn=extra_state_fn,
        load_extra_state_fn=load_extra_state_fn,
        load_fallback=load_fallback,
        durable_checkpoint=durable_checkpoint,
        recovery_mode=recovery_mode,
    )
