"""Megatron-core integration for lm_resiliency.

Provides GEMINI (in-memory checkpointing) and SCOUT (fault detection)
for megatron-core distributed training.

Usage:
    from lm_resiliency.integrations.megatron import enable_resiliency

    resiliency = enable_resiliency(
        model=model,
        optimizer=optimizer,
        opt_param_scheduler=opt_param_scheduler,
        interval=10,
        load_fallback=lambda: load_checkpoint_from_disk(),
    )
"""

from lm_resiliency.integrations.megatron.adapter import MegatronAdapter
from lm_resiliency.integrations.megatron.training import (
    MegatronResiliency,
    enable_megatron_resiliency,
    enable_resiliency,
)

__all__ = [
    "MegatronAdapter",
    "MegatronResiliency",
    "enable_resiliency",
    "enable_megatron_resiliency",
]
