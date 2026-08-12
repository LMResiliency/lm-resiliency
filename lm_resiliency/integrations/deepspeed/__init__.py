"""DeepSpeed integration for lm_resiliency.

Provides GEMINI (in-memory checkpointing) and SCOUT (fault detection)
for DeepSpeed distributed training (ZeRO Stage 1/2/3).

Usage:
    from lm_resiliency.integrations.deepspeed import enable_resiliency

    model_engine, optimizer, _, _ = deepspeed.initialize(...)

    resiliency = enable_resiliency(
        engine=model_engine,
        interval=10,
        load_fallback=lambda: load_from_disk(),
    )
"""

from lm_resiliency.integrations.deepspeed.adapter import DeepSpeedAdapter
from lm_resiliency.integrations.deepspeed.training import (
    DeepSpeedResiliency,
    enable_deepspeed_resiliency,
    enable_resiliency,
)

__all__ = [
    "DeepSpeedAdapter",
    "DeepSpeedResiliency",
    "enable_resiliency",
    "enable_deepspeed_resiliency",
]
