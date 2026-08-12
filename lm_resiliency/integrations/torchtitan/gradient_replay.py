"""Compatibility import for native PyTorch FSDP2 gradient replay."""

from lm_resiliency.integrations.pytorch.gradient_replay import (
    replay_fsdp_gradient_communication,
)

__all__ = ["replay_fsdp_gradient_communication"]
