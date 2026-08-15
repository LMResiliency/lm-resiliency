# mypy: ignore-errors
"""Shared utilities for detection modules."""

from __future__ import annotations

import contextlib
import time
from typing import Any, Generator

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.checkpointing.rng import capture_rng_state, restore_rng_state


@contextlib.contextmanager
def deterministic_mode(enabled: bool = True) -> Generator[None, None, None]:
    """Context manager: enable deterministic algorithms, restore on exit."""
    prev = torch.are_deterministic_algorithms_enabled()
    if enabled:
        torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        if enabled:
            torch.use_deterministic_algorithms(prev)


@contextlib.contextmanager
def synchronized_replay_rng(
    group: dist.ProcessGroup | None,
    source_global_rank: int,
    enabled: bool = True,
) -> Generator[dict[str, Any] | None, None, None]:
    """Use one peer's RNG state for replay, then restore every rank's state.

    Deterministic algorithms do not make stochastic modules equivalent: dropout and
    framework RNG trackers still consume rank-local generator state. Replay therefore
    broadcasts the source peer's complete RNG snapshot (CPU, CUDA, Python, NumPy, and
    Megatron's tracker when present), installs it only for the replay invocation, and
    restores the original local snapshot afterward. Diagnostic replay consequently
    neither advances nor otherwise perturbs the training RNG streams.
    """
    if not enabled:
        yield None
        return

    local_state = capture_rng_state()
    shared_state = local_state
    if dist.is_available() and dist.is_initialized():
        payload = [local_state if dist.get_rank() == source_global_rank else None]
        dist.broadcast_object_list(payload, src=source_global_rank, group=group)
        shared_state = payload[0]

    restore_rng_state(shared_state)
    try:
        yield shared_state
    finally:
        restore_rng_state(local_state)


def timed_forward(
    layer: nn.Module, activation: torch.Tensor, device: torch.device
) -> tuple[torch.Tensor, float]:
    """Run a single forward pass with wall-clock timing (sync before and after)."""
    output, elapsed_ms = timed_call(layer, (activation,), {}, device)
    return output, elapsed_ms


def timed_call(
    layer: nn.Module,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    device: torch.device,
) -> tuple[Any, float]:
    """Run a structured module invocation with wall-clock CUDA timing."""
    torch.cuda.synchronize(device)
    start = time.perf_counter()

    with torch.no_grad():
        output = layer(*args, **kwargs)

    torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return output, elapsed_ms
