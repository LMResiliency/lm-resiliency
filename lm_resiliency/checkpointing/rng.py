"""RNG-state capture/restore for bitwise recovery of a *stochastic* forward.

A recovery only reproduces the never-failed trajectory bitwise if the random-number
generators resume exactly where they were at the checkpointed step — otherwise a
stochastic forward (dropout, stochastic depth, MoE routing noise, …) draws different
masks after the restart and the trajectory diverges. Model + optimizer state alone is not
enough.

This captures every generator a training step may consume, per rank:
  * ``torch`` CPU + the current CUDA device (each rank drives its own device),
  * Python ``random`` and NumPy (if installed),
  * **Megatron's model-parallel CUDA RNG tracker** — Megatron dropout draws from this
    forked tracker, *not* the default CUDA generator, so capturing only the default
    generator would silently miss it.

The captured dict is small (a few KB) and picklable, so it rides inside the
checkpoint's per-rank non-tensor payload, inheriting node-local flush and peer
replication. Each step's shard carries its own RNG. Restore must run **after** any
post-reload work that could consume RNG (a Megatron optimizer-state dummy step, etc.)
and **before** the first resumed forward.
"""

from __future__ import annotations

import random
from typing import Any

import torch

RNG_KEY = "__rng__"  # reserved non-tensor-data key the checkpoint carries RNG under


def capture_rng_state() -> dict[str, Any]:
    """Snapshot every RNG a step may consume, on this rank. Cheap; call at checkpoint time."""
    state: dict[str, Any] = {
        "torch_cpu": torch.get_rng_state(),
        "python": random.getstate(),
    }
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state()  # current device only
    try:
        import numpy as np

        state["numpy"] = np.random.get_state()
    except ImportError:
        pass
    tracker = _megatron_tracker_states()
    if tracker is not None:
        state["megatron_tracker"] = tracker
    return state


def restore_rng_state(state: dict[str, Any] | None) -> None:
    """Restore generators captured by ``capture_rng_state``. Run just before resuming."""
    if not state:
        return
    if "torch_cpu" in state:
        torch.set_rng_state(_as_byte_cpu(state["torch_cpu"]))
    if "python" in state:
        random.setstate(state["python"])
    if "torch_cuda" in state and torch.cuda.is_available():
        torch.cuda.set_rng_state(_as_byte_cpu(state["torch_cuda"]))
    if "numpy" in state:
        try:
            import numpy as np

            np.random.set_state(state["numpy"])
        except ImportError:
            pass
    if "megatron_tracker" in state:
        _restore_megatron_tracker(state["megatron_tracker"])


def _as_byte_cpu(t: Any) -> torch.Tensor:
    """RNG states must be a CPU ByteTensor; torch.load may hand back a differing view."""
    return t.cpu().to(torch.uint8) if isinstance(t, torch.Tensor) else t


# ── Megatron model-parallel RNG tracker (guarded; no-op off Megatron) ─────────────
def _megatron_tracker_states() -> dict[str, torch.Tensor] | None:
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
    except Exception:  # noqa: BLE001 — Megatron not installed / not initialized
        return None
    try:
        states = get_cuda_rng_tracker().get_states()
    except Exception:  # noqa: BLE001 — tracker not set up on this rank
        return None
    # Clone so a later fork()/step can't mutate the snapshot before it is serialized.
    return {k: v.clone() for k, v in states.items()} if states else None


def _restore_megatron_tracker(states: dict[str, torch.Tensor]) -> None:
    try:
        from megatron.core.tensor_parallel import get_cuda_rng_tracker
    except Exception:  # noqa: BLE001
        return
    get_cuda_rng_tracker().set_states({k: _as_byte_cpu(v) for k, v in states.items()})
