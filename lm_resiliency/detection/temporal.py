# mypy: ignore-errors
"""Bounded temporal baselines for hierarchical replay straggler detection."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._pytree import tree_flatten

SCOUT_TEMPORAL_KEY = "__scout_temporal__"
_STATE_VERSION = 1


@dataclass
class TemporalAssessment:
    """Temporal anomalies for one replay timing vector."""

    rank_bitmap: list[int]
    group_slowdown: bool


@dataclass
class _Baseline:
    peer_windows: list[list[float]]
    group_window: list[float] = field(default_factory=list)


class TemporalBaselineStore:
    """In-process hot store for bounded per-peer and peer-group timing history."""

    def __init__(
        self,
        *,
        window_size: int = 32,
        min_samples: int = 5,
        slowdown_ratio: float = 1.25,
        threshold_sigma: float = 4.0,
        max_keys: int = 256,
    ) -> None:
        if window_size < 1:
            raise ValueError("temporal window_size must be positive")
        if min_samples < 1 or min_samples > window_size:
            raise ValueError("temporal min_samples must be in [1, window_size]")
        if slowdown_ratio <= 1.0:
            raise ValueError("temporal slowdown_ratio must be greater than 1")
        self._window_size = window_size
        self._min_samples = min_samples
        self._slowdown_ratio = slowdown_ratio
        self._threshold_sigma = threshold_sigma
        self._max_keys = max_keys
        self._baselines: OrderedDict[str, _Baseline] = OrderedDict()

    def assess(self, key: str, times_ms: list[float]) -> TemporalAssessment:
        """Compare current peer timings with history without updating it."""
        baseline = self._baselines.get(key)
        if baseline is None or len(baseline.peer_windows) != len(times_ms):
            return TemporalAssessment([0] * len(times_ms), False)

        rank_bitmap = [
            int(self._is_slow(value, history))
            for value, history in zip(times_ms, baseline.peer_windows)
        ]
        group_value = statistics.median(times_ms) if times_ms else 0.0
        return TemporalAssessment(
            rank_bitmap=rank_bitmap,
            group_slowdown=self._is_slow(group_value, baseline.group_window),
        )

    def observe_clean(self, key: str, times_ms: list[float]) -> None:
        """Add a clean round to the bounded baseline."""
        if not times_ms or not all(math.isfinite(value) and value >= 0 for value in times_ms):
            return
        baseline = self._baselines.get(key)
        if baseline is None or len(baseline.peer_windows) != len(times_ms):
            baseline = _Baseline(peer_windows=[[] for _ in times_ms])
            self._baselines[key] = baseline
        self._baselines.move_to_end(key)
        for history, value in zip(baseline.peer_windows, times_ms):
            history.append(float(value))
            del history[: -self._window_size]
        baseline.group_window.append(float(statistics.median(times_ms)))
        del baseline.group_window[: -self._window_size]
        while len(self._baselines) > self._max_keys:
            self._baselines.popitem(last=False)

    def state_dict(self) -> dict[str, Any]:
        """Return compact, picklable state suitable for GEMINI checkpoint extras."""
        return {
            "version": _STATE_VERSION,
            "baselines": {
                key: {
                    "peers": [history[-self._window_size :] for history in value.peer_windows],
                    "group": value.group_window[-self._window_size :],
                }
                for key, value in self._baselines.items()
            },
        }

    def load_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore valid bounded histories and ignore incompatible state."""
        if not state or state.get("version") != _STATE_VERSION:
            return
        restored: OrderedDict[str, _Baseline] = OrderedDict()
        for key, raw in list((state.get("baselines") or {}).items())[-self._max_keys :]:
            peers = raw.get("peers") if isinstance(raw, dict) else None
            group = raw.get("group") if isinstance(raw, dict) else None
            if (
                not isinstance(key, str)
                or not isinstance(peers, list)
                or not isinstance(group, list)
            ):
                continue
            peer_windows = [self._valid_history(history) for history in peers]
            restored[key] = _Baseline(
                peer_windows=peer_windows,
                group_window=self._valid_history(group),
            )
        self._baselines = restored

    def _valid_history(self, values: Any) -> list[float]:
        if not isinstance(values, list):
            return []
        return [
            float(value)
            for value in values[-self._window_size :]
            if isinstance(value, (int, float)) and math.isfinite(value) and value >= 0
        ]

    def _is_slow(self, value: float, history: list[float]) -> bool:
        if len(history) < self._min_samples:
            return False
        center = statistics.median(history)
        deviations = [abs(sample - center) for sample in history]
        mad = statistics.median(deviations)
        robust_sigma = 1.4826 * mad
        ratio_threshold = center * self._slowdown_ratio
        dispersion_threshold = center + self._threshold_sigma * robust_sigma
        return value > max(ratio_threshold, dispersion_threshold)


def replay_baseline_key(
    *,
    layer_id: int,
    replay_mode: str,
    invocation: Any,
    peer_ranks: list[int],
    device: torch.device,
) -> str:
    """Build a stable key from layer, mode, input structure, group, and hardware."""
    leaves, spec = tree_flatten((invocation.args, invocation.kwargs))
    descriptors = []
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor):
            local = leaf.to_local() if type(leaf).__name__ == "DTensor" else leaf
            descriptors.append(
                {
                    "kind": "tensor",
                    "shape": list(local.shape),
                    "dtype": str(local.dtype),
                    "requires_grad": bool(local.requires_grad),
                }
            )
        elif leaf is None or isinstance(leaf, (bool, int, float, str)):
            descriptors.append({"kind": type(leaf).__name__, "value": leaf})
        else:
            descriptors.append(
                {
                    "kind": "object",
                    "type": f"{type(leaf).__module__}.{type(leaf).__qualname__}",
                }
            )

    hardware = "cpu"
    if device.type == "cuda" and torch.cuda.is_available():
        hardware = torch.cuda.get_device_name(device)
    payload = {
        "layer": layer_id,
        "mode": replay_mode,
        "tree": str(spec),
        "inputs": descriptors,
        "group": peer_ranks,
        "hardware": hardware,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
