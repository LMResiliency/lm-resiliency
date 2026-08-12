"""Training-side progress instrumentation for the OOB hang daemon."""

from __future__ import annotations

import functools
import hashlib
import json
import threading
from collections.abc import Sequence
from contextlib import contextmanager
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.all_to_all_replay import (
    AllToAllReplayRecipe,
    TensorReplaySpec,
)
from lm_resiliency.detection.op_tracker import OpTracker
from lm_resiliency.detection.stage_instrumentation import DiagnosticStageMonitor

_COLLECTIVE_NAMES = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_to_all",
    "all_to_all_single",
    "broadcast",
    "gather",
    "reduce",
    "reduce_scatter",
    "reduce_scatter_tensor",
    "scatter",
    "barrier",
)

_COLLECTIVE_PARAMETERS = {
    "all_reduce": ("tensor", "op", "group", "async_op"),
    "all_gather": ("tensor_list", "tensor", "group", "async_op"),
    "all_gather_into_tensor": ("output_tensor", "input_tensor", "group", "async_op"),
    "all_to_all": ("output_tensor_list", "input_tensor_list", "group", "async_op"),
    "all_to_all_single": (
        "output",
        "input",
        "output_split_sizes",
        "input_split_sizes",
        "group",
        "async_op",
    ),
    "broadcast": ("tensor", "src", "group", "async_op", "group_src"),
    "gather": ("tensor", "gather_list", "dst", "group", "async_op", "group_dst"),
    "reduce": ("tensor", "dst", "op", "group", "async_op", "group_dst"),
    "reduce_scatter": ("output", "input_list", "op", "group", "async_op"),
    "reduce_scatter_tensor": ("output", "input", "op", "group", "async_op"),
    "scatter": ("tensor", "scatter_list", "src", "group", "async_op", "group_src"),
    "barrier": ("group", "async_op", "device_ids"),
}


def _bind_collective_arguments(
    name: str, args: tuple[Any, ...], kwargs: dict[str, Any]
) -> dict[str, Any]:
    bound = dict(zip(_COLLECTIVE_PARAMETERS[name], args))
    bound.update(kwargs)
    return bound


def _tensor_spec(value: Any, *, include_shape: bool = True) -> Any:
    if isinstance(value, torch.Tensor):
        spec: dict[str, Any] = {"dtype": str(value.dtype)}
        if include_shape:
            spec["shape"] = list(value.shape)
        return spec
    if isinstance(value, (list, tuple)):
        return [_tensor_spec(item, include_shape=include_shape) for item in value]
    if value is None:
        return None
    return type(value).__name__


def _stable_value(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_stable_value(item) for item in value]
    return str(value)


def _preferred_argument(bound: dict[str, Any], preferred: str, fallback: str) -> Any:
    value = bound.get(preferred)
    return value if value is not None else bound.get(fallback)


def _tensor_replay_specs(value: Any) -> tuple[TensorReplaySpec, ...]:
    if isinstance(value, torch.Tensor):
        return (
            TensorReplaySpec(
                shape=tuple(int(size) for size in value.shape),
                dtype=value.dtype,
                numel=value.numel(),
                element_size=value.element_size(),
            ),
        )
    if isinstance(value, (list, tuple)):
        return tuple(spec for tensor in value for spec in _tensor_replay_specs(tensor))
    return ()


def _explicit_split_sizes(value: Any) -> tuple[int, ...] | None:
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    return tuple(int(size) for size in value)


def _split_count(value: Any) -> int:
    split_sizes = _explicit_split_sizes(value)
    return len(split_sizes) if split_sizes is not None else 0


def _equal_split_sizes(
    specs: tuple[TensorReplaySpec, ...], group_ranks: tuple[int, ...] | None
) -> tuple[int, ...] | None:
    if len(specs) != 1 or not specs[0].shape or not group_ranks:
        return None
    leading_size = specs[0].shape[0]
    world_size = len(group_ranks)
    if world_size == 0 or leading_size % world_size:
        return None
    return (leading_size // world_size,) * world_size


def _all_to_all_replay_recipe(
    name: str,
    bound: dict[str, Any],
    *,
    group_ranks: tuple[int, ...] | None,
    sequence: int,
) -> AllToAllReplayRecipe:
    if name == "all_to_all_single":
        inputs = _tensor_replay_specs(bound.get("input"))
        outputs = _tensor_replay_specs(bound.get("output"))
        input_splits = _explicit_split_sizes(bound.get("input_split_sizes"))
        output_splits = _explicit_split_sizes(bound.get("output_split_sizes"))
        if input_splits is None:
            input_splits = _equal_split_sizes(inputs, group_ranks)
        if output_splits is None:
            output_splits = _equal_split_sizes(outputs, group_ranks)
    else:
        inputs = _tensor_replay_specs(bound.get("input_tensor_list"))
        outputs = _tensor_replay_specs(bound.get("output_tensor_list"))
        input_splits = tuple(spec.shape[0] for spec in inputs if spec.shape)
        output_splits = tuple(spec.shape[0] for spec in outputs if spec.shape)

    return AllToAllReplayRecipe(
        sequence=sequence,
        collective=name,
        group_ranks=group_ranks,
        inputs=inputs,
        outputs=outputs,
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        async_op=bool(bound.get("async_op", False)),
        group=bound.get("group"),
    )


def collective_metadata_fingerprint(
    name: str,
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    group_ranks: tuple[int, ...] | None,
) -> int:
    """Return a stable description of rank-invariant collective arguments.

    Dynamic all-to-all and gather/scatter shapes may legitimately differ by
    rank, so only their rank-invariant type and control metadata is compared.
    """

    bound = _bind_collective_arguments(name, args, kwargs)
    metadata: dict[str, Any] = {
        "collective": name,
        "group_ranks": list(group_ranks) if group_ranks is not None else "default",
    }

    if name in {"all_reduce", "broadcast", "reduce"}:
        metadata["tensor"] = _tensor_spec(bound.get("tensor"))
    elif name == "all_gather":
        # all_gather supports uneven inputs, but every rank has the same ordered
        # output-list schema.
        metadata["tensor_list"] = _tensor_spec(bound.get("tensor_list"))
        metadata["input_dtype"] = _tensor_spec(bound.get("tensor"), include_shape=False)
    elif name == "all_gather_into_tensor":
        metadata["output"] = _tensor_spec(bound.get("output_tensor"))
        metadata["input"] = _tensor_spec(bound.get("input_tensor"))
    elif name == "all_to_all":
        metadata["output_types"] = _tensor_spec(
            bound.get("output_tensor_list"), include_shape=False
        )
        metadata["input_types"] = _tensor_spec(bound.get("input_tensor_list"), include_shape=False)
    elif name == "all_to_all_single":
        metadata["output_type"] = _tensor_spec(bound.get("output"), include_shape=False)
        metadata["input_type"] = _tensor_spec(bound.get("input"), include_shape=False)
        metadata["output_split_count"] = _split_count(bound.get("output_split_sizes"))
        metadata["input_split_count"] = _split_count(bound.get("input_split_sizes"))
    elif name == "gather":
        metadata["input_type"] = _tensor_spec(bound.get("tensor"), include_shape=False)
    elif name in {"reduce_scatter", "reduce_scatter_tensor"}:
        metadata["output"] = _tensor_spec(bound.get("output"))
        input_name = "input_list" if name == "reduce_scatter" else "input"
        metadata["input"] = _tensor_spec(bound.get(input_name))
    elif name == "scatter":
        metadata["output_type"] = _tensor_spec(bound.get("tensor"), include_shape=False)

    if name in {"all_reduce", "reduce", "reduce_scatter", "reduce_scatter_tensor"}:
        metadata["reduce_op"] = _stable_value(bound.get("op", dist.ReduceOp.SUM))
    if name in {"broadcast", "scatter"}:
        metadata["source"] = _stable_value(_preferred_argument(bound, "group_src", "src"))
    if name in {"gather", "reduce"}:
        metadata["destination"] = _stable_value(_preferred_argument(bound, "group_dst", "dst"))

    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode()
    fingerprint = int.from_bytes(
        hashlib.blake2b(encoded, digest_size=8).digest(), byteorder="big", signed=True
    )
    return fingerprint or 1


class HangInstrumentation:
    """Publish Python-visible collective and transformer-layer progress."""

    def __init__(
        self,
        model: nn.Module,
        layers: Sequence[nn.Module],
        rank: int,
        *,
        progress_event: Any | None = None,
    ) -> None:
        del model
        self._tracker = OpTracker(rank, progress_event=progress_event)
        self._hooks: list[Any] = []
        self._saved_collectives: dict[str, Any] = {}
        self._collective_wrappers: dict[str, Any] = {}
        self._group_ranks_cache: dict[int, tuple[int, ...] | None] = {}
        self._all_to_all_recipes: list[AllToAllReplayRecipe] = []
        self._all_to_all_capture_suspensions = 0
        self._lock = threading.Lock()
        self._stage_monitor = DiagnosticStageMonitor(self._tracker, self._lock)
        self._closed = False
        for layer in layers:
            self._hooks.append(layer.register_forward_pre_hook(self._layer_boundary))
            self._hooks.append(layer.register_forward_hook(self._layer_boundary))
        self._patch_collectives()

    @property
    def tracker(self) -> OpTracker:
        return self._tracker

    @property
    def stage_monitor(self) -> DiagnosticStageMonitor:
        return self._stage_monitor

    @property
    def all_to_all_recipes(self) -> tuple[AllToAllReplayRecipe, ...]:
        """All Python-visible AllToAll operations since the last step boundary."""
        with self._lock:
            return tuple(self._all_to_all_recipes)

    def step_boundary(self) -> None:
        with self._lock:
            self._all_to_all_recipes.clear()
            self._tracker.step_boundary()

    @contextmanager
    def suspend_all_to_all_capture(self):
        """Avoid recapturing diagnostic AllToAll replay as training traffic."""
        with self._lock:
            self._all_to_all_capture_suspensions += 1
        try:
            yield
        finally:
            with self._lock:
                self._all_to_all_capture_suspensions -= 1

    def _layer_boundary(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs
        with self._lock:
            self._tracker.advance()

    def _patch_collectives(self) -> None:
        for name in _COLLECTIVE_NAMES:
            original = getattr(dist, name, None)
            if original is None:
                continue
            self._saved_collectives[name] = original

            @functools.wraps(original)
            def wrapped(*args: Any, __name=name, __original=original, **kwargs: Any):
                bound = _bind_collective_arguments(__name, args, kwargs)
                group_ranks = self._group_ranks(bound.get("group"))
                fingerprint = collective_metadata_fingerprint(
                    __name,
                    args,
                    kwargs,
                    group_ranks=group_ranks,
                )
                with self._lock:
                    if (
                        __name in {"all_to_all", "all_to_all_single"}
                        and self._all_to_all_capture_suspensions == 0
                    ):
                        self._all_to_all_recipes.append(
                            _all_to_all_replay_recipe(
                                __name,
                                bound,
                                group_ranks=group_ranks,
                                sequence=len(self._all_to_all_recipes),
                            )
                        )
                    self._tracker.advance(fingerprint, force_signal=True)
                result = __original(*args, **kwargs)
                with self._lock:
                    self._tracker.advance(force_signal=True)
                return result

            setattr(dist, name, wrapped)
            self._collective_wrappers[name] = wrapped

    def _group_ranks(self, group: Any) -> tuple[int, ...] | None:
        key = 0 if group is None else id(group)
        if key in self._group_ranks_cache:
            return self._group_ranks_cache[key]
        ranks: tuple[int, ...] | None = None
        try:
            if dist.is_initialized():
                if group is None:
                    ranks = tuple(range(dist.get_world_size()))
                else:
                    ranks = tuple(dist.get_process_group_ranks(group))
        except (RuntimeError, ValueError):
            pass
        self._group_ranks_cache[key] = ranks
        return ranks

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for name, original in self._saved_collectives.items():
            if getattr(dist, name, None) is self._collective_wrappers.get(name):
                setattr(dist, name, original)
        self._saved_collectives.clear()
        self._collective_wrappers.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        self._tracker.close()
