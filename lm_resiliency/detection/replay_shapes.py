"""Unified replay-shape plans for fixed-shape and dynamic-shape workloads."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.detection.layer_replay import ReplayInvocation
from lm_resiliency.detection.topology import ReplayPeerRole, normalize_replay_peer_role


@dataclass(frozen=True)
class ReplayShape:
    """One logical input shape selected for a replay round.

    ``dimensions=None`` means to replay the captured invocation unchanged. Concrete
    dimensions are interpreted by the workload's materializer. For a post-dispatch
    MoE expert stage, ``dimensions=(n_exec,)`` is the physical row count for each
    expert GEMM.
    """

    shape_id: str
    dimensions: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        if not self.shape_id:
            raise ValueError("replay shape_id cannot be empty")
        if self.dimensions is not None and any(dimension < 0 for dimension in self.dimensions):
            raise ValueError("replay shape dimensions must be non-negative")

    @classmethod
    def captured(cls) -> ReplayShape:
        """The one identity shape used by ordinary dense replay."""
        return cls(shape_id="captured", dimensions=None)


@dataclass(frozen=True)
class ReplayShapePlan:
    """An ordered list of shapes rotated by the common replay harness."""

    shapes: tuple[ReplayShape, ...]
    source_id: str = "configured"

    def __post_init__(self) -> None:
        if not self.shapes:
            raise ValueError("replay shape plan cannot be empty")
        shape_ids = [shape.shape_id for shape in self.shapes]
        if len(shape_ids) != len(set(shape_ids)):
            raise ValueError("replay shape IDs must be unique")
        if not self.source_id:
            raise ValueError("replay shape plan source_id cannot be empty")

    @classmethod
    def dense(cls) -> ReplayShapePlan:
        """Build the single captured-shape plan used by dense models."""
        return cls(shapes=(ReplayShape.captured(),), source_id="dense-captured")

    @classmethod
    def from_dimensions(
        cls,
        dimensions: Iterable[Sequence[int]],
        *,
        source_id: str = "configured-shapes",
    ) -> ReplayShapePlan:
        """Build a plan from an ordered list of concrete logical dimensions."""
        values = tuple(tuple(int(value) for value in shape) for shape in dimensions)
        return cls(
            shapes=tuple(
                ReplayShape(
                    shape_id=f"shape-{index}-{'x'.join(str(value) for value in shape)}",
                    dimensions=shape,
                )
                for index, shape in enumerate(values)
            ),
            source_id=source_id,
        )

    @classmethod
    def from_moe_catalog(cls, catalog: Any) -> ReplayShapePlan:
        """Convert qualified MoE representatives into the common shape list."""
        recipes = tuple(catalog.replay_recipes)
        if not recipes:
            raise ValueError("MoE catalog has no replay recipes")
        return cls(
            shapes=tuple(
                ReplayShape(
                    shape_id=(
                        f"{recipe.execution_class}:{recipe.regime_id}:"
                        f"n_exec={recipe.n_exec}:{recipe.fingerprint_id[:12]}"
                    ),
                    dimensions=(int(recipe.n_exec),),
                )
                for recipe in recipes
            ),
            source_id=f"moe-catalog:{catalog.identifier}",
        )

    @property
    def identifier(self) -> str:
        payload = {
            "source_id": self.source_id,
            "shapes": [
                {
                    "shape_id": shape.shape_id,
                    "dimensions": list(shape.dimensions) if shape.dimensions is not None else None,
                }
                for shape in self.shapes
            ],
        }
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def signature(self, shape: ReplayShape) -> int:
        """Return a stable signed int64 signature for cross-peer shape consensus."""
        if shape not in self.shapes:
            raise ValueError("replay shape does not belong to this plan")
        digest = hashlib.sha256(f"{self.identifier}:{shape.shape_id}".encode("utf-8")).digest()
        return int.from_bytes(digest[:8], byteorder="big", signed=True)


class ReplayShapePlanMismatch(ValueError):
    """Raised when replay rotation state belongs to a different shape plan."""


class ReplayShapeScheduler:
    """Rotate a shape plan once per successful replay check."""

    def __init__(self, plan: ReplayShapePlan) -> None:
        self.plan = plan
        self._position = 0
        self._cycle = 0

    @property
    def current_shape(self) -> ReplayShape:
        return self.plan.shapes[self._position]

    @property
    def position(self) -> int:
        return self._position

    @property
    def completed_cycles(self) -> int:
        return self._cycle

    def advance(self) -> None:
        self._position += 1
        if self._position == len(self.plan.shapes):
            self._position = 0
            self._cycle += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "plan_id": self.plan.identifier,
            "position": self._position,
            "cycle": self._cycle,
        }

    def load_state_dict(self, state: Mapping[str, Any] | None) -> None:
        if not state:
            return
        if state.get("plan_id") != self.plan.identifier:
            raise ReplayShapePlanMismatch(
                "cannot restore replay position into a different shape plan"
            )
        position = int(state["position"])
        cycle = int(state["cycle"])
        if not 0 <= position < len(self.plan.shapes) or cycle < 0:
            raise ValueError("invalid replay shape scheduler state")
        self._position = position
        self._cycle = cycle


ReplayShapeMaterializer = Callable[[ReplayInvocation, ReplayShape], ReplayInvocation]


@dataclass(frozen=True)
class LeadingDimensionMaterializer:
    """Resize a post-dispatch invocation along one physical work dimension.

    Every tensor leaf whose selected dimension matches the first input tensor's
    source extent is resized consistently, including ``grad_output``. Rows are
    truncated or repeated deterministically. This is suitable for an expert-stage
    boundary where that dimension represents packed routed tokens; complex grouped
    layouts should provide a backend-specific materializer instead.
    """

    dimension: int = 0

    def __post_init__(self) -> None:
        if self.dimension < 0:
            raise ValueError("materialized replay dimension must be non-negative")

    def __call__(
        self,
        invocation: ReplayInvocation,
        shape: ReplayShape,
    ) -> ReplayInvocation:
        if shape.dimensions is None or len(shape.dimensions) != 1:
            raise ValueError("LeadingDimensionMaterializer requires one concrete replay dimension")
        target_extent = shape.dimensions[0]
        inputs, _ = tree_flatten((invocation.args, invocation.kwargs))
        source = next(
            (
                leaf
                for leaf in inputs
                if isinstance(leaf, torch.Tensor) and leaf.ndim > self.dimension
            ),
            None,
        )
        if source is None:
            raise ValueError(f"captured invocation has no tensor with dimension {self.dimension}")
        if type(source).__name__ == "DTensor":
            raise ValueError(
                "LeadingDimensionMaterializer does not support DTensor inputs; "
                "provide a framework-specific materializer"
            )
        source_extent = int(source.shape[self.dimension])
        if target_extent > 0 and source_extent == 0:
            raise ValueError("cannot expand an empty captured replay dimension")
        if target_extent == source_extent:
            return invocation

        args, resized_args = self._resize_tree(
            invocation.args,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        kwargs, resized_kwargs = self._resize_tree(
            invocation.kwargs,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        grad_output, _ = self._resize_tree(
            invocation.grad_output,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        if resized_args + resized_kwargs == 0:
            raise ValueError("replay shape materializer did not resize an input tensor")
        return ReplayInvocation(
            args=args,
            kwargs=kwargs,
            input_requires_grad=list(invocation.input_requires_grad),
            grad_output=grad_output,
            autocast_enabled=invocation.autocast_enabled,
            autocast_device_type=invocation.autocast_device_type,
            autocast_dtype=invocation.autocast_dtype,
        )

    def _resize_tree(
        self,
        value: Any,
        *,
        source_extent: int,
        target_extent: int,
    ) -> tuple[Any, int]:
        leaves, spec = tree_flatten(value)
        resized = 0
        output = []
        for leaf in leaves:
            if (
                isinstance(leaf, torch.Tensor)
                and type(leaf).__name__ != "DTensor"
                and leaf.ndim > self.dimension
                and int(leaf.shape[self.dimension]) == source_extent
            ):
                output.append(
                    _resize_tensor_dimension(
                        leaf,
                        dimension=self.dimension,
                        target_extent=target_extent,
                    )
                )
                resized += 1
            else:
                output.append(leaf)
        return tree_unflatten(output, spec), resized


@dataclass(frozen=True)
class GroupedExpertMaterializer:
    """Materialize packed grouped-expert inputs for a qualified physical shape.

    Grouped expert modules receive a packed token tensor and a one-dimensional
    per-expert count tensor. Resizing only the packed tensors leaves the counts
    inconsistent with the grouped GEMM. A scalar ``n_exec`` recipe is applied to
    every local expert, so the materialized count vector is
    ``[n_exec] * local_expert_count`` and the packed extent is their sum.

    ``counts_input`` selects the count tensor by positional index or keyword.
    ``alignment`` can enforce a backend's per-expert row alignment.
    """

    counts_input: int | str = 1
    dimension: int = 0
    alignment: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.counts_input, int) and self.counts_input < 0:
            raise ValueError("grouped-expert counts input index must be non-negative")
        if isinstance(self.counts_input, str) and not self.counts_input:
            raise ValueError("grouped-expert counts input name cannot be empty")
        if self.dimension < 0:
            raise ValueError("materialized replay dimension must be non-negative")
        if self.alignment < 1:
            raise ValueError("grouped-expert alignment must be positive")

    def __call__(
        self,
        invocation: ReplayInvocation,
        shape: ReplayShape,
    ) -> ReplayInvocation:
        if shape.dimensions is None or len(shape.dimensions) != 1:
            raise ValueError(
                "GroupedExpertMaterializer requires one concrete per-expert replay dimension"
            )

        counts = self._counts_tensor(invocation)
        if counts.ndim != 1 or counts.numel() == 0:
            raise ValueError("grouped-expert counts must be a non-empty one-dimensional tensor")
        if counts.dtype == torch.bool or counts.is_floating_point() or counts.is_complex():
            raise ValueError("grouped-expert counts must use an integer dtype")
        if bool(torch.any(counts < 0).item()):
            raise ValueError("grouped-expert counts must be non-negative")

        n_exec = shape.dimensions[0]
        if n_exec % self.alignment:
            raise ValueError(
                f"per-expert replay extent {n_exec} is not aligned to {self.alignment} rows"
            )
        target_extent = n_exec * counts.numel()

        source = self._source_tensor(invocation, counts)
        source_extent = int(source.shape[self.dimension])
        counted_extent = int(counts.sum(dtype=torch.int64).item())
        if counted_extent != source_extent:
            raise ValueError(
                "grouped-expert counts must sum to the captured token extent "
                f"({counted_extent} != {source_extent})"
            )
        if target_extent > 0 and source_extent == 0:
            raise ValueError("cannot expand an empty captured replay dimension")

        args, resized_args = self._resize_tree(
            invocation.args,
            excluded=counts,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        kwargs, resized_kwargs = self._resize_tree(
            invocation.kwargs,
            excluded=counts,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        grad_output, _ = self._resize_tree(
            invocation.grad_output,
            excluded=None,
            source_extent=source_extent,
            target_extent=target_extent,
        )
        if target_extent != source_extent and resized_args + resized_kwargs == 0:
            raise ValueError("grouped-expert materializer did not resize a token tensor")

        new_counts = torch.full(
            (counts.numel(),),
            n_exec,
            dtype=counts.dtype,
            device=counts.device,
        )
        if isinstance(self.counts_input, int):
            args = list(args)
            if self.counts_input >= len(args):
                raise ValueError(
                    f"grouped-expert counts input index {self.counts_input} is out of range"
                )
            args[self.counts_input] = new_counts
            args = tuple(args)
        else:
            kwargs = dict(kwargs)
            if self.counts_input not in kwargs:
                raise ValueError(f"grouped-expert counts input {self.counts_input!r} is missing")
            kwargs[self.counts_input] = new_counts

        return ReplayInvocation(
            args=args,
            kwargs=kwargs,
            input_requires_grad=list(invocation.input_requires_grad),
            grad_output=grad_output,
            autocast_enabled=invocation.autocast_enabled,
            autocast_device_type=invocation.autocast_device_type,
            autocast_dtype=invocation.autocast_dtype,
        )

    def _counts_tensor(self, invocation: ReplayInvocation) -> torch.Tensor:
        if isinstance(self.counts_input, int):
            if self.counts_input >= len(invocation.args):
                raise ValueError(
                    f"grouped-expert counts input index {self.counts_input} is out of range"
                )
            counts = invocation.args[self.counts_input]
        else:
            if self.counts_input not in invocation.kwargs:
                raise ValueError(f"grouped-expert counts input {self.counts_input!r} is missing")
            counts = invocation.kwargs[self.counts_input]
        if not isinstance(counts, torch.Tensor):
            raise ValueError("grouped-expert counts input must be a tensor")
        return counts

    def _source_tensor(
        self,
        invocation: ReplayInvocation,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        inputs, _ = tree_flatten((invocation.args, invocation.kwargs))
        source = next(
            (
                leaf
                for leaf in inputs
                if isinstance(leaf, torch.Tensor)
                and leaf is not counts
                and type(leaf).__name__ != "DTensor"
                and leaf.ndim > self.dimension
            ),
            None,
        )
        if source is None:
            raise ValueError(
                f"captured grouped-expert invocation has no tensor with dimension {self.dimension}"
            )
        return source

    def _resize_tree(
        self,
        value: Any,
        *,
        excluded: torch.Tensor | None,
        source_extent: int,
        target_extent: int,
    ) -> tuple[Any, int]:
        leaves, spec = tree_flatten(value)
        resized = 0
        output = []
        for leaf in leaves:
            if (
                isinstance(leaf, torch.Tensor)
                and leaf is not excluded
                and type(leaf).__name__ != "DTensor"
                and leaf.ndim > self.dimension
                and int(leaf.shape[self.dimension]) == source_extent
            ):
                output.append(
                    _resize_tensor_dimension(
                        leaf,
                        dimension=self.dimension,
                        target_extent=target_extent,
                    )
                )
                resized += 1
            else:
                output.append(leaf)
        return tree_unflatten(output, spec), resized


@dataclass(frozen=True)
class ReplayWorkload:
    """Bind replay modules, a shape list, and input construction into one API."""

    shape_plan: ReplayShapePlan = field(default_factory=ReplayShapePlan.dense)
    replay_modules: tuple[nn.Module, ...] = ()
    materializer: ReplayShapeMaterializer | None = None
    peer_role: ReplayPeerRole = ReplayPeerRole.DENSE

    def __post_init__(self) -> None:
        object.__setattr__(self, "peer_role", normalize_replay_peer_role(self.peer_role))

    @classmethod
    def dense(
        cls,
        replay_modules: Sequence[nn.Module] | None = None,
    ) -> ReplayWorkload:
        return cls(
            shape_plan=ReplayShapePlan.dense(),
            replay_modules=tuple(replay_modules or ()),
        )

    @classmethod
    def from_moe_catalog(
        cls,
        catalog: Any,
        *,
        replay_modules: Sequence[nn.Module],
        materializer: ReplayShapeMaterializer | None = None,
    ) -> ReplayWorkload:
        modules = tuple(replay_modules)
        if not modules:
            raise ValueError("MoE replay requires at least one post-dispatch replay module")
        return cls(
            shape_plan=ReplayShapePlan.from_moe_catalog(catalog),
            replay_modules=modules,
            materializer=materializer or LeadingDimensionMaterializer(),
            peer_role=ReplayPeerRole.EXPERT,
        )

    @classmethod
    def from_shapes(
        cls,
        dimensions: Iterable[Sequence[int]],
        *,
        replay_modules: Sequence[nn.Module],
        materializer: ReplayShapeMaterializer,
        source_id: str = "configured-shapes",
        peer_role: ReplayPeerRole | str = ReplayPeerRole.DENSE,
    ) -> ReplayWorkload:
        """Build a dynamic workload directly from a list of logical shapes."""
        modules = tuple(replay_modules)
        if not modules:
            raise ValueError("dynamic replay requires at least one replay module")
        if materializer is None:
            raise ValueError("dynamic replay requires a shape materializer")
        return cls(
            shape_plan=ReplayShapePlan.from_dimensions(dimensions, source_id=source_id),
            replay_modules=modules,
            materializer=materializer,
            peer_role=normalize_replay_peer_role(peer_role),
        )

    def materialize(
        self,
        invocation: ReplayInvocation,
        shape: ReplayShape,
    ) -> ReplayInvocation:
        if self.materializer is not None:
            return self.materializer(invocation, shape)
        if shape.dimensions is None:
            return invocation
        raise ValueError(
            f"replay shape {shape.shape_id!r} is concrete but no materializer was configured"
        )


def _resize_tensor_dimension(
    tensor: torch.Tensor,
    *,
    dimension: int,
    target_extent: int,
) -> torch.Tensor:
    source_extent = int(tensor.shape[dimension])
    if target_extent == source_extent:
        return tensor
    if target_extent == 0:
        return tensor.narrow(dimension, 0, 0)
    indices = torch.arange(target_extent, device=tensor.device) % source_extent
    return tensor.index_select(dimension, indices)
