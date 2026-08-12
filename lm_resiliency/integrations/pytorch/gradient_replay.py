"""Training-shaped FSDP2/HSDP gradient communication for SCOUT replay.

SCOUT computes diagnostic parameter gradients with ``torch.autograd.grad`` so replay
does not alter live training gradients. That intentionally bypasses FSDP2's normal
post-backward gradient communication. This module closes the workload gap with scratch
buffers on the actual FSDP process groups:

* FSDP: ReduceScatter across the shard group.
* HSDP: the same ReduceScatter followed by AllReduce across the replicate group.

The scratch buffers match the layer's default dim-0 FSDP sharding volume and reduction
dtype. Their values are not applied to parameters or optimizer state.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

_SHARD_DIM_NAMES = ("dp_shard", "fsdp", "dp")
_REPLICATE_DIM_NAMES = ("dp_replicate",)
_warned_layer_types: set[type[nn.Module]] = set()


def replay_fsdp_gradient_communication(
    layer: nn.Module,
    gradients: Sequence[torch.Tensor | None],
) -> None:
    """Replay the sampled FSDP2 layer's gradient collectives on scratch buffers.

    This function is installed by the native PyTorch FSDP2/HSDP adapter. It deliberately
    uses public collectives instead of mutating ``Parameter.grad`` or reaching into
    FSDP's reducer lifecycle. Private FSDP state is consulted when available to recover
    the exact process groups and reduction dtype; DeviceMesh is the compatibility
    fallback.
    """
    materialized = [_local_tensor(gradient) for gradient in gradients if gradient is not None]
    if not materialized or not dist.is_available() or not dist.is_initialized():
        return

    groups = _resolve_gradient_groups(layer)
    if groups is None:
        layer_type = type(layer)
        if layer_type not in _warned_layer_types:
            _warned_layer_types.add(layer_type)
            logger.warning(
                "SCOUT could not resolve the FSDP gradient process groups for %s; "
                "gradient communication replay is unavailable for this layer type",
                layer_type.__name__,
            )
        return

    shard_group, replicate_group, reduce_dtype = groups
    shard_world_size = dist.get_world_size(shard_group)
    if shard_world_size < 2:
        return

    dtype = reduce_dtype or materialized[0].dtype
    device = materialized[0].device
    padded_numel = sum(_dim0_padded_numel(gradient, shard_world_size) for gradient in materialized)
    if padded_numel == 0:
        return

    # FSDP2 batches a layer's padded parameter gradients into one ReduceScatter.
    # Trigger-equivalent communication requires its dtype and byte volume, not the
    # optimizer-visible result, so scratch storage avoids mutating live gradients.
    reduce_scatter_input = torch.empty(padded_numel, dtype=dtype, device=device)
    reduce_scatter_output = torch.empty(
        padded_numel // shard_world_size,
        dtype=dtype,
        device=device,
    )
    dist.reduce_scatter_tensor(
        reduce_scatter_output,
        reduce_scatter_input,
        group=shard_group,
    )

    if replicate_group is not None and dist.get_world_size(replicate_group) > 1:
        dist.all_reduce(reduce_scatter_output, group=replicate_group)


def _resolve_gradient_groups(
    layer: nn.Module,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup | None, torch.dtype | None] | None:
    state = getattr(layer, "_fsdp_state", None)
    param_group = None
    if state is not None:
        param_group = getattr(state, "_fsdp_param_group", None)
        if param_group is None:
            param_group = getattr(state, "_param_group", None)

    shard_group = _first_attr(
        param_group,
        "_reduce_scatter_process_group",
        "reduce_scatter_process_group",
    )
    replicate_group = _first_attr(
        param_group,
        "_all_reduce_process_group",
        "all_reduce_process_group",
    )
    reduce_dtype = _first_attr(param_group, "_reduce_dtype", "reduce_dtype")
    if reduce_dtype is None and state is not None:
        mixed_precision = _first_attr(state, "_mp_policy", "_mixed_precision_policy")
        reduce_dtype = _first_attr(mixed_precision, "reduce_dtype")
    if shard_group is not None:
        return shard_group, replicate_group, reduce_dtype

    mesh = getattr(state, "_device_mesh", None) if state is not None else None
    if mesh is None:
        for parameter in layer.parameters():
            mesh = getattr(parameter, "device_mesh", None)
            if mesh is not None:
                break
    if mesh is None:
        return None

    names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
    if names:
        shard_index = _find_named_dim(names, _SHARD_DIM_NAMES)
        if shard_index is None and len(names) == 1:
            shard_index = 0
        if shard_index is None:
            return None
        replicate_index = _find_named_dim(names, _REPLICATE_DIM_NAMES)
    else:
        mesh_tensor = getattr(mesh, "mesh", None)
        if mesh_tensor is None or mesh_tensor.ndim != 1:
            return None
        shard_index = 0
        replicate_index = None

    shard_group = mesh.get_group(shard_index)
    replicate_group = mesh.get_group(replicate_index) if replicate_index is not None else None
    return shard_group, replicate_group, reduce_dtype


def _dim0_padded_numel(tensor: torch.Tensor, world_size: int) -> int:
    """Return FSDP's default dim-0 padded full-gradient storage size."""
    if tensor.numel() == 0:
        return 0
    if tensor.ndim == 0:
        return world_size
    rows = int(tensor.shape[0])
    if rows == 0:
        return 0
    row_numel = tensor.numel() // rows
    local_rows = math.ceil(rows / world_size)
    return local_rows * world_size * row_numel


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Use a TP-local gradient when the unsharded FSDP parameter remains a DTensor."""
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _find_named_dim(names: Sequence[str], candidates: Sequence[str]) -> int | None:
    for candidate in candidates:
        try:
            return names.index(candidate)
        except ValueError:
            continue
    return None


def _first_attr(value: Any, *names: str) -> Any:
    if value is None:
        return None
    for name in names:
        result = getattr(value, name, None)
        if result is not None:
            return result
    return None
