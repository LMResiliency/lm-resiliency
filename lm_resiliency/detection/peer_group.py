"""Peer group formation for fault detection.

Auto-discovers the DP/FSDP peer group from the model's DeviceMesh without
requiring the user to pass explicit process groups.

    # User code — zero topology knowledge needed:
    group, nccl_group = form_detection_groups(model)

Discovery chain:
  1. DTensor parameters → param.device_mesh
  2. FSDP2 module state → module._fsdp_state._device_mesh
  3. Active DeviceMesh context → _mesh_resources.get_current_mesh()
  4. Fallback: WORLD group (pure DDP — all ranks are DP peers)

For HSDP, replay peers vary only ``dp_replicate`` while retaining the same
``dp_shard`` coordinate.
"""

from __future__ import annotations

import itertools
import logging
from typing import Sequence

import torch
import torch.distributed as dist
import torch.nn as nn

logger = logging.getLogger(__name__)

_DP_DIM_NAMES = ("dp", "dp_replicate", "dp_shard", "fsdp", "efsdp")


def form_detection_groups(
    model: nn.Module | None = None,
    device_mesh=None,
) -> tuple[dist.ProcessGroup, dist.ProcessGroup]:
    """Create independent Gloo + NCCL peer groups for fault detection.

    Auto-discovers the replay-peer dimension from the model's DeviceMesh and
    creates new process groups with the same rank membership. These groups are
    independent from the training communicators — safe for OOB use.

    Args:
        model: The training model. Used to extract DeviceMesh from DTensor
            parameters or FSDP2 state. Can be None if device_mesh is provided.
        device_mesh: Explicit DeviceMesh. If provided, skips auto-discovery.

    Returns:
        (gloo_group, nccl_group): Independent process groups for C3 operations.
            gloo_group for scalar consensus, nccl_group for GPU tensor ops.
    """
    mesh = device_mesh or _infer_mesh(model)
    all_subgroups = _all_dp_subgroups(mesh)
    my_rank = dist.get_rank()
    sync_group = None
    if len(all_subgroups) > 1:
        sync_group = dist.new_group(
            ranks=list(range(dist.get_world_size())),
            backend="gloo",
        )

    # dist.new_group is collective — all ranks must participate in every call.
    # Create all subgroups in order, keep only this rank's.
    my_gloo: dist.ProcessGroup | None = None
    my_nccl: dist.ProcessGroup | None = None
    for ranks in all_subgroups:
        g = dist.new_group(ranks=ranks, backend="gloo")
        n = dist.new_group(ranks=ranks, backend="nccl")
        if my_rank in ranks:
            my_gloo = g
            my_nccl = n
            # Eagerly initialize this NCCL communicator before another disjoint
            # subgroup is allowed to initialize.
            dist.barrier(
                group=n,
                device_ids=([torch.cuda.current_device()] if torch.cuda.is_available() else None),
            )
        if sync_group is not None:
            dist.barrier(group=sync_group)

    assert my_gloo is not None and my_nccl is not None
    peer_ranks = dist.get_process_group_ranks(my_gloo)

    logger.info(
        f"Formed detection peer group: {len(peer_ranks)} ranks "
        f"(peer slice from mesh {_describe_mesh(mesh)})"
    )

    return my_gloo, my_nccl


def parallelism_device_mesh(parallelism_info, *, expert: bool = False):
    """Extract a full replay mesh from DeviceMesh- or framework-shaped metadata."""
    if parallelism_info is None:
        return None
    if getattr(parallelism_info, "mesh", None) is not None and hasattr(
        parallelism_info, "mesh_dim_names"
    ):
        return parallelism_info
    mesh = getattr(parallelism_info, "device_mesh", None)
    if mesh is not None:
        return mesh
    get_mesh = getattr(parallelism_info, "get_mesh", None)
    if callable(get_mesh):
        try:
            return get_mesh("sparse" if expert else "dense")
        except (AssertionError, KeyError, RuntimeError, ValueError):
            pass

        # TorchTitan exposes individual dimensions publicly rather than the
        # internal dense and sparse global-mesh aliases.
        get_optional_mesh = getattr(parallelism_info, "get_optional_mesh", None)
        if callable(get_optional_mesh):
            names = (
                ("pp", "dp_replicate", "efsdp", "ep", "etp")
                if expert
                else ("pp", "dp_replicate", "fsdp", "tp")
            )
            active_names = []
            for name in names:
                try:
                    if get_optional_mesh(name) is not None:
                        active_names.append(name)
                except (AssertionError, KeyError, RuntimeError, ValueError):
                    continue
            if active_names:
                dimensions = active_names[0] if len(active_names) == 1 else active_names
                try:
                    return get_mesh(dimensions)
                except (AssertionError, KeyError, RuntimeError, ValueError):
                    pass
    return None


def get_peer_ranks(
    model: nn.Module | None = None,
    device_mesh=None,
) -> list[int]:
    """Return the list of global ranks in this rank's DP peer group.

    Same discovery logic as form_detection_groups(), but returns the rank list
    without creating process groups. Useful for passing to the OOB daemon.
    """
    mesh = device_mesh or _infer_mesh(model)
    return _extract_dp_ranks(mesh)


def _infer_mesh(model: nn.Module | None):
    """Auto-discover DeviceMesh from model or active context.

    Returns DeviceMesh or None (for pure DDP fallback).
    """
    # 1. From DTensor parameters
    if model is not None:
        mesh = _mesh_from_dtensor_params(model)
        if mesh is not None:
            return mesh

        # 2. From FSDP2 module state
        mesh = _mesh_from_fsdp2(model)
        if mesh is not None:
            return mesh

    # 3. From active context manager
    mesh = _mesh_from_context()
    if mesh is not None:
        return mesh

    # 4. No mesh found — pure DDP
    return None


def _mesh_from_dtensor_params(model: nn.Module):
    """Extract DeviceMesh from the first DTensor parameter found."""
    try:
        from torch.distributed.tensor import DTensor
    except ImportError:
        return None

    for param in model.parameters():
        if isinstance(param, DTensor):
            return _prefer_root_replica_mesh(param.device_mesh)
    return None


def _mesh_from_fsdp2(model: nn.Module):
    """Extract DeviceMesh from FSDP2 module state."""
    for module in model.modules():
        state = getattr(module, "_fsdp_state", None)
        if state is not None:
            mesh = getattr(state, "_device_mesh", None)
            if mesh is not None:
                return _prefer_root_replica_mesh(mesh)
    return None


def _mesh_from_context():
    """Get DeviceMesh from the active context manager stack."""
    try:
        from torch.distributed.device_mesh import _mesh_resources

        return _mesh_resources.get_current_mesh()
    except (ImportError, RuntimeError):
        return None


def _all_dp_subgroups(mesh) -> list[list[int]]:
    """Return all distinct DP subgroups from the mesh.

    For pure DDP (mesh=None), returns a single group with all ranks.
    For a 2D mesh (dp=2, tp=4), returns 4 groups of 2 ranks each.
    For HSDP, returns one replica group per fixed shard coordinate.
    """
    if mesh is None:
        return [list(range(dist.get_world_size()))]

    dim_names = mesh.mesh_dim_names
    if dim_names is None:
        peer_dim = 0
    elif not any(name in _DP_DIM_NAMES for name in dim_names):
        return [[rank] for rank in mesh.mesh.flatten().tolist()]
    else:
        peer_dim = _find_peer_dim(mesh)
    return _all_dim_subgroups(mesh, peer_dim)


def _all_dim_subgroups(mesh, selected_dim: int) -> list[list[int]]:
    """Return every one-dimensional subgroup along ``selected_dim``."""
    rank_tensor = mesh.mesh
    other_dims = [index for index in range(rank_tensor.ndim) if index != selected_dim]
    other_sizes = [rank_tensor.shape[index] for index in other_dims]
    subgroups = []
    for indices in itertools.product(*(range(size) for size in other_sizes)):
        item = [slice(None)] * rank_tensor.ndim
        for dim, index in zip(other_dims, indices):
            item[dim] = index
        subgroups.append(sorted(rank_tensor[tuple(item)].flatten().tolist()))
    return subgroups


def _extract_dp_ranks(mesh) -> list[int]:
    """Given a DeviceMesh (or None), return the DP peer ranks for this rank.

    For a named HSDP mesh, natural replicas take priority over state-shard
    peers, so the returned ranks retain this rank's shard coordinate.
    """
    if mesh is None:
        return list(range(dist.get_world_size()))

    dim_names = mesh.mesh_dim_names
    if dim_names is None:
        peer_dim = 0
    elif not any(name in _DP_DIM_NAMES for name in dim_names):
        return [dist.get_rank()]
    else:
        peer_dim = _find_peer_dim(mesh)
    peer_group = mesh.get_group(peer_dim)
    return dist.get_process_group_ranks(peer_group)


def _find_dp_dim(dim_names: Sequence[str]) -> int:
    """Find the DP dimension index from mesh dimension names."""
    for target in _DP_DIM_NAMES:
        for i, name in enumerate(dim_names):
            if name == target:
                return i

    raise ValueError(
        f"No DP dimension found in mesh dims: {list(dim_names)}. "
        f"Expected one of {list(_DP_DIM_NAMES)}. "
        f"Pass device_mesh= or use enable_resiliency(group=...) as fallback."
    )


def _find_peer_dim(mesh) -> int:
    """Select natural replicas when present, otherwise state-shard peers."""
    dim_names = tuple(mesh.mesh_dim_names or ())
    if "dp" in dim_names:
        return dim_names.index("dp")

    if "dp_replicate" in dim_names:
        replica_dim = dim_names.index("dp_replicate")
        if int(mesh.mesh.shape[replica_dim]) > 1:
            return replica_dim

    for name in ("dp_shard", "fsdp", "efsdp"):
        if name in dim_names:
            return dim_names.index(name)

    if "dp_replicate" in dim_names:
        return dim_names.index("dp_replicate")
    return _find_dp_dim(dim_names)


def _prefer_root_replica_mesh(mesh):
    """Recover a parent DP x model-parallel mesh from a TP-only submesh."""
    get_root_mesh = getattr(mesh, "_get_root_mesh", None)
    if not callable(get_root_mesh):
        return mesh
    root = get_root_mesh()
    root_names = tuple(getattr(root, "mesh_dim_names", None) or ())
    if any(name in _DP_DIM_NAMES for name in root_names):
        return root
    return mesh


def _describe_mesh(mesh) -> str:
    """Human-readable mesh description for logging."""
    if mesh is None:
        return "None (pure DDP, all ranks)"
    dim_names = mesh.mesh_dim_names
    if dim_names is not None:
        dims = [f"{name}={mesh.size(i)}" for i, name in enumerate(dim_names)]
        return f"({', '.join(dims)})"
    return f"(unnamed, shape={list(mesh.mesh.shape)})"
