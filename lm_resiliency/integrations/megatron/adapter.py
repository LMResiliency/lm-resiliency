"""Megatron-core FrameworkAdapter implementation.

Bridges megatron-core's distributed training components with lm_resiliency's
in-memory checkpointing and fault detection.

Handles:
- TP-sharded model state extraction (each rank holds its own shard)
- DistributedOptimizer state (DP-sharded optimizer params)
- Virtual pipeline parallelism (multiple model chunks per rank)
- Parallel group discovery for SCOUT peer groups
"""

from __future__ import annotations

from typing import Any

import torch
import torch.distributed as dist

from lm_resiliency.adapters import FrameworkAdapter, ParallelismInfo
from lm_resiliency.detection.topology import (
    ReplayPeerGroup,
    ReplayPeerRole,
    normalize_replay_peer_role,
)
from lm_resiliency.integrations._common import create_gloo_peer_group


class MegatronAdapter(FrameworkAdapter):
    """Adapter bridging megatron-core with lm_resiliency.

    Each rank saves its LOCAL shard of model/optimizer state (TP-local,
    PP-local, DP-sharded optimizer). On recovery, each rank loads its own
    shard — no cross-rank resharding needed.

    Args:
        model: List of model chunks (one per virtual pipeline stage on this rank).
            Each chunk is typically a DistributedDataParallel-wrapped MegatronModule.
        optimizer: Megatron's DistributedOptimizer or MixedPrecisionOptimizer.
        opt_param_scheduler: LR scheduler (Megatron's OptimizerParamScheduler).
        extra_state: Additional state to checkpoint (e.g., iteration count, RNG state).
    """

    def __init__(
        self,
        model: list[Any],
        optimizer: Any,
        opt_param_scheduler: Any = None,
        extra_state: dict[str, Any] | None = None,
    ) -> None:
        self._model = model
        self._optimizer = optimizer
        self._opt_param_scheduler = opt_param_scheduler
        self._extra_state = extra_state or {}

    def get_state_dict(self) -> dict[str, Any]:
        """Extract local training state for in-memory checkpointing.

        Returns the TP-local, PP-local model state and DP-sharded optimizer
        state. Each rank's state is self-contained — no cross-rank communication.
        """
        state: dict[str, Any] = {}

        for i, model_chunk in enumerate(self._model):
            module = _unwrap_model_chunk(model_chunk)
            state[f"model_{i}"] = module.state_dict()

        state["optimizer"] = self._optimizer.state_dict()

        if hasattr(self._optimizer, "save_parameter_state"):
            param_state = {}
            self._optimizer.save_parameter_state(param_state)
            state["optimizer_param_state"] = param_state

        if self._opt_param_scheduler is not None:
            state["opt_param_scheduler"] = self._opt_param_scheduler.state_dict()

        state.update(self._extra_state)
        return state

    def collect_checkpoint_tensors(self) -> list[torch.Tensor]:
        """Collect raw tensor references for near-zero-overhead checkpointing.

        Returns direct references to model parameters and optimizer state tensors
        (exp_avg, exp_avg_sq, fp32 master copies). No state_dict() call, no dict
        assembly — just the underlying storage that save_tensors() can async-copy.

        The tensor list order is deterministic (model chunks in order, then
        optimizer state keyed by parameter). On recovery, load_checkpoint_tensors()
        writes back into these same references.
        """
        tensors: list[torch.Tensor] = []

        # Model parameters (TP-local shards)
        for model_chunk in self._model:
            module = _unwrap_model_chunk(model_chunk)
            for p in module.parameters():
                tensors.append(p.data)

        # DistributedOptimizer master shards + their optimizer state. The fp32 main
        # params (base_opt's params) are the optimizer's source of truth — distinct
        # storage from the model params (which the optimizer re-syncs *from* the mains
        # each step). Checkpointing only the model params would resume the optimizer from
        # stale master weights, so the mains must be captured alongside their moments.
        base_opt = _get_base_optimizer(self._optimizer)
        for param_group in base_opt.param_groups:
            for p in param_group["params"]:
                tensors.append(p.data)
                state = base_opt.state.get(p)
                if state is None:
                    continue
                for key in sorted(state.keys()):
                    v = state[key]
                    if isinstance(v, torch.Tensor):
                        tensors.append(v)

        return tensors

    def materialize_optimizer_state(self) -> None:
        """Allocate DistributedOptimizer state on a fresh engine using Megatron's own
        ``_init_optimizer_states_with_dummy_values`` — which Megatron ships expressly so a
        distributed-checkpoint load can replace the states in-place. Without it a
        just-restarted optimizer holds no exp_avg/exp_avg_sq for the reload to copy into,
        so collect() returns only the model params (fewer tensors than were saved)."""
        if _get_base_optimizer(self._optimizer).state:
            return  # already allocated (e.g. in-process reload) — nothing to materialize
        for opt in _state_initializable_optimizers(self._optimizer):
            opt._init_optimizer_states_with_dummy_values()

    def load_checkpoint_tensors(self, saved_tensors: list[torch.Tensor]) -> None:
        """Restore checkpoint by copying saved tensors back into live references.

        Inverse of collect_checkpoint_tensors(). Copies from CPU buffers (loaded
        from disk) back into the GPU tensors that the model/optimizer hold.
        """
        live_tensors = self.collect_checkpoint_tensors()
        assert len(saved_tensors) == len(live_tensors), (
            f"Tensor count mismatch: saved {len(saved_tensors)} vs live {len(live_tensors)}"
        )
        for live, saved in zip(live_tensors, saved_tensors):
            live.data.copy_(saved)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Apply a checkpoint state dict to all framework components."""
        for i, model_chunk in enumerate(self._model):
            key = f"model_{i}"
            if key in state_dict:
                module = _unwrap_model_chunk(model_chunk)
                module.load_state_dict(state_dict[key])

        if "optimizer" in state_dict:
            self._optimizer.load_state_dict(state_dict["optimizer"])

        if "optimizer_param_state" in state_dict and hasattr(
            self._optimizer, "load_parameter_state"
        ):
            self._optimizer.load_parameter_state(state_dict["optimizer_param_state"])

        if "opt_param_scheduler" in state_dict and self._opt_param_scheduler is not None:
            self._opt_param_scheduler.load_state_dict(state_dict["opt_param_scheduler"])

    def get_parallelism_info(self) -> ParallelismInfo:
        """Return parallelism configuration from megatron-core's mpu."""
        from megatron.core import mpu

        tp = mpu.get_tensor_model_parallel_world_size()
        pp = mpu.get_pipeline_model_parallel_world_size()
        world_size = dist.get_world_size()
        try:
            discovered_dp = mpu.get_data_parallel_world_size(with_context_parallel=False)
            if not isinstance(discovered_dp, int):
                raise TypeError("data-parallel world size is not an integer")
            dp = discovered_dp
        except (AttributeError, TypeError):
            cp_getter = getattr(mpu, "get_context_parallel_world_size", None)
            cp = int(cp_getter()) if callable(cp_getter) else 1
            dp = world_size // (tp * pp * cp)

        # Check if DP uses replication (HSDP-like)
        # megatron-core doesn't split DP into replicate/shard natively,
        # but when using DistributedOptimizer, optimizer state is sharded across DP
        dp_replicate = 1
        dp_shard = dp
        if not _optimizer_is_distributed(self._optimizer):
            dp_replicate = dp
            dp_shard = 1

        return ParallelismInfo(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp,
            pp=pp,
            world_size=world_size,
        )

    @property
    def rank(self) -> int:
        return dist.get_rank()

    @property
    def world_size(self) -> int:
        return dist.get_world_size()

    def get_dp_group(self) -> dist.ProcessGroup:
        """Return the data-parallel process group from megatron-core."""
        from megatron.core import mpu

        return mpu.get_data_parallel_group()

    def get_replay_peer_group(
        self,
        role: ReplayPeerRole | str,
    ) -> ReplayPeerGroup:
        """Resolve equivalent dense or expert replicas from Megatron parallel state."""
        peer_role = normalize_replay_peer_role(role)
        if not dist.is_available() or not dist.is_initialized():
            return ReplayPeerGroup(peer_role, None, None)

        from megatron.core import mpu

        if peer_role is ReplayPeerRole.EXPERT:
            nccl_group = mpu.get_expert_data_parallel_group()
            gloo_getter = getattr(mpu, "get_expert_data_parallel_group_gloo", None)
        else:
            try:
                nccl_group = mpu.get_data_parallel_group(with_context_parallel=False)
            except TypeError:
                nccl_group = mpu.get_data_parallel_group()
            gloo_getter = getattr(mpu, "get_data_parallel_group_gloo", None)

        gloo_group = None
        if callable(gloo_getter):
            try:
                gloo_group = gloo_getter()
            except (AssertionError, RuntimeError, TypeError):
                gloo_group = None
        if gloo_group is None:
            gloo_group = create_gloo_peer_group(nccl_group)
        return ReplayPeerGroup(peer_role, gloo_group, nccl_group)

    def get_repeated_layers(self) -> list[torch.nn.Module] | None:
        """Return the transformer layer ModuleList for SCOUT replay.

        Virtual-pipeline chunks are flattened in local execution order so layer
        rotation covers every chunk owned by this rank.
        """
        all_layers: list[torch.nn.Module] = []
        for model_chunk in self._model:
            module = _unwrap_model_chunk(model_chunk)
            layers = _find_transformer_layers(module)
            if layers is not None:
                all_layers.extend(layers)
        return all_layers or None

    def get_base_optimizers(self) -> list[torch.optim.Optimizer]:
        """Return every base optimizer invoked by Megatron's outer wrapper."""
        return _get_base_optimizers(self._optimizer)

    @property
    def optimizer_state_is_sharded(self) -> bool:
        """Whether raw optimizer tensors differ across ordinary DP peers."""
        return _optimizer_is_distributed(self._optimizer)

    def get_optimizer_replica_group(self) -> dist.ProcessGroup | None:
        """Return Megatron's HSDP-like inter-optimizer-instance group."""
        if not _optimizer_is_distributed(self._optimizer) or not dist.is_initialized():
            return None
        try:
            from megatron.core import mpu

            getter = getattr(mpu, "get_inter_distributed_optimizer_instance_group", None)
            if not callable(getter):
                return None
            try:
                group = getter(check_initialized=False)
            except TypeError:
                group = getter()
            if group is None or dist.get_world_size(group) < 2:
                return None
            return group
        except (AssertionError, RuntimeError):
            return None


def _unwrap_model_chunk(model_chunk: Any) -> Any:
    """Unwrap DDP/Float16Module wrappers to get the base MegatronModule."""
    module = model_chunk
    if hasattr(module, "module"):
        module = module.module
    if hasattr(module, "module"):
        module = module.module
    return module


def _find_transformer_layers(module: Any) -> torch.nn.ModuleList | None:
    """Find TransformerBlock.layers within a megatron-core model module."""
    # Direct path: module.decoder.layers (GPTModel pattern)
    decoder = getattr(module, "decoder", None)
    if decoder is not None:
        layers = getattr(decoder, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers

    # Alternative: module.encoder.layers
    encoder = getattr(module, "encoder", None)
    if encoder is not None:
        layers = getattr(encoder, "layers", None)
        if isinstance(layers, torch.nn.ModuleList):
            return layers

    # Fallback: search for any TransformerBlock
    for _, child in module.named_modules():
        if type(child).__name__ == "TransformerBlock":
            layers = getattr(child, "layers", None)
            if isinstance(layers, torch.nn.ModuleList):
                return layers

    return None


def _get_base_optimizer(optimizer: Any) -> Any:
    """Unwrap to the base torch.optim.Optimizer that holds param_groups/state.

    Megatron's DistributedOptimizer stores it in .optimizer;
    MixedPrecisionOptimizer may nest further.
    """
    opt = optimizer
    for attr in ("optimizer", "_inner", "optim"):
        inner = getattr(opt, attr, None)
        if inner is not None and hasattr(inner, "param_groups"):
            return inner
    if hasattr(opt, "param_groups"):
        return opt
    raise AttributeError(
        f"Cannot find base optimizer with param_groups in {type(optimizer).__name__}"
    )


def _get_base_optimizers(optimizer: Any) -> list[torch.optim.Optimizer]:
    """Unwrap a single optimizer or every member of a ChainedOptimizer."""
    chain = getattr(optimizer, "chained_optimizers", None)
    candidates = list(chain) if chain is not None else [optimizer]
    optimizers: list[torch.optim.Optimizer] = []
    seen: set[int] = set()
    for candidate in candidates:
        base = _get_base_optimizer(candidate)
        if id(base) not in seen:
            optimizers.append(base)
            seen.add(id(base))
    return optimizers


def _optimizer_is_distributed(optimizer: Any) -> bool:
    """Check if the optimizer shards state across DP ranks."""
    chain = getattr(optimizer, "chained_optimizers", None)
    if chain is not None:
        return any(_optimizer_is_distributed(candidate) for candidate in chain)
    type_name = type(optimizer).__name__
    return "DistributedOptimizer" in type_name or "distributed" in type_name.lower()


def _state_initializable_optimizers(optimizer: Any) -> list[Any]:
    """The optimizer(s) exposing Megatron's native state materializer.

    ``get_megatron_optimizer`` wraps the per-model-group DistributedOptimizers in a
    ChainedOptimizer; return each that can init its own dummy states.
    """
    chain = getattr(optimizer, "chained_optimizers", None)
    opts = list(chain) if chain is not None else [optimizer]
    return [o for o in opts if hasattr(o, "_init_optimizer_states_with_dummy_values")]
