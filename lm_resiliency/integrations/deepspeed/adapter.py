"""DeepSpeed FrameworkAdapter implementation.

Bridges DeepSpeed's engine with lm_resiliency's in-memory checkpointing
and fault detection.

Handles:
- ZeRO Stage 1/2: each rank holds full model params (bf16/fp16 flat groups)
  + its partition of fp32 master weights + optimizer states (exp_avg, exp_avg_sq)
- ZeRO Stage 3: each rank holds its partition of fp16 params (fp16_partitioned_groups_flat)
  + its partition of fp32 master weights + optimizer states
- Data-parallel group discovery for SCOUT peer groups
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist

from lm_resiliency.adapters import FrameworkAdapter, ParallelismInfo, materialize_adam_state
from lm_resiliency.detection.topology import (
    ReplayPeerGroup,
    ReplayPeerRole,
    normalize_replay_peer_role,
)
from lm_resiliency.integrations._common import (
    create_gloo_peer_group,
    notify_checkpoint_tensor_load,
)

if TYPE_CHECKING:
    pass


class DeepSpeedAdapter(FrameworkAdapter):
    """Adapter bridging DeepSpeed engine with lm_resiliency.

    Each rank saves its LOCAL partition of optimizer state. For ZeRO Stage 1/2,
    the model params are replicated (each rank has full bf16 params via all-gather),
    but optimizer state is partitioned. For ZeRO Stage 3, both params and optimizer
    are partitioned.

    The tensor collection order is deterministic: model params (full or partitioned),
    then fp32 master weight partitions, then optimizer state tensors.

    Args:
        engine: A DeepSpeed engine instance (from deepspeed.initialize()).
    """

    def __init__(self, engine: Any) -> None:
        self._engine = engine

    def get_state_dict(self) -> dict[str, Any]:
        """Extract local training state for in-memory checkpointing."""
        state: dict[str, Any] = {}
        state["module"] = self._engine.module.state_dict()
        state["optimizer"] = self._engine.optimizer.state_dict()
        if self._engine.lr_scheduler is not None:
            state["lr_scheduler"] = self._engine.lr_scheduler.state_dict()
        state["global_steps"] = self._engine.global_steps
        return state

    def collect_checkpoint_tensors(self) -> list[torch.Tensor]:
        """Collect raw tensor references for near-zero-overhead checkpointing.

        For ZeRO Stage 1/2:
          - bit16_groups_flat[i]: flattened bf16/fp16 params per group (full on each rank)
          - single_partition_of_fp32_groups[i]: this rank's fp32 master weight partition
          - optimizer.state[partition]: exp_avg, exp_avg_sq tensors

        For ZeRO Stage 3:
          - fp16_partitioned_groups_flat[i]: this rank's partition of fp16 params
          - fp32_partitioned_groups_flat[i]: this rank's fp32 master weights
          - optimizer.state[partition]: exp_avg, exp_avg_sq tensors

        Returns direct references — no dict assembly, no copies.
        """
        optimizer = self._engine.optimizer
        stage = self._zero_stage()

        tensors: list[torch.Tensor] = []

        if stage <= 2:
            tensors.extend(self._collect_stage1_2(optimizer))
        else:
            tensors.extend(self._collect_stage3(optimizer))

        return tensors

    def _collect_stage1_2(self, optimizer: Any) -> list[torch.Tensor]:
        """Collect tensors for ZeRO Stage 1/2."""
        tensors: list[torch.Tensor] = []

        # bf16/fp16 flat param groups (full model on each rank)
        for flat_group in optimizer.bit16_groups_flat:
            tensors.append(flat_group.data)

        # fp32 master weight partitions (this rank's shard)
        for fp32_partition in optimizer.single_partition_of_fp32_groups:
            tensors.append(fp32_partition.data)

        # Optimizer state (exp_avg, exp_avg_sq, etc.) keyed on fp32 partitions
        base_opt = optimizer.optimizer
        for fp32_partition in optimizer.single_partition_of_fp32_groups:
            state = base_opt.state.get(fp32_partition)
            if state is None:
                continue
            for key in sorted(state.keys()):
                v = state[key]
                if isinstance(v, torch.Tensor):
                    tensors.append(v)

        return tensors

    def _collect_stage3(self, optimizer: Any) -> list[torch.Tensor]:
        """Collect tensors for ZeRO Stage 3."""
        tensors: list[torch.Tensor] = []

        # fp16 partitioned params (this rank's partition)
        for flat_partition in optimizer.fp16_partitioned_groups_flat:
            if flat_partition is not None:
                tensors.append(flat_partition.data)

        # fp32 master weight partitions
        for fp32_partition in optimizer.fp32_partitioned_groups_flat:
            if fp32_partition.numel() > 0:
                tensors.append(fp32_partition.data)

        # Optimizer state
        base_opt = optimizer.optimizer
        for fp32_partition in optimizer.fp32_partitioned_groups_flat:
            if fp32_partition.numel() == 0:
                continue
            state = base_opt.state.get(fp32_partition)
            if state is None:
                continue
            for key in sorted(state.keys()):
                v = state[key]
                if isinstance(v, torch.Tensor):
                    tensors.append(v)

        return tensors

    def materialize_optimizer_state(self) -> None:
        """Allocate the base optimizer's Adam state on a fresh engine (ZeRO's momentum
        buffers are lazily created on the first step, so a just-restarted engine has
        none for the reload to copy into). Materialize on the fp32 master partitions —
        exactly the parameters the base optimizer steps — matching collect()'s ordering."""
        optimizer = self._engine.optimizer
        base_opt = optimizer.optimizer
        if self._zero_stage() <= 2:
            partitions = optimizer.single_partition_of_fp32_groups
        else:
            partitions = [p for p in optimizer.fp32_partitioned_groups_flat if p.numel() > 0]
        materialize_adam_state(base_opt, partitions)

    def load_checkpoint_tensors(self, saved_tensors: list[torch.Tensor]) -> None:
        """Restore checkpoint by copying saved tensors back into live references.

        Inverse of collect_checkpoint_tensors(). Copies from CPU buffers
        back into the GPU/CPU tensors that the engine holds.
        """
        live_tensors = self.collect_checkpoint_tensors()
        assert len(saved_tensors) == len(live_tensors), (
            f"Tensor count mismatch: saved {len(saved_tensors)} vs live {len(live_tensors)}"
        )
        for live, saved in zip(live_tensors, saved_tensors):
            live.data.copy_(saved)
        notify_checkpoint_tensor_load(self)

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Apply a checkpoint state dict to all framework components."""
        if "module" in state_dict:
            self._engine.module.load_state_dict(state_dict["module"])
        if "optimizer" in state_dict:
            self._engine.optimizer.load_state_dict(state_dict["optimizer"])
        if "lr_scheduler" in state_dict and self._engine.lr_scheduler is not None:
            self._engine.lr_scheduler.load_state_dict(state_dict["lr_scheduler"])
        if "global_steps" in state_dict:
            self._engine.global_steps = state_dict["global_steps"]

    def get_parallelism_info(self) -> ParallelismInfo:
        """Return parallelism configuration from DeepSpeed engine.

        ZeRO Stage 1/2: model is replicated across DP ranks, optimizer is partitioned.
          → dp_replicate = dp_world_size (full model replicas exist)
          → dp_shard = 1 (model NOT sharded)

        ZeRO Stage 3: both model and optimizer are partitioned.
          → dp_replicate = 1 (no full replicas)
          → dp_shard = dp_world_size
        """
        stage = self._zero_stage()
        dp_size = self._engine.dp_world_size
        world_size = self._engine.world_size

        if stage >= 3:
            dp_replicate = 1
            dp_shard = dp_size
        else:
            dp_replicate = dp_size
            dp_shard = 1

        tp = getattr(self._engine, "tp_world_size", 1) or 1
        pp = getattr(self._engine, "pp_world_size", 1) or 1

        return ParallelismInfo(
            dp_replicate=dp_replicate,
            dp_shard=dp_shard,
            tp=tp,
            pp=pp,
            world_size=world_size,
        )

    @property
    def rank(self) -> int:
        return self._engine.global_rank

    @property
    def world_size(self) -> int:
        return self._engine.world_size

    def get_dp_group(self) -> dist.ProcessGroup | None:
        """Return the data-parallel process group from the engine."""
        return self._engine.data_parallel_group

    def get_replay_peer_group(
        self,
        role: ReplayPeerRole | str,
        replay_modules: tuple[torch.nn.Module, ...] = (),
    ) -> ReplayPeerGroup:
        """Resolve equivalent dense or expert replicas from DeepSpeed groups."""
        peer_role = normalize_replay_peer_role(role)
        if not dist.is_available() or not dist.is_initialized():
            return ReplayPeerGroup(peer_role, None, None)
        if peer_role is ReplayPeerRole.EXPERT:
            nccl_group = self._expert_data_parallel_group(replay_modules)
        else:
            nccl_group = self._engine.data_parallel_group
        return ReplayPeerGroup(
            peer_role,
            create_gloo_peer_group(nccl_group),
            nccl_group,
        )

    def _expert_data_parallel_group(
        self,
        replay_modules: tuple[torch.nn.Module, ...],
    ) -> dist.ProcessGroup:
        groups = getattr(self._engine, "expert_data_parallel_group", None) or {}
        names = {
            str(group_name)
            for module in replay_modules
            for parameter in module.parameters()
            if (group_name := getattr(parameter, "group_name", None)) is not None
        }
        if len(names) > 1:
            raise ValueError(
                "SCOUT expert replay modules span multiple DeepSpeed expert groups: "
                f"{sorted(names)}"
            )
        if names:
            name = next(iter(names))
            if name not in groups:
                raise ValueError(f"DeepSpeed expert data group {name!r} is unavailable")
            return groups[name]
        if len(groups) == 1:
            return next(iter(groups.values()))
        if not groups:
            raise ValueError("DeepSpeed has no initialized expert-data-parallel group")
        raise ValueError(
            "DeepSpeed has multiple expert groups; replay module parameters must "
            "identify their group_name"
        )

    def get_repeated_layers(self) -> list[torch.nn.Module] | None:
        """Return the transformer layer ModuleList for SCOUT replay.

        Navigates common model hierarchies (HuggingFace, custom models)
        to find transformer blocks.
        """
        module = self._engine.module
        return _find_transformer_layers(module)

    def get_base_optimizers(self) -> list[torch.optim.Optimizer]:
        """Return the optimizer(s) that apply updates after ZeRO preprocessing."""
        optimizer = self._engine.optimizer
        candidates = [
            getattr(optimizer, "optimizer", None),
            getattr(optimizer, "backup_optimizer", None),
        ]
        if not any(candidate is not None for candidate in candidates) and hasattr(
            optimizer, "param_groups"
        ):
            candidates.append(optimizer)
        base_optimizers: list[torch.optim.Optimizer] = []
        seen: set[int] = set()
        for candidate in candidates:
            if (
                candidate is not None
                and hasattr(candidate, "param_groups")
                and id(candidate) not in seen
            ):
                base_optimizers.append(candidate)
                seen.add(id(candidate))
        if base_optimizers:
            return base_optimizers
        raise AttributeError(f"Cannot find DeepSpeed base optimizer in {type(optimizer).__name__}")

    def _zero_stage(self) -> int:
        """Return the ZeRO optimization stage (0, 1, 2, or 3)."""
        return self._engine.zero_optimization_stage()


def _find_transformer_layers(module: Any) -> list[torch.nn.Module] | None:
    """Find repeated transformer layers in a model module.

    Searches common patterns:
    - model.layers (Llama, Mistral)
    - model.transformer.h (GPT-2, GPT-J)
    - model.model.layers (HuggingFace wrapped)
    - encoder.layers / decoder.layers
    """
    # Direct: module.model.layers (HF LlamaModel)
    model = getattr(module, "model", None)
    if model is not None:
        layers = getattr(model, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and layers:
            return list(layers)

    # Direct: module.transformer.h (GPT-2/GPT-J)
    transformer = getattr(module, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if isinstance(h, torch.nn.ModuleList) and h:
            return list(h)
        layers = getattr(transformer, "layers", None)
        if isinstance(layers, torch.nn.ModuleList) and layers:
            return list(layers)

    # Direct: module.layers
    layers = getattr(module, "layers", None)
    if isinstance(layers, torch.nn.ModuleList) and layers:
        return list(layers)

    # Encoder/decoder
    for name in ("encoder", "decoder"):
        sub = getattr(module, name, None)
        if sub is not None:
            layers = getattr(sub, "layers", None) or getattr(sub, "layer", None)
            if isinstance(layers, torch.nn.ModuleList) and layers:
                return list(layers)

    # PipelineModule stores this rank's local stage as an execution list.
    forward_funcs = getattr(module, "forward_funcs", None)
    if forward_funcs is not None:
        local_layers = [item for item in forward_funcs if isinstance(item, torch.nn.Module)]
        if local_layers:
            return local_layers

    # Fallback: find the largest ModuleList
    best = None
    for _, child in module.named_modules():
        if isinstance(child, torch.nn.ModuleList) and child:
            if best is None or len(child) > len(best):
                best = child
    return list(best) if best is not None else None
