"""Native PyTorch FSDP2/HSDP resiliency via GEMINI's local-shard fast path.

FSDP2 shards parameters as DTensor. Rather than the state_dict route (which
can't copy DTensor), this checkpoints each rank's **local shard** through
``save_tensors`` — exactly the DeepSpeed/Megatron mechanism — so recovery reloads the
local shards in place.

The native PyTorch entry point uses this runtime when the model carries FSDP2
DTensor parameters.
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from functools import partial
from typing import Any, Callable

import torch
import torch.distributed as dist
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.adapters import ParallelismInfo
from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import DurableCheckpointConfig
from lm_resiliency.checkpointing.manager import (
    InMemoryCheckpointManager,
    RecoveryMode,
    _local_shard,
)
from lm_resiliency.checkpointing.rng import RNG_KEY
from lm_resiliency.detection.c3 import C3
from lm_resiliency.detection.layer_replay import (
    FSDP_PARAMETER_ALL_GATHER,
    ReplayInvocation,
    ReplayResult,
)
from lm_resiliency.detection.peer_group import (
    form_detection_groups,
    parallelism_device_mesh,
)
from lm_resiliency.detection.replay_harness import ModelReplayHarness, ReplayHarnessConfig
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.detection.temporal import SCOUT_TEMPORAL_KEY
from lm_resiliency.handle import ResiliencyHandle
from lm_resiliency.integrations._checkpoint_certification import (
    CheckpointCertificationCoordinator,
)
from lm_resiliency.integrations._common import (
    build_checkpoint_manager,
    build_durable_checkpoint,
    checkpoint_extra,
    recover_with_fallback,
    restore_checkpoint_extra,
)
from lm_resiliency.integrations.pytorch.gradient_replay import (
    replay_fsdp_gradient_communication,
)
from lm_resiliency.orchestration import (
    OrchestrationHooks,
    _bind_orchestration,
    _resolve_orchestration_callbacks,
)

logger = logging.getLogger(__name__)


def _prepare_root_managed_fsdp_invocation(
    module: torch.nn.Module,
    invocation: ReplayInvocation,
) -> ReplayInvocation:
    """Wrap local replay tensors for a root-managed DTensor boundary module."""
    if getattr(module, "_fsdp_state", None) is not None:
        return invocation

    try:
        from torch.distributed.tensor import DTensor, Replicate
    except ImportError:
        return invocation

    parameter = next(
        (candidate for candidate in module.parameters() if isinstance(candidate, DTensor)),
        None,
    )
    if parameter is None:
        return invocation

    placements = tuple(Replicate() for _ in parameter.placements)
    leaves, spec = tree_flatten((invocation.args, invocation.kwargs, invocation.grad_output))
    wrapped = [
        (
            DTensor.from_local(
                leaf,
                device_mesh=parameter.device_mesh,
                placements=placements,
                run_check=False,
                shape=leaf.shape,
                stride=leaf.stride(),
            )
            if isinstance(leaf, torch.Tensor) and not isinstance(leaf, DTensor)
            else leaf
        )
        for leaf in leaves
    ]
    args, kwargs, grad_output = tree_unflatten(wrapped, spec)
    return ReplayInvocation(
        args=args,
        kwargs=kwargs,
        input_requires_grad=list(invocation.input_requires_grad),
        grad_output=grad_output,
        autocast_enabled=invocation.autocast_enabled,
        autocast_device_type=invocation.autocast_device_type,
        autocast_dtype=invocation.autocast_dtype,
    )


def _materialize_pure_fsdp_evidence(
    tensor_groups: dict[str, list[torch.Tensor]],
) -> dict[str, list[torch.Tensor]]:
    """Materialize global DTensor values when pure FSDP has no replica oracle."""

    def materialize(tensor: torch.Tensor) -> torch.Tensor:
        full_tensor = getattr(tensor, "full_tensor", None)
        if type(tensor).__name__ == "DTensor" and callable(full_tensor):
            return full_tensor()
        return tensor

    return {
        name: [materialize(tensor) for tensor in tensors] for name, tensors in tensor_groups.items()
    }


def has_dtensor_params(model: torch.nn.Module) -> bool:
    """True if any parameter is sharded by FSDP2 or tensor parallelism."""
    return any(type(p).__name__ == "DTensor" for p in model.parameters())


def has_fsdp_modules(model: torch.nn.Module) -> bool:
    """Whether FSDP2 attached runtime state to any module."""
    return any(getattr(module, "_fsdp_state", None) is not None for module in model.modules())


class PyTorchFSDPResiliency(ResiliencyHandle):
    """GEMINI and SCOUT for a native FSDP2/HSDP model.

    Checkpoints model params and Adam optimizer moments as local shards; recovery
    materializes lazily-created optimizer state, then copies reloaded shards in place.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        ckpt_config: InMemoryCkptConfig | None = None,
        detection_config: ReplayHarnessConfig | None = None,
        device: torch.device | None = None,
        fault_callback: Callable[[ReplayResult], None] | None = None,
        oob_fault_callback: SCOUTFaultCallback | None = None,
        group: dist.ProcessGroup | None = None,
        nccl_group: dist.ProcessGroup | None = None,
        parallelism_info: Any | None = None,
        extra_state_fn: Callable[[], dict[str, Any]] | None = None,
        load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
        durable_checkpoint: DurableCheckpointConfig | None = None,
    ) -> None:
        super().__init__()
        self._model = model
        self._optimizer = optimizer
        self._device = device or torch.device("cuda")
        self._parallelism_info = infer_parallelism_info(model, parallelism_info)
        self._has_fsdp = has_fsdp_modules(model) or _effective_dp_shard(self._parallelism_info) > 1
        self._is_hsdp = _is_hsdp_model(model, self._parallelism_info)
        self._fault_callback = fault_callback
        self._extra_state_fn = extra_state_fn
        self._load_extra_state_fn = load_extra_state_fn
        self._local_shard_c3: C3 | None = None

        self.ckpt_manager, self._ckpt_interval = build_checkpoint_manager(
            ckpt_config,
            manager_factory=InMemoryCheckpointManager,
            parallelism_info=self._parallelism_info,
        )
        self._compare_updated_weights = False
        if detection_config is not None:
            if (group is None) != (nccl_group is None):
                raise ValueError("group and nccl_group must be supplied together")
            if group is None:
                topology_model = model
                if (
                    detection_config.workload is not None
                    and detection_config.workload.replay_modules
                ):
                    topology_model = detection_config.workload.replay_modules[0]
                workload = detection_config.workload
                expert = bool(workload is not None and workload.peer_role.value == "expert")
                group, nccl_group = form_detection_groups(
                    topology_model,
                    device_mesh=parallelism_device_mesh(
                        parallelism_info,
                        expert=expert,
                    ),
                )
            if self._is_hsdp and group is not None and dist.is_initialized():
                self._local_shard_c3 = C3(group=group)
            try:
                self.replay_harness = ModelReplayHarness(
                    model=model,
                    optimizer=None,
                    group=group,
                    nccl_group=nccl_group,
                    device=self._device,
                    config=(
                        replace(
                            detection_config,
                            compare_parameter_state=self._is_hsdp,
                        )
                        if self._has_fsdp
                        else detection_config
                    ),
                    callback=self._fault_callback,
                    oob_fault_callback=oob_fault_callback,
                    gradient_communication=(
                        replay_fsdp_gradient_communication if self._has_fsdp else None
                    ),
                    invocation_preparer=(
                        _prepare_root_managed_fsdp_invocation if self._has_fsdp else None
                    ),
                    evidence_preparer=(
                        _materialize_pure_fsdp_evidence
                        if self._has_fsdp and not self._is_hsdp
                        else None
                    ),
                )
                peer_count = (
                    dist.get_world_size(group)
                    if group is not None and dist.is_initialized()
                    else _world_size()
                )
                self._compare_updated_weights = self._is_hsdp or (
                    not self._has_fsdp and peer_count > 1
                )
                if not self._compare_updated_weights and _is_log_rank():
                    logger.info(
                        "SCOUT parameter-state and updated-weight comparison requires "
                        "equivalent HSDP shard replicas and is unavailable for pure FSDP. "
                        "Layer replay remains active."
                    )
            except ValueError as exc:
                logger.warning(f"SCOUT: {exc} Detection disabled.")

        self._cadence = ResiliencyCadence.from_component_intervals(
            checkpoint_interval=self._ckpt_interval,
            detection_interval=(
                detection_config.check_interval
                if self.replay_harness is not None and detection_config is not None
                else 0
            ),
        )
        self.durable_checkpoint = build_durable_checkpoint(
            durable_checkpoint,
            self.replay_harness,
        )
        self._certification = CheckpointCertificationCoordinator(
            checkpoint_manager=self.ckpt_manager,
            replay_harness=self.replay_harness,
            durable_checkpoint=self.durable_checkpoint,
            cadence=self._cadence,
            checkpoint_tensors=self._collect,
            checkpoint_extra=self._checkpoint_extra,
            fault_callback=self._fault_callback,
            logger=logger,
        )
        self._set_prepare_recovery_callback(
            lambda failure_kind, all_ranks_accessible, step: self._certification.prepare_recovery(
                failure_kind=failure_kind,
                all_ranks_accessible=all_ranks_accessible,
                check_all_recipes=self._check_all_replay_recipes,
                step=step,
            )
        )

        # Post-step hook: save local shards and run replay at their configured intervals.
        self._register_hook(optimizer.register_step_post_hook(self._post_step))

    # ── the sharded tensor set (params + Adam moments, in a stable order) ────────
    def _collect(self) -> list[torch.Tensor]:
        # Model params, then every optimizer-state tensor in a stable (sorted-key) order.
        # Including "step" matters: without the restored step counter the recovered Adam
        # uses the wrong bias correction and the resumed trajectory diverges from baseline.
        tensors: list[torch.Tensor] = list(self._model.parameters())
        for p in self._model.parameters():
            st = self._optimizer.state.get(p, {})
            for key in sorted(st.keys()):
                v = st[key]
                if isinstance(v, torch.Tensor):
                    tensors.append(v)
        return tensors

    def _materialize_optimizer_state(self) -> None:
        """Create optimizer tensors that PyTorch allocates lazily on the first step."""
        if isinstance(self._optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            step_on_parameter_device = bool(
                self._optimizer.defaults.get("capturable") or self._optimizer.defaults.get("fused")
            )
            for parameter in self._model.parameters():
                state = self._optimizer.state[parameter]
                if "exp_avg" in state:
                    continue
                step_device = parameter.device if step_on_parameter_device else torch.device("cpu")
                state["step"] = torch.zeros((), dtype=torch.float32, device=step_device)
                state["exp_avg"] = torch.zeros_like(parameter)
                state["exp_avg_sq"] = torch.zeros_like(parameter)
            return

        if isinstance(self._optimizer, torch.optim.SGD):
            for group in self._optimizer.param_groups:
                if float(group.get("momentum", 0.0)) == 0.0:
                    continue
                for parameter in group["params"]:
                    self._optimizer.state[parameter].setdefault(
                        "momentum_buffer",
                        torch.zeros_like(parameter),
                    )

    def _post_step(self, opt: Any, args: Any, kwargs: Any) -> None:
        del opt, args, kwargs
        step = self._advance_step()

        if self.replay_harness is None:
            self._certification.post_step(step)
            return

        replay_result = None
        if self.replay_harness.replay_due(step):
            replay = self.replay_harness.step
            if self._compare_updated_weights and self.replay_harness.optimizer_replay_due(step):
                replay = partial(
                    self.replay_harness.step,
                    optimizer=self._optimizer,
                    allow_local_dtensor_shards=True,
                )
            replay_result = self._run_with_unsharded_model(replay)
            self._certification.apply_result(step, replay_result)
        else:
            replay_result = self.replay_harness.step()

    def _run_with_unsharded_model(self, replay: Callable[[], ReplayResult | None]):
        """Materialize FSDP2 params while the sampled layer is replayed directly."""
        assert self.replay_harness is not None
        target = self.replay_harness.target_layer
        communication_ranks = _fsdp_communication_ranks(
            target,
            self._parallelism_info,
        )
        message_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in target.parameters()
        )
        shard_bitmap = None
        if self._is_hsdp:
            shard_bitmap = self._check_local_parameter_shards(target)
        replay_modules = (target, *self.replay_harness.dense_boundary_modules)
        owners: list[torch.nn.Module] = []
        for module in replay_modules:
            owner = module if callable(getattr(module, "unshard", None)) else self._model
            if all(owner is not existing for existing in owners):
                owners.append(owner)
        self._synchronize_device()
        materialize_start = time.perf_counter()
        for owner in owners:
            unshard = getattr(owner, "unshard", None)
            if callable(unshard):
                unshard()
        self._synchronize_device()
        materialize_ms = (time.perf_counter() - materialize_start) * 1000.0
        try:
            result = replay()
            if result is not None:
                self.replay_harness.add_communication_timing(
                    result,
                    name=FSDP_PARAMETER_ALL_GATHER,
                    elapsed_ms=materialize_ms,
                    group_ranks=communication_ranks,
                    topology_role="fsdp",
                    message_bytes=message_bytes,
                )
                if shard_bitmap is not None:
                    result.sdc_source_bitmaps["local_parameter_shard"] = shard_bitmap
                    if any(shard_bitmap):
                        result.sdc_sources.append("local_parameter_shard")
                        result.sdc_bitmap = [
                            int(bool(current) or bool(shard))
                            for current, shard in zip(result.sdc_bitmap, shard_bitmap)
                        ]
            return result
        finally:
            for owner in reversed(owners):
                reshard = getattr(owner, "reshard", None)
                if callable(reshard):
                    reshard()

    def _synchronize_device(self) -> None:
        if self._device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.synchronize(self._device)

    def _check_local_parameter_shards(
        self,
        target: torch.nn.Module,
    ) -> list[int]:
        if self._local_shard_c3 is None:
            assert self.replay_harness is not None
            return self.replay_harness.check_local_parameter_shards()
        local_result = self._local_shard_c3.run_tensor_sequence(
            [_local_shard(parameter) for parameter in target.parameters()]
        )
        return local_result.bitmap

    def try_recover(self, mode: RecoveryMode | str | None = None) -> int:
        """Reload local shards from node-local (in place). Returns the step, or -1."""
        if self.ckpt_manager is None:
            return -1
        result = self.ckpt_manager.load_tensors(mode=mode)
        if result is None:
            return -1
        self._materialize_optimizer_state()
        saved, step, extra = result
        targets = self._collect()
        if len(saved) != len(targets):
            raise RuntimeError(
                "FSDP2 checkpoint tensor layout does not match the live model and "
                f"optimizer: saved={len(saved)}, live={len(targets)}"
            )
        for index, (target, checkpoint_tensor) in enumerate(zip(targets, saved)):
            local_target = _local_shard(target)
            if (
                checkpoint_tensor.shape != local_target.shape
                or checkpoint_tensor.dtype != local_target.dtype
            ):
                raise RuntimeError(
                    "FSDP2 checkpoint tensor layout does not match the live model "
                    f"and optimizer at index {index}: checkpoint="
                    f"{tuple(checkpoint_tensor.shape)}/{checkpoint_tensor.dtype}, "
                    f"live={tuple(local_target.shape)}/{local_target.dtype}"
                )
        with torch.no_grad():
            for target, latest in zip(targets, saved):
                _local_shard(target).detach().copy_(latest)
        values = extra or {}
        if self._load_extra_state_fn is not None:
            self._load_extra_state_fn(
                {
                    key: value
                    for key, value in values.items()
                    if key not in (RNG_KEY, SCOUT_TEMPORAL_KEY)
                }
            )
        self._restore_step(step)
        restore_checkpoint_extra(values, self.replay_harness)
        return step

    def _checkpoint_extra(self) -> dict[str, Any]:
        return checkpoint_extra(self.replay_harness, self._extra_state_fn)

    def check_now(self) -> ReplayResult | None:
        """Run SCOUT immediately when a training forward has supplied an activation."""
        if self.replay_harness is None or not self.replay_harness.has_replay_capture:
            return None
        replay = (
            self.replay_harness.check_shape_cycle
            if self.ckpt_manager is not None or self.durable_checkpoint is not None
            else self.replay_harness.check
        )
        if self._compare_updated_weights:
            replay = partial(
                replay,
                optimizer=self._optimizer,
                allow_local_dtensor_shards=True,
            )
        return self._run_with_unsharded_model(replay)

    def _check_all_replay_recipes(self) -> ReplayResult | None:
        if self.replay_harness is None or not self.replay_harness.has_replay_capture:
            return None
        replay = partial(
            self.replay_harness.check_shape_cycle,
            preserve_scheduler=True,
            optimizer=self._optimizer if self._compare_updated_weights else None,
            allow_local_dtensor_shards=self._compare_updated_weights,
        )
        return self._run_with_unsharded_model(replay)

    @property
    def step_count(self) -> int:
        return self._step_count

    @step_count.setter
    def step_count(self, value: int) -> None:
        self._restore_step(value)


def enable_fsdp2_resiliency(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    ckpt_config: InMemoryCkptConfig | None = None,
    detection_config: ReplayHarnessConfig | None = None,
    device: torch.device | None = None,
    fault_callback: Callable[[ReplayResult], None] | None = None,
    oob_fault_callback: SCOUTFaultCallback | None = None,
    orchestration: OrchestrationHooks | None = None,
    group: dist.ProcessGroup | None = None,
    nccl_group: dist.ProcessGroup | None = None,
    load_fallback: Callable[[], int | None] | None = None,
    parallelism_info: Any | None = None,
    extra_state_fn: Callable[[], dict[str, Any]] | None = None,
    load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
    durable_checkpoint: DurableCheckpointConfig | None = None,
    recovery_mode: RecoveryMode | str | None = None,
) -> PyTorchFSDPResiliency:
    """Build the native FSDP2/HSDP runtime and recover its local shards."""
    fault_callback, oob_fault_callback = _resolve_orchestration_callbacks(
        orchestration,
        fault_callback,
        oob_fault_callback,
    )
    res = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=ckpt_config,
        detection_config=detection_config,
        device=device,
        fault_callback=fault_callback,
        oob_fault_callback=oob_fault_callback,
        group=group,
        nccl_group=nccl_group,
        parallelism_info=parallelism_info,
        extra_state_fn=extra_state_fn,
        load_extra_state_fn=load_extra_state_fn,
        durable_checkpoint=durable_checkpoint,
    )
    _bind_orchestration(orchestration, res)
    recover_with_fallback(res, load_fallback, recovery_mode)
    return res


def _is_hsdp_model(model: torch.nn.Module, parallelism_info: Any | None) -> bool:
    info = infer_parallelism_info(model, parallelism_info)
    return int(info.dp_replicate) > 1 and _effective_dp_shard(info) > 1


def _effective_dp_shard(parallelism_info: Any) -> int:
    """Return the FSDP degree, including TorchTitan's folded CP dimension."""
    return int(parallelism_info.dp_shard) * int(getattr(parallelism_info, "cp", 1))


def infer_parallelism_info(
    model: torch.nn.Module,
    parallelism_info: Any | None = None,
) -> Any:
    """Infer FSDP2/HSDP dimensions from the parameter DeviceMesh."""
    if parallelism_info is not None and hasattr(parallelism_info, "dp_replicate"):
        return parallelism_info

    meshes = []
    explicit_mesh = parallelism_device_mesh(parallelism_info)
    if explicit_mesh is not None:
        meshes.append(explicit_mesh)
    meshes.extend(
        mesh
        for parameter in model.parameters()
        if (mesh := getattr(parameter, "device_mesh", None)) is not None
    )
    for mesh in meshes:
        names = getattr(mesh, "mesh_dim_names", None)
        if names:
            sizes = {name: int(mesh.size(index)) for index, name in enumerate(names)}
            return ParallelismInfo(
                dp_replicate=sizes.get("dp_replicate", 1),
                dp_shard=sizes.get(
                    "dp_shard",
                    sizes.get("fsdp", sizes.get("efsdp", sizes.get("dp", 1))),
                ),
                tp=sizes.get("tp", 1),
                pp=sizes.get("pp", 1),
                world_size=_world_size(),
            )
        mesh_tensor = getattr(mesh, "mesh", None)
        dp_shard = int(mesh.size(0)) if mesh_tensor is not None and mesh_tensor.ndim == 1 else 1
        return ParallelismInfo(dp_shard=dp_shard, world_size=_world_size())

    return ParallelismInfo(world_size=_world_size())


def _fsdp_communication_ranks(
    module: torch.nn.Module,
    parallelism_info: Any | None,
) -> tuple[int, ...] | None:
    """Return this rank's FSDP state-shard group for Cross-PG localization."""
    meshes = []
    explicit = parallelism_device_mesh(parallelism_info)
    if explicit is not None:
        meshes.append(explicit)
    for parameter in module.parameters():
        mesh = getattr(parameter, "device_mesh", None)
        if mesh is not None and all(mesh is not existing for existing in meshes):
            meshes.append(mesh)
    for candidate in module.modules():
        state = getattr(candidate, "_fsdp_state", None)
        mesh = getattr(state, "_device_mesh", None)
        if mesh is not None and all(mesh is not existing for existing in meshes):
            meshes.append(mesh)

    for mesh in meshes:
        names = tuple(getattr(mesh, "mesh_dim_names", None) or ())
        dimensions: list[str | int] = [
            name for name in ("dp_shard", "fsdp", "efsdp", "dp") if name in names
        ]
        mesh_tensor = getattr(mesh, "mesh", None)
        if not dimensions and mesh_tensor is not None and mesh_tensor.ndim == 1:
            dimensions.append(0)
        for dimension in dimensions:
            try:
                group = mesh.get_group(dimension)
                ranks = tuple(int(rank) for rank in dist.get_process_group_ranks(group))
            except (AssertionError, KeyError, RuntimeError, TypeError, ValueError):
                continue
            if len(ranks) > 1:
                return ranks
    return None


def _world_size() -> int:
    return dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1


def _is_log_rank() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
