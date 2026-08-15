# mypy: ignore-errors
"""Model Replay Harness: captures activations during training, replays on demand.

Minimal integration: one line to attach, zero changes to training logic.

    harness = ModelReplayHarness(model, optimizer, group=dp_group, nccl_group=...)
    # training loop runs as normal — detection is fully automatic

The harness:
  1. Identifies the repeated hidden layers in the model.
  2. Captures a selected layer's complete args, kwargs, and backward signal.
  3. Materializes the next shape from one common dense/MoE replay plan.
  4. Registers an optimizer post-step hook to auto-trigger detection every N steps.
  5. Confirms timing anomalies, decomposes them, and rotates shape/layer coverage.

Captures are overwritten every step — only the most recent step's tensors are held.
Memory cost is bounded to one layer invocation plus its output gradients.
"""

from __future__ import annotations

import logging
import statistics
from typing import Any, Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.detection.all_to_all_replay import (
    AllToAllReplayExecutor,
    AllToAllReplayRecipe,
)
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.config import ReplayHarnessConfig
from lm_resiliency.detection.cross_pg import CrossPGCoordinator, CrossPGResult
from lm_resiliency.detection.hang_instrumentation import HangInstrumentation
from lm_resiliency.detection.layer_replay import (
    GradientCommunicationReplay,
    LayerReplayDetector,
    ReplayEvidencePreparer,
    ReplayInvocation,
    ReplayInvocationPreparer,
    ReplayResult,
    replay_result_has_fault,
    replay_result_has_sdc,
)
from lm_resiliency.detection.oob_service import OOBHangConfig, OOBHangService
from lm_resiliency.detection.optimizer_step import (
    OptimizerReplayBatch,
    OptimizerStepCheckUnsupported,
    OptimizerStepEvidence,
    collect_updated_weights,
)
from lm_resiliency.detection.replay_analysis import (
    has_timing_candidate as _has_timing_candidate,
)
from lm_resiliency.detection.replay_analysis import (
    merge_replay_rounds as _merge_replay_rounds,
)
from lm_resiliency.detection.replay_analysis import (
    merge_sdc_source_bitmaps as _merge_sdc_source_bitmaps,
)
from lm_resiliency.detection.replay_shapes import (
    ReplayShape,
    ReplayShapeScheduler,
    ReplayWorkload,
)
from lm_resiliency.detection.reports import SCOUTFaultCallback
from lm_resiliency.detection.stage_instrumentation import (
    CheckpointOperation,
    InstrumentedDataLoader,
    checkpoint_io,
    instrument_dataloader,
)
from lm_resiliency.detection.temporal import (
    TemporalAssessment,
    TemporalBaselineStore,
    replay_baseline_key,
)
from lm_resiliency.detection.topology import ReplayPeerGroup

logger = logging.getLogger(__name__)


def enable_replay_detection(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    group: dist.ProcessGroup | None = None,
    nccl_group: dist.ProcessGroup | None = None,
    device: torch.device | None = None,
    check_interval: int = 50,
    callback: Callable[[ReplayResult], None] | None = None,
    oob_fault_callback: SCOUTFaultCallback | None = None,
    workload: ReplayWorkload | None = None,
    peer_group: ReplayPeerGroup | None = None,
) -> ModelReplayHarness:
    """Enable automatic SDC and straggler detection and return its lifecycle handle.

    One line in the training script — no further changes needed:

        enable_replay_detection(model, optimizer, group=dp_gloo, nccl_group=dp_nccl)

    Detection runs automatically every `check_interval` optimizer steps.
    Faults are reported via `callback` (or logged as warnings if no callback).

    Args:
        model: The full model. Repeated hidden layers are auto-detected.
        optimizer: Training optimizer (hooks into post-step).
        group: Gloo process group for the DP peer group.
        nccl_group: NCCL process group for GPU tensor operations.
        device: CUDA device. Defaults to current device.
        check_interval: Run detection every N steps.
        callback: Called with ReplayResult when a fault is detected.
        oob_fault_callback: Called with JSON-ready hang and DataLoader-stall reports.
        workload: Common dense/MoE replay modules, shape list, and materializer.
            Omit for the one-shape dense default.
    """
    has_concrete_shapes = workload is not None and any(
        shape.dimensions is not None for shape in workload.shape_plan.shapes
    )
    config = ReplayHarnessConfig(
        check_interval=check_interval,
        capture_inputs_by_value=has_concrete_shapes,
        workload=workload,
        scale_factors=[0.1, 1.0, 10.0],
    )
    return ModelReplayHarness(
        model,
        optimizer=optimizer,
        group=group,
        nccl_group=nccl_group,
        device=device,
        config=config,
        callback=callback,
        oob_fault_callback=oob_fault_callback,
        peer_group=peer_group,
    )


def _as_replay_modules(module: nn.Module | None) -> Sequence[nn.Module] | None:
    if isinstance(module, nn.ModuleDict):
        modules = tuple(module.values())
    elif isinstance(module, (nn.ModuleList, nn.Sequential)):
        modules = module
    else:
        return None
    if not modules:
        return None
    return modules


def find_repeated_layers(model: nn.Module) -> Sequence[nn.Module] | None:
    """Find the repeated hidden layer block in an LLM.

    Handles common patterns:
      - model.layers (Llama, Mistral)
      - model.model.layers (HuggingFace wrapper)
      - model.transformer.h (GPT-2, GPT-Neo)
      - model.transformer.layers (some custom)

    Returns the ModuleList containing the repeated blocks, or None.
    """
    candidates = [
        ("layers",),
        ("model", "layers"),
        ("transformer", "h"),
        ("transformer", "layers"),
    ]

    for path in candidates:
        module = model
        for attr in path:
            module = getattr(module, attr, None)
            if module is None:
                break
        repeated = _as_replay_modules(module)
        if repeated is not None:
            return repeated

    for _, child in model.named_modules():
        repeated = _as_replay_modules(child)
        if repeated is not None and len(repeated) > 1:
            first_type = type(repeated[0])
            if all(type(module) is first_type for module in repeated):
                return repeated

    return None


def _find_module_at_paths(
    model: nn.Module,
    paths: Sequence[tuple[str, ...]],
) -> nn.Module | None:
    roots = [model]
    seen = {id(model)}
    root = model
    while True:
        wrapped = getattr(root, "module", None)
        if not isinstance(wrapped, nn.Module) or id(wrapped) in seen:
            break
        roots.append(wrapped)
        seen.add(id(wrapped))
        root = wrapped

    for root in roots:
        for path in paths:
            module: Any = root
            for attr in path:
                module = getattr(module, attr, None)
                if module is None:
                    break
            if isinstance(module, nn.Module):
                return module
    return None


def find_embedding_layer(model: nn.Module) -> nn.Module | None:
    """Find a model's token-embedding replay boundary."""
    return _find_module_at_paths(
        model,
        (
            ("model", "embed_tokens"),
            ("transformer", "wte"),
            ("tok_embeddings",),
            ("embed_tokens",),
            ("embeddings",),
            ("embedding",),
            ("embed",),
        ),
    )


def find_output_layer(model: nn.Module) -> nn.Module | None:
    """Find a model's language-model output replay boundary."""
    return _find_module_at_paths(
        model,
        (
            ("lm_head",),
            ("output_layer",),
            ("output",),
            ("head",),
        ),
    )


class ModelReplayHarness:
    """Captures activations during training and replays for fault detection.

    Attaches to a model's repeated hidden layer block. During training,
    forward hooks silently capture the input activation and attach tensor hooks
    to capture the selected layer's output gradients. An optimizer post-step
    hook auto-triggers detection every N steps — no manual step() call needed.

    Args:
        model: The full model (e.g., LlamaForCausalLM, GPT, etc.).
        optimizer: Training optimizer. If provided, registers a post-step hook
            for automatic detection. If None, use step() manually.
        group: Gloo process group for scalar C3 operations.
        nccl_group: NCCL process group for GPU tensor C3 and broadcast.
        device: CUDA device for replay.
        config: ReplayHarnessConfig with layer_index, check_interval, etc.
        layers: Backward-compatible name for explicit replay modules.
        replay_modules: Explicit module boundaries to capture and replay. For a
            decoupled MoE replay, pass post-dispatch expert modules so their inputs
            are the routed-token tensors rather than the outer transformer input.
            Prefer ``config.workload.replay_modules`` for the unified shape API.
        callback: Called with ReplayResult when a fault is detected. If None,
            faults are logged as warnings.
        oob_fault_callback: Called with JSON-ready hang and DataLoader-stall reports.
        gradient_communication: Optional framework-specific gradient communication
            replay invoked with the diagnostic parameter gradients.
        peer_group: Framework-resolved equivalent peers for the workload's dense
            or expert model-state role.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        group: dist.ProcessGroup | None = None,
        nccl_group: dist.ProcessGroup | None = None,
        device: torch.device | None = None,
        config: ReplayHarnessConfig | None = None,
        layers: nn.ModuleList | Sequence[nn.Module] | None = None,
        replay_modules: nn.ModuleList | Sequence[nn.Module] | None = None,
        callback: Callable[[ReplayResult], None] | None = None,
        oob_fault_callback: SCOUTFaultCallback | None = None,
        gradient_communication: GradientCommunicationReplay | None = None,
        invocation_preparer: ReplayInvocationPreparer | None = None,
        evidence_preparer: ReplayEvidencePreparer | None = None,
        peer_group: ReplayPeerGroup | None = None,
    ) -> None:
        self._config = config or ReplayHarnessConfig()
        self._workload = self._config.workload or ReplayWorkload.dense()
        self._shape_scheduler = ReplayShapeScheduler(self._workload.shape_plan)
        self._device = device or torch.device("cuda")
        if peer_group is not None and (group is not None or nccl_group is not None):
            raise ValueError("pass peer_group or group/nccl_group, not both")
        if peer_group is None:
            if (group is None) != (nccl_group is None):
                raise ValueError("group and nccl_group must be supplied together")
            peer_group = ReplayPeerGroup(
                role=self._workload.peer_role,
                group=group,
                nccl_group=nccl_group,
            )
        elif peer_group.role is not self._workload.peer_role:
            raise ValueError(
                f"{self._workload.peer_role.value} replay requires a matching "
                f"peer group, got {peer_group.role.value}"
            )
        self._peer_group = peer_group
        self._group = peer_group.group
        self._nccl_group = peer_group.nccl_group
        self._hooks: list[Any] = []
        self._layer_hooks: list[Any] = []
        self._boundary_hooks: list[Any] = []
        self._callback = callback
        self._oob_fault_callback = oob_fault_callback
        self._gradient_communication = gradient_communication
        self._invocation_preparer = invocation_preparer
        self._evidence_preparer = evidence_preparer
        self._optimizer = optimizer
        self._optimizer_step_check_disabled = False
        self._is_dense_catalog = (
            self._workload.shape_plan.source_id == "dense-captured"
            and self._workload.materializer is None
            and replay_modules is None
        )

        if layers is not None and replay_modules is not None:
            raise ValueError("Pass only one of layers= or replay_modules=")
        if replay_modules is not None and self._workload.replay_modules:
            raise ValueError(
                "Pass replay modules through either replay_modules= or "
                "ReplayHarnessConfig.workload, not both"
            )
        if self._workload.replay_modules:
            self._layers = self._workload.replay_modules
        elif replay_modules is not None:
            self._layers = replay_modules
        elif layers is not None:
            self._layers = layers
        else:
            self._layers = find_repeated_layers(model)
            if self._layers is None:
                raise ValueError(
                    "Cannot auto-detect repeated layers. Pass `layers=model.layers` explicitly."
                )

        if (
            any(shape.dimensions is not None for shape in self._workload.shape_plan.shapes)
            and not self._config.capture_inputs_by_value
        ):
            raise ValueError(
                "concrete replay shapes require capture_inputs_by_value=True so "
                "shape materialization cannot read reused training buffers"
            )
        if self._config.layer_index >= len(self._layers):
            raise ValueError(
                f"layer_index={self._config.layer_index} but model has "
                f"{len(self._layers)} repeated layers."
            )

        if self._config.straggler_confirmation_rounds < 1:
            raise ValueError("straggler_confirmation_rounds must be positive")
        if self._config.straggler_min_slowdown_ratio <= 1.0:
            raise ValueError("straggler_min_slowdown_ratio must be greater than 1")
        if self._config.straggler_min_slowdown_ms < 0:
            raise ValueError("straggler_min_slowdown_ms must be non-negative")
        if self._config.dataloader_latency_threshold_s <= 0:
            raise ValueError("dataloader_latency_threshold_s must be positive")
        if self._config.dataloader_min_slowdown_ratio <= 1.0:
            raise ValueError("dataloader_min_slowdown_ratio must be greater than 1")
        if self._config.dataloader_confirmation_rounds < 1:
            raise ValueError("dataloader_confirmation_rounds must be positive")
        if self._config.checkpoint_io_latency_threshold_s <= 0:
            raise ValueError("checkpoint_io_latency_threshold_s must be positive")
        if self._config.checkpoint_io_min_slowdown_ratio <= 1.0:
            raise ValueError("checkpoint_io_min_slowdown_ratio must be greater than 1")
        if self._config.checkpoint_io_confirmation_rounds < 1:
            raise ValueError("checkpoint_io_confirmation_rounds must be positive")
        self._validate_dense_recipe_intervals()

        self._target_layer_index = self._config.layer_index
        self._target_layer = self._layers[self._target_layer_index]
        self._detector: LayerReplayDetector | None = None
        self._dense_recipe_modules: dict[str, nn.Module] = {}
        if self._is_dense_catalog:
            embedding = find_embedding_layer(model)
            output = find_output_layer(model)
            if embedding is not None:
                self._dense_recipe_modules["embedding"] = embedding
            if output is not None:
                self._dense_recipe_modules["output"] = output
        self._dense_recipe_invocations: dict[str, ReplayInvocation] = {}
        self._dense_recipe_capture_steps: dict[str, int] = {}

        self._invocation: ReplayInvocation | None = None
        self._activation: torch.Tensor | None = None
        self._grad_output: torch.Tensor | None = None
        self._step_count = 0
        self._last_result: ReplayResult | None = None
        self._captured_inputs_owned = False
        self._captured_step: int | None = None
        self._hang_instrumentation: HangInstrumentation | None = None
        self._all_to_all_replay_recipes: tuple[AllToAllReplayRecipe, ...] = ()
        self._all_to_all_executor = AllToAllReplayExecutor(self._device)
        self._oob_service: OOBHangService | None = None
        self._temporal = TemporalBaselineStore(
            window_size=self._config.temporal_window_size,
            min_samples=self._config.temporal_min_samples,
            slowdown_ratio=self._config.temporal_slowdown_ratio,
            threshold_sigma=self._config.temporal_threshold_sigma,
        )
        self._cross_pg = CrossPGCoordinator()

        self._register_hooks()
        self._register_dense_boundary_hooks()
        self._start_oob_hang_detection(model)

        if optimizer is not None:
            self._optimizer_hook = optimizer.register_step_post_hook(self._optimizer_step_hook)
            self._hooks.append(self._optimizer_hook)

    def _get_detector(self) -> LayerReplayDetector:
        if self._detector is None:
            self._detector = LayerReplayDetector(
                group=self._group,
                nccl_group=self._nccl_group,
                broadcast_src=self._config.broadcast_src,
                device=self._device,
                deterministic=self._config.deterministic,
                synchronize_rng=self._config.synchronize_rng,
                compare_parameter_state=self._config.compare_parameter_state,
                gradient_communication=self._gradient_communication,
                invocation_preparer=self._invocation_preparer,
                evidence_preparer=self._evidence_preparer,
                straggler_min_slowdown_ratio=self._config.straggler_min_slowdown_ratio,
                straggler_min_slowdown_ms=self._config.straggler_min_slowdown_ms,
            )
        return self._detector

    def _validate_dense_recipe_intervals(self) -> None:
        for name in (
            "embedding_check_interval",
            "hidden_check_interval",
            "output_check_interval",
            "optimizer_check_interval",
        ):
            interval = getattr(self._config, name)
            if interval is None:
                continue
            if interval < 0:
                raise ValueError(f"{name} must be non-negative")

    def _dense_recipe_interval(self, recipe_id: str) -> int:
        configured = getattr(self._config, f"{recipe_id}_check_interval")
        if configured is not None:
            return configured
        return self._config.check_interval

    def _dense_recipe_due(self, recipe_id: str, *, scheduled: bool) -> bool:
        configured = getattr(self._config, f"{recipe_id}_check_interval")
        if configured == 0:
            return False
        if not scheduled:
            return True
        return self._dense_recipe_scheduled_at(recipe_id, self._step_count)

    def _dense_recipe_scheduled_at(self, recipe_id: str, step: int) -> bool:
        if self._config.check_interval <= 0:
            return False
        interval = self._dense_recipe_interval(recipe_id)
        return interval > 0 and step % interval == 0

    def _optimizer_step_hook(self, optimizer, args, kwargs) -> None:
        """Post-step hook: auto-triggers detection at the configured interval."""
        result = self.step(optimizer=optimizer)
        if result is not None:
            self._last_result = result
            if replay_result_has_fault(result):
                if self._callback:
                    self._callback(result)
                else:
                    logger.warning(
                        f"Fault detected at step {self._step_count}: "
                        f"sdc={result.sdc_bitmap}, straggler={result.straggler_bitmap}"
                    )

    def _register_hooks(self) -> None:
        fwd_hook = self._target_layer.register_forward_hook(self._fwd_hook_fn, with_kwargs=True)
        self._layer_hooks.append(fwd_hook)

    def _register_dense_boundary_hooks(self) -> None:
        for recipe_id, module in self._dense_recipe_modules.items():
            if not self._dense_recipe_due(recipe_id, scheduled=False):
                continue
            forward = module.register_forward_hook(
                self._dense_forward_hook(recipe_id),
                with_kwargs=True,
            )
            self._boundary_hooks.append(forward)

    def _dense_forward_hook(self, recipe_id: str):
        def hook(
            module: nn.Module,
            args: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> None:
            del module
            leaves, spec = tree_flatten((args, kwargs))
            own_storage = self._capture_inputs_by_value_now(recipe_id)
            captured = [
                _capture_tensor(leaf, clone=own_storage) if isinstance(leaf, torch.Tensor) else leaf
                for leaf in leaves
            ]
            captured_args, captured_kwargs = tree_unflatten(captured, spec)
            autocast_enabled, autocast_dtype = _autocast_state(self._device.type)
            invocation = ReplayInvocation(
                args=captured_args,
                kwargs=captured_kwargs,
                input_requires_grad=[
                    bool(leaf.requires_grad) if isinstance(leaf, torch.Tensor) else False
                    for leaf in leaves
                ],
                autocast_enabled=autocast_enabled,
                autocast_device_type=self._device.type,
                autocast_dtype=autocast_dtype,
            )
            self._dense_recipe_invocations[recipe_id] = invocation
            self._dense_recipe_capture_steps[recipe_id] = self._step_count + 1
            _register_output_gradient_capture(
                output,
                clone=own_storage,
                callback=lambda grad_output: self._store_dense_grad_output(
                    recipe_id,
                    invocation,
                    grad_output,
                ),
            )

        return hook

    def _store_dense_grad_output(
        self,
        recipe_id: str,
        invocation: ReplayInvocation,
        grad_output: Any,
    ) -> None:
        if self._dense_recipe_invocations.get(recipe_id) is invocation:
            invocation.grad_output = grad_output

    def _fwd_hook_fn(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        del module
        leaves, spec = tree_flatten((args, kwargs))
        input_requires_grad = [
            bool(leaf.requires_grad) if isinstance(leaf, torch.Tensor) else False for leaf in leaves
        ]
        own_storage = self._capture_inputs_by_value_now("hidden")
        captured_leaves = [
            _capture_tensor(leaf, clone=own_storage) if isinstance(leaf, torch.Tensor) else leaf
            for leaf in leaves
        ]
        captured_args, captured_kwargs = tree_unflatten(captured_leaves, spec)
        autocast_enabled, autocast_dtype = _autocast_state(self._device.type)
        invocation = ReplayInvocation(
            args=captured_args,
            kwargs=captured_kwargs,
            input_requires_grad=input_requires_grad,
            autocast_enabled=autocast_enabled,
            autocast_device_type=self._device.type,
            autocast_dtype=autocast_dtype,
        )
        self._invocation = invocation
        self._activation = _first_tensor(captured_args, captured_kwargs)
        self._grad_output = None
        self._captured_inputs_owned = own_storage
        self._captured_step = self._step_count + 1
        _register_output_gradient_capture(
            output,
            clone=own_storage,
            callback=lambda grad_output: self._store_hidden_grad_output(
                invocation,
                grad_output,
            ),
        )

    def _store_hidden_grad_output(
        self,
        invocation: ReplayInvocation,
        grad_output: Any,
    ) -> None:
        if self._invocation is invocation:
            invocation.grad_output = grad_output
            self._grad_output = _first_tensor(grad_output)

    def _capture_inputs_by_value_now(self, recipe_id: str | None = None) -> bool:
        """Whether this invocation needs storage that survives until replay.

        Automatic checks clone only the step on which detection will run. Manual
        mode has no future cadence signal, so it retains the latest invocation by
        value. This bounds the additional memory traffic in the production path.
        """
        if not self._config.capture_inputs_by_value:
            return False
        interval = (
            self._dense_recipe_interval(recipe_id)
            if self._is_dense_catalog and recipe_id is not None
            else self._config.check_interval
        )
        return interval == 0 or (self._step_count + 1) % interval == 0

    def step(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        allow_local_dtensor_shards: bool = False,
        optimizer_step_tensors: OptimizerStepEvidence | None = None,
        complete_shape_cycle: bool = False,
    ) -> ReplayResult | None:
        """Call after each training step. Runs detection if interval is reached.

        ``complete_shape_cycle`` replays every configured shape against the same
        captured state. GEMINI uses this mode before checkpoint capture.

        Returns ReplayResult if detection ran this step, None otherwise.
        """
        self._step_count += 1
        self._snapshot_all_to_all_recipes()
        result = None
        if self._is_dense_catalog:
            if any(
                self._dense_recipe_due(recipe_id, scheduled=True)
                for recipe_id in ("embedding", "hidden", "output", "optimizer")
            ):
                result = self._check_dense_recipes(
                    optimizer=optimizer or self._optimizer,
                    allow_local_dtensor_shards=allow_local_dtensor_shards,
                    optimizer_step_tensors=optimizer_step_tensors,
                    scheduled=not complete_shape_cycle,
                    rotate_hidden=True,
                )
        elif (
            self._config.check_interval > 0 and self._step_count % self._config.check_interval == 0
        ):
            if self._captured_step == self._step_count and self.has_capture:
                check = self.check_shape_cycle if complete_shape_cycle else self.check
                result = check(
                    optimizer=optimizer or self._optimizer,
                    allow_local_dtensor_shards=allow_local_dtensor_shards,
                    optimizer_step_tensors=optimizer_step_tensors,
                )
            else:
                logger.debug("SCOUT scheduled replay skipped: selected layer has no capture")
        if result is not None:
            self._attach_all_to_all_replay(result)
        if self._hang_instrumentation is not None:
            self._hang_instrumentation.step_boundary()
        return result

    def mark_step_boundary(self) -> None:
        """Publish an externally counted step without running scheduled replay."""
        if self._hang_instrumentation is not None:
            self._hang_instrumentation.step_boundary()

    def check(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        allow_local_dtensor_shards: bool = False,
        optimizer_step_tensors: OptimizerStepEvidence | None = None,
        _scheduled: bool = False,
        _rotate_hidden: bool = True,
    ) -> ReplayResult:
        """Run layer replay detection now, using the most recently captured tensors.

        Raises RuntimeError if no activation has been captured yet.
        """
        self._snapshot_all_to_all_recipes()
        if self._is_dense_catalog:
            if (
                self._invocation is None
                and not self._dense_recipe_invocations
                and optimizer_step_tensors is None
                and optimizer is None
                and self._optimizer is None
            ):
                raise RuntimeError(
                    "No activation captured yet. Ensure at least one forward pass "
                    "has run before calling check()."
                )
            result = self._check_dense_recipes(
                optimizer=optimizer,
                allow_local_dtensor_shards=allow_local_dtensor_shards,
                optimizer_step_tensors=optimizer_step_tensors,
                scheduled=_scheduled,
                rotate_hidden=_rotate_hidden,
            )
            if result is None:
                raise RuntimeError("No activation captured for an enabled dense replay recipe")
            self._attach_all_to_all_replay(result)
            return result

        if self._activation is None or self._invocation is None:
            raise RuntimeError(
                "No activation captured yet. Ensure at least one forward pass "
                "has run before calling check()."
            )

        result = self._check_current_shape(
            optimizer=optimizer,
            allow_local_dtensor_shards=allow_local_dtensor_shards,
            optimizer_step_tensors=optimizer_step_tensors,
        )
        result.shape_cycle_size = len(self.replay_shapes)
        result.completed_shape_cycle = len(self.replay_shapes) == 1
        self._attach_all_to_all_replay(result)
        self._last_result = result
        if self._config.rotate_layers:
            self._rotate_target_layer()
        return result

    def _check_dense_recipes(
        self,
        *,
        optimizer: torch.optim.Optimizer | None,
        allow_local_dtensor_shards: bool,
        optimizer_step_tensors: OptimizerStepEvidence | None,
        scheduled: bool,
        rotate_hidden: bool,
    ) -> ReplayResult | None:
        has_module_evidence = False
        for recipe_id in ("embedding", "hidden", "output"):
            if not self._dense_recipe_due(recipe_id, scheduled=scheduled):
                continue
            if recipe_id == "hidden":
                has_module_evidence = self._invocation is not None and (
                    not scheduled or self._captured_step == self._step_count
                )
            else:
                has_module_evidence = (
                    recipe_id in self._dense_recipe_modules
                    and recipe_id in self._dense_recipe_invocations
                    and (
                        not scheduled
                        or self._dense_recipe_capture_steps.get(recipe_id) == self._step_count
                    )
                )
            if has_module_evidence:
                break
        optimizer_due = self._dense_recipe_due("optimizer", scheduled=scheduled)
        has_optimizer_evidence = optimizer_due and (
            optimizer_step_tensors is not None
            or (
                not self._optimizer_step_check_disabled
                and (optimizer is not None or self._optimizer is not None)
            )
        )
        if not has_module_evidence and not has_optimizer_evidence:
            if scheduled:
                logger.debug("SCOUT scheduled dense replay skipped: no current evidence")
                return None
            raise RuntimeError("No activation captured for an enabled dense replay recipe")

        detector = self._get_detector()
        results: list[ReplayResult] = []

        for recipe_id in ("embedding", "hidden", "output"):
            if not self._dense_recipe_due(recipe_id, scheduled=scheduled):
                continue
            if recipe_id == "hidden":
                if self._invocation is None or (
                    scheduled and self._captured_step != self._step_count
                ):
                    logger.debug("SCOUT hidden recipe skipped: no current capture")
                    continue
                result = self._check_current_shape(
                    optimizer=None,
                    allow_local_dtensor_shards=allow_local_dtensor_shards,
                    optimizer_step_tensors=None,
                    include_optimizer=False,
                )
            else:
                module = self._dense_recipe_modules.get(recipe_id)
                invocation = self._dense_recipe_invocations.get(recipe_id)
                if (
                    module is None
                    or invocation is None
                    or (
                        scheduled
                        and self._dense_recipe_capture_steps.get(recipe_id) != self._step_count
                    )
                ):
                    logger.debug("SCOUT %s recipe skipped: no current capture", recipe_id)
                    continue
                layer_id = -1 if recipe_id == "embedding" else len(self._layers)
                result = self._replay_dense_module(
                    detector,
                    module,
                    invocation,
                    layer_id=layer_id,
                )
            _annotate_recipe_result(result, recipe_id)
            results.append(result)

        optimizer_results: dict[str, C3Result] = {}
        if self._dense_recipe_due("optimizer", scheduled=scheduled):
            optimizer_results = self._compare_optimizer_step(
                detector,
                optimizer or self._optimizer,
                allow_local_dtensor_shards=allow_local_dtensor_shards,
                optimizer_step_tensors=optimizer_step_tensors,
            )

        if not results and not optimizer_results:
            if scheduled:
                logger.debug("SCOUT scheduled dense replay skipped: no current evidence")
                return None
            raise RuntimeError("No activation captured for an enabled dense replay recipe")

        if results:
            aggregate = _aggregate_dense_recipe_results(results)
        else:
            width = len(next(iter(optimizer_results.values())).bitmap)
            aggregate = ReplayResult(
                sdc_bitmap=[0] * width,
                straggler_bitmap=[0] * width,
                replay_time_ms=0.0,
                layer_id=self._target_layer_index,
                peer_ranks=detector.peer_ranks,
                replay_times_ms=[0.0] * width,
                replay_mode="optimizer",
                spatial_straggler_bitmap=[0] * width,
                replay_shape_id="captured",
                checked_shape_ids=["captured"],
                checked_shapes=[None],
                shape_cycle_size=1,
                completed_shape_cycle=True,
                completed_scheduled_cycle=True,
                dense_replay=True,
            )

        if optimizer_results:
            prefixed = {
                f"optimizer.{name}": c3_result for name, c3_result in optimizer_results.items()
            }
            aggregate.c3_results.update(prefixed)
            _merge_sdc_source_bitmaps(
                aggregate,
                {name: c3_result.bitmap for name, c3_result in prefixed.items()},
            )
            aggregate.checked_recipe_ids.append("optimizer")

        aggregate.checked_shape_ids = ["captured"]
        aggregate.checked_shapes = [None]
        aggregate.shape_cycle_size = 1
        aggregate.completed_shape_cycle = True
        aggregate.completed_scheduled_cycle = True
        aggregate.dense_replay = True
        self._last_result = aggregate
        if (
            rotate_hidden
            and self._config.rotate_layers
            and "hidden" in aggregate.checked_recipe_ids
        ):
            self._rotate_target_layer()
        return aggregate

    def _replay_dense_module(
        self,
        detector: LayerReplayDetector,
        module: nn.Module,
        invocation: ReplayInvocation,
        *,
        layer_id: int,
    ) -> ReplayResult:
        scale_factors = self._config.scale_factors if invocation.grad_output is None else []
        result = detector.replay_invocation(
            layer=module,
            invocation=invocation,
            layer_id=layer_id,
            scale_factors=scale_factors,
        )
        return self._confirm_and_classify(detector, module, invocation, result)

    def check_shape_cycle(
        self,
        optimizer: torch.optim.Optimizer | None = None,
        *,
        allow_local_dtensor_shards: bool = False,
        optimizer_step_tensors: OptimizerStepEvidence | None = None,
        preserve_scheduler: bool = False,
    ) -> ReplayResult:
        """Replay every configured shape against one captured state.

        A numerical fault stops the sweep early. Failure-time validation sets
        ``preserve_scheduler`` so the emergency sweep does not consume the
        normal rotating recipe schedule.
        """
        self._snapshot_all_to_all_recipes()
        if self._is_dense_catalog:
            result = self._check_dense_recipes(
                optimizer=optimizer,
                allow_local_dtensor_shards=allow_local_dtensor_shards,
                optimizer_step_tensors=optimizer_step_tensors,
                scheduled=False,
                rotate_hidden=not preserve_scheduler,
            )
            if result is None:
                raise RuntimeError("No activation captured for an enabled dense replay recipe")
            self._attach_all_to_all_replay(result)
            return result

        if self._activation is None or self._invocation is None:
            raise RuntimeError(
                "No activation captured yet. Ensure at least one forward pass "
                "has run before calling check_shape_cycle()."
            )

        scheduler_state = self._shape_scheduler.state_dict()
        temporal_state = self._temporal.state_dict()
        expected = len(self.replay_shapes)
        results: list[ReplayResult] = []
        try:
            for index in range(expected):
                result = self._check_current_shape(
                    optimizer=optimizer if index == 0 else None,
                    allow_local_dtensor_shards=allow_local_dtensor_shards,
                    optimizer_step_tensors=optimizer_step_tensors if index == 0 else None,
                )
                results.append(result)
                if replay_result_has_sdc(result):
                    break
        finally:
            if preserve_scheduler:
                self._shape_scheduler.load_state_dict(scheduler_state)
                self._temporal.load_state_dict(temporal_state)

        aggregate = _aggregate_shape_cycle(results, expected=expected)
        self._attach_all_to_all_replay(aggregate)
        self._last_result = aggregate
        if (
            aggregate.completed_shape_cycle
            and self._config.rotate_layers
            and not preserve_scheduler
        ):
            self._rotate_target_layer()
        return aggregate

    def _check_current_shape(
        self,
        optimizer: torch.optim.Optimizer | None,
        *,
        allow_local_dtensor_shards: bool,
        optimizer_step_tensors: OptimizerStepEvidence | None,
        include_optimizer: bool = True,
    ) -> ReplayResult:
        """Replay the scheduler's current shape and advance after success."""
        cycle_before = self._shape_scheduler.completed_cycles

        detector = self._get_detector()
        replay_shape = self._shape_scheduler.current_shape
        agreed, outliers = detector.replay_shape_consensus(
            self._workload.shape_plan.signature(replay_shape)
        )
        if not agreed:
            raise RuntimeError(
                "replay shape schedule differs across peers; "
                f"local_shape={replay_shape.shape_id!r}, outliers={outliers}"
            )
        invocation = self._workload.materialize(self._invocation, replay_shape)
        layer = self._target_layer
        layer_id = self._target_layer_index
        scale_factors = self._config.scale_factors if invocation.grad_output is None else []
        result = detector.replay_invocation(
            layer=layer,
            invocation=invocation,
            layer_id=layer_id,
            scale_factors=scale_factors,
        )
        result = self._confirm_and_classify(detector, layer, invocation, result)
        optimizer_bitmaps = (
            self._compare_optimizer_step(
                detector,
                optimizer or self._optimizer,
                allow_local_dtensor_shards=allow_local_dtensor_shards,
                optimizer_step_tensors=optimizer_step_tensors,
            )
            if include_optimizer
            else {}
        )
        result.c3_results.update(optimizer_bitmaps)
        _merge_sdc_source_bitmaps(
            result,
            {name: c3_result.bitmap for name, c3_result in optimizer_bitmaps.items()},
        )
        result.replay_shape_id = replay_shape.shape_id
        result.replay_shape = replay_shape.dimensions
        result.checked_shape_ids = [replay_shape.shape_id]
        result.checked_shapes = [replay_shape.dimensions]
        result.shape_cycle_size = len(self.replay_shapes)
        result.completed_shape_cycle = len(self.replay_shapes) == 1
        self._shape_scheduler.advance()
        result.completed_scheduled_cycle = self._shape_scheduler.completed_cycles > cycle_before
        result.scheduled_cycle = self._shape_scheduler.completed_cycles
        return result

    def _snapshot_all_to_all_recipes(self) -> None:
        if self._hang_instrumentation is not None:
            self._all_to_all_replay_recipes = self._hang_instrumentation.all_to_all_recipes

    def _attach_all_to_all_replay(self, result: ReplayResult) -> None:
        policy = self._config.all_to_all_policy
        recipes = self._all_to_all_replay_recipes
        if policy is None or not recipes or "all_to_all" in result.checked_recipe_ids:
            return
        result.checked_recipe_ids.append("all_to_all")

        detector = self._get_detector()
        count_result = detector.compare_structure(len(recipes))
        result.c3_results["all_to_all.recipe_count"] = count_result
        _merge_sdc_source_bitmaps(
            result,
            {"all_to_all.recipe_count": count_result.bitmap},
        )
        if count_result.status is not C3Status.AGREE:
            return

        instrumentation = self._hang_instrumentation
        for recipe_index, recipe in enumerate(recipes):
            outcomes = []
            replay_ok = True
            try:
                if instrumentation is None:
                    outcomes = self._all_to_all_executor.replay(recipe, policy)
                else:
                    with instrumentation.suspend_all_to_all_capture():
                        outcomes = self._all_to_all_executor.replay(recipe, policy)
            except (RuntimeError, ValueError) as exc:
                replay_ok = False
                logger.warning(
                    "SCOUT AllToAll replay recipe %s is unavailable: %s",
                    recipe_index,
                    exc,
                )

            status_name = f"all_to_all.{recipe_index}.execution"
            status_result = detector.compare_expected_boolean(replay_ok)
            result.c3_results[status_name] = status_result
            _merge_sdc_source_bitmaps(
                result,
                {status_name: status_result.bitmap},
            )
            if status_result.status is not C3Status.AGREE:
                continue

            matrix_count_name = f"all_to_all.{recipe_index}.matrix_count"
            matrix_count_result = detector.compare_structure(len(outcomes))
            result.c3_results[matrix_count_name] = matrix_count_result
            _merge_sdc_source_bitmaps(
                result,
                {matrix_count_name: matrix_count_result.bitmap},
            )
            if matrix_count_result.status is not C3Status.AGREE:
                continue

            for matrix_index, outcome in enumerate(outcomes):
                prefix = f"all_to_all.{recipe_index}.{matrix_index}"
                contract_name = f"{prefix}.contract"
                contract_result = detector.compare_structure(outcome.comparison_signature)
                result.c3_results[contract_name] = contract_result
                _merge_sdc_source_bitmaps(
                    result,
                    {contract_name: contract_result.bitmap},
                )
                if contract_result.status is not C3Status.AGREE:
                    continue

                output_name = f"{prefix}.output"
                output_result = detector.compare_expected_boolean(outcome.correct)
                result.c3_results[output_name] = output_result
                _merge_sdc_source_bitmaps(
                    result,
                    {output_name: output_result.bitmap},
                )
                detector.add_communication_timing(
                    result,
                    name=f"all_to_all_replay.{outcome.matrix.name}",
                    elapsed_ms=outcome.latency_ms,
                    group_ranks=outcome.group_ranks,
                    topology_role="ep",
                    message_bytes=max(
                        outcome.input_bytes,
                        outcome.output_bytes,
                    ),
                    sequence=outcome.sequence,
                )

    def _compare_optimizer_step(
        self,
        detector: LayerReplayDetector,
        optimizer: torch.optim.Optimizer | None,
        *,
        allow_local_dtensor_shards: bool,
        optimizer_step_tensors: OptimizerStepEvidence | None,
    ) -> dict[str, C3Result]:
        if isinstance(optimizer_step_tensors, OptimizerReplayBatch):
            return detector.replay_optimizer_batch(optimizer_step_tensors)
        if optimizer_step_tensors is not None:
            return detector.compare_tensor_groups(optimizer_step_tensors)
        if self._optimizer_step_check_disabled or optimizer is None:
            return {}
        try:
            tensor_groups = collect_updated_weights(
                optimizer,
                list(self._target_layer.parameters()),
                allow_local_dtensor_shards=allow_local_dtensor_shards,
            )
        except OptimizerStepCheckUnsupported as exc:
            self._optimizer_step_check_disabled = True
            logger.warning(
                "SCOUT optimizer-step verification disabled: %s. Layer replay remains enabled.",
                exc,
            )
            return {}
        return detector.compare_tensor_groups(tensor_groups)

    def _confirm_and_classify(
        self,
        detector: LayerReplayDetector,
        layer: nn.Module,
        invocation: ReplayInvocation,
        initial: ReplayResult,
    ) -> ReplayResult:
        """Confirm timing candidates, apply temporal detection, and decompose faults."""
        key = replay_baseline_key(
            layer_id=initial.layer_id,
            replay_mode=initial.replay_mode,
            invocation=invocation,
            peer_ranks=initial.peer_ranks,
            device=self._device,
        )
        rounds = [initial]
        assessments = [self._temporal_assessment(key, initial)]
        had_candidate = _has_timing_candidate(initial, assessments[0])

        if had_candidate:
            for _ in range(1, self._config.straggler_confirmation_rounds):
                replay = detector.replay_invocation(
                    layer=layer,
                    invocation=invocation,
                    layer_id=initial.layer_id,
                    scale_factors=(
                        self._config.scale_factors if invocation.grad_output is None else []
                    ),
                )
                rounds.append(replay)
                assessment = self._temporal_assessment(key, replay)
                assessments.append(assessment)
                had_candidate = had_candidate or _has_timing_candidate(replay, assessment)

        result = _merge_replay_rounds(
            rounds,
            assessments,
            required=self._config.straggler_confirmation_rounds,
        )
        if (
            not had_candidate
            and not any(result.sdc_bitmap)
            and result.replay_times_ms
            and self._config.enable_temporal
        ):
            self._temporal.observe_clean(key, result.replay_times_ms)

        if any(result.straggler_bitmap) or result.temporal_group_slowdown:
            result.straggler_detail = detector.localize_invocation_straggler(
                layer,
                invocation,
                layer_id=result.layer_id,
            )
            result.collective_timings.extend(result.straggler_detail.collective_timings)
            if result.temporal_group_slowdown and result.straggler_detail.straggler_type == "none":
                compute = statistics.median(result.straggler_detail.compute_times_ms)
                communication = statistics.median(result.straggler_detail.comm_times_ms)
                result.straggler_detail.straggler_type = (
                    "shared_communication" if communication > compute else "shared_compute"
                )
        return result

    def _temporal_assessment(self, key: str, result: ReplayResult) -> TemporalAssessment:
        if not self._config.enable_temporal:
            return TemporalAssessment([0] * len(result.straggler_bitmap), False)
        return self._temporal.assess(key, result.replay_times_ms)

    def _rotate_target_layer(self) -> None:
        for hook in self._layer_hooks:
            hook.remove()
        self._layer_hooks.clear()
        self._target_layer_index = (self._target_layer_index + 1) % len(self._layers)
        self._target_layer = self._layers[self._target_layer_index]
        self._invocation = None
        self._activation = None
        self._grad_output = None
        self._captured_inputs_owned = False
        self._captured_step = None
        self._register_hooks()

    def temporal_state_dict(self) -> dict[str, Any]:
        """Compact temporal and replay-rotation state for checkpoint persistence."""
        state = self._temporal.state_dict()
        state["replay_shape_scheduler"] = self._shape_scheduler.state_dict()
        return state

    def load_temporal_state_dict(self, state: dict[str, Any] | None) -> None:
        """Restore temporal baselines and replay-shape position after a restart."""
        self._temporal.load_state_dict(state)
        if state:
            self._shape_scheduler.load_state_dict(state.get("replay_shape_scheduler"))

    def check_local_parameter_shards(self) -> list[int]:
        """Compare the sampled layer's local shards before an HSDP unshard."""
        return self._get_detector().compare_local_parameter_shards(self._target_layer)

    def add_communication_timing(
        self,
        result: ReplayResult,
        *,
        name: str,
        elapsed_ms: float,
        group_ranks: Sequence[int] | None = None,
        topology_role: str | None = None,
        message_bytes: int = 0,
        sequence: int = 0,
    ) -> None:
        """Merge a framework-visible communication boundary into replay evidence."""
        self._get_detector().add_communication_timing(
            result,
            name=name,
            elapsed_ms=elapsed_ms,
            group_ranks=group_ranks,
            topology_role=topology_role,
            message_bytes=message_bytes,
            sequence=sequence,
        )

    def finalize_communication_localization(
        self,
        result: ReplayResult | None,
    ) -> CrossPGResult:
        """Gather confirmed PG timings and attach a machine-level diagnosis."""
        cross_pg = self._cross_pg.localize(
            list(result.collective_timings) if result is not None else []
        )
        if result is not None:
            result.cross_pg_result = cross_pg
        return cross_pg

    def _start_oob_hang_detection(self, model: nn.Module) -> None:
        if not dist.is_initialized():
            return
        peer_ranks = (
            dist.get_process_group_ranks(self._group)
            if self._group is not None
            else list(range(dist.get_world_size()))
        )
        if len(peer_ranks) < 2:
            return
        global_rank = dist.get_rank()
        self._oob_service = OOBHangService(
            global_rank=global_rank,
            peer_ranks=peer_ranks,
            config=OOBHangConfig(
                stall_threshold_s=self._config.hang_stall_threshold_s,
                confirmation_interval_s=self._config.hang_confirmation_interval_s,
                dataloader_latency_threshold_s=self._config.dataloader_latency_threshold_s,
                dataloader_min_slowdown_ratio=self._config.dataloader_min_slowdown_ratio,
                dataloader_confirmation_rounds=self._config.dataloader_confirmation_rounds,
                checkpoint_io_latency_threshold_s=(self._config.checkpoint_io_latency_threshold_s),
                checkpoint_io_min_slowdown_ratio=(self._config.checkpoint_io_min_slowdown_ratio),
                checkpoint_io_confirmation_rounds=(self._config.checkpoint_io_confirmation_rounds),
                state_dir=self._config.hang_state_dir,
                master_addr=self._config.hang_master_addr,
                master_port=self._config.hang_master_port,
            ),
            report_callback=self._oob_fault_callback,
        )
        self._hang_instrumentation = HangInstrumentation(
            model,
            self._layers,
            global_rank,
            progress_event=self._oob_service.progress_event,
        )
        self._oob_service.start()

    @property
    def last_result(self) -> ReplayResult | None:
        """Most recent detection result (from auto or manual check)."""
        return self._last_result

    @property
    def all_to_all_replay_recipes(self) -> tuple[AllToAllReplayRecipe, ...]:
        """AllToAll recipes captured from the training window used by the last check."""
        return self._all_to_all_replay_recipes

    def instrument_dataloader(
        self,
        dataloader: Any,
        *,
        name: str = "train",
    ) -> InstrumentedDataLoader[Any]:
        """Sample ``next(dataloader)`` latency at the replay detection interval."""
        monitor = (
            self._hang_instrumentation.stage_monitor
            if self._hang_instrumentation is not None
            else None
        )
        return instrument_dataloader(
            dataloader,
            monitor,
            name=name,
            detection_interval=self._config.check_interval,
        )

    def checkpoint_io(
        self,
        operation: CheckpointOperation,
        *,
        name: str = "framework",
    ):
        """Publish a framework checkpoint read or write to the OOB daemon."""
        monitor = (
            self._hang_instrumentation.stage_monitor
            if self._hang_instrumentation is not None
            else None
        )
        return checkpoint_io(monitor, operation, name=name)

    @property
    def target_layer(self) -> nn.Module:
        """The sampled layer whose inputs are captured and replayed."""
        return self._target_layer

    @property
    def has_capture(self) -> bool:
        """Whether at least one forward pass has been captured."""
        return self._activation is not None

    @property
    def has_replay_capture(self) -> bool:
        """Whether any enabled replay surface has a captured invocation."""
        return self.has_capture or (self._is_dense_catalog and bool(self._dense_recipe_invocations))

    @property
    def has_grad(self) -> bool:
        """Whether backward pass grad_output has been captured."""
        return self._grad_output is not None

    @property
    def step_count(self) -> int:
        return self._step_count

    @property
    def replay_shapes(self) -> tuple[ReplayShape, ...]:
        """Ordered shape list shared by dense and dynamic-shape replay."""
        return self._workload.shape_plan.shapes

    @property
    def replay_shape_plan_id(self) -> str:
        """Stable identifier bound into durable checkpoint certification."""
        return self._workload.shape_plan.identifier

    @property
    def current_replay_shape(self) -> ReplayShape:
        """Shape that the next successful replay check will execute."""
        return self._shape_scheduler.current_shape

    @property
    def replay_shape_cycle_steps(self) -> int | None:
        """Maximum training steps for one shape cycle at the configured cadence."""
        if self._config.check_interval <= 0:
            return None
        return len(self.replay_shapes) * self._config.check_interval

    def optimizer_replay_due(self, step: int) -> bool:
        """Whether framework-owned optimizer state should be captured at ``step``."""
        if self._config.check_interval <= 0:
            return False
        if not self._is_dense_catalog:
            interval = self._config.check_interval
        else:
            interval = self._dense_recipe_interval("optimizer")
        return interval > 0 and step % interval == 0

    def replay_due(self, step: int) -> bool:
        """Whether any configured replay recipe is scheduled at ``step``."""
        if self._config.check_interval <= 0:
            return False
        if not self._is_dense_catalog:
            return step % self._config.check_interval == 0
        return any(
            self._dense_recipe_scheduled_at(recipe_id, step)
            for recipe_id in ("embedding", "hidden", "output", "optimizer")
        )

    @property
    def dense_boundary_modules(self) -> tuple[nn.Module, ...]:
        """Dense modules that may require framework-specific materialization."""
        if not self._is_dense_catalog:
            return ()
        return tuple(
            module
            for recipe_id, module in self._dense_recipe_modules.items()
            if self._dense_recipe_due(recipe_id, scheduled=False)
        )

    def remove_hooks(self) -> None:
        """Remove all registered hooks. Call when done with detection."""
        oob_service = getattr(self, "_oob_service", None)
        if oob_service is not None:
            oob_service.close()
            self._oob_service = None
        instrumentation = getattr(self, "_hang_instrumentation", None)
        if instrumentation is not None:
            instrumentation.close()
            self._hang_instrumentation = None
        for hook in self._layer_hooks:
            hook.remove()
        self._layer_hooks.clear()
        for hook in self._boundary_hooks:
            hook.remove()
        self._boundary_hooks.clear()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self) -> None:
        if hasattr(self, "_hooks"):
            self.remove_hooks()


def _annotate_recipe_result(result: ReplayResult, recipe_id: str) -> None:
    result.checked_recipe_ids = [recipe_id]
    result.sdc_source_bitmaps = {
        f"{recipe_id}.{name}": bitmap for name, bitmap in result.sdc_source_bitmaps.items()
    }
    result.sdc_sources = [name for name, bitmap in result.sdc_source_bitmaps.items() if any(bitmap)]
    result.c3_results = {
        f"{recipe_id}.{name}": c3_result for name, c3_result in result.c3_results.items()
    }
    if result.timing_c3_result is not None:
        result.c3_results[f"{recipe_id}.timing"] = result.timing_c3_result
    result.replay_mode = f"{recipe_id}:{result.replay_mode}"


def _aggregate_dense_recipe_results(results: Sequence[ReplayResult]) -> ReplayResult:
    checked_recipe_ids = [
        recipe_id for result in results for recipe_id in result.checked_recipe_ids
    ]
    c3_results = {
        name: c3_result for result in results for name, c3_result in result.c3_results.items()
    }
    aggregate = _aggregate_shape_cycle(results, expected=len(results))
    aggregate.checked_recipe_ids = checked_recipe_ids
    aggregate.c3_results = c3_results
    aggregate.replay_shape_id = "captured"
    aggregate.replay_shape = None
    aggregate.checked_shape_ids = ["captured"]
    aggregate.checked_shapes = [None]
    aggregate.shape_cycle_size = 1
    aggregate.completed_shape_cycle = True
    aggregate.completed_scheduled_cycle = True
    aggregate.dense_replay = True
    aggregate.replay_mode = results[0].replay_mode if len(results) == 1 else "dense_recipe_set"
    return aggregate


def _aggregate_shape_cycle(
    results: Sequence[ReplayResult],
    *,
    expected: int,
) -> ReplayResult:
    """Merge per-shape evidence into one checkpoint-certification result."""
    if not results:
        raise ValueError("cannot aggregate an empty replay shape cycle")
    aggregate = results[0]
    width = len(aggregate.sdc_bitmap)
    if any(
        len(result.sdc_bitmap) != width or len(result.straggler_bitmap) != width
        for result in results
    ):
        raise RuntimeError("replay shape results have incompatible peer-group widths")

    source_names = {source for result in results for source in result.sdc_source_bitmaps}
    aggregate.sdc_source_bitmaps = {
        source: [
            int(
                any(result.sdc_source_bitmaps.get(source, [0] * width)[index] for result in results)
            )
            for index in range(width)
        ]
        for source in source_names
    }
    aggregate.sdc_bitmap = [
        int(any(result.sdc_bitmap[index] for result in results)) for index in range(width)
    ]
    aggregate.sdc_sources = [
        source for source, bitmap in aggregate.sdc_source_bitmaps.items() if any(bitmap)
    ]
    aggregate.c3_results = {
        f"{result.replay_shape_id}.{name}": c3_result
        for result in results
        for name, c3_result in result.c3_results.items()
    }
    aggregate.checked_recipe_ids = [
        recipe_id for result in results for recipe_id in result.checked_recipe_ids
    ]
    aggregate.straggler_bitmap = [
        int(any(result.straggler_bitmap[index] for result in results)) for index in range(width)
    ]
    aggregate.spatial_straggler_bitmap = [
        int(
            any(
                (result.spatial_straggler_bitmap or result.straggler_bitmap)[index]
                for result in results
            )
        )
        for index in range(width)
    ]
    aggregate.temporal_straggler_bitmap = [
        int(any((result.temporal_straggler_bitmap or [0] * width)[index] for result in results))
        for index in range(width)
    ]
    aggregate.temporal_group_slowdown = any(result.temporal_group_slowdown for result in results)
    aggregate.straggler_confirmations = max(result.straggler_confirmations for result in results)
    aggregate.straggler_detail = next(
        (result.straggler_detail for result in results if result.straggler_detail is not None),
        None,
    )
    aggregate.collective_timings = [
        timing for result in results for timing in result.collective_timings
    ]
    aggregate.replay_time_ms = sum(result.replay_time_ms for result in results)
    timing_vectors = [
        result.replay_times_ms for result in results if len(result.replay_times_ms) == width
    ]
    if len(timing_vectors) == len(results):
        aggregate.replay_times_ms = [
            sum(vector[index] for vector in timing_vectors) for index in range(width)
        ]
    aggregate.replay_shape_id = "shape-cycle"
    aggregate.replay_shape = None
    aggregate.checked_shape_ids = [
        shape_id for result in results for shape_id in result.checked_shape_ids
    ]
    aggregate.checked_shapes = [shape for result in results for shape in result.checked_shapes]
    aggregate.shape_cycle_size = expected
    aggregate.completed_shape_cycle = len(results) == expected
    aggregate.completed_scheduled_cycle = any(
        result.completed_scheduled_cycle for result in results
    )
    aggregate.scheduled_cycle = max(result.scheduled_cycle for result in results)
    if expected > 1:
        aggregate.replay_mode = "shape_cycle"
    return aggregate


def _capture_tensor(value: torch.Tensor, *, clone: bool) -> torch.Tensor:
    captured = value.detach()
    return captured.clone() if clone else captured


def _autocast_state(device_type: str) -> tuple[bool, torch.dtype | None]:
    enabled = torch.is_autocast_enabled(device_type)
    return enabled, torch.get_autocast_dtype(device_type) if enabled else None


def _register_output_gradient_capture(
    output: Any,
    *,
    clone: bool,
    callback: Callable[[Any], None],
) -> None:
    """Capture output gradients without wrapping module outputs in autograd views."""
    leaves, spec = tree_flatten(output)
    gradients: list[Any] = [None] * len(leaves)

    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or not leaf.requires_grad:
            continue

        def capture(gradient: torch.Tensor, *, index: int = index) -> None:
            gradients[index] = _capture_tensor(gradient, clone=clone)
            callback(tree_unflatten(list(gradients), spec))

        leaf.register_hook(capture)


def _first_tensor(*values: Any) -> torch.Tensor | None:
    leaves, _ = tree_flatten(values)
    return next((leaf for leaf in leaves if isinstance(leaf, torch.Tensor)), None)
