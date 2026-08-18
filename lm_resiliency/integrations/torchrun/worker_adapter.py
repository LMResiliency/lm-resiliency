"""Worker adapters for zero-import torchrun activation.

Torchrun owns process and rendezvous lifecycle, but it does not know which
framework objects represent training state. Bootstrap monitoring is installed
before the user module starts and selects framework-specific attachment logic
as the module imports its training stack.
"""

from __future__ import annotations

import builtins
import hashlib
import importlib
import importlib.util
import inspect
import os
import threading
import time
import types
import weakref
from collections.abc import Callable, Iterable, Mapping, MutableMapping
from dataclasses import dataclass, fields, replace
from pathlib import Path
from typing import Any, Protocol, Union, get_args, get_origin, get_type_hints

_toml = importlib.import_module(
    "tomllib" if importlib.util.find_spec("tomllib") is not None else "tomli"
)

_ACTIVATE_ENV = "LM_RESILIENCY_TORCHRUN_BOOTSTRAP"
_CONFIG_ENV = "LM_RESILIENCY_TORCHRUN_WORKER_CONFIG"
_POLICY_DIGEST_ENV = "LM_RESILIENCY_TORCHRUN_WORKER_POLICY_DIGEST"
_RUN_ID_ENV = "LM_RESILIENCY_TORCHRUN_RUN_ID"
_NODE_ID_ENV = "LM_RESILIENCY_TORCHRUN_NODE_ID"
_LOCAL_WORLD_SIZE_ENV = "LOCAL_WORLD_SIZE"
_CONTEXT_PATH_ENV = "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT"
_EXPECTED_GENERATION_ENV = "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION"
_ATTACHED_ENV = "LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED"
_GENERATION_ENV = "LM_RESILIENCY_GENERATION"
_CHECKPOINT_STEP_ENV = "LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP"
_MAX_WORKER_CONFIG_BYTES = 1 << 20
_BUILTIN_PYTORCH = "pytorch"
_BUILTIN_TORCHTITAN = "torchtitan"
_BUILTIN_MEGATRON = "megatron"
_BUILTIN_DEEPSPEED = "deepspeed"
_ROOT_CONFIG_FIELDS = {
    "adapter",
    "schema_version",
    "interval",
    "enable_checkpoint",
    "enable_detection",
    "checkpoint",
    "replay",
}


class TorchrunWorkerAdapterError(RuntimeError):
    """Raised when automatic worker instrumentation cannot attach safely."""


@dataclass(frozen=True, slots=True)
class TorchrunWorkerContext:
    """Launcher state supplied to a worker adapter before user code starts."""

    run_id: str
    node_id: str
    local_world_size: int
    restart_context_path: Path
    config_path: Path | None = None
    generation: int = 0
    logical_node_slot: int | None = None
    first_global_rank: int | None = None
    checkpoint_step: int | None = None
    checkpoint_id: str | None = None
    checkpoint_source: str | None = None
    recovery_mode: str | None = None
    topology_digest: str | None = None
    restart_deadline_unix_ms: int | None = None

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.node_id, "node_id")
        if (
            isinstance(self.local_world_size, bool)
            or not isinstance(self.local_world_size, int)
            or self.local_world_size < 1
        ):
            raise ValueError("local_world_size must be a positive integer")
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise ValueError("restart_context_path must be absolute")
        if self.config_path is not None:
            if not isinstance(self.config_path, Path):
                raise TypeError("config_path must be pathlib.Path")
            if not self.config_path.is_absolute():
                raise ValueError("config_path must be absolute")
        for name, value in (
            ("generation", self.generation),
            ("logical_node_slot", self.logical_node_slot),
            ("first_global_rank", self.first_global_rank),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        if self.checkpoint_step is not None and (
            isinstance(self.checkpoint_step, bool)
            or not isinstance(self.checkpoint_step, int)
            or self.checkpoint_step < 1
        ):
            raise ValueError("checkpoint_step must be a positive integer")
        if self.restart_deadline_unix_ms is not None and (
            isinstance(self.restart_deadline_unix_ms, bool)
            or not isinstance(self.restart_deadline_unix_ms, int)
            or self.restart_deadline_unix_ms < 1
        ):
            raise ValueError("restart_deadline_unix_ms must be a positive integer")
        if self.checkpoint_id is not None:
            _nonempty(self.checkpoint_id, "checkpoint_id")
        if self.checkpoint_source not in {None, "gemini", "durable"}:
            raise ValueError("checkpoint_source must be 'gemini' or 'durable'")
        if self.recovery_mode not in {None, "latest", "recovery_verified"}:
            raise ValueError("recovery_mode must be 'latest' or 'recovery_verified'")
        if self.topology_digest is not None:
            _nonempty(self.topology_digest, "topology_digest")
        recovery_fields = (
            self.logical_node_slot,
            self.first_global_rank,
            self.checkpoint_step,
            self.checkpoint_source,
            self.recovery_mode,
            self.topology_digest,
            self.restart_deadline_unix_ms,
        )
        if self.generation == 0 and (
            self.checkpoint_id is not None or any(value is not None for value in recovery_fields)
        ):
            raise ValueError("initial worker context must not contain recovery state")
        if self.generation > 0 and any(value is None for value in recovery_fields):
            raise ValueError("replacement worker context requires complete recovery state")
        if (
            self.logical_node_slot is not None
            and self.first_global_rank is not None
            and self.first_global_rank != self.logical_node_slot * self.local_world_size
        ):
            raise ValueError("first_global_rank does not match logical_node_slot")
        if self.checkpoint_source == "gemini" and self.checkpoint_id is not None:
            raise ValueError("GEMINI recovery must not set checkpoint_id")
        if self.checkpoint_source == "durable" and self.checkpoint_id is None:
            raise ValueError("durable recovery requires checkpoint_id")


class TorchrunWorkerAdapter(Protocol):
    """Framework-specific worker instrumentation installed before user code."""

    def install(self, context: TorchrunWorkerContext) -> None:
        """Install framework hooks for one worker process."""


class _SingleAttachAdapter:
    """Common lifecycle for adapters that attach at one framework hook."""

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        before_attach: Callable[[], None] | None = None,
        on_attach: Callable[[], None] | None = None,
    ) -> None:
        self._options = dict(options or {})
        self._before_attach = before_attach
        self._on_attach = on_attach
        self._lock = threading.RLock()
        self._installed = False
        self._context: TorchrunWorkerContext | None = None
        self._handle: Any | None = None
        self._restore_hook: Callable[[], None] | None = None
        self._restore_cleanup_hook: Callable[[], None] | None = None

    @property
    def attached(self) -> bool:
        """Whether automatic attachment has completed."""

        with self._lock:
            return self._handle is not None

    @property
    def handle(self) -> Any | None:
        """The attached resiliency handle, if attachment has completed."""

        with self._lock:
            return self._handle

    def install(self, context: TorchrunWorkerContext) -> None:
        if not isinstance(context, TorchrunWorkerContext):
            raise TypeError("context must be TorchrunWorkerContext")
        if context.checkpoint_source == "durable":
            raise TorchrunWorkerAdapterError(
                "built-in worker adapters require GEMINI restart contexts; "
                "durable recovery requires a custom worker adapter"
            )
        with self._lock:
            if self._installed:
                if self._context != context:
                    raise TorchrunWorkerAdapterError(
                        f"{type(self).__name__} is already installed for another context"
                    )
                return
            self._installed = True
            self._context = context
            try:
                self._restore_hook = self._install_hook()
            except BaseException:
                self._installed = False
                self._context = None
                raise

    def close(self) -> None:
        """Close the resiliency handle before restoring framework hooks."""

        with self._lock:
            if not self._installed:
                return
            handle = self._handle
            self._handle = None
            self._restore_locked()
            self._restore_cleanup_locked()
            os.environ.pop(_ATTACHED_ENV, None)
            self._installed = False
            self._context = None
        close = getattr(handle, "close", None)
        if callable(close):
            close()

    def _install_hook(self) -> Callable[[], None]:
        raise NotImplementedError

    def _enable(self, *objects: Any) -> Any:
        raise NotImplementedError

    def _install_cleanup_hook(self, *objects: Any) -> Callable[[], None]:
        del objects
        import torch.distributed as dist

        original = dist.destroy_process_group
        adapter = self

        def destroy_process_group(group: Any | None = None) -> None:
            world = getattr(dist.group, "WORLD", None)
            if group is None or group is world:
                _close_before_teardown(adapter, lambda: original(group))
                return
            original(group)

        dist.destroy_process_group = destroy_process_group

        def restore() -> None:
            if dist.destroy_process_group is destroy_process_group:
                dist.destroy_process_group = original

        return restore

    def _attach(self, *objects: Any) -> Any:
        if self._before_attach is not None:
            self._before_attach()
        attached_now = False
        with self._lock:
            if self._handle is None:
                handle = self._enable(*objects)
                try:
                    self._validate_recovery(handle)
                    self._restore_cleanup_hook = self._install_cleanup_hook(*objects)
                except BaseException:
                    close = getattr(handle, "close", None)
                    if callable(close):
                        try:
                            close()
                        except Exception:
                            pass
                    raise
                self._handle = handle
                os.environ[_ATTACHED_ENV] = "1"
                self._restore_locked()
                attached_now = True
            result = self._handle
        if attached_now and self._on_attach is not None:
            self._on_attach()
        return result

    def _validate_recovery(self, handle: Any) -> None:
        assert self._context is not None
        expected = self._context.checkpoint_step
        if expected is None:
            return
        observed = getattr(
            handle,
            "recovered_step",
            getattr(handle, "step_count", None),
        )
        if isinstance(observed, bool) or not isinstance(observed, int):
            raise TorchrunWorkerAdapterError(
                "framework resiliency handle does not expose the recovered step"
            )
        if observed != expected:
            raise TorchrunWorkerAdapterError(
                f"framework recovered step {observed}, expected manager-selected step {expected}"
            )

    def _restore_locked(self) -> None:
        restore = self._restore_hook
        self._restore_hook = None
        if restore is not None:
            restore()

    def _restore_cleanup_locked(self) -> None:
        restore = self._restore_cleanup_hook
        self._restore_cleanup_hook = None
        if restore is not None:
            restore()


class NativePyTorchAdapter(_SingleAttachAdapter):
    """Attach to one unambiguous native PyTorch model/optimizer pair.

    The existing PyTorch integration owns DDP, FSDP2, HSDP, and model-parallel
    topology discovery. This adapter only discovers the root module and
    optimizer that would be passed to ``enable_resiliency()`` explicitly.
    """

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        before_attach: Callable[[], None] | None = None,
        on_attach: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            options,
            before_attach=before_attach,
            on_attach=on_attach,
        )
        self._optimizers: list[weakref.ReferenceType[Any]] = []
        self._distributed_models: list[weakref.ReferenceType[Any]] = []
        self._attached_model: Any | None = None
        self._attached_optimizer: Any | None = None
        self._original_module_call: Callable[..., Any] | None = None
        self._original_optimizer_init: Callable[..., None] | None = None
        self._original_optimizer_init_subclass: Any = None
        self._original_ddp_init: Callable[..., None] | None = None
        self._original_fsdp_init: Callable[..., None] | None = None
        self._original_fully_shard: Callable[..., Any] | None = None
        self._optimizer_initializers: list[tuple[type[Any], Callable[..., None] | None]] = []
        self._optimizer_init_state = threading.local()

    def _install_hook(self) -> Callable[[], None]:
        import torch.nn
        import torch.optim
        from torch.nn.parallel import DistributedDataParallel

        self._original_module_call = torch.nn.Module.__call__
        self._original_optimizer_init = torch.optim.Optimizer.__init__
        self._original_optimizer_init_subclass = torch.optim.Optimizer.__dict__.get(
            "__init_subclass__"
        )
        self._original_ddp_init = DistributedDataParallel.__init__
        adapter = self

        def module_call(instance: Any, *args: Any, **kwargs: Any) -> Any:
            if adapter._distributed_world_size() == 1:
                adapter._attach_if_candidate(instance)
            elif not adapter.attached:
                raise TorchrunWorkerAdapterError(
                    "distributed native PyTorch automatic attachment requires a "
                    "DDP or FSDP construction boundary on every rank"
                )
            assert adapter._original_module_call is not None
            return adapter._original_module_call(instance, *args, **kwargs)

        def optimizer_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            assert adapter._original_optimizer_init is not None
            adapter._original_optimizer_init(instance, *args, **kwargs)
            with adapter._lock:
                adapter._optimizers.append(weakref.ref(instance))

        def ddp_init(instance: Any, *args: Any, **kwargs: Any) -> None:
            assert adapter._original_ddp_init is not None
            adapter._original_ddp_init(instance, *args, **kwargs)
            adapter._register_distributed_model(instance)

        torch.nn.Module.__call__ = module_call
        torch.optim.Optimizer.__init__ = optimizer_init
        DistributedDataParallel.__init__ = ddp_init
        for optimizer_type in _optimizer_types(torch.optim.Optimizer):
            self._wrap_optimizer_initializer(optimizer_type)

        original_init_subclass = self._original_optimizer_init_subclass

        def optimizer_init_subclass(
            optimizer_type: type[Any],
            **kwargs: Any,
        ) -> None:
            if original_init_subclass is None:
                super(torch.optim.Optimizer, optimizer_type).__init_subclass__(**kwargs)
            else:
                original_init_subclass.__get__(None, optimizer_type)(**kwargs)
            adapter._wrap_optimizer_initializer(optimizer_type)

        torch.optim.Optimizer.__init_subclass__ = classmethod(optimizer_init_subclass)

        fsdp_type = None
        fully_shard_module = None
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel

            fsdp_type = FullyShardedDataParallel
            self._original_fsdp_init = fsdp_type.__init__

            def fsdp_init(instance: Any, *args: Any, **kwargs: Any) -> None:
                assert adapter._original_fsdp_init is not None
                adapter._original_fsdp_init(instance, *args, **kwargs)
                adapter._register_distributed_model(instance)

            fsdp_type.__init__ = fsdp_init
        except ImportError:
            pass
        try:
            fully_shard_module = importlib.import_module("torch.distributed.fsdp")
            fully_shard = getattr(fully_shard_module, "fully_shard", None)
            if callable(fully_shard):
                self._original_fully_shard = fully_shard

                def wrapped_fully_shard(
                    module: Any,
                    *args: Any,
                    **kwargs: Any,
                ) -> Any:
                    assert adapter._original_fully_shard is not None
                    result = adapter._original_fully_shard(module, *args, **kwargs)
                    adapter._register_distributed_model(module)
                    return result

                fully_shard_module.fully_shard = wrapped_fully_shard
        except ImportError:
            pass

        def restore() -> None:
            assert self._original_module_call is not None
            assert self._original_optimizer_init is not None
            assert self._original_ddp_init is not None
            torch.nn.Module.__call__ = self._original_module_call
            torch.optim.Optimizer.__init__ = self._original_optimizer_init
            if self._original_optimizer_init_subclass is None:
                del torch.optim.Optimizer.__init_subclass__
            else:
                torch.optim.Optimizer.__init_subclass__ = self._original_optimizer_init_subclass
            DistributedDataParallel.__init__ = self._original_ddp_init
            for optimizer_type, original_init in self._optimizer_initializers:
                if original_init is None:
                    del optimizer_type.__init__
                else:
                    optimizer_type.__init__ = original_init
            self._optimizer_initializers.clear()
            if fsdp_type is not None and self._original_fsdp_init is not None:
                fsdp_type.__init__ = self._original_fsdp_init
            if fully_shard_module is not None and self._original_fully_shard is not None:
                fully_shard_module.fully_shard = self._original_fully_shard

        return restore

    @staticmethod
    def _distributed_world_size() -> int:
        import torch.distributed as dist

        return dist.get_world_size() if dist.is_initialized() else 1

    def _register_distributed_model(self, model: Any) -> None:
        with self._lock:
            self._distributed_models.append(weakref.ref(model))
        self._attach_distributed_candidate()

    def _attach_distributed_candidate(self) -> None:
        if self._distributed_world_size() <= 1:
            return
        with self._lock:
            candidates = _outermost_models(
                model
                for reference in self._distributed_models
                if (model := reference()) is not None
            )
        if len(candidates) == 1:
            self._attach_if_candidate(
                candidates[0],
                require_optimizer=False,
                defer_no_match=True,
            )

    def _attach_if_candidate(
        self,
        model: Any,
        *,
        require_optimizer: bool = True,
        defer_no_match: bool = False,
    ) -> None:
        with self._lock:
            if self._handle is not None:
                return
            trainable = {
                id(parameter) for parameter in model.parameters() if parameter.requires_grad
            }
            if not trainable:
                return
            optimizers = []
            for reference in self._optimizers:
                optimizer = reference()
                if optimizer is not None:
                    optimizers.append(optimizer)
            model_parameters = {id(parameter) for parameter in model.parameters()}
            matches = [
                optimizer
                for optimizer in optimizers
                if trainable
                <= {
                    id(parameter)
                    for group in optimizer.param_groups
                    for parameter in group["params"]
                }
                <= model_parameters
            ]
            if len(matches) != 1:
                if not optimizers:
                    if not require_optimizer:
                        return
                    reason = "no optimizer was constructed before the first model forward"
                elif not matches:
                    if defer_no_match:
                        return
                    reason = "no optimizer owns exactly one model's trainable parameters"
                else:
                    reason = "multiple optimizers own the model's trainable parameters"
                raise TorchrunWorkerAdapterError(f"cannot attach native PyTorch adapter: {reason}")
            self._attached_model = model
            self._attached_optimizer = matches[0]
        self._attach(model, matches[0])

    def _enable(self, model: Any, optimizer: Any) -> Any:
        assert self._context is not None
        from lm_resiliency.integrations.pytorch import enable_resiliency

        checkpoint, replay, root = _feature_options(self._options, self._context)
        return enable_resiliency(
            model,
            optimizer,
            interval=root["interval"],
            enable_checkpoint=root["enable_checkpoint"],
            enable_detection=root["enable_detection"],
            checkpoint=checkpoint,
            replay=replay,
            recovery_mode=self._context.recovery_mode,
            **_recovery_options(self._context),
        )

    def _wrap_optimizer_initializer(self, optimizer_type: type[Any]) -> None:
        if any(existing is optimizer_type for existing, _original in self._optimizer_initializers):
            return
        declared_init = optimizer_type.__dict__.get("__init__")
        original_init = declared_init if callable(declared_init) else None
        inherited_init = optimizer_type.__init__
        delegate = original_init or inherited_init
        adapter = self

        def concrete_optimizer_init(
            instance: Any,
            *args: Any,
            **kwargs: Any,
        ) -> None:
            depth = getattr(adapter._optimizer_init_state, "depth", 0)
            adapter._optimizer_init_state.depth = depth + 1
            try:
                delegate(instance, *args, **kwargs)
            finally:
                adapter._optimizer_init_state.depth = depth
            if depth == 0:
                adapter._attach_distributed_candidate()

        self._optimizer_initializers.append((optimizer_type, original_init))
        optimizer_type.__init__ = concrete_optimizer_init


NativePyTorchDDPAdapter = NativePyTorchAdapter


class TorchTitanWorkerAdapter(_SingleAttachAdapter):
    """Attach to TorchTitan's initialized Trainer before ``Trainer.train``."""

    def _install_hook(self) -> Callable[[], None]:
        try:
            module = importlib.import_module("torchtitan.train")
            trainer_type = module.Trainer
            original = trainer_type.train
        except (ImportError, AttributeError) as error:
            raise TorchrunWorkerAdapterError(
                "TorchTitan adapter requires torchtitan.train.Trainer"
            ) from error
        adapter = self

        def train(trainer: Any, *args: Any, **kwargs: Any) -> Any:
            adapter._attach(trainer)
            return original(trainer, *args, **kwargs)

        trainer_type.train = train
        return lambda: setattr(trainer_type, "train", original)

    def _enable(self, trainer: Any) -> Any:
        assert self._context is not None
        from lm_resiliency.integrations.torchtitan import enable_resiliency

        checkpoint, replay, root = _feature_options(self._options, self._context)
        return enable_resiliency(
            trainer,
            interval=root["interval"],
            enable_checkpoint=root["enable_checkpoint"],
            enable_detection=root["enable_detection"],
            ckpt_config=checkpoint,
            detection_config=replay,
            recovery_mode=self._context.recovery_mode,
            **_recovery_options(self._context),
        )


class MegatronWorkerAdapter(_SingleAttachAdapter):
    """Attach to Megatron's model chunks, optimizer, and scheduler."""

    def _install_hook(self) -> Callable[[], None]:
        try:
            module = importlib.import_module("megatron.training.training")
            original_setup = getattr(module, "setup_model_and_optimizer")
            original_train = getattr(module, "train")
        except (ImportError, AttributeError) as error:
            raise TorchrunWorkerAdapterError(
                "Megatron adapter requires megatron.training.training"
            ) from error
        adapter = self

        def setup(*args: Any, **kwargs: Any) -> Any:
            result = original_setup(*args, **kwargs)
            if not isinstance(result, tuple) or len(result) != 3:
                raise TorchrunWorkerAdapterError(
                    "Megatron setup_model_and_optimizer must return (model, optimizer, scheduler)"
                )
            adapter._attach(*result)
            return result

        def train(*args: Any, **kwargs: Any) -> Any:
            if not adapter.attached:
                try:
                    bound = inspect.signature(original_train).bind_partial(*args, **kwargs)
                except (TypeError, ValueError) as error:
                    raise TorchrunWorkerAdapterError(
                        "cannot inspect megatron.training.training.train arguments"
                    ) from error
                arguments = bound.arguments
                required = ("model", "optimizer", "opt_param_scheduler")
                missing = tuple(name for name in required if name not in arguments)
                if missing:
                    raise TorchrunWorkerAdapterError(
                        "Megatron train must receive model, optimizer, and opt_param_scheduler"
                    )
                adapter._attach(*(arguments[name] for name in required))
            return original_train(*args, **kwargs)

        setattr(module, "setup_model_and_optimizer", setup)
        setattr(module, "train", train)

        def restore() -> None:
            setattr(module, "setup_model_and_optimizer", original_setup)
            setattr(module, "train", original_train)

        return restore

    def _enable(self, model: Any, optimizer: Any, scheduler: Any) -> Any:
        if not isinstance(model, list) or not model:
            raise TorchrunWorkerAdapterError(
                "Megatron adapter requires a non-empty model chunk list"
            )
        if optimizer is None:
            raise TorchrunWorkerAdapterError("Megatron adapter requires an optimizer")
        assert self._context is not None
        from lm_resiliency.integrations.megatron.training import enable_resiliency

        module = importlib.import_module("megatron.training.training")
        args = module.get_args()
        holder: dict[str, Any] = {}

        def capture_loop_state() -> dict[str, Any]:
            handle = holder.get("handle")
            if handle is None:
                raise TorchrunWorkerAdapterError(
                    "Megatron loop state was captured before resiliency attachment completed"
                )
            state_dict = getattr(scheduler, "state_dict", None)
            if not callable(state_dict):
                raise TorchrunWorkerAdapterError("Megatron scheduler does not expose state_dict()")
            return {
                "torchrun_megatron_loop": {
                    "iteration": _nonnegative_int(handle.step_count, "Megatron iteration"),
                    "consumed_train_samples": _nonnegative_int(
                        args.consumed_train_samples,
                        "Megatron consumed_train_samples",
                    ),
                    "skipped_train_samples": _nonnegative_int(
                        args.skipped_train_samples,
                        "Megatron skipped_train_samples",
                    ),
                    "scheduler": state_dict(),
                }
            }

        def restore_loop_state(state: dict[str, Any]) -> None:
            loop_state = state.get("torchrun_megatron_loop")
            if not isinstance(loop_state, Mapping):
                raise TorchrunWorkerAdapterError(
                    "Megatron checkpoint is missing torchrun loop state"
                )
            expected = {
                "iteration",
                "consumed_train_samples",
                "skipped_train_samples",
                "scheduler",
            }
            if set(loop_state) != expected:
                raise TorchrunWorkerAdapterError(
                    "Megatron torchrun loop state has unexpected fields"
                )
            args.iteration = _nonnegative_int(
                loop_state["iteration"],
                "Megatron iteration",
            )
            args.consumed_train_samples = _nonnegative_int(
                loop_state["consumed_train_samples"],
                "Megatron consumed_train_samples",
            )
            args.skipped_train_samples = _nonnegative_int(
                loop_state["skipped_train_samples"],
                "Megatron skipped_train_samples",
            )
            load_state_dict = getattr(scheduler, "load_state_dict", None)
            if not callable(load_state_dict):
                raise TorchrunWorkerAdapterError(
                    "Megatron scheduler does not expose load_state_dict()"
                )
            load_state_dict(loop_state["scheduler"])

        checkpoint, replay, root = _feature_options(self._options, self._context)
        handle = enable_resiliency(
            model,
            optimizer,
            scheduler,
            interval=root["interval"],
            enable_checkpoint=root["enable_checkpoint"],
            enable_detection=root["enable_detection"],
            ckpt_config=checkpoint,
            detection_config=replay,
            recovery_mode=self._context.recovery_mode,
            extra_state_fn=capture_loop_state,
            load_extra_state_fn=restore_loop_state,
            **_recovery_options(self._context),
        )
        holder["handle"] = handle
        args.iteration = handle.step_count
        return handle

    def _install_cleanup_hook(self, *objects: Any) -> Callable[[], None]:
        del objects
        module = importlib.import_module("megatron.core.parallel_state")
        original = module.destroy_model_parallel
        adapter = self

        def destroy_model_parallel(*args: Any, **kwargs: Any) -> Any:
            return _close_before_teardown(
                adapter,
                lambda: original(*args, **kwargs),
            )

        module.destroy_model_parallel = destroy_model_parallel

        def restore() -> None:
            if module.destroy_model_parallel is destroy_model_parallel:
                module.destroy_model_parallel = original

        return restore


class DeepSpeedWorkerAdapter(_SingleAttachAdapter):
    """Attach to the engine returned by ``deepspeed.initialize``."""

    def __init__(
        self,
        options: Mapping[str, Any] | None = None,
        *,
        before_attach: Callable[[], None] | None = None,
        on_attach: Callable[[], None] | None = None,
    ) -> None:
        super().__init__(
            options,
            before_attach=before_attach,
            on_attach=on_attach,
        )
        self._engine: Any | None = None
        self._original_load_checkpoint: Callable[..., Any] | None = None

    def _install_hook(self) -> Callable[[], None]:
        try:
            module = importlib.import_module("deepspeed")
            original = getattr(module, "initialize")
        except (ImportError, AttributeError) as error:
            raise TorchrunWorkerAdapterError(
                "DeepSpeed adapter requires deepspeed.initialize"
            ) from error
        adapter = self

        def initialize(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            if not isinstance(result, tuple) or not result:
                raise TorchrunWorkerAdapterError(
                    "deepspeed.initialize must return a tuple beginning with the engine"
                )
            adapter._attach(result[0])
            return result

        setattr(module, "initialize", initialize)
        return lambda: setattr(module, "initialize", original)

    def _enable(self, engine: Any) -> Any:
        if engine is None:
            raise TorchrunWorkerAdapterError("DeepSpeed adapter requires an engine")
        assert self._context is not None
        from lm_resiliency.integrations.deepspeed.training import enable_resiliency

        checkpoint, replay, root = _feature_options(self._options, self._context)
        handle = enable_resiliency(
            engine,
            interval=root["interval"],
            enable_checkpoint=root["enable_checkpoint"],
            enable_detection=root["enable_detection"],
            ckpt_config=checkpoint,
            detection_config=replay,
            recovery_mode=self._context.recovery_mode,
            **_recovery_options(self._context),
        )
        self._engine = engine
        return handle

    def _validate_recovery(self, handle: Any) -> None:
        super()._validate_recovery(handle)
        assert self._context is not None
        if self._context.checkpoint_step is None:
            return
        engine = self._engine
        load_checkpoint = getattr(engine, "load_checkpoint", None)
        if not callable(load_checkpoint):
            return
        self._original_load_checkpoint = load_checkpoint

        def reject_framework_load(*_args: Any, **_kwargs: Any) -> Any:
            raise TorchrunWorkerAdapterError(
                "DeepSpeed load_checkpoint() cannot run after manager-selected "
                "GEMINI recovery; use a custom worker adapter to coordinate "
                "framework checkpoint client state"
            )

        engine.load_checkpoint = reject_framework_load

    def _install_cleanup_hook(self, engine: Any) -> Callable[[], None]:
        original = getattr(engine, "destroy", None)
        if not callable(original):
            return super()._install_cleanup_hook(engine)
        adapter = self

        def destroy(*args: Any, **kwargs: Any) -> Any:
            return _close_before_teardown(
                adapter,
                lambda: original(*args, **kwargs),
            )

        engine.destroy = destroy

        def restore() -> None:
            if engine.destroy is destroy:
                engine.destroy = original

        return restore

    def close(self) -> None:
        engine = self._engine
        original_load_checkpoint = self._original_load_checkpoint
        if engine is not None and original_load_checkpoint is not None:
            engine.load_checkpoint = original_load_checkpoint
        self._original_load_checkpoint = None
        self._engine = None
        super().close()


class _AutoFrameworkAdapter:
    """Infer one built-in adapter from framework imports in the worker."""

    _SPECIALIZED = {
        _BUILTIN_DEEPSPEED: DeepSpeedWorkerAdapter,
        _BUILTIN_MEGATRON: MegatronWorkerAdapter,
        _BUILTIN_TORCHTITAN: TorchTitanWorkerAdapter,
    }

    def __init__(self, options: Mapping[str, Any] | None = None) -> None:
        self._options = dict(options or {})
        self._lock = threading.RLock()
        self._state = threading.local()
        self._context: TorchrunWorkerContext | None = None
        self._native: NativePyTorchAdapter | None = None
        self._delegate: TorchrunWorkerAdapter | None = None
        self._selected_framework: str | None = None
        self._inference_error: TorchrunWorkerAdapterError | None = None
        self._original_import: Callable[..., Any] | None = None
        self._original_import_module: Callable[..., Any] | None = None
        self._import_wrapper: Callable[..., Any] | None = None
        self._import_module_wrapper: Callable[..., Any] | None = None

    @property
    def selected_framework(self) -> str | None:
        """Framework selected from observed imports, if any."""

        with self._lock:
            return self._selected_framework

    @property
    def delegate(self) -> TorchrunWorkerAdapter | None:
        """Concrete framework adapter installed by inference."""

        with self._lock:
            return self._delegate if self._delegate is not None else self._native

    def install(self, context: TorchrunWorkerContext) -> None:
        if not isinstance(context, TorchrunWorkerContext):
            raise TypeError("context must be TorchrunWorkerContext")
        if context.checkpoint_source == "durable":
            raise TorchrunWorkerAdapterError(
                "built-in worker adapters require GEMINI restart contexts; "
                "durable recovery requires a custom worker adapter"
            )
        with self._lock:
            if self._context is not None:
                if self._context != context:
                    raise TorchrunWorkerAdapterError(
                        "automatic framework adapter is already installed for another context"
                    )
                return
            self._context = context
            self._original_import = builtins.__import__
            self._original_import_module = importlib.import_module

            def monitored_import(
                name: str,
                globals: Mapping[str, Any] | None = None,
                locals: Mapping[str, Any] | None = None,
                fromlist: tuple[str, ...] = (),
                level: int = 0,
            ) -> Any:
                assert self._original_import is not None
                return self._run_import(
                    name,
                    self._original_import,
                    name,
                    globals,
                    locals,
                    fromlist,
                    level,
                )

            def monitored_import_module(name: str, package: str | None = None) -> Any:
                assert self._original_import_module is not None
                return self._run_import(
                    name,
                    self._original_import_module,
                    name,
                    package,
                )

            self._import_wrapper = monitored_import
            self._import_module_wrapper = monitored_import_module
            builtins.__import__ = monitored_import
            importlib.import_module = monitored_import_module

    def close(self) -> None:
        with self._lock:
            self._restore_imports_locked()
            delegates = tuple(
                adapter for adapter in (self._delegate, self._native) if adapter is not None
            )
            self._delegate = None
            self._native = None
            self._context = None
        for adapter in delegates:
            close = getattr(adapter, "close", None)
            if callable(close):
                close()

    def _run_import(
        self,
        observed_name: str,
        operation: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        depth = getattr(self._state, "depth", 0)
        outermost = depth == 0
        if outermost:
            with self._lock:
                self._state.import_snapshot = (
                    self._selected_framework,
                    self._native is not None,
                )
        self._state.depth = depth + 1
        try:
            result = operation(*args, **kwargs)
            if not getattr(self._state, "suppress_observation", False):
                root = self._supported_root(observed_name)
                if root is not None:
                    self._observe_root(root)
            return result
        except BaseException:
            if outermost:
                selected_framework, had_native = self._state.import_snapshot
                self._rollback_import_selection(selected_framework, had_native)
            raise
        finally:
            self._state.depth = depth
            if outermost:
                del self._state.import_snapshot

    def _rollback_import_selection(
        self,
        selected_framework: str | None,
        had_native: bool,
    ) -> None:
        with self._lock:
            if self._selected_framework == selected_framework and (
                (self._native is not None) == had_native
            ):
                return
            delegates = tuple(
                adapter for adapter in (self._delegate, self._native) if adapter is not None
            )
            self._delegate = None
            self._native = None
            self._selected_framework = selected_framework
            self._inference_error = None
        for adapter in delegates:
            adapter.close()
        if selected_framework is None and had_native:
            with self._lock:
                self._install_tentative_pytorch_locked()

    def _supported_root(self, module_name: str) -> str | None:
        if not isinstance(module_name, str) or not module_name or module_name.startswith("."):
            return None
        root = module_name.split(".", 1)[0]
        if root not in {*self._SPECIALIZED, "torch"}:
            return None
        return root

    def _observe_root(self, root: str) -> None:
        self._state.suppress_observation = True
        try:
            with self._lock:
                if root == "torch":
                    self._install_tentative_pytorch_locked()
                else:
                    self._select_specialized_locked(root)
        finally:
            self._state.suppress_observation = False

    def _install_tentative_pytorch_locked(self) -> None:
        if self._selected_framework is not None or self._native is not None:
            return
        assert self._context is not None
        adapter = NativePyTorchAdapter(
            self._options,
            before_attach=self._raise_if_inference_failed,
            on_attach=self._native_attached,
        )
        adapter.install(self._context)
        self._native = adapter

    def _select_specialized_locked(self, framework: str) -> None:
        if self._selected_framework is not None:
            if self._selected_framework != framework:
                error = TorchrunWorkerAdapterError(
                    "multiple supported training frameworks were imported: "
                    f"{self._selected_framework!r} and {framework!r}"
                )
                self._inference_error = error
                raise error
            return
        if self._native is not None:
            if self._native.attached:
                error = TorchrunWorkerAdapterError(
                    f"{framework} was imported after the native PyTorch adapter attached"
                )
                self._inference_error = error
                raise error
            self._native.close()
            self._native = None
        assert self._context is not None
        adapter_type = self._SPECIALIZED[framework]
        self._selected_framework = framework
        adapter = adapter_type(
            self._options,
            before_attach=self._raise_if_inference_failed,
            on_attach=lambda: self._specialized_attached(framework),
        )
        self._delegate = adapter
        try:
            adapter.install(self._context)
        except BaseException:
            self._delegate = None
            self._selected_framework = None
            raise

    def _native_attached(self) -> None:
        with self._lock:
            if self._native is None:
                raise TorchrunWorkerAdapterError(
                    "native PyTorch adapter attached after framework selection changed"
                )
            self._delegate = self._native
            self._native = None
            self._selected_framework = _BUILTIN_PYTORCH
            self._restore_imports_locked()

    def _specialized_attached(self, framework: str) -> None:
        with self._lock:
            if self._selected_framework != framework:
                raise TorchrunWorkerAdapterError(
                    "framework adapter attached after inference state changed"
                )
            self._restore_imports_locked()

    def _raise_if_inference_failed(self) -> None:
        with self._lock:
            if self._inference_error is not None:
                raise self._inference_error

    def _restore_imports_locked(self) -> None:
        if (
            self._import_wrapper is not None
            and builtins.__import__ is self._import_wrapper
            and self._original_import is not None
        ):
            builtins.__import__ = self._original_import
        if (
            self._import_module_wrapper is not None
            and importlib.import_module is self._import_module_wrapper
            and self._original_import_module is not None
        ):
            importlib.import_module = self._original_import_module
        self._import_wrapper = None
        self._import_module_wrapper = None


_bootstrap_lock = threading.Lock()
_installed_adapter: TorchrunWorkerAdapter | None = None


def bootstrap_worker_from_environment(
    environment: Mapping[str, str] | None = None,
) -> TorchrunWorkerAdapter | None:
    """Install automatic inference or a configured custom worker adapter."""

    global _installed_adapter
    environ = os.environ if environment is None else environment
    if environ.get(_ACTIVATE_ENV) != "1":
        return None
    with _bootstrap_lock:
        if _installed_adapter is not None:
            return _installed_adapter
        context = _context_from_environment(environ)
        generation_target = environ if isinstance(environ, MutableMapping) else os.environ
        generation_target[_GENERATION_ENV] = str(context.generation)
        generation_target[_CHECKPOINT_STEP_ENV] = str(context.checkpoint_step or 0)
        options = _load_config(
            context.config_path,
            expected_digest=environ.get(_POLICY_DIGEST_ENV),
        )
        adapter_spec = options.pop("adapter", None)
        if adapter_spec is None:
            adapter: TorchrunWorkerAdapter = _AutoFrameworkAdapter(options)
        else:
            adapter = _load_custom_adapter(adapter_spec, context)
        adapter.install(context)
        _installed_adapter = adapter
        return adapter


def get_torchrun_worker_context(
    environment: Mapping[str, str] | None = None,
) -> TorchrunWorkerContext:
    """Return the current torchrun worker identity and recovery decision."""

    environ = os.environ if environment is None else environment
    context = _context_from_environment(environ)
    target = environ if isinstance(environ, MutableMapping) else os.environ
    target[_GENERATION_ENV] = str(context.generation)
    target[_CHECKPOINT_STEP_ENV] = str(context.checkpoint_step or 0)
    return context


def configure_worker_context_environment(
    *,
    run_id: str,
    node_id: str,
    restart_context_path: Path,
    environment: dict[str, str] | None = None,
) -> None:
    """Configure child-process environment for explicit torchrun integrations."""

    if not isinstance(restart_context_path, Path):
        raise TypeError("restart_context_path must be pathlib.Path")
    if not restart_context_path.is_absolute():
        raise ValueError("restart_context_path must be absolute")
    target = os.environ if environment is None else environment
    values = {
        _RUN_ID_ENV: _nonempty(run_id, "run_id"),
        _NODE_ID_ENV: _nonempty(node_id, "node_id"),
        _CONTEXT_PATH_ENV: str(restart_context_path),
    }
    for name, value in values.items():
        existing = target.get(name)
        if existing is not None and existing != value:
            raise TorchrunWorkerAdapterError(f"conflicting worker environment {name}")
    target.update(values)
    target.pop(_EXPECTED_GENERATION_ENV, None)
    target.pop(_CHECKPOINT_STEP_ENV, None)


def configure_worker_generation_environment(
    generation: int,
    environment: dict[str, str] | None = None,
) -> None:
    """Publish the generation selected by the rendezvous handler."""

    if isinstance(generation, bool) or not isinstance(generation, int) or generation < 0:
        raise ValueError("generation must be a non-negative integer")
    target = os.environ if environment is None else environment
    target[_EXPECTED_GENERATION_ENV] = str(generation)


def disable_worker_bootstrap_environment(
    environment: dict[str, str] | None = None,
) -> None:
    """Disable automatic instrumentation while retaining explicit context."""

    target = os.environ if environment is None else environment
    target.pop(_ACTIVATE_ENV, None)
    target.pop(_CONFIG_ENV, None)
    target.pop(_POLICY_DIGEST_ENV, None)


def configure_worker_bootstrap_environment(
    *,
    run_id: str,
    node_id: str,
    restart_context_path: Path,
    config_path: Path | None,
    policy_digest: str | None = None,
    environment: dict[str, str] | None = None,
) -> None:
    """Configure child-process environment for automatic worker bootstrap."""

    if config_path is not None:
        if not isinstance(config_path, Path):
            raise TypeError("config_path must be pathlib.Path")
        if not config_path.is_absolute():
            raise ValueError("config_path must be absolute")
    if policy_digest is not None:
        policy_digest = _nonempty(policy_digest, "policy_digest")
    target = os.environ if environment is None else environment
    before = dict(target)
    configure_worker_context_environment(
        run_id=run_id,
        node_id=node_id,
        restart_context_path=restart_context_path,
        environment=target,
    )
    values = {
        _ACTIVATE_ENV: "1",
    }
    if config_path is not None:
        values[_CONFIG_ENV] = str(config_path)
    elif _CONFIG_ENV in target:
        target.clear()
        target.update(before)
        raise TorchrunWorkerAdapterError(f"conflicting worker environment {_CONFIG_ENV}")
    if policy_digest is not None:
        values[_POLICY_DIGEST_ENV] = policy_digest
    for name, value in values.items():
        existing = target.get(name)
        if existing is not None and existing != value:
            target.clear()
            target.update(before)
            raise TorchrunWorkerAdapterError(f"conflicting worker environment {name}")
    for name, value in values.items():
        target[name] = value
    bootstrap_dir = str(Path(__file__).with_name("_worker_bootstrap"))
    paths = [item for item in target.get("PYTHONPATH", "").split(os.pathsep) if item]
    if bootstrap_dir not in paths:
        target["PYTHONPATH"] = os.pathsep.join([bootstrap_dir, *paths])


def _context_from_environment(environment: Mapping[str, str]) -> TorchrunWorkerContext:
    run_id = _required_environment(environment, _RUN_ID_ENV)
    node_id = _required_environment(environment, _NODE_ID_ENV)
    local_world_size = _positive_int(
        _required_environment(environment, _LOCAL_WORLD_SIZE_ENV),
        "local_world_size",
    )
    local_rank = _nonnegative_int(
        _required_environment(environment, "LOCAL_RANK"),
        "local_rank",
    )
    global_rank = _nonnegative_int(
        _required_environment(environment, "RANK"),
        "global_rank",
    )
    group_rank = _nonnegative_int(
        _required_environment(environment, "GROUP_RANK"),
        "group_rank",
    )
    world_size = _positive_int(
        _required_environment(environment, "WORLD_SIZE"),
        "world_size",
    )
    if local_rank >= local_world_size:
        raise TorchrunWorkerAdapterError("LOCAL_RANK must be smaller than LOCAL_WORLD_SIZE")
    if global_rank >= world_size:
        raise TorchrunWorkerAdapterError("RANK must be smaller than WORLD_SIZE")
    if global_rank != group_rank * local_world_size + local_rank:
        raise TorchrunWorkerAdapterError(
            "torchrun rank does not match GROUP_RANK, LOCAL_RANK, and LOCAL_WORLD_SIZE"
        )
    restart_context_path = Path(_required_environment(environment, _CONTEXT_PATH_ENV)).expanduser()
    if not restart_context_path.is_absolute():
        raise TorchrunWorkerAdapterError("restart context path must be absolute")
    config_value = environment.get(_CONFIG_ENV)
    config_path = Path(config_value).expanduser() if config_value else None
    if config_path is not None and not config_path.is_absolute():
        raise TorchrunWorkerAdapterError("worker config path must be absolute")
    generation = _nonnegative_int(
        _required_environment(environment, _EXPECTED_GENERATION_ENV),
        "generation",
    )
    context_fields: dict[str, Any] = {}
    if generation == 0:
        if restart_context_path.exists():
            raise TorchrunWorkerAdapterError(
                "initial worker generation must not have a restart context"
            )
    else:
        from ._simple_runtime import SimpleRestartContextFile

        restart = SimpleRestartContextFile(restart_context_path).read()
        if restart is None:
            raise TorchrunWorkerAdapterError("successor worker requires a restart context")
        if restart.run_id != run_id:
            raise TorchrunWorkerAdapterError("restart context belongs to another run")
        if restart.node_id != node_id:
            raise TorchrunWorkerAdapterError("restart context belongs to another node")
        if restart.local_world_size != local_world_size:
            raise TorchrunWorkerAdapterError("restart context changes local_world_size")
        if restart.generation != generation:
            raise TorchrunWorkerAdapterError(
                f"restart context generation {restart.generation} does not match "
                f"rendezvous generation {generation}"
            )
        if restart.expected_world_size != world_size:
            raise TorchrunWorkerAdapterError("restart context changes WORLD_SIZE")
        if restart.logical_node_slot != group_rank:
            raise TorchrunWorkerAdapterError("restart context changes GROUP_RANK")
        if restart.first_global_rank + local_rank != global_rank:
            raise TorchrunWorkerAdapterError("restart context changes RANK")
        if time.time_ns() // 1_000_000 >= restart.restart_deadline_unix_ms:
            raise TorchrunWorkerAdapterError("restart context deadline elapsed")
        context_fields = {
            "logical_node_slot": restart.logical_node_slot,
            "first_global_rank": restart.first_global_rank,
            "checkpoint_step": restart.checkpoint_step,
            "checkpoint_id": restart.checkpoint_id,
            "checkpoint_source": restart.checkpoint_source,
            "recovery_mode": restart.recovery_mode,
            "topology_digest": restart.topology_digest,
            "restart_deadline_unix_ms": restart.restart_deadline_unix_ms,
        }
    return TorchrunWorkerContext(
        run_id=run_id,
        node_id=node_id,
        local_world_size=local_world_size,
        restart_context_path=restart_context_path,
        config_path=config_path,
        generation=generation,
        **context_fields,
    )


def _load_custom_adapter(
    spec: str,
    context: TorchrunWorkerContext,
) -> TorchrunWorkerAdapter:
    if spec.count(":") != 1:
        raise TorchrunWorkerAdapterError("custom adapter must use the 'module:factory' form")
    module_name, attribute = spec.split(":", 1)
    module = importlib.import_module(_nonempty(module_name, "adapter module"))
    factory = getattr(module, _nonempty(attribute, "adapter factory"), None)
    if not callable(factory):
        raise TorchrunWorkerAdapterError("worker adapter factory is not callable")
    adapter = factory(context)
    if not callable(getattr(adapter, "install", None)):
        raise TorchrunWorkerAdapterError(
            "worker adapter factory must return an object with install(context)"
        )
    return adapter


def _load_config(
    path: Path | None,
    *,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    if path is None:
        if expected_digest is not None:
            raise TorchrunWorkerAdapterError(
                "worker policy digest was configured without a worker config"
            )
        return {}
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise TorchrunWorkerAdapterError(f"failed to read worker config {path}") from error
    return _parse_config_bytes(encoded, path=path, expected_digest=expected_digest)


def _parse_config_bytes(
    encoded: bytes,
    *,
    path: Path,
    expected_digest: str | None = None,
) -> dict[str, Any]:
    if len(encoded) > _MAX_WORKER_CONFIG_BYTES:
        raise TorchrunWorkerAdapterError("worker config is too large")
    observed_digest = hashlib.sha256(b"lm-resiliency/worker-policy/v1\0" + encoded).hexdigest()
    if expected_digest is not None and observed_digest != expected_digest:
        raise TorchrunWorkerAdapterError("worker config changed after rendezvous policy agreement")
    try:
        payload = _toml.loads(encoded.decode("utf-8"))
    except (UnicodeError, _toml.TOMLDecodeError) as error:
        raise TorchrunWorkerAdapterError(f"failed to read worker config {path}") from error
    if not isinstance(payload, dict):
        raise TorchrunWorkerAdapterError("worker config must be a TOML table")
    unknown = set(payload) - _ROOT_CONFIG_FIELDS
    if unknown:
        raise TorchrunWorkerAdapterError(f"unknown worker config fields: {sorted(unknown)!r}")
    if "schema_version" not in payload:
        raise TorchrunWorkerAdapterError("worker config requires schema_version")
    version = payload["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise TorchrunWorkerAdapterError("worker config schema_version must be integer 1")
    adapter = payload.get("adapter")
    if adapter is not None:
        try:
            normalized = _nonempty(adapter, "adapter")
        except (TypeError, ValueError) as error:
            raise TorchrunWorkerAdapterError(
                "worker config adapter must be a non-empty string"
            ) from error
        if normalized.count(":") != 1:
            raise TorchrunWorkerAdapterError(
                "worker config adapter must use the 'module:factory' form"
            )
        payload["adapter"] = normalized
    return payload


def _validate_worker_config_bytes(encoded: bytes, *, path: Path) -> None:
    payload = _parse_config_bytes(encoded, path=path)
    _feature_options(payload, types.SimpleNamespace(run_id="policy-validation"))


def _feature_options(
    payload: Mapping[str, Any],
    context: TorchrunWorkerContext,
) -> tuple[Any | None, Any | None, dict[str, Any]]:
    from lm_resiliency.checkpointing.config import InMemoryCkptConfig
    from lm_resiliency.detection.config import ReplayHarnessConfig

    interval = _config_positive_int(payload.get("interval", 10), "interval")
    enable_checkpoint = _config_bool(payload.get("enable_checkpoint", True), "enable_checkpoint")
    enable_detection = _config_bool(payload.get("enable_detection", True), "enable_detection")
    checkpoint_values = _dataclass_options(
        InMemoryCkptConfig,
        payload.get("checkpoint", {}),
        excluded={"enable", "interval"},
        section="checkpoint",
    )
    replay_values = _dataclass_options(
        ReplayHarnessConfig,
        payload.get("replay", {}),
        excluded={"check_interval", "workload", "all_to_all_policy"},
        section="replay",
    )
    checkpoint = None
    if enable_checkpoint:
        checkpoint = InMemoryCkptConfig(**checkpoint_values)
        if checkpoint.run_id is None:
            checkpoint = replace(checkpoint, run_id=context.run_id)
    replay = None
    if enable_detection:
        replay = ReplayHarnessConfig(**replay_values)
    return (
        checkpoint,
        replay,
        {
            "interval": interval,
            "enable_checkpoint": enable_checkpoint,
            "enable_detection": enable_detection,
        },
    )


def _recovery_options(context: TorchrunWorkerContext) -> dict[str, Any]:
    if context.checkpoint_step is None:
        return {}
    return {
        "_recovery_step": context.checkpoint_step,
        "_expected_topology_id": context.topology_digest,
    }


def _dataclass_options(
    cls: type[Any],
    payload: object,
    *,
    excluded: set[str],
    section: str,
) -> dict[str, Any]:
    if not isinstance(payload, Mapping):
        raise TorchrunWorkerAdapterError(f"{section} must be a TOML table")
    annotations = get_type_hints(cls)
    allowed = {field.name for field in fields(cls)} - excluded
    unknown = set(payload) - allowed
    if unknown:
        raise TorchrunWorkerAdapterError(f"unknown {section} fields: {sorted(unknown)!r}")
    result: dict[str, Any] = {}
    for name, value in payload.items():
        result[name] = _normalize_typed_value(
            value,
            annotations[name],
            f"{section}.{name}",
        )
    return result


def _normalize_typed_value(value: Any, annotation: Any, name: str) -> Any:
    origin = get_origin(annotation)
    arguments = get_args(annotation)
    if origin in (Union, types.UnionType):
        if value is None and type(None) in arguments:
            return None
        options = [argument for argument in arguments if argument is not type(None)]
        for option in options:
            try:
                return _normalize_typed_value(value, option, name)
            except TorchrunWorkerAdapterError:
                pass
        raise TorchrunWorkerAdapterError(f"{name} has the wrong type")
    if origin is list:
        if not isinstance(value, list):
            raise TorchrunWorkerAdapterError(f"{name} must be a list")
        return [_normalize_typed_value(item, arguments[0], f"{name} item") for item in value]
    if annotation is bool:
        return _config_bool(value, name)
    if annotation is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TorchrunWorkerAdapterError(f"{name} must be an integer")
        return value
    if annotation is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TorchrunWorkerAdapterError(f"{name} must be a number")
        return float(value)
    if annotation is str:
        if not isinstance(value, str):
            raise TorchrunWorkerAdapterError(f"{name} must be a string")
        return value
    raise TorchrunWorkerAdapterError(f"{name} is not configurable in this adapter")


def _required_environment(environment: Mapping[str, str], name: str) -> str:
    value = environment.get(name)
    try:
        return _nonempty(value, name)
    except (TypeError, ValueError) as error:
        raise TorchrunWorkerAdapterError(f"{name} must be configured") from error


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TorchrunWorkerAdapterError(f"{name} must be a positive integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and value.isdecimal():
        normalized = int(value)
    else:
        raise TorchrunWorkerAdapterError(f"{name} must be a positive integer")
    if normalized < 1:
        raise TorchrunWorkerAdapterError(f"{name} must be a positive integer")
    return normalized


def _nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise TorchrunWorkerAdapterError(f"{name} must be a non-negative integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and value.isdecimal():
        normalized = int(value)
    else:
        raise TorchrunWorkerAdapterError(f"{name} must be a non-negative integer")
    return normalized


def _config_positive_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise TorchrunWorkerAdapterError(f"{name} must be a positive integer")
    return value


def _config_bool(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TorchrunWorkerAdapterError(f"{name} must be a boolean")
    return value


def _close_before_teardown(
    adapter: _SingleAttachAdapter,
    teardown: Callable[[], Any],
) -> Any:
    close_error: BaseException | None = None
    close_traceback = None
    try:
        adapter.close()
    except BaseException as error:
        close_error = error
        close_traceback = error.__traceback__
    try:
        result = teardown()
    except BaseException as teardown_error:
        if close_error is not None:
            raise close_error.with_traceback(close_traceback) from teardown_error
        raise
    if close_error is not None:
        raise close_error.with_traceback(close_traceback)
    return result


def _optimizer_types(root: type[Any]) -> tuple[type[Any], ...]:
    result: list[type[Any]] = []
    pending = list(root.__subclasses__())
    seen: set[type[Any]] = set()
    while pending:
        optimizer_type = pending.pop()
        if optimizer_type in seen:
            continue
        seen.add(optimizer_type)
        result.append(optimizer_type)
        pending.extend(optimizer_type.__subclasses__())
    return tuple(result)


def _outermost_models(models: Iterable[Any]) -> tuple[Any, ...]:
    candidates: list[Any] = []
    seen: set[int] = set()
    for model in models:
        identity = id(model)
        if identity not in seen:
            seen.add(identity)
            candidates.append(model)
    parameter_ids = {
        id(model): {id(parameter) for parameter in model.parameters()} for model in candidates
    }
    roots = []
    for candidate in candidates:
        nested = False
        candidate_parameters = parameter_ids[id(candidate)]
        for other in candidates:
            if other is candidate:
                continue
            if any(module is candidate for module in other.modules()):
                nested = True
                break
            other_parameters = parameter_ids[id(other)]
            if candidate_parameters and candidate_parameters < other_parameters:
                nested = True
                break
        if not nested:
            roots.append(candidate)
    return tuple(roots)


def _reset_worker_adapter_for_tests() -> None:
    global _installed_adapter
    with _bootstrap_lock:
        adapter = _installed_adapter
        _installed_adapter = None
    close = getattr(adapter, "close", None)
    if callable(close):
        close()


__all__ = [
    "DeepSpeedWorkerAdapter",
    "MegatronWorkerAdapter",
    "NativePyTorchAdapter",
    "NativePyTorchDDPAdapter",
    "TorchTitanWorkerAdapter",
    "TorchrunWorkerAdapter",
    "TorchrunWorkerAdapterError",
    "TorchrunWorkerContext",
    "bootstrap_worker_from_environment",
    "configure_worker_context_environment",
    "configure_worker_bootstrap_environment",
    "configure_worker_generation_environment",
    "get_torchrun_worker_context",
]
