"""Framework discovery, logical target resolution, and iteration hooks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn

from lm_resiliency.dispatch import _select_framework
from lm_resiliency.fault_injection.config import FaultTarget

_LAYER_PARENTS = {
    "block",
    "blocks",
    "decoder_layers",
    "h",
    "layer",
    "layers",
    "transformer_blocks",
}
_UNSET = object()


@dataclass(slots=True)
class TrainingContext:
    """Rank-local framework objects required by the injector."""

    framework: str
    models: tuple[nn.Module, ...]
    optimizer: Any
    step_owner: Any
    step_attribute: str
    inferred_completed_iterations: int
    is_pipeline_engine: bool = False
    step_counter_attribute: str | None = None
    _cleanups: list[Callable[[], None]] = field(default_factory=list)

    def resolve_module(self, target: FaultTarget) -> nn.Module:
        """Resolve an explicit path or framework-neutral logical component."""
        if target.model_part >= len(self.models):
            raise LookupError(
                f"model_part {target.model_part} is unavailable; "
                f"framework exposes {len(self.models)} part(s)"
            )
        model = self.models[target.model_part]
        candidates = [model]
        unwrapped = _unwrap_module(model)
        if unwrapped is not model:
            candidates.append(unwrapped)
        candidate_modules = [dict(candidate.named_modules()) for candidate in candidates]
        if target.module_path is not None:
            for modules in candidate_modules:
                if target.module_path in modules:
                    return modules[target.module_path]
        if target.component is not None:
            for modules in candidate_modules:
                resolved = _resolve_logical_module(
                    modules,
                    target=target,
                    framework=self.framework,
                    require_global_layer_metadata=(
                        self.framework in {"megatron", "torchtitan"}
                        or self.is_pipeline_engine
                        or len(self.models) > 1
                    ),
                )
                if resolved is not None:
                    return resolved
        available = sorted(candidate_modules[-1])
        sample = ", ".join(repr(name) for name in available[:10])
        selector = target.module_path or (
            f"{target.component}[{target.index}]" if target.index is not None else target.component
        )
        global_hint = ""
        if (
            target.component is not None
            and _is_layer_component(target.component)
            and (
                self.framework in {"megatron", "torchtitan"}
                or self.is_pipeline_engine
                or len(self.models) > 1
            )
        ):
            global_hint = (
                "; pipeline-sharded logical layers require global layer metadata "
                "such as layer_number or global_layer_index"
            )
        raise LookupError(
            f"target {selector!r} was not found in model_part {target.model_part}; "
            f"available paths include: {sample}{global_hint}"
        )

    def resolve_parameter(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None = None,
    ) -> torch.Tensor:
        """Resolve a parameter on the selected logical module."""
        parameter = self._resolve_model_parameter(target, parameter_name=parameter_name)
        return self._parameter_storage(parameter, target)

    def resolve_gradient_parameter(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None = None,
    ) -> torch.Tensor:
        """Resolve the model parameter whose materialized gradient is hooked."""
        return self._resolve_model_parameter(target, parameter_name=parameter_name)

    def resolve_optimizer_state(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None,
        state_key: str | None,
    ) -> torch.Tensor:
        """Resolve one optimizer-state tensor associated with the target parameter."""
        tensor, _owner_identity = self.resolve_optimizer_state_with_owner(
            target,
            parameter_name=parameter_name,
            state_key=state_key,
        )
        return tensor

    def resolve_optimizer_state_with_owner(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None,
        state_key: str | None,
    ) -> tuple[torch.Tensor, int]:
        """Resolve optimizer state and the optimizer whose load replaces it."""
        parameter = self._resolve_model_parameter(target, parameter_name=parameter_name)
        if self.framework == "deepspeed":
            resolved = _resolve_deepspeed_optimizer_state(
                self.optimizer,
                parameter,
                state_key=state_key,
            )
            if resolved is not None:
                return resolved
        for optimizer in _base_optimizers(self.optimizer):
            state = optimizer.state.get(parameter, {})
            if state_key is not None:
                value = state.get(state_key)
                if isinstance(value, torch.Tensor):
                    return value, id(optimizer)
                continue
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    return value, id(optimizer)
        suffix = "" if state_key is None else f" {state_key!r}"
        raise LookupError(f"optimizer state tensor{suffix} is unavailable")

    def _resolve_model_parameter(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None = None,
    ) -> torch.Tensor:
        module = self.resolve_module(target)
        preferred = parameter_name
        if preferred is None and target.surface.value in {"weight", "bias"}:
            preferred = target.surface.value
        if preferred is not None:
            parameter = getattr(module, preferred, None)
            if isinstance(parameter, torch.Tensor):
                return parameter
            raise LookupError(
                f"module {type(module).__name__} has no tensor parameter {preferred!r}"
            )
        for parameter in module.parameters(recurse=False):
            return parameter
        nested = sorted(module.named_parameters(recurse=True), key=lambda item: item[0])
        if nested:
            return nested[0][1]
        raise LookupError(f"module {type(module).__name__} exposes no parameters")

    def _parameter_storage(
        self,
        parameter: torch.Tensor,
        target: FaultTarget,
    ) -> torch.Tensor:
        if (
            self.framework == "deepspeed"
            and target.surface.value in {"weight", "bias"}
            and parameter.numel() == 0
        ):
            partition = getattr(parameter, "ds_tensor", None)
            if isinstance(partition, torch.Tensor) and partition.numel() > 0:
                return partition
        return parameter

    def register_step_callback(self, callback: Callable[[], None]) -> None:
        """Run a callback after every completed framework optimizer boundary."""
        if self.framework == "deepspeed":
            if self.is_pipeline_engine:
                self._register_deepspeed_pipeline_callback(callback)
            else:
                self._register_deepspeed_step_callback(callback)
            return

        register = getattr(self.optimizer, "register_step_post_hook", None)
        if callable(register):
            handle = register(lambda *_args, **_kwargs: callback())
            self._cleanups.append(handle.remove)
            return

        owner = self.step_owner
        attribute = self.step_attribute
        original = getattr(owner, attribute, None)
        if not callable(original):
            raise TypeError(f"{self.framework} optimizer boundary {attribute!r} is not callable")

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            result = original(*args, **kwargs)
            callback()
            return result

        setattr(owner, attribute, wrapped)

        def restore() -> None:
            if getattr(owner, attribute, None) is wrapped:
                setattr(owner, attribute, original)

        self._cleanups.append(restore)

    def register_state_replacement_callback(
        self,
        callback: Callable[[str, frozenset[int]], None],
    ) -> None:
        """Observe model or optimizer state loads that replace active fault state."""
        seen: set[int] = set()
        for model in self.models:
            for candidate in (model, _unwrap_module(model)):
                if id(candidate) in seen:
                    continue
                seen.add(id(candidate))
                register_pre = getattr(candidate, "register_load_state_dict_pre_hook", None)
                register_post = getattr(candidate, "register_load_state_dict_post_hook", None)
                if callable(register_pre) and callable(register_post):
                    pending: dict[str, frozenset[int]] = {"identities": frozenset()}

                    def capture_loaded_parameters(
                        module: nn.Module,
                        state_dict: dict[str, Any],
                        prefix: str,
                        *_args: Any,
                        _pending: dict[str, frozenset[int]] = pending,
                        **_kwargs: Any,
                    ) -> None:
                        _pending["identities"] = frozenset(
                            id(parameter)
                            for name, parameter in module.named_parameters(recurse=True)
                            if f"{prefix}{name}" in state_dict
                        )

                    def report_loaded_parameters(
                        *_args: Any,
                        _pending: dict[str, frozenset[int]] = pending,
                        **_kwargs: Any,
                    ) -> None:
                        callback("model", _pending["identities"])
                        _pending["identities"] = frozenset()

                    pre_handle = register_pre(capture_loaded_parameters)
                    post_handle = register_post(report_loaded_parameters)
                    self._cleanups.extend((post_handle.remove, pre_handle.remove))
        for optimizer in _base_optimizers(self.optimizer):
            if id(optimizer) in seen:
                continue
            seen.add(id(optimizer))
            register = getattr(optimizer, "register_load_state_dict_post_hook", None)
            if callable(register):
                handle = register(
                    lambda *_args, _identity=id(optimizer), **_kwargs: callback(
                        "optimizer",
                        frozenset({_identity}),
                    )
                )
                self._cleanups.append(handle.remove)

    def _register_deepspeed_step_callback(self, callback: Callable[[], None]) -> None:
        owner = self.step_owner
        attribute = self.step_attribute
        original = getattr(owner, attribute, None)
        if not callable(original):
            raise TypeError(f"deepspeed optimizer boundary {attribute!r} is not callable")
        counter_attribute = self.step_counter_attribute or "global_steps"

        def wrapped(*args: Any, **kwargs: Any) -> Any:
            before = int(getattr(owner, counter_attribute))
            result = original(*args, **kwargs)
            after = int(getattr(owner, counter_attribute))
            for _ in range(max(0, after - before)):
                callback()
            return result

        setattr(owner, attribute, wrapped)

        def restore() -> None:
            if getattr(owner, attribute, None) is wrapped:
                setattr(owner, attribute, original)

        self._cleanups.append(restore)

    def _register_deepspeed_pipeline_callback(self, callback: Callable[[], None]) -> None:
        engine = self.step_owner
        instruction_map = getattr(engine, "_INSTRUCTION_MAP", None)
        if not isinstance(instruction_map, dict):
            raise TypeError("DeepSpeed PipelineEngine exposes no instruction map")
        optimizer_instruction = next(
            (
                instruction
                for instruction in instruction_map
                if getattr(instruction, "__name__", "") == "OptimizerStep"
            ),
            None,
        )
        if optimizer_instruction is None:
            raise RuntimeError("DeepSpeed PipelineEngine has no OptimizerStep instruction")
        original_map = engine.__dict__.get("_INSTRUCTION_MAP", _UNSET)
        original_instruction = instruction_map[optimizer_instruction]
        local_map = dict(instruction_map)

        def wrapped_instruction(bound_engine: Any, *args: Any, **kwargs: Any) -> Any:
            result = original_instruction(bound_engine, *args, **kwargs)
            callback()
            return result

        local_map[optimizer_instruction] = wrapped_instruction
        engine._INSTRUCTION_MAP = local_map

        def restore() -> None:
            if getattr(engine, "_INSTRUCTION_MAP", None) is not local_map:
                return
            if original_map is _UNSET:
                engine.__dict__.pop("_INSTRUCTION_MAP", None)
            else:
                engine._INSTRUCTION_MAP = original_map

        self._cleanups.append(restore)

    def close(self) -> None:
        for cleanup in reversed(self._cleanups):
            cleanup()
        self._cleanups.clear()


def resolve_training_context(
    target: Any,
    optimizer: Any | None,
) -> TrainingContext:
    """Infer the training framework and its optimizer-step boundary."""
    selected = _select_framework(target, "auto")
    inferred = 0
    is_pipeline = False
    step_counter_attribute: str | None = None
    if selected == "pytorch":
        if optimizer is None:
            raise TypeError("PyTorch fault injection requires an optimizer")
        models = (_require_module(target, "PyTorch target"),)
        step_owner = optimizer
        step_attribute = "step"
    elif selected == "deepspeed":
        if optimizer is not None:
            raise TypeError("do not pass optimizer with a DeepSpeed engine")
        models = (_require_module(getattr(target, "module", None), "DeepSpeed module"),)
        optimizer = getattr(target, "optimizer", None)
        if optimizer is None:
            raise TypeError("DeepSpeed engine exposes no optimizer")
        is_pipeline = callable(getattr(target, "_exec_optimizer_step", None)) and any(
            cls.__name__ == "PipelineEngine" for cls in type(target).__mro__
        )
        step_owner = target
        step_attribute = "_exec_optimizer_step" if is_pipeline else "step"
        step_counter_attribute = "global_steps"
        inferred = int(getattr(target, "global_steps", 0))
    elif selected == "torchtitan":
        if optimizer is not None:
            raise TypeError("do not pass optimizer with a TorchTitan trainer")
        parts = getattr(target, "model_parts", None)
        if parts is None:
            raise TypeError("TorchTitan trainer exposes no model_parts")
        models = tuple(
            _require_module(part, f"TorchTitan model_part {index}")
            for index, part in enumerate(parts)
        )
        optimizer = getattr(target, "optimizers", None)
        if optimizer is None:
            raise TypeError("TorchTitan trainer exposes no optimizers")
        step_owner = optimizer
        step_attribute = "step"
        inferred = int(getattr(target, "step", 0))
    else:
        if optimizer is None:
            raise TypeError("Megatron fault injection requires an optimizer")
        if not isinstance(target, (list, tuple)):
            raise TypeError("Megatron fault injection requires model chunks")
        models = tuple(
            _require_module(_unwrap_module(chunk), f"Megatron model chunk {index}")
            for index, chunk in enumerate(target)
        )
        step_owner = optimizer
        step_attribute = "step"
    if not models:
        raise ValueError(f"{selected} target exposes no model parts")
    return TrainingContext(
        framework=selected,
        models=models,
        optimizer=optimizer,
        step_owner=step_owner,
        step_attribute=step_attribute,
        inferred_completed_iterations=inferred,
        is_pipeline_engine=is_pipeline,
        step_counter_attribute=step_counter_attribute,
    )


def _resolve_logical_module(
    modules: dict[str, nn.Module],
    *,
    target: FaultTarget,
    framework: str,
    require_global_layer_metadata: bool = False,
) -> nn.Module | None:
    component = target.component
    if component is None:
        return None
    index = target.index
    if component in modules and index is None:
        return modules[component]
    normalized = component.lower().replace("-", "_")
    if _is_layer_component(normalized):
        if index is None:
            raise ValueError(f"logical component {component!r} requires index")
        metadata_match = _resolve_global_layer_metadata(
            modules,
            index,
            framework=framework,
            layer_number_base=target.metadata.get("layer_number_base"),
        )
        if metadata_match is not None:
            return metadata_match
        if require_global_layer_metadata:
            return None
        for name, module in modules.items():
            pieces = name.split(".")
            if (
                len(pieces) >= 2
                and pieces[-1] == str(index)
                and pieces[-2].lower() in _LAYER_PARENTS
            ):
                return module
    if normalized in {"embedding", "token_embedding"}:
        return _resolve_embedding(modules)
    if normalized in {"output", "lm_head"}:
        return _resolve_output(modules, normalized)
    if normalized == "expert":
        if index is None:
            raise ValueError("logical component 'expert' requires index")
        return _resolve_global_expert(modules, index, target)
    return None


def _resolve_embedding(modules: dict[str, nn.Module]) -> nn.Module | None:
    candidates = [
        (name, module)
        for name, module in modules.items()
        if (
            any(token in name.lower() for token in ("embed", "tok_embeddings"))
            or name.lower().split(".")[-1] == "wte"
        )
    ]
    if not candidates:
        return None
    token_candidates = [
        (name, module)
        for name, module in candidates
        if any(
            token in name.lower()
            for token in (
                "token_embedding",
                "token_embeddings",
                "tok_embedding",
                "tok_embeddings",
                "embed_tokens",
                "word_embedding",
                "word_embeddings",
            )
        )
        or name.lower().split(".")[-1] == "wte"
    ]
    selected = token_candidates or candidates
    unique = {id(module): (name, module) for name, module in selected}
    if len(unique) == 1:
        return next(iter(unique.values()))[1]
    paths = ", ".join(sorted(name for name, _module in unique.values()))
    raise ValueError(f"logical embedding target is ambiguous; specify module_path from: {paths}")


def _resolve_output(
    modules: dict[str, nn.Module],
    component: str,
) -> nn.Module | None:
    preferred = [
        (name, module)
        for name, module in modules.items()
        if name.lower().split(".")[-1] in {"lm_head", "output_layer"}
    ]
    selected = preferred
    if not selected and component == "output":
        selected = [
            (name, module)
            for name, module in modules.items()
            if name.lower().split(".")[-1] == "output"
        ]
    if not selected:
        return None
    unique = {id(module): (name, module) for name, module in selected}
    if len(unique) == 1:
        return next(iter(unique.values()))[1]
    paths = ", ".join(sorted(name for name, _module in unique.values()))
    raise ValueError(f"logical output target is ambiguous; specify module_path from: {paths}")


def _is_layer_component(component: str) -> bool:
    normalized = component.lower().replace("-", "_")
    return normalized in {"transformer_block", "transformer_layer", "layer"}


def _resolve_global_layer_metadata(
    modules: dict[str, nn.Module],
    index: int,
    *,
    framework: str,
    layer_number_base: Any,
) -> nn.Module | None:
    direct: list[nn.Module] = []
    numbered: list[tuple[nn.Module, int]] = []
    for module in modules.values():
        for attribute in ("global_layer_index", "global_layer_idx"):
            value = getattr(module, attribute, None)
            if isinstance(value, int) and value == index:
                direct.append(module)
        layer_number = getattr(module, "layer_number", None)
        if isinstance(layer_number, int):
            numbered.append((module, layer_number))
    match = _unique_module(direct, index)
    if match is not None:
        return match
    if not numbered:
        return None
    if layer_number_base is None:
        if framework != "megatron":
            return None
        base = 1
    else:
        if isinstance(layer_number_base, bool) or not isinstance(layer_number_base, int):
            raise TypeError("target metadata layer_number_base must be an integer")
        if layer_number_base not in {0, 1}:
            raise ValueError("target metadata layer_number_base must be 0 or 1")
        base = layer_number_base
    return _unique_module(
        [module for module, number in numbered if number - base == index],
        index,
    )


def _resolve_global_expert(
    modules: dict[str, nn.Module],
    index: int,
    target: FaultTarget,
) -> nn.Module | None:
    direct: list[nn.Module] = []
    local_candidates: list[tuple[int, nn.Module]] = []
    for name, module in modules.items():
        for attribute in ("global_expert_index", "global_expert_id"):
            value = getattr(module, attribute, None)
            if isinstance(value, int) and not isinstance(value, bool) and value == index:
                direct.append(module)
        pieces = name.split(".")
        if not pieces or not pieces[-1].isdigit() or "expert" not in name.lower():
            continue
        local_candidates.append((int(pieces[-1]), module))
    match = _unique_module(direct, index)
    if match is not None:
        return match
    if not local_candidates:
        return None
    expert_parallel_rank = _target_or_module_int(
        target,
        modules,
        metadata_key="expert_parallel_rank",
        attributes=("expert_parallel_rank", "expert_model_parallel_rank", "ep_rank"),
    )
    num_local_experts = _target_or_module_int(
        target,
        modules,
        metadata_key="num_local_experts",
        attributes=("num_local_experts",),
    )
    if expert_parallel_rank is None or num_local_experts is None:
        return None
    if expert_parallel_rank < 0 or num_local_experts <= 0:
        raise ValueError("expert topology metadata must be non-negative and non-empty")
    return _unique_module(
        [
            module
            for local_index, module in local_candidates
            if expert_parallel_rank * num_local_experts + local_index == index
        ],
        index,
    )


def _target_or_module_int(
    target: FaultTarget,
    modules: dict[str, nn.Module],
    *,
    metadata_key: str,
    attributes: tuple[str, ...],
) -> int | None:
    value = target.metadata.get(metadata_key)
    if value is not None:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"target metadata {metadata_key} must be an integer")
        return value
    found = {
        candidate
        for module in modules.values()
        for attribute in attributes
        if isinstance((candidate := getattr(module, attribute, None)), int)
        and not isinstance(candidate, bool)
    }
    if len(found) > 1:
        raise LookupError(f"conflicting {metadata_key} values are exposed by the model")
    return next(iter(found), None)


def _resolve_deepspeed_optimizer_state(
    optimizer: Any,
    parameter: torch.Tensor,
    *,
    state_key: str | None,
) -> tuple[torch.Tensor, int] | None:
    base_optimizers = _base_optimizers(optimizer)
    default_owner = id(base_optimizers[0]) if base_optimizers else id(optimizer)
    mapping = getattr(parameter, "_hp_mapping", None)
    if mapping is not None:
        if getattr(mapping, "optim_fragment", None) is None:
            initialize = getattr(optimizer, "_lazy_init_hp_params_optimizer_state", None)
            if callable(initialize):
                initialize()
        keys = (
            (state_key,)
            if state_key is not None
            else tuple(sorted(getattr(mapping, "get_optim_state_keys", lambda: ())()))
        )
        for key in keys:
            try:
                value = mapping.get_optim_state_fragment(key)
            except (KeyError, ValueError):
                continue
            if isinstance(value, torch.Tensor) and value.numel() > 1:
                return value, default_owner
        return None

    zero_optimizer = getattr(parameter, "_z3_optimizer", None)
    get_partition = getattr(zero_optimizer, "_get_fp32_opt_state_partition", None)
    if not callable(get_partition):
        return None
    get_param_id = getattr(zero_optimizer, "get_param_id", None)
    positions = getattr(zero_optimizer, "grad_position", None)
    flat_groups = getattr(zero_optimizer, "fp32_partitioned_groups_flat", None)
    base_optimizer = getattr(zero_optimizer, "optimizer", None)
    if not callable(get_param_id) or not isinstance(positions, dict) or flat_groups is None:
        return None
    group_idx, _offset, _numel = positions[get_param_id(parameter)]
    swappable = getattr(zero_optimizer, "_swappable_optimizer_subgroup", None)
    if callable(swappable) and swappable(group_idx):
        raise LookupError("offloaded DeepSpeed ZeRO-3 optimizer state is not supported")
    flat_parameter = flat_groups[group_idx]
    state = getattr(base_optimizer, "state", {}).get(flat_parameter, {})
    keys = (state_key,) if state_key is not None else tuple(sorted(state))
    for key in keys:
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or value.numel() <= 1:
            continue
        partition, _ = get_partition(
            parameter,
            release_swap_buffers=False,
            optim_state_key=key,
        )
        if isinstance(partition, torch.Tensor) and partition.numel() > 1:
            return partition, id(base_optimizer)
    return None


def _unique_module(candidates: list[nn.Module], index: int) -> nn.Module | None:
    unique = {id(module): module for module in candidates}
    if len(unique) > 1:
        raise LookupError(f"global layer index {index} resolves to multiple modules")
    return next(iter(unique.values()), None)


def _base_optimizers(value: Any) -> tuple[torch.optim.Optimizer, ...]:
    found: list[torch.optim.Optimizer] = []
    seen: set[int] = set()

    def visit(candidate: Any) -> None:
        if candidate is None or id(candidate) in seen:
            return
        seen.add(id(candidate))
        if isinstance(candidate, torch.optim.Optimizer):
            found.append(candidate)
            return
        for attribute in (
            "optimizers",
            "chained_optimizers",
        ):
            children = getattr(candidate, attribute, None)
            if children is not None and not isinstance(children, (str, bytes)):
                try:
                    for child in children:
                        visit(child)
                except TypeError:
                    pass
        for attribute in ("optimizer", "_inner", "optim"):
            visit(getattr(candidate, attribute, None))

    visit(value)
    return tuple(found)


def _require_module(value: Any, label: str) -> nn.Module:
    if not isinstance(value, nn.Module):
        raise TypeError(f"{label} must be a torch.nn.Module")
    return value


def _unwrap_module(module: nn.Module) -> nn.Module:
    current = module
    seen = {id(current)}
    for _ in range(4):
        child = getattr(current, "module", None)
        if not isinstance(child, nn.Module) or id(child) in seen:
            break
        current = child
        seen.add(id(current))
    return current


__all__ = ["TrainingContext", "resolve_training_context"]
