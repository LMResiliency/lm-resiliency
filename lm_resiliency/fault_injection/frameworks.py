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


@dataclass(slots=True)
class TrainingContext:
    """Rank-local framework objects required by the injector."""

    framework: str
    models: tuple[nn.Module, ...]
    optimizer: Any
    step_owner: Any
    step_attribute: str
    inferred_completed_iterations: int
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
                    component=target.component,
                    index=target.index,
                )
                if resolved is not None:
                    return resolved
        available = sorted(candidate_modules[-1])
        sample = ", ".join(repr(name) for name in available[:10])
        selector = target.module_path or (
            f"{target.component}[{target.index}]" if target.index is not None else target.component
        )
        raise LookupError(
            f"target {selector!r} was not found in model_part {target.model_part}; "
            f"available paths include: {sample}"
        )

    def resolve_parameter(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None = None,
    ) -> torch.Tensor:
        """Resolve a parameter on the selected logical module."""
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
        raise LookupError(f"module {type(module).__name__} exposes no direct parameters")

    def resolve_optimizer_state(
        self,
        target: FaultTarget,
        *,
        parameter_name: str | None,
        state_key: str | None,
    ) -> torch.Tensor:
        """Resolve one optimizer-state tensor associated with the target parameter."""
        parameter = self.resolve_parameter(target, parameter_name=parameter_name)
        for optimizer in _base_optimizers(self.optimizer):
            state = optimizer.state.get(parameter, {})
            if state_key is not None:
                value = state.get(state_key)
                if isinstance(value, torch.Tensor):
                    return value
                continue
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor) and value.numel() > 1:
                    return value
        suffix = "" if state_key is None else f" {state_key!r}"
        raise LookupError(f"optimizer state tensor{suffix} is unavailable")

    def register_step_callback(self, callback: Callable[[], None]) -> None:
        """Run a callback after every completed framework optimizer boundary."""
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
    )


def _resolve_logical_module(
    modules: dict[str, nn.Module],
    *,
    component: str,
    index: int | None,
) -> nn.Module | None:
    if component in modules and index is None:
        return modules[component]
    normalized = component.lower().replace("-", "_")
    if normalized in {"transformer_block", "transformer_layer", "layer"}:
        if index is None:
            raise ValueError(f"logical component {component!r} requires index")
        for name, module in modules.items():
            pieces = name.split(".")
            if (
                len(pieces) >= 2
                and pieces[-1] == str(index)
                and pieces[-2].lower() in _LAYER_PARENTS
            ):
                return module
    if normalized in {"embedding", "token_embedding"}:
        for name, module in modules.items():
            if any(token in name.lower() for token in ("embed", "tok_embeddings")):
                return module
    if normalized in {"output", "lm_head"}:
        for name, module in modules.items():
            if name.lower().endswith(("lm_head", "output_layer", "output")):
                return module
    if normalized == "expert":
        if index is None:
            raise ValueError("logical component 'expert' requires index")
        for name, module in modules.items():
            pieces = name.split(".")
            if pieces[-1:] == [str(index)] and "expert" in name.lower():
                return module
    return None


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
