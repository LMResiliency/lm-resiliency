"""Training-framework model discovery for fault injection."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from lm_resiliency.dispatch import _select_framework
from lm_resiliency.fault_injection.config import FaultTarget


@dataclass(frozen=True, slots=True)
class ResolvedTrainingModels:
    """Rank-local model parts exposed by one training framework."""

    framework: str
    models: tuple[nn.Module, ...]

    def resolve_module(self, target: FaultTarget) -> nn.Module:
        """Resolve a campaign model-part and module path."""
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
        for candidate in candidates:
            modules = dict(candidate.named_modules())
            if target.module in modules:
                return modules[target.module]
        available = sorted(dict(candidates[-1].named_modules()))
        sample = ", ".join(repr(name) for name in available[:8])
        raise LookupError(
            f"module {target.module!r} was not found in model_part "
            f"{target.model_part}; available paths include: {sample}"
        )


def resolve_training_models(target: Any, framework: str) -> ResolvedTrainingModels:
    """Resolve initialized framework objects to rank-local model parts."""
    selected = _select_framework(target, framework)
    if selected == "pytorch":
        models = (_require_module(target, "PyTorch target"),)
    elif selected == "deepspeed":
        models = (_require_module(getattr(target, "module", None), "DeepSpeed module"),)
    elif selected == "torchtitan":
        parts = getattr(target, "model_parts", None)
        if parts is None:
            models = (_require_module(target, "TorchTitan target"),)
        else:
            models = tuple(
                _require_module(part, f"TorchTitan model_part {index}")
                for index, part in enumerate(parts)
            )
    else:
        if not isinstance(target, (list, tuple)):
            raise TypeError("Megatron fault injection requires model chunks in a list or tuple")
        models = tuple(
            _require_module(_unwrap_module(chunk), f"Megatron model chunk {index}")
            for index, chunk in enumerate(target)
        )
    if not models:
        raise ValueError(f"{selected} target exposes no model parts")
    return ResolvedTrainingModels(selected, models)


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


__all__ = ["ResolvedTrainingModels", "resolve_training_models"]
