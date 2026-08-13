"""Safe in-process fault execution for model and optimizer surfaces."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.fault_injection.config import (
    CorruptionOperation,
    FailureType,
    FaultMagnitude,
    FaultScope,
    FaultSpec,
    FaultSurface,
)
from lm_resiliency.fault_injection.executors import FaultExecutionRequest
from lm_resiliency.fault_injection.frameworks import TrainingContext
from lm_resiliency.fault_injection.reports import (
    FaultInjectionRecord,
    InjectionStatus,
)

_LOCAL_TYPES = {
    FailureType.TENSOR_CORRUPTION,
    FailureType.STALE_STATE,
    FailureType.DROP,
    FailureType.DUPLICATE,
    FailureType.REORDER,
    FailureType.DELAY,
}
_TENSOR_SURFACES = {
    FaultSurface.INPUT,
    FaultSurface.OUTPUT,
    FaultSurface.WEIGHT,
    FaultSurface.BIAS,
    FaultSurface.GRADIENT,
    FaultSurface.OPTIMIZER_STATE,
}
_FLOW_SURFACES = {
    FaultSurface.INPUT,
    FaultSurface.OUTPUT,
    FaultSurface.GRADIENT,
}
_MODULE_DELAY_SURFACES = {
    FaultSurface.INPUT,
    FaultSurface.OUTPUT,
    FaultSurface.COMPUTE,
}
_INTEGER_VIEW = {
    torch.float16: (torch.int16, 16),
    torch.bfloat16: (torch.int16, 16),
    torch.float32: (torch.int32, 32),
    torch.float64: (torch.int64, 64),
}
_SCALE_UP = {
    FaultMagnitude.CATASTROPHIC: 1e6,
    FaultMagnitude.LARGE: 100.0,
    FaultMagnitude.MEDIUM: 10.0,
    FaultMagnitude.SUBTLE: 2.0,
    FaultMagnitude.NEAR_INVISIBLE: 1.0001,
}
_NOISE_STD = {
    FaultMagnitude.CATASTROPHIC: 1e6,
    FaultMagnitude.LARGE: 1e2,
    FaultMagnitude.MEDIUM: 1.0,
    FaultMagnitude.SUBTLE: 1e-3,
    FaultMagnitude.NEAR_INVISIBLE: 1e-7,
}


@dataclass(slots=True)
class _History:
    previous: torch.Tensor | None = None
    latest: torch.Tensor | None = None

    def observe(self, tensor: torch.Tensor) -> None:
        self.previous = self.latest
        self.latest = tensor.detach().clone()


@dataclass(slots=True)
class LocalFaultEffect:
    """Live local hooks and restoration state for one fault action."""

    record: FaultInjectionRecord
    target_key: tuple[Any, ...]
    on_done: Callable[[tuple[Any, ...]], None]
    remaining_calls: int | None
    handles: list[Any] = field(default_factory=list)
    cleanup_callbacks: list[Callable[[], None]] = field(default_factory=list)
    done: bool = False

    def verify(self, evidence: dict[str, Any]) -> None:
        if self.record.verified:
            return
        self.record.verified = True
        self.record.status = InjectionStatus.ACTIVE
        self.record.activated_at_ns = time.monotonic_ns()
        self.record.evidence = dict(evidence)

    def matched(self) -> None:
        if self.remaining_calls is None:
            return
        self.remaining_calls -= 1
        if self.remaining_calls == 0:
            self.complete()

    def complete(self, evidence: dict[str, Any] | None = None) -> None:
        if self.done:
            return
        try:
            self._cleanup()
        except Exception as error:
            self.record.status = InjectionStatus.FAILED
            self.record.error = f"fault cleanup failed: {error}"
            self.record.completed_at_ns = time.monotonic_ns()
            self.done = True
            self.on_done(self.target_key)
            raise
        if evidence:
            merged = dict(self.record.evidence)
            merged.update(evidence)
            self.record.evidence = merged
        if self.record.verified:
            self.record.status = InjectionStatus.COMPLETED
        elif self.record.status is InjectionStatus.PENDING:
            self.record.status = InjectionStatus.CANCELLED
        self.record.completed_at_ns = time.monotonic_ns()
        self.done = True
        self.on_done(self.target_key)

    def fail(self, error: Exception) -> None:
        cleanup_error: Exception | None = None
        try:
            self._cleanup()
        except Exception as caught:
            cleanup_error = caught
        self.record.status = InjectionStatus.FAILED
        self.record.error = str(error)
        if cleanup_error is not None:
            self.record.error += f"; cleanup also failed: {cleanup_error}"
        self.record.completed_at_ns = time.monotonic_ns()
        self.done = True
        self.on_done(self.target_key)

    def _cleanup(self) -> None:
        first_error: Exception | None = None
        for handle in self.handles:
            try:
                handle.remove()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self.handles.clear()
        for callback in reversed(self.cleanup_callbacks):
            try:
                callback()
            except Exception as error:
                if first_error is None:
                    first_error = error
        self.cleanup_callbacks.clear()
        if first_error is not None:
            raise first_error


class LocalFaultExecutor:
    """Built-in safe executor for model, gradient, optimizer, and delay faults."""

    name = "local"

    def __init__(self, context: TrainingContext, rank: int) -> None:
        self._context = context
        self._rank = rank
        self._history: dict[tuple[Any, ...], _History] = {}
        self._observer_handles: list[Any] = []
        self._active_keys: set[tuple[Any, ...]] = set()

    def supports(self, fault: FaultSpec) -> bool:
        surface = fault.target.surface
        if fault.type is FailureType.TENSOR_CORRUPTION:
            return surface in _TENSOR_SURFACES
        if fault.type in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
            return surface in _TENSOR_SURFACES
        if fault.type in {FailureType.DROP, FailureType.REORDER}:
            return surface in _FLOW_SURFACES
        if fault.type is FailureType.DELAY:
            return surface in _MODULE_DELAY_SURFACES
        return False

    def prepare_history(self, faults: tuple[FaultSpec, ...]) -> None:
        """Install observers required by stale and duplicate actions."""
        for fault in faults:
            if fault.target.execution_rank != self._rank:
                continue
            if fault.type not in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
                continue
            if not self.supports(fault):
                continue
            key = _target_key(fault)
            if key in self._history:
                continue
            self._history[key] = _History()
            self._install_history_observer(fault, key)

    def refresh_state_history(self, faults: tuple[FaultSpec, ...]) -> None:
        """Capture state surfaces after a completed optimizer iteration."""
        for fault in faults:
            if fault.target.execution_rank != self._rank:
                continue
            if fault.type not in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
                continue
            if fault.target.surface not in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.OPTIMIZER_STATE,
            }:
                continue
            key = _target_key(fault)
            history = self._history.setdefault(key, _History())
            try:
                tensor = self._state_tensor(fault)
            except LookupError:
                continue
            history.observe(tensor)

    def activate(
        self,
        request: FaultExecutionRequest,
        record: FaultInjectionRecord,
    ) -> LocalFaultEffect:
        fault = request.fault
        if not self.supports(fault):
            raise ValueError(
                f"local executor does not support {fault.type.value} on "
                f"{fault.target.surface.value}"
            )
        key = _target_key(fault)
        if key in self._active_keys:
            raise RuntimeError("another fault is already active on the same target")
        self._active_keys.add(key)
        effect = LocalFaultEffect(
            record=record,
            target_key=key,
            on_done=self._active_keys.discard,
            remaining_calls=request.lifetime.matching_calls,
        )
        try:
            if fault.type is FailureType.DELAY:
                self._activate_delay(request, effect)
            elif fault.target.surface in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.OPTIMIZER_STATE,
            }:
                self._activate_state(request, effect)
            elif fault.target.surface in {FaultSurface.INPUT, FaultSurface.OUTPUT}:
                self._activate_module_value(request, effect)
            elif fault.target.surface is FaultSurface.GRADIENT:
                self._activate_gradient(request, effect)
            else:
                raise ValueError(f"unsupported local target surface {fault.target.surface.value}")
        except Exception as error:
            effect.fail(error)
            raise
        return effect

    def close(self) -> None:
        for handle in self._observer_handles:
            handle.remove()
        self._observer_handles.clear()
        self._history.clear()
        self._active_keys.clear()

    def _activate_delay(
        self,
        request: FaultExecutionRequest,
        effect: LocalFaultEffect,
    ) -> None:
        module = self._context.resolve_module(request.fault.target)
        delay_ms = float(request.fault.parameters["delay_ms"])

        def delay(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            try:
                time.sleep(delay_ms / 1000.0)
                effect.verify({"delay_ms": delay_ms})
                effect.matched()
            except Exception as error:
                if not effect.done:
                    effect.fail(error)
                raise
            return None

        effect.handles.append(module.register_forward_pre_hook(delay, with_kwargs=True))

    def _activate_state(
        self,
        request: FaultExecutionRequest,
        effect: LocalFaultEffect,
    ) -> None:
        fault = request.fault
        tensor = self._state_tensor(fault)
        if request.lifetime.matching_calls is None:
            restoration, affected = self._mutate_state_tensor(tensor, request)
            effect.cleanup_callbacks.append(restoration)
            effect.verify({"affected_elements": affected})
            return

        module = self._context.resolve_module(fault.target)
        restoration: Callable[[], None] | None = None

        def inject(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            nonlocal restoration
            try:
                if restoration is None:
                    restoration, affected = self._mutate_state_tensor(tensor, request)
                    effect.cleanup_callbacks.append(restoration)
                    effect.verify({"affected_elements": affected})
            except Exception as error:
                if not effect.done:
                    effect.fail(error)
                raise
            return None

        def count(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            effect.matched()
            return None

        effect.handles.append(module.register_forward_pre_hook(inject, with_kwargs=True))
        effect.handles.append(
            module.register_forward_hook(count, with_kwargs=True, always_call=True)
        )

    def _activate_module_value(
        self,
        request: FaultExecutionRequest,
        effect: LocalFaultEffect,
    ) -> None:
        fault = request.fault
        module = self._context.resolve_module(fault.target)
        history = self._history.get(_target_key(fault))
        if fault.target.surface is FaultSurface.INPUT:

            def transform_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                try:
                    transformed, affected = _transform_tree(
                        (args, kwargs),
                        request,
                        history,
                    )
                    effect.verify({"affected_elements": affected})
                    effect.matched()
                    new_args, new_kwargs = transformed
                    return tuple(new_args), dict(new_kwargs)
                except Exception as error:
                    if not effect.done:
                        effect.fail(error)
                    raise

            effect.handles.append(
                module.register_forward_pre_hook(transform_input, with_kwargs=True)
            )
            return

        def transform_output(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
        ) -> Any:
            try:
                transformed, affected = _transform_tree(output, request, history)
                effect.verify({"affected_elements": affected})
                effect.matched()
                return transformed
            except Exception as error:
                if not effect.done:
                    effect.fail(error)
                raise

        effect.handles.append(
            module.register_forward_hook(
                transform_output,
                with_kwargs=True,
                always_call=True,
            )
        )

    def _activate_gradient(
        self,
        request: FaultExecutionRequest,
        effect: LocalFaultEffect,
    ) -> None:
        fault = request.fault
        parameter = self._context.resolve_parameter(
            fault.target,
            parameter_name=_parameter_name(fault),
        )
        history = self._history.get(_target_key(fault))

        def transform_gradient(gradient: torch.Tensor) -> torch.Tensor:
            try:
                transformed, affected = _transform_tensor(gradient, request, history)
                effect.verify({"affected_elements": affected})
                effect.matched()
                return transformed
            except Exception as error:
                if not effect.done:
                    effect.fail(error)
                raise

        effect.handles.append(parameter.register_hook(transform_gradient))

    def _state_tensor(self, fault: FaultSpec) -> torch.Tensor:
        if fault.target.surface is FaultSurface.OPTIMIZER_STATE:
            return self._context.resolve_optimizer_state(
                fault.target,
                parameter_name=_parameter_name(fault),
                state_key=(
                    None
                    if fault.parameters.get("state_key") is None
                    else str(fault.parameters["state_key"])
                ),
            )
        return self._context.resolve_parameter(
            fault.target,
            parameter_name=_parameter_name(fault),
        )

    def _mutate_state_tensor(
        self,
        tensor: torch.Tensor,
        request: FaultExecutionRequest,
    ) -> tuple[Callable[[], None], int]:
        original = tensor.detach().clone()
        history = self._history.get(_target_key(request.fault))
        transformed, affected = _transform_tensor(tensor, request, history)
        with torch.no_grad():
            tensor.copy_(transformed)

        def restore() -> None:
            with torch.no_grad():
                tensor.copy_(original)

        return restore, affected

    def _install_history_observer(
        self,
        fault: FaultSpec,
        key: tuple[Any, ...],
    ) -> None:
        history = self._history[key]
        surface = fault.target.surface
        if surface in {
            FaultSurface.WEIGHT,
            FaultSurface.BIAS,
            FaultSurface.OPTIMIZER_STATE,
        }:
            return
        module = self._context.resolve_module(fault.target)
        if surface is FaultSurface.INPUT:

            def observe_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> None:
                tensor = _first_float_tensor((args, kwargs))
                if tensor is not None:
                    history.observe(tensor)
                return None

            self._observer_handles.append(
                module.register_forward_pre_hook(observe_input, with_kwargs=True)
            )
            return
        if surface is FaultSurface.OUTPUT:

            def observe_output(
                _module: nn.Module,
                _args: tuple[Any, ...],
                _kwargs: dict[str, Any],
                output: Any,
            ) -> None:
                tensor = _first_float_tensor(output)
                if tensor is not None:
                    history.observe(tensor)
                return None

            self._observer_handles.append(
                module.register_forward_hook(
                    observe_output,
                    with_kwargs=True,
                    always_call=True,
                )
            )
            return
        if surface is FaultSurface.GRADIENT:
            parameter = self._context.resolve_parameter(
                fault.target,
                parameter_name=_parameter_name(fault),
            )

            def observe_gradient(gradient: torch.Tensor) -> None:
                history.observe(gradient)
                return None

            self._observer_handles.append(parameter.register_hook(observe_gradient))


def _transform_tree(
    value: Any,
    request: FaultExecutionRequest,
    history: _History | None,
) -> tuple[Any, int]:
    leaves, spec = tree_flatten(value)
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or not leaf.is_floating_point():
            continue
        transformed, affected = _transform_tensor(leaf, request, history)
        leaves[index] = transformed
        return tree_unflatten(leaves, spec), affected
    raise TypeError("target value contains no floating-point tensor")


def _transform_tensor(
    tensor: torch.Tensor,
    request: FaultExecutionRequest,
    history: _History | None,
) -> tuple[torch.Tensor, int]:
    if tensor.numel() == 0:
        raise ValueError("fault target tensor must be non-empty")
    if not tensor.is_floating_point():
        raise TypeError("fault target tensor must be floating point")
    transformed = tensor.clone().contiguous()
    fault = request.fault
    scope = FaultScope(fault.parameters.get("scope", FaultScope.SINGLE.value))
    indices = _target_indices(transformed, scope)
    before = transformed.detach().view(-1).index_select(0, indices).clone()
    if fault.type is FailureType.TENSOR_CORRUPTION:
        with torch.no_grad():
            _apply_corruption(transformed, indices, request)
    elif fault.type in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
        if history is None or history.previous is None:
            raise RuntimeError("stale or duplicate injection has no prior observed value")
        previous = history.previous.to(device=transformed.device, dtype=transformed.dtype)
        if previous.shape != transformed.shape:
            raise RuntimeError("prior observed value shape does not match the target")
        with torch.no_grad():
            transformed.view(-1).index_copy_(
                0,
                indices,
                previous.contiguous().view(-1).index_select(0, indices),
            )
    elif fault.type is FailureType.DROP:
        with torch.no_grad():
            transformed.view(-1).index_fill_(0, indices, 0.0)
    elif fault.type is FailureType.REORDER:
        if transformed.shape[0] < 2:
            raise RuntimeError("reorder requires a leading dimension of at least two")
        transformed = torch.flip(transformed, dims=(0,))
        indices = torch.arange(
            transformed.numel(),
            device=transformed.device,
            dtype=torch.long,
        )
        before = tensor.detach().contiguous().view(-1).clone()
    else:
        raise ValueError(f"unsupported tensor fault type {fault.type.value!r}")
    after = transformed.detach().view(-1).index_select(0, indices)
    if _same_tensor_values(before, after):
        raise RuntimeError("fault injection did not change the selected tensor values")
    return transformed, int(indices.numel())


def _apply_corruption(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    request: FaultExecutionRequest,
) -> None:
    fault = request.fault
    operation = CorruptionOperation(fault.parameters["operation"])
    magnitude = FaultMagnitude(fault.parameters.get("magnitude", FaultMagnitude.MEDIUM.value))
    flat = tensor.view(-1)
    selected = flat.index_select(0, indices)
    if operation is CorruptionOperation.SINGLE_BITFLIP:
        _flip_bits(tensor, indices, magnitude, count=1)
    elif operation is CorruptionOperation.MULTI_BITFLIP:
        _flip_bits(tensor, indices, magnitude, count=4)
    elif operation is CorruptionOperation.SET_VALUE:
        value = _numeric_value(fault.parameters["value"])
        flat.index_fill_(0, indices, value)
    elif operation is CorruptionOperation.SCALE:
        factor = float(fault.parameters.get("factor", _SCALE_UP[magnitude]))
        flat.index_copy_(0, indices, selected * factor)
    elif operation is CorruptionOperation.NOISE:
        standard_deviation = float(fault.parameters.get("std", _NOISE_STD[magnitude]))
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(request.seed)
        noise = torch.randn(
            selected.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )
        flat.index_copy_(0, indices, selected + noise * standard_deviation)
    else:
        flat.index_copy_(0, indices, -selected)


def _flip_bits(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    magnitude: FaultMagnitude,
    *,
    count: int,
) -> None:
    try:
        integer_dtype, width = _INTEGER_VIEW[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"bit flips do not support dtype {tensor.dtype}") from error
    base = {
        FaultMagnitude.NEAR_INVISIBLE: 0,
        FaultMagnitude.SUBTLE: width // 4,
        FaultMagnitude.MEDIUM: width // 2,
        FaultMagnitude.LARGE: (3 * width) // 4,
        FaultMagnitude.CATASTROPHIC: width - 2,
    }[magnitude]
    start = min(base, width - 1 - count)
    mask = sum(1 << bit for bit in range(start, start + count))
    integer_flat = tensor.view(integer_dtype).view(-1)
    selected = integer_flat.index_select(0, indices)
    integer_flat.index_copy_(0, indices, torch.bitwise_xor(selected, mask))


def _target_indices(tensor: torch.Tensor, scope: FaultScope) -> torch.Tensor:
    numel = tensor.numel()
    device = tensor.device
    if scope is FaultScope.SINGLE:
        return torch.tensor([numel // 2], device=device, dtype=torch.long)
    if scope is FaultScope.ROW:
        row_size = tensor.shape[-1] if tensor.ndim >= 2 else min(256, numel)
        return torch.arange(row_size, device=device, dtype=torch.long)
    if scope is FaultScope.PERCENT_1:
        count = max(1, numel // 100)
    elif scope is FaultScope.PERCENT_10:
        count = max(1, numel // 10)
    else:
        count = numel
    if count == numel:
        return torch.arange(numel, device=device, dtype=torch.long)
    step = max(1, numel // count)
    return torch.arange(0, numel, step, device=device, dtype=torch.long)[:count]


def _first_float_tensor(value: Any) -> torch.Tensor | None:
    leaves, _ = tree_flatten(value)
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point() and leaf.numel():
            return leaf
    return None


def _target_key(fault: FaultSpec) -> tuple[Any, ...]:
    target = fault.target
    return (
        target.execution_rank,
        target.model_part,
        target.component,
        target.index,
        target.module_path,
        target.surface.value,
        _parameter_name(fault),
        fault.parameters.get("state_key"),
    )


def _parameter_name(fault: FaultSpec) -> str | None:
    value = fault.parameters.get("parameter")
    return None if value is None else str(value)


def _numeric_value(value: Any) -> float:
    if isinstance(value, str):
        normalized = value.lower()
        if normalized == "nan":
            return float("nan")
        if normalized in {"inf", "+inf", "infinity", "+infinity"}:
            return float("inf")
        if normalized in {"-inf", "-infinity"}:
            return float("-inf")
    number = float(value)
    if math.isnan(number):
        return float("nan")
    return number


def _same_tensor_values(left: torch.Tensor, right: torch.Tensor) -> bool:
    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    equal = left == right
    if left.is_floating_point():
        equal = equal | (torch.isnan(left) & torch.isnan(right))
    return bool(torch.all(equal).item())


__all__ = ["LocalFaultEffect", "LocalFaultExecutor"]
