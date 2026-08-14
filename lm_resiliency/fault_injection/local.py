"""Safe in-process fault execution for model and optimizer surfaces."""

from __future__ import annotations

import math
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.fault_injection.config import (
    CorruptionOperation,
    FailureType,
    FaultIncident,
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


class _NoOpFaultError(RuntimeError):
    """The selected target already has the requested faulty value."""


class _UnavailableHistoryError(RuntimeError):
    """A stale or duplicate fault has no earlier observed value."""


class _UnsupportedTargetTensorError(TypeError):
    """A runtime hook value exposes no supported tensor to transform."""


@dataclass(slots=True)
class _History:
    previous: torch.Tensor | None = None
    latest: torch.Tensor | None = None
    previous_shape: torch.Size | None = None
    latest_shape: torch.Size | None = None
    observation_error: str | None = None

    def observe(
        self,
        tensor: torch.Tensor,
        indices: torch.Tensor | None = None,
    ) -> None:
        self.observation_error = None
        self.previous = self.latest
        self.previous_shape = self.latest_shape
        local = _local_shard(tensor)
        self.latest_shape = local.shape
        self.latest = local.detach().clone() if indices is None else _read_linear(local, indices)

    def reject(self, error: Exception) -> None:
        self.previous = self.latest
        self.previous_shape = self.latest_shape
        self.latest = None
        self.latest_shape = None
        self.observation_error = str(error)


@dataclass(slots=True)
class LocalFaultEffect:
    """Live local hooks and restoration state for one fault action."""

    record: FaultInjectionRecord
    target_key: tuple[Any, ...]
    on_done: Callable[[tuple[Any, ...]], None]
    remaining_calls: int | None
    handles: list[Any] = field(default_factory=list)
    cleanup_callbacks: list[Callable[[bool, bool], None]] = field(default_factory=list)
    state_replaced: bool = False
    replacement_identity: int | None = None
    done: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)
    cancel_event: threading.Event = field(default_factory=threading.Event, repr=False)

    def verify(self, evidence: dict[str, Any]) -> None:
        with self.lock:
            if self.done or self.record.verified:
                return
            with self.record._lock:
                self.record.verified = True
                self.record.status = InjectionStatus.ACTIVE
                self.record.activated_at_ns = time.monotonic_ns()
                self.record.evidence = dict(evidence)

    def matched(self) -> None:
        with self.lock:
            if self.done or self.remaining_calls is None:
                return
            self.remaining_calls -= 1
            if self.remaining_calls == 0:
                self.complete()

    def complete(
        self,
        evidence: dict[str, Any] | None = None,
        *,
        cancelled: bool = False,
        preserve_replaced_state: bool = False,
    ) -> None:
        self.cancel_event.set()
        with self.lock:
            if self.done:
                return
            try:
                self._cleanup(
                    preserve_replaced_state=preserve_replaced_state,
                    replacement_confirmed=self.state_replaced,
                )
            except Exception as error:
                with self.record._lock:
                    self.record.status = InjectionStatus.FAILED
                    self.record.error = f"fault cleanup failed: {error}"
                    self.record.completed_at_ns = time.monotonic_ns()
                self.done = True
                self.on_done(self.target_key)
                raise
            with self.record._lock:
                if evidence:
                    merged = dict(self.record.evidence)
                    merged.update(evidence)
                    self.record.evidence = merged
                if cancelled:
                    self.record.status = InjectionStatus.CANCELLED
                elif self.record.verified:
                    self.record.status = InjectionStatus.COMPLETED
                elif self.record.status is InjectionStatus.PENDING:
                    self.record.status = InjectionStatus.CANCELLED
                self.record.completed_at_ns = time.monotonic_ns()
            self.done = True
            self.on_done(self.target_key)

    def mark_state_replaced(self) -> None:
        """Preserve values loaded by an external checkpoint or recovery path."""
        with self.lock:
            if not self.done:
                self.state_replaced = True

    def fail(
        self,
        error: BaseException,
        *,
        propagate_cleanup_error: bool = False,
    ) -> None:
        self.cancel_event.set()
        with self.lock:
            if self.done:
                return
            cleanup_error: Exception | None = None
            try:
                self._cleanup()
            except Exception as caught:
                cleanup_error = caught
            with self.record._lock:
                self.record.status = InjectionStatus.FAILED
                self.record.error = str(error)
                if cleanup_error is not None:
                    self.record.error += f"; cleanup also failed: {cleanup_error}"
                self.record.completed_at_ns = time.monotonic_ns()
            self.done = True
            self.on_done(self.target_key)
            if cleanup_error is not None and propagate_cleanup_error:
                raise RuntimeError(
                    f"fault rollback cleanup failed: {cleanup_error}"
                ) from cleanup_error

    def _cleanup(
        self,
        *,
        preserve_replaced_state: bool = False,
        replacement_confirmed: bool = False,
    ) -> None:
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
                callback(preserve_replaced_state, replacement_confirmed)
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
        self._observer_handles: dict[tuple[Any, ...], list[Any]] = {}
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

    def validate_targets(self, faults: tuple[FaultSpec, ...]) -> None:
        """Resolve local targets without installing hooks or cloning tensors."""
        for fault in faults:
            if fault.target.execution_rank != self._rank or not self.supports(fault):
                continue
            if fault.target.surface in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.GRADIENT,
                FaultSurface.OPTIMIZER_STATE,
            }:
                self._context.resolve_parameter(
                    fault.target,
                    parameter_name=_parameter_name(fault),
                )
            else:
                self._context.resolve_module(fault.target)

    def validate_schedule(self, incidents: tuple[FaultIncident, ...]) -> None:
        """Reject local effects whose possible active windows share a target."""
        scheduled: list[tuple[tuple[Any, ...], FaultIncident, FaultSpec]] = []
        for incident in incidents:
            if incident.trigger.probability <= 0.0:
                continue
            local_faults = tuple(
                fault
                for fault in incident.faults
                if fault.target.execution_rank == self._rank and self.supports(fault)
            )
            if (
                local_faults
                and incident.lifetime.matching_calls is not None
                and incident.lifetime.matching_calls > 1
                and _trigger_candidate_count(incident) > 1
            ):
                raise ValueError(
                    "local matching_calls incidents require a single trigger candidate "
                    "because matching operations may span training iterations"
                )
            for fault in local_faults:
                key = self._schedule_target_key(fault)
                for other_key, other_incident, other_fault in scheduled:
                    if other_incident is incident:
                        continue
                    if key != other_key:
                        continue
                    if not _incident_windows_may_overlap(incident, other_incident):
                        continue
                    raise ValueError(
                        "fault incidents may not overlap on the same resolved target: "
                        f"{other_incident.incident_id}/{other_fault.fault_id} and "
                        f"{incident.incident_id}/{fault.fault_id}"
                    )
                scheduled.append((key, incident, fault))

    def validate_activations(
        self,
        requests: tuple[FaultExecutionRequest, ...],
    ) -> None:
        """Validate immediately armed local effects without mutating training state."""
        seen: set[tuple[Any, ...]] = set()
        for request in requests:
            fault = request.fault
            if fault.target.execution_rank != self._rank or not self.supports(fault):
                continue
            target_key = self._resolved_target_key(fault)
            if target_key in seen or target_key in self._active_keys:
                raise RuntimeError("another fault is already active on the same target")
            seen.add(target_key)
            if fault.target.surface not in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.OPTIMIZER_STATE,
            }:
                continue
            tensor = self._state_tensor(fault)
            history = self._history.get(self._history_key(fault))
            try:
                _indices, original, transformed = _prepare_state_values(
                    tensor,
                    request,
                    history,
                )
            except _UnavailableHistoryError:
                continue
            _validate_state_retirement(original, transformed, request)

    def sync_history(self, faults: tuple[FaultSpec, ...]) -> None:
        """Collect stale-state history only around upcoming scheduled faults."""
        desired: dict[tuple[Any, ...], FaultSpec] = {}
        for fault in faults:
            if fault.target.execution_rank != self._rank:
                continue
            if fault.type not in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
                continue
            if not self.supports(fault):
                continue
            try:
                key = self._history_key(fault)
            except LookupError:
                # Optimizer state may not exist until the first optimizer step.
                # Retry at the next history boundary instead of aborting enablement.
                continue
            desired.setdefault(key, fault)

        removal_error: Exception | None = None
        for key in set(self._history) - set(desired):
            for handle in self._observer_handles.pop(key, ()):
                try:
                    handle.remove()
                except Exception as error:
                    if removal_error is None:
                        removal_error = error
            self._history.pop(key, None)
        if removal_error is not None:
            raise removal_error

        for key, fault in desired.items():
            if key in self._history:
                continue
            self._history[key] = _History()
            self._install_history_observer(fault, key)

        for key, fault in desired.items():
            if fault.target.surface not in {
                FaultSurface.WEIGHT,
                FaultSurface.BIAS,
                FaultSurface.OPTIMIZER_STATE,
            }:
                continue
            try:
                tensor = self._state_tensor(fault)
            except LookupError:
                continue
            local = _local_shard(tensor)
            scope = FaultScope(fault.parameters.get("scope", FaultScope.SINGLE.value))
            _observe_history(self._history[key], local, scope)

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
        key = self._resolved_target_key(fault)
        if key in self._active_keys:
            raise RuntimeError("another fault is already active on the same target")
        self._active_keys.add(key)
        effect = LocalFaultEffect(
            record=record,
            target_key=key,
            on_done=self._active_keys.discard,
            remaining_calls=request.lifetime.matching_calls,
            replacement_identity=self._replacement_identity(fault),
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
        except _UnavailableHistoryError as error:
            effect.fail(error)
            return effect
        except Exception as error:
            effect.fail(error)
            raise
        return effect

    def close(self) -> None:
        first_error: Exception | None = None
        for handles in self._observer_handles.values():
            for handle in handles:
                try:
                    handle.remove()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        self._observer_handles.clear()
        self._history.clear()
        self._active_keys.clear()
        if first_error is not None:
            raise first_error

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
            if effect.cancel_event.wait(delay_ms / 1000.0):
                return None
            with effect.lock:
                if effect.done or effect.cancel_event.is_set():
                    return None
                try:
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
        restoration: Callable[[bool, bool], None] | None = None

        def inject(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            nonlocal restoration
            with effect.lock:
                if effect.done:
                    return None
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
            with effect.lock:
                if not effect.done:
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
        history = self._history.get(self._history_key(fault))
        if fault.target.surface is FaultSurface.INPUT:

            def transform_input(
                _module: nn.Module,
                args: tuple[Any, ...],
                kwargs: dict[str, Any],
            ) -> tuple[tuple[Any, ...], dict[str, Any]]:
                with effect.lock:
                    if effect.done:
                        return args, kwargs
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
                        if isinstance(
                            error,
                            (
                                _NoOpFaultError,
                                _UnavailableHistoryError,
                                _UnsupportedTargetTensorError,
                            ),
                        ):
                            return args, kwargs
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
            with effect.lock:
                if effect.done:
                    return output
                try:
                    transformed, affected = _transform_tree(output, request, history)
                    effect.verify({"affected_elements": affected})
                    effect.matched()
                    return transformed
                except Exception as error:
                    if not effect.done:
                        effect.fail(error)
                    if isinstance(
                        error,
                        (
                            _NoOpFaultError,
                            _UnavailableHistoryError,
                            _UnsupportedTargetTensorError,
                        ),
                    ):
                        return output
                    raise

        effect.handles.append(
            module.register_forward_hook(
                transform_output,
                with_kwargs=True,
            )
        )

    def _activate_gradient(
        self,
        request: FaultExecutionRequest,
        effect: LocalFaultEffect,
    ) -> None:
        fault = request.fault
        parameter = self._context.resolve_gradient_parameter(
            fault.target,
            parameter_name=_parameter_name(fault),
        )
        history = self._history.get(self._history_key(fault))

        def transform_gradient(gradient: torch.Tensor) -> torch.Tensor:
            with effect.lock:
                if effect.done:
                    return gradient
                try:
                    transformed, affected = _transform_tensor(gradient, request, history)
                    effect.verify({"affected_elements": affected})
                    effect.matched()
                    return transformed
                except Exception as error:
                    if not effect.done:
                        effect.fail(error)
                    if isinstance(
                        error,
                        (
                            _NoOpFaultError,
                            _UnavailableHistoryError,
                            _UnsupportedTargetTensorError,
                        ),
                    ):
                        return gradient
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

    def _replacement_identity(self, fault: FaultSpec) -> int | None:
        parameter_name = _parameter_name(fault)
        if fault.target.surface in {FaultSurface.WEIGHT, FaultSurface.BIAS}:
            return id(
                self._context.resolve_gradient_parameter(
                    fault.target,
                    parameter_name=parameter_name,
                )
            )
        if fault.target.surface is FaultSurface.OPTIMIZER_STATE:
            _tensor, owner_identity = self._context.resolve_optimizer_state_with_owner(
                fault.target,
                parameter_name=parameter_name,
                state_key=(
                    None
                    if fault.parameters.get("state_key") is None
                    else str(fault.parameters["state_key"])
                ),
            )
            return owner_identity
        return None

    def _resolved_target_key(self, fault: FaultSpec) -> tuple[Any, ...]:
        surface = fault.target.surface
        parameter_name = _parameter_name(fault)
        if surface in {
            FaultSurface.WEIGHT,
            FaultSurface.BIAS,
            FaultSurface.GRADIENT,
        }:
            target = self._context.resolve_gradient_parameter(
                fault.target,
                parameter_name=parameter_name,
            )
        elif surface is FaultSurface.OPTIMIZER_STATE:
            parameter = self._context.resolve_gradient_parameter(
                fault.target,
                parameter_name=parameter_name,
            )
            _tensor, owner_identity, resolved_state_key = (
                self._context.resolve_optimizer_state_with_identity(
                    fault.target,
                    parameter_name=parameter_name,
                    state_key=(
                        None
                        if fault.parameters.get("state_key") is None
                        else str(fault.parameters["state_key"])
                    ),
                )
            )
            return (
                fault.target.execution_rank,
                surface.value,
                id(parameter),
                owner_identity,
                resolved_state_key,
            )
        else:
            target = self._context.resolve_module(fault.target)
        target_kind = (
            "parameter_state"
            if surface in {FaultSurface.WEIGHT, FaultSurface.BIAS}
            else surface.value
        )
        return (
            fault.target.execution_rank,
            target_kind,
            id(target),
        )

    def _schedule_target_key(self, fault: FaultSpec) -> tuple[Any, ...]:
        if fault.target.surface is not FaultSurface.OPTIMIZER_STATE:
            return self._resolved_target_key(fault)
        parameter = self._context.resolve_gradient_parameter(
            fault.target,
            parameter_name=_parameter_name(fault),
        )
        return (
            fault.target.execution_rank,
            FaultSurface.OPTIMIZER_STATE.value,
            id(parameter),
        )

    def _history_key(self, fault: FaultSpec) -> tuple[Any, ...]:
        scope = FaultScope(fault.parameters.get("scope", FaultScope.SINGLE.value))
        return (*self._resolved_target_key(fault), scope.value)

    def _mutate_state_tensor(
        self,
        tensor: torch.Tensor,
        request: FaultExecutionRequest,
    ) -> tuple[Callable[[bool, bool], None], int]:
        tensor = _local_shard(tensor)
        history = self._history.get(self._history_key(request.fault))
        changed_indices, original, transformed = _prepare_state_values(
            tensor,
            request,
            history,
        )
        _validate_state_retirement(original, transformed, request)
        retirement_delta = transformed - original
        _synchronize_state_mutation(tensor)
        with torch.no_grad():
            _write_linear(tensor, changed_indices, transformed)

        def restore(
            preserve_replaced_state: bool,
            replacement_confirmed: bool,
        ) -> None:
            if replacement_confirmed:
                return
            _synchronize_state_mutation(tensor)
            with torch.no_grad():
                current = _read_linear(tensor, changed_indices)
                finite = torch.isfinite(retirement_delta)
                already_restored = _elementwise_same(current, original)
                still_injected = _elementwise_same(current, transformed)
                preserve = (
                    ~still_injected if preserve_replaced_state else torch.zeros_like(still_injected)
                )
                restore_exact = still_injected & ~already_restored & ~preserve
                if bool(torch.any(restore_exact).item()):
                    current[restore_exact] = original[restore_exact]
                retire_finite = finite & ~already_restored & ~still_injected & ~preserve
                if bool(torch.any(retire_finite).item()):
                    current[retire_finite] -= retirement_delta[retire_finite]
                restore_nonfinite = ~finite & ~already_restored & ~still_injected & ~preserve
                if bool(torch.any(restore_nonfinite).item()):
                    current[restore_nonfinite] = original[restore_nonfinite]
                _write_linear(tensor, changed_indices, current)

        return restore, int(changed_indices.numel())

    def _install_history_observer(
        self,
        fault: FaultSpec,
        key: tuple[Any, ...],
    ) -> None:
        history = self._history[key]
        surface = fault.target.surface
        scope = FaultScope(fault.parameters.get("scope", FaultScope.SINGLE.value))
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
                    _observe_history(history, tensor, scope)
                return None

            self._observer_handles.setdefault(key, []).append(
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
                    _observe_history(history, tensor, scope)
                return None

            self._observer_handles.setdefault(key, []).append(
                module.register_forward_hook(
                    observe_output,
                    with_kwargs=True,
                    always_call=True,
                )
            )
            return
        if surface is FaultSurface.GRADIENT:
            parameter = self._context.resolve_gradient_parameter(
                fault.target,
                parameter_name=_parameter_name(fault),
            )

            def observe_gradient(gradient: torch.Tensor) -> None:
                _observe_history(history, gradient, scope)
                return None

            self._observer_handles.setdefault(key, []).append(
                parameter.register_hook(observe_gradient)
            )


def _transform_tree(
    value: Any,
    request: FaultExecutionRequest,
    history: _History | None,
) -> tuple[Any, int]:
    leaves, spec = tree_flatten(value)
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or not leaf.is_floating_point() or leaf.numel() == 0:
            continue
        transformed, affected = _transform_tensor(leaf, request, history)
        leaves[index] = transformed
        return tree_unflatten(leaves, spec), affected
    raise _UnsupportedTargetTensorError("target value contains no non-empty floating-point tensor")


def _prepare_state_values(
    tensor: torch.Tensor,
    request: FaultExecutionRequest,
    history: _History | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    tensor = _local_shard(tensor)
    if tensor.numel() == 0:
        raise ValueError("fault target tensor must be non-empty")
    if not tensor.is_floating_point():
        raise TypeError("fault target tensor must be floating point")
    scope = FaultScope(request.fault.parameters.get("scope", FaultScope.SINGLE.value))
    indices = _target_indices(tensor, scope)
    original = _read_linear(tensor, indices)
    transformed = original.clone()
    if request.fault.type is FailureType.TENSOR_CORRUPTION:
        local_indices = torch.arange(
            transformed.numel(),
            device=transformed.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            _apply_corruption(transformed, local_indices, request)
    elif request.fault.type in {FailureType.STALE_STATE, FailureType.DUPLICATE}:
        if history is not None and history.observation_error is not None:
            raise _UnsupportedTargetTensorError(
                f"history observation failed: {history.observation_error}"
            )
        if history is None or history.previous is None:
            raise _UnavailableHistoryError(
                "stale or duplicate injection has no prior observed value"
            )
        if history.previous_shape != tensor.shape:
            raise RuntimeError("prior observed value shape does not match the target")
        previous = _local_shard(history.previous).to(
            device=tensor.device,
            dtype=tensor.dtype,
        )
        if previous.shape == tensor.shape:
            transformed = _read_linear(previous, indices)
        elif previous.shape == original.shape:
            transformed = previous.detach().clone()
        else:
            raise RuntimeError("prior observed value shape does not match the target")
    else:
        raise ValueError(f"unsupported state fault type {request.fault.type.value!r}")
    if _same_tensor_values(original, transformed):
        raise _NoOpFaultError("fault injection did not change the selected tensor values")
    return indices, original, transformed


def _transform_tensor(
    tensor: torch.Tensor,
    request: FaultExecutionRequest,
    history: _History | None,
) -> tuple[torch.Tensor, int]:
    reference = tensor
    tensor = _local_shard(tensor)
    if tensor.numel() == 0:
        raise _UnsupportedTargetTensorError("fault target tensor must be non-empty")
    if tensor.layout is not torch.strided:
        raise _UnsupportedTargetTensorError(
            f"fault target tensor layout {tensor.layout} is not supported"
        )
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
        if history is not None and history.observation_error is not None:
            raise _UnsupportedTargetTensorError(
                f"history observation failed: {history.observation_error}"
            )
        if history is None or history.previous is None:
            raise _UnavailableHistoryError(
                "stale or duplicate injection has no prior observed value"
            )
        if history.previous_shape != transformed.shape:
            raise _UnsupportedTargetTensorError(
                "prior observed value shape does not match the target"
            )
        previous = history.previous.to(device=transformed.device, dtype=transformed.dtype)
        if previous.numel() != indices.numel():
            raise _UnsupportedTargetTensorError(
                "prior observed value scope does not match the target"
            )
        with torch.no_grad():
            transformed.view(-1).index_copy_(
                0,
                indices,
                previous.contiguous().view(-1),
            )
    elif fault.type is FailureType.DROP:
        with torch.no_grad():
            transformed.view(-1).index_fill_(0, indices, 0.0)
    elif fault.type is FailureType.REORDER:
        if transformed.ndim == 0 or transformed.shape[0] < 2:
            raise _UnsupportedTargetTensorError(
                "reorder requires a leading dimension of at least two"
            )
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
        raise _NoOpFaultError("fault injection did not change the selected tensor values")
    return _rewrap_local(reference, transformed), int(indices.numel())


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
        if math.isfinite(value):
            limits = torch.finfo(tensor.dtype)
            if value < limits.min or value > limits.max:
                raise _UnsupportedTargetTensorError(
                    f"set_value {value} is outside dtype {tensor.dtype} range"
                )
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
        raise _UnsupportedTargetTensorError(
            f"bit flips do not support dtype {tensor.dtype}"
        ) from error
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


def _observe_history(
    history: _History,
    tensor: torch.Tensor,
    scope: FaultScope,
) -> None:
    try:
        local = _local_shard(tensor)
        if local.numel() == 0:
            raise _UnsupportedTargetTensorError("fault target tensor must be non-empty")
        if local.layout is not torch.strided:
            raise _UnsupportedTargetTensorError(
                f"fault target tensor layout {local.layout} is not supported"
            )
        history.observe(local, _target_indices(local, scope))
    except _UnsupportedTargetTensorError as error:
        history.reject(error)


def _linear_coordinates(
    indices: torch.Tensor,
    shape: torch.Size,
) -> tuple[torch.Tensor, ...]:
    if len(shape) == 0:
        return ()
    remaining = indices
    coordinates: list[torch.Tensor] = []
    for dimension in reversed(shape):
        coordinates.append(torch.remainder(remaining, dimension))
        remaining = torch.div(remaining, dimension, rounding_mode="floor")
    return tuple(reversed(coordinates))


def _read_linear(tensor: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    if tensor.ndim == 0:
        return tensor.detach().reshape(1).clone()
    return tensor[_linear_coordinates(indices, tensor.shape)].detach().clone()


def _write_linear(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    values: torch.Tensor,
) -> None:
    if tensor.ndim == 0:
        tensor.copy_(values.reshape(()))
        return
    tensor[_linear_coordinates(indices, tensor.shape)] = values


def _first_float_tensor(value: Any) -> torch.Tensor | None:
    leaves, _ = tree_flatten(value)
    for leaf in leaves:
        if isinstance(leaf, torch.Tensor) and leaf.is_floating_point():
            return leaf
    return None


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


def _elementwise_same(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    equal = left == right
    if left.is_floating_point():
        equal = equal | (torch.isnan(left) & torch.isnan(right))
    return equal


def _synchronize_state_mutation(tensor: torch.Tensor) -> None:
    """Finish asynchronous device reads before mutating checkpoint source storage."""
    if tensor.is_cuda:
        torch.cuda.synchronize(tensor.device)


def _incident_windows_may_overlap(
    left: FaultIncident,
    right: FaultIncident,
) -> bool:
    left_count = _trigger_candidate_count(left)
    right_count = _trigger_candidate_count(right)
    if min(left_count, right_count) <= 10_000:
        if left_count <= right_count:
            return any(
                _candidate_window_hits_incident(value, left, right) for value in _triggers(left)
            )
        return any(
            _candidate_window_hits_incident(value, right, left) for value in _triggers(right)
        )

    left_start, left_end = _incident_window_bounds(left)
    right_start, right_end = _incident_window_bounds(right)
    return left_start <= right_end and right_start <= left_end


def _candidate_window_hits_incident(
    iteration: int,
    owner: FaultIncident,
    other: FaultIncident,
) -> bool:
    owner_duration = _incident_iteration_duration(owner)
    other_duration = _incident_iteration_duration(other)
    low = 1 if other_duration is None else max(1, iteration - other_duration + 1)
    high = math.inf if owner_duration is None else iteration + owner_duration - 1
    return _trigger_has_candidate_between(other, low, high)


def _trigger_has_candidate_between(
    incident: FaultIncident,
    low: int,
    high: float,
) -> bool:
    trigger = incident.trigger
    if trigger.range is not None:
        candidate = trigger.range.start
        if candidate < low:
            candidate += (
                (low - candidate + trigger.range.every - 1) // trigger.range.every
            ) * trigger.range.every
        return candidate <= trigger.range.end and candidate <= high
    position = bisect_left(trigger.at, low)
    return position < len(trigger.at) and trigger.at[position] <= high


def _trigger_candidate_count(incident: FaultIncident) -> int:
    trigger = incident.trigger
    if trigger.range is None:
        return len(trigger.at)
    return 1 + (trigger.range.end - trigger.range.start) // trigger.range.every


def _triggers(incident: FaultIncident):
    trigger = incident.trigger
    if trigger.range is None:
        yield from trigger.at
        return
    yield from range(trigger.range.start, trigger.range.end + 1, trigger.range.every)


def _incident_window_bounds(incident: FaultIncident) -> tuple[int, float]:
    trigger = incident.trigger
    first = trigger.range.start if trigger.range is not None else trigger.at[0]
    last = trigger.range.end if trigger.range is not None else trigger.at[-1]
    duration = _incident_iteration_duration(incident)
    return first, math.inf if duration is None else last + duration - 1


def _incident_iteration_duration(incident: FaultIncident) -> int | None:
    if incident.lifetime.iterations is not None:
        return incident.lifetime.iterations
    if incident.lifetime.matching_calls is not None:
        return None if incident.lifetime.matching_calls > 1 else 1
    return None


def _local_shard(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    if type(tensor).__name__ == "DTensor" and callable(to_local):
        local = to_local()
        if not isinstance(local, torch.Tensor):
            raise TypeError("DTensor.to_local() must return a tensor")
        return local
    return tensor


def _rewrap_local(reference: torch.Tensor, local: torch.Tensor) -> torch.Tensor:
    if type(reference).__name__ != "DTensor":
        return local
    factory = getattr(type(reference), "from_local", None)
    if not callable(factory):
        raise TypeError("DTensor type exposes no from_local constructor")
    return factory(
        local,
        device_mesh=reference.device_mesh,
        placements=reference.placements,
        run_check=False,
        shape=reference.shape,
        stride=reference.stride(),
    )


def _validate_state_retirement(
    original: torch.Tensor,
    transformed: torch.Tensor,
    request: FaultExecutionRequest,
) -> None:
    if request.lifetime.iterations is None:
        return
    original = _local_shard(original)
    transformed = _local_shard(transformed)
    delta = transformed - original
    if not bool(torch.all(torch.isfinite(delta)).item()):
        raise ValueError(
            "bounded weight, bias, and optimizer_state faults must have a finite "
            "retirement delta; use an until lifetime with recovery for non-finite corruption"
        )


__all__ = ["LocalFaultEffect", "LocalFaultExecutor"]
