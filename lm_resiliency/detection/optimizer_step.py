# mypy: ignore-errors
"""C3 payload collection and isolated optimizer-transition replay."""

from __future__ import annotations

import copy
import logging
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

OPTIMIZER_UPDATED_WEIGHT = "optimizer_updated_weight"
OPTIMIZER_REPLAY_INPUT = "optimizer_replay_input"
OPTIMIZER_REPLAY_STATUS = "optimizer_replay_status"
DEFAULT_REPLAY_SLICE_NUMEL = 64 * 1024

OPTIMIZER_STATUS_OK = 0
_STATUS_STEP_NOT_OBSERVED = 1
_STATUS_CAPTURE_FAILED = 2


class OptimizerStepCheckUnsupported(RuntimeError):
    """The optimizer layout cannot be compared by the built-in collector."""


@dataclass
class _TransitionCapture:
    optimizer: torch.optim.Optimizer
    group: dict[str, Any]
    parameter: torch.Tensor
    offset: int
    length: int
    parameter_before: torch.Tensor
    gradient: torch.Tensor
    state: dict[str, Any]
    optimizer_tensors: dict[str, torch.Tensor]

    def source_payload(self) -> dict[str, Any]:
        """Return the bounded copied transition state broadcast by the source peer."""
        return {
            "optimizer_type": (
                f"{type(self.optimizer).__module__}.{type(self.optimizer).__qualname__}"
            ),
            "defaults": _clone_value(self.optimizer.defaults),
            "group": _clone_value(self.group),
            "parameter_before": self.parameter_before.detach().clone(),
            "gradient": self.gradient.detach().clone(),
            "state": _clone_value(self.state),
            "optimizer_tensors": _clone_value(self.optimizer_tensors),
        }

    def replay(self, payload: Mapping[str, Any]) -> torch.Tensor:
        """Run a source-broadcast transition on state disjoint from training."""
        return _replay_optimizer_payload(
            self.optimizer,
            self.parameter.requires_grad,
            payload,
        )


def _replay_optimizer_payload(
    optimizer: torch.optim.Optimizer,
    requires_grad: bool,
    payload: Mapping[str, Any],
) -> torch.Tensor:
    """Execute one copied source transition through the local optimizer kernel."""
    optimizer_type = f"{type(optimizer).__module__}.{type(optimizer).__qualname__}"
    if payload["optimizer_type"] != optimizer_type:
        raise OptimizerStepCheckUnsupported(
            "source optimizer type differs from the local optimizer: "
            f"source={payload['optimizer_type']}, local={optimizer_type}"
        )
    parameter_before = payload["parameter_before"]
    gradient = payload["gradient"]
    if not isinstance(parameter_before, torch.Tensor) or not isinstance(
        gradient,
        torch.Tensor,
    ):
        raise OptimizerStepCheckUnsupported(
            "source optimizer replay parameter and gradient must be tensors"
        )
    replay_parameter = nn.Parameter(
        parameter_before.detach().clone(),
        requires_grad=requires_grad,
    )
    replay_parameter.grad = gradient.detach().clone()

    replay_optimizer = _clone_optimizer_shell(
        optimizer,
        replay_parameter,
        payload["defaults"],
        payload["group"],
        payload["state"],
        payload["optimizer_tensors"],
    )
    replay_optimizer.step()
    return replay_parameter.detach()


@dataclass
class OptimizerReplayRecipe:
    """One scheduled optimizer transition awaiting source broadcast and replay."""

    optimizer: torch.optim.Optimizer
    status: int
    capture: _TransitionCapture | None

    def source_payload(self) -> dict[str, Any]:
        if self.capture is None:
            raise OptimizerStepCheckUnsupported("source optimizer transition was not captured")
        return self.capture.source_payload()

    def replay(self, payload: Mapping[str, Any]) -> torch.Tensor:
        capture = self.capture
        if capture is not None:
            return capture.replay(payload)
        parameter = next(
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
            if isinstance(parameter, torch.Tensor)
        )
        return _replay_optimizer_payload(
            self.optimizer,
            parameter.requires_grad,
            payload,
        )


@dataclass(frozen=True)
class OptimizerReplayBatch:
    """Optimizer recipes captured at one framework optimizer boundary."""

    recipes: tuple[OptimizerReplayRecipe, ...]


OptimizerStepEvidence = OptimizerReplayBatch | dict[str, list[torch.Tensor]]


class OptimizerStepReplay:
    """Replay one rotating optimizer-owned flat slice on cloned state.

    Framework wrappers such as DeepSpeed ZeRO and Megatron DistributedOptimizer
    preprocess gradients before invoking a base ``torch.optim.Optimizer``. Hooks
    on that base optimizer therefore capture the effective gradient at the right
    boundary without patching either framework.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        slice_numel: int = DEFAULT_REPLAY_SLICE_NUMEL,
    ) -> None:
        if slice_numel < 1:
            raise ValueError("slice_numel must be positive")
        if not callable(getattr(optimizer, "register_step_pre_hook", None)):
            raise OptimizerStepCheckUnsupported(
                f"{type(optimizer).__name__} does not expose optimizer pre-step hooks"
            )

        self._optimizer = optimizer
        self._slice_numel = slice_numel
        self._slice_cursors: dict[int, tuple[int, int]] = {}
        self._invocation_target = 0
        self._observed_invocations = 0
        self._armed = False
        self._scheduled = False
        self._status = _STATUS_STEP_NOT_OBSERVED
        self._capture: _TransitionCapture | None = None
        self._warning_emitted = False
        self._pre_hook = optimizer.register_step_pre_hook(self._capture_pre_step)

    def arm(self) -> None:
        """Arm capture for the next invocation of this base optimizer."""
        self._armed = True
        self._scheduled = True
        self._status = _STATUS_STEP_NOT_OBSERVED
        self._capture = None
        self._observed_invocations = 0

    def cancel(self) -> None:
        """Discard a pending capture, normally after the outer step raises."""
        self._armed = False
        self._scheduled = False
        self._capture = None

    def consume(self) -> OptimizerReplayRecipe | None:
        """Return one scheduled transition recipe without replaying local inputs."""
        if not self._scheduled:
            return None

        self._scheduled = False
        self._armed = False
        capture = self._capture
        self._capture = None
        status = self._status
        self._advance_invocation_target()

        return OptimizerReplayRecipe(
            optimizer=self._optimizer,
            status=status,
            capture=capture,
        )

    def remove(self) -> None:
        """Remove the base-optimizer hooks."""
        self._pre_hook.remove()

    def _capture_pre_step(
        self,
        optimizer: torch.optim.Optimizer,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> None:
        del args, kwargs
        if not self._scheduled:
            return
        invocation_index = self._observed_invocations
        self._observed_invocations += 1
        if not self._armed or invocation_index != self._invocation_target:
            return
        self._armed = False
        try:
            self._capture = self._snapshot_transition(optimizer, invocation_index)
            self._status = OPTIMIZER_STATUS_OK
        except Exception as exc:
            self._capture = None
            self._status = _STATUS_CAPTURE_FAILED
            self._warn_once("optimizer transition capture failed", exc)

    def _snapshot_transition(
        self,
        optimizer: torch.optim.Optimizer,
        invocation_index: int,
    ) -> _TransitionCapture:
        entries = [
            (group, parameter_index, parameter)
            for group in optimizer.param_groups
            for parameter_index, parameter in enumerate(group["params"])
            if isinstance(parameter, torch.Tensor) and _local_tensor(parameter).numel() > 0
        ]
        if not entries:
            raise OptimizerStepCheckUnsupported("optimizer has no tensor parameters")

        parameter_index, slice_offset = self._slice_cursors.get(invocation_index, (0, 0))
        entry_index = parameter_index % len(entries)
        group, parameter_index, parameter = entries[entry_index]
        local_parameter = _local_tensor(parameter)
        offset = min(slice_offset, local_parameter.numel() - 1)
        length = min(self._slice_numel, local_parameter.numel() - offset)
        self._advance_slice_cursor(
            invocation_index,
            entry_index,
            len(entries),
            offset,
            length,
            local_parameter.numel(),
        )

        gradient = parameter.grad
        if gradient is None:
            raise OptimizerStepCheckUnsupported("sampled optimizer parameter has no gradient")
        local_gradient = _local_tensor(gradient)
        if local_gradient.is_sparse:
            raise OptimizerStepCheckUnsupported("sparse optimizer gradients are not supported")
        if local_gradient.numel() != local_parameter.numel():
            raise OptimizerStepCheckUnsupported(
                "optimizer parameter and gradient have different local sizes"
            )

        end = offset + length
        parameter_before = local_parameter.detach().reshape(-1)[offset:end].clone()
        gradient_slice = local_gradient.detach().reshape(-1)[offset:end].clone()
        state = {
            key: _clone_state_value(value, local_parameter.numel(), offset, end)
            for key, value in optimizer.state.get(parameter, {}).items()
        }
        optimizer_tensors = {
            name: value.detach().clone()
            for name, value in optimizer.__dict__.items()
            if isinstance(value, torch.Tensor)
        }
        cloned_group = _clone_parameter_group(group, parameter_index)
        return _TransitionCapture(
            optimizer=optimizer,
            group=cloned_group,
            parameter=parameter,
            offset=offset,
            length=length,
            parameter_before=parameter_before,
            gradient=gradient_slice,
            state=state,
            optimizer_tensors=optimizer_tensors,
        )

    def _advance_slice_cursor(
        self,
        invocation_index: int,
        entry_index: int,
        entry_count: int,
        offset: int,
        length: int,
        parameter_numel: int,
    ) -> None:
        next_offset = offset + length
        if next_offset < parameter_numel:
            self._slice_cursors[invocation_index] = (entry_index, next_offset)
        else:
            self._slice_cursors[invocation_index] = (
                (entry_index + 1) % entry_count,
                0,
            )

    def _advance_invocation_target(self) -> None:
        if self._observed_invocations > 0:
            self._invocation_target = (self._invocation_target + 1) % self._observed_invocations

    def _warn_once(self, message: str, exc: Exception) -> None:
        if self._warning_emitted:
            return
        self._warning_emitted = True
        logger.warning("SCOUT %s for %s: %s", message, type(self._optimizer).__name__, exc)


def collect_optimizer_replays(
    replays: Sequence[OptimizerStepReplay],
) -> OptimizerReplayBatch | None:
    """Collect scheduled recipes for source broadcast in the replay detector."""
    recipes: list[OptimizerReplayRecipe] = []
    for replay in replays:
        recipe = replay.consume()
        if recipe is None:
            continue
        recipes.append(recipe)
    if not recipes:
        return None
    return OptimizerReplayBatch(tuple(recipes))


def collect_updated_weights(
    optimizer: torch.optim.Optimizer,
    parameters: Sequence[nn.Parameter],
    *,
    allow_local_dtensor_shards: bool = False,
) -> dict[str, list[torch.Tensor]]:
    """Collect the updated weights for one sampled layer.

    This runs after the real optimizer step. It does not clone or re-execute the
    optimizer. C3 compares the updated weights across equivalent replicas.
    """
    selected = {id(parameter) for parameter in parameters}
    sampled_parameters: list[nn.Parameter] = []

    for group in optimizer.param_groups:
        for parameter in group["params"]:
            if id(parameter) not in selected:
                continue
            if type(parameter).__name__ == "DTensor" and not allow_local_dtensor_shards:
                raise OptimizerStepCheckUnsupported(
                    "DTensor parameters require a confirmed HSDP replica group"
                )
            sampled_parameters.append(parameter)

    if not sampled_parameters:
        raise OptimizerStepCheckUnsupported(
            "the sampled layer has no parameters owned by this optimizer"
        )

    updated_parameters = [_local_tensor(parameter).detach() for parameter in sampled_parameters]
    return {OPTIMIZER_UPDATED_WEIGHT: updated_parameters}


def _local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to_local() if type(tensor).__name__ == "DTensor" else tensor


def _clone_parameter_group(
    group: dict[str, Any],
    parameter_index: int,
) -> dict[str, Any]:
    cloned: dict[str, Any] = {}
    for key, value in group.items():
        if key == "params":
            continue
        if key == "param_names" and isinstance(value, Sequence):
            cloned[key] = [value[parameter_index]]
        else:
            cloned[key] = _clone_value(value)
    return cloned


def _clone_state_value(
    value: Any,
    parameter_numel: int,
    offset: int,
    end: int,
) -> Any:
    if isinstance(value, torch.Tensor):
        local = _local_tensor(value)
        if local.numel() == parameter_numel and local.ndim > 0:
            return local.detach().reshape(-1)[offset:end].clone()
        return local.detach().clone()
    if isinstance(value, dict):
        return {
            key: _clone_state_value(item, parameter_numel, offset, end)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_clone_state_value(item, parameter_numel, offset, end) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_state_value(item, parameter_numel, offset, end) for item in value)
    return copy.deepcopy(value)


def _clone_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().clone()
    if isinstance(value, dict):
        return {key: _clone_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_value(item) for item in value)
    return copy.deepcopy(value)


def _clone_optimizer_shell(
    optimizer: torch.optim.Optimizer,
    parameter: nn.Parameter,
    defaults: Mapping[str, Any],
    group: Mapping[str, Any],
    state: Mapping[str, Any],
    optimizer_tensors: Mapping[str, torch.Tensor],
) -> torch.optim.Optimizer:
    """Shallow-copy optimizer code while replacing every mutable tensor input."""
    replay_optimizer = copy.copy(optimizer)
    for name, value in optimizer.__dict__.items():
        if name != "step" and name not in replay_optimizer.__dict__:
            setattr(replay_optimizer, name, value)
    replay_optimizer.defaults = _clone_value(defaults)
    replay_optimizer.param_groups = [{**_clone_value(group), "params": [parameter]}]
    replay_optimizer.state = defaultdict(dict, {parameter: _clone_value(state)})

    for name, value in optimizer_tensors.items():
        setattr(replay_optimizer, name, value.detach().clone())

    for name in (
        "_optimizer_step_pre_hooks",
        "_optimizer_step_post_hooks",
        "_optimizer_state_dict_pre_hooks",
        "_optimizer_state_dict_post_hooks",
        "_optimizer_load_state_dict_pre_hooks",
        "_optimizer_load_state_dict_post_hooks",
    ):
        setattr(replay_optimizer, name, OrderedDict())
    return replay_optimizer
