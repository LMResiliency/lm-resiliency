"""Tests for sampled optimizer-step output collection."""

import pytest
import torch
import torch.nn as nn

from lm_resiliency.detection.optimizer_step import (
    OPTIMIZER_STATUS_OK,
    OptimizerReplayRecipe,
    OptimizerStepCheckUnsupported,
    OptimizerStepReplay,
    collect_optimizer_replays,
    collect_updated_weights,
)


class RuntimeKernelOptimizer(torch.optim.Optimizer):
    """Optimizer whose step depends on an attribute omitted by ``copy.copy``."""

    def __init__(self, parameters, *, lr):
        super().__init__(parameters, {"lr": lr})
        self.runtime_kernel = self._update

    @staticmethod
    def _update(parameter, gradient, lr):
        parameter.add_(gradient, alpha=-lr)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    self.runtime_kernel(parameter, parameter.grad, group["lr"])


def _single_recipe(replay: OptimizerStepReplay) -> OptimizerReplayRecipe:
    batch = collect_optimizer_replays([replay])
    assert batch is not None
    assert len(batch.recipes) == 1
    recipe = batch.recipes[0]
    assert recipe.status == OPTIMIZER_STATUS_OK
    assert recipe.capture is not None
    return recipe


def _actual_updated_slice(recipe: OptimizerReplayRecipe) -> torch.Tensor:
    assert recipe.capture is not None
    capture = recipe.capture
    end = capture.offset + capture.length
    return capture.parameter.detach().reshape(-1)[capture.offset : end]


def test_collects_only_sampled_layer_updated_weights():
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 2))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    model(torch.randn(3, 4)).sum().backward()
    optimizer.step()

    groups = collect_updated_weights(
        optimizer,
        list(model[0].parameters()),
    )

    assert list(groups) == ["optimizer_updated_weight"]
    assert len(groups["optimizer_updated_weight"]) == 2
    for collected, parameter in zip(
        groups["optimizer_updated_weight"],
        model[0].parameters(),
    ):
        assert collected.data_ptr() == parameter.data_ptr()
        assert torch.equal(collected, parameter)


def test_rejects_layer_not_owned_by_optimizer():
    sampled_layer = nn.Linear(4, 4)
    optimized_layer = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(optimized_layer.parameters(), lr=0.1)

    with pytest.raises(OptimizerStepCheckUnsupported, match="no parameters"):
        collect_updated_weights(optimizer, list(sampled_layer.parameters()))


@pytest.mark.parametrize(
    ("optimizer_type", "kwargs"),
    [
        (torch.optim.AdamW, {"lr": 0.01, "weight_decay": 0.1}),
        (torch.optim.Adam, {"lr": 0.01}),
        (torch.optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    ],
)
def test_isolated_replay_matches_real_optimizer_transition(optimizer_type, kwargs):
    parameter = nn.Parameter(torch.randn(32))
    optimizer = optimizer_type([parameter], **kwargs)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    parameter.grad = torch.randn_like(parameter)
    replay.arm()
    optimizer.step()

    recipe = _single_recipe(replay)
    replayed = recipe.replay(recipe.source_payload())
    actual = _actual_updated_slice(recipe)
    assert torch.equal(actual, replayed)
    replay.remove()


def test_isolated_replay_preserves_optimizer_runtime_kernel():
    parameter = nn.Parameter(torch.randn(16))
    optimizer = RuntimeKernelOptimizer([parameter], lr=0.01)
    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    parameter.grad = torch.randn_like(parameter)
    replay.arm()
    optimizer.step()

    recipe = _single_recipe(replay)
    replayed = recipe.replay(recipe.source_payload())
    actual = _actual_updated_slice(recipe)
    assert torch.equal(actual, replayed)
    replay.remove()


def test_isolated_replay_does_not_copy_framework_step_wrapper():
    parameter = nn.Parameter(torch.randn(16))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    original_step = optimizer.step
    wrapper_calls = 0

    def wrapped_step():
        nonlocal wrapper_calls
        wrapper_calls += 1
        return original_step()

    optimizer.step = wrapped_step
    parameter.grad = torch.randn_like(parameter)
    replay.arm()
    optimizer.step()

    recipe = _single_recipe(replay)
    replayed = recipe.replay(recipe.source_payload())
    actual = _actual_updated_slice(recipe)
    assert wrapper_calls == 1
    assert torch.equal(actual, replayed)
    replay.remove()


def test_replay_does_not_mutate_live_parameter_or_optimizer_state():
    parameter = nn.Parameter(torch.randn(16))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    parameter.grad = torch.randn_like(parameter)
    replay.arm()
    optimizer.step()
    parameter_after_real_step = parameter.detach().clone()
    state_after_real_step = {
        key: value.detach().clone() if isinstance(value, torch.Tensor) else value
        for key, value in optimizer.state[parameter].items()
    }

    recipe = replay.consume()
    assert recipe is not None
    recipe.replay(recipe.source_payload())
    assert torch.equal(parameter, parameter_after_real_step)
    for key, expected in state_after_real_step.items():
        actual = optimizer.state[parameter][key]
        if isinstance(expected, torch.Tensor):
            assert torch.equal(actual, expected)
        else:
            assert actual == expected
    replay.remove()


def test_source_replay_is_isolated_from_real_update_corruption():
    parameter = nn.Parameter(torch.randn(8))
    optimizer = torch.optim.AdamW([parameter], lr=0.01)
    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    def corrupt_after_step(opt, args, kwargs):
        del opt, args, kwargs
        parameter.data[0].add_(1.0)

    corruption_hook = optimizer.register_step_post_hook(corrupt_after_step)
    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    parameter.grad = torch.randn_like(parameter)
    replay.arm()
    optimizer.step()

    recipe = _single_recipe(replay)
    actual = _actual_updated_slice(recipe)
    replayed = recipe.replay(recipe.source_payload())
    assert not torch.equal(actual, replayed)
    corruption_hook.remove()
    replay.remove()


def test_replay_capture_is_inactive_until_armed():
    parameter = nn.Parameter(torch.randn(8))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    replay = OptimizerStepReplay(optimizer)

    parameter.grad = torch.randn_like(parameter)
    optimizer.step()

    assert replay.consume() is None
    replay.remove()


def test_replay_rotates_across_flat_parameter_slices():
    parameter = nn.Parameter(torch.randn(12))
    optimizer = torch.optim.SGD([parameter], lr=0.1)
    replay = OptimizerStepReplay(optimizer, slice_numel=5)
    observed_lengths = []

    for _ in range(3):
        parameter.grad = torch.randn_like(parameter)
        replay.arm()
        optimizer.step()
        recipe = replay.consume()
        assert recipe is not None
        assert recipe.capture is not None
        observed_lengths.append(recipe.capture.length)

    assert observed_lengths == [5, 5, 2]
    replay.remove()


def test_replay_rotates_across_multiple_base_optimizer_invocations():
    first = nn.Parameter(torch.randn(5))
    second = nn.Parameter(torch.randn(3))
    optimizer = torch.optim.SGD(
        [{"params": [first]}, {"params": [second]}],
        lr=0.1,
    )
    all_groups = optimizer.param_groups
    replay = OptimizerStepReplay(optimizer, slice_numel=8)
    observed_lengths = []

    for _ in range(2):
        replay.arm()
        for group, parameter in zip(all_groups, (first, second)):
            optimizer.param_groups = [group]
            parameter.grad = torch.randn_like(parameter)
            optimizer.step()
        optimizer.param_groups = all_groups
        recipe = replay.consume()
        assert recipe is not None
        assert recipe.capture is not None
        observed_lengths.append(recipe.capture.length)

    assert observed_lengths == [5, 3]
    replay.remove()


def test_peer_replays_source_payload_instead_of_local_transition():
    source_parameter = nn.Parameter(torch.linspace(-1.0, 1.0, 16))
    peer_parameter = nn.Parameter(torch.linspace(5.0, 8.0, 16))
    source_optimizer = torch.optim.AdamW([source_parameter], lr=0.01)
    peer_optimizer = torch.optim.AdamW([peer_parameter], lr=0.01)

    source_parameter.grad = torch.linspace(0.1, 0.4, 16)
    peer_parameter.grad = torch.linspace(-0.7, -0.2, 16)
    source_optimizer.step()
    peer_optimizer.step()

    source_replay = OptimizerStepReplay(source_optimizer, slice_numel=8)
    peer_replay = OptimizerStepReplay(peer_optimizer, slice_numel=8)
    source_parameter.grad = torch.linspace(0.5, 0.8, 16)
    peer_parameter.grad = torch.linspace(-0.3, 0.3, 16)
    source_replay.arm()
    peer_replay.arm()
    source_optimizer.step()
    peer_optimizer.step()

    source_recipe = _single_recipe(source_replay)
    peer_recipe = _single_recipe(peer_replay)
    source_payload = source_recipe.source_payload()

    source_result = source_recipe.replay(source_payload)
    peer_result = peer_recipe.replay(source_payload)
    peer_local_result = peer_recipe.replay(peer_recipe.source_payload())

    assert torch.equal(source_result, peer_result)
    assert not torch.equal(source_result, peer_local_result)
    source_replay.remove()
    peer_replay.remove()
