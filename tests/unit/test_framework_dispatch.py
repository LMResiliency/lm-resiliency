"""Tests for the package-root automatic framework dispatcher."""

import inspect
import typing
from unittest.mock import patch

import pytest
import torch.nn as nn

import lm_resiliency
from lm_resiliency import ResiliencySession, enable_resiliency
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig


def test_public_dispatch_return_annotation_is_runtime_resolvable():
    hints = typing.get_type_hints(lm_resiliency.enable_resiliency)

    assert hints["return"] is ResiliencySession


class FakeDeepSpeedEngine:
    def __init__(self):
        self.module = nn.Linear(4, 4)
        self.optimizer = object()

    def step(self):
        pass

    def zero_optimization_stage(self):
        return 2


class FakeTorchTitanTrainer:
    def __init__(self):
        self.model_parts = [nn.Linear(4, 4)]
        self.optimizers = object()
        self.lr_schedulers = object()
        self.parallel_dims = object()
        self.checkpointer = object()

    def train(self):
        pass


def test_plain_module_dispatches_to_pytorch():
    model = nn.Linear(4, 4)
    optimizer = object()
    checkpoint = InMemoryCkptConfig()
    replay = ReplayHarnessConfig()
    sentinel = object()

    with patch(
        "lm_resiliency.api.enable_resiliency",
        return_value=sentinel,
    ) as enable_pytorch:
        result = enable_resiliency(
            model,
            optimizer,
            checkpoint=checkpoint,
            replay=replay,
        )

    assert result is sentinel
    assert enable_pytorch.call_args.args == (model, optimizer)
    assert enable_pytorch.call_args.kwargs["checkpoint"] is checkpoint
    assert enable_pytorch.call_args.kwargs["replay"] is replay


def test_deepspeed_engine_dispatches_and_translates_configs():
    engine = FakeDeepSpeedEngine()
    checkpoint = InMemoryCkptConfig()
    replay = ReplayHarnessConfig()
    sentinel = object()

    with patch(
        "lm_resiliency.integrations.deepspeed.enable_resiliency",
        return_value=sentinel,
    ) as enable_deepspeed:
        result = enable_resiliency(
            engine,
            checkpoint=checkpoint,
            replay=replay,
        )

    assert result is sentinel
    assert enable_deepspeed.call_args.args == (engine,)
    assert enable_deepspeed.call_args.kwargs["ckpt_config"] is checkpoint
    assert enable_deepspeed.call_args.kwargs["detection_config"] is replay


def test_model_chunk_list_dispatches_to_megatron():
    model = [nn.Linear(4, 4), nn.Linear(4, 4)]
    optimizer = object()
    scheduler = object()
    restore = object()
    sentinel = object()

    with patch(
        "lm_resiliency.integrations.megatron.enable_resiliency",
        return_value=sentinel,
    ) as enable_megatron:
        result = enable_resiliency(
            model,
            optimizer,
            opt_param_scheduler=scheduler,
            load_extra_state_fn=restore,
        )

    assert result is sentinel
    assert enable_megatron.call_args.args == (model, optimizer)
    assert enable_megatron.call_args.kwargs["opt_param_scheduler"] is scheduler
    assert enable_megatron.call_args.kwargs["load_extra_state_fn"] is restore


def test_all_native_modules_dispatch_to_pytorch():
    model = nn.Linear(4, 4)
    optimizer = object()
    sentinel = object()

    with patch(
        "lm_resiliency.api.enable_resiliency",
        return_value=sentinel,
    ) as enable_pytorch:
        result = enable_resiliency(model, optimizer)

    assert result is sentinel
    assert enable_pytorch.call_args.args == (model, optimizer)


def test_framework_override_handles_custom_torchtitan_wrapper():
    model = nn.Linear(4, 4)
    optimizer = object()

    with patch(
        "lm_resiliency.integrations.torchtitan.enable_resiliency",
        return_value=object(),
    ) as enable_torchtitan:
        enable_resiliency(model, optimizer, framework="torchtitan")

    enable_torchtitan.assert_called_once()


def test_torchtitan_trainer_dispatches_without_explicit_optimizer():
    trainer = FakeTorchTitanTrainer()
    sentinel = object()

    with patch(
        "lm_resiliency.integrations.torchtitan.enable_resiliency",
        return_value=sentinel,
    ) as enable_torchtitan:
        result = enable_resiliency(trainer, interval=3)

    assert result is sentinel
    assert enable_torchtitan.call_args.args == (trainer, None)
    assert enable_torchtitan.call_args.kwargs["interval"] == 3


def test_dispatcher_rejects_ambiguous_or_unsupported_calls():
    model = nn.Linear(4, 4)

    with pytest.raises(TypeError, match="requires an optimizer"):
        enable_resiliency(model)
    with pytest.raises(TypeError, match="could not infer"):
        enable_resiliency(object(), object())
    with pytest.raises(ValueError, match="unsupported framework"):
        enable_resiliency(model, object(), framework="unknown")
    with pytest.raises(TypeError, match="does not accept: group"):
        enable_resiliency(FakeDeepSpeedEngine(), group=object())


def test_dispatcher_signature_supports_external_managers():
    parameters = inspect.signature(enable_resiliency).parameters

    assert "checkpoint" in parameters
    assert "orchestration" in parameters
    assert "engine" not in parameters
    assert "replay_callback" not in parameters
