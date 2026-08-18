"""Tests for binding lm-resiliency to TorchTitan's production Trainer surface."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch.nn as nn

from lm_resiliency.integrations.torchtitan import _SchedulerStepHook, enable_resiliency
from lm_resiliency.integrations.torchtitan.adapter import TorchTitanAdapter


class _Stateful:
    def __init__(self, state):
        self.state = dict(state)

    def state_dict(self):
        return dict(self.state)

    def load_state_dict(self, state):
        self.state = dict(state)

    def step(self, increment=1):
        self.state["last_epoch"] = int(self.state.get("last_epoch", 0)) + increment
        return self.state["last_epoch"]


class _Checkpointer:
    def __init__(self):
        self.loads = 0

    def load(self, *args, **kwargs):
        del args, kwargs
        self.loads += 1
        return False


class _Trainer:
    def __init__(self):
        self.model_parts = [nn.Linear(4, 4)]
        self.optimizers = object()
        self.lr_schedulers = _Stateful({"last_epoch": 2})
        self.dataloader = _Stateful({"index": 8})
        self.parallel_dims = SimpleNamespace(
            dp_replicate=2,
            dp_shard=1,
            tp=1,
            pp=1,
            world_size=2,
        )
        self.checkpointer = _Checkpointer()
        self.step = 2
        self.ntokens_seen = 64

    def train(self):
        pass

    def state_dict(self):
        return {"step": self.step, "ntokens_seen": self.ntokens_seen}

    def load_state_dict(self, state):
        self.step = state["step"]
        self.ntokens_seen = state["ntokens_seen"]


def test_trainer_adapter_captures_and_restores_framework_owned_state():
    trainer = _Trainer()
    adapter = TorchTitanAdapter(trainer)

    state = adapter.get_extra_state_dict()
    trainer.lr_schedulers.state = {"last_epoch": 99}
    trainer.dataloader.state = {"index": 99}
    trainer.step = 99
    trainer.ntokens_seen = 99
    adapter.load_extra_state_dict(state)

    assert trainer.lr_schedulers.state == {"last_epoch": 2}
    assert trainer.dataloader.state == {"index": 8}
    assert trainer.state_dict() == {"step": 2, "ntokens_seen": 64}


def test_trainer_entry_point_binds_framework_objects_and_durable_load():
    trainer = _Trainer()
    handle = MagicMock()
    handle.recovered_step = -1

    with patch(
        "lm_resiliency.integrations.torchtitan._enable_pytorch_resiliency",
        return_value=handle,
    ) as enable_pytorch:
        result = enable_resiliency(trainer, interval=3)

    assert result is handle
    assert enable_pytorch.call_args.args == (trainer.model_parts[0], trainer.optimizers)
    assert enable_pytorch.call_args.kwargs["parallelism_info"] is trainer.parallel_dims
    assert callable(enable_pytorch.call_args.kwargs["extra_state_fn"])
    assert callable(enable_pytorch.call_args.kwargs["load_extra_state_fn"])
    register_step_hook = enable_pytorch.call_args.kwargs["_step_hook_registrar"]
    assert callable(register_step_hook)

    observations = []
    hook = register_step_hook(
        lambda optimizer, _args, _kwargs: observations.append(
            (optimizer, trainer.lr_schedulers.state["last_epoch"])
        )
    )
    assert trainer.lr_schedulers.step(increment=3) == 5
    assert observations == [(trainer.optimizers, 5)]
    hook.remove()
    assert trainer.lr_schedulers.step(increment=1) == 6

    trainer.step = 7
    trainer.checkpointer.load(step=-1)
    assert trainer.checkpointer.loads == 1
    handle._restore_step.assert_not_called()

    restore_load = handle.add_close_callback.call_args.args[0]
    restore_load()
    trainer.checkpointer.load(step=-1)
    assert trainer.checkpointer.loads == 2


def test_trainer_records_only_successful_durable_load():
    trainer = _Trainer()
    trainer.checkpointer.load = MagicMock(return_value=True)
    handle = MagicMock()
    handle.recovered_step = -1

    with patch(
        "lm_resiliency.integrations.torchtitan._enable_pytorch_resiliency",
        return_value=handle,
    ):
        enable_resiliency(trainer)

    trainer.step = 7
    assert trainer.checkpointer.load(step=-1) is True
    handle._restore_step.assert_called_once_with(7)


def test_scheduler_hook_preserves_later_wrapper():
    scheduler = _Stateful({})
    optimizer = object()
    hook = _SchedulerStepHook(scheduler, optimizer, lambda *_args: None)
    later_wrapper = MagicMock(return_value=17)
    scheduler.step = later_wrapper

    hook.remove()

    assert scheduler.step is later_wrapper


def test_trainer_skips_durable_load_after_gemini_recovery():
    trainer = _Trainer()
    handle = MagicMock()
    handle.recovered_step = 5

    with patch(
        "lm_resiliency.integrations.torchtitan._enable_pytorch_resiliency",
        return_value=handle,
    ):
        enable_resiliency(trainer)

    assert trainer.checkpointer.load(step=-1) is True
    assert trainer.checkpointer.loads == 0
    handle._restore_step.assert_not_called()
