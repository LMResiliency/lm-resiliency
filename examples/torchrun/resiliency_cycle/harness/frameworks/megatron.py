"""Megatron Core driver for torchrun fault injection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist

from examples.production_loops.megatron import (
    MICRO_BATCH_SIZE,
    _arguments,
    _build_model_optimizer_scheduler,
    _forward_step,
    _install_training_services,
    _tokens,
)
from lm_resiliency.integrations.megatron import MegatronAdapter, enable_resiliency

from ..runtime import DriverConfig, clone_tensors, close_resources


class MegatronDriver:
    framework = "megatron"
    expected_recipes = {"hidden"}

    def __init__(self, config: DriverConfig) -> None:
        environment = __import__("os").environ
        self.rank = int(environment["RANK"])
        self.world_size = int(environment["WORLD_SIZE"])
        local_rank = int(environment["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")

        from megatron.core.num_microbatches_calculator import (
            init_num_microbatches_calculator,
        )
        from megatron.training import training

        self._training = training
        self.args = _arguments(self.rank, self.world_size, config.total_steps)
        init_num_microbatches_calculator(
            rank=self.rank,
            global_batch_size=self.args.global_batch_size,
            micro_batch_size=self.args.micro_batch_size,
            data_parallel_size=self.args.data_parallel_size,
            decrease_batch_size_if_needed=False,
            step_batch_size_schedule=None,
            seq_length=self.args.seq_length,
        )
        _install_training_services(training, self.args)
        self.model, self.optimizer, self.scheduler, self.model_config = (
            _build_model_optimizer_scheduler(self.device)
        )
        self._extra_state: dict[str, Any] = {"absolute_step": 0}
        self.handle = enable_resiliency(
            [self.model],
            self.optimizer,
            self.scheduler,
            interval=1,
            ckpt_config=config.checkpoint,
            detection_config=config.replay,
            device=self.device,
            fault_callback=config.fault_callback,
            orchestration=config.orchestration,
            extra_state_fn=self._extra_state_dict,
            load_extra_state_fn=self._load_extra_state_dict,
            **config.recovery_options(),
        )
        if self.args.iteration != self.handle.step_count:
            raise RuntimeError("Megatron loop iteration does not match the recovered GEMINI step")
        self._adapter = MegatronAdapter([self.model], self.optimizer, self.scheduler)

    def _extra_state_dict(self) -> dict[str, Any]:
        return {
            "fault_validation": dict(self._extra_state),
            "iteration": int(self._extra_state["absolute_step"]),
            "consumed_train_samples": (
                int(self.args.consumed_train_samples) + MICRO_BATCH_SIZE * self.world_size
            ),
            "skipped_train_samples": int(self.args.skipped_train_samples),
            "scheduler": self.scheduler.state_dict(),
        }

    def _load_extra_state_dict(self, value: dict[str, Any]) -> None:
        state = value.get("fault_validation", {})
        if not isinstance(state, dict):
            raise RuntimeError("Megatron fault-validation state is malformed")
        self._extra_state.clear()
        self._extra_state.update(state)
        self.args.iteration = _nonnegative_int(value.get("iteration"), "iteration")
        self.args.consumed_train_samples = _nonnegative_int(
            value.get("consumed_train_samples"),
            "consumed_train_samples",
        )
        self.args.skipped_train_samples = _nonnegative_int(
            value.get("skipped_train_samples"),
            "skipped_train_samples",
        )
        scheduler = value.get("scheduler")
        if not isinstance(scheduler, dict):
            raise RuntimeError("Megatron scheduler state is malformed")
        self.scheduler.load_state_dict(scheduler)

    def _data_iterator(self, start_step: int) -> Iterator[tuple[torch.Tensor, ...]]:
        step = start_step
        while True:
            yield _tokens(self.rank, step, self.device)
            step += 1

    def run(
        self,
        *,
        before_step: Callable[[int], None],
        after_step: Callable[[int, float | None], None],
        total_steps: int,
    ) -> None:
        from megatron.core.pipeline_parallel.schedules import (
            get_forward_backward_func,
        )

        data_iterator = self._data_iterator(self.handle.step_count)
        forward_backward = get_forward_backward_func()
        for step in range(self.handle.step_count + 1, total_steps + 1):
            self._extra_state["absolute_step"] = step
            self.args.curr_iteration = step - 1
            before_step(step)
            result = self._training.train_step(
                _forward_step,
                data_iterator,
                [self.model],
                self.optimizer,
                self.scheduler,
                self.model_config,
                forward_backward,
                iteration=step - 1,
            )
            loss_dict = result[0]
            loss = loss_dict.get("lm loss")
            self.args.iteration = step
            self.args.consumed_train_samples += MICRO_BATCH_SIZE * self.world_size
            after_step(
                step,
                float(loss.detach()) if isinstance(loss, torch.Tensor) else None,
            )

    def verification_state(self) -> dict[str, list[torch.Tensor]]:
        tensors = self._adapter.collect_checkpoint_tensors()
        model_count = sum(1 for parameter in self.model.parameters())
        return {
            "model": clone_tensors(tensors[:model_count]),
            "optimizer": clone_tensors(tensors[model_count:]),
        }

    def framework_state(self) -> dict[str, Any]:
        return {
            "absolute_step": int(self._extra_state["absolute_step"]),
            "consumed_train_samples": int(self.args.consumed_train_samples),
            "iteration": int(self.args.iteration),
            "scheduler_num_steps": int(self.scheduler.num_steps),
            "skipped_train_samples": int(self.args.skipped_train_samples),
            "step_count": int(self.handle.step_count),
        }

    def close(self) -> None:
        def destroy_model_parallel() -> None:
            if not dist.is_initialized():
                return
            from megatron.core import parallel_state as mpu

            mpu.destroy_model_parallel()

        close_resources(
            ("resiliency handle", self.handle.close),
            ("Megatron model parallel", destroy_model_parallel),
            (
                "default process group",
                lambda: dist.destroy_process_group() if dist.is_initialized() else None,
            ),
        )


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RuntimeError(f"Megatron {name} must be a non-negative integer")
    return value
