"""DeepSpeed ZeRO-2 driver for torchrun fault injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist

from examples.production_loops.deepspeed import (
    MICRO_BATCH_SIZE,
    TinyCausalLM,
    _tokens,
)
from lm_resiliency.integrations.deepspeed import enable_resiliency
from lm_resiliency.integrations.deepspeed.adapter import DeepSpeedAdapter

from ..runtime import DriverConfig, clone_tensors, close_resources


class DeepSpeedDriver:
    framework = "deepspeed"
    expected_recipes = {"hidden"}

    def __init__(self, config: DriverConfig) -> None:
        import deepspeed

        environment = __import__("os").environ
        self.rank = int(environment["RANK"])
        self.world_size = int(environment["WORLD_SIZE"])
        local_rank = int(environment["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        deepspeed.init_distributed(dist_backend="nccl")
        torch.manual_seed(123)
        model = TinyCausalLM()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
        self.engine, _, _, _ = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config={
                "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
                "gradient_accumulation_steps": 1,
                "train_batch_size": MICRO_BATCH_SIZE * self.world_size,
                "zero_optimization": {"stage": 2},
                "bf16": {"enabled": True},
                "steps_per_print": 1_000,
            },
        )
        self.handle = enable_resiliency(
            self.engine,
            interval=1,
            ckpt_config=config.checkpoint,
            detection_config=config.replay,
            device=self.device,
            fault_callback=config.fault_callback,
            orchestration=config.orchestration,
            **config.recovery_options(),
        )
        self._adapter = DeepSpeedAdapter(self.engine)

    def run(
        self,
        *,
        before_step: Callable[[int], None],
        after_step: Callable[[int, float | None], None],
        total_steps: int,
    ) -> None:
        for step in range(self.handle.step_count + 1, total_steps + 1):
            before_step(step)
            tokens, labels = _tokens(self.rank, step - 1, self.engine.device)
            loss = self.engine(tokens, labels)
            self.engine.backward(loss)
            self.engine.step()
            after_step(step, float(loss.detach()))

    def verification_state(self) -> dict[str, list[torch.Tensor]]:
        tensors = self._adapter.collect_checkpoint_tensors()
        model_group_count = len(self.engine.optimizer.bit16_groups_flat)
        return {
            "model": clone_tensors(tensors[:model_group_count]),
            "optimizer": clone_tensors(tensors[model_group_count:]),
        }

    def framework_state(self) -> dict[str, Any]:
        return {
            "global_steps": int(self.engine.global_steps),
            "step_count": int(self.handle.step_count),
        }

    def close(self) -> None:
        close_resources(
            ("resiliency handle", self.handle.close),
            ("DeepSpeed engine", self.engine.destroy),
            (
                "default process group",
                lambda: dist.destroy_process_group() if dist.is_initialized() else None,
            ),
        )
