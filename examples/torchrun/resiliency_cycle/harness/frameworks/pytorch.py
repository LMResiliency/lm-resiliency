"""Native PyTorch DDP driver for torchrun fault injection."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from examples.production_loops.pytorch import TinyCausalLM, _tokens
from lm_resiliency.integrations.pytorch import enable_resiliency

from ..runtime import DriverConfig, clone_tensors


class PyTorchDriver:
    framework = "pytorch"
    expected_recipes = {"embedding", "hidden", "output", "optimizer"}

    def __init__(self, config: DriverConfig) -> None:
        self.rank = int(__import__("os").environ["RANK"])
        self.world_size = int(__import__("os").environ["WORLD_SIZE"])
        local_rank = int(__import__("os").environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        self.device = torch.device("cuda", local_rank)
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
        torch.manual_seed(123)
        self.model = DistributedDataParallel(
            TinyCausalLM().to(self.device),
            device_ids=[local_rank],
        )
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=2e-4)
        self._extra_state: dict[str, Any] = {"absolute_step": 0}
        self.handle = enable_resiliency(
            self.model,
            self.optimizer,
            interval=1,
            checkpoint=config.checkpoint,
            replay=config.replay,
            device=self.device,
            fault_callback=config.fault_callback,
            orchestration=config.orchestration,
            extra_state_fn=lambda: dict(self._extra_state),
            load_extra_state_fn=self._load_extra_state,
            recovery_mode=config.recovery_mode,
        )

    def _load_extra_state(self, value: dict[str, Any]) -> None:
        self._extra_state.clear()
        self._extra_state.update(value)

    def run(
        self,
        *,
        before_step: Callable[[int], None],
        after_step: Callable[[int, float | None], None],
        total_steps: int,
    ) -> None:
        for step in range(self.handle.step_count + 1, total_steps + 1):
            self._extra_state["absolute_step"] = step
            before_step(step)
            tokens, labels = _tokens(self.rank, step - 1, self.device)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = self.model(tokens, labels)
            loss.backward()
            self.optimizer.step()
            after_step(step, float(loss.detach()))

    def verification_state(self) -> dict[str, list[torch.Tensor]]:
        model = [parameter for parameter in self.model.parameters()]
        optimizer: list[torch.Tensor] = []
        for parameter in self.model.parameters():
            state = self.optimizer.state.get(parameter, {})
            for key in sorted(state):
                value = state[key]
                if isinstance(value, torch.Tensor):
                    optimizer.append(value)
        return {
            "model": clone_tensors(model),
            "optimizer": clone_tensors(optimizer),
        }

    def framework_state(self) -> dict[str, Any]:
        return {
            "absolute_step": int(self._extra_state["absolute_step"]),
            "step_count": int(self.handle.step_count),
        }

    def close(self) -> None:
        self.handle.close()
        if dist.is_initialized():
            dist.destroy_process_group()
