"""TorchTitan Trainer driver for torchrun fault injection."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from typing import Any

import torch
import torch.distributed as dist

from examples.production_loops.torchtitan import _job_config, _register_train_spec
from lm_resiliency.integrations.torchtitan import enable_resiliency

from ..runtime import DriverConfig, clone_tensors, tensor_leaves


class _ObservedLoader:
    def __init__(
        self,
        loader: Any,
        *,
        trainer: Any,
        before_step: Callable[[int], None],
    ) -> None:
        self._loader = loader
        self._trainer = trainer
        self._before_step = before_step
        self._iterator: Iterator[Any] | None = None
        self._observed_step = -1

    @property
    def index(self) -> int:
        return int(self._loader.index)

    def __iter__(self) -> _ObservedLoader:
        self._iterator = iter(self._loader)
        return self

    def __next__(self) -> Any:
        step = int(self._trainer.step)
        if step != self._observed_step:
            self._before_step(step)
            self._observed_step = step
        if self._iterator is None:
            self._iterator = iter(self._loader)
        return next(self._iterator)

    def state_dict(self) -> dict[str, Any]:
        return self._loader.state_dict()

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self._loader.load_state_dict(state_dict)


class TorchTitanDriver:
    framework = "torchtitan"
    expected_recipes = {"hidden"}

    def __init__(self, config: DriverConfig) -> None:
        from torchtitan.train import Trainer

        self.rank = int(__import__("os").environ["RANK"])
        self.world_size = int(__import__("os").environ["WORLD_SIZE"])
        torch.manual_seed(123)
        model_name = _register_train_spec()
        self.trainer = Trainer(
            _job_config(
                config.campaign_dir / "torchtitan-runtime",
                model_name,
                self.world_size,
                config.total_steps,
            )
        )
        self.device = self.trainer.device
        self.handle = enable_resiliency(
            self.trainer,
            interval=1,
            ckpt_config=config.checkpoint,
            detection_config=config.replay,
            device=self.device,
            fault_callback=config.fault_callback,
            orchestration=config.orchestration,
            recovery_mode=config.recovery_mode,
        )
        self._latest_loss: float | None = None
        self._original_loss = self.trainer.loss_fn

        def observed_loss(*args: Any, **kwargs: Any) -> torch.Tensor:
            loss = self._original_loss(*args, **kwargs)
            self._latest_loss = float(loss.detach())
            return loss

        self.trainer.loss_fn = observed_loss
        self._original_scheduler_step = self.trainer.lr_schedulers.step

    def run(
        self,
        *,
        before_step: Callable[[int], None],
        after_step: Callable[[int, float | None], None],
        total_steps: int,
    ) -> None:
        del total_steps
        if self.trainer.dataloader is None:
            raise RuntimeError("TorchTitan fault validation requires a dataloader")
        self.trainer.dataloader = _ObservedLoader(
            self.trainer.dataloader,
            trainer=self.trainer,
            before_step=before_step,
        )

        def scheduler_step(*args: Any, **kwargs: Any) -> Any:
            result = self._original_scheduler_step(*args, **kwargs)
            after_step(int(self.trainer.step), self._latest_loss)
            return result

        self.trainer.lr_schedulers.step = scheduler_step
        self.trainer.train()

    def verification_state(self) -> dict[str, list[torch.Tensor]]:
        model = [
            parameter
            for model_part in self.trainer.model_parts
            for parameter in model_part.parameters()
        ]
        optimizer = tensor_leaves(self.trainer.optimizers.state_dict())
        return {
            "model": clone_tensors(model),
            "optimizer": clone_tensors(optimizer),
        }

    def framework_state(self) -> dict[str, Any]:
        scheduler_state = self.trainer.lr_schedulers.state_dict()
        dataloader = self.trainer.dataloader
        return {
            "dataloader_index": int(dataloader.index),
            "ntokens_seen": int(self.trainer.ntokens_seen),
            "scheduler_last_epoch": int(scheduler_state["last_epoch"]),
            "step_count": int(self.handle.step_count),
            "trainer_step": int(self.trainer.step),
        }

    def close(self) -> None:
        self.trainer.loss_fn = self._original_loss
        self.trainer.lr_schedulers.step = self._original_scheduler_step
        self.handle.close()
        self.trainer.close()
        if dist.is_initialized():
            dist.destroy_process_group()
