"""torchtitan FrameworkAdapter implementation."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.adapters import FrameworkAdapter, ParallelismInfo

if TYPE_CHECKING:
    from torchtitan.train import Trainer  # torchtitan 0.2.x moved Trainer here


class TorchTitanAdapter(FrameworkAdapter):
    """Adapter bridging torchtitan's Trainer with lm_resiliency.

    Extracts model, optimizer, LR scheduler, dataloader, and train state
    from a torchtitan Trainer instance.
    """

    def __init__(self, trainer: "Trainer") -> None:
        self._trainer = trainer

    @property
    def model(self) -> nn.Module:
        """Return the rank-local model surface used by GEMINI and SCOUT."""
        if len(self._trainer.model_parts) == 1:
            return self._trainer.model_parts[0]
        return nn.ModuleList(self._trainer.model_parts)

    @property
    def optimizer(self) -> Any:
        """Return TorchTitan's framework-owned optimizer container."""
        return self._trainer.optimizers

    def get_extra_state_dict(self) -> dict[str, Any]:
        """Capture TorchTitan-owned state not held by model or optimizer tensors."""
        state = {
            "lr_scheduler": self._trainer.lr_schedulers.state_dict(),
            "train_state": self._trainer.state_dict(),
        }
        if self._trainer.dataloader is not None:
            state["dataloader"] = self._trainer.dataloader.state_dict()
        return state

    def load_extra_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore TorchTitan-owned state after GEMINI reloads model and optimizer."""
        if "lr_scheduler" in state_dict:
            self._trainer.lr_schedulers.load_state_dict(state_dict["lr_scheduler"])
        if "train_state" in state_dict:
            self._trainer.load_state_dict(state_dict["train_state"])
        if "dataloader" in state_dict and self._trainer.dataloader is not None:
            self._trainer.dataloader.load_state_dict(state_dict["dataloader"])

    def get_state_dict(self) -> dict[str, Any]:
        from torch.distributed.checkpoint.state_dict import get_model_state_dict

        state: dict[str, Any] = {}
        for i, model_part in enumerate(self._trainer.model_parts):
            state[f"model_{i}"] = get_model_state_dict(model_part)
        state["optimizer"] = self._trainer.optimizers.state_dict()
        state.update(self.get_extra_state_dict())
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        from torch.distributed.checkpoint.state_dict import set_model_state_dict

        for i, model_part in enumerate(self._trainer.model_parts):
            key = f"model_{i}"
            if key in state_dict:
                set_model_state_dict(model_part, state_dict[key])
        if "optimizer" in state_dict:
            self._trainer.optimizers.load_state_dict(state_dict["optimizer"])
        self.load_extra_state_dict(state_dict)

    def get_parallelism_info(self) -> ParallelismInfo:
        pd = self._trainer.parallel_dims
        return ParallelismInfo(
            dp_replicate=pd.dp_replicate,
            dp_shard=pd.dp_shard,
            tp=pd.tp,
            pp=pd.pp,
            world_size=pd.world_size,
        )

    @property
    def rank(self) -> int:
        return dist.get_rank() if dist.is_initialized() else 0

    @property
    def world_size(self) -> int:
        return dist.get_world_size() if dist.is_initialized() else 1
