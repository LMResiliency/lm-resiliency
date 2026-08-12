"""Run lm-resiliency inside TorchTitan's production Trainer loop.

The job uses TorchTitan's Llama 3 debug model and deterministic synthetic token
data. TorchTitan owns model construction, parallelization, forward/backward,
optimizer stepping, scheduling, and checkpoint loading.

Run on one eight-GPU host:

    torchrun --standalone --nproc-per-node=8 --module \
      examples.production_loops.torchtitan \
      --artifact-dir /tmp/torchtitan-production-loop

Run the same program on two eight-GPU hosts:

    torchrun --nnodes=2 --nproc-per-node=8 --module \
      --node-rank=$NODE_RANK \
      --master-addr=$MASTER_ADDR --master-port=29800 \
      examples.production_loops.torchtitan \
      --artifact-dir /tmp/torchtitan-production-loop

Add ``--inject-fault`` to exercise localization and checkpoint rejection.
"""

from __future__ import annotations

import argparse
import copy
import json
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.distributed as dist

from examples.production_loops._common import (
    ReplayFaultCampaign,
    add_run_arguments,
)
from lm_resiliency import InMemoryCkptConfig, ReplayHarnessConfig, enable_resiliency


class SyntheticTokenLoader:
    """Stateful deterministic token source used by the real TorchTitan Trainer."""

    def __init__(
        self,
        *,
        dp_rank: int,
        batch_size: int,
        sequence_length: int,
        vocabulary_size: int,
        campaign: ReplayFaultCampaign,
    ) -> None:
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.campaign = campaign
        self.index = 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
            self.campaign.start_step(self.index + 1)
            generator = torch.Generator().manual_seed(1_000_003 * self.dp_rank + self.index)
            tokens = torch.randint(
                0,
                self.vocabulary_size,
                (self.batch_size, self.sequence_length + 1),
                generator=generator,
            )
            self.index += 1
            yield {"input": tokens[:, :-1]}, tokens[:, 1:]

    def state_dict(self) -> dict[str, int]:
        return {"index": self.index}

    def load_state_dict(self, state_dict: dict[str, int]) -> None:
        self.index = int(state_dict["index"])


def _build_synthetic_dataloader(
    *,
    dp_rank: int,
    job_config: Any,
    campaign: ReplayFaultCampaign,
    **_: Any,
) -> SyntheticTokenLoader:
    return SyntheticTokenLoader(
        dp_rank=dp_rank,
        batch_size=job_config.training.local_batch_size,
        sequence_length=job_config.training.seq_len,
        vocabulary_size=2048,
        campaign=campaign,
    )


def _register_train_spec(campaign: ReplayFaultCampaign) -> str:
    from functools import partial

    from torchtitan.models.llama3 import get_train_spec, llama3_args
    from torchtitan.protocols.train_spec import register_train_spec

    name = "lm_resiliency_llama3_production_loop"
    base = get_train_spec()
    spec = replace(
        base,
        model_args={"debugmodel": copy.deepcopy(llama3_args["debugmodel"])},
        build_dataloader_fn=partial(
            _build_synthetic_dataloader,
            campaign=campaign,
        ),
        build_tokenizer_fn=None,
        build_validator_fn=None,
        state_dict_adapter=None,
    )
    register_train_spec(name, spec)
    return name


def _trainer_type(campaign: ReplayFaultCampaign):
    from torchtitan.train import Trainer

    class ProductionResilientTrainer(Trainer):
        resiliency = None

        def train(self) -> None:
            self.resiliency = enable_resiliency(
                self,
                interval=1,
                checkpoint=InMemoryCkptConfig(
                    enable=True,
                    interval=1,
                    replication_jump=max(1, dist.get_world_size() // 2),
                    disk_flush_interval=0,
                    disk_folder=str(self._lm_resiliency_artifact_dir / "gemini"),
                ),
                replay=ReplayHarnessConfig(
                    check_interval=1,
                    rotate_layers=False,
                    enable_temporal=False,
                    scale_factors=[],
                    straggler_min_slowdown_ratio=100.0,
                    straggler_min_slowdown_ms=10_000.0,
                ),
                fault_callback=campaign.record_fault,
                orchestration=campaign.orchestration,
            )
            campaign.bind(self.resiliency)
            super().train()

    return ProductionResilientTrainer


def _job_config(
    artifact_dir: Path,
    model_name: str,
    world_size: int,
    steps: int,
):
    from torchtitan.config import JobConfig

    config = JobConfig()
    config.job.description = "lm-resiliency production-loop validation"
    config.job.dump_folder = str(artifact_dir / "torchtitan")
    config.model.name = model_name
    config.model.flavor = "debugmodel"
    config.training.steps = steps
    config.training.local_batch_size = 2
    config.training.global_batch_size = 2 * world_size
    config.training.seq_len = 32
    config.training.dtype = "bfloat16"
    config.optimizer.implementation = "fused"
    config.lr_scheduler.warmup_steps = 0
    config.parallelism.data_parallel_replicate_degree = world_size
    config.parallelism.data_parallel_shard_degree = 1
    config.activation_checkpoint.mode = "none"
    config.checkpoint.enable = False
    config.validation.enable = False
    config.metrics.log_freq = 100
    return config


def main(*, inject_fault_by_default: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_run_arguments(
        parser,
        inject_fault_by_default=inject_fault_by_default,
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    environment = __import__("os").environ
    world_size = int(environment["WORLD_SIZE"])
    campaign = ReplayFaultCampaign.from_args(
        args,
        rank=int(environment["RANK"]),
        world_size=world_size,
    )
    model_name = _register_train_spec(campaign)
    trainer_cls = _trainer_type(campaign)
    trainer = trainer_cls(
        _job_config(
            args.artifact_dir,
            model_name,
            world_size,
            args.steps,
        )
    )
    trainer._lm_resiliency_artifact_dir = args.artifact_dir

    try:
        trainer.train()
        handle = trainer.resiliency
        if handle is None:
            raise AssertionError("TorchTitan Trainer did not retain the resiliency handle")
        if trainer.step != args.steps or handle.step_count != args.steps:
            raise AssertionError(
                f"step mismatch: trainer={trainer.step}, resiliency={handle.step_count}"
            )
        if trainer.lr_schedulers.schedulers[0].last_epoch != args.steps:
            raise AssertionError("TorchTitan scheduler did not advance through the production loop")
        handle.ckpt_manager.maybe_wait()
        result = handle.replay_harness.last_result
        campaign.validate(
            handle,
            result,
            {"embedding", "hidden", "output", "optimizer"},
        )

        summary = {
            "framework": "torchtitan",
            "framework_loop": "torchtitan.train.Trainer.train",
            "model": "torchtitan Llama 3 debugmodel",
            "world_size": dist.get_world_size(),
            "steps": trainer.step,
            "resiliency_steps": handle.step_count,
            "checkpoint_step": handle.ckpt_manager._last_saved_step,
            "checked_recipes": sorted(result.checked_recipe_ids),
            "dataloader_index": trainer.dataloader.index,
            **campaign.summary(),
        }
        if dist.get_rank() == 0:
            (args.artifact_dir / "torchtitan-production-loop.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        campaign.close()
        if trainer.resiliency is not None:
            trainer.resiliency.close()
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
