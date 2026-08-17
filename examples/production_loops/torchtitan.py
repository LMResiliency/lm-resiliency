"""Run TorchTitan's native production Trainer loop with injected resiliency.

The job uses TorchTitan's Llama 3 debug model and deterministic synthetic token
data. TorchTitan owns model construction, parallelization, forward/backward,
optimizer stepping, scheduling, and checkpoint loading.

Run on one eight-GPU host:

    export LM_RESILIENCY_RESTART_CONTEXT="${LM_RESILIENCY_RESTART_CONTEXT:-/tmp/lm-resiliency-torchtitan-context/context.json}"
    torchrun --rdzv-backend=lm_resiliency \
      --rdzv-endpoint=/tmp/lm-resiliency-torchtitan-rdzv \
      --rdzv-id=torchtitan-production \
      --rdzv-conf="store_type=file,node_id=node-a,active_nodes=node-a,\
local_world_size=8,\
worker_adapter=torchtitan,\
worker_config=$PWD/examples/production_loops/worker_resiliency.toml" \
      --nnodes=1:1 --nproc-per-node=8 --module \
      examples.production_loops.torchtitan \
      --artifact-dir /tmp/torchtitan-production-loop
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

from examples.production_loops._common import add_run_arguments, require_resiliency_adapter


class SyntheticTokenLoader:
    """Stateful deterministic token source used by the real TorchTitan Trainer."""

    def __init__(
        self,
        *,
        dp_rank: int,
        batch_size: int,
        sequence_length: int,
        vocabulary_size: int,
    ) -> None:
        self.dp_rank = dp_rank
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.vocabulary_size = vocabulary_size
        self.index = 0

    def __iter__(self) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        while True:
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
    **_: Any,
) -> SyntheticTokenLoader:
    return SyntheticTokenLoader(
        dp_rank=dp_rank,
        batch_size=job_config.training.local_batch_size,
        sequence_length=job_config.training.seq_len,
        vocabulary_size=2048,
    )


def _register_train_spec() -> str:
    from functools import partial

    from torchtitan.models.llama3 import get_train_spec, llama3_args
    from torchtitan.protocols.train_spec import register_train_spec

    name = "lm_resiliency_llama3_production_loop"
    base = get_train_spec()
    spec = replace(
        base,
        model_args={"debugmodel": copy.deepcopy(llama3_args["debugmodel"])},
        build_dataloader_fn=partial(_build_synthetic_dataloader),
        build_tokenizer_fn=None,
        build_validator_fn=None,
        state_dict_adapter=None,
    )
    register_train_spec(name, spec)
    return name


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_run_arguments(parser)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    environment = __import__("os").environ
    world_size = int(environment["WORLD_SIZE"])
    model_name = _register_train_spec()
    from torchtitan.train import Trainer

    trainer = Trainer(
        _job_config(
            args.artifact_dir,
            model_name,
            world_size,
            args.steps,
        )
    )
    try:
        trainer.train()
        require_resiliency_adapter()
        if trainer.step != args.steps:
            raise AssertionError(f"step mismatch: trainer={trainer.step}")
        if trainer.lr_schedulers.schedulers[0].last_epoch != args.steps:
            raise AssertionError("TorchTitan scheduler did not advance through the production loop")

        summary = {
            "framework": "torchtitan",
            "framework_loop": "torchtitan.train.Trainer.train",
            "model": "torchtitan Llama 3 debugmodel",
            "world_size": dist.get_world_size(),
            "steps": trainer.step,
            "dataloader_index": trainer.dataloader.index,
            "resiliency_adapter_attached": True,
        }
        if dist.get_rank() == 0:
            (args.artifact_dir / "torchtitan-production-loop.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        if dist.is_initialized():
            dist.barrier()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
