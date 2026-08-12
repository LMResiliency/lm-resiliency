"""Train a tiny causal LM through a native PyTorch DDP training loop.

PyTorch owns distributed gradient synchronization through DDP.
The application owns forward, backward, optimizer stepping, and token loading.
Only the deterministic token source is synthetic.

Run on one eight-GPU host:

    torchrun --standalone --nproc-per-node=8 --module \
      examples.production_loops.pytorch \
      --artifact-dir /tmp/pytorch-production-loop

Run the same program on two eight-GPU hosts:

    torchrun --nnodes=2 --nproc-per-node=8 --module \
      --node-rank=$NODE_RANK \
      --master-addr=$MASTER_ADDR --master-port=29803 \
      examples.production_loops.pytorch \
      --artifact-dir /tmp/pytorch-production-loop

Add ``--inject-fault`` to exercise localization and checkpoint rejection.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from examples.production_loops._common import (
    ReplayFaultCampaign,
    add_run_arguments,
)
from lm_resiliency import InMemoryCkptConfig, ReplayHarnessConfig, enable_resiliency

VOCABULARY_SIZE = 256
SEQUENCE_LENGTH = 16
MICRO_BATCH_SIZE = 2


class CausalBlock(nn.Module):
    def __init__(self, hidden_size: int = 64, heads: int = 4) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(hidden_size)
        self.attention = nn.MultiheadAttention(
            hidden_size,
            heads,
            dropout=0.0,
            batch_first=True,
        )
        self.mlp_norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

    def forward(
        self,
        hidden: torch.Tensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        attended, _ = self.attention(
            normalized,
            normalized,
            normalized,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = hidden + attended
        return hidden + self.mlp(self.mlp_norm(hidden))


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 64, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCABULARY_SIZE, hidden_size)
        self.layers = nn.ModuleList([CausalBlock(hidden_size) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, VOCABULARY_SIZE, bias=False)

    def forward(
        self,
        tokens: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        hidden = self.embed(tokens)
        causal_mask = torch.triu(
            torch.ones(
                SEQUENCE_LENGTH,
                SEQUENCE_LENGTH,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, causal_mask)
        logits = self.output(self.final_norm(hidden))
        return torch.nn.functional.cross_entropy(
            logits.reshape(-1, VOCABULARY_SIZE),
            labels.reshape(-1),
        )


def _tokens(rank: int, step: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1_000_003 * rank + step)
    values = torch.randint(
        0,
        VOCABULARY_SIZE,
        (MICRO_BATCH_SIZE, SEQUENCE_LENGTH + 1),
        generator=generator,
    ).to(device)
    return values[:, :-1], values[:, 1:]


def main(*, inject_fault_by_default: bool = False) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_run_arguments(
        parser,
        inject_fault_by_default=inject_fault_by_default,
    )
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size < 3:
        raise RuntimeError("SCOUT localization requires at least three DDP replicas")
    campaign = ReplayFaultCampaign.from_args(
        args,
        rank=rank,
        world_size=world_size,
    )

    torch.manual_seed(123)
    model = DistributedDataParallel(
        TinyCausalLM().to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    resiliency = enable_resiliency(
        model,
        optimizer,
        interval=1,
        checkpoint=InMemoryCkptConfig(
            enable=True,
            interval=1,
            replication_jump=max(1, world_size // 2),
            disk_flush_interval=0,
            disk_folder=str(args.artifact_dir / "gemini"),
        ),
        replay=ReplayHarnessConfig(
            check_interval=1,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
            straggler_min_slowdown_ratio=100.0,
            straggler_min_slowdown_ms=10_000.0,
        ),
        device=device,
        fault_callback=campaign.record_fault,
        orchestration=campaign.orchestration,
    )
    campaign.bind(resiliency)

    try:
        for step in range(args.steps):
            campaign.start_step(step + 1)
            tokens, labels = _tokens(rank, step, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, labels)
            loss.backward()
            optimizer.step()

        resiliency.ckpt_manager.maybe_wait()
        result = resiliency.replay_harness.last_result
        if resiliency.step_count != args.steps:
            raise AssertionError("PyTorch and lm-resiliency optimizer steps did not remain aligned")
        campaign.validate(
            resiliency,
            result,
            {"embedding", "hidden", "output", "optimizer"},
        )

        summary = {
            "framework": "pytorch",
            "framework_loop": "DDP forward/backward/AdamW.step",
            "model": "tiny causal language model",
            "parallelism": "DDP",
            "world_size": world_size,
            "steps": args.steps,
            "resiliency_steps": resiliency.step_count,
            "checkpoint_step": resiliency.ckpt_manager._last_saved_step,
            "checked_recipes": sorted(result.checked_recipe_ids),
            **campaign.summary(),
        }
        if rank == 0:
            (args.artifact_dir / "pytorch-production-loop.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        campaign.close()
        resiliency.close()
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
