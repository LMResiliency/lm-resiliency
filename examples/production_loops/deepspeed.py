"""Train a tiny causal LM through DeepSpeed's production engine lifecycle.

DeepSpeed owns distributed model execution, backward, ZeRO-2 optimizer state,
and optimizer stepping.
Only the deterministic token source is synthetic.

Run on one eight-GPU host:

    torchrun --rdzv-backend=lm_resiliency \
      --rdzv-endpoint=/tmp/lm-resiliency-deepspeed-rdzv \
      --rdzv-id=deepspeed-production \
      --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-deepspeed-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/production_loops/policies/resiliency.toml" \
      --nnodes=1:1 --nproc-per-node=8 --module \
      examples.production_loops.deepspeed \
      --artifact-dir /tmp/deepspeed-production-loop
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import deepspeed
import torch
import torch.distributed as dist
import torch.nn as nn

from examples.production_loops._common import add_run_arguments, require_resiliency_adapter

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_run_arguments(parser)
    args = parser.parse_args()
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    deepspeed.init_distributed(dist_backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(123)
    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE,
            "gradient_accumulation_steps": 1,
            "train_batch_size": MICRO_BATCH_SIZE * world_size,
            "zero_optimization": {"stage": 2},
            "bf16": {"enabled": True},
            "steps_per_print": 1_000,
        },
    )
    try:
        for step in range(args.steps):
            tokens, labels = _tokens(rank, step, engine.device)
            loss = engine(tokens, labels)
            engine.backward(loss)
            engine.step()

        require_resiliency_adapter()
        summary = {
            "framework": "deepspeed",
            "framework_loop": "DeepSpeedEngine.backward/step",
            "model": "tiny causal language model",
            "world_size": world_size,
            "steps": engine.global_steps,
            "zero_stage": 2,
            "resiliency_adapter_attached": True,
        }
        if rank == 0:
            (args.artifact_dir / "deepspeed-production-loop.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        engine.destroy()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
