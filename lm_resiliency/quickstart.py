"""Train and resume a tiny causal LM with native PyTorch on CPU.

This entry point is intentionally single-process so it can run on a laptop or
in CI. It exercises the complete application-owned training loop and GEMINI
checkpoint recovery. SCOUT replay and multi-rank localization require the
distributed production-loop examples in the source repository.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from lm_resiliency import InMemoryCkptConfig, enable_resiliency

VOCABULARY_SIZE = 128
SEQUENCE_LENGTH = 12
BATCH_SIZE = 4


class TinyBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_size)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.mlp(self.norm(hidden))


class TinyCausalLM(nn.Module):
    def __init__(self, hidden_size: int = 32, layers: int = 2) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCABULARY_SIZE, hidden_size)
        self.layers = nn.ModuleList([TinyBlock(hidden_size) for _ in range(layers)])
        self.output = nn.Linear(hidden_size, VOCABULARY_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        for layer in self.layers:
            hidden = layer(hidden)
        return self.output(hidden)


def _batch(step: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(10_000 + step)
    values = torch.randint(
        0,
        VOCABULARY_SIZE,
        (BATCH_SIZE, SEQUENCE_LENGTH + 1),
        generator=generator,
    )
    return values[:, :-1], values[:, 1:]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--interval", type=int, default=2)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("/tmp/lm-resiliency-quickstart/checkpoints"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be greater than zero")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than zero")

    torch.manual_seed(7)

    model = TinyCausalLM()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    resiliency = enable_resiliency(
        model,
        optimizer,
        interval=args.interval,
        enable_detection=False,
        checkpoint=InMemoryCkptConfig(
            disk_flush_interval=1,
            disk_folder=str(args.checkpoint_dir),
            pin_memory=False,
            verify_integrity=True,
        ),
        device=torch.device("cpu"),
    )

    start_step = resiliency.step_count
    loss_value: float | None = None
    for step in range(start_step, args.steps):
        tokens, labels = _batch(step)
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCABULARY_SIZE),
            labels.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach())

    checkpoint_step = resiliency.flush_for_restart()
    summary = {
        "checkpoint_step": checkpoint_step,
        "completed_step": resiliency.step_count,
        "final_loss": round(loss_value, 6) if loss_value is not None else None,
        "recovered_step": resiliency.recovered_step,
        "start_step": start_step,
    }
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
