"""Measure native-PyTorch healthy-path overhead for one protection mode."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from lm_resiliency import InMemoryCkptConfig, ReplayHarnessConfig, enable_resiliency

VOCABULARY_SIZE = 256


class CausalBlock(nn.Module):
    def __init__(self, hidden_size: int, heads: int) -> None:
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

    def forward(self, hidden: torch.Tensor, causal_mask: torch.Tensor) -> torch.Tensor:
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
    def __init__(self, *, hidden_size: int, layers: int, heads: int) -> None:
        super().__init__()
        self.embed = nn.Embedding(VOCABULARY_SIZE, hidden_size)
        self.layers = nn.ModuleList([CausalBlock(hidden_size, heads) for _ in range(layers)])
        self.final_norm = nn.LayerNorm(hidden_size)
        self.output = nn.Linear(hidden_size, VOCABULARY_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden = self.embed(tokens)
        sequence_length = tokens.shape[1]
        causal_mask = torch.triu(
            torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=tokens.device,
            ),
            diagonal=1,
        )
        for layer in self.layers:
            hidden = layer(hidden, causal_mask)
        return self.output(self.final_norm(hidden))


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        raise ValueError("cannot compute a percentile from no samples")
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * fraction


def replication_jump(world_size: int) -> int:
    """Return the pairwise GEMINI replication jump for a supported world size."""
    if world_size < 1:
        raise ValueError("world size must be positive")
    if world_size > 1 and world_size % 2:
        raise ValueError("GEMINI healthy-path modes require an even world size")
    return max(1, world_size // 2)


def aggregate_step_latencies(rank_results: list[dict[str, Any]]) -> list[float]:
    """Collapse rank timings to the slowest latency for each synchronous step."""
    sample_counts = {len(result["step_times_ms"]) for result in rank_results}
    if len(sample_counts) != 1:
        raise ValueError("all ranks must report the same number of timed steps")
    sample_count = sample_counts.pop() if sample_counts else 0
    return [
        max(float(result["step_times_ms"][offset]) for result in rank_results)
        for offset in range(sample_count)
    ]


def _tokens(
    rank: int,
    step: int,
    *,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1_000_003 * rank + step)
    values = torch.randint(
        0,
        VOCABULARY_SIZE,
        (batch_size, sequence_length + 1),
        generator=generator,
    ).to(device)
    return values[:, :-1], values[:, 1:]


def _train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    rank: int,
    step: int,
    args: argparse.Namespace,
    device: torch.device,
) -> None:
    tokens, labels = _tokens(
        rank,
        step,
        batch_size=args.batch_size,
        sequence_length=args.sequence_length,
        device=device,
    )
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=device.type == "cuda",
    ):
        logits = model(tokens)
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, VOCABULARY_SIZE),
            labels.reshape(-1),
        )
    loss.backward()
    optimizer.step()


def _tree_tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(
            _tree_tensor_bytes(key) + _tree_tensor_bytes(item) for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return sum(_tree_tensor_bytes(item) for item in value)
    return 0


def _max_rss_bytes() -> int:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value if sys.platform == "darwin" else value * 1024


def _process_peak_rss_bytes(pid: int | None, *, proc_root: Path = Path("/proc")) -> int | None:
    """Read Linux's peak resident set for a live SCOUT daemon."""
    if pid is None:
        return None
    try:
        status = (proc_root / str(pid) / "status").read_text()
    except OSError:
        return None
    for line in status.splitlines():
        if line.startswith("VmHWM:"):
            fields = line.split()
            if len(fields) == 3 and fields[2] == "kB":
                return int(fields[1]) * 1024
    return None


def _oob_daemon_peak_rss_bytes(handle: Any) -> int | None:
    replay_harness = getattr(handle, "replay_harness", None)
    service = getattr(replay_harness, "_oob_service", None)
    return _process_peak_rss_bytes(getattr(service, "pid", None))


def _wait_for_oob_daemon(handle: Any) -> None:
    replay_harness = getattr(handle, "replay_harness", None)
    service = getattr(replay_harness, "_oob_service", None)
    wait_until_ready = getattr(service, "wait_until_ready", None)
    if not callable(wait_until_ready):
        raise RuntimeError("SCOUT benchmark could not access its OOB daemon readiness signal")
    wait_until_ready()


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_worktree_dirty() -> bool | None:
    completed = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        check=False,
        capture_output=True,
        text=True,
    )
    return bool(completed.stdout.strip()) if completed.returncode == 0 else None


def _recorded_worktree_dirty() -> bool | None:
    recorded = os.environ.get("LM_BENCHMARK_WORKTREE_DIRTY")
    if recorded is None:
        return _git_worktree_dirty()
    if recorded == "1":
        return True
    if recorded == "0":
        return False
    return None


def _run(args: argparse.Namespace) -> dict[str, Any]:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("--device cuda requires an available CUDA device")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        backend = "cpu:gloo,cuda:nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"

    dist.init_process_group(backend=backend)
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.manual_seed(args.seed)

    model: nn.Module = TinyCausalLM(
        hidden_size=args.hidden_size,
        layers=args.layers,
        heads=args.heads,
    ).to(device)
    model = DistributedDataParallel(
        model,
        device_ids=[local_rank] if device.type == "cuda" else None,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    enable_checkpoint = args.mode in {"gemini", "combined"}
    enable_detection = args.mode in {"scout", "combined"}
    effective_pin_memory = enable_checkpoint and device.type == "cuda" and args.pin_memory
    checkpoint_replication_jump = replication_jump(world_size) if enable_checkpoint else 1
    handle = None
    if enable_checkpoint or enable_detection:
        handle = enable_resiliency(
            model,
            optimizer,
            interval=args.interval,
            enable_checkpoint=enable_checkpoint,
            enable_detection=enable_detection,
            checkpoint=InMemoryCkptConfig(
                enable=enable_checkpoint,
                interval=args.interval,
                replication_jump=checkpoint_replication_jump,
                replication_chunk_size=args.replication_chunk_size,
                disk_flush_interval=0,
                pin_memory=effective_pin_memory,
            ),
            replay=ReplayHarnessConfig(
                check_interval=args.interval,
                rotate_layers=True,
                enable_temporal=False,
                all_to_all_policy=None,
                scale_factors=[],
                hang_stall_threshold_s=300.0,
            ),
            group=dist.group.WORLD,
            nccl_group=dist.group.WORLD,
            device=device,
        )

    try:
        for step in range(args.warmup_steps):
            _train_step(model, optimizer, rank, step, args, device)

        if handle is not None and handle.ckpt_manager is not None:
            handle.ckpt_manager.maybe_wait()
        if enable_detection:
            _wait_for_oob_daemon(handle)
        dist.barrier()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)

        step_times_ms: list[float] = []
        cuda_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        measured_started = time.perf_counter()
        for offset in range(args.steps):
            step = args.warmup_steps + offset
            if device.type == "cuda":
                started = torch.cuda.Event(enable_timing=True)
                completed = torch.cuda.Event(enable_timing=True)
                started.record()
                _train_step(model, optimizer, rank, step, args, device)
                completed.record()
                cuda_events.append((started, completed))
            else:
                started_at = time.perf_counter()
                _train_step(model, optimizer, rank, step, args, device)
                step_times_ms.append((time.perf_counter() - started_at) * 1000.0)

        if device.type == "cuda":
            torch.cuda.synchronize(device)
            step_times_ms = [started.elapsed_time(completed) for started, completed in cuda_events]
        measured_seconds = time.perf_counter() - measured_started
        dist.barrier()

        peak_parent_host_memory_bytes = _max_rss_bytes()
        peak_oob_host_memory_bytes = (
            _oob_daemon_peak_rss_bytes(handle) if enable_detection else None
        )
        rank_result = {
            "rank": rank,
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "duration_seconds": measured_seconds,
            "step_times_ms": step_times_ms,
            "peak_parent_host_memory_bytes": peak_parent_host_memory_bytes,
            "peak_oob_host_memory_bytes": peak_oob_host_memory_bytes,
            "peak_host_memory_bytes": peak_parent_host_memory_bytes
            + (peak_oob_host_memory_bytes or 0),
            "peak_gpu_memory_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        rank_results: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(rank_results, rank_result)

        if rank != 0:
            return {}
        complete_results = [result for result in rank_results if result is not None]
        device_names = [result["device_name"] for result in complete_results]
        if device.type == "cuda" and len(set(device_names)) != 1:
            raise RuntimeError(
                "healthy-path qualification requires equivalent GPU models on every rank: "
                f"{device_names}"
            )
        job_step_times = aggregate_step_latencies(complete_results)
        max_duration = max(result["duration_seconds"] for result in complete_results)
        tokens = args.steps * args.batch_size * args.sequence_length * world_size
        return {
            "schema_version": 1,
            "commit_sha": os.environ.get("GITHUB_SHA") or _git_revision(),
            "worktree_dirty": _recorded_worktree_dirty(),
            "mode": args.mode,
            "environment": {
                "host": platform.node(),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda_runtime": torch.version.cuda,
                "device": args.device,
                "device_name": device_names[0] if device.type == "cuda" else None,
                "device_names_by_rank": device_names,
            },
            "topology": {
                "hosts": 1,
                "world_size": world_size,
                "data_parallel": world_size,
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "context_parallel": 1,
                "sequence_parallel": 1,
                "expert_parallel": 1,
            },
            "workload": {
                "model": "native-pytorch-tiny-causal-lm",
                "seed": args.seed,
                "steps": args.steps,
                "warmup_steps": args.warmup_steps,
                "batch_size_per_rank": args.batch_size,
                "sequence_length": args.sequence_length,
                "hidden_size": args.hidden_size,
                "layers": args.layers,
                "heads": args.heads,
                "checkpoint_bytes_per_rank": _tree_tensor_bytes(model.state_dict())
                + _tree_tensor_bytes(optimizer.state_dict()),
            },
            "protection": {
                "checkpoint_interval": args.interval if enable_checkpoint else None,
                "replay_interval": args.interval if enable_detection else None,
                "pin_memory": effective_pin_memory if enable_checkpoint else None,
                "replication": enable_checkpoint and world_size > 1,
                "replication_chunk_size": (
                    args.replication_chunk_size if enable_checkpoint else None
                ),
                "qualification_boundary": (
                    "SCOUT healthy-path overhead only; exact localization requires at least three peers"
                    if enable_detection and world_size < 3
                    else None
                ),
            },
            "metrics": {
                "throughput_tokens_per_second": tokens / max_duration,
                "step_latency_p50_ms": percentile(job_step_times, 0.50),
                "step_latency_p95_ms": percentile(job_step_times, 0.95),
                "step_latencies_ms": job_step_times,
                "peak_parent_host_memory_bytes": max(
                    result["peak_parent_host_memory_bytes"] for result in complete_results
                ),
                "peak_oob_host_memory_bytes": (
                    max(
                        result["peak_oob_host_memory_bytes"]
                        for result in complete_results
                        if result["peak_oob_host_memory_bytes"] is not None
                    )
                    if any(
                        result["peak_oob_host_memory_bytes"] is not None
                        for result in complete_results
                    )
                    else None
                ),
                "peak_host_memory_bytes": max(
                    result["peak_host_memory_bytes"] for result in complete_results
                ),
                "peak_gpu_memory_bytes": (
                    max(result["peak_gpu_memory_bytes"] for result in complete_results)
                    if device.type == "cuda"
                    else None
                ),
            },
            "ranks": complete_results,
        }
    finally:
        if handle is not None:
            handle.close()
        dist.barrier()
        dist.destroy_process_group()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("baseline", "gemini", "scout", "combined"), required=True
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--replication-chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if min(args.steps, args.interval, args.batch_size, args.sequence_length, args.hidden_size) < 1:
        parser.error(
            "steps, interval, batch size, sequence length, and hidden size must be positive"
        )
    if args.warmup_steps < 0:
        parser.error("warmup steps cannot be negative")
    if args.hidden_size % args.heads:
        parser.error("hidden size must be divisible by heads")

    result = _run(args)
    if result:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
