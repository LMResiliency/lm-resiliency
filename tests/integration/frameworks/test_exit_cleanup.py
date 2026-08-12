"""Validate automatic process-exit cleanup through real framework integrations.

Run one framework at a time:

    python tests/integration/frameworks/test_exit_cleanup.py \
      --case pytorch --artifact-dir /tmp/automatic-exit-cleanup

The parent launches two torchrun workers.
Each worker enables GEMINI and SCOUT, completes one training step, and exits
without calling ``close()`` or destroying its process group.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from lm_resiliency import InMemoryCkptConfig, ReplayHarnessConfig, enable_resiliency

_WORLD_SIZE = 2


class Block(nn.Module):
    def __init__(self, width: int = 32) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.linear = nn.Linear(width, width)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + torch.nn.functional.gelu(self.linear(self.norm(inputs)))


class Decoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([Block(), Block()])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = Decoder()

    @property
    def layers(self) -> nn.ModuleList:
        return self.decoder.layers

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


def _checkpoint_config(artifact_dir: Path, case: str, rank: int) -> InMemoryCkptConfig:
    return InMemoryCkptConfig(
        interval=1,
        replication_jump=1,
        disk_flush_interval=0,
        disk_folder=str(artifact_dir / f"checkpoints-{case}-{rank}"),
    )


def _replay_config() -> ReplayHarnessConfig:
    return ReplayHarnessConfig(
        check_interval=1,
        rotate_layers=False,
        enable_temporal=False,
        scale_factors=[],
        straggler_min_slowdown_ratio=100.0,
        straggler_min_slowdown_ms=10_000.0,
    )


def _observe_exit_cleanup(
    handle: Any,
    *,
    case: str,
    rank: int,
    artifact_dir: Path,
    step_owner: Any,
    step_attribute: str,
    original_step: Any,
) -> None:
    checkpoint_manager = getattr(
        handle,
        "ckpt_manager",
        getattr(handle, "_ckpt_manager", None),
    )
    replay_harness = getattr(
        handle,
        "replay_harness",
        getattr(handle, "_replay_harness", None),
    )
    if checkpoint_manager is None or replay_harness is None:
        raise AssertionError(f"{case}: GEMINI and SCOUT must both be active")

    observations = {
        "checkpoint_closed": False,
        "replay_closed": False,
    }
    original_checkpoint_close = checkpoint_manager.close
    original_replay_close = replay_harness.remove_hooks
    original_handle_close = handle.close

    def close_checkpoint() -> None:
        original_checkpoint_close()
        observations["checkpoint_closed"] = True

    def close_replay() -> None:
        original_replay_close()
        observations["replay_closed"] = True

    checkpoint_manager.close = close_checkpoint
    replay_harness.remove_hooks = close_replay

    def close_and_record() -> None:
        result = {
            "case": case,
            "rank": rank,
            "status": "error",
            **observations,
        }
        try:
            original_handle_close()
            result.update(
                {
                    **observations,
                    "handle_closed": bool(
                        getattr(handle, "closed", getattr(handle, "_closed", False))
                    ),
                    "step_hook_restored": (
                        not handle._hooks
                        if hasattr(handle, "_hooks")
                        else getattr(step_owner, step_attribute) == original_step
                    ),
                    "status": "passed",
                }
            )
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            artifact_dir.mkdir(parents=True, exist_ok=True)
            marker = artifact_dir / f"{case}-rank-{rank}.json"
            marker.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    handle.close = close_and_record


def _common_enable_kwargs(
    artifact_dir: Path,
    case: str,
    rank: int,
) -> dict[str, Any]:
    return {
        "interval": 1,
        "checkpoint": _checkpoint_config(artifact_dir, case, rank),
        "replay": _replay_config(),
    }


def _build_pytorch(
    device: torch.device,
    artifact_dir: Path,
    *,
    torchtitan: bool,
) -> tuple[Any, torch.optim.Optimizer, Any]:
    model = DistributedDataParallel(
        Model().to(device),
        device_ids=[device.index],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    kwargs: dict[str, Any] = _common_enable_kwargs(
        artifact_dir,
        "torchtitan" if torchtitan else "pytorch",
        dist.get_rank(),
    )
    if torchtitan:
        from torchtitan.distributed import ParallelDims

        parallel_dims = ParallelDims(
            dp_replicate=_WORLD_SIZE,
            dp_shard=1,
            cp=1,
            tp=1,
            pp=1,
            ep=1,
            etp=1,
            world_size=_WORLD_SIZE,
        )
        parallel_dims.build_mesh()
        kwargs.update(
            framework="torchtitan",
            parallelism_info=parallel_dims,
        )
    handle = enable_resiliency(model, optimizer, **kwargs)
    return model, optimizer, handle


def _build_megatron(
    device: torch.device,
    artifact_dir: Path,
) -> tuple[Any, torch.optim.Optimizer, Any]:
    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(create_gloo_process_groups=True)
    model = Model().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    handle = enable_resiliency(
        [model],
        optimizer,
        **_common_enable_kwargs(artifact_dir, "megatron", dist.get_rank()),
    )
    return model, optimizer, handle


def _build_deepspeed(
    device: torch.device,
    artifact_dir: Path,
) -> tuple[Any, Any, Any]:
    import deepspeed

    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    engine, _, _, _ = deepspeed.initialize(
        model=model,
        optimizer=optimizer,
        config={
            "train_micro_batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,
            "train_batch_size": 2 * _WORLD_SIZE,
            "zero_optimization": {"stage": 2},
            "steps_per_print": 1_000,
        },
    )
    handle = enable_resiliency(
        engine,
        **_common_enable_kwargs(artifact_dir, "deepspeed", dist.get_rank()),
    )
    return engine, engine, handle


def _run_worker(case: str, artifact_dir: Path) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    if dist.get_world_size() != _WORLD_SIZE:
        raise RuntimeError(f"this integration test requires {_WORLD_SIZE} ranks")
    atexit.register(lambda: dist.destroy_process_group() if dist.is_initialized() else None)

    torch.manual_seed(20260809)
    device = torch.device("cuda", local_rank)
    if case == "pytorch":
        model, step_owner, handle = _build_pytorch(
            device,
            artifact_dir,
            torchtitan=False,
        )
        step_attribute = "step"
        original_step = step_owner.step
    elif case == "torchtitan":
        model, step_owner, handle = _build_pytorch(
            device,
            artifact_dir,
            torchtitan=True,
        )
        step_attribute = "step"
        original_step = step_owner.step
    elif case == "megatron":
        model, step_owner, handle = _build_megatron(device, artifact_dir)
        step_attribute = "step"
        original_step = handle._original_step
    else:
        model, step_owner, handle = _build_deepspeed(device, artifact_dir)
        step_attribute = handle._step_attribute
        original_step = handle._original_step

    if case in {"pytorch", "torchtitan"}:
        original_step = step_owner.step

    _observe_exit_cleanup(
        handle,
        case=case,
        rank=dist.get_rank(),
        artifact_dir=artifact_dir,
        step_owner=step_owner,
        step_attribute=step_attribute,
        original_step=original_step,
    )

    inputs = torch.linspace(-1.0, 1.0, 2 * 4 * 32, device=device).reshape(2, 4, 32)
    if case == "deepspeed":
        loss = model(inputs).float().square().mean()
        model.backward(loss)
        model.step()
    else:
        step_owner.zero_grad(set_to_none=True)
        model(inputs).float().square().mean().backward()
        step_owner.step()

    dist.barrier()
    # Deliberately omit handle.close() and dist.destroy_process_group().


def _run_parent(case: str, artifact_dir: Path) -> None:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    for marker in artifact_dir.glob(f"{case}-rank-*.json"):
        marker.unlink()
    for checkpoint_dir in artifact_dir.glob(f"checkpoints-{case}-*"):
        shutil.rmtree(checkpoint_dir)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={_WORLD_SIZE}",
        __file__,
        "--worker",
        "--case",
        case,
        "--artifact-dir",
        str(artifact_dir),
    ]
    environment = os.environ.copy()
    root = str(Path(__file__).parents[3])
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in (root, environment.get("PYTHONPATH")) if part
    )
    subprocess.run(
        command,
        cwd=root,
        env=environment,
        check=True,
        timeout=300,
    )

    results = []
    for rank in range(_WORLD_SIZE):
        marker = artifact_dir / f"{case}-rank-{rank}.json"
        if not marker.exists():
            raise AssertionError(f"{case}: rank {rank} produced no exit-cleanup marker")
        result = json.loads(marker.read_text())
        expected = {
            "status": "passed",
            "checkpoint_closed": True,
            "replay_closed": True,
            "handle_closed": True,
            "step_hook_restored": True,
        }
        for key, value in expected.items():
            if result.get(key) != value:
                raise AssertionError(
                    f"{case}: rank {rank} reported {key}={result.get(key)!r}, "
                    f"expected {value!r}: {result}"
                )
        results.append(result)

    print(
        f"PASS automatic-exit-cleanup {case}: {json.dumps(results, sort_keys=True)}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        choices=("pytorch", "torchtitan", "megatron", "deepspeed"),
        required=True,
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()

    if args.worker:
        _run_worker(args.case, args.artifact_dir)
    else:
        _run_parent(args.case, args.artifact_dir)


if __name__ == "__main__":
    main()
