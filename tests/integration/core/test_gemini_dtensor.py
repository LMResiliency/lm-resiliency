"""GEMINI checkpoints FSDP2 DTensor params (torchtitan's case) — 2-rank local shards.

FSDP2 (torch ``fully_shard``) makes each parameter a **DTensor** sharded across ranks.
GEMINI's save_tensors fast path checkpoints each rank's **local shard** (``to_local()``)
— a plain-tensor view — so the async copy, flush, and in-place recovery copy never run a
distributed op. This test shards a model across 2 ranks (each param's local shard is a
real fraction of the whole), checkpoints via save_tensors, flushes to node-local, corrupts
the live shards, reloads, and verifies each rank's local shard is restored bitwise.

Run:  torchrun --nproc_per_node=2 tests/integration/core/test_gemini_dtensor.py    (2 GPUs)
"""

from __future__ import annotations

import tempfile

import torch
import torch.distributed as dist
import torch.nn as nn


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    assert world == 2, "run with --nproc_per_node=2"
    torch.cuda.set_device(rank)

    from torch.distributed.fsdp import fully_shard

    from lm_resiliency.checkpointing.config import InMemoryCkptConfig
    from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager

    torch.manual_seed(0)  # same init on both ranks, then FSDP2 shards it
    model = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64)).cuda()
    fully_shard(model)  # → DTensor params, sharded across the 2 ranks
    params = list(model.parameters())
    is_dtensor = type(params[0]).__name__ == "DTensor"
    local0 = params[0].to_local()
    if rank == 0:
        print(
            f"param0: {type(params[0]).__name__}, global {tuple(params[0].shape)}, "
            f"local {tuple(local0.shape)}"
        )
    assert is_dtensor and local0.shape[0] < params[0].shape[0], "params not sharded (need DTensor)"

    tmp = tempfile.mkdtemp(prefix=f"lm_dtensor_r{rank}_")
    cfg = InMemoryCkptConfig(
        enable=True, interval=1, disk_folder=tmp, replication_jump=1, disk_flush_interval=0
    )
    mgr = InMemoryCheckpointManager(cfg)
    mgr.save_tensors(params, step=3)  # DTensor → local shard checkpointed
    mgr.maybe_wait()

    with torch.no_grad():
        ref = [p.to_local().detach().clone() for p in params]
        mgr.flush_for_restart()  # local shards → node-local
        for p in params:
            p.to_local().detach().zero_()  # corrupt live shards
        res = InMemoryCheckpointManager(cfg).load_tensors()  # fresh manager → node-local
        assert res is not None, "load_tensors returned None"
        loaded, step, _extra = res
        for p, latest in zip(params, loaded):
            p.to_local().detach().copy_(latest)  # in-place into the local shard
        ok = all(torch.allclose(p.to_local(), r) for p, r in zip(params, ref))

    # every rank must have restored its own local shard
    okt = torch.tensor([1 if (ok and step == 3) else 0])
    dist.all_reduce(okt, op=dist.ReduceOp.MIN)
    assert okt.item() == 1, (
        f"rank {rank}: DTensor local-shard round-trip failed (ok={ok}, step={step})"
    )
    if rank == 0:
        print(
            "GEMINI DTENSOR OK: FSDP2 local shards checkpointed + reloaded bitwise on both ranks."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
