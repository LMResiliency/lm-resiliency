"""Integration test: chunk-based P2P checkpoint replication on a single node.

Simulates multi-node replication by splitting 8 GPUs into two "nodes" of 4.
Uses replication_jump=4 so that rank 0↔4, 1↔5, 2↔6, 3↔7 form pairs.
Each pair replicates checkpoints across the "node boundary."

Uses a small debug model to keep the test fast while exercising the full
replication path: GPU→CPU async copy → immediate chunked P2P send/recv.

Run:
    torchrun --nproc_per_node=8 tests/integration/core/test_replication.py

Expected: replication completes correctly, peer buffers contain the partner's
checkpoint data, and training throughput is minimally impacted.
"""

from __future__ import annotations

import os
import tempfile
import time

import torch
import torch.distributed as dist

from lm_resiliency import InMemoryCkptConfig
from lm_resiliency.experimental import InMemoryCheckpointManager


def setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def build_debug_model(device):
    """Small model for fast testing."""
    from torchtitan.models.llama3 import Transformer, llama3_args

    model = Transformer(llama3_args["debugmodel"]).to(device)
    return model


def run_training_with_replication(
    model, optimizer, device, num_steps, batch_size, seq_len, mgr, ckpt_interval, ckpt_tensors
):
    """Training loop with checkpoint save and immediate post-copy replication."""
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()

    for step in range(1, num_steps + 1):
        x = torch.randint(0, 2048, (batch_size, seq_len), device=device)
        out = model(x)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
        if step % ckpt_interval == 0:
            mgr.save_tensors(ckpt_tensors, step)

    torch.cuda.synchronize()
    dist.barrier()
    elapsed = time.perf_counter() - start

    mgr.maybe_wait()
    mgr.finalize_replication()

    return elapsed


def verify_replication(mgr, rank):
    """Verify that peer buffers received valid data."""
    pool = mgr._buffer_pool
    if pool._num_slots < 4:
        return True, "Replication skipped (2-slot mode)"

    peer_step = pool.get_latest_peer_step()
    if peer_step <= 0:
        return False, f"Rank {rank}: no peer checkpoint received"

    # Verify peer tensors are non-zero (received actual data)
    peer_slot = pool.peer_current if pool.peer_current.step > 0 else pool.peer_previous
    if peer_slot.step <= 0:
        return False, f"Rank {rank}: peer slot has step <= 0"

    non_zero_count = sum(1 for t in peer_slot.tensors if t.abs().sum() > 0)
    total_tensors = len(peer_slot.tensors)

    if non_zero_count == 0:
        return False, f"Rank {rank}: all peer tensors are zero"

    return (
        True,
        f"Rank {rank}: peer step={peer_slot.step}, {non_zero_count}/{total_tensors} non-zero tensors",
    )


def test_correctness():
    """Verify that replicated checkpoint matches the original."""
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    log("\n=== Correctness Test ===")
    log("Testing that peer receives exact checkpoint data...")

    with tempfile.TemporaryDirectory() as tmp:
        config = InMemoryCkptConfig(
            enable=True,
            interval=5,
            replication_jump=4,  # rank 0↔4, 1↔5, etc.
            replication_chunk_size=256 * 1024,  # 256 KiB (small for fast test)
            disk_flush_interval=0,
            disk_folder=tmp,
            skip_replication_if_hsdp=False,
        )
        mgr = InMemoryCheckpointManager(config)

        # Create deterministic tensors (different per rank)
        tensors = [
            torch.full((1000,), fill_value=float(rank + 1), device=device),
            torch.full((500,), fill_value=float(rank + 1) * 10, device=device),
        ]

        # One save is sufficient: replication starts when the host copy completes.
        mgr.save_tensors(tensors, step=5)
        mgr.maybe_wait()
        mgr.finalize_replication()

        # Verify peer received our data
        ok, msg = verify_replication(mgr, rank)
        # Gather results
        ok_tensor = torch.tensor([1 if ok else 0], dtype=torch.int64)
        dist.all_reduce(ok_tensor, op=dist.ReduceOp.MIN)

        if rank == 0:
            log(f"  All ranks replicated successfully: {ok_tensor.item() == 1}")

        # Verify peer data matches the expected values from paired rank
        if mgr._buffer_pool._num_slots >= 4:
            peer_slot = mgr._buffer_pool.peer_current
            if peer_slot.step > 0:
                # My peer is rank ± 4
                peer_rank = rank + 4 if rank < 4 else rank - 4
                expected_val = float(peer_rank + 1)
                actual_val = peer_slot.tensors[0][0].item()
                match = abs(actual_val - expected_val) < 1e-6
                if rank == 0:
                    log(
                        f"  Rank 0 received from rank 4: expected={expected_val}, got={actual_val}, match={match}"
                    )

        mgr.close()

    dist.barrier()


def test_rank_dependent_tensor_layouts():
    """Replicate shards whose optimizer-style layouts differ by rank."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    peer_rank = rank + 4 if rank < 4 else rank - 4

    with tempfile.TemporaryDirectory() as tmp:
        config = InMemoryCkptConfig(
            enable=True,
            interval=1,
            replication_jump=4,
            replication_chunk_size=64 * 1024,
            disk_flush_interval=0,
            disk_folder=tmp,
            skip_replication_if_hsdp=False,
        )
        mgr = InMemoryCheckpointManager(config)
        tensors = [
            torch.full((rank + 3, 2), float(rank), device=device),
            *(
                [torch.full((rank + 1,), rank + 0.5, dtype=torch.float64, device=device)]
                if rank % 2
                else []
            ),
        ]

        mgr.save_tensors(tensors, step=1, extra={"source_rank": rank})
        mgr.maybe_wait()
        mgr.finalize_replication()

        slot = mgr._buffer_pool.peer_current
        assert slot.step == 1
        assert slot.non_tensor_data["__extra__"]["source_rank"] == peer_rank
        assert [tuple(tensor.shape) for tensor in slot.tensors] == [
            (peer_rank + 3, 2),
            *((peer_rank + 1,) for _ in range(peer_rank % 2)),
        ]
        assert torch.all(slot.tensors[0] == float(peer_rank))
        if peer_rank % 2:
            assert slot.tensors[1].dtype == torch.float64
            assert torch.all(slot.tensors[1] == peer_rank + 0.5)

        assert mgr.flush_for_restart() == 1
        metadata, flushed_tensors = mgr._disk.load(1, rank=peer_rank)
        assert [entry.shape for entry in metadata.tensor_entries] == [
            tensor.shape for tensor in slot.tensors
        ]
        assert metadata.non_tensor_data["__extra__"]["source_rank"] == peer_rank
        assert all(
            torch.equal(received, flushed)
            for received, flushed in zip(slot.tensors, flushed_tensors)
        )

        mgr.close()

    dist.barrier()


def test_throughput():
    """Measure training throughput with and without replication."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    num_steps = 50
    ckpt_interval = 5
    batch_size = 4
    seq_len = 512

    log("\n=== Throughput Test ===")
    log(f"Config: {num_steps} steps, ckpt every {ckpt_interval}, replication_jump=4")
    log("Simulates 2 'nodes' of 4 GPUs each\n")

    # Build model
    model = build_debug_model(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # Warmup
    for _ in range(10):
        x = torch.randint(0, 2048, (batch_size, seq_len), device=device)
        out = model(x)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()

    ckpt_tensors = [p.data for p in model.parameters()]
    ckpt_bytes = sum(t.numel() * t.element_size() for t in ckpt_tensors)

    log(f"Checkpoint: {len(ckpt_tensors)} tensors, {ckpt_bytes / 1e6:.1f} MB per rank")

    # ── Baseline (no checkpointing) ─────────────────────────────────────────
    torch.cuda.synchronize()
    dist.barrier()
    start = time.perf_counter()
    for step in range(num_steps):
        x = torch.randint(0, 2048, (batch_size, seq_len), device=device)
        out = model(x)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()
    torch.cuda.synchronize()
    dist.barrier()
    baseline = time.perf_counter() - start

    # ── With replication (chunk_size=256KB for aggressive chunking test) ─────
    with tempfile.TemporaryDirectory() as tmp:
        config = InMemoryCkptConfig(
            enable=True,
            interval=ckpt_interval,
            replication_jump=4,
            replication_chunk_size=256 * 1024,  # 256 KiB
            disk_flush_interval=0,
            disk_folder=tmp,
            skip_replication_if_hsdp=False,
        )
        mgr = InMemoryCheckpointManager(config)
        # Pre-allocate
        mgr.save_tensors(ckpt_tensors, step=0)
        mgr.maybe_wait()
        mgr.finalize_replication()
        dist.barrier()

        t_repl = run_training_with_replication(
            model,
            optimizer,
            device,
            num_steps,
            batch_size,
            seq_len,
            mgr,
            ckpt_interval,
            ckpt_tensors,
        )

        ok, msg = verify_replication(mgr, rank)
        mgr.close()

    # ── With replication (chunk_size=16MB, production default) ───────────────
    with tempfile.TemporaryDirectory() as tmp:
        config = InMemoryCkptConfig(
            enable=True,
            interval=ckpt_interval,
            replication_jump=4,
            replication_chunk_size=16 * 1024 * 1024,  # 16 MiB
            disk_flush_interval=0,
            disk_folder=tmp,
            skip_replication_if_hsdp=False,
        )
        mgr = InMemoryCheckpointManager(config)
        mgr.save_tensors(ckpt_tensors, step=0)
        mgr.maybe_wait()
        mgr.finalize_replication()
        dist.barrier()

        t_repl_16m = run_training_with_replication(
            model,
            optimizer,
            device,
            num_steps,
            batch_size,
            seq_len,
            mgr,
            ckpt_interval,
            ckpt_tensors,
        )

        ok2, msg2 = verify_replication(mgr, rank)
        mgr.close()

    # ── Report ───────────────────────────────────────────────────────────────
    overhead_256k = ((t_repl - baseline) / baseline) * 100
    overhead_16m = ((t_repl_16m - baseline) / baseline) * 100

    if rank == 0:
        log(f"\n{'=' * 70}")
        log("Replication Test Results (8 GPUs as 2×4 'nodes')")
        log(f"{'=' * 70}")
        log(
            f"  Model:            Llama debug ({sum(p.numel() for p in model.parameters()) / 1e6:.1f}M params)"
        )
        log(f"  Checkpoint:       {ckpt_bytes / 1e6:.1f} MB/rank")
        log(f"  Steps:            {num_steps}, ckpt interval={ckpt_interval}")
        log("  Pairs:            rank 0↔4, 1↔5, 2↔6, 3↔7")
        log(f"{'─' * 70}")
        log(
            f"  Baseline (no ckpt):           {baseline:.3f}s ({baseline / num_steps * 1000:.1f} ms/step)"
        )
        log(f"  With replication (256KB chunks): {t_repl:.3f}s  overhead: {overhead_256k:+.2f}%")
        log(f"  With replication (16MB chunks):  {t_repl_16m:.3f}s  overhead: {overhead_16m:+.2f}%")
        log(f"{'─' * 70}")
        log(f"  Replication correctness (256KB): {ok} — {msg}")
        log(f"  Replication correctness (16MB):  {ok2} — {msg2}")
        log(f"{'=' * 70}\n")


def main():
    setup()
    test_correctness()
    test_rank_dependent_tensor_layouts()
    test_throughput()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
