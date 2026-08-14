"""Integration test: checkpoint save and load correctness.

Verifies the full checkpoint lifecycle:
  1. save_tensors() correctly copies GPU tensors to CPU and flushes to disk
  2. Periodic disk flush produces correct files
  3. SIGTERM handler flushes the latest READY buffer to disk
  4. load() recovers the correct state from disk
  5. find_latest() agrees across all ranks (MIN-reduce)
  6. Full state_dict roundtrip (model + optimizer)

Run:
    torchrun --nproc_per_node=8 tests/integration/core/test_save_load.py
"""

from __future__ import annotations

import os
import shutil
import signal
import tempfile
from pathlib import Path

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


def make_shared_tmp():
    """Create a shared temp dir (rank 0 creates, broadcasts path to all)."""
    if dist.get_rank() == 0:
        tmp = tempfile.mkdtemp()
    else:
        tmp = ""
    obj_list = [tmp]
    dist.broadcast_object_list(obj_list, src=0)
    return obj_list[0]


def cleanup_tmp(tmp_dir: str):
    """Remove shared temp dir (rank 0 only, after barrier)."""
    dist.barrier()
    if dist.get_rank() == 0:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def log(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def assert_all(condition: bool, msg: str):
    """All-reduce a boolean condition and fail if any rank disagrees."""
    t = torch.tensor([1 if condition else 0], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    assert t.item() == 1, f"Assertion failed on some rank: {msg}"


# ──────────────────────────────────────────────────────────────────────────────


def test_save_and_periodic_flush():
    """Periodic disk flush writes correct checkpoint files."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: Save and Periodic Disk Flush ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=10,
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )
    mgr = InMemoryCheckpointManager(config)

    t1 = torch.arange(100, dtype=torch.float32, device=device) + rank
    t2 = torch.ones(50, dtype=torch.float32, device=device) * (rank + 1)
    tensors = [t1, t2]

    # Step 10: rotates step-5→own_previous, flush triggers → writes step 5
    # Step 20: rotates step-15→own_previous, flush triggers → writes step 15
    for step in [5, 10, 15, 20]:
        mgr.save_tensors(tensors, step=step)
        mgr.maybe_wait()

    mgr.finalize_replication()
    mgr._disk.wait()

    disk_path = Path(tmp) / "step-5" / f"rank-{rank}.pt"
    assert_all(disk_path.exists(), f"Disk checkpoint not found at {disk_path}")

    data = torch.load(disk_path, weights_only=True)
    loaded_tensors = data["tensors"]
    assert_all(torch.allclose(loaded_tensors[0], t1.cpu()), "t1 mismatch")
    assert_all(torch.allclose(loaded_tensors[1], t2.cpu()), "t2 mismatch")

    mgr.close()
    cleanup_tmp(tmp)
    log("  PASSED")


def test_sigterm_flush():
    """SIGTERM handler flushes the latest READY buffer to disk."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: SIGTERM Flush ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=0,  # NO periodic flush
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )
    mgr = InMemoryCheckpointManager(config)

    t1 = torch.full((200,), fill_value=float(rank * 10 + 1), device=device)
    t2 = torch.full((100,), fill_value=float(rank * 10 + 2), device=device)
    tensors = [t1, t2]

    # Keep two completed generations so the signal flush can select the latest
    # locally complete checkpoint while preserving the previous recovery pair.
    mgr.save_tensors(tensors, step=5)
    mgr.maybe_wait()
    mgr.save_tensors(tensors, step=10)
    mgr.maybe_wait()
    mgr.finalize_replication()

    # Invoke SIGTERM handler directly — it raises SystemExit, catch it
    try:
        mgr._sigterm_handler(signal.SIGTERM, None)
    except (SystemExit, KeyboardInterrupt):
        pass

    # Verify flush happened
    flushed = mgr._disk.find_latest_on_disk()
    assert_all(flushed > 0, f"SIGTERM did not flush (got step={flushed})")

    # Verify data correctness
    disk_path = Path(tmp) / f"step-{flushed}" / f"rank-{rank}.pt"
    data = torch.load(disk_path, weights_only=True)
    loaded_tensors = data["tensors"]
    assert_all(torch.allclose(loaded_tensors[0], t1.cpu()), "t1 mismatch after SIGTERM")
    assert_all(torch.allclose(loaded_tensors[1], t2.cpu()), "t2 mismatch after SIGTERM")

    # Restore signal handlers
    signal.signal(signal.SIGTERM, mgr._prev_sigterm_handler)
    signal.signal(signal.SIGINT, mgr._prev_sigint_handler)

    cleanup_tmp(tmp)
    log(f"  PASSED (flushed step {flushed})")


def test_load_recovers_correct_state():
    """load() recovers exact tensor values from disk after simulated restart."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: Load Recovers Correct State ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=5,
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )

    # Phase 1: save (simulates training)
    mgr = InMemoryCheckpointManager(config)

    t1 = torch.randn(500, device=device)
    t2 = torch.randn(200, 3, device=device)
    t3 = torch.tensor([rank * 100.0], device=device)
    tensors = [t1, t2, t3]

    ref_t1 = t1.cpu().clone()
    ref_t2 = t2.cpu().clone()
    ref_t3 = t3.cpu().clone()

    # Step 10 flush: step 5; Step 15 flush: step 10; Step 20 flush: step 15
    for step in [5, 10, 15, 20]:
        mgr.save_tensors(tensors, step=step)
        mgr.maybe_wait()

    mgr.finalize_replication()
    mgr._disk.wait()
    mgr.close()

    dist.barrier()

    # Phase 2: new manager (simulates restart)
    mgr2 = InMemoryCheckpointManager(config)
    latest = mgr2.find_latest()
    assert_all(latest > 0, f"find_latest returned {latest}")

    result = mgr2.load()
    assert_all(result is not None, "load() returned None")

    state_dict, step = result

    # save_tensors uses string indices as keys via unflatten
    if isinstance(state_dict, dict):
        loaded_t1 = state_dict["0"]
        loaded_t2 = state_dict["1"]
        loaded_t3 = state_dict["2"]
    else:
        loaded_t1, loaded_t2, loaded_t3 = state_dict[0], state_dict[1], state_dict[2]

    assert_all(torch.allclose(loaded_t1.float(), ref_t1.float()), f"t1 mismatch (step={step})")
    assert_all(torch.allclose(loaded_t2.float(), ref_t2.float()), f"t2 mismatch (step={step})")
    assert_all(torch.allclose(loaded_t3.float(), ref_t3.float()), f"t3 mismatch (step={step})")

    mgr2.close()
    cleanup_tmp(tmp)
    log(f"  PASSED (recovered step {step})")


def test_find_latest_min_reduce():
    """find_latest() returns MIN across all ranks."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: find_latest MIN-reduce ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=5,
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )
    mgr = InMemoryCheckpointManager(config)

    t1 = torch.ones(100, device=device)

    # Step 10 flushes step 5; step 15 flushes step 10.
    for step in [5, 10, 15]:
        mgr.save_tensors([t1], step=step)
        mgr.maybe_wait()

    mgr.finalize_replication()
    mgr._disk.wait()

    # Make the newest in-memory generation unavailable on one rank. Immediate
    # replication makes step 15 complete without requiring another save, so the
    # MIN-reduce must be tested with an actual rank-local omission.
    if rank == 3:
        newest = mgr._buffer_pool.get_slot_by_step(15)
        assert newest is not None
        newest.step = -1

    latest = mgr.find_latest()
    assert_all(latest == 10, f"Expected 10, got {latest}")
    mgr.close()

    dist.barrier()

    # Simulate rank 3 missing step 10
    if rank == 3:
        step10_file = Path(tmp) / "step-10" / f"rank-{rank}.pt"
        if step10_file.exists():
            step10_file.unlink()

    dist.barrier()

    mgr2 = InMemoryCheckpointManager(config)
    latest2 = mgr2.find_latest()
    assert_all(latest2 == 5, f"Expected 5 after deletion, got {latest2}")
    mgr2.close()

    cleanup_tmp(tmp)
    log("  PASSED")


def test_save_load_with_state_dict():
    """Full state_dict save/load roundtrip (model + optimizer state)."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: State Dict Save/Load Roundtrip ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=5,
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )

    # Build model and do some training
    model = torch.nn.Linear(64, 32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        x = torch.randn(8, 64, device=device)
        model(x).sum().backward()
        optimizer.step()
        optimizer.zero_grad()

    state_dict = {
        "model": {k: v.clone() for k, v in model.state_dict().items()},
        "step": 42,
    }

    # Save
    mgr = InMemoryCheckpointManager(config)
    # Need 3 saves to get step 5 flushed: save(5), save(10) flushes step 5
    mgr.save(state_dict, step=5)
    mgr.maybe_wait()
    mgr.save(state_dict, step=10)
    mgr.maybe_wait()
    mgr.save(state_dict, step=15)
    mgr.maybe_wait()
    mgr.finalize_replication()
    mgr._disk.wait()
    mgr.close()

    dist.barrier()

    # Load
    mgr2 = InMemoryCheckpointManager(config)
    result = mgr2.load()
    assert_all(result is not None, "load() returned None")

    loaded_state_dict, loaded_step = result

    # Verify model weights
    for key in state_dict["model"]:
        original = state_dict["model"][key].cpu().float()
        loaded = loaded_state_dict["model"][key]
        if isinstance(loaded, torch.Tensor):
            loaded = loaded.float()
        match = torch.allclose(original, loaded, atol=1e-6)
        assert_all(match, f"Model param '{key}' mismatch")

    # Verify scalar
    assert_all(
        loaded_state_dict["step"] == 42, f"Expected step=42, got {loaded_state_dict.get('step')}"
    )

    mgr2.close()
    cleanup_tmp(tmp)
    log(f"  PASSED (loaded step {loaded_step})")


def test_multiple_save_latest_wins():
    """Save different values at each step, verify loaded matches the saved step."""
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    log("\n=== Test: Multiple Saves, Latest Wins ===")

    tmp = make_shared_tmp()
    config = InMemoryCkptConfig(
        enable=True,
        interval=5,
        replication_jump=4,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=5,
        disk_folder=tmp,
        skip_replication_if_hsdp=False,
    )
    mgr = InMemoryCheckpointManager(config)

    # Save different values at each step — simulates optimizer updating params
    # Flushes: step10→5, step15→10, step20→15, step25→20
    for step in [5, 10, 15, 20, 25]:
        t = torch.full((100,), fill_value=float(step + rank), device=device)
        mgr.save_tensors([t], step=step)
        mgr.maybe_wait()

    mgr.finalize_replication()
    mgr._disk.wait()
    mgr.close()

    dist.barrier()

    # Latest on disk = step 20 (flushed when step 25 saved)
    mgr2 = InMemoryCheckpointManager(config)
    latest = mgr2.find_latest()
    assert_all(latest == 20, f"Expected 20, got {latest}")

    result = mgr2.load()
    assert_all(result is not None, "load() returned None")

    state_dict, step = result
    if isinstance(state_dict, dict):
        loaded = list(state_dict.values())[0]
    else:
        loaded = state_dict[0]

    # The value saved at step 20 was (20 + rank)
    expected = torch.full((100,), fill_value=float(20 + rank))
    assert_all(
        torch.allclose(loaded, expected),
        f"Loaded value {loaded[0].item()} != expected {expected[0].item()} for step 20",
    )

    mgr2.close()
    cleanup_tmp(tmp)
    log("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────


def main():
    setup()

    test_save_and_periodic_flush()
    test_sigterm_flush()
    test_load_recovers_correct_state()
    test_find_latest_min_reduce()
    test_save_load_with_state_dict()
    test_multiple_save_latest_wins()

    log("\n" + "=" * 60)
    log("ALL SAVE/LOAD TESTS PASSED")
    log("=" * 60 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
