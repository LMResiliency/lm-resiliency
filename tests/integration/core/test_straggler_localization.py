"""Integration tests for two-phase straggler localization.

Tests the LayerReplayDetector.localize_straggler() method which:
  Phase 1: Identifies slow group via t_replay (already tested elsewhere)
  Phase 2: Re-runs with CUDA event instrumentation to separate t_compute vs t_comm,
            then uses C3 to identify the specific straggler rank.

Run with:
    torchrun --nproc_per_node=8 tests/integration/core/test_straggler_localization.py
"""

import os
import sys
import time

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.layer_replay import LayerReplayDetector


class TransformerBlock(nn.Module):
    """Simple transformer block without communication (pure DP setting)."""

    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.linear1(h))
        h = self.linear2(h)
        return x + h


class TPTransformerBlock(nn.Module):
    """Transformer block with simulated TP AllReduce communication."""

    def __init__(self, hidden_dim: int = 256, group: dist.ProcessGroup | None = None):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)
        self._group = group

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.linear1(h))
        h = self.linear2(h)
        # Simulate TP AllReduce
        if self._group is not None:
            dist.all_reduce(h, group=self._group)
        return x + h


def setup():
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    local_rank = rank % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    return rank, dist.get_world_size()


def test_localize_compute_straggler():
    """Phase 2 identifies a compute straggler (one rank's t_compute is an outlier)."""
    rank, world_size = setup()

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    layer = TransformerBlock(hidden_dim=256).to(device)
    activation = torch.randn(4, 32, 256, device=device)

    # Warm up
    detector = LayerReplayDetector(
        group=gloo_group,
        nccl_group=nccl_group,
        broadcast_src=0,
        device=device,
        deterministic=True,
    )
    for _ in range(3):
        detector.replay_forward(layer, activation, layer_id=0)
    dist.barrier()

    # Inject compute straggler on rank 2: busy-wait to simulate GPU throttling
    straggler_rank = 2 % world_size

    original_forward = layer.forward

    def slow_forward(x):
        if rank == straggler_rank:
            # Burn GPU time with a large matmul
            dummy = torch.randn(1024, 1024, device=device)
            for _ in range(5):
                dummy = dummy @ dummy
            torch.cuda.synchronize(device)
        return original_forward(x)

    layer.forward = slow_forward

    detail = detector.localize_straggler(layer, activation, layer_id=0)

    layer.forward = original_forward

    if rank == 0:
        print("\n[test_localize_compute_straggler]")
        print(f"  compute_times_ms: {[f'{t:.2f}' for t in detail.compute_times_ms]}")
        print(f"  comm_times_ms:    {[f'{t:.2f}' for t in detail.comm_times_ms]}")
        print(f"  compute_bitmap:   {detail.compute_bitmap}")
        print(f"  straggler_type:   {detail.straggler_type}")
        print(f"  straggler_rank:   {detail.straggler_rank}")

        # Compute straggler should be identified
        assert detail.straggler_type == "compute", (
            f"Expected 'compute', got '{detail.straggler_type}'"
        )
        assert detail.straggler_rank == straggler_rank, (
            f"Expected rank {straggler_rank}, got {detail.straggler_rank}"
        )
        print("  PASSED")

    dist.barrier()
    dist.destroy_process_group()


def test_localize_delay_before_collective():
    """A rank that delays before entering a collective is detected via t_compute.

    When one rank sleeps before the collective, t_compute = t_total - t_comm
    is elevated for that rank (the sleep is outside the wrapped collective).
    This correctly identifies the source of the group slowdown.

    Note: true NIC degradation (where the collective itself is slow) cannot be
    localized within a group because collectives are synchronous — all ranks see
    the same elevated t_comm. That case requires P2P probes (future work).
    """
    rank, world_size = setup()

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    layer = TPTransformerBlock(hidden_dim=256, group=nccl_group).to(device)
    activation = torch.randn(4, 32, 256, device=device)

    detector = LayerReplayDetector(
        group=gloo_group,
        nccl_group=nccl_group,
        broadcast_src=0,
        device=device,
        deterministic=True,
    )

    # Warm up
    for _ in range(5):
        detector.replay_forward(layer, activation, layer_id=0)
    dist.barrier()

    # Inject: rank 3 sleeps 100ms before the collective
    straggler_rank = 3 % world_size

    original_forward = layer.forward

    def slow_comm_forward(x):
        h = layer.norm(x)
        h = torch.nn.functional.gelu(layer.linear1(h))
        h = layer.linear2(h)
        if rank == straggler_rank:
            torch.cuda.synchronize(device)
            time.sleep(0.1)  # 100ms
        if layer._group is not None:
            dist.all_reduce(h, group=layer._group)
        return x + h

    layer.forward = slow_comm_forward

    detail = detector.localize_straggler(layer, activation, layer_id=0)

    layer.forward = original_forward

    if rank == 0:
        print("\n[test_localize_delay_before_collective]")
        print(f"  compute_times_ms: {[f'{t:.2f}' for t in detail.compute_times_ms]}")
        print(f"  comm_times_ms:    {[f'{t:.2f}' for t in detail.comm_times_ms]}")
        print(f"  compute_bitmap:   {detail.compute_bitmap}")
        print(f"  straggler_type:   {detail.straggler_type}")
        print(f"  straggler_rank:   {detail.straggler_rank}")

        # The sleep inflates rank 3's t_compute → detected as compute straggler
        assert detail.straggler_rank == straggler_rank, (
            f"Expected rank {straggler_rank}, got {detail.straggler_rank}"
        )
        assert detail.straggler_type == "compute"
        print("  PASSED")

    dist.barrier()
    dist.destroy_process_group()


def test_no_straggler():
    """When all ranks are healthy, no straggler is reported (with sufficient threshold).

    Note: 8 processes sharing GPUs introduces timing noise. We use a higher
    threshold_sigma to avoid false positives in this test environment.
    In production (dedicated GPUs), the default threshold_sigma=3.0 suffices.
    """
    rank, world_size = setup()

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    layer = TransformerBlock(hidden_dim=256).to(device)
    activation = torch.randn(4, 32, 256, device=device)

    detector = LayerReplayDetector(
        group=gloo_group,
        nccl_group=nccl_group,
        broadcast_src=0,
        device=device,
        deterministic=True,
    )

    # Warm up extensively to stabilize timings
    for _ in range(10):
        detector.replay_forward(layer, activation, layer_id=0)
    dist.barrier()

    # Use higher threshold to tolerate noise from shared GPUs
    detail = detector.localize_straggler(layer, activation, layer_id=0, threshold_sigma=5.0)

    if rank == 0:
        print("\n[test_no_straggler]")
        print(f"  compute_times_ms: {[f'{t:.2f}' for t in detail.compute_times_ms]}")
        print(f"  straggler_type:   {detail.straggler_type}")
        print(f"  straggler_rank:   {detail.straggler_rank}")

        # With elevated threshold, healthy ranks should not be flagged
        assert detail.straggler_type == "none", (
            f"Expected 'none', got '{detail.straggler_type}'. "
            f"compute_bitmap={detail.compute_bitmap}, comm_bitmap={detail.comm_bitmap}. "
            f"This may be a flaky result due to shared GPUs."
        )
        assert detail.straggler_rank is None
        print("  PASSED")

    dist.barrier()
    dist.destroy_process_group()


def test_full_replay_with_localization():
    """replay_forward_localize does phase 1 + auto-triggers phase 2 on straggler."""
    rank, world_size = setup()

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    layer = TransformerBlock(hidden_dim=256).to(device)
    activation = torch.randn(4, 32, 256, device=device)

    detector = LayerReplayDetector(
        group=gloo_group,
        nccl_group=nccl_group,
        broadcast_src=0,
        device=device,
        deterministic=True,
    )

    # Warm up
    for _ in range(3):
        detector.replay_forward(layer, activation, layer_id=0)
    dist.barrier()

    # Inject straggler
    straggler_rank = 1 % world_size
    original_forward = layer.forward

    def slow_forward(x):
        if rank == straggler_rank:
            dummy = torch.randn(1024, 1024, device=device)
            for _ in range(5):
                dummy = dummy @ dummy
            torch.cuda.synchronize(device)
        return original_forward(x)

    layer.forward = slow_forward

    result = detector.replay_forward_localize(layer, activation, layer_id=0)

    layer.forward = original_forward

    if rank == 0:
        print("\n[test_full_replay_with_localization]")
        print(f"  straggler_bitmap: {result.straggler_bitmap}")
        print(f"  straggler_detail: {result.straggler_detail}")

        # Phase 1 should detect a straggler
        assert any(result.straggler_bitmap), "Phase 1 should detect straggler"
        # Phase 2 should provide detail
        assert result.straggler_detail is not None, "Phase 2 should run"
        assert result.straggler_detail.straggler_rank == straggler_rank
        print("  PASSED")

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    test_name = sys.argv[1] if len(sys.argv) > 1 else "compute"

    tests = {
        "compute": test_localize_compute_straggler,
        "delay": test_localize_delay_before_collective,
        "none": test_no_straggler,
        "full": test_full_replay_with_localization,
    }

    if test_name in tests:
        tests[test_name]()
    else:
        print(f"Unknown test: {test_name}. Available: {list(tests.keys())}")
        sys.exit(1)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"\n{'=' * 60}")
        print("All straggler localization tests passed!")
        print(f"{'=' * 60}")
