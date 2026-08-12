"""Integration test: ModelReplayHarness in distributed setting.

Verifies the full path: hook capture → broadcast → replay → C3 comparison.

Run:
    torchrun --nproc_per_node=8 tests/integration/core/test_replay_harness.py
"""

from __future__ import annotations

import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.replay_harness import ModelReplayHarness, ReplayHarnessConfig


def setup():
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def log(msg: str):
    if dist.get_rank() == 0:
        print(msg, flush=True)


def assert_all(condition: bool, msg: str):
    t = torch.tensor([1 if condition else 0], dtype=torch.int64)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    assert t.item() == 1, f"Assertion failed on some rank: {msg}"


# ──────────────────────────────────────────────────────────────────────────────
# Model definition (shared across tests)
# ──────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 4)
        self.linear2 = nn.Linear(hidden_dim * 4, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = torch.nn.functional.gelu(self.linear1(h))
        return x + self.linear2(h)


class DropoutBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.linear(x))


class SimpleLLM(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 256, vocab_size: int = 1000):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class DropoutLLM(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([DropoutBlock(hidden_dim) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class StructuredBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, freqs_cis, attention_masks, positions=None):
        del freqs_cis
        output = self.linear(x) + attention_masks
        if positions is not None:
            output = output + positions.unsqueeze(-1)
        return torch.relu(output)


class StructuredLLM(nn.Module):
    def __init__(self, hidden_dim: int = 256):
        super().__init__()
        self.layers = nn.ModuleList([StructuredBlock(hidden_dim) for _ in range(2)])

    def forward(self, x, freqs_cis, attention_masks, positions):
        for layer in self.layers:
            x = layer(x, freqs_cis, attention_masks, positions=positions)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# Tests
# ──────────────────────────────────────────────────────────────────────────────


def test_harness_no_sdc():
    """All ranks have identical model → check() detects no SDC."""
    log("\n=== Test: Harness — no SDC (identical models) ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    # All ranks share the same model weights (simulates DP replicas)
    torch.manual_seed(42)
    model = SimpleLLM(num_layers=4, hidden_dim=256).to(device)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    config = ReplayHarnessConfig(layer_index=0, check_interval=0)
    harness = ModelReplayHarness(
        model, group=gloo_group, nccl_group=nccl_group, device=device, config=config
    )

    # Simulate a training step: forward + backward
    x = torch.randint(0, 1000, (4, 32), device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    assert_all(harness.has_capture, "Should have captured activation")
    assert_all(harness.has_grad, "Should have captured grad_output")

    # Run detection
    result = harness.check()

    assert_all(
        all(b == 0 for b in result.sdc_bitmap),
        f"Expected no SDC, got {result.sdc_bitmap}",
    )
    assert_all(result.replay_time_ms > 0, "Replay time should be positive")

    harness.remove_hooks()
    dist.barrier()
    log(f"  PASSED (replay_time={result.replay_time_ms:.2f}ms)")


def test_dropout_replay_synchronizes_and_restores_rng():
    """Different rank-local RNG streams still produce equivalent replay outputs."""
    log("\n=== Test: Harness — synchronized dropout RNG ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    model = DropoutLLM().to(device)
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    harness = ModelReplayHarness(
        model,
        group=gloo_group,
        nccl_group=nccl_group,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            rotate_layers=False,
            scale_factors=[],
            synchronize_rng=True,
        ),
    )

    torch.cuda.manual_seed(1000 + rank)
    x = torch.randn(4, 16, 256, device=device)
    model(x).sum().backward()

    # Deliberately leave every rank at a different CUDA RNG state before replay.
    torch.cuda.manual_seed(2000 + rank)
    state_before = torch.cuda.get_rng_state().clone()
    result = harness.check()
    state_after = torch.cuda.get_rng_state()

    assert_all(not any(result.sdc_bitmap), f"dropout replay diverged: {result.sdc_bitmap}")
    assert_all(torch.equal(state_before, state_after), "replay changed the training CUDA RNG")
    harness.remove_hooks()
    dist.barrier()
    log("  PASSED")


def test_harness_detects_sdc():
    """One rank has corrupted weights → check() detects SDC on that rank."""
    log("\n=== Test: Harness — detects SDC (corrupted weights) ===")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleLLM(num_layers=4, hidden_dim=256).to(device)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    config = ReplayHarnessConfig(layer_index=0, check_interval=0, scale_factors=[])
    harness = ModelReplayHarness(
        model, group=gloo_group, nccl_group=nccl_group, device=device, config=config
    )

    # Simulate a training step
    x = torch.randint(0, 1000, (4, 32), device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    # Corrupt rank 2's layer weights AFTER capture (simulates SDC during replay)
    if rank == 2:
        with torch.no_grad():
            model.layers[0].linear1.weight[0, 0] = 99999.0

    result = harness.check()

    expected = [0] * world_size
    expected[2] = 1
    assert_all(
        result.sdc_bitmap == expected,
        f"Expected {expected}, got {result.sdc_bitmap}",
    )

    harness.remove_hooks()
    dist.barrier()
    log("  PASSED (detected SDC on rank 2)")


def test_harness_step_interval():
    """step() only triggers check at the configured interval."""
    log("\n=== Test: Harness — step interval ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleLLM(num_layers=4, hidden_dim=256).to(device)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    config = ReplayHarnessConfig(layer_index=0, check_interval=3)
    harness = ModelReplayHarness(
        model, group=gloo_group, nccl_group=nccl_group, device=device, config=config
    )

    # Run 3 training steps
    for i in range(3):
        x = torch.randint(0, 1000, (4, 32), device=device)
        out = model(x)
        loss = out.sum()
        loss.backward()
        model.zero_grad()

        result = harness.step()
        if i < 2:
            assert_all(result is None, f"Step {i + 1}: should not trigger check")
        else:
            assert_all(result is not None, f"Step {i + 1}: should trigger check")
            assert_all(
                all(b == 0 for b in result.sdc_bitmap),
                f"Expected no SDC, got {result.sdc_bitmap}",
            )

    harness.remove_hooks()
    dist.barrier()
    log("  PASSED")


def test_harness_forward_only():
    """When no backward has run, check() uses forward-only replay."""
    log("\n=== Test: Harness — forward-only (no backward captured) ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleLLM(num_layers=4, hidden_dim=256).to(device)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    config = ReplayHarnessConfig(layer_index=0, check_interval=0, scale_factors=[])
    harness = ModelReplayHarness(
        model, group=gloo_group, nccl_group=nccl_group, device=device, config=config
    )

    # Forward only — no backward
    x = torch.randint(0, 1000, (4, 32), device=device)
    with torch.no_grad():
        _ = model(x)

    assert_all(harness.has_capture, "Should have captured activation")
    assert_all(not harness.has_grad, "Should NOT have grad_output")

    result = harness.check()

    assert_all(
        all(b == 0 for b in result.sdc_bitmap),
        f"Expected no SDC, got {result.sdc_bitmap}",
    )

    harness.remove_hooks()
    dist.barrier()
    log(f"  PASSED (forward-only replay_time={result.replay_time_ms:.2f}ms)")


def test_harness_auto_detect_layers():
    """Auto-detection of repeated layers works in distributed setting."""
    log("\n=== Test: Harness — auto-detect layers ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")

    torch.manual_seed(42)
    model = SimpleLLM(num_layers=4, hidden_dim=256).to(device)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")

    config = ReplayHarnessConfig(layer_index=2, check_interval=0, scale_factors=[])
    # No explicit layers= argument — relies on auto-detection
    harness = ModelReplayHarness(
        model, group=gloo_group, nccl_group=nccl_group, device=device, config=config
    )

    x = torch.randint(0, 1000, (4, 32), device=device)
    out = model(x)
    loss = out.sum()
    loss.backward()

    result = harness.check()

    assert_all(
        all(b == 0 for b in result.sdc_bitmap),
        f"Expected no SDC, got {result.sdc_bitmap}",
    )

    harness.remove_hooks()
    dist.barrier()
    log(f"  PASSED (layer_index=2, replay_time={result.replay_time_ms:.2f}ms)")


def test_structured_invocation_replay():
    """Replay preserves positional tensors and keyword arguments."""
    log("\n=== Test: Harness — structured args/kwargs ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    model = StructuredLLM().to(device)
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    harness = ModelReplayHarness(
        model,
        group=gloo_group,
        nccl_group=nccl_group,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            rotate_layers=False,
            scale_factors=[],
        ),
    )
    x = torch.randn(2, 16, 256, device=device, requires_grad=True)
    freqs_cis = torch.randn(16, 8, device=device)
    attention_masks = torch.randn(2, 16, 256, device=device)
    positions = torch.arange(16, device=device).expand(2, -1)
    model(x, freqs_cis, attention_masks, positions).sum().backward()

    result = harness.check()

    assert_all(result.sdc_bitmap == [0] * dist.get_world_size(), str(result.sdc_bitmap))
    assert_all(result.replay_mode == "forward_backward", result.replay_mode)
    harness.remove_hooks()
    dist.barrier()
    log("  PASSED")


def test_isolated_parameter_gradient_comparison():
    """A parameter-gradient-only fault is localized independently."""
    log("\n=== Test: Harness — isolated parameter gradient ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    model = SimpleLLM(num_layers=2, hidden_dim=256).to(device)
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    harness = ModelReplayHarness(
        model,
        group=gloo_group,
        nccl_group=nccl_group,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            rotate_layers=False,
            scale_factors=[],
        ),
    )
    x = torch.randint(0, 1000, (4, 16), device=device)
    model(x).sum().backward()
    victim = 2

    def corrupt_parameter_gradient(gradient):
        return gradient + 1.0 if rank == victim else gradient

    hook = model.layers[0].linear1.weight.register_hook(corrupt_parameter_gradient)
    result = harness.check()
    hook.remove()

    expected = [0] * dist.get_world_size()
    expected[victim] = 1
    assert_all(
        result.sdc_source_bitmaps["parameter_gradient"] == expected,
        str(result.sdc_source_bitmaps),
    )
    assert_all(
        not any(result.sdc_source_bitmaps["output"]),
        str(result.sdc_source_bitmaps),
    )
    assert_all(
        not any(result.sdc_source_bitmaps["input_gradient"]),
        str(result.sdc_source_bitmaps),
    )
    harness.remove_hooks()
    dist.barrier()
    log("  PASSED")


def test_optimizer_step_updated_weight_comparison():
    """A divergent post-step weight is localized through its own C3 source."""
    log("\n=== Test: Harness — optimizer updated weight ===")
    rank = dist.get_rank()
    device = torch.device(f"cuda:{rank}")
    torch.manual_seed(42)
    model = SimpleLLM(num_layers=2, hidden_dim=256).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    harness = ModelReplayHarness(
        model,
        group=gloo_group,
        nccl_group=nccl_group,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            rotate_layers=False,
            scale_factors=[],
        ),
    )
    x = torch.randint(0, 1000, (4, 16), device=device)
    model(x).sum().backward()
    optimizer.step()

    victim = 2
    if rank == victim:
        with torch.no_grad():
            model.layers[0].linear1.weight[0, 0].add_(1.0)

    result = harness.check(optimizer=optimizer)

    expected = [0] * dist.get_world_size()
    expected[victim] = 1
    assert_all(
        result.sdc_source_bitmaps["optimizer_updated_weight"] == expected,
        str(result.sdc_source_bitmaps),
    )
    harness.remove_hooks()
    dist.barrier()
    log("  PASSED")


# ──────────────────────────────────────────────────────────────────────────────


def main():
    setup()

    tests = {
        "no-sdc": test_harness_no_sdc,
        "dropout-rng": test_dropout_replay_synchronizes_and_restores_rng,
        "weight-sdc": test_harness_detects_sdc,
        "step-interval": test_harness_step_interval,
        "forward-only": test_harness_forward_only,
        "auto-layers": test_harness_auto_detect_layers,
        "structured": test_structured_invocation_replay,
        "parameter-gradient": test_isolated_parameter_gradient_comparison,
        "optimizer-updated-weight": test_optimizer_step_updated_weight_comparison,
    }
    selected = sys.argv[1:] or list(tests)
    unknown = sorted(set(selected) - set(tests))
    if unknown:
        raise SystemExit(f"unknown replay harness tests: {unknown}; choices={sorted(tests)}")
    for name in selected:
        tests[name]()

    log("\n" + "=" * 60)
    log("ALL REPLAY HARNESS TESTS PASSED")
    log("=" * 60 + "\n")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
