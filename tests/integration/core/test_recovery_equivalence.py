"""Integration test: a failed+recovered run must equal a never-failed run.

This is the definitive correctness test for recovery — it checks not just that the
checkpoint *round-trips*, but that training *resumes on the same trajectory*. It
covers all three pieces of training state end-to-end:

  * model weights,
  * optimizer state (Adam — real exp_avg / exp_avg_sq momentum, not stateless SGD),
  * dataloader position (a real position-dependent loader — the resumed run must
    read the same next batches).

Method: run to 2N uninterrupted (baseline); run to N, checkpoint, DISCARD all live
state (simulating a process restart), recover from the GEMINI in-memory checkpoint
(model + optimizer + dataloader via extra_state_fn), continue to 2N. The two final
states must be identical. A negative control (recover but DON'T restore the loader)
must diverge — proving the equivalence is non-trivial and the test detects errors.

Single process (no torchrun needed): GEMINI's manager runs with world_size=1. Runs
on GPU if available, else CPU, under deterministic algorithms.

Run:
    python tests/integration/core/test_recovery_equivalence.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parents[3])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager

DIM = 32
HIDDEN = 64
BS = 8
TOTAL_STEPS = 16
FAIL_AT = 8
MODEL_SEED = 1234

torch.manual_seed(0)
DATA = torch.randn(4096, DIM)  # fixed dataset shared by every run


class StatefulLoader:
    """A real position-dependent loader: the batch depends on the cursor, and the
    cursor is checkpointed/restored. If the position isn't restored, the resumed
    run reads different data and diverges."""

    def __init__(self, data: torch.Tensor, batch_size: int) -> None:
        self.data = data
        self.bs = batch_size
        self.pos = 0

    def next_batch(self, device) -> torch.Tensor:
        idx = [(self.pos + i) % len(self.data) for i in range(self.bs)]
        self.pos = (self.pos + self.bs) % len(self.data)
        return self.data[idx].to(device)

    def state_dict(self) -> dict:
        return {"pos": self.pos}  # non-tensor extra state

    def load_state_dict(self, sd: dict) -> None:
        self.pos = sd["pos"]


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(DIM, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, HIDDEN),
            nn.ReLU(),
            nn.Linear(HIDDEN, DIM),
        )

    def forward(self, x):
        return self.net(x)


def _make_model_opt(device):
    model = MLP().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    return model, opt


def _mgr(disk_dir: str) -> InMemoryCheckpointManager:
    return InMemoryCheckpointManager(
        InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=disk_dir,
            verify_integrity=True,
        )
    )


def _train_step(model, opt, batch):
    loss = model(batch).sum()
    loss.backward()
    opt.step()
    opt.zero_grad()


def run(device, disk_dir, fail_at=None, restore_loader=True):
    """Train TOTAL_STEPS. If fail_at set, discard+recover from GEMINI at that step."""
    torch.manual_seed(MODEL_SEED)
    model, opt = _make_model_opt(device)
    loader = StatefulLoader(DATA, BS)
    mgr = _mgr(disk_dir)
    positions: list[int] = []

    step = 0
    while step < TOTAL_STEPS:
        step += 1
        if fail_at is not None and step == fail_at + 1:
            # Simulate a crash+restart: persist in-memory → disk, throw away ALL
            # live state, then recover from the checkpoint.
            mgr.flush_for_restart()
            model, opt = _make_model_opt(device)  # fresh process
            loader = StatefulLoader(DATA, BS)  # fresh loader (pos=0)
            mgr = _mgr(disk_dir)
            recovered = mgr.load()
            assert recovered is not None, "recovery found no checkpoint"
            sd, rstep = recovered
            assert rstep == fail_at, f"recovered step {rstep}, expected {fail_at}"
            model.load_state_dict(sd["model"])
            opt.load_state_dict(sd["optimizer"])
            if restore_loader:
                loader.load_state_dict(sd["extra"]["loader"])

        positions.append(loader.pos)
        _train_step(model, opt, loader.next_batch(device))

        sd = {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "extra": {"loader": loader.state_dict()},
        }
        mgr.save(sd, step)
        mgr.maybe_wait()

    mgr.close()
    return model, opt, positions


# ── comparison helpers ────────────────────────────────────────────────────────
def _models_equal(a, b) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return all(torch.allclose(sa[k].cpu(), sb[k].cpu(), atol=1e-6, rtol=1e-5) for k in sa)


def _adam_state_equal(a, b) -> tuple[bool, str]:
    sa = a.state_dict()["state"]
    sb = b.state_dict()["state"]
    if set(sa) != set(sb):
        return False, "optimizer state param sets differ"
    n = 0
    for pid in sa:
        for key in ("exp_avg", "exp_avg_sq"):
            ta, tb = sa[pid][key].cpu(), sb[pid][key].cpu()
            if not torch.allclose(ta, tb, atol=1e-6, rtol=1e-5):
                return False, f"param {pid} {key} differs"
            n += 1
    return True, f"{n} Adam momentum/variance tensors match"


def main() -> None:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    print(f"equivalence test on {device}")

    with tempfile.TemporaryDirectory() as d:
        base_model, base_opt, base_pos = run(device, os.path.join(d, "A"))
        rec_model, rec_opt, rec_pos = run(device, os.path.join(d, "B"), fail_at=FAIL_AT)
        neg_model, _, neg_pos = run(
            device, os.path.join(d, "C"), fail_at=FAIL_AT, restore_loader=False
        )

    # sanity: training is non-trivial (the two halves consume distinct data)
    assert base_pos == list(range(0, TOTAL_STEPS * BS, BS)), "unexpected batch order"

    # 1) model weights match
    assert _models_equal(base_model, rec_model), "MODEL WEIGHTS diverged after recovery"
    # 2) optimizer (Adam) state matches
    ok, detail = _adam_state_equal(base_opt, rec_opt)
    assert ok, f"OPTIMIZER STATE diverged after recovery: {detail}"
    # 3) dataloader resumed to the same positions (exact data)
    assert rec_pos == base_pos, f"DATALOADER positions diverged: {rec_pos} vs {base_pos}"

    # negative control: NOT restoring the loader must change the trajectory —
    # proves the dataloader position genuinely matters and the test isn't vacuous.
    assert neg_pos != base_pos, "negative control: loader positions unexpectedly equal"
    assert not _models_equal(base_model, neg_model), (
        "negative control: weights matched even WITHOUT restoring the loader — "
        "the test would not catch a broken dataloader-resume"
    )

    print("  model weights: match")
    print(f"  optimizer state: {detail}")
    print(f"  dataloader: resumed to identical positions {rec_pos[FAIL_AT:]}")
    print("  negative control: diverged as expected without loader restore")
    print("\nRECOVERY EQUIVALENCE TEST PASSED (model + optimizer + dataloader)")


if __name__ == "__main__":
    main()
