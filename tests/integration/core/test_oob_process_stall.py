"""Exercise OOB localization while a real training rank is SIGSTOP'ed.

Run one case at a time:
    torchrun --standalone --nproc-per-node=4 \
        tests/integration/core/test_oob_process_stall.py transient
    torchrun --standalone --nproc-per-node=4 \
        tests/integration/core/test_oob_process_stall.py persistent
"""

from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch.distributed as dist

from lm_resiliency.detection.oob_service import OOBHangConfig, OOBHangService
from lm_resiliency.detection.op_tracker import OpTracker

WORLD_SIZE = 4
VICTIM = 2


def _wait_for_daemon(service: OOBHangService, status_path: Path) -> None:
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if status_path.exists() and status_path.read_text() == "ready":
            return
        assert service.is_alive, f"OOB daemon exited with {service.exitcode}"
        time.sleep(0.05)
    raise AssertionError(f"OOB daemon did not become ready: {status_path}")


def _resume_later(pid: int, delay_s: float) -> subprocess.Popen:
    command = f"sleep {delay_s}; kill -CONT {pid}"
    return subprocess.Popen(["bash", "-c", command], start_new_session=True)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"transient", "persistent"}:
        raise SystemExit("usage: test_oob_process_stall.py [transient|persistent]")
    case = sys.argv[1]
    pause_s = 0.10 if case == "transient" else 1.20
    threshold_s = 0.50 if case == "transient" else 0.25

    dist.init_process_group("gloo")
    rank = dist.get_rank()
    assert dist.get_world_size() == WORLD_SIZE
    control_dir = Path(f"/tmp/scout-oob-process-stall-{os.environ['MASTER_PORT']}-{case}")
    if rank == 0:
        shutil.rmtree(control_dir, ignore_errors=True)
    dist.barrier()

    reports = []
    service = OOBHangService(
        global_rank=rank,
        peer_ranks=list(range(WORLD_SIZE)),
        config=OOBHangConfig(
            stall_threshold_s=threshold_s,
            confirmation_interval_s=0.03,
            rendezvous_timeout_s=30.0,
            state_dir=str(control_dir),
            master_addr="127.0.0.1",
        ),
        report_callback=reports.append,
    )
    tracker = OpTracker(rank, progress_event=service.progress_event)
    service.start()
    _wait_for_daemon(
        service,
        control_dir / "oob_daemons" / f"rank-{rank}.status",
    )

    for _ in range(3):
        tracker.advance(force_signal=True)
    dist.barrier()

    resume = None
    if rank == VICTIM:
        resume = _resume_later(os.getpid(), pause_s)
        os.kill(os.getpid(), signal.SIGSTOP)
    if case == "transient" or rank != VICTIM:
        for _ in range(10):
            tracker.advance(force_signal=True)
            time.sleep(0.02)
    if resume is not None:
        assert resume.wait(timeout=5.0) == 0

    dist.barrier()
    if case == "persistent":
        # Leave the resumed victim behind while all OOB daemons time out together.
        observed = control_dir / "persistent-observed"
        deadline = time.time() + 5.0
        while time.time() < deadline and not observed.exists():
            if rank == 0:
                if any(report.get("kind") == "hang" for report in reports):
                    observed.write_text("observed")
                    break
            time.sleep(0.02)
        assert observed.exists(), "persistent SIGSTOP was not observed before timeout"
    else:
        observation_deadline = time.time() + max(0.8, threshold_s * 2.0)
        while time.time() < observation_deadline:
            tracker.advance(force_signal=True)
            time.sleep(0.03)
    if rank == 0:
        reports = [report for report in reports if report.get("kind") == "hang"]
        if case == "persistent":
            assert reports, "persistent SIGSTOP did not produce an OOB hang report"
            assert reports[0]["failed_ranks"] == [VICTIM], reports[0]
            assert reports[0]["scope"] == "rank", reports[0]
        else:
            assert not reports, f"sub-threshold SIGSTOP produced a false report: {reports}"

    dist.barrier()
    service.close()
    tracker.close()
    dist.barrier()
    if rank == 0:
        shutil.rmtree(control_dir, ignore_errors=True)
        print(
            f"OOB PROCESS STALL OK: {case} SIGSTOP for {pause_s:.2f}s "
            f"with threshold {threshold_s:.2f}s"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
