"""Separate-process OOB hang daemon integration test.

Run:
    torchrun --standalone --nproc_per_node=4 tests/integration/core/test_oob_daemon.py
"""

from __future__ import annotations

import os
import shutil
import time
from pathlib import Path

import torch.distributed as dist

from lm_resiliency.detection.oob_service import OOBHangConfig, OOBHangService
from lm_resiliency.detection.op_tracker import DiagnosticStage, OpTracker


def _run_case(
    *,
    case: str,
    rank: int,
    world_size: int,
    victim: int,
    advances: int,
    victim_advances: int,
    metadata_fingerprint: int = 0,
    victim_metadata_fingerprint: int = 0,
) -> dict:
    rank = dist.get_rank()
    control_dir = Path(f"/tmp/scout-oob-integration-{os.environ['MASTER_PORT']}-{case}")
    if rank == 0:
        shutil.rmtree(control_dir, ignore_errors=True)
    dist.barrier()

    reports = []
    service = OOBHangService(
        global_rank=rank,
        peer_ranks=list(range(world_size)),
        config=OOBHangConfig(
            stall_threshold_s=0.25,
            confirmation_interval_s=0.05,
            rendezvous_timeout_s=30.0,
            state_dir=str(control_dir),
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        ),
        report_callback=reports.append,
    )
    tracker = OpTracker(rank, progress_event=service.progress_event)
    service.start()

    local_advances = victim_advances if rank == victim else advances
    local_metadata = victim_metadata_fingerprint if rank == victim else metadata_fingerprint
    for index in range(local_advances):
        tracker.advance(metadata_fingerprint=local_metadata if index == local_advances - 1 else 0)

    time.sleep(1.0)
    assert service.is_alive, f"rank {rank} OOB daemon exited with {service.exitcode}"

    deadline = time.time() + 10.0
    if rank == 0:
        while time.time() < deadline:
            if any(report.get("kind") == "hang" for report in reports):
                break
            time.sleep(0.05)
    dist.barrier()
    service.close()
    tracker.close()
    dist.barrier()
    if rank == 0:
        reports = [report for report in reports if report.get("kind") == "hang"]
        assert reports, "OOB daemon did not report the stalled group"
        assert reports[-1]["failed_ranks"] == [victim], reports[-1]
        assert reports[-1]["scope"] == "rank"
        shutil.rmtree(control_dir, ignore_errors=True)
        return reports[-1]
    return {}


def _run_dataloader_case(
    *,
    rank: int,
    world_size: int,
    victim: int,
) -> dict:
    control_dir = Path(f"/tmp/scout-oob-integration-{os.environ['MASTER_PORT']}-dataloader")
    if rank == 0:
        shutil.rmtree(control_dir, ignore_errors=True)
    dist.barrier()

    reports = []
    service = OOBHangService(
        global_rank=rank,
        peer_ranks=list(range(world_size)),
        config=OOBHangConfig(
            stall_threshold_s=5.0,
            confirmation_interval_s=0.03,
            dataloader_latency_threshold_s=0.15,
            dataloader_min_slowdown_ratio=2.0,
            dataloader_confirmation_rounds=2,
            rendezvous_timeout_s=30.0,
            state_dir=str(control_dir),
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        ),
        report_callback=reports.append,
    )
    tracker = OpTracker(rank, progress_event=service.progress_event)
    service.start()

    status_path = control_dir / "oob_daemons" / f"rank-{rank}.status"
    ready_deadline = time.time() + 15.0
    while time.time() < ready_deadline:
        if status_path.exists() and status_path.read_text() == "ready":
            break
        assert service.is_alive, f"rank {rank} OOB daemon exited with {service.exitcode}"
        time.sleep(0.05)
    else:
        raise AssertionError(f"rank {rank} OOB daemon did not become ready")
    dist.barrier()

    token = tracker.stage_started(DiagnosticStage.DATA_LOADING, key=123)
    if rank == victim:
        time.sleep(1.0)
    tracker.stage_finished(token)
    time.sleep(0.5)

    if rank == 0:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if any(report.get("kind") == "data_stall" for report in reports):
                break
            time.sleep(0.05)
    dist.barrier()
    service.close()
    tracker.close()
    dist.barrier()
    if rank == 0:
        reports = [report for report in reports if report.get("kind") == "data_stall"]
        assert reports, "OOB daemon did not report the sampled DataLoader stall"
        report = reports[-1]
        assert report["failed_ranks"] == [victim], report
        assert report["dataloader_active"], report
        assert report["dataloader_confirmations"] >= 2, report
        shutil.rmtree(control_dir, ignore_errors=True)
        return report
    return {}


def _run_checkpoint_case(
    *,
    rank: int,
    world_size: int,
    victim: int,
) -> dict:
    control_dir = Path(f"/tmp/scout-oob-integration-{os.environ['MASTER_PORT']}-checkpoint")
    if rank == 0:
        shutil.rmtree(control_dir, ignore_errors=True)
    dist.barrier()

    reports = []
    service = OOBHangService(
        global_rank=rank,
        peer_ranks=list(range(world_size)),
        config=OOBHangConfig(
            stall_threshold_s=5.0,
            confirmation_interval_s=0.03,
            dataloader_latency_threshold_s=5.0,
            checkpoint_io_latency_threshold_s=0.15,
            checkpoint_io_min_slowdown_ratio=2.0,
            checkpoint_io_confirmation_rounds=2,
            rendezvous_timeout_s=30.0,
            state_dir=str(control_dir),
            master_addr=os.environ.get("MASTER_ADDR", "127.0.0.1"),
        ),
        report_callback=reports.append,
    )
    tracker = OpTracker(rank, progress_event=service.progress_event)
    service.start()

    status_path = control_dir / "oob_daemons" / f"rank-{rank}.status"
    ready_deadline = time.time() + 15.0
    while time.time() < ready_deadline:
        if status_path.exists() and status_path.read_text() == "ready":
            break
        assert service.is_alive, f"rank {rank} OOB daemon exited with {service.exitcode}"
        time.sleep(0.05)
    else:
        raise AssertionError(f"rank {rank} OOB daemon did not become ready")
    dist.barrier()

    token = tracker.stage_started(DiagnosticStage.CHECKPOINT_WRITE, key=456)
    if rank == victim:
        time.sleep(1.0)
    tracker.stage_finished(token)
    time.sleep(0.5)

    if rank == 0:
        deadline = time.time() + 10.0
        while time.time() < deadline:
            if any(report.get("kind") == "checkpoint_stall" for report in reports):
                break
            time.sleep(0.05)
    dist.barrier()
    service.close()
    tracker.close()
    dist.barrier()
    if rank == 0:
        reports = [report for report in reports if report.get("kind") == "checkpoint_stall"]
        assert reports, "OOB daemon did not report the checkpoint write stall"
        report = reports[-1]
        assert report["failed_ranks"] == [victim], report
        assert report["stage_active"], report
        assert report["stage_kind"] == "checkpoint_write", report
        assert report["stage_confirmations"] >= 2, report
        shutil.rmtree(control_dir, ignore_errors=True)
        return report
    return {}


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4
    victim = 2

    progress_report = _run_case(
        case="progress",
        rank=rank,
        world_size=world_size,
        victim=victim,
        advances=10,
        victim_advances=5,
    )
    if rank == 0:
        assert progress_report["mismatch_kind"] == "progress", progress_report

    metadata_report = _run_case(
        case="metadata",
        rank=rank,
        world_size=world_size,
        victim=victim,
        advances=1,
        victim_advances=1,
        metadata_fingerprint=123,
        victim_metadata_fingerprint=999,
    )
    if rank == 0:
        assert metadata_report["mismatch_kind"] == "collective_metadata", metadata_report
        assert metadata_report["collective_metadata"] == [123, 123, 999, 123]
    _run_dataloader_case(
        rank=rank,
        world_size=world_size,
        victim=victim,
    )
    _run_checkpoint_case(
        rank=rank,
        world_size=world_size,
        victim=victim,
    )
    if rank == 0:
        print(
            "SCOUT OOB DAEMON OK: separate Gloo processes localized progress, "
            "collective-metadata mismatches, DataLoader latency, and checkpoint I/O."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
