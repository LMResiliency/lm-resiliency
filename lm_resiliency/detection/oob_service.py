# mypy: ignore-errors
"""Separate-process CPU/Gloo service for SCOUT hang localization."""

from __future__ import annotations

import ctypes
import logging
import multiprocessing
import os
import queue
import re
import signal
import threading
import traceback
from dataclasses import dataclass
from datetime import timedelta

import torch.distributed as dist

from lm_resiliency.detection.hang_detector import (
    HangDetectionDaemon,
    HangLocalizationResult,
)
from lm_resiliency.detection.reports import SCOUTFaultCallback, SCOUTFaultReport

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OOBHangConfig:
    stall_threshold_s: float = 30.0
    confirmation_interval_s: float = 1.0
    dataloader_latency_threshold_s: float = 5.0
    dataloader_min_slowdown_ratio: float = 2.0
    dataloader_confirmation_rounds: int = 2
    checkpoint_io_latency_threshold_s: float = 30.0
    checkpoint_io_min_slowdown_ratio: float = 2.0
    checkpoint_io_confirmation_rounds: int = 2
    rendezvous_timeout_s: float = 120.0
    state_dir: str | None = None
    master_addr: str | None = None
    master_port: int | None = None


class OOBHangService:
    """Own and supervise one independent daemon process for a training rank.

    ``report_callback`` runs in a parent-process dispatch thread. Without a callback,
    the child logs each report.
    """

    def __init__(
        self,
        *,
        global_rank: int,
        peer_ranks: list[int],
        config: OOBHangConfig,
        report_callback: SCOUTFaultCallback | None = None,
    ) -> None:
        self._global_rank = global_rank
        self._peer_ranks = peer_ranks
        self._config = config
        self._context = multiprocessing.get_context("spawn")
        self._progress_event = self._context.Event()
        self._process: multiprocessing.Process | None = None
        self._report_callback = report_callback
        self._report_queue = self._context.Queue() if report_callback is not None else None
        self._report_thread: threading.Thread | None = None

    @property
    def progress_event(self):
        """Local training-to-daemon progress notification."""
        return self._progress_event

    def start(self) -> None:
        if self._process is not None:
            return
        if self._report_queue is not None:
            self._report_thread = threading.Thread(
                target=self._dispatch_reports,
                name=f"scout-oob-reports-rank-{self._global_rank}",
                daemon=True,
            )
            self._report_thread.start()
        self._process = self._context.Process(
            target=_daemon_main,
            args=(
                self._global_rank,
                self._peer_ranks,
                self._config,
                self._progress_event,
                self._report_queue,
            ),
            name=f"scout-oob-rank-{self._global_rank}",
            daemon=True,
        )
        self._process.start()

    @property
    def pid(self) -> int | None:
        return self._process.pid if self._process is not None else None

    @property
    def is_alive(self) -> bool:
        return self._process is not None and self._process.is_alive()

    @property
    def exitcode(self) -> int | None:
        return self._process.exitcode if self._process is not None else None

    def close(self) -> None:
        process = self._process
        self._process = None
        if process is None:
            return
        if process.is_alive():
            process.terminate()
        process.join(timeout=5.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=2.0)
        if self._report_queue is not None:
            self._report_queue.put(None)
        if self._report_thread is not None:
            self._report_thread.join(timeout=2.0)
            self._report_thread = None

    def _dispatch_reports(self) -> None:
        assert self._report_queue is not None
        while True:
            try:
                report = self._report_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if report is None:
                return
            try:
                assert self._report_callback is not None
                self._report_callback(report)
            except Exception:
                logger.exception("SCOUT OOB report callback failed")


def _daemon_main(
    global_rank: int,
    peer_ranks: list[int],
    config: OOBHangConfig,
    progress_event,
    report_queue,
) -> None:
    _set_parent_death_signal()
    local_rank = peer_ranks.index(global_rank)
    address = config.master_addr or os.environ.get("MASTER_ADDR", "127.0.0.1")
    port = config.master_port or _group_port(peer_ranks)
    init_method = _rendezvous_method(config, peer_ranks, address, port)
    daemon: HangDetectionDaemon | None = None
    store = None
    try:
        _write_daemon_status(config, global_rank, f"joining {init_method} as {local_rank}")
        store = _init_daemon_process_group(
            init_method=init_method,
            address=address,
            port=port,
            local_rank=local_rank,
            world_size=len(peer_ranks),
            timeout=timedelta(seconds=config.rendezvous_timeout_s),
        )
        daemon = HangDetectionDaemon(
            rank=global_rank,
            world_size=len(peer_ranks),
            group=dist.group.WORLD,
            stall_threshold_s=config.stall_threshold_s,
            confirmation_interval_s=config.confirmation_interval_s,
            peer_ranks=peer_ranks,
            progress_event=progress_event,
            dataloader_latency_threshold_s=config.dataloader_latency_threshold_s,
            dataloader_min_slowdown_ratio=config.dataloader_min_slowdown_ratio,
            dataloader_confirmation_rounds=config.dataloader_confirmation_rounds,
            checkpoint_io_latency_threshold_s=config.checkpoint_io_latency_threshold_s,
            checkpoint_io_min_slowdown_ratio=config.checkpoint_io_min_slowdown_ratio,
            checkpoint_io_confirmation_rounds=config.checkpoint_io_confirmation_rounds,
        )
        _write_daemon_status(config, global_rank, "ready")

        def report(result: HangLocalizationResult) -> None:
            if local_rank != 0:
                return
            if result.is_hang and result.culprit_ranks:
                failed = result.culprit_ranks
            else:
                failed = (
                    result.stage_culprit_ranks
                    or result.culprit_ranks
                    or ([result.culprit_rank] if result.culprit_rank is not None else peer_ranks)
                )
            if result.stage_active and result.stage_kind in {
                "checkpoint_read",
                "checkpoint_write",
            }:
                kind = "checkpoint_stall"
            elif result.dataloader_active:
                kind = "data_stall"
            else:
                kind = "hang"
            report_payload: SCOUTFaultReport = {
                "failed_ranks": failed,
                "kind": kind,
                "scope": "rank" if len(failed) < len(peer_ranks) else "peer_group",
                "op_ids": result.op_ids,
                "steps": result.steps,
                "collective_metadata": result.metadata_fingerprints,
                "mismatch_kind": result.mismatch_kind,
                "stall_duration_s": result.stall_duration_s,
                "dataloader_active": result.dataloader_active,
                "dataloader_key": result.dataloader_key,
                "dataloader_sequence": result.dataloader_sequence,
                "dataloader_bitmap": result.dataloader_bitmap,
                "dataloader_latencies_ms": result.dataloader_latencies_ms,
                "dataloader_culprit_ranks": result.dataloader_culprit_ranks,
                "dataloader_confirmations": result.dataloader_confirmations,
                "stage_active": result.stage_active,
                "stage_kind": result.stage_kind,
                "stage_key": result.stage_key,
                "stage_sequence": result.stage_sequence,
                "stage_bitmap": result.stage_bitmap,
                "stage_latencies_ms": result.stage_latencies_ms,
                "stage_culprit_ranks": result.stage_culprit_ranks,
                "stage_confirmations": result.stage_confirmations,
            }
            logger.error("SCOUT OOB hang report: %s", report_payload)
            if report_queue is not None:
                report_queue.put(report_payload)

        daemon.run(on_hang=report)
    except BaseException:
        _write_daemon_status(config, global_rank, traceback.format_exc())
        raise
    finally:
        if daemon is not None:
            daemon.close()
        if dist.is_initialized():
            dist.destroy_process_group()
        del store


def _init_daemon_process_group(
    *,
    init_method: str,
    address: str,
    port: int,
    local_rank: int,
    world_size: int,
    timeout: timedelta,
):
    """Initialize OOB Gloo without reusing torchrun's elastic agent store."""
    if init_method.startswith("tcp://"):
        store = dist.TCPStore(
            address,
            port,
            world_size,
            local_rank == 0,
            timeout,
        )
        dist.init_process_group(
            backend="gloo",
            store=store,
            rank=local_rank,
            world_size=world_size,
            timeout=timeout,
        )
        return store

    dist.init_process_group(
        backend="gloo",
        init_method=init_method,
        rank=local_rank,
        world_size=world_size,
        timeout=timeout,
    )
    return None


def _group_port(peer_ranks: list[int]) -> int:
    base = int(os.environ.get("LM_SCOUT_OOB_PORT", int(os.environ.get("MASTER_PORT", 29500)) + 100))
    group_hash = sum((index + 1) * (rank + 1) for index, rank in enumerate(peer_ranks))
    return 1024 + ((base - 1024 + group_hash) % (65535 - 1024))


def _rendezvous_method(
    config: OOBHangConfig,
    peer_ranks: list[int],
    address: str,
    port: int,
) -> str:
    if config.master_addr is not None or config.master_port is not None:
        return f"tcp://{address}:{port}"
    if not config.state_dir:
        return f"tcp://{address}:{port}"
    run_id = (
        os.environ.get("TORCHELASTIC_RUN_ID")
        or os.environ.get("LM_RUN_ID")
        or os.environ.get("MASTER_PORT")
        or "default"
    )
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    group_id = "-".join(str(rank) for rank in peer_ranks)
    root = os.path.abspath(os.path.join(config.state_dir, "oob_rendezvous"))
    os.makedirs(root, exist_ok=True)
    return f"file://{root}/{run_id}-group-{group_id}"


def _set_parent_death_signal() -> None:
    if os.name != "posix":
        return
    try:
        libc = ctypes.CDLL(None)
        libc.prctl(1, signal.SIGTERM)
    except Exception:
        pass


def _write_daemon_status(config: OOBHangConfig, rank: int, status: str) -> None:
    if not config.state_dir:
        return
    try:
        from pathlib import Path

        root = Path(config.state_dir) / "oob_daemons"
        root.mkdir(parents=True, exist_ok=True)
        path = root / f"rank-{rank}.status"
        temporary = path.with_suffix(f".tmp.{os.getpid()}")
        temporary.write_text(status)
        os.replace(temporary, path)
    except Exception:
        logger.exception("could not publish SCOUT OOB daemon status")
