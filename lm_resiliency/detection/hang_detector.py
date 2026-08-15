"""Out-of-Band Hang Detection Daemon.

Runs as a separate process from training. Monitors the training process's
execution progress via shared memory and triggers C³ consensus across the
peer group when a stall is detected.

Runtime contract (see docs/scout.md#out-of-band-detection):
  - One daemon per GPU, separate process from training
  - Reads progress and pending-collective metadata from shared memory
  - If progress stalls beyond a threshold, triggers C³ to localize the culprit
  - Independent communication channel (own process group) — never blocked by
    training-side hangs

Design choices:
  - Out-of-band: daemon stays alive even when training hangs (fate-sharing broken)
  - Progress consensus identifies a lagging or diverged rank
  - Metadata consensus handles incompatible public Python collective calls at
    the same progress value
  - Equal progress and metadata remain group-scoped unless communication overlap
    or an external fabric diagnostic identifies a physical culprit

Usage:
    # In a separate process (e.g., launched by a supervisor):
    daemon = HangDetectionDaemon(
        rank=0, world_size=8,
        group=gloo_group,
        stall_threshold_s=30.0,
    )
    daemon.run()  # blocking loop

    # Or run one detection round (useful for testing):
    result = daemon.check_once()
"""

from __future__ import annotations

import logging
import statistics
import time
from collections import Counter
from dataclasses import dataclass, field
from threading import Event
from typing import Any, Callable

import torch
import torch.distributed as dist

from lm_resiliency.detection.c3 import C3, C3Mode
from lm_resiliency.detection.op_tracker import DiagnosticStage, OpTrackerReader, ProgressSnapshot

logger = logging.getLogger(__name__)


@dataclass
class HangLocalizationResult:
    """Result of a single hang detection round.

    Attributes:
        is_hang: Whether a stall was detected on the local rank.
        culprit_rank: The rank identified as the outlier, or None if no outlier.
        bitmap: C³ output bitmap (bit i = 1 means rank i diverges).
        op_ids: All ranks' op_ids at the time of detection.
        local_op_id: This rank's op_id.
        stall_duration_s: How long the local rank has been stalled (0 if not stalled).
        metadata_fingerprints: Compact collective descriptions from all ranks.
        mismatch_kind: ``progress``, ``collective_metadata``, or
            a typed diagnostic-stage latency when localized.
        dataloader_active: Whether the culprit or peer group is inside a sampled
            DataLoader call.
    """

    is_hang: bool
    culprit_rank: int | None
    bitmap: list[int]
    op_ids: list[int]
    local_op_id: int
    stall_duration_s: float
    steps: list[int] = field(default_factory=list)
    local_step: int = -1
    metadata_fingerprints: list[int] = field(default_factory=list)
    local_metadata_fingerprint: int = 0
    mismatch_kind: str | None = None
    culprit_ranks: list[int] = field(default_factory=list)
    is_dataloader_straggler: bool = False
    dataloader_active: bool = False
    dataloader_key: int = 0
    dataloader_sequence: int = 0
    dataloader_bitmap: list[int] = field(default_factory=list)
    dataloader_culprit_ranks: list[int] = field(default_factory=list)
    dataloader_latencies_ms: list[float] = field(default_factory=list)
    dataloader_confirmations: int = 0
    is_stage_straggler: bool = False
    stage_active: bool = False
    stage_kind: str | None = None
    stage_key: int = 0
    stage_sequence: int = 0
    stage_bitmap: list[int] = field(default_factory=list)
    stage_culprit_ranks: list[int] = field(default_factory=list)
    stage_latencies_ms: list[float] = field(default_factory=list)
    stage_confirmations: int = 0


class HangDetectionDaemon:
    """OOB daemon that detects and localizes communication hangs.

    Monitors the local training process via shared memory and coordinates
    with peer daemons via C³ to localize the culprit rank.

    The daemon operates from local progress notifications:
      1. Wait for a training-side progress event or the stall timeout
      2. Reset the timeout whenever visible progress changes
      3. On timeout, compare progress and metadata through C³
      4. Require stall_threshold_s before reporting a hard hang

    Args:
        rank: This daemon's rank (same as the training rank it monitors).
        world_size: Total number of ranks in the peer group.
        group: Gloo process group for C³ communication (daemon's own group,
            independent from training's NCCL groups).
        stall_threshold_s: How long an op_id must be unchanged before reporting a
            hard hang. Should be shorter than the NCCL watchdog timeout (default
            10 min) to enable early localization.
        confirmation_interval_s: Delay between C³ confirmation rounds while
            progress remains stalled.
    """

    def __init__(
        self,
        rank: int,
        world_size: int,
        group: dist.ProcessGroup,
        stall_threshold_s: float = 30.0,
        confirmation_interval_s: float = 1.0,
        peer_ranks: list[int] | None = None,
        progress_event: Any | None = None,
        dataloader_latency_threshold_s: float = 5.0,
        dataloader_min_slowdown_ratio: float = 2.0,
        dataloader_confirmation_rounds: int = 2,
        checkpoint_io_latency_threshold_s: float = 30.0,
        checkpoint_io_min_slowdown_ratio: float = 2.0,
        checkpoint_io_confirmation_rounds: int = 2,
        tracker_name: str | None = None,
        tracker_token: bytes | None = None,
    ) -> None:
        self._rank = rank
        self._world_size = world_size
        self._group = group
        self._stall_threshold_s = stall_threshold_s
        self._confirmation_interval_s = confirmation_interval_s
        self._progress_event = progress_event if progress_event is not None else Event()
        self._peer_ranks = peer_ranks or list(range(world_size))
        self._local_index = self._peer_ranks.index(rank) if rank in self._peer_ranks else 0
        self._dataloader_latency_threshold_ms = dataloader_latency_threshold_s * 1000.0
        self._checkpoint_io_latency_threshold_ms = checkpoint_io_latency_threshold_s * 1000.0
        self._consensus_trigger_threshold_s = min(
            stall_threshold_s,
            dataloader_latency_threshold_s,
            checkpoint_io_latency_threshold_s,
        )
        self._dataloader_min_slowdown_ratio = dataloader_min_slowdown_ratio
        self._dataloader_confirmation_rounds = dataloader_confirmation_rounds
        self._checkpoint_io_min_slowdown_ratio = checkpoint_io_min_slowdown_ratio
        self._checkpoint_io_confirmation_rounds = checkpoint_io_confirmation_rounds
        if dataloader_latency_threshold_s <= 0:
            raise ValueError("dataloader_latency_threshold_s must be positive")
        if dataloader_min_slowdown_ratio <= 1:
            raise ValueError("dataloader_min_slowdown_ratio must be greater than 1")
        if dataloader_confirmation_rounds < 1:
            raise ValueError("dataloader_confirmation_rounds must be positive")
        if checkpoint_io_latency_threshold_s <= 0:
            raise ValueError("checkpoint_io_latency_threshold_s must be positive")
        if checkpoint_io_min_slowdown_ratio <= 1:
            raise ValueError("checkpoint_io_min_slowdown_ratio must be greater than 1")
        if checkpoint_io_confirmation_rounds < 1:
            raise ValueError("checkpoint_io_confirmation_rounds must be positive")
        if confirmation_interval_s <= 0:
            raise ValueError("confirmation_interval_s must be positive")

        self._c3 = C3(group=group)
        self._reader = OpTrackerReader(
            rank=rank,
            shm_name=tracker_name,
            owner_token=tracker_token,
        )
        self._last_op_id: int = -1
        self._last_step: int = -1
        self._last_change_time: float = time.time()
        self._last_reported_progress: tuple[int, int] | None = None
        self._last_reported_stage: tuple[str | None, int, int, tuple[int, ...]] | None = None
        self._stage_candidate: tuple[str | None, int, int, tuple[int, ...]] | None = None
        self._stage_confirmations = 0
        self._running = False

    def start(self) -> bool:
        """Attach to the training process's shared memory.

        Returns True if successfully attached, False if training process
        hasn't started yet (shared memory not found).
        """
        if not self._reader.attach():
            logger.warning(
                f"Rank {self._rank}: training process shared memory not found. "
                "Will retry after the next progress notification or timeout."
            )
            return False
        snapshot = self._reader.read_snapshot()
        self._last_op_id = snapshot.op_id
        self._last_step = snapshot.step
        self._last_change_time = time.time()
        self._progress_event.clear()
        return True

    def _observe_stall(self) -> float:
        """Read progress after an event or timeout and return its idle duration."""
        snapshot = self._reader.read_snapshot()
        op_id, step = snapshot.op_id, snapshot.step

        if op_id == -1:
            return 0.0

        if (step, op_id) != (self._last_step, self._last_op_id):
            self._last_op_id = op_id
            self._last_step = step
            self._last_change_time = time.time()
            self._last_reported_progress = None
            return 0.0

        return time.time() - self._last_change_time

    def check_once(self) -> HangLocalizationResult:
        """Run a single C³ detection round across the peer group.

        All daemons in the group should call this at approximately the same
        time (coordinated by each daemon independently detecting a stall on
        its local rank, or triggered externally).

        Returns:
            HangLocalizationResult identifying the culprit (if any).
        """
        snapshot = self._reader.read_snapshot()
        op_id = snapshot.op_id
        step = snapshot.step
        metadata_fingerprint = snapshot.metadata_fingerprint
        progress_unchanged = (step, op_id) == (self._last_step, self._last_op_id)
        stall_s = time.time() - self._last_change_time if progress_unchanged else 0.0

        progress_id = (int(step) << 32) | (int(op_id) & 0xFFFFFFFF)
        progress_result = self._c3.run_scalar(progress_id, mode=C3Mode.EXACT)
        bitmap = progress_result.bitmap

        # Gather explicit values for diagnostics after consensus on the encoded pair.
        active_elapsed_ns = _active_stage_elapsed_ns(snapshot)
        local_tensor = torch.tensor(
            [
                op_id,
                step,
                metadata_fingerprint,
                int(snapshot.active_stage),
                snapshot.active_stage_key,
                snapshot.active_stage_sequence,
                active_elapsed_ns,
                int(snapshot.completed_stage),
                snapshot.completed_stage_key,
                snapshot.completed_stage_sequence,
                snapshot.completed_stage_duration_ns,
            ],
            dtype=torch.int64,
        )
        gathered = [torch.zeros_like(local_tensor) for _ in range(self._world_size)]
        dist.all_gather(gathered, local_tensor, group=self._group)
        all_op_ids = [int(t[0].item()) for t in gathered]
        all_steps = [int(t[1].item()) for t in gathered]
        all_metadata = [int(t[2].item()) for t in gathered]
        stage_rows = [[int(value) for value in tensor.tolist()] for tensor in gathered]

        mismatch_kind = None
        culprit_ranks: list[int] = []
        if any(b == 1 for b in bitmap):
            mismatch_kind = "progress"
            culprit_ranks = [self._peer_ranks[index] for index, value in enumerate(bitmap) if value]
        elif len(set(zip(all_steps, all_op_ids))) == 1 and len(set(all_metadata)) > 1:
            metadata_counts = Counter(all_metadata)
            majority_fingerprint, count = metadata_counts.most_common(1)[0]
            if count > self._world_size // 2:
                bitmap = [int(fingerprint != majority_fingerprint) for fingerprint in all_metadata]
                mismatch_kind = "collective_metadata"
                culprit_ranks = [
                    self._peer_ranks[index] for index, value in enumerate(bitmap) if value
                ]
        culprit = culprit_ranks[0] if culprit_ranks else None
        stage, stage_key, stage_sequence, stage_bitmap, stage_latencies_ms = (
            self._compare_stage_latency(stage_rows)
        )
        stage_culprit_ranks = [
            self._peer_ranks[index] for index, value in enumerate(stage_bitmap) if value
        ]
        is_stage_straggler = bool(stage_culprit_ranks)
        if culprit is None and stage_culprit_ranks:
            culprit = stage_culprit_ranks[0]
        if mismatch_kind is None and is_stage_straggler:
            mismatch_kind = _stage_mismatch_kind(stage)

        associated_stage = stage
        if culprit_ranks:
            associated_stage = _active_stage_for_culprits(
                stage_rows,
                culprit_ranks,
                self._peer_ranks,
            )
        elif not is_stage_straggler:
            associated_stage = _common_active_stage(stage_rows)
        stage_active = associated_stage is not DiagnosticStage.NONE
        stage_kind = _stage_name(stage) if stage is not DiagnosticStage.NONE else None

        is_dataloader_straggler = stage is DiagnosticStage.DATA_LOADING and is_stage_straggler
        dataloader_active = stage_active and associated_stage is DiagnosticStage.DATA_LOADING
        dataloader_key = stage_key if dataloader_active else 0
        dataloader_sequence = stage_sequence if dataloader_active else 0
        dataloader_bitmap = stage_bitmap if dataloader_active else [0] * self._world_size
        dataloader_culprit_ranks = stage_culprit_ranks if dataloader_active else []
        dataloader_latencies_ms = stage_latencies_ms if dataloader_active else []

        is_hang = stall_s > self._stall_threshold_s

        result = HangLocalizationResult(
            is_hang=is_hang,
            culprit_rank=culprit,
            bitmap=bitmap,
            op_ids=all_op_ids,
            local_op_id=op_id,
            stall_duration_s=stall_s,
            steps=all_steps,
            local_step=step,
            metadata_fingerprints=all_metadata,
            local_metadata_fingerprint=metadata_fingerprint,
            mismatch_kind=mismatch_kind,
            culprit_ranks=culprit_ranks,
            is_dataloader_straggler=is_dataloader_straggler,
            dataloader_active=dataloader_active,
            dataloader_key=dataloader_key,
            dataloader_sequence=dataloader_sequence,
            dataloader_bitmap=dataloader_bitmap,
            dataloader_culprit_ranks=dataloader_culprit_ranks,
            dataloader_latencies_ms=dataloader_latencies_ms,
            is_stage_straggler=is_stage_straggler,
            stage_active=stage_active,
            stage_kind=stage_kind,
            stage_key=stage_key,
            stage_sequence=stage_sequence,
            stage_bitmap=stage_bitmap,
            stage_culprit_ranks=stage_culprit_ranks,
            stage_latencies_ms=stage_latencies_ms,
        )

        if is_hang and culprit is not None:
            logger.warning(
                f"Hang localized: rank(s) {culprit_ranks} differ in {mismatch_kind}. "
                f"op_ids={all_op_ids}, metadata={all_metadata}, bitmap={bitmap}"
            )
        elif is_hang and culprit is None:
            logger.warning(
                f"No strict-majority culprit at op_id={op_id} after {stall_s:.1f}s. "
                "Check configured fabric diagnostics."
            )

        return result

    def _compare_stage_latency(
        self,
        rows: list[list[int]],
    ) -> tuple[DiagnosticStage, int, int, list[int], list[float]]:
        """Run C³ on the longest active Python-visible diagnostic stage."""
        candidates = [
            (row[6], index, _stage(row[3]), row[4], row[5])
            for index, row in enumerate(rows)
            if _stage(row[3]) is not DiagnosticStage.NONE and row[6] > 0
        ]
        if not candidates:
            return DiagnosticStage.NONE, 0, 0, [0] * self._world_size, []

        _, _, kind, key, sequence = max(candidates)
        duration_ns = [
            _matching_stage_duration_ns(
                row,
                kind=kind,
                key=key,
                sequence=sequence,
            )
            for row in rows
        ]
        local_duration_ms = duration_ns[self._local_index] / 1_000_000.0
        timing_result = self._c3.run_scalar(
            local_duration_ms,
            mode=C3Mode.STATISTICAL,
        )
        bitmap = timing_result.bitmap
        values_ms = [float(value) for value in timing_result.evidence]
        center = statistics.median(values_ms) if values_ms else 0.0
        if kind is DiagnosticStage.DATA_LOADING:
            threshold_ms = self._dataloader_latency_threshold_ms
            slowdown_ratio = self._dataloader_min_slowdown_ratio
        else:
            threshold_ms = self._checkpoint_io_latency_threshold_ms
            slowdown_ratio = self._checkpoint_io_min_slowdown_ratio
        threshold = max(threshold_ms, center * slowdown_ratio)
        high_bitmap = [
            int(bool(flagged) and value > threshold) for flagged, value in zip(bitmap, values_ms)
        ]
        return kind, key, sequence, high_bitmap, values_ms

    def run(
        self,
        max_rounds: int | None = None,
        *,
        on_hang: Callable[[HangLocalizationResult], None] | None = None,
    ) -> None:
        """Main daemon loop. Blocks until stopped or max_rounds reached.

        Args:
            max_rounds: If set, exit after this many actionable detection rounds
                (for testing). If None, run indefinitely until stop() is called.
        """
        self._running = True
        rounds = 0

        # Wait for training process to start
        while self._running:
            if self.start():
                break
            self._progress_event.wait(self._confirmation_interval_s)
            self._progress_event.clear()

        wait_timeout_s = self._consensus_trigger_threshold_s
        while self._running:
            progressed = self._progress_event.wait(wait_timeout_s)
            if not self._running:
                break
            if progressed:
                self._progress_event.clear()
                self._observe_stall()
                wait_timeout_s = self._consensus_trigger_threshold_s
                continue

            stall_s = self._observe_stall()
            if stall_s < self._consensus_trigger_threshold_s:
                wait_timeout_s = max(
                    self._consensus_trigger_threshold_s - stall_s,
                    0.001,
                )
                continue

            result = self.check_once()
            progress = (result.local_step, result.local_op_id)
            stage_event = (
                result.stage_kind,
                result.stage_key,
                result.stage_sequence,
                tuple(result.stage_culprit_ranks),
            )
            if result.is_stage_straggler:
                if stage_event == self._stage_candidate:
                    self._stage_confirmations += 1
                else:
                    self._stage_candidate = stage_event
                    self._stage_confirmations = 1
            else:
                self._stage_candidate = None
                self._stage_confirmations = 0
            result.stage_confirmations = self._stage_confirmations
            if result.stage_kind == "data_loading":
                required_confirmations = self._dataloader_confirmation_rounds
                result.dataloader_confirmations = self._stage_confirmations
            else:
                required_confirmations = self._checkpoint_io_confirmation_rounds
            stage_confirmed = (
                result.is_stage_straggler and self._stage_confirmations >= required_confirmations
            )
            report_hang = result.is_hang and progress != self._last_reported_progress
            report_stage = (
                stage_confirmed and stage_event != self._last_reported_stage and not result.is_hang
            )
            if (report_hang or report_stage) and on_hang is not None:
                on_hang(result)
                if report_hang:
                    self._last_reported_progress = progress
                if report_stage:
                    self._last_reported_stage = stage_event
            if result.is_hang or stage_confirmed:
                rounds += 1
                if max_rounds is not None and rounds >= max_rounds:
                    break
            wait_timeout_s = self._confirmation_interval_s

    def stop(self) -> None:
        """Signal the daemon loop to exit."""
        self._running = False
        self._progress_event.set()

    def close(self) -> None:
        """Clean up resources."""
        self.stop()
        self._reader.close()


def _active_stage_elapsed_ns(snapshot: ProgressSnapshot) -> int:
    if snapshot.active_stage is DiagnosticStage.NONE or snapshot.active_stage_started_ns <= 0:
        return 0
    return max(0, time.monotonic_ns() - snapshot.active_stage_started_ns)


def _stage(value: int) -> DiagnosticStage:
    try:
        return DiagnosticStage(value)
    except ValueError:
        return DiagnosticStage.NONE


def _stage_name(stage: DiagnosticStage) -> str:
    return stage.name.lower()


def _stage_mismatch_kind(stage: DiagnosticStage) -> str:
    if stage is DiagnosticStage.DATA_LOADING:
        return "dataloader_latency"
    return f"{_stage_name(stage)}_latency"


def _matching_stage_duration_ns(
    row: list[int],
    *,
    kind: DiagnosticStage,
    key: int,
    sequence: int,
) -> int:
    kind_value = int(kind)
    if (row[3], row[4], row[5]) == (kind_value, key, sequence):
        return row[6]
    if (row[7], row[8], row[9]) == (kind_value, key, sequence):
        return row[10]
    return 0


def _active_stage_for_culprits(
    rows: list[list[int]],
    culprit_ranks: list[int],
    peer_ranks: list[int],
) -> DiagnosticStage:
    states = {
        _stage(rows[peer_ranks.index(rank)][3]) for rank in culprit_ranks if rank in peer_ranks
    }
    if len(states) == 1:
        return next(iter(states))
    return DiagnosticStage.NONE


def _common_active_stage(rows: list[list[int]]) -> DiagnosticStage:
    states = {_stage(row[3]) for row in rows}
    if len(states) == 1:
        return next(iter(states))
    return DiagnosticStage.NONE
