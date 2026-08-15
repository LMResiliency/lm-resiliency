"""Op Tracker: training-side component that publishes execution progress to shared memory.

The training process increments an op_id after each operation (compute or collective).
The OOB hang detection daemon reads this value to determine if the process is stuck.

Usage (training process):

    tracker = OpTracker(rank=dist.get_rank())
    # ... in training loop, after each collective or compute op:
    tracker.advance()

The shared memory region contains the current op_id, heartbeat timestamp, step,
and a compact description of the pending collective. The daemon reads them
without synchronization. A torn read is acceptable because localization is
confirmed across ranks after a sustained stall.
"""

from __future__ import annotations

import enum
import os
import struct
import time
from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any

# Layout: op_id (int64), heartbeat (float64), step (int64),
# collective metadata fingerprint (int64), active diagnostic stage
# (kind, key, sequence, start_ns), and most recently completed diagnostic stage
# (kind, key, sequence, duration_ns).
_OWNER_FMT = "q16s"
_OWNER_SIZE = struct.calcsize(_OWNER_FMT)
_PROGRESS_FMT = "qd" + ("q" * 10)
_SHM_SIZE = _OWNER_SIZE + struct.calcsize(_PROGRESS_FMT)
_SHM_NAME_PREFIX = "scout_op_tracker_rank_"
_DEFAULT_OWNER_TOKEN = bytes(16)
_PROGRESS_SIGNAL_INTERVAL_NS = 500_000_000


class DiagnosticStage(enum.IntEnum):
    """Python-visible operations whose latency SCOUT compares out of band."""

    NONE = 0
    DATA_LOADING = 1
    CHECKPOINT_READ = 2
    CHECKPOINT_WRITE = 3


@dataclass(frozen=True)
class StageToken:
    """Identity returned when a diagnostic stage starts."""

    kind: DiagnosticStage
    key: int
    sequence: int
    started_ns: int


@dataclass(frozen=True)
class ProgressSnapshot:
    """One lock-free shared-memory observation."""

    op_id: int
    heartbeat: float
    step: int
    metadata_fingerprint: int
    active_stage: DiagnosticStage
    active_stage_key: int
    active_stage_sequence: int
    active_stage_started_ns: int
    completed_stage: DiagnosticStage
    completed_stage_key: int
    completed_stage_sequence: int
    completed_stage_duration_ns: int


def _shm_name(rank: int, namespace: str | None = None) -> str:
    return namespace or f"{_SHM_NAME_PREFIX}{rank}"


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _diagnostic_stage(value: int) -> DiagnosticStage:
    try:
        return DiagnosticStage(value)
    except ValueError:
        return DiagnosticStage.NONE


class OpTracker:
    """Training-side op progress publisher.

    Writes current op_id and heartbeat to shared memory so the OOB daemon
    can detect hangs without any coupling to the training process.

    The tracker auto-increments op_id on each advance() call. The training
    framework hooks (forward hooks, collective wrappers) call advance() at
    each operation boundary.

    Args:
        rank: This process's global rank.
    """

    def __init__(
        self,
        rank: int,
        progress_event: Any | None = None,
        *,
        shm_name: str | None = None,
        owner_token: bytes | None = None,
    ) -> None:
        self._rank = rank
        self._progress_event = progress_event
        self._last_progress_signal_ns = 0
        self._op_id: int = 0
        self._step: int = 0
        self._metadata_fingerprint: int = 0
        self._stage_counts: dict[tuple[DiagnosticStage, int], int] = {}
        self._active_stage = DiagnosticStage.NONE
        self._active_stage_key: int = 0
        self._active_stage_sequence: int = 0
        self._active_stage_started_ns: int = 0
        self._completed_stage = DiagnosticStage.NONE
        self._completed_stage_key: int = 0
        self._completed_stage_sequence: int = 0
        self._completed_stage_duration_ns: int = 0
        self._shm_name = _shm_name(rank, shm_name)
        self._owner_token = _DEFAULT_OWNER_TOKEN if owner_token is None else owner_token
        if len(self._owner_token) != 16:
            raise ValueError("SCOUT op tracker owner token must contain exactly 16 bytes")

        # Never attach to another live publisher. A dead owner may leave its
        # segment behind after SIGKILL, in which case the exact channel name is
        # safe to reclaim.
        try:
            self._shm = shared_memory.SharedMemory(name=self._shm_name, create=True, size=_SHM_SIZE)
        except FileExistsError:
            existing = shared_memory.SharedMemory(name=self._shm_name, create=False)
            try:
                if existing.size < _OWNER_SIZE:
                    raise FileExistsError(
                        f"SCOUT op tracker channel {self._shm_name!r} has an unknown live owner"
                    )
                owner_pid, _ = struct.unpack_from(_OWNER_FMT, existing.buf, 0)
                if _pid_is_alive(owner_pid):
                    raise FileExistsError(
                        f"SCOUT op tracker channel {self._shm_name!r} is owned by live "
                        f"process {owner_pid}"
                    )
                existing.unlink()
            finally:
                existing.close()
            self._shm = shared_memory.SharedMemory(
                name=self._shm_name,
                create=True,
                size=_SHM_SIZE,
            )

        struct.pack_into(_OWNER_FMT, self._shm.buf, 0, os.getpid(), self._owner_token)
        self._write(force_signal=True)

    def advance(self, metadata_fingerprint: int = 0, *, force_signal: bool = False) -> None:
        """Mark that the next op has started. Called by training hooks."""
        self._op_id += 1
        self._metadata_fingerprint = metadata_fingerprint
        self._write(force_signal=force_signal or metadata_fingerprint != 0)

    def step_boundary(self) -> None:
        """Mark the start of a new training step. Resets op_id to 0."""
        self._step += 1
        self._op_id = 0
        self._metadata_fingerprint = 0
        self._write(force_signal=True)

    def stage_started(self, kind: DiagnosticStage, key: int) -> StageToken:
        """Publish a diagnostic stage before entering a potentially blocking call."""
        if kind is DiagnosticStage.NONE:
            raise ValueError("NONE is not a measurable diagnostic stage")
        if self._active_stage is not DiagnosticStage.NONE:
            raise RuntimeError(f"diagnostic stage {self._active_stage.name} is already active")
        event = (kind, int(key))
        sequence = self._stage_counts.get(event, 0) + 1
        self._stage_counts[event] = sequence
        started_ns = time.monotonic_ns()
        self._active_stage = kind
        self._active_stage_key = int(key)
        self._active_stage_sequence = sequence
        self._active_stage_started_ns = started_ns
        self._metadata_fingerprint = 0
        self._op_id += 1
        self._write(force_signal=True)
        return StageToken(kind, int(key), sequence, started_ns)

    def stage_finished(self, token: StageToken) -> int:
        """Publish completion and return the measured duration in nanoseconds."""
        identity = (
            self._active_stage,
            self._active_stage_key,
            self._active_stage_sequence,
            self._active_stage_started_ns,
        )
        expected = (token.kind, token.key, token.sequence, token.started_ns)
        if identity != expected:
            raise RuntimeError("diagnostic stage token does not match the active stage")
        duration_ns = max(0, time.monotonic_ns() - token.started_ns)
        self._completed_stage = token.kind
        self._completed_stage_key = token.key
        self._completed_stage_sequence = token.sequence
        self._completed_stage_duration_ns = duration_ns
        self._active_stage = DiagnosticStage.NONE
        self._active_stage_key = 0
        self._active_stage_sequence = 0
        self._active_stage_started_ns = 0
        self._op_id += 1
        self._write(force_signal=True)
        return duration_ns

    @property
    def op_id(self) -> int:
        return self._op_id

    @property
    def step(self) -> int:
        return self._step

    @property
    def metadata_fingerprint(self) -> int:
        return self._metadata_fingerprint

    def _write(self, *, force_signal: bool = False) -> None:
        struct.pack_into(
            _PROGRESS_FMT,
            self._shm.buf,
            _OWNER_SIZE,
            self._op_id,
            time.time(),
            self._step,
            self._metadata_fingerprint,
            int(self._active_stage),
            self._active_stage_key,
            self._active_stage_sequence,
            self._active_stage_started_ns,
            int(self._completed_stage),
            self._completed_stage_key,
            self._completed_stage_sequence,
            self._completed_stage_duration_ns,
        )
        self._signal_progress(force=force_signal)

    def _signal_progress(self, *, force: bool) -> None:
        if self._progress_event is None:
            return
        now_ns = time.monotonic_ns()
        if force or now_ns - self._last_progress_signal_ns >= _PROGRESS_SIGNAL_INTERVAL_NS:
            self._progress_event.set()
            self._last_progress_signal_ns = now_ns

    def close(self) -> None:
        """Release shared memory. Call during shutdown."""
        try:
            self._shm.close()
            current = shared_memory.SharedMemory(name=self._shm_name, create=False)
            try:
                _, token = struct.unpack_from(_OWNER_FMT, current.buf, 0)
                if token == self._owner_token:
                    current.unlink()
            finally:
                current.close()
        except FileNotFoundError:
            pass

    def __del__(self) -> None:
        if hasattr(self, "_shm"):
            try:
                self._shm.close()
            except Exception:
                pass


class OpTrackerReader:
    """Read-only view of a rank's op tracker state (used by the OOB daemon).

    Args:
        rank: The rank whose shared memory to read.
    """

    def __init__(
        self,
        rank: int,
        *,
        shm_name: str | None = None,
        owner_token: bytes | None = None,
    ) -> None:
        self._rank = rank
        self._shm_name = _shm_name(rank, shm_name)
        self._owner_token = _DEFAULT_OWNER_TOKEN if owner_token is None else owner_token
        if len(self._owner_token) != 16:
            raise ValueError("SCOUT op tracker owner token must contain exactly 16 bytes")
        self._shm: shared_memory.SharedMemory | None = None

    def attach(self) -> bool:
        """Attach to the training process's shared memory. Returns False if not found."""
        try:
            candidate = shared_memory.SharedMemory(name=self._shm_name, create=False)
            if candidate.size < _SHM_SIZE:
                candidate.close()
                raise RuntimeError(
                    f"SCOUT op tracker channel {self._shm_name!r} has an incompatible layout"
                )
            _, token = struct.unpack_from(_OWNER_FMT, candidate.buf, 0)
            if token != self._owner_token:
                candidate.close()
                raise RuntimeError(
                    f"SCOUT op tracker channel {self._shm_name!r} ownership token mismatch"
                )
            self._shm = candidate
            return True
        except FileNotFoundError:
            return False

    def read(self) -> tuple[int, float, int]:
        """Read the backward-compatible (op_id, heartbeat_timestamp, step) state."""
        op_id, heartbeat, step, _ = self.read_with_metadata()
        return op_id, heartbeat, step

    def read_with_metadata(self) -> tuple[int, float, int, int]:
        """Read progress plus the pending collective metadata fingerprint."""
        snapshot = self.read_snapshot()
        return (
            snapshot.op_id,
            snapshot.heartbeat,
            snapshot.step,
            snapshot.metadata_fingerprint,
        )

    def read_snapshot(self) -> ProgressSnapshot:
        """Read progress, collective metadata, and diagnostic-stage timing."""
        if self._shm is None:
            return ProgressSnapshot(
                op_id=-1,
                heartbeat=0.0,
                step=-1,
                metadata_fingerprint=0,
                active_stage=DiagnosticStage.NONE,
                active_stage_key=0,
                active_stage_sequence=0,
                active_stage_started_ns=0,
                completed_stage=DiagnosticStage.NONE,
                completed_stage_key=0,
                completed_stage_sequence=0,
                completed_stage_duration_ns=0,
            )
        values = struct.unpack_from(_PROGRESS_FMT, self._shm.buf, _OWNER_SIZE)
        return ProgressSnapshot(
            op_id=values[0],
            heartbeat=values[1],
            step=values[2],
            metadata_fingerprint=values[3],
            active_stage=_diagnostic_stage(values[4]),
            active_stage_key=values[5],
            active_stage_sequence=values[6],
            active_stage_started_ns=values[7],
            completed_stage=_diagnostic_stage(values[8]),
            completed_stage_key=values[9],
            completed_stage_sequence=values[10],
            completed_stage_duration_ns=values[11],
        )

    def close(self) -> None:
        if self._shm is not None:
            self._shm.close()
            self._shm = None

    def __del__(self) -> None:
        if hasattr(self, "_shm") and self._shm is not None:
            try:
                self._shm.close()
            except Exception:
                pass
