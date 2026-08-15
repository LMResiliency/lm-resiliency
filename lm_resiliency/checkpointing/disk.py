# mypy: ignore-errors
"""Async disk serialization and loading for checkpoint persistence."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import threading
import zlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import torch

from lm_resiliency.checkpointing.state_dict import FlatStateDictMetadata

logger = logging.getLogger(__name__)

_STEP_DIR_PATTERN = re.compile(r"step-(\d+)")
_STATUS_FILE = "GEMINI_CHECKPOINT_STATUS"
_STATUS_SCHEMA_VERSION = 1

# Chunk + thread the CRC so it stays sub-second on multi-GB shards. zlib.crc32
# releases the GIL, so per-chunk tasks scale near-linearly with worker count
# (measured ~25 GB/s at 8 workers → a 12 GB shard in ~0.5 s).
_CRC_CHUNK = 64 * 1024 * 1024  # 64 MiB
_CRC_WORKERS = min(8, os.cpu_count() or 1)


class ChecksumMismatch(Exception):
    """Raised when a loaded shard fails its stored checksum."""


@dataclass(frozen=True, slots=True)
class CheckpointStatus:
    """Persisted GEMINI candidate, recovery pointer, and restart policy."""

    candidate_step: int = -1
    recovery_verified_step: int = -1
    recovery_mode: str = "latest"

    def to_dict(self) -> dict[str, int | str]:
        return {
            "schema_version": _STATUS_SCHEMA_VERSION,
            "candidate_step": self.candidate_step,
            "recovery_verified_step": self.recovery_verified_step,
            "recovery_mode": self.recovery_mode,
        }

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> CheckpointStatus:
        if value.get("schema_version") != _STATUS_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported GEMINI checkpoint status schema {value.get('schema_version')!r}"
            )
        return cls(
            candidate_step=int(value.get("candidate_step", -1)),
            recovery_verified_step=int(value.get("recovery_verified_step", -1)),
            recovery_mode=str(value.get("recovery_mode", "latest")),
        )


class CheckpointStatusStore:
    """Atomically persist one rank's checkpoint trust metadata.

    Rank-scoped files avoid read-modify-write races when multiple local workers
    share a node-local checkpoint directory. The unscoped filename remains a
    read-only fallback for checkpoints written by older releases.
    """

    def __init__(self, folder: str | Path, rank: int | None = None) -> None:
        self._folder = Path(folder)
        self._rank = rank
        self._path = self._folder / self.filename(rank)
        self._legacy_path = self._folder / _STATUS_FILE

    @staticmethod
    def filename(rank: int | None) -> str:
        if rank is None:
            return _STATUS_FILE
        return f"{_STATUS_FILE}.rank-{rank}"

    @property
    def path(self) -> Path:
        return self._path

    def read(self) -> CheckpointStatus:
        if self._path.exists():
            return self._read_path(self._path)
        if self._rank is not None:
            peer_status = self._read_peer_fallback()
            if peer_status != CheckpointStatus():
                return peer_status
        if self._legacy_path.exists():
            return self._read_path(self._legacy_path)
        return CheckpointStatus()

    def write(self, status: CheckpointStatus) -> None:
        self._folder.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(
            status.to_dict(),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        descriptor, temporary = tempfile.mkstemp(
            dir=self._folder,
            prefix=f".{self._path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self._path)
            directory = os.open(self._folder, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    @staticmethod
    def _read_path(path: Path) -> CheckpointStatus:
        with path.open("r", encoding="utf-8") as handle:
            return CheckpointStatus.from_dict(json.load(handle))

    def _read_peer_fallback(self) -> CheckpointStatus:
        """Use surviving rank sidecars only as a conservative verified pointer."""
        statuses = [
            self._read_path(path)
            for path in sorted(self._folder.glob(f"{_STATUS_FILE}.rank-*"))
            if path != self._path
        ]
        if not statuses:
            return CheckpointStatus()
        verified_steps = [status.recovery_verified_step for status in statuses]
        verified_step = min(verified_steps) if all(step > 0 for step in verified_steps) else -1
        return CheckpointStatus(
            candidate_step=-1,
            recovery_verified_step=verified_step,
            recovery_mode="recovery_verified",
        )


def shard_checksums(
    tensors: list[torch.Tensor],
    chunk_size: int = _CRC_CHUNK,
    workers: int = _CRC_WORKERS,
) -> list[int]:
    """Chunked, multithreaded CRC-32 over raw bytes — guards at-rest / in-flight
    bit-flips (the failure mode layer replay cannot see: replay validates
    *compute*, checksums validate the stored/transferred *bytes*).

    Returns a flat list of per-chunk CRCs in deterministic (tensor, offset) order,
    so save and load produce identical lists for identical shard bytes. The uint8
    reinterpret lets it handle dtypes NumPy can't represent directly (bf16).
    """
    bufs = [t.detach().cpu().contiguous().flatten().view(torch.uint8).numpy() for t in tensors]
    spans = [(b, o) for b in bufs for o in range(0, len(b), chunk_size)]
    if not spans:
        return []

    def _crc(span) -> int:
        buf, off = span
        return zlib.crc32(buf[off : off + chunk_size]) & 0xFFFFFFFF

    if workers <= 1 or len(spans) == 1:
        return [_crc(s) for s in spans]
    try:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(_crc, spans))
    except RuntimeError:
        # Python disables ThreadPoolExecutor before normal atexit callbacks run.
        # An abnormal-exit checkpoint flush must still compute integrity metadata,
        # so retain the same deterministic serial result.
        logger.warning("CRC thread pool unavailable; computing checksums serially")
        return [_crc(s) for s in spans]


class DiskSerializer:
    """Handles async checkpoint flush to disk and loading from disk.

    Checkpoints are stored as: {folder}/step-{step}/rank-{rank}.pt
    Each file contains (metadata, tensors[, checksums]) for that rank.

    Args:
        folder: Checkpoint directory.
        rank: This shard's rank id.
        integrity: If True, store a per-tensor CRC-32 on save and verify it on
            load (raising ChecksumMismatch on a corrupt shard).
    """

    def __init__(self, folder: str, rank: int = 0, integrity: bool = False) -> None:
        self._folder = Path(folder)
        self._rank = rank
        self._integrity = integrity
        self._write_thread: threading.Thread | None = None
        self._write_error: BaseException | None = None
        self._latest_flushed_step: int = -1

    @property
    def latest_flushed_step(self) -> int:
        return self._latest_flushed_step

    def save_async(
        self,
        metadata: FlatStateDictMetadata,
        tensors: list[torch.Tensor],
        step: int,
    ) -> None:
        """Flush checkpoint to disk in a background thread."""
        self.wait()  # ensure previous write is done

        self._write_error = None

        # Clone tensors to avoid races with buffer rotation
        tensors_copy = [t.clone() for t in tensors]
        checksums = shard_checksums(tensors_copy) if self._integrity else None

        def _write():
            try:
                step_dir = self._folder / f"step-{step}"
                step_dir.mkdir(parents=True, exist_ok=True)
                save_path = step_dir / f"rank-{self._rank}.pt"
                torch.save(
                    {"metadata": metadata, "tensors": tensors_copy, "checksums": checksums},
                    save_path,
                )
                self._latest_flushed_step = step
                logger.info(f"Flushed checkpoint step {step} to {save_path}")
            except BaseException as e:
                self._write_error = e
                logger.error(f"Disk flush failed for step {step}: {e}")

        self._write_thread = threading.Thread(target=_write, daemon=True)
        self._write_thread.start()

    def wait(self) -> None:
        """Wait for in-flight disk write to complete."""
        if self._write_thread is not None:
            self._write_thread.join()
            self._write_thread = None
        if self._write_error is not None:
            err = self._write_error
            self._write_error = None
            raise RuntimeError(f"Disk write failed: {err}") from err

    def save_sync(
        self,
        metadata: FlatStateDictMetadata,
        tensors: list[torch.Tensor],
        step: int,
        rank: int | None = None,
    ) -> Path:
        """Synchronously write a shard to disk, addressable by an arbitrary rank.

        Unlike save_async (which always writes this node's own rank), the rank
        argument lets a node persist a *peer's* replicated shard under the peer's
        rank id. This is what makes a dead node's shard recoverable from disk:
        the surviving peer flushes rank-{peer}.pt before it, too, is restarted.

        Tensors are cloned so the caller may keep mutating its buffers.
        """
        target_rank = self._rank if rank is None else rank
        step_dir = self._folder / f"step-{step}"
        step_dir.mkdir(parents=True, exist_ok=True)
        save_path = step_dir / f"rank-{target_rank}.pt"
        tensors_copy = [t.clone() for t in tensors]
        checksums = shard_checksums(tensors_copy) if self._integrity else None
        torch.save(
            {"metadata": metadata, "tensors": tensors_copy, "checksums": checksums},
            save_path,
        )
        self._latest_flushed_step = max(self._latest_flushed_step, step)
        logger.info(f"Flushed checkpoint step {step} (rank {target_rank}) to {save_path}")
        return save_path

    def load(
        self, step: int, rank: int | None = None
    ) -> tuple[FlatStateDictMetadata, list[torch.Tensor]]:
        """Load checkpoint from disk for the given step (and optional rank)."""
        target_rank = self._rank if rank is None else rank
        step_dir = self._folder / f"step-{step}"
        load_path = step_dir / f"rank-{target_rank}.pt"
        if not load_path.exists():
            raise FileNotFoundError(f"No checkpoint at {load_path}")

        data = torch.load(load_path, weights_only=False)
        tensors = data["tensors"]
        if self._integrity:
            stored = data.get("checksums")
            if stored is not None:
                actual = shard_checksums(tensors)
                if actual != stored:
                    bad = sum(1 for a, b in zip(actual, stored) if a != b) + abs(
                        len(actual) - len(stored)
                    )
                    raise ChecksumMismatch(f"{load_path}: {bad} shard chunk(s) failed checksum")
        return data["metadata"], tensors

    def has_rank(self, step: int, rank: int) -> bool:
        """Whether a shard file for the given step and rank exists on this node."""
        return (self._folder / f"step-{step}" / f"rank-{rank}.pt").exists()

    def find_latest_on_disk(self) -> int:
        """Scan the checkpoint folder and return the latest step, or -1."""
        if not self._folder.exists():
            return -1

        latest = -1
        for entry in self._folder.iterdir():
            match = _STEP_DIR_PATTERN.fullmatch(entry.name)
            if match and entry.is_dir():
                rank_file = entry / f"rank-{self._rank}.pt"
                if rank_file.exists():
                    step = int(match.group(1))
                    latest = max(latest, step)
        return latest

    def cleanup_older_than(self, keep_step: int) -> None:
        """Remove disk checkpoints older than keep_step."""
        if not self._folder.exists():
            return
        import shutil

        for entry in self._folder.iterdir():
            match = _STEP_DIR_PATTERN.fullmatch(entry.name)
            if match and entry.is_dir():
                step = int(match.group(1))
                if step < keep_step:
                    shutil.rmtree(entry, ignore_errors=True)
                    logger.info(f"Removed old disk checkpoint step-{step}")

    def close(self) -> None:
        self.wait()
