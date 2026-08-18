"""SCOUT-gated certification for framework-owned durable checkpoints."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Any, ContextManager, Protocol

import torch
import torch.distributed as dist

from lm_resiliency.detection.layer_replay import ReplayResult, replay_result_has_sdc

_SCHEMA_VERSION = 1
_LATEST_MANIFEST = "LATEST_SCOUT_VALIDATED"
_PENDING_MANIFEST = "PENDING_SCOUT_CANDIDATE"

logger = logging.getLogger(__name__)


class DurableCheckpointAdapter(Protocol):
    """Framework operations used by the certification coordinator.

    ``save_candidate`` and ``load_checkpoint`` are called on every rank and may
    invoke collective framework checkpoint APIs. ``commit_candidate`` and
    ``quarantine_candidate`` are optional metadata/retention notifications called
    only on the manifest-writer rank; they must not contain collectives.
    """

    def save_candidate(
        self,
        candidate: DurableCheckpointRecord,
    ) -> Mapping[str, Any] | None: ...

    def load_checkpoint(
        self,
        checkpoint: DurableCheckpointRecord,
    ) -> int: ...


@dataclass(frozen=True, slots=True)
class CallbackDurableCheckpointAdapter:
    """Build a durable adapter from framework checkpoint callbacks."""

    save_candidate_fn: Callable[["DurableCheckpointRecord"], Mapping[str, Any] | None]
    load_checkpoint_fn: Callable[["DurableCheckpointRecord"], int]
    commit_candidate_fn: (
        Callable[["DurableCheckpointRecord", "DurableCheckpointRecord | None"], None] | None
    ) = None
    quarantine_candidate_fn: Callable[["DurableCheckpointRecord", str], None] | None = None

    def save_candidate(
        self,
        candidate: DurableCheckpointRecord,
    ) -> Mapping[str, Any] | None:
        return self.save_candidate_fn(candidate)

    def load_checkpoint(self, checkpoint: DurableCheckpointRecord) -> int:
        return self.load_checkpoint_fn(checkpoint)

    def commit_candidate(
        self,
        checkpoint: DurableCheckpointRecord,
        previous: DurableCheckpointRecord | None,
    ) -> None:
        if self.commit_candidate_fn is not None:
            self.commit_candidate_fn(checkpoint, previous)

    def quarantine_candidate(
        self,
        checkpoint: DurableCheckpointRecord,
        reason: str,
    ) -> None:
        if self.quarantine_candidate_fn is not None:
            self.quarantine_candidate_fn(checkpoint, reason)


@dataclass(frozen=True, slots=True)
class DurableCheckpointConfig:
    """Configuration for SCOUT-certified framework checkpoints.

    ``manifest_dir`` must be on the same durable shared storage as, or at least
    have the same failure durability as, the framework checkpoints.
    """

    manifest_dir: str
    environment_id: str
    adapter: DurableCheckpointAdapter
    writer_rank: int = 0

    def __post_init__(self) -> None:
        if not self.manifest_dir:
            raise ValueError("durable checkpoint manifest_dir cannot be empty")
        if not self.environment_id:
            raise ValueError("durable checkpoint environment_id cannot be empty")
        if self.writer_rank < 0:
            raise ValueError("durable checkpoint writer_rank cannot be negative")


@dataclass(frozen=True, slots=True)
class ShapeValidation:
    """One clean shape replay contributing to a candidate's certification."""

    shape_id: str
    step: int


@dataclass(frozen=True, slots=True)
class DurableCheckpointRecord:
    """Persistent candidate or recovery-verified checkpoint metadata."""

    checkpoint_id: str
    step: int
    validation_epoch: int
    shape_plan_id: str
    environment_id: str
    expected_shape_ids: tuple[str, ...]
    checked_shapes: tuple[ShapeValidation, ...] = ()
    status: str = "writing"
    verdict: str | None = None
    completed_step: int | None = None
    artifacts: dict[str, Any] = field(default_factory=dict)

    @property
    def checked_shape_ids(self) -> tuple[str, ...]:
        return tuple(item.shape_id for item in self.checked_shapes)

    @property
    def complete(self) -> bool:
        return self.checked_shape_ids == self.expected_shape_ids

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": _SCHEMA_VERSION,
            "checkpoint_id": self.checkpoint_id,
            "step": self.step,
            "validation_epoch": self.validation_epoch,
            "shape_plan_id": self.shape_plan_id,
            "environment_id": self.environment_id,
            "expected_shape_ids": list(self.expected_shape_ids),
            "checked_shapes": [
                {"shape_id": item.shape_id, "step": item.step} for item in self.checked_shapes
            ],
            "status": self.status,
            "verdict": self.verdict,
            "completed_step": self.completed_step,
            "artifacts": self.artifacts,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> DurableCheckpointRecord:
        if value.get("schema_version") != _SCHEMA_VERSION:
            raise ValueError(
                f"unsupported durable checkpoint manifest schema {value.get('schema_version')!r}"
            )
        record = cls(
            checkpoint_id=str(value["checkpoint_id"]),
            step=int(value["step"]),
            validation_epoch=int(value["validation_epoch"]),
            shape_plan_id=str(value["shape_plan_id"]),
            environment_id=str(value["environment_id"]),
            expected_shape_ids=tuple(str(item) for item in value["expected_shape_ids"]),
            checked_shapes=tuple(
                ShapeValidation(shape_id=str(item["shape_id"]), step=int(item["step"]))
                for item in value.get("checked_shapes", ())
            ),
            status=str(value["status"]),
            verdict=(None if value.get("verdict") is None else str(value["verdict"])),
            completed_step=(
                None if value.get("completed_step") is None else int(value["completed_step"])
            ),
            artifacts=dict(value.get("artifacts") or {}),
        )
        if not record.checkpoint_id or not record.expected_shape_ids:
            raise ValueError("durable checkpoint manifest has empty identity or shape plan")
        return record


class DurableCheckpointEvent(str, Enum):
    """Result of applying one SCOUT replay result to a pending candidate."""

    NONE = "none"
    PROGRESS = "progress"
    COMMITTED = "committed"
    REJECTED = "rejected"


class DurableCheckpointCoordinator:
    """Promote framework candidates after the following clean shape cycle."""

    def __init__(
        self,
        config: DurableCheckpointConfig,
        *,
        shape_plan_id: str,
        shape_ids: Sequence[str],
        process_group: dist.ProcessGroup | None = None,
        checkpoint_io: Callable[[str, str], ContextManager[None]] | None = None,
        topology_digest: str | None = None,
    ) -> None:
        if not shape_plan_id:
            raise ValueError("durable checkpoint shape_plan_id cannot be empty")
        if not shape_ids or len(shape_ids) != len(set(shape_ids)):
            raise ValueError("durable checkpoint shape IDs must be non-empty and unique")
        if topology_digest is not None and (
            not isinstance(topology_digest, str) or not topology_digest.strip()
        ):
            raise ValueError("durable checkpoint topology_digest must be a non-empty string")

        self.config = config
        self.shape_plan_id = shape_plan_id
        self.shape_ids = tuple(shape_ids)
        self._process_group = process_group
        self._checkpoint_io = checkpoint_io
        self.topology_digest = topology_digest
        self._rank = dist.get_rank(process_group) if dist.is_initialized() else 0
        self._world_size = dist.get_world_size(process_group) if dist.is_initialized() else 1
        if config.writer_rank >= self._world_size:
            raise ValueError(
                f"writer_rank={config.writer_rank} outside checkpoint group "
                f"of size {self._world_size}"
            )
        self._store = _DurableManifestStore(config.manifest_dir)
        self._pending: DurableCheckpointRecord | None = None

        stale = self._store.read_pending()
        latest = self._store.read_latest()
        if stale is not None and self._is_writer:
            if (
                latest is not None
                and latest.checkpoint_id == stale.checkpoint_id
                and latest.status in ("recovery_verified", "validated")
                and latest.verdict == "clean"
            ):
                # Commit publishes the validated pointer before deleting the
                # pending marker. A crash in that small window is already safe.
                self._store.clear_pending()
            else:
                self._reject_record(
                    stale,
                    "validation incomplete when the checkpoint coordinator restarted",
                )
        self._barrier()
        self._latest = self._store.read_latest()
        self._next_epoch = self._store.next_validation_epoch()

    @property
    def pending(self) -> DurableCheckpointRecord | None:
        return self._pending

    @property
    def latest_validated(self) -> DurableCheckpointRecord | None:
        return self._latest

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    def begin_candidate(
        self,
        *,
        step: int,
        first_shape_id: str,
    ) -> DurableCheckpointRecord:
        """Write one framework candidate for the supplied validation epoch."""
        if self._pending is not None:
            raise RuntimeError("a durable checkpoint candidate is already pending")
        starts = self._all_gather_object(
            (
                self.shape_plan_id,
                self.shape_ids,
                first_shape_id,
                self.config.environment_id,
            )
        )
        if any(item != starts[0] for item in starts[1:]):
            raise RuntimeError(
                "durable checkpoint shape plan or scheduler position differs "
                "across checkpoint ranks"
            )
        expected = _rotate_shape_ids(self.shape_ids, first_shape_id)
        epoch = self._next_epoch
        checkpoint_id = f"scout-step-{int(step)}-epoch-{epoch}-{self.shape_plan_id[:12]}"
        record = DurableCheckpointRecord(
            checkpoint_id=checkpoint_id,
            step=int(step),
            validation_epoch=epoch,
            shape_plan_id=self.shape_plan_id,
            environment_id=self.config.environment_id,
            expected_shape_ids=expected,
        )
        if self._is_writer:
            self._store.write_pending(record)
        self._barrier()

        local_error: Exception | None = None
        artifacts: Mapping[str, Any] | None = None
        try:
            with self._measure_checkpoint_io("write", record.checkpoint_id):
                artifacts = self.config.adapter.save_candidate(record)
        except Exception as exc:  # noqa: BLE001 - synchronize failure across ranks
            local_error = exc
        if not self._all_ranks_succeeded(local_error is None):
            if self._is_writer:
                self._reject_record(record, "framework candidate checkpoint write failed")
            self._barrier()
            if local_error is not None:
                raise local_error
            raise RuntimeError("framework candidate checkpoint write failed on another rank")

        record = replace(
            record,
            status="candidate",
            artifacts=dict(artifacts or {}) if self._is_writer else {},
        )
        manifest_error: Exception | None = None
        if self._is_writer:
            try:
                # Validate JSON serializability before the candidate receives evidence.
                json.dumps(record.to_dict())
                self._store.write_pending(record)
            except Exception as exc:  # noqa: BLE001 - synchronize metadata failures
                manifest_error = exc
        if not self._all_ranks_succeeded(manifest_error is None):
            if self._is_writer:
                self._reject_record(
                    replace(record, artifacts={}),
                    "candidate artifact metadata could not be persisted",
                )
            self._barrier()
            if manifest_error is not None:
                raise manifest_error
            raise RuntimeError(
                "candidate artifact metadata could not be persisted on the writer rank"
            )
        self._barrier()
        self._pending = record
        self._next_epoch = epoch + 1
        return record

    def observe(
        self,
        result: ReplayResult | None,
        *,
        step: int,
    ) -> DurableCheckpointEvent:
        """Apply one scheduled SCOUT result to the pending candidate."""
        if self._pending is None:
            return DurableCheckpointEvent.NONE
        candidate = self._pending

        reason: str | None = None
        checked: tuple[str, ...] = ()
        if result is None:
            reason = "scheduled SCOUT replay did not produce evidence"
        elif replay_result_has_sdc(result):
            reason = "SCOUT detected numerical divergence"
        else:
            checked = tuple(result.checked_shape_ids)
            if not checked:
                reason = "SCOUT result did not identify any checked replay shape"
            else:
                offset = len(candidate.checked_shapes)
                expected = candidate.expected_shape_ids[offset : offset + len(checked)]
                if checked != expected:
                    reason = "SCOUT replay shape order differs from the candidate validation plan"

        observations = self._all_gather_object((reason, checked))
        rejection = next(
            (item_reason for item_reason, _ in observations if item_reason is not None),
            None,
        )
        if rejection is not None:
            self.reject(rejection)
            return DurableCheckpointEvent.REJECTED
        if any(item_checked != checked for _, item_checked in observations):
            self.reject("SCOUT checked different replay shapes across checkpoint ranks")
            return DurableCheckpointEvent.REJECTED

        candidate = replace(
            candidate,
            checked_shapes=candidate.checked_shapes
            + tuple(ShapeValidation(shape_id=shape_id, step=int(step)) for shape_id in checked),
        )
        self._pending = candidate
        if candidate.complete:
            self._commit(candidate, completed_step=int(step))
            return DurableCheckpointEvent.COMMITTED

        if self._is_writer:
            self._store.write_pending(candidate)
        self._barrier()
        return DurableCheckpointEvent.PROGRESS

    def reject(self, reason: str) -> None:
        """Reject the pending candidate without changing the validated pointer."""
        candidate = self._pending
        if candidate is None:
            return
        if self._is_writer:
            self._reject_record(candidate, reason)
        self._barrier()
        self._pending = None

    def load_latest_validated(self) -> int | None:
        """Load exactly the checkpoint named by ``LATEST_SCOUT_VALIDATED``."""
        checkpoint = self._store.read_latest()
        if checkpoint is None:
            return None
        if (
            checkpoint.status not in ("recovery_verified", "validated")
            or checkpoint.verdict != "clean"
        ):
            raise RuntimeError("latest SCOUT checkpoint manifest is not recovery-verified")
        if not checkpoint.complete:
            raise RuntimeError("latest SCOUT checkpoint manifest has incomplete shape coverage")
        with self._measure_checkpoint_io("read", checkpoint.checkpoint_id):
            loaded_step = self.config.adapter.load_checkpoint(checkpoint)
        if loaded_step is None:
            raise RuntimeError("framework durable checkpoint loader did not return the loaded step")
        if int(loaded_step) != checkpoint.step:
            raise RuntimeError(
                "framework loaded a different durable checkpoint step: "
                f"manifest={checkpoint.step}, loaded={loaded_step}"
            )
        return int(loaded_step)

    def _commit(self, candidate: DurableCheckpointRecord, *, completed_step: int) -> None:
        committed = replace(
            candidate,
            status="recovery_verified",
            verdict="clean",
            completed_step=completed_step,
        )
        previous = self._latest
        if self._is_writer:
            self._store.commit(committed)
            callback = getattr(self.config.adapter, "commit_candidate", None)
            if callback is not None:
                try:
                    callback(committed, previous)
                except Exception:  # noqa: BLE001 - the committed pointer is authoritative
                    logger.exception(
                        "durable checkpoint commit notification failed for %s",
                        committed.checkpoint_id,
                    )
        self._barrier()
        self._latest = committed
        self._pending = None

    def _reject_record(self, candidate: DurableCheckpointRecord, reason: str) -> None:
        rejected = replace(candidate, status="rejected", verdict=reason)
        self._store.reject(rejected)
        callback = getattr(self.config.adapter, "quarantine_candidate", None)
        if callback is not None:
            try:
                callback(rejected, reason)
            except Exception:  # noqa: BLE001 - quarantine metadata is already durable
                logger.exception(
                    "durable checkpoint quarantine notification failed for %s",
                    rejected.checkpoint_id,
                )

    @property
    def _is_writer(self) -> bool:
        return self._rank == self.config.writer_rank

    def _barrier(self) -> None:
        if self._world_size > 1:
            dist.barrier(group=self._process_group)

    def _all_ranks_succeeded(self, local_success: bool) -> bool:
        if self._world_size == 1:
            return local_success
        device = torch.device("cpu")
        try:
            backend = str(dist.get_backend(self._process_group)).lower()
        except Exception:  # noqa: BLE001 - best-effort backend selection
            backend = ""
        if backend == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL durable-checkpoint status consensus requires CUDA")
            device = torch.device("cuda", torch.cuda.current_device())
        value = torch.tensor([int(local_success)], dtype=torch.int32, device=device)
        dist.all_reduce(value, op=dist.ReduceOp.MIN, group=self._process_group)
        return bool(value.item())

    def _all_gather_object(self, value: Any) -> list[Any]:
        if self._world_size == 1:
            return [value]
        gathered: list[Any] = [None] * self._world_size
        dist.all_gather_object(gathered, value, group=self._process_group)
        return gathered

    def _measure_checkpoint_io(
        self,
        operation: str,
        checkpoint_id: str,
    ) -> ContextManager[None]:
        if self._checkpoint_io is None:
            return nullcontext()
        return self._checkpoint_io(operation, checkpoint_id)


def _rotate_shape_ids(shape_ids: tuple[str, ...], first_shape_id: str) -> tuple[str, ...]:
    try:
        index = shape_ids.index(first_shape_id)
    except ValueError as exc:
        raise ValueError(
            f"first replay shape {first_shape_id!r} is not in the configured plan"
        ) from exc
    return shape_ids[index:] + shape_ids[:index]


class _DurableManifestStore:
    """Atomic JSON metadata store; checkpoint bytes remain framework-owned."""

    def __init__(self, folder: str) -> None:
        self.folder = Path(folder)
        self.records = self.folder / "records"
        self.folder.mkdir(parents=True, exist_ok=True)
        self.records.mkdir(parents=True, exist_ok=True)

    def read_latest(self) -> DurableCheckpointRecord | None:
        return self._read(self.folder / _LATEST_MANIFEST)

    def read_pending(self) -> DurableCheckpointRecord | None:
        return self._read(self.folder / _PENDING_MANIFEST)

    def write_pending(self, record: DurableCheckpointRecord) -> None:
        self._write_record(record)
        self._atomic_write(self.folder / _PENDING_MANIFEST, record.to_dict())

    def commit(self, record: DurableCheckpointRecord) -> None:
        self._write_record(record)
        self._atomic_write(self.folder / _LATEST_MANIFEST, record.to_dict())
        self._unlink_pending()

    def reject(self, record: DurableCheckpointRecord) -> None:
        self._write_record(record)
        self._unlink_pending()

    def clear_pending(self) -> None:
        self._unlink_pending()

    def next_validation_epoch(self) -> int:
        epochs = [
            record.validation_epoch
            for path in self.records.glob("*.json")
            if (record := self._read(path)) is not None
        ]
        return max(epochs, default=0) + 1

    def _write_record(self, record: DurableCheckpointRecord) -> None:
        self._atomic_write(
            self.records / f"{record.checkpoint_id}.json",
            record.to_dict(),
        )

    def _read(self, path: Path) -> DurableCheckpointRecord | None:
        if not path.exists():
            return None
        with path.open("r", encoding="utf-8") as handle:
            return DurableCheckpointRecord.from_dict(json.load(handle))

    def _unlink_pending(self) -> None:
        path = self.folder / _PENDING_MANIFEST
        try:
            path.unlink()
        except FileNotFoundError:
            return
        self._fsync_directory()

    def _atomic_write(self, path: Path, payload: Mapping[str, Any]) -> None:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        fd, temporary = tempfile.mkstemp(
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
        )
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
            self._fsync_directory(path.parent)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass

    def _fsync_directory(self, folder: Path | None = None) -> None:
        descriptor = os.open(folder or self.folder, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
