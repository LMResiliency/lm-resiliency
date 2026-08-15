# mypy: ignore-errors
"""InMemoryCheckpointManager: orchestrates the 2-phase in-memory checkpoint pipeline."""

from __future__ import annotations

import atexit
import enum
import hashlib
import json
import logging
import os
import pickle
import signal
import threading
import uuid
from pathlib import Path
from typing import Any, Callable

import torch
import torch.distributed as dist

from lm_resiliency.checkpointing.buffer import BufferPool, SlotState
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.copy import AsyncDeviceCopier
from lm_resiliency.checkpointing.disk import (
    CheckpointStatus,
    CheckpointStatusStore,
    DiskSerializer,
    atomic_copy_file,
)
from lm_resiliency.checkpointing.replication import ChunkedGlooBackend, PeerReplicator
from lm_resiliency.checkpointing.state_dict import (
    FlatStateDictMetadata,
    TensorEntry,
    flatten,
    unflatten,
)

try:  # DTensor (FSDP2 / torchtitan sharded params) — guarded for older torch
    from torch.distributed.tensor import DTensor
except Exception:  # noqa: BLE001
    DTensor = ()  # isinstance(x, ()) is always False → no-op on torch without DTensor

logger = logging.getLogger(__name__)


def _local_shard(t: torch.Tensor) -> torch.Tensor:
    """The local shard of a DTensor (FSDP2/torchtitan), else the tensor itself.

    The save_tensors fast path checkpoints raw tensor references; a DTensor is a sharded
    view, so we checkpoint each rank's *local* shard (``to_local()`` aliases the same
    storage, so a later in-place ``copy_`` on recovery writes straight back into the
    param). GEMINI thus stays plain-tensor throughout and never runs a distributed op."""
    return t.to_local() if isinstance(t, DTensor) else t


# Reserved non-tensor-data key for the opaque per-step ``extra`` blob a save_tensors caller
# rides alongside its shard (e.g. RNG state). Distinct from the "0".."N" tensor-skeleton keys.
_EXTRA_KEY = "__extra__"
_REPLICATION_META_MARKER = "lm_resiliency.peer-shard-metadata"

# Upper bound on how long the SIGTERM flush waits for an in-flight replica to land before
# abandoning it. On a coordinated stop every rank is signalled at once, so a rank can block
# on dist.recv from a peer that is already gone; without a bound the flush would stall until
# the launcher SIGKILLs the worker (losing the whole checkpoint). Well under the launcher's
# worker-shutdown grace, and ample for a real in-flight replica to complete.
_FLUSH_REPLICATION_TIMEOUT_S = 10.0


class RecoveryMode(str, enum.Enum):
    """Checkpoint trust level selected by failure recovery."""

    LATEST_GEMINI = "latest"
    RECOVERY_VERIFIED = "recovery_verified"


def _dumps_meta(
    non_tensor_data: Any,
    tensor_entries: list[TensorEntry] | None,
) -> bytes:
    """Serialize a peer shard's full reconstruction metadata."""
    if non_tensor_data is None and tensor_entries is None:
        return b""
    return pickle.dumps((_REPLICATION_META_MARKER, 1, non_tensor_data, tensor_entries))


def _loads_meta(blob: bytes) -> tuple[Any, list[TensorEntry] | None]:
    """Deserialize peer metadata, accepting the legacy skeleton-only payload."""
    if not blob:
        return None, None
    payload = pickle.loads(blob)
    if (
        isinstance(payload, tuple)
        and len(payload) == 4
        and payload[:2] == (_REPLICATION_META_MARKER, 1)
    ):
        return payload[2], payload[3]
    return payload, None


class InMemoryCheckpointManager:
    """Orchestrates async GPU-to-CPU capture and immediate peer replication.

    Pipeline:
      Phase 1 (step N): non-blocking GPU-to-CPU copy into own_current
      Phase 2 (copy completion): replicate own_current to its peer immediately

    Correct placement in training loop:
        for step in range(steps):
            forward()
            backward()
            optimizer.step()
            if step % interval == 0:
                mgr.save(...)   # AFTER optimizer.step() — captures updated params+optimizer

    Args:
        config: Checkpoint configuration.
        parallelism_info: ParallelismInfo or any object with dp_replicate/dp_shard attrs.
        process_group: Existing process group for coordination (default: WORLD).
    """

    def __init__(
        self,
        config: InMemoryCkptConfig,
        parallelism_info: Any | None = None,
        parallel_dims: Any | None = None,
        process_group: dist.ProcessGroup | None = None,
    ) -> None:
        self.config = config

        if not config.enable:
            return

        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1
        self._process_group = process_group

        par_info = parallelism_info or parallel_dims
        self._skip_replication = self._should_skip_replication(par_info)

        num_slots = 2 if self._skip_replication else 4
        self._buffer_pool = BufferPool(num_slots=num_slots, pin_memory=config.pin_memory)
        self._copier = AsyncDeviceCopier()
        self._metadata: FlatStateDictMetadata | None = None
        self._last_saved_step: int = -1
        self._completion_thread: threading.Thread | None = None
        self._completion_error: BaseException | None = None
        self._replication_source: Any | None = None

        # Peer replication
        jump = config.replication_jump
        if jump < 0:
            jump = torch.cuda.device_count() if torch.cuda.is_available() else 1
        if not self._skip_replication and self._world_size > 1:
            backend = ChunkedGlooBackend(
                replication_jump=jump,
                chunk_size=config.replication_chunk_size,
                world_size=self._world_size,
                rank=self._rank,
            )
            self._replicator = PeerReplicator(backend, enabled=True)
        else:
            self._replicator = PeerReplicator(None, enabled=False)

        self._run_id = self._resolve_checkpoint_run_id(config.run_id)
        self.config.run_id = self._run_id
        topology = self._checkpoint_topology(jump)
        self._require_exact_checkpoint_contract(self._run_id, topology)
        self._topology_id = hashlib.sha256(
            json.dumps(topology, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()

        # Node-local (fast) tier. GEMINI is deliberately single-tier: durable /
        # global checkpointing is the pre-training framework's job (reached via the
        # caller's load_fallback when GEMINI has nothing to load — e.g. a fresh
        # allocation whose node-local tier is empty).
        self._disk = DiskSerializer(
            folder=config.disk_folder,
            rank=self._rank,
            integrity=config.verify_integrity,
            run_id=self._run_id,
            topology_id=self._topology_id,
            namespace=True,
        )
        self._checkpoint_status = CheckpointStatusStore(
            self._disk._folder,
            rank=self._rank,
            run_id=self._run_id,
            topology_id=self._topology_id,
        )
        self._save_count = 0
        self._restart_dest_fn: Callable[[], str | Path | None] | None = None

        self._prev_sigterm_handler = signal.getsignal(signal.SIGTERM)
        self._prev_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGTERM, self._sigterm_handler)
        signal.signal(signal.SIGINT, self._sigterm_handler)
        self._exit_flush_registered = False

    def _resolve_checkpoint_run_id(self, configured: str | None) -> str:
        """Resolve one stable run id, coordinating a fresh id when necessary."""
        if configured is not None:
            if not isinstance(configured, str) or not configured.strip():
                raise ValueError("checkpoint run_id must be a non-empty string")
            return configured

        value = os.environ.get("LM_RESILIENCY_RUN_ID")
        if value and value.strip():
            return value
        value = os.environ.get("TORCHELASTIC_RUN_ID")
        if value and value.strip() and value.strip().lower() != "none":
            return value

        source = 0
        if dist.is_initialized() and self._process_group is not None:
            source = dist.get_process_group_ranks(self._process_group)[0]
        value = uuid.uuid4().hex if self._rank == source else ""
        if dist.is_initialized() and dist.get_world_size(self._process_group) > 1:
            values = [value]
            dist.broadcast_object_list(values, src=source, group=self._process_group)
            value = values[0]
        return value

    def _checkpoint_topology(self, replication_jump: int) -> dict[str, object]:
        group_ranks = list(range(self._world_size))
        if dist.is_initialized() and self._process_group is not None:
            group_ranks = list(dist.get_process_group_ranks(self._process_group))
        peer_map: dict[str, int] = {}
        if not self._skip_replication and self._world_size > 1:
            segment_size = replication_jump * 2
            for rank in group_ranks:
                segment_start = (rank // segment_size) * segment_size
                offset = rank - segment_start
                peer_map[str(rank)] = (
                    rank + replication_jump
                    if offset < replication_jump
                    else rank - replication_jump
                )
        return {
            "world_size": self._world_size,
            "checkpoint_group_ranks": group_ranks,
            "replication_jump": replication_jump if peer_map else None,
            "replication_peers": peer_map,
        }

    def _require_exact_checkpoint_contract(self, run_id: str, topology: dict[str, object]) -> None:
        """Fail before disk eligibility if checkpoint ranks disagree on identity."""
        if not dist.is_initialized() or dist.get_world_size(self._process_group) == 1:
            return
        encoded = json.dumps(
            {"run_id": run_id, "topology": topology},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        digest = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big") & ((1 << 63) - 1)
        device = torch.device("cpu")
        if str(dist.get_backend(self._process_group)).lower() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL checkpoint identity agreement requires a CUDA device")
            device = torch.device("cuda", torch.cuda.current_device())
        lower = torch.tensor([digest], dtype=torch.int64, device=device)
        upper = lower.clone()
        dist.all_reduce(lower, op=dist.ReduceOp.MIN, group=self._process_group)
        dist.all_reduce(upper, op=dist.ReduceOp.MAX, group=self._process_group)
        if lower.item() != upper.item():
            raise RuntimeError(
                "checkpoint ranks disagree on run_id or topology; node-local recovery is unsafe"
            )

    def _should_skip_replication(self, par_info: Any | None) -> bool:
        if not self.config.skip_replication_if_hsdp:
            return False
        if par_info is None:
            return False
        if hasattr(par_info, "has_natural_replicas"):
            return par_info.has_natural_replicas
        dp_replicate = getattr(par_info, "dp_replicate", 1)
        dp_shard = getattr(par_info, "dp_shard", 1)
        return dp_replicate > 1 and dp_shard > 1

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def save(
        self,
        state_dict: dict[str, Any],
        step: int,
    ) -> None:
        """Non-blocking save from a state_dict.

        Flattens the dict, launches an async GPU-to-CPU copy, and starts peer
        replication as soon as that copy completes. Call AFTER optimizer.step().
        """
        if not self.config.enable:
            return

        metadata, tensors = flatten(state_dict)

        if not self._buffer_pool.allocated:
            self._buffer_pool.allocate(metadata.tensor_entries)
            self._metadata = metadata

        self._do_save(tensors, metadata, step)

    def save_tensors(
        self,
        tensors: list[torch.Tensor],
        step: int,
        extra: dict[str, Any] | None = None,
    ) -> None:
        """Near-zero-overhead save: copy parameter tensor references directly.

        Avoids state_dict() and flatten(). Launches one async copy per tensor.
        Call AFTER optimizer.step() — the tensors won't be written again until
        the next optimizer.step(), giving the async copy a full forward+backward
        window to complete.

        Args:
            tensors: Direct references to parameter/state tensors on GPU.
            step: Current training step.
            extra: Optional small, schema-supported non-tensor state to checkpoint alongside
                this step's shard (e.g. RNG state for bitwise stochastic-forward recovery).
                Rides the per-rank non-tensor payload, so it inherits node-local
                flush and peer replication; ``load_tensors`` returns it back.
        """
        if not self.config.enable:
            return

        # Checkpoint the *local* shard of any DTensor (FSDP2/torchtitan) — a plain-tensor
        # view aliasing the same storage, so entries, the async copy, and the in-place
        # recovery copy all operate on the local bytes without any distributed op.
        tensors = [_local_shard(t) for t in tensors]

        if not self._buffer_pool.allocated:
            entries = [
                TensorEntry(key_path=(str(i),), shape=t.shape, dtype=t.dtype, device=t.device)
                for i, t in enumerate(tensors)
            ]
            self._buffer_pool.allocate(entries)
            self._metadata = FlatStateDictMetadata(
                tensor_entries=entries,
                non_tensor_data={str(i): None for i in range(len(tensors))},
            )

        # Fresh per-save non-tensor payload (so each buffer slot snapshots *its* step's
        # extra, not a shared mutable dict), carrying this step's extra under a reserved key.
        non_tensor = {str(i): None for i in range(len(tensors))}
        if extra is not None:
            non_tensor[_EXTRA_KEY] = extra
        md = FlatStateDictMetadata(
            tensor_entries=self._metadata.tensor_entries, non_tensor_data=non_tensor
        )
        self._do_save(tensors, md, step)

    def maybe_wait(self) -> None:
        """Wait for the host copy and launch replication, without waiting for transfer."""
        if not self.config.enable:
            return
        self._wait_for_completion_worker()

    def finalize_replication(self, timeout_s: float | None = None) -> bool:
        """Finish the active host copy and peer exchange.

        Records the received peer step and returns both the peer-receive slot and
        the local send slot to a clean READY state.
        Returns whether a complete aligned local/peer generation is available.

        An active source requires wait() even when the backend threads have
        already finished. The receive buffer is not committed to its slot until
        this method records the received step and metadata.
        """
        if not self.config.enable:
            return False
        self._wait_for_completion_worker()
        if not self._replicator.enabled:
            return True
        source = self._replication_source
        if source is None:
            return True
        # A bounded wait is used by the SIGTERM flush path, where a coordinated
        # stop can close the peer connection. Normal capture rotation waits
        # without a timeout because replication may legitimately span steps.
        try:
            peer_step, peer_meta = self._replicator.wait(timeout_s=timeout_s)
        except Exception as e:  # noqa: BLE001 — durability must survive a broken peer
            logger.warning(
                f"replication did not finish before flush ({e}); flushing complete buffers"
            )
            peer_step, peer_meta = 0, b""
        replicated_step = source.step
        replication_succeeded = peer_step > 0 and peer_step == replicated_step
        if peer_step > 0 and not replication_succeeded:
            logger.warning(
                "replication received peer step %s while sending step %s; "
                "dropping the unaligned generation",
                peer_step,
                replicated_step,
            )
        if replication_succeeded:
            self._buffer_pool.peer_current.step = peer_step
            peer_non_tensor, peer_entries = _loads_meta(peer_meta)
            self._buffer_pool.peer_current.non_tensor_data = peer_non_tensor
            self._buffer_pool.peer_current.tensor_entries = peer_entries
            self._buffer_pool.peer_current.state = SlotState.READY
        if source is not None and source.state == SlotState.REPLICATING:
            source.state = SlotState.READY
        self._replication_source = None
        return replication_succeeded

    def _collective_min_step(self, local_step: int) -> int:
        """MIN-reduce a local checkpoint step across the configured process group."""
        if not dist.is_initialized() or self._world_size == 1:
            return int(local_step)

        device = torch.device("cpu")
        try:
            backend = dist.get_backend(self._process_group)
        except Exception:  # noqa: BLE001 - best-effort backend detection
            backend = ""
        if str(backend).lower() == "nccl":
            if not torch.cuda.is_available():
                raise RuntimeError("NCCL checkpoint step reduction requires a CUDA device")
            device = torch.device("cuda", torch.cuda.current_device())

        tensor = torch.tensor([local_step], dtype=torch.int64, device=device)
        dist.all_reduce(tensor, op=dist.ReduceOp.MIN, group=self._process_group)
        return int(tensor.item())

    def _consistent_step(
        self,
        disk: DiskSerializer,
        *,
        before_step: int | None = None,
    ) -> int:
        """Newest exact generation present on every rank below ``before_step``."""
        candidate = self._collective_min_step(disk.find_latest_on_disk(before_step=before_step))
        while candidate > 0:
            local_step = (
                candidate
                if disk.has_rank(candidate, self._rank)
                else disk.find_latest_on_disk(before_step=candidate)
            )
            agreed = self._collective_min_step(local_step)
            if agreed == candidate:
                return candidate
            candidate = agreed
        return -1

    def _latest_memory_step(self) -> int:
        """Latest complete own checkpoint currently resident in local memory."""
        if self._metadata is None or not self._buffer_pool.allocated:
            return -1
        return max(
            (
                slot.step
                for slot in self._buffer_pool.own_slots
                if slot.state in (SlotState.READY, SlotState.REPLICATING) and slot.step > 0
            ),
            default=-1,
        )

    def _consistent_memory_step(self) -> int:
        """Latest exact in-memory step retained by every rank."""
        step = self._collective_min_step(self._latest_memory_step())
        if step <= 0:
            return -1
        local_has_step = self._memory_slot_by_step(step) is not None
        return step if self._collective_min_step(int(local_has_step)) == 1 else -1

    def _latest_recovery_steps(self) -> tuple[int, int]:
        """Return (memory_step, disk_step), each made rank-consistent."""
        return self._consistent_memory_step(), self._consistent_step(self._disk)

    def _resolved_recovery_mode(
        self,
        mode: RecoveryMode | str | None,
    ) -> RecoveryMode:
        if mode is not None:
            return RecoveryMode(mode)
        status = self._checkpoint_status.read()
        try:
            return RecoveryMode(status.recovery_mode)
        except ValueError:
            logger.warning(
                "Unknown persisted GEMINI recovery mode %r; using verified recovery",
                status.recovery_mode,
            )
            return RecoveryMode.RECOVERY_VERIFIED

    def _activate_recovery_mode(
        self,
        mode: RecoveryMode | str | None,
    ) -> RecoveryMode:
        """Persist an actual recovery decision and invalidate unsafe candidates."""
        resolved = self._resolved_recovery_mode(mode)
        status = self._checkpoint_status.read()
        if status.recovery_mode != resolved.value or (
            resolved is RecoveryMode.RECOVERY_VERIFIED and status.candidate_step > 0
        ):
            self.set_recovery_mode(resolved)
        return resolved

    def _verified_disk_step(self) -> int:
        status = self._checkpoint_status.read()
        local_step = status.recovery_verified_step
        if local_step > 0 and not self._disk.has_rank(local_step, self._rank):
            local_step = -1
        return self._collective_min_step(local_step)

    def _recovery_steps(
        self,
        mode: RecoveryMode | str | None,
    ) -> tuple[int, int]:
        resolved = self._resolved_recovery_mode(mode)
        if resolved is RecoveryMode.RECOVERY_VERIFIED:
            return -1, self._verified_disk_step()
        return self._latest_recovery_steps()

    def _memory_slot_by_step(self, step: int) -> Any | None:
        if self._metadata is None or not self._buffer_pool.allocated:
            return None
        return next(
            (
                slot
                for slot in self._buffer_pool.own_slots
                if slot.step == step and slot.state in (SlotState.READY, SlotState.REPLICATING)
            ),
            None,
        )

    def find_latest(self, mode: RecoveryMode | str | None = None) -> int:
        """Collective: latest step recoverable across all ranks from memory or disk.

        -1 if none (caller then falls back to framework recovery, such as on a
        fresh allocation whose node-local tier is empty).
        """
        if not self.config.enable:
            return -1
        resolved = self._resolved_recovery_mode(mode)
        memory_step, disk_step = self._recovery_steps(resolved)
        if disk_step > memory_step:
            loaded = self._load_latest_collectively_validated_disk_shard(
                disk_step,
                allow_older=resolved is RecoveryMode.LATEST_GEMINI,
            )
            disk_step = loaded[2] if loaded is not None else -1
        step = max(memory_step, disk_step)
        return step if step > 0 else -1

    def local_recovery_step(self, mode: RecoveryMode | str | None = None) -> int:
        """Return this rank's newest eligible checkpoint without a collective.

        Recovery observers use this method while the current process group may be
        unhealthy. The external manager can combine rank-local decisions; the
        restarted job still uses :meth:`find_latest` to select a rank-consistent
        generation before loading checkpoint data.
        """
        if not self.config.enable:
            return -1
        resolved = self._resolved_recovery_mode(mode)
        if resolved is RecoveryMode.RECOVERY_VERIFIED:
            status = self._checkpoint_status.read()
            step = status.recovery_verified_step
            return step if step > 0 and self._disk.has_rank(step, self._rank) else -1
        step = max(self._latest_memory_step(), self._disk.find_latest_on_disk())
        return step if step > 0 else -1

    def load(
        self,
        mode: RecoveryMode | str | None = None,
    ) -> tuple[dict[str, Any], int] | None:
        """Recover the latest rank-consistent checkpoint from memory or node-local disk.

        All ranks call collectively; each tier's step is the MIN across ranks so
        every rank has a valid shard. Returns None if there is none (e.g.
        correlated loss of a replication pair, or an empty node-local tier on a
        fresh allocation) — caller invokes framework recovery via load_fallback.
        """
        if not self.config.enable:
            return None

        resolved = self._activate_recovery_mode(mode)
        memory_step, disk_step = self._recovery_steps(resolved)
        if memory_step <= 0 and disk_step <= 0:
            return None

        if memory_step >= disk_step and memory_step > 0:
            slot = self._memory_slot_by_step(memory_step)
            if slot is not None:
                metadata = self._slot_metadata(slot)
                state_dict = unflatten(metadata, [t.clone() for t in slot.tensors])
                logger.info(f"Loaded checkpoint from memory at step {memory_step}")
                return state_dict, memory_step

        latest = disk_step
        if latest <= 0:
            return None
        loaded = self._load_latest_collectively_validated_disk_shard(
            latest,
            allow_older=resolved is RecoveryMode.LATEST_GEMINI,
        )
        if loaded is None:
            return None
        metadata, tensors, latest = loaded
        state_dict = unflatten(metadata, tensors)
        self._metadata = metadata
        logger.info(f"Loaded checkpoint from {self._disk._folder} at step {latest}")
        return state_dict, latest

    def load_tensors(
        self,
        mode: RecoveryMode | str | None = None,
    ) -> tuple[list[torch.Tensor], int, dict[str, Any] | None] | None:
        """Recover checkpoint as a flat tensor list (inverse of save_tensors).

        Returns ``(tensors, step, extra)`` where ``extra`` is the non-tensor blob the
        matching ``save_tensors(..., extra=)`` stored for the recovered step (or None).
        """
        if not self.config.enable:
            return None

        resolved = self._activate_recovery_mode(mode)
        memory_step, disk_step = self._recovery_steps(resolved)
        if memory_step <= 0 and disk_step <= 0:
            return None

        if memory_step >= disk_step and memory_step > 0:
            slot = self._memory_slot_by_step(memory_step)
            if slot is not None:
                metadata = self._slot_metadata(slot)
                extra = (metadata.non_tensor_data or {}).get(_EXTRA_KEY)
                logger.info(f"Loaded checkpoint tensors from memory at step {memory_step}")
                return [t.clone() for t in slot.tensors], memory_step, extra

        latest = disk_step
        if latest <= 0:
            return None
        loaded = self._load_latest_collectively_validated_disk_shard(
            latest,
            allow_older=resolved is RecoveryMode.LATEST_GEMINI,
        )
        if loaded is None:
            return None
        metadata, tensors, latest = loaded
        self._metadata = metadata
        extra = (metadata.non_tensor_data or {}).get(_EXTRA_KEY)
        logger.info(f"Loaded checkpoint tensors from {self._disk._folder} at step {latest}")
        return tensors, latest, extra

    def _load_collectively_validated_disk_shard(
        self,
        step: int,
    ) -> tuple[FlatStateDictMetadata, list[torch.Tensor]] | None:
        """Load locally, then require every rank to accept its selected shard."""
        loaded: tuple[FlatStateDictMetadata, list[torch.Tensor]] | None = None
        local_error: Exception | None = None
        try:
            loaded = self._disk.load(step)
        except Exception as error:  # noqa: BLE001 - every rank must reach the validity vote
            local_error = error

        all_valid = self._collective_min_step(int(loaded is not None)) == 1
        if all_valid:
            return loaded

        if local_error is not None:
            logger.error(
                "Checkpoint validation failed at step %s: %s — all ranks will "
                "consider an older eligible generation",
                step,
                local_error,
            )
        else:
            logger.error(
                "A peer rejected its checkpoint shard at step %s — all ranks will "
                "consider an older eligible generation",
                step,
            )
        return None

    def _load_latest_collectively_validated_disk_shard(
        self,
        step: int,
        *,
        allow_older: bool,
    ) -> tuple[FlatStateDictMetadata, list[torch.Tensor], int] | None:
        """Load the newest collectively valid generation, optionally descending."""
        candidate = step
        while candidate > 0:
            loaded = self._load_collectively_validated_disk_shard(candidate)
            if loaded is not None:
                return loaded[0], loaded[1], candidate
            if not allow_older:
                break
            candidate = self._consistent_step(self._disk, before_step=candidate)
        return None

    @property
    def checkpoint_status(self) -> CheckpointStatus:
        """Current persisted candidate, verified pointer, and recovery mode."""
        return self._checkpoint_status.read()

    def set_recovery_mode(self, mode: RecoveryMode | str) -> None:
        """Persist the recovery source selected by SCOUT or the launcher."""
        resolved = RecoveryMode(mode)
        status = self._checkpoint_status.read()
        self._checkpoint_status.write(
            CheckpointStatus(
                candidate_step=(
                    -1 if resolved is RecoveryMode.RECOVERY_VERIFIED else status.candidate_step
                ),
                recovery_verified_step=status.recovery_verified_step,
                recovery_mode=resolved.value,
            )
        )

    def clear_candidate(self) -> None:
        """Clear candidate lineage without changing the selected recovery mode."""
        status = self._checkpoint_status.read()
        self._checkpoint_status.write(
            CheckpointStatus(
                candidate_step=-1,
                recovery_verified_step=status.recovery_verified_step,
                recovery_mode=status.recovery_mode,
            )
        )

    def reject_candidate(self) -> None:
        """Reject the current candidate and require verified recovery."""
        self.set_recovery_mode(RecoveryMode.RECOVERY_VERIFIED)

    def persist_cycle_boundary(self, step: int) -> CheckpointStatus:
        """Persist a cycle checkpoint and atomically advance its trust roles.

        The previous CANDIDATE becomes RECOVERY_VERIFIED. The supplied completed
        in-memory generation is written with its received peer replica and becomes
        the new CANDIDATE.
        """
        return self._persist_boundary(step, verify_current=False)

    def persist_verified_boundary(self, step: int) -> CheckpointStatus:
        """Persist a clean dense checkpoint as immediately recovery-verified."""
        return self._persist_boundary(step, verify_current=True)

    def _persist_boundary(
        self,
        step: int,
        *,
        verify_current: bool,
    ) -> CheckpointStatus:
        if not self.config.enable or self._metadata is None:
            raise RuntimeError("cannot persist a GEMINI cycle boundary before a checkpoint save")

        replication_succeeded = self.finalize_replication()
        if self._replicator.enabled and not replication_succeeded:
            raise RuntimeError(
                f"cannot persist checkpoint step {step}: peer replication did not complete"
            )

        own_slot = next(
            (
                slot
                for slot in self._buffer_pool.own_slots
                if slot.step == step and slot.state == SlotState.READY
            ),
            None,
        )
        local_step = own_slot.step if own_slot is not None else -1
        if self._collective_min_step(local_step) != step:
            raise RuntimeError(f"checkpoint step {step} is not complete on every checkpoint rank")
        assert own_slot is not None
        self._disk.save_sync(
            self._slot_metadata(own_slot),
            own_slot.tensors,
            step,
            rank=self._rank,
        )

        peer_rank = self._replicator.peer_rank
        if self._buffer_pool._num_slots >= 4 and peer_rank >= 0:
            peer_slot = next(
                (
                    slot
                    for slot in (
                        self._buffer_pool.peer_current,
                        self._buffer_pool.peer_previous,
                    )
                    if slot.step == step and slot.state == SlotState.READY
                ),
                None,
            )
            if peer_slot is None:
                raise RuntimeError(f"checkpoint step {step} has no completed peer replica")
            self._disk.save_sync(
                self._slot_metadata(peer_slot),
                peer_slot.tensors,
                step,
                rank=peer_rank,
            )

        previous = self._checkpoint_status.read()
        verified_step = step
        candidate_step = -1
        if not verify_current:
            verified_step = (
                previous.candidate_step
                if previous.candidate_step > 0
                else previous.recovery_verified_step
            )
            candidate_step = step
        status = CheckpointStatus(
            candidate_step=candidate_step,
            recovery_verified_step=verified_step,
            recovery_mode=RecoveryMode.LATEST_GEMINI.value,
        )
        self._checkpoint_status.write(status)
        return status

    def flush_for_restart(self) -> int:
        """Persist the latest complete in-memory checkpoint to node-local disk/NVMe.

        This is GEMINI's durability boundary for the torchrun restart model.
        torchrun restarts *all* worker processes on a failure, freeing their
        address space — so the in-memory buffers do not survive. Before the
        restart happens (SIGTERM grace window, or on demand from the recovery
        supervisor), we flush the buffers to node-local NVMe so the restarted
        worker can reload them without touching a remote/global filesystem.

        This node persists **every** READY buffer it holds:
          - own slots  → rank-{self._rank}.pt   (this rank's shard at the
            current and previous steps)
          - peer slots → rank-{peer_rank}.pt     (the replica of the paired rank,
            so that rank's shard is recoverable *even if its node died*)

        Replication starts when the current D2H copy completes. While that
        exchange is in flight, ``own_previous`` and ``peer_previous`` retain the
        prior aligned generation. Flushing every READY slot therefore
        materializes a rank-consistent fallback even when the current exchange
        fails.

        Peer buffers carry the sender's tensor metadata and are resized when its
        shard layout differs from this rank's layout.

        Blocking. Returns the latest own step flushed, or -1 if nothing flushed.
        """
        flushed_step = self._flush_slots(self._disk)
        self._disarm_exit_flush()
        return flushed_step

    def set_restart_destination(
        self,
        resolver: Callable[[], str | Path | None] | None,
    ) -> None:
        """Configure an optional mirror destination for signal-triggered flushes.

        The caller owns all policy. GEMINI only resolves the path after flushing
        node-local shards and copies the exact serialized files there.
        """
        self._restart_dest_fn = resolver

    def copy_to(self, destination: str | Path) -> int:
        """Copy already-flushed own and peer shards to ``destination``.

        Returns the latest copied step, or ``-1`` when no flushed shard exists.
        """
        if not self.config.enable:
            return -1
        source = self._disk._folder
        if not source.exists():
            return -1
        destination = DiskSerializer.namespace_folder(destination, self._run_id, self._topology_id)
        ranks = {self._rank}
        peer_rank = self._replicator.peer_rank
        if peer_rank >= 0:
            ranks.add(peer_rank)

        latest = -1
        for step_dir in sorted(source.glob("step-*")):
            try:
                step = int(step_dir.name.split("-", 1)[1])
            except ValueError:
                continue
            for rank in ranks:
                shard = step_dir / f"rank-{rank}.pt"
                if not shard.exists():
                    continue
                output = destination / step_dir.name / shard.name
                output.parent.mkdir(parents=True, exist_ok=True)
                atomic_copy_file(shard, output)
                latest = max(latest, step)
        status = self.checkpoint_status
        for rank in ranks:
            CheckpointStatusStore(
                destination,
                rank=rank,
                run_id=self._run_id,
                topology_id=self._topology_id,
            ).write(status)
        return latest

    def _flush_slots(self, disk: DiskSerializer) -> int:
        """Persist every READY buffer (own steps under self._rank, peer replicas
        under the peer's rank, each with its own non-tensor skeleton) to `disk`."""
        if not self.config.enable or self._metadata is None:
            return -1

        # Land any in-flight D2H copy and P2P replication into the buffers first.
        self.finalize_replication(timeout_s=_FLUSH_REPLICATION_TIMEOUT_S)

        own_step = -1
        for slot in self._buffer_pool.own_slots:
            if slot.state == SlotState.READY and slot.step > 0:
                disk.save_sync(self._slot_metadata(slot), slot.tensors, slot.step, rank=self._rank)
                own_step = max(own_step, slot.step)

        peer_rank = self._replicator.peer_rank
        if self._buffer_pool._num_slots >= 4 and peer_rank >= 0:
            for slot in (self._buffer_pool.peer_current, self._buffer_pool.peer_previous):
                if slot.state == SlotState.READY and slot.step > 0:
                    # Use the *peer's* non-tensor metadata (received via replication),
                    # so peer recovery reconstructs the sender's state, not this rank's.
                    disk.save_sync(
                        self._slot_metadata(slot), slot.tensors, slot.step, rank=peer_rank
                    )

        return own_step

    def _slot_metadata(self, slot: Any) -> FlatStateDictMetadata:
        """Metadata for flushing a slot: shared tensor structure + the slot's own
        non-tensor skeleton (per step, and per rank for a replica slot)."""
        non_tensor = slot.non_tensor_data
        if non_tensor is None:
            non_tensor = self._metadata.non_tensor_data
        return FlatStateDictMetadata(
            tensor_entries=slot.tensor_entries or self._metadata.tensor_entries,
            non_tensor_data=non_tensor,
        )

    def flush_to_disk(self, step: int | None = None) -> None:
        """Trigger async disk flush of the latest or specified in-memory checkpoint."""
        if not self.config.enable or self._metadata is None:
            return
        target_step = step or self._last_saved_step
        slot = self._buffer_pool.get_slot_by_step(target_step)
        if slot is not None:
            self._disk.save_async(self._slot_metadata(slot), slot.tensors, target_step)

    def close(self) -> None:
        """Cleanup: wait for all pending operations and release resources."""
        if not self.config.enable:
            return
        self._disarm_exit_flush()
        signal.signal(signal.SIGTERM, self._prev_sigterm_handler)
        signal.signal(signal.SIGINT, self._prev_sigint_handler)
        self.finalize_replication()
        self._replicator.close()
        self._disk.close()

    # ──────────────────────────────────────────────────────────────────────────
    # Internal
    # ──────────────────────────────────────────────────────────────────────────

    def _arm_exit_flush(self) -> None:
        """Preserve a surviving peer replica when training exits by exception.

        A hard failure in one rank commonly breaks a collective in its peer before
        the supervisor can deliver SIGTERM. Python still runs atexit handlers for
        that uncaught exception, giving the peer a final chance to persist its own
        shard and the failed rank's completed replica.
        """
        if not self._exit_flush_registered:
            atexit.register(self._flush_on_exit)
            self._exit_flush_registered = True

    def _disarm_exit_flush(self) -> None:
        if self._exit_flush_registered:
            atexit.unregister(self._flush_on_exit)
            self._exit_flush_registered = False

    def _flush_and_mirror_for_restart(self) -> int:
        flushed_step = self.flush_for_restart()
        logger.warning(
            "Flushed step %s (own + peer shards) before process exit",
            flushed_step,
        )
        destination = self._restart_dest_fn() if self._restart_dest_fn is not None else None
        if destination is not None:
            copied_step = self.copy_to(destination)
            logger.warning(
                "Copied step %s (own + peer shards) to restart destination %s",
                copied_step,
                destination,
            )
        return flushed_step

    def _flush_on_exit(self) -> None:
        """Best-effort flush when an uncaught exception terminates this worker."""
        self._disarm_exit_flush()
        if self._metadata is None:
            return
        logger.warning("Process exiting without checkpoint-manager close; flushing restart state")
        try:
            self._flush_and_mirror_for_restart()
        except BaseException:  # noqa: BLE001 - interpreter shutdown must continue
            logger.exception("Failed to flush restart state during interpreter shutdown")

    def _sigterm_handler(self, signum: int, frame: Any) -> None:
        """Flush the latest complete in-memory checkpoint to disk on SIGTERM.

        Job schedulers (SLURM, Kubernetes) and torchrun send SIGTERM before
        SIGKILL with a grace period (typically 30-60s). We use this window to
        persist the latest complete in-memory checkpoint, including this rank's own
        shard and the peer replica it holds, to node-local NVMe via
        flush_for_restart().
        """
        logger.warning("SIGTERM received — flushing in-memory checkpoint to disk")

        if self._metadata is not None:
            self._flush_and_mirror_for_restart()
        self._disarm_exit_flush()

        # Chain to previous handler
        if signum == signal.SIGTERM:
            prev = self._prev_sigterm_handler
        else:
            prev = self._prev_sigint_handler

        if callable(prev):
            prev(signum, frame)
        elif signum == signal.SIGINT:
            raise KeyboardInterrupt
        else:
            raise SystemExit(128 + signum)

    def _do_save(
        self,
        tensors: list[torch.Tensor],
        metadata: FlatStateDictMetadata | None,
        step: int,
    ) -> None:
        """Start a host copy whose completion immediately launches replication."""
        self._prepare_next_capture()

        # Async GPU→CPU copy into own_current
        target_slot = self._buffer_pool.own_current
        self._copier.start_copy(tensors, target_slot.tensors)
        target_slot.step = step
        target_slot.state = SlotState.COPYING
        target_slot.tensor_entries = metadata.tensor_entries if metadata is not None else None
        # Capture this step's non-tensor skeleton per slot (owned snapshot from
        # flatten), so an older slot flushes with its own step's values.
        target_slot.non_tensor_data = metadata.non_tensor_data if metadata is not None else None

        self._finalize_save(step, metadata)
        self._start_completion_worker(target_slot)

    def _prepare_next_capture(self) -> None:
        """Retire the active exchange, then free aligned local and peer slots."""
        if self._save_count == 0:
            return
        replication_succeeded = self.finalize_replication()
        if not self._replicator.enabled or replication_succeeded:
            self._buffer_pool.rotate()
            return

        # Keep own_previous/peer_previous as the last aligned recovery pair.
        # The incomplete current exchange is overwritten by the next capture.
        for slot in (self._buffer_pool.own_current, self._buffer_pool.peer_current):
            slot.step = -1
            slot.state = SlotState.EMPTY
            slot.non_tensor_data = None
            slot.tensor_entries = None

    def _start_completion_worker(self, target_slot: Any) -> None:
        """Wait for D2H completion and launch the peer exchange in the background."""
        self._completion_error = None

        def _complete_and_replicate() -> None:
            try:
                self._copier.wait()
                target_slot.state = SlotState.READY
                if not self._replicator.enabled:
                    return

                peer_slot = self._buffer_pool.peer_current
                peer_slot.step = -1
                peer_slot.state = SlotState.EMPTY
                peer_slot.non_tensor_data = None
                self._replicator.start_replication(
                    send_tensors=target_slot.tensors,
                    send_step=target_slot.step,
                    recv_buffers=peer_slot.tensors,
                    send_meta=_dumps_meta(
                        target_slot.non_tensor_data,
                        target_slot.tensor_entries,
                    ),
                )
                target_slot.state = SlotState.REPLICATING
                self._replication_source = target_slot
            except BaseException as exc:  # noqa: BLE001 - surface on the owner thread
                self._completion_error = exc

        self._completion_thread = threading.Thread(
            target=_complete_and_replicate,
            daemon=True,
            name=f"gemini-copy-step-{target_slot.step}",
        )
        self._completion_thread.start()

    def _wait_for_completion_worker(self) -> None:
        thread = self._completion_thread
        if thread is not None:
            thread.join()
            self._completion_thread = None
        if self._completion_error is not None:
            error = self._completion_error
            self._completion_error = None
            raise RuntimeError("checkpoint copy or replication launch failed") from error

    def _finalize_save(
        self,
        step: int,
        metadata: FlatStateDictMetadata | None,
    ) -> None:
        """Update bookkeeping and optionally flush to disk."""
        self._last_saved_step = step
        self._save_count += 1
        if metadata is not None:
            self._metadata = metadata
        self._arm_exit_flush()

        if (
            self.config.disk_flush_interval > 0
            and step % self.config.disk_flush_interval == 0
            and self._buffer_pool.own_previous.step > 0
        ):
            slot = self._buffer_pool.own_previous
            self._disk.save_async(
                metadata=self._slot_metadata(slot),
                tensors=slot.tensors,
                step=slot.step,
            )
