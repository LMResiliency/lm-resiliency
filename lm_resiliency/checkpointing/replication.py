"""Peer-to-peer checkpoint replication with pluggable backends.

Design: Chunk-Based Replication Without Training Interference
=============================================================

Problem: Checkpoint replication (CPU→NIC→fabric→NIC→CPU) shares the same NIC
as FSDP all-gather (GPU→NIC→fabric→NIC→GPU). Both use RDMA. How to avoid
checkpoint traffic delaying all-gather without profiling the training timeline?

Key insight: FSDP prefetches the next layer's all-gather during the current
layer's compute. The prefetch buffer = one layer's compute time. As long as
the all-gather isn't delayed beyond this buffer, forward computation doesn't
stall and training throughput is unchanged.

    Layer N:    [════════ compute (T_compute) ════════]
    Layer N+1:  [AG]...............[wait]...............[compute starts]
                |←──── prefetch buffer = T_compute ────→|

The maximum delay a checkpoint chunk can impose on an all-gather = time to
drain one in-flight chunk through the NIC:

    max_ag_delay = chunk_size / nic_bandwidth

By choosing chunk_size such that max_ag_delay << prefetch_buffer, we guarantee
zero impact on training throughput without any profiling or scheduling.

Example (Llama 3.1 8B, FSDP across 4 nodes):
    T_compute per layer:    9.4 ms (measured on single node with NVLink-only FSDP;
                            may differ on multi-node where AG uses inter-node NIC)
    AG time per layer:      ~1 ms (computed: 47MB / 50 GB/s per NIC)
    Prefetch buffer:        ~8.4 ms (= T_compute - AG_time)

With chunk_size chosen by estimate_chunk_size(max_ag_delay_fraction=0.05):
    chunk_size = 21 MB, max_ag_delay = 420 μs, safety margin = 20×

Backend note: GlooBackend uses TCP (~3 GB/s) — suitable for correctness
testing but too slow for production. Production deployments should use an
RDMA-native backend (e.g., NIXL, ibverbs, or NCCL on a separate group)
to achieve full NIC bandwidth (~50 GB/s per NIC).
"""

from __future__ import annotations

import json
import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


def estimate_chunk_size(
    nic_bandwidth_gbps: float = 400.0,
    layer_compute_ms: float = 9.4,
    ag_time_ms: float = 1.0,
    max_ag_delay_fraction: float = 0.05,
) -> int:
    """Estimate the replication chunk size that won't delay training.

    Computes the largest chunk size such that the worst-case head-of-line
    blocking delay on an all-gather stays within a fraction of the available
    prefetch buffer.

    The prefetch buffer is the slack between when an all-gather completes and
    when computation actually needs the result:

        prefetch_buffer = layer_compute_time - ag_time

    We set:

        chunk_size = prefetch_buffer × max_ag_delay_fraction × nic_bandwidth

    This guarantees that even if a checkpoint chunk is fully in-flight when an
    AG arrives, the AG is delayed by at most (max_ag_delay_fraction × buffer),
    leaving (1 - fraction) × buffer of slack remaining.

    Args:
        nic_bandwidth_gbps: Per-GPU NIC bandwidth in Gbps (e.g., 400 for CX-7).
        layer_compute_ms: Forward compute time per transformer layer in ms.
            Measure with: total_forward_time / num_layers.
        ag_time_ms: All-gather time per layer in ms. For hierarchical AG with
            N NICs per node: layer_params_bytes * (N_nodes-1) / N_nodes / N_nics / nic_bw.
        max_ag_delay_fraction: Maximum fraction of the prefetch buffer that one
            chunk can consume. Default 0.05 (5%) provides 20× safety margin.

    Returns:
        Recommended chunk size in bytes. Clamped to [64 KiB, 64 MiB].

    Example:
        >>> estimate_chunk_size(nic_bandwidth_gbps=400, layer_compute_ms=9.4,
        ...                     ag_time_ms=1.0, max_ag_delay_fraction=0.05)
        21000000  # ~21 MB → max AG delay = 420 μs (5% of 8.4ms buffer)
    """
    nic_bw_bytes_per_ms = (nic_bandwidth_gbps / 8) * 1e9 / 1000  # bytes/ms
    prefetch_buffer_ms = layer_compute_ms - ag_time_ms
    if prefetch_buffer_ms <= 0:
        return 64 * 1024  # minimum: network is the bottleneck

    max_delay_ms = prefetch_buffer_ms * max_ag_delay_fraction
    chunk_bytes = int(max_delay_ms * nic_bw_bytes_per_ms)

    min_chunk = 64 * 1024  # 64 KiB
    max_chunk = 64 * 1024 * 1024  # 64 MiB
    return max(min_chunk, min(chunk_bytes, max_chunk))


def _send_bytes(peer_rank: int, group, blob: bytes) -> None:
    """Send a length-prefixed byte blob (serialized non-tensor metadata)."""
    length = torch.tensor([len(blob)], dtype=torch.int64)
    dist.send(length, dst=peer_rank, group=group)
    if blob:
        buf = torch.frombuffer(bytearray(blob), dtype=torch.uint8)
        dist.send(buf, dst=peer_rank, group=group)


def _recv_bytes(peer_rank: int, group) -> bytes:
    """Receive a length-prefixed byte blob."""
    length = torch.tensor([0], dtype=torch.int64)
    dist.recv(length, src=peer_rank, group=group)
    n = int(length.item())
    if n == 0:
        return b""
    buf = torch.empty(n, dtype=torch.uint8)
    dist.recv(buf, src=peer_rank, group=group)
    return buf.numpy().tobytes()


def _tensor_layout(tensors: list[torch.Tensor]) -> list[dict[str, Any]]:
    """Return a JSON-safe tensor layout for allocating a peer's shard."""
    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype).removeprefix("torch."),
        }
        for tensor in tensors
    ]


def _prepare_receive_buffers(
    buffers: list[torch.Tensor],
    layout: list[dict[str, Any]],
) -> None:
    """Resize a receive slot when the peer owns a different shard layout."""
    current = _tensor_layout(buffers)
    if current == layout:
        return

    pin_memory = bool(buffers) and buffers[0].is_pinned()
    allocated = []
    for spec in layout:
        dtype = getattr(torch, spec["dtype"], None)
        if not isinstance(dtype, torch.dtype):
            raise ValueError(f"Unsupported peer checkpoint dtype: {spec['dtype']}")
        allocated.append(
            torch.empty(
                tuple(spec["shape"]),
                dtype=dtype,
                device="cpu",
                pin_memory=pin_memory,
            )
        )
    buffers[:] = allocated


def _send_header(
    peer_rank: int,
    group: dist.ProcessGroup,
    tensors: list[torch.Tensor],
    step: int,
    meta: bytes,
) -> None:
    step_tensor = torch.tensor([step], dtype=torch.int64)
    dist.send(step_tensor, dst=peer_rank, group=group)
    _send_bytes(peer_rank, group, meta)
    layout = json.dumps(_tensor_layout(tensors), separators=(",", ":")).encode()
    _send_bytes(peer_rank, group, layout)


def _recv_header(
    peer_rank: int,
    group: dist.ProcessGroup,
    buffers: list[torch.Tensor],
) -> tuple[int, bytes]:
    step_tensor = torch.tensor([0], dtype=torch.int64)
    dist.recv(step_tensor, src=peer_rank, group=group)
    meta = _recv_bytes(peer_rank, group)
    layout = json.loads(_recv_bytes(peer_rank, group))
    _prepare_receive_buffers(buffers, layout)
    return int(step_tensor.item()), meta


class ReplicationBackend(ABC):
    """Abstract interface for P2P transfer of a checkpoint shard between peers.

    A shard is the tensor list plus an opaque non-tensor metadata blob (the
    serialized skeleton: dataloader/RNG/step values). Both must cross so recovery
    can reconstruct the source rank's full state, not just its tensors.
    """

    @abstractmethod
    def send(self, tensors: list[torch.Tensor], step: int, meta: bytes = b"") -> None:
        """Send tensor list + non-tensor metadata blob to the paired peer. Blocking."""

    @abstractmethod
    def recv(self, buffers: list[torch.Tensor]) -> tuple[int, bytes]:
        """Receive tensors into buffers, resizing for the peer layout if needed.
        Returns (step, meta_blob). Blocking."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    @property
    def peer_rank(self) -> int:
        """Global rank of the paired peer this backend replicates to/from.

        Returns -1 if the backend has no peer (e.g., replication disabled).
        """
        return -1


class _GlooPeerBackend(ReplicationBackend):
    """Shared rank pairing and process-group setup for Gloo backends."""

    def __init__(
        self,
        replication_jump: int,
        backend_name: str,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError(f"torch.distributed must be initialized before {backend_name}")

        self._rank = rank if rank is not None else dist.get_rank()
        self._world_size = world_size if world_size is not None else dist.get_world_size()

        if replication_jump <= 0:
            raise ValueError(f"replication_jump must be > 0, got {replication_jump}")
        if self._world_size % (replication_jump * 2) != 0:
            raise ValueError(
                f"world_size ({self._world_size}) must be divisible by "
                f"2 * replication_jump ({2 * replication_jump})"
            )

        self._jump = replication_jump
        self._peer_rank = self._compute_peer_rank()
        self._group = self._create_group()

    @property
    def peer_rank(self) -> int:
        return self._peer_rank

    def _compute_peer_rank(self) -> int:
        """Compute the peer rank using replication_jump within a segment."""
        segment_size = self._jump * 2
        segment_start = (self._rank // segment_size) * segment_size
        offset_in_segment = self._rank - segment_start
        if offset_in_segment < self._jump:
            return self._rank + self._jump
        return self._rank - self._jump

    def _create_group(self) -> dist.ProcessGroup:
        """Create Gloo process groups for all replication pairs."""
        segment_size = self._jump * 2
        my_group: dist.ProcessGroup | None = None

        for seg_start in range(0, self._world_size, segment_size):
            for offset in range(self._jump):
                rank_a = seg_start + offset
                rank_b = seg_start + offset + self._jump
                group = dist.new_group(ranks=[rank_a, rank_b], backend="gloo")
                if self._rank in (rank_a, rank_b):
                    my_group = group

        assert my_group is not None, f"Rank {self._rank} not found in any replication group"
        return my_group

    def close(self) -> None:
        pass  # Process groups are cleaned up by dist.destroy_process_group()


class GlooBackend(_GlooPeerBackend):
    """Gloo-based P2P replication using torch.distributed send/recv.

    Forms a dedicated Gloo process group between paired ranks.
    Rank pairing: rank N is paired with rank N + replication_jump (modulo world_size,
    within the same replication segment).
    """

    def __init__(
        self,
        replication_jump: int,
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        super().__init__(
            replication_jump,
            "GlooBackend",
            world_size=world_size,
            rank=rank,
        )
        logger.info(f"GlooBackend: rank {self._rank} paired with rank {self._peer_rank}")

    def send(self, tensors: list[torch.Tensor], step: int, meta: bytes = b"") -> None:
        _send_header(self._peer_rank, self._group, tensors, step, meta)
        for tensor in tensors:
            cpu_tensor = tensor.contiguous()
            dist.send(cpu_tensor, dst=self._peer_rank, group=self._group)

    def recv(self, buffers: list[torch.Tensor]) -> tuple[int, bytes]:
        step, meta = _recv_header(self._peer_rank, self._group, buffers)
        for buf in buffers:
            dist.recv(buf, src=self._peer_rank, group=self._group)
        return step, meta


class ChunkedGlooBackend(_GlooPeerBackend):
    """Chunk-based Gloo P2P replication that bounds training interference.

    Sends checkpoint data in fixed-size chunks with only one chunk in-flight
    at a time. This bounds the maximum head-of-line blocking delay on FSDP
    all-gather operations to: chunk_size / nic_bandwidth.

    The NIC round-robins between NCCL's training QP and Gloo's replication
    sends at chunk boundaries. Small chunks = fine-grained interleaving =
    AG sees minimal extra latency.

    Args:
        replication_jump: Spacing between paired ranks.
        chunk_size: Maximum bytes per send operation. Controls the max AG delay:
            max_ag_delay = chunk_size / nic_bw. Use estimate_chunk_size() to
            compute an appropriate value.
        world_size: Override world size (default: from dist).
        rank: Override rank (default: from dist).
    """

    def __init__(
        self,
        replication_jump: int,
        chunk_size: int = 16 * 1024 * 1024,  # 16 MiB default
        world_size: int | None = None,
        rank: int | None = None,
    ) -> None:
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before ChunkedGlooBackend")
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be > 0, got {chunk_size}")

        self._chunk_size = chunk_size
        super().__init__(
            replication_jump,
            "ChunkedGlooBackend",
            world_size=world_size,
            rank=rank,
        )
        logger.info(
            f"ChunkedGlooBackend: rank {self._rank} paired with rank {self._peer_rank}, "
            f"chunk_size={chunk_size / 1024:.0f} KiB"
        )

    def send(self, tensors: list[torch.Tensor], step: int, meta: bytes = b"") -> None:
        _send_header(self._peer_rank, self._group, tensors, step, meta)

        for tensor in tensors:
            flat = tensor.contiguous().view(-1)
            chunk_elems = max(1, self._chunk_size // flat.element_size())

            offset = 0
            while offset < flat.numel():
                end = min(offset + chunk_elems, flat.numel())
                chunk = flat[offset:end]
                dist.send(chunk, dst=self._peer_rank, group=self._group)
                offset = end

    def recv(self, buffers: list[torch.Tensor]) -> tuple[int, bytes]:
        step, meta = _recv_header(self._peer_rank, self._group, buffers)

        for buf in buffers:
            flat = buf.contiguous().view(-1)
            chunk_elems = max(1, self._chunk_size // flat.element_size())

            offset = 0
            while offset < flat.numel():
                end = min(offset + chunk_elems, flat.numel())
                chunk = flat[offset:end]
                dist.recv(chunk, src=self._peer_rank, group=self._group)
                offset = end

        return step, meta


class PeerReplicator:
    """Orchestrates P2P replication using a pluggable backend.

    Handles the background threading and no-op behavior when replication is disabled.
    """

    def __init__(
        self,
        backend: ReplicationBackend | None,
        enabled: bool = True,
    ) -> None:
        self._backend = backend
        self._enabled = enabled and backend is not None
        self._send_thread: threading.Thread | None = None
        self._recv_thread: threading.Thread | None = None
        self._send_error: BaseException | None = None
        self._recv_error: BaseException | None = None

    @property
    def enabled(self) -> bool:
        return self._enabled

    @property
    def peer_rank(self) -> int:
        """Global rank of the paired peer, or -1 if replication is disabled."""
        if self._backend is None:
            return -1
        return self._backend.peer_rank

    def start_replication(
        self,
        send_tensors: list[torch.Tensor],
        send_step: int,
        recv_buffers: list[torch.Tensor],
        send_meta: bytes = b"",
    ) -> None:
        """Start non-blocking P2P exchange: send own shard, receive peer's.

        send_meta is the serialized non-tensor metadata for the shard being sent;
        the peer's is captured and returned from wait(). Both run in bg threads.
        """
        if not self._enabled:
            return

        self.wait()  # ensure previous replication completed

        self._send_error = None
        self._recv_error = None

        def _send():
            try:
                self._backend.send(send_tensors, send_step, send_meta)
            except BaseException as e:
                self._send_error = e

        def _recv():
            try:
                step, meta = self._backend.recv(recv_buffers)
                self._recv_step = step
                self._recv_meta = meta
            except BaseException as e:
                self._recv_error = e

        self._recv_step = -1
        self._recv_meta = b""
        self._send_thread = threading.Thread(target=_send, daemon=True)
        self._recv_thread = threading.Thread(target=_recv, daemon=True)
        self._send_thread.start()
        self._recv_thread.start()

    def wait(self, timeout_s: float | None = None) -> tuple[int, bytes]:
        """Wait for in-flight replication. Returns (peer step, peer meta blob).

        ``timeout_s`` bounds the wait: with it set (the SIGTERM-flush path), a peer that
        stalled or died mid-transfer — the common case when a coordinated stop SIGTERMs
        every rank at once, so a rank blocks on ``dist.recv`` from a peer that is already
        gone — must not stall the flush until the launcher SIGKILLs the worker (losing the
        checkpoint entirely). We abandon this cycle's replica and proceed; the send/recv
        threads are daemons, so the dangling one dies with the process. Unbounded (default)
        for the normal training path, where replication legitimately spans steps.
        """
        if self._send_thread is not None:
            self._send_thread.join(timeout_s)
        if self._recv_thread is not None:
            self._recv_thread.join(timeout_s)

        timed_out = self.in_flight  # a bounded join left a thread blocked on a dead peer
        self._send_thread = None
        self._recv_thread = None

        # A replication failure/timeout is a *degraded replica*, not a fatal error: the
        # peer vanished so this cycle's replica is dropped. It must NOT crash the training
        # step or the SIGTERM flush — the last complete replicated step still stands, and
        # recovery MIN-reduces to it. Log, clear (so we don't re-raise on the next wait),
        # and report no peer data this cycle; every caller treats step <= 0 as "no replica".
        err = self._send_error or self._recv_error
        if timed_out or err is not None:
            reason = "did not finish within the wait window" if timed_out else f"failed ({err})"
            logger.warning(f"replication {reason}; dropping this cycle's replica")
            self._send_error = self._recv_error = None
            return -1, b""

        return getattr(self, "_recv_step", -1), getattr(self, "_recv_meta", b"")

    @property
    def in_flight(self) -> bool:
        if self._send_thread is not None and self._send_thread.is_alive():
            return True
        if self._recv_thread is not None and self._recv_thread.is_alive():
            return True
        return False

    def close(self) -> None:
        self.wait()
        if self._backend is not None:
            self._backend.close()
