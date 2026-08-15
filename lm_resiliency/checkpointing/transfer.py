"""GEMINI point-to-point checkpoint transport.

Moving checkpoint bytes between machines is part of GEMINI's design. This module
sits alongside the steady-state replicator and provides a one-shot,
endpoint-addressed transfer of a shard from the node holding its replica to a
replacement endpoint. The caller supplies and coordinates the endpoints; GEMINI
owns only the transport.

Backends
--------
* ``NixlCheckpointTransfer`` — production path. Uses NIXL (NVIDIA Inference Xfer
  Library) for one-sided RDMA/GPUDirect reads: the peer registers the shard
  buffers and publishes descriptors plus agent metadata through a caller-provided
  metadata store; the destination issues a READ transfer. One-sided, so
  the peer does not have to synchronously participate in a collective.
* ``TorchDistTransfer`` — portable fallback using Gloo send/recv over a dedicated
  pair process group. Requires both endpoints to participate, coordinated by the
  caller. Always available wherever torch.distributed is.

Both operate on flat tensor lists matching the shard's saved layout, so the
result plugs straight into GEMINI's unflatten/load path.
"""

from __future__ import annotations

import hashlib
import json
import logging
import socket
import threading
import time
from abc import ABC, abstractmethod
from datetime import timedelta
from typing import Any, Protocol

import torch

from lm_resiliency.checkpointing.disk import shard_checksums

logger = logging.getLogger(__name__)

_TRANSFER_PROTOCOL_VERSION = 1
_TERMINAL_FAILURE_STATES = {"FAILED", "ERROR", "CANCELLED", "CANCELED", "ABORTED"}


def _tensor_manifest(tensors: list[torch.Tensor]) -> list[dict[str, object]]:
    return [
        {
            "shape": list(tensor.shape),
            "dtype": str(tensor.dtype),
            "numel": tensor.numel(),
            "bytes": tensor.numel() * tensor.element_size(),
        }
        for tensor in tensors
    ]


def _payload_manifest(
    tensors: list[torch.Tensor],
    *,
    key: str,
    source_rank: int,
    destination_rank: int,
    destination_host: str,
) -> dict[str, object]:
    cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors]
    return {
        "protocol_version": _TRANSFER_PROTOCOL_VERSION,
        "key": key,
        "source_rank": source_rank,
        "destination_rank": destination_rank,
        "source_host": socket.gethostname(),
        "destination_host": destination_host,
        "tensors": _tensor_manifest(cpu_tensors),
        "checksums": shard_checksums(cpu_tensors),
    }


def _validate_manifest(
    manifest: object,
    buffers: list[torch.Tensor],
    *,
    key: str,
    source_rank: int,
    destination_rank: int,
    peer_host: str,
) -> None:
    if not isinstance(manifest, dict):
        raise RuntimeError("checkpoint transfer manifest must be a mapping")
    expected_fields = {
        "protocol_version",
        "key",
        "source_rank",
        "destination_rank",
        "source_host",
        "destination_host",
        "tensors",
        "checksums",
    }
    if set(manifest) != expected_fields:
        raise RuntimeError("checkpoint transfer manifest has unexpected or missing fields")
    if manifest["protocol_version"] != _TRANSFER_PROTOCOL_VERSION:
        raise RuntimeError("checkpoint transfer protocol version mismatch")
    if manifest["key"] != key:
        raise RuntimeError("checkpoint transfer key mismatch")
    if manifest["source_rank"] != source_rank or manifest["destination_rank"] != destination_rank:
        raise RuntimeError("checkpoint transfer endpoint mismatch")
    if peer_host and manifest["source_host"] != peer_host:
        raise RuntimeError(
            f"checkpoint transfer source host mismatch: expected {peer_host!r}, "
            f"got {manifest['source_host']!r}"
        )
    local_host = socket.gethostname()
    if manifest["destination_host"] and manifest["destination_host"] != local_host:
        raise RuntimeError(
            f"checkpoint transfer destination host mismatch: expected {local_host!r}, "
            f"got {manifest['destination_host']!r}"
        )
    if manifest["tensors"] != _tensor_manifest(buffers):
        raise RuntimeError("checkpoint transfer tensor shape, dtype, or byte-count mismatch")
    checksums = manifest["checksums"]
    if not isinstance(checksums, list) or any(type(value) is not int for value in checksums):
        raise RuntimeError("checkpoint transfer checksum manifest is invalid")


def _transfer_tag(key: str) -> int:
    digest = hashlib.blake2s(key.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, "big") % ((1 << 30) - 8)


def _validate_request(key: str, *, rank: int, peer_rank: int, peer_host: str) -> None:
    if not key:
        raise ValueError("checkpoint transfer key must be non-empty")
    if rank == peer_rank:
        raise ValueError("checkpoint transfer peer must differ from the local rank")
    if not peer_host:
        raise ValueError("checkpoint transfer peer_host must be non-empty")


class TransferMetadataStore(Protocol):
    """Rendezvous store required by one-sided checkpoint transfer backends."""

    def put_transfer_meta(self, key: str, value: dict[str, Any]) -> None: ...

    def get_transfer_meta(
        self,
        key: str,
        *,
        wait: bool = False,
        timeout_s: float | None = None,
    ) -> dict[str, Any]: ...


class CheckpointTransfer(ABC):
    """Move a rank's shard from its replica holder to a replacement endpoint.

    ``key`` names the shard (symmetric across both endpoints, used for NIXL
    metadata rendezvous). ``peer_rank``/``peer_host`` identify the transport
    endpoint: torch_dist uses the rank for send/recv, NIXL uses the host to
    locate the remote agent.
    """

    @abstractmethod
    def serve(self, tensors: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        """Source side: make a shard available to pull under ``key``."""

    @abstractmethod
    def fetch(self, buffers: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        """Destination side: pull the shard published under ``key`` into ``buffers``."""

    @abstractmethod
    def close(self) -> None: ...


class NixlCheckpointTransfer(CheckpointTransfer):
    """One-sided RDMA/GPUDirect shard transfer via NIXL.

    Workflow (replica holder = server, replacement endpoint = initiator):
      peer.serve():
        - register the shard buffers with the NIXL agent
        - publish {agent_metadata, xfer_descriptors} to the control channel
          under ``key``
      destination.fetch():
        - read the peer's published metadata from the control channel
        - add_remote_agent(peer_metadata); register local receive buffers
        - initialize_xfer("READ", local_descs, remote_descs, peer_agent)
        - transfer + poll until DONE

    NIXL API names track the ``nixl`` python bindings; validate against the
    version installed on the target cluster (the binding surface has evolved).
    Control-channel plumbing is injected so this class stays transport-agnostic.
    """

    def __init__(
        self,
        agent_name: str,
        metadata_store: TransferMetadataStore,
        chunk_size: int = 64 * 1024 * 1024,
        *,
        rank: int = -1,
        timeout_s: float = 120.0,
        poll_interval_s: float = 0.001,
    ) -> None:
        if timeout_s <= 0:
            raise ValueError("checkpoint transfer timeout must be positive")
        if poll_interval_s <= 0:
            raise ValueError("NIXL poll interval must be positive")
        if rank < 0:
            import torch.distributed as dist

            if not dist.is_initialized():
                raise ValueError("NIXL checkpoint transfer requires the local rank")
            rank = dist.get_rank()
        self._control = metadata_store
        self._chunk_size = chunk_size
        self._rank = rank
        self._timeout_s = timeout_s
        self._poll_interval_s = poll_interval_s
        self._served: dict[str, list[torch.Tensor]] = {}
        try:
            from nixl._api import nixl_agent, nixl_agent_config  # type: ignore

            self._agent = nixl_agent(agent_name, nixl_agent_config())
            self._reg: dict[str, object] = {}
        except Exception as e:  # pragma: no cover - depends on cluster
            raise RuntimeError(f"NIXL unavailable: {e}") from e

    def serve(self, tensors: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        _validate_request(key, rank=self._rank, peer_rank=peer_rank, peer_host=peer_host)
        old_reg = self._reg.pop(key, None)
        if old_reg is not None:
            self._agent.deregister_memory(old_reg)
        reg = self._agent.register_memory(tensors)
        self._reg[key] = reg
        self._served[key] = tensors
        self._control.put_transfer_meta(
            key,
            {
                "transport": "nixl",
                "manifest": _payload_manifest(
                    tensors,
                    key=key,
                    source_rank=self._rank,
                    destination_rank=peer_rank,
                    destination_host=peer_host,
                ),
                "agent_meta": self._agent.get_agent_metadata(),
                "descs": self._agent.get_serialized_descs(reg),
            },
        )
        logger.info(f"NIXL serve: published {len(tensors)} tensors under {key}")

    def fetch(self, buffers: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        _validate_request(key, rank=self._rank, peer_rank=peer_rank, peer_host=peer_host)
        meta = self._control.get_transfer_meta(key, wait=True, timeout_s=self._timeout_s)
        if meta.get("transport") != "nixl":
            raise RuntimeError("checkpoint transfer metadata backend mismatch")
        manifest = meta.get("manifest")
        _validate_manifest(
            manifest,
            buffers,
            key=key,
            source_rank=peer_rank,
            destination_rank=self._rank,
            peer_host=peer_host,
        )
        peer_agent = self._agent.add_remote_agent(meta["agent_meta"])
        remote_descs = self._agent.deserialize_descs(meta["descs"])
        local_reg = self._agent.register_memory(buffers)
        try:
            local_descs = self._agent.get_xfer_descs(local_reg)
            xfer = self._agent.initialize_xfer(
                "READ", local_descs, remote_descs, peer_agent, key.encode()
            )
            self._agent.transfer(xfer)
            deadline = time.monotonic() + self._timeout_s
            while True:
                raw_state = self._agent.check_xfer_state(xfer)
                state = str(raw_state).upper().rsplit(".", 1)[-1]
                if state == "DONE":
                    break
                if state in _TERMINAL_FAILURE_STATES:
                    raise RuntimeError(f"NIXL checkpoint transfer {key!r} failed: {raw_state}")
                if time.monotonic() >= deadline:
                    cancel = getattr(self._agent, "cancel_xfer", None)
                    if callable(cancel):
                        cancel(xfer)
                    raise TimeoutError(
                        f"NIXL checkpoint transfer {key!r} exceeded {self._timeout_s:.1f}s"
                    )
                time.sleep(self._poll_interval_s)
            expected = manifest["checksums"]
            actual = shard_checksums([buffer.detach().cpu().contiguous() for buffer in buffers])
            if actual != expected:
                raise RuntimeError(f"NIXL checkpoint transfer {key!r} failed checksum")
        finally:
            self._agent.deregister_memory(local_reg)
            remove_remote = getattr(self._agent, "remove_remote_agent", None)
            if callable(remove_remote):
                remove_remote(peer_agent)
        logger.info(f"NIXL fetch: pulled {len(buffers)} tensors for {key}")

    def close(self) -> None:
        try:
            for key, reg in getattr(self, "_reg", {}).items():
                self._agent.deregister_memory(reg)
            self._reg.clear()
            self._served.clear()
        except Exception:  # pragma: no cover
            pass


class TorchDistTransfer(CheckpointTransfer):
    """Portable fallback: chunked Gloo send/recv over a dedicated pair group.

    Both endpoints must call collectively. The caller pairs the replacement rank
    with the replica holder and invokes ``serve`` and ``fetch`` at the same recovery
    step. Reuses the same chunking discipline as GEMINI's
    ChunkedGlooBackend so a partial NIC is not overwhelmed.
    """

    def __init__(
        self,
        rank: int,
        chunk_size: int = 64 * 1024 * 1024,
        *,
        timeout_s: float = 120.0,
        process_group: object | None = None,
    ) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for TorchDistTransfer")
        if chunk_size <= 0:
            raise ValueError("checkpoint transfer chunk_size must be positive")
        if timeout_s <= 0:
            raise ValueError("checkpoint transfer timeout must be positive")
        self._dist = dist
        actual_rank = dist.get_rank()
        if rank != actual_rank:
            raise ValueError(
                f"TorchDistTransfer rank {rank} does not match distributed rank {actual_rank}"
            )
        self._rank = rank
        self._chunk_size = chunk_size
        self._timeout_s = timeout_s
        self._wait_timeout = timedelta(seconds=timeout_s)
        self._provided_group = process_group
        self._groups: dict[int, object] = {}
        self._group_lock = threading.Lock()
        if process_group is not None and str(dist.get_backend(process_group)).lower() != "gloo":
            raise ValueError("TorchDistTransfer requires a dedicated Gloo process group")

    def _group_for_peer(self, peer_rank: int) -> object:
        if self._provided_group is not None:
            return self._provided_group
        with self._group_lock:
            group = self._groups.get(peer_rank)
            if group is None:
                group = self._dist.new_group(
                    ranks=sorted((self._rank, peer_rank)),
                    backend="gloo",
                    timeout=self._wait_timeout,
                    use_local_synchronization=True,
                )
                self._groups[peer_rank] = group
            return group

    def _wait(self, work: object, *, operation: str) -> None:
        try:
            completed = work.wait(timeout=self._wait_timeout)
        except Exception as error:
            raise TimeoutError(
                f"checkpoint transfer {operation} exceeded {self._timeout_s:.1f}s"
            ) from error
        if completed is False:
            raise TimeoutError(f"checkpoint transfer {operation} exceeded {self._timeout_s:.1f}s")

    def _send(
        self,
        tensor: torch.Tensor,
        *,
        peer_rank: int,
        group: object,
        tag: int,
        operation: str,
    ) -> None:
        work = self._dist.isend(tensor, dst=peer_rank, group=group, tag=tag)
        self._wait(work, operation=operation)

    def _recv(
        self,
        tensor: torch.Tensor,
        *,
        peer_rank: int,
        group: object,
        tag: int,
        operation: str,
    ) -> None:
        work = self._dist.irecv(tensor, src=peer_rank, group=group, tag=tag)
        self._wait(work, operation=operation)

    def serve(self, tensors: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        _validate_request(key, rank=self._rank, peer_rank=peer_rank, peer_host=peer_host)
        cpu_tensors = [tensor.detach().cpu().contiguous() for tensor in tensors]
        manifest = _payload_manifest(
            cpu_tensors,
            key=key,
            source_rank=self._rank,
            destination_rank=peer_rank,
            destination_host=peer_host,
        )
        encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
        tag = _transfer_tag(key)
        group = self._group_for_peer(peer_rank)
        self._send(
            torch.tensor([len(encoded)], dtype=torch.int64),
            peer_rank=peer_rank,
            group=group,
            tag=tag,
            operation=f"{key!r} manifest length send",
        )
        self._send(
            torch.tensor(list(encoded), dtype=torch.uint8),
            peer_rank=peer_rank,
            group=group,
            tag=tag + 1,
            operation=f"{key!r} manifest send",
        )
        accepted = torch.zeros(1, dtype=torch.int32)
        self._recv(
            accepted,
            peer_rank=peer_rank,
            group=group,
            tag=tag + 2,
            operation=f"{key!r} preflight acknowledgement",
        )
        if accepted.item() != 1:
            raise RuntimeError(f"checkpoint transfer {key!r} was rejected by the destination")
        for tensor in cpu_tensors:
            flat = tensor.view(-1)
            elems = max(1, self._chunk_size // flat.element_size())
            for off in range(0, flat.numel(), elems):
                self._send(
                    flat[off : off + elems],
                    peer_rank=peer_rank,
                    group=group,
                    tag=tag + 3,
                    operation=f"{key!r} payload send",
                )
        verified = torch.zeros(1, dtype=torch.int32)
        self._recv(
            verified,
            peer_rank=peer_rank,
            group=group,
            tag=tag + 4,
            operation=f"{key!r} checksum acknowledgement",
        )
        if verified.item() != 1:
            raise RuntimeError(f"checkpoint transfer {key!r} failed destination checksum")

    def fetch(self, buffers: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        _validate_request(key, rank=self._rank, peer_rank=peer_rank, peer_host=peer_host)
        tag = _transfer_tag(key)
        group = self._group_for_peer(peer_rank)
        encoded_size = torch.zeros(1, dtype=torch.int64)
        self._recv(
            encoded_size,
            peer_rank=peer_rank,
            group=group,
            tag=tag,
            operation=f"{key!r} manifest length receive",
        )
        size = int(encoded_size.item())
        if size <= 0 or size > 64 * 1024 * 1024:
            raise RuntimeError(f"checkpoint transfer {key!r} manifest size is invalid")
        encoded = torch.empty(size, dtype=torch.uint8)
        self._recv(
            encoded,
            peer_rank=peer_rank,
            group=group,
            tag=tag + 1,
            operation=f"{key!r} manifest receive",
        )
        try:
            manifest = json.loads(bytes(encoded.tolist()))
            _validate_manifest(
                manifest,
                buffers,
                key=key,
                source_rank=peer_rank,
                destination_rank=self._rank,
                peer_host=peer_host,
            )
        except Exception:
            self._send(
                torch.zeros(1, dtype=torch.int32),
                peer_rank=peer_rank,
                group=group,
                tag=tag + 2,
                operation=f"{key!r} preflight rejection",
            )
            raise
        self._send(
            torch.ones(1, dtype=torch.int32),
            peer_rank=peer_rank,
            group=group,
            tag=tag + 2,
            operation=f"{key!r} preflight acknowledgement",
        )
        cpu_buffers = [
            torch.empty(buffer.shape, dtype=buffer.dtype, device="cpu") for buffer in buffers
        ]
        for buffer in cpu_buffers:
            flat = buffer.view(-1)
            elems = max(1, self._chunk_size // flat.element_size())
            for off in range(0, flat.numel(), elems):
                self._recv(
                    flat[off : off + elems],
                    peer_rank=peer_rank,
                    group=group,
                    tag=tag + 3,
                    operation=f"{key!r} payload receive",
                )
        expected = manifest["checksums"]
        actual = shard_checksums(cpu_buffers)
        verified = actual == expected
        self._send(
            torch.tensor([int(verified)], dtype=torch.int32),
            peer_rank=peer_rank,
            group=group,
            tag=tag + 4,
            operation=f"{key!r} checksum acknowledgement",
        )
        if not verified:
            raise RuntimeError(f"checkpoint transfer {key!r} failed checksum")
        for destination, source in zip(buffers, cpu_buffers):
            destination.copy_(source.to(destination.device))

    def close(self) -> None:
        for group in self._groups.values():
            self._dist.destroy_process_group(group)
        self._groups.clear()


def make_transfer(
    backend: str,
    *,
    rank: int,
    agent_name: str | None = None,
    metadata_store: TransferMetadataStore | None = None,
    control: TransferMetadataStore | None = None,
    chunk_size: int = 64 * 1024 * 1024,
    timeout_s: float = 120.0,
    process_group: object | None = None,
    allow_backend_fallback: bool = False,
) -> CheckpointTransfer:
    """Construct the requested transfer backend.

    ``nixl`` is preferred for production (RDMA/GPUDirect, one-sided). Backend
    fallback is disabled by default because torch_dist requires synchronous
    participation from both endpoints. Set ``allow_backend_fallback=True`` only
    when the caller has arranged that two-sided protocol explicitly.
    ``control`` is a backward-compatible alias for ``metadata_store``.
    """
    if backend not in {"nixl", "torch_dist"}:
        raise ValueError(f"unsupported checkpoint transfer backend: {backend}")
    if metadata_store is not None and control is not None and metadata_store is not control:
        raise ValueError("pass only one of metadata_store or control")
    metadata_store = metadata_store if metadata_store is not None else control
    agent_name = agent_name or f"gemini-transfer-{rank}"

    if backend == "nixl":
        if metadata_store is None:
            raise ValueError("the nixl backend requires a metadata_store")
        try:
            return NixlCheckpointTransfer(
                agent_name,
                metadata_store,
                chunk_size,
                rank=rank,
                timeout_s=timeout_s,
            )
        except Exception as e:
            if not allow_backend_fallback:
                raise RuntimeError(
                    "NIXL backend unavailable and two-sided fallback was not enabled"
                ) from e
            logger.warning(
                "NIXL backend unavailable (%s); using explicitly enabled two-sided "
                "torch_dist fallback",
                e,
            )
    return TorchDistTransfer(
        rank,
        chunk_size,
        timeout_s=timeout_s,
        process_group=process_group,
    )
