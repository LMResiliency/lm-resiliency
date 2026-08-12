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

import logging
from abc import ABC, abstractmethod
from typing import Any, Protocol

import torch

logger = logging.getLogger(__name__)


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
    ) -> None:
        self._control = metadata_store
        self._chunk_size = chunk_size
        self._served: dict[str, list[torch.Tensor]] = {}
        try:
            from nixl._api import nixl_agent, nixl_agent_config  # type: ignore

            self._agent = nixl_agent(agent_name, nixl_agent_config())
            self._reg: dict[str, object] = {}
        except Exception as e:  # pragma: no cover - depends on cluster
            raise RuntimeError(f"NIXL unavailable: {e}") from e

    def serve(self, tensors: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        reg = self._agent.register_memory(tensors)
        self._reg[key] = reg
        self._served[key] = tensors
        self._control.put_transfer_meta(
            key,
            {
                "agent_meta": self._agent.get_agent_metadata(),
                "descs": self._agent.get_serialized_descs(reg),
            },
        )
        logger.info(f"NIXL serve: published {len(tensors)} tensors under {key}")

    def fetch(self, buffers: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        meta = self._control.get_transfer_meta(key, wait=True)
        peer_agent = self._agent.add_remote_agent(meta["agent_meta"])
        remote_descs = self._agent.deserialize_descs(meta["descs"])
        local_reg = self._agent.register_memory(buffers)
        local_descs = self._agent.get_xfer_descs(local_reg)
        xfer = self._agent.initialize_xfer(
            "READ", local_descs, remote_descs, peer_agent, key.encode()
        )
        self._agent.transfer(xfer)
        while self._agent.check_xfer_state(xfer) != "DONE":
            pass
        logger.info(f"NIXL fetch: pulled {len(buffers)} tensors for {key}")

    def close(self) -> None:
        try:
            for key, reg in getattr(self, "_reg", {}).items():
                self._agent.deregister_memory(reg)
        except Exception:  # pragma: no cover
            pass


class TorchDistTransfer(CheckpointTransfer):
    """Portable fallback: chunked Gloo send/recv over a dedicated pair group.

    Both endpoints must call collectively. The caller pairs the replacement rank
    with the replica holder and invokes ``serve`` and ``fetch`` at the same recovery
    step. Reuses the same chunking discipline as GEMINI's
    ChunkedGlooBackend so a partial NIC is not overwhelmed.
    """

    def __init__(self, rank: int, chunk_size: int = 64 * 1024 * 1024) -> None:
        import torch.distributed as dist

        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized for TorchDistTransfer")
        self._dist = dist
        self._rank = rank
        self._chunk_size = chunk_size

    def serve(self, tensors: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        for t in tensors:
            flat = t.contiguous().view(-1)
            elems = max(1, self._chunk_size // flat.element_size())
            for off in range(0, flat.numel(), elems):
                self._dist.send(flat[off : off + elems], dst=peer_rank)

    def fetch(self, buffers: list[torch.Tensor], key: str, peer_rank: int, peer_host: str) -> None:
        for b in buffers:
            flat = b.contiguous().view(-1)
            elems = max(1, self._chunk_size // flat.element_size())
            for off in range(0, flat.numel(), elems):
                self._dist.recv(flat[off : off + elems], src=peer_rank)

    def close(self) -> None:
        pass


def make_transfer(
    backend: str,
    *,
    rank: int,
    agent_name: str | None = None,
    metadata_store: TransferMetadataStore | None = None,
    control: TransferMetadataStore | None = None,
    chunk_size: int = 64 * 1024 * 1024,
) -> CheckpointTransfer:
    """Construct the requested transfer backend, falling back to torch_dist.

    "nixl" is preferred for production (RDMA/GPUDirect, one-sided). If the nixl
    package is unavailable we log and fall back to the always-present Gloo path.
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
            return NixlCheckpointTransfer(agent_name, metadata_store, chunk_size)
        except Exception as e:
            logger.warning(f"NIXL backend unavailable ({e}); falling back to torch_dist")
    return TorchDistTransfer(rank, chunk_size)
