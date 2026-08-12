"""Asynchronous GPU→CPU tensor copy using a dedicated CUDA stream.

Timeline:
    Compute stream:  [backward] [optimizer.step] [next forward] [next backward] ...
    Copy stream:                [D2H copies ────────────────────────────────────]

The checkpoint manager waits for the copy event in a background completion
worker and starts peer replication as soon as the host buffers are complete.
"""

from __future__ import annotations

import torch


class AsyncDeviceCopier:
    """Manages non-blocking GPU→CPU copies on a dedicated CUDA stream.

    Per-tensor copies run on a separate stream so they don't block the default
    compute stream. A manager-owned worker waits for the completion event without
    blocking the training thread.
    """

    def __init__(self) -> None:
        self._stream: torch.cuda.Stream | None = None
        self._event: torch.cuda.Event | None = None
        self._in_flight = False

    @property
    def in_flight(self) -> bool:
        return self._in_flight

    def _get_stream(self) -> torch.cuda.Stream:
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        return self._stream

    def start_copy(self, src_tensors: list[torch.Tensor], dst_buffers: list[torch.Tensor]) -> None:
        """Start non-blocking copy from GPU tensors to pinned CPU buffers.

        The copy runs on a dedicated stream and does not block the compute stream.
        """
        if self._in_flight:
            raise RuntimeError("Previous copy still in flight. Call wait() first.")

        if len(src_tensors) != len(dst_buffers):
            raise ValueError(
                f"src ({len(src_tensors)}) and dst ({len(dst_buffers)}) length mismatch"
            )

        if torch.cuda.is_available() and any(t.is_cuda for t in src_tensors):
            stream = self._get_stream()

            # Wait for the current compute stream to finish writing the tensors
            # (ensures optimizer.step() writes are visible to the copy stream)
            stream.wait_stream(torch.cuda.current_stream())

            with torch.cuda.stream(stream):
                for src, dst in zip(src_tensors, dst_buffers):
                    dst.copy_(src, non_blocking=True)

            # Record event on copy stream for later synchronization
            self._event = torch.cuda.Event()
            self._event.record(stream)
        else:
            # CPU fallback
            for src, dst in zip(src_tensors, dst_buffers):
                dst.copy_(src)
            self._event = None

        self._in_flight = True

    def wait(self) -> None:
        """Block until the in-flight copy is complete."""
        if not self._in_flight:
            return
        if self._event is not None:
            self._event.synchronize()
            self._event = None
        self._in_flight = False

    def is_complete(self) -> bool:
        """Check if the copy has completed without blocking."""
        if not self._in_flight:
            return True
        if self._event is not None:
            return self._event.query()
        return True
