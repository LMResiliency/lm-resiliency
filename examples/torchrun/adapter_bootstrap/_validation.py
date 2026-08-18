"""Validation-only assertions for the native torchrun integration."""

from __future__ import annotations

import os
import threading
from types import TracebackType


def assert_torchrun_adapter_attached() -> None:
    """Fail when automatic torchrun worker instrumentation did not attach."""
    if os.environ.get("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED") != "1":
        raise RuntimeError("the torchrun worker adapter did not attach")


class ObserveTorchrunAdapterAttachment:
    """Latch the transient attachment marker across framework-owned teardown."""

    def __init__(self) -> None:
        self._attached = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ObserveTorchrunAdapterAttachment:
        self._observe()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()
        return self

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del error, traceback
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=1)
        self._observe()
        if error_type is None and not self._attached:
            raise RuntimeError("the torchrun worker adapter did not attach")

    def _poll(self) -> None:
        while not self._stop.wait(0.01):
            self._observe()

    def _observe(self) -> None:
        if os.environ.get("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED") == "1":
            self._attached = True
