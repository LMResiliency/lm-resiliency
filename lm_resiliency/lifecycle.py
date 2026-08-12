"""Process-level cleanup for enabled resiliency integrations."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections import OrderedDict
from typing import Protocol, TypeVar

logger = logging.getLogger(__name__)


class _Closable(Protocol):
    def close(self) -> None: ...


_Handle = TypeVar("_Handle", bound=_Closable)
_handles: OrderedDict[int, _Closable] = OrderedDict()
_lock = threading.RLock()


def register_automatic_cleanup(handle: _Handle) -> _Handle:
    """Keep a handle alive and close it automatically at normal process exit."""
    if not callable(getattr(handle, "close", None)):
        raise TypeError("automatic cleanup requires a handle with close()")

    with _lock:
        key = id(handle)
        if key in _handles:
            return handle
        _handles[key] = handle

        # Keep this callback newer than component-specific exit handlers so it
        # runs first under atexit's last-in, first-out ordering.
        atexit.unregister(_close_registered_handles)
        atexit.register(_close_registered_handles)
    return handle


def unregister_automatic_cleanup(handle: _Closable) -> None:
    """Remove a handle that has already been closed explicitly."""
    with _lock:
        _handles.pop(id(handle), None)
        if not _handles:
            atexit.unregister(_close_registered_handles)


def _close_registered_handles() -> None:
    """Close live handles in reverse registration order."""
    with _lock:
        handles = list(reversed(_handles.values()))
        _handles.clear()

    for handle in handles:
        try:
            handle.close()
        except Exception:
            logger.exception(
                "Automatic resiliency cleanup failed for %s",
                type(handle).__name__,
            )

    with _lock:
        if not _handles:
            atexit.unregister(_close_registered_handles)


def _discard_inherited_handles() -> None:
    """Do not close parent-owned training resources in a forked child."""
    global _lock
    _handles.clear()
    _lock = threading.RLock()
    atexit.unregister(_close_registered_handles)


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_discard_inherited_handles)
