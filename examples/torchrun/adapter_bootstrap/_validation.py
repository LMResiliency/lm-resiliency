"""Validation-only assertions for the native torchrun integration."""

from __future__ import annotations

import os


def assert_torchrun_adapter_attached() -> None:
    """Fail when automatic torchrun worker instrumentation did not attach."""
    if os.environ.get("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED") != "1":
        raise RuntimeError("the torchrun worker adapter did not attach")
