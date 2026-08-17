"""Framework-neutral controls for user-owned production-loop examples."""

from __future__ import annotations

import argparse
import os


def _positive_steps(value: str) -> int:
    steps = int(value)
    if steps < 1:
        raise argparse.ArgumentTypeError("steps must be positive")
    return steps


def add_run_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    """Add training arguments shared by the framework-native examples."""
    parser.add_argument("--steps", type=_positive_steps, default=10)


def require_resiliency_adapter() -> None:
    """Fail when a production-loop example was not activated by torchrun."""
    if os.environ.get("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED") != "1":
        raise RuntimeError("the torchrun worker adapter did not attach")
