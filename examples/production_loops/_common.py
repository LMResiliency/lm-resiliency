"""Framework-neutral controls for user-owned production-loop examples."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class RunArguments:
    """Arguments shared by every framework-native production-loop example."""

    validation_output_dir: Path
    steps: int


_CHECKPOINT_STEP_ENV = "LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP"


def _positive_steps(value: str) -> int:
    steps = int(value)
    if steps < 1:
        raise argparse.ArgumentTypeError("steps must be positive")
    return steps


def torchrun_checkpoint_step() -> int:
    """Return the manager-selected resume position published by bootstrap."""
    value = os.environ.get(_CHECKPOINT_STEP_ENV, "0")
    if not value.isdecimal():
        raise RuntimeError(f"{_CHECKPOINT_STEP_ENV} must be a non-negative integer")
    return int(value)


def training_step_range(current_step: int, target_step: int) -> range:
    """Return remaining deterministic training steps for one validation run."""
    for value, name in ((current_step, "current_step"), (target_step, "target_step")):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
    if target_step <= current_step:
        raise ValueError("target_step must exceed the recovered current_step")
    return range(current_step, target_step)


def parse_run_arguments(arguments: Sequence[str] | None = None) -> RunArguments:
    """Parse shared arguments and prepare the validation output directory."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-output-dir", type=Path, required=True)
    parser.add_argument("--steps", type=_positive_steps, default=10)
    parsed = parser.parse_args(arguments)
    parsed.validation_output_dir.mkdir(parents=True, exist_ok=True)
    return RunArguments(
        validation_output_dir=parsed.validation_output_dir,
        steps=parsed.steps,
    )


def write_validation_summary(
    output_dir: Path,
    summary: Mapping[str, Any],
    *,
    writer: bool,
) -> None:
    """Write and print a framework validation summary from one worker."""
    if not writer:
        return
    framework = summary.get("framework")
    if not isinstance(framework, str) or not framework:
        raise ValueError("validation summary requires a non-empty framework")
    payload = dict(summary)
    (output_dir / f"{framework}-production-loop.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, sort_keys=True), flush=True)
