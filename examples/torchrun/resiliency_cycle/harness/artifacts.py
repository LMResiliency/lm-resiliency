"""File and process helpers shared by torchrun fault injection."""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} does not contain a JSON object")
    return value


def atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def wait_for_paths(
    paths: Sequence[Path],
    *,
    processes: Sequence[subprocess.Popen[bytes]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"torchrun agent failed with exit codes {failed}")
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"timed out waiting for artifacts: {missing}")
        time.sleep(0.1)
