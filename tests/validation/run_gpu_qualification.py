"""Run the smallest trusted GPU qualification tier and publish JSON evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

_SCHEMA_VERSION = 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _nvidia_smi() -> dict[str, Any]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,uuid,driver_version,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as error:
        return {"command": command, "error": f"{type(error).__name__}: {error}"}
    return {
        "command": command,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _environment() -> dict[str, Any]:
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0
    devices = []
    for index in range(device_count):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    nccl_version = torch.cuda.nccl.version() if cuda_available else None
    return {
        "captured_at": _timestamp(),
        "host": platform.node(),
        "platform": platform.platform(),
        "python": sys.version,
        "python_executable": sys.executable,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
        "device_count": device_count,
        "devices": devices,
        "nvidia_smi": _nvidia_smi(),
        "runner": {
            "name": os.environ.get("RUNNER_NAME"),
            "os": os.environ.get("RUNNER_OS"),
            "arch": os.environ.get("RUNNER_ARCH"),
        },
    }


def _commands(artifact_dir: Path) -> list[tuple[str, list[str]]]:
    python = sys.executable
    return [
        (
            "recovery-equivalence",
            [python, "tests/integration/core/test_recovery_equivalence.py"],
        ),
        (
            "replay-and-collectives",
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "tests/integration/core/test_replay_harness.py",
                "no-sdc",
                "dropout-rng",
                "structured",
            ],
        ),
        (
            "fsdp2-checkpoint-recovery",
            [
                python,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc-per-node=2",
                "tests/integration/core/test_gemini_dtensor.py",
            ],
        ),
        (
            "automatic-exit-cleanup",
            [
                python,
                "tests/integration/frameworks/test_exit_cleanup.py",
                "--case",
                "pytorch",
                "--artifact-dir",
                str(artifact_dir / "exit-cleanup"),
            ],
        ),
    ]


def _run_command(
    name: str,
    command: list[str],
    artifact_dir: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    log_path = artifact_dir / f"{name}.log"
    started_at = _timestamp()
    started = time.monotonic()
    print(f"::group::{name}: {shlex.join(command)}", flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=_REPOSITORY_ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
        returncode = process.wait()
    print("::endgroup::", flush=True)
    return {
        "name": name,
        "command": command,
        "started_at": started_at,
        "completed_at": _timestamp(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "returncode": returncode,
        "status": "passed" if returncode == 0 else "failed",
        "log": log_path.name,
        "log_sha256": _sha256(log_path),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_summary_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "## GPU qualification",
        "",
        f"- Status: **{summary['status']}**",
        f"- Commit: `{summary['commit_sha']}`",
        f"- Completed: `{summary['completed_at']}`",
        f"- Topology: `{summary['topology']['hosts']} host / "
        f"{summary['topology']['visible_gpus']} visible GPUs`",
        "",
        "| Check | Result | Duration |",
        "|---|---|---:|",
    ]
    for result in summary["results"]:
        lines.append(
            f"| `{result['name']}` | {result['status']} | {result['duration_seconds']:.3f}s |"
        )
    if summary.get("error"):
        lines.extend(["", f"Error: `{summary['error']}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(artifact_dir: Path) -> None:
    lines = []
    for path in sorted(item for item in artifact_dir.rglob("*") if item.is_file()):
        if path.name == "checksums.txt":
            continue
        lines.append(f"{_sha256(path)}  {path.relative_to(artifact_dir)}")
    (artifact_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--minimum-gpus", type=int, default=2)
    args = parser.parse_args()
    if args.minimum_gpus < 1:
        parser.error("--minimum-gpus must be positive")

    artifact_dir = args.artifact_dir.resolve()
    artifact_dir.mkdir(parents=True, exist_ok=True)
    environment = _environment()
    _write_json(artifact_dir / "environment.json", environment)

    command_specs = _commands(artifact_dir)
    (artifact_dir / "commands.txt").write_text(
        "\n".join(shlex.join(command) for _, command in command_specs) + "\n",
        encoding="utf-8",
    )
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "repository": os.environ.get("GITHUB_REPOSITORY"),
        "commit_sha": os.environ.get("GITHUB_SHA") or _git_revision(),
        "ref": os.environ.get("GITHUB_REF"),
        "event": os.environ.get("GITHUB_EVENT_NAME"),
        "started_at": _timestamp(),
        "completed_at": None,
        "status": "running",
        "topology": {
            "hosts": 1,
            "required_gpus": args.minimum_gpus,
            "visible_gpus": environment["device_count"],
        },
        "results": [],
    }

    if not environment["cuda_available"] or environment["device_count"] < args.minimum_gpus:
        summary["status"] = "failed"
        summary["error"] = (
            f"requires at least {args.minimum_gpus} CUDA GPUs; found {environment['device_count']}"
        )
    else:
        command_environment = os.environ.copy()
        root = str(_REPOSITORY_ROOT)
        command_environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (root, command_environment.get("PYTHONPATH")) if part
        )
        for name, command in command_specs:
            result = _run_command(name, command, artifact_dir, command_environment)
            summary["results"].append(result)
            _write_json(artifact_dir / "summary.json", summary)
        summary["status"] = (
            "passed"
            if all(result["status"] == "passed" for result in summary["results"])
            else "failed"
        )

    summary["completed_at"] = _timestamp()
    _write_json(artifact_dir / "summary.json", summary)
    _write_summary_markdown(artifact_dir / "summary.md", summary)
    _write_checksums(artifact_dir)
    return 0 if summary["status"] == "passed" else 1


def _git_revision() -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


if __name__ == "__main__":
    raise SystemExit(main())
