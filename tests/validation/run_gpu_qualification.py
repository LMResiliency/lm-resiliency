"""Run the smallest trusted GPU qualification tier and publish JSON evidence."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
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
        "schema_version": _SCHEMA_VERSION,
        "captured_at": _timestamp(),
        "hardware": {
            "host": platform.node(),
            "platform": platform.platform(),
            "hosts": 1,
            "world_size": 2,
            "gpu_count": device_count,
            "devices": devices,
            "nvidia_smi": _nvidia_smi(),
        },
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
            "cuda_available": cuda_available,
            "cuda_runtime": torch.version.cuda,
            "nccl": list(nccl_version) if isinstance(nccl_version, tuple) else nccl_version,
            "frameworks": {
                "lm-resiliency": importlib.metadata.version("lm-resiliency"),
                "pytorch": torch.__version__,
            },
        },
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
        "command_id": name,
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


def _write_summary_markdown(path: Path, summary: dict[str, Any], commit_sha: str) -> None:
    lines = [
        "## GPU qualification",
        "",
        f"- Status: **{summary['status']}**",
        f"- Commit: `{commit_sha}`",
        f"- Completed: `{summary['completed_at']}`",
        f"- Topology: `{summary['topology']['hosts']} host / "
        f"{summary['topology']['gpu_count']} visible GPUs`",
        "",
        "| Check | Result | Duration |",
        "|---|---|---:|",
    ]
    for result in summary["results"]:
        lines.append(
            f"| `{result['command_id']}` | {result['status']} | "
            f"{result.get('duration_seconds', 0.0):.3f}s |"
        )
    reasons = [result.get("reason") for result in summary["results"] if result.get("reason")]
    if reasons:
        lines.extend(["", f"Error: `{reasons[0]}`"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _seal_evidence(
    artifact_dir: Path,
    summary: dict[str, Any],
    *,
    commit_sha: str,
    ref: str,
) -> None:
    repository = os.environ.get("GITHUB_REPOSITORY", "LMResiliency/lm-resiliency")
    run_id = os.environ.get("GITHUB_RUN_ID")
    artifact_name = f"gpu-qualification-{commit_sha}"
    artifact_url = (
        f"https://github.com/{repository}/actions/runs/{run_id}" if run_id else str(artifact_dir)
    )
    command = [
        sys.executable,
        str(_REPOSITORY_ROOT / "scripts" / "validation_evidence.py"),
        "seal",
        "--bundle",
        str(artifact_dir),
        "--campaign-id",
        "core-gpu-qualification",
        "--tier",
        "single-host-gpu",
        "--repository",
        repository,
        "--commit",
        commit_sha,
        "--ref",
        ref,
        "--artifact-name",
        artifact_name,
        "--artifact-url",
        artifact_url,
        "--framework",
        "pytorch",
        "--framework",
        "gemini",
        "--framework",
        "scout",
        "--boundary",
        "One host and two ranks; multi-host transport is not qualified.",
        "--boundary",
        "Tiny deterministic workloads qualify recovery and localization, not convergence.",
    ]
    completed = subprocess.run(command, cwd=_REPOSITORY_ROOT, check=False)
    if completed.returncode:
        raise RuntimeError("failed to seal GPU qualification evidence")


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
        "\n".join(f"[{name}] {shlex.join(command)}" for name, command in command_specs) + "\n",
        encoding="utf-8",
    )
    commit_sha = os.environ.get("GITHUB_SHA") or _git_revision()
    ref = os.environ.get("GITHUB_REF") or _git_ref()
    summary: dict[str, Any] = {
        "schema_version": _SCHEMA_VERSION,
        "campaign_id": "core-gpu-qualification",
        "started_at": _timestamp(),
        "completed_at": None,
        "status": "running",
        "topology": {
            "hosts": 1,
            "world_size": 2,
            "gpu_count": environment["hardware"]["gpu_count"],
        },
        "seed": None,
        "configuration": {"minimum_gpus": args.minimum_gpus},
        "counts": {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "errors": 0},
        "metrics": {},
        "results": [],
    }

    gpu_count = environment["hardware"]["gpu_count"]
    if not environment["software"]["cuda_available"] or gpu_count < args.minimum_gpus:
        summary["status"] = "failed"
        reason = f"requires at least {args.minimum_gpus} CUDA GPUs; found {gpu_count}"
        summary["results"] = [
            {"command_id": name, "status": "skipped", "reason": reason} for name, _ in command_specs
        ]
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
    summary["counts"] = {
        "total": len(summary["results"]),
        "passed": sum(result["status"] == "passed" for result in summary["results"]),
        "failed": sum(result["status"] == "failed" for result in summary["results"]),
        "skipped": sum(result["status"] == "skipped" for result in summary["results"]),
        "errors": sum(result["status"] == "error" for result in summary["results"]),
    }
    summary["metrics"] = {
        "duration_seconds": round(
            sum(result.get("duration_seconds", 0.0) for result in summary["results"]), 3
        )
    }
    _write_json(artifact_dir / "summary.json", summary)
    _write_summary_markdown(artifact_dir / "summary.md", summary, commit_sha)
    _seal_evidence(artifact_dir, summary, commit_sha=commit_sha, ref=ref)
    return 0 if summary["status"] == "passed" else 1


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    revision = completed.stdout.strip()
    if completed.returncode or not revision:
        raise RuntimeError("could not determine the repository revision")
    return revision


def _git_ref() -> str:
    completed = subprocess.run(
        ["git", "symbolic-ref", "--short", "HEAD"],
        cwd=_REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else "detached"


if __name__ == "__main__":
    raise SystemExit(main())
