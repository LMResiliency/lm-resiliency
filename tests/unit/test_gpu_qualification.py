"""Tests for the automated GPU qualification evidence harness."""

from __future__ import annotations

import json
import sys

import pytest

from tests.validation import run_gpu_qualification


def test_gpu_qualification_commands_cover_core_two_gpu_paths(tmp_path):
    commands = dict(run_gpu_qualification._commands(tmp_path))

    assert set(commands) == {
        "recovery-equivalence",
        "replay-and-collectives",
        "fsdp2-checkpoint-recovery",
        "automatic-exit-cleanup",
    }
    assert "--nproc-per-node=2" in commands["replay-and-collectives"]
    assert "--nproc-per-node=2" in commands["fsdp2-checkpoint-recovery"]
    case_index = commands["automatic-exit-cleanup"].index("--case")
    assert commands["automatic-exit-cleanup"][case_index + 1] == "pytorch"


def test_gpu_qualification_writes_failure_evidence_without_required_gpus(tmp_path, monkeypatch):
    artifact_dir = tmp_path / "evidence"
    monkeypatch.setattr(
        run_gpu_qualification,
        "_environment",
        lambda: {
            "schema_version": 1,
            "captured_at": "2026-08-14T00:00:00+00:00",
            "hardware": {
                "host": "test",
                "platform": "test",
                "hosts": 1,
                "world_size": 2,
                "gpu_count": 0,
                "devices": [],
                "nvidia_smi": {},
            },
            "software": {
                "python": "3.12",
                "python_executable": sys.executable,
                "cuda_available": False,
                "cuda_runtime": None,
                "nccl": None,
                "frameworks": {"lm-resiliency": "0.2.0", "pytorch": "2.13.0"},
            },
            "runner": {"name": None, "os": None, "arch": None},
        },
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_gpu_qualification.py",
            "--artifact-dir",
            str(artifact_dir),
            "--minimum-gpus",
            "2",
        ],
    )
    monkeypatch.setenv("GITHUB_SHA", "a" * 40)
    monkeypatch.setenv("GITHUB_REF", "refs/heads/test")

    assert run_gpu_qualification.main() == 1
    summary = json.loads((artifact_dir / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["status"] == "failed"
    assert summary["topology"] == {"hosts": 1, "world_size": 2, "gpu_count": 0}
    assert summary["counts"] == {
        "total": 4,
        "passed": 0,
        "failed": 0,
        "skipped": 4,
        "errors": 0,
    }
    assert {result["reason"] for result in summary["results"]} == {
        "requires at least 2 CUDA GPUs; found 0"
    }
    assert (artifact_dir / "environment.json").exists()
    assert (artifact_dir / "commands.txt").exists()
    assert (artifact_dir / "summary.md").exists()
    assert (artifact_dir / "manifest.json").exists()
    assert (artifact_dir / "checksums.txt").exists()


def test_git_revision_rejects_tracked_worktree_changes(monkeypatch):
    class Completed:
        returncode = 0
        stdout = " M tracked.py\n"

    monkeypatch.setattr(
        run_gpu_qualification.subprocess, "run", lambda *args, **kwargs: Completed()
    )

    with pytest.raises(RuntimeError, match="clean tracked worktree"):
        run_gpu_qualification._git_revision()
