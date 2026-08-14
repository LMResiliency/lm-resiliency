"""Tests for the automated GPU qualification evidence harness."""

from __future__ import annotations

import json
import sys

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
            "cuda_available": False,
            "device_count": 0,
            "devices": [],
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

    assert run_gpu_qualification.main() == 1
    summary = json.loads((artifact_dir / "summary.json").read_text())
    assert summary["schema_version"] == 1
    assert summary["status"] == "failed"
    assert summary["topology"] == {"hosts": 1, "required_gpus": 2, "visible_gpus": 0}
    assert summary["error"] == "requires at least 2 CUDA GPUs; found 0"
    assert (artifact_dir / "environment.json").exists()
    assert (artifact_dir / "commands.txt").exists()
    assert (artifact_dir / "summary.md").exists()
    assert (artifact_dir / "checksums.txt").exists()
