import json
import os
import subprocess
import sys
from pathlib import Path


def _run_example(checkpoint_dir: Path, *, steps: int) -> dict:
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ""
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lm_resiliency.quickstart",
            "--steps",
            str(steps),
            "--interval",
            "1",
            "--checkpoint-dir",
            str(checkpoint_dir),
        ],
        check=True,
        capture_output=True,
        env=environment,
        text=True,
    )
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_quickstart_trains_and_resumes(tmp_path: Path) -> None:
    checkpoint_dir = tmp_path / "checkpoints"

    initial = _run_example(checkpoint_dir, steps=2)
    assert initial["start_step"] == 0
    assert initial["recovered_step"] == -1
    assert initial["completed_step"] == 2
    assert initial["checkpoint_step"] == 2

    resumed = _run_example(checkpoint_dir, steps=3)
    assert resumed["start_step"] == 2
    assert resumed["recovered_step"] == 2
    assert resumed["completed_step"] == 3
    assert resumed["checkpoint_step"] == 3
