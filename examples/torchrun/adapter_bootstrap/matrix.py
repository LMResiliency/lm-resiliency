"""Launch the unchanged production loops through the torchrun adapters."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from lm_resiliency.integrations.torchrun import TorchrunLaunchConfig

from .framework_worker import FRAMEWORKS


def _write_worker_policy(path: Path, *, replication_jump: int) -> None:
    path.write_text(
        "\n".join(
            (
                "schema_version = 1",
                "interval = 1",
                "enable_checkpoint = true",
                "enable_detection = true",
                "",
                "[checkpoint]",
                f"replication_jump = {replication_jump}",
                "disk_flush_interval = 0",
                f"disk_folder = {json.dumps(str((path.parent / 'checkpoints').resolve()))}",
                "",
                "[replay]",
                "rotate_layers = false",
                "enable_temporal = false",
                "scale_factors = []",
                "straggler_min_slowdown_ratio = 100.0",
                "straggler_min_slowdown_ms = 10000.0",
                "",
            )
        ),
        encoding="utf-8",
    )


def _run_framework(
    framework: str,
    *,
    nproc_per_node: int,
    output_dir: Path,
    policy: Path,
    steps: int,
    torchrun: str,
) -> dict[str, object]:
    framework_dir = output_dir / framework
    framework_dir.mkdir(parents=True, exist_ok=True)
    launch = TorchrunLaunchConfig(
        run_id=f"torchrun-validation-{framework}-{os.getpid()}",
        rendezvous_endpoint=str(framework_dir / "rdzv"),
        restart_context_path=(framework_dir / "context" / "context.json").resolve(),
        min_nodes=1,
        max_nodes=1,
        nproc_per_node=nproc_per_node,
        max_restarts=0,
        torchrun=torchrun,
        store_type="file",
        worker_config=policy.resolve(),
    )
    subprocess.run(
        launch.command(
            module="examples.torchrun.adapter_bootstrap.framework_worker",
            module_args=(
                f"--framework={framework}",
                f"--validation-output-dir={framework_dir}",
                f"--steps={steps}",
            ),
        ),
        check=True,
    )
    summary_path = framework_dir / f"{framework}-production-loop.json"
    if not summary_path.is_file():
        raise RuntimeError(f"{framework} did not publish its validation summary")
    value = json.loads(summary_path.read_text(encoding="utf-8"))
    if value.get("framework") != framework:
        raise RuntimeError(f"{framework} published a mismatched validation summary")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--framework",
        action="append",
        choices=(*FRAMEWORKS, "all"),
        default=[],
    )
    parser.add_argument("--nproc-per-node", type=int, default=8)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--torchrun", default=str(Path(sys.executable).with_name("torchrun")))
    parser.add_argument("--validation-output-dir", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.nproc_per_node < 2 or arguments.nproc_per_node % 2:
        raise ValueError("--nproc-per-node must be an even integer of at least two")
    if arguments.steps < 1:
        raise ValueError("--steps must be positive")
    frameworks = arguments.framework or ["all"]
    selected = FRAMEWORKS if "all" in frameworks else tuple(dict.fromkeys(frameworks))
    output_dir = arguments.validation_output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    policy = output_dir / "worker.toml"
    _write_worker_policy(policy, replication_jump=arguments.nproc_per_node // 2)
    summaries = {
        framework: _run_framework(
            framework,
            nproc_per_node=arguments.nproc_per_node,
            output_dir=output_dir,
            policy=policy,
            steps=arguments.steps,
            torchrun=arguments.torchrun,
        )
        for framework in selected
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summaries, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
