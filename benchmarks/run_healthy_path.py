"""Run isolated healthy-path modes and enforce regression thresholds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _free_local_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def regression_percent(baseline: float, current: float, *, direction: str) -> float:
    if baseline <= 0:
        raise ValueError("baseline metrics must be positive")
    if direction == "higher":
        return (baseline - current) / baseline * 100.0
    if direction == "lower":
        return (current - baseline) / baseline * 100.0
    raise ValueError(f"unsupported metric direction: {direction}")


def summarize_results(
    runs: dict[str, dict[str, Any]], thresholds: dict[str, Any]
) -> dict[str, Any]:
    baseline = runs["baseline"]["metrics"]
    comparisons = []
    violations = []
    for metric, policy in thresholds["metrics"].items():
        for mode, limit in policy["maximum_regression_percent"].items():
            if mode not in runs:
                continue
            regression = regression_percent(
                float(baseline[metric]),
                float(runs[mode]["metrics"][metric]),
                direction=policy["direction"],
            )
            comparison = {
                "metric": metric,
                "mode": mode,
                "baseline": baseline[metric],
                "current": runs[mode]["metrics"][metric],
                "regression_percent": regression,
                "maximum_regression_percent": limit,
                "passed": regression <= limit,
            }
            comparisons.append(comparison)
            if not comparison["passed"]:
                violations.append(comparison)
    return {
        "schema_version": 1,
        "status": "passed" if not violations else "failed",
        "commit_sha": runs["baseline"].get("commit_sha"),
        "thresholds": thresholds,
        "comparisons": comparisons,
        "violations": violations,
        "runs": runs,
    }


def _write_markdown(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "## Healthy-path performance",
        "",
        f"- Status: **{summary['status']}**",
        f"- Commit: `{summary['commit_sha']}`",
        "",
        "| Mode | Metric | Regression | Limit | Result |",
        "|---|---|---:|---:|---|",
    ]
    for comparison in summary["comparisons"]:
        lines.append(
            f"| `{comparison['mode']}` | `{comparison['metric']}` | "
            f"{comparison['regression_percent']:.2f}% | "
            f"{comparison['maximum_regression_percent']:.2f}% | "
            f"{'passed' if comparison['passed'] else 'failed'} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_checksums(output_dir: Path) -> None:
    lines = []
    for path in sorted(item for item in output_dir.iterdir() if item.is_file()):
        if path.name == "checksums.txt":
            continue
        lines.append(f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}")
    (output_dir / "checksums.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--thresholds",
        type=Path,
        default=ROOT / "benchmarks" / "thresholds.json",
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--world-size", type=int, default=2)
    parser.add_argument("--modes", nargs="+", default=["baseline", "gemini", "scout", "combined"])
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--interval", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=512)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--replication-chunk-size", type=int, default=16 * 1024 * 1024)
    parser.add_argument("--pin-memory", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce", action="store_true")
    args = parser.parse_args()
    if args.world_size < 1:
        parser.error("world size must be positive")
    if "baseline" not in args.modes:
        parser.error("modes must include baseline")
    if args.device == "cpu" and {"scout", "combined"}.intersection(args.modes):
        parser.error("SCOUT healthy-path modes require --device cuda")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    runs = {}
    commands = []
    for mode in args.modes:
        output = args.output_dir / f"{mode}.json"
        command = [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--master-addr=127.0.0.1",
            f"--master-port={_free_local_port()}",
            f"--nproc-per-node={args.world_size}",
            "benchmarks/healthy_path.py",
            "--mode",
            mode,
            "--device",
            args.device,
            "--output",
            str(output),
            "--steps",
            str(args.steps),
            "--warmup-steps",
            str(args.warmup_steps),
            "--interval",
            str(args.interval),
            "--batch-size",
            str(args.batch_size),
            "--sequence-length",
            str(args.sequence_length),
            "--hidden-size",
            str(args.hidden_size),
            "--layers",
            str(args.layers),
            "--heads",
            str(args.heads),
            "--replication-chunk-size",
            str(args.replication_chunk_size),
            "--pin-memory" if args.pin_memory else "--no-pin-memory",
        ]
        commands.append(command)
        environment = os.environ.copy()
        environment["PYTHONPATH"] = os.pathsep.join(
            part for part in (str(ROOT), environment.get("PYTHONPATH")) if part
        )
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
        runs[mode] = json.loads(output.read_text())

    thresholds = json.loads(args.thresholds.read_text())
    summary = summarize_results(runs, thresholds)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "commands.json").write_text(
        json.dumps(commands, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_markdown(args.output_dir / "summary.md", summary)
    _write_checksums(args.output_dir)
    print(json.dumps(summary, sort_keys=True))
    return 1 if args.enforce and summary["status"] != "passed" else 0


if __name__ == "__main__":
    raise SystemExit(main())
