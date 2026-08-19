"""Evidence and numerical verification for torchrun pressure validation."""

from __future__ import annotations

import math
from collections.abc import Sequence
from pathlib import Path

import torch

from .artifacts import read_json
from .campaign import PressureEvent

LOSS_MAX_ABS_DIFF = 1e-2
TORCHTITAN_LOSS_MAX_ABS_DIFF = 1e-1
MODEL_MAX_ABS_DIFF = 2e-3
OPTIMIZER_MAX_ABS_DIFF = 5e-5
DEEPSPEED_OPTIMIZER_MAX_ABS_DIFF = 1e-3


def loss_difference_limit(framework: str) -> float:
    if framework == "torchtitan":
        return TORCHTITAN_LOSS_MAX_ABS_DIFF
    return LOSS_MAX_ABS_DIFF


def optimizer_difference_limit(framework: str) -> float:
    if framework == "deepspeed":
        return DEEPSPEED_OPTIMIZER_MAX_ABS_DIFF
    return OPTIMIZER_MAX_ABS_DIFF


def loss_difference(expected: float, actual: float, *, rank: int, step: str) -> float:
    """Return a finite loss difference or reject invalid numerical evidence."""

    difference = abs(expected - actual)
    if not all(math.isfinite(value) for value in (expected, actual, difference)):
        raise AssertionError(f"rank {rank} loss at step {step} is non-finite")
    return difference


def checkpoint_topology_digest(reports: Sequence[dict[str, object]]) -> str:
    """Return the one live GEMINI topology identity agreed by every report."""

    values = [report.get("topology_digest") for report in reports]
    if not values or any(not isinstance(value, str) or not value for value in values):
        raise AssertionError("incident report checkpoint topology is malformed")
    unique_values = set(values)
    if len(unique_values) != 1:
        raise AssertionError("incident reports disagree on checkpoint topology")
    return unique_values.pop()


def validate_fault_reports(
    reports: Sequence[dict[str, object]],
    *,
    expected_generation: int,
    expected_checkpoint_step: int,
    fault_rank: int,
    world_size: int,
    require_injection: bool = False,
) -> None:
    expected_bitmap = [int(rank == fault_rank) for rank in range(world_size)]
    topology_digest = checkpoint_topology_digest(reports)
    for report in reports:
        if report["generation"] != expected_generation:
            raise AssertionError("fault report generation mismatch")
        if report["sdc_bitmap"] != expected_bitmap:
            raise AssertionError(f"SCOUT localization mismatch: {report}")
        if require_injection:
            _validate_injection(report)
        decision = report["decision"]
        if not isinstance(decision, dict):
            raise AssertionError("fault report recovery decision is malformed")
        if (
            decision["recovery_mode"] != "recovery_verified"
            or decision["checkpoint_source"] != "gemini"
            or decision["checkpoint_step"] != expected_checkpoint_step
            or decision.get("topology_digest") != topology_digest
            or not decision["available"]
        ):
            raise AssertionError(f"invalid recovery decision: {decision}")


def validate_isolated_replacement_reports(
    reports: Sequence[dict[str, object]],
    *,
    event: PressureEvent,
    expected_generation: int,
) -> None:
    for report in reports:
        if report["generation"] != expected_generation:
            raise AssertionError("replacement report generation mismatch")
        if report["incident_id"] != event.incident_id:
            raise AssertionError("replacement report incident mismatch")
        if report["checkpoint_step"] != event.checkpoint_step:
            raise AssertionError("replacement report checkpoint mismatch")
        if report.get("replacement_rank") != event.fault_rank:
            raise AssertionError("replacement report target-rank mismatch")
        if event.injection_executor is not None:
            _validate_injection(report)


def validate_restart_reports(
    reports: Sequence[dict[str, object]],
    *,
    event: PressureEvent,
    expected_generation: int,
) -> None:
    for report in reports:
        if report["generation"] != expected_generation:
            raise AssertionError("restart report generation mismatch")
        if report["incident_id"] != event.incident_id:
            raise AssertionError("restart report incident mismatch")
        if report["checkpoint_step"] != event.checkpoint_step:
            raise AssertionError("restart report checkpoint mismatch")
        if event.injection_executor is not None:
            _validate_injection(report)


def _validate_injection(report: dict[str, object]) -> None:
    injection = report.get("injection")
    if not isinstance(injection, dict):
        raise AssertionError("incident report injection evidence is malformed")
    if (
        injection.get("failure_type") != report.get("failure_type")
        or injection.get("status") != "completed"
        or injection.get("verified") is not True
        or injection.get("injection_succeeded") is not True
    ):
        raise AssertionError(f"incident report injection evidence is invalid: {injection}")


def compare_baseline(
    fault_campaign_dir: Path,
    *,
    generations: int,
    world_size: int,
) -> tuple[float, float, float]:
    baseline_dir = fault_campaign_dir / "baseline-artifacts"
    campaign_dir = fault_campaign_dir / "campaign-artifacts"
    campaign_losses = _merge_losses(
        campaign_dir,
        generations=generations,
        world_size=world_size,
    )
    maximum_model_difference = 0.0
    maximum_optimizer_difference = 0.0
    maximum_loss_difference = 0.0
    framework: str | None = None
    for rank in range(world_size):
        baseline = read_json(baseline_dir / f"baseline-r{rank}.json")
        campaign = read_json(campaign_dir / f"final-g{generations - 1}-r{rank}.json")
        if baseline["framework"] != campaign["framework"]:
            raise AssertionError(f"rank {rank} framework identity diverged")
        observed_framework = baseline["framework"]
        if not isinstance(observed_framework, str):
            raise AssertionError(f"rank {rank} framework identity is malformed")
        if framework is None:
            framework = observed_framework
        elif framework != observed_framework:
            raise AssertionError("baseline ranks disagree on the training framework")
        if baseline["rng_digest"] != campaign["rng_digest"]:
            raise AssertionError(f"rank {rank} final RNG state diverged")
        if baseline["framework_state"] != campaign["framework_state"]:
            raise AssertionError(f"rank {rank} final framework state diverged")
        baseline_losses = {key: float(value) for key, value in baseline["losses"].items()}
        for step, expected in baseline_losses.items():
            actual = float(campaign_losses[rank][step])
            difference = loss_difference(expected, actual, rank=rank, step=step)
            maximum_loss_difference = max(maximum_loss_difference, difference)
            loss_limit = loss_difference_limit(observed_framework)
            if difference > loss_limit:
                raise AssertionError(
                    f"rank {rank} loss at step {step} differs by {difference:.3e}, "
                    f"above {loss_limit:.1e}"
                )
        baseline_state = _load_verification_state(baseline_dir / f"baseline-state-r{rank}.pt")
        campaign_state = _load_verification_state(
            campaign_dir / f"final-g{generations - 1}-state-r{rank}.pt"
        )
        maximum_model_difference = max(
            maximum_model_difference,
            _tensor_max_abs_diff(
                baseline_state["model"],
                campaign_state["model"],
                limit=MODEL_MAX_ABS_DIFF,
                label="model",
                rank=rank,
            ),
        )
        maximum_optimizer_difference = max(
            maximum_optimizer_difference,
            _tensor_max_abs_diff(
                baseline_state["optimizer"],
                campaign_state["optimizer"],
                limit=optimizer_difference_limit(observed_framework),
                label="optimizer",
                rank=rank,
            ),
        )
    return (
        maximum_loss_difference,
        maximum_model_difference,
        maximum_optimizer_difference,
    )


def _merge_losses(
    artifact_dir: Path,
    *,
    generations: int,
    world_size: int,
) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {rank: {} for rank in range(world_size)}
    for generation in range(generations):
        for rank in range(world_size):
            path = artifact_dir / f"losses-g{generation}-r{rank}.json"
            if path.exists():
                merged[rank].update(read_json(path)["losses"])
    for rank in range(world_size):
        final = read_json(artifact_dir / f"final-g{generations - 1}-r{rank}.json")
        merged[rank].update(final["losses"])
    return merged


def _load_verification_state(path: Path) -> dict[str, list[torch.Tensor]]:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, dict) or set(value) != {"model", "optimizer"}:
        raise AssertionError(f"{path} does not contain a verification state")
    result: dict[str, list[torch.Tensor]] = {}
    for key in ("model", "optimizer"):
        tensors = value[key]
        if not isinstance(tensors, list) or not all(
            isinstance(tensor, torch.Tensor) for tensor in tensors
        ):
            raise AssertionError(f"{path} has invalid {key} tensors")
        result[key] = tensors
    return result


def _assert_same_tensor_layout(
    baseline: Sequence[torch.Tensor],
    campaign: Sequence[torch.Tensor],
    *,
    label: str,
    rank: int,
) -> None:
    if len(baseline) != len(campaign):
        raise AssertionError(f"rank {rank} final {label} tensor count diverged")
    for index, (expected, actual) in enumerate(zip(baseline, campaign)):
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            raise AssertionError(f"rank {rank} final {label} tensor {index} layout diverged")


def _tensor_max_abs_diff(
    baseline: Sequence[torch.Tensor],
    campaign: Sequence[torch.Tensor],
    *,
    limit: float,
    label: str,
    rank: int,
) -> float:
    _assert_same_tensor_layout(baseline, campaign, label=label, rank=rank)
    maximum = 0.0
    for index, (expected, actual) in enumerate(zip(baseline, campaign)):
        if not expected.is_floating_point():
            if not torch.equal(expected, actual):
                raise AssertionError(f"rank {rank} final {label} tensor {index} diverged")
            continue
        if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
            raise AssertionError(f"rank {rank} final {label} tensor {index} is non-finite")
        difference = float((expected - actual).abs().max()) if expected.numel() else 0.0
        maximum = max(maximum, difference)
        if difference > limit:
            raise AssertionError(
                f"rank {rank} final {label} tensor {index} differs by "
                f"{difference:.3e}, above {limit:.1e}"
            )
    return maximum
