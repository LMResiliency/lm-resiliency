"""Run an eight-GPU DDP fault matrix and validate SCOUT localization."""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from collections.abc import Callable, Container
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from examples.fault_injection.compare import compare_artifacts
from examples.production_loops.pytorch import TinyCausalLM, _tokens
from lm_resiliency import (
    FaultCampaign,
    FaultIncident,
    FaultInjectionSession,
    InMemoryCkptConfig,
    OrchestrationHooks,
    ReplayHarnessConfig,
    enable_fault_injection,
    enable_resiliency,
)
from lm_resiliency.fault_injection.injector import _probability_selected


@dataclass
class LocalizationCollector:
    """Collect normalized reports emitted by enable_resiliency()."""

    training_iteration: int = 0
    reports: list[dict[str, Any]] = field(default_factory=list)

    def record(self, report: dict[str, Any]) -> None:
        self.reports.append(
            {
                "training_iteration": self.training_iteration,
                **dict(report),
            }
        )


class EvaluationStateReset:
    """Restore the last clean state after faults that can contaminate later cases."""

    def __init__(
        self,
        model: DistributedDataParallel,
        optimizer: torch.optim.Optimizer,
        reset_iterations: Container[int],
        hold_iterations: Container[int] = (),
    ) -> None:
        self._model = model.module
        self._optimizer = optimizer
        self._reset_iterations = reset_iterations
        self._hold_iterations = hold_iterations
        self._model_state: dict[str, Any]
        self._optimizer_state: dict[str, Any]
        self._completed_iterations = 0
        self.restored_iterations: list[int] = []
        self._capture()
        self._handle = optimizer.register_step_post_hook(self._after_step)

    def close(self) -> None:
        self._handle.remove()

    def _after_step(self, *_args: Any, **_kwargs: Any) -> None:
        self._completed_iterations += 1
        if self._completed_iterations in self._hold_iterations:
            return
        if self._completed_iterations in self._reset_iterations:
            device = next(self._model.parameters()).device
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            self._model.load_state_dict(copy.deepcopy(self._model_state))
            self._optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))
            self.restored_iterations.append(self._completed_iterations)
            return
        self._capture()

    def _capture(self) -> None:
        self._model_state = copy.deepcopy(self._model.state_dict())
        self._optimizer_state = copy.deepcopy(self._optimizer.state_dict())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument(
        "--campaign",
        type=Path,
        default=Path(__file__).with_name("campaign.json"),
    )
    parser.add_argument("--steps", type=int)
    args = parser.parse_args()

    campaign = FaultCampaign.from_json(args.campaign)
    steps = args.steps if args.steps is not None else _last_scheduled_iteration(campaign) + 1
    _validate_run(campaign, steps)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != 8:
        raise RuntimeError("the systematic campaign requires exactly eight DDP replicas")
    _validate_target_ranks(campaign, world_size)

    torch.manual_seed(123)
    model = DistributedDataParallel(
        TinyCausalLM().to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    localizations = LocalizationCollector()

    # Register SCOUT first so detection runs before a one-iteration fault is retired.
    resiliency = enable_resiliency(
        model,
        optimizer,
        interval=1,
        checkpoint=InMemoryCkptConfig(
            enable=True,
            interval=1,
            replication_jump=max(1, world_size // 2),
            disk_flush_interval=0,
            disk_folder=str(args.artifact_dir / "gemini"),
        ),
        replay=ReplayHarnessConfig(
            check_interval=1,
            layer_index=0,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
            straggler_min_slowdown_ratio=100.0,
            straggler_min_slowdown_ms=10.0,
            straggler_confirmation_rounds=1,
        ),
        device=device,
        orchestration=OrchestrationHooks(report_fault=localizations.record),
    )
    faults = None
    state_reset = EvaluationStateReset(
        model,
        optimizer,
        _state_reset_iterations(campaign),
        _state_hold_iterations(campaign),
    )
    try:
        faults = enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
        )
        for step in range(1, steps + 1):
            localizations.training_iteration = step
            tokens, labels = _tokens(rank, step - 1, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, labels)
            loss.backward()
            optimizer.step()
            _complete_permanent_incidents(faults, campaign, step)

        if resiliency.ckpt_manager is not None:
            resiliency.ckpt_manager.maybe_wait()
        if resiliency.step_count != steps or faults.completed_iterations != steps:
            raise AssertionError(
                "training, resiliency, and fault-injection iteration clocks diverged"
            )
        final_result = (
            None if resiliency.replay_harness is None else resiliency.replay_harness.last_result
        )
        if final_result is None or any(final_result.sdc_bitmap):
            raise AssertionError("the final post-fault SCOUT replay was not clean")

        local_records = [record.to_dict() for record in faults.records]
        gathered_records: list[list[dict[str, Any]] | None] | None = (
            [None] * world_size if rank == 0 else None
        )
        dist.gather_object(local_records, gathered_records, dst=0)
        dist.barrier()
        if rank == 0:
            if gathered_records is None:
                raise AssertionError("rank 0 did not allocate injection record collection")
            injection_records = sorted(
                (record for rank_records in gathered_records for record in (rank_records or ())),
                key=lambda record: (
                    int(record["iteration"]),
                    str(record["injection_id"]),
                ),
            )
            injection_path = args.artifact_dir / "injection.json"
            localization_path = args.artifact_dir / "localization.json"
            evaluation_path = args.artifact_dir / "evaluation.json"
            _write_json(
                injection_path,
                {
                    "schema_version": 1,
                    "campaign": campaign.name,
                    "manifest_identity": campaign.manifest_identity,
                    "manifest": campaign.to_dict(),
                    "framework": faults.framework,
                    "world_size": world_size,
                    "completed_iterations": faults.completed_iterations,
                    "state_reset_iterations": state_reset.restored_iterations,
                    "injections": injection_records,
                },
            )
            _write_json(
                localization_path,
                {
                    "schema_version": 1,
                    "campaign": campaign.name,
                    "manifest_identity": campaign.manifest_identity,
                    "framework": "pytorch",
                    "reports": localizations.reports,
                },
            )
            evaluation = compare_artifacts(
                injection_path,
                localization_path,
                evaluation_path,
            )
            print(json.dumps(evaluation, sort_keys=True), flush=True)
            if not evaluation["summary"]["passed"]:
                raise AssertionError(
                    "SCOUT did not detect and localize every successfully injected occurrence"
                )
    finally:
        _teardown(
            faults,
            state_reset,
            resiliency,
            active_error=sys.exc_info()[1],
            destroy_process_group=dist.destroy_process_group,
        )


def _validate_run(campaign: FaultCampaign, steps: int) -> None:
    _validate_call_bounded_lifetimes(campaign)
    if steps <= 0:
        raise ValueError("--steps must be positive")
    latest = _last_scheduled_iteration(campaign)
    if steps <= latest:
        raise ValueError("--steps must include at least one clean post-fault iteration")


def _validate_target_ranks(campaign: FaultCampaign, world_size: int) -> None:
    invalid = sorted(
        {
            fault.target.execution_rank
            for incident in campaign.incidents
            for fault in incident.faults
            if fault.target.execution_rank >= world_size
        }
    )
    if invalid:
        raise ValueError(f"campaign targets unavailable global ranks: {invalid}")


def _last_scheduled_iteration(campaign: FaultCampaign) -> int:
    _validate_call_bounded_lifetimes(campaign)
    latest: list[int] = []
    for incident in campaign.incidents:
        if incident.trigger.at:
            last_trigger = incident.trigger.at[-1]
        else:
            trigger_range = incident.trigger.range
            if trigger_range is None:
                raise AssertionError("validated incident has no trigger schedule")
            steps = (trigger_range.end - trigger_range.start) // trigger_range.every
            last_trigger = trigger_range.start + steps * trigger_range.every
        if incident.lifetime.iterations is not None:
            last_trigger += incident.lifetime.iterations - 1
        latest.append(last_trigger)
    return max(latest)


def _validate_call_bounded_lifetimes(campaign: FaultCampaign) -> None:
    if any(
        incident.lifetime.matching_calls is not None and incident.lifetime.matching_calls != 1
        for incident in campaign.incidents
    ):
        raise ValueError(
            "the evaluation example supports matching_calls=1 only because "
            "framework call multiplicity cannot be inferred from optimizer iterations"
        )


@dataclass(frozen=True)
class _ResetIterationSchedule:
    exact: frozenset[int]
    ranged: tuple[tuple[FaultIncident, int], ...]
    campaign_seed: int

    def __contains__(self, iteration: object) -> bool:
        if not isinstance(iteration, int):
            return False
        return iteration in self.exact or any(
            incident.trigger.range is not None
            and incident.trigger.range.matches(iteration - offset)
            and _incident_selected(incident, iteration - offset, self.campaign_seed)
            for incident, offset in self.ranged
        )


@dataclass(frozen=True)
class _HoldIterationSchedule:
    exact: tuple[tuple[int, int], ...]
    ranged: tuple[tuple[FaultIncident, int], ...]
    campaign_seed: int

    def __contains__(self, iteration: object) -> bool:
        if not isinstance(iteration, int):
            return False
        if any(start <= iteration <= end for start, end in self.exact):
            return True
        for incident, offset in self.ranged:
            trigger_range = incident.trigger.range
            if trigger_range is None or iteration < trigger_range.start:
                continue
            trigger = (
                trigger_range.start
                + ((iteration - trigger_range.start) // trigger_range.every) * trigger_range.every
            )
            if (
                trigger <= trigger_range.end
                and 0 <= iteration - trigger < offset
                and _incident_selected(incident, trigger, self.campaign_seed)
            ):
                return True
        return False


def _state_reset_iterations(campaign: FaultCampaign) -> Container[int]:
    eligible = _state_reset_incidents(campaign)
    offsets = {incident.incident_id: _state_reset_offset(incident) for incident in eligible}
    ranged = tuple(
        (incident, offsets[incident.incident_id])
        for incident in eligible
        if incident.trigger.range is not None
    )
    exact = frozenset(
        iteration + offsets[incident.incident_id]
        for incident in eligible
        if incident.trigger.at
        for iteration in incident.trigger.at
        if _incident_selected(incident, iteration, campaign.seed)
    )
    if not ranged:
        return set(exact)
    return _ResetIterationSchedule(
        exact=exact,
        ranged=ranged,
        campaign_seed=campaign.seed,
    )


def _state_hold_iterations(campaign: FaultCampaign) -> Container[int]:
    eligible = _state_reset_incidents(campaign)
    offsets = {incident.incident_id: _state_reset_offset(incident) for incident in eligible}
    exact = tuple(
        (iteration, iteration + offsets[incident.incident_id] - 1)
        for incident in eligible
        if incident.trigger.at and offsets[incident.incident_id] > 0
        for iteration in incident.trigger.at
        if _incident_selected(incident, iteration, campaign.seed)
    )
    ranged = tuple(
        (incident, offsets[incident.incident_id])
        for incident in eligible
        if incident.trigger.range is not None and offsets[incident.incident_id] > 0
    )
    return _HoldIterationSchedule(
        exact=exact,
        ranged=ranged,
        campaign_seed=campaign.seed,
    )


def _state_reset_incidents(campaign: FaultCampaign) -> tuple[FaultIncident, ...]:
    gradient_affecting_surfaces = {
        "input",
        "output",
        "weight",
        "bias",
        "gradient",
        "optimizer_state",
    }
    return tuple(
        incident
        for incident in campaign.incidents
        if any(
            fault.type.value != "delay"
            and fault.target.surface.value in gradient_affecting_surfaces
            for fault in incident.faults
        )
    )


def _state_reset_offset(incident: FaultIncident) -> int:
    if incident.lifetime.iterations is not None:
        return incident.lifetime.iterations - 1
    if incident.lifetime.matching_calls is not None:
        if incident.lifetime.matching_calls != 1:
            raise ValueError(
                "the evaluation example supports matching_calls=1 for gradient-affecting incidents"
            )
        return 0
    if incident.lifetime.until in {"recovery", "replacement"}:
        return 0
    raise ValueError(
        "the evaluation example does not support campaign_end lifetimes for "
        "gradient-affecting incidents"
    )


def _complete_permanent_incidents(
    faults: FaultInjectionSession,
    campaign: FaultCampaign,
    iteration: int,
) -> None:
    boundaries = {
        incident.lifetime.until
        for incident in campaign.incidents
        if incident.trigger.matches(iteration)
        and incident.lifetime.permanent
        and _incident_selected(incident, iteration, campaign.seed)
    }
    if "recovery" in boundaries:
        faults.notify_recovery()
    if "replacement" in boundaries:
        faults.notify_replacement()


def _incident_selected(
    incident: FaultIncident,
    iteration: int,
    campaign_seed: int,
) -> bool:
    return _probability_selected(
        campaign_seed,
        incident.incident_id,
        iteration,
        incident.trigger.probability,
    )


def _teardown(
    faults: FaultInjectionSession | None,
    state_reset: EvaluationStateReset,
    resiliency: Any,
    *,
    active_error: BaseException | None,
    destroy_process_group: Callable[[], None],
) -> None:
    cleanup_errors: list[BaseException] = []
    actions = (
        None if faults is None else faults.close,
        state_reset.close,
        resiliency.close,
        destroy_process_group,
    )
    for action in actions:
        if action is None:
            continue
        try:
            action()
        except BaseException as error:
            cleanup_errors.append(error)

    if active_error is not None:
        for error in cleanup_errors:
            _add_exception_note(active_error, f"example teardown also failed: {error}")
        return
    if cleanup_errors:
        first_error = cleanup_errors[0]
        for error in cleanup_errors[1:]:
            _add_exception_note(first_error, f"additional teardown failure: {error}")
        raise first_error


def _add_exception_note(error: BaseException, note: str) -> None:
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = list(getattr(error, "__notes__", ()))
    notes.append(note)
    error.__notes__ = notes


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
