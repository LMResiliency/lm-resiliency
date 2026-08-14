"""Run an eight-GPU DDP fault matrix and validate SCOUT localization."""

from __future__ import annotations

import argparse
import copy
import json
import os
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
        reset_iterations: set[int],
    ) -> None:
        self._model = model.module
        self._optimizer = optimizer
        self._reset_iterations = frozenset(reset_iterations)
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
        if self._completed_iterations in self._reset_iterations:
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
        if faults is not None:
            faults.close()
        state_reset.close()
        resiliency.close()
        dist.destroy_process_group()


def _validate_run(campaign: FaultCampaign, steps: int) -> None:
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
    return max(
        iteration
        for incident in campaign.incidents
        for iteration in _scheduled_iterations(incident)
    )


def _scheduled_iterations(incident: FaultIncident) -> tuple[int, ...]:
    if incident.trigger.at:
        return incident.trigger.at
    trigger_range = incident.trigger.range
    if trigger_range is None:
        raise AssertionError("validated incident has no trigger schedule")
    return tuple(
        range(
            trigger_range.start,
            trigger_range.end + 1,
            trigger_range.every,
        )
    )


def _state_reset_iterations(campaign: FaultCampaign) -> set[int]:
    state_surfaces = {"weight", "bias", "gradient", "optimizer_state"}
    return {
        iteration
        for incident in campaign.incidents
        if any(fault.target.surface.value in state_surfaces for fault in incident.faults)
        for iteration in _scheduled_iterations(incident)
    }


def _complete_permanent_incidents(
    faults: FaultInjectionSession,
    campaign: FaultCampaign,
    iteration: int,
) -> None:
    boundaries = {
        incident.lifetime.until
        for incident in campaign.incidents
        if iteration in _scheduled_iterations(incident) and incident.lifetime.permanent
    }
    if "recovery" in boundaries:
        faults.notify_recovery()
    if "replacement" in boundaries:
        faults.notify_replacement()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
