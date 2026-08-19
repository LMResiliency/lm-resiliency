"""Framework-neutral worker runtime for torchrun fault injection."""

from __future__ import annotations

import hashlib
import os
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import torch
import torch.distributed as dist

from lm_resiliency import (
    FaultInjectionSession,
    InjectionStatus,
    InMemoryCkptConfig,
    OrchestrationHooks,
    RecoveryDecision,
    ReplayHarnessConfig,
)
from lm_resiliency.integrations.torchrun import TorchrunWorkerContext

from .artifacts import atomic_json, atomic_torch, read_json
from .campaign import PressureEvent
from .replay_fault import ReplayFaultCampaign


class FrameworkDriver(Protocol):
    """Minimal framework surface consumed by the shared campaign runtime."""

    framework: str
    device: torch.device
    rank: int
    world_size: int
    handle: Any
    expected_recipes: set[str]

    def run(
        self,
        *,
        before_step: Callable[[int], None],
        after_step: Callable[[int, float | None], None],
        total_steps: int,
    ) -> None: ...

    def verification_state(self) -> dict[str, list[torch.Tensor]]: ...

    def framework_state(self) -> dict[str, Any]: ...

    def fault_injection_objects(self) -> tuple[Any, Any | None]: ...

    def close(self) -> None: ...


@dataclass(frozen=True, slots=True)
class DriverConfig:
    """Common resiliency configuration supplied to a framework driver."""

    campaign_dir: Path
    checkpoint: InMemoryCkptConfig
    replay: ReplayHarnessConfig
    recovery_mode: str | None
    recovery_step: int | None
    expected_topology_id: str | None
    fault_callback: Callable[[Any], None]
    orchestration: OrchestrationHooks
    total_steps: int

    def recovery_options(self) -> dict[str, Any]:
        """Return the exact manager-selected recovery constraints."""

        return {
            "recovery_mode": self.recovery_mode,
            "_recovery_step": self.recovery_step,
            "_expected_topology_id": self.expected_topology_id,
        }


def replay_config() -> ReplayHarnessConfig:
    return ReplayHarnessConfig(
        check_interval=1,
        rotate_layers=False,
        enable_temporal=False,
        scale_factors=[],
        straggler_min_slowdown_ratio=100.0,
        straggler_min_slowdown_ms=10_000.0,
    )


def checkpoint_config(
    *,
    campaign_dir: Path,
    framework: str,
    mode: str,
    replication_jump: int,
) -> InMemoryCkptConfig:
    run_id = os.environ.get("TORCHELASTIC_RUN_ID")
    if not run_id:
        raise RuntimeError("fault-injection worker requires TORCHELASTIC_RUN_ID")
    return InMemoryCkptConfig(
        enable=True,
        interval=1,
        replication_jump=replication_jump,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=0,
        disk_folder=str(campaign_dir / f"{mode}-{framework}-checkpoints"),
        run_id=run_id,
        verify_integrity=True,
        pin_memory=True,
    )


def checkpoint_is_safe_for_replacement(
    checkpoint_manager: Any,
    handle: Any,
    *,
    expected_step: int,
) -> bool:
    """Verify that the contaminated step was neither saved nor made recoverable."""

    last_saved_step = int(checkpoint_manager._last_saved_step)
    verified_step = checkpoint_manager.checkpoint_status.recovery_verified_step
    flushed_step = handle.flush_for_restart()
    recoverable_step = checkpoint_manager.local_recovery_step("recovery_verified")
    safe = (
        last_saved_step in (-1, expected_step)
        and verified_step == expected_step
        and flushed_step in (-1, expected_step)
        and recoverable_step == expected_step
    )
    if not safe:
        print(
            "replacement checkpoint diagnostics: "
            f"rank={os.environ.get('RANK', 'unknown')} "
            f"expected={expected_step} saved={last_saved_step} verified={verified_step} "
            f"flushed={flushed_step} recoverable={recoverable_step}",
            flush=True,
        )
    return safe


def close_resources(*actions: tuple[str, Callable[[], None]]) -> None:
    """Run every cleanup action and re-raise the first failure."""

    failures: list[tuple[str, BaseException, Any]] = []
    for name, action in actions:
        try:
            action()
        except BaseException as error:
            failures.append((name, error, error.__traceback__))
    if not failures:
        return

    for name, error, _ in failures[1:]:
        print(f"additional {name} cleanup failure: {error}", file=sys.stderr)
    _, first_error, first_traceback = failures[0]
    raise first_error.with_traceback(first_traceback)


def checkpoint_topology_digest(handle: Any) -> str:
    """Return the live GEMINI job-wide topology identity."""

    manager = _checkpoint_manager(handle)
    topology_digest = None if manager is None else manager.topology_id
    if not isinstance(topology_digest, str) or not topology_digest:
        raise AssertionError("GEMINI did not publish a checkpoint topology digest")
    return topology_digest


def tensor_digest(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def rng_digest(device: torch.device) -> str:
    return tensor_digest(
        [
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device),
        ]
    )


def clone_tensors(values: Sequence[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().cpu().clone() for value in values]


def tensor_leaves(value: Any) -> list[torch.Tensor]:
    """Collect tensors from nested framework state in deterministic order."""
    tensors: list[torch.Tensor] = []
    if isinstance(value, torch.Tensor):
        tensors.append(value)
    elif isinstance(value, Mapping):
        for key in sorted(value, key=str):
            tensors.extend(tensor_leaves(value[key]))
    elif isinstance(value, (list, tuple)):
        for item in value:
            tensors.extend(tensor_leaves(item))
    return tensors


class CampaignRuntime:
    """Shared incident handling around one framework-native training driver."""

    def __init__(
        self,
        *,
        campaign_dir: Path,
        context: TorchrunWorkerContext | None,
        event: PressureEvent | None,
        framework: str,
        generation: int,
        mode: str,
        node_id: str,
        rank: int,
        replay_campaign: ReplayFaultCampaign,
        total_steps: int,
    ) -> None:
        self.campaign_dir = campaign_dir
        self.context = context
        self.event = event
        self.framework = framework
        self.generation = generation
        self.mode = mode
        self.node_id = node_id
        self.rank = rank
        self.replay_campaign = replay_campaign
        self.total_steps = total_steps
        self.artifact_dir = campaign_dir / f"{mode}-artifacts"
        self.losses: dict[str, float] = {}
        self.faults: list[Any] = []
        self.decisions: list[RecoveryDecision] = []
        self.fault_session: FaultInjectionSession | None = None

    def bind_fault_session(self, session: FaultInjectionSession) -> None:
        self.fault_session = session

    def record_fault(self, result: Any) -> None:
        self.faults.append(result)
        self.replay_campaign.record_fault(result)

    @property
    def orchestration(self) -> OrchestrationHooks:
        return OrchestrationHooks(report_recovery=self.decisions.append)

    def initialize(self, driver: FrameworkDriver) -> None:
        self.replay_campaign.bind(driver.handle)
        if self.generation == 0:
            _assert_all(
                driver.handle.step_count == 0,
                "fresh worker recovered unexpectedly",
                driver.device,
            )
            return
        if self.context is None or self.context.checkpoint_step is None:
            raise AssertionError("successor generation requires a recovery context")
        expected = read_json(
            self.artifact_dir
            / "checkpoints"
            / f"step-{self.context.checkpoint_step}-rank-{self.rank}.json"
        )
        state = driver.verification_state()
        topology_digest = checkpoint_topology_digest(driver.handle)
        recovered = (
            driver.handle.step_count == self.context.checkpoint_step
            and topology_digest == self.context.topology_digest
            and tensor_digest([*state["model"], *state["optimizer"]]) == expected["state_digest"]
            and rng_digest(driver.device) == expected["rng_digest"]
            and driver.framework_state() == expected["framework_state"]
        )
        _assert_all(recovered, "GEMINI fresh-process recovery was not exact", driver.device)
        atomic_json(
            self.artifact_dir / f"recovery-g{self.generation}-r{self.rank}.json",
            {
                "checkpoint_step": self.context.checkpoint_step,
                "framework": self.framework,
                "generation": self.generation,
                "logical_node_slot": self.context.logical_node_slot,
                "node_id": self.node_id,
                "rank": self.rank,
                "recovered_exact": recovered,
                "topology_digest": topology_digest,
            },
        )

    def before_step(self, step: int) -> None:
        self.replay_campaign.start_step(step)

    def after_step(
        self,
        driver: FrameworkDriver,
        step: int,
        loss: float | None,
    ) -> None:
        if loss is not None:
            self.losses[str(step)] = loss
        checkpoint_manager = _checkpoint_manager(driver.handle)
        if checkpoint_manager is None:
            raise AssertionError("GEMINI did not create a checkpoint manager")
        checkpoint_manager.maybe_wait()
        injection = self._collect_injection(step)
        if checkpoint_manager.checkpoint_status.recovery_verified_step == step:
            atomic_json(
                self.artifact_dir / "checkpoints" / f"step-{step}-rank-{self.rank}.json",
                self._snapshot(driver, step),
            )
        if self.event is not None and self.event.kind == "restart" and step == self.event.step:
            restart_checkpoint = driver.handle.flush_for_restart()
            _assert_all(
                restart_checkpoint == self.event.checkpoint_step,
                "restart-only failure did not flush the latest verified checkpoint",
                driver.device,
            )
            self._write_incident(
                "restart",
                checkpoint_step=self.event.checkpoint_step,
                topology_digest=checkpoint_topology_digest(driver.handle),
                extra={"injection": injection},
            )
            self._wait_for_restart()
        if self.faults:
            self._validate_and_report_fault(driver, checkpoint_manager, injection)
            self._wait_for_restart()
        if (
            self.event is not None
            and self.event.kind == "replacement"
            and not self.event.scout_localized
            and step == self.event.step
        ):
            self._validate_and_report_isolated_replacement(driver, injection)
            self._wait_for_restart()

    def finish(self, driver: FrameworkDriver) -> None:
        checkpoint_manager = _checkpoint_manager(driver.handle)
        replay_harness = _replay_harness(driver.handle)
        if checkpoint_manager is None or replay_harness is None:
            raise AssertionError("GEMINI and SCOUT must both be enabled")
        checkpoint_manager.maybe_wait()
        self.replay_campaign.validate(
            driver.handle,
            replay_harness.last_result,
            driver.expected_recipes,
        )
        final = self._snapshot(driver, self.total_steps)
        final.update(
            {
                "framework": self.framework,
                "generation": self.generation,
                "losses": self.losses,
                "node_id": self.node_id,
                "rank": self.rank,
                "topology_digest": checkpoint_topology_digest(driver.handle),
            }
        )
        prefix = "baseline" if self.mode == "baseline" else f"final-g{self.generation}"
        atomic_torch(
            self.artifact_dir / f"{prefix}-state-r{self.rank}.pt",
            driver.verification_state(),
        )
        atomic_json(self.artifact_dir / f"{prefix}-r{self.rank}.json", final)
        dist.barrier()

    def close(self) -> None:
        close_resources(
            *(
                (("fault injection session", self.fault_session.close),)
                if self.fault_session is not None
                else ()
            ),
            ("replay fault campaign", self.replay_campaign.close),
        )

    def _snapshot(self, driver: FrameworkDriver, step: int) -> dict[str, Any]:
        state = driver.verification_state()
        return {
            "framework_state": driver.framework_state(),
            "model_digest": tensor_digest(state["model"]),
            "optimizer_digest": tensor_digest(state["optimizer"]),
            "rng_digest": rng_digest(driver.device),
            "state_digest": tensor_digest([*state["model"], *state["optimizer"]]),
            "step": step,
        }

    def _validate_and_report_fault(
        self,
        driver: FrameworkDriver,
        checkpoint_manager: Any,
        injection: Mapping[str, Any],
    ) -> None:
        if len(self.faults) != 1 or len(self.decisions) != 1:
            raise AssertionError("SCOUT emitted an unexpected fault/decision count")
        fault = self.faults[0]
        if self.event is None or self.event.kind != "replacement" or self.event.fault_rank is None:
            replay_harness = _replay_harness(driver.handle)
            replay_config = getattr(replay_harness, "_config", None)
            runtime_topology = {
                "compare_updated_weights": getattr(
                    driver.handle,
                    "_compare_updated_weights",
                    None,
                ),
                "has_fsdp": getattr(driver.handle, "_has_fsdp", None),
                "is_hsdp": getattr(driver.handle, "_is_hsdp", None),
                "optimizer_interval": (
                    None if replay_config is None else replay_config.optimizer_check_interval
                ),
            }
            c3_statuses = {
                name: {
                    "bitmap": list(result.bitmap),
                    "evidence": list(result.evidence),
                    "status": result.status.value,
                }
                for name, result in fault.c3_results.items()
            }
            raise AssertionError(
                "SCOUT reported an unscheduled replacement fault: "
                f"sdc_bitmap={fault.sdc_bitmap}, sdc_sources={fault.sdc_sources}, "
                f"c3_results={c3_statuses}, runtime_topology={runtime_topology}"
            )
        decision = self.decisions[0]
        expected_bitmap = [
            int(candidate == self.event.fault_rank) for candidate in fault.peer_ranks
        ]
        expected_step = self.event.checkpoint_step
        localized = (
            fault.peer_ranks == list(range(driver.world_size))
            and fault.sdc_bitmap == expected_bitmap
            and not any(fault.straggler_bitmap)
            and any(source.startswith("hidden.") for source in fault.sdc_sources)
        )
        selected = (
            decision["failure_kind"] == "sdc"
            and decision["recovery_mode"] == "recovery_verified"
            and decision["checkpoint_source"] == "gemini"
            and decision["checkpoint_step"] == expected_step
            and decision["checkpoint_id"] is None
            and decision["available"]
        )
        checkpoint_safe = checkpoint_is_safe_for_replacement(
            checkpoint_manager,
            driver.handle,
            expected_step=expected_step,
        )
        _assert_all(localized, "SCOUT did not localize the injected rank", driver.device)
        _assert_all(selected, "SCOUT selected the wrong recovery checkpoint", driver.device)
        _assert_all(
            checkpoint_safe,
            "GEMINI exposed or flushed the contaminated checkpoint",
            driver.device,
        )
        self._write_incident(
            "fault",
            checkpoint_step=expected_step,
            topology_digest=checkpoint_topology_digest(driver.handle),
            extra={
                "decision": decision,
                "peer_ranks": list(fault.peer_ranks),
                "sdc_bitmap": list(fault.sdc_bitmap),
                "sdc_sources": list(fault.sdc_sources),
                "injection": injection,
            },
        )

    def _validate_and_report_isolated_replacement(
        self,
        driver: FrameworkDriver,
        injection: Mapping[str, Any],
    ) -> None:
        if self.event is None or self.event.fault_rank is None:
            raise AssertionError("isolated replacement requires a target rank")
        if self.faults or self.decisions:
            raise AssertionError(
                "isolated replacement unexpectedly produced a SCOUT recovery decision"
            )
        checkpoint_manager = _checkpoint_manager(driver.handle)
        if checkpoint_manager is None:
            raise AssertionError("GEMINI did not create a checkpoint manager")
        status = checkpoint_manager.checkpoint_status
        disk = getattr(checkpoint_manager, "_disk", None)
        selected_available = (
            status.is_recovery_verified(self.event.checkpoint_step)
            and disk is not None
            and disk.has_rank(self.event.checkpoint_step, driver.rank)
        )
        _assert_all(
            selected_available,
            "isolated replacement did not preserve the selected clean checkpoint",
            driver.device,
        )
        self._write_incident(
            "fault",
            checkpoint_step=self.event.checkpoint_step,
            topology_digest=checkpoint_topology_digest(driver.handle),
            extra={
                "failure_type": self.event.failure_type.value,
                "injection": injection,
                "replacement_rank": self.event.fault_rank,
            },
        )

    def _collect_injection(self, step: int) -> dict[str, Any]:
        if self.event is None or step != self.event.step:
            return {}
        if self.event.injection_executor is None:
            return {}
        if self.fault_session is None:
            raise AssertionError("campaign incident requires a fault injection session")
        local_records = [
            record.to_dict()
            for record in self.fault_session.records
            if record.incident_id == self.event.incident_id and record.iteration == step
        ]
        gathered: list[list[dict[str, Any]] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, local_records)
        records = [
            record
            for rank_records in gathered
            if rank_records is not None
            for record in rank_records
        ]
        if len(records) != 1:
            raise AssertionError(
                f"expected one verified injection record for {self.event.incident_id}, "
                f"got {len(records)}"
            )
        record = records[0]
        if (
            record["failure_type"] != self.event.failure_type.value
            or record["status"] != InjectionStatus.COMPLETED.value
            or not record["verified"]
            or not record["injection_succeeded"]
        ):
            raise AssertionError(f"fault injection evidence is invalid: {record}")
        return record

    def _write_incident(
        self,
        kind: str,
        *,
        checkpoint_step: int,
        topology_digest: str,
        extra: Mapping[str, Any] | None = None,
    ) -> None:
        if self.event is None:
            raise AssertionError("incident report requires an active event")
        report = {
            "checkpoint_step": checkpoint_step,
            "failure_type": self.event.failure_type.value,
            "framework": self.framework,
            "generation": self.generation,
            "incident_id": self.event.incident_id,
            "node_id": self.node_id,
            "rank": self.rank,
            "topology_digest": topology_digest,
        }
        report.update(extra or {})
        atomic_json(
            self.artifact_dir / f"{kind}-g{self.generation}-r{self.rank}.json",
            report,
        )

        atomic_json(
            self.artifact_dir / f"losses-g{self.generation}-r{self.rank}.json",
            {
                "framework": self.framework,
                "generation": self.generation,
                "losses": self.losses,
                "rank": self.rank,
            },
        )

    @staticmethod
    def _wait_for_restart() -> None:
        while True:
            time.sleep(1)


def _checkpoint_manager(handle: Any) -> Any | None:
    manager = getattr(handle, "ckpt_manager", None)
    return manager if manager is not None else getattr(handle, "_ckpt_manager", None)


def _replay_harness(handle: Any) -> Any | None:
    harness = getattr(handle, "replay_harness", None)
    return harness if harness is not None else getattr(handle, "_replay_harness", None)


def _assert_all(condition: bool, message: str, device: torch.device) -> None:
    value = torch.tensor([int(condition)], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if not value.item():
        raise AssertionError(message)
