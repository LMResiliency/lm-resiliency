"""Campaign and GPU-node topology for torchrun pressure validation."""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from lm_resiliency import (
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
)

from .artifacts import atomic_json

DEFAULT_ACTIVE_NODES = 8
DEFAULT_STANDBY_NODES = 8
RESTARTS_PER_REPLACEMENT = 2
CAMPAIGN_FILENAME = "campaign.json"
STATE_FILENAME = "state.json"
SINGLE_NODE_REPLACEMENTS = {
    FailureType.TENSOR_CORRUPTION: 0,
    FailureType.RESOURCE_UNAVAILABLE: 1,
    FailureType.COLLECTIVE_DESYNC: 2,
    FailureType.NETWORK_PARTITION: 3,
}
SINGLE_NODE_FAILURE_ORDER = (
    FailureType.HANG,
    FailureType.TENSOR_CORRUPTION,
    FailureType.STALE_STATE,
    FailureType.DROP,
    FailureType.DUPLICATE,
    FailureType.REORDER,
    FailureType.DELAY,
    FailureType.TIMEOUT,
    FailureType.RESOURCE_UNAVAILABLE,
    FailureType.EXCEPTION,
    FailureType.RESOURCE_EXHAUSTION,
    FailureType.PROCESS_TERMINATION,
    FailureType.CHECKPOINT_CORRUPTION,
    FailureType.CHECKPOINT_TRUNCATION,
    FailureType.CHECKPOINT_MISSING,
    FailureType.IO_ERROR,
    FailureType.PAYLOAD_CORRUPTION,
    FailureType.COLLECTIVE_DESYNC,
    FailureType.MESSAGE_DROP,
    FailureType.NETWORK_PARTITION,
    FailureType.CONFIG_DRIFT,
)


@dataclass(frozen=True, slots=True)
class PressureEvent:
    """One manager action derived from a fault-campaign incident."""

    incident_id: str
    kind: str
    step: int
    fault_rank: int | None
    failure_type: FailureType
    scout_localized: bool = False
    injection_executor: str | None = None
    selected_checkpoint_step: int | None = None

    @property
    def checkpoint_step(self) -> int:
        if self.selected_checkpoint_step is not None:
            return self.selected_checkpoint_step
        return self.step if self.kind == "restart" else self.step - 1


@dataclass(frozen=True, slots=True)
class GpuNodePlacement:
    """One validation GPU modeled as an independent torchrun node."""

    node_label: str
    gpu_id: str
    remote: bool


@dataclass(frozen=True, slots=True)
class PressureTopology:
    """Validated active/standby layout for one campaign."""

    placements: tuple[GpuNodePlacement, ...]
    world_size: int
    replication_jump: int


def default_pressure_campaign() -> FaultCampaign:
    incidents: list[FaultIncident] = []
    for replacement_index in range(DEFAULT_STANDBY_NODES):
        base_step = replacement_index * (RESTARTS_PER_REPLACEMENT + 1)
        for restart_index in range(RESTARTS_PER_REPLACEMENT):
            step = base_step + restart_index + 1
            incident_id = f"restart-{replacement_index + 1:02d}-{restart_index + 1}"
            incidents.append(
                FaultIncident(
                    incident_id=incident_id,
                    trigger=IncidentTrigger(at=(step,)),
                    lifetime=IncidentLifetime(matching_calls=1),
                    faults=(
                        FaultSpec(
                            fault_id=f"{incident_id}-process-stall",
                            type=FailureType.HANG,
                            target=FaultTarget(
                                rank=0,
                                surface=FaultSurface.PROCESS,
                                operation="manager_restart",
                            ),
                        ),
                    ),
                )
            )
        step = base_step + RESTARTS_PER_REPLACEMENT + 1
        incident_id = f"replacement-{replacement_index + 1:02d}"
        incidents.append(
            FaultIncident(
                incident_id=incident_id,
                trigger=IncidentTrigger(at=(step,)),
                lifetime=IncidentLifetime(matching_calls=1),
                faults=(
                    FaultSpec(
                        fault_id=f"{incident_id}-replay-sdc",
                        type=FailureType.TENSOR_CORRUPTION,
                        target=FaultTarget(
                            rank=replacement_index,
                            component="transformer_block",
                            index=0,
                            surface=FaultSurface.OUTPUT,
                            metadata={"injection_mode": "scout_replay_only"},
                        ),
                        parameters={"operation": "sign_flip", "scope": "100%"},
                    ),
                ),
            )
        )
    return FaultCampaign(
        name="torchrun-pressure-8-active-8-standby",
        seed=17,
        incidents=tuple(incidents),
        metadata={
            "active_nodes": DEFAULT_ACTIVE_NODES,
            "profile": "torchrun_pressure",
            "standby_nodes": DEFAULT_STANDBY_NODES,
            "total_steps": len(incidents) + 1,
        },
    )


def single_node_pressure_campaign() -> FaultCampaign:
    """Return the all-failure-types profile for four active and four standby GPUs."""

    incidents: list[FaultIncident] = []
    for index, failure_type in enumerate(SINGLE_NODE_FAILURE_ORDER):
        step = index * 2 + 1
        replacement_rank = SINGLE_NODE_REPLACEMENTS.get(failure_type)
        recovery_action = "replacement" if replacement_rank is not None else "restart"
        target_rank = replacement_rank if replacement_rank is not None else (step - 1) % 4
        parameters: dict[str, object] = {}
        if failure_type is FailureType.TENSOR_CORRUPTION:
            parameters = {"operation": "sign_flip", "scope": "100%"}
        elif failure_type is FailureType.DELAY:
            parameters = {"delay_ms": 5.0}
        target_metadata = {
            "executor": "isolated_validation",
            "recovery_action": recovery_action,
        }
        if failure_type is FailureType.TENSOR_CORRUPTION:
            target_metadata["injection_mode"] = "scout_replay_only"
        if replacement_rank is not None:
            target_metadata["checkpoint_step"] = (
                step - 1 if failure_type is FailureType.TENSOR_CORRUPTION else step - 2
            )
        incident_id = f"{step:02d}-{failure_type.value}"
        incidents.append(
            FaultIncident(
                incident_id=incident_id,
                trigger=IncidentTrigger(at=(step,)),
                lifetime=IncidentLifetime(matching_calls=1),
                faults=(
                    FaultSpec(
                        fault_id=f"{incident_id}-fault",
                        type=failure_type,
                        target=FaultTarget(
                            rank=target_rank,
                            surface=FaultSurface.RESOURCE,
                            operation=(
                                "manager_restart" if failure_type is FailureType.HANG else None
                            ),
                            resource=f"resiliency-cycle:{failure_type.value}",
                            metadata=target_metadata,
                        ),
                        parameters=parameters,
                    ),
                ),
            )
        )
    return FaultCampaign(
        name="torchrun-resiliency-cycle-all-failure-types",
        seed=29,
        incidents=tuple(incidents),
        metadata={
            "active_nodes": 4,
            "profile": "torchrun_pressure",
            "standby_nodes": 4,
            "total_steps": incidents[-1].trigger.at[0] + 1,
        },
    )


def pressure_events(campaign: FaultCampaign) -> tuple[PressureEvent, ...]:
    metadata = dict(campaign.metadata)
    if metadata.get("profile") != "torchrun_pressure":
        raise ValueError("fault campaign metadata.profile must be 'torchrun_pressure'")
    events: list[PressureEvent] = []
    for incident in campaign.incidents:
        if (
            len(incident.trigger.at) != 1
            or incident.trigger.range is not None
            or incident.trigger.probability != 1.0
            or len(incident.faults) != 1
        ):
            raise ValueError("pressure incidents require one deterministic trigger and fault")
        fault = incident.faults[0]
        recovery_action = fault.target.metadata.get("recovery_action")
        scout_localized = fault.target.metadata.get("injection_mode") == "scout_replay_only"
        injection_executor = fault.target.metadata.get("executor")
        selected_checkpoint_step = fault.target.metadata.get("checkpoint_step")
        if recovery_action is not None:
            if recovery_action not in {"restart", "replacement"}:
                raise ValueError("fault target metadata.recovery_action is invalid")
            if fault.target.metadata.get("executor") != "isolated_validation":
                raise ValueError(
                    "multi-type pressure incidents require the isolated validation executor"
                )
            kind = str(recovery_action)
            fault_rank = fault.target.rank
            if fault_rank is None:
                raise ValueError("pressure incidents require an explicit target rank")
            if kind == "replacement" and incident.trigger.at[0] < 2:
                raise ValueError(
                    "replacement incidents must follow at least one clean checkpoint step"
                )
            if selected_checkpoint_step is not None and (
                isinstance(selected_checkpoint_step, bool)
                or not isinstance(selected_checkpoint_step, int)
                or selected_checkpoint_step < 1
                or selected_checkpoint_step >= incident.trigger.at[0]
            ):
                raise ValueError(
                    "fault target metadata.checkpoint_step must select an earlier "
                    "positive training step"
                )
            if scout_localized and (
                fault.type is not FailureType.TENSOR_CORRUPTION
                or fault.parameters.get("operation") != "sign_flip"
                or fault.parameters.get("scope") != "100%"
            ):
                raise ValueError(
                    "SCOUT replay incidents require tensor sign_flip over 100% "
                    "of the selected tensor"
                )
        elif fault.type is FailureType.HANG and fault.target.surface is FaultSurface.PROCESS:
            if fault.target.operation != "manager_restart":
                raise ValueError("restart incidents require operation='manager_restart'")
            kind = "restart"
            fault_rank = None
        elif (
            fault.type is FailureType.TENSOR_CORRUPTION
            and fault.target.surface is FaultSurface.OUTPUT
            and fault.target.rank is not None
        ):
            if (
                fault.target.component != "transformer_block"
                or fault.target.index != 0
                or fault.target.metadata.get("injection_mode") != "scout_replay_only"
                or fault.parameters.get("operation") != "sign_flip"
                or fault.parameters.get("scope") != "100%"
            ):
                raise ValueError(
                    "replacement incidents require the supported replay-only "
                    "transformer-block sign_flip over 100% of the selected tensor"
                )
            if incident.trigger.at[0] < 2:
                raise ValueError(
                    "replacement incidents must follow at least one clean checkpoint step"
                )
            kind = "replacement"
            fault_rank = fault.target.rank
            scout_localized = True
        else:
            raise ValueError(f"unsupported pressure incident {incident.incident_id!r}")
        events.append(
            PressureEvent(
                incident_id=incident.incident_id,
                kind=kind,
                step=incident.trigger.at[0],
                fault_rank=fault_rank,
                failure_type=fault.type,
                scout_localized=scout_localized,
                injection_executor=(
                    str(injection_executor) if injection_executor is not None else None
                ),
                selected_checkpoint_step=selected_checkpoint_step,
            )
        )
    events.sort(key=lambda event: event.step)
    if len({event.step for event in events}) != len(events):
        raise ValueError("pressure incident steps must be unique")
    for previous, current in zip(events, events[1:]):
        if (
            previous.injection_executor is not None
            and current.injection_executor is not None
            and current.step - previous.step < 2
        ):
            raise ValueError(
                "public fault-injection incidents require one clean iteration "
                "between scheduled steps"
            )
    return tuple(events)


def load_campaign_bundle(path: Path) -> tuple[FaultCampaign, tuple[PressureEvent, ...]]:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    manifest_path = path / CAMPAIGN_FILENAME
    if manifest_path.exists():
        campaign = FaultCampaign.from_json(manifest_path)
    else:
        campaign = default_pressure_campaign()
        atomic_json(manifest_path, campaign.to_dict())
    return campaign, pressure_events(campaign)


def require_fresh_campaign_run(path: Path) -> None:
    """Reject bundle reuse so stale evidence cannot satisfy controller waits."""

    unexpected = sorted(entry.name for entry in path.iterdir() if entry.name != CAMPAIGN_FILENAME)
    if unexpected:
        raise ValueError(
            "fault campaign directory must be fresh and contain only campaign.json; "
            f"found stale or unrelated entries: {unexpected!r}"
        )


def campaign_layout(
    *,
    gpus: str,
    remote_gpus: str,
    remote_enabled: bool,
    campaign: FaultCampaign,
    events: Sequence[PressureEvent],
) -> PressureTopology:
    local_gpu_ids = _gpu_ids(gpus, "--gpus")
    remote_gpu_ids = _gpu_ids(remote_gpus, "--remote-gpus") if remote_enabled else ()
    placements = tuple(
        [
            *(
                GpuNodePlacement(f"local-gpu-{index:02d}", gpu_id, False)
                for index, gpu_id in enumerate(local_gpu_ids)
            ),
            *(
                GpuNodePlacement(f"remote-gpu-{index:02d}", gpu_id, True)
                for index, gpu_id in enumerate(remote_gpu_ids)
            ),
        ]
    )
    active_nodes = campaign.metadata.get("active_nodes")
    standby_nodes = campaign.metadata.get("standby_nodes")
    if (
        isinstance(active_nodes, bool)
        or not isinstance(active_nodes, int)
        or active_nodes < 4
        or active_nodes % 2
    ):
        raise ValueError("campaign metadata.active_nodes must be an even integer of at least four")
    if isinstance(standby_nodes, bool) or not isinstance(standby_nodes, int) or standby_nodes < 1:
        raise ValueError("campaign metadata.standby_nodes must be a positive integer")
    replacement_events = [event for event in events if event.kind == "replacement"]
    if len(replacement_events) != standby_nodes:
        raise ValueError("campaign must contain one replacement event per standby")
    replacement_ranks = [event.fault_rank for event in replacement_events]
    if len(set(replacement_ranks)) != len(replacement_ranks):
        raise ValueError("replacement events must target distinct logical ranks")
    if any(rank is None or rank < 0 or rank >= active_nodes for rank in replacement_ranks):
        raise ValueError("replacement event target rank is outside the active world")
    if len(placements) != active_nodes + standby_nodes:
        raise ValueError(
            "supplied GPU-node count must equal campaign active_nodes plus standby_nodes"
        )
    replication_jump = active_nodes // 2
    return PressureTopology(
        placements=placements,
        world_size=active_nodes,
        replication_jump=replication_jump,
    )


def _gpu_ids(value: str, name: str) -> tuple[str, ...]:
    gpu_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpu_ids:
        raise ValueError(f"{name} must provide at least one GPU ID")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"{name} must not contain duplicate GPU IDs")
    return gpu_ids
