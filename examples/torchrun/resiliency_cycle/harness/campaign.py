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


@dataclass(frozen=True, slots=True)
class PressureEvent:
    """One manager action derived from a fault-campaign incident."""

    incident_id: str
    kind: str
    step: int
    fault_rank: int | None

    @property
    def checkpoint_step(self) -> int:
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
        if fault.type is FailureType.HANG and fault.target.surface is FaultSurface.PROCESS:
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
            kind = "replacement"
            fault_rank = fault.target.rank
        else:
            raise ValueError(f"unsupported pressure incident {incident.incident_id!r}")
        events.append(
            PressureEvent(
                incident_id=incident.incident_id,
                kind=kind,
                step=incident.trigger.at[0],
                fault_rank=fault_rank,
            )
        )
    events.sort(key=lambda event: event.step)
    if len({event.step for event in events}) != len(events):
        raise ValueError("pressure incident steps must be unique")
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
