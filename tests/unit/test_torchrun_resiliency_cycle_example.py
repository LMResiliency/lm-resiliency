"""Tests for the torchrun resiliency-cycle example."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from examples.torchrun.resiliency_cycle.harness.campaign import (
    CAMPAIGN_FILENAME,
    STATE_FILENAME,
    campaign_layout,
    default_pressure_campaign,
    load_campaign_bundle,
    pressure_events,
)
from examples.torchrun.resiliency_cycle.harness.replay_fault import (
    ReplayFaultCampaign,
)
from examples.torchrun.resiliency_cycle.harness.runtime import (
    checkpoint_is_safe_for_replacement,
)
from examples.torchrun.resiliency_cycle.harness.verify import (
    loss_difference_limit,
    optimizer_difference_limit,
)
from examples.torchrun.resiliency_cycle.pressure import (
    _validate_replacement_coverage,
)


class _CheckpointManager:
    def __init__(self, step: int) -> None:
        self._last_saved_step = step
        self.checkpoint_status = SimpleNamespace(recovery_verified_step=step)


def _result(*, fault: bool) -> SimpleNamespace:
    return SimpleNamespace(
        peer_ranks=[0, 1],
        sdc_bitmap=[1, 0] if fault else [0, 0],
        straggler_bitmap=[0, 0],
        sdc_sources=["hidden.output"] if fault else [],
        checked_recipe_ids={"embedding", "hidden", "output", "optimizer"},
    )


def test_default_pressure_campaign_has_sixteen_restarts_and_eight_replacements() -> None:
    campaign = default_pressure_campaign()
    events = pressure_events(campaign)

    assert len(events) == 24
    assert [event.step for event in events] == list(range(1, 25))
    assert sum(event.kind == "restart" for event in events) == 16
    replacements = [event for event in events if event.kind == "replacement"]
    assert len(replacements) == 8
    assert [event.fault_rank for event in replacements] == list(range(8))
    assert campaign.metadata["total_steps"] == 25


def test_sixteen_gpu_layout_treats_every_gpu_as_one_node() -> None:
    campaign = default_pressure_campaign()
    events = pressure_events(campaign)
    topology = campaign_layout(
        gpus="0,1,2,3,4,5,6,7",
        remote_gpus="0,1,2,3,4,5,6,7",
        remote_enabled=True,
        campaign=campaign,
        events=events,
    )

    assert len(topology.placements) == 16
    assert len({placement.node_label for placement in topology.placements}) == 16
    assert sum(placement.remote for placement in topology.placements) == 8
    assert topology.world_size == 8
    assert topology.replication_jump == 4
    assert topology.topology_digest == "ddp-world-8-replication-jump-4-gpu-nodes"


def test_campaign_bundle_creates_only_manifest_before_execution(tmp_path: Path) -> None:
    campaign, events = load_campaign_bundle(tmp_path)

    assert campaign == default_pressure_campaign()
    assert events == pressure_events(campaign)
    assert (tmp_path / CAMPAIGN_FILENAME).is_file()
    assert not (tmp_path / STATE_FILENAME).exists()


def test_fault_campaign_corrupts_replay_but_not_training_forward() -> None:
    target = nn.Identity()
    manager = _CheckpointManager(step=1)
    handle = SimpleNamespace(
        replay_harness=SimpleNamespace(target_layer=target),
        ckpt_manager=manager,
    )
    campaign = ReplayFaultCampaign(
        steps=5,
        rank=0,
        world_size=2,
        inject_fault=True,
        fault_step=2,
        fault_rank=0,
    )
    campaign.bind(handle)
    campaign.start_step(2)

    value = torch.zeros(2)
    torch.testing.assert_close(target(value), value)
    torch.testing.assert_close(target(value), value + 1.0)

    campaign.record_fault(_result(fault=True))
    campaign.recovery_decisions.append(
        {
            "failure_kind": "sdc",
            "recovery_mode": "recovery_verified",
            "checkpoint_source": "gemini",
            "checkpoint_step": 1,
            "available": True,
        }
    )
    manager._last_saved_step = 5
    manager.checkpoint_status.recovery_verified_step = 5

    campaign.validate(
        handle,
        _result(fault=False),
        {"embedding", "hidden", "output", "optimizer"},
    )
    campaign.close()


def test_fault_campaign_requires_two_clean_post_fault_steps() -> None:
    args = SimpleNamespace(
        steps=5,
        inject_fault=True,
        fault_step=4,
        fault_rank=-1,
    )

    with pytest.raises(ValueError, match="post-fault"):
        ReplayFaultCampaign.from_args(args, rank=0, world_size=2)


@pytest.mark.parametrize("flushed_step", [-1, 2])
def test_replacement_accepts_verified_disk_checkpoint_after_process_restart(
    flushed_step: int,
) -> None:
    state = {"flushed": False}

    def flush_for_restart() -> int:
        state["flushed"] = True
        return flushed_step

    manager = SimpleNamespace(
        _last_saved_step=-1,
        checkpoint_status=SimpleNamespace(recovery_verified_step=2),
        local_recovery_step=lambda mode: (
            2 if state["flushed"] and mode == "recovery_verified" else -1
        ),
    )
    handle = SimpleNamespace(flush_for_restart=flush_for_restart)

    assert checkpoint_is_safe_for_replacement(manager, handle, expected_step=2)


@pytest.mark.parametrize(
    ("gpus", "remote_gpus", "message"),
    [
        ("0,0,1,2,3,4,5,6", "0,1,2,3,4,5,6,7", "duplicate GPU IDs"),
        ("0,1,2,3,4,5,6,7", "0,1,2,3,4,5,6", "must equal"),
    ],
)
def test_campaign_layout_rejects_invalid_gpu_fleets(
    gpus: str,
    remote_gpus: str,
    message: str,
) -> None:
    campaign = default_pressure_campaign()

    with pytest.raises(ValueError, match=message):
        campaign_layout(
            gpus=gpus,
            remote_gpus=remote_gpus,
            remote_enabled=True,
            campaign=campaign,
            events=pressure_events(campaign),
        )


def test_replacement_coverage_matches_campaign_incident_count() -> None:
    _validate_replacement_coverage(
        current_nodes=("standby-a", "node-b", "node-c", "node-d"),
        initial_nodes=("node-a", "node-b", "node-c", "node-d"),
        replacement_failures=1,
    )

    with pytest.raises(AssertionError, match="unexpected number"):
        _validate_replacement_coverage(
            current_nodes=("standby-a", "standby-b", "node-c", "node-d"),
            initial_nodes=("node-a", "node-b", "node-c", "node-d"),
            replacement_failures=1,
        )


def test_deepspeed_uses_bf16_continuation_tolerance() -> None:
    assert optimizer_difference_limit("deepspeed") == 1e-3
    assert optimizer_difference_limit("pytorch") == 5e-5


def test_torchtitan_uses_bf16_loss_continuation_tolerance() -> None:
    assert loss_difference_limit("torchtitan") == 1e-1
    assert loss_difference_limit("pytorch") == 1e-2
