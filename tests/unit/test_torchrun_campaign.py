"""Tests for the torchrun restart and replacement pressure campaign."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from examples.torchrun_resiliency.pressure import (
    CAMPAIGN_FILENAME,
    STATE_FILENAME,
    _campaign_layout,
    _checkpoint_is_safe_for_replacement,
    _default_pressure_campaign,
    _load_campaign_bundle,
    _pressure_events,
)


def _args(
    *,
    gpus: str,
    remote_gpus: str = "",
    remote_host: str | None = None,
) -> Namespace:
    return Namespace(
        gpus=gpus,
        remote_gpus=remote_gpus,
        remote_host=remote_host,
    )


def test_default_pressure_campaign_has_sixteen_restarts_and_eight_replacements() -> None:
    campaign = _default_pressure_campaign()
    events = _pressure_events(campaign)

    assert len(events) == 24
    assert [event.step for event in events] == list(range(1, 25))
    assert sum(event.kind == "restart" for event in events) == 16
    replacements = [event for event in events if event.kind == "replacement"]
    assert len(replacements) == 8
    assert [event.fault_rank for event in replacements] == list(range(8))
    assert campaign.metadata["total_steps"] == 25


def test_sixteen_gpu_layout_treats_every_gpu_as_one_node() -> None:
    campaign = _default_pressure_campaign()
    events = _pressure_events(campaign)
    placements, world_size, replication_jump, topology_digest = _campaign_layout(
        _args(
            gpus="0,1,2,3,4,5,6,7",
            remote_gpus="0,1,2,3,4,5,6,7",
            remote_host="remote",
        ),
        campaign,
        events,
    )

    assert len(placements) == 16
    assert len({label for label, _gpu, _remote in placements}) == 16
    assert sum(remote for _label, _gpu, remote in placements) == 8
    assert world_size == 8
    assert replication_jump == 4
    assert topology_digest == "ddp-world-8-replication-jump-4-gpu-nodes"


def test_campaign_bundle_creates_only_manifest_before_execution(tmp_path: Path) -> None:
    campaign, events = _load_campaign_bundle(tmp_path)

    assert campaign == _default_pressure_campaign()
    assert events == _pressure_events(campaign)
    assert (tmp_path / CAMPAIGN_FILENAME).is_file()
    assert not (tmp_path / STATE_FILENAME).exists()


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

    assert _checkpoint_is_safe_for_replacement(manager, handle, expected_step=2)


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
    campaign = _default_pressure_campaign()

    with pytest.raises(ValueError, match=message):
        _campaign_layout(
            _args(gpus=gpus, remote_gpus=remote_gpus, remote_host="remote"),
            campaign,
            _pressure_events(campaign),
        )
