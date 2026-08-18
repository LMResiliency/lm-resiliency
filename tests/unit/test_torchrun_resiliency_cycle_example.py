"""Tests for the torchrun resiliency-cycle example."""

from __future__ import annotations

import io
import shlex
import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
import torch.nn as nn

from examples.torchrun.resiliency_cycle.harness import launch
from examples.torchrun.resiliency_cycle.harness.campaign import (
    CAMPAIGN_FILENAME,
    STATE_FILENAME,
    GpuNodePlacement,
    PressureTopology,
    campaign_layout,
    default_pressure_campaign,
    load_campaign_bundle,
    pressure_events,
    require_fresh_campaign_run,
)
from examples.torchrun.resiliency_cycle.harness.frameworks.megatron import (
    MegatronDriver,
)
from examples.torchrun.resiliency_cycle.harness.launch import (
    LaunchedAgent,
    PressureLaunchOptions,
)
from examples.torchrun.resiliency_cycle.harness.replay_fault import (
    ReplayFaultCampaign,
)
from examples.torchrun.resiliency_cycle.harness.runtime import (
    DriverConfig,
    checkpoint_is_safe_for_replacement,
)
from examples.torchrun.resiliency_cycle.harness.verify import (
    checkpoint_topology_digest,
    loss_difference_limit,
    optimizer_difference_limit,
)
from examples.torchrun.resiliency_cycle.harness.worker import (
    _validate_worker_context,
)
from examples.torchrun.resiliency_cycle.pressure import (
    _cleanup_controller,
    _validate_replacement_coverage,
)
from lm_resiliency import FaultCampaign


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


def test_campaign_bundle_creates_only_manifest_before_execution(tmp_path: Path) -> None:
    campaign, events = load_campaign_bundle(tmp_path)

    assert campaign == default_pressure_campaign()
    assert events == pressure_events(campaign)
    assert (tmp_path / CAMPAIGN_FILENAME).is_file()
    assert not (tmp_path / STATE_FILENAME).exists()


def test_campaign_bundle_rejects_stale_run_artifacts(tmp_path: Path) -> None:
    load_campaign_bundle(tmp_path)
    (tmp_path / STATE_FILENAME).write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="must be fresh"):
        require_fresh_campaign_run(tmp_path)


def test_campaign_rejects_unsupported_replay_corruption() -> None:
    value = default_pressure_campaign().to_dict()
    replacement = next(
        incident
        for incident in value["incidents"]
        if incident["faults"][0]["type"] == "tensor_corruption"
    )
    replacement["faults"][0]["parameters"]["operation"] = "scale"

    with pytest.raises(ValueError, match="sign_flip"):
        pressure_events(FaultCampaign.from_dict(value))


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

    value = torch.tensor([1.0, -2.0])
    torch.testing.assert_close(target(value), value)
    torch.testing.assert_close(target(value), -value)

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


def test_driver_config_forwards_exact_recovery_constraints(tmp_path: Path) -> None:
    config = DriverConfig(
        campaign_dir=tmp_path,
        checkpoint=object(),  # type: ignore[arg-type]
        replay=object(),  # type: ignore[arg-type]
        recovery_mode="recovery_verified",
        recovery_step=7,
        expected_topology_id="checkpoint-topology",
        fault_callback=lambda _result: None,
        orchestration=object(),  # type: ignore[arg-type]
        total_steps=9,
    )

    assert config.recovery_options() == {
        "recovery_mode": "recovery_verified",
        "_recovery_step": 7,
        "_expected_topology_id": "checkpoint-topology",
    }


def test_generation_zero_context_does_not_require_recovery_source() -> None:
    _validate_worker_context(SimpleNamespace(local_world_size=1, generation=0))


def test_incident_reports_require_one_checkpoint_topology() -> None:
    reports = [
        {"topology_digest": "checkpoint-topology"},
        {"topology_digest": "checkpoint-topology"},
    ]

    assert checkpoint_topology_digest(reports) == "checkpoint-topology"
    reports[1]["topology_digest"] = "other-topology"
    with pytest.raises(AssertionError, match="disagree"):
        checkpoint_topology_digest(reports)


def test_gpu_environment_does_not_assume_network_interface() -> None:
    assert launch._gpu_environment("3") == {"CUDA_VISIBLE_DEVICES": "3"}


def test_remote_install_uses_selected_python(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[str]] = []

    def run(command: list[str], **_kwargs: Any) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", run)
    options = PressureLaunchOptions(
        fault_campaign_dir=tmp_path / "campaign",
        framework="pytorch",
        run_id="remote-install",
        timeout=10,
        remote_host="worker-b",
        remote_python="/opt/lm/bin/python",
        remote_source_dir=tmp_path / "remote-source",
    )

    launch.prepare_remote_source(options)

    assert shlex.split(calls[-1][-1])[:4] == [
        "/opt/lm/bin/python",
        "-m",
        "pip",
        "install",
    ]


def test_partial_managed_launch_is_cleaned_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = SimpleNamespace()
    first = LaunchedAgent(process=process, log=io.BytesIO())
    calls = 0
    cleaned: list[LaunchedAgent] = []

    def start(**_kwargs: Any) -> LaunchedAgent:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("launch failed")
        return first

    def cleanup(
        _options: PressureLaunchOptions,
        agents: list[LaunchedAgent],
        *,
        remote_run_id: str,
    ) -> None:
        assert remote_run_id == "partial-launch"
        cleaned.extend(agents)

    monkeypatch.setattr(launch, "_launch_managed_agent", start)
    monkeypatch.setattr(launch, "cleanup_agents", cleanup)
    options = PressureLaunchOptions(
        fault_campaign_dir=tmp_path,
        framework="pytorch",
        run_id="partial-launch",
        timeout=10,
    )
    topology = PressureTopology(
        placements=(
            GpuNodePlacement("node-a", "0", False),
            GpuNodePlacement("node-b", "1", False),
        ),
        world_size=1,
        replication_jump=1,
    )

    with pytest.raises(RuntimeError, match="launch failed"):
        launch.launch_managed_agents(
            options,
            topology,
            endpoint=None,
            max_restarts=1,
        )

    assert cleaned == [first]


def test_agent_cleanup_runs_when_coordinator_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[LaunchedAgent] = []
    options = PressureLaunchOptions(
        fault_campaign_dir=tmp_path,
        framework="pytorch",
        run_id="close-failure",
        timeout=10,
    )
    agent = LaunchedAgent(process=SimpleNamespace(), log=io.BytesIO())
    coordinator = SimpleNamespace(close=lambda: (_ for _ in ()).throw(RuntimeError("store failed")))

    def cleanup(
        _options: PressureLaunchOptions,
        agents: list[LaunchedAgent],
        *,
        remote_run_id: str,
    ) -> None:
        assert remote_run_id == "close-failure"
        cleaned.extend(agents)

    monkeypatch.setattr(
        "examples.torchrun.resiliency_cycle.pressure.cleanup_agents",
        cleanup,
    )

    with pytest.raises(RuntimeError, match="store failed"):
        _cleanup_controller(coordinator, options=options, agents=[agent])

    assert cleaned == [agent]


def test_megatron_extra_state_restores_sample_counters() -> None:
    loaded_scheduler: list[dict[str, int]] = []
    driver = MegatronDriver.__new__(MegatronDriver)
    driver.world_size = 4
    driver._extra_state = {"absolute_step": 2}
    driver.args = SimpleNamespace(
        iteration=1,
        consumed_train_samples=8,
        skipped_train_samples=3,
    )
    driver.scheduler = SimpleNamespace(
        num_steps=2,
        state_dict=lambda: {"num_steps": 2},
        load_state_dict=loaded_scheduler.append,
    )
    driver.handle = SimpleNamespace(step_count=2)

    saved = driver._extra_state_dict()
    assert saved["iteration"] == 2
    assert saved["consumed_train_samples"] == 16
    assert saved["skipped_train_samples"] == 3

    driver.args.iteration = 0
    driver.args.consumed_train_samples = 0
    driver.args.skipped_train_samples = 0
    driver._load_extra_state_dict(saved)

    assert driver.framework_state()["iteration"] == 2
    assert driver.framework_state()["consumed_train_samples"] == 16
    assert driver.framework_state()["skipped_train_samples"] == 3
    assert loaded_scheduler == [{"num_steps": 2}]


def test_deepspeed_uses_bf16_continuation_tolerance() -> None:
    assert optimizer_difference_limit("deepspeed") == 1e-3
    assert optimizer_difference_limit("pytorch") == 5e-5


def test_torchtitan_uses_bf16_loss_continuation_tolerance() -> None:
    assert loss_difference_limit("torchtitan") == 1e-1
    assert loss_difference_limit("pytorch") == 1e-2
