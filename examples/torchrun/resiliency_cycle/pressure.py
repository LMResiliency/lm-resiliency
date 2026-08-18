"""Orchestrate repeated resiliency cycles through torchrun.

The outer process is a validation controller, not a training worker. It starts
an uninterrupted baseline, launches one torchrun agent per supplied GPU, acts
as the recovery manager, and compares the final managed run with the baseline.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from collections.abc import Sequence
from pathlib import Path

from torch.distributed import FileStore, Store

from lm_resiliency.integrations.torchrun import (
    TorchrunInitialPlacement,
    TorchrunRecoveryCoordinator,
    TorchrunRecoveryRequest,
    TorchrunSuccessorPlacement,
)

from .harness.artifacts import atomic_json, read_json, wait_for_paths
from .harness.campaign import (
    STATE_FILENAME,
    PressureEvent,
    campaign_layout,
    load_campaign_bundle,
    require_fresh_campaign_run,
)
from .harness.launch import (
    LaunchedAgent,
    PressureLaunchOptions,
    cleanup_agents,
    create_tcp_store,
    launch_baseline,
    launch_managed_agents,
    prepare_remote_source,
    synthetic_node_id,
)
from .harness.verify import (
    MODEL_MAX_ABS_DIFF,
    checkpoint_topology_digest,
    compare_baseline,
    loss_difference_limit,
    optimizer_difference_limit,
    validate_fault_reports,
    validate_restart_reports,
)


def _wait_for_initial_admission(
    coordinator: TorchrunRecoveryCoordinator,
    *,
    expected_nodes: frozenset[str],
    world_size: int,
    processes: Sequence[subprocess.Popen[bytes]],
    timeout: float,
) -> TorchrunInitialPlacement:
    deadline = time.monotonic() + timeout
    while True:
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"torchrun agent failed with exit codes {failed}")
        placement = coordinator.initial_placement(
            active_node_count=world_size,
            allocated_node_count=len(expected_nodes),
        )
        if placement is not None:
            observed = frozenset([*placement.active_node_ids, *placement.standby_node_ids])
            if observed != expected_nodes:
                raise AssertionError("registered machine identities do not match the fleet")
            return placement
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for automatic initial node admission")
        time.sleep(0.1)


def _publish_plan(
    *,
    coordinator: TorchrunRecoveryCoordinator,
    generation: int,
    current_nodes: Sequence[str],
    event: PressureEvent,
    replacement_node: str | None,
    quarantined: Sequence[str],
    topology_digest: str,
) -> TorchrunSuccessorPlacement:
    replacement = None
    if event.kind == "replacement":
        if event.fault_rank is None or replacement_node is None:
            raise AssertionError("replacement event requires a target rank and standby")
        replacement = (event.fault_rank, replacement_node)
        recovery_mode = "recovery_verified"
        reason_code = "sdc_detected"
    else:
        recovery_mode = "latest"
        reason_code = "process_stall"
    return coordinator.publish_successor(
        generation=generation,
        active_node_ids=current_nodes,
        quarantined_node_ids=quarantined,
        request=TorchrunRecoveryRequest(
            plan_id=f"pressure-{event.kind}-{generation + 1}",
            intent_id=event.incident_id,
            reason_code=reason_code,
            recovery_mode=recovery_mode,
            checkpoint_source="gemini",
            checkpoint_step=event.checkpoint_step,
            checkpoint_manifest_id=(f"gemini-{recovery_mode}-{event.checkpoint_step}"),
            topology_digest=topology_digest,
            restart_deadline_unix_ms=time.time_ns() // 1_000_000 + 120_000,
        ),
        local_world_size=1,
        replacement=replacement,
    )


def _validate_replacement_coverage(
    *,
    current_nodes: Sequence[str],
    initial_nodes: Sequence[str],
    replacement_failures: int,
) -> None:
    retained_initial = set(current_nodes) & set(initial_nodes)
    expected_retained = len(initial_nodes) - replacement_failures
    if len(retained_initial) != expected_retained:
        raise AssertionError("pressure campaign replaced an unexpected number of initial GPU-nodes")


def _cleanup_controller(
    coordinator: TorchrunRecoveryCoordinator | None,
    *,
    options: PressureLaunchOptions,
    agents: Sequence[LaunchedAgent],
) -> None:
    try:
        if coordinator is not None:
            coordinator.close()
    finally:
        cleanup_agents(options, agents, remote_run_id=options.run_id)


def _orchestrate(args: argparse.Namespace) -> None:
    campaign_dir = args.fault_campaign_dir.resolve()
    options = PressureLaunchOptions(
        fault_campaign_dir=campaign_dir,
        framework=args.framework,
        run_id=args.run_id,
        timeout=args.timeout,
        remote_host=args.remote_host,
        remote_python=args.remote_python,
        remote_source_dir=args.remote_source_dir,
        rendezvous_host=args.rdzv_host,
    )
    campaign, events = load_campaign_bundle(campaign_dir)
    require_fresh_campaign_run(campaign_dir)
    topology = campaign_layout(
        gpus=args.gpus,
        remote_gpus=args.remote_gpus,
        remote_enabled=options.remote_enabled,
        campaign=campaign,
        events=events,
    )
    standby_count = int(campaign.metadata["standby_nodes"])
    total_steps = int(campaign.metadata["total_steps"])
    if total_steps <= events[-1].step:
        raise ValueError("campaign metadata.total_steps must include a clean final step")
    if options.remote_enabled:
        if not options.remote_python:
            raise ValueError("--remote-python is required with --remote-host")
        if options.remote_source_dir is None:
            raise ValueError("--remote-source-dir is required with --remote-host")
        if not options.rendezvous_host:
            raise ValueError("--rdzv-host is required with --remote-host")
        prepare_remote_source(options)
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                options.remote_host,
                "test",
                "-d",
                str(campaign_dir.parent),
            ],
            check=True,
        )
    campaign_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(campaign_dir, 0o700)
    for name in (
        "baseline-artifacts",
        "campaign-artifacts",
    ):
        (campaign_dir / name).mkdir(parents=True, exist_ok=True)

    launch_baseline(options, topology)

    control_store: Store
    endpoint: str | None
    if options.remote_enabled:
        assert options.rendezvous_host is not None
        tcp_store = create_tcp_store(options.rendezvous_host, options.timeout)
        control_store = tcp_store
        endpoint = f"{tcp_store.host}:{tcp_store.port}"
    else:
        endpoint = None
        control_store = FileStore(str(campaign_dir / "rdzv"))
    agents = launch_managed_agents(
        options,
        topology,
        endpoint=endpoint,
        max_restarts=len(events),
    )
    processes = [agent.process for agent in agents]
    artifacts = campaign_dir / "campaign-artifacts"
    expected_nodes = frozenset(
        synthetic_node_id(placement.node_label) for placement in topology.placements
    )
    coordinator: TorchrunRecoveryCoordinator | None = None
    try:
        coordinator = TorchrunRecoveryCoordinator(control_store, run_id=options.run_id)
        initial = _wait_for_initial_admission(
            coordinator,
            expected_nodes=expected_nodes,
            world_size=topology.world_size,
            processes=processes,
            timeout=options.timeout,
        )
        if len(initial.standby_node_ids) != standby_count:
            raise AssertionError("initial admission selected the wrong standby count")
        current_nodes = list(initial.active_node_ids)
        replacements = iter(initial.standby_node_ids)
        quarantined: list[str] = []
        completed_events: list[str] = []
        observed_topology_digest: str | None = None
        for generation, event in enumerate(events):
            report_prefix = "fault" if event.kind == "replacement" else "restart"
            report_paths = [
                artifacts / f"{report_prefix}-g{generation}-r{rank}.json"
                for rank in range(topology.world_size)
            ]
            wait_for_paths(report_paths, processes=processes, timeout=options.timeout)
            reports = [read_json(path) for path in report_paths]
            topology_digest = checkpoint_topology_digest(reports)
            if observed_topology_digest is not None and topology_digest != observed_topology_digest:
                raise AssertionError("GEMINI checkpoint topology changed across generations")
            observed_topology_digest = topology_digest
            replacement_node = None
            if event.kind == "replacement":
                assert event.fault_rank is not None
                validate_fault_reports(
                    reports,
                    expected_generation=generation,
                    expected_checkpoint_step=event.checkpoint_step,
                    fault_rank=event.fault_rank,
                    world_size=topology.world_size,
                )
                replacement_node = next(replacements)
            else:
                validate_restart_reports(
                    reports,
                    event=event,
                    expected_generation=generation,
                )
            successor = _publish_plan(
                coordinator=coordinator,
                generation=generation,
                current_nodes=current_nodes,
                event=event,
                replacement_node=replacement_node,
                quarantined=quarantined,
                topology_digest=topology_digest,
            )
            current_nodes = list(successor.active_node_ids)
            quarantined = list(successor.quarantined_node_ids)
            completed_events.append(event.incident_id)
            atomic_json(
                campaign_dir / STATE_FILENAME,
                {
                    "campaign": campaign.name,
                    "completed_incidents": completed_events,
                    "generation": successor.generation,
                    "manifest_identity": campaign.manifest_identity,
                    "quarantined_nodes": quarantined,
                },
            )
            recovery_paths = [
                artifacts / f"recovery-g{successor.generation}-r{rank}.json"
                for rank in range(topology.world_size)
            ]
            wait_for_paths(recovery_paths, processes=processes, timeout=options.timeout)
            recovery = [read_json(path) for path in recovery_paths]
            if not all(item["recovered_exact"] for item in recovery):
                raise AssertionError("GEMINI recovery was not exact on every rank")
            if checkpoint_topology_digest(recovery) != topology_digest:
                raise AssertionError("successor workers recovered a different topology")
            if event.kind == "replacement":
                assert event.fault_rank is not None and replacement_node is not None
                if recovery[event.fault_rank]["node_id"] != replacement_node:
                    raise AssertionError("selected standby did not inherit the faulty logical rank")
                if recovery[event.fault_rank]["logical_node_slot"] != event.fault_rank:
                    raise AssertionError("selected standby did not inherit the faulty logical slot")

        final_paths = [
            artifacts / f"final-g{len(events)}-r{rank}.json" for rank in range(topology.world_size)
        ]
        wait_for_paths(final_paths, processes=processes, timeout=options.timeout)
        loss_diff, model_diff, optimizer_diff = compare_baseline(
            campaign_dir,
            generations=len(events) + 1,
            world_size=topology.world_size,
        )
        replacement_failures = sum(event.kind == "replacement" for event in events)
        _validate_replacement_coverage(
            current_nodes=current_nodes,
            initial_nodes=initial.active_node_ids,
            replacement_failures=replacement_failures,
        )
        coordinator.close()
        coordinator = None
        for process in processes:
            process.wait(timeout=options.timeout)
            if process.returncode != 0:
                raise RuntimeError(f"torchrun agent failed with exit code {process.returncode}")
        atomic_json(
            campaign_dir / "summary.json",
            {
                "campaign": campaign.name,
                "campaign_manifest_identity": campaign.manifest_identity,
                "checkpoint_topology_digest": observed_topology_digest,
                "framework": options.framework,
                "final_nodes": current_nodes,
                "final_step": total_steps,
                "gpu_nodes": [
                    {
                        "gpu_id": placement.gpu_id,
                        "location": "remote" if placement.remote else "local",
                        "node_id": synthetic_node_id(placement.node_label),
                        "node_label": placement.node_label,
                    }
                    for placement in topology.placements
                ],
                "initial_nodes": list(initial.active_node_ids),
                "localization": "exact",
                "replacement_failures": replacement_failures,
                "replication_jump": topology.replication_jump,
                "recoveries": "bitwise exact",
                "restart_only_failures": sum(event.kind == "restart" for event in events),
                "standbys": list(initial.standby_node_ids),
                "total_failures": len(events),
                "world_size": topology.world_size,
                "training_vs_baseline": {
                    "loss_max_abs_diff": loss_diff,
                    "loss_max_abs_diff_limit": loss_difference_limit(options.framework),
                    "model_max_abs_diff": model_diff,
                    "model_max_abs_diff_limit": MODEL_MAX_ABS_DIFF,
                    "optimizer_max_abs_diff": optimizer_diff,
                    "optimizer_max_abs_diff_limit": optimizer_difference_limit(options.framework),
                    "rng": "bitwise exact",
                },
            },
        )
    finally:
        _cleanup_controller(coordinator, options=options, agents=agents)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument("--fault-campaign-dir", type=Path, required=True)
    orchestrate.add_argument(
        "--framework",
        choices=("pytorch", "deepspeed", "megatron", "torchtitan"),
        required=True,
    )
    orchestrate.add_argument("--gpus", required=True)
    orchestrate.add_argument("--remote-gpus", default="")
    orchestrate.add_argument("--remote-host")
    orchestrate.add_argument("--remote-python")
    orchestrate.add_argument("--remote-source-dir", type=Path)
    orchestrate.add_argument("--rdzv-host")
    orchestrate.add_argument("--run-id", default=f"torchrun-pressure-{os.getpid()}")
    orchestrate.add_argument("--timeout", type=float, default=1_800.0)
    return parser


def main() -> None:
    _orchestrate(_parser().parse_args())


if __name__ == "__main__":
    main()
