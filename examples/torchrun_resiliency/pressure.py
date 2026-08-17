"""Run repeated restart and replacement pressure through native torchrun.

The orchestrator launches one torchrun agent per supplied GPU. Eight agents
train while eight remain parked as standbys. The generated campaign alternates
same-node job restarts with SCOUT-localized replay-only SDC that requires node
replacement. GEMINI supplies every manager-selected recovery point, and the
final state is compared with an uninterrupted baseline.

Run across two eight-GPU hosts:

    python -m examples.torchrun_resiliency.pressure orchestrate \
      --fault-campaign-dir /tmp/lm-resiliency-torchrun-pressure
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed import FileStore, Store, TCPStore
from torch.nn.parallel import DistributedDataParallel

from examples.production_loops.pytorch import TinyCausalLM, _tokens
from examples.torchrun_resiliency._replay_fault import ReplayFaultCampaign
from lm_resiliency import (
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
    InMemoryCkptConfig,
    OrchestrationHooks,
    RecoveryDecision,
    ReplayHarnessConfig,
)
from lm_resiliency.integrations.pytorch import enable_resiliency
from lm_resiliency.integrations.torchrun._protocol import RestartPlan, SlotAssignment
from lm_resiliency.integrations.torchrun._simple_runtime import (
    SimpleRecoveryPlanStore,
    SimpleRestartContextFile,
    _node_id_from_machine_id,
)

EXPECTED_RECIPES = {"embedding", "hidden", "output", "optimizer"}
LOSS_MAX_ABS_DIFF = 1e-2
MODEL_MAX_ABS_DIFF = 2e-3
OPTIMIZER_MAX_ABS_DIFF = 5e-5
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


def _default_pressure_campaign() -> FaultCampaign:
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


def _pressure_events(campaign: FaultCampaign) -> tuple[PressureEvent, ...]:
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
            kind = "restart"
            fault_rank = None
        elif (
            fault.type is FailureType.TENSOR_CORRUPTION
            and fault.target.surface is FaultSurface.OUTPUT
            and fault.target.rank is not None
        ):
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


def _load_campaign_bundle(path: Path) -> tuple[FaultCampaign, tuple[PressureEvent, ...]]:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path, 0o700)
    manifest_path = path / CAMPAIGN_FILENAME
    if manifest_path.exists():
        campaign = FaultCampaign.from_json(manifest_path)
    else:
        campaign = _default_pressure_campaign()
        _atomic_json(manifest_path, campaign.to_dict())
    return campaign, _pressure_events(campaign)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"{path} does not contain a JSON object")
    return value


def _atomic_torch(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("wb") as handle:
        torch.save(dict(value), handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _tensor_digest(values: Sequence[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in values:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _model_tensors(model: DistributedDataParallel) -> list[torch.Tensor]:
    return [parameter.detach().cpu().clone() for parameter in model.parameters()]


def _optimizer_tensors(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for parameter in model.parameters():
        state = optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                tensors.append(value.detach().cpu().clone())
    return tensors


def _training_state_digest(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
) -> str:
    return _tensor_digest([*_model_tensors(model), *_optimizer_tensors(model, optimizer)])


def _rng_digest(device: torch.device) -> str:
    return _tensor_digest(
        [
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device),
        ]
    )


def _assert_all(condition: bool, message: str, device: torch.device) -> None:
    value = torch.tensor([int(condition)], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if not value.item():
        raise AssertionError(message)


def _checkpoint_is_safe_for_replacement(
    checkpoint_manager: Any,
    handle: Any,
    *,
    expected_step: int,
) -> bool:
    """Verify recovery safety without relying on process-local save history."""
    verified_step = checkpoint_manager.checkpoint_status.recovery_verified_step
    flushed_step = handle.flush_for_restart()
    recoverable_step = checkpoint_manager.local_recovery_step("recovery_verified")
    safe = (
        verified_step == expected_step
        and flushed_step in (-1, expected_step)
        and recoverable_step == expected_step
    )
    if not safe:
        print(
            "replacement checkpoint diagnostics: "
            f"rank={os.environ.get('RANK', 'unknown')} "
            f"expected={expected_step} verified={verified_step} "
            f"flushed={flushed_step} recoverable={recoverable_step}",
            flush=True,
        )
    return safe


def _snapshot(
    *,
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    extra_state: Mapping[str, Any],
    step: int,
) -> dict[str, Any]:
    return {
        "extra_state": dict(extra_state),
        "model_digest": _tensor_digest(_model_tensors(model)),
        "optimizer_digest": _tensor_digest(_optimizer_tensors(model, optimizer)),
        "rng_digest": _rng_digest(device),
        "state_digest": _training_state_digest(model, optimizer),
        "step": step,
    }


def _verification_state(
    model: DistributedDataParallel,
    optimizer: torch.optim.Optimizer,
) -> dict[str, list[torch.Tensor]]:
    return {
        "model": _model_tensors(model),
        "optimizer": _optimizer_tensors(model, optimizer),
    }


def _replay_config() -> ReplayHarnessConfig:
    return ReplayHarnessConfig(
        check_interval=1,
        rotate_layers=False,
        enable_temporal=False,
        scale_factors=[],
        straggler_min_slowdown_ratio=100.0,
        straggler_min_slowdown_ms=10_000.0,
    )


def _checkpoint_config(args: argparse.Namespace) -> InMemoryCkptConfig:
    return InMemoryCkptConfig(
        enable=True,
        interval=1,
        replication_jump=args.replication_jump,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=0,
        disk_folder=str(args.checkpoint_dir),
        run_id=args.checkpoint_run_id,
        verify_integrity=True,
        pin_memory=True,
    )


def _worker(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != args.world_size:
        raise RuntimeError(f"expected {args.world_size} ranks, got {world_size}")

    context = None
    generation = 0
    if args.context_path is not None and args.context_path.exists():
        context = SimpleRestartContextFile(args.context_path).read()
        if context is None:
            raise AssertionError("restart context path exists without a context")
        generation = context.generation
    os.environ["LM_RESILIENCY_GENERATION"] = str(generation)

    manifest = FaultCampaign.from_json(args.fault_campaign_dir / CAMPAIGN_FILENAME)
    events = _pressure_events(manifest)
    total_steps = int(manifest.metadata["total_steps"])
    event = None if args.mode == "baseline" or generation >= len(events) else events[generation]
    replacement_event = event is not None and event.kind == "replacement"
    replay_campaign = ReplayFaultCampaign(
        steps=total_steps,
        rank=rank,
        world_size=world_size,
        inject_fault=replacement_event,
        fault_step=event.step if replacement_event else total_steps,
        fault_rank=event.fault_rank if replacement_event else 0,
    )
    external_state: dict[str, Any] = {"absolute_step": 0}
    restored_extra: dict[str, Any] = {}
    losses: dict[str, float] = {}
    faults: list[Any] = []
    decisions: list[RecoveryDecision] = []

    def record_fault(result: Any) -> None:
        faults.append(result)
        replay_campaign.record_fault(result)

    def load_extra_state(value: dict[str, Any]) -> None:
        restored_extra.clear()
        restored_extra.update(value)
        external_state.clear()
        external_state.update(value)

    torch.manual_seed(123)
    model = DistributedDataParallel(
        TinyCausalLM().to(device),
        device_ids=[local_rank],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    handle = enable_resiliency(
        model,
        optimizer,
        interval=1,
        checkpoint=_checkpoint_config(args),
        replay=_replay_config(),
        device=device,
        fault_callback=record_fault,
        orchestration=OrchestrationHooks(report_recovery=decisions.append),
        extra_state_fn=lambda: dict(external_state),
        load_extra_state_fn=load_extra_state,
        recovery_mode=(context.recovery_mode if context is not None else None),
    )
    if handle.ckpt_manager is None or handle.replay_harness is None:
        raise AssertionError("GEMINI and SCOUT must both be enabled")
    ckpt_manager = handle.ckpt_manager
    replay_harness = handle.replay_harness
    replay_campaign.bind(handle)

    try:
        if context is None:
            _assert_all(handle.recovered_step == -1, "fresh worker recovered unexpectedly", device)
        else:
            expected = _read_json(
                args.artifact_dir
                / "checkpoints"
                / f"step-{context.checkpoint_step}-rank-{rank}.json"
            )
            recovered = (
                handle.recovered_step == context.checkpoint_step
                and handle.step_count == context.checkpoint_step
                and restored_extra == expected["extra_state"]
                and _training_state_digest(model, optimizer) == expected["state_digest"]
                and _rng_digest(device) == expected["rng_digest"]
            )
            _assert_all(recovered, "GEMINI fresh-process recovery was not exact", device)
            _atomic_json(
                args.artifact_dir / f"recovery-g{generation}-r{rank}.json",
                {
                    "checkpoint_step": context.checkpoint_step,
                    "generation": generation,
                    "logical_node_slot": context.logical_node_slot,
                    "node_id": args.node_id,
                    "rank": rank,
                    "recovered_exact": recovered,
                },
            )

        for step in range(handle.step_count + 1, total_steps + 1):
            external_state["absolute_step"] = step
            replay_campaign.start_step(step)
            tokens, labels = _tokens(rank, step - 1, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = model(tokens, labels)
            loss.backward()
            optimizer.step()
            losses[str(step)] = float(loss.detach())

            ckpt_manager.maybe_wait()
            status = ckpt_manager.checkpoint_status
            if status.recovery_verified_step == step:
                _atomic_json(
                    args.artifact_dir / "checkpoints" / f"step-{step}-rank-{rank}.json",
                    _snapshot(
                        model=model,
                        optimizer=optimizer,
                        device=device,
                        extra_state=external_state,
                        step=step,
                    ),
                )

            if event is not None and event.kind == "restart" and step == event.step:
                restart_checkpoint = handle.flush_for_restart()
                _assert_all(
                    restart_checkpoint == event.checkpoint_step,
                    "restart-only failure did not flush the latest verified checkpoint",
                    device,
                )
                _atomic_json(
                    args.artifact_dir / f"restart-g{generation}-r{rank}.json",
                    {
                        "checkpoint_step": event.checkpoint_step,
                        "generation": generation,
                        "incident_id": event.incident_id,
                        "node_id": args.node_id,
                        "rank": rank,
                    },
                )
                _atomic_json(
                    args.artifact_dir / f"losses-g{generation}-r{rank}.json",
                    {"generation": generation, "losses": losses, "rank": rank},
                )
                while True:
                    time.sleep(1)

            if faults:
                if len(faults) != 1 or len(decisions) != 1:
                    raise AssertionError("SCOUT emitted an unexpected fault/decision count")
                if event is None or event.kind != "replacement" or event.fault_rank is None:
                    raise AssertionError("SCOUT reported an unscheduled replacement fault")
                fault = faults[0]
                decision = decisions[0]
                expected_bitmap = [
                    int(candidate == event.fault_rank) for candidate in fault.peer_ranks
                ]
                expected_step = event.checkpoint_step
                localized = (
                    fault.peer_ranks == list(range(args.world_size))
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
                checkpoint_safe = _checkpoint_is_safe_for_replacement(
                    ckpt_manager,
                    handle,
                    expected_step=expected_step,
                )
                _assert_all(localized, "SCOUT did not localize the injected rank", device)
                _assert_all(selected, "SCOUT selected the wrong recovery checkpoint", device)
                _assert_all(
                    checkpoint_safe,
                    "GEMINI exposed or flushed the contaminated checkpoint",
                    device,
                )
                _atomic_json(
                    args.artifact_dir / f"fault-g{generation}-r{rank}.json",
                    {
                        "checkpoint_step": expected_step,
                        "decision": decision,
                        "generation": generation,
                        "incident_id": event.incident_id,
                        "node_id": args.node_id,
                        "peer_ranks": list(fault.peer_ranks),
                        "rank": rank,
                        "sdc_bitmap": list(fault.sdc_bitmap),
                        "sdc_sources": list(fault.sdc_sources),
                    },
                )
                _atomic_json(
                    args.artifact_dir / f"losses-g{generation}-r{rank}.json",
                    {"generation": generation, "losses": losses, "rank": rank},
                )
                while True:
                    time.sleep(1)

        ckpt_manager.maybe_wait()
        result = replay_harness.last_result
        replay_campaign.validate(handle, result, EXPECTED_RECIPES)
        final = _snapshot(
            model=model,
            optimizer=optimizer,
            device=device,
            extra_state=external_state,
            step=total_steps,
        )
        final.update(
            {
                "generation": generation,
                "losses": losses,
                "node_id": args.node_id,
                "rank": rank,
            }
        )
        prefix = "baseline" if args.mode == "baseline" else f"final-g{generation}"
        _atomic_json(args.artifact_dir / f"{prefix}-r{rank}.json", final)
        _atomic_torch(
            args.artifact_dir / f"{prefix}-state-r{rank}.pt",
            _verification_state(model, optimizer),
        )
        dist.barrier()
    finally:
        replay_campaign.close()
        handle.close()
        if dist.is_initialized():
            dist.destroy_process_group()


def _wait_for(
    paths: Sequence[Path],
    *,
    processes: Sequence[subprocess.Popen[bytes]],
    timeout: float,
) -> None:
    deadline = time.monotonic() + timeout
    while not all(path.exists() for path in paths):
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"torchrun agent failed with exit codes {failed}")
        if time.monotonic() >= deadline:
            missing = [str(path) for path in paths if not path.exists()]
            raise TimeoutError(f"timed out waiting for artifacts: {missing}")
        time.sleep(0.1)


def _remote_enabled(args: argparse.Namespace) -> bool:
    return args.remote_host is not None


def _synthetic_machine_id(node_label: str) -> str:
    return hashlib.sha256(
        f"lm-resiliency/torchrun/campaign/{node_label}".encode("utf-8")
    ).hexdigest()[:32]


def _synthetic_node_id(node_label: str) -> str:
    return _node_id_from_machine_id(_synthetic_machine_id(node_label))


def _prepare_machine_id_file(
    args: argparse.Namespace,
    *,
    node_label: str,
    remote: bool,
) -> Path:
    path = (
        Path("/tmp") / args.run_id / "machine-ids" / f"{node_label}.machine-id"
        if remote
        else args.fault_campaign_dir / "machine-ids" / f"{node_label}.machine-id"
    )
    machine_id = _synthetic_machine_id(node_label)
    if remote:
        command = (
            "umask 077 && "
            f"mkdir -p {shlex.quote(str(path.parent))} && "
            f"printf '%s\\n' {shlex.quote(machine_id)} > {shlex.quote(str(path))}"
        )
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.remote_host,
                command,
            ],
            check=True,
        )
    else:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        path.write_text(machine_id + "\n", encoding="ascii")
        os.chmod(path, 0o600)
    return path


def _launch_process(
    *,
    args: argparse.Namespace,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    remote: bool,
) -> tuple[subprocess.Popen[bytes], Any]:
    log = log_path.open("wb")
    source_root = args.remote_source_dir if remote else Path(__file__).resolve().parents[2]
    if source_root is None:
        raise ValueError("source root is not initialized")
    process_environment = dict(environment)
    existing_python_path = process_environment.get("PYTHONPATH")
    process_environment["PYTHONPATH"] = (
        f"{source_root}:{existing_python_path}" if existing_python_path else str(source_root)
    )
    if remote:
        remote_shell = (
            f"cd {shlex.quote(str(args.remote_source_dir))} && "
            f"{shlex.join(['env', *[f'{key}={value}' for key, value in process_environment.items()], *command])}"
        )
        remote_command = [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.remote_host,
            remote_shell,
        ]
        process = subprocess.Popen(
            remote_command,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    else:
        local_environment = dict(os.environ)
        local_environment.update(process_environment)
        process = subprocess.Popen(
            list(command),
            env=local_environment,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return process, log


def _tcp_store(host: str, timeout: float) -> TCPStore:
    return TCPStore(
        host,
        0,
        is_master=True,
        multi_tenant=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout),
    )


def _prepare_remote_source(args: argparse.Namespace) -> None:
    if not _remote_enabled(args):
        return
    source_root = Path(__file__).resolve().parents[2]
    remote_source = args.remote_source_dir
    if remote_source is None:
        raise ValueError("--remote-source-dir is required with --remote-host")
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.remote_host,
            "mkdir",
            "-p",
            str(remote_source),
        ],
        check=True,
    )
    subprocess.run(
        [
            "rsync",
            "-a",
            "--exclude=.git",
            "--exclude=.venv",
            "--exclude=.mypy_cache",
            "--exclude=.pytest_cache",
            "--exclude=__pycache__",
            "--exclude=*.egg-info",
            f"{source_root}/",
            f"{args.remote_host}:{remote_source}/",
        ],
        check=True,
    )
    site_packages_code = 'import sysconfig; print(sysconfig.get_paths()["purelib"])'
    install_command = (
        f"site_packages=$({shlex.quote(args.remote_python)} -c "
        f"{shlex.quote(site_packages_code)}) && "
        f'/usr/bin/pip install --upgrade --no-deps --target "$site_packages" '
        f"-e {shlex.quote(str(remote_source))}"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.remote_host,
            install_command,
        ],
        check=True,
    )


def _terminate_remote_run(args: argparse.Namespace, run_id: str) -> None:
    if not _remote_enabled(args):
        return
    pattern = f"[r]dzv-id={re.escape(run_id)}"
    command = (
        f"pkill -TERM -f -- {shlex.quote(pattern)} || true; "
        "sleep 1; "
        f"pkill -KILL -f -- {shlex.quote(pattern)} || true"
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            args.remote_host,
            command,
        ],
        check=False,
        timeout=10,
    )


def _launch_baseline(
    args: argparse.Namespace,
    *,
    placements: Sequence[tuple[str, str, bool]],
    world_size: int,
    replication_jump: int,
) -> None:
    store_host = args.rdzv_host if _remote_enabled(args) else "127.0.0.1"
    store = _tcp_store(store_host, args.timeout)
    endpoint = f"{store.host}:{store.port}"
    launched: list[tuple[subprocess.Popen[bytes], Any]] = []
    try:
        for node_label, gpu_id, remote in placements[:world_size]:
            python = args.remote_python if remote else sys.executable
            torchrun = str(Path(python).with_name("torchrun"))
            command = [
                torchrun,
                f"--nnodes={world_size}",
                "--nproc-per-node=1",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={endpoint}",
                f"--rdzv-id={args.run_id}-baseline",
                "--rdzv-conf=is_host=false,read_timeout=120",
                "--module",
                "examples.torchrun_resiliency.pressure",
                "worker",
                "--mode=baseline",
                f"--artifact-dir={args.fault_campaign_dir / 'baseline-artifacts'}",
                f"--checkpoint-dir={args.fault_campaign_dir / 'baseline-checkpoints'}",
                f"--checkpoint-run-id={args.run_id}-baseline",
                f"--fault-campaign-dir={args.fault_campaign_dir}",
                f"--node-id=baseline-{node_label}",
                f"--world-size={world_size}",
                f"--replication-jump={replication_jump}",
            ]
            launched.append(
                _launch_process(
                    args=args,
                    command=command,
                    environment={
                        "CUDA_VISIBLE_DEVICES": gpu_id,
                        "GLOO_SOCKET_IFNAME": "ens32",
                        "NCCL_SOCKET_IFNAME": "ens32",
                    },
                    log_path=args.fault_campaign_dir / f"baseline-{node_label}.log",
                    remote=remote,
                )
            )
        for process, _log in launched:
            process.wait(timeout=args.timeout)
            if process.returncode != 0:
                raise RuntimeError(
                    f"baseline torchrun agent failed with exit code {process.returncode}"
                )
    finally:
        for process, log in launched:
            if process.poll() is None:
                process.terminate()
        _terminate_remote_run(args, f"{args.run_id}-baseline")
        for process, _log in launched:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for _process, log in launched:
            log.close()


def _launch_agent(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    max_restarts: int,
    node_label: str,
    remote: bool,
    endpoint: str | None,
    world_size: int,
    replication_jump: int,
    node_count: int,
) -> tuple[subprocess.Popen[bytes], Any]:
    node_id = _synthetic_node_id(node_label)
    machine_id_path = _prepare_machine_id_file(
        args,
        node_label=node_label,
        remote=remote,
    )
    python = args.remote_python if remote else sys.executable
    torchrun = str(Path(python).with_name("torchrun"))
    context_path = (
        Path("/tmp") / args.run_id / node_label / "restart-context.json"
        if remote
        else args.fault_campaign_dir / "contexts" / node_label / "restart-context.json"
    )
    if endpoint is None:
        rendezvous_endpoint = str(args.fault_campaign_dir / "rdzv")
        rendezvous_config = "store_type=file"
    else:
        rendezvous_endpoint = endpoint
        rendezvous_config = "store_type=tcp,is_host=false,read_timeout=120"
    command = [
        torchrun,
        f"--nnodes={world_size}:{node_count}",
        "--nproc-per-node=1",
        f"--max-restarts={max_restarts}",
        "--monitor-interval=0.1",
        "--rdzv-backend=lm_resiliency",
        f"--rdzv-endpoint={rendezvous_endpoint}",
        f"--rdzv-id={args.run_id}",
        "--rdzv-conf="
        f"{rendezvous_config},lm_resiliency_restart_context_path={context_path},"
        "lm_resiliency_join_timeout_ms=120000,lm_resiliency_poll_interval_ms=100,"
        "lm_resiliency_heartbeat_timeout_ms=10000",
        "--module",
        "examples.torchrun_resiliency.pressure",
        "worker",
        "--mode=campaign",
        f"--artifact-dir={args.fault_campaign_dir / 'campaign-artifacts'}",
        f"--checkpoint-dir={args.fault_campaign_dir / 'campaign-checkpoints'}",
        f"--checkpoint-run-id={args.run_id}-campaign",
        f"--context-path={context_path}",
        f"--fault-campaign-dir={args.fault_campaign_dir}",
        f"--node-id={node_id}",
        f"--world-size={world_size}",
        f"--replication-jump={replication_jump}",
    ]
    return _launch_process(
        args=args,
        command=command,
        environment={
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "GLOO_SOCKET_IFNAME": "ens32",
            "LM_RESILIENCY_MACHINE_ID_PATH": str(machine_id_path),
            "NCCL_SOCKET_IFNAME": "ens32",
        },
        log_path=args.fault_campaign_dir / f"{node_label}.log",
        remote=remote,
    )


def _validate_fault_reports(
    reports: Sequence[dict[str, Any]],
    *,
    expected_generation: int,
    expected_checkpoint_step: int,
    fault_rank: int,
    world_size: int,
) -> None:
    expected_bitmap = [int(rank == fault_rank) for rank in range(world_size)]
    for report in reports:
        if report["generation"] != expected_generation:
            raise AssertionError("fault report generation mismatch")
        if report["sdc_bitmap"] != expected_bitmap:
            raise AssertionError(f"SCOUT localization mismatch: {report}")
        decision = report["decision"]
        if (
            decision["recovery_mode"] != "recovery_verified"
            or decision["checkpoint_source"] != "gemini"
            or decision["checkpoint_step"] != expected_checkpoint_step
            or not decision["available"]
        ):
            raise AssertionError(f"invalid recovery decision: {decision}")


def _validate_restart_reports(
    reports: Sequence[dict[str, Any]],
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


def _wait_for_initial_admission(
    plans: SimpleRecoveryPlanStore,
    *,
    expected_nodes: frozenset[str],
    world_size: int,
    standby_count: int,
    processes: Sequence[subprocess.Popen[bytes]],
    timeout: float,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    deadline = time.monotonic() + timeout
    while True:
        failed = [process.returncode for process in processes if process.poll() not in (None, 0)]
        if failed:
            raise RuntimeError(f"torchrun agent failed with exit codes {failed}")
        initial_nodes = plans.read_initial_nodes()
        if initial_nodes is not None:
            registered_nodes = plans.registered_nodes(max_nodes=len(expected_nodes))
            registered_set = frozenset(registered_nodes)
            unexpected = registered_set - expected_nodes
            if unexpected:
                raise AssertionError(f"unexpected machine identities registered: {unexpected}")
            if registered_set == expected_nodes:
                standbys = tuple(
                    node_id for node_id in registered_nodes if node_id not in initial_nodes
                )
                if len(initial_nodes) != world_size or len(standbys) != standby_count:
                    raise AssertionError("initial admission selected the wrong fleet size")
                return initial_nodes, standbys
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for automatic initial node admission")
        time.sleep(0.1)


def _publish_plan(
    *,
    plans: SimpleRecoveryPlanStore,
    run_id: str,
    generation: int,
    current_nodes: Sequence[str],
    event: PressureEvent,
    replacement_node: str | None,
    quarantined: Sequence[str],
    world_size: int,
    topology_digest: str,
) -> list[str]:
    next_nodes = list(current_nodes)
    next_quarantined = list(quarantined)
    if event.kind == "replacement":
        if event.fault_rank is None or replacement_node is None:
            raise AssertionError("replacement event requires a target rank and standby")
        failed_node = next_nodes[event.fault_rank]
        next_nodes[event.fault_rank] = replacement_node
        next_quarantined.append(failed_node)
        recovery_mode = "recovery_verified"
        reason_code = "sdc_detected"
    else:
        recovery_mode = "latest"
        reason_code = "process_stall"
    plan = RestartPlan(
        plan_id=f"pressure-{event.kind}-{generation + 1}",
        intent_id=event.incident_id,
        run_id=run_id,
        from_generation=generation,
        to_generation=generation + 1,
        incident_ids=(event.incident_id,),
        reason_code=reason_code,
        recovery_mode=recovery_mode,
        checkpoint_source="gemini",
        checkpoint_step=event.checkpoint_step,
        checkpoint_id=None,
        checkpoint_manifest_id=f"gemini-{recovery_mode}-{event.checkpoint_step}",
        slot_assignments=tuple(
            SlotAssignment(slot, node_id, slot, 1) for slot, node_id in enumerate(next_nodes)
        ),
        quarantined_node_ids=tuple(next_quarantined),
        expected_world_size=world_size,
        topology_digest=topology_digest,
        restart_deadline_unix_ms=time.time_ns() // 1_000_000 + 120_000,
    )
    plans.publish(plan)
    return next_nodes


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
            if not path.exists():
                continue
            merged[rank].update(_read_json(path)["losses"])
    for rank in range(world_size):
        final = _read_json(artifact_dir / f"final-g{generations - 1}-r{rank}.json")
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


def _compare_baseline(
    args: argparse.Namespace,
    *,
    generations: int,
    world_size: int,
) -> tuple[float, float, float]:
    baseline_dir = args.fault_campaign_dir / "baseline-artifacts"
    campaign_dir = args.fault_campaign_dir / "campaign-artifacts"
    campaign_losses = _merge_losses(
        campaign_dir,
        generations=generations,
        world_size=world_size,
    )
    maximum_model_difference = 0.0
    maximum_optimizer_difference = 0.0
    maximum_loss_difference = 0.0
    for rank in range(world_size):
        baseline = _read_json(baseline_dir / f"baseline-r{rank}.json")
        campaign = _read_json(campaign_dir / f"final-g{generations - 1}-r{rank}.json")
        if baseline["rng_digest"] != campaign["rng_digest"]:
            raise AssertionError(f"rank {rank} final RNG state diverged")
        if baseline["extra_state"] != campaign["extra_state"]:
            raise AssertionError(f"rank {rank} final extra state diverged")
        baseline_losses = {key: float(value) for key, value in baseline["losses"].items()}
        for step, expected in baseline_losses.items():
            actual = float(campaign_losses[rank][step])
            difference = abs(expected - actual)
            maximum_loss_difference = max(maximum_loss_difference, difference)
            if difference > LOSS_MAX_ABS_DIFF:
                raise AssertionError(
                    f"rank {rank} loss at step {step} differs by {difference:.3e}, "
                    f"above {LOSS_MAX_ABS_DIFF:.1e}"
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
                limit=OPTIMIZER_MAX_ABS_DIFF,
                label="optimizer",
                rank=rank,
            ),
        )
    return (
        maximum_loss_difference,
        maximum_model_difference,
        maximum_optimizer_difference,
    )


def _gpu_ids(value: str, name: str) -> tuple[str, ...]:
    gpu_ids = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpu_ids:
        raise ValueError(f"{name} must provide at least one GPU ID")
    if len(gpu_ids) != len(set(gpu_ids)):
        raise ValueError(f"{name} must not contain duplicate GPU IDs")
    return gpu_ids


def _campaign_layout(
    args: argparse.Namespace,
    campaign: FaultCampaign,
    events: Sequence[PressureEvent],
) -> tuple[
    tuple[tuple[str, str, bool], ...],
    int,
    int,
    str,
]:
    local_gpu_ids = _gpu_ids(args.gpus, "--gpus")
    remote_gpu_ids = _gpu_ids(args.remote_gpus, "--remote-gpus") if _remote_enabled(args) else ()
    placements = tuple(
        [
            *(
                (f"local-gpu-{index:02d}", gpu_id, False)
                for index, gpu_id in enumerate(local_gpu_ids)
            ),
            *(
                (f"remote-gpu-{index:02d}", gpu_id, True)
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
    world_size = active_nodes
    if world_size < 4 or world_size % 2:
        raise ValueError("active GPU-node count must be even and at least four")
    replication_jump = world_size // 2
    topology_digest = f"ddp-world-{world_size}-replication-jump-{replication_jump}-gpu-nodes"
    return placements, world_size, replication_jump, topology_digest


def _orchestrate(args: argparse.Namespace) -> None:
    campaign, events = _load_campaign_bundle(args.fault_campaign_dir)
    placements, world_size, replication_jump, topology_digest = _campaign_layout(
        args,
        campaign,
        events,
    )
    standby_count = int(campaign.metadata["standby_nodes"])
    total_steps = int(campaign.metadata["total_steps"])
    if total_steps <= events[-1].step:
        raise ValueError("campaign metadata.total_steps must include a clean final step")
    if _remote_enabled(args):
        if not args.remote_python:
            raise ValueError("--remote-python is required with --remote-host")
        if args.remote_source_dir is None:
            raise ValueError("--remote-source-dir is required with --remote-host")
        if not args.rdzv_host:
            raise ValueError("--rdzv-host is required with --remote-host")
        _prepare_remote_source(args)
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.remote_host,
                "test",
                "-d",
                str(args.fault_campaign_dir.parent),
            ],
            check=True,
        )
    args.fault_campaign_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.fault_campaign_dir, 0o700)
    for path in (
        args.fault_campaign_dir / "baseline-artifacts",
        args.fault_campaign_dir / "baseline-checkpoints",
        args.fault_campaign_dir / "campaign-artifacts",
        args.fault_campaign_dir / "campaign-checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)

    _launch_baseline(
        args,
        placements=placements,
        world_size=world_size,
        replication_jump=replication_jump,
    )

    control_store: Store
    endpoint: str | None
    if _remote_enabled(args):
        tcp_store = _tcp_store(args.rdzv_host, args.timeout)
        control_store = tcp_store
        endpoint = f"{tcp_store.host}:{tcp_store.port}"
    else:
        endpoint = None
        control_store = FileStore(str(args.fault_campaign_dir / "rdzv"))
    launched = [
        _launch_agent(
            args=args,
            gpu_id=gpu_id,
            max_restarts=len(events),
            node_label=node_label,
            remote=remote,
            endpoint=endpoint,
            world_size=world_size,
            replication_jump=replication_jump,
            node_count=len(placements),
        )
        for node_label, gpu_id, remote in placements
    ]
    processes = [process for process, _log in launched]
    logs = [log for _process, log in launched]
    artifacts = args.fault_campaign_dir / "campaign-artifacts"
    plans = SimpleRecoveryPlanStore(control_store, run_id=args.run_id)
    expected_nodes = frozenset(_synthetic_node_id(label) for label, _gpu, _remote in placements)
    quarantined: list[str] = []
    try:
        initial_nodes, standby_nodes = _wait_for_initial_admission(
            plans,
            expected_nodes=expected_nodes,
            world_size=world_size,
            standby_count=standby_count,
            processes=processes,
            timeout=args.timeout,
        )
        current_nodes = list(initial_nodes)
        replacements = iter(standby_nodes)
        completed_events: list[str] = []
        for generation, event in enumerate(events):
            report_prefix = "fault" if event.kind == "replacement" else "restart"
            report_paths = [
                artifacts / f"{report_prefix}-g{generation}-r{rank}.json"
                for rank in range(world_size)
            ]
            _wait_for(report_paths, processes=processes, timeout=args.timeout)
            reports = [_read_json(path) for path in report_paths]
            replacement_node = None
            if event.kind == "replacement":
                assert event.fault_rank is not None
                _validate_fault_reports(
                    reports,
                    expected_generation=generation,
                    expected_checkpoint_step=event.checkpoint_step,
                    fault_rank=event.fault_rank,
                    world_size=world_size,
                )
                failed_node = current_nodes[event.fault_rank]
                replacement_node = next(replacements)
            else:
                _validate_restart_reports(
                    reports,
                    event=event,
                    expected_generation=generation,
                )
                failed_node = None
            current_nodes = _publish_plan(
                plans=plans,
                run_id=args.run_id,
                generation=generation,
                current_nodes=current_nodes,
                event=event,
                replacement_node=replacement_node,
                quarantined=quarantined,
                world_size=world_size,
                topology_digest=topology_digest,
            )
            if failed_node is not None:
                quarantined.append(failed_node)
            completed_events.append(event.incident_id)
            _atomic_json(
                args.fault_campaign_dir / STATE_FILENAME,
                {
                    "campaign": campaign.name,
                    "completed_incidents": completed_events,
                    "generation": generation + 1,
                    "manifest_identity": campaign.manifest_identity,
                    "quarantined_nodes": quarantined,
                },
            )
            recovery_paths = [
                artifacts / f"recovery-g{generation + 1}-r{rank}.json" for rank in range(world_size)
            ]
            _wait_for(recovery_paths, processes=processes, timeout=args.timeout)
            recovery = [_read_json(path) for path in recovery_paths]
            if not all(item["recovered_exact"] for item in recovery):
                raise AssertionError("GEMINI recovery was not exact on every rank")
            if event.kind == "replacement":
                assert event.fault_rank is not None and replacement_node is not None
                if recovery[event.fault_rank]["node_id"] != replacement_node:
                    raise AssertionError("selected standby did not inherit the faulty logical rank")
                if recovery[event.fault_rank]["logical_node_slot"] != event.fault_rank:
                    raise AssertionError("selected standby did not inherit the faulty logical slot")

        final_paths = [
            artifacts / f"final-g{len(events)}-r{rank}.json" for rank in range(world_size)
        ]
        _wait_for(final_paths, processes=processes, timeout=args.timeout)
        loss_max_abs_diff, model_max_abs_diff, optimizer_max_abs_diff = _compare_baseline(
            args,
            generations=len(events) + 1,
            world_size=world_size,
        )
        if set(current_nodes) & set(initial_nodes):
            raise AssertionError("pressure campaign did not replace every initial GPU-node")
        plans.close_run()
        for process in processes:
            process.wait(timeout=args.timeout)
            if process.returncode != 0:
                raise RuntimeError(f"torchrun agent failed with exit code {process.returncode}")
        _atomic_json(
            args.fault_campaign_dir / "summary.json",
            {
                "campaign": campaign.name,
                "campaign_manifest_identity": campaign.manifest_identity,
                "final_nodes": current_nodes,
                "final_step": total_steps,
                "gpu_nodes": [
                    {
                        "gpu_id": gpu_id,
                        "location": "remote" if remote else "local",
                        "node_id": _synthetic_node_id(node_label),
                        "node_label": node_label,
                    }
                    for node_label, gpu_id, remote in placements
                ],
                "initial_nodes": list(initial_nodes),
                "localization": "exact",
                "replacement_failures": sum(event.kind == "replacement" for event in events),
                "replication_jump": replication_jump,
                "recoveries": "bitwise exact",
                "restart_only_failures": sum(event.kind == "restart" for event in events),
                "standbys": list(standby_nodes),
                "total_failures": len(events),
                "world_size": world_size,
                "training_vs_baseline": {
                    "loss_max_abs_diff": loss_max_abs_diff,
                    "loss_max_abs_diff_limit": LOSS_MAX_ABS_DIFF,
                    "model_max_abs_diff": model_max_abs_diff,
                    "model_max_abs_diff_limit": MODEL_MAX_ABS_DIFF,
                    "optimizer_max_abs_diff": optimizer_max_abs_diff,
                    "optimizer_max_abs_diff_limit": OPTIMIZER_MAX_ABS_DIFF,
                    "rng": "bitwise exact",
                },
            },
        )
    finally:
        plans.close_run()
        for process in processes:
            if process.poll() is None:
                process.terminate()
        _terminate_remote_run(args, args.run_id)
        for process in processes:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        for log in logs:
            log.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)

    worker = subparsers.add_parser("worker")
    worker.add_argument("--mode", choices=("baseline", "campaign"), required=True)
    worker.add_argument("--artifact-dir", type=Path, required=True)
    worker.add_argument("--checkpoint-dir", type=Path, required=True)
    worker.add_argument("--checkpoint-run-id", required=True)
    worker.add_argument("--context-path", type=Path)
    worker.add_argument("--fault-campaign-dir", type=Path, required=True)
    worker.add_argument("--node-id", required=True)
    worker.add_argument("--replication-jump", type=int, required=True)
    worker.add_argument("--world-size", type=int, required=True)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument(
        "--fault-campaign-dir",
        type=Path,
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
    args = _parser().parse_args()
    if args.command == "worker":
        _worker(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
