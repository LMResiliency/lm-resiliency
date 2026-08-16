"""Run a complete LM Resiliency lifecycle through native torchrun.

The orchestrator launches an uninterrupted baseline and a managed campaign with
four active GPU workers plus two parked standbys. SCOUT localizes replay-only
SDC at logical rank 3, GEMINI exposes the preceding recovery-verified step, and
the manager publishes one simplified torchrun recovery plan. The campaign
repeats this sequence twice and compares the final state and losses with the
uninterrupted baseline.

Run on one host with at least six GPUs:

    python -m examples.production_loops.torchrun orchestrate \
      --workspace /tmp/lm-resiliency-torchrun-production
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.distributed as dist
from torch.distributed import FileStore, Store, TCPStore
from torch.nn.parallel import DistributedDataParallel

from examples.production_loops._common import ReplayFaultCampaign
from examples.production_loops.pytorch import TinyCausalLM, _tokens
from lm_resiliency import (
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
)

ACTIVE_NODES = ("node-a", "node-b", "node-c", "node-d")
STANDBY_NODES = ("node-e", "node-f")
FAULT_RANK = 3
FAULT_STEPS = (3, 6)
TOTAL_STEPS = 8
WORLD_SIZE = len(ACTIVE_NODES)
REPLICATION_JUMP = WORLD_SIZE // 2
TOPOLOGY_DIGEST = "ddp-world-4-replication-jump-2"
EXPECTED_RECIPES = {"embedding", "hidden", "output", "optimizer"}
MODEL_MAX_ABS_DIFF = 1e-10
OPTIMIZER_MAX_ABS_DIFF = 1e-9
REMOTE_NODES = frozenset({"node-c", "node-d", "node-f"})


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
        replication_jump=REPLICATION_JUMP,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=0,
        disk_folder=str(args.checkpoint_dir),
        run_id=args.checkpoint_run_id,
        verify_integrity=True,
        pin_memory=True,
    )


def _fault_step(generation: int) -> int | None:
    return FAULT_STEPS[generation] if generation < len(FAULT_STEPS) else None


def _worker(args: argparse.Namespace) -> None:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    if world_size != WORLD_SIZE:
        raise RuntimeError(f"expected {WORLD_SIZE} ranks, got {world_size}")

    context = None
    generation = 0
    if args.context_path is not None and args.context_path.exists():
        context = SimpleRestartContextFile(args.context_path).read()
        if context is None:
            raise AssertionError("restart context path exists without a context")
        generation = context.generation

    fault_step = None if args.mode == "baseline" else _fault_step(generation)
    campaign = ReplayFaultCampaign(
        steps=TOTAL_STEPS,
        rank=rank,
        world_size=world_size,
        inject_fault=fault_step is not None,
        fault_step=fault_step or TOTAL_STEPS,
        fault_rank=FAULT_RANK,
    )
    external_state: dict[str, Any] = {"absolute_step": 0}
    restored_extra: dict[str, Any] = {}
    losses: dict[str, float] = {}
    faults: list[Any] = []
    decisions: list[RecoveryDecision] = []

    def record_fault(result: Any) -> None:
        faults.append(result)
        campaign.record_fault(result)

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
    campaign.bind(handle)

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

        for step in range(handle.step_count + 1, TOTAL_STEPS + 1):
            external_state["absolute_step"] = step
            campaign.start_step(step)
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

            if faults:
                if len(faults) != 1 or len(decisions) != 1:
                    raise AssertionError("SCOUT emitted an unexpected fault/decision count")
                fault = faults[0]
                decision = decisions[0]
                expected_bitmap = [int(candidate == FAULT_RANK) for candidate in fault.peer_ranks]
                expected_step = step - 1
                localized = (
                    fault.peer_ranks == list(range(WORLD_SIZE))
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
                checkpoint_safe = (
                    ckpt_manager._last_saved_step == expected_step
                    and ckpt_manager.checkpoint_status.recovery_verified_step == expected_step
                    and handle.flush_for_restart() == expected_step
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
        campaign.validate(handle, result, EXPECTED_RECIPES)
        final = _snapshot(
            model=model,
            optimizer=optimizer,
            device=device,
            extra_state=external_state,
            step=TOTAL_STEPS,
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
        campaign.close()
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


def _remote_node(args: argparse.Namespace, node_id: str) -> bool:
    return _remote_enabled(args) and node_id in REMOTE_NODES


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


def _launch_baseline(args: argparse.Namespace, gpu_ids: Sequence[str]) -> None:
    if _remote_enabled(args):
        remote_gpu_ids = tuple(item.strip() for item in args.remote_gpus.split(",") if item.strip())
        if len(gpu_ids) < 2 or len(remote_gpu_ids) < 2:
            raise ValueError("two-host baseline requires two GPUs on each host")
        store = _tcp_store(args.rdzv_host, args.timeout)
        endpoint = f"{store.host}:{store.port}"
        launched: list[tuple[subprocess.Popen[bytes], Any]] = []
        try:
            for label, remote, visible_devices, python in (
                ("local", False, ",".join(gpu_ids[:2]), sys.executable),
                ("remote", True, ",".join(remote_gpu_ids[:2]), args.remote_python),
            ):
                torchrun = str(Path(python).with_name("torchrun"))
                command = [
                    torchrun,
                    "--nnodes=2",
                    "--nproc-per-node=2",
                    "--rdzv-backend=c10d",
                    f"--rdzv-endpoint={endpoint}",
                    f"--rdzv-id={args.run_id}-baseline",
                    "--rdzv-conf=is_host=false,read_timeout=120",
                    "--module",
                    "examples.production_loops.torchrun",
                    "worker",
                    "--mode=baseline",
                    f"--artifact-dir={args.workspace / 'baseline-artifacts'}",
                    f"--checkpoint-dir={args.workspace / 'baseline-checkpoints'}",
                    f"--checkpoint-run-id={args.run_id}-baseline",
                    f"--node-id=baseline-{label}",
                ]
                launched.append(
                    _launch_process(
                        args=args,
                        command=command,
                        environment={
                            "CUDA_VISIBLE_DEVICES": visible_devices,
                            "GLOO_SOCKET_IFNAME": "ens32",
                            "NCCL_SOCKET_IFNAME": "ens32",
                        },
                        log_path=args.workspace / f"baseline-{label}.log",
                        remote=remote,
                    )
                )
            for process, _log in launched:
                process.wait(timeout=args.timeout)
                if process.returncode != 0:
                    raise RuntimeError(
                        f"two-host baseline failed with exit code {process.returncode}"
                    )
        finally:
            for process, log in launched:
                if process.poll() is None:
                    process.terminate()
                log.close()
        return

    local_torchrun = Path(sys.executable).with_name("torchrun")
    log_path = args.workspace / "baseline.log"
    environment = dict(os.environ)
    environment["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids[:WORLD_SIZE])
    command = [
        str(local_torchrun),
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        "--module",
        "examples.production_loops.torchrun",
        "worker",
        "--mode=baseline",
        f"--artifact-dir={args.workspace / 'baseline-artifacts'}",
        f"--checkpoint-dir={args.workspace / 'baseline-checkpoints'}",
        f"--checkpoint-run-id={args.run_id}-baseline",
        "--node-id=baseline",
    ]
    with log_path.open("wb") as log:
        completed = subprocess.run(
            command,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=args.timeout,
            check=False,
        )
    if completed.returncode != 0:
        raise RuntimeError(f"baseline failed with exit code {completed.returncode}: {log_path}")


def _launch_agent(
    *,
    args: argparse.Namespace,
    gpu_id: str,
    node_id: str,
    endpoint: str | None,
) -> tuple[subprocess.Popen[bytes], Any]:
    remote = _remote_node(args, node_id)
    python = args.remote_python if remote else sys.executable
    torchrun = str(Path(python).with_name("torchrun"))
    context_path = (
        Path("/tmp") / args.run_id / node_id / "restart-context.json"
        if remote
        else args.workspace / "contexts" / node_id / "restart-context.json"
    )
    if endpoint is None:
        rendezvous_endpoint = str(args.workspace / "rdzv")
        rendezvous_config = "store_type=file"
    else:
        rendezvous_endpoint = endpoint
        rendezvous_config = "store_type=tcp,is_host=false,read_timeout=120"
    command = [
        torchrun,
        f"--nnodes={WORLD_SIZE}:{WORLD_SIZE + len(STANDBY_NODES)}",
        "--nproc-per-node=1",
        "--max-restarts=4",
        "--monitor-interval=0.1",
        "--rdzv-backend=lm_resiliency",
        f"--rdzv-endpoint={rendezvous_endpoint}",
        f"--rdzv-id={args.run_id}",
        "--rdzv-conf="
        f"{rendezvous_config},node_id={node_id},"
        f"active_nodes={';'.join(ACTIVE_NODES)},local_world_size=1,"
        f"restart_context_path={context_path},join_timeout_ms=120000,"
        "poll_interval_ms=100,heartbeat_timeout_ms=10000",
        "--module",
        "examples.production_loops.torchrun",
        "worker",
        "--mode=campaign",
        f"--artifact-dir={args.workspace / 'campaign-artifacts'}",
        f"--checkpoint-dir={args.workspace / 'campaign-checkpoints'}",
        f"--checkpoint-run-id={args.run_id}-campaign",
        f"--context-path={context_path}",
        f"--node-id={node_id}",
    ]
    return _launch_process(
        args=args,
        command=command,
        environment={
            "CUDA_VISIBLE_DEVICES": gpu_id,
            "GLOO_SOCKET_IFNAME": "ens32",
            "NCCL_SOCKET_IFNAME": "ens32",
        },
        log_path=args.workspace / f"{node_id}.log",
        remote=remote,
    )


def _validate_fault_reports(
    reports: Sequence[dict[str, Any]],
    *,
    expected_generation: int,
    expected_checkpoint_step: int,
) -> None:
    expected_bitmap = [0, 0, 0, 1]
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


def _publish_plan(
    *,
    plans: SimpleRecoveryPlanStore,
    run_id: str,
    generation: int,
    current_nodes: Sequence[str],
    replacement_node: str,
    quarantined: Sequence[str],
    checkpoint_step: int,
) -> list[str]:
    next_nodes = list(current_nodes)
    failed_node = next_nodes[FAULT_RANK]
    next_nodes[FAULT_RANK] = replacement_node
    plan = RestartPlan(
        plan_id=f"scout-replacement-{generation + 1}",
        intent_id=f"localized-{failed_node}-sdc",
        run_id=run_id,
        from_generation=generation,
        to_generation=generation + 1,
        incident_ids=(f"injected-sdc-step-{checkpoint_step + 1}",),
        reason_code="sdc_detected",
        recovery_mode="recovery_verified",
        checkpoint_source="gemini",
        checkpoint_step=checkpoint_step,
        checkpoint_id=None,
        checkpoint_manifest_id=f"gemini-recovery-verified-{checkpoint_step}",
        slot_assignments=tuple(
            SlotAssignment(slot, node_id, slot, 1) for slot, node_id in enumerate(next_nodes)
        ),
        quarantined_node_ids=tuple((*quarantined, failed_node)),
        expected_world_size=WORLD_SIZE,
        topology_digest=TOPOLOGY_DIGEST,
        restart_deadline_unix_ms=time.time_ns() // 1_000_000 + 120_000,
    )
    plans.publish(plan)
    return next_nodes


def _merge_losses(artifact_dir: Path, *, generations: int) -> dict[int, dict[str, float]]:
    merged: dict[int, dict[str, float]] = {rank: {} for rank in range(WORLD_SIZE)}
    for generation in range(generations):
        for rank in range(WORLD_SIZE):
            path = artifact_dir / f"losses-g{generation}-r{rank}.json"
            if not path.exists():
                continue
            merged[rank].update(_read_json(path)["losses"])
    for rank in range(WORLD_SIZE):
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


def _compare_baseline(args: argparse.Namespace, *, generations: int) -> tuple[float, float]:
    baseline_dir = args.workspace / "baseline-artifacts"
    campaign_dir = args.workspace / "campaign-artifacts"
    campaign_losses = _merge_losses(campaign_dir, generations=generations)
    maximum_model_difference = 0.0
    maximum_optimizer_difference = 0.0
    for rank in range(WORLD_SIZE):
        baseline = _read_json(baseline_dir / f"baseline-r{rank}.json")
        campaign = _read_json(campaign_dir / f"final-g{generations - 1}-r{rank}.json")
        if baseline["rng_digest"] != campaign["rng_digest"]:
            raise AssertionError(f"rank {rank} final RNG state diverged")
        if baseline["extra_state"] != campaign["extra_state"]:
            raise AssertionError(f"rank {rank} final extra state diverged")
        baseline_losses = {key: float(value) for key, value in baseline["losses"].items()}
        if baseline_losses != campaign_losses[rank]:
            raise AssertionError(
                f"rank {rank} losses diverged: "
                f"baseline={baseline_losses}, campaign={campaign_losses[rank]}"
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
    return maximum_model_difference, maximum_optimizer_difference


def _orchestrate(args: argparse.Namespace) -> None:
    gpu_ids = tuple(item.strip() for item in args.gpus.split(",") if item.strip())
    needed = 3 if _remote_enabled(args) else WORLD_SIZE + len(STANDBY_NODES)
    if len(gpu_ids) < needed:
        raise ValueError(f"--gpus must provide at least {needed} GPU IDs")
    if _remote_enabled(args):
        if not args.remote_python:
            raise ValueError("--remote-python is required with --remote-host")
        if args.remote_source_dir is None:
            raise ValueError("--remote-source-dir is required with --remote-host")
        if not args.rdzv_host:
            raise ValueError("--rdzv-host is required with --remote-host")
        remote_gpu_ids = tuple(item.strip() for item in args.remote_gpus.split(",") if item.strip())
        if len(remote_gpu_ids) < 3:
            raise ValueError("--remote-gpus must provide at least three GPU IDs")
        _prepare_remote_source(args)
        subprocess.run(
            [
                "ssh",
                "-o",
                "BatchMode=yes",
                args.remote_host,
                "test",
                "-d",
                str(args.workspace.parent),
            ],
            check=True,
        )
    args.workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(args.workspace, 0o700)
    for path in (
        args.workspace / "baseline-artifacts",
        args.workspace / "baseline-checkpoints",
        args.workspace / "campaign-artifacts",
        args.workspace / "campaign-checkpoints",
    ):
        path.mkdir(parents=True, exist_ok=True)

    _launch_baseline(args, gpu_ids)

    control_store: Store
    endpoint: str | None
    placements: tuple[tuple[str, str], ...]
    if _remote_enabled(args):
        tcp_store = _tcp_store(args.rdzv_host, args.timeout)
        control_store = tcp_store
        endpoint = f"{tcp_store.host}:{tcp_store.port}"
        remote_gpu_ids = tuple(item.strip() for item in args.remote_gpus.split(",") if item.strip())
        placements = (
            ("node-a", gpu_ids[0]),
            ("node-b", gpu_ids[1]),
            ("node-c", remote_gpu_ids[0]),
            ("node-d", remote_gpu_ids[1]),
            ("node-e", gpu_ids[2]),
            ("node-f", remote_gpu_ids[2]),
        )
    else:
        endpoint = None
        control_store = FileStore(str(args.workspace / "rdzv"))
        placements = tuple(zip((*ACTIVE_NODES, *STANDBY_NODES), gpu_ids))
    launched = [
        _launch_agent(args=args, gpu_id=gpu_id, node_id=node_id, endpoint=endpoint)
        for node_id, gpu_id in placements
    ]
    processes = [process for process, _log in launched]
    logs = [log for _process, log in launched]
    artifacts = args.workspace / "campaign-artifacts"
    plans = SimpleRecoveryPlanStore(control_store, run_id=args.run_id)
    current_nodes = list(ACTIVE_NODES)
    quarantined: list[str] = []
    try:
        for generation, (fault_step, replacement_node) in enumerate(
            zip(FAULT_STEPS, STANDBY_NODES)
        ):
            report_paths = [
                artifacts / f"fault-g{generation}-r{rank}.json" for rank in range(WORLD_SIZE)
            ]
            _wait_for(report_paths, processes=processes, timeout=args.timeout)
            reports = [_read_json(path) for path in report_paths]
            _validate_fault_reports(
                reports,
                expected_generation=generation,
                expected_checkpoint_step=fault_step - 1,
            )
            failed_node = current_nodes[FAULT_RANK]
            current_nodes = _publish_plan(
                plans=plans,
                run_id=args.run_id,
                generation=generation,
                current_nodes=current_nodes,
                replacement_node=replacement_node,
                quarantined=quarantined,
                checkpoint_step=fault_step - 1,
            )
            quarantined.append(failed_node)
            recovery_paths = [
                artifacts / f"recovery-g{generation + 1}-r{rank}.json" for rank in range(WORLD_SIZE)
            ]
            _wait_for(recovery_paths, processes=processes, timeout=args.timeout)
            recovery = [_read_json(path) for path in recovery_paths]
            if not all(item["recovered_exact"] for item in recovery):
                raise AssertionError("GEMINI recovery was not exact on every rank")
            if recovery[FAULT_RANK]["node_id"] != replacement_node:
                raise AssertionError("selected standby did not inherit the faulty logical rank")
            if recovery[FAULT_RANK]["logical_node_slot"] != FAULT_RANK:
                raise AssertionError("selected standby did not inherit the faulty logical slot")

        final_paths = [
            artifacts / f"final-g{len(FAULT_STEPS)}-r{rank}.json" for rank in range(WORLD_SIZE)
        ]
        _wait_for(final_paths, processes=processes, timeout=args.timeout)
        model_max_abs_diff, optimizer_max_abs_diff = _compare_baseline(
            args,
            generations=len(FAULT_STEPS) + 1,
        )
        plans.close_run()
        for process in processes:
            process.wait(timeout=args.timeout)
            if process.returncode != 0:
                raise RuntimeError(f"torchrun agent failed with exit code {process.returncode}")
        _atomic_json(
            args.workspace / "summary.json",
            {
                "active_nodes": list(ACTIVE_NODES),
                "final_nodes": current_nodes,
                "fault_rank": FAULT_RANK,
                "fault_steps": list(FAULT_STEPS),
                "final_step": TOTAL_STEPS,
                "localization": "exact",
                "recoveries": "bitwise exact",
                "standbys": list(STANDBY_NODES),
                "training_vs_baseline": {
                    "losses": "exact",
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
    worker.add_argument("--node-id", required=True)

    orchestrate = subparsers.add_parser("orchestrate")
    orchestrate.add_argument(
        "--workspace",
        type=Path,
        default=Path(tempfile.mkdtemp(prefix="lm-resiliency-torchrun-gpu-")),
    )
    orchestrate.add_argument("--gpus", default="0,1,2,3,4,5")
    orchestrate.add_argument("--remote-gpus", default="0,1,2")
    orchestrate.add_argument("--remote-host")
    orchestrate.add_argument("--remote-python")
    orchestrate.add_argument("--remote-source-dir", type=Path)
    orchestrate.add_argument("--rdzv-host")
    orchestrate.add_argument("--run-id", default=f"torchrun-production-{os.getpid()}")
    orchestrate.add_argument("--timeout", type=float, default=600.0)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "worker":
        _worker(args)
    else:
        _orchestrate(args)


if __name__ == "__main__":
    main()
