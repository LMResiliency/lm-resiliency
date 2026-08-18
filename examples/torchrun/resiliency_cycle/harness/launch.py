"""Local and SSH launch support for torchrun pressure validation."""

from __future__ import annotations

import hashlib
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import BinaryIO

from torch.distributed import TCPStore

from lm_resiliency.integrations.torchrun import (
    TorchrunLaunchConfig,
    derive_torchrun_node_id,
)

from .campaign import GpuNodePlacement, PressureTopology


@dataclass(frozen=True, slots=True)
class PressureLaunchOptions:
    """Environment-specific options for the two-host validation harness."""

    fault_campaign_dir: Path
    framework: str
    run_id: str
    timeout: float
    remote_host: str | None = None
    remote_python: str | None = None
    remote_source_dir: Path | None = None
    rendezvous_host: str | None = None

    @property
    def remote_enabled(self) -> bool:
        return self.remote_host is not None


@dataclass(slots=True)
class LaunchedAgent:
    process: subprocess.Popen[bytes]
    log: BinaryIO


def synthetic_machine_id(node_label: str) -> str:
    return hashlib.sha256(
        f"lm-resiliency/torchrun/campaign/{node_label}".encode("utf-8")
    ).hexdigest()[:32]


def synthetic_node_id(node_label: str) -> str:
    return derive_torchrun_node_id(synthetic_machine_id(node_label))


def create_tcp_store(host: str, timeout: float) -> TCPStore:
    return TCPStore(
        host,
        0,
        is_master=True,
        multi_tenant=True,
        wait_for_workers=False,
        timeout=timedelta(seconds=timeout),
    )


def prepare_remote_source(options: PressureLaunchOptions) -> None:
    if not options.remote_enabled:
        return
    if options.remote_source_dir is None or options.remote_python is None:
        raise ValueError("remote source and Python are required for remote validation")
    source_root = Path(__file__).resolve().parents[4]
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            options.remote_host,
            "mkdir",
            "-p",
            str(options.remote_source_dir),
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
            f"{options.remote_host}:{options.remote_source_dir}/",
        ],
        check=True,
    )
    install_command = shlex.join(
        [
            options.remote_python,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "--no-deps",
            "-e",
            str(options.remote_source_dir),
        ]
    )
    subprocess.run(
        [
            "ssh",
            "-o",
            "BatchMode=yes",
            options.remote_host,
            install_command,
        ],
        check=True,
    )


def terminate_remote_run(options: PressureLaunchOptions, run_id: str) -> None:
    if not options.remote_enabled:
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
            options.remote_host,
            command,
        ],
        check=False,
        timeout=10,
    )


def launch_baseline(
    options: PressureLaunchOptions,
    topology: PressureTopology,
) -> None:
    store_host = options.rendezvous_host if options.remote_enabled else "127.0.0.1"
    if store_host is None:
        raise ValueError("rendezvous host is required for remote validation")
    store = create_tcp_store(store_host, options.timeout)
    endpoint = f"{store.host}:{store.port}"
    launched: list[LaunchedAgent] = []
    try:
        for placement in topology.placements[: topology.world_size]:
            python = options.remote_python if placement.remote else sys.executable
            if python is None:
                raise ValueError("remote Python is required for remote placements")
            command = [
                str(Path(python).with_name("torchrun")),
                f"--nnodes={topology.world_size}",
                "--nproc-per-node=1",
                "--rdzv-backend=c10d",
                f"--rdzv-endpoint={endpoint}",
                f"--rdzv-id={options.run_id}-baseline",
                "--rdzv-conf=is_host=false,read_timeout=120",
                "--module",
                "examples.torchrun.resiliency_cycle.harness.worker",
                f"--framework={options.framework}",
                "--mode=baseline",
                f"--fault-campaign-dir={options.fault_campaign_dir}",
            ]
            launched.append(
                _launch_process(
                    options=options,
                    command=command,
                    environment=_gpu_environment(placement.gpu_id),
                    log_path=(options.fault_campaign_dir / f"baseline-{placement.node_label}.log"),
                    remote=placement.remote,
                )
            )
        for agent in launched:
            agent.process.wait(timeout=options.timeout)
            if agent.process.returncode != 0:
                raise RuntimeError(
                    f"baseline torchrun agent failed with exit code {agent.process.returncode}"
                )
    finally:
        cleanup_agents(
            options,
            launched,
            remote_run_id=f"{options.run_id}-baseline",
        )


def launch_managed_agents(
    options: PressureLaunchOptions,
    topology: PressureTopology,
    *,
    endpoint: str | None,
    max_restarts: int,
) -> list[LaunchedAgent]:
    launched: list[LaunchedAgent] = []
    try:
        for placement in topology.placements:
            launched.append(
                _launch_managed_agent(
                    options=options,
                    placement=placement,
                    endpoint=endpoint,
                    max_restarts=max_restarts,
                    topology=topology,
                )
            )
    except BaseException:
        cleanup_agents(options, launched, remote_run_id=options.run_id)
        raise
    return launched


def cleanup_agents(
    options: PressureLaunchOptions,
    agents: Sequence[LaunchedAgent],
    *,
    remote_run_id: str,
) -> None:
    for agent in agents:
        if agent.process.poll() is None:
            agent.process.terminate()
    terminate_remote_run(options, remote_run_id)
    for agent in agents:
        try:
            agent.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            agent.process.kill()
            agent.process.wait(timeout=10)
        agent.log.close()


def _launch_managed_agent(
    *,
    options: PressureLaunchOptions,
    placement: GpuNodePlacement,
    endpoint: str | None,
    max_restarts: int,
    topology: PressureTopology,
) -> LaunchedAgent:
    machine_id_path = _prepare_machine_id_file(
        options,
        node_label=placement.node_label,
        remote=placement.remote,
    )
    python = options.remote_python if placement.remote else sys.executable
    if python is None:
        raise ValueError("remote Python is required for remote placements")
    context_path = (
        Path("/tmp") / options.run_id / placement.node_label / "restart-context.json"
        if placement.remote
        else (
            options.fault_campaign_dir / "contexts" / placement.node_label / "restart-context.json"
        )
    )
    launch = TorchrunLaunchConfig(
        run_id=options.run_id,
        rendezvous_endpoint=(
            str(options.fault_campaign_dir / "rdzv") if endpoint is None else endpoint
        ),
        restart_context_path=context_path,
        min_nodes=topology.world_size,
        max_nodes=len(topology.placements),
        nproc_per_node=1,
        max_restarts=max_restarts,
        torchrun=str(Path(python).with_name("torchrun")),
        store_type="file" if endpoint is None else "tcp",
        is_host=None if endpoint is None else False,
        join_timeout_ms=120_000,
        poll_interval_ms=100,
        heartbeat_timeout_ms=10_000,
    )
    command = launch.command(
        module="examples.torchrun.resiliency_cycle.harness.worker",
        module_args=(
            f"--framework={options.framework}",
            "--mode=campaign",
            f"--fault-campaign-dir={options.fault_campaign_dir}",
        ),
    )
    environment = _gpu_environment(placement.gpu_id)
    environment["LM_RESILIENCY_MACHINE_ID_PATH"] = str(machine_id_path)
    return _launch_process(
        options=options,
        command=command,
        environment=environment,
        log_path=options.fault_campaign_dir / f"{placement.node_label}.log",
        remote=placement.remote,
    )


def _prepare_machine_id_file(
    options: PressureLaunchOptions,
    *,
    node_label: str,
    remote: bool,
) -> Path:
    path = (
        Path("/tmp") / options.run_id / "machine-ids" / f"{node_label}.machine-id"
        if remote
        else options.fault_campaign_dir / "machine-ids" / f"{node_label}.machine-id"
    )
    machine_id = synthetic_machine_id(node_label)
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
                options.remote_host,
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
    options: PressureLaunchOptions,
    command: Sequence[str],
    environment: Mapping[str, str],
    log_path: Path,
    remote: bool,
) -> LaunchedAgent:
    log = log_path.open("wb")
    try:
        source_root = options.remote_source_dir if remote else Path(__file__).resolve().parents[4]
        if source_root is None:
            raise ValueError("source root is not initialized")
        process_environment = dict(environment)
        existing_python_path = process_environment.get("PYTHONPATH")
        process_environment["PYTHONPATH"] = (
            f"{source_root}:{existing_python_path}" if existing_python_path else str(source_root)
        )
        if remote:
            remote_shell = (
                f"cd {shlex.quote(str(options.remote_source_dir))} && "
                f"{shlex.join(['env', *[f'{key}={value}' for key, value in process_environment.items()], *command])}"
            )
            process = subprocess.Popen(
                [
                    "ssh",
                    "-o",
                    "BatchMode=yes",
                    options.remote_host,
                    remote_shell,
                ],
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
    except BaseException:
        log.close()
        raise
    return LaunchedAgent(process, log)


def _gpu_environment(gpu_id: str) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": gpu_id}
