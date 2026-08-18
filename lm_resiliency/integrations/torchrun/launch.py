"""Typed command construction for the LM Resiliency torchrun backend."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TorchrunLaunchConfig:
    """Framework-neutral options shared by LM Resiliency torchrun agents."""

    run_id: str
    rendezvous_endpoint: str
    restart_context_path: Path
    min_nodes: int
    max_nodes: int
    nproc_per_node: int
    max_restarts: int
    torchrun: str = "torchrun"
    store_type: str = "tcp"
    is_host: bool | None = None
    read_timeout_seconds: int = 120
    join_timeout_ms: int = 300_000
    poll_interval_ms: int = 250
    heartbeat_timeout_ms: int = 10_000
    monitor_interval_seconds: float = 0.1
    worker_config: Path | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("run_id", self.run_id),
            ("rendezvous_endpoint", self.rendezvous_endpoint),
            ("torchrun", self.torchrun),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-empty string")
            if "," in value:
                raise ValueError(f"{name} must not contain commas")
        for name, value in (
            ("min_nodes", self.min_nodes),
            ("max_nodes", self.max_nodes),
            ("nproc_per_node", self.nproc_per_node),
            ("read_timeout_seconds", self.read_timeout_seconds),
            ("join_timeout_ms", self.join_timeout_ms),
            ("poll_interval_ms", self.poll_interval_ms),
            ("heartbeat_timeout_ms", self.heartbeat_timeout_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value < 1:
                raise ValueError(f"{name} must be positive")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be at least min_nodes")
        if isinstance(self.max_restarts, bool) or not isinstance(self.max_restarts, int):
            raise TypeError("max_restarts must be an integer")
        if self.max_restarts < 0:
            raise ValueError("max_restarts must be non-negative")
        if (
            isinstance(self.monitor_interval_seconds, bool)
            or not isinstance(self.monitor_interval_seconds, (int, float))
            or self.monitor_interval_seconds <= 0
        ):
            raise ValueError("monitor_interval_seconds must be positive")
        if self.store_type not in {"file", "tcp"}:
            raise ValueError("store_type must be 'file' or 'tcp'")
        if self.is_host is not None and not isinstance(self.is_host, bool):
            raise TypeError("is_host must be a boolean or None")
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise ValueError("restart_context_path must be absolute")
        if "," in str(self.restart_context_path):
            raise ValueError("restart_context_path must not contain commas")
        if self.worker_config is not None:
            if not isinstance(self.worker_config, Path):
                raise TypeError("worker_config must be pathlib.Path")
            if not self.worker_config.is_absolute():
                raise ValueError("worker_config must be absolute")
            if "," in str(self.worker_config):
                raise ValueError("worker_config must not contain commas")

    def command(
        self,
        *,
        module: str,
        module_args: Sequence[str] = (),
    ) -> list[str]:
        """Build one torchrun command without launching a subprocess."""

        if not isinstance(module, str) or not module.strip():
            raise ValueError("module must be a non-empty string")
        if any(not isinstance(argument, str) for argument in module_args):
            raise TypeError("module_args must contain only strings")
        rendezvous = [
            f"store_type={self.store_type}",
            f"read_timeout={self.read_timeout_seconds}",
            f"lm_resiliency_restart_context_path={self.restart_context_path}",
            f"lm_resiliency_join_timeout_ms={self.join_timeout_ms}",
            f"lm_resiliency_poll_interval_ms={self.poll_interval_ms}",
            f"lm_resiliency_heartbeat_timeout_ms={self.heartbeat_timeout_ms}",
        ]
        if self.is_host is not None:
            rendezvous.append(f"is_host={str(self.is_host).lower()}")
        if self.worker_config is not None:
            rendezvous.append(f"lm_resiliency_worker_config={self.worker_config}")
        return [
            self.torchrun,
            f"--nnodes={self.min_nodes}:{self.max_nodes}",
            f"--nproc-per-node={self.nproc_per_node}",
            f"--max-restarts={self.max_restarts}",
            f"--monitor-interval={self.monitor_interval_seconds}",
            "--rdzv-backend=lm_resiliency",
            f"--rdzv-endpoint={self.rendezvous_endpoint}",
            f"--rdzv-id={self.run_id}",
            f"--rdzv-conf={','.join(rendezvous)}",
            "--module",
            module,
            *module_args,
        ]


__all__ = ["TorchrunLaunchConfig"]
