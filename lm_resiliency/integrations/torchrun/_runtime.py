"""Runtime configuration and node-local restart-context handoff for torchrun."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
import socket
import stat
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Callable, ClassVar

try:
    _toml = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job.
    _toml = importlib.import_module("tomli")

from torch.distributed import DistStoreError, PrefixStore, Store
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousConnectionError,
    RendezvousHandler,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationClockError,
    AgentRegistrationCorrupt,
    AgentRegistrationLost,
    AgentRegistrationManager,
    AgentRegistrationUnavailable,
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistoryCorrupt,
    AgentRegistrationHistoryError,
    AgentRegistrationHistoryReader,
)
from lm_resiliency.integrations.torchrun._agent_registration_records import (
    HeldAgentRegistration,
)
from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreConflict,
    ControlStoreEntry,
    ControlStoreWrite,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    GenerationStateCorrupt,
    GenerationStateError,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    ProtocolValidationError,
    RankAssignment,
    RestartContext,
    RestartPlan,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication import (
    RestartPlanPublicationReadConflict,
    RestartPlanPublicationReadCorrupt,
    RestartPlanPublicationReader,
)
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    RestartPlanPersistedRecoveryState,
)

_BACKEND = "lm_resiliency"
_NODE_ID_ENV = "LM_RESILIENCY_NODE_ID"
_LOCAL_WORLD_SIZE_ENV = "LM_RESILIENCY_LOCAL_WORLD_SIZE"
_ENVIRONMENT_DIGEST_ENV = "LM_RESILIENCY_ENVIRONMENT_DIGEST"
_RESOURCE_IDS_ENV = "LM_RESILIENCY_RESOURCE_IDS"
_RESTART_CONTEXT_ENV = "LM_RESILIENCY_RESTART_CONTEXT"
_MAX_CONTEXT_BYTES = 64 * 1024
_MAX_POLICY_BYTES = 64 * 1024
_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_CLOSED_SCHEMA_VERSION = 1
_ARRIVAL_SCHEMA_VERSION = 1
_ARRIVAL_ATTEMPT_SCHEMA_VERSION = 1
_ARRIVAL_COMPLETION_SCHEMA_VERSION = 1
_ARRIVAL_CONSUMPTION_SCHEMA_VERSION = 1
_SHARED_FIELDS = {
    "control_endpoint",
    "replacement_only",
    "max_replacement_generations",
    "registration_lease_duration_ms",
    "poll_interval_ms",
    "join_timeout_ms",
}
_RUNTIME_FIELDS = _SHARED_FIELDS | {
    "config",
    "environment_digest",
    "local_world_size",
    "node_id",
    "resource_ids",
    "restart_context_path",
}


class TorchrunRuntimeConfigError(ValueError):
    """Raised when torchrun runtime configuration is unsafe or contradictory."""


class RestartContextFileError(RuntimeError):
    """Raised when the node-local restart-context file is unsafe or malformed."""


class _DuplicateJsonFieldError(ValueError):
    """Raised when strict JSON decoding observes a duplicate object field."""


@dataclass(frozen=True, slots=True)
class TorchrunRendezvousPolicy:
    """Shared replacement-only policy resolved identically by every agent."""

    SCHEMA_VERSION: ClassVar[int] = 1

    control_endpoint: str
    replacement_only: bool = True
    max_replacement_generations: int = 2
    registration_lease_duration_ms: int = 30_000
    poll_interval_ms: int = 1_000
    join_timeout_ms: int = 300_000

    def __post_init__(self) -> None:
        _nonempty_string(self.control_endpoint, "control_endpoint")
        if not isinstance(self.replacement_only, bool):
            raise TorchrunRuntimeConfigError("replacement_only must be a boolean")
        if not self.replacement_only:
            raise TorchrunRuntimeConfigError(
                "replacement_only=false is unsupported by the lm_resiliency backend"
            )
        _positive_integer(
            self.max_replacement_generations,
            "max_replacement_generations",
        )
        _positive_integer(
            self.registration_lease_duration_ms,
            "registration_lease_duration_ms",
        )
        _positive_integer(self.poll_interval_ms, "poll_interval_ms")
        _positive_integer(self.join_timeout_ms, "join_timeout_ms")
        if self.registration_lease_duration_ms <= self.poll_interval_ms:
            raise TorchrunRuntimeConfigError(
                "registration_lease_duration_ms must exceed poll_interval_ms"
            )
        if self.join_timeout_ms <= self.poll_interval_ms:
            raise TorchrunRuntimeConfigError("join_timeout_ms must exceed poll_interval_ms")

    @property
    def digest(self) -> str:
        """Return the canonical digest of the shared policy fields."""

        return _canonical_digest(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "schema_version": self.SCHEMA_VERSION,
            "control_endpoint": self.control_endpoint,
            "replacement_only": self.replacement_only,
            "max_replacement_generations": self.max_replacement_generations,
            "registration_lease_duration_ms": self.registration_lease_duration_ms,
            "poll_interval_ms": self.poll_interval_ms,
            "join_timeout_ms": self.join_timeout_ms,
        }


@dataclass(frozen=True, slots=True)
class TorchrunRuntimeConfig:
    """Resolved shared policy and node-local torchrun inputs."""

    policy: TorchrunRendezvousPolicy
    run_id: str
    endpoint: str
    min_nodes: int
    max_nodes: int
    local_world_size: int
    node_id: str
    environment_digest: str
    resource_ids: tuple[str, ...]
    restart_context_path: Path
    local_addr: str | None
    source_path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.policy, TorchrunRendezvousPolicy):
            raise TypeError("policy must be TorchrunRendezvousPolicy")
        _nonempty_string(self.run_id, "run_id")
        _nonempty_string(self.endpoint, "endpoint")
        _positive_integer(self.min_nodes, "min_nodes")
        _positive_integer(self.max_nodes, "max_nodes")
        if self.max_nodes <= self.min_nodes:
            raise TorchrunRuntimeConfigError(
                "max_nodes must exceed min_nodes to provide standby capacity"
            )
        _positive_integer(self.local_world_size, "local_world_size")
        _nonempty_string(self.node_id, "node_id")
        _nonempty_string(self.environment_digest, "environment_digest")
        object.__setattr__(
            self,
            "resource_ids",
            _resource_ids(self.resource_ids, "resource_ids"),
        )
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise TorchrunRuntimeConfigError("restart_context_path must be absolute")
        if self.local_addr is not None:
            _nonempty_string(self.local_addr, "local_addr")
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise TorchrunRuntimeConfigError("source_path must be an absolute pathlib.Path")

    @property
    def registration_digest(self) -> str:
        """Return the shared runtime digest registered by every agent."""

        return _canonical_digest(
            {
                "policy": self.policy.to_dict(),
                "run_id": self.run_id,
                "endpoint": self.endpoint,
                "min_nodes": self.min_nodes,
                "max_nodes": self.max_nodes,
                "local_world_size": self.local_world_size,
            }
        )

    @property
    def agent_environment_digest(self) -> str:
        """Return the workload and runtime identity recorded for this agent."""

        return _canonical_digest(
            {
                "runtime": self.registration_digest,
                "workload_environment": self.environment_digest,
            }
        )

    def build_agent_identity(
        self,
        *,
        agent_id: str,
        hostname: str,
    ) -> AgentIdentity:
        """Build the immutable agent identity for this handler incarnation."""

        return AgentIdentity(
            run_id=self.run_id,
            node_id=self.node_id,
            agent_id=_nonempty_string(agent_id, "agent_id"),
            hostname=_nonempty_string(hostname, "hostname"),
            local_world_size=self.local_world_size,
            resource_ids=self.resource_ids,
            environment_digest=self.agent_environment_digest,
        )

    @classmethod
    def from_parameters(
        cls,
        params: RendezvousParameters,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> TorchrunRuntimeConfig:
        if not isinstance(params, RendezvousParameters):
            raise TypeError("params must be RendezvousParameters")
        if params.backend != _BACKEND:
            raise TorchrunRuntimeConfigError(
                f"rendezvous backend must be {_BACKEND!r}, got {params.backend!r}"
            )
        unknown = set(params.config) - _RUNTIME_FIELDS
        if unknown:
            raise TorchrunRuntimeConfigError(
                f"unknown rendezvous configuration fields: {sorted(unknown)!r}"
            )
        source_path = Path(
            _nonempty_string(params.get("config"), "rendezvous config path")
        ).expanduser()
        if not source_path.is_absolute():
            raise TorchrunRuntimeConfigError("rendezvous config path must be absolute")
        policy_values = _load_policy_values(source_path)
        for field in _SHARED_FIELDS:
            value = params.get(field)
            if value is not None:
                policy_values[field] = value
        policy = _policy_from_values(policy_values)
        resolved_environment = MappingProxyType(
            dict(os.environ if environment is None else environment)
        )
        node_id = _node_value(
            params.get("node_id"),
            resolved_environment.get(_NODE_ID_ENV),
            "node_id",
            _NODE_ID_ENV,
        )
        local_world_size = _node_integer_value(
            params.get("local_world_size"),
            resolved_environment.get(_LOCAL_WORLD_SIZE_ENV),
            "local_world_size",
            _LOCAL_WORLD_SIZE_ENV,
        )
        environment_digest = _node_value(
            params.get("environment_digest"),
            resolved_environment.get(_ENVIRONMENT_DIGEST_ENV),
            "environment_digest",
            _ENVIRONMENT_DIGEST_ENV,
        )
        resource_ids = _node_resource_ids(
            params.get("resource_ids"),
            resolved_environment.get(_RESOURCE_IDS_ENV),
        )
        restart_context_path = Path(
            _node_value(
                params.get("restart_context_path"),
                resolved_environment.get(_RESTART_CONTEXT_ENV),
                "restart_context_path",
                _RESTART_CONTEXT_ENV,
            )
        ).expanduser()
        return cls(
            policy=policy,
            run_id=params.run_id,
            endpoint=params.endpoint,
            min_nodes=params.min_nodes,
            max_nodes=params.max_nodes,
            local_world_size=local_world_size,
            node_id=node_id,
            environment_digest=environment_digest,
            resource_ids=resource_ids,
            restart_context_path=restart_context_path,
            local_addr=params.local_addr,
            source_path=source_path,
        )


class SlotAwareRendezvousHandler(RendezvousHandler):
    """Admit the initial fixed-size worker group and park passive standbys."""

    def __init__(
        self,
        config: TorchrunRuntimeConfig,
        *,
        control_store: ControlStore,
        rendezvous_store: Store,
        clock: Callable[[], int],
        agent_id: str | None = None,
        hostname: str | None = None,
        bootstrap_store_info: RendezvousStoreInfo | None = None,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(config, TorchrunRuntimeConfig):
            raise TypeError("config must be TorchrunRuntimeConfig")
        if not callable(clock):
            raise TypeError("clock must be callable")
        if not callable(monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if bootstrap_store_info is not None and not isinstance(
            bootstrap_store_info,
            RendezvousStoreInfo,
        ):
            raise TypeError("bootstrap_store_info must be RendezvousStoreInfo or None")
        self._config = config
        self._control_store = control_store
        self._rendezvous_store = rendezvous_store
        self._bootstrap_store_info = bootstrap_store_info
        self._monotonic_clock = monotonic_clock
        self._clock = clock
        self._agent_identity = config.build_agent_identity(
            agent_id=agent_id or secrets.token_hex(16),
            hostname=hostname or socket.getfqdn(),
        )
        self._registration_manager = AgentRegistrationManager(
            control_store,
            agent_identity=self._agent_identity,
            lease_duration_ms=config.policy.registration_lease_duration_ms,
            clock=clock,
        )
        self._generation_reader = GenerationStateReader(
            control_store,
            run_id=config.run_id,
        )
        self._publication_reader = RestartPlanPublicationReader(
            control_store,
            run_id=config.run_id,
        )
        self._restart_context = RestartContextFile(config.restart_context_path)
        run_digest = hashlib.sha256(config.run_id.encode("utf-8")).hexdigest()
        self._run_digest = run_digest
        self._closure_key = f"{_CONTROL_PREFIX}/runs/{run_digest}/rendezvous-closed"
        self._compatibility_key = f"{_CONTROL_PREFIX}/runs/{run_digest}/rendezvous-compatibility"
        self._closure_value = json.dumps(
            {
                "run_id": config.run_id,
                "schema_version": _CLOSED_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._compatibility_value = json.dumps(
            {
                "environment_digest": self._agent_identity.environment_digest,
                "local_world_size": self._agent_identity.local_world_size,
                "run_id": config.run_id,
                "schema_version": 1,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        self._registration_lock = threading.RLock()
        self._registration: HeldAgentRegistration | None = None
        self._registration_inflight = False
        self._registration_thread: threading.Thread | None = None
        self._registration_error: Exception | None = None
        self._heartbeat_error: Exception | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._heartbeat_last_now_unix_ms = 0
        self._stop_heartbeat_event = threading.Event()
        self._release_thread: threading.Thread | None = None
        self._release_error: Exception | None = None
        self._closure_thread: threading.Thread | None = None
        self._closure_error: Exception | None = None
        self._closure_read_thread: threading.Thread | None = None
        self._closure_read_entry: ControlStoreEntry | None = None
        self._closure_read_error: Exception | None = None
        self._state_changed_event = threading.Event()
        self._closed_event = threading.Event()
        self._closure_lock = threading.Lock()
        self._closure_read_lock = threading.Lock()
        self._shutdown_lock = threading.Lock()

    @property
    def agent_identity(self) -> AgentIdentity:
        """Return the immutable identity registered by this handler."""

        return self._agent_identity

    @property
    def use_agent_store(self) -> bool:
        """Return whether the supplied bootstrap endpoint is agent-owned."""

        return self._bootstrap_store_info is not None

    @property
    def closure_key(self) -> str:
        """Return the run-scoped immutable closure key."""

        return self._closure_key

    def get_backend(self) -> str:
        return _BACKEND

    def get_run_id(self) -> str:
        return self._config.run_id

    def next_rendezvous(self) -> RendezvousInfo:
        """Admit one authoritative fixed-size generation and park passive standbys."""

        formation_deadline = self._monotonic_clock() + self._config.policy.join_timeout_ms / 1_000
        context_written = False
        try:
            if self._is_closed_bounded(formation_deadline):
                raise RendezvousClosedError
            self._ensure_registered(formation_deadline)
            while True:
                current = self._read_generation()
                if current is None:
                    slot = None
                    recovery_state = None
                    admission_deadline = formation_deadline
                else:
                    slot, recovery_state, admission_deadline = self._generation_admission(
                        current,
                        formation_deadline,
                    )
                if self._is_closed_bounded(
                    admission_deadline if current is None or slot is not None else None
                ):
                    raise RendezvousClosedError
                self._raise_heartbeat_error()
                if current is not None:
                    if slot is not None:
                        context_written = self._prepare_restart_context(
                            current,
                            recovery_state,
                        )
                        attempt, completion = self._wait_for_assigned_arrivals(
                            current,
                            slot,
                            admission_deadline,
                        )
                        bootstrap_store = self._bootstrap_store(
                            current.snapshot.record.assignment.generation,
                        )
                        bootstrap = self._build_bootstrap(
                            slot,
                            bootstrap_store,
                            admission_deadline,
                        )
                        if self.is_closed():
                            raise RendezvousClosedError
                        self._raise_heartbeat_error()
                        if self._read_generation() != current:
                            raise RendezvousConnectionError("generation changed during rendezvous")
                        self._publish_arrival_consumption(
                            current,
                            current.snapshot.record.assignment,
                            attempt,
                            slot,
                            completion,
                            admission_deadline,
                        )
                        self._validate_final_admission_state(
                            current,
                            recovery_state,
                        )
                        return RendezvousInfo(
                            bootstrap_store,
                            slot,
                            self._config.min_nodes,
                            bootstrap,
                        )
                    wait_deadline = None
                else:
                    wait_deadline = formation_deadline
                if not self._wait_for_change(wait_deadline):
                    raise RendezvousTimeoutError
                if current is not None and slot is None:
                    formation_deadline = (
                        self._monotonic_clock() + self._config.policy.join_timeout_ms / 1_000
                    )
        except BaseException:
            if context_written:
                try:
                    self._restart_context.clear()
                except (OSError, RestartContextFileError):
                    pass
            self._cleanup_local_resources()
            raise

    def is_closed(self) -> bool:
        if self._closed_event.is_set():
            return True
        try:
            entry = self._read_closure_entry()
            if entry is None:
                return False
        except RendezvousStateError:
            raise
        except Exception as error:
            raise RendezvousConnectionError("failed to read rendezvous closure state") from error
        self._closed_event.set()
        self._state_changed_event.set()
        return True

    def _is_closed_bounded(self, deadline: float | None) -> bool:
        if self._closed_event.is_set():
            return True
        with self._closure_read_lock:
            thread = self._closure_read_thread
            if thread is None:
                self._closure_read_entry = None
                self._closure_read_error = None
                thread = threading.Thread(
                    target=self._read_closure_worker,
                    name=f"lm-resiliency-read-close-{self._config.node_id}",
                    daemon=True,
                )
                self._closure_read_thread = thread
                thread.start()
        while thread.is_alive():
            if self._closed_event.is_set():
                return True
            if deadline is not None:
                remaining = deadline - self._monotonic_clock()
                if remaining <= 0:
                    raise RendezvousTimeoutError
            else:
                remaining = self._config.policy.poll_interval_ms / 1_000
            thread.join(
                timeout=min(
                    remaining,
                    self._config.policy.poll_interval_ms / 1_000,
                )
            )
        with self._closure_read_lock:
            entry = self._closure_read_entry
            error = self._closure_read_error
            self._closure_read_thread = None
            self._closure_read_entry = None
            self._closure_read_error = None
        if (
            entry is None
            and error is None
            and deadline is not None
            and self._monotonic_clock() >= deadline
        ):
            raise RendezvousTimeoutError
        if error is not None:
            if isinstance(error, RendezvousStateError):
                raise error
            raise RendezvousConnectionError("failed to read rendezvous closure state") from error
        if entry is None:
            return False
        self._closed_event.set()
        self._state_changed_event.set()
        return True

    def _read_closure_worker(self) -> None:
        entry: ControlStoreEntry | None = None
        error: Exception | None = None
        try:
            entry = self._read_closure_entry()
        except Exception as closure_error:
            error = closure_error
        with self._closure_read_lock:
            self._closure_read_entry = entry
            self._closure_read_error = error
        self._state_changed_event.set()

    def set_closed(self) -> None:
        self._closed_event.set()
        self._state_changed_event.set()
        try:
            self._publish_closure_bounded()
        finally:
            self._cleanup_local_resources()

    def _publish_closure_bounded(self) -> None:
        with self._closure_lock:
            thread = self._closure_thread
            if thread is None:
                self._closure_error = None
                thread = threading.Thread(
                    target=self._publish_closure_worker,
                    name=f"lm-resiliency-close-{self._config.node_id}",
                    daemon=True,
                )
                self._closure_thread = thread
                thread.start()
        thread.join(timeout=self._bounded_cleanup_timeout_seconds())
        if thread.is_alive():
            raise RendezvousConnectionError("timed out while publishing rendezvous closure state")
        with self._closure_lock:
            error = self._closure_error
            self._closure_thread = None
            self._closure_error = None
        if error is None:
            return
        if isinstance(error, RendezvousStateError):
            raise error
        raise RendezvousConnectionError("failed to publish rendezvous closure state") from error

    def _publish_closure_worker(self) -> None:
        error: Exception | None = None
        try:
            entry = self._read_closure_entry()
            if entry is None:
                try:
                    entry = self._control_store.compare_set(
                        self._closure_key,
                        expected_revision=None,
                        value=self._closure_value,
                    )
                except ControlStoreConflict:
                    entry = self._control_store.get(self._closure_key)
                    if entry is None:
                        raise RendezvousStateError(
                            "rendezvous closure changed without a current record"
                        )
            self._validate_closure_entry(entry)
        except Exception as closure_error:
            error = closure_error
        with self._closure_lock:
            self._closure_error = error
        self._state_changed_event.set()

    def num_nodes_waiting(self) -> int:
        """Hide passive standbys until a committed replacement plan exists."""

        if self.is_closed():
            return 0
        self._raise_heartbeat_error()
        return 0

    def shutdown(self) -> bool:
        self._closed_event.set()
        self._state_changed_event.set()
        return self._cleanup_local_resources()

    def _cleanup_local_resources(self) -> bool:
        with self._shutdown_lock:
            heartbeat_stopped = self._stop_heartbeat()
            registration_released = heartbeat_stopped and self._release_registration(
                raise_errors=False
            )
            thread = self._heartbeat_thread
            return (
                heartbeat_stopped
                and registration_released
                and (thread is None or not thread.is_alive())
            )

    def _ensure_registered(self, deadline: float) -> None:
        self._raise_heartbeat_error()
        with self._registration_lock:
            if self._registration is not None:
                self._start_heartbeat_locked()
                return
            thread = self._registration_thread
            if thread is None:
                self._registration_inflight = True
                self._registration_error = None
                thread = threading.Thread(
                    target=self._registration_worker,
                    name=f"lm-resiliency-register-{self._config.node_id}",
                    daemon=True,
                )
                self._registration_thread = thread
                thread.start()
        while thread.is_alive():
            if self._closed_event.is_set() or self._stop_heartbeat_event.is_set():
                raise RendezvousClosedError
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                raise RendezvousTimeoutError
            thread.join(
                timeout=min(
                    remaining,
                    self._config.policy.poll_interval_ms / 1_000,
                )
            )
        with self._registration_lock:
            error = self._registration_error
            registration = self._registration
        if error is not None:
            if isinstance(error, AgentRegistrationCorrupt):
                raise RendezvousStateError("agent registration state is corrupt") from error
            if isinstance(
                error,
                (
                    AgentRegistrationClockError,
                    AgentRegistrationLost,
                    AgentRegistrationUnavailable,
                ),
            ):
                raise RendezvousConnectionError(
                    "failed to acquire the agent registration"
                ) from error
            raise RendezvousConnectionError("agent registration backend failed") from error
        if registration is None:
            if self._closed_event.is_set() or self._stop_heartbeat_event.is_set():
                raise RendezvousClosedError
            raise RendezvousConnectionError(
                "agent registration completed without installing ownership"
            )

    def _registration_worker(self) -> None:
        registration: HeldAgentRegistration | None = None
        error: Exception | None = None
        try:
            registration = self._registration_manager.register()
        except Exception as registration_error:
            error = registration_error
        release_late_registration = False
        with self._registration_lock:
            self._registration_inflight = False
            self._registration_error = error
            if registration is not None:
                if self._closed_event.is_set() or self._stop_heartbeat_event.is_set():
                    release_late_registration = True
                else:
                    self._registration = registration
                    self._start_heartbeat_locked()
        self._state_changed_event.set()
        if release_late_registration and registration is not None:
            try:
                self._registration_manager.release(registration)
            except Exception:
                # The registration is lease-bounded; a failed best-effort release
                # must not keep the registration worker or shutdown path alive.
                pass

    def _start_heartbeat_locked(self) -> None:
        thread = self._heartbeat_thread
        if thread is None or not thread.is_alive():
            thread = threading.Thread(
                target=self._heartbeat_loop,
                name=f"lm-resiliency-registration-{self._config.node_id}",
                daemon=True,
            )
            thread.start()
            self._heartbeat_thread = thread

    def _heartbeat_loop(self) -> None:
        while True:
            with self._registration_lock:
                registration = self._registration
                if registration is None:
                    return
                try:
                    delay_seconds = self._renewal_delay_seconds(registration)
                except (
                    AgentRegistrationClockError,
                    AgentRegistrationLost,
                ) as error:
                    self._heartbeat_error = error
                    self._state_changed_event.set()
                    return
            if self._stop_heartbeat_event.wait(delay_seconds):
                return
            with self._registration_lock:
                registration = self._registration
                if registration is None:
                    return
                try:
                    self._registration = self._registration_manager.renew(registration)
                except (
                    AgentRegistrationClockError,
                    AgentRegistrationCorrupt,
                    AgentRegistrationLost,
                    AgentRegistrationUnavailable,
                ) as error:
                    self._heartbeat_error = error
                    self._state_changed_event.set()
                    return
                except Exception as error:
                    self._heartbeat_error = error
                    self._state_changed_event.set()
                    return

    def _renewal_delay_seconds(
        self,
        registration: HeldAgentRegistration,
    ) -> float:
        now_unix_ms = self._clock()
        if isinstance(now_unix_ms, bool) or not isinstance(now_unix_ms, int) or now_unix_ms < 1:
            raise AgentRegistrationClockError(
                "agent registration heartbeat clock returned an invalid time"
            )
        if now_unix_ms < self._heartbeat_last_now_unix_ms:
            raise AgentRegistrationClockError("agent registration heartbeat clock moved backward")
        self._heartbeat_last_now_unix_ms = now_unix_ms
        if now_unix_ms < registration.granted_at_unix_ms:
            raise AgentRegistrationClockError(
                "agent registration heartbeat clock precedes the authoritative grant"
            )
        remaining_ms = registration.expires_at_unix_ms - now_unix_ms
        if remaining_ms <= 0:
            raise AgentRegistrationLost("agent registration expired before the next heartbeat")
        return remaining_ms / 3_000

    def _stop_heartbeat(self) -> bool:
        self._stop_heartbeat_event.set()
        self._state_changed_event.set()
        thread = self._heartbeat_thread
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=self._bounded_cleanup_timeout_seconds())
            except RuntimeError:
                return False
        return thread is None or not thread.is_alive()

    def _release_registration(self, *, raise_errors: bool) -> bool:
        with self._registration_lock:
            registration = self._registration
            if registration is None:
                return True
            thread = self._release_thread
            if thread is None:
                self._release_error = None
                thread = threading.Thread(
                    target=self._release_registration_worker,
                    args=(registration,),
                    name=f"lm-resiliency-release-{self._config.node_id}",
                    daemon=True,
                )
                self._release_thread = thread
                thread.start()
        if thread is not threading.current_thread():
            thread.join(timeout=self._bounded_cleanup_timeout_seconds())
        if thread.is_alive():
            return False
        with self._registration_lock:
            error = self._release_error
            self._release_thread = None
            self._release_error = None
        if error is None:
            return True
        if raise_errors:
            if isinstance(
                error,
                (
                    AgentRegistrationClockError,
                    AgentRegistrationCorrupt,
                    AgentRegistrationLost,
                    AgentRegistrationUnavailable,
                ),
            ):
                raise RendezvousConnectionError(
                    "failed to release the agent registration"
                ) from error
            raise RendezvousConnectionError(
                "agent registration backend failed during release"
            ) from error
        return False

    def _bounded_cleanup_timeout_seconds(self) -> float:
        return max(
            self._config.policy.poll_interval_ms / 500,
            0.1,
        )

    def _release_registration_worker(
        self,
        registration: HeldAgentRegistration,
    ) -> None:
        error: Exception | None = None
        try:
            self._registration_manager.release(registration)
        except Exception as release_error:
            error = release_error
        with self._registration_lock:
            if error is None and self._registration == registration:
                self._registration = None
            self._release_error = error
        self._state_changed_event.set()

    def _raise_heartbeat_error(self) -> None:
        error = self._heartbeat_error
        if error is None:
            return
        if isinstance(error, AgentRegistrationCorrupt):
            raise RendezvousStateError(
                "agent registration heartbeat found corrupt state"
            ) from error
        raise RendezvousConnectionError("agent registration heartbeat failed") from error

    def _read_generation(self) -> CurrentGeneration | None:
        try:
            return self._generation_reader.current()
        except GenerationStateCorrupt as error:
            raise RendezvousStateError("generation state is corrupt") from error
        except GenerationStateError as error:
            raise RendezvousConnectionError(
                "generation state changed repeatedly during rendezvous"
            ) from error
        except Exception as error:
            raise RendezvousConnectionError("generation backend failed") from error

    def _bootstrap_store(self, generation: int) -> Store:
        prefix = (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/bootstrap/"
        )
        return PrefixStore(prefix, self._rendezvous_store)

    def _build_bootstrap(
        self,
        slot: int,
        store: Store,
        deadline: float,
    ) -> RendezvousStoreInfo:
        bootstrap = self._bootstrap_store_info
        if bootstrap is not None:
            store.set_timeout(timedelta(milliseconds=self._config.policy.join_timeout_ms))
            return bootstrap
        remaining = deadline - self._monotonic_clock()
        if remaining <= 0:
            raise RendezvousTimeoutError
        try:
            store.set_timeout(timedelta(seconds=max(remaining, 0.001)))
            keys = [
                RendezvousStoreInfo.MASTER_ADDR_KEY,
                RendezvousStoreInfo.MASTER_PORT_KEY,
            ]
            if slot == 0 and not store.check(keys):
                bootstrap = RendezvousStoreInfo.build(
                    slot,
                    store,
                    self._config.local_addr,
                )
            else:
                bootstrap = RendezvousStoreInfo(
                    store.get(RendezvousStoreInfo.MASTER_ADDR_KEY).decode("utf-8"),
                    int(store.get(RendezvousStoreInfo.MASTER_PORT_KEY).decode("utf-8")),
                )
            store.set_timeout(timedelta(milliseconds=self._config.policy.join_timeout_ms))
            return bootstrap
        except DistStoreError as error:
            raise RendezvousTimeoutError from error
        except Exception as error:
            if self._monotonic_clock() >= deadline:
                raise RendezvousTimeoutError from error
            raise RendezvousConnectionError(
                "failed to publish rendezvous bootstrap information"
            ) from error

    def _read_closure_entry(self) -> ControlStoreEntry | None:
        entry = self._control_store.get(self._closure_key)
        if entry is not None:
            self._validate_closure_entry(entry)
            return entry
        if not self._control_store.has_history(self._closure_key):
            return None
        entry = self._control_store.get(self._closure_key)
        if entry is None:
            raise RendezvousStateError("rendezvous closure record was deleted after publication")
        self._validate_closure_entry(entry)
        return entry

    def _validate_assigned_registration_history(self) -> None:
        try:
            history = AgentRegistrationHistoryReader(
                self._control_store,
                run_id=self._config.run_id,
                node_id=self._config.node_id,
            ).read()
        except AgentRegistrationHistoryCorrupt as error:
            raise RendezvousStateError("assigned agent registration history is corrupt") from error
        except AgentRegistrationHistoryError as error:
            raise RendezvousConnectionError(
                "assigned agent registration history changed repeatedly"
            ) from error
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read assigned agent registration history"
            ) from error
        with self._registration_lock:
            registration = self._registration
        if (
            registration is None
            or history.current is None
            or history.current.record != registration.record
        ):
            raise RendezvousConnectionError(
                "assigned node registration no longer belongs to this handler"
            )
        for authority in history.authorities:
            identity = authority.registration.record.agent_identity
            if (
                identity.environment_digest != self._agent_identity.environment_digest
                or identity.local_world_size != self._agent_identity.local_world_size
            ):
                raise RendezvousStateError(
                    "assigned node registration history is incompatible with "
                    "the committed workload environment"
                )

    def _wait_for_assigned_arrivals(
        self,
        current: CurrentGeneration,
        slot: int,
        deadline: float,
    ) -> tuple[int, ControlStoreEntry]:
        assignment = current.snapshot.record.assignment
        with self._registration_lock:
            registration = self._registration
        if registration is None:
            raise RendezvousConnectionError(
                "assigned node lost its registration before the arrival barrier"
            )
        attempt = self._claim_shared_arrival_attempt(
            assignment=assignment,
            slot=slot,
            registration=registration,
            deadline=deadline,
        )
        completion = self._read_arrival_completion(
            assignment,
            attempt,
            None,
        )
        if completion is not None:
            return attempt, completion
        arrival_value = self._arrival_value(
            generation=assignment.generation,
            attempt=attempt,
            slot=slot,
            registration=registration,
        )
        arrival_key = self._arrival_key(
            assignment.generation,
            attempt,
            slot,
        )
        try:
            entry = self._control_store.get(arrival_key)
            if entry is None:
                try:
                    entry = self._control_store.compare_set(
                        arrival_key,
                        expected_revision=None,
                        value=arrival_value,
                    )
                except ControlStoreConflict:
                    entry = self._control_store.get(arrival_key)
                    if entry is None:
                        raise RendezvousStateError(
                            "rendezvous arrival changed without a current record"
                        )
            elif entry.value != arrival_value:
                completion = self._read_arrival_completion(
                    assignment,
                    attempt,
                    None,
                )
                if completion is not None:
                    raise RendezvousConnectionError(
                        "rendezvous attempt completed before this agent arrived"
                    )
                try:
                    entry = self._control_store.compare_set(
                        arrival_key,
                        expected_revision=entry.revision,
                        value=arrival_value,
                    )
                except ControlStoreConflict as error:
                    raise RendezvousConnectionError(
                        "rendezvous arrival changed during agent replacement"
                    ) from error
            self._validate_arrival_record(
                entry,
                generation=assignment.generation,
                attempt=attempt,
                slot=slot,
                node_id=self._config.node_id,
            )
        except RendezvousStateError:
            raise
        except Exception as error:
            raise RendezvousConnectionError("failed to publish rendezvous arrival") from error

        while True:
            if self.is_closed():
                raise RendezvousClosedError
            self._raise_heartbeat_error()
            completion = self._read_arrival_completion(
                assignment,
                attempt,
                None,
            )
            if completion is not None:
                return attempt, completion
            if slot == 0:
                self._try_complete_arrival_attempt(
                    assignment,
                    attempt,
                    registration,
                )
            if not self._wait_for_change(deadline):
                raise RendezvousTimeoutError

    def _claim_shared_arrival_attempt(
        self,
        *,
        assignment: RankAssignment,
        slot: int,
        registration: HeldAgentRegistration,
        deadline: float,
    ) -> int:
        generation = assignment.generation
        key = self._arrival_attempt_key(generation)
        while True:
            if self.is_closed():
                raise RendezvousClosedError
            self._raise_heartbeat_error()
            try:
                entry = self._control_store.get(key)
                if entry is None:
                    if slot != 0:
                        if not self._wait_for_change(deadline):
                            raise RendezvousTimeoutError
                        continue
                    attempt = 1
                    value = self._arrival_attempt_value(
                        generation=generation,
                        attempt=attempt,
                        registration=registration,
                    )
                    try:
                        committed = self._control_store.compare_set(
                            key,
                            expected_revision=None,
                            value=value,
                        )
                    except ControlStoreConflict:
                        continue
                else:
                    attempt, _, _ = self._validate_arrival_attempt_entry(
                        entry,
                        generation=generation,
                        leader_node_id=assignment.slot_to_node_id[0],
                    )
                    completion = self._read_arrival_completion(
                        assignment,
                        attempt,
                        None,
                    )
                    own_consumption = None
                    if completion is not None:
                        own_consumption = self._read_arrival_consumption(
                            assignment,
                            attempt,
                            slot,
                            completion,
                            registration,
                        )
                    if completion is not None and own_consumption is None:
                        committed = entry
                    elif (
                        slot == 0
                        and completion is not None
                        and self._arrival_consumption_conditions(
                            assignment,
                            attempt,
                            completion,
                        )
                        is not None
                    ):
                        next_attempt = attempt + 1
                        value = self._arrival_attempt_value(
                            generation=generation,
                            attempt=next_attempt,
                            registration=registration,
                        )
                        advanced = self._advance_arrival_attempt(
                            assignment,
                            entry,
                            attempt,
                            next_attempt,
                            value,
                            registration,
                            completion,
                        )
                        if advanced is None:
                            continue
                        committed = advanced
                        attempt = next_attempt
                    elif completion is not None:
                        if not self._wait_for_change(deadline):
                            raise RendezvousTimeoutError
                        continue
                    else:
                        committed = entry
                self._validate_arrival_attempt_entry(
                    committed,
                    generation=generation,
                    expected_attempt=attempt,
                    leader_node_id=assignment.slot_to_node_id[0],
                )
                return attempt
            except (RendezvousClosedError, RendezvousStateError, RendezvousTimeoutError):
                raise
            except Exception as error:
                raise RendezvousConnectionError(
                    "failed to claim a rendezvous arrival attempt"
                ) from error

    def _advance_arrival_attempt(
        self,
        assignment: RankAssignment,
        current_entry: ControlStoreEntry,
        current_attempt: int,
        next_attempt: int,
        value: bytes,
        registration: HeldAgentRegistration,
        completion: ControlStoreEntry,
    ) -> ControlStoreEntry | None:
        conditions = self._arrival_consumption_conditions(
            assignment,
            current_attempt,
            completion,
        )
        if conditions is None:
            return None
        completion_key = self._arrival_completion_key(
            assignment.generation,
            current_attempt,
        )
        conditions[completion_key] = completion.revision
        with self._registration_lock:
            current_registration = self._registration
        if current_registration is None or current_registration.record != registration.record:
            raise RendezvousConnectionError("arrival leader no longer owns its agent registration")
        now_unix_ms = self._clock()
        if (
            isinstance(now_unix_ms, bool)
            or not isinstance(now_unix_ms, int)
            or now_unix_ms < current_registration.granted_at_unix_ms
            or now_unix_ms >= current_registration.expires_at_unix_ms
        ):
            raise RendezvousConnectionError(
                "rendezvous attempt advance is outside the registration window"
            )
        attempt_key = self._arrival_attempt_key(assignment.generation)
        try:
            result = self._control_store.compare_set_many_guarded(
                {
                    attempt_key: ControlStoreWrite(
                        expected_revision=current_entry.revision,
                        value=value,
                    )
                },
                guard_key=self._registration_manager.registration_key,
                expected_guard_revision=current_registration.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=current_registration.expires_at_unix_ms,
                conditions=conditions,
            )
        except ControlStoreConflict:
            return None
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to advance the rendezvous arrival attempt"
            ) from error
        committed = result[attempt_key]
        self._validate_arrival_attempt_entry(
            committed,
            generation=assignment.generation,
            expected_attempt=next_attempt,
            leader_node_id=assignment.slot_to_node_id[0],
        )
        self._state_changed_event.set()
        return committed

    def _arrival_attempt_key(self, generation: int) -> str:
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/arrival-attempt"
        )

    def _arrival_attempt_value(
        self,
        *,
        generation: int,
        attempt: int,
        registration: HeldAgentRegistration,
    ) -> bytes:
        return json.dumps(
            {
                "attempt": attempt,
                "generation": generation,
                "leader_agent_id": registration.record.agent_identity.agent_id,
                "leader_registration_id": registration.record.registration_id,
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_ATTEMPT_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_arrival_attempt_entry(
        self,
        entry: ControlStoreEntry,
        *,
        generation: int,
        expected_attempt: int | None = None,
        leader_node_id: str | None = None,
    ) -> tuple[int, str, str]:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("rendezvous arrival attempt record is malformed") from error
        expected_fields = {
            "attempt",
            "generation",
            "leader_agent_id",
            "leader_registration_id",
            "run_id",
            "schema_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise RendezvousStateError("rendezvous arrival attempt record has invalid fields")
        attempt = payload["attempt"]
        if (
            isinstance(attempt, bool)
            or not isinstance(attempt, int)
            or attempt < 1
            or isinstance(payload["generation"], bool)
            or not isinstance(payload["generation"], int)
            or payload["schema_version"] != _ARRIVAL_ATTEMPT_SCHEMA_VERSION
            or isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["run_id"] != self._config.run_id
            or payload["generation"] != generation
            or (expected_attempt is not None and attempt != expected_attempt)
        ):
            raise RendezvousStateError(
                "rendezvous arrival attempt record does not match its generation"
            )
        for field in ("leader_agent_id", "leader_registration_id"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise RendezvousStateError(f"rendezvous arrival attempt record has invalid {field}")
        expected_guard_key = (
            None
            if attempt == 1 or leader_node_id is None
            else agent_registration_key(self._config.run_id, leader_node_id)
        )
        if (
            entry.mutation_sequence != attempt
            or entry.value_sequence != attempt
            or entry.lifetime_sequence != 1
            or (attempt == 1 and entry.guard_key is not None)
            or (
                attempt > 1 and leader_node_id is not None and entry.guard_key != expected_guard_key
            )
        ):
            raise RendezvousStateError(
                "rendezvous arrival attempt record has invalid store provenance"
            )
        return (
            attempt,
            payload["leader_registration_id"],
            payload["leader_agent_id"],
        )

    def _arrival_key(self, generation: int, attempt: int, slot: int) -> str:
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/attempt-{attempt}/arrivals/slot-{slot}"
        )

    def _arrival_value(
        self,
        *,
        generation: int,
        attempt: int,
        slot: int,
        registration: HeldAgentRegistration,
    ) -> bytes:
        return json.dumps(
            {
                "agent_id": registration.record.agent_identity.agent_id,
                "attempt": attempt,
                "generation": generation,
                "logical_node_slot": slot,
                "node_id": registration.record.agent_identity.node_id,
                "registration_id": registration.record.registration_id,
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_arrival_record(
        self,
        entry: ControlStoreEntry,
        *,
        generation: int,
        attempt: int,
        slot: int,
        node_id: str,
    ) -> tuple[str, str]:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("rendezvous arrival record is malformed") from error
        expected_fields = {
            "agent_id",
            "attempt",
            "generation",
            "logical_node_slot",
            "node_id",
            "registration_id",
            "run_id",
            "schema_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise RendezvousStateError("rendezvous arrival record has invalid fields")
        if (
            isinstance(payload["generation"], bool)
            or not isinstance(payload["generation"], int)
            or isinstance(payload["attempt"], bool)
            or not isinstance(payload["attempt"], int)
            or isinstance(payload["logical_node_slot"], bool)
            or not isinstance(payload["logical_node_slot"], int)
            or payload["schema_version"] != _ARRIVAL_SCHEMA_VERSION
            or isinstance(payload["schema_version"], bool)
            or not isinstance(payload["schema_version"], int)
            or payload["run_id"] != self._config.run_id
            or payload["generation"] != generation
            or payload["attempt"] != attempt
            or payload["logical_node_slot"] != slot
            or payload["node_id"] != node_id
        ):
            raise RendezvousStateError("rendezvous arrival record does not match its assigned slot")
        for field in ("agent_id", "registration_id"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise RendezvousStateError(f"rendezvous arrival record has invalid {field}")
        if entry.lifetime_sequence != 1 or entry.guard_key is not None:
            raise RendezvousStateError("rendezvous arrival record has invalid store provenance")
        return payload["registration_id"], payload["agent_id"]

    def _validate_arrived_registration(
        self,
        *,
        node_id: str,
        registration_id: str,
        agent_id: str,
    ) -> HeldAgentRegistration:
        current = self._current_assigned_registration(node_id)
        identity = current.record.agent_identity
        if current.record.registration_id != registration_id or identity.agent_id != agent_id:
            raise RendezvousStateError(
                "rendezvous arrival does not match the current agent registration"
            )
        return current

    def _current_assigned_registration(
        self,
        node_id: str,
    ) -> HeldAgentRegistration:
        try:
            history = AgentRegistrationHistoryReader(
                self._control_store,
                run_id=self._config.run_id,
                node_id=node_id,
            ).read()
        except AgentRegistrationHistoryCorrupt as error:
            raise RendezvousStateError("arrived agent registration history is corrupt") from error
        except AgentRegistrationHistoryError as error:
            raise RendezvousConnectionError(
                "arrived agent registration history changed repeatedly"
            ) from error
        current = history.current
        if current is None:
            raise RendezvousConnectionError("arrived agent no longer has a current registration")
        now_unix_ms = self._clock()
        if (
            isinstance(now_unix_ms, bool)
            or not isinstance(now_unix_ms, int)
            or now_unix_ms < current.granted_at_unix_ms
        ):
            raise RendezvousConnectionError(
                "rendezvous arrival clock is invalid for its registration"
            )
        if current.expires_at_unix_ms <= now_unix_ms:
            raise RendezvousConnectionError("arrived agent registration has expired")
        return current

    def _validate_final_admission_state(
        self,
        current: CurrentGeneration,
        recovery_state: RestartPlanPersistedRecoveryState | None = None,
    ) -> None:
        if self._is_closed_bounded(None):
            raise RendezvousClosedError
        if self._read_generation() != current:
            raise RendezvousConnectionError("generation changed before rendezvous admission")
        if recovery_state is None:
            return
        refreshed = self._read_recovery_state()
        if refreshed != recovery_state:
            raise RendezvousConnectionError(
                "restart-plan recovery state changed before rendezvous admission"
            )
        self._replacement_deadline(refreshed.plan, None)
        expected_context = self._restart_context_for_plan(refreshed.plan)
        try:
            persisted_context = self._restart_context.read()
        except (OSError, RestartContextFileError) as error:
            raise RendezvousStateError(
                "failed to validate the replacement restart context"
            ) from error
        if persisted_context != expected_context:
            raise RendezvousStateError(
                "replacement restart context changed before rendezvous admission"
            )

    def _try_complete_arrival_attempt(
        self,
        assignment: RankAssignment,
        attempt: int,
        leader_registration: HeldAgentRegistration,
    ) -> None:
        with self._registration_lock:
            current_leader_registration = self._registration
        if (
            current_leader_registration is None
            or current_leader_registration.record != leader_registration.record
        ):
            raise RendezvousConnectionError("arrival leader no longer owns its agent registration")
        leader_registration = current_leader_registration
        completion_key = self._arrival_completion_key(
            assignment.generation,
            attempt,
        )
        if self._control_store.get(completion_key) is not None:
            return
        arrivals: dict[str, dict[str, Any]] = {}
        conditions: dict[str, int | None] = {}
        deadline_unix_ms = leader_registration.expires_at_unix_ms
        attempt_key = self._arrival_attempt_key(assignment.generation)
        try:
            attempt_entry = self._control_store.get(attempt_key)
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read the rendezvous arrival attempt"
            ) from error
        if attempt_entry is None:
            raise RendezvousStateError("rendezvous arrival attempt disappeared before completion")
        self._validate_arrival_attempt_entry(
            attempt_entry,
            generation=assignment.generation,
            expected_attempt=attempt,
            leader_node_id=assignment.slot_to_node_id[0],
        )
        conditions[attempt_key] = attempt_entry.revision
        for assigned_slot, node_id in assignment.slot_to_node_id.items():
            arrival_key = self._arrival_key(
                assignment.generation,
                attempt,
                assigned_slot,
            )
            try:
                entry = self._control_store.get(arrival_key)
            except Exception as error:
                raise RendezvousConnectionError("failed to read rendezvous arrivals") from error
            if entry is None:
                return
            registration_id, agent_id = self._validate_arrival_record(
                entry,
                generation=assignment.generation,
                attempt=attempt,
                slot=assigned_slot,
                node_id=node_id,
            )
            current_registration = self._validate_arrived_registration(
                node_id=node_id,
                registration_id=registration_id,
                agent_id=agent_id,
            )
            registration_key = agent_registration_key(
                self._config.run_id,
                node_id,
            )
            conditions[arrival_key] = entry.revision
            if registration_key != self._registration_manager.registration_key:
                conditions[registration_key] = current_registration.fencing_token
            deadline_unix_ms = min(
                deadline_unix_ms,
                current_registration.expires_at_unix_ms,
            )
            arrivals[str(assigned_slot)] = {
                "agent_id": agent_id,
                "arrival_revision": entry.revision,
                "node_id": node_id,
                "registration_id": registration_id,
                "registration_revision": current_registration.fencing_token,
            }
        now_unix_ms = self._clock()
        if (
            isinstance(now_unix_ms, bool)
            or not isinstance(now_unix_ms, int)
            or now_unix_ms < leader_registration.granted_at_unix_ms
            or now_unix_ms >= deadline_unix_ms
        ):
            raise RendezvousConnectionError(
                "rendezvous completion clock is outside the registration window"
            )
        value = self._arrival_completion_value(
            generation=assignment.generation,
            attempt=attempt,
            arrivals=arrivals,
        )
        try:
            result = self._control_store.compare_set_many_guarded(
                {
                    completion_key: ControlStoreWrite(
                        expected_revision=None,
                        value=value,
                        require_never_created=True,
                    )
                },
                guard_key=self._registration_manager.registration_key,
                expected_guard_revision=leader_registration.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=deadline_unix_ms,
                conditions=conditions,
            )
        except ControlStoreConflict:
            return
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to commit the rendezvous arrival barrier"
            ) from error
        self._validate_arrival_completion_entry(
            result[completion_key],
            assignment=assignment,
            attempt=attempt,
            registration=leader_registration,
        )
        self._state_changed_event.set()

    def _read_arrival_completion(
        self,
        assignment: RankAssignment,
        attempt: int,
        registration: HeldAgentRegistration | None,
    ) -> ControlStoreEntry | None:
        key = self._arrival_completion_key(assignment.generation, attempt)
        try:
            entry = self._control_store.get(key)
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read the rendezvous arrival completion"
            ) from error
        if entry is None:
            return None
        self._validate_arrival_completion_entry(
            entry,
            assignment=assignment,
            attempt=attempt,
            registration=registration,
        )
        return entry

    def _arrival_completion_key(self, generation: int, attempt: int) -> str:
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/attempt-{attempt}/complete"
        )

    def _arrival_completion_value(
        self,
        *,
        generation: int,
        attempt: int,
        arrivals: Mapping[str, Mapping[str, Any]],
    ) -> bytes:
        return json.dumps(
            {
                "arrivals": arrivals,
                "attempt": attempt,
                "generation": generation,
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_COMPLETION_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_arrival_completion_entry(
        self,
        entry: ControlStoreEntry,
        *,
        assignment: RankAssignment,
        attempt: int,
        registration: HeldAgentRegistration | None,
    ) -> Mapping[str, Mapping[str, Any]]:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("rendezvous arrival completion is malformed") from error
        expected_fields = {
            "arrivals",
            "attempt",
            "generation",
            "run_id",
            "schema_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise RendezvousStateError("rendezvous arrival completion has invalid fields")
        if (
            payload["run_id"] != self._config.run_id
            or payload["generation"] != assignment.generation
            or payload["attempt"] != attempt
            or payload["schema_version"] != _ARRIVAL_COMPLETION_SCHEMA_VERSION
            or any(
                isinstance(payload[field], bool) or not isinstance(payload[field], int)
                for field in ("generation", "attempt", "schema_version")
            )
        ):
            raise RendezvousStateError("rendezvous arrival completion does not match its attempt")
        arrivals = payload["arrivals"]
        expected_slots = {
            str(slot): node_id for slot, node_id in assignment.slot_to_node_id.items()
        }
        if not isinstance(arrivals, dict) or set(arrivals) != set(expected_slots):
            raise RendezvousStateError("rendezvous arrival completion has invalid slot coverage")
        own_arrival: Mapping[str, Any] | None = None
        for slot, node_id in expected_slots.items():
            arrival = arrivals[slot]
            fields = {
                "agent_id",
                "arrival_revision",
                "node_id",
                "registration_id",
                "registration_revision",
            }
            if not isinstance(arrival, dict) or set(arrival) != fields:
                raise RendezvousStateError(
                    "rendezvous arrival completion has invalid arrival metadata"
                )
            if arrival["node_id"] != node_id:
                raise RendezvousStateError(
                    "rendezvous arrival completion has invalid node assignment"
                )
            for field in ("agent_id", "registration_id"):
                if not isinstance(arrival[field], str) or not arrival[field]:
                    raise RendezvousStateError(f"rendezvous arrival completion has invalid {field}")
            for field in ("arrival_revision", "registration_revision"):
                if (
                    isinstance(arrival[field], bool)
                    or not isinstance(arrival[field], int)
                    or arrival[field] < 1
                ):
                    raise RendezvousStateError(f"rendezvous arrival completion has invalid {field}")
            if node_id == self._config.node_id:
                own_arrival = arrival
        if registration is not None and (
            own_arrival is None
            or own_arrival["registration_id"] != registration.record.registration_id
            or own_arrival["agent_id"] != registration.record.agent_identity.agent_id
        ):
            raise RendezvousConnectionError(
                "rendezvous completion does not include this agent registration"
            )
        slot_zero = arrivals["0"]
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
            or entry.guard_key
            != agent_registration_key(
                self._config.run_id,
                expected_slots["0"],
            )
            or entry.guard_revision != slot_zero["registration_revision"]
        ):
            raise RendezvousStateError("rendezvous arrival completion has invalid store provenance")
        return arrivals

    def _publish_arrival_consumption(
        self,
        current: CurrentGeneration,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        deadline: float,
    ) -> None:
        completion_key = self._arrival_completion_key(assignment.generation, attempt)
        generation_snapshot_key = self._generation_reader.snapshot_key(assignment.generation)
        while True:
            with self._registration_lock:
                registration = self._registration
            if registration is None:
                raise RendezvousConnectionError(
                    "assigned node lost its registration before consuming rendezvous"
                )
            key = self._arrival_consumption_key(
                assignment.generation,
                attempt,
                slot,
                registration.record.registration_id,
            )
            value = self._arrival_consumption_value(
                generation=assignment.generation,
                attempt=attempt,
                slot=slot,
                registration=registration,
                completion=completion,
            )
            try:
                existing = self._control_store.get(key)
            except Exception as error:
                raise RendezvousConnectionError("failed to read rendezvous consumption") from error
            if existing is not None:
                self._validate_arrival_consumption_entry(
                    existing,
                    assignment=assignment,
                    attempt=attempt,
                    slot=slot,
                    completion=completion,
                    registration=registration,
                )
                return
            now_unix_ms = self._clock()
            if (
                isinstance(now_unix_ms, bool)
                or not isinstance(now_unix_ms, int)
                or now_unix_ms < registration.granted_at_unix_ms
                or now_unix_ms >= registration.expires_at_unix_ms
            ):
                raise RendezvousConnectionError(
                    "rendezvous consumption is outside the registration window"
                )
            try:
                result = self._control_store.compare_set_many_guarded(
                    {
                        key: ControlStoreWrite(
                            expected_revision=None,
                            value=value,
                            require_never_created=True,
                        )
                    },
                    guard_key=self._registration_manager.registration_key,
                    expected_guard_revision=registration.fencing_token,
                    not_before_unix_ms=now_unix_ms,
                    deadline_unix_ms=registration.expires_at_unix_ms,
                    conditions={
                        completion_key: completion.revision,
                        self._generation_reader.head_key: current.head_revision,
                        generation_snapshot_key: current.snapshot.revision,
                        self._closure_key: None,
                    },
                )
            except ControlStoreConflict:
                try:
                    existing = self._control_store.get(key)
                except Exception as error:
                    raise RendezvousConnectionError(
                        "failed to resolve concurrent rendezvous consumption"
                    ) from error
                if existing is not None:
                    self._validate_arrival_consumption_entry(
                        existing,
                        assignment=assignment,
                        attempt=attempt,
                        slot=slot,
                        completion=completion,
                        registration=registration,
                    )
                    return
                if self._is_closed_bounded(deadline):
                    raise RendezvousClosedError
                if self._read_generation() != current:
                    raise RendezvousConnectionError(
                        "generation changed before rendezvous consumption"
                    )
                if not self._wait_for_change(deadline):
                    raise RendezvousTimeoutError
                continue
            except Exception as error:
                raise RendezvousConnectionError(
                    "failed to publish rendezvous consumption"
                ) from error
            self._validate_arrival_consumption_entry(
                result[key],
                assignment=assignment,
                attempt=attempt,
                slot=slot,
                completion=completion,
                registration=registration,
            )
            self._state_changed_event.set()
            return

    def _arrival_consumption_conditions(
        self,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
    ) -> dict[str, int] | None:
        conditions: dict[str, int] = {}
        for slot, node_id in assignment.slot_to_node_id.items():
            registration = self._current_assigned_registration(node_id)
            entry = self._read_arrival_consumption(
                assignment,
                attempt,
                slot,
                completion,
                registration,
            )
            if entry is None:
                return None
            conditions[
                self._arrival_consumption_key(
                    assignment.generation,
                    attempt,
                    slot,
                    registration.record.registration_id,
                )
            ] = entry.revision
            registration_key = agent_registration_key(self._config.run_id, node_id)
            if registration_key != self._registration_manager.registration_key:
                conditions[registration_key] = registration.fencing_token
        return conditions

    def _read_arrival_consumption(
        self,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        registration: HeldAgentRegistration,
    ) -> ControlStoreEntry | None:
        key = self._arrival_consumption_key(
            assignment.generation,
            attempt,
            slot,
            registration.record.registration_id,
        )
        try:
            entry = self._control_store.get(key)
        except Exception as error:
            raise RendezvousConnectionError("failed to read rendezvous consumption") from error
        if entry is None:
            return None
        self._validate_arrival_consumption_entry(
            entry,
            assignment=assignment,
            attempt=attempt,
            slot=slot,
            completion=completion,
            registration=registration,
        )
        return entry

    def _arrival_consumption_key(
        self,
        generation: int,
        attempt: int,
        slot: int,
        registration_id: str,
    ) -> str:
        registration_digest = hashlib.sha256(registration_id.encode("utf-8")).hexdigest()
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/attempt-{attempt}/consumed/"
            f"slot-{slot}/registration-{registration_digest}"
        )

    def _arrival_consumption_value(
        self,
        *,
        generation: int,
        attempt: int,
        slot: int,
        registration: HeldAgentRegistration,
        completion: ControlStoreEntry,
    ) -> bytes:
        return json.dumps(
            {
                "agent_id": registration.record.agent_identity.agent_id,
                "attempt": attempt,
                "completion_digest": hashlib.sha256(completion.value).hexdigest(),
                "generation": generation,
                "logical_node_slot": slot,
                "node_id": registration.record.agent_identity.node_id,
                "registration_id": registration.record.registration_id,
                "registration_revision": registration.fencing_token,
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_CONSUMPTION_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_arrival_consumption_entry(
        self,
        entry: ControlStoreEntry,
        *,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        registration: HeldAgentRegistration | None,
    ) -> None:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("rendezvous consumption is malformed") from error
        expected_fields = {
            "agent_id",
            "attempt",
            "completion_digest",
            "generation",
            "logical_node_slot",
            "node_id",
            "registration_id",
            "registration_revision",
            "run_id",
            "schema_version",
        }
        if not isinstance(payload, dict) or set(payload) != expected_fields:
            raise RendezvousStateError("rendezvous consumption has invalid fields")
        expected_node_id = assignment.slot_to_node_id.get(slot)
        expected_digest = hashlib.sha256(completion.value).hexdigest()
        if (
            expected_node_id is None
            or payload["run_id"] != self._config.run_id
            or payload["generation"] != assignment.generation
            or payload["attempt"] != attempt
            or payload["logical_node_slot"] != slot
            or payload["node_id"] != expected_node_id
            or payload["completion_digest"] != expected_digest
            or payload["schema_version"] != _ARRIVAL_CONSUMPTION_SCHEMA_VERSION
            or any(
                isinstance(payload[field], bool) or not isinstance(payload[field], int)
                for field in (
                    "generation",
                    "attempt",
                    "logical_node_slot",
                    "registration_revision",
                    "schema_version",
                )
            )
            or payload["registration_revision"] < 1
        ):
            raise RendezvousStateError(
                "rendezvous consumption does not match its completed attempt"
            )
        for field in ("agent_id", "registration_id"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise RendezvousStateError(f"rendezvous consumption has invalid {field}")
        if (
            not isinstance(payload["completion_digest"], str)
            or len(payload["completion_digest"]) != 64
        ):
            raise RendezvousStateError("rendezvous consumption has invalid completion digest")
        if registration is not None and (
            payload["registration_id"] != registration.record.registration_id
            or payload["agent_id"] != registration.record.agent_identity.agent_id
        ):
            raise RendezvousConnectionError(
                "rendezvous attempt was consumed by another agent registration"
            )
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
            or entry.guard_key
            != agent_registration_key(
                self._config.run_id,
                expected_node_id,
            )
            or entry.guard_revision != payload["registration_revision"]
        ):
            raise RendezvousStateError("rendezvous consumption has invalid store provenance")

    def _generation_admission(
        self,
        current: CurrentGeneration,
        formation_deadline: float,
    ) -> tuple[int | None, RestartPlanPersistedRecoveryState | None, float]:
        assignment = current.snapshot.record.assignment
        if assignment.active_nodes != self._config.min_nodes:
            raise RendezvousStateError(
                "generation assignment active node count does not match min_nodes"
            )
        if assignment.local_world_size != self._config.local_world_size:
            raise RendezvousStateError(
                "generation assignment local world size does not match runtime configuration"
            )
        recovery_state: RestartPlanPersistedRecoveryState | None = None
        admission_deadline = formation_deadline
        if assignment.generation > 0:
            if assignment.generation > self._config.policy.max_replacement_generations:
                raise RendezvousStateError(
                    "replacement generation exceeds the configured replacement budget"
                )
            recovery_state = self._read_recovery_state()
            if recovery_state is None:
                raise RendezvousStateError(
                    "replacement generation lacks an authoritative restart-plan publication"
                )
            plan = recovery_state.plan
            expected_assignment = RankAssignment.from_assignments(
                run_id=plan.run_id,
                generation=plan.to_generation,
                assignments=plan.slot_assignments,
                topology_digest=plan.topology_digest,
            )
            if (
                plan.run_id != self._config.run_id
                or plan.to_generation != assignment.generation
                or expected_assignment != assignment
                or plan.expected_world_size
                != self._config.min_nodes * self._config.local_world_size
            ):
                raise RendezvousStateError(
                    "replacement plan does not match the committed generation assignment"
                )
            admission_deadline = self._replacement_deadline(
                plan,
                formation_deadline,
            )
        for slot, node_id in assignment.slot_to_node_id.items():
            if node_id == self._config.node_id:
                self._ensure_job_compatibility()
                self._validate_assigned_registration_history()
                return slot, recovery_state, admission_deadline
        return None, recovery_state, admission_deadline

    def _read_recovery_state(self) -> RestartPlanPersistedRecoveryState | None:
        try:
            return self._publication_reader.read_recovery_state()
        except RestartPlanPublicationReadCorrupt as error:
            raise RendezvousStateError(
                "restart-plan publication is corrupt during replacement rendezvous"
            ) from error
        except RestartPlanPublicationReadConflict as error:
            raise RendezvousConnectionError(
                "restart-plan publication changed during replacement rendezvous"
            ) from error
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read restart-plan publication during replacement rendezvous"
            ) from error

    def _replacement_deadline(
        self,
        plan: RestartPlan,
        formation_deadline: float | None,
    ) -> float:
        now_unix_ms = self._clock()
        if isinstance(now_unix_ms, bool) or not isinstance(now_unix_ms, int) or now_unix_ms < 1:
            raise RendezvousConnectionError("replacement rendezvous clock returned an invalid time")
        remaining_seconds = (plan.restart_deadline_unix_ms - now_unix_ms) / 1_000
        if remaining_seconds <= 0:
            raise RendezvousTimeoutError
        deadline = self._monotonic_clock() + remaining_seconds
        return deadline if formation_deadline is None else min(deadline, formation_deadline)

    def _prepare_restart_context(
        self,
        current: CurrentGeneration,
        recovery_state: RestartPlanPersistedRecoveryState | None,
    ) -> bool:
        if current.snapshot.record.assignment.generation == 0:
            try:
                self._restart_context.clear()
            except (OSError, RestartContextFileError) as error:
                raise RendezvousStateError(
                    "failed to clear stale initial restart context"
                ) from error
            return False
        if recovery_state is None:
            raise RendezvousStateError("replacement generation lacks an authoritative restart plan")
        context = self._restart_context_for_plan(recovery_state.plan)
        try:
            self._restart_context.write(context)
        except (OSError, RestartContextFileError) as error:
            raise RendezvousStateError(
                "failed to publish the replacement restart context"
            ) from error
        return True

    def _restart_context_for_plan(self, plan: RestartPlan) -> RestartContext:
        try:
            return RestartContext.from_plan(plan, self._config.node_id)
        except ProtocolValidationError as error:
            raise RendezvousStateError(
                "replacement plan cannot produce this node's restart context"
            ) from error

    def _ensure_job_compatibility(self) -> None:
        try:
            entry = self._control_store.get(self._compatibility_key)
            if entry is None:
                if self._control_store.has_history(self._compatibility_key):
                    entry = self._control_store.get(self._compatibility_key)
                    if entry is None:
                        raise RendezvousStateError("rendezvous compatibility record was deleted")
                else:
                    try:
                        entry = self._control_store.compare_set(
                            self._compatibility_key,
                            expected_revision=None,
                            value=self._compatibility_value,
                        )
                    except ControlStoreConflict:
                        entry = self._control_store.get(self._compatibility_key)
                        if entry is None:
                            raise RendezvousStateError(
                                "rendezvous compatibility changed without a current record"
                            )
            if entry.value != self._compatibility_value:
                raise RendezvousStateError(
                    "assigned node is incompatible with the committed workload environment"
                )
            if (
                entry.mutation_sequence != 1
                or entry.value_sequence != 1
                or entry.lifetime_sequence != 1
                or entry.guard_key is not None
            ):
                raise RendezvousStateError("rendezvous compatibility record is not immutable")
        except RendezvousStateError:
            raise
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to validate rendezvous workload compatibility"
            ) from error

    def _wait_for_change(self, deadline: float | None) -> bool:
        if deadline is None:
            timeout = self._config.policy.poll_interval_ms / 1_000
        else:
            remaining = deadline - self._monotonic_clock()
            if remaining <= 0:
                return False
            timeout = min(
                remaining,
                self._config.policy.poll_interval_ms / 1_000,
            )
        self._state_changed_event.wait(timeout)
        self._state_changed_event.clear()
        return deadline is None or self._monotonic_clock() < deadline

    def _validate_closure_entry(self, entry: ControlStoreEntry) -> None:
        if entry.value != self._closure_value:
            raise RendezvousStateError("rendezvous closure record is malformed")
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
        ):
            raise RendezvousStateError("rendezvous closure record is not immutable")
        if entry.guard_key is not None:
            raise RendezvousStateError("rendezvous closure record must be unguarded")


@dataclass(frozen=True, slots=True)
class RestartContextFile:
    """Atomically persist and validate one owner-only restart context."""

    path: Path

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if not self.path.is_absolute():
            raise RestartContextFileError("restart-context path must be absolute")
        if self.path.name in {"", ".", ".."}:
            raise RestartContextFileError("restart-context path must name a file")

    def write(self, context: RestartContext) -> None:
        if not isinstance(context, RestartContext):
            raise TypeError("context must be RestartContext")
        encoded = (context.to_json() + "\n").encode("utf-8")
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise RestartContextFileError("restart context is too large")
        parent_descriptor = self._open_parent(create=True)
        assert parent_descriptor is not None
        descriptor = -1
        temporary = ""
        try:
            self._reject_existing_symlink(parent_descriptor)
            descriptor, temporary = self._create_temporary_file(parent_descriptor)
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary,
                self.path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary = ""
            self._fsync_directory(parent_descriptor)
        except OSError as error:
            raise RestartContextFileError(
                f"failed to publish restart context at {self.path}"
            ) from error
        finally:
            if descriptor >= 0:
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            if temporary:
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)

    def read(self) -> RestartContext:
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            raise RestartContextFileError(
                f"restart-context directory does not exist for {self.path}"
            )
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                self.path.name,
                flags,
                dir_fd=parent_descriptor,
            )
        except OSError as error:
            os.close(parent_descriptor)
            raise RestartContextFileError(
                f"failed to open restart context at {self.path}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            self._validate_owned_private_file(metadata)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                encoded = stream.read(_MAX_CONTEXT_BYTES + 1)
            if len(encoded) > _MAX_CONTEXT_BYTES:
                raise RestartContextFileError("restart-context file is too large")
        finally:
            os.close(descriptor)
            os.close(parent_descriptor)
        try:
            value = json.loads(
                encoded,
                object_pairs_hook=_strict_object,
            )
            if not isinstance(value, Mapping):
                raise RestartContextFileError("restart-context JSON must contain an object")
            return RestartContext.from_dict(value)
        except _DuplicateJsonFieldError as error:
            raise RestartContextFileError(str(error)) from error
        except RestartContextFileError:
            raise
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise RestartContextFileError("restart-context file is malformed") from error

    def clear(self) -> None:
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            return
        try:
            self._reject_existing_symlink(parent_descriptor)
            try:
                os.unlink(self.path.name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                self._fsync_directory(parent_descriptor)
                return
            except OSError as error:
                raise RestartContextFileError(
                    f"failed to remove restart context at {self.path}"
                ) from error
            self._fsync_directory(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def _open_parent(self, *, create: bool) -> int | None:
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            current = os.open(os.sep, directory_flags)
        except OSError as error:
            raise RestartContextFileError(
                "failed to open the filesystem root for restart-context traversal"
            ) from error
        try:
            for component in self.path.parent.parts[1:]:
                if component in {"", ".", ".."}:
                    raise RestartContextFileError("restart-context parent path is not canonical")
                try:
                    following = os.open(
                        component,
                        directory_flags,
                        dir_fd=current,
                    )
                except FileNotFoundError:
                    if not create:
                        return None
                    try:
                        os.mkdir(component, mode=0o700, dir_fd=current)
                    except FileExistsError:
                        pass
                    except OSError as error:
                        raise RestartContextFileError(
                            f"failed to create restart-context directory component {component!r}"
                        ) from error
                    try:
                        following = os.open(
                            component,
                            directory_flags,
                            dir_fd=current,
                        )
                    except OSError as error:
                        raise RestartContextFileError(
                            "restart-context parent path must contain only real directories"
                        ) from error
                except OSError as error:
                    raise RestartContextFileError(
                        "restart-context parent path must contain only real directories"
                    ) from error
                os.close(current)
                current = following
            self._validate_owned_private_directory(os.fstat(current))
            result = current
            current = -1
            return result
        finally:
            if current >= 0:
                os.close(current)

    @staticmethod
    def _validate_owned_private_directory(metadata: os.stat_result) -> None:
        if not stat.S_ISDIR(metadata.st_mode):
            raise RestartContextFileError("restart-context parent is not a directory")
        if metadata.st_uid != os.geteuid():
            raise RestartContextFileError("restart-context directory is owned by another user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RestartContextFileError(
                "restart-context directory must not grant group or other permissions"
            )

    def _reject_existing_symlink(self, parent_descriptor: int) -> None:
        try:
            metadata = os.stat(
                self.path.name,
                dir_fd=parent_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        except OSError as error:
            raise RestartContextFileError(
                f"failed to inspect restart context at {self.path}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RestartContextFileError("restart-context path must not be a symlink")

    def _create_temporary_file(self, parent_descriptor: int) -> tuple[int, str]:
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            name = f".{self.path.name}.{secrets.token_hex(16)}.tmp"
            try:
                descriptor = os.open(
                    name,
                    flags,
                    0o600,
                    dir_fd=parent_descriptor,
                )
            except FileExistsError:
                continue
            except OSError as error:
                raise RestartContextFileError(
                    f"failed to create temporary restart context for {self.path}"
                ) from error
            return descriptor, name
        raise RestartContextFileError(
            f"failed to allocate a temporary restart context for {self.path}"
        )

    @staticmethod
    def _validate_owned_private_file(metadata: os.stat_result) -> None:
        if not stat.S_ISREG(metadata.st_mode):
            raise RestartContextFileError("restart-context path is not a regular file")
        if metadata.st_uid != os.geteuid():
            raise RestartContextFileError("restart-context file is owned by another user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RestartContextFileError(
                "restart-context file must not grant group or other permissions"
            )

    @staticmethod
    def _fsync_directory(descriptor: int) -> None:
        os.fsync(descriptor)


def _load_policy_values(path: Path) -> dict[str, object]:
    descriptor = -1
    flags = (
        os.O_RDONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    try:
        descriptor = os.open(path, flags)
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise TorchrunRuntimeConfigError("rendezvous config path must be a regular file")
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = -1
            encoded = stream.read(_MAX_POLICY_BYTES + 1)
        if len(encoded) > _MAX_POLICY_BYTES:
            raise TorchrunRuntimeConfigError("rendezvous config file is too large")
        value = _toml.loads(encoded.decode("utf-8"))
    except TorchrunRuntimeConfigError:
        raise
    except (OSError, UnicodeDecodeError, _toml.TOMLDecodeError) as error:
        raise TorchrunRuntimeConfigError(f"failed to load rendezvous config {path}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    if not isinstance(value, dict):
        raise TorchrunRuntimeConfigError("rendezvous config must contain a TOML table")
    actual_schema = value.get("schema_version")
    if type(actual_schema) is not int or actual_schema != 1:
        raise TorchrunRuntimeConfigError(
            f"schema_version must be the integer 1, got {actual_schema!r}"
        )
    unknown = set(value) - _SHARED_FIELDS - {"schema_version"}
    if unknown:
        raise TorchrunRuntimeConfigError(f"unknown rendezvous config fields: {sorted(unknown)!r}")
    return {field: value[field] for field in _SHARED_FIELDS if field in value}


def _policy_from_values(values: Mapping[str, object]) -> TorchrunRendezvousPolicy:
    try:
        return TorchrunRendezvousPolicy(
            control_endpoint=_nonempty_string(
                values.get("control_endpoint"),
                "control_endpoint",
            ),
            replacement_only=_boolean(
                values.get("replacement_only", True),
                "replacement_only",
            ),
            max_replacement_generations=_integer(
                values.get("max_replacement_generations", 2),
                "max_replacement_generations",
            ),
            registration_lease_duration_ms=_integer(
                values.get("registration_lease_duration_ms", 30_000),
                "registration_lease_duration_ms",
            ),
            poll_interval_ms=_integer(
                values.get("poll_interval_ms", 1_000),
                "poll_interval_ms",
            ),
            join_timeout_ms=_integer(
                values.get("join_timeout_ms", 300_000),
                "join_timeout_ms",
            ),
        )
    except TorchrunRuntimeConfigError:
        raise
    except (TypeError, ValueError) as error:
        raise TorchrunRuntimeConfigError("rendezvous policy is invalid") from error


def _canonical_digest(value: Mapping[str, object]) -> str:
    encoded = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _node_value(
    parameter: object,
    environment: object,
    parameter_name: str,
    environment_name: str,
) -> str:
    parameter_value = None if parameter is None else _nonempty_string(parameter, parameter_name)
    environment_value = (
        None if environment is None else _nonempty_string(environment, environment_name)
    )
    if (
        parameter_value is not None
        and environment_value is not None
        and parameter_value != environment_value
    ):
        raise TorchrunRuntimeConfigError(f"{parameter_name} conflicts with {environment_name}")
    value = parameter_value or environment_value
    if value is None:
        raise TorchrunRuntimeConfigError(f"{parameter_name} or {environment_name} must be provided")
    return value


def _node_integer_value(
    parameter: object,
    environment: object,
    parameter_name: str,
    environment_name: str,
) -> int:
    parameter_value = None if parameter is None else _positive_integer(parameter, parameter_name)
    environment_value = (
        None if environment is None else _positive_integer(environment, environment_name)
    )
    if (
        parameter_value is not None
        and environment_value is not None
        and parameter_value != environment_value
    ):
        raise TorchrunRuntimeConfigError(f"{parameter_name} conflicts with {environment_name}")
    value = parameter_value if parameter_value is not None else environment_value
    if value is None:
        raise TorchrunRuntimeConfigError(f"{parameter_name} or {environment_name} must be provided")
    return value


def _node_resource_ids(
    parameter: object,
    environment: object,
) -> tuple[str, ...]:
    parameter_value = None if parameter is None else _resource_ids(parameter, "resource_ids")
    environment_value = (
        None if environment is None else _resource_ids(environment, _RESOURCE_IDS_ENV)
    )
    if (
        parameter_value is not None
        and environment_value is not None
        and parameter_value != environment_value
    ):
        raise TorchrunRuntimeConfigError(f"resource_ids conflicts with {_RESOURCE_IDS_ENV}")
    value = parameter_value if parameter_value is not None else environment_value
    if value is None:
        raise TorchrunRuntimeConfigError(f"resource_ids or {_RESOURCE_IDS_ENV} must be provided")
    return value


def _resource_ids(value: object, path: str) -> tuple[str, ...]:
    items: Sequence[object]
    if isinstance(value, str):
        normalized = value.strip()
        if normalized == "[]":
            items = ()
        else:
            if normalized.startswith("[") or normalized.endswith("]"):
                raise TorchrunRuntimeConfigError(
                    f"{path} must be a semicolon-delimited resource list or []"
                )
            items = tuple(normalized.split(";"))
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (bytes, bytearray),
    ):
        items = value
    else:
        raise TorchrunRuntimeConfigError(
            f"{path} must be a semicolon-delimited resource list or sequence"
        )
    result = tuple(
        _nonempty_string(item, f"{path}[{index}]").strip() for index, item in enumerate(items)
    )
    if len(result) != len(set(result)):
        raise TorchrunRuntimeConfigError(f"{path} resource IDs must be unique")
    return tuple(sorted(result))


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateJsonFieldError(f"JSON contains duplicate field {key!r}")
        value[key] = item
    return value


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TorchrunRuntimeConfigError(f"{path} must be a non-empty string")
    return value


def _integer(value: object, path: str) -> int:
    if isinstance(value, bool):
        raise TorchrunRuntimeConfigError(f"{path} must be an integer")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise TorchrunRuntimeConfigError(f"{path} must be an integer")
    try:
        return int(value)
    except ValueError as error:
        raise TorchrunRuntimeConfigError(f"{path} must be an integer") from error


def _positive_integer(value: object, path: str) -> int:
    parsed = _integer(value, path)
    if parsed < 1:
        raise TorchrunRuntimeConfigError(f"{path} must be positive")
    return parsed


def _boolean(value: object, path: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise TorchrunRuntimeConfigError(f"{path} must be a boolean")


__all__ = [
    "RestartContextFile",
    "RestartContextFileError",
    "SlotAwareRendezvousHandler",
    "TorchrunRendezvousPolicy",
    "TorchrunRuntimeConfig",
    "TorchrunRuntimeConfigError",
]
