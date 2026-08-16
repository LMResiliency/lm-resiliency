"""Runtime configuration and node-local restart-context handoff for torchrun."""

from __future__ import annotations

import errno
import fcntl
import hashlib
import importlib
import json
import math
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
    AgentRegistrationHistory,
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
    ControlStoreDeadlineExceeded,
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
_MAX_CONTEXT_METADATA_BYTES = 1_024
_MAX_CONTEXT_INVALIDATION_BYTES = 1_024
_MAX_POLICY_BYTES = 64 * 1024
_CONTEXT_METADATA_SCHEMA_VERSION = 2
_CONTEXT_INVALIDATION_SCHEMA_VERSION = 1
_CONTEXT_LOCK_TIMEOUT_SECONDS = 1.0
_CONTEXT_LOCK_POLL_SECONDS = 0.01
_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_CLOSED_SCHEMA_VERSION = 1
_ARRIVAL_SCHEMA_VERSION = 1
_ARRIVAL_ATTEMPT_SCHEMA_VERSION = 1
_ARRIVAL_COMPLETION_SCHEMA_VERSION = 1
_ARRIVAL_CONSUMPTION_SCHEMA_VERSION = 1
_ARRIVAL_ADMISSION_SCHEMA_VERSION = 1
_ARRIVAL_RETURN_SCHEMA_VERSION = 2
_SUPPORTED_REPLACEMENT_GENERATIONS = 1
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


_RestartContextCleanupToken = str


@dataclass(frozen=True, slots=True)
class _RestartContextOwnership:
    registration_id: str
    mutation_sequence: int

    def __post_init__(self) -> None:
        if not isinstance(self.registration_id, str) or not self.registration_id:
            raise TypeError("registration_id must be a non-empty string")
        if (
            isinstance(self.mutation_sequence, bool)
            or not isinstance(self.mutation_sequence, int)
            or self.mutation_sequence < 1
        ):
            raise TypeError("mutation_sequence must be a positive integer")


class RestartContextFileError(RuntimeError):
    """Raised when the node-local restart-context file is unsafe or malformed."""

    def __init__(
        self,
        message: str,
        *,
        published_token: _RestartContextCleanupToken | None = None,
    ) -> None:
        super().__init__(message)
        self.published_token = published_token


class _DuplicateJsonFieldError(ValueError):
    """Raised when strict JSON decoding observes a duplicate object field."""


@dataclass(frozen=True, slots=True)
class TorchrunRendezvousPolicy:
    """Shared replacement-only policy resolved identically by every agent."""

    SCHEMA_VERSION: ClassVar[int] = 1

    control_endpoint: str
    replacement_only: bool = True
    max_replacement_generations: int = _SUPPORTED_REPLACEMENT_GENERATIONS
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
        if self.policy.max_replacement_generations > _SUPPORTED_REPLACEMENT_GENERATIONS:
            raise TorchrunRuntimeConfigError(
                "the torchrun runtime currently supports exactly one replacement generation"
            )
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
        self._restart_context = RestartContextFile(
            config.restart_context_path,
            monotonic_clock=monotonic_clock,
        )
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
        self._replacement_clock_lock = threading.Lock()
        self._last_replacement_now_unix_ms: int | None = None
        self._admitted_assignment_lock = threading.Lock()
        self._admitted_assignment: RankAssignment | None = None
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
        context_token: _RestartContextCleanupToken | None = None
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
                        if recovery_state is not None:
                            self._acknowledge_prior_replacement_return(
                                current,
                                slot,
                                recovery_state,
                                admission_deadline,
                            )
                        context_token = self._prepare_restart_context(
                            current,
                            recovery_state,
                            admission_deadline,
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
                        self._validate_final_admission_state(
                            current,
                            recovery_state,
                            admission_deadline,
                            expected_context_token=context_token,
                        )
                        self._publish_arrival_consumption(
                            current,
                            current.snapshot.record.assignment,
                            attempt,
                            slot,
                            completion,
                            admission_deadline,
                            recovery_state,
                        )
                        if recovery_state is not None:
                            self._wait_for_replacement_admission(
                                current,
                                current.snapshot.record.assignment,
                                attempt,
                                slot,
                                completion,
                                recovery_state,
                                admission_deadline,
                                context_token,
                            )
                        self._record_admitted_assignment(
                            current.snapshot.record.assignment,
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
            try:
                if context_token is not None:
                    try:
                        self._invalidate_and_clear_restart_context(context_token)
                    except (OSError, RestartContextFileError, RendezvousStateError):
                        pass
            finally:
                try:
                    self._cleanup_local_resources()
                except Exception:
                    pass
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
        """Expose only live plan-selected replacement nodes to healthy workers."""

        if self.is_closed():
            return 0
        self._raise_heartbeat_error()
        with self._admitted_assignment_lock:
            admitted = self._admitted_assignment
        if admitted is None:
            return 0
        try:
            current = self._read_generation()
            if (
                current is None
                or current.snapshot.record.assignment.generation <= admitted.generation
            ):
                return 0
            successor = current.snapshot.record.assignment
            if successor.generation != admitted.generation + 1:
                raise RendezvousStateError("replacement signal skipped an admitted generation")
            recovery_state = self._read_recovery_state()
        except RendezvousConnectionError:
            return 0
        if recovery_state is None:
            raise RendezvousStateError(
                "replacement generation lacks an authoritative restart-plan publication"
            )
        self._validate_replacement_generation(current, recovery_state)
        admitted_nodes = set(admitted.slot_to_node_id.values())
        successor_nodes = set(successor.slot_to_node_id.values())
        replacement_nodes = successor_nodes - admitted_nodes
        if not replacement_nodes:
            raise RendezvousStateError("replacement generation does not admit a new standby node")
        if len(replacement_nodes) > self._config.max_nodes - self._config.min_nodes:
            raise RendezvousStateError("replacement generation exceeds configured standby capacity")
        try:
            now_unix_ms = self._replacement_now_unix_ms(recovery_state)
            if now_unix_ms >= recovery_state.plan.restart_deadline_unix_ms:
                return 0
            for node_id in sorted(replacement_nodes):
                registration = self._replacement_signal_registration(node_id)
                if registration is None:
                    return 0
                self._validate_registration_window(
                    registration,
                    now_unix_ms=now_unix_ms,
                )
        except RendezvousConnectionError:
            return 0
        return len(replacement_nodes)

    def _record_admitted_assignment(self, assignment: RankAssignment) -> None:
        with self._admitted_assignment_lock:
            previous = self._admitted_assignment
            if previous is not None and assignment.generation < previous.generation:
                raise RendezvousStateError("admitted generation moved backward")
            self._admitted_assignment = assignment

    def _replacement_signal_registration(
        self,
        node_id: str,
    ) -> HeldAgentRegistration | None:
        try:
            history = AgentRegistrationHistoryReader(
                self._control_store,
                run_id=self._config.run_id,
                node_id=node_id,
            ).read()
        except AgentRegistrationHistoryCorrupt as error:
            raise RendezvousStateError(
                "selected replacement registration history is corrupt"
            ) from error
        except AgentRegistrationHistoryError:
            return None
        except Exception:
            return None
        registration = history.current
        if registration is None:
            return None
        identity = registration.record.agent_identity
        if identity.run_id != self._config.run_id or identity.node_id != node_id:
            raise RendezvousStateError("selected replacement registration has the wrong identity")
        if (
            identity.local_world_size != self._config.local_world_size
            or identity.environment_digest != self._agent_identity.environment_digest
        ):
            raise RendezvousStateError(
                "selected replacement registration is incompatible with the workload"
            )
        return registration

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
        self._current_restart_context_ownership()

    def _current_restart_context_ownership(self) -> _RestartContextOwnership:
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
            or not history.authorities
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
        current_authority = history.authorities[-1]
        if current_authority.registration != history.current:
            raise RendezvousStateError(
                "assigned node registration history has an invalid current authority"
            )
        return _RestartContextOwnership(
            registration_id=history.current.record.registration_id,
            mutation_sequence=current_authority.mutation_sequence,
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
                    admission = None
                    own_return = None
                    if completion is not None and generation > 0:
                        admission = self._read_replacement_admission(
                            assignment,
                            attempt,
                            completion,
                            None,
                        )
                        if admission is not None:
                            own_return = self._read_arrival_return(
                                assignment,
                                attempt,
                                slot,
                                admission,
                            )
                    if admission is not None and own_return is not None:
                        assert completion is not None
                        if (
                            slot == 0
                            and self._arrival_return_conditions(
                                assignment,
                                attempt,
                                completion,
                                admission,
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
                        else:
                            if not self._wait_for_change(deadline):
                                raise RendezvousTimeoutError
                            continue
                    else:
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
        conditions: dict[str, int]
        if assignment.generation > 0:
            admission = self._read_replacement_admission(
                assignment,
                current_attempt,
                completion,
                None,
            )
            if admission is None:
                return None
            conditions = {
                self._arrival_admission_key(
                    assignment.generation,
                    current_attempt,
                ): admission.revision
            }
            return_conditions = self._arrival_return_conditions(
                assignment,
                current_attempt,
                completion,
                admission,
            )
            if return_conditions is None:
                return None
            conditions.update(return_conditions)
        else:
            consumption_conditions = self._arrival_consumption_conditions(
                assignment,
                current_attempt,
                completion,
            )
            if consumption_conditions is None:
                return None
            conditions = consumption_conditions
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
        current = self._read_current_assigned_registration(node_id)
        self._validate_registration_window(
            current,
            now_unix_ms=self._clock(),
        )
        return current

    def _try_current_assigned_registration(
        self,
        node_id: str,
    ) -> HeldAgentRegistration | None:
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
            return None
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
            return None
        return current

    def _read_current_assigned_registration(
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
        return current

    @staticmethod
    def _validate_registration_window(
        registration: HeldAgentRegistration,
        *,
        now_unix_ms: int,
    ) -> None:
        if (
            isinstance(now_unix_ms, bool)
            or not isinstance(now_unix_ms, int)
            or now_unix_ms < registration.granted_at_unix_ms
        ):
            raise RendezvousConnectionError(
                "rendezvous arrival clock is invalid for its registration"
            )
        if registration.expires_at_unix_ms <= now_unix_ms:
            raise RendezvousConnectionError("arrived agent registration has expired")

    def _validate_final_admission_state(
        self,
        current: CurrentGeneration,
        recovery_state: RestartPlanPersistedRecoveryState | None = None,
        admission_deadline: float | None = None,
        *,
        validate_deadline: bool = True,
        expected_context_token: _RestartContextCleanupToken | None = None,
    ) -> None:
        if self._is_closed_bounded(admission_deadline):
            raise RendezvousClosedError
        if self._read_generation() != current:
            raise RendezvousConnectionError("generation changed before rendezvous admission")
        if recovery_state is None:
            if expected_context_token is not None:
                raise RendezvousStateError(
                    "initial rendezvous unexpectedly carries restart-context ownership"
                )
            return
        if expected_context_token is None:
            raise RendezvousStateError("replacement rendezvous lacks restart-context ownership")
        refreshed = self._read_recovery_state()
        if refreshed != recovery_state:
            raise RendezvousConnectionError(
                "restart-plan recovery state changed before rendezvous admission"
            )
        if admission_deadline is None:
            raise RendezvousStateError(
                "replacement rendezvous lacks its original admission deadline"
            )
        if validate_deadline:
            self._validate_replacement_deadline(refreshed, admission_deadline)
        expected_context = self._restart_context_for_plan(refreshed.plan)
        try:
            persisted_token, persisted_context = self._restart_context.read_with_token()
        except (OSError, RestartContextFileError) as error:
            raise RendezvousStateError(
                "failed to validate the replacement restart context"
            ) from error
        if persisted_context != expected_context or persisted_token != expected_context_token:
            raise RendezvousStateError(
                "replacement restart context changed before rendezvous admission"
            )

    def _validate_current_arrival_attempt(
        self,
        assignment: RankAssignment,
        expected_attempt: int,
    ) -> None:
        key = self._arrival_attempt_key(assignment.generation)
        try:
            entry = self._control_store.get(key)
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to revalidate the rendezvous arrival attempt"
            ) from error
        if entry is None:
            raise RendezvousConnectionError(
                "rendezvous arrival attempt disappeared before admission"
            )
        self._validate_arrival_attempt_entry(
            entry,
            generation=assignment.generation,
            expected_attempt=expected_attempt,
            leader_node_id=assignment.slot_to_node_id[0],
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
        recovery_state: RestartPlanPersistedRecoveryState | None = None,
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
            store_deadline_unix_ms = registration.expires_at_unix_ms
            if recovery_state is not None:
                store_deadline_unix_ms = min(
                    store_deadline_unix_ms,
                    recovery_state.plan.restart_deadline_unix_ms,
                )
            if (
                isinstance(now_unix_ms, bool)
                or not isinstance(now_unix_ms, int)
                or now_unix_ms < registration.granted_at_unix_ms
                or now_unix_ms >= store_deadline_unix_ms
                or self._monotonic_clock() >= deadline
            ):
                if recovery_state is not None:
                    raise RendezvousTimeoutError
                raise RendezvousConnectionError("rendezvous consumption is outside its window")
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
                    deadline_unix_ms=store_deadline_unix_ms,
                    conditions={
                        completion_key: completion.revision,
                        self._generation_reader.head_key: current.head_revision,
                        generation_snapshot_key: current.snapshot.revision,
                        self._closure_key: None,
                    },
                )
            except ControlStoreDeadlineExceeded as error:
                if recovery_state is not None:
                    raise RendezvousTimeoutError from error
                raise RendezvousConnectionError(
                    "rendezvous consumption missed its registration window"
                ) from error
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
        state = self._arrival_consumption_state(
            assignment,
            attempt,
            completion,
        )
        return None if state is None else state[0]

    def _arrival_consumption_state(
        self,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
    ) -> tuple[dict[str, int], dict[str, dict[str, Any]]] | None:
        conditions: dict[str, int] = {}
        consumptions: dict[str, dict[str, Any]] = {}
        for slot, node_id in assignment.slot_to_node_id.items():
            registration = self._try_current_assigned_registration(node_id)
            if registration is None:
                return None
            entry = self._read_arrival_consumption(
                assignment,
                attempt,
                slot,
                completion,
                registration,
            )
            if entry is None:
                return None
            payload = self._validate_arrival_consumption_entry(
                entry,
                assignment=assignment,
                attempt=attempt,
                slot=slot,
                completion=completion,
                registration=registration,
            )
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
            consumptions[str(slot)] = {
                "agent_id": payload["agent_id"],
                "consumption_revision": entry.revision,
                "consumption_transaction_sequence": entry.transaction_sequence,
                "node_id": node_id,
                "registration_id": payload["registration_id"],
                "registration_revision": registration.fencing_token,
            }
        return conditions, consumptions

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
    ) -> Mapping[str, Any]:
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
        return payload

    def _wait_for_replacement_admission(
        self,
        current: CurrentGeneration,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState,
        deadline: float,
        context_token: _RestartContextCleanupToken | None,
    ) -> ControlStoreEntry:
        if context_token is None:
            raise RendezvousStateError("replacement rendezvous lacks restart-context ownership")
        while True:
            admission = self._read_replacement_admission(
                assignment,
                attempt,
                completion,
                recovery_state,
            )
            if admission is not None:
                self._validate_replacement_return(
                    current,
                    assignment,
                    attempt,
                    slot,
                    completion,
                    admission,
                    recovery_state,
                    deadline,
                    context_token,
                )
                return admission
            if slot == 0:
                self._try_commit_replacement_admission(
                    current,
                    assignment,
                    attempt,
                    completion,
                    recovery_state,
                    deadline,
                )
                admission = self._read_replacement_admission(
                    assignment,
                    attempt,
                    completion,
                    recovery_state,
                )
                if admission is not None:
                    self._validate_replacement_return(
                        current,
                        assignment,
                        attempt,
                        slot,
                        completion,
                        admission,
                        recovery_state,
                        deadline,
                        context_token,
                    )
                    return admission
            if not self._wait_for_change(deadline):
                admission = self._read_replacement_admission(
                    assignment,
                    attempt,
                    completion,
                    recovery_state,
                )
                if admission is not None:
                    self._validate_replacement_return(
                        current,
                        assignment,
                        attempt,
                        slot,
                        completion,
                        admission,
                        recovery_state,
                        deadline,
                        context_token,
                    )
                    return admission
                raise RendezvousTimeoutError

    def _validate_replacement_return(
        self,
        current: CurrentGeneration,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        admission: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState,
        deadline: float,
        context_token: _RestartContextCleanupToken,
    ) -> None:
        self._validate_final_admission_state(
            current,
            recovery_state,
            deadline,
            validate_deadline=False,
            expected_context_token=context_token,
        )
        self._raise_heartbeat_error()
        with self._registration_lock:
            registration = self._registration
        if registration is None:
            raise RendezvousConnectionError("replacement agent no longer owns its registration")
        registration_history = self._read_assigned_registration_history(self._config.node_id)
        current_registration = registration_history.current
        if current_registration is None:
            raise RendezvousConnectionError(
                "replacement agent no longer has a current registration"
            )
        if current_registration.record != registration.record:
            raise RendezvousConnectionError(
                "replacement agent registration changed before admission"
            )
        consumptions = self._validate_replacement_admission_entry(
            admission,
            assignment=assignment,
            attempt=attempt,
            completion=completion,
            recovery_state=recovery_state,
        )
        metadata = consumptions[str(slot)]
        admission_registration = next(
            (
                authority.registration
                for authority in registration_history.authorities
                if authority.registration.fencing_token == metadata["registration_revision"]
            ),
            None,
        )
        if (
            metadata["registration_id"] != current_registration.record.registration_id
            or metadata["agent_id"] != current_registration.record.agent_identity.agent_id
            or admission_registration is None
            or admission_registration.record != current_registration.record
        ):
            raise RendezvousConnectionError(
                "replacement admission does not include this agent registration"
            )
        consumption = self._read_arrival_consumption(
            assignment,
            attempt,
            slot,
            completion,
            current_registration,
        )
        if consumption is None or (
            metadata["consumption_revision"] != consumption.revision
            or metadata["consumption_transaction_sequence"] != consumption.transaction_sequence
        ):
            raise RendezvousConnectionError(
                "replacement admission does not include this agent consumption"
            )
        self._validate_current_arrival_attempt(assignment, attempt)
        now_unix_ms = self._replacement_now_unix_ms(recovery_state)
        self._validate_registration_window(
            current_registration,
            now_unix_ms=now_unix_ms,
        )
        self._validate_replacement_deadline(
            recovery_state,
            deadline,
            now_unix_ms=now_unix_ms,
        )

    def _read_assigned_registration_history(
        self,
        node_id: str,
    ) -> AgentRegistrationHistory:
        try:
            return AgentRegistrationHistoryReader(
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
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read arrived agent registration history"
            ) from error

    def _acknowledge_prior_replacement_return(
        self,
        current: CurrentGeneration,
        slot: int,
        recovery_state: RestartPlanPersistedRecoveryState,
        deadline: float,
    ) -> None:
        assignment = current.snapshot.record.assignment
        attempt_entry = self._control_store.get(self._arrival_attempt_key(assignment.generation))
        if attempt_entry is None:
            return
        attempt, _, _ = self._validate_arrival_attempt_entry(
            attempt_entry,
            generation=assignment.generation,
            leader_node_id=assignment.slot_to_node_id[0],
        )
        completion = self._read_arrival_completion(
            assignment,
            attempt,
            None,
        )
        if completion is None:
            return
        admission = self._read_replacement_admission(
            assignment,
            attempt,
            completion,
            recovery_state,
        )
        if admission is None:
            return
        self._publish_replacement_return_acknowledgement(
            current,
            assignment,
            attempt,
            slot,
            completion,
            admission,
            recovery_state,
            deadline=deadline,
        )

    def _publish_replacement_return_acknowledgement(
        self,
        current: CurrentGeneration,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        completion: ControlStoreEntry,
        admission: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState,
        *,
        deadline: float,
    ) -> ControlStoreEntry:
        consumptions = self._validate_replacement_admission_entry(
            admission,
            assignment=assignment,
            attempt=attempt,
            completion=completion,
            recovery_state=recovery_state,
        )
        metadata = consumptions[str(slot)]
        consumption_key = self._arrival_consumption_key(
            assignment.generation,
            attempt,
            slot,
            metadata["registration_id"],
        )
        consumption = self._control_store.get(consumption_key)
        if consumption is None or consumption.revision != metadata["consumption_revision"]:
            raise RendezvousStateError("replacement admission references a missing consumption")
        key = self._arrival_return_key(
            assignment.generation,
            attempt,
            slot,
        )
        while True:
            existing = self._control_store.get(key)
            if existing is not None:
                self._validate_arrival_return_entry(
                    existing,
                    assignment=assignment,
                    attempt=attempt,
                    slot=slot,
                    admission=admission,
                    consumption=consumption,
                    registration=None,
                )
                return existing
            with self._registration_lock:
                registration = self._registration
            if registration is None:
                raise RendezvousConnectionError(
                    "replacement agent lost its registration before re-entering rendezvous"
                )
            value = self._arrival_return_value(
                generation=assignment.generation,
                attempt=attempt,
                slot=slot,
                admission=admission,
                consumption=consumption,
                registration=registration,
            )
            now_unix_ms = self._replacement_now_unix_ms(recovery_state)
            deadline_unix_ms = min(
                registration.expires_at_unix_ms,
                recovery_state.plan.restart_deadline_unix_ms,
            )
            if self._monotonic_clock() >= deadline:
                raise RendezvousTimeoutError
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
                    deadline_unix_ms=deadline_unix_ms,
                    conditions={
                        self._arrival_admission_key(
                            assignment.generation,
                            attempt,
                        ): admission.revision,
                        consumption_key: consumption.revision,
                        self._generation_reader.head_key: current.head_revision,
                        self._generation_reader.snapshot_key(
                            assignment.generation,
                        ): current.snapshot.revision,
                        self._closure_key: None,
                    },
                )
            except ControlStoreDeadlineExceeded as error:
                raise RendezvousTimeoutError from error
            except ControlStoreConflict:
                existing = self._control_store.get(key)
                if existing is not None:
                    self._validate_arrival_return_entry(
                        existing,
                        assignment=assignment,
                        attempt=attempt,
                        slot=slot,
                        admission=admission,
                        consumption=consumption,
                        registration=None,
                    )
                    return existing
                with self._registration_lock:
                    refreshed = self._registration
                if (
                    refreshed is not None
                    and refreshed.record == registration.record
                    and refreshed.fencing_token != registration.fencing_token
                ):
                    continue
                raise RendezvousConnectionError(
                    "replacement return acknowledgement lost its admission fence"
                ) from None
            except Exception as error:
                raise RendezvousConnectionError(
                    "failed to publish replacement return acknowledgement"
                ) from error
            entry = result[key]
            self._validate_arrival_return_entry(
                entry,
                assignment=assignment,
                attempt=attempt,
                slot=slot,
                admission=admission,
                consumption=consumption,
                registration=registration,
            )
            self._state_changed_event.set()
            return entry

    def _read_arrival_return(
        self,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        admission: ControlStoreEntry,
    ) -> ControlStoreEntry | None:
        try:
            entry = self._control_store.get(
                self._arrival_return_key(
                    assignment.generation,
                    attempt,
                    slot,
                )
            )
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to read replacement return acknowledgement"
            ) from error
        if entry is None:
            return None
        self._validate_arrival_return_entry(
            entry,
            assignment=assignment,
            attempt=attempt,
            slot=slot,
            admission=admission,
            consumption=None,
            registration=None,
        )
        return entry

    def _arrival_return_conditions(
        self,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
        admission: ControlStoreEntry,
    ) -> dict[str, int] | None:
        consumptions = self._validate_replacement_admission_entry(
            admission,
            assignment=assignment,
            attempt=attempt,
            completion=completion,
            recovery_state=None,
        )
        conditions: dict[str, int] = {}
        for slot, node_id in assignment.slot_to_node_id.items():
            metadata = consumptions[str(slot)]
            key = self._arrival_return_key(
                assignment.generation,
                attempt,
                slot,
            )
            try:
                entry = self._control_store.get(key)
            except Exception as error:
                raise RendezvousConnectionError(
                    "failed to read replacement return acknowledgement"
                ) from error
            if entry is None:
                return None
            payload = self._validate_arrival_return_entry(
                entry,
                assignment=assignment,
                attempt=attempt,
                slot=slot,
                admission=admission,
                consumption=None,
                registration=None,
            )
            if payload["consumption_revision"] != metadata["consumption_revision"]:
                raise RendezvousStateError(
                    "replacement return acknowledgement does not match its admission metadata"
                )
            if entry.guard_key != agent_registration_key(
                self._config.run_id,
                node_id,
            ):
                raise RendezvousStateError(
                    "replacement return acknowledgement has invalid registration provenance"
                )
            conditions[key] = entry.revision
        return conditions

    def _arrival_return_key(
        self,
        generation: int,
        attempt: int,
        slot: int,
    ) -> str:
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/attempt-{attempt}/returned/"
            f"slot-{slot}"
        )

    def _arrival_return_value(
        self,
        *,
        generation: int,
        attempt: int,
        slot: int,
        admission: ControlStoreEntry,
        consumption: ControlStoreEntry,
        registration: HeldAgentRegistration,
    ) -> bytes:
        return json.dumps(
            {
                "admission_digest": hashlib.sha256(admission.value).hexdigest(),
                "agent_id": registration.record.agent_identity.agent_id,
                "attempt": attempt,
                "consumption_revision": consumption.revision,
                "generation": generation,
                "logical_node_slot": slot,
                "node_id": registration.record.agent_identity.node_id,
                "registration_id": registration.record.registration_id,
                "registration_revision": registration.fencing_token,
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_RETURN_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_arrival_return_entry(
        self,
        entry: ControlStoreEntry,
        *,
        assignment: RankAssignment,
        attempt: int,
        slot: int,
        admission: ControlStoreEntry,
        consumption: ControlStoreEntry | None,
        registration: HeldAgentRegistration | None,
    ) -> Mapping[str, Any]:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("replacement return acknowledgement is malformed") from error
        expected_fields = {
            "admission_digest",
            "agent_id",
            "attempt",
            "consumption_revision",
            "generation",
            "logical_node_slot",
            "node_id",
            "registration_id",
            "registration_revision",
            "run_id",
            "schema_version",
        }
        expected_node_id = assignment.slot_to_node_id.get(slot)
        integer_fields = (
            "attempt",
            "consumption_revision",
            "generation",
            "registration_revision",
            "schema_version",
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_fields
            or payload["run_id"] != self._config.run_id
            or payload["generation"] != assignment.generation
            or payload["attempt"] != attempt
            or payload["logical_node_slot"] != slot
            or payload["node_id"] != expected_node_id
            or payload["admission_digest"] != hashlib.sha256(admission.value).hexdigest()
            or payload["schema_version"] != _ARRIVAL_RETURN_SCHEMA_VERSION
            or isinstance(payload["logical_node_slot"], bool)
            or not isinstance(payload["logical_node_slot"], int)
            or payload["logical_node_slot"] < 0
            or any(
                isinstance(payload[field], bool)
                or not isinstance(payload[field], int)
                or payload[field] < 1
                for field in integer_fields
            )
        ):
            raise RendezvousStateError(
                "replacement return acknowledgement does not match its admission"
            )
        for field in ("agent_id", "registration_id"):
            if not isinstance(payload[field], str) or not payload[field]:
                raise RendezvousStateError(
                    f"replacement return acknowledgement has invalid {field}"
                )
        if consumption is not None and payload["consumption_revision"] != consumption.revision:
            raise RendezvousStateError(
                "replacement return acknowledgement does not match its consumption"
            )
        if registration is not None and (
            payload["registration_id"] != registration.record.registration_id
            or payload["agent_id"] != registration.record.agent_identity.agent_id
            or payload["registration_revision"] != registration.fencing_token
        ):
            raise RendezvousConnectionError(
                "replacement return acknowledgement belongs to another registration"
            )
        if (
            entry.value
            != json.dumps(
                payload,
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            or entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
            or entry.guard_key
            != agent_registration_key(
                self._config.run_id,
                expected_node_id,
            )
            or entry.guard_revision != payload["registration_revision"]
            or entry.transaction_sequence <= admission.transaction_sequence
        ):
            raise RendezvousStateError(
                "replacement return acknowledgement has invalid store provenance"
            )
        return payload

    def _try_commit_replacement_admission(
        self,
        current: CurrentGeneration,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState,
        deadline: float,
    ) -> None:
        admission_key = self._arrival_admission_key(
            assignment.generation,
            attempt,
        )
        if self._control_store.get(admission_key) is not None:
            return
        state = self._arrival_consumption_state(
            assignment,
            attempt,
            completion,
        )
        if state is None:
            return
        readiness_conditions, consumptions = state
        conditions: dict[str, int | None] = dict(readiness_conditions)
        with self._registration_lock:
            leader_registration = self._registration
        if leader_registration is None:
            raise RendezvousConnectionError(
                "replacement leader lost its registration before group admission"
            )
        slot_zero = consumptions["0"]
        if (
            slot_zero["registration_id"] != leader_registration.record.registration_id
            or slot_zero["agent_id"] != leader_registration.record.agent_identity.agent_id
        ):
            raise RendezvousConnectionError(
                "replacement leader registration changed before group admission"
            )
        completion_key = self._arrival_completion_key(
            assignment.generation,
            attempt,
        )
        attempt_key = self._arrival_attempt_key(assignment.generation)
        snapshot_key = self._generation_reader.snapshot_key(assignment.generation)
        attempt_entry = self._control_store.get(attempt_key)
        if attempt_entry is None:
            raise RendezvousStateError(
                "replacement rendezvous attempt disappeared before group admission"
            )
        self._validate_arrival_attempt_entry(
            attempt_entry,
            generation=assignment.generation,
            expected_attempt=attempt,
            leader_node_id=assignment.slot_to_node_id[0],
        )
        conditions.update(
            {
                attempt_key: attempt_entry.revision,
                completion_key: completion.revision,
                self._generation_reader.head_key: current.head_revision,
                snapshot_key: current.snapshot.revision,
                self._closure_key: None,
            }
        )
        now_unix_ms = self._replacement_now_unix_ms(recovery_state)
        store_deadline_unix_ms = recovery_state.plan.restart_deadline_unix_ms
        for metadata in consumptions.values():
            registration = self._try_current_assigned_registration(metadata["node_id"])
            if registration is None:
                return
            if (
                registration.record.registration_id != metadata["registration_id"]
                or registration.record.agent_identity.agent_id != metadata["agent_id"]
            ):
                return
            metadata["registration_revision"] = registration.fencing_token
            store_deadline_unix_ms = min(
                store_deadline_unix_ms,
                registration.expires_at_unix_ms,
            )
            registration_key = agent_registration_key(
                self._config.run_id,
                metadata["node_id"],
            )
            if registration_key != self._registration_manager.registration_key:
                conditions[registration_key] = registration.fencing_token
        if self._monotonic_clock() >= deadline or now_unix_ms >= store_deadline_unix_ms:
            raise RendezvousTimeoutError
        value = self._arrival_admission_value(
            generation=assignment.generation,
            attempt=attempt,
            recovery_state=recovery_state,
            completion=completion,
            consumptions=consumptions,
        )
        try:
            result = self._control_store.compare_set_many_guarded(
                {
                    admission_key: ControlStoreWrite(
                        expected_revision=None,
                        value=value,
                        require_never_created=True,
                    )
                },
                guard_key=self._registration_manager.registration_key,
                expected_guard_revision=leader_registration.fencing_token,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=store_deadline_unix_ms,
                conditions=conditions,
            )
        except ControlStoreDeadlineExceeded as error:
            raise RendezvousTimeoutError from error
        except ControlStoreConflict:
            return
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to commit replacement group admission"
            ) from error
        self._validate_replacement_admission_entry(
            result[admission_key],
            assignment=assignment,
            attempt=attempt,
            completion=completion,
            recovery_state=recovery_state,
        )
        self._state_changed_event.set()

    def _read_replacement_admission(
        self,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState | None,
    ) -> ControlStoreEntry | None:
        key = self._arrival_admission_key(
            assignment.generation,
            attempt,
        )
        try:
            entry = self._control_store.get(key)
        except Exception as error:
            raise RendezvousConnectionError("failed to read replacement group admission") from error
        if entry is None:
            return None
        self._validate_replacement_admission_entry(
            entry,
            assignment=assignment,
            attempt=attempt,
            completion=completion,
            recovery_state=recovery_state,
        )
        return entry

    def _arrival_admission_key(self, generation: int, attempt: int) -> str:
        return (
            f"{_CONTROL_PREFIX}/runs/{self._run_digest}/rendezvous/"
            f"generation-{generation}/attempt-{attempt}/admitted"
        )

    def _arrival_admission_value(
        self,
        *,
        generation: int,
        attempt: int,
        recovery_state: RestartPlanPersistedRecoveryState,
        completion: ControlStoreEntry,
        consumptions: Mapping[str, Mapping[str, Any]],
    ) -> bytes:
        return json.dumps(
            {
                "attempt": attempt,
                "completion_digest": hashlib.sha256(completion.value).hexdigest(),
                "consumptions": consumptions,
                "generation": generation,
                "plan_id": recovery_state.plan.plan_id,
                "restart_deadline_unix_ms": (recovery_state.plan.restart_deadline_unix_ms),
                "run_id": self._config.run_id,
                "schema_version": _ARRIVAL_ADMISSION_SCHEMA_VERSION,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")

    def _validate_replacement_admission_entry(
        self,
        entry: ControlStoreEntry,
        *,
        assignment: RankAssignment,
        attempt: int,
        completion: ControlStoreEntry,
        recovery_state: RestartPlanPersistedRecoveryState | None,
    ) -> Mapping[str, Mapping[str, Any]]:
        try:
            payload = json.loads(
                entry.value.decode("utf-8"),
                object_pairs_hook=_strict_object,
            )
        except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as error:
            raise RendezvousStateError("replacement group admission is malformed") from error
        expected_fields = {
            "attempt",
            "completion_digest",
            "consumptions",
            "generation",
            "plan_id",
            "restart_deadline_unix_ms",
            "run_id",
            "schema_version",
        }
        integer_fields = (
            "attempt",
            "generation",
            "restart_deadline_unix_ms",
            "schema_version",
        )
        if (
            not isinstance(payload, dict)
            or set(payload) != expected_fields
            or payload["run_id"] != self._config.run_id
            or payload["generation"] != assignment.generation
            or payload["attempt"] != attempt
            or payload["schema_version"] != _ARRIVAL_ADMISSION_SCHEMA_VERSION
            or any(
                isinstance(payload[field], bool) or not isinstance(payload[field], int)
                for field in integer_fields
            )
            or payload["restart_deadline_unix_ms"] < 1
            or payload["completion_digest"] != hashlib.sha256(completion.value).hexdigest()
            or not isinstance(payload["plan_id"], str)
            or not payload["plan_id"]
        ):
            raise RendezvousStateError(
                "replacement group admission does not match its rendezvous attempt"
            )
        if recovery_state is not None and (
            payload["plan_id"] != recovery_state.plan.plan_id
            or payload["restart_deadline_unix_ms"] != recovery_state.plan.restart_deadline_unix_ms
        ):
            raise RendezvousStateError(
                "replacement group admission does not match its restart plan"
            )
        if entry.value != json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8"):
            raise RendezvousStateError("replacement group admission contains noncanonical bytes")
        consumptions = payload["consumptions"]
        expected_slots = {
            str(slot): node_id for slot, node_id in assignment.slot_to_node_id.items()
        }
        if not isinstance(consumptions, dict) or set(consumptions) != set(expected_slots):
            raise RendezvousStateError("replacement group admission has invalid slot coverage")
        for slot, node_id in expected_slots.items():
            metadata = consumptions[slot]
            fields = {
                "agent_id",
                "consumption_revision",
                "consumption_transaction_sequence",
                "node_id",
                "registration_id",
                "registration_revision",
            }
            if (
                not isinstance(metadata, dict)
                or set(metadata) != fields
                or metadata["node_id"] != node_id
            ):
                raise RendezvousStateError(
                    "replacement group admission has invalid readiness metadata"
                )
            for field in ("agent_id", "registration_id"):
                if not isinstance(metadata[field], str) or not metadata[field]:
                    raise RendezvousStateError(f"replacement group admission has invalid {field}")
            for field in (
                "consumption_revision",
                "consumption_transaction_sequence",
                "registration_revision",
            ):
                if (
                    isinstance(metadata[field], bool)
                    or not isinstance(metadata[field], int)
                    or metadata[field] < 1
                ):
                    raise RendezvousStateError(f"replacement group admission has invalid {field}")
        slot_zero = consumptions["0"]
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
            or entry.committed_at_unix_ms is None
            or entry.committed_at_unix_ms >= payload["restart_deadline_unix_ms"]
            or entry.transaction_sequence <= completion.transaction_sequence
            or any(
                entry.transaction_sequence <= metadata["consumption_transaction_sequence"]
                for metadata in consumptions.values()
            )
        ):
            raise RendezvousStateError("replacement group admission has invalid store provenance")
        if recovery_state is not None and (
            entry.committed_at_unix_ms < recovery_state.publication.committed_at_unix_ms
        ):
            raise RendezvousStateError("replacement group admission predates its restart plan")
        return consumptions

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
        slot = next(
            (
                slot
                for slot, node_id in assignment.slot_to_node_id.items()
                if node_id == self._config.node_id
            ),
            None,
        )
        if slot is None:
            return None, None, formation_deadline
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
            self._validate_replacement_generation(current, recovery_state)
            admission_deadline = self._replacement_deadline(
                recovery_state,
                formation_deadline,
            )
        self._ensure_job_compatibility()
        self._validate_assigned_registration_history()
        return slot, recovery_state, admission_deadline

    def _validate_replacement_generation(
        self,
        current: CurrentGeneration,
        recovery_state: RestartPlanPersistedRecoveryState,
    ) -> None:
        assignment = current.snapshot.record.assignment
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
            or plan.expected_world_size != self._config.min_nodes * self._config.local_world_size
        ):
            raise RendezvousStateError(
                "replacement plan does not match the committed generation assignment"
            )

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
        recovery_state: RestartPlanPersistedRecoveryState,
        formation_deadline: float | None,
    ) -> float:
        now_unix_ms = self._replacement_now_unix_ms(recovery_state)
        plan = recovery_state.plan
        remaining_seconds = (plan.restart_deadline_unix_ms - now_unix_ms) / 1_000
        if remaining_seconds <= 0:
            raise RendezvousTimeoutError
        deadline = self._monotonic_clock() + remaining_seconds
        return deadline if formation_deadline is None else min(deadline, formation_deadline)

    def _replacement_now_unix_ms(
        self,
        recovery_state: RestartPlanPersistedRecoveryState,
    ) -> int:
        try:
            now_unix_ms = self._control_store.observe_time_unix_ms()
        except Exception as error:
            raise RendezvousConnectionError(
                "failed to observe authoritative replacement rendezvous time"
            ) from error
        if isinstance(now_unix_ms, bool) or not isinstance(now_unix_ms, int) or now_unix_ms < 1:
            raise RendezvousConnectionError(
                "control-store replacement rendezvous clock returned an invalid time"
            )
        publication_time = recovery_state.publication.committed_at_unix_ms
        with self._replacement_clock_lock:
            if now_unix_ms < publication_time or (
                self._last_replacement_now_unix_ms is not None
                and now_unix_ms < self._last_replacement_now_unix_ms
            ):
                raise RendezvousConnectionError(
                    "replacement rendezvous clock regressed behind trusted state"
                )
            self._last_replacement_now_unix_ms = now_unix_ms
        return now_unix_ms

    def _validate_replacement_deadline(
        self,
        recovery_state: RestartPlanPersistedRecoveryState,
        admission_deadline: float,
        *,
        now_unix_ms: int | None = None,
    ) -> None:
        observed_unix_ms = (
            self._replacement_now_unix_ms(recovery_state) if now_unix_ms is None else now_unix_ms
        )
        if (
            self._monotonic_clock() >= admission_deadline
            or observed_unix_ms >= recovery_state.plan.restart_deadline_unix_ms
        ):
            raise RendezvousTimeoutError

    def _prepare_restart_context(
        self,
        current: CurrentGeneration,
        recovery_state: RestartPlanPersistedRecoveryState | None,
        deadline: float,
    ) -> _RestartContextCleanupToken | None:
        if current.snapshot.record.assignment.generation == 0:
            try:
                self._restart_context.clear_stale_for_run(
                    self._config.run_id,
                    deadline=deadline,
                )
            except (OSError, RestartContextFileError) as error:
                raise RendezvousStateError(
                    "failed to clear stale initial restart context"
                ) from error
            return None
        if recovery_state is None:
            raise RendezvousStateError("replacement generation lacks an authoritative restart plan")
        context = self._restart_context_for_plan(recovery_state.plan)
        ownership = self._current_restart_context_ownership()
        try:
            return self._restart_context.write(
                context,
                deadline=deadline,
                ownership=ownership,
            )
        except (OSError, RestartContextFileError) as error:
            if isinstance(error, RestartContextFileError) and error.published_token is not None:
                self._invalidate_and_clear_restart_context(error.published_token)
            raise RendezvousStateError(
                "failed to publish the replacement restart context"
            ) from error

    def _invalidate_and_clear_restart_context(
        self,
        cleanup_token: _RestartContextCleanupToken,
    ) -> None:
        invalidation_error: OSError | RestartContextFileError | None = None
        try:
            self._restart_context.invalidate(cleanup_token)
        except (OSError, RestartContextFileError) as error:
            invalidation_error = error
        try:
            self._restart_context.clear_if_token(
                cleanup_token,
                deadline=self._context_cleanup_deadline(),
            )
        except (OSError, RestartContextFileError) as cleanup_error:
            if invalidation_error is not None:
                raise RendezvousStateError(
                    "failed to invalidate the replacement restart context"
                ) from cleanup_error

    def _context_cleanup_deadline(self) -> float:
        return self._monotonic_clock() + _CONTEXT_LOCK_TIMEOUT_SECONDS

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
    monotonic_clock: Callable[[], float] = time.monotonic

    def __post_init__(self) -> None:
        if not isinstance(self.path, Path):
            raise TypeError("path must be pathlib.Path")
        if not callable(self.monotonic_clock):
            raise TypeError("monotonic_clock must be callable")
        if not self.path.is_absolute():
            raise RestartContextFileError("restart-context path must be absolute")
        if self.path.name in {"", ".", ".."}:
            raise RestartContextFileError("restart-context path must name a file")

    def write(
        self,
        context: RestartContext,
        *,
        deadline: float | None = None,
        ownership: _RestartContextOwnership | None = None,
    ) -> _RestartContextCleanupToken:
        if not isinstance(context, RestartContext):
            raise TypeError("context must be RestartContext")
        if ownership is not None and not isinstance(ownership, _RestartContextOwnership):
            raise TypeError("ownership must be _RestartContextOwnership or None")
        cleanup_token = secrets.token_hex(32)
        encoded = (context.to_json() + "\n").encode("utf-8")
        metadata = self._encode_metadata(cleanup_token, encoded, ownership)
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise RestartContextFileError("restart context is too large")
        parent_descriptor = self._open_parent(create=True)
        assert parent_descriptor is not None
        context_descriptor = -1
        context_temporary = ""
        metadata_descriptor = -1
        metadata_temporary = ""
        metadata_published = False
        try:
            self._acquire_parent_lock(parent_descriptor, deadline=deadline)
            self._reject_existing_symlink(parent_descriptor)
            existing_metadata = self._read_encoded(
                parent_descriptor,
                missing_ok=True,
                name=self._metadata_name(),
                maximum_bytes=_MAX_CONTEXT_METADATA_BYTES,
            )
            existing_ownership = (
                None
                if existing_metadata is None
                else self._decode_metadata_record(existing_metadata)[2]
            )
            self._validate_replacement_ownership(
                existing=existing_ownership,
                replacement=ownership,
            )
            context_descriptor, context_temporary = self._create_temporary_file(parent_descriptor)
            metadata_descriptor, metadata_temporary = self._create_temporary_file(
                parent_descriptor,
                target_name=self._metadata_name(),
            )
            os.fchmod(context_descriptor, 0o600)
            os.fchmod(metadata_descriptor, 0o600)
            with os.fdopen(context_descriptor, "wb") as stream:
                context_descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            with os.fdopen(metadata_descriptor, "wb") as stream:
                metadata_descriptor = -1
                stream.write(metadata)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                metadata_temporary,
                self._metadata_name(),
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            metadata_temporary = ""
            metadata_published = True
            os.replace(
                context_temporary,
                self.path.name,
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            context_temporary = ""
            self._fsync_directory(parent_descriptor)
            return cleanup_token
        except OSError as error:
            raise RestartContextFileError(
                f"failed to publish restart context at {self.path}",
                published_token=cleanup_token if metadata_published else None,
            ) from error
        finally:
            for descriptor in (context_descriptor, metadata_descriptor):
                if descriptor < 0:
                    continue
                try:
                    os.close(descriptor)
                except OSError:
                    pass
            for temporary in (context_temporary, metadata_temporary):
                if not temporary:
                    continue
                try:
                    os.unlink(temporary, dir_fd=parent_descriptor)
                except OSError:
                    pass
            os.close(parent_descriptor)

    def read(self) -> RestartContext:
        return self.read_with_token()[1]

    def read_with_token(
        self,
    ) -> tuple[_RestartContextCleanupToken, RestartContext]:
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            raise RestartContextFileError(
                f"restart-context directory does not exist for {self.path}"
            )
        try:
            self._acquire_parent_lock(parent_descriptor, deadline=None)
            encoded = self._read_encoded(parent_descriptor)
            assert encoded is not None
            metadata = self._read_encoded(
                parent_descriptor,
                name=self._metadata_name(),
                maximum_bytes=_MAX_CONTEXT_METADATA_BYTES,
            )
            assert metadata is not None
            context = self._decode_context(encoded)
            cleanup_token = self._decode_metadata(metadata, encoded)
            if self._is_invalidated(parent_descriptor, cleanup_token):
                raise RestartContextFileError("restart context has been invalidated")
            return cleanup_token, context
        finally:
            os.close(parent_descriptor)

    def invalidate(
        self,
        cleanup_token: _RestartContextCleanupToken,
    ) -> None:
        """Durably invalidate one published context without taking its directory lock."""

        self._validate_cleanup_token(cleanup_token)
        encoded = (
            json.dumps(
                {
                    "schema_version": _CONTEXT_INVALIDATION_SCHEMA_VERSION,
                    "cleanup_token": cleanup_token,
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        parent_descriptor = self._open_parent(create=True)
        assert parent_descriptor is not None
        descriptor = -1
        temporary = ""
        try:
            descriptor, temporary = self._create_temporary_file(
                parent_descriptor,
                target_name=self._invalidation_name(cleanup_token),
            )
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(
                temporary,
                self._invalidation_name(cleanup_token),
                src_dir_fd=parent_descriptor,
                dst_dir_fd=parent_descriptor,
            )
            temporary = ""
            self._fsync_directory(parent_descriptor)
        except OSError as error:
            raise RestartContextFileError(
                f"failed to invalidate restart context at {self.path}"
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

    def clear(self, *, deadline: float | None = None) -> None:
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            return
        try:
            self._acquire_parent_lock(parent_descriptor, deadline=deadline)
            self._reject_existing_symlink(parent_descriptor)
            self._unlink_context_pair(parent_descriptor)
            self._fsync_directory(parent_descriptor)
        finally:
            os.close(parent_descriptor)

    def clear_stale_for_run(
        self,
        run_id: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        if not isinstance(run_id, str) or not run_id:
            raise TypeError("run_id must be a non-empty string")
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            return False
        try:
            self._acquire_parent_lock(parent_descriptor, deadline=deadline)
            self._reject_existing_symlink(parent_descriptor)
            encoded = self._read_encoded(parent_descriptor, missing_ok=True)
            if encoded is None:
                metadata = self._read_encoded(
                    parent_descriptor,
                    missing_ok=True,
                    name=self._metadata_name(),
                    maximum_bytes=_MAX_CONTEXT_METADATA_BYTES,
                )
                if metadata is None:
                    return False
                try:
                    os.unlink(self._metadata_name(), dir_fd=parent_descriptor)
                    self._fsync_directory(parent_descriptor)
                except OSError as error:
                    raise RestartContextFileError(
                        f"failed to remove orphaned restart-context metadata at {self.path}"
                    ) from error
                return True
            context = self._decode_context(encoded)
            if context.run_id == run_id:
                raise RestartContextFileError(
                    "refusing to remove a restart context for the current run"
                )
            self._unlink_context_pair(parent_descriptor)
            self._fsync_directory(parent_descriptor)
            return True
        finally:
            os.close(parent_descriptor)

    def clear_if_token(
        self,
        cleanup_token: _RestartContextCleanupToken,
        *,
        deadline: float | None = None,
    ) -> bool:
        self._validate_cleanup_token(cleanup_token)
        parent_descriptor = self._open_parent(create=False)
        if parent_descriptor is None:
            return False
        try:
            self._acquire_parent_lock(parent_descriptor, deadline=deadline)
            self._reject_existing_symlink(parent_descriptor)
            metadata = self._read_encoded(
                parent_descriptor,
                missing_ok=True,
                name=self._metadata_name(),
                maximum_bytes=_MAX_CONTEXT_METADATA_BYTES,
            )
            if metadata is None:
                return False
            current_token, _, _ = self._decode_metadata_record(metadata)
            if current_token != cleanup_token:
                return False
            self._unlink_context_pair(parent_descriptor)
            self._fsync_directory(parent_descriptor)
            return True
        finally:
            os.close(parent_descriptor)

    def _unlink_context_pair(self, parent_descriptor: int) -> None:
        for name in (self.path.name, self._metadata_name()):
            try:
                os.unlink(name, dir_fd=parent_descriptor)
            except FileNotFoundError:
                continue
            except OSError as error:
                raise RestartContextFileError(
                    f"failed to remove restart context at {self.path}"
                ) from error

    def _acquire_parent_lock(
        self,
        parent_descriptor: int,
        *,
        deadline: float | None,
    ) -> None:
        now = self.monotonic_clock()
        if isinstance(now, bool) or not isinstance(now, (int, float)) or not math.isfinite(now):
            raise RestartContextFileError("restart-context clock returned an invalid time")
        logical_deadline = now + _CONTEXT_LOCK_TIMEOUT_SECONDS if deadline is None else deadline
        if (
            isinstance(logical_deadline, bool)
            or not isinstance(logical_deadline, (int, float))
            or not math.isfinite(logical_deadline)
        ):
            raise TypeError("deadline must be a monotonic number or None")
        wall_deadline = time.monotonic() + max(0.0, logical_deadline - now)
        previous_logical_now = float(now)
        while True:
            try:
                fcntl.flock(
                    parent_descriptor,
                    fcntl.LOCK_EX | fcntl.LOCK_NB,
                )
                return
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN}:
                    raise RestartContextFileError(
                        f"failed to lock restart-context directory for {self.path}"
                    ) from error
            logical_now = self.monotonic_clock()
            if (
                isinstance(logical_now, bool)
                or not isinstance(logical_now, (int, float))
                or not math.isfinite(logical_now)
                or logical_now < previous_logical_now
            ):
                raise RestartContextFileError("restart-context clock returned an invalid time")
            previous_logical_now = float(logical_now)
            wall_now = time.monotonic()
            if logical_now >= logical_deadline or wall_now >= wall_deadline:
                raise RestartContextFileError(
                    f"timed out locking restart-context directory for {self.path}"
                )
            time.sleep(
                min(
                    _CONTEXT_LOCK_POLL_SECONDS,
                    max(0.0, wall_deadline - wall_now),
                )
            )

    def _read_encoded(
        self,
        parent_descriptor: int,
        *,
        missing_ok: bool = False,
        name: str | None = None,
        maximum_bytes: int = _MAX_CONTEXT_BYTES,
    ) -> bytes | None:
        target_name = self.path.name if name is None else name
        flags = (
            os.O_RDONLY
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_NONBLOCK", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        try:
            descriptor = os.open(
                target_name,
                flags,
                dir_fd=parent_descriptor,
            )
        except FileNotFoundError:
            if missing_ok:
                return None
            raise RestartContextFileError(
                f"failed to open restart context at {self.path}"
            ) from None
        except OSError as error:
            raise RestartContextFileError(
                f"failed to open restart context at {self.path}"
            ) from error
        try:
            metadata = os.fstat(descriptor)
            self._validate_owned_private_file(metadata)
            with os.fdopen(descriptor, "rb", closefd=False) as stream:
                encoded = stream.read(maximum_bytes + 1)
            if len(encoded) > maximum_bytes:
                raise RestartContextFileError("restart-context file is too large")
            return encoded
        finally:
            os.close(descriptor)

    def _is_invalidated(
        self,
        parent_descriptor: int,
        cleanup_token: _RestartContextCleanupToken,
    ) -> bool:
        encoded = self._read_encoded(
            parent_descriptor,
            missing_ok=True,
            name=self._invalidation_name(cleanup_token),
            maximum_bytes=_MAX_CONTEXT_INVALIDATION_BYTES,
        )
        if encoded is None:
            return False
        try:
            value = json.loads(
                encoded,
                object_pairs_hook=_strict_object,
            )
            if (
                not isinstance(value, Mapping)
                or set(value) != {"schema_version", "cleanup_token"}
                or type(value["schema_version"]) is not int
                or value["schema_version"] != _CONTEXT_INVALIDATION_SCHEMA_VERSION
            ):
                raise RestartContextFileError("restart-context invalidation marker is malformed")
            return self._validate_cleanup_token(value["cleanup_token"]) == cleanup_token
        except _DuplicateJsonFieldError as error:
            raise RestartContextFileError(str(error)) from error
        except RestartContextFileError:
            raise
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise RestartContextFileError(
                "restart-context invalidation marker is malformed"
            ) from error

    @staticmethod
    def _decode_context(encoded: bytes) -> RestartContext:
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

    @classmethod
    def _encode_metadata(
        cls,
        cleanup_token: _RestartContextCleanupToken,
        context: bytes,
        ownership: _RestartContextOwnership | None,
    ) -> bytes:
        cls._validate_cleanup_token(cleanup_token)
        encoded = (
            json.dumps(
                {
                    "schema_version": _CONTEXT_METADATA_SCHEMA_VERSION,
                    "cleanup_token": cleanup_token,
                    "context_sha256": hashlib.sha256(context).hexdigest(),
                    "registration_id": (None if ownership is None else ownership.registration_id),
                    "registration_mutation_sequence": (
                        None if ownership is None else ownership.mutation_sequence
                    ),
                },
                allow_nan=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        if len(encoded) > _MAX_CONTEXT_METADATA_BYTES:
            raise RestartContextFileError("restart-context metadata is too large")
        return encoded

    @classmethod
    def _decode_metadata(
        cls,
        encoded: bytes,
        context: bytes,
    ) -> _RestartContextCleanupToken:
        cleanup_token, context_digest, _ = cls._decode_metadata_record(encoded)
        if not secrets.compare_digest(
            context_digest,
            hashlib.sha256(context).hexdigest(),
        ):
            raise RestartContextFileError("restart-context metadata does not match the context")
        return cleanup_token

    @classmethod
    def _decode_metadata_record(
        cls,
        encoded: bytes,
    ) -> tuple[_RestartContextCleanupToken, str, _RestartContextOwnership | None]:
        try:
            value = json.loads(
                encoded,
                object_pairs_hook=_strict_object,
            )
            if not isinstance(value, Mapping):
                raise RestartContextFileError(
                    "restart-context metadata JSON must contain an object"
                )
            expected = {
                "schema_version",
                "cleanup_token",
                "context_sha256",
                "registration_id",
                "registration_mutation_sequence",
            }
            if set(value) != expected:
                raise RestartContextFileError("restart-context metadata fields are invalid")
            if (
                type(value["schema_version"]) is not int
                or value["schema_version"] != _CONTEXT_METADATA_SCHEMA_VERSION
            ):
                raise RestartContextFileError("restart-context metadata schema version is invalid")
            cleanup_token = cls._validate_cleanup_token(value["cleanup_token"])
            context_digest = value["context_sha256"]
            if (
                not isinstance(context_digest, str)
                or len(context_digest) != 64
                or context_digest != context_digest.lower()
            ):
                raise RestartContextFileError("restart-context metadata digest is invalid")
            try:
                int(context_digest, 16)
            except ValueError as error:
                raise RestartContextFileError(
                    "restart-context metadata digest is invalid"
                ) from error
            registration_id = value["registration_id"]
            mutation_sequence = value["registration_mutation_sequence"]
            if registration_id is None and mutation_sequence is None:
                ownership = None
            elif (
                isinstance(registration_id, str)
                and registration_id
                and (type(mutation_sequence) is int and mutation_sequence >= 1)
            ):
                ownership = _RestartContextOwnership(
                    registration_id=registration_id,
                    mutation_sequence=mutation_sequence,
                )
            else:
                raise RestartContextFileError("restart-context metadata ownership is invalid")
            return cleanup_token, context_digest, ownership
        except _DuplicateJsonFieldError as error:
            raise RestartContextFileError(str(error)) from error
        except RestartContextFileError:
            raise
        except (TypeError, ValueError, UnicodeDecodeError) as error:
            raise RestartContextFileError("restart-context metadata is malformed") from error

    @staticmethod
    def _validate_replacement_ownership(
        *,
        existing: _RestartContextOwnership | None,
        replacement: _RestartContextOwnership | None,
    ) -> None:
        if existing is None:
            return
        if replacement is None:
            raise RestartContextFileError(
                "unowned restart context cannot replace registration-owned state"
            )
        if replacement.mutation_sequence < existing.mutation_sequence or (
            replacement.mutation_sequence == existing.mutation_sequence
            and replacement.registration_id != existing.registration_id
        ):
            raise RestartContextFileError(
                "stale registration cannot replace a newer restart context"
            )

    @staticmethod
    def _validate_cleanup_token(value: object) -> _RestartContextCleanupToken:
        if not isinstance(value, str) or len(value) != 64 or value != value.lower():
            raise RestartContextFileError("restart-context cleanup token is invalid")
        try:
            int(value, 16)
        except ValueError as error:
            raise RestartContextFileError("restart-context cleanup token is invalid") from error
        return value

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

    def _create_temporary_file(
        self,
        parent_descriptor: int,
        *,
        target_name: str | None = None,
    ) -> tuple[int, str]:
        filename = self.path.name if target_name is None else target_name
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        for _ in range(128):
            name = f".{filename}.{secrets.token_hex(16)}.tmp"
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

    def _invalidation_name(self, cleanup_token: _RestartContextCleanupToken) -> str:
        self._validate_cleanup_token(cleanup_token)
        identity = f"{self.path.name}\0{cleanup_token}".encode("utf-8")
        digest = hashlib.sha256(identity).hexdigest()
        return f".restart-context-invalidated-{digest}"

    def _metadata_name(self) -> str:
        digest = hashlib.sha256(self.path.name.encode("utf-8")).hexdigest()
        return f".restart-context-metadata-{digest}"

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
                values.get(
                    "max_replacement_generations",
                    _SUPPORTED_REPLACEMENT_GENERATIONS,
                ),
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
