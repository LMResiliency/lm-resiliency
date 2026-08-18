"""Minimal torchrun integration for manager-owned recovery plans."""

from __future__ import annotations

import hashlib
import json
import os
import secrets
import stat
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

from torch.distributed import PrefixStore, Store
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
from torch.distributed.elastic.rendezvous.c10d_rendezvous_backend import create_backend

from lm_resiliency.integrations.torchrun._protocol import (
    ProtocolValidationError,
    RestartContext,
    RestartPlan,
)

_BACKEND = "lm_resiliency"
_PREFIX = "lm_resiliency/simple/v1"
_MAX_CONTEXT_BYTES = 1 << 20
_MAX_REGISTRATION_BYTES = 1 << 20
_MAX_WORKER_CONFIG_BYTES = 1 << 20
_MACHINE_ID_PATH_ENV = "LM_RESILIENCY_MACHINE_ID_PATH"
_DEFAULT_MACHINE_ID_PATH = Path("/etc/machine-id")
_ALLOWED_CONFIG = {
    "is_host",
    "lm_resiliency_heartbeat_timeout_ms",
    "lm_resiliency_join_timeout_ms",
    "lm_resiliency_poll_interval_ms",
    "lm_resiliency_restart_context_path",
    "lm_resiliency_worker_config",
    "read_timeout",
    "store_type",
    "timeout",
}


class SimpleRuntimeError(RuntimeError):
    """Base error for the simplified torchrun integration."""


class RecoveryPlanConflict(SimpleRuntimeError):
    """Raised when a recovery plan conflicts with committed state."""


class RecoveryPlanCorrupt(SimpleRuntimeError):
    """Raised when a persisted recovery plan is malformed."""


class SimpleRestartContextFile:
    """Atomic owner-only storage for one canonical worker restart context."""

    def __init__(self, path: Path) -> None:
        if not isinstance(path, Path):
            raise TypeError("path must be pathlib.Path")
        if not path.is_absolute():
            raise ValueError("path must be absolute")
        self._path = path

    def write(self, context: RestartContext) -> None:
        if not isinstance(context, RestartContext):
            raise TypeError("context must be RestartContext")
        self.prepare()
        parent = self._path.parent
        encoded = context.to_json().encode("utf-8")
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise SimpleRuntimeError("restart context is too large")
        temporary = parent / f".{self._path.name}.{secrets.token_hex(16)}.tmp"
        descriptor = os.open(
            temporary,
            os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_WRONLY,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "wb", closefd=True) as stream:
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            directory = os.open(parent, os.O_DIRECTORY | os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass

    def read(self) -> RestartContext | None:
        try:
            metadata = self._path.lstat()
        except FileNotFoundError:
            return None
        if not stat.S_ISREG(metadata.st_mode):
            raise SimpleRuntimeError("restart context must be a regular file")
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise SimpleRuntimeError("restart context must be owner-only")
        encoded = self._path.read_bytes()
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise SimpleRuntimeError("restart context is too large")
        try:
            value = json.loads(encoded, object_pairs_hook=_reject_duplicate_fields)
            if not isinstance(value, Mapping):
                raise ValueError("restart context must be a JSON object")
            context = RestartContext.from_dict(value)
        except (TypeError, ValueError, ProtocolValidationError) as error:
            raise SimpleRuntimeError("restart context is malformed") from error
        if context.to_json().encode("utf-8") != encoded:
            raise SimpleRuntimeError("restart context is not canonical JSON")
        return context

    def clear(self) -> None:
        try:
            self._path.unlink()
        except FileNotFoundError:
            return
        directory = os.open(self._path.parent, os.O_DIRECTORY | os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)

    def prepare(self) -> None:
        """Create and validate the owner-only parent directory."""

        parent = self._path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._validate_private(parent)

    @staticmethod
    def _validate_private(path: Path) -> None:
        metadata = path.stat()
        if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
            raise SimpleRuntimeError("restart-context directory must be owner-only")


@dataclass(frozen=True, slots=True)
class SimpleRuntimeConfig:
    """Per-agent settings supplied through ``torchrun --rdzv-conf``."""

    run_id: str
    node_id: str
    min_nodes: int
    max_nodes: int
    restart_context_path: Path
    join_timeout_ms: int = 300_000
    poll_interval_ms: int = 250
    heartbeat_timeout_ms: int = 10_000
    worker_config: Path | None = None

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.node_id, "node_id")
        _positive_int(self.min_nodes, "min_nodes")
        _positive_int(self.max_nodes, "max_nodes")
        if self.max_nodes < self.min_nodes:
            raise ValueError("max_nodes must be greater than or equal to min_nodes")
        _positive_int(self.join_timeout_ms, "join_timeout_ms")
        _positive_int(self.poll_interval_ms, "poll_interval_ms")
        _positive_int(self.heartbeat_timeout_ms, "heartbeat_timeout_ms")
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise ValueError("restart_context_path must be absolute")
        if self.worker_config is not None:
            if not isinstance(self.worker_config, Path):
                raise TypeError("worker_config must be pathlib.Path")
            if not self.worker_config.is_absolute():
                raise ValueError("worker_config must be absolute")

    @classmethod
    def from_parameters(
        cls,
        params: RendezvousParameters,
        *,
        environment: Mapping[str, str] | None = None,
    ) -> SimpleRuntimeConfig:
        if not isinstance(params, RendezvousParameters):
            raise TypeError("params must be RendezvousParameters")
        if params.backend != _BACKEND:
            raise ValueError(f"rendezvous backend must be {_BACKEND!r}")
        unknown = set(params.config) - _ALLOWED_CONFIG
        if unknown:
            raise ValueError(f"unknown rendezvous configuration: {sorted(unknown)!r}")
        environ = os.environ if environment is None else environment
        machine_id_path = Path(
            environ.get(_MACHINE_ID_PATH_ENV, str(_DEFAULT_MACHINE_ID_PATH))
        ).expanduser()
        if not machine_id_path.is_absolute():
            raise ValueError(f"{_MACHINE_ID_PATH_ENV} must be an absolute path")
        node_id = _machine_node_id(machine_id_path)
        restart_context_path = Path(
            _nonempty(
                params.get("lm_resiliency_restart_context_path"),
                "lm_resiliency_restart_context_path",
            )
        )
        restart_context_path = restart_context_path.expanduser()
        worker_config_value = _optional_nonempty(
            params.get("lm_resiliency_worker_config"),
            "lm_resiliency_worker_config",
        )
        config = cls(
            run_id=params.run_id,
            node_id=node_id,
            min_nodes=params.min_nodes,
            max_nodes=params.max_nodes,
            restart_context_path=restart_context_path,
            join_timeout_ms=_optional_int(
                params.get("lm_resiliency_join_timeout_ms"),
                "lm_resiliency_join_timeout_ms",
                300_000,
            ),
            poll_interval_ms=_optional_int(
                params.get("lm_resiliency_poll_interval_ms"),
                "lm_resiliency_poll_interval_ms",
                250,
            ),
            heartbeat_timeout_ms=_optional_int(
                params.get("lm_resiliency_heartbeat_timeout_ms"),
                "lm_resiliency_heartbeat_timeout_ms",
                10_000,
            ),
            worker_config=(
                None if worker_config_value is None else Path(worker_config_value).expanduser()
            ),
        )
        return config


@dataclass(frozen=True, slots=True)
class _NodeAssignment:
    """One logical node slot used only for rendezvous admission."""

    logical_node_slot: int
    node_id: str


class SimpleRecoveryPlanStore:
    """One create-once recovery plan per generation in a c10d store."""

    def __init__(self, store: Store, *, run_id: str) -> None:
        if not isinstance(store, Store):
            raise TypeError("store must be torch.distributed.Store")
        self._store = store
        self._run_id = _nonempty(run_id, "run_id")
        digest = hashlib.sha256(run_id.encode("utf-8")).hexdigest()
        self._prefix = f"{_PREFIX}/runs/{digest}"
        self._generation_key = f"{self._prefix}/generation"
        self._registrations_key = f"{self._prefix}/initial/registrations"
        self._initial_nodes_key = f"{self._prefix}/initial/nodes"
        self._max_nodes_key = f"{self._prefix}/initial/max_nodes"
        self._store.compare_set(self._generation_key, b"", b"0")

    @property
    def prefix(self) -> str:
        return self._prefix

    def current_generation(self) -> int:
        return _decode_generation(self._store.get(self._generation_key))

    def register_node(self, node_id: str, agent_id: str, *, max_nodes: int) -> None:
        node_id = _nonempty(node_id, "node_id")
        agent_id = _nonempty(agent_id, "agent_id")
        _positive_int(max_nodes, "max_nodes")
        encoded_max_nodes = str(max_nodes).encode("ascii")
        stored_max_nodes = self._store.compare_set(
            self._max_nodes_key,
            b"",
            encoded_max_nodes,
        )
        if stored_max_nodes != encoded_max_nodes:
            raise RecoveryPlanCorrupt("torchrun agents disagree on max_nodes")
        record = _canonical_json(
            {
                "agent_id": agent_id,
                "node_id": node_id,
                "schema_version": 1,
            }
        )
        self._store.append(self._registrations_key, record + b"\n")
        self._decode_registrations(max_nodes=max_nodes)

    def ensure_initial_nodes(
        self,
        *,
        min_nodes: int,
        max_nodes: int,
    ) -> tuple[str, ...] | None:
        _positive_int(min_nodes, "min_nodes")
        _positive_int(max_nodes, "max_nodes")
        if max_nodes < min_nodes:
            raise ValueError("max_nodes must be greater than or equal to min_nodes")
        registrations = self._decode_registrations(max_nodes=max_nodes)
        if len(registrations) < min_nodes:
            return None
        expected = tuple(node_id for node_id, _agent_id in registrations[:min_nodes])
        encoded = _canonical_json(
            {
                "node_ids": list(expected),
                "schema_version": 1,
            }
        )
        stored = self._store.compare_set(self._initial_nodes_key, b"", encoded)
        selected = self._decode_initial_nodes(stored, expected_count=min_nodes)
        if selected != expected:
            raise RecoveryPlanCorrupt("initial node assignment conflicts with registration order")
        return selected

    def read_initial_nodes(self) -> tuple[str, ...] | None:
        if not self._store.check([self._initial_nodes_key]):
            return None
        return self._decode_initial_nodes(self._store.get(self._initial_nodes_key))

    def registered_nodes(self, *, max_nodes: int | None = None) -> tuple[str, ...]:
        if self._store.check([self._max_nodes_key]):
            committed_max_nodes = _decode_positive_integer(
                self._store.get(self._max_nodes_key),
                "registered max_nodes",
            )
        elif max_nodes is not None:
            committed_max_nodes = _positive_int(max_nodes, "max_nodes")
            self._store.compare_set(
                self._max_nodes_key,
                b"",
                str(committed_max_nodes).encode("ascii"),
            )
        else:
            raise RecoveryPlanCorrupt("registered allocation is missing max_nodes")
        if max_nodes is not None and max_nodes != committed_max_nodes:
            raise RecoveryPlanCorrupt("manager max_nodes does not match registered allocation")
        return tuple(
            node_id
            for node_id, _agent_id in self._decode_registrations(max_nodes=committed_max_nodes)
        )

    def read(self, generation: int) -> RestartPlan | None:
        if generation < 1:
            return None
        key = self._plan_key(generation)
        if not self._store.check([key]):
            return None
        return self._decode_plan(self._store.get(key), expected_generation=generation)

    def publish(self, plan: RestartPlan) -> None:
        if not isinstance(plan, RestartPlan):
            raise TypeError("plan must be RestartPlan")
        if plan.run_id != self._run_id:
            raise ValueError("plan belongs to another run")
        current_generation = self.current_generation()
        if current_generation == plan.to_generation:
            if self.read(plan.to_generation) != plan:
                raise RecoveryPlanConflict("another plan is committed for this generation")
            return
        if current_generation != plan.from_generation:
            raise RecoveryPlanConflict(
                f"cannot advance generation {plan.from_generation} to {plan.to_generation}; "
                f"current generation is {current_generation}"
            )
        encoded = plan.to_json().encode("utf-8")
        key = self._plan_key(plan.to_generation)
        stored = self._store.compare_set(key, b"", encoded)
        if stored != encoded:
            existing = self._decode_plan(stored, expected_generation=plan.to_generation)
            if existing != plan:
                raise RecoveryPlanConflict("another plan is committed for this generation")
        current = self._store.compare_set(
            self._generation_key,
            str(plan.from_generation).encode("ascii"),
            str(plan.to_generation).encode("ascii"),
        )
        observed = _decode_generation(current)
        if observed != plan.to_generation:
            raise RecoveryPlanConflict(
                f"cannot advance generation {plan.from_generation} to {plan.to_generation}; "
                f"current generation is {observed}"
            )

    def close_run(self) -> None:
        """Wake parked agents and prevent further rendezvous."""
        self._store.set(f"{self._prefix}/closed", b"1")

    def _plan_key(self, generation: int) -> str:
        return f"{self._prefix}/plans/{generation}"

    def _decode_registrations(
        self,
        *,
        max_nodes: int,
    ) -> tuple[tuple[str, str], ...]:
        encoded = self._store.get(self._registrations_key)
        if len(encoded) > _MAX_REGISTRATION_BYTES:
            raise RecoveryPlanCorrupt("initial node registrations are too large")
        registrations: list[tuple[str, str]] = []
        node_agents: dict[str, str] = {}
        agent_nodes: dict[str, str] = {}
        for line in encoded.splitlines():
            try:
                value = json.loads(line, object_pairs_hook=_reject_duplicate_fields)
            except (TypeError, ValueError, UnicodeError) as error:
                raise RecoveryPlanCorrupt("initial node registration is malformed") from error
            if (
                not isinstance(value, Mapping)
                or set(value) != {"agent_id", "node_id", "schema_version"}
                or type(value.get("schema_version")) is not int
                or value.get("schema_version") != 1
            ):
                raise RecoveryPlanCorrupt("initial node registration has invalid fields")
            try:
                node_id = _nonempty(value.get("node_id"), "node_id")
                agent_id = _nonempty(value.get("agent_id"), "agent_id")
            except ValueError as error:
                raise RecoveryPlanCorrupt(
                    "initial node registration has invalid identities"
                ) from error
            if _canonical_json(dict(value)) != line:
                raise RecoveryPlanCorrupt("initial node registration is not canonical JSON")
            existing_agent = node_agents.get(node_id)
            existing_node = agent_nodes.get(agent_id)
            if existing_agent is not None and existing_agent != agent_id:
                raise RecoveryPlanCorrupt(
                    "multiple torchrun agents resolved to the same machine identity"
                )
            if existing_node is not None and existing_node != node_id:
                raise RecoveryPlanCorrupt(
                    "one torchrun agent registered multiple machine identities"
                )
            if existing_agent is None:
                node_agents[node_id] = agent_id
                agent_nodes[agent_id] = node_id
                registrations.append((node_id, agent_id))
        if len(registrations) > max_nodes:
            raise RecoveryPlanCorrupt("registered nodes exceed torchrun max_nodes")
        return tuple(registrations)

    @staticmethod
    def _decode_initial_nodes(
        encoded: bytes,
        *,
        expected_count: int | None = None,
    ) -> tuple[str, ...]:
        try:
            value = json.loads(encoded, object_pairs_hook=_reject_duplicate_fields)
        except (TypeError, ValueError, UnicodeError) as error:
            raise RecoveryPlanCorrupt("initial node assignment is malformed") from error
        if (
            not isinstance(value, Mapping)
            or set(value) != {"node_ids", "schema_version"}
            or type(value.get("schema_version")) is not int
            or value.get("schema_version") != 1
            or not isinstance(value.get("node_ids"), list)
        ):
            raise RecoveryPlanCorrupt("initial node assignment has invalid fields")
        try:
            node_ids = tuple(_nonempty(item, "node_id") for item in value["node_ids"])
        except ValueError as error:
            raise RecoveryPlanCorrupt("initial node assignment has invalid identities") from error
        if not node_ids or len(node_ids) != len(set(node_ids)):
            raise RecoveryPlanCorrupt("initial node assignment must be non-empty and unique")
        if expected_count is not None and len(node_ids) != expected_count:
            raise RecoveryPlanCorrupt("initial node assignment has the wrong size")
        if _canonical_json(dict(value)) != encoded:
            raise RecoveryPlanCorrupt("initial node assignment is not canonical JSON")
        return node_ids

    def _decode_plan(self, encoded: bytes, *, expected_generation: int) -> RestartPlan:
        try:
            value = json.loads(encoded, object_pairs_hook=_reject_duplicate_fields)
            if not isinstance(value, Mapping):
                raise ValueError("plan must be a JSON object")
            plan = RestartPlan.from_dict(value)
        except (TypeError, ValueError, ProtocolValidationError) as error:
            raise RecoveryPlanCorrupt("persisted recovery plan is malformed") from error
        if plan.to_json().encode("utf-8") != encoded:
            raise RecoveryPlanCorrupt("persisted recovery plan is not canonical JSON")
        if plan.run_id != self._run_id or plan.to_generation != expected_generation:
            raise RecoveryPlanCorrupt("persisted recovery plan has the wrong identity")
        return plan


class SimpleRendezvousHandler(RendezvousHandler):
    """Consume manager-owned plans and park unselected standby agents."""

    def __init__(
        self,
        config: SimpleRuntimeConfig,
        *,
        store: Store,
        local_addr: str | None,
        clock: Callable[[], int] = lambda: time.time_ns() // 1_000_000,
        monotonic_clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._store = store
        self._local_addr = local_addr
        self._clock = clock
        self._monotonic_clock = monotonic_clock
        self._context = SimpleRestartContextFile(config.restart_context_path)
        self._context.prepare()
        self._worker_policy_digest = _worker_policy_digest(config.worker_config)
        self._plans = SimpleRecoveryPlanStore(store, run_id=config.run_id)
        self._agent_id = secrets.token_hex(16)
        self._plans.register_node(
            config.node_id,
            self._agent_id,
            max_nodes=config.max_nodes,
        )
        self._closed = threading.Event()
        self._heartbeat_error: Exception | None = None
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            name=f"lm-resiliency-simple-heartbeat-{config.node_id}",
            daemon=True,
        )
        self._last_generation: int | None = None
        self._last_assignment: tuple[_NodeAssignment, ...] = ()
        self._heartbeat_observations: dict[str, tuple[str, int, float, bool]] = {}
        self._heartbeat_observation_lock = threading.Lock()
        self._heartbeat.start()

    @property
    def use_agent_store(self) -> bool:
        return False

    def get_backend(self) -> str:
        return _BACKEND

    def get_run_id(self) -> str:
        return self._config.run_id

    def next_rendezvous(self) -> RendezvousInfo:
        admission_deadline = self._monotonic_clock() + self._config.join_timeout_ms / 1_000
        while True:
            self._raise_if_closed()
            self._raise_heartbeat_error()
            generation = self._plans.current_generation()
            if self._last_generation is not None and generation <= self._last_generation:
                if self._monotonic_clock() >= admission_deadline:
                    raise RendezvousTimeoutError(
                        "timed out waiting for a manager-owned successor generation"
                    )
                self._closed.wait(self._config.poll_interval_ms / 1_000)
                continue
            plan = self._plans.read(generation)
            assignment = self._assignment(generation, plan)
            if assignment is None:
                if self._monotonic_clock() >= admission_deadline:
                    raise RendezvousTimeoutError("timed out waiting for initial node registrations")
                self._closed.wait(self._config.poll_interval_ms / 1_000)
                continue
            slot = next(
                (
                    item.logical_node_slot
                    for item in assignment
                    if item.node_id == self._config.node_id
                ),
                None,
            )
            if slot is None:
                self._closed.wait(self._config.poll_interval_ms / 1_000)
                continue
            deadline = self._monotonic_clock() + self._config.join_timeout_ms / 1_000
            if generation == 0:
                try:
                    self._context.clear()
                except SimpleRuntimeError as error:
                    raise RendezvousStateError("failed to clear stale restart context") from error
            else:
                assert plan is not None
                self._check_plan_deadline(plan)
                try:
                    self._context.write(RestartContext.from_plan(plan, self._config.node_id))
                except (ProtocolValidationError, SimpleRuntimeError) as error:
                    raise RendezvousStateError("failed to publish restart context") from error
            self._store.set(
                self._ready_key(generation, self._config.node_id),
                _canonical_json(
                    {
                        "agent_id": self._agent_id,
                        "policy_digest": self._worker_policy_digest,
                        "schema_version": 1,
                    }
                ),
            )
            if not self._wait_for_group(generation, assignment, deadline):
                continue
            if self._plans.current_generation() != generation:
                continue
            if plan is not None:
                self._check_plan_deadline(plan)
            group_store = PrefixStore(
                f"{self._plans.prefix}/bootstrap/{generation}/",
                self._store,
            )
            self._last_generation = generation
            self._last_assignment = assignment
            bootstrap = self._build_bootstrap(slot, group_store, deadline)
            if plan is not None:
                self._check_plan_deadline(plan)
            from .worker_adapter import configure_worker_generation_environment

            configure_worker_generation_environment(generation)
            return RendezvousInfo(
                store=group_store,
                rank=slot,
                world_size=len(assignment),
                bootstrap_store_info=bootstrap,
            )

    def num_nodes_waiting(self) -> int:
        if self._last_generation is None:
            return 0
        try:
            generation = self._plans.current_generation()
            if generation <= self._last_generation:
                return 0
            plan = self._plans.read(generation)
            if plan is None:
                return 0
            try:
                self._check_plan_deadline(plan)
            except RendezvousTimeoutError:
                return 0
            previous = {item.node_id for item in self._last_assignment}
            selected = {item.node_id for item in plan.slot_assignments} - previous
            if not selected:
                return 1
            return sum(self._heartbeat_is_live(node_id) for node_id in selected)
        except (RecoveryPlanCorrupt, ValueError):
            raise RendezvousStateError("recovery-plan state is corrupt")
        except Exception:
            return 0

    def is_closed(self) -> bool:
        if self._closed.is_set():
            return True
        return self._store.check([self._closed_key()])

    def set_closed(self) -> None:
        self._store.set(self._closed_key(), b"1")
        self._closed.set()

    def shutdown(self) -> bool:
        self._closed.set()
        self._heartbeat.join(timeout=max(self._config.poll_interval_ms / 500, 0.1))
        return not self._heartbeat.is_alive()

    def _assignment(
        self,
        generation: int,
        plan: RestartPlan | None,
    ) -> tuple[_NodeAssignment, ...] | None:
        if generation == 0:
            node_ids = self._plans.ensure_initial_nodes(
                min_nodes=self._config.min_nodes,
                max_nodes=self._config.max_nodes,
            )
            if node_ids is None:
                return None
            return tuple(
                _NodeAssignment(
                    logical_node_slot=slot,
                    node_id=node_id,
                )
                for slot, node_id in enumerate(node_ids)
            )
        if plan is None:
            raise RendezvousStateError("current generation has no recovery plan")
        return tuple(
            _NodeAssignment(
                logical_node_slot=item.logical_node_slot,
                node_id=item.node_id,
            )
            for item in plan.slot_assignments
        )

    def _wait_for_group(
        self,
        generation: int,
        assignment: tuple[_NodeAssignment, ...],
        deadline: float,
    ) -> bool:
        keys = [self._ready_key(generation, item.node_id) for item in assignment]
        while True:
            self._raise_if_closed()
            self._raise_heartbeat_error()
            if self._plans.current_generation() != generation:
                return False
            if self._store.check(keys):
                records = [
                    self._ready_agent_record(generation, item.node_id) for item in assignment
                ]
                if all(record is not None for record in records):
                    digests = {record[1] for record in records if record is not None}
                    if len(digests) != 1:
                        raise RendezvousStateError(
                            "assigned nodes disagree on the LM Resiliency worker policy"
                        )
                    if all(
                        self._heartbeat_is_live(item.node_id, agent_id=record[0])
                        for item, record in zip(assignment, records)
                        if record is not None
                    ):
                        return True
            if self._monotonic_clock() >= deadline:
                raise RendezvousTimeoutError("timed out waiting for assigned nodes")
            self._closed.wait(self._config.poll_interval_ms / 1_000)

    def _build_bootstrap(
        self,
        slot: int,
        store: Store,
        deadline: float,
    ) -> RendezvousStoreInfo:
        remaining = deadline - self._monotonic_clock()
        if remaining <= 0:
            raise RendezvousTimeoutError("timed out before bootstrap")
        store.set_timeout(timedelta(seconds=remaining))
        try:
            return RendezvousStoreInfo.build(slot, store, self._local_addr)
        except Exception as error:
            raise RendezvousConnectionError("failed to build worker bootstrap store") from error

    def _check_plan_deadline(self, plan: RestartPlan) -> None:
        now = self._clock()
        if isinstance(now, bool) or not isinstance(now, int) or now < 1:
            raise RendezvousStateError("clock returned an invalid unix time")
        if now >= plan.restart_deadline_unix_ms:
            raise RendezvousTimeoutError("recovery plan deadline elapsed")

    def _heartbeat_loop(self) -> None:
        interval = max(self._config.heartbeat_timeout_ms / 3_000, 0.1)
        sequence = 0
        while not self._closed.is_set():
            try:
                payload = json.dumps(
                    {
                        "agent_id": self._agent_id,
                        "node_id": self._config.node_id,
                        "sequence": sequence,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                self._store.set(self._heartbeat_key(self._config.node_id), payload)
                sequence += 1
            except Exception as error:
                self._heartbeat_error = error
                return
            self._closed.wait(interval)

    def _ready_agent_record(self, generation: int, node_id: str) -> tuple[str, str] | None:
        try:
            payload = json.loads(
                self._store.get(self._ready_key(generation, node_id)),
                object_pairs_hook=_reject_duplicate_fields,
            )
        except (TypeError, ValueError, UnicodeError, json.JSONDecodeError):
            return None
        if (
            not isinstance(payload, Mapping)
            or set(payload) != {"agent_id", "policy_digest", "schema_version"}
            or type(payload.get("schema_version")) is not int
            or payload.get("schema_version") != 1
        ):
            return None
        try:
            agent_id = _nonempty(payload.get("agent_id"), "agent_id")
            policy_digest = _nonempty(payload.get("policy_digest"), "policy_digest")
        except ValueError:
            return None
        return agent_id, policy_digest

    def _heartbeat_is_live(self, node_id: str, *, agent_id: str | None = None) -> bool:
        key = self._heartbeat_key(node_id)
        if not self._store.check([key]):
            return False
        try:
            payload = json.loads(
                self._store.get(key),
                object_pairs_hook=_reject_duplicate_fields,
            )
            if not isinstance(payload, Mapping):
                return False
            observed_agent_id = payload.get("agent_id")
            sequence = payload.get("sequence")
            if (
                set(payload) != {"agent_id", "node_id", "sequence"}
                or payload.get("node_id") != node_id
                or not isinstance(observed_agent_id, str)
                or not observed_agent_id
                or (agent_id is not None and observed_agent_id != agent_id)
                or isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
            ):
                return False
            now = self._monotonic_clock()
            if isinstance(now, bool) or not isinstance(now, (int, float)) or now < 0:
                return False
            with self._heartbeat_observation_lock:
                previous = self._heartbeat_observations.get(node_id)
                if previous is None or previous[0] != observed_agent_id or sequence < previous[1]:
                    self._heartbeat_observations[node_id] = (
                        observed_agent_id,
                        sequence,
                        float(now),
                        False,
                    )
                    return False
                if sequence > previous[1]:
                    self._heartbeat_observations[node_id] = (
                        observed_agent_id,
                        sequence,
                        float(now),
                        True,
                    )
                    return True
                return previous[3] and (
                    0 <= float(now) - previous[2] < self._config.heartbeat_timeout_ms / 1_000
                )
        except (ArithmeticError, TypeError, ValueError, json.JSONDecodeError):
            return False

    def _raise_if_closed(self) -> None:
        if self._closed.is_set():
            raise RendezvousClosedError
        if self._store.check([self._closed_key()]):
            raise SystemExit(0)

    def _raise_heartbeat_error(self) -> None:
        if self._heartbeat_error is not None:
            raise RendezvousConnectionError("standby heartbeat failed") from self._heartbeat_error

    def _ready_key(self, generation: int, node_id: str) -> str:
        return f"{self._plans.prefix}/ready/{generation}/{_key(node_id)}"

    def _heartbeat_key(self, node_id: str) -> str:
        return f"{self._plans.prefix}/heartbeats/{_key(node_id)}"

    def _closed_key(self) -> str:
        return f"{self._plans.prefix}/closed"


def get_rendezvous_handler_creator() -> Callable[[RendezvousParameters], RendezvousHandler]:
    """Return the handler creator discovered through ``torchrun.handlers``."""

    return _create_rendezvous_handler


def _create_rendezvous_handler(params: RendezvousParameters) -> RendezvousHandler:
    try:
        config = SimpleRuntimeConfig.from_parameters(params)
        backend, store = create_backend(params)
        handler = SimpleRendezvousHandler(
            config,
            store=store,
            local_addr=params.local_addr,
        )
        from .worker_adapter import (
            configure_worker_bootstrap_environment,
            configure_worker_context_environment,
            disable_worker_bootstrap_environment,
        )

        try:
            configure_worker_context_environment(
                run_id=config.run_id,
                node_id=config.node_id,
                restart_context_path=config.restart_context_path,
            )
        except Exception as error:
            handler.shutdown()
            raise SimpleRuntimeError("failed to configure worker context") from error
        if config.worker_config is not None:
            try:
                configure_worker_bootstrap_environment(
                    run_id=config.run_id,
                    node_id=config.node_id,
                    restart_context_path=config.restart_context_path,
                    config_path=config.worker_config,
                    policy_digest=handler._worker_policy_digest,
                )
            except Exception as error:
                handler.shutdown()
                raise SimpleRuntimeError("failed to configure worker bootstrap") from error
        else:
            disable_worker_bootstrap_environment()
        handler._backend_owner = backend  # Keep the backend alive with its store.
        return handler
    except (TypeError, ValueError, SimpleRuntimeError) as error:
        raise RendezvousConnectionError("failed to initialize lm_resiliency handler") from error


def _optional_nonempty(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _nonempty(value, name)


def _optional_int(value: object, name: str, default: int) -> int:
    return default if value is None else _positive_int(value, name)


def _decode_positive_integer(encoded: bytes, name: str) -> int:
    try:
        value = encoded.decode("ascii")
    except UnicodeError as error:
        raise RecoveryPlanCorrupt(f"{name} is malformed") from error
    try:
        return _positive_int(value, name)
    except ValueError as error:
        raise RecoveryPlanCorrupt(f"{name} is malformed") from error


def _worker_policy_digest(path: Path | None) -> str:
    if path is None:
        return hashlib.sha256(b"lm-resiliency/worker-policy/none/v1").hexdigest()
    try:
        encoded = path.read_bytes()
    except OSError as error:
        raise SimpleRuntimeError(f"failed to read worker config {path}") from error
    if len(encoded) > _MAX_WORKER_CONFIG_BYTES:
        raise SimpleRuntimeError("worker config is too large")
    from .worker_adapter import (
        TorchrunWorkerAdapterError,
        _validate_worker_config_bytes,
    )

    try:
        _validate_worker_config_bytes(encoded, path=path)
    except TorchrunWorkerAdapterError as error:
        raise SimpleRuntimeError(f"invalid worker config {path}: {error}") from error
    return hashlib.sha256(b"lm-resiliency/worker-policy/v1\0" + encoded).hexdigest()


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a positive integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and value.isdecimal():
        normalized = int(value)
    else:
        raise ValueError(f"{name} must be a positive integer")
    if normalized < 1:
        raise ValueError(f"{name} must be a positive integer")
    return normalized


def _nonempty(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _key(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _machine_node_id(path: Path) -> str:
    try:
        metadata = path.stat()
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("machine identity path must be a regular file")
        encoded = path.read_bytes()
    except OSError as error:
        raise ValueError(f"failed to read machine identity from {path}") from error
    if len(encoded) > 128:
        raise ValueError("machine identity file is too large")
    try:
        machine_id = encoded.decode("ascii").strip().lower()
    except UnicodeDecodeError as error:
        raise ValueError("machine identity must be ASCII") from error
    return _node_id_from_machine_id(machine_id)


def _node_id_from_machine_id(machine_id: str) -> str:
    if (
        len(machine_id) != 32
        or any(character not in "0123456789abcdef" for character in machine_id)
        or machine_id == "0" * 32
    ):
        raise ValueError("machine identity must be 32 nonzero hexadecimal characters")
    digest = hashlib.sha256(
        b"lm-resiliency/torchrun/node/v1\0" + machine_id.encode("ascii")
    ).hexdigest()
    return f"machine-{digest}"


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_generation(encoded: bytes) -> int:
    try:
        generation = int(encoded.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as error:
        raise RecoveryPlanCorrupt("current generation is malformed") from error
    if generation < 0:
        raise RecoveryPlanCorrupt("current generation must not be negative")
    return generation


def _reject_duplicate_fields(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"duplicate JSON field {key!r}")
        value[key] = item
    return value


__all__ = [
    "RecoveryPlanConflict",
    "RecoveryPlanCorrupt",
    "SimpleRecoveryPlanStore",
    "SimpleRestartContextFile",
    "SimpleRendezvousHandler",
    "SimpleRuntimeConfig",
    "get_rendezvous_handler_creator",
]
