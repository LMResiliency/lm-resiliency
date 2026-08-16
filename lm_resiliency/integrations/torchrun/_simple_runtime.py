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
    SlotAssignment,
)

_BACKEND = "lm_resiliency"
_PREFIX = "lm_resiliency/simple/v1"
_MAX_CONTEXT_BYTES = 1 << 20
_ALLOWED_CONFIG = {
    "active_nodes",
    "heartbeat_timeout_ms",
    "is_host",
    "join_timeout_ms",
    "local_world_size",
    "node_id",
    "poll_interval_ms",
    "read_timeout",
    "restart_context_path",
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
        parent = self._path.parent
        parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        self._validate_private(parent)
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
    active_nodes: tuple[str, ...]
    local_world_size: int
    restart_context_path: Path
    join_timeout_ms: int = 300_000
    poll_interval_ms: int = 250
    heartbeat_timeout_ms: int = 10_000

    def __post_init__(self) -> None:
        _nonempty(self.run_id, "run_id")
        _nonempty(self.node_id, "node_id")
        if not self.active_nodes:
            raise ValueError("active_nodes must not be empty")
        if len(set(self.active_nodes)) != len(self.active_nodes):
            raise ValueError("active_nodes must contain unique node IDs")
        for node_id in self.active_nodes:
            _nonempty(node_id, "active_nodes item")
        _positive_int(self.local_world_size, "local_world_size")
        _positive_int(self.join_timeout_ms, "join_timeout_ms")
        _positive_int(self.poll_interval_ms, "poll_interval_ms")
        _positive_int(self.heartbeat_timeout_ms, "heartbeat_timeout_ms")
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise ValueError("restart_context_path must be absolute")

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
        node_id = _configured(
            params.get("node_id"),
            environ.get("LM_RESILIENCY_NODE_ID"),
            "node_id",
        )
        active_nodes = tuple(
            item.strip()
            for item in _configured(
                params.get("active_nodes"),
                environ.get("LM_RESILIENCY_ACTIVE_NODES"),
                "active_nodes",
            ).split(";")
            if item.strip()
        )
        local_world_size = _configured_int(
            params.get("local_world_size"),
            environ.get("LM_RESILIENCY_LOCAL_WORLD_SIZE"),
            "local_world_size",
        )
        restart_context_path = Path(
            _configured(
                params.get("restart_context_path"),
                environ.get("LM_RESILIENCY_RESTART_CONTEXT"),
                "restart_context_path",
            )
        ).expanduser()
        config = cls(
            run_id=params.run_id,
            node_id=node_id,
            active_nodes=active_nodes,
            local_world_size=local_world_size,
            restart_context_path=restart_context_path,
            join_timeout_ms=_optional_int(
                params.get("join_timeout_ms"),
                "join_timeout_ms",
                300_000,
            ),
            poll_interval_ms=_optional_int(
                params.get("poll_interval_ms"),
                "poll_interval_ms",
                250,
            ),
            heartbeat_timeout_ms=_optional_int(
                params.get("heartbeat_timeout_ms"),
                "heartbeat_timeout_ms",
                10_000,
            ),
        )
        if len(config.active_nodes) != params.min_nodes:
            raise ValueError("active_nodes must contain exactly min_nodes node IDs")
        if params.max_nodes < len(config.active_nodes):
            raise ValueError("max_nodes cannot be smaller than active_nodes")
        return config


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
        self._store.compare_set(self._generation_key, b"", b"0")

    @property
    def prefix(self) -> str:
        return self._prefix

    def current_generation(self) -> int:
        return _decode_generation(self._store.get(self._generation_key))

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
        self._plans = SimpleRecoveryPlanStore(store, run_id=config.run_id)
        self._context = SimpleRestartContextFile(config.restart_context_path)
        self._agent_id = secrets.token_hex(16)
        self._closed = threading.Event()
        self._heartbeat_error: Exception | None = None
        self._heartbeat = threading.Thread(
            target=self._heartbeat_loop,
            name=f"lm-resiliency-simple-heartbeat-{config.node_id}",
            daemon=True,
        )
        self._last_generation: int | None = None
        self._last_assignment: tuple[SlotAssignment, ...] = ()
        self._heartbeat.start()

    @property
    def use_agent_store(self) -> bool:
        return False

    def get_backend(self) -> str:
        return _BACKEND

    def get_run_id(self) -> str:
        return self._config.run_id

    def next_rendezvous(self) -> RendezvousInfo:
        while True:
            self._raise_if_closed()
            self._raise_heartbeat_error()
            generation = self._plans.current_generation()
            plan = self._plans.read(generation)
            assignment = self._assignment(generation, plan)
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
                self._agent_id.encode("utf-8"),
            )
            self._wait_for_group(generation, assignment, deadline)
            if self._plans.current_generation() != generation:
                continue
            if plan is not None:
                self._check_plan_deadline(plan)
            group_store = PrefixStore(
                f"{self._plans.prefix}/bootstrap/{generation}/",
                self._store,
            )
            bootstrap = self._build_bootstrap(slot, group_store, deadline)
            self._last_generation = generation
            self._last_assignment = assignment
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
            previous = {item.node_id for item in self._last_assignment}
            selected = {item.node_id for item in plan.slot_assignments} - previous
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
    ) -> tuple[SlotAssignment, ...]:
        if generation == 0:
            return tuple(
                SlotAssignment(
                    logical_node_slot=slot,
                    node_id=node_id,
                    first_global_rank=slot * self._config.local_world_size,
                    local_world_size=self._config.local_world_size,
                )
                for slot, node_id in enumerate(self._config.active_nodes)
            )
        if plan is None:
            raise RendezvousStateError("current generation has no recovery plan")
        for item in plan.slot_assignments:
            if item.local_world_size != self._config.local_world_size:
                raise RendezvousStateError("recovery plan changes local_world_size")
        return plan.slot_assignments

    def _wait_for_group(
        self,
        generation: int,
        assignment: tuple[SlotAssignment, ...],
        deadline: float,
    ) -> None:
        keys = [self._ready_key(generation, item.node_id) for item in assignment]
        while not self._store.check(keys):
            self._raise_if_closed()
            self._raise_heartbeat_error()
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
        while not self._closed.is_set():
            try:
                payload = json.dumps(
                    {
                        "agent_id": self._agent_id,
                        "local_world_size": self._config.local_world_size,
                        "node_id": self._config.node_id,
                        "updated_at_unix_ms": self._clock(),
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
                self._store.set(self._heartbeat_key(self._config.node_id), payload)
            except Exception as error:
                self._heartbeat_error = error
                return
            self._closed.wait(interval)

    def _heartbeat_is_live(self, node_id: str) -> bool:
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
            updated = payload.get("updated_at_unix_ms")
            local_world_size = payload.get("local_world_size")
            return (
                isinstance(updated, int)
                and not isinstance(updated, bool)
                and self._clock() - updated < self._config.heartbeat_timeout_ms
                and local_world_size == self._config.local_world_size
            )
        except (TypeError, ValueError, json.JSONDecodeError):
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
        handler._backend_owner = backend  # Keep the backend alive with its store.
        return handler
    except (TypeError, ValueError, SimpleRuntimeError) as error:
        raise RendezvousConnectionError("failed to initialize lm_resiliency handler") from error


def _configured(primary: object, fallback: object, name: str) -> str:
    primary_value = None if primary is None else _nonempty(primary, name)
    fallback_value = None if fallback is None else _nonempty(fallback, name)
    if primary_value is not None and fallback_value is not None and primary_value != fallback_value:
        raise ValueError(f"conflicting {name} values")
    value = primary_value if primary_value is not None else fallback_value
    if value is None:
        raise ValueError(f"{name} must be configured")
    return value


def _configured_int(primary: object, fallback: object, name: str) -> int:
    return _positive_int(_configured(primary, fallback, name), name)


def _optional_int(value: object, name: str, default: int) -> int:
    return default if value is None else _positive_int(value, name)


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
