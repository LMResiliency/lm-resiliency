"""Runtime configuration and node-local restart-context handoff for torchrun."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import secrets
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, ClassVar

try:
    _toml = importlib.import_module("tomllib")
except ModuleNotFoundError:  # pragma: no cover - exercised by the Python 3.10 CI job.
    _toml = importlib.import_module("tomli")

from torch.distributed.elastic.rendezvous import RendezvousParameters

from lm_resiliency.integrations.torchrun._protocol import AgentIdentity, RestartContext

_BACKEND = "lm_resiliency"
_NODE_ID_ENV = "LM_RESILIENCY_NODE_ID"
_LOCAL_WORLD_SIZE_ENV = "LM_RESILIENCY_LOCAL_WORLD_SIZE"
_ENVIRONMENT_DIGEST_ENV = "LM_RESILIENCY_ENVIRONMENT_DIGEST"
_RESTART_CONTEXT_ENV = "LM_RESILIENCY_RESTART_CONTEXT"
_MAX_CONTEXT_BYTES = 64 * 1024
_MAX_POLICY_BYTES = 64 * 1024
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
    "restart_context_path",
}


class TorchrunRuntimeConfigError(ValueError):
    """Raised when torchrun runtime configuration is unsafe or contradictory."""


class RestartContextFileError(RuntimeError):
    """Raised when the node-local restart-context file is unsafe or malformed."""


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
            resource_ids=(),
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
            restart_context_path=restart_context_path,
            local_addr=params.local_addr,
            source_path=source_path,
        )


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


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise RestartContextFileError(f"restart-context JSON contains duplicate field {key!r}")
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
    "TorchrunRendezvousPolicy",
    "TorchrunRuntimeConfig",
    "TorchrunRuntimeConfigError",
]
