"""Runtime configuration and node-local restart-context handoff for torchrun."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import stat
import tempfile
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

from lm_resiliency.integrations.torchrun._protocol import RestartContext

_BACKEND = "lm_resiliency"
_NODE_ID_ENV = "LM_RESILIENCY_NODE_ID"
_RESTART_CONTEXT_ENV = "LM_RESILIENCY_RESTART_CONTEXT"
_MAX_CONTEXT_BYTES = 64 * 1024
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
        """Return the canonical digest agents register for drift detection."""

        encoded = json.dumps(
            self.to_dict(),
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

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
    node_id: str
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
        _nonempty_string(self.node_id, "node_id")
        if not isinstance(self.restart_context_path, Path):
            raise TypeError("restart_context_path must be pathlib.Path")
        if not self.restart_context_path.is_absolute():
            raise TorchrunRuntimeConfigError("restart_context_path must be absolute")
        if self.local_addr is not None:
            _nonempty_string(self.local_addr, "local_addr")
        if not isinstance(self.source_path, Path) or not self.source_path.is_absolute():
            raise TorchrunRuntimeConfigError("source_path must be an absolute pathlib.Path")

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
            node_id=node_id,
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

    def write(self, context: RestartContext) -> None:
        if not isinstance(context, RestartContext):
            raise TypeError("context must be RestartContext")
        parent = self._prepare_parent()
        self._reject_existing_symlink()
        encoded = (context.to_json() + "\n").encode("utf-8")
        if len(encoded) > _MAX_CONTEXT_BYTES:
            raise RestartContextFileError("restart context is too large")
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=parent,
        )
        try:
            os.fchmod(descriptor, 0o600)
            with os.fdopen(descriptor, "wb") as stream:
                descriptor = -1
                stream.write(encoded)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.path)
            self._fsync_directory(parent)
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
            try:
                os.unlink(temporary)
            except OSError:
                pass

    def read(self) -> RestartContext:
        self._validate_parent()
        flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        try:
            descriptor = os.open(self.path, flags)
        except OSError as error:
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
        self._validate_parent()
        self._reject_existing_symlink()
        try:
            self.path.unlink()
        except FileNotFoundError:
            self._fsync_directory(self.path.parent)
            return
        except OSError as error:
            raise RestartContextFileError(
                f"failed to remove restart context at {self.path}"
            ) from error
        self._fsync_directory(self.path.parent)

    def _prepare_parent(self) -> Path:
        parent = self.path.parent
        try:
            parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        except OSError as error:
            raise RestartContextFileError(
                f"failed to create restart-context directory {parent}"
            ) from error
        self._validate_parent()
        return parent

    def _validate_parent(self) -> None:
        parent = self.path.parent
        try:
            metadata = parent.lstat()
        except OSError as error:
            raise RestartContextFileError(
                f"failed to inspect restart-context directory {parent}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RestartContextFileError("restart-context parent directory must not be a symlink")
        if not stat.S_ISDIR(metadata.st_mode):
            raise RestartContextFileError("restart-context parent is not a directory")
        if metadata.st_uid != os.geteuid():
            raise RestartContextFileError("restart-context directory is owned by another user")
        if stat.S_IMODE(metadata.st_mode) & 0o077:
            raise RestartContextFileError(
                "restart-context directory must not grant group or other permissions"
            )

    def _reject_existing_symlink(self) -> None:
        try:
            metadata = self.path.lstat()
        except FileNotFoundError:
            return
        except OSError as error:
            raise RestartContextFileError(
                f"failed to inspect restart context at {self.path}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode):
            raise RestartContextFileError("restart-context path must not be a symlink")

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
    def _fsync_directory(directory: Path) -> None:
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def _load_policy_values(path: Path) -> dict[str, object]:
    try:
        with path.open("rb") as stream:
            value = _toml.load(stream)
    except (OSError, _toml.TOMLDecodeError) as error:
        raise TorchrunRuntimeConfigError(f"failed to load rendezvous config {path}") from error
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
