# mypy: ignore-errors
"""Offline discovery and replay scheduling for MoE execution regimes.

Token count is not itself a hardware trigger. It selects a physical execution plan:
kernel code, launch geometry, tiling and tail paths, workspace, and resource pressure.
This module catalogs those physical fingerprints for the exact training configuration
and chooses a bounded set of shapes that SCOUT can rotate through online.

The discovery core is intentionally framework-independent. A framework adapter runs
the real post-dispatch expert stage and supplies :class:`ExecutionObservation`
objects, either directly or through :class:`TorchCudaExecutionProfiler`.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import tempfile
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from lm_resiliency.detection.replay_shapes import ReplayShapePlan

CATALOG_SCHEMA_VERSION = 5
_EQUIVALENCE_POLICIES = {"exact_launch", "plan_and_pressure"}
_CTA_SEMANTIC_EVIDENCE_SOURCES = {
    "backend_api",
    "generated_metadata",
    "instrumented",
    "source_audit",
}
_REQUIRED_CATALOG_ENVIRONMENT_FIELDS = (
    "backend",
    "container_digest",
    "cublas",
    "cuda_build",
    "cuda_graphs",
    "driver",
    "gpu_capability",
    "gpu_model",
    "model",
    "model_commit",
    "nccl",
    "overlap",
    "parallelism",
    "precision",
    "precision_recipe",
    "sm_count",
    "torch",
    "workspace_policy",
)
_PLACEHOLDER_LABELS = {
    "",
    "n/a",
    "none",
    "null",
    "placeholder",
    "tbd",
    "todo",
    "unknown",
    "unset",
}


class RegimeDiscoveryError(RuntimeError):
    """Base error for an invalid or incomplete regime-discovery run."""


class ConflictingObservationError(RegimeDiscoveryError):
    """The same homogeneous workload produced incompatible physical fingerprints."""


class CatalogEnvironmentMismatch(RegimeDiscoveryError):
    """A catalog was loaded under a different hardware/software/model environment."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _frozen_mapping(values: Mapping[str, Any] | None) -> tuple[tuple[str, str], ...]:
    if not values:
        return ()
    return tuple(sorted((str(key), _canonical_json(value)) for key, value in values.items()))


def _thaw_mapping(values: Sequence[tuple[str, str]]) -> dict[str, Any]:
    return {key: json.loads(value) for key, value in values}


@dataclass(frozen=True)
class MoEExecutionEnvironment:
    """Configuration identity under which a regime catalog is valid.

    Values should include the GPU generation and SM count, driver/CUDA/PyTorch and
    GEMM backend versions, precision/FP8 recipe, expert dimensions, alignment policy,
    parallelism, workspace policy, overlap mode, and CUDA Graph configuration.
    Arbitrary JSON-compatible values are accepted so framework adapters can record
    additional execution-defining settings without changing the schema.
    """

    attributes: tuple[tuple[str, str], ...]

    @classmethod
    def from_mapping(cls, attributes: Mapping[str, Any]) -> MoEExecutionEnvironment:
        if not attributes:
            raise ValueError("MoE execution environment cannot be empty")
        return cls(attributes=_frozen_mapping(attributes))

    @cached_property
    def identifier(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return _thaw_mapping(self.attributes)

    def difference(self, other: MoEExecutionEnvironment) -> dict[str, tuple[Any, Any]]:
        left = self.to_dict()
        right = other.to_dict()
        return {
            key: (left.get(key, "<missing>"), right.get(key, "<missing>"))
            for key in sorted(set(left) | set(right))
            if left.get(key, "<missing>") != right.get(key, "<missing>")
        }

    def validate_for_catalog(self) -> None:
        """Reject incomplete or placeholder execution identities."""
        attributes = self.to_dict()
        extra = attributes.get("extra")
        extra = extra if isinstance(extra, Mapping) else {}
        missing = [
            name
            for name in _REQUIRED_CATALOG_ENVIRONMENT_FIELDS
            if name not in attributes and name not in extra
        ]
        placeholders = [
            name
            for name in _REQUIRED_CATALOG_ENVIRONMENT_FIELDS
            if name not in missing
            and _contains_placeholder(
                attributes.get(name, extra.get(name)),
                allow_none=name == "overlap",
            )
        ]
        if extra and _contains_placeholder(extra):
            placeholders.append("extra")
        try:
            if (
                "sm_count" not in missing
                and int(attributes.get("sm_count", extra.get("sm_count"))) < 1
            ):
                placeholders.append("sm_count")
        except (TypeError, ValueError):
            if "sm_count" not in missing:
                placeholders.append("sm_count")
        placeholders = sorted(set(placeholders))
        if missing or placeholders:
            details = []
            if missing:
                details.append(f"missing fields: {', '.join(missing)}")
            if placeholders:
                details.append(f"placeholder fields: {', '.join(placeholders)}")
            raise RegimeDiscoveryError(
                "MoE execution environment is incomplete for catalog construction ("
                + "; ".join(details)
                + ")"
            )


def _environment_attribute(environment: MoEExecutionEnvironment, name: str) -> Any:
    attributes = environment.to_dict()
    if name in attributes:
        return attributes[name]
    extra = attributes.get("extra")
    if isinstance(extra, Mapping) and name in extra:
        return extra[name]
    raise KeyError(name)


def current_moe_environment(
    *,
    backend: str,
    backend_version: str,
    model: Mapping[str, Any],
    precision: str,
    parallelism: Mapping[str, Any],
    extra: Mapping[str, Any] | None = None,
    device: int | torch.device = 0,
) -> MoEExecutionEnvironment:
    """Capture the stable environment fields that bind an offline catalog."""
    attributes: dict[str, Any] = {
        "backend": {"name": backend, "version": backend_version},
        "model": dict(model),
        "parallelism": dict(parallelism),
        "precision": precision,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_git": torch.version.git_version,
        "cuda_build": str(torch.version.cuda),
        "cudnn": str(torch.backends.cudnn.version()),
    }
    try:
        attributes["nccl"] = ".".join(str(value) for value in torch.cuda.nccl.version())
    except Exception:  # noqa: BLE001 -- unavailable on CPU or non-NCCL builds
        pass
    if torch.cuda.is_available():
        index = device.index if isinstance(device, torch.device) else int(device)
        index = torch.cuda.current_device() if index is None else index
        properties = torch.cuda.get_device_properties(index)
        attributes.update(
            {
                "gpu_model": properties.name,
                "gpu_capability": [properties.major, properties.minor],
                "sm_count": properties.multi_processor_count,
            }
        )
        try:
            import pynvml

            pynvml.nvmlInit()
            driver = pynvml.nvmlSystemGetDriverVersion()
            attributes["driver"] = driver.decode() if isinstance(driver, bytes) else str(driver)
        except Exception:  # noqa: BLE001 -- NVML is an optional deployment dependency
            try:
                result = subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=driver_version",
                        "--format=csv,noheader",
                    ],
                    capture_output=True,
                    check=True,
                    text=True,
                    timeout=5,
                )
                versions = {line.strip() for line in result.stdout.splitlines() if line.strip()}
                if len(versions) == 1:
                    attributes["driver"] = versions.pop()
            except (OSError, subprocess.SubprocessError):
                pass
    else:
        attributes["gpu_model"] = "cpu-only-discovery"
    if extra:
        attributes["extra"] = dict(extra)
    return MoEExecutionEnvironment.from_mapping(attributes)


@dataclass(frozen=True)
class KernelLaunch:
    """Failure-relevant identity of one GPU kernel launch."""

    name: str
    grid: tuple[int, int, int] | None = None
    block: tuple[int, int, int] | None = None
    shared_memory_bytes: int | None = None
    registers_per_thread: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "grid": list(self.grid) if self.grid is not None else None,
            "block": list(self.block) if self.block is not None else None,
            "shared_memory_bytes": self.shared_memory_bytes,
            "registers_per_thread": self.registers_per_thread,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> KernelLaunch:
        return cls(
            name=str(value["name"]),
            grid=_triple(value.get("grid")),
            block=_triple(value.get("block")),
            shared_memory_bytes=_optional_int(value.get("shared_memory_bytes")),
            registers_per_thread=_optional_int(value.get("registers_per_thread")),
        )


@dataclass(frozen=True)
class CTASemantics:
    """Auditable semantic partition of one kernel's scalable work.

    Roles are disjoint work classes. For a direct kernel their counts partition the
    launch CTAs; for a persistent kernel they partition its logical work items.
    Compression requires an independently derived partition with different evidence.
    """

    mapping_class: str
    role_counts: tuple[tuple[str, int], ...]
    qualification_role_counts: tuple[tuple[str, int], ...]
    derivation_source: str
    derivation_digest: str
    qualification_source: str
    qualification_digest: str

    @classmethod
    def create(
        cls,
        *,
        mapping_class: str,
        role_counts: Mapping[str, int] | Sequence[tuple[str, int]],
        qualification_role_counts: Mapping[str, int] | Sequence[tuple[str, int]],
        derivation_source: str,
        derivation_digest: str,
        qualification_source: str,
        qualification_digest: str,
    ) -> CTASemantics:
        return cls(
            mapping_class=_label_or_unknown(mapping_class),
            role_counts=_frozen_role_counts(role_counts),
            qualification_role_counts=_frozen_role_counts(qualification_role_counts),
            derivation_source=_label_or_unknown(derivation_source),
            derivation_digest=_label_or_unknown(derivation_digest),
            qualification_source=_label_or_unknown(qualification_source),
            qualification_digest=_label_or_unknown(qualification_digest),
        )

    @property
    def roles(self) -> tuple[str, ...]:
        return tuple(role for role, _count in self.role_counts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mapping_class": self.mapping_class,
            "role_counts": dict(self.role_counts),
            "qualification_role_counts": dict(self.qualification_role_counts),
            "derivation_source": self.derivation_source,
            "derivation_digest": self.derivation_digest,
            "qualification_source": self.qualification_source,
            "qualification_digest": self.qualification_digest,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CTASemantics:
        return cls.create(
            mapping_class=str(value.get("mapping_class", "unknown")),
            role_counts=value.get("role_counts", {}),
            qualification_role_counts=value.get("qualification_role_counts", {}),
            derivation_source=str(value.get("derivation_source", "unknown")),
            derivation_digest=str(value.get("derivation_digest", "unknown")),
            qualification_source=str(value.get("qualification_source", "unknown")),
            qualification_digest=str(value.get("qualification_digest", "unknown")),
        )

    def is_compression_ready(
        self,
        kernel: KernelLaunch,
        persistent_work_items: int,
    ) -> bool:
        expected_work = persistent_work_items if persistent_work_items else _cta_count(kernel)
        return (
            _known_label(self.mapping_class)
            and bool(self.role_counts)
            and self.role_counts == self.qualification_role_counts
            and all(_known_label(role) and count > 0 for role, count in self.role_counts)
            and sum(count for _role, count in self.role_counts) == expected_work
            and self.derivation_source in _CTA_SEMANTIC_EVIDENCE_SOURCES
            and self.qualification_source in _CTA_SEMANTIC_EVIDENCE_SOURCES
            and self.derivation_source != self.qualification_source
            and _valid_sha256(self.derivation_digest)
            and _valid_sha256(self.qualification_digest)
            and self.derivation_digest != self.qualification_digest
        )


@dataclass(frozen=True)
class ExecutionFingerprint:
    """Physical execution properties used as the regime-equivalence key.

    Timing measurements are deliberately excluded: they are observations of a regime,
    not its identity. Framework adapters should put every categorical distinction that
    can select a different fault path into this fingerprint. Unknown distinctions must
    remain separate until GPU validation establishes that they are equivalent.
    """

    kernels: tuple[KernelLaunch, ...]
    algorithm_ids: tuple[str, ...] = ()
    tile_shapes: tuple[str, ...] = ()
    tail_path: str = "unknown"
    workspace_bytes: int | None = None
    pressure_class: str = "unknown"
    overlap_class: str = "unknown"
    persistent_work_items: tuple[int, ...] = ()
    cta_semantics: tuple[CTASemantics, ...] = ()
    extra: tuple[tuple[str, str], ...] = ()

    @classmethod
    def create(
        cls,
        *,
        kernels: Sequence[KernelLaunch],
        algorithm_ids: Sequence[str] = (),
        tile_shapes: Sequence[str] = (),
        tail_path: str = "unknown",
        workspace_bytes: int | None = None,
        pressure_class: str = "unknown",
        overlap_class: str = "unknown",
        persistent_work_items: Sequence[int] = (),
        cta_semantics: Sequence[CTASemantics] = (),
        extra: Mapping[str, Any] | None = None,
    ) -> ExecutionFingerprint:
        if not kernels:
            raise ValueError("an execution fingerprint requires at least one GPU kernel")
        return cls(
            kernels=tuple(kernels),
            algorithm_ids=tuple(_label_or_unknown(value) for value in algorithm_ids),
            tile_shapes=tuple(_label_or_unknown(value) for value in tile_shapes),
            tail_path=_label_or_unknown(tail_path),
            workspace_bytes=_optional_int(workspace_bytes),
            pressure_class=_label_or_unknown(pressure_class),
            overlap_class=_label_or_unknown(overlap_class),
            persistent_work_items=tuple(int(value) for value in persistent_work_items),
            cta_semantics=tuple(cta_semantics),
            extra=_frozen_mapping(extra),
        )

    @cached_property
    def identifier(self) -> str:
        return _digest(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "kernels": [kernel.to_dict() for kernel in self.kernels],
            "algorithm_ids": list(self.algorithm_ids),
            "tile_shapes": list(self.tile_shapes),
            "tail_path": self.tail_path,
            "workspace_bytes": self.workspace_bytes,
            "pressure_class": self.pressure_class,
            "overlap_class": self.overlap_class,
            "persistent_work_items": list(self.persistent_work_items),
            "cta_semantics": [semantics.to_dict() for semantics in self.cta_semantics],
            "extra": _thaw_mapping(self.extra),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionFingerprint:
        return cls.create(
            kernels=[KernelLaunch.from_dict(item) for item in value["kernels"]],
            algorithm_ids=value.get("algorithm_ids", ()),
            tile_shapes=value.get("tile_shapes", ()),
            tail_path=str(value.get("tail_path", "unknown")),
            workspace_bytes=value.get("workspace_bytes"),
            pressure_class=str(value.get("pressure_class", "unknown")),
            overlap_class=str(value.get("overlap_class", "unknown")),
            persistent_work_items=value.get("persistent_work_items", ()),
            cta_semantics=tuple(
                CTASemantics.from_dict(item) for item in value.get("cta_semantics", ())
            ),
            extra=value.get("extra", {}),
        )


@dataclass(frozen=True, order=True)
class ProfileLocation:
    """Where an expert-stage shape was observed in the training model."""

    layer_id: str
    expert_id: int
    ep_position: int
    execution_class: str = "default"

    def to_dict(self) -> dict[str, Any]:
        return {
            "layer_id": self.layer_id,
            "expert_id": self.expert_id,
            "ep_position": self.ep_position,
            "execution_class": self.execution_class,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfileLocation:
        return cls(
            layer_id=str(value["layer_id"]),
            expert_id=int(value["expert_id"]),
            ep_position=int(value["ep_position"]),
            execution_class=str(value.get("execution_class", "default")),
        )


@dataclass(frozen=True, order=True)
class ProfileRequest:
    """One shape/location that the offline profiler must execute."""

    n_exec: int
    location: ProfileLocation

    def __post_init__(self) -> None:
        if self.n_exec < 0:
            raise ValueError("n_exec must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return {"n_exec": self.n_exec, "location": self.location.to_dict()}

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ProfileRequest:
        return cls(
            n_exec=int(value["n_exec"]),
            location=ProfileLocation.from_dict(value["location"]),
        )


@dataclass(frozen=True)
class ExecutionObservation:
    """One measured physical execution fingerprint."""

    request: ProfileRequest
    fingerprint: ExecutionFingerprint

    @property
    def n_exec(self) -> int:
        return self.request.n_exec

    @property
    def location(self) -> ProfileLocation:
        return self.request.location

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_exec": self.n_exec,
            "location": self.location.to_dict(),
            "fingerprint": self.fingerprint.to_dict(),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionObservation:
        return cls(
            request=ProfileRequest.from_dict(value),
            fingerprint=ExecutionFingerprint.from_dict(value["fingerprint"]),
        )


@dataclass(frozen=True)
class ExecutionRegime:
    """Shapes proven equivalent by their physical execution fingerprint."""

    regime_id: str
    execution_class: str
    fingerprints: tuple[ExecutionFingerprint, ...]
    shape_fingerprint_ids: tuple[tuple[int, str], ...]
    n_exec_values: tuple[int, ...]
    representatives: tuple[int, ...]
    locations: tuple[ProfileLocation, ...]

    def __post_init__(self) -> None:
        if not self.fingerprints or not self.n_exec_values or not self.representatives:
            raise ValueError(
                "an execution regime requires fingerprints, shapes, and representatives"
            )
        if not set(self.representatives).issubset(self.n_exec_values):
            raise ValueError("regime representatives must belong to its n_exec values")
        known_ids = {fingerprint.identifier for fingerprint in self.fingerprints}
        shape_map = dict(self.shape_fingerprint_ids)
        if set(shape_map) != set(self.n_exec_values):
            raise ValueError("regime shape/fingerprint map must cover every n_exec exactly")
        if not set(shape_map.values()).issubset(known_ids):
            raise ValueError("regime shape/fingerprint map refers to an unknown fingerprint")
        uncovered = [
            n_exec for n_exec in self.n_exec_values if not self.representatives_cover_shape(n_exec)
        ]
        if uncovered:
            raise ValueError(
                f"regime representatives do not cover per-role work counts for shapes {uncovered}"
            )

    @property
    def fingerprint(self) -> ExecutionFingerprint:
        """Backward-friendly access to the first audited raw fingerprint."""
        return self.fingerprints[0]

    @cached_property
    def _shape_fingerprint_map(self) -> dict[int, str]:
        return dict(self.shape_fingerprint_ids)

    @cached_property
    def _fingerprints_by_id(self) -> dict[str, ExecutionFingerprint]:
        return {fingerprint.identifier: fingerprint for fingerprint in self.fingerprints}

    def fingerprint_for_shape(self, n_exec: int) -> ExecutionFingerprint:
        identifier = self._shape_fingerprint_map.get(n_exec)
        if identifier is None:
            raise KeyError(f"n_exec={n_exec} is not in regime {self.regime_id}")
        return self._fingerprints_by_id[identifier]

    def representatives_cover_shape(self, n_exec: int) -> bool:
        """Whether a representative dominates a shape's per-role scalable work."""
        target = self.fingerprint_for_shape(n_exec)
        return any(
            _fingerprint_covers(self.fingerprint_for_shape(representative), target)
            for representative in self.representatives
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "regime_id": self.regime_id,
            "execution_class": self.execution_class,
            "fingerprints": [fingerprint.to_dict() for fingerprint in self.fingerprints],
            "shape_fingerprint_ids": [
                [n_exec, fingerprint_id] for n_exec, fingerprint_id in self.shape_fingerprint_ids
            ],
            "n_exec_values": list(self.n_exec_values),
            "representatives": list(self.representatives),
            "locations": [location.to_dict() for location in self.locations],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ExecutionRegime:
        return cls(
            regime_id=str(value["regime_id"]),
            execution_class=str(value["execution_class"]),
            fingerprints=tuple(
                ExecutionFingerprint.from_dict(item) for item in value["fingerprints"]
            ),
            shape_fingerprint_ids=tuple(
                (int(item[0]), str(item[1])) for item in value["shape_fingerprint_ids"]
            ),
            n_exec_values=tuple(int(item) for item in value["n_exec_values"]),
            representatives=tuple(int(item) for item in value["representatives"]),
            locations=tuple(ProfileLocation.from_dict(item) for item in value["locations"]),
        )


@dataclass(frozen=True)
class ReplayRecipe:
    """One representative physical shape in a catalog rotation."""

    regime_id: str
    execution_class: str
    n_exec: int
    fingerprint_id: str


@dataclass(frozen=True)
class MoERegimeCatalog:
    """Versioned, environment-bound collection of MoE execution regimes."""

    environment: MoEExecutionEnvironment
    ep_position: int
    regimes: tuple[ExecutionRegime, ...]
    profiled_requests: tuple[ProfileRequest, ...]
    observation_counts: tuple[tuple[ProfileRequest, int], ...]
    equivalence_policy: str = "plan_and_pressure"
    schema_version: int = CATALOG_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != CATALOG_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported MoE catalog schema {self.schema_version}; "
                f"expected {CATALOG_SCHEMA_VERSION}"
            )
        if not self.regimes:
            raise ValueError("MoE regime catalog cannot be empty")
        if self.equivalence_policy not in _EQUIVALENCE_POLICIES:
            raise ValueError(f"unsupported regime equivalence policy {self.equivalence_policy!r}")
        ids = [regime.regime_id for regime in self.regimes]
        if len(ids) != len(set(ids)):
            raise ValueError("MoE regime IDs must be unique")
        if len(self.profiled_requests) != len(set(self.profiled_requests)):
            raise ValueError("profiled requests must be unique")
        counts = dict(self.observation_counts)
        if len(counts) != len(self.observation_counts):
            raise ValueError("observation counts must contain each request exactly once")
        if set(counts) != set(self.profiled_requests):
            raise ValueError("observation counts must cover every profiled request exactly")
        if any(count < 1 for count in counts.values()):
            raise ValueError("observation counts must be positive")
        self.environment.validate_for_catalog()
        for regime in self.regimes:
            keys = {
                _regime_equivalence_key(
                    regime.fingerprint_for_shape(n_exec),
                    policy=self.equivalence_policy,
                    n_exec=n_exec,
                )
                for n_exec in regime.n_exec_values
            }
            if len(keys) != 1:
                raise ValueError(f"regime {regime.regime_id} contains incompatible execution plans")
            expected_id = f"{regime.execution_class}-{next(iter(keys))[:12]}"
            if regime.regime_id != expected_id:
                raise ValueError(
                    f"regime ID {regime.regime_id!r} does not match its execution fingerprint"
                )

    @cached_property
    def identifier(self) -> str:
        return _digest(self.to_dict())

    @property
    def replay_recipes(self) -> tuple[ReplayRecipe, ...]:
        return tuple(
            ReplayRecipe(
                regime_id=regime.regime_id,
                execution_class=regime.execution_class,
                n_exec=n_exec,
                fingerprint_id=regime.fingerprint_for_shape(n_exec).identifier,
            )
            for regime in self.regimes
            for n_exec in regime.representatives
        )

    def validate_representative_sm_coverage(
        self,
        observed_sm_ids: Mapping[tuple[str, int, int], Iterable[int]],
    ) -> None:
        """Require instrumented evidence that every representative kernel visited every SM."""
        sm_count = int(_environment_attribute(self.environment, "sm_count"))
        expected = set(range(sm_count))
        gaps = []
        for regime in self.regimes:
            for n_exec in regime.representatives:
                fingerprint = regime.fingerprint_for_shape(n_exec)
                for kernel_index in range(len(fingerprint.kernels)):
                    key = (regime.regime_id, n_exec, kernel_index)
                    observed = {int(sm_id) for sm_id in observed_sm_ids.get(key, ())}
                    invalid = observed - expected
                    if invalid:
                        raise ValueError(f"invalid SM IDs for {key}: {sorted(invalid)}")
                    missing = expected - observed
                    if missing:
                        gaps.append((key, missing))
        if gaps:
            preview = ", ".join(f"{key}: missing {len(missing)} SMs" for key, missing in gaps[:8])
            suffix = " ..." if len(gaps) > 8 else ""
            raise RegimeDiscoveryError(
                f"representative SM coverage is incomplete for {len(gaps)} kernels: "
                f"{preview}{suffix}"
            )

    @property
    def cycle_size(self) -> int:
        return len(self.replay_recipes)

    def to_replay_shape_plan(self) -> ReplayShapePlan:
        """Convert catalog representatives to the common dense/MoE replay shape list."""
        from lm_resiliency.detection.replay_shapes import ReplayShapePlan

        return ReplayShapePlan.from_moe_catalog(self)

    def validate_environment(self, current: MoEExecutionEnvironment) -> None:
        difference = self.environment.difference(current)
        if difference:
            details = ", ".join(
                f"{key}: catalog={left!r}, current={right!r}"
                for key, (left, right) in difference.items()
            )
            raise CatalogEnvironmentMismatch(f"MoE regime catalog is stale: {details}")

    def regimes_for_shape(
        self, n_exec: int, *, execution_class: str = "default"
    ) -> tuple[ExecutionRegime, ...]:
        return tuple(
            regime
            for regime in self.regimes
            if regime.execution_class == execution_class and n_exec in regime.n_exec_values
        )

    def missing_shapes(
        self, n_exec_values: Iterable[int], *, execution_class: str = "default"
    ) -> tuple[int, ...]:
        return tuple(
            sorted(
                n_exec
                for n_exec in set(int(value) for value in n_exec_values)
                if not self.regimes_for_shape(n_exec, execution_class=execution_class)
            )
        )

    def missing_profile_requests(
        self, expected: Iterable[ProfileRequest]
    ) -> tuple[ProfileRequest, ...]:
        """Return layer/expert/shape requests absent from offline profiling."""
        observed = set(self.profiled_requests)
        return tuple(sorted(set(expected) - observed))

    def observation_count(self, request: ProfileRequest) -> int:
        """Return the number of raw observations retained for one request."""
        return dict(self.observation_counts).get(request, 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "environment": self.environment.to_dict(),
            "environment_id": self.environment.identifier,
            "ep_position": self.ep_position,
            "equivalence_policy": self.equivalence_policy,
            "profiled_requests": [request.to_dict() for request in self.profiled_requests],
            "observation_counts": [
                {"request": request.to_dict(), "count": count}
                for request, count in self.observation_counts
            ],
            "regimes": [regime.to_dict() for regime in self.regimes],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> MoERegimeCatalog:
        catalog = cls(
            schema_version=int(value["schema_version"]),
            environment=MoEExecutionEnvironment.from_mapping(value["environment"]),
            ep_position=int(value["ep_position"]),
            equivalence_policy=str(value.get("equivalence_policy", "exact_launch")),
            profiled_requests=tuple(
                ProfileRequest.from_dict(item) for item in value["profiled_requests"]
            ),
            observation_counts=tuple(
                (ProfileRequest.from_dict(item["request"]), int(item["count"]))
                for item in value["observation_counts"]
            ),
            regimes=tuple(ExecutionRegime.from_dict(item) for item in value["regimes"]),
        )
        expected = value.get("environment_id")
        if expected is not None and expected != catalog.environment.identifier:
            raise ValueError("MoE catalog environment checksum does not match its contents")
        return catalog

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    @classmethod
    def load(cls, path: str | Path) -> MoERegimeCatalog:
        return cls.from_dict(json.loads(Path(path).read_text()))


def discover_execution_regimes(
    observations: Iterable[ExecutionObservation],
    *,
    environment: MoEExecutionEnvironment,
    representatives_per_regime: int = 1,
    equivalence_policy: str = "plan_and_pressure",
    expected_requests: Iterable[ProfileRequest] | None = None,
    minimum_observations_per_request: int = 3,
    max_replay_recipes: int | None = None,
) -> MoERegimeCatalog:
    """Partition measured shapes by physical fingerprint.

    Every observation in one catalog must target the same EP position. Within an
    execution class, the same ``n_exec`` must produce one fingerprint across every
    sampled layer and expert. A conflict means the workload is heterogeneous or the
    profiler is unstable; callers must split it into explicit execution classes or
    fix the measurement rather than allowing unsafe catalog compression.
    """
    values = tuple(observations)
    if not values:
        raise RegimeDiscoveryError("cannot discover MoE regimes without observations")
    if representatives_per_regime < 1:
        raise ValueError("representatives_per_regime must be positive")
    if minimum_observations_per_request < 1:
        raise ValueError("minimum_observations_per_request must be positive")
    if max_replay_recipes is not None and max_replay_recipes < 1:
        raise ValueError("max_replay_recipes must be positive when provided")
    if equivalence_policy not in _EQUIVALENCE_POLICIES:
        raise ValueError(f"unsupported regime equivalence policy {equivalence_policy!r}")
    environment.validate_for_catalog()
    ep_positions = {observation.location.ep_position for observation in values}
    if len(ep_positions) != 1:
        raise RegimeDiscoveryError(
            f"build a separate MoE regime catalog for each EP position; got {sorted(ep_positions)}"
        )
    ep_position = next(iter(ep_positions))
    observation_counts_by_request = Counter(observation.request for observation in values)
    profiled_requests = tuple(sorted(observation_counts_by_request))
    if expected_requests is not None:
        missing = tuple(sorted(set(expected_requests) - set(profiled_requests)))
        if missing:
            preview = ", ".join(
                f"layer={request.location.layer_id}/expert={request.location.expert_id}/"
                f"n_exec={request.n_exec}"
                for request in missing[:8]
            )
            suffix = " ..." if len(missing) > 8 else ""
            raise RegimeDiscoveryError(
                f"offline MoE profiling missed {len(missing)} required requests: {preview}{suffix}"
            )
    under_sampled = tuple(
        sorted(
            (request, count)
            for request, count in observation_counts_by_request.items()
            if count < minimum_observations_per_request
        )
    )
    if under_sampled:
        preview = ", ".join(
            f"layer={request.location.layer_id}/expert={request.location.expert_id}/"
            f"n_exec={request.n_exec}: {count}"
            for request, count in under_sampled[:8]
        )
        suffix = " ..." if len(under_sampled) > 8 else ""
        raise RegimeDiscoveryError(
            f"offline MoE profiling has {len(under_sampled)} requests with fewer than "
            f"{minimum_observations_per_request} observations: {preview}{suffix}"
        )

    by_shape: dict[tuple[str, int], set[str]] = defaultdict(set)
    for observation in values:
        by_shape[(observation.location.execution_class, observation.n_exec)].add(
            observation.fingerprint.identifier
        )
    conflicts = {key: identifiers for key, identifiers in by_shape.items() if len(identifiers) > 1}
    if conflicts:
        details = ", ".join(
            f"{execution_class}/n_exec={n_exec}: {len(identifiers)} fingerprints"
            for (execution_class, n_exec), identifiers in sorted(conflicts.items())
        )
        raise ConflictingObservationError(
            "homogeneous MoE workloads produced conflicting execution fingerprints; "
            "use separate execution_class values for heterogeneous layers/experts or "
            f"stabilize the profiler ({details})"
        )

    grouped: dict[tuple[str, str], list[ExecutionObservation]] = defaultdict(list)
    for observation in values:
        regime_key = _regime_equivalence_key(
            observation.fingerprint,
            policy=equivalence_policy,
            n_exec=observation.n_exec,
        )
        grouped[(observation.location.execution_class, regime_key)].append(observation)

    regimes: list[ExecutionRegime] = []
    for (execution_class, regime_key), members in sorted(grouped.items()):
        n_exec_values = tuple(sorted({member.n_exec for member in members}))
        locations = tuple(sorted({member.location for member in members}))
        fingerprints_by_id = {
            member.fingerprint.identifier: member.fingerprint for member in members
        }
        fingerprints_by_shape = {member.n_exec: member.fingerprint for member in members}
        representatives = _select_coverage_representatives(
            n_exec_values,
            fingerprints_by_shape,
            representatives_per_regime,
        )
        fingerprints = tuple(
            fingerprints_by_id[identifier] for identifier in sorted(fingerprints_by_id)
        )
        shape_fingerprint_ids = tuple(
            sorted({(member.n_exec, member.fingerprint.identifier) for member in members})
        )
        regimes.append(
            ExecutionRegime(
                regime_id=f"{execution_class}-{regime_key[:12]}",
                execution_class=execution_class,
                fingerprints=fingerprints,
                shape_fingerprint_ids=shape_fingerprint_ids,
                n_exec_values=n_exec_values,
                representatives=representatives,
                locations=locations,
            )
        )

    catalog = MoERegimeCatalog(
        environment=environment,
        ep_position=ep_position,
        profiled_requests=profiled_requests,
        observation_counts=tuple(sorted(observation_counts_by_request.items())),
        equivalence_policy=equivalence_policy,
        regimes=tuple(regimes),
    )
    if (
        equivalence_policy == "plan_and_pressure"
        and max_replay_recipes is not None
        and catalog.cycle_size > max_replay_recipes
    ):
        raise RegimeDiscoveryError(
            f"compressed MoE catalog requires {catalog.cycle_size} replay recipes, "
            f"exceeding the configured maximum of {max_replay_recipes}"
        )
    return catalog


def profile_requests(
    requests: Iterable[ProfileRequest],
    profiler: Callable[[ProfileRequest], ExecutionFingerprint],
    *,
    repetitions: int = 3,
) -> tuple[ExecutionObservation, ...]:
    """Run repeated profiles for every requested shape/location."""
    if repetitions < 1:
        raise ValueError("repetitions must be positive")
    return tuple(
        ExecutionObservation(request=request, fingerprint=profiler(request))
        for request in requests
        for _ in range(repetitions)
    )


def build_profile_requests(
    *,
    layer_ids: Iterable[str | int],
    expert_ids: Iterable[int],
    ep_position: int,
    n_exec_values: Iterable[int],
    execution_class: str = "default",
) -> tuple[ProfileRequest, ...]:
    """Build the complete layer x expert x physical-shape profiling manifest."""
    layers = tuple(sorted({str(layer_id) for layer_id in layer_ids}))
    experts = tuple(sorted({int(expert_id) for expert_id in expert_ids}))
    shapes = tuple(sorted({int(n_exec) for n_exec in n_exec_values}))
    if not layers or not experts or not shapes:
        raise ValueError("layer_ids, expert_ids, and n_exec_values must all be non-empty")
    return tuple(
        ProfileRequest(
            n_exec=n_exec,
            location=ProfileLocation(
                layer_id=layer_id,
                expert_id=expert_id,
                ep_position=ep_position,
                execution_class=execution_class,
            ),
        )
        for layer_id in layers
        for expert_id in experts
        for n_exec in shapes
    )


@dataclass(frozen=True)
class ExecutionHints:
    """Backend knowledge not available in a generic Kineto trace."""

    algorithm_ids: tuple[str, ...] = ()
    tile_shapes: tuple[str, ...] = ()
    tail_path: str = "unknown"
    workspace_bytes: int | None = None
    pressure_class: str = "unknown"
    overlap_class: str = "unknown"
    persistent_work_items: tuple[int, ...] = ()
    cta_semantics: tuple[CTASemantics, ...] = ()
    extra: Mapping[str, Any] = field(default_factory=dict)


class TorchCudaExecutionProfiler:
    """Profile the real expert stage with PyTorch/Kineto CUDA traces.

    ``workload`` must execute exactly one post-dispatch expert-stage invocation for
    the supplied request. ``hints`` supplies backend facts such as the cuBLASLt or
    grouped-GEMM algorithm ID, tile shape, explicit tail class, and workspace size.
    Kineto supplies the kernel sequence and launch geometry. Both are required for a
    strong trigger-equivalence catalog; a kernel name alone is not treated as enough.
    """

    def __init__(
        self,
        workload: Callable[[ProfileRequest], None],
        *,
        hints: Callable[[ProfileRequest, tuple[KernelLaunch, ...]], ExecutionHints],
        warmup: int = 3,
        device: int | torch.device = 0,
    ) -> None:
        if warmup < 0:
            raise ValueError("warmup must be non-negative")
        self._workload = workload
        self._hints = hints
        self._warmup = warmup
        self._device = device

    def __call__(self, request: ProfileRequest) -> ExecutionFingerprint:
        if not torch.cuda.is_available():
            raise RuntimeError("TorchCudaExecutionProfiler requires a CUDA GPU")
        for _ in range(self._warmup):
            self._workload(request)
        torch.cuda.synchronize(self._device)

        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True,
        ) as profiler:
            self._workload(request)
            torch.cuda.synchronize(self._device)

        with tempfile.TemporaryDirectory(prefix="scout-moe-profile-") as directory:
            trace_path = Path(directory) / "kineto.json"
            profiler.export_chrome_trace(str(trace_path))
            launches = kernel_launches_from_kineto_trace(json.loads(trace_path.read_text()))
        if not launches:
            raise RegimeDiscoveryError(
                f"Kineto recorded no CUDA kernels for n_exec={request.n_exec}"
            )

        hints = self._hints(request, launches)
        return ExecutionFingerprint.create(
            kernels=launches,
            algorithm_ids=hints.algorithm_ids,
            tile_shapes=hints.tile_shapes,
            tail_path=hints.tail_path,
            workspace_bytes=hints.workspace_bytes,
            pressure_class=hints.pressure_class,
            overlap_class=hints.overlap_class,
            persistent_work_items=hints.persistent_work_items,
            cta_semantics=hints.cta_semantics,
            extra=hints.extra,
        )


def kernel_launches_from_kineto_trace(trace: Mapping[str, Any]) -> tuple[KernelLaunch, ...]:
    """Extract ordered CUDA kernel launch identities from a Chrome/Kineto trace."""
    launches: list[KernelLaunch] = []
    for event in trace.get("traceEvents", ()):
        category = str(event.get("cat", "")).lower()
        if "kernel" not in category:
            continue
        args = event.get("args") or {}
        launches.append(
            KernelLaunch(
                name=str(event.get("name", "<unnamed>")),
                grid=_trace_triple(args, "grid"),
                block=_trace_triple(args, "block"),
                shared_memory_bytes=_trace_int(
                    args,
                    "shared memory",
                    "shared_memory",
                    "shared memory (bytes)",
                ),
                registers_per_thread=_trace_int(
                    args,
                    "registers per thread",
                    "registers_per_thread",
                ),
            )
        )
    return tuple(launches)


@dataclass(frozen=True)
class ScheduledReplay:
    """Recipe selected at one replay boundary."""

    recipe: ReplayRecipe
    position: int
    cycle: int
    completes_cycle: bool


class MoEReplayScheduler:
    """Deterministically rotate through every recipe in a regime catalog."""

    def __init__(
        self,
        catalog: MoERegimeCatalog,
        *,
        replay_interval: int,
        first_replay_step: int | None = None,
    ) -> None:
        if replay_interval < 1:
            raise ValueError("replay_interval must be positive")
        self.catalog = catalog
        self.replay_interval = replay_interval
        self.first_replay_step = replay_interval if first_replay_step is None else first_replay_step
        if self.first_replay_step < 1:
            raise ValueError("first_replay_step must be positive")
        self._position = 0
        self._cycle = 0

    @property
    def detection_bound_steps(self) -> int:
        return self.catalog.cycle_size * self.replay_interval

    @property
    def completed_cycles(self) -> int:
        return self._cycle

    def recipe_for_step(self, step: int) -> ScheduledReplay | None:
        if step < self.first_replay_step:
            return None
        if (step - self.first_replay_step) % self.replay_interval:
            return None
        return self.next_recipe()

    def next_recipe(self) -> ScheduledReplay:
        recipes = self.catalog.replay_recipes
        position = self._position
        completes_cycle = position == len(recipes) - 1
        result = ScheduledReplay(
            recipe=recipes[position],
            position=position,
            cycle=self._cycle,
            completes_cycle=completes_cycle,
        )
        self._position = (position + 1) % len(recipes)
        if completes_cycle:
            self._cycle += 1
        return result

    def state_dict(self) -> dict[str, Any]:
        return {
            "catalog_id": self.catalog.identifier,
            "position": self._position,
            "cycle": self._cycle,
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("catalog_id") != self.catalog.identifier:
            raise CatalogEnvironmentMismatch(
                "cannot restore MoE replay position into a different catalog"
            )
        position = int(state["position"])
        cycle = int(state["cycle"])
        if not 0 <= position < self.catalog.cycle_size or cycle < 0:
            raise ValueError("invalid MoE replay scheduler state")
        self._position = position
        self._cycle = cycle


def save_observations(observations: Iterable[ExecutionObservation], path: str | Path) -> None:
    """Persist raw profiler observations as JSON Lines for offline inspection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(_canonical_json(observation.to_dict()) + "\n" for observation in observations)
    )


def load_observations(path: str | Path) -> tuple[ExecutionObservation, ...]:
    """Load raw JSON Lines observations produced by :func:`save_observations`."""
    observations = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        if not line.strip():
            continue
        try:
            observations.append(ExecutionObservation.from_dict(json.loads(line)))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"invalid observation at line {line_number}: {exc}") from exc
    return tuple(observations)


def save_profile_requests(requests: Iterable[ProfileRequest], path: str | Path) -> None:
    """Persist a complete profiling manifest as a JSON array."""
    values = tuple(requests)
    if not values:
        raise ValueError("profiling manifest cannot be empty")
    if len(values) != len(set(values)):
        raise ValueError("profiling manifest contains duplicate requests")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps([request.to_dict() for request in values], indent=2, sort_keys=True) + "\n"
    )


def load_profile_requests(path: str | Path) -> tuple[ProfileRequest, ...]:
    """Load a profiling manifest produced by :func:`save_profile_requests`."""
    try:
        payload = json.loads(Path(path).read_text())
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid profiling manifest JSON: {exc}") from exc
    if not isinstance(payload, list) or not payload:
        raise ValueError("profiling manifest must be a non-empty JSON array")
    try:
        requests = tuple(ProfileRequest.from_dict(item) for item in payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"invalid profiling manifest request: {exc}") from exc
    if len(requests) != len(set(requests)):
        raise ValueError("profiling manifest contains duplicate requests")
    return requests


def _regime_equivalence_key(
    fingerprint: ExecutionFingerprint,
    *,
    policy: str,
    n_exec: int,
) -> str:
    """Project a raw launch trace onto the declared physical-equivalence policy.

    ``plan_and_pressure`` keeps kernel plans, CTA mapping semantics, persistent versus
    direct scheduling, and operating-pressure classes exact. Grid extents, semantic
    per-role work counts, and scalable persistent-work counts are retained for audit
    but removed from the grouping key. Representative selection separately retains a
    minimum set of shapes whose independently qualified role counts dominate every
    member. Unknown or unqualified semantics conservatively separate every shape.
    """
    if policy == "exact_launch":
        return _exact_regime_key(fingerprint, n_exec)
    compression_ready = (
        _known_labels(fingerprint.algorithm_ids)
        and _known_labels(fingerprint.tile_shapes)
        and _known_label(fingerprint.tail_path)
        and fingerprint.workspace_bytes is not None
        and fingerprint.workspace_bytes >= 0
        and _known_label(fingerprint.pressure_class)
        and _known_label(fingerprint.overlap_class, allow_none=True)
        and len(fingerprint.persistent_work_items) == len(fingerprint.kernels)
        and all(value >= 0 for value in fingerprint.persistent_work_items)
        and len(fingerprint.cta_semantics) == len(fingerprint.kernels)
        and all(
            semantics.is_compression_ready(kernel, persistent_work)
            for semantics, kernel, persistent_work in zip(
                fingerprint.cta_semantics,
                fingerprint.kernels,
                fingerprint.persistent_work_items,
                strict=True,
            )
        )
        and _known_extra(fingerprint.extra)
        and all(
            _known_label(kernel.name)
            and _valid_launch_shape(kernel.grid)
            and _valid_launch_shape(kernel.block)
            for kernel in fingerprint.kernels
        )
    )
    if not compression_ready:
        return _exact_regime_key(fingerprint, n_exec)

    projected = fingerprint.to_dict()
    projected["persistent_work_modes"] = [
        "persistent" if work_items else "direct" for work_items in fingerprint.persistent_work_items
    ]
    for kernel in projected["kernels"]:
        kernel["grid"] = None
    for semantics in projected["cta_semantics"]:
        semantics["role_counts"] = None
        semantics["qualification_role_counts"] = None
        semantics["derivation_source"] = None
        semantics["derivation_digest"] = None
        semantics["qualification_source"] = None
        semantics["qualification_digest"] = None
    projected["persistent_work_items"] = None
    return _digest({"equivalence": "plan_and_pressure", "fingerprint": projected})


def _exact_regime_key(fingerprint: ExecutionFingerprint, n_exec: int) -> str:
    return _digest(
        {
            "equivalence": "exact_launch",
            "fingerprint_id": fingerprint.identifier,
            "n_exec": n_exec,
        }
    )


def _select_coverage_representatives(
    values: Sequence[int],
    fingerprints: Mapping[int, ExecutionFingerprint],
    requested_count: int,
) -> tuple[int, ...]:
    if not values:
        raise ValueError("cannot select a representative from an empty regime")
    # Keep one deterministic member of each Pareto-maximal coverage class.
    maximal: list[int] = []
    for candidate in values:
        relations = [
            (
                existing,
                _fingerprint_covers(fingerprints[existing], fingerprints[candidate]),
                _fingerprint_covers(fingerprints[candidate], fingerprints[existing]),
            )
            for existing in maximal
        ]
        if any(
            existing_covers and not candidate_covers
            for _, existing_covers, candidate_covers in relations
        ):
            continue

        equivalent = [
            existing
            for existing, existing_covers, candidate_covers in relations
            if existing_covers and candidate_covers
        ]
        winner = max(
            (candidate, *equivalent),
            key=lambda value: (
                _fingerprint_coverage_size(fingerprints[value]),
                value,
            ),
        )
        maximal = [
            existing
            for existing, existing_covers, candidate_covers in relations
            if not candidate_covers and not (existing_covers and candidate_covers)
        ]
        maximal.append(winner)

    maximal_set = set(maximal)
    selected: set[int] = set()
    while maximal_set:
        representative = max(
            maximal_set,
            key=lambda candidate: (
                _fingerprint_coverage_size(fingerprints[candidate]),
                candidate,
            ),
        )
        selected.add(representative)
        maximal_set = {
            candidate
            for candidate in maximal_set
            if not (
                _fingerprint_covers(
                    fingerprints[representative],
                    fingerprints[candidate],
                )
                and _fingerprint_covers(
                    fingerprints[candidate],
                    fingerprints[representative],
                )
            )
        }

    if any(
        not any(
            _fingerprint_covers(fingerprints[representative], fingerprints[target])
            for representative in selected
        )
        for target in values
    ):
        raise AssertionError("maximal per-role representatives did not cover the regime")

    target_count = min(len(values), max(requested_count, len(selected)))
    indexed_values = {value: index for index, value in enumerate(values)}
    while len(selected) < target_count:
        candidate = max(
            (value for value in values if value not in selected),
            key=lambda value: (
                min(abs(indexed_values[value] - indexed_values[chosen]) for chosen in selected),
                value,
            ),
        )
        selected.add(candidate)
    return tuple(sorted(selected))


def _fingerprint_covers(
    candidate: ExecutionFingerprint,
    target: ExecutionFingerprint,
) -> bool:
    if candidate.identifier == target.identifier:
        return True
    if len(candidate.kernels) != len(target.kernels):
        return False
    if len(candidate.persistent_work_items) != len(candidate.kernels):
        return False
    if len(target.persistent_work_items) != len(target.kernels):
        return False
    if len(candidate.cta_semantics) != len(candidate.kernels):
        return False
    if len(target.cta_semantics) != len(target.kernels):
        return False
    return all(
        candidate_kernel.grid is not None
        and target_kernel.grid is not None
        and candidate_work >= target_work
        and candidate_semantics.mapping_class == target_semantics.mapping_class
        and _role_counts_cover(
            candidate_semantics.role_counts,
            target_semantics.role_counts,
        )
        for candidate_kernel, target_kernel, candidate_work, target_work, candidate_semantics, target_semantics in zip(
            candidate.kernels,
            target.kernels,
            candidate.persistent_work_items,
            target.persistent_work_items,
            candidate.cta_semantics,
            target.cta_semantics,
            strict=True,
        )
    )


def _fingerprint_coverage_size(fingerprint: ExecutionFingerprint) -> int:
    if len(fingerprint.persistent_work_items) != len(fingerprint.kernels):
        return 0
    if len(fingerprint.cta_semantics) != len(fingerprint.kernels):
        return 0
    return sum(
        sum(count for _role, count in semantics.role_counts)
        for _kernel, _persistent_work, semantics in zip(
            fingerprint.kernels,
            fingerprint.persistent_work_items,
            fingerprint.cta_semantics,
            strict=True,
        )
    )


def _cta_count(kernel: KernelLaunch) -> int:
    if kernel.grid is None:
        return 0
    return kernel.grid[0] * kernel.grid[1] * kernel.grid[2]


def _frozen_role_counts(
    values: Mapping[str, int] | Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    items = values.items() if isinstance(values, Mapping) else values
    counts: dict[str, int] = {}
    for role, count in items:
        label = _label_or_unknown(role)
        if label in counts:
            raise ValueError(f"duplicate CTA semantic role {label!r}")
        counts[label] = int(count)
    return tuple(sorted(counts.items()))


def _role_counts_cover(
    candidate: Sequence[tuple[str, int]],
    target: Sequence[tuple[str, int]],
) -> bool:
    candidate_counts = dict(candidate)
    return all(candidate_counts.get(role, 0) >= count for role, count in target)


def _valid_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _label_or_unknown(value: Any) -> str:
    return "unknown" if value is None else str(value)


def _contains_placeholder(value: Any, *, allow_none: bool = False) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        normalized = value.strip().lower()
        if allow_none and normalized == "none":
            return False
        return normalized in _PLACEHOLDER_LABELS or (
            normalized.startswith("<") and normalized.endswith(">")
        )
    if isinstance(value, Mapping):
        return not value or any(
            _contains_placeholder(item, allow_none=key == "overlap") for key, item in value.items()
        )
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray)):
        return not value or any(
            _contains_placeholder(item, allow_none=allow_none) for item in value
        )
    return False


def _known_label(value: Any, *, allow_none: bool = False) -> bool:
    return isinstance(value, str) and not _contains_placeholder(value, allow_none=allow_none)


def _known_labels(values: Sequence[str]) -> bool:
    return bool(values) and all(_known_label(value) for value in values)


def _known_extra(values: Sequence[tuple[str, str]]) -> bool:
    return all(
        not _contains_placeholder(json.loads(value), allow_none=key == "overlap")
        for key, value in values
    )


def _valid_launch_shape(value: tuple[int, int, int] | None) -> bool:
    return value is not None and all(dimension > 0 for dimension in value)


def _triple(value: Any) -> tuple[int, int, int] | None:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip().strip("[]()")
        parts = [part.strip() for part in cleaned.replace("x", ",").split(",") if part.strip()]
    elif isinstance(value, Sequence):
        parts = list(value)
    else:
        return None
    if not parts:
        return None
    integers = [int(part) for part in parts[:3]]
    integers.extend([1] * (3 - len(integers)))
    return integers[0], integers[1], integers[2]


def _trace_triple(args: Mapping[str, Any], name: str) -> tuple[int, int, int] | None:
    direct = args.get(name)
    if direct is not None:
        return _triple(direct)
    values = [args.get(f"{name} {axis}", args.get(f"{name}_{axis}")) for axis in ("x", "y", "z")]
    if not any(value is not None for value in values):
        return None
    return _triple([1 if value is None else value for value in values])


def _trace_int(args: Mapping[str, Any], *names: str) -> int | None:
    for name in names:
        if args.get(name) is not None:
            return int(args[name])
    return None
