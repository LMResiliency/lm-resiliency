"""Framework-aware runtime for scheduled fault campaigns."""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.fault_injection.config import (
    FaultCampaign,
    FaultLocation,
    FaultMagnitude,
    FaultPersistence,
    FaultScope,
    FaultSpec,
    FaultType,
)
from lm_resiliency.fault_injection.frameworks import resolve_training_models
from lm_resiliency.fault_injection.reports import (
    CampaignReport,
    FaultEvaluation,
    FaultInjectionRecord,
    InjectionStatus,
    LocalizationResult,
)

_SCALE_UP = {
    FaultMagnitude.CATASTROPHIC: 1e6,
    FaultMagnitude.LARGE: 100.0,
    FaultMagnitude.MEDIUM: 10.0,
    FaultMagnitude.SUBTLE: 2.0,
    FaultMagnitude.NEAR_INVISIBLE: 1.0001,
}
_SCALE_DOWN = {
    FaultMagnitude.CATASTROPHIC: 1e-6,
    FaultMagnitude.LARGE: 0.01,
    FaultMagnitude.MEDIUM: 0.1,
    FaultMagnitude.SUBTLE: 0.5,
    FaultMagnitude.NEAR_INVISIBLE: 0.9999,
}
_NOISE_STD = {
    FaultMagnitude.CATASTROPHIC: 1e6,
    FaultMagnitude.LARGE: 1e2,
    FaultMagnitude.MEDIUM: 1.0,
    FaultMagnitude.SUBTLE: 1e-3,
    FaultMagnitude.NEAR_INVISIBLE: 1e-7,
}
_INTEGER_VIEW = {
    torch.float16: (torch.int16, 16),
    torch.bfloat16: (torch.int16, 16),
    torch.float32: (torch.int32, 32),
    torch.float64: (torch.int64, 64),
}


@dataclass(slots=True)
class _Restoration:
    tensor: torch.Tensor
    indices: torch.Tensor
    values: torch.Tensor
    restored: bool = False

    def restore(self) -> None:
        if self.restored:
            return
        with torch.no_grad():
            self.tensor.view(-1).index_copy_(0, self.indices, self.values)
        self.restored = True


@dataclass(slots=True)
class _PendingInjection:
    record: FaultInjectionRecord
    handles: list[Any] = field(default_factory=list)
    restoration: _Restoration | None = None
    completed: bool = False

    def remove_hooks(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class FaultInjectionSession:
    """Rank-local runtime bound to initialized framework training objects."""

    def __init__(
        self,
        target: Any,
        campaign: FaultCampaign,
        *,
        framework: str = "auto",
        rank: int | None = None,
    ) -> None:
        self.campaign = campaign
        self._models = resolve_training_models(target, framework)
        self.framework = self._models.framework
        self.rank = _distributed_rank() if rank is None else int(rank)
        if self.rank < 0:
            raise ValueError("fault injection rank must be non-negative")
        self._records: list[FaultInjectionRecord] = []
        self._triggered: set[tuple[str, int]] = set()
        self._pending: list[_PendingInjection] = []
        self._persistent: list[_Restoration] = []
        self._closed = False

    @property
    def records(self) -> tuple[FaultInjectionRecord, ...]:
        """Ground-truth records created on this rank."""
        return tuple(self._records)

    def trigger(self, step: int) -> tuple[FaultInjectionRecord, ...]:
        """Activate faults scheduled for one training step on this rank."""
        if self._closed:
            raise RuntimeError("fault injection session is closed")
        if step <= 0:
            raise ValueError("fault injection step must be positive")
        created: list[FaultInjectionRecord] = []
        for spec in self.campaign.faults:
            key = (spec.fault_id, step)
            if step not in spec.steps or key in self._triggered:
                continue
            self._triggered.add(key)
            if self.rank != spec.target.rank:
                continue
            record = self._new_record(spec, step)
            self._records.append(record)
            created.append(record)
            if not _probability_selected(spec, step, self.rank):
                record.status = InjectionStatus.SKIPPED_PROBABILITY
                continue
            try:
                module = self._models.resolve_module(spec.target)
                self._schedule(module, spec, record)
            except Exception as error:
                record.status = InjectionStatus.FAILED
                record.error = str(error)
                raise
        return tuple(created)

    def evaluate(
        self,
        results: tuple[LocalizationResult | Mapping[str, Any], ...]
        | list[LocalizationResult | Mapping[str, Any]] = (),
    ) -> CampaignReport:
        """Compare neutral localization results with verified ground truth."""
        normalized = tuple(
            LocalizationResult.from_dict(result) if isinstance(result, Mapping) else result
            for result in results
        )
        if not all(isinstance(result, LocalizationResult) for result in normalized):
            raise TypeError("localization results must be LocalizationResult instances or mappings")
        by_injection: dict[str, LocalizationResult] = {}
        for result in normalized:
            if result.injection_id in by_injection:
                raise ValueError(f"duplicate localization result for {result.injection_id!r}")
            by_injection[result.injection_id] = result
        known_injections = {record.injection_id for record in self._records}
        unknown = sorted(set(by_injection) - known_injections)
        if unknown:
            raise ValueError(f"localization results reference unknown injections: {unknown}")

        evaluations: list[FaultEvaluation] = []
        for record in self._records:
            result = by_injection.get(record.injection_id)
            reported_ranks = result.failed_ranks if result is not None else ()
            detected = bool(result is not None and result.detected)
            localized = bool(
                record.injection_succeeded and detected and record.expected_rank in reported_ranks
            )
            unexpected = tuple(rank for rank in reported_ranks if rank != record.expected_rank)
            kind_matches = (
                None
                if result is None or result.kind is None
                else result.kind == record.expected_kind
            )
            component_matches = (
                None
                if result is None or result.component is None
                else result.component == record.module
            )
            evaluations.append(
                FaultEvaluation(
                    injection_id=record.injection_id,
                    injection_succeeded=record.injection_succeeded,
                    detected=detected,
                    localized=localized,
                    expected_rank=record.expected_rank,
                    reported_ranks=reported_ranks,
                    unexpected_ranks=unexpected,
                    kind_matches=kind_matches,
                    component_matches=component_matches,
                    latency_ms=None if result is None else result.latency_ms,
                )
            )
        return CampaignReport(
            campaign=self.campaign.name,
            manifest=self.campaign.to_dict(),
            framework=self.framework,
            rank=self.rank,
            injections=tuple(self._records),
            localizations=normalized,
            evaluations=tuple(evaluations),
            metadata=self.campaign.metadata,
        )

    def restore(self) -> None:
        """Restore injection-time values for persistent parameter faults."""
        for restoration in reversed(self._persistent):
            restoration.restore()
        self._persistent.clear()

    def close(self, *, restore: bool = True) -> None:
        """Remove pending hooks and optionally restore persistent faults."""
        if self._closed:
            return
        for pending in self._pending:
            pending.remove_hooks()
            if pending.restoration is not None:
                pending.restoration.restore()
            if pending.record.status is InjectionStatus.PENDING:
                pending.record.status = InjectionStatus.CANCELLED
        self._pending.clear()
        if restore:
            self.restore()
        self._closed = True

    def __enter__(self) -> "FaultInjectionSession":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

    def _new_record(self, spec: FaultSpec, step: int) -> FaultInjectionRecord:
        return FaultInjectionRecord(
            injection_id=f"{spec.fault_id}@{step}",
            fault_id=spec.fault_id,
            step=step,
            framework=self.framework,
            rank=self.rank,
            expected_rank=spec.target.rank,
            model_part=spec.target.model_part,
            module=spec.target.module,
            location=spec.target.location.value,
            fault_type=spec.fault_type.value,
            expected_kind=spec.expected_kind,
            magnitude=spec.magnitude.value,
            scope=spec.scope.value,
            persistence=spec.persistence.value,
            probability=spec.probability,
            seed=spec.seed,
            call_index=spec.call_index,
            delay_ms=spec.delay_ms,
            triggered_at_ns=time.monotonic_ns(),
        )

    def _schedule(
        self,
        module: nn.Module,
        spec: FaultSpec,
        record: FaultInjectionRecord,
    ) -> None:
        if spec.target.location is FaultLocation.OUTPUT:
            self._schedule_output(module, spec, record)
            return
        tensor = _target_tensor(module, spec.target.location)
        if spec.persistence is FaultPersistence.PERSISTENT:
            restoration, affected = _inject_tensor(tensor, spec)
            self._persistent.append(restoration)
            _mark_injected(record, affected)
            return
        self._schedule_transient_parameter(module, tensor, spec, record)

    def _schedule_output(
        self,
        module: nn.Module,
        spec: FaultSpec,
        record: FaultInjectionRecord,
    ) -> None:
        pending = _PendingInjection(record)
        calls = 0

        def inject_output(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            output: Any,
        ) -> Any | None:
            nonlocal calls
            calls += 1
            if calls != spec.call_index:
                return None
            pending.remove_hooks()
            pending.completed = True
            try:
                if spec.fault_type is FaultType.DELAY:
                    record.injected_at_ns = time.monotonic_ns()
                    time.sleep(spec.delay_ms / 1000.0)
                    record.status = InjectionStatus.INJECTED
                    return None
                corrupted, affected = _inject_output_tree(output, spec)
                _mark_injected(record, affected)
                return corrupted
            except Exception as error:
                record.status = InjectionStatus.FAILED
                record.error = str(error)
                raise

        pending.handles.append(
            module.register_forward_hook(
                inject_output,
                with_kwargs=True,
                always_call=True,
            )
        )
        self._pending.append(pending)

    def _schedule_transient_parameter(
        self,
        module: nn.Module,
        tensor: torch.Tensor,
        spec: FaultSpec,
        record: FaultInjectionRecord,
    ) -> None:
        pending = _PendingInjection(record)
        calls = 0

        def inject_parameter(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
        ) -> None:
            nonlocal calls
            calls += 1
            if calls != spec.call_index:
                return None
            try:
                pending.restoration, affected = _inject_tensor(tensor, spec)
                _mark_injected(record, affected)
            except Exception as error:
                pending.remove_hooks()
                pending.completed = True
                record.status = InjectionStatus.FAILED
                record.error = str(error)
                raise
            return None

        def restore_parameter(
            _module: nn.Module,
            _args: tuple[Any, ...],
            _kwargs: dict[str, Any],
            _output: Any,
        ) -> None:
            if pending.restoration is None:
                return None
            pending.restoration.restore()
            pending.remove_hooks()
            pending.completed = True
            return None

        pending.handles.append(module.register_forward_pre_hook(inject_parameter, with_kwargs=True))
        pending.handles.append(
            module.register_forward_hook(
                restore_parameter,
                with_kwargs=True,
                always_call=True,
            )
        )
        self._pending.append(pending)


def enable_fault_injection(
    target: Any,
    campaign: FaultCampaign,
    *,
    framework: str | None = None,
    rank: int | None = None,
) -> FaultInjectionSession:
    """Bind a fault campaign to initialized framework training objects."""
    return campaign.bind(target, framework=framework, rank=rank)


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _probability_selected(spec: FaultSpec, step: int, rank: int) -> bool:
    if spec.probability >= 1.0:
        return True
    if spec.probability <= 0.0:
        return False
    mixed_seed = (
        (spec.seed & ((1 << 64) - 1))
        ^ ((step * 0x9E3779B185EBCA87) & ((1 << 64) - 1))
        ^ ((rank * 0xC2B2AE3D27D4EB4F) & ((1 << 64) - 1))
    )
    return random.Random(mixed_seed).random() < spec.probability


def _target_tensor(module: nn.Module, location: FaultLocation) -> torch.Tensor:
    target = getattr(module, location.value, None)
    if not isinstance(target, torch.Tensor):
        raise LookupError(f"module {type(module).__name__} has no tensor {location.value!r}")
    if not target.is_floating_point():
        raise TypeError("numerical fault targets must use a floating-point tensor")
    if not target.is_contiguous():
        raise ValueError("numerical fault targets must be contiguous")
    if target.numel() == 0:
        raise ValueError("numerical fault target must be non-empty")
    return target


def _inject_tensor(
    tensor: torch.Tensor,
    spec: FaultSpec,
) -> tuple[_Restoration, int]:
    indices = _target_indices(tensor, spec.scope)
    with torch.no_grad():
        flat = tensor.view(-1)
        original = flat.index_select(0, indices).clone()
        _apply_fault(tensor, indices, spec)
        modified = flat.index_select(0, indices)
        if torch.equal(original, modified):
            raise RuntimeError("fault injection did not change the selected tensor values")
    return _Restoration(tensor, indices, original), int(indices.numel())


def _inject_output_tree(output: Any, spec: FaultSpec) -> tuple[Any, int]:
    leaves, tree_spec = tree_flatten(output)
    for index, leaf in enumerate(leaves):
        if not isinstance(leaf, torch.Tensor) or leaf.numel() == 0:
            continue
        if not leaf.is_floating_point():
            continue
        corrupted = leaf.clone().contiguous()
        indices = _target_indices(corrupted, spec.scope)
        before = corrupted.view(-1).index_select(0, indices).clone()
        with torch.no_grad():
            _apply_fault(corrupted, indices, spec)
            after = corrupted.view(-1).index_select(0, indices)
            if torch.equal(before, after):
                raise RuntimeError("fault injection did not change the selected output values")
        leaves[index] = corrupted
        return tree_unflatten(leaves, tree_spec), int(indices.numel())
    raise TypeError("target module output contains no non-empty floating-point tensor")


def _target_indices(tensor: torch.Tensor, scope: FaultScope) -> torch.Tensor:
    numel = tensor.numel()
    device = tensor.device
    if scope is FaultScope.SINGLE:
        return torch.tensor([numel // 2], device=device, dtype=torch.long)
    if scope is FaultScope.ROW:
        row_size = tensor.shape[-1] if tensor.ndim >= 2 else min(256, numel)
        return torch.arange(row_size, device=device, dtype=torch.long)
    if scope is FaultScope.PERCENT_1:
        count = max(1, numel // 100)
    elif scope is FaultScope.PERCENT_10:
        count = max(1, numel // 10)
    else:
        count = numel
    if count == numel:
        return torch.arange(numel, device=device, dtype=torch.long)
    step = max(1, numel // count)
    return torch.arange(0, numel, step, device=device, dtype=torch.long)[:count]


def _apply_fault(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    spec: FaultSpec,
) -> None:
    flat = tensor.view(-1)
    selected = flat.index_select(0, indices)
    if spec.fault_type is FaultType.SINGLE_BITFLIP:
        _flip_bits(tensor, indices, spec.magnitude, count=1)
    elif spec.fault_type is FaultType.MULTI_BITFLIP:
        _flip_bits(tensor, indices, spec.magnitude, count=4)
    elif spec.fault_type is FaultType.STUCK_AT_ZERO:
        flat.index_fill_(0, indices, 0.0)
    elif spec.fault_type is FaultType.STUCK_AT_ONE:
        flat.index_fill_(0, indices, 1.0)
    elif spec.fault_type is FaultType.SCALE_UP:
        flat.index_copy_(0, indices, selected * _SCALE_UP[spec.magnitude])
    elif spec.fault_type is FaultType.SCALE_DOWN:
        flat.index_copy_(0, indices, selected * _SCALE_DOWN[spec.magnitude])
    elif spec.fault_type is FaultType.GAUSSIAN_NOISE:
        generator = torch.Generator(device=tensor.device)
        generator.manual_seed(spec.seed)
        noise = torch.randn(
            selected.shape,
            dtype=tensor.dtype,
            device=tensor.device,
            generator=generator,
        )
        flat.index_copy_(0, indices, selected + noise * _NOISE_STD[spec.magnitude])
    elif spec.fault_type is FaultType.SIGN_FLIP:
        flat.index_copy_(0, indices, -selected)
    elif spec.fault_type is FaultType.SET_NAN:
        flat.index_fill_(0, indices, float("nan"))
    elif spec.fault_type is FaultType.SET_INF:
        flat.index_fill_(0, indices, float("inf"))
    else:
        raise ValueError(f"fault type {spec.fault_type.value!r} cannot modify tensor values")


def _flip_bits(
    tensor: torch.Tensor,
    indices: torch.Tensor,
    magnitude: FaultMagnitude,
    *,
    count: int,
) -> None:
    try:
        integer_dtype, width = _INTEGER_VIEW[tensor.dtype]
    except KeyError as error:
        raise TypeError(f"bit flips do not support dtype {tensor.dtype}") from error
    if not tensor.is_contiguous():
        raise ValueError("bit-flip target must be contiguous")
    base = {
        FaultMagnitude.NEAR_INVISIBLE: 0,
        FaultMagnitude.SUBTLE: width // 4,
        FaultMagnitude.MEDIUM: width // 2,
        FaultMagnitude.LARGE: (3 * width) // 4,
        FaultMagnitude.CATASTROPHIC: width - 2,
    }[magnitude]
    start = min(base, width - 1 - count)
    mask = sum(1 << bit for bit in range(start, start + count))
    integer_flat = tensor.view(integer_dtype).view(-1)
    selected = integer_flat.index_select(0, indices)
    integer_flat.index_copy_(0, indices, torch.bitwise_xor(selected, mask))


def _mark_injected(record: FaultInjectionRecord, affected: int) -> None:
    record.status = InjectionStatus.INJECTED
    record.affected_elements = affected
    record.injected_at_ns = time.monotonic_ns()


__all__ = ["FaultInjectionSession", "enable_fault_injection"]
