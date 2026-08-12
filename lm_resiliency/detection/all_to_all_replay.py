"""Policy-driven replay for representative AllToAll traffic matrices."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from math import prod
from typing import Any, Sequence

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class TensorReplaySpec:
    """Tensor storage needed to reconstruct collective pressure."""

    shape: tuple[int, ...]
    dtype: torch.dtype
    numel: int
    element_size: int

    @property
    def bytes(self) -> int:
        return self.numel * self.element_size


@dataclass(frozen=True)
class AllToAllReplayRecipe:
    """Payload-free capture of one EP dispatch or combine collective."""

    sequence: int
    collective: str
    group_ranks: tuple[int, ...] | None
    inputs: tuple[TensorReplaySpec, ...]
    outputs: tuple[TensorReplaySpec, ...]
    input_split_sizes: tuple[int, ...] | None
    output_split_sizes: tuple[int, ...] | None
    async_op: bool
    group: Any | None = field(default=None, repr=False, compare=False)

    @property
    def input_bytes(self) -> int:
        return sum(spec.bytes for spec in self.inputs)

    @property
    def output_bytes(self) -> int:
        return sum(spec.bytes for spec in self.outputs)


@dataclass(frozen=True)
class AllToAllCapture:
    """Globally reconstructed metadata for one captured AllToAll."""

    sequence: int
    collective: str
    group_ranks: tuple[int, ...]
    dtype: torch.dtype
    trailing_shape: tuple[int, ...]
    element_size: int
    observed_splits: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        size = len(self.group_ranks)
        if size < 1 or len(self.observed_splits) != size:
            raise ValueError("AllToAll capture must contain one row per group rank")
        if any(len(row) != size for row in self.observed_splits):
            raise ValueError("captured AllToAll split matrix must be square")
        if any(value < 0 for row in self.observed_splits for value in row):
            raise ValueError("captured AllToAll splits must be non-negative")
        if self.element_size <= 0 or self.bytes_per_unit <= 0:
            raise ValueError("captured AllToAll row width must be positive")

    @property
    def group_size(self) -> int:
        return len(self.group_ranks)

    @property
    def bytes_per_unit(self) -> int:
        return prod(self.trailing_shape) * self.element_size


@dataclass(frozen=True)
class AllToAllTrafficMatrix:
    """One policy-selected traffic matrix indexed by source then destination."""

    name: str
    splits: tuple[tuple[int, ...], ...]

    def __post_init__(self) -> None:
        if not self.name or not self.splits:
            raise ValueError("AllToAll traffic matrix must have a name and rows")
        width = len(self.splits)
        if any(len(row) != width for row in self.splits):
            raise ValueError("AllToAll traffic matrix must be square")
        if any(value < 0 for row in self.splits for value in row):
            raise ValueError("AllToAll traffic splits must be non-negative")
        if not any(value for row in self.splits for value in row):
            raise ValueError("AllToAll traffic matrix must contain traffic")


class AllToAllReplayPolicy(ABC):
    """Generate bounded representative matrices from a captured collective."""

    @abstractmethod
    def generate(self, capture: AllToAllCapture) -> Sequence[AllToAllTrafficMatrix]:
        """Return deterministic matrices in the same order on equivalent replicas."""


@dataclass(frozen=True)
class BalancedAndPermutationPolicy(AllToAllReplayPolicy):
    """Replay one dense and one sparse matrix with bounded per-rank payloads.

    The balanced matrix exercises concurrent traffic among peers.
    The cyclic permutation matrix sends each rank's full payload over one route
    while keeping every rank's send and receive volume equal.
    """

    max_payload_bytes_per_rank: int = 4 * 1024 * 1024

    def __post_init__(self) -> None:
        if self.max_payload_bytes_per_rank <= 0:
            raise ValueError("max_payload_bytes_per_rank must be positive")

    def generate(self, capture: AllToAllCapture) -> Sequence[AllToAllTrafficMatrix]:
        world_size = capture.group_size
        if world_size < 1:
            return ()
        units = max(1, self.max_payload_bytes_per_rank // capture.bytes_per_unit)

        balanced = tuple(
            _balanced_row(
                units,
                world_size,
                start=(source + capture.sequence) % world_size,
            )
            for source in range(world_size)
        )
        matrices = [AllToAllTrafficMatrix("balanced", balanced)]

        offset = 0 if world_size == 1 else 1 + capture.sequence % (world_size - 1)
        permutation = tuple(
            tuple(
                units if destination == (source + offset) % world_size else 0
                for destination in range(world_size)
            )
            for source in range(world_size)
        )
        if permutation != balanced:
            matrices.append(
                AllToAllTrafficMatrix(
                    f"cyclic_permutation_{offset}",
                    permutation,
                )
            )
        return matrices


@dataclass(frozen=True)
class AllToAllReplayOutcome:
    """Local correctness and timing from one representative matrix."""

    matrix: AllToAllTrafficMatrix
    sequence: int
    group_ranks: tuple[int, ...]
    dtype: torch.dtype
    trailing_shape: tuple[int, ...]
    latency_ms: float
    input_bytes: int
    output_bytes: int
    correct: bool

    @property
    def comparison_signature(self) -> tuple[Any, ...]:
        """Metadata that must match across equivalent replay groups."""
        return (
            self.matrix.name,
            self.matrix.splits,
            str(self.dtype),
            self.trailing_shape,
            self.input_bytes,
            self.output_bytes,
        )


class AllToAllReplayExecutor:
    """Materialize, execute, and verify policy-selected AllToAll recipes."""

    def __init__(self, device: torch.device) -> None:
        self._device = device

    def replay(
        self,
        recipe: AllToAllReplayRecipe,
        policy: AllToAllReplayPolicy,
    ) -> list[AllToAllReplayOutcome]:
        capture = _gather_capture(recipe)
        global_rank = dist.get_rank()
        try:
            local_rank = capture.group_ranks.index(global_rank)
        except ValueError as exc:
            raise RuntimeError("current rank is absent from captured AllToAll group") from exc

        matrices = _generate_policy_matrices(
            capture,
            policy,
            group=recipe.group,
        )

        outcomes = []
        for matrix in matrices:
            outcomes.append(
                self._execute_matrix(
                    capture,
                    matrix,
                    local_rank=local_rank,
                    group=recipe.group,
                )
            )
        return outcomes

    def _execute_matrix(
        self,
        capture: AllToAllCapture,
        matrix: AllToAllTrafficMatrix,
        *,
        local_rank: int,
        group: Any | None,
    ) -> AllToAllReplayOutcome:
        input_splits = matrix.splits[local_rank]
        output_splits = tuple(row[local_rank] for row in matrix.splits)
        input_tensor = _materialize_input(
            input_splits,
            source=local_rank,
            world_size=capture.group_size,
            trailing_shape=capture.trailing_shape,
            dtype=capture.dtype,
            device=self._device,
        )
        output_tensor = torch.empty(
            (sum(output_splits), *capture.trailing_shape),
            dtype=capture.dtype,
            device=self._device,
        )

        _synchronize(self._device)
        started = time.perf_counter()
        dist.all_to_all_single(
            output_tensor,
            input_tensor,
            output_split_sizes=list(output_splits),
            input_split_sizes=list(input_splits),
            group=group,
            async_op=False,
        )
        _synchronize(self._device)
        latency_ms = (time.perf_counter() - started) * 1000.0

        expected = _materialize_output(
            output_splits,
            destination=local_rank,
            world_size=capture.group_size,
            trailing_shape=capture.trailing_shape,
            dtype=capture.dtype,
            device=self._device,
        )
        return AllToAllReplayOutcome(
            matrix=matrix,
            sequence=capture.sequence,
            group_ranks=capture.group_ranks,
            dtype=capture.dtype,
            trailing_shape=capture.trailing_shape,
            latency_ms=latency_ms,
            input_bytes=input_tensor.numel() * input_tensor.element_size(),
            output_bytes=output_tensor.numel() * output_tensor.element_size(),
            correct=torch.equal(output_tensor, expected),
        )


@dataclass(frozen=True)
class _RankCapture:
    sequence: int
    collective: str
    dtype: torch.dtype
    trailing_shape: tuple[int, ...]
    element_size: int
    input_splits: tuple[int, ...]
    output_splits: tuple[int, ...]


def _gather_capture(recipe: AllToAllReplayRecipe) -> AllToAllCapture:
    group_ranks = recipe.group_ranks
    if group_ranks is None:
        group_ranks = tuple(range(dist.get_world_size()))
    local = _rank_capture(recipe, len(group_ranks))
    gathered: list[Any] = [None] * len(group_ranks)
    dist.all_gather_object(gathered, local, group=recipe.group)
    captures = [item for item in gathered if isinstance(item, _RankCapture)]
    if len(captures) != len(group_ranks):
        raise RuntimeError("incomplete AllToAll replay metadata")

    reference = captures[0]
    for capture in captures[1:]:
        if (
            capture.sequence != reference.sequence
            or capture.collective != reference.collective
            or capture.dtype != reference.dtype
            or capture.trailing_shape != reference.trailing_shape
            or capture.element_size != reference.element_size
        ):
            raise RuntimeError("AllToAll replay metadata differs within the process group")

    rows = tuple(capture.input_splits for capture in captures)
    for destination, capture in enumerate(captures):
        expected = tuple(row[destination] for row in rows)
        if capture.output_splits != expected:
            raise RuntimeError("captured AllToAll input and output split matrices are inconsistent")
    return AllToAllCapture(
        sequence=reference.sequence,
        collective=reference.collective,
        group_ranks=group_ranks,
        dtype=reference.dtype,
        trailing_shape=reference.trailing_shape,
        element_size=reference.element_size,
        observed_splits=rows,
    )


def _generate_policy_matrices(
    capture: AllToAllCapture,
    policy: AllToAllReplayPolicy,
    *,
    group: Any | None,
) -> list[AllToAllTrafficMatrix]:
    matrices: list[AllToAllTrafficMatrix] = []
    error = ""
    try:
        matrices = list(policy.generate(capture))
        if not matrices:
            raise ValueError("policy returned no traffic matrices")
        if any(not isinstance(matrix, AllToAllTrafficMatrix) for matrix in matrices):
            raise TypeError("policy returned an unsupported matrix type")
        if any(len(matrix.splits) != capture.group_size for matrix in matrices):
            raise ValueError("policy returned a matrix with the wrong group size")
    except Exception as exc:  # noqa: BLE001 - synchronize policy failures before replay
        error = f"{type(exc).__name__}: {exc}"

    contract = tuple((matrix.name, matrix.splits) for matrix in matrices)
    local = (not error, error, contract)
    gathered: list[Any] = [None] * capture.group_size
    dist.all_gather_object(gathered, local, group=group)
    failures = [str(item[1]) for item in gathered if item is not None and not item[0]]
    if failures:
        raise RuntimeError(
            "AllToAll replay policy failed within the process group: "
            + "; ".join(sorted(set(failures)))
        )
    contracts = [item[2] for item in gathered if item is not None]
    if len(contracts) != capture.group_size or any(item != contracts[0] for item in contracts[1:]):
        raise RuntimeError(
            "AllToAll replay policy generated different matrices within the process group"
        )
    return matrices


def _rank_capture(recipe: AllToAllReplayRecipe, world_size: int) -> _RankCapture:
    if recipe.input_split_sizes is None or recipe.output_split_sizes is None:
        raise RuntimeError("captured AllToAll split sizes are unavailable")
    if len(recipe.input_split_sizes) != world_size or len(recipe.output_split_sizes) != world_size:
        raise RuntimeError("captured AllToAll split count differs from group size")

    input_spec = _common_spec(recipe.inputs, recipe.input_split_sizes)
    output_spec = _common_spec(recipe.outputs, recipe.output_split_sizes)
    if (
        input_spec.dtype != output_spec.dtype
        or input_spec.shape[1:] != output_spec.shape[1:]
        or input_spec.element_size != output_spec.element_size
    ):
        raise RuntimeError("captured AllToAll input and output layouts differ")
    return _RankCapture(
        sequence=recipe.sequence,
        collective=recipe.collective,
        dtype=input_spec.dtype,
        trailing_shape=input_spec.shape[1:],
        element_size=input_spec.element_size,
        input_splits=recipe.input_split_sizes,
        output_splits=recipe.output_split_sizes,
    )


def _common_spec(
    specs: tuple[TensorReplaySpec, ...],
    splits: tuple[int, ...],
) -> TensorReplaySpec:
    if len(specs) == 1:
        spec = specs[0]
        if not spec.shape or spec.shape[0] != sum(splits):
            raise RuntimeError("captured AllToAll tensor shape does not match its splits")
        return spec
    if len(specs) != len(splits) or not specs:
        raise RuntimeError("captured AllToAll tensor-list layout is unsupported")
    reference = specs[0]
    for spec, split in zip(specs, splits):
        if (
            not spec.shape
            or spec.shape[0] != split
            or spec.shape[1:] != reference.shape[1:]
            or spec.dtype != reference.dtype
            or spec.element_size != reference.element_size
        ):
            raise RuntimeError("captured AllToAll tensor-list layouts differ")
    return TensorReplaySpec(
        shape=(sum(splits), *reference.shape[1:]),
        dtype=reference.dtype,
        numel=sum(spec.numel for spec in specs),
        element_size=reference.element_size,
    )


def _balanced_row(
    units: int,
    world_size: int,
    *,
    start: int,
) -> tuple[int, ...]:
    base, remainder = divmod(units, world_size)
    row = [base] * world_size
    for index in range(remainder):
        row[(start + index) % world_size] += 1
    return tuple(row)


def _materialize_input(
    splits: Sequence[int],
    *,
    source: int,
    world_size: int,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _concatenate_segments(
        [
            _payload_segment(
                count,
                source=source,
                destination=destination,
                world_size=world_size,
                trailing_shape=trailing_shape,
                dtype=dtype,
                device=device,
            )
            for destination, count in enumerate(splits)
        ],
        trailing_shape=trailing_shape,
        dtype=dtype,
        device=device,
    )


def _materialize_output(
    splits: Sequence[int],
    *,
    destination: int,
    world_size: int,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    return _concatenate_segments(
        [
            _payload_segment(
                count,
                source=source,
                destination=destination,
                world_size=world_size,
                trailing_shape=trailing_shape,
                dtype=dtype,
                device=device,
            )
            for source, count in enumerate(splits)
        ],
        trailing_shape=trailing_shape,
        dtype=dtype,
        device=device,
    )


def _payload_segment(
    count: int,
    *,
    source: int,
    destination: int,
    world_size: int,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    shape = (count, *trailing_shape)
    numel = count * prod(trailing_shape)
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=device)
    seed = source * world_size + destination + 1
    values = (torch.arange(numel, dtype=torch.int64, device=device) + seed * 17) % 97
    if dtype == torch.bool:
        values = values.remainder(2)
    return values.to(dtype=dtype).reshape(shape)


def _concatenate_segments(
    segments: Sequence[torch.Tensor],
    *,
    trailing_shape: tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    if not segments:
        return torch.empty((0, *trailing_shape), dtype=dtype, device=device)
    return torch.cat(list(segments), dim=0)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


__all__ = [
    "AllToAllCapture",
    "AllToAllReplayExecutor",
    "AllToAllReplayOutcome",
    "AllToAllReplayPolicy",
    "AllToAllReplayRecipe",
    "AllToAllTrafficMatrix",
    "BalancedAndPermutationPolicy",
    "TensorReplaySpec",
]
