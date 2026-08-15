# mypy: ignore-errors
"""Layer Replay Detector: orchestrates SDC and straggler detection.

Replays a sampled model layer across a peer group with identical inputs,
extracting two signals:
  - Output signal → SDC localization (bitwise divergence = hardware fault)
  - Timing signal → straggler localization (compute outlier = degraded GPU/NIC)

This module owns the broadcast logic, replay execution, and result assembly.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency.detection._utils import (
    deterministic_mode,
    synchronized_replay_rng,
    timed_call,
)
from lm_resiliency.detection.c3 import C3, C3Mode, C3Result, C3Status
from lm_resiliency.detection.cross_pg import (
    CollectiveTimingSample,
    CrossPGResult,
)
from lm_resiliency.detection.optimizer_step import (
    OPTIMIZER_REPLAY_INPUT,
    OPTIMIZER_REPLAY_STATUS,
    OPTIMIZER_STATUS_OK,
    OPTIMIZER_UPDATED_WEIGHT,
    OptimizerReplayBatch,
)

_COMM_OPS = (
    "all_reduce",
    "all_gather",
    "all_gather_into_tensor",
    "all_to_all",
    "all_to_all_single",
    "broadcast",
    "reduce_scatter",
    "reduce_scatter_tensor",
)

_COMM_PARAMETERS = {
    "all_reduce": ("tensor", "op", "group", "async_op"),
    "all_gather": ("tensor_list", "tensor", "group", "async_op"),
    "all_gather_into_tensor": ("output_tensor", "input_tensor", "group", "async_op"),
    "all_to_all": ("output_tensor_list", "input_tensor_list", "group", "async_op"),
    "all_to_all_single": (
        "output",
        "input",
        "output_split_sizes",
        "input_split_sizes",
        "group",
        "async_op",
    ),
    "broadcast": ("tensor", "src", "group", "async_op", "group_src"),
    "reduce_scatter": ("output", "input_list", "op", "group", "async_op"),
    "reduce_scatter_tensor": ("output", "input", "op", "group", "async_op"),
}

_COMM_INPUT_NAMES = {
    "all_reduce": ("tensor",),
    "all_gather": ("tensor",),
    "all_gather_into_tensor": ("input_tensor",),
    "all_to_all": ("input_tensor_list",),
    "all_to_all_single": ("input",),
    "broadcast": ("tensor",),
    "reduce_scatter": ("input_list",),
    "reduce_scatter_tensor": ("input",),
}

GradientCommunicationReplay = Callable[
    [nn.Module, Sequence[torch.Tensor | None]],
    None,
]

PARAMETER_STATE = "parameter_state"
REPLAY_INPUT = "replay_input"
REPLAY_RNG_STATE = "replay_rng_state"
FSDP_PARAMETER_ALL_GATHER = "fsdp_parameter_all_gather"


@dataclass
class OpTiming:
    """Latency of a single op (submodule or collective) during instrumented replay."""

    name: str
    type: str  # "compute" or "communication"
    time_ms: float
    group_ranks: tuple[int, ...] = ()
    message_bytes: int = 0
    sequence: int = 0


@dataclass
class StragglerDetail:
    """Detailed straggler localization result from two-phase detection."""

    straggler_rank: int | None
    straggler_type: str  # "compute" or "none"
    compute_times_ms: list[float]
    comm_times_ms: list[float]
    compute_bitmap: list[int]
    communication_bitmap: list[int] = field(default_factory=list)
    op_timings: list[OpTiming] = field(default_factory=list)
    collective_timings: list[CollectiveTimingSample] = field(default_factory=list)


@dataclass
class ReplayResult:
    """Result of a single layer replay detection round."""

    sdc_bitmap: list[int]
    straggler_bitmap: list[int]
    replay_time_ms: float
    layer_id: int
    straggler_detail: StragglerDetail | None = None
    peer_ranks: list[int] = field(default_factory=list)
    replay_times_ms: list[float] = field(default_factory=list)
    sdc_sources: list[str] = field(default_factory=list)
    sdc_source_bitmaps: dict[str, list[int]] = field(default_factory=dict)
    replay_mode: str = "forward"
    spatial_straggler_bitmap: list[int] = field(default_factory=list)
    temporal_straggler_bitmap: list[int] = field(default_factory=list)
    straggler_confirmations: int = 0
    temporal_group_slowdown: bool = False
    replay_shape_id: str = "captured"
    replay_shape: tuple[int, ...] | None = None
    checked_shape_ids: list[str] = field(default_factory=lambda: ["captured"])
    checked_shapes: list[tuple[int, ...] | None] = field(default_factory=lambda: [None])
    shape_cycle_size: int = 1
    completed_shape_cycle: bool = True
    completed_scheduled_cycle: bool = False
    scheduled_cycle: int = 0
    checked_recipe_ids: list[str] = field(default_factory=list)
    dense_replay: bool = False
    c3_results: dict[str, C3Result] = field(default_factory=dict)
    timing_c3_result: C3Result | None = None
    communication_times_ms: dict[str, float] = field(default_factory=dict)
    communication_peer_times_ms: dict[str, list[float]] = field(default_factory=dict)
    collective_timings: list[CollectiveTimingSample] = field(default_factory=list)
    cross_pg_result: CrossPGResult | None = None


@dataclass
class ReplayInvocation:
    """Captured layer call, including structured inputs and backward signal."""

    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    input_requires_grad: list[bool] = field(default_factory=list)
    grad_output: Any | None = None
    autocast_enabled: bool = False
    autocast_device_type: str = "cuda"
    autocast_dtype: torch.dtype | None = None


ReplayInvocationPreparer = Callable[
    [nn.Module, ReplayInvocation],
    ReplayInvocation,
]
ReplayEvidencePreparer = Callable[
    [dict[str, list[torch.Tensor]]],
    dict[str, list[torch.Tensor]],
]


@dataclass(frozen=True)
class _BroadcastTensorSpec:
    """Metadata for one tensor leaf in a source-owned nested payload."""

    index: int
    shape: tuple[int, ...]
    dtype: torch.dtype
    device_type: str


@dataclass(frozen=True)
class _CollectiveEvent:
    """CUDA timing events plus the process-group identity of one collective."""

    name: str
    group_ranks: tuple[int, ...]
    message_bytes: int
    sequence: int
    start: torch.cuda.Event
    end: torch.cuda.Event


def replay_result_has_fault(result: ReplayResult | None) -> bool:
    """Whether a replay result contains a numerical or confirmed timing anomaly."""
    return result is not None and (
        replay_result_has_sdc(result)
        or any(result.straggler_bitmap)
        or result.temporal_group_slowdown
        or bool(result.cross_pg_result and result.cross_pg_result.confirmed)
    )


def replay_result_has_sdc(result: ReplayResult | None) -> bool:
    """Whether replay found numerical corruption that must block checkpoint capture."""
    if result is None:
        return False
    if any(result.sdc_bitmap):
        return True
    return any(
        c3_result.status is C3Status.INCONCLUSIVE
        for c3_result in getattr(result, "c3_results", {}).values()
    )


def replay_result_certifies_checkpoint(result: ReplayResult | None) -> bool:
    """Whether SCOUT completed the full configured shape plan without numerical SDC."""
    return result is not None and result.completed_shape_cycle and not replay_result_has_sdc(result)


class LayerReplayDetector:
    """Detects SDC and stragglers by replaying a model layer across peers.

    All ranks in the C3 peer group replay the same layer with identical inputs.
    One replay produces both timing (straggler) and output (SDC) signals.

    The detector:
      1. Broadcasts one rank's activation to all peers.
      2. Each rank replays the layer forward (and optionally backward).
      3. Compares outputs via C3 (SDC detection) — stays on GPU.
      4. Compares timing via C3 (straggler detection).

    Args:
        group: Process group for scalar C3 (AllGather). Should support Gloo.
        nccl_group: NCCL process group for GPU tensor C3 signature exchange.
        broadcast_src: Rank (within group) that provides the reference activation.
        device: CUDA device for replay execution.
        deterministic: Whether to enforce deterministic algorithms during replay.
        synchronize_rng: Whether every peer should use the broadcast source's RNG
            state during replay without advancing its training RNG streams.
        compare_parameter_state: Whether sampled-layer parameters are equivalent
            replicas that should be compared exactly alongside replay outputs.
        gradient_communication: Optional framework adapter that replays gradient
            communication using the diagnostic parameter gradients.
    """

    def __init__(
        self,
        group: dist.ProcessGroup | None = None,
        nccl_group: dist.ProcessGroup | None = None,
        broadcast_src: int = 0,
        device: torch.device | None = None,
        deterministic: bool = True,
        synchronize_rng: bool = True,
        compare_parameter_state: bool = True,
        gradient_communication: GradientCommunicationReplay | None = None,
        invocation_preparer: ReplayInvocationPreparer | None = None,
        evidence_preparer: ReplayEvidencePreparer | None = None,
        straggler_min_slowdown_ratio: float = 1.1,
        straggler_min_slowdown_ms: float = 2.0,
    ) -> None:
        self._group = group
        self._nccl_group = nccl_group
        self._broadcast_src = broadcast_src
        self._device = device or torch.device("cuda")
        self._c3 = C3(group=group, nccl_group=nccl_group)
        self._rank = dist.get_rank(group) if group else dist.get_rank()
        self._peer_ranks = (
            dist.get_process_group_ranks(group)
            if group is not None
            else list(range(dist.get_world_size()))
        )
        self._deterministic = deterministic
        self._synchronize_rng = synchronize_rng
        self._compare_parameter_state = compare_parameter_state
        self._gradient_communication = gradient_communication
        self._invocation_preparer = invocation_preparer
        self._evidence_preparer = evidence_preparer
        self._straggler_min_slowdown_ratio = straggler_min_slowdown_ratio
        self._straggler_min_slowdown_ms = straggler_min_slowdown_ms

    @property
    def peer_ranks(self) -> list[int]:
        return self._peer_ranks.copy()

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def replay_forward(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        layer_id: int = 0,
    ) -> ReplayResult:
        """Replay a layer's forward pass and detect SDC + stragglers."""
        return self.replay_invocation(
            layer,
            ReplayInvocation(args=(activation,), kwargs={}),
            layer_id=layer_id,
            scale_factors=[],
        )

    def compare_local_parameter_shards(self, layer: nn.Module) -> list[int]:
        """Compare corresponding local parameter shards within an HSDP replica group."""
        local_shards = [
            parameter.to_local() if type(parameter).__name__ == "DTensor" else parameter.detach()
            for parameter in layer.parameters()
        ]
        return self._c3.run_tensor_sequence(local_shards).bitmap

    def compare_tensor_groups(
        self,
        tensor_groups: dict[str, list[torch.Tensor]],
    ) -> dict[str, C3Result]:
        """Compare named diagnostic surfaces and retain each C3 verdict."""
        return self._c3.run_tensor_groups(tensor_groups)

    def compare_structure(self, value: Any) -> C3Result:
        """Compare a compact structured replay contract exactly."""
        return self._c3.run_structure(value)

    def compare_expected_boolean(self, value: bool) -> C3Result:
        """Compare a local pass/fail result against the expected true value."""
        raw = self._c3.run_scalar(int(value), mode=C3Mode.EXACT)
        evidence = [bool(item) for item in raw.evidence]
        if all(evidence):
            return C3Result(C3Status.AGREE, [0] * len(evidence), raw.evidence)
        if sum(evidence) > len(evidence) // 2:
            return C3Result(
                C3Status.ATTRIBUTED,
                [int(not item) for item in evidence],
                raw.evidence,
            )
        return C3Result(
            C3Status.INCONCLUSIVE,
            [0] * len(evidence),
            raw.evidence,
        )

    def replay_optimizer_batch(
        self,
        batch: OptimizerReplayBatch,
    ) -> dict[str, C3Result]:
        """Broadcast source optimizer recipes, replay them, and compare outputs."""
        count_result = self._c3.run_scalar(len(batch.recipes), mode=C3Mode.EXACT)
        results = {"optimizer_replay_count": count_result}
        if count_result.status is not C3Status.AGREE:
            return results

        recipe_count = int(count_result.evidence[0])
        for index in range(recipe_count):
            recipe = batch.recipes[index]
            status_result = self._c3.run_scalar(recipe.status, mode=C3Mode.EXACT)
            source_status = int(status_result.evidence[self._broadcast_src])
            status_name = _optimizer_result_name(
                OPTIMIZER_REPLAY_STATUS,
                index,
                recipe_count,
            )
            results[status_name] = (
                C3Result(
                    C3Status.AGREE,
                    [0] * len(status_result.bitmap),
                    status_result.evidence,
                )
                if source_status == OPTIMIZER_STATUS_OK
                else C3Result(
                    C3Status.INCONCLUSIVE,
                    [0] * len(status_result.bitmap),
                    status_result.evidence,
                )
            )
            if source_status != OPTIMIZER_STATUS_OK:
                continue

            source_payload = recipe.source_payload() if self._rank == self._broadcast_src else None
            payload = self._broadcast_structure(source_payload)
            input_name = _optimizer_result_name(
                OPTIMIZER_REPLAY_INPUT,
                index,
                recipe_count,
            )
            input_result = self._c3.run_structure(payload)
            results[input_name] = input_result
            if input_result.status is not C3Status.AGREE:
                continue

            replayed: torch.Tensor | None = None
            replay_status = OPTIMIZER_STATUS_OK
            try:
                replayed = recipe.replay(payload)
            except Exception:  # noqa: BLE001 - compare failures across all peers
                replay_status = 1
            replay_status_result = self._c3.run_scalar(
                replay_status,
                mode=C3Mode.EXACT,
            )
            replay_status_result = _expected_status_result(
                replay_status_result,
                expected=OPTIMIZER_STATUS_OK,
            )
            replay_status_name = _optimizer_result_name(
                f"{OPTIMIZER_REPLAY_STATUS}.execution",
                index,
                recipe_count,
            )
            results[replay_status_name] = replay_status_result
            if replay_status_result.status is not C3Status.AGREE:
                continue

            assert replayed is not None
            output_name = _optimizer_result_name(
                OPTIMIZER_UPDATED_WEIGHT,
                index,
                recipe_count,
            )
            results[output_name] = self._c3.run_tensor_sequence([replayed])
        return results

    def replay_shape_consensus(self, signature: int) -> tuple[bool, list[int]]:
        """Verify that every peer selected the same replay-shape plan entry."""
        result = self._c3.run_scalar(signature, mode=C3Mode.EXACT)
        return len(set(int(value) for value in result.evidence)) == 1, result.bitmap

    def add_communication_timing(
        self,
        result: ReplayResult,
        *,
        name: str,
        elapsed_ms: float,
        group_ranks: Sequence[int] | None = None,
        topology_role: str | None = None,
        message_bytes: int = 0,
        sequence: int = 0,
    ) -> None:
        """Compare an adapter-visible communication boundary and merge its evidence."""
        raw = self._c3.run_scalar(elapsed_ms, mode=C3Mode.STATISTICAL)
        peer_times = [float(value) for value in raw.evidence]
        bitmap = _slow_outlier_bitmap(
            raw.bitmap,
            peer_times,
            self._straggler_min_slowdown_ratio,
            self._straggler_min_slowdown_ms,
        )
        timing_result = C3Result(
            C3Status.ATTRIBUTED if any(bitmap) else C3Status.AGREE,
            bitmap,
            raw.evidence,
        )
        result.c3_results[f"{name}.timing"] = timing_result
        result.communication_times_ms[name] = float(elapsed_ms)
        result.communication_peer_times_ms[name] = peer_times
        result.straggler_bitmap = _merge_bitmaps(result.straggler_bitmap, bitmap)
        result.spatial_straggler_bitmap = _merge_bitmaps(
            result.spatial_straggler_bitmap,
            bitmap,
        )

        detail = result.straggler_detail
        if detail is None:
            detail = StragglerDetail(
                straggler_rank=None,
                straggler_type="none",
                compute_times_ms=[],
                comm_times_ms=peer_times,
                compute_bitmap=[0] * len(bitmap),
                communication_bitmap=bitmap.copy(),
            )
            result.straggler_detail = detail
        else:
            detail.comm_times_ms = peer_times
            detail.communication_bitmap = _merge_bitmaps(
                detail.communication_bitmap,
                bitmap,
            )
        detail.op_timings.append(
            OpTiming(
                name=name,
                type="communication",
                time_ms=float(elapsed_ms),
                group_ranks=tuple(group_ranks or ()),
                message_bytes=message_bytes,
                sequence=sequence,
            )
        )
        if group_ranks:
            sample = CollectiveTimingSample(
                collective=name,
                group_ranks=tuple(int(rank) for rank in group_ranks),
                message_bytes=message_bytes,
                sequence=sequence,
                latency_ms=float(elapsed_ms),
                slow=bool(bitmap[self._rank]),
                topology_role=topology_role or name,
            )
            result.collective_timings.append(sample)
            detail.collective_timings.append(sample)
        _refresh_straggler_detail(detail, self._peer_ranks)

    def replay_forward_backward(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        grad_output: torch.Tensor,
        layer_id: int = 0,
    ) -> ReplayResult:
        """Replay forward + backward pass for deeper fault coverage."""
        return self.replay_invocation(
            layer,
            ReplayInvocation(
                args=(activation,),
                kwargs={},
                input_requires_grad=[True],
                grad_output=(grad_output,),
            ),
            layer_id=layer_id,
            scale_factors=[],
        )

    def replay_with_scaling(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        layer_id: int = 0,
        scale_factors: list[float] | None = None,
    ) -> ReplayResult:
        """Replay with multiple input scales to improve SDC detection coverage."""
        return self.replay_invocation(
            layer,
            ReplayInvocation(args=(activation,), kwargs={}),
            layer_id=layer_id,
            scale_factors=scale_factors,
        )

    def replay_invocation(
        self,
        layer: nn.Module,
        invocation: ReplayInvocation,
        layer_id: int = 0,
        scale_factors: list[float] | None = None,
    ) -> ReplayResult:
        """Replay a complete captured module invocation."""
        precondition_results = {}
        if self._compare_parameter_state:
            precondition_results[PARAMETER_STATE] = self._c3.run_tensor_sequence(
                [parameter.detach() for parameter in layer.parameters()]
            )
        shared = self._broadcast_invocation(invocation)
        precondition_results[REPLAY_INPUT] = self._c3.run_structure(
            (
                shared.args,
                shared.kwargs,
                shared.grad_output,
                shared.autocast_enabled,
                shared.autocast_device_type,
                shared.autocast_dtype,
            )
        )
        invocation_preparer = getattr(self, "_invocation_preparer", None)
        execution = (
            invocation_preparer(layer, shared) if invocation_preparer is not None else shared
        )
        rng_source = self._peer_ranks[self._broadcast_src]
        with (
            synchronized_replay_rng(
                self._group,
                source_global_rank=rng_source,
                enabled=self._synchronize_rng,
            ) as shared_rng_state,
            _replay_autocast(execution),
        ):
            if shared_rng_state is not None:
                precondition_results[REPLAY_RNG_STATE] = self._c3.run_structure(shared_rng_state)
            if execution.grad_output is not None:
                tensor_groups, elapsed_ms = self._check_forward_backward_sdc(layer, execution)
                replay_mode = "forward_backward"
            elif scale_factors:
                tensor_groups, elapsed_ms = self._check_scaled_sdc(
                    layer,
                    execution.args,
                    execution.kwargs,
                    scale_factors=scale_factors,
                )
                replay_mode = "scaled"
            else:
                tensor_groups, elapsed_ms = self._check_forward_sdc(
                    layer,
                    execution.args,
                    execution.kwargs,
                )
                replay_mode = "forward"

        evidence_preparer = getattr(self, "_evidence_preparer", None)
        if evidence_preparer is not None:
            tensor_groups = evidence_preparer(tensor_groups)
        c3_results = {
            **precondition_results,
            **self._c3.run_tensor_groups(tensor_groups),
        }
        source_bitmaps = {name: result.bitmap for name, result in c3_results.items()}
        sdc_bitmap = _merge_bitmaps(*source_bitmaps.values())
        timing_result = self._c3.run_scalar(elapsed_ms, mode=C3Mode.STATISTICAL)
        straggler_bitmap = timing_result.bitmap
        replay_times = timing_result.evidence
        straggler_bitmap = _slow_outlier_bitmap(
            straggler_bitmap,
            [float(value) for value in replay_times],
            self._straggler_min_slowdown_ratio,
            self._straggler_min_slowdown_ms,
        )
        return ReplayResult(
            sdc_bitmap=sdc_bitmap,
            straggler_bitmap=straggler_bitmap,
            replay_time_ms=elapsed_ms,
            layer_id=layer_id,
            peer_ranks=self._peer_ranks.copy(),
            replay_times_ms=[float(t) for t in replay_times],
            sdc_sources=[source for source, bitmap in source_bitmaps.items() if any(bitmap)],
            sdc_source_bitmaps=source_bitmaps,
            replay_mode=replay_mode,
            spatial_straggler_bitmap=straggler_bitmap.copy(),
            c3_results=c3_results,
            timing_c3_result=timing_result,
        )

    def replay_forward_localize(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        layer_id: int = 0,
    ) -> ReplayResult:
        """Replay, detect, then localize if straggler found."""
        result = self.replay_forward(layer, activation, layer_id=layer_id)

        if any(result.straggler_bitmap):
            result.straggler_detail = self.localize_straggler(layer, activation, layer_id=layer_id)

        return result

    def localize_straggler(
        self,
        layer: nn.Module,
        activation: torch.Tensor,
        layer_id: int = 0,
        threshold_sigma: float = 3.0,
    ) -> StragglerDetail:
        """Phase 2 straggler localization via instrumented replay.

        Re-runs the layer with CUDA event instrumentation to decompose total
        replay time into t_compute vs t_comm, then uses C3 to identify which
        rank's compute time is an outlier.
        """
        invocation = ReplayInvocation(args=(activation,), kwargs={})
        return self.localize_invocation_straggler(
            layer, invocation, layer_id=layer_id, threshold_sigma=threshold_sigma
        )

    def localize_invocation_straggler(
        self,
        layer: nn.Module,
        invocation: ReplayInvocation,
        layer_id: int = 0,
        threshold_sigma: float = 3.0,
    ) -> StragglerDetail:
        """Instrument a complete captured invocation to separate compute and communication."""
        shared = self._broadcast_invocation(invocation)
        invocation_preparer = getattr(self, "_invocation_preparer", None)
        execution = (
            invocation_preparer(layer, shared) if invocation_preparer is not None else shared
        )

        with (
            deterministic_mode(self._deterministic),
            _replay_autocast(execution),
        ):
            t_compute, t_comm, op_timings = self._instrumented_replay(
                layer,
                execution.args,
                execution.kwargs,
            )
        collective_timings = self._classify_collective_timings(op_timings)

        # Single allgather for both compute and comm times (replaces 3 allgathers)
        local = torch.tensor([t_compute, t_comm], dtype=torch.float64)
        gathered = [torch.zeros(2, dtype=torch.float64) for _ in range(self._c3._world_size)]
        dist.all_gather(gathered, local, group=self._group)
        compute_times = [g[0].item() for g in gathered]
        comm_times = [g[1].item() for g in gathered]

        compute_bitmap = self._c3.classify_evidence(
            compute_times, C3Mode.STATISTICAL, threshold_sigma
        ).bitmap
        compute_bitmap = _slow_outlier_bitmap(
            compute_bitmap,
            compute_times,
            self._straggler_min_slowdown_ratio,
            self._straggler_min_slowdown_ms,
        )
        communication_bitmap = self._c3.classify_evidence(
            comm_times, C3Mode.STATISTICAL, threshold_sigma
        ).bitmap
        communication_bitmap = _slow_outlier_bitmap(
            communication_bitmap,
            comm_times,
            self._straggler_min_slowdown_ratio,
            self._straggler_min_slowdown_ms,
        )

        compute_indices = {index for index, value in enumerate(compute_bitmap) if value}
        communication_indices = {index for index, value in enumerate(communication_bitmap) if value}
        if compute_indices and communication_indices:
            index = min(compute_indices | communication_indices)
            straggler_rank = self._peer_ranks[index]
            straggler_type = (
                "mixed"
                if index in compute_indices and index in communication_indices
                else ("compute" if index in compute_indices else "communication")
            )
        elif compute_indices:
            index = min(compute_indices)
            straggler_rank = self._peer_ranks[index]
            straggler_type = "compute"
        elif communication_indices:
            index = min(communication_indices)
            straggler_rank = self._peer_ranks[index]
            straggler_type = "communication"
        else:
            straggler_rank = None
            straggler_type = "none"

        return StragglerDetail(
            straggler_rank=straggler_rank,
            straggler_type=straggler_type,
            compute_times_ms=compute_times,
            comm_times_ms=comm_times,
            compute_bitmap=compute_bitmap,
            communication_bitmap=communication_bitmap,
            op_timings=op_timings,
            collective_timings=collective_timings,
        )

    def _classify_collective_timings(
        self,
        op_timings: list[OpTiming],
    ) -> list[CollectiveTimingSample]:
        """Compare corresponding collective timings across equivalent replay peers."""
        local_ops = [timing for timing in op_timings if timing.type == "communication"]
        if not local_ops:
            return []
        gathered: list[Any] = [None] * self._c3._world_size
        dist.all_gather_object(gathered, local_ops, group=self._group)
        peer_ops = [list(value or ()) for value in gathered]
        samples: list[CollectiveTimingSample] = []
        for index, local in enumerate(local_ops):
            if any(index >= len(items) for items in peer_ops):
                continue
            comparable = [items[index] for items in peer_ops]
            signature = _collective_timing_signature(local)
            if any(_collective_timing_signature(item) != signature for item in comparable):
                continue
            times = [float(item.time_ms) for item in comparable]
            raw = C3.classify_evidence(times, C3Mode.STATISTICAL)
            bitmap = _slow_outlier_bitmap(
                raw.bitmap,
                times,
                self._straggler_min_slowdown_ratio,
                self._straggler_min_slowdown_ms,
            )
            samples.append(
                CollectiveTimingSample(
                    collective=local.name,
                    group_ranks=local.group_ranks,
                    message_bytes=local.message_bytes,
                    sequence=local.sequence,
                    latency_ms=local.time_ms,
                    slow=bool(bitmap[self._rank]),
                )
            )
        return samples

    # ──────────────────────────────────────────────────────────────────────────
    # SDC detection (inlined from SDCLocalizer)
    # ──────────────────────────────────────────────────────────────────────────

    def _check_forward_sdc(
        self,
        layer: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[dict[str, list[torch.Tensor]], float]:
        """Replay forward pass and check for SDC via bitwise output comparison."""
        with deterministic_mode(self._deterministic):
            output, elapsed_ms = timed_call(layer, args, kwargs, self._device)

        return {"output": _tensor_leaves(output)}, elapsed_ms

    def _check_forward_backward_sdc(
        self,
        layer: nn.Module,
        invocation: ReplayInvocation,
    ) -> tuple[dict[str, list[torch.Tensor]], float]:
        """Replay forward + backward without mutating live parameter gradients."""
        input_leaves, input_spec = tree_flatten((invocation.args, invocation.kwargs))
        grad_mask = invocation.input_requires_grad or [False] * len(input_leaves)
        replay_leaves: list[Any] = []
        differentiable_inputs: list[torch.Tensor] = []
        for index, leaf in enumerate(input_leaves):
            if isinstance(leaf, torch.Tensor):
                value = leaf.clone().detach()
                wants_grad = (
                    index < len(grad_mask)
                    and grad_mask[index]
                    and (value.is_floating_point() or value.is_complex())
                )
                value.requires_grad_(wants_grad)
                if wants_grad:
                    differentiable_inputs.append(value)
                replay_leaves.append(value)
            else:
                replay_leaves.append(leaf)
        args, kwargs = tree_unflatten(replay_leaves, input_spec)

        with deterministic_mode(self._deterministic):
            torch.cuda.synchronize(self._device)
            start = time.perf_counter()

            # Capture parameter objects before forward. FSDP may swap the module's
            # registered parameters back to sharded DTensors in its post-forward hook;
            # autograd must target the unsharded leaves actually used by this graph.
            params = [p for p in layer.parameters() if p.requires_grad]
            output = layer(*args, **kwargs)

            # Use completed forward latency for spatial straggler localization.
            # FSDP backward hooks enter synchronized collectives, where one delayed
            # contributor makes every healthy peer wait and erases the rank outlier.
            torch.cuda.synchronize(self._device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

            outputs, grad_outputs = _differentiable_outputs_and_grads(
                output, invocation.grad_output
            )
            targets: list[torch.Tensor] = [*differentiable_inputs, *params]
            gradients: tuple[torch.Tensor | None, ...]
            if outputs and targets:
                gradients = torch.autograd.grad(
                    outputs,
                    targets,
                    grad_outputs=grad_outputs,
                    allow_unused=True,
                    retain_graph=False,
                )
            else:
                gradients = ()

            input_count = len(differentiable_inputs)
            all_parameter_gradients = list(gradients[input_count:])

            if self._gradient_communication is not None and all_parameter_gradients:
                self._gradient_communication(layer, all_parameter_gradients)
                # Communication still runs for trigger-equivalent failure coverage,
                # but its group-wide wait is not used for compute localization.
            torch.cuda.synchronize(self._device)

        input_gradients = [g for g in gradients[:input_count] if g is not None]
        parameter_gradients = [g for g in all_parameter_gradients if g is not None]
        return {
            "output": _tensor_leaves(output),
            "input_gradient": input_gradients,
            "parameter_gradient": parameter_gradients,
        }, elapsed_ms

    def _check_scaled_sdc(
        self,
        layer: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        scale_factors: list[float] | None = None,
    ) -> tuple[dict[str, list[torch.Tensor]], float]:
        """Replay with multiple input scales for broader SDC coverage."""
        if scale_factors is None:
            scale_factors = [0.1, 1.0, 10.0]

        with deterministic_mode(self._deterministic):
            torch.cuda.synchronize(self._device)
            start = time.perf_counter()

            outputs = []
            with torch.no_grad():
                for scale in scale_factors:
                    scaled_args, scaled_kwargs = _scale_first_tensor(args, kwargs, scale)
                    output = layer(*scaled_args, **scaled_kwargs)
                    outputs.extend(_tensor_leaves(output))

            torch.cuda.synchronize(self._device)
            elapsed_ms = (time.perf_counter() - start) * 1000.0

        return {"output": outputs}, elapsed_ms

    # ──────────────────────────────────────────────────────────────────────────
    # Straggler localization (inlined from StragglerLocalizer)
    # ──────────────────────────────────────────────────────────────────────────

    def _instrumented_replay(
        self,
        layer: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[float, float, list[OpTiming]]:
        """Replay with CUDA events at every submodule and collective boundary."""
        stream = torch.cuda.current_stream(self._device)
        submodule_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
        comm_events: list[_CollectiveEvent] = []

        hooks = self._attach_submodule_timing_hooks(layer, stream, submodule_events)
        saved_ops = self._patch_comm_ops(stream, comm_events)

        evt_start = torch.cuda.Event(enable_timing=True)
        evt_end = torch.cuda.Event(enable_timing=True)

        try:
            torch.cuda.synchronize(self._device)
            evt_start.record(stream)

            with torch.no_grad():
                _ = layer(*args, **kwargs)

            evt_end.record(stream)
            torch.cuda.synchronize(self._device)
        finally:
            self._restore_comm_ops(saved_ops)
            for h in hooks:
                h.remove()

        return self._compute_timing_breakdown(evt_start, evt_end, submodule_events, comm_events)

    def _attach_submodule_timing_hooks(
        self,
        layer: nn.Module,
        stream: torch.cuda.Stream,
        events_out: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
    ) -> list:
        """Register pre/post hooks on immediate children to record CUDA timing events."""
        hooks = []

        def make_pre(name):
            def hook(module, input):
                evt = torch.cuda.Event(enable_timing=True)
                evt.record(stream)
                module._replay_start_evt = evt

            return hook

        def make_post(name):
            def hook(module, input, output):
                end_evt = torch.cuda.Event(enable_timing=True)
                end_evt.record(stream)
                start_evt = getattr(module, "_replay_start_evt", None)
                if start_evt is not None:
                    events_out.append((name, start_evt, end_evt))
                    del module._replay_start_evt

            return hook

        for name, child in layer.named_children():
            hooks.append(child.register_forward_pre_hook(make_pre(name)))
            hooks.append(child.register_forward_hook(make_post(name)))

        return hooks

    def _patch_comm_ops(
        self,
        stream: torch.cuda.Stream,
        events_out: list[_CollectiveEvent],
    ) -> dict[str, object]:
        """Monkey-patch dist collective ops with timing wrappers. Returns originals."""
        saved = {}

        def wrap(orig_fn, op_name):
            def wrapped(*args, **kwargs):
                bound = dict(zip(_COMM_PARAMETERS[op_name], args))
                bound.update(kwargs)
                group_ranks = _process_group_ranks(bound.get("group"))
                message_bytes = _collective_message_bytes(
                    op_name,
                    bound,
                )
                start_evt = torch.cuda.Event(enable_timing=True)
                end_evt = torch.cuda.Event(enable_timing=True)
                start_evt.record(stream)
                result = orig_fn(*args, **kwargs)
                end_evt.record(stream)
                events_out.append(
                    _CollectiveEvent(
                        name=op_name,
                        group_ranks=group_ranks,
                        message_bytes=message_bytes,
                        sequence=len(events_out),
                        start=start_evt,
                        end=end_evt,
                    )
                )
                return result

            return wrapped

        for op_name in _COMM_OPS:
            orig = getattr(dist, op_name, None)
            if orig is not None:
                saved[op_name] = orig
                setattr(dist, op_name, wrap(orig, op_name))

        return saved

    def _restore_comm_ops(self, saved: dict[str, object]) -> None:
        """Restore original dist collective functions."""
        for op_name, orig_fn in saved.items():
            setattr(dist, op_name, orig_fn)

    def _compute_timing_breakdown(
        self,
        evt_start: torch.cuda.Event,
        evt_end: torch.cuda.Event,
        submodule_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]],
        comm_events: list[_CollectiveEvent],
    ) -> tuple[float, float, list[OpTiming]]:
        """Compute t_compute, t_comm, and per-op timings from recorded CUDA events."""
        t_total = evt_start.elapsed_time(evt_end)
        op_timings: list[OpTiming] = []

        t_comm = 0.0
        for event in comm_events:
            t = event.start.elapsed_time(event.end)
            t_comm += t
            op_timings.append(
                OpTiming(
                    name=event.name,
                    type="communication",
                    time_ms=t,
                    group_ranks=event.group_ranks,
                    message_bytes=event.message_bytes,
                    sequence=event.sequence,
                )
            )

        for name, s, e in submodule_events:
            op_timings.append(OpTiming(name=name, type="compute", time_ms=s.elapsed_time(e)))

        t_compute = t_total - t_comm
        return t_compute, t_comm, op_timings

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _broadcast_activation(self, tensor: torch.Tensor) -> torch.Tensor:
        """Broadcast activation from broadcast_src to all ranks in the group."""
        group = self._nccl_group or self._group
        metadata = [(tuple(tensor.shape), tensor.dtype)]
        metadata_src = self._peer_ranks[self._broadcast_src]
        dist.broadcast_object_list(metadata, src=metadata_src, group=self._group)
        shape, dtype = metadata[0]
        if self._rank == self._broadcast_src:
            t = tensor.detach().to(device=self._device, copy=True).contiguous()
        else:
            t = torch.empty(shape, dtype=dtype, device=self._device)
        global_src = dist.get_global_rank(group, self._broadcast_src)
        dist.broadcast(t, src=global_src, group=group)
        return t

    def _broadcast_invocation(self, invocation: ReplayInvocation) -> ReplayInvocation:
        """Broadcast every tensor leaf while retaining the captured call structure."""
        combined = (invocation.args, invocation.kwargs, invocation.grad_output)
        leaves, spec = tree_flatten(combined)
        local_tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
        metadata: list[Any] = [
            {
                "tensors": [(tuple(tensor.shape), tensor.dtype) for tensor in local_tensors],
                "input_requires_grad": list(invocation.input_requires_grad),
                "autocast_enabled": invocation.autocast_enabled,
                "autocast_device_type": invocation.autocast_device_type,
                "autocast_dtype": invocation.autocast_dtype,
            }
        ]
        metadata_src = self._peer_ranks[self._broadcast_src]
        dist.broadcast_object_list(metadata, src=metadata_src, group=self._group)
        source_tensors = metadata[0]["tensors"]
        if len(source_tensors) != len(local_tensors):
            raise RuntimeError(
                "Replay invocation tensor structure differs across peers: "
                f"source has {len(source_tensors)}, local rank has {len(local_tensors)}"
            )

        tensor_index = 0
        shared_leaves = []
        tensor_group = self._nccl_group or self._group
        tensor_src = self._peer_ranks[self._broadcast_src]
        for leaf in leaves:
            if not isinstance(leaf, torch.Tensor):
                shared_leaves.append(leaf)
                continue
            shape, dtype = source_tensors[tensor_index]
            tensor_index += 1
            if self._rank == self._broadcast_src:
                shared = leaf.detach().to(device=self._device, copy=True).contiguous()
            else:
                shared = torch.empty(shape, dtype=dtype, device=self._device)
            dist.broadcast(shared, src=tensor_src, group=tensor_group)
            shared_leaves.append(shared)

        args, kwargs, grad_output = tree_unflatten(shared_leaves, spec)
        return ReplayInvocation(
            args=args,
            kwargs=kwargs,
            input_requires_grad=list(metadata[0]["input_requires_grad"]),
            grad_output=grad_output,
            autocast_enabled=bool(metadata[0]["autocast_enabled"]),
            autocast_device_type=str(metadata[0]["autocast_device_type"]),
            autocast_dtype=metadata[0]["autocast_dtype"],
        )

    def _broadcast_structure(self, value: Any | None) -> Any:
        """Broadcast a source-owned nested value with CPU or CUDA tensor leaves."""
        metadata_src = self._peer_ranks[self._broadcast_src]
        source_tensors: list[torch.Tensor] = []
        metadata: list[Any] = [None]
        if self._rank == self._broadcast_src:
            if value is None:
                raise RuntimeError("optimizer replay source has no captured payload")
            template = _encode_broadcast_structure(value, source_tensors)
            metadata[0] = (
                template,
                [
                    _BroadcastTensorSpec(
                        index=index,
                        shape=tuple(tensor.shape),
                        dtype=tensor.dtype,
                        device_type=tensor.device.type,
                    )
                    for index, tensor in enumerate(source_tensors)
                ],
            )
        dist.broadcast_object_list(metadata, src=metadata_src, group=self._group)
        template, tensor_specs = metadata[0]

        received_tensors: list[torch.Tensor] = []
        for tensor_spec in tensor_specs:
            tensor_group = self._group
            if tensor_spec.device_type == "cuda":
                tensor_group = self._nccl_group or self._group
                device = self._device
            else:
                device = torch.device("cpu")
            if self._rank == self._broadcast_src:
                tensor = (
                    source_tensors[tensor_spec.index]
                    .detach()
                    .to(device=device, copy=True)
                    .contiguous()
                )
            else:
                tensor = torch.empty(
                    tensor_spec.shape,
                    dtype=tensor_spec.dtype,
                    device=device,
                )
            dist.broadcast(tensor, src=metadata_src, group=tensor_group)
            received_tensors.append(tensor)
        return _decode_broadcast_structure(template, received_tensors)


def _tensor_leaves(value: Any, *, detach: bool = True) -> list[torch.Tensor]:
    leaves, _ = tree_flatten(value)
    tensors = [leaf for leaf in leaves if isinstance(leaf, torch.Tensor)]
    return [tensor.detach() for tensor in tensors] if detach else tensors


def _replay_autocast(invocation: ReplayInvocation):
    """Restore the mixed-precision context captured at the module boundary."""
    if not invocation.autocast_enabled:
        return nullcontext()
    return torch.autocast(
        device_type=invocation.autocast_device_type,
        dtype=invocation.autocast_dtype,
        enabled=True,
    )


def _encode_broadcast_structure(
    value: Any,
    tensors: list[torch.Tensor],
) -> Any:
    if isinstance(value, torch.Tensor):
        index = len(tensors)
        tensors.append(value)
        return _BroadcastTensorSpec(
            index=index,
            shape=tuple(value.shape),
            dtype=value.dtype,
            device_type=value.device.type,
        )
    if isinstance(value, dict):
        return {key: _encode_broadcast_structure(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_encode_broadcast_structure(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_encode_broadcast_structure(item, tensors) for item in value)
    return value


def _decode_broadcast_structure(
    value: Any,
    tensors: Sequence[torch.Tensor],
) -> Any:
    if isinstance(value, _BroadcastTensorSpec):
        return tensors[value.index]
    if isinstance(value, dict):
        return {key: _decode_broadcast_structure(item, tensors) for key, item in value.items()}
    if isinstance(value, list):
        return [_decode_broadcast_structure(item, tensors) for item in value]
    if isinstance(value, tuple):
        return tuple(_decode_broadcast_structure(item, tensors) for item in value)
    return value


def _differentiable_outputs_and_grads(
    output: Any,
    grad_output: Any,
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """Align a backward hook's structured gradients with differentiable outputs."""
    output_leaves, output_spec = tree_flatten(output)
    grad_leaves, grad_spec = tree_flatten(grad_output)
    all_outputs = [tensor for tensor in output_leaves if isinstance(tensor, torch.Tensor)]

    if output_spec == grad_spec:
        pairs = (
            (tensor, grad_leaves[index])
            for index, tensor in enumerate(output_leaves)
            if isinstance(tensor, torch.Tensor)
        )
    elif len(all_outputs) == len(grad_leaves):
        pairs = zip(all_outputs, grad_leaves)
    elif len(all_outputs) == 1 and grad_leaves:
        gradient = next(
            (leaf for leaf in grad_leaves if isinstance(leaf, torch.Tensor)),
            None,
        )
        pairs = [(all_outputs[0], gradient)]
    else:
        pairs = [(tensor, None) for tensor in all_outputs]

    outputs: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    for tensor, gradient in pairs:
        if not tensor.requires_grad:
            continue
        outputs.append(tensor)
        if isinstance(gradient, torch.Tensor) and gradient.shape == tensor.shape:
            gradients.append(gradient)
        else:
            gradients.append(torch.ones_like(tensor))
    return outputs, gradients


def _scale_first_tensor(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    scale: float,
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    leaves, spec = tree_flatten((args, kwargs))
    scaled = []
    found = False
    for leaf in leaves:
        if not found and isinstance(leaf, torch.Tensor) and leaf.is_floating_point():
            scaled.append(leaf * scale)
            found = True
        else:
            scaled.append(leaf)
    return tree_unflatten(scaled, spec)


def _process_group_ranks(group: Any) -> tuple[int, ...]:
    if not dist.is_available() or not dist.is_initialized():
        return (0,)
    if group is None:
        return tuple(range(dist.get_world_size()))
    try:
        return tuple(int(rank) for rank in dist.get_process_group_ranks(group))
    except (RuntimeError, ValueError):
        return ()


def _collective_message_bytes(name: str, bound: dict[str, Any]) -> int:
    return sum(_tensor_bytes(bound.get(argument)) for argument in _COMM_INPUT_NAMES.get(name, ()))


def _tensor_bytes(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _collective_timing_signature(
    timing: OpTiming,
) -> tuple[str, int, int, int]:
    return (
        timing.name,
        len(timing.group_ranks),
        timing.message_bytes,
        timing.sequence,
    )


def _merge_bitmaps(*bitmaps: Sequence[int]) -> list[int]:
    if not bitmaps:
        return []
    return [int(any(values)) for values in zip(*bitmaps)]


def _refresh_straggler_detail(
    detail: StragglerDetail,
    peer_ranks: Sequence[int],
) -> None:
    compute = {index for index, value in enumerate(detail.compute_bitmap) if value}
    communication = {index for index, value in enumerate(detail.communication_bitmap) if value}
    affected = compute | communication
    if not affected:
        detail.straggler_rank = None
        detail.straggler_type = "none"
        return
    index = min(affected)
    detail.straggler_rank = peer_ranks[index] if index < len(peer_ranks) else index
    if index in compute and index in communication:
        detail.straggler_type = "mixed"
    elif index in communication:
        detail.straggler_type = "communication"
    else:
        detail.straggler_type = "compute"


def _optimizer_result_name(base: str, index: int, count: int) -> str:
    return base if count == 1 else f"{base}.{index}"


def _expected_status_result(
    result: C3Result,
    *,
    expected: int,
) -> C3Result:
    bitmap = [int(int(value) != expected) for value in result.evidence]
    if not any(bitmap):
        status = C3Status.AGREE
    elif all(bitmap):
        status = C3Status.INCONCLUSIVE
        bitmap = [0] * len(bitmap)
    else:
        status = C3Status.ATTRIBUTED
    return C3Result(status, bitmap, result.evidence)


def _slow_outlier_bitmap(
    bitmap: Sequence[int],
    values: Sequence[float],
    min_slowdown_ratio: float,
    min_slowdown_ms: float,
) -> list[int]:
    if not values:
        return list(bitmap)
    ordered = sorted(float(value) for value in values)
    center = ordered[len(ordered) // 2]
    threshold = max(
        center * min_slowdown_ratio,
        center + min_slowdown_ms,
    )
    return [
        int(bool(flagged) and float(value) > threshold) for flagged, value in zip(bitmap, values)
    ]
