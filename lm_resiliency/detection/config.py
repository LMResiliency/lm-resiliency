"""Configuration objects for SCOUT detection."""

from __future__ import annotations

from dataclasses import dataclass, field

from lm_resiliency.detection.all_to_all_replay import (
    AllToAllReplayPolicy,
    BalancedAndPermutationPolicy,
)
from lm_resiliency.detection.replay_shapes import ReplayWorkload


@dataclass
class ReplayHarnessConfig:
    """Configuration for model replay and OOB hang detection.

    Args:
        layer_index: Repeated hidden layer to monitor initially.
        check_interval: Detection cadence in optimizer steps. Zero is manual only.
        broadcast_src: Rank that provides the reference activation for replay.
        deterministic: Enforce deterministic algorithms during replay.
        synchronize_rng: Use the broadcast source's RNG state during replay and
            restore every rank's training RNG state afterward.
        capture_inputs_by_value: Own a clone of captured replay inputs and backward
            signals on detection steps. This is intended for MoE post-dispatch
            buffers whose storage may be reused before replay.
        workload: Replay modules and ordered replay-shape plan. Dense replay uses
            one captured identity shape by default. Dynamic-shape workloads provide
            a materializer and multiple concrete shapes, commonly from a qualified
            MoE execution-regime catalog.
        compare_parameter_state: Compare the sampled layer's exact parameter
            state across replica-compatible peers. Disable this when peers own
            different parameter shards without corresponding replicas.
        scale_factors: Optional input scales for broader SDC coverage.
        rotate_layers: Move capture hooks to the next layer after each check.
        embedding_check_interval: Embedding recipe cadence in optimizer steps.
            ``None`` inherits ``check_interval`` and zero disables the recipe.
        hidden_check_interval: Hidden-layer recipe cadence in optimizer steps.
            ``None`` inherits ``check_interval`` and zero disables the recipe.
        output_check_interval: Language-model output recipe cadence in optimizer
            steps. ``None`` inherits ``check_interval`` and zero disables it.
        optimizer_check_interval: Optimizer recipe cadence in optimizer steps.
            ``None`` inherits ``check_interval`` and zero disables it.
        all_to_all_policy: Policy used to generate representative AllToAll
            traffic matrices. ``None`` disables AllToAll replay.
        straggler_confirmation_rounds: Matching rounds required before reporting.
        straggler_min_slowdown_ms: Minimum absolute excess over the peer median.
            This rejects sub-millisecond launch and synchronization jitter even when
            the relative slowdown threshold is crossed by a short replay.
        enable_temporal: Compare timings with bounded clean-round history.
        hang_stall_threshold_s: Time without progress before OOB hang reporting.
        dataloader_latency_threshold_s: Minimum sampled ``next()`` latency
            before OOB comparison can report a rank.
        dataloader_min_slowdown_ratio: Minimum slowdown versus the peer median.
        dataloader_confirmation_rounds: Consecutive OOB observations required.
        checkpoint_io_latency_threshold_s: Minimum checkpoint read or write
            latency before OOB comparison can report a rank.
        checkpoint_io_min_slowdown_ratio: Minimum checkpoint I/O slowdown
            versus the peer median.
        checkpoint_io_confirmation_rounds: Consecutive OOB observations required.
        hang_state_dir: Optional directory for OOB status and, when no TCP
            endpoint is configured, file rendezvous.
        hang_master_addr: Optional OOB TCP rendezvous address. An explicit TCP
            endpoint takes precedence over file rendezvous.
        hang_master_port: Optional OOB TCP rendezvous port. An explicit TCP
            endpoint takes precedence over file rendezvous.
    """

    layer_index: int = 0
    check_interval: int = 50
    broadcast_src: int = 0
    deterministic: bool = True
    synchronize_rng: bool = True
    capture_inputs_by_value: bool = False
    workload: ReplayWorkload | None = None
    compare_parameter_state: bool = True
    scale_factors: list[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    rotate_layers: bool = True
    embedding_check_interval: int | None = None
    hidden_check_interval: int | None = None
    output_check_interval: int | None = None
    optimizer_check_interval: int | None = None
    all_to_all_policy: AllToAllReplayPolicy | None = field(
        default_factory=BalancedAndPermutationPolicy
    )
    straggler_confirmation_rounds: int = 2
    straggler_min_slowdown_ratio: float = 1.1
    straggler_min_slowdown_ms: float = 2.0
    enable_temporal: bool = True
    temporal_window_size: int = 32
    temporal_min_samples: int = 5
    temporal_slowdown_ratio: float = 1.25
    temporal_threshold_sigma: float = 4.0
    hang_stall_threshold_s: float = 30.0
    hang_confirmation_interval_s: float = 1.0
    dataloader_latency_threshold_s: float = 5.0
    dataloader_min_slowdown_ratio: float = 2.0
    dataloader_confirmation_rounds: int = 2
    checkpoint_io_latency_threshold_s: float = 30.0
    checkpoint_io_min_slowdown_ratio: float = 2.0
    checkpoint_io_confirmation_rounds: int = 2
    hang_state_dir: str | None = None
    hang_master_addr: str | None = None
    hang_master_port: int | None = None
