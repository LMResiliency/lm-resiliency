"""Shared lifecycle helpers for framework integrations."""

from __future__ import annotations

import logging
from typing import Any, Callable

import torch.distributed as dist

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointConfig,
    DurableCheckpointCoordinator,
)
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.checkpointing.rng import RNG_KEY, capture_rng_state, restore_rng_state
from lm_resiliency.detection.layer_replay import ReplayResult, replay_result_has_fault
from lm_resiliency.detection.temporal import SCOUT_TEMPORAL_KEY

_UNSET = object()


def build_checkpoint_manager(
    config: InMemoryCkptConfig | None,
    *,
    manager_factory: Callable[..., Any],
    parallelism_info: Any = _UNSET,
    parallelism_info_fn: Callable[[], Any] | None = None,
    process_group: Any = _UNSET,
) -> tuple[Any | None, int]:
    """Create an enabled GEMINI checkpoint manager."""
    if config is None or not config.enable or config.interval <= 0:
        return None, 0
    if parallelism_info_fn is not None:
        parallelism_info = parallelism_info_fn()
    manager_kwargs = {"config": config}
    if parallelism_info is not _UNSET:
        manager_kwargs["parallelism_info"] = parallelism_info
    if process_group is not _UNSET:
        manager_kwargs["process_group"] = process_group
    return manager_factory(**manager_kwargs), config.interval


def build_durable_checkpoint(
    config: DurableCheckpointConfig | None,
    replay_harness: Any | None,
) -> DurableCheckpointCoordinator | None:
    """Bind framework checkpoint callbacks to the active SCOUT shape plan."""
    if config is None:
        return None
    if replay_harness is None:
        raise ValueError("durable checkpoint certification requires SCOUT replay detection")
    return DurableCheckpointCoordinator(
        config,
        shape_plan_id=replay_harness.replay_shape_plan_id,
        shape_ids=[shape.shape_id for shape in replay_harness.replay_shapes],
        checkpoint_io=lambda operation, name: replay_harness.checkpoint_io(
            operation,
            name=name,
        ),
    )


def create_gloo_peer_group(
    process_group: dist.ProcessGroup | None,
) -> dist.ProcessGroup | None:
    """Create a Gloo group over the same ranks as a framework DP group."""
    if process_group is None:
        return None
    ranks = tuple(dist.get_process_group_ranks(process_group))
    memberships: list[tuple[int, ...] | None] = [None] * dist.get_world_size()
    dist.all_gather_object(memberships, ranks)
    unique_memberships = sorted({membership for membership in memberships if membership})
    local_rank = dist.get_rank()
    local_group = None
    for membership in unique_memberships:
        group = dist.new_group(ranks=list(membership), backend="gloo")
        if local_rank in membership:
            local_group = group
    if local_group is None:
        raise RuntimeError("current rank is absent from every framework peer group")
    return local_group


def checkpoint_extra(
    replay_harness: Any | None,
    extra_state_fn: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Capture common RNG and SCOUT temporal checkpoint metadata."""
    extra = {RNG_KEY: capture_rng_state()}
    if extra_state_fn is not None:
        extra.update(extra_state_fn())
    if replay_harness is not None:
        extra[SCOUT_TEMPORAL_KEY] = replay_harness.temporal_state_dict()
    return extra


def restore_checkpoint_extra(
    extra: dict[str, Any] | None,
    replay_harness: Any | None,
    load_extra_state_fn: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    """Restore replay baselines, then RNG immediately before training resumes."""
    values = extra or {}
    if replay_harness is not None:
        replay_harness.load_temporal_state_dict(values.get(SCOUT_TEMPORAL_KEY))
    if load_extra_state_fn is not None:
        load_extra_state_fn(
            {
                key: value
                for key, value in values.items()
                if key not in (RNG_KEY, SCOUT_TEMPORAL_KEY)
            }
        )
    restore_rng_state(values.get(RNG_KEY))


def prepare_checkpoint_tensor_load(adapter: Any, saved_tensors: list[Any]) -> None:
    """Materialize lazy optimizer state only when the checkpoint contains it."""
    live_count = len(adapter.collect_checkpoint_tensors())
    if len(saved_tensors) > live_count:
        adapter.materialize_optimizer_state()


def report_replay_result(
    result: ReplayResult | None,
    callback: Callable[[ReplayResult], None] | None,
    *,
    step: int,
    log: logging.Logger,
) -> None:
    """Dispatch a replay fault through the configured integration callback."""
    if not replay_result_has_fault(result):
        return
    if callback is not None:
        callback(result)
        return
    log.warning(
        "SCOUT fault at step %s: sdc=%s, straggler=%s",
        step,
        result.sdc_bitmap,
        result.straggler_bitmap,
    )


def recover_with_fallback(
    resiliency: Any,
    load_fallback: Callable[[], int | None] | None,
    recovery_mode: RecoveryMode | str | None = None,
) -> None:
    """Try GEMINI first and apply a framework fallback resume step."""
    recovered = resiliency.try_recover(mode=recovery_mode)
    if recovered >= 0:
        return
    durable = getattr(
        resiliency,
        "durable_checkpoint",
        getattr(resiliency, "_durable_checkpoint", None),
    )
    if durable is not None:
        durable_step = durable.load_latest_validated()
        if durable_step is not None and durable_step > 0:
            resiliency.step_count = durable_step
        # Never invoke an unconstrained latest-checkpoint fallback while the
        # SCOUT manifest protocol is active.
        return
    if load_fallback is None:
        return
    with resiliency.checkpoint_io("read", name="fallback"):
        fallback_step = load_fallback()
    if fallback_step is not None and fallback_step > 0:
        resiliency.step_count = fallback_step
