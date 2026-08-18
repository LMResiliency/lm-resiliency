"""Manager-side coordination for fixed-size torchrun recovery."""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Sequence
from dataclasses import dataclass

from torch.distributed import Store

from ._protocol import RestartPlan, SlotAssignment
from ._simple_runtime import (
    _PLAN_LEASE_RENEW_INTERVAL_SECONDS,
    SimpleRecoveryPlanStore,
    _node_id_from_machine_id,
)

_PLAN_LEASE_MAX_CONSECUTIVE_FAILURES = 3


@dataclass(frozen=True, slots=True)
class TorchrunInitialPlacement:
    """Committed active nodes and registered standbys for generation zero."""

    active_node_ids: tuple[str, ...]
    standby_node_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TorchrunRecoveryRequest:
    """Manager-owned checkpoint and failure decision for one restart."""

    plan_id: str
    intent_id: str
    reason_code: str
    recovery_mode: str
    checkpoint_source: str
    checkpoint_step: int
    checkpoint_manifest_id: str
    topology_digest: str
    restart_deadline_unix_ms: int
    checkpoint_id: str | None = None


@dataclass(frozen=True, slots=True)
class TorchrunSuccessorPlacement:
    """Committed node placement for one successor generation."""

    generation: int
    active_node_ids: tuple[str, ...]
    quarantined_node_ids: tuple[str, ...]


class TorchrunRecoveryCoordinator:
    """Publish fixed-size successor generations through a c10d store."""

    def __init__(self, store: Store, *, run_id: str) -> None:
        self._run_id = run_id
        self._plans = SimpleRecoveryPlanStore(store, run_id=run_id)
        self._lease_lock = threading.Lock()
        self._lease_stop: threading.Event | None = None
        self._lease_thread: threading.Thread | None = None
        self._lease_error: Exception | None = None

    @property
    def current_generation(self) -> int:
        """Return the latest committed manager generation."""

        self.check_health()
        return self._plans.current_generation()

    def check_health(self) -> None:
        """Raise if background recovery-plan lease renewal failed persistently."""

        with self._lease_lock:
            error = self._lease_error
        if error is not None:
            raise RuntimeError("recovery plan lease renewal failed") from error

    def initial_placement(
        self,
        *,
        active_node_count: int,
        allocated_node_count: int,
    ) -> TorchrunInitialPlacement | None:
        """Return generation-zero placement after the complete fleet registers."""

        self.check_health()
        if isinstance(active_node_count, bool) or not isinstance(active_node_count, int):
            raise TypeError("active_node_count must be an integer")
        if isinstance(allocated_node_count, bool) or not isinstance(allocated_node_count, int):
            raise TypeError("allocated_node_count must be an integer")
        if active_node_count < 1:
            raise ValueError("active_node_count must be positive")
        if allocated_node_count < active_node_count:
            raise ValueError("allocated_node_count must be at least active_node_count")
        active = self._plans.read_initial_nodes()
        if active is None:
            return None
        registered = self._plans.registered_nodes(max_nodes=allocated_node_count)
        if len(registered) < allocated_node_count:
            return None
        if len(active) != active_node_count:
            raise RuntimeError("committed initial placement has the wrong active-node count")
        active_set = set(active)
        standbys = tuple(node_id for node_id in registered if node_id not in active_set)
        if len(standbys) != allocated_node_count - active_node_count:
            raise RuntimeError("committed initial placement has the wrong standby count")
        return TorchrunInitialPlacement(active, standbys)

    def publish_successor(
        self,
        *,
        generation: int,
        active_node_ids: Sequence[str],
        quarantined_node_ids: Sequence[str],
        request: TorchrunRecoveryRequest,
        local_world_size: int,
        replacement: tuple[int, str] | None = None,
    ) -> TorchrunSuccessorPlacement:
        """Commit one same-node restart or logical-slot replacement."""

        self.check_health()
        if not isinstance(request, TorchrunRecoveryRequest):
            raise TypeError("request must be TorchrunRecoveryRequest")
        if isinstance(generation, bool) or not isinstance(generation, int):
            raise TypeError("generation must be an integer")
        if generation < 0:
            raise ValueError("generation must be non-negative")
        if isinstance(local_world_size, bool) or not isinstance(local_world_size, int):
            raise TypeError("local_world_size must be an integer")
        if local_world_size < 1:
            raise ValueError("local_world_size must be positive")
        active = list(active_node_ids)
        if not active or len(active) != len(set(active)):
            raise ValueError("active_node_ids must be non-empty and unique")
        quarantined = list(quarantined_node_ids)
        if len(quarantined) != len(set(quarantined)):
            raise ValueError("quarantined_node_ids must be unique")
        if set(active) & set(quarantined):
            raise ValueError("active and quarantined node IDs must be disjoint")
        current_generation = self._plans.current_generation()
        if current_generation != generation:
            raise RuntimeError(
                f"manager generation {generation} is stale; "
                f"committed generation is {current_generation}"
            )
        if generation == 0:
            committed_active = self._plans.read_initial_nodes()
            if committed_active is None:
                raise RuntimeError("generation-zero placement is not committed")
            committed_quarantined: tuple[str, ...] = ()
        else:
            committed_plan = self._plans.read(generation)
            if committed_plan is None:
                raise RuntimeError(f"generation {generation} has no committed recovery plan")
            committed_active = tuple(
                assignment.node_id for assignment in committed_plan.slot_assignments
            )
            committed_quarantined = committed_plan.quarantined_node_ids
            committed_local_world_size = committed_plan.slot_assignments[0].local_world_size
            if local_world_size != committed_local_world_size:
                raise RuntimeError("local_world_size does not match the committed worker topology")
            if request.topology_digest != committed_plan.topology_digest:
                raise RuntimeError(
                    "recovery request topology_digest does not match the committed topology"
                )
        if tuple(active) != committed_active:
            raise RuntimeError("active_node_ids do not match the committed placement")
        if tuple(quarantined) != committed_quarantined:
            raise RuntimeError("quarantined_node_ids do not match the committed quarantine history")
        remaining_seconds = (request.restart_deadline_unix_ms - time.time_ns() // 1_000_000) / 1_000
        if remaining_seconds <= 0:
            raise ValueError("restart deadline must be in the future")
        if replacement is not None:
            slot, replacement_node_id = replacement
            if isinstance(slot, bool) or not isinstance(slot, int):
                raise TypeError("replacement slot must be an integer")
            if slot < 0 or slot >= len(active):
                raise ValueError("replacement slot is outside the active placement")
            if not isinstance(replacement_node_id, str) or not replacement_node_id.strip():
                raise ValueError("replacement node ID must be non-empty")
            failed_node_id = active[slot]
            if replacement_node_id in active or replacement_node_id in quarantined:
                raise ValueError("replacement node is already active or quarantined")
            registered = set(self._plans.registered_nodes())
            if replacement_node_id not in registered:
                raise ValueError("replacement node is not registered in this torchrun allocation")
            active[slot] = replacement_node_id
            quarantined.append(failed_node_id)
        plan = RestartPlan(
            plan_id=request.plan_id,
            intent_id=request.intent_id,
            run_id=self._run_id,
            from_generation=generation,
            to_generation=generation + 1,
            incident_ids=(request.intent_id,),
            reason_code=request.reason_code,
            recovery_mode=request.recovery_mode,
            checkpoint_source=request.checkpoint_source,
            checkpoint_step=request.checkpoint_step,
            checkpoint_id=request.checkpoint_id,
            checkpoint_manifest_id=request.checkpoint_manifest_id,
            slot_assignments=tuple(
                SlotAssignment(
                    logical_node_slot=slot,
                    node_id=node_id,
                    first_global_rank=slot * local_world_size,
                    local_world_size=local_world_size,
                )
                for slot, node_id in enumerate(active)
            ),
            quarantined_node_ids=tuple(quarantined),
            expected_world_size=len(active) * local_world_size,
            topology_digest=request.topology_digest,
            restart_deadline_unix_ms=request.restart_deadline_unix_ms,
        )
        self._plans.publish(plan)
        self._start_plan_lease(plan, remaining_seconds=remaining_seconds)
        return TorchrunSuccessorPlacement(
            generation=plan.to_generation,
            active_node_ids=tuple(active),
            quarantined_node_ids=tuple(quarantined),
        )

    def close(self) -> None:
        """Wake parked agents and prevent further rendezvous."""

        self._stop_plan_lease()
        self._plans.close_run()
        self.check_health()

    def _start_plan_lease(
        self,
        plan: RestartPlan,
        *,
        remaining_seconds: float,
    ) -> None:
        self._stop_plan_lease()
        stop = threading.Event()
        deadline = time.monotonic() + remaining_seconds
        plans = self._plans

        def publish_remaining(remaining: float) -> None:
            plans.renew_plan_lease(
                plan,
                remaining_ms=0 if remaining <= 0 else math.ceil(remaining * 1_000),
            )

        initial_failures = 0
        try:
            publish_remaining(remaining_seconds)
        except Exception:
            # The plan is already committed and cannot be rolled back. Keep the
            # coordinator alive so the renewal thread can retry this first lease
            # publication instead of stranding the successor generation.
            initial_failures = 1

        def renew() -> None:
            consecutive_failures = initial_failures
            wait_before_publish = initial_failures == 0
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    try:
                        publish_remaining(0)
                    except Exception:
                        pass
                    return
                if wait_before_publish and stop.wait(
                    min(_PLAN_LEASE_RENEW_INTERVAL_SECONDS, remaining)
                ):
                    return
                wait_before_publish = True
                try:
                    publish_remaining(deadline - time.monotonic())
                    consecutive_failures = 0
                except Exception as error:
                    consecutive_failures += 1
                    if consecutive_failures >= _PLAN_LEASE_MAX_CONSECUTIVE_FAILURES:
                        with self._lease_lock:
                            if self._lease_stop is stop and not stop.is_set():
                                self._lease_error = error

        thread = threading.Thread(
            target=renew,
            name=f"lm-resiliency-plan-lease-{plan.to_generation}",
            daemon=True,
        )
        with self._lease_lock:
            self._lease_stop = stop
            self._lease_thread = thread
            self._lease_error = None
        thread.start()

    def _stop_plan_lease(self) -> None:
        with self._lease_lock:
            stop = self._lease_stop
            thread = self._lease_thread
            self._lease_stop = None
            self._lease_thread = None
        if stop is not None:
            stop.set()
        if thread is not None:
            thread.join(timeout=max(_PLAN_LEASE_RENEW_INTERVAL_SECONDS * 2, 0.1))


def derive_torchrun_node_id(machine_id: str) -> str:
    """Return the domain-separated node ID used by torchrun admission."""

    return _node_id_from_machine_id(machine_id)


__all__ = [
    "TorchrunInitialPlacement",
    "TorchrunRecoveryCoordinator",
    "TorchrunRecoveryRequest",
    "TorchrunSuccessorPlacement",
    "derive_torchrun_node_id",
]
