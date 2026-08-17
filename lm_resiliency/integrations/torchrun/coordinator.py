"""Manager-side coordination for fixed-size torchrun recovery."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch.distributed import Store

from ._protocol import RestartPlan, SlotAssignment
from ._simple_runtime import SimpleRecoveryPlanStore, _node_id_from_machine_id


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

    @property
    def current_generation(self) -> int:
        """Return the latest committed manager generation."""

        return self._plans.current_generation()

    def initial_placement(
        self,
        *,
        active_node_count: int,
        allocated_node_count: int,
    ) -> TorchrunInitialPlacement | None:
        """Return generation-zero placement after the complete fleet registers."""

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
        return TorchrunSuccessorPlacement(
            generation=plan.to_generation,
            active_node_ids=tuple(active),
            quarantined_node_ids=tuple(quarantined),
        )

    def close(self) -> None:
        """Wake parked agents and prevent further rendezvous."""

        self._plans.close_run()


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
