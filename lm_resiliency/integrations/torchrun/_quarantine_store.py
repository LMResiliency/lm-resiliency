"""Fail-closed quarantine reads and transaction writes for torchrun replacement."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
    ControlStoreWrite,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._protocol import RestartIntent, RestartPlan
from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class QuarantineStateError(RuntimeError):
    """Base error for persisted node-quarantine state."""


class QuarantineStateCorrupt(QuarantineStateError):
    """Raised when persisted quarantine state is malformed or contradictory."""


class QuarantineLeaseLost(QuarantineStateError):
    """Raised when quarantine writes use a stale or foreign coordinator lease."""


@dataclass(frozen=True, slots=True)
class StoredNodeQuarantine:
    """One verified permanent node quarantine and its store provenance."""

    record: NodeQuarantineRecord
    entry: ControlStoreEntry

    def __post_init__(self) -> None:
        if not isinstance(self.record, NodeQuarantineRecord):
            raise TypeError("StoredNodeQuarantine.record must be NodeQuarantineRecord")
        if not isinstance(self.entry, ControlStoreEntry):
            raise TypeError("StoredNodeQuarantine.entry must be ControlStoreEntry")


class NodeQuarantineRepository:
    """Prepare plan-authorized writes and read permanent node exclusions."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._coordinator_lease_key = f"{self._run_prefix}/coordinator-lease"

    @property
    def coordinator_lease_key(self) -> str:
        return self._coordinator_lease_key

    def quarantine_key(self, node_id: str) -> str:
        return node_quarantine_key(self._run_id, node_id)

    def get(self, node_id: str) -> StoredNodeQuarantine | None:
        normalized_node_id = _nonempty_string(node_id, "node_id")
        key = self.quarantine_key(normalized_node_id)
        for _ in range(_MAX_READ_ATTEMPTS):
            entry = self._store.get(key)
            if entry is None:
                if not self._store.has_history(key):
                    return None
                if self._store.get(key) is not None:
                    continue
                raise QuarantineStateCorrupt(
                    f"node quarantine for {normalized_node_id!r} was deleted"
                )
            stored = self._decode_entry(entry, node_id=normalized_node_id)
            if self._store.get(key) == entry:
                return stored
        raise QuarantineStateError(
            f"node quarantine for {normalized_node_id!r} changed repeatedly during read"
        )

    def prepare_plan_writes(
        self,
        plan: RestartPlan,
        intent: RestartIntent,
        lease: HeldCoordinatorLease,
        *,
        authorized_resource_ids_by_node: Mapping[str, Sequence[str]],
        resource_to_node_id: Mapping[str, str],
    ) -> Mapping[str, ControlStoreWrite]:
        """Build create-once writes from coordinator-authorized fault evidence."""
        self._validate_lease(lease)
        self._validate_plan_intent(plan, intent)
        quarantined_nodes = set(plan.quarantined_node_ids)
        resources_by_node = _resources_by_node(authorized_resource_ids_by_node)
        if set(resources_by_node) != quarantined_nodes:
            raise ValueError(
                "authorized_resource_ids_by_node keys must exactly match quarantined nodes"
            )
        resource_owners = _resource_owners(resource_to_node_id)
        records: dict[str, NodeQuarantineRecord] = {}
        for node_id in sorted(quarantined_nodes):
            resource_ids = resources_by_node[node_id]
            wrong_owner = sorted(
                resource_id
                for resource_id in resource_ids
                if resource_owners.get(resource_id) != node_id
            )
            if wrong_owner:
                raise ValueError(
                    f"quarantine resources are not trusted as owned by {node_id!r}: {wrong_owner!r}"
                )
            records[node_id] = NodeQuarantineRecord(
                run_id=plan.run_id,
                node_id=node_id,
                plan_id=plan.plan_id,
                intent_id=plan.intent_id,
                from_generation=plan.from_generation,
                effective_generation=plan.to_generation,
                incident_ids=plan.incident_ids,
                reason_code=plan.reason_code,
                resource_ids=resource_ids,
                coordinator_id=lease.record.coordinator_id,
                lease_id=lease.record.lease_id,
                coordinator_lease_duration_ms=lease.record.lease_duration_ms,
                coordinator_fencing_token=lease.fencing_token,
            )
        return MappingProxyType(
            {
                self.quarantine_key(node_id): ControlStoreWrite(
                    expected_revision=None,
                    value=record.to_json(),
                    require_never_created=True,
                )
                for node_id, record in records.items()
            }
        )

    def _validate_plan_intent(
        self,
        plan: RestartPlan,
        intent: RestartIntent,
    ) -> None:
        if not isinstance(plan, RestartPlan):
            raise TypeError("plan must be RestartPlan")
        if not isinstance(intent, RestartIntent):
            raise TypeError("intent must be RestartIntent")
        if plan.run_id != self._run_id or intent.run_id != self._run_id:
            raise ValueError("restart plan and intent must belong to the repository run")
        if plan.intent_id != intent.intent_id:
            raise ValueError("restart plan does not reference the supplied intent")
        if plan.from_generation != intent.generation:
            raise ValueError("restart plan generation does not match the supplied intent")
        if plan.incident_ids != intent.incident_ids:
            raise ValueError("restart plan incidents do not match the supplied intent")
        if plan.reason_code != intent.reason_code:
            raise ValueError("restart plan reason does not match the supplied intent")
        if (
            intent.minimum_recovery_mode == "recovery_verified"
            and plan.recovery_mode != "recovery_verified"
        ):
            raise ValueError("restart plan recovery mode is weaker than the supplied intent")
        unsupported = sorted(set(plan.quarantined_node_ids) - set(intent.suspected_node_ids))
        if unsupported:
            raise ValueError(
                f"restart plan quarantines nodes outside the intent scope: {unsupported!r}"
            )

    def _validate_lease(self, lease: HeldCoordinatorLease) -> None:
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if lease.record.run_id != self._run_id:
            raise QuarantineLeaseLost("coordinator lease belongs to another run")
        entry = self._store.get(self._coordinator_lease_key)
        if entry is None or entry.revision != lease.fencing_token:
            raise QuarantineLeaseLost(
                "coordinator lease changed before quarantine write preparation"
            )
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise QuarantineStateCorrupt("coordinator lease is malformed") from error
        if entry.committed_at_unix_ms is None:
            raise QuarantineStateCorrupt("coordinator lease has no authoritative grant time")
        if record != lease.record or entry.committed_at_unix_ms != lease.granted_at_unix_ms:
            raise QuarantineLeaseLost("coordinator lease handle does not match persisted ownership")

    def _decode_entry(
        self,
        entry: ControlStoreEntry,
        *,
        node_id: str,
    ) -> StoredNodeQuarantine:
        try:
            record = NodeQuarantineRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise QuarantineStateCorrupt("persisted node quarantine is malformed") from error
        if record.run_id != self._run_id or record.node_id != node_id:
            raise QuarantineStateCorrupt("persisted node quarantine belongs to another run or node")
        if (
            entry.mutation_sequence != 1
            or entry.value_sequence != 1
            or entry.lifetime_sequence != 1
        ):
            raise QuarantineStateCorrupt("immutable node quarantine has noninitial store sequences")
        if entry.committed_at_unix_ms is None:
            raise QuarantineStateCorrupt(
                "persisted node quarantine has no authoritative commit time"
            )
        if entry.guard_key != self._coordinator_lease_key:
            raise QuarantineStateCorrupt(
                "persisted node quarantine was not guarded by the run coordinator lease"
            )
        guard_revision = entry.guard_revision
        guard_value_digest = entry.guard_value_digest
        guard_mutation_sequence = entry.guard_mutation_sequence
        guard_value_sequence = entry.guard_value_sequence
        guard_lifetime_sequence = entry.guard_lifetime_sequence
        guard_committed_at_unix_ms = entry.guard_committed_at_unix_ms
        if any(
            value is None
            for value in (
                guard_revision,
                guard_value_digest,
                guard_mutation_sequence,
                guard_value_sequence,
                guard_lifetime_sequence,
                guard_committed_at_unix_ms,
            )
        ):
            raise QuarantineStateCorrupt(
                "persisted node quarantine has incomplete guard provenance"
            )
        assert guard_mutation_sequence is not None
        assert guard_value_sequence is not None
        assert guard_lifetime_sequence is not None
        assert guard_revision is not None
        assert guard_value_digest is not None
        assert guard_committed_at_unix_ms is not None
        if guard_mutation_sequence < 2 * guard_lifetime_sequence - 1:
            raise QuarantineStateCorrupt(
                "node quarantine guard mutation sequence cannot support its lifetime"
            )
        if guard_value_sequence < guard_lifetime_sequence:
            raise QuarantineStateCorrupt(
                "node quarantine guard value sequence cannot support its lifetime"
            )
        if guard_value_sequence > guard_mutation_sequence - guard_lifetime_sequence + 1:
            raise QuarantineStateCorrupt(
                "node quarantine guard value sequence includes deletion mutations"
            )
        expected_lease = CoordinatorLeaseRecord(
            run_id=record.run_id,
            coordinator_id=record.coordinator_id,
            lease_id=record.lease_id,
            lease_duration_ms=record.coordinator_lease_duration_ms,
        )
        if guard_revision != record.coordinator_fencing_token:
            raise QuarantineStateCorrupt(
                "node quarantine fencing token does not match guard provenance"
            )
        if guard_value_digest != hashlib.sha256(expected_lease.to_json()).hexdigest():
            raise QuarantineStateCorrupt(
                "node quarantine lease identity does not match guard provenance"
            )
        if (
            entry.committed_at_unix_ms < guard_committed_at_unix_ms
            or entry.committed_at_unix_ms
            >= guard_committed_at_unix_ms + record.coordinator_lease_duration_ms
        ):
            raise QuarantineStateCorrupt(
                "node quarantine committed outside its coordinator lease window"
            )
        return StoredNodeQuarantine(record=record, entry=entry)


def node_quarantine_key(run_id: str, node_id: str) -> str:
    """Derive one run/node-scoped quarantine key without exposing identities."""
    normalized_run_id = _nonempty_string(run_id, "run_id")
    normalized_node_id = _nonempty_string(node_id, "node_id")
    run_digest = hashlib.sha256(normalized_run_id.encode("utf-8")).hexdigest()
    node_digest = hashlib.sha256(normalized_node_id.encode("utf-8")).hexdigest()
    return f"{_CONTROL_PREFIX}/runs/{run_digest}/node-quarantines/{node_digest}"


def _resources_by_node(value: object) -> dict[str, tuple[str, ...]]:
    if not isinstance(value, Mapping):
        raise TypeError("authorized_resource_ids_by_node must be a mapping")
    result: dict[str, tuple[str, ...]] = {}
    for node_id, resource_ids in value.items():
        normalized_node_id = _nonempty_string(
            node_id,
            "authorized_resource_ids_by_node key",
        )
        normalized_resource_ids = _strings(
            resource_ids,
            f"authorized_resource_ids_by_node[{normalized_node_id!r}]",
        )
        result[normalized_node_id] = normalized_resource_ids
    if len(result) != len(value):
        raise ValueError("authorized_resource_ids_by_node keys must be unique")
    return result


def _resource_owners(value: object) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise TypeError("resource_to_node_id must be a mapping")
    result: dict[str, str] = {}
    for resource_id, node_id in value.items():
        normalized_resource_id = _nonempty_string(
            resource_id,
            "resource_to_node_id key",
        )
        result[normalized_resource_id] = _nonempty_string(
            node_id,
            f"resource_to_node_id[{normalized_resource_id!r}]",
        )
    if len(result) != len(value):
        raise ValueError("resource_to_node_id keys must be unique")
    return result


def _strings(value: object, path: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
        raise ValueError(f"{path} must be an array")
    result = tuple(_nonempty_string(item, f"{path}[{index}]") for index, item in enumerate(value))
    if len(result) != len(set(result)):
        raise ValueError(f"{path} values must be unique")
    return result


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "NodeQuarantineRepository",
    "QuarantineLeaseLost",
    "QuarantineStateCorrupt",
    "QuarantineStateError",
    "StoredNodeQuarantine",
    "node_quarantine_key",
]
