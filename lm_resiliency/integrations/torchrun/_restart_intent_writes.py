"""Authenticated transaction writes for opening torchrun restart intents."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreWrite,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._protocol import RestartIntent
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


class RestartIntentPreparationError(RuntimeError):
    """Base error for preparing a restart-intent open transaction."""


class RestartIntentPreparationConflict(RestartIntentPreparationError):
    """Raised when the supplied generation is no longer current."""


class RestartIntentPreparationLeaseLost(RestartIntentPreparationError):
    """Raised when the supplied coordinator lease is stale or foreign."""


class RestartIntentPreparationDeadlineElapsed(RestartIntentPreparationError):
    """Raised when no valid commit window remains."""


class RestartIntentPreparationCorrupt(RestartIntentPreparationError):
    """Raised when persisted coordinator ownership is malformed."""


@dataclass(frozen=True, slots=True)
class PreparedRestartIntentOpen:
    """One immutable set of inputs for a future guarded store transaction."""

    record: RestartIntentRecord
    head: RestartIntentHeadRecord
    intent_key: str
    intent_head_key: str
    coordinator_lease_key: str
    expected_guard_revision: int
    generation_head_key: str
    expected_generation_head_revision: int
    generation_snapshot_key: str
    expected_generation_snapshot_revision: int
    coordinator_lease_granted_at_unix_ms: int
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, RestartIntentRecord):
            raise TypeError("PreparedRestartIntentOpen.record must be RestartIntentRecord")
        if not isinstance(self.head, RestartIntentHeadRecord):
            raise TypeError("PreparedRestartIntentOpen.head must be RestartIntentHeadRecord")
        if (
            self.head.run_id != self.record.intent.run_id
            or self.head.generation != self.record.intent.generation
            or self.head.intent_id != self.record.intent.intent_id
            or self.head.intent_digest != self.record.digest
        ):
            raise ValueError("PreparedRestartIntentOpen head does not identify its intent record")
        for path, key in (
            ("intent_key", self.intent_key),
            ("intent_head_key", self.intent_head_key),
            ("coordinator_lease_key", self.coordinator_lease_key),
            ("generation_head_key", self.generation_head_key),
            ("generation_snapshot_key", self.generation_snapshot_key),
        ):
            _nonempty_string(key, f"PreparedRestartIntentOpen.{path}")
        for path, integer_value in (
            ("expected_guard_revision", self.expected_guard_revision),
            (
                "expected_generation_head_revision",
                self.expected_generation_head_revision,
            ),
            (
                "expected_generation_snapshot_revision",
                self.expected_generation_snapshot_revision,
            ),
            (
                "coordinator_lease_granted_at_unix_ms",
                self.coordinator_lease_granted_at_unix_ms,
            ),
            ("not_before_unix_ms", self.not_before_unix_ms),
            ("deadline_unix_ms", self.deadline_unix_ms),
        ):
            _positive_integer(integer_value, f"PreparedRestartIntentOpen.{path}")
        if self.expected_guard_revision != self.record.coordinator_fencing_token:
            raise ValueError(
                "PreparedRestartIntentOpen guard revision does not match its intent record"
            )
        keys = {
            self.intent_key,
            self.intent_head_key,
            self.coordinator_lease_key,
            self.generation_head_key,
            self.generation_snapshot_key,
        }
        if len(keys) != 5:
            raise ValueError("PreparedRestartIntentOpen key roles must be distinct")
        if self.not_before_unix_ms < self.coordinator_lease_granted_at_unix_ms:
            raise ValueError("PreparedRestartIntentOpen cannot precede its coordinator lease grant")
        if self.not_before_unix_ms >= self.deadline_unix_ms:
            raise ValueError(
                "PreparedRestartIntentOpen.not_before_unix_ms must precede its deadline"
            )
        lease_expiry_unix_ms = (
            self.coordinator_lease_granted_at_unix_ms + self.record.coordinator_lease_duration_ms
        )
        if self.deadline_unix_ms > lease_expiry_unix_ms:
            raise ValueError("PreparedRestartIntentOpen deadline exceeds its coordinator lease")
        if self.deadline_unix_ms > self.record.intent.prepare_deadline_unix_ms:
            raise ValueError("PreparedRestartIntentOpen deadline exceeds its restart intent")

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return MappingProxyType(
            {
                self.intent_head_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.head.to_json(),
                ),
                self.intent_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.record.to_json(),
                    require_never_created=True,
                ),
            }
        )

    @property
    def conditions(self) -> Mapping[str, int | None]:
        return MappingProxyType(
            {
                self.generation_head_key: self.expected_generation_head_revision,
                self.generation_snapshot_key: self.expected_generation_snapshot_revision,
            }
        )


class RestartIntentWriteRepository:
    """Validate authority and prepare one restart-intent open transaction."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._generation_reader = GenerationStateReader(store, run_id=self._run_id)
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._intent_head_key = f"{self._run_prefix}/restart-intent-head"

    @property
    def coordinator_lease_key(self) -> str:
        return self._generation_reader.coordinator_lease_key

    @property
    def generation_head_key(self) -> str:
        return self._generation_reader.head_key

    @property
    def intent_head_key(self) -> str:
        return self._intent_head_key

    def intent_key(self, intent_id: str) -> str:
        normalized_intent_id = _nonempty_string(intent_id, "intent_id")
        intent_digest = hashlib.sha256(normalized_intent_id.encode("utf-8")).hexdigest()
        return f"{self._run_prefix}/restart-intents/{intent_digest}"

    def prepare_open(
        self,
        lease: HeldCoordinatorLease,
        current: CurrentGeneration,
        intent: RestartIntent,
    ) -> PreparedRestartIntentOpen:
        self._validate_lease(lease)
        self._validate_current(current)
        self._validate_intent(intent, current)
        not_before_unix_ms = max(
            lease.granted_at_unix_ms,
            current.snapshot.committed_at_unix_ms,
        )
        if lease.expires_at_unix_ms <= not_before_unix_ms:
            raise RestartIntentPreparationLeaseLost(
                "coordinator lease expired before restart-intent preparation"
            )
        if intent.prepare_deadline_unix_ms <= not_before_unix_ms:
            raise RestartIntentPreparationDeadlineElapsed(
                "restart-intent preparation deadline has already elapsed"
            )
        record = RestartIntentRecord(
            intent=intent,
            generation_snapshot_digest=current.snapshot.record.digest,
            coordinator_id=lease.record.coordinator_id,
            lease_id=lease.record.lease_id,
            coordinator_lease_duration_ms=lease.record.lease_duration_ms,
            coordinator_fencing_token=lease.fencing_token,
        )
        return PreparedRestartIntentOpen(
            record=record,
            head=RestartIntentHeadRecord(
                run_id=self._run_id,
                generation=intent.generation,
                intent_id=intent.intent_id,
                intent_digest=record.digest,
            ),
            intent_key=self.intent_key(intent.intent_id),
            intent_head_key=self._intent_head_key,
            coordinator_lease_key=self.coordinator_lease_key,
            expected_guard_revision=lease.fencing_token,
            generation_head_key=self.generation_head_key,
            expected_generation_head_revision=current.head_revision,
            generation_snapshot_key=self._generation_reader.snapshot_key(intent.generation),
            expected_generation_snapshot_revision=current.snapshot.revision,
            coordinator_lease_granted_at_unix_ms=lease.granted_at_unix_ms,
            not_before_unix_ms=not_before_unix_ms,
            deadline_unix_ms=min(
                lease.expires_at_unix_ms,
                intent.prepare_deadline_unix_ms,
            ),
        )

    def _validate_lease(self, lease: HeldCoordinatorLease) -> None:
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if lease.record.run_id != self._run_id:
            raise ValueError("coordinator lease belongs to another run")
        entry = self._store.get(self.coordinator_lease_key)
        if entry is None or entry.revision != lease.fencing_token:
            raise RestartIntentPreparationLeaseLost(
                "coordinator lease changed before restart-intent preparation"
            )
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentPreparationCorrupt("coordinator lease is malformed") from error
        if entry.committed_at_unix_ms is None:
            raise RestartIntentPreparationCorrupt(
                "coordinator lease has no authoritative grant time"
            )
        if record != lease.record or entry.committed_at_unix_ms != lease.granted_at_unix_ms:
            raise RestartIntentPreparationLeaseLost(
                "coordinator lease handle does not match persisted ownership"
            )

    def _validate_current(self, current: CurrentGeneration) -> None:
        if not isinstance(current, CurrentGeneration):
            raise TypeError("current must be CurrentGeneration")
        if self._generation_reader.current() != current:
            raise RestartIntentPreparationConflict(
                "current generation does not match the committed generation head"
            )

    def _validate_intent(
        self,
        intent: RestartIntent,
        current: CurrentGeneration,
    ) -> None:
        if not isinstance(intent, RestartIntent):
            raise TypeError("intent must be RestartIntent")
        if intent.run_id != self._run_id:
            raise ValueError("restart intent belongs to another run")
        assignment = current.snapshot.record.assignment
        if intent.generation != assignment.generation:
            raise ValueError("restart intent does not target the current generation")
        active_nodes = set(assignment.slot_to_node_id.values())
        unknown_nodes = sorted(set(intent.suspected_node_ids) - active_nodes)
        if unknown_nodes:
            raise ValueError(
                f"restart intent suspects nodes outside the current generation: {unknown_nodes!r}"
            )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "PreparedRestartIntentOpen",
    "RestartIntentPreparationConflict",
    "RestartIntentPreparationCorrupt",
    "RestartIntentPreparationDeadlineElapsed",
    "RestartIntentPreparationError",
    "RestartIntentPreparationLeaseLost",
    "RestartIntentWriteRepository",
]
