"""Authenticated preparation of the first torchrun restart-intent opening."""

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


class RestartIntentOpenPreparationError(RuntimeError):
    """Base error for preparing the first restart-intent opening."""


class RestartIntentOpenPreparationConflict(RestartIntentOpenPreparationError):
    """Raised when the supplied generation or lifecycle state changed."""


class RestartIntentOpenPreparationLeaseLost(RestartIntentOpenPreparationError):
    """Raised when the supplied coordinator lease is stale or foreign."""


class RestartIntentOpenPreparationDeadlineElapsed(RestartIntentOpenPreparationError):
    """Raised when no valid commit window remains."""


class RestartIntentOpenPreparationCorrupt(RestartIntentOpenPreparationError):
    """Raised when persisted ownership or lifecycle state is malformed."""


@dataclass(frozen=True, slots=True)
class PreparedInitialRestartIntentOpen:
    """Immutable inputs for the first restart-intent guarded transaction."""

    record: RestartIntentRecord
    head: RestartIntentHeadRecord
    current: CurrentGeneration
    intent_key: str
    intent_head_key: str
    coordinator_lease_key: str
    expected_guard_revision: int
    generation_head_key: str
    generation_snapshot_key: str
    coordinator_lease_granted_at_unix_ms: int
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, RestartIntentRecord):
            raise TypeError("PreparedInitialRestartIntentOpen.record must be RestartIntentRecord")
        if not isinstance(self.head, RestartIntentHeadRecord):
            raise TypeError("PreparedInitialRestartIntentOpen.head must be RestartIntentHeadRecord")
        if not isinstance(self.current, CurrentGeneration):
            raise TypeError("PreparedInitialRestartIntentOpen.current must be CurrentGeneration")
        if (
            self.head.run_id != self.record.intent.run_id
            or self.head.generation != self.record.intent.generation
            or self.head.intent_id != self.record.intent.intent_id
            or self.head.intent_digest != self.record.digest
        ):
            raise ValueError(
                "PreparedInitialRestartIntentOpen head does not identify its intent record"
            )
        assignment = self.current.snapshot.record.assignment
        if (
            assignment.run_id != self.record.intent.run_id
            or assignment.generation != self.record.intent.generation
            or self.current.snapshot.record.digest != self.record.generation_snapshot_digest
        ):
            raise ValueError(
                "PreparedInitialRestartIntentOpen generation does not identify its intent record"
            )
        run_digest = hashlib.sha256(self.record.intent.run_id.encode("utf-8")).hexdigest()
        run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        intent_digest = hashlib.sha256(self.record.intent.intent_id.encode("utf-8")).hexdigest()
        expected_keys = {
            "intent_key": f"{run_prefix}/restart-intents/{intent_digest}",
            "intent_head_key": f"{run_prefix}/restart-intent-head",
            "coordinator_lease_key": f"{run_prefix}/coordinator-lease",
            "generation_head_key": f"{run_prefix}/generation-head",
            "generation_snapshot_key": (
                f"{run_prefix}/generations/{self.record.intent.generation}"
            ),
        }
        for path, expected_key in expected_keys.items():
            if getattr(self, path) != expected_key:
                raise ValueError(f"PreparedInitialRestartIntentOpen.{path} is not canonical")
        for path, integer_value in (
            ("expected_guard_revision", self.expected_guard_revision),
            ("generation_head_revision", self.current.head_revision),
            ("generation_snapshot_revision", self.current.snapshot.revision),
            (
                "generation_snapshot_committed_at_unix_ms",
                self.current.snapshot.committed_at_unix_ms,
            ),
            (
                "coordinator_lease_granted_at_unix_ms",
                self.coordinator_lease_granted_at_unix_ms,
            ),
            ("not_before_unix_ms", self.not_before_unix_ms),
            ("deadline_unix_ms", self.deadline_unix_ms),
        ):
            _positive_integer(integer_value, f"PreparedInitialRestartIntentOpen.{path}")
        if self.expected_guard_revision != self.record.coordinator_fencing_token:
            raise ValueError(
                "PreparedInitialRestartIntentOpen guard revision does not match its intent record"
            )
        if self.not_before_unix_ms < self.coordinator_lease_granted_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentOpen cannot precede its coordinator lease grant"
            )
        if self.not_before_unix_ms < self.current.snapshot.committed_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentOpen cannot precede its generation snapshot"
            )
        if self.not_before_unix_ms >= self.deadline_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentOpen.not_before_unix_ms must precede its deadline"
            )
        lease_expiry_unix_ms = (
            self.coordinator_lease_granted_at_unix_ms + self.record.coordinator_lease_duration_ms
        )
        if self.deadline_unix_ms > lease_expiry_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentOpen deadline exceeds its coordinator lease"
            )
        if self.deadline_unix_ms > self.record.intent.prepare_deadline_unix_ms:
            raise ValueError("PreparedInitialRestartIntentOpen deadline exceeds its restart intent")

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return MappingProxyType(
            {
                self.intent_head_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.head.to_json(),
                    require_never_created=True,
                ),
                self.intent_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.record.to_json(),
                    require_never_created=True,
                ),
            }
        )

    @property
    def conditions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                self.generation_head_key: self.current.head_revision,
                self.generation_snapshot_key: self.current.snapshot.revision,
            }
        )


class RestartIntentOpenPreparer:
    """Authenticate and prepare the first restart intent for one run."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._generation_reader = GenerationStateReader(store, run_id=self._run_id)
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._intent_head_key = f"{self._run_prefix}/restart-intent-head"
        self._lifecycle_head_key = f"{self._run_prefix}/restart-intent-lifecycle-head"

    @property
    def coordinator_lease_key(self) -> str:
        return self._generation_reader.coordinator_lease_key

    @property
    def generation_head_key(self) -> str:
        return self._generation_reader.head_key

    @property
    def intent_head_key(self) -> str:
        return self._intent_head_key

    @property
    def lifecycle_head_key(self) -> str:
        return self._lifecycle_head_key

    def intent_key(self, intent_id: str) -> str:
        normalized_intent_id = _nonempty_string(intent_id, "intent_id")
        intent_digest = hashlib.sha256(normalized_intent_id.encode("utf-8")).hexdigest()
        return f"{self._run_prefix}/restart-intents/{intent_digest}"

    def prepare_initial_open(
        self,
        lease: HeldCoordinatorLease,
        current: CurrentGeneration,
        intent: RestartIntent,
    ) -> PreparedInitialRestartIntentOpen:
        self._validate_lease(lease)
        self._validate_current(current)
        self._validate_intent(intent, current)
        self._require_never_opened()
        not_before_unix_ms = max(
            lease.granted_at_unix_ms,
            current.snapshot.committed_at_unix_ms,
        )
        if lease.expires_at_unix_ms <= not_before_unix_ms:
            raise RestartIntentOpenPreparationLeaseLost(
                "coordinator lease expired before restart-intent preparation"
            )
        if intent.prepare_deadline_unix_ms <= not_before_unix_ms:
            raise RestartIntentOpenPreparationDeadlineElapsed(
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
        return PreparedInitialRestartIntentOpen(
            record=record,
            head=RestartIntentHeadRecord(
                run_id=self._run_id,
                generation=intent.generation,
                intent_id=intent.intent_id,
                intent_digest=record.digest,
            ),
            current=current,
            intent_key=self.intent_key(intent.intent_id),
            intent_head_key=self._intent_head_key,
            coordinator_lease_key=self.coordinator_lease_key,
            expected_guard_revision=lease.fencing_token,
            generation_head_key=self.generation_head_key,
            generation_snapshot_key=self._generation_reader.snapshot_key(intent.generation),
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
            raise RestartIntentOpenPreparationLeaseLost(
                "coordinator lease changed before restart-intent preparation"
            )
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise RestartIntentOpenPreparationCorrupt("coordinator lease is malformed") from error
        if entry.committed_at_unix_ms is None:
            raise RestartIntentOpenPreparationCorrupt(
                "coordinator lease has no authoritative grant time"
            )
        if record != lease.record or entry.committed_at_unix_ms != lease.granted_at_unix_ms:
            raise RestartIntentOpenPreparationLeaseLost(
                "coordinator lease handle does not match persisted ownership"
            )

    def _validate_current(self, current: CurrentGeneration) -> None:
        if not isinstance(current, CurrentGeneration):
            raise TypeError("current must be CurrentGeneration")
        if self._generation_reader.current() != current:
            raise RestartIntentOpenPreparationConflict(
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

    def _require_never_opened(self) -> None:
        if self._store.get(self._intent_head_key) is not None:
            raise RestartIntentOpenPreparationConflict(
                "a restart intent is already current or closed"
            )
        if self._store.has_history(self._intent_head_key):
            raise RestartIntentOpenPreparationConflict("restart-intent lifecycle already exists")
        if self._store.get(self._lifecycle_head_key) is not None or self._store.has_history(
            self._lifecycle_head_key
        ):
            raise RestartIntentOpenPreparationCorrupt(
                "restart-intent lifecycle exists without current-head history"
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
    "PreparedInitialRestartIntentOpen",
    "RestartIntentOpenPreparationConflict",
    "RestartIntentOpenPreparationCorrupt",
    "RestartIntentOpenPreparationDeadlineElapsed",
    "RestartIntentOpenPreparationError",
    "RestartIntentOpenPreparationLeaseLost",
    "RestartIntentOpenPreparer",
]
