"""Immutable transaction inputs for opening the first restart intent."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._generation_reader import CurrentGeneration
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class PreparedInitialRestartIntentOpen:
    """Immutable inputs for the first restart-intent guarded transaction."""

    record: RestartIntentRecord
    head: RestartIntentHeadRecord
    current: CurrentGeneration
    lease: HeldCoordinatorLease
    intent_key: str
    intent_head_key: str
    coordinator_lease_key: str
    generation_head_key: str
    generation_snapshot_key: str
    not_before_unix_ms: int
    deadline_unix_ms: int

    def __post_init__(self) -> None:
        if not isinstance(self.record, RestartIntentRecord):
            raise TypeError("PreparedInitialRestartIntentOpen.record must be RestartIntentRecord")
        if not isinstance(self.head, RestartIntentHeadRecord):
            raise TypeError("PreparedInitialRestartIntentOpen.head must be RestartIntentHeadRecord")
        if not isinstance(self.current, CurrentGeneration):
            raise TypeError("PreparedInitialRestartIntentOpen.current must be CurrentGeneration")
        if not isinstance(self.lease, HeldCoordinatorLease):
            raise TypeError("PreparedInitialRestartIntentOpen.lease must be HeldCoordinatorLease")
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
        active_nodes = set(assignment.slot_to_node_id.values())
        unknown_nodes = sorted(set(self.record.intent.suspected_node_ids) - active_nodes)
        if unknown_nodes:
            raise ValueError(
                "PreparedInitialRestartIntentOpen suspects nodes outside its "
                f"generation: {unknown_nodes!r}"
            )
        if (
            self.lease.record.run_id != self.record.intent.run_id
            or self.lease.record.coordinator_id != self.record.coordinator_id
            or self.lease.record.lease_id != self.record.lease_id
            or self.lease.record.lease_duration_ms != self.record.coordinator_lease_duration_ms
            or self.lease.fencing_token != self.record.coordinator_fencing_token
        ):
            raise ValueError(
                "PreparedInitialRestartIntentOpen lease does not authorize its intent record"
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
            ("generation_head_revision", self.current.head_revision),
            ("generation_snapshot_revision", self.current.snapshot.revision),
            (
                "generation_snapshot_committed_at_unix_ms",
                self.current.snapshot.committed_at_unix_ms,
            ),
            ("coordinator_lease_granted_at_unix_ms", self.lease.granted_at_unix_ms),
            ("not_before_unix_ms", self.not_before_unix_ms),
            ("deadline_unix_ms", self.deadline_unix_ms),
        ):
            _positive_integer(integer_value, f"PreparedInitialRestartIntentOpen.{path}")
        if self.not_before_unix_ms < self.lease.granted_at_unix_ms:
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
        if self.deadline_unix_ms > self.lease.expires_at_unix_ms:
            raise ValueError(
                "PreparedInitialRestartIntentOpen deadline exceeds its coordinator lease"
            )
        if self.deadline_unix_ms > self.record.intent.prepare_deadline_unix_ms:
            raise ValueError("PreparedInitialRestartIntentOpen deadline exceeds its restart intent")

    @property
    def expected_guard_revision(self) -> int:
        return self.lease.fencing_token

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


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = ["PreparedInitialRestartIntentOpen"]
