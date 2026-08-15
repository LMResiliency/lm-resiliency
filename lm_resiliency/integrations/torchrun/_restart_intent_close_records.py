"""Immutable record writes for closing the first restart intent."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class InitialRestartIntentClosureRecords:
    """Linked records and store inputs for closing the first restart intent."""

    opened: CommittedInitialRestartIntentOpen
    lifecycle: RestartIntentLifecycleRecord
    lifecycle_head: RestartIntentLifecycleHeadRecord
    closed_head: RestartIntentClosedHeadRecord
    intent_key: str
    intent_head_key: str
    closure_key: str
    lifecycle_head_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError(
                "InitialRestartIntentClosureRecords.opened must be "
                "CommittedInitialRestartIntentOpen"
            )
        if not isinstance(self.lifecycle, RestartIntentLifecycleRecord):
            raise TypeError(
                "InitialRestartIntentClosureRecords.lifecycle must be RestartIntentLifecycleRecord"
            )
        if not isinstance(self.lifecycle_head, RestartIntentLifecycleHeadRecord):
            raise TypeError(
                "InitialRestartIntentClosureRecords.lifecycle_head must be "
                "RestartIntentLifecycleHeadRecord"
            )
        if not isinstance(self.closed_head, RestartIntentClosedHeadRecord):
            raise TypeError(
                "InitialRestartIntentClosureRecords.closed_head must be "
                "RestartIntentClosedHeadRecord"
            )
        open_head = self.opened.prepared.head
        if self.lifecycle.closed_intent != open_head:
            raise ValueError(
                "InitialRestartIntentClosureRecords lifecycle does not close its open intent"
            )
        if (
            self.lifecycle_head.run_id != open_head.run_id
            or self.lifecycle_head.closure_index != 1
            or self.lifecycle_head.generation != open_head.generation
            or self.lifecycle_head.intent_id != open_head.intent_id
            or self.lifecycle_head.lifecycle_digest != self.lifecycle.digest
        ):
            raise ValueError(
                "InitialRestartIntentClosureRecords lifecycle head does not identify its closure"
            )
        if (
            self.closed_head.run_id != self.lifecycle_head.run_id
            or self.closed_head.closure_index != self.lifecycle_head.closure_index
            or self.closed_head.generation != self.lifecycle_head.generation
            or self.closed_head.intent_id != self.lifecycle_head.intent_id
            or self.closed_head.lifecycle_head_digest != self.lifecycle_head.digest
        ):
            raise ValueError(
                "InitialRestartIntentClosureRecords closed head does not identify its lifecycle"
            )
        run_digest = hashlib.sha256(open_head.run_id.encode("utf-8")).hexdigest()
        run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        expected_keys = {
            "intent_key": self.opened.prepared.intent_key,
            "intent_head_key": self.opened.prepared.intent_head_key,
            "closure_key": f"{run_prefix}/restart-intent-closures/1",
            "lifecycle_head_key": self.opened.prepared.lifecycle_head_key,
        }
        for path, expected_key in expected_keys.items():
            if getattr(self, path) != expected_key:
                raise ValueError(f"InitialRestartIntentClosureRecords.{path} is not canonical")
        if len(set(expected_keys.values())) != len(expected_keys):
            raise ValueError("InitialRestartIntentClosureRecords key roles must be distinct")

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return MappingProxyType(
            {
                self.intent_head_key: ControlStoreWrite(
                    expected_revision=self.opened.head_entry.revision,
                    value=self.closed_head.to_json(),
                ),
                self.lifecycle_head_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.lifecycle_head.to_json(),
                    require_never_created=True,
                ),
                self.closure_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.lifecycle.to_json(),
                    require_never_created=True,
                ),
            }
        )

    @property
    def conditions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                self.intent_key: self.opened.intent_entry.revision,
            }
        )


__all__ = ["InitialRestartIntentClosureRecords"]
