"""Immutable transaction records for publishing one torchrun restart plan."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._generation_reader import CurrentGeneration
from lm_resiliency.integrations.torchrun._generation_records import GenerationHeadRecord
from lm_resiliency.integrations.torchrun._quarantine_store import node_quarantine_key
from lm_resiliency.integrations.torchrun._restart_plan_state import RestartPlanCandidateState

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"


@dataclass(frozen=True, slots=True)
class RestartPlanPublicationRecords:
    """Canonical records and store inputs for one restart-plan publication."""

    candidate: RestartPlanCandidateState
    current: CurrentGeneration

    def __post_init__(self) -> None:
        if not isinstance(self.candidate, RestartPlanCandidateState):
            raise TypeError(
                "RestartPlanPublicationRecords.candidate must be RestartPlanCandidateState"
            )
        if not isinstance(self.current, CurrentGeneration):
            raise TypeError("RestartPlanPublicationRecords.current must be CurrentGeneration")
        generation_state = self.candidate.placement_state.generation_state
        if self.current.snapshot.record != generation_state.from_snapshot:
            raise ValueError(
                "RestartPlanPublicationRecords current generation does not match its candidate"
            )
        _positive_integer(
            self.current.head_revision,
            "RestartPlanPublicationRecords.current.head_revision",
        )
        _positive_integer(
            self.current.snapshot.revision,
            "RestartPlanPublicationRecords.current.snapshot.revision",
        )
        manifest_source = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot
        _positive_integer(
            manifest_source.revision,
            "RestartPlanPublicationRecords manifest source revision",
        )
        if (
            self.manifest_source_generation_snapshot_key == self.source_generation_snapshot_key
            and manifest_source.revision != self.current.snapshot.revision
        ):
            raise ValueError(
                "RestartPlanPublicationRecords current and manifest source revisions disagree"
            )

    @property
    def run_prefix(self) -> str:
        run_digest = hashlib.sha256(self.candidate.plan.run_id.encode("utf-8")).hexdigest()
        return f"{_CONTROL_PREFIX}/runs/{run_digest}"

    @property
    def plan_key(self) -> str:
        return f"{self.run_prefix}/restart-plans/{self.candidate.plan.to_generation}"

    @property
    def recovery_manifest_key(self) -> str:
        return f"{self.plan_key}/recovery-manifest"

    @property
    def generation_head_key(self) -> str:
        return f"{self.run_prefix}/generation-head"

    @property
    def source_generation_snapshot_key(self) -> str:
        return f"{self.run_prefix}/generations/{self.candidate.plan.from_generation}"

    @property
    def successor_generation_snapshot_key(self) -> str:
        return f"{self.run_prefix}/generations/{self.candidate.plan.to_generation}"

    @property
    def manifest_source_generation_snapshot_key(self) -> str:
        source_generation = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.manifest.source_generation
        return f"{self.run_prefix}/generations/{source_generation}"

    @property
    def quarantine_keys(self) -> Mapping[str, str]:
        records = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state
        return MappingProxyType(
            {
                node_id: node_quarantine_key(self.candidate.plan.run_id, node_id)
                for node_id in records.quarantine_records
            }
        )

    @property
    def registration_keys(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                node_id: agent_registration_key(self.candidate.plan.run_id, node_id)
                for node_id in self.candidate.placement_state.registration_histories
            }
        )

    @property
    def generation_head(self) -> GenerationHeadRecord:
        generation_state = self.candidate.placement_state.generation_state
        return GenerationHeadRecord(
            run_id=self.candidate.plan.run_id,
            generation=self.candidate.plan.to_generation,
            snapshot_digest=generation_state.to_snapshot.digest,
        )

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        generation_state = self.candidate.placement_state.generation_state
        manifest_state = (
            self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state
        )
        quarantine_state = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state
        writes = {
            self.generation_head_key: ControlStoreWrite(
                expected_revision=self.current.head_revision,
                value=self.generation_head.to_json(),
            ),
            self.successor_generation_snapshot_key: ControlStoreWrite(
                expected_revision=None,
                value=generation_state.to_snapshot.to_json(),
                require_never_created=True,
            ),
            self.recovery_manifest_key: ControlStoreWrite(
                expected_revision=None,
                value=manifest_state.resolved_manifest.record.to_json(),
                require_never_created=True,
            ),
            self.plan_key: ControlStoreWrite(
                expected_revision=None,
                value=generation_state.record.to_json(),
                require_never_created=True,
            ),
        }
        writes.update(
            {
                self.quarantine_keys[node_id]: ControlStoreWrite(
                    expected_revision=None,
                    value=record.to_json(),
                    require_never_created=True,
                )
                for node_id, record in quarantine_state.quarantine_records.items()
            }
        )
        return MappingProxyType(writes)

    @property
    def conditions(self) -> Mapping[str, int]:
        manifest_source = self.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot
        conditions = {
            self.source_generation_snapshot_key: self.current.snapshot.revision,
            self.manifest_source_generation_snapshot_key: manifest_source.revision,
        }
        for node_id, history in self.candidate.placement_state.registration_histories.items():
            registration = history.current
            if registration is None:
                raise AssertionError("validated placement lost its current registration")
            conditions[self.registration_keys[node_id]] = registration.fencing_token
        return MappingProxyType(conditions)


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = ["RestartPlanPublicationRecords"]
