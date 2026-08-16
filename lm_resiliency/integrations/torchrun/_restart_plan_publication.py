"""Read-only preparation of one complete restart-plan publication."""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    GenerationStateCorrupt,
    GenerationStateError,
    GenerationStateReader,
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._quarantine_store import node_quarantine_key
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleConflict,
    RestartPlanPublicationLifecycleCorrupt,
    RestartPlanPublicationLifecycleFence,
    RestartPlanPublicationLifecycleReader,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_records import (
    PreparedRestartPlanPublication,
    RestartPlanPublicationAuthority,
    RestartPlanPublicationRecords,
    recovery_manifest_key,
    restart_plan_key,
)
from lm_resiliency.integrations.torchrun._restart_plan_records import RestartPlanRecord
from lm_resiliency.integrations.torchrun._restart_plan_state import (
    PersistedRestartPlanPublication,
)

_MAX_READ_ATTEMPTS = 8


class RestartPlanPublicationPreparationError(RuntimeError):
    """Base error for authenticating restart-plan publication authority."""


class RestartPlanPublicationPreparationConflict(RestartPlanPublicationPreparationError):
    """Raised when coordinator authority changes repeatedly during preparation."""


class RestartPlanPublicationPreparationLeaseLost(RestartPlanPublicationPreparationError):
    """Raised when the plan's coordinator authority is absent, stale, or expired."""


class RestartPlanPublicationPreparationCorrupt(RestartPlanPublicationPreparationError):
    """Raised when durable coordinator authority is contradictory."""


class RestartPlanPublicationPreparationClockError(RestartPlanPublicationPreparationError):
    """Raised when the coordinator preparation clock is invalid."""


class RestartPlanPublicationAuthorityPreparer:
    """Authenticate publication records against the live coordinator lease."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        clock: Callable[[], int],
    ) -> None:
        self._run_id = _nonempty_string(run_id, "run_id")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._clock_lock = threading.Lock()
        self._last_now_unix_ms = 0
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    def prepare(
        self,
        records: RestartPlanPublicationRecords,
    ) -> RestartPlanPublicationAuthority:
        """Return authenticated, non-mutating publication authority."""

        if not isinstance(records, RestartPlanPublicationRecords):
            raise TypeError("records must be RestartPlanPublicationRecords")
        if records.candidate.plan.run_id != self._run_id:
            raise ValueError("restart-plan publication records belong to another run")
        authority, now_unix_ms = self._read_stable_authority()
        self._require_exact_authority(records, authority)
        required_observation = max(
            authority.lease.granted_at_unix_ms,
            records.current.snapshot.committed_at_unix_ms,
            records.candidate.placement_state.observed_at_unix_ms,
            records.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot.committed_at_unix_ms,
        )
        if now_unix_ms < required_observation:
            raise RestartPlanPublicationPreparationClockError(
                "restart-plan publication clock precedes authoritative inputs"
            )
        deadline_unix_ms = min(
            records.deadline_unix_ms,
            authority.lease.expires_at_unix_ms,
        )
        if now_unix_ms >= deadline_unix_ms:
            raise RestartPlanPublicationPreparationLeaseLost(
                "restart-plan publication authority window elapsed before preparation"
            )
        try:
            return RestartPlanPublicationAuthority(
                records=records,
                coordinator_authority=authority,
                observed_at_unix_ms=now_unix_ms,
            )
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationPreparationCorrupt(
                "durable coordinator authority cannot authorize restart-plan publication"
            ) from error

    def _read_stable_authority(
        self,
    ) -> tuple[CoordinatorLeaseAuthority, int]:
        for _ in range(_MAX_READ_ATTEMPTS):
            history = self._read_history()
            now_unix_ms = self._now_unix_ms()
            if history != self._read_history():
                continue
            if not history:
                raise RestartPlanPublicationPreparationLeaseLost(
                    "no live coordinator lease can authorize restart-plan publication"
                )
            return history[-1], now_unix_ms
        raise RestartPlanPublicationPreparationConflict(
            "coordinator lease history changed repeatedly during publication preparation"
        )

    def _read_history(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        try:
            return self._lease_history_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartPlanPublicationPreparationCorrupt(
                "coordinator lease history is corrupt"
            ) from error
        except CoordinatorLeaseHistoryError as error:
            raise RestartPlanPublicationPreparationConflict(
                "coordinator lease history changed repeatedly during preparation"
            ) from error

    def _require_exact_authority(
        self,
        records: RestartPlanPublicationRecords,
        authority: CoordinatorLeaseAuthority,
    ) -> None:
        plan_record = records.candidate.placement_state.generation_state.record
        lease = authority.lease
        if (
            lease.record.run_id != self._run_id
            or lease.record.coordinator_id != plan_record.coordinator_id
            or lease.record.lease_id != plan_record.lease_id
            or lease.record.lease_duration_ms != plan_record.coordinator_lease_duration_ms
            or lease.fencing_token != plan_record.coordinator_fencing_token
        ):
            raise RestartPlanPublicationPreparationLeaseLost(
                "restart plan is not authorized by the live coordinator lease"
            )

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            try:
                now_unix_ms = _positive_integer(
                    self._clock(),
                    "restart-plan publication preparation clock",
                )
            except (TypeError, ValueError) as error:
                raise RestartPlanPublicationPreparationClockError(
                    "restart-plan publication preparation clock is invalid"
                ) from error
            if now_unix_ms < self._last_now_unix_ms:
                raise RestartPlanPublicationPreparationClockError(
                    "restart-plan publication preparation clock moved backward"
                )
            self._last_now_unix_ms = now_unix_ms
            return now_unix_ms


class RestartPlanPublicationError(RuntimeError):
    """Base error for preparing one complete restart-plan publication."""


class RestartPlanPublicationConflict(RestartPlanPublicationError):
    """Raised when publication inputs change or no closed intent is available."""


class RestartPlanPublicationLeaseLost(RestartPlanPublicationError):
    """Raised when the plan's coordinator authority is no longer live."""


class RestartPlanPublicationClockError(RestartPlanPublicationError):
    """Raised when the publication preparation clock is unsafe."""


class RestartPlanPublicationCorrupt(RestartPlanPublicationError):
    """Raised when durable publication dependencies are contradictory."""


class RestartPlanPublicationReadError(RuntimeError):
    """Base error for reading the latest committed restart plan."""


class RestartPlanPublicationReadConflict(RestartPlanPublicationReadError):
    """Raised when generation or publication state changes repeatedly."""


class RestartPlanPublicationReadCorrupt(RestartPlanPublicationReadError):
    """Raised when the latest publication is incomplete or contradictory."""


@dataclass(frozen=True, slots=True)
class _PublicationEntries:
    plan: ControlStoreEntry
    manifest: ControlStoreEntry
    successor_snapshot: ControlStoreEntry
    generation_head: ControlStoreEntry
    quarantines: tuple[tuple[str, ControlStoreEntry], ...]


class RestartPlanPublicationReader:
    """Read one stable latest restart-plan publication without mutation."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._generation_reader = GenerationStateReader(
            store,
            run_id=self._run_id,
        )

    def read(self) -> PersistedRestartPlanPublication | None:
        """Return the stable publication for the current generation, if any."""

        for _ in range(_MAX_READ_ATTEMPTS):
            current = self._read_current()
            if current is None or current.snapshot.record.assignment.generation == 0:
                return None
            generation = current.snapshot.record.assignment.generation
            first = self._read_entries(generation)
            if self._read_current() != current:
                continue
            second = self._read_entries(generation)
            if second != first or self._read_current() != current:
                continue
            publication = self._decode(first, generation)
            self._validate_current(publication, current)
            return publication
        raise RestartPlanPublicationReadConflict(
            "restart-plan publication changed repeatedly during read"
        )

    def _read_current(self) -> CurrentGeneration | None:
        try:
            return self._generation_reader.current()
        except GenerationStateCorrupt as error:
            raise RestartPlanPublicationReadCorrupt(
                "generation state is corrupt while reading restart-plan publication"
            ) from error
        except GenerationStateError as error:
            raise RestartPlanPublicationReadConflict(
                "generation state changed repeatedly while reading restart-plan publication"
            ) from error

    def _read_entries(self, generation: int) -> _PublicationEntries:
        plan = self._require_entry(
            restart_plan_key(self._run_id, generation),
            "restart plan",
        )
        try:
            plan_record = RestartPlanRecord.from_json(plan.value)
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationReadCorrupt(
                "restart-plan publication contains a malformed plan record"
            ) from error
        quarantines = tuple(
            (
                node_id,
                self._require_entry(
                    node_quarantine_key(self._run_id, node_id),
                    f"quarantine record for {node_id!r}",
                ),
            )
            for node_id in sorted(plan_record.plan.quarantined_node_ids)
        )
        return _PublicationEntries(
            plan=plan,
            manifest=self._require_entry(
                recovery_manifest_key(self._run_id, generation),
                "recovery manifest",
            ),
            successor_snapshot=self._require_entry(
                self._generation_reader.snapshot_key(generation),
                "successor generation snapshot",
            ),
            generation_head=self._require_entry(
                self._generation_reader.head_key,
                "generation head",
            ),
            quarantines=quarantines,
        )

    def _require_entry(self, key: str, path: str) -> ControlStoreEntry:
        entry = self._store.get(key)
        if entry is None:
            raise RestartPlanPublicationReadCorrupt(
                f"latest restart-plan publication is missing its {path}"
            )
        return entry

    def _decode(
        self,
        entries: _PublicationEntries,
        generation: int,
    ) -> PersistedRestartPlanPublication:
        try:
            return PersistedRestartPlanPublication.from_entries(
                run_id=self._run_id,
                to_generation=generation,
                plan_entry=entries.plan,
                manifest_entry=entries.manifest,
                successor_snapshot_entry=entries.successor_snapshot,
                generation_head_entry=entries.generation_head,
                quarantine_entries=dict(entries.quarantines),
            )
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationReadCorrupt(
                "latest restart-plan publication is corrupt"
            ) from error

    def _validate_current(
        self,
        publication: PersistedRestartPlanPublication,
        current: CurrentGeneration,
    ) -> None:
        successor_entry = publication.successor_snapshot_entry
        if (
            successor_entry.committed_at_unix_ms is None
            or successor_entry.guard_mutation_sequence is None
            or successor_entry.guard_value_sequence is None
            or successor_entry.guard_lifetime_sequence is None
            or successor_entry.guard_committed_at_unix_ms is None
        ):
            raise RestartPlanPublicationReadCorrupt(
                "latest restart-plan successor lacks authoritative metadata"
            )
        expected_snapshot = StoredGenerationSnapshot(
            record=publication.successor_snapshot,
            revision=successor_entry.revision,
            committed_at_unix_ms=successor_entry.committed_at_unix_ms,
            transaction_sequence=successor_entry.transaction_sequence,
            guard_mutation_sequence=successor_entry.guard_mutation_sequence,
            guard_value_sequence=successor_entry.guard_value_sequence,
            guard_lifetime_sequence=successor_entry.guard_lifetime_sequence,
            guard_committed_at_unix_ms=successor_entry.guard_committed_at_unix_ms,
        )
        if (
            current.snapshot != expected_snapshot
            or current.head_revision != publication.generation_head_entry.revision
        ):
            raise RestartPlanPublicationReadCorrupt(
                "latest restart-plan publication does not match the current generation"
            )


class RestartPlanPublicationPreparer:
    """Compose authenticated authority and lifecycle state without mutation."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        clock: Callable[[], int],
    ) -> None:
        self._authority_preparer = RestartPlanPublicationAuthorityPreparer(
            store,
            run_id=run_id,
            clock=clock,
        )
        self._lifecycle_reader = RestartPlanPublicationLifecycleReader(
            store,
            run_id=run_id,
        )

    def prepare(
        self,
        records: RestartPlanPublicationRecords,
    ) -> PreparedRestartPlanPublication:
        """Return one authenticated, lifecycle-fenced publication value."""

        authority = self._prepare_authority(records)
        lifecycle_fence = self._read_lifecycle()
        try:
            return PreparedRestartPlanPublication(
                authority=authority,
                lifecycle_fence=lifecycle_fence,
            )
        except TypeError as error:
            raise RestartPlanPublicationCorrupt(
                "authenticated publication inputs have invalid types"
            ) from error
        except ValueError as error:
            raise RestartPlanPublicationConflict(
                "restart-plan publication inputs changed during preparation"
            ) from error

    def _prepare_authority(
        self,
        records: RestartPlanPublicationRecords,
    ) -> RestartPlanPublicationAuthority:
        try:
            return self._authority_preparer.prepare(records)
        except RestartPlanPublicationPreparationClockError as error:
            raise RestartPlanPublicationClockError(str(error)) from error
        except RestartPlanPublicationPreparationLeaseLost as error:
            raise RestartPlanPublicationLeaseLost(str(error)) from error
        except RestartPlanPublicationPreparationConflict as error:
            raise RestartPlanPublicationConflict(str(error)) from error
        except RestartPlanPublicationPreparationCorrupt as error:
            raise RestartPlanPublicationCorrupt(str(error)) from error

    def _read_lifecycle(self) -> RestartPlanPublicationLifecycleFence:
        try:
            return self._lifecycle_reader.read()
        except RestartPlanPublicationLifecycleConflict as error:
            raise RestartPlanPublicationConflict(str(error)) from error
        except RestartPlanPublicationLifecycleCorrupt as error:
            raise RestartPlanPublicationCorrupt(str(error)) from error


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "RestartPlanPublicationAuthorityPreparer",
    "RestartPlanPublicationClockError",
    "RestartPlanPublicationConflict",
    "RestartPlanPublicationCorrupt",
    "RestartPlanPublicationError",
    "RestartPlanPublicationLeaseLost",
    "RestartPlanPublicationPreparationClockError",
    "RestartPlanPublicationPreparationConflict",
    "RestartPlanPublicationPreparationCorrupt",
    "RestartPlanPublicationPreparationError",
    "RestartPlanPublicationPreparationLeaseLost",
    "RestartPlanPublicationPreparer",
    "RestartPlanPublicationReadConflict",
    "RestartPlanPublicationReadCorrupt",
    "RestartPlanPublicationReadError",
    "RestartPlanPublicationReader",
]
