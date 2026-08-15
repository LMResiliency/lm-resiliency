"""Lease-fenced mutation of immutable torchrun generation state."""

from __future__ import annotations

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreClockError,
    ControlStoreConflict,
    ControlStoreDeadlineExceeded,
    ControlStoreTooEarly,
    ControlStoreWrite,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    GenerationStateCorrupt,
    GenerationStateError,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationHeadRecord,
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import RankAssignment


class GenerationStateConflict(GenerationStateError):
    """Raised when initialization or successor commit observes newer state."""


class GenerationStateLeaseLost(GenerationStateError):
    """Raised when the coordinator lease is stale or expired."""


class GenerationStateClockError(GenerationStateError):
    """Raised when authoritative store time contradicts the lease timeline."""


class GenerationStateManager(GenerationStateReader):
    """Commit immutable generation snapshots under a coordinator lease."""

    def initialize(
        self,
        lease: HeldCoordinatorLease,
        assignment: RankAssignment,
    ) -> CurrentGeneration:
        self._validate_lease(lease)
        self._validate_assignment(assignment)
        if assignment.generation != 0:
            raise ValueError("initial generation assignment must use generation zero")
        if self.current() is not None:
            raise GenerationStateConflict("generation state is already initialized")
        snapshot_record = self._snapshot_record(
            lease,
            assignment,
            previous_snapshot_digest=None,
        )
        return self._commit(
            lease=lease,
            snapshot_record=snapshot_record,
            expected_head_revision=None,
        )

    def commit_successor(
        self,
        lease: HeldCoordinatorLease,
        current: CurrentGeneration,
        assignment: RankAssignment,
    ) -> CurrentGeneration:
        self._validate_lease(lease)
        self._validate_current(current)
        self._validate_assignment(assignment)
        previous = current.snapshot.record.assignment
        if assignment.generation != previous.generation + 1:
            raise ValueError("successor assignment must advance generation by exactly one")
        if assignment.active_nodes != previous.active_nodes:
            raise ValueError("successor assignment must preserve active node count")
        if assignment.local_world_size != previous.local_world_size:
            raise ValueError("successor assignment must preserve local world size")
        if assignment.slot_to_rank_range != previous.slot_to_rank_range:
            raise ValueError("successor assignment must preserve logical rank ranges")
        if assignment.topology_digest != previous.topology_digest:
            raise ValueError("successor assignment must preserve topology digest")
        snapshot_record = self._snapshot_record(
            lease,
            assignment,
            previous_snapshot_digest=current.snapshot.record.digest,
        )
        return self._commit(
            lease=lease,
            snapshot_record=snapshot_record,
            expected_head_revision=current.head_revision,
        )

    def _commit(
        self,
        *,
        lease: HeldCoordinatorLease,
        snapshot_record: GenerationSnapshotRecord,
        expected_head_revision: int | None,
    ) -> CurrentGeneration:
        head_record = GenerationHeadRecord(
            run_id=self._run_id,
            generation=snapshot_record.assignment.generation,
            snapshot_digest=snapshot_record.digest,
        )
        snapshot_key = self.snapshot_key(snapshot_record.assignment.generation)
        try:
            committed = self._store.compare_set_many_guarded(
                {
                    self._head_key: ControlStoreWrite(
                        expected_revision=expected_head_revision,
                        value=head_record.to_json(),
                    ),
                    snapshot_key: ControlStoreWrite(
                        expected_revision=None,
                        value=snapshot_record.to_json(),
                    ),
                },
                guard_key=self._coordinator_lease_key,
                expected_guard_revision=lease.fencing_token,
                not_before_unix_ms=lease.granted_at_unix_ms,
                deadline_unix_ms=lease.expires_at_unix_ms,
            )
        except ControlStoreConflict as error:
            if error.key == self._coordinator_lease_key:
                raise GenerationStateLeaseLost(
                    "coordinator lease changed before generation commit"
                ) from error
            raise GenerationStateConflict(
                f"generation state changed at {error.key!r} before commit"
            ) from error
        except ControlStoreDeadlineExceeded as error:
            raise GenerationStateLeaseLost(
                "coordinator lease expired before generation commit"
            ) from error
        except (ControlStoreTooEarly, ControlStoreClockError) as error:
            raise GenerationStateClockError(
                "control-store time contradicts the coordinator lease"
            ) from error
        expected_keys = {self._head_key, snapshot_key}
        if set(committed) != expected_keys:
            raise GenerationStateCorrupt(
                "generation transaction returned an unexpected committed key set"
            )
        head_entry = committed[self._head_key]
        committed_head = self._decode_head(head_entry)
        if committed_head != head_record:
            raise GenerationStateCorrupt(
                "generation transaction returned an unexpected head record"
            )
        snapshot = self._decode_snapshot(
            committed[snapshot_key],
            expected_generation=snapshot_record.assignment.generation,
        )
        if snapshot.record != snapshot_record:
            raise GenerationStateCorrupt(
                "generation transaction returned an unexpected snapshot record"
            )
        self._validate_head_link(committed_head, head_entry, snapshot)
        return CurrentGeneration(
            snapshot=snapshot,
            head_revision=head_entry.revision,
        )

    def _snapshot_record(
        self,
        lease: HeldCoordinatorLease,
        assignment: RankAssignment,
        *,
        previous_snapshot_digest: str | None,
    ) -> GenerationSnapshotRecord:
        return GenerationSnapshotRecord(
            assignment=assignment,
            previous_snapshot_digest=previous_snapshot_digest,
            coordinator_id=lease.record.coordinator_id,
            lease_id=lease.record.lease_id,
            coordinator_lease_duration_ms=lease.record.lease_duration_ms,
            coordinator_fencing_token=lease.fencing_token,
        )

    def _validate_lease(self, lease: HeldCoordinatorLease) -> None:
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if lease.record.run_id != self._run_id:
            raise ValueError("coordinator lease belongs to another run")
        entry = self._store.get(self._coordinator_lease_key)
        if entry is None or entry.revision != lease.fencing_token:
            raise GenerationStateLeaseLost("coordinator lease changed before generation validation")
        try:
            record = CoordinatorLeaseRecord.from_json(entry.value)
        except (TypeError, ValueError) as error:
            raise GenerationStateCorrupt("coordinator lease is malformed") from error
        if entry.committed_at_unix_ms is None:
            raise GenerationStateCorrupt("coordinator lease has no authoritative grant time")
        if record != lease.record or entry.committed_at_unix_ms != lease.granted_at_unix_ms:
            raise GenerationStateLeaseLost(
                "coordinator lease handle does not match persisted ownership"
            )

    def _validate_assignment(self, assignment: RankAssignment) -> None:
        if not isinstance(assignment, RankAssignment):
            raise TypeError("assignment must be RankAssignment")
        if assignment.run_id != self._run_id:
            raise ValueError("rank assignment belongs to another run")

    def _validate_current(self, current: CurrentGeneration) -> None:
        if not isinstance(current, CurrentGeneration):
            raise TypeError("current must be CurrentGeneration")
        observed = self.current()
        if observed != current:
            raise GenerationStateConflict(
                "current generation does not match the committed generation head"
            )


__all__ = [
    "GenerationStateClockError",
    "GenerationStateConflict",
    "GenerationStateLeaseLost",
    "GenerationStateManager",
]
