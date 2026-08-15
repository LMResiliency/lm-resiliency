"""Fail-closed reads of the current initial restart-intent opening."""

from __future__ import annotations

import hashlib

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStore,
    ControlStoreEntry,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateCorrupt,
    GenerationStateReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_records import (
    PreparedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentHeadRecord,
    RestartIntentRecord,
)

_CONTROL_PREFIX = "lm_resiliency/torchrun/v1"
_MAX_READ_ATTEMPTS = 8


class RestartIntentOpenStateError(RuntimeError):
    """Base error for persisted restart-intent opening reads."""


class RestartIntentOpenStateCorrupt(RestartIntentOpenStateError):
    """Raised when persisted opening state is malformed or contradictory."""


class RestartIntentOpenStateClosed(RestartIntentOpenStateError):
    """Raised when a closed marker prevents reconstructing a current opening.

    This error does not authenticate the closure. Callers must use the
    lifecycle reader before treating the intent as validly closed.
    """


class RestartIntentOpenStateReader:
    """Read and verify the current initial restart-intent opening."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        run_digest = hashlib.sha256(self._run_id.encode("utf-8")).hexdigest()
        self._run_prefix = f"{_CONTROL_PREFIX}/runs/{run_digest}"
        self._intent_head_key = f"{self._run_prefix}/restart-intent-head"
        self._lifecycle_head_key = f"{self._run_prefix}/restart-intent-lifecycle-head"
        self._generation_reader = GenerationStateReader(store, run_id=self._run_id)
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

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

    def read(self) -> CommittedInitialRestartIntentOpen | None:
        """Return one stable, verified opening or ``None`` before first open."""

        for _ in range(_MAX_READ_ATTEMPTS):
            head_entry = self._store.get(self._intent_head_key)
            head_has_history = self._store.has_history(self._intent_head_key)
            lifecycle_entry = self._store.get(self._lifecycle_head_key)
            lifecycle_has_history = self._store.has_history(self._lifecycle_head_key)
            if not self._state_is_stable(
                head_entry,
                head_has_history,
                lifecycle_entry,
                lifecycle_has_history,
            ):
                continue
            if head_entry is None:
                if head_has_history:
                    raise RestartIntentOpenStateCorrupt(
                        "current restart-intent head disappeared after lifecycle creation"
                    )
                if lifecycle_entry is not None or lifecycle_has_history:
                    raise RestartIntentOpenStateCorrupt(
                        "restart-intent lifecycle exists without current-head history"
                    )
                return None
            if not head_has_history:
                raise RestartIntentOpenStateCorrupt(
                    "live restart-intent head has no durable history"
                )
            head = self._decode_head(head_entry)
            if isinstance(head, RestartIntentClosedHeadRecord):
                raise RestartIntentOpenStateClosed(
                    "current restart-intent head is a closed marker; "
                    "closure evidence is not verified by this reader"
                )
            if lifecycle_entry is not None or lifecycle_has_history:
                raise RestartIntentOpenStateCorrupt(
                    "open restart-intent head coexists with lifecycle state"
                )
            intent_key = self.intent_key(head.intent_id)
            intent_entry = self._store.get(intent_key)
            if intent_entry is None:
                raise RestartIntentOpenStateCorrupt(
                    "current restart-intent head references a missing intent"
                )
            intent_has_history = self._store.has_history(intent_key)
            if not intent_has_history:
                raise RestartIntentOpenStateCorrupt(
                    "live immutable restart intent has no durable history"
                )
            try:
                generation_result = self._generation_reader.current_with_history()
                lease_history = self._lease_history_reader.read()
            except (
                CoordinatorLeaseHistoryCorrupt,
                GenerationStateCorrupt,
            ) as error:
                raise RestartIntentOpenStateCorrupt(
                    "restart-intent opening dependencies are corrupt"
                ) from error
            if not self._state_is_stable(
                head_entry,
                head_has_history,
                lifecycle_entry,
                lifecycle_has_history,
                intent_key=intent_key,
                intent_entry=intent_entry,
                intent_has_history=intent_has_history,
            ):
                continue
            if generation_result is None:
                raise RestartIntentOpenStateCorrupt(
                    "restart intent exists without committed generation state"
                )
            current, generation_history = generation_result
            try:
                record = RestartIntentRecord.from_json(intent_entry.value)
            except (TypeError, ValueError) as error:
                raise RestartIntentOpenStateCorrupt(
                    "immutable restart-intent record is malformed"
                ) from error
            if intent_entry.value != record.to_json():
                raise RestartIntentOpenStateCorrupt(
                    "immutable restart-intent record is noncanonical"
                )
            if (
                head.run_id != self._run_id
                or record.intent.run_id != self._run_id
                or head.generation != record.intent.generation
                or head.intent_id != record.intent.intent_id
                or head.intent_digest != record.digest
            ):
                raise RestartIntentOpenStateCorrupt(
                    "current restart-intent head does not identify its immutable record"
                )
            if (
                current.snapshot.record.assignment.generation != record.intent.generation
                or current.snapshot.record.digest != record.generation_snapshot_digest
            ):
                raise RestartIntentOpenStateCorrupt(
                    "current generation does not match the open restart intent"
                )
            opening_authority = self._opening_authority(
                record,
                intent_entry,
                lease_history,
            )
            try:
                prepared = PreparedInitialRestartIntentOpen(
                    record=record,
                    head=head,
                    current=current,
                    lease=opening_authority.lease,
                    intent_key=intent_key,
                    intent_head_key=self._intent_head_key,
                    lifecycle_head_key=self._lifecycle_head_key,
                    coordinator_lease_key=self._generation_reader.coordinator_lease_key,
                    generation_head_key=self._generation_reader.head_key,
                    generation_snapshot_key=self._generation_reader.snapshot_key(
                        record.intent.generation
                    ),
                    coordinator_lease_transaction_sequence=(opening_authority.transaction_sequence),
                    coordinator_lease_mutation_sequence=(opening_authority.mutation_sequence),
                    coordinator_lease_value_sequence=(opening_authority.value_sequence),
                    coordinator_lease_lifetime_sequence=(opening_authority.lifetime_sequence),
                    generation_lease_id_history=tuple(
                        snapshot.record.lease_id for snapshot in generation_history
                    ),
                    generation_fencing_token_history=tuple(
                        snapshot.record.coordinator_fencing_token for snapshot in generation_history
                    ),
                    not_before_unix_ms=_commit_time(intent_entry),
                    deadline_unix_ms=min(
                        opening_authority.lease.expires_at_unix_ms,
                        record.intent.prepare_deadline_unix_ms,
                    ),
                )
                return CommittedInitialRestartIntentOpen(
                    prepared=prepared,
                    intent_entry=intent_entry,
                    head_entry=head_entry,
                )
            except (TypeError, ValueError) as error:
                raise RestartIntentOpenStateCorrupt(
                    "persisted restart-intent opening is contradictory"
                ) from error
        raise RestartIntentOpenStateError("restart-intent opening changed repeatedly during read")

    def _decode_head(
        self,
        entry: ControlStoreEntry,
    ) -> RestartIntentHeadRecord | RestartIntentClosedHeadRecord:
        try:
            head = RestartIntentHeadRecord.from_json(entry.value)
        except (TypeError, ValueError) as open_error:
            try:
                closed = RestartIntentClosedHeadRecord.from_json(entry.value)
            except (TypeError, ValueError) as closed_error:
                raise RestartIntentOpenStateCorrupt(
                    "current restart-intent head is malformed"
                ) from closed_error
            if closed.run_id != self._run_id:
                raise RestartIntentOpenStateCorrupt(
                    "closed restart-intent head belongs to another run"
                ) from open_error
            if entry.value != closed.to_json():
                raise RestartIntentOpenStateCorrupt("closed restart-intent head is noncanonical")
            return closed
        if head.run_id != self._run_id:
            raise RestartIntentOpenStateCorrupt(
                "current restart-intent head belongs to another run"
            )
        return head

    def _opening_authority(
        self,
        record: RestartIntentRecord,
        entry: ControlStoreEntry,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> CoordinatorLeaseAuthority:
        provenance = (
            entry.guard_key,
            entry.guard_revision,
            entry.guard_value_digest,
            entry.guard_mutation_sequence,
            entry.guard_value_sequence,
            entry.guard_lifetime_sequence,
            entry.guard_committed_at_unix_ms,
        )
        if any(value is None for value in provenance):
            raise RestartIntentOpenStateCorrupt(
                "immutable restart intent has incomplete lease provenance"
            )
        (
            guard_key,
            guard_revision,
            guard_value_digest,
            guard_mutation_sequence,
            guard_value_sequence,
            guard_lifetime_sequence,
            guard_committed_at_unix_ms,
        ) = provenance
        if (
            guard_key != self._generation_reader.coordinator_lease_key
            or guard_revision != record.coordinator_fencing_token
            or guard_value_digest != record.coordinator_lease_digest
        ):
            raise RestartIntentOpenStateCorrupt(
                "immutable restart intent has invalid lease provenance"
            )
        lease_record = CoordinatorLeaseRecord(
            run_id=record.intent.run_id,
            coordinator_id=record.coordinator_id,
            lease_id=record.lease_id,
            lease_duration_ms=record.coordinator_lease_duration_ms,
        )
        matches = tuple(
            authority
            for authority in lease_history
            if (
                authority.lease.record == lease_record
                and authority.lease.fencing_token == guard_revision
                and authority.lease.granted_at_unix_ms == guard_committed_at_unix_ms
                and authority.mutation_sequence == guard_mutation_sequence
                and authority.value_sequence == guard_value_sequence
                and authority.lifetime_sequence == guard_lifetime_sequence
            )
        )
        if len(matches) != 1:
            raise RestartIntentOpenStateCorrupt(
                "opening coordinator lease is absent from durable lease history"
            )
        return matches[0]

    def _state_is_stable(
        self,
        head_entry: ControlStoreEntry | None,
        head_has_history: bool,
        lifecycle_entry: ControlStoreEntry | None,
        lifecycle_has_history: bool,
        *,
        intent_key: str | None = None,
        intent_entry: ControlStoreEntry | None = None,
        intent_has_history: bool | None = None,
    ) -> bool:
        if (
            self._store.get(self._intent_head_key) != head_entry
            or self._store.has_history(self._intent_head_key) != head_has_history
            or self._store.get(self._lifecycle_head_key) != lifecycle_entry
            or self._store.has_history(self._lifecycle_head_key) != lifecycle_has_history
        ):
            return False
        if intent_key is None:
            return True
        return (
            self._store.get(intent_key) == intent_entry
            and self._store.has_history(intent_key) == intent_has_history
        )


def _commit_time(entry: ControlStoreEntry) -> int:
    if entry.committed_at_unix_ms is None:
        raise RestartIntentOpenStateCorrupt("restart-intent entry has no authoritative commit time")
    return entry.committed_at_unix_ms


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "RestartIntentOpenStateClosed",
    "RestartIntentOpenStateCorrupt",
    "RestartIntentOpenStateError",
    "RestartIntentOpenStateReader",
]
