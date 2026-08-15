"""Authenticated preparation of the first restart-intent closure."""

from __future__ import annotations

import threading
from collections.abc import Callable

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    GenerationStateCorrupt,
    GenerationStateError,
)
from lm_resiliency.integrations.torchrun._restart_intent_close import (
    PreparedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_reader import (
    RestartIntentOpenStateClosed,
    RestartIntentOpenStateCorrupt,
    RestartIntentOpenStateError,
    RestartIntentOpenStateReader,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
)

_MAX_READ_ATTEMPTS = 8


class RestartIntentClosurePreparationError(RuntimeError):
    """Base error for preparing the first restart-intent closure."""


class RestartIntentClosurePreparationConflict(RestartIntentClosurePreparationError):
    """Raised when the current restart-intent state changes or is not open."""


class RestartIntentClosurePreparationLeaseLost(RestartIntentClosurePreparationError):
    """Raised when the supplied closing lease is stale or expired."""


class RestartIntentClosurePreparationCorrupt(RestartIntentClosurePreparationError):
    """Raised when persisted opening or lease history is contradictory."""


class RestartIntentClosurePreparationClockError(RestartIntentClosurePreparationError):
    """Raised when the coordinator preparation clock moves backward."""


class RestartIntentClosurePreparer:
    """Read and authenticate inputs for the first closure transaction."""

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
        self._open_reader = RestartIntentOpenStateReader(store, run_id=self._run_id)
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    def prepare_initial_closure(
        self,
        lease: HeldCoordinatorLease,
    ) -> PreparedInitialRestartIntentClosure:
        """Return authenticated, non-mutating inputs for the first closure."""

        self._validate_lease_type(lease)
        opened, lease_history = self._read_stable_inputs()
        if opened is None:
            raise RestartIntentClosurePreparationConflict("no current restart intent is open")
        if not lease_history or lease_history[-1].lease != lease:
            raise RestartIntentClosurePreparationLeaseLost(
                "closing coordinator lease is not the live durable authority"
            )
        opening_authority = _opening_authority(opened)
        try:
            opening_index = lease_history.index(opening_authority)
        except ValueError as error:
            raise RestartIntentClosurePreparationCorrupt(
                "opening coordinator lease is absent from durable lease history"
            ) from error
        now_unix_ms = self._now_unix_ms()
        if now_unix_ms < lease.granted_at_unix_ms:
            raise RestartIntentClosurePreparationClockError(
                "closure preparation clock precedes the authoritative lease grant"
            )
        not_before_unix_ms = max(
            opened.committed_at_unix_ms,
            lease.granted_at_unix_ms,
            now_unix_ms,
        )
        if not_before_unix_ms >= lease.expires_at_unix_ms:
            raise RestartIntentClosurePreparationLeaseLost(
                "closing coordinator lease expired before closure preparation"
            )
        records = _closure_records(opened, lease)
        try:
            return PreparedInitialRestartIntentClosure(
                records=records,
                lease_authority_chain=lease_history[opening_index:],
                not_before_unix_ms=not_before_unix_ms,
                deadline_unix_ms=lease.expires_at_unix_ms,
            )
        except ValueError as error:
            raise RestartIntentClosurePreparationCorrupt(
                "persisted lease history cannot authorize restart-intent closure"
            ) from error

    def _read_stable_inputs(
        self,
    ) -> tuple[
        CommittedInitialRestartIntentOpen | None,
        tuple[CoordinatorLeaseAuthority, ...],
    ]:
        for _ in range(_MAX_READ_ATTEMPTS):
            opened = self._read_open()
            lease_history = self._read_lease_history()
            if opened == self._read_open() and lease_history == self._read_lease_history():
                return opened, lease_history
        raise RestartIntentClosurePreparationConflict(
            "restart-intent closure inputs changed repeatedly during preparation"
        )

    def _read_open(self) -> CommittedInitialRestartIntentOpen | None:
        try:
            return self._open_reader.read()
        except RestartIntentOpenStateClosed as error:
            raise RestartIntentClosurePreparationConflict(
                "restart intent is already closed"
            ) from error
        except RestartIntentOpenStateCorrupt as error:
            raise RestartIntentClosurePreparationCorrupt(
                "persisted restart-intent opening is corrupt"
            ) from error
        except RestartIntentOpenStateError as error:
            raise RestartIntentClosurePreparationConflict(
                "restart-intent opening changed repeatedly during preparation"
            ) from error
        except (CoordinatorLeaseHistoryCorrupt, GenerationStateCorrupt) as error:
            raise RestartIntentClosurePreparationCorrupt(
                "restart-intent opening dependencies are corrupt"
            ) from error
        except (CoordinatorLeaseHistoryError, GenerationStateError) as error:
            raise RestartIntentClosurePreparationConflict(
                "restart-intent opening dependencies changed repeatedly during preparation"
            ) from error

    def _read_lease_history(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        try:
            return self._lease_history_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartIntentClosurePreparationCorrupt(
                "coordinator lease history is corrupt"
            ) from error
        except CoordinatorLeaseHistoryError as error:
            raise RestartIntentClosurePreparationConflict(
                "coordinator lease history changed repeatedly during preparation"
            ) from error

    def _validate_lease_type(self, lease: HeldCoordinatorLease) -> None:
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if lease.record.run_id != self._run_id:
            raise ValueError("coordinator lease belongs to another run")

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            now_unix_ms = _positive_integer(
                self._clock(),
                "restart-intent closure preparation clock",
            )
            if now_unix_ms < self._last_now_unix_ms:
                raise RestartIntentClosurePreparationClockError(
                    "restart-intent closure preparation clock moved backward"
                )
            self._last_now_unix_ms = now_unix_ms
            return now_unix_ms


def _opening_authority(
    opened: CommittedInitialRestartIntentOpen,
) -> CoordinatorLeaseAuthority:
    prepared = opened.prepared
    return CoordinatorLeaseAuthority(
        lease=prepared.lease,
        transaction_sequence=prepared.coordinator_lease_transaction_sequence,
        mutation_sequence=prepared.coordinator_lease_mutation_sequence,
        value_sequence=prepared.coordinator_lease_value_sequence,
        lifetime_sequence=prepared.coordinator_lease_lifetime_sequence,
    )


def _closure_records(
    opened: CommittedInitialRestartIntentOpen,
    lease: HeldCoordinatorLease,
) -> InitialRestartIntentClosureRecords:
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=opened.prepared.head,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    lifecycle_head = RestartIntentLifecycleHeadRecord(
        run_id=opened.prepared.record.intent.run_id,
        closure_index=1,
        generation=opened.prepared.record.intent.generation,
        intent_id=opened.prepared.record.intent.intent_id,
        lifecycle_digest=lifecycle.digest,
    )
    run_prefix = opened.prepared.intent_head_key.rsplit("/", 1)[0]
    return InitialRestartIntentClosureRecords(
        opened=opened,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=RestartIntentClosedHeadRecord(
            run_id=lifecycle_head.run_id,
            closure_index=lifecycle_head.closure_index,
            generation=lifecycle_head.generation,
            intent_id=lifecycle_head.intent_id,
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=opened.prepared.intent_key,
        intent_head_key=opened.prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/1",
        lifecycle_head_key=opened.prepared.lifecycle_head_key,
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
    "RestartIntentClosurePreparationClockError",
    "RestartIntentClosurePreparationConflict",
    "RestartIntentClosurePreparationCorrupt",
    "RestartIntentClosurePreparationError",
    "RestartIntentClosurePreparationLeaseLost",
    "RestartIntentClosurePreparer",
]
