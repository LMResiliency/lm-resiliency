"""Authenticated, non-mutating preparation of restart acknowledgements."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Callable

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import HeldCoordinatorLease
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    AuthenticatedRestartAckState,
    PreparedRestartAckWrite,
    RestartAckReceiptRecord,
    RestartAckWriteRecords,
)
from lm_resiliency.integrations.torchrun._restart_ack_state_reader import (
    RestartAckStateConflict,
    RestartAckStateCorrupt,
    RestartAckStateLeaseLost,
    RestartAckStateReader,
    RestartAckStateRegistrationLost,
)


class RestartAckPreparationError(RuntimeError):
    """Base error for restart-acknowledgement preparation."""


class RestartAckPreparationConflict(RestartAckPreparationError):
    """Raised when the restart-intent state changes or is no longer open."""


class RestartAckPreparationRegistrationLost(RestartAckPreparationError):
    """Raised when the authenticated agent registration changed or expired."""


class RestartAckPreparationLeaseLost(RestartAckPreparationError):
    """Raised when the coordinator lease changed or expired."""


class RestartAckPreparationDeadlineElapsed(RestartAckPreparationError):
    """Raised when the restart-intent preparation deadline elapsed."""


class RestartAckPreparationClockError(RestartAckPreparationError):
    """Raised when the coordinator preparation clock is invalid."""


class RestartAckPreparationCorrupt(RestartAckPreparationError):
    """Raised when persisted acknowledgement dependencies are contradictory."""


class RestartAckPreparer:
    """Add a bounded commit window to authenticated acknowledgement state."""

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
        self._state_reader = RestartAckStateReader(store, run_id=self._run_id)

    def prepare(
        self,
        receipt: RestartAckReceiptRecord,
        lease: HeldCoordinatorLease,
    ) -> PreparedRestartAckWrite:
        """Return authenticated, non-mutating inputs for one receipt write."""

        state = self._read_state(receipt, lease)
        now_unix_ms = self._now_unix_ms()
        authoritative_times = (
            state.receipt.received_at_unix_ms,
            state.opened.committed_at_unix_ms,
            state.registration.granted_at_unix_ms,
            state.coordinator_authority.lease.granted_at_unix_ms,
        )
        if now_unix_ms < max(authoritative_times):
            raise RestartAckPreparationClockError(
                "restart-acknowledgement preparation clock precedes authoritative state"
            )
        if now_unix_ms >= state.registration.expires_at_unix_ms:
            raise RestartAckPreparationRegistrationLost(
                "authenticated agent registration expired before preparation"
            )
        coordinator_lease = state.coordinator_authority.lease
        if now_unix_ms >= coordinator_lease.expires_at_unix_ms:
            raise RestartAckPreparationLeaseLost(
                "coordinator lease expired before acknowledgement preparation"
            )
        intent_deadline = state.receipt.intent_record.intent.prepare_deadline_unix_ms
        if now_unix_ms >= intent_deadline:
            raise RestartAckPreparationDeadlineElapsed(
                "restart-intent preparation deadline elapsed before acknowledgement preparation"
            )
        try:
            records = _write_records(state)
            return PreparedRestartAckWrite(
                records=records,
                registration_authority=state.registration_authority,
                coordinator_authority=state.coordinator_authority,
                not_before_unix_ms=now_unix_ms,
                deadline_unix_ms=min(
                    state.registration.expires_at_unix_ms,
                    coordinator_lease.expires_at_unix_ms,
                    intent_deadline,
                ),
            )
        except (TypeError, ValueError) as error:
            raise RestartAckPreparationCorrupt(
                "persisted state cannot authorize the restart acknowledgement"
            ) from error

    def _read_state(
        self,
        receipt: RestartAckReceiptRecord,
        lease: HeldCoordinatorLease,
    ) -> AuthenticatedRestartAckState:
        try:
            return self._state_reader.read(receipt, lease)
        except RestartAckStateRegistrationLost as error:
            raise RestartAckPreparationRegistrationLost(str(error)) from error
        except RestartAckStateLeaseLost as error:
            raise RestartAckPreparationLeaseLost(str(error)) from error
        except RestartAckStateConflict as error:
            raise RestartAckPreparationConflict(str(error)) from error
        except RestartAckStateCorrupt as error:
            raise RestartAckPreparationCorrupt(str(error)) from error

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            try:
                now_unix_ms = _positive_integer(
                    self._clock(),
                    "restart-acknowledgement preparation clock",
                )
            except (TypeError, ValueError) as error:
                raise RestartAckPreparationClockError(
                    "restart-acknowledgement preparation clock is invalid"
                ) from error
            if now_unix_ms < self._last_now_unix_ms:
                raise RestartAckPreparationClockError(
                    "restart-acknowledgement preparation clock moved backward"
                )
            self._last_now_unix_ms = now_unix_ms
            return now_unix_ms


def _write_records(state: AuthenticatedRestartAckState) -> RestartAckWriteRecords:
    acknowledgement = state.receipt.acknowledgement
    node_digest = hashlib.sha256(acknowledgement.node_id.encode("utf-8")).hexdigest()
    return RestartAckWriteRecords(
        receipt=state.receipt,
        opened=state.opened,
        acknowledgement_key=(f"{state.opened.prepared.intent_key}/acknowledgements/{node_digest}"),
        agent_registration_key=agent_registration_key(
            acknowledgement.run_id,
            acknowledgement.node_id,
        ),
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
    "RestartAckPreparationClockError",
    "RestartAckPreparationConflict",
    "RestartAckPreparationCorrupt",
    "RestartAckPreparationDeadlineElapsed",
    "RestartAckPreparationError",
    "RestartAckPreparationLeaseLost",
    "RestartAckPreparationRegistrationLost",
    "RestartAckPreparer",
]
