"""Stable reads of authenticated restart-acknowledgement state."""

from __future__ import annotations

from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistory,
    AgentRegistrationHistoryCorrupt,
    AgentRegistrationHistoryError,
    AgentRegistrationHistoryReader,
)
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
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    AuthenticatedRestartAckState,
    RestartAckReceiptRecord,
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

_MAX_READ_ATTEMPTS = 8


class RestartAckStateReaderError(RuntimeError):
    """Base error for restart-acknowledgement dependency reads."""


class RestartAckStateConflict(RestartAckStateReaderError):
    """Raised when the restart-intent state changes or is no longer open."""


class RestartAckStateRegistrationLost(RestartAckStateReaderError):
    """Raised when the authenticated agent registration is no longer current."""


class RestartAckStateLeaseLost(RestartAckStateReaderError):
    """Raised when the supplied coordinator lease is no longer current."""


class RestartAckStateCorrupt(RestartAckStateReaderError):
    """Raised when persisted acknowledgement dependencies are contradictory."""


class RestartAckStateReader:
    """Read one stable, authenticated restart-acknowledgement state."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._open_reader = RestartIntentOpenStateReader(
            store,
            run_id=self._run_id,
        )
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    def read(
        self,
        receipt: RestartAckReceiptRecord,
        lease: HeldCoordinatorLease,
    ) -> AuthenticatedRestartAckState:
        """Return one stable authenticated dependency snapshot."""

        self._validate_inputs(receipt, lease)
        registration_reader = self._registration_history_reader(receipt.acknowledgement.node_id)
        for _ in range(_MAX_READ_ATTEMPTS):
            opened = self._read_open()
            registration_history = self._read_registration_history(registration_reader)
            lease_history = self._read_lease_history()
            if (
                opened != self._read_open()
                or registration_history != self._read_registration_history(registration_reader)
                or lease_history != self._read_lease_history()
            ):
                continue
            return self._authenticate(
                receipt,
                lease,
                opened=opened,
                registration_history=registration_history,
                lease_history=lease_history,
            )
        raise RestartAckStateConflict(
            "restart-acknowledgement dependencies changed repeatedly during read"
        )

    def _authenticate(
        self,
        receipt: RestartAckReceiptRecord,
        lease: HeldCoordinatorLease,
        *,
        opened: CommittedInitialRestartIntentOpen | None,
        registration_history: AgentRegistrationHistory,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> AuthenticatedRestartAckState:
        if opened is None:
            raise RestartAckStateConflict("no current restart intent is open")
        if receipt.intent_record != opened.prepared.record:
            raise RestartAckStateConflict(
                "restart acknowledgement does not answer the current restart intent"
            )
        registration = registration_history.current
        if registration is None or registration != receipt.authenticated_registration:
            raise RestartAckStateRegistrationLost(
                "authenticated agent registration is no longer current"
            )
        registration_authority = registration_history.authorities[-1]
        if not lease_history or lease_history[-1].lease != lease:
            raise RestartAckStateLeaseLost("coordinator lease is not the live durable authority")
        try:
            return AuthenticatedRestartAckState(
                receipt=receipt,
                opened=opened,
                registration_authority=registration_authority,
                coordinator_authority=lease_history[-1],
            )
        except (TypeError, ValueError) as error:
            raise RestartAckStateCorrupt(
                "restart acknowledgement contradicts persisted state"
            ) from error

    def _read_open(self) -> CommittedInitialRestartIntentOpen | None:
        try:
            return self._open_reader.read()
        except RestartIntentOpenStateClosed as error:
            raise RestartAckStateConflict(
                "restart intent closed before acknowledgement authentication"
            ) from error
        except RestartIntentOpenStateCorrupt as error:
            raise RestartAckStateCorrupt("persisted restart-intent opening is corrupt") from error
        except RestartIntentOpenStateError as error:
            raise RestartAckStateConflict(
                "restart-intent opening changed repeatedly during read"
            ) from error
        except (CoordinatorLeaseHistoryCorrupt, GenerationStateCorrupt) as error:
            raise RestartAckStateCorrupt(
                "restart-intent opening dependencies are corrupt"
            ) from error
        except (CoordinatorLeaseHistoryError, GenerationStateError) as error:
            raise RestartAckStateConflict(
                "restart-intent opening dependencies changed repeatedly during read"
            ) from error

    @staticmethod
    def _read_registration_history(
        reader: AgentRegistrationHistoryReader,
    ) -> AgentRegistrationHistory:
        try:
            return reader.read()
        except AgentRegistrationHistoryCorrupt as error:
            raise RestartAckStateCorrupt("agent registration history is corrupt") from error
        except AgentRegistrationHistoryError as error:
            raise RestartAckStateConflict(
                "agent registration history changed repeatedly during read"
            ) from error

    def _registration_history_reader(
        self,
        node_id: str,
    ) -> AgentRegistrationHistoryReader:
        return AgentRegistrationHistoryReader(
            self._store,
            run_id=self._run_id,
            node_id=node_id,
        )

    def _read_lease_history(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        try:
            return self._lease_history_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartAckStateCorrupt("coordinator lease history is corrupt") from error
        except CoordinatorLeaseHistoryError as error:
            raise RestartAckStateConflict(
                "coordinator lease history changed repeatedly during read"
            ) from error

    def _validate_inputs(
        self,
        receipt: RestartAckReceiptRecord,
        lease: HeldCoordinatorLease,
    ) -> None:
        if not isinstance(receipt, RestartAckReceiptRecord):
            raise TypeError("receipt must be RestartAckReceiptRecord")
        if not isinstance(lease, HeldCoordinatorLease):
            raise TypeError("lease must be HeldCoordinatorLease")
        if receipt.acknowledgement.run_id != self._run_id:
            raise ValueError("restart acknowledgement belongs to another run")
        if lease.record.run_id != self._run_id:
            raise ValueError("coordinator lease belongs to another run")


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "RestartAckStateConflict",
    "RestartAckStateCorrupt",
    "RestartAckStateLeaseLost",
    "RestartAckStateReader",
    "RestartAckStateReaderError",
    "RestartAckStateRegistrationLost",
]
