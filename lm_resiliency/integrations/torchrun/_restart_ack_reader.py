"""Stable reads of persisted restart acknowledgements."""

from __future__ import annotations

import hashlib

from lm_resiliency.integrations.torchrun._agent_registration_history import (
    AgentRegistrationAuthority,
)
from lm_resiliency.integrations.torchrun._agent_registration_history_reader import (
    AgentRegistrationHistory,
    AgentRegistrationHistoryCorrupt,
    AgentRegistrationHistoryError,
    AgentRegistrationHistoryReader,
)
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
    GenerationStateCorrupt,
    GenerationStateError,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckCollection,
)
from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
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


class RestartAckReadError(RuntimeError):
    """Base error for persisted restart-acknowledgement reads."""


class RestartAckReadConflict(RestartAckReadError):
    """Raised when the current restart-intent state changes repeatedly."""


class RestartAckReadCorrupt(RestartAckReadError):
    """Raised when persisted acknowledgement state is contradictory."""


class RestartAckReader:
    """Read one node's receipt for the current restart intent."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        node_id: str,
    ) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._node_id = _nonempty_string(node_id, "node_id")
        self._open_reader = RestartIntentOpenStateReader(
            store,
            run_id=self._run_id,
        )
        self._registration_reader = AgentRegistrationHistoryReader(
            store,
            run_id=self._run_id,
            node_id=self._node_id,
        )
        self._lease_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    def read(self) -> PersistedRestartAck | None:
        """Return the current intent's receipt for this node, if committed."""

        for _ in range(_MAX_READ_ATTEMPTS):
            opened = self._read_open()
            acknowledgement_key = _acknowledgement_key(opened, self._node_id)
            receipt_state = self._receipt_state(acknowledgement_key)
            registration_history = self._read_registration_history()
            lease_history = self._read_lease_history()
            if (
                opened != self._read_open()
                or receipt_state != self._receipt_state(acknowledgement_key)
                or registration_history != self._read_registration_history()
                or lease_history != self._read_lease_history()
            ):
                continue
            receipt_entry = _current_receipt(receipt_state)
            if receipt_entry is None:
                return None
            return self._authenticate(
                receipt_entry,
                opened=opened,
                registration_history=registration_history,
                lease_history=lease_history,
            )
        raise RestartAckReadConflict(
            "restart-acknowledgement dependencies changed repeatedly during read"
        )

    def _authenticate(
        self,
        receipt_entry: ControlStoreEntry,
        *,
        opened: CommittedInitialRestartIntentOpen,
        registration_history: AgentRegistrationHistory,
        lease_history: tuple[CoordinatorLeaseAuthority, ...],
    ) -> PersistedRestartAck:
        try:
            receipt = RestartAckReceiptRecord.from_json(receipt_entry.value)
        except (TypeError, ValueError) as error:
            raise RestartAckReadCorrupt("persisted restart acknowledgement is malformed") from error
        if receipt.acknowledgement.node_id != self._node_id:
            raise RestartAckReadCorrupt("persisted restart acknowledgement belongs to another node")
        registration_authority = _registration_authority(
            registration_history,
            receipt,
        )
        coordinator_authority = _coordinator_authority(
            lease_history,
            receipt_entry,
        )
        try:
            return PersistedRestartAck(
                receipt=receipt,
                receipt_entry=receipt_entry,
                opened=opened,
                registration_authority=registration_authority,
                coordinator_authority=coordinator_authority,
            )
        except (TypeError, ValueError) as error:
            raise RestartAckReadCorrupt(
                "persisted restart acknowledgement contradicts its durable dependencies"
            ) from error

    def _read_open(self) -> CommittedInitialRestartIntentOpen:
        try:
            opened = self._open_reader.read()
        except RestartIntentOpenStateClosed as error:
            raise RestartAckReadConflict(
                "restart intent closed during acknowledgement read"
            ) from error
        except RestartIntentOpenStateCorrupt as error:
            raise RestartAckReadCorrupt("persisted restart-intent opening is corrupt") from error
        except RestartIntentOpenStateError as error:
            raise RestartAckReadConflict(
                "restart-intent opening changed repeatedly during read"
            ) from error
        except (CoordinatorLeaseHistoryCorrupt, GenerationStateCorrupt) as error:
            raise RestartAckReadCorrupt(
                "restart-intent opening dependencies are corrupt"
            ) from error
        except (CoordinatorLeaseHistoryError, GenerationStateError) as error:
            raise RestartAckReadConflict(
                "restart-intent opening dependencies changed repeatedly during read"
            ) from error
        if opened is None:
            raise RestartAckReadConflict("no current restart intent is open")
        return opened

    def _read_registration_history(self) -> AgentRegistrationHistory:
        try:
            return self._registration_reader.read()
        except AgentRegistrationHistoryCorrupt as error:
            raise RestartAckReadCorrupt("agent registration history is corrupt") from error
        except AgentRegistrationHistoryError as error:
            raise RestartAckReadConflict(
                "agent registration history changed repeatedly during read"
            ) from error

    def _read_lease_history(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        try:
            return self._lease_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartAckReadCorrupt("coordinator lease history is corrupt") from error
        except CoordinatorLeaseHistoryError as error:
            raise RestartAckReadConflict(
                "coordinator lease history changed repeatedly during read"
            ) from error

    def _receipt_state(
        self,
        acknowledgement_key: str,
    ) -> tuple[tuple[ControlStoreEntry, ...], ControlStoreEntry | None, bool]:
        return (
            self._store.get_history(acknowledgement_key),
            self._store.get(acknowledgement_key),
            self._store.has_history(acknowledgement_key),
        )


class RestartAckCollectionReadError(RuntimeError):
    """Base error for stable restart-acknowledgement collection reads."""


class RestartAckCollectionReadConflict(RestartAckCollectionReadError):
    """Raised when acknowledgement state changes repeatedly during collection."""


class RestartAckCollectionReadCorrupt(RestartAckCollectionReadError):
    """Raised when persisted acknowledgement state is contradictory."""


class RestartAckCollector:
    """Read one stable acknowledgement-or-absence value for every active node."""

    def __init__(self, store: ControlStore, *, run_id: str) -> None:
        self._store = store
        self._run_id = _nonempty_string(run_id, "run_id")
        self._open_reader = RestartIntentOpenStateReader(
            store,
            run_id=self._run_id,
        )
        self._receipt_readers: dict[str, RestartAckReader] = {}

    def collect(self) -> RestartAckCollection:
        """Return one stable active-node acknowledgement collection."""

        for _ in range(_MAX_READ_ATTEMPTS):
            opened = self._read_open()
            node_ids = _active_node_ids(opened)
            first = self._read_receipts(node_ids)
            if not _same_committed_opening(opened, self._read_open()):
                continue
            second = self._read_receipts(node_ids)
            current_opened = self._read_open()
            if first != second or not _same_committed_opening(opened, current_opened):
                continue
            try:
                return RestartAckCollection(
                    opened=current_opened,
                    receipts_by_node_id=second,
                )
            except (TypeError, ValueError) as error:
                raise RestartAckCollectionReadCorrupt(
                    "persisted acknowledgements contradict the current opening"
                ) from error
        raise RestartAckCollectionReadConflict(
            "restart acknowledgements changed repeatedly during collection"
        )

    def _read_receipts(
        self,
        node_ids: tuple[str, ...],
    ) -> dict[str, PersistedRestartAck | None]:
        receipts: dict[str, PersistedRestartAck | None] = {}
        for node_id in node_ids:
            reader = self._receipt_readers.get(node_id)
            if reader is None:
                reader = RestartAckReader(
                    self._store,
                    run_id=self._run_id,
                    node_id=node_id,
                )
                self._receipt_readers[node_id] = reader
            try:
                receipts[node_id] = reader.read()
            except RestartAckReadCorrupt as error:
                raise RestartAckCollectionReadCorrupt(
                    f"persisted acknowledgement for node {node_id!r} is corrupt"
                ) from error
            except RestartAckReadConflict as error:
                raise RestartAckCollectionReadConflict(
                    f"acknowledgement for node {node_id!r} changed repeatedly"
                ) from error
        return receipts

    def _read_open(self) -> CommittedInitialRestartIntentOpen:
        try:
            opened = self._open_reader.read()
        except RestartIntentOpenStateClosed as error:
            raise RestartAckCollectionReadConflict(
                "restart intent closed during acknowledgement collection"
            ) from error
        except RestartIntentOpenStateCorrupt as error:
            raise RestartAckCollectionReadCorrupt(
                "persisted restart-intent opening is corrupt"
            ) from error
        except RestartIntentOpenStateError as error:
            raise RestartAckCollectionReadConflict(
                "restart-intent opening changed repeatedly during collection"
            ) from error
        except (CoordinatorLeaseHistoryCorrupt, GenerationStateCorrupt) as error:
            raise RestartAckCollectionReadCorrupt(
                "restart-intent opening dependencies are corrupt"
            ) from error
        except (CoordinatorLeaseHistoryError, GenerationStateError) as error:
            raise RestartAckCollectionReadConflict(
                "restart-intent opening dependencies changed repeatedly"
            ) from error
        if opened is None:
            raise RestartAckCollectionReadConflict("no current restart intent is open")
        return opened


def _current_receipt(
    state: tuple[tuple[ControlStoreEntry, ...], ControlStoreEntry | None, bool],
) -> ControlStoreEntry | None:
    history, current, has_history = state
    if bool(history) != has_history:
        raise RestartAckReadCorrupt(
            "restart-acknowledgement history contradicts its durable marker"
        )
    if current is None:
        if history:
            raise RestartAckReadCorrupt("restart acknowledgement was deleted after commit")
        return None
    if len(history) != 1 or history[0] != current:
        raise RestartAckReadCorrupt("restart acknowledgement is not one immutable creation")
    return current


def _registration_authority(
    history: AgentRegistrationHistory,
    receipt: RestartAckReceiptRecord,
) -> AgentRegistrationAuthority:
    matches = tuple(
        authority
        for authority in history.authorities
        if authority.registration == receipt.authenticated_registration
    )
    if len(matches) != 1:
        raise RestartAckReadCorrupt(
            "restart acknowledgement has no unique agent-registration authority"
        )
    return matches[0]


def _coordinator_authority(
    history: tuple[CoordinatorLeaseAuthority, ...],
    receipt_entry: ControlStoreEntry,
) -> CoordinatorLeaseAuthority:
    matches = tuple(
        authority
        for authority in history
        if (
            receipt_entry.guard_revision == authority.lease.fencing_token
            and receipt_entry.guard_value_digest
            == hashlib.sha256(authority.lease.record.to_json()).hexdigest()
            and receipt_entry.guard_committed_at_unix_ms == authority.lease.granted_at_unix_ms
            and receipt_entry.guard_mutation_sequence == authority.mutation_sequence
            and receipt_entry.guard_value_sequence == authority.value_sequence
            and receipt_entry.guard_lifetime_sequence == authority.lifetime_sequence
        )
    )
    if len(matches) != 1:
        raise RestartAckReadCorrupt(
            "restart acknowledgement has no unique coordinator-lease authority"
        )
    return matches[0]


def _acknowledgement_key(
    opened: CommittedInitialRestartIntentOpen,
    node_id: str,
) -> str:
    node_digest = hashlib.sha256(node_id.encode("utf-8")).hexdigest()
    return f"{opened.prepared.intent_key}/acknowledgements/{node_digest}"


def _active_node_ids(
    opened: CommittedInitialRestartIntentOpen,
) -> tuple[str, ...]:
    slot_to_node_id = opened.prepared.current.snapshot.record.assignment.slot_to_node_id
    return tuple(slot_to_node_id[slot_id] for slot_id in sorted(slot_to_node_id))


def _same_committed_opening(
    left: CommittedInitialRestartIntentOpen,
    right: CommittedInitialRestartIntentOpen,
) -> bool:
    return (
        left.prepared.intent_key == right.prepared.intent_key
        and left.intent_entry == right.intent_entry
        and left.prepared.intent_head_key == right.prepared.intent_head_key
        and left.head_entry == right.head_entry
    )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = [
    "RestartAckCollectionReadConflict",
    "RestartAckCollectionReadCorrupt",
    "RestartAckCollectionReadError",
    "RestartAckCollector",
    "RestartAckReadConflict",
    "RestartAckReadCorrupt",
    "RestartAckReadError",
    "RestartAckReader",
]
