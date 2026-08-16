"""Stable multi-node collections of persisted restart acknowledgements."""

from __future__ import annotations

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
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
from lm_resiliency.integrations.torchrun._restart_ack_reader import (
    RestartAckReadConflict,
    RestartAckReadCorrupt,
    RestartAckReader,
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
]
