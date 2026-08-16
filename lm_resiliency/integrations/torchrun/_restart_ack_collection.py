"""Immutable collections of restart acknowledgements for one intent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointInventoryEvent,
    ProtocolValidationError,
    checkpoint_inventory_digest,
    validate_worker_identity,
)
from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
    PersistedInitialRestartIntentOpen,
    RestartIntentOpening,
)


@dataclass(frozen=True, slots=True)
class RestartAckCollection:
    """One exact active-node acknowledgement observation."""

    opened: RestartIntentOpening
    receipts_by_node_id: Mapping[str, PersistedRestartAck | None]

    def __post_init__(self) -> None:
        if not isinstance(
            self.opened,
            (CommittedInitialRestartIntentOpen, PersistedInitialRestartIntentOpen),
        ):
            raise TypeError("RestartAckCollection.opened must be a restart-intent opening")
        if not isinstance(self.receipts_by_node_id, Mapping):
            raise TypeError("RestartAckCollection.receipts_by_node_id must be a mapping")
        normalized: dict[str, PersistedRestartAck | None] = {}
        for node_id, receipt in self.receipts_by_node_id.items():
            normalized_node_id = _nonempty_string(
                node_id,
                "RestartAckCollection.receipts_by_node_id key",
            )
            if receipt is not None and not isinstance(receipt, PersistedRestartAck):
                raise TypeError(
                    "RestartAckCollection receipt values must be PersistedRestartAck or None"
                )
            normalized[normalized_node_id] = receipt
        active_node_ids = self.active_node_ids
        if set(normalized) != set(active_node_ids):
            raise ValueError(
                "RestartAckCollection receipt keys do not exactly match active generation nodes"
            )
        for node_id, receipt in normalized.items():
            if receipt is None:
                continue
            if not _same_committed_opening(receipt.opened, self.opened):
                raise ValueError(
                    "RestartAckCollection receipt belongs to another restart intent opening"
                )
            if receipt.receipt.acknowledgement.node_id != node_id:
                raise ValueError(
                    "RestartAckCollection receipt is stored under another node identity"
                )
        object.__setattr__(
            self,
            "receipts_by_node_id",
            MappingProxyType({node_id: normalized[node_id] for node_id in active_node_ids}),
        )

    @property
    def active_node_ids(self) -> tuple[str, ...]:
        slot_to_node_id = self.opened.generation_snapshot.record.assignment.slot_to_node_id
        return tuple(dict.fromkeys(slot_to_node_id[slot_id] for slot_id in sorted(slot_to_node_id)))

    @property
    def received_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id
            for node_id in self.active_node_ids
            if self.receipts_by_node_id[node_id] is not None
        )

    @property
    def missing_node_ids(self) -> tuple[str, ...]:
        return tuple(
            node_id for node_id in self.active_node_ids if self.receipts_by_node_id[node_id] is None
        )

    @property
    def successful_node_ids(self) -> tuple[str, ...]:
        successful_node_ids: list[str] = []
        for node_id in self.active_node_ids:
            receipt = self.receipts_by_node_id[node_id]
            if receipt is not None and receipt.receipt.acknowledgement.success:
                successful_node_ids.append(node_id)
        return tuple(successful_node_ids)

    @property
    def failed_node_ids(self) -> tuple[str, ...]:
        failed_node_ids: list[str] = []
        for node_id in self.active_node_ids:
            receipt = self.receipts_by_node_id[node_id]
            if receipt is not None and not receipt.receipt.acknowledgement.success:
                failed_node_ids.append(node_id)
        return tuple(failed_node_ids)


@dataclass(frozen=True, slots=True)
class RestartAckEvidence:
    """Authenticate latest inventory events against one stable collection."""

    collection: RestartAckCollection

    def __post_init__(self) -> None:
        if not isinstance(self.collection, RestartAckCollection):
            raise TypeError("RestartAckEvidence.collection must be RestartAckCollection")

    def authorizes_latest_inventory(
        self,
        event: CheckpointInventoryEvent,
    ) -> bool:
        """Return whether one latest event has exact preparation evidence."""

        if not isinstance(event, CheckpointInventoryEvent):
            raise TypeError("event must be CheckpointInventoryEvent")
        if event.trust != "latest":
            return False
        assignment = self.collection.opened.generation_snapshot.record.assignment
        try:
            validate_worker_identity(event.reporter, assignment)
        except ProtocolValidationError:
            return False
        receipt = self.collection.receipts_by_node_id[event.reporter.node_id]
        if receipt is None:
            return False
        acknowledgement = receipt.receipt.acknowledgement
        return (
            acknowledgement.success
            and acknowledgement.agent_id == event.reporter.agent_id
            and acknowledgement.flushed_step == event.step
            and acknowledgement.inventory_event_digests.get(event.event_id)
            == checkpoint_inventory_digest(event)
        )


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _same_committed_opening(
    left: RestartIntentOpening,
    right: RestartIntentOpening,
) -> bool:
    return (
        left.intent_key == right.intent_key
        and left.intent_entry == right.intent_entry
        and left.intent_head_key == right.intent_head_key
        and left.head_entry == right.head_entry
    )


__all__ = ["RestartAckCollection", "RestartAckEvidence"]
