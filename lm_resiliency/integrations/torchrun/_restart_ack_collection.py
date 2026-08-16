"""Immutable collections of restart acknowledgements for one intent."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)


@dataclass(frozen=True, slots=True)
class RestartAckCollection:
    """One exact active-node acknowledgement observation."""

    opened: CommittedInitialRestartIntentOpen
    receipts_by_node_id: Mapping[str, PersistedRestartAck | None]

    def __post_init__(self) -> None:
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError("RestartAckCollection.opened must be CommittedInitialRestartIntentOpen")
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
            if receipt.opened != self.opened:
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
        slot_to_node_id = self.opened.prepared.current.snapshot.record.assignment.slot_to_node_id
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


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


__all__ = ["RestartAckCollection"]
