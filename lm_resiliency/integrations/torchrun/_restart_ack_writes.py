"""Immutable transaction records for one restart acknowledgement receipt."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from lm_resiliency.integrations.torchrun._agent_registration import (
    agent_registration_key,
)
from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    CommittedInitialRestartIntentOpen,
)


@dataclass(frozen=True, slots=True)
class RestartAckWriteRecords:
    """Immutable receipt write and read conditions for one active node."""

    receipt: RestartAckReceiptRecord
    opened: CommittedInitialRestartIntentOpen
    acknowledgement_key: str
    agent_registration_key: str

    def __post_init__(self) -> None:
        if not isinstance(self.receipt, RestartAckReceiptRecord):
            raise TypeError("RestartAckWriteRecords.receipt must be RestartAckReceiptRecord")
        if not isinstance(self.opened, CommittedInitialRestartIntentOpen):
            raise TypeError(
                "RestartAckWriteRecords.opened must be CommittedInitialRestartIntentOpen"
            )
        if self.receipt.intent_record != self.opened.prepared.record:
            raise ValueError(
                "RestartAckWriteRecords receipt does not answer its committed restart intent"
            )
        if self.receipt.received_at_unix_ms < self.opened.committed_at_unix_ms:
            raise ValueError("RestartAckWriteRecords receipt predates its committed restart intent")
        acknowledgement = self.receipt.acknowledgement
        active_nodes = set(
            self.opened.prepared.current.snapshot.record.assignment.slot_to_node_id.values()
        )
        if acknowledgement.node_id not in active_nodes:
            raise ValueError(
                "RestartAckWriteRecords acknowledgement node is not active in its generation"
            )
        expected_registration_key = agent_registration_key(
            acknowledgement.run_id,
            acknowledgement.node_id,
        )
        if self.agent_registration_key != expected_registration_key:
            raise ValueError("RestartAckWriteRecords.agent_registration_key is not canonical")
        node_digest = hashlib.sha256(acknowledgement.node_id.encode("utf-8")).hexdigest()
        expected_acknowledgement_key = (
            f"{self.opened.prepared.intent_key}/acknowledgements/{node_digest}"
        )
        if self.acknowledgement_key != expected_acknowledgement_key:
            raise ValueError("RestartAckWriteRecords.acknowledgement_key is not canonical")
        keys = {
            self.acknowledgement_key,
            self.agent_registration_key,
            self.opened.prepared.intent_key,
            self.opened.prepared.intent_head_key,
        }
        if len(keys) != 4:
            raise ValueError("RestartAckWriteRecords transaction key roles must be distinct")

    @property
    def writes(self) -> Mapping[str, ControlStoreWrite]:
        return MappingProxyType(
            {
                self.acknowledgement_key: ControlStoreWrite(
                    expected_revision=None,
                    value=self.receipt.to_json(),
                    require_never_created=True,
                )
            }
        )

    @property
    def conditions(self) -> Mapping[str, int]:
        return MappingProxyType(
            {
                self.opened.prepared.intent_key: self.opened.intent_entry.revision,
                self.opened.prepared.intent_head_key: self.opened.head_entry.revision,
                self.agent_registration_key: self.receipt.registration_fencing_token,
            }
        )


__all__ = ["RestartAckWriteRecords"]
