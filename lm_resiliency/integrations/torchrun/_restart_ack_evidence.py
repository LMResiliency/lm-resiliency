"""Latest-checkpoint evidence derived from restart acknowledgements."""

from __future__ import annotations

from dataclasses import dataclass

from lm_resiliency.integrations.torchrun._protocol import (
    CheckpointInventoryEvent,
    ProtocolValidationError,
    checkpoint_inventory_digest,
    validate_worker_identity,
)
from lm_resiliency.integrations.torchrun._restart_ack_collection import (
    RestartAckCollection,
)


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
        assignment = self.collection.opened.prepared.current.snapshot.record.assignment
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


__all__ = ["RestartAckEvidence"]
