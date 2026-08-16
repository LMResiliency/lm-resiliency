"""Contract tests for canonical persisted restart acknowledgements."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    AgentIdentity,
    RankAssignment,
    RestartAck,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_ack_execution import (
    RestartAckExecutor,
)
from lm_resiliency.integrations.torchrun._restart_ack_persisted import (
    PersistedRestartAck,
)
from lm_resiliency.integrations.torchrun._restart_ack_preparation import (
    RestartAckPreparer,
)
from lm_resiliency.integrations.torchrun._restart_ack_records import (
    RestartAckReceiptRecord,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    RestartIntentOpenExecutor,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms


def _persisted() -> PersistedRestartAck:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=1_000,
        clock=clock,
    ).acquire()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=0,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )
    current = GenerationStateManager(store, run_id=RUN_ID).initialize(
        lease,
        assignment,
    )
    intent = RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=("node-b",),
        prepare_deadline_unix_ms=1_500,
    )
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_open(lease, current, intent)
    )
    registration = AgentRegistrationManager(
        store,
        agent_identity=AgentIdentity(
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            hostname="host-node-a",
            local_world_size=2,
            resource_ids=("gpu-node-a-0", "gpu-node-a-1"),
            environment_digest="environment-v1",
        ),
        lease_duration_ms=400,
        clock=clock,
    ).register()
    receipt = RestartAckReceiptRecord(
        acknowledgement=RestartAck(
            intent_id=intent.intent_id,
            run_id=RUN_ID,
            node_id="node-a",
            agent_id="agent-a",
            generation=0,
            flushed_step=40,
            inventory_event_digests={"inventory-a": "b" * 64},
            transferred_owner_ranks=(0, 1),
            transferred_peer_ranks=(2, 3),
            success=True,
            reason="prepared",
        ),
        intent_record=opened.prepared.record,
        agent_registration=registration.record,
        registration_fencing_token=registration.fencing_token,
        registration_granted_at_unix_ms=registration.granted_at_unix_ms,
        received_at_unix_ms=clock.now_unix_ms,
    )
    prepared = RestartAckPreparer(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).prepare(receipt, lease)
    committed = RestartAckExecutor(store, run_id=RUN_ID).execute(prepared)
    return PersistedRestartAck.from_entry(
        run_id=RUN_ID,
        receipt_entry=committed.receipt_entry,
        opened=prepared.records.opened,
        registration_authority=prepared.registration_authority,
        coordinator_authority=prepared.coordinator_authority,
    )


def test_persisted_restart_ack_decodes_one_canonical_receipt():
    persisted = _persisted()

    assert persisted.receipt_entry.value == persisted.receipt.to_json()
    assert persisted.receipt.intent_record == persisted.opened.prepared.record
    assert (
        persisted.receipt.authenticated_registration
        == persisted.registration_authority.registration
    )
    assert persisted.committed_at_unix_ms == 1_000
    assert persisted.transaction_sequence > persisted.opened.transaction_sequence


def test_persisted_restart_ack_is_immutable():
    persisted = _persisted()

    with pytest.raises(AttributeError):
        persisted.receipt_entry = persisted.receipt_entry


def test_persisted_restart_ack_rejects_malformed_or_noncanonical_receipt():
    persisted = _persisted()

    with pytest.raises(ValueError, match="malformed"):
        PersistedRestartAck.from_entry(
            run_id=RUN_ID,
            receipt_entry=replace(persisted.receipt_entry, value=b"{}"),
            opened=persisted.opened,
            registration_authority=persisted.registration_authority,
            coordinator_authority=persisted.coordinator_authority,
        )

    with pytest.raises(ValueError, match="noncanonical"):
        replace(
            persisted,
            receipt_entry=replace(
                persisted.receipt_entry,
                value=b" " + persisted.receipt.to_json(),
            ),
        )


def test_persisted_restart_ack_rejects_wrong_intent_or_registration():
    persisted = _persisted()

    with pytest.raises(ValueError, match="restart intent"):
        replace(
            persisted,
            receipt=replace(
                persisted.receipt,
                intent_record=replace(
                    persisted.receipt.intent_record,
                    intent=replace(
                        persisted.receipt.intent_record.intent,
                        intent_id="intent-b",
                    ),
                ),
            ),
        )
    with pytest.raises(ValueError, match="registration authority"):
        replace(
            persisted,
            registration_authority=replace(
                persisted.registration_authority,
                registration=replace(
                    persisted.registration_authority.registration,
                    fencing_token=persisted.registration_authority.registration.fencing_token + 1,
                ),
            ),
        )


def test_persisted_restart_ack_rejects_receipt_before_intent_opening():
    persisted = _persisted()
    later_opened = replace(
        persisted.opened,
        intent_entry=replace(
            persisted.opened.intent_entry,
            committed_at_unix_ms=persisted.receipt.received_at_unix_ms + 1,
        ),
        head_entry=replace(
            persisted.opened.head_entry,
            committed_at_unix_ms=persisted.receipt.received_at_unix_ms + 1,
        ),
    )

    with pytest.raises(ValueError, match="predates"):
        replace(persisted, opened=later_opened)


def test_persisted_restart_ack_rejects_cross_run_coordinator_authority():
    persisted = _persisted()

    with pytest.raises(ValueError, match="another run"):
        replace(
            persisted,
            coordinator_authority=replace(
                persisted.coordinator_authority,
                lease=replace(
                    persisted.coordinator_authority.lease,
                    record=replace(
                        persisted.coordinator_authority.lease.record,
                        run_id="other-run",
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("entry_changes", "message"),
    [
        ({"mutation_sequence": 2, "value_sequence": 2}, "immutable creation"),
        ({"guard_value_digest": "0" * 64}, "lease provenance"),
        ({"transaction_sequence": 1}, "does not follow"),
        ({"committed_at_unix_ms": None}, "commit time"),
        ({"committed_at_unix_ms": 1_500}, "authority window"),
    ],
)
def test_persisted_restart_ack_rejects_invalid_entry(entry_changes, message):
    persisted = _persisted()

    with pytest.raises(ValueError, match=message):
        replace(
            persisted,
            receipt_entry=replace(persisted.receipt_entry, **entry_changes),
        )


def test_persisted_restart_ack_requires_expected_types_and_run():
    persisted = _persisted()

    with pytest.raises(TypeError, match="receipt"):
        replace(persisted, receipt={})
    with pytest.raises(TypeError, match="ControlStoreEntry"):
        PersistedRestartAck.from_entry(
            run_id=RUN_ID,
            receipt_entry=cast(Any, {}),
            opened=persisted.opened,
            registration_authority=persisted.registration_authority,
            coordinator_authority=persisted.coordinator_authority,
        )
    with pytest.raises(ValueError, match="another run"):
        PersistedRestartAck.from_entry(
            run_id="other-run",
            receipt_entry=persisted.receipt_entry,
            opened=persisted.opened,
            registration_authority=persisted.registration_authority,
            coordinator_authority=persisted.coordinator_authority,
        )
