"""Contract tests for canonical persisted initial-closure state."""

from __future__ import annotations

import threading
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_close_records import (
    InitialRestartIntentClosureRecords,
)
from lm_resiliency.integrations.torchrun._restart_intent_lifecycle_state import (
    PersistedInitialRestartIntentClosure,
)
from lm_resiliency.integrations.torchrun._restart_intent_open import (
    RestartIntentOpenPreparer,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_execution import (
    RestartIntentOpenExecutor,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentClosedHeadRecord,
    RestartIntentLifecycleHeadRecord,
    RestartIntentLifecycleRecord,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms

    def set(self, now_unix_ms: int) -> None:
        with self._lock:
            self.now_unix_ms = now_unix_ms


def _state() -> tuple[
    PersistedInitialRestartIntentClosure,
    InitialRestartIntentClosureRecords,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    lease_manager = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    )
    generation_manager = GenerationStateManager(store, run_id=RUN_ID)
    lease = lease_manager.acquire()
    generation_manager.initialize(
        lease,
        RankAssignment.from_assignments(
            run_id=RUN_ID,
            generation=0,
            assignments=(
                SlotAssignment(0, "node-a", 0, 2),
                SlotAssignment(1, "node-b", 2, 2),
            ),
            topology_digest="topology-v1",
        ),
    )
    current = generation_manager.current()
    assert current is not None
    opened = RestartIntentOpenExecutor(store, run_id=RUN_ID).execute_initial_open(
        RestartIntentOpenPreparer(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).prepare_initial_open(
            lease,
            current,
            RestartIntent(
                intent_id="intent-a",
                run_id=RUN_ID,
                generation=0,
                incident_ids=("incident-a",),
                reason_code="attributed_sdc",
                minimum_recovery_mode="recovery_verified",
                suspected_node_ids=("node-b",),
                prepare_deadline_unix_ms=1_200,
            ),
        )
    )
    lifecycle = RestartIntentLifecycleRecord(
        closed_intent=opened.prepared.head,
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=lease.fencing_token,
    )
    lifecycle_head = RestartIntentLifecycleHeadRecord(
        run_id=RUN_ID,
        closure_index=1,
        generation=0,
        intent_id="intent-a",
        lifecycle_digest=lifecycle.digest,
    )
    run_prefix = opened.prepared.intent_head_key.rsplit("/", 1)[0]
    records = InitialRestartIntentClosureRecords(
        opened=opened,
        lifecycle=lifecycle,
        lifecycle_head=lifecycle_head,
        closed_head=RestartIntentClosedHeadRecord(
            run_id=RUN_ID,
            closure_index=1,
            generation=0,
            intent_id="intent-a",
            lifecycle_head_digest=lifecycle_head.digest,
        ),
        intent_key=opened.prepared.intent_key,
        intent_head_key=opened.prepared.intent_head_key,
        closure_key=f"{run_prefix}/restart-intent-closures/1",
        lifecycle_head_key=opened.prepared.lifecycle_head_key,
    )
    clock.set(1_010)
    committed = store.compare_set_many_guarded(
        records.writes,
        guard_key=opened.prepared.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=1_010,
        deadline_unix_ms=lease.expires_at_unix_ms,
        conditions=records.conditions,
    )
    head_history = store.get_history(records.intent_head_key)
    assert len(head_history) == 2
    state = PersistedInitialRestartIntentClosure.from_entries(
        run_id=RUN_ID,
        intent_entry=opened.intent_entry,
        open_head_entry=head_history[0],
        closed_head_entry=committed[records.intent_head_key],
        lifecycle_entry=committed[records.closure_key],
        lifecycle_head_entry=committed[records.lifecycle_head_key],
    )
    return state, records


def test_persisted_initial_closure_decodes_one_canonical_transaction():
    state, records = _state()

    assert state.intent == records.opened.prepared.record
    assert state.open_head == records.opened.prepared.head
    assert state.closed_head == records.closed_head
    assert state.lifecycle == records.lifecycle
    assert state.lifecycle_head == records.lifecycle_head
    assert state.opened_at_unix_ms == records.opened.committed_at_unix_ms
    assert state.closed_at_unix_ms == 1_010
    assert state.closing_transaction_sequence > state.opening_transaction_sequence


def test_persisted_initial_closure_is_immutable():
    state, _ = _state()

    with pytest.raises(AttributeError):
        state.closed_head = state.closed_head


def test_persisted_initial_closure_requires_entry_and_record_types():
    state, _ = _state()

    with pytest.raises(TypeError, match="ControlStoreEntry"):
        replace(state, intent_entry={})
    with pytest.raises(TypeError, match="RestartIntentRecord"):
        replace(state, intent={})
    with pytest.raises(TypeError, match="ControlStoreEntry"):
        PersistedInitialRestartIntentClosure.from_entries(
            run_id=RUN_ID,
            intent_entry=cast(Any, {}),
            open_head_entry=state.open_head_entry,
            closed_head_entry=state.closed_head_entry,
            lifecycle_entry=state.lifecycle_entry,
            lifecycle_head_entry=state.lifecycle_head_entry,
        )


@pytest.mark.parametrize(
    "entry_name",
    [
        "intent_entry",
        "open_head_entry",
        "closed_head_entry",
        "lifecycle_entry",
        "lifecycle_head_entry",
    ],
)
def test_persisted_initial_closure_rejects_malformed_entry(entry_name):
    state, _ = _state()

    with pytest.raises(ValueError, match="malformed"):
        PersistedInitialRestartIntentClosure.from_entries(
            run_id=RUN_ID,
            intent_entry=replace(state.intent_entry, value=b"{}")
            if entry_name == "intent_entry"
            else state.intent_entry,
            open_head_entry=replace(state.open_head_entry, value=b"{}")
            if entry_name == "open_head_entry"
            else state.open_head_entry,
            closed_head_entry=replace(state.closed_head_entry, value=b"{}")
            if entry_name == "closed_head_entry"
            else state.closed_head_entry,
            lifecycle_entry=replace(state.lifecycle_entry, value=b"{}")
            if entry_name == "lifecycle_entry"
            else state.lifecycle_entry,
            lifecycle_head_entry=replace(state.lifecycle_head_entry, value=b"{}")
            if entry_name == "lifecycle_head_entry"
            else state.lifecycle_head_entry,
        )


def test_persisted_initial_closure_rejects_unlinked_records():
    state, _ = _state()
    lifecycle = replace(
        state.lifecycle,
        closed_intent=replace(state.open_head, intent_id="intent-b"),
    )

    with pytest.raises(ValueError, match="do not form one"):
        replace(
            state,
            lifecycle=lifecycle,
            lifecycle_entry=replace(
                state.lifecycle_entry,
                value=lifecycle.to_json(),
            ),
        )


@pytest.mark.parametrize(
    ("entry_name", "changes", "message"),
    [
        (
            "intent_entry",
            {"mutation_sequence": 2},
            "immutable initial creation",
        ),
        (
            "lifecycle_head_entry",
            {
                "lifetime_sequence": 2,
                "mutation_sequence": 3,
                "value_sequence": 2,
            },
            "immutable initial creation",
        ),
        (
            "closed_head_entry",
            {"mutation_sequence": 3, "value_sequence": 3},
            "replace exactly one",
        ),
    ],
)
def test_persisted_initial_closure_rejects_invalid_store_sequences(
    entry_name,
    changes,
    message,
):
    state, _ = _state()

    with pytest.raises(ValueError, match=message):
        replace(
            state,
            **{entry_name: replace(getattr(state, entry_name), **changes)},
        )


def test_persisted_initial_closure_rejects_reused_head_revision():
    state, _ = _state()

    with pytest.raises(ValueError, match="replace exactly one"):
        replace(
            state,
            closed_head_entry=replace(
                state.closed_head_entry,
                revision=state.open_head_entry.revision,
            ),
        )


def test_persisted_initial_closure_rejects_noncanonical_guard_key():
    state, _ = _state()

    with pytest.raises(ValueError, match="guard provenance"):
        replace(
            state,
            lifecycle_entry=replace(
                state.lifecycle_entry,
                guard_key="other/coordinator-lease",
            ),
        )


def test_persisted_initial_closure_rejects_split_open_transaction():
    state, _ = _state()

    with pytest.raises(ValueError, match="opening entries"):
        replace(
            state,
            open_head_entry=replace(
                state.open_head_entry,
                transaction_sequence=state.open_head_entry.transaction_sequence + 1,
            ),
        )


def test_persisted_initial_closure_rejects_split_close_transaction():
    state, _ = _state()

    with pytest.raises(ValueError, match="closure entries"):
        replace(
            state,
            lifecycle_entry=replace(
                state.lifecycle_entry,
                transaction_sequence=state.lifecycle_entry.transaction_sequence + 1,
            ),
        )


def test_persisted_initial_closure_rejects_authority_substitution():
    state, _ = _state()

    with pytest.raises(ValueError, match="opening authority"):
        replace(
            state,
            intent_entry=replace(
                state.intent_entry,
                guard_revision=state.intent.coordinator_fencing_token + 1,
            ),
            open_head_entry=replace(
                state.open_head_entry,
                guard_revision=state.intent.coordinator_fencing_token + 1,
            ),
        )

    with pytest.raises(ValueError, match="closing authority"):
        replace(
            state,
            lifecycle_entry=replace(state.lifecycle_entry, guard_revision=2),
            lifecycle_head_entry=replace(
                state.lifecycle_head_entry,
                guard_revision=2,
            ),
            closed_head_entry=replace(
                state.closed_head_entry,
                guard_revision=2,
            ),
        )


def test_persisted_initial_closure_rejects_noncausal_close():
    state, _ = _state()

    with pytest.raises(ValueError, match="does not follow"):
        replace(
            state,
            closed_head_entry=replace(
                state.closed_head_entry,
                transaction_sequence=state.opening_transaction_sequence,
                committed_at_unix_ms=state.opened_at_unix_ms,
            ),
            lifecycle_entry=replace(
                state.lifecycle_entry,
                transaction_sequence=state.opening_transaction_sequence,
                committed_at_unix_ms=state.opened_at_unix_ms,
            ),
            lifecycle_head_entry=replace(
                state.lifecycle_head_entry,
                transaction_sequence=state.opening_transaction_sequence,
                committed_at_unix_ms=state.opened_at_unix_ms,
            ),
        )


def test_persisted_initial_closure_is_run_scoped():
    state, _ = _state()

    with pytest.raises(ValueError, match="another run"):
        PersistedInitialRestartIntentClosure.from_entries(
            run_id="other-run",
            intent_entry=state.intent_entry,
            open_head_entry=state.open_head_entry,
            closed_head_entry=state.closed_head_entry,
            lifecycle_entry=state.lifecycle_entry,
            lifecycle_head_entry=state.lifecycle_head_entry,
        )
    with pytest.raises(ValueError, match="non-empty"):
        PersistedInitialRestartIntentClosure.from_entries(
            run_id="",
            intent_entry=state.intent_entry,
            open_head_entry=state.open_head_entry,
            closed_head_entry=state.closed_head_entry,
            lifecycle_entry=state.lifecycle_entry,
            lifecycle_head_entry=state.lifecycle_head_entry,
        )
