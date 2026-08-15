"""Contract tests for initial restart-intent closure records."""

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


def _records() -> InitialRestartIntentClosureRecords:
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
    open_preparer = RestartIntentOpenPreparer(store, run_id=RUN_ID, clock=clock)
    open_executor = RestartIntentOpenExecutor(store, run_id=RUN_ID)
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
    opened = open_executor.execute_initial_open(
        open_preparer.prepare_initial_open(
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
                prepare_deadline_unix_ms=1_050,
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
    return InitialRestartIntentClosureRecords(
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


def test_initial_closure_records_build_immutable_store_inputs():
    records = _records()

    assert set(records.writes) == {
        records.intent_head_key,
        records.lifecycle_head_key,
        records.closure_key,
    }
    assert (
        records.writes[records.intent_head_key].expected_revision
        == records.opened.head_entry.revision
    )
    assert not records.writes[records.intent_head_key].require_never_created
    assert records.writes[records.intent_head_key].value == records.closed_head.to_json()
    assert records.writes[records.lifecycle_head_key].require_never_created
    assert records.writes[records.lifecycle_head_key].value == records.lifecycle_head.to_json()
    assert records.writes[records.closure_key].require_never_created
    assert records.writes[records.closure_key].value == records.lifecycle.to_json()
    assert records.conditions == {
        records.intent_key: records.opened.intent_entry.revision,
    }
    with pytest.raises(TypeError):
        cast(Any, records.writes)["other"] = next(iter(records.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, records.conditions)["other"] = 1


def test_initial_closure_records_are_immutable():
    records = _records()

    with pytest.raises(AttributeError):
        records.intent_key = "other"


def test_initial_closure_records_require_expected_types():
    records = _records()

    for field, value, message in (
        ("opened", {}, "CommittedInitialRestartIntentOpen"),
        ("lifecycle", {}, "RestartIntentLifecycleRecord"),
        ("lifecycle_head", {}, "RestartIntentLifecycleHeadRecord"),
        ("closed_head", {}, "RestartIntentClosedHeadRecord"),
    ):
        with pytest.raises(TypeError, match=message):
            replace(records, **{field: value})


def test_initial_closure_records_reject_unlinked_values():
    records = _records()
    wrong_intent = replace(records.lifecycle.closed_intent, intent_id="other")
    wrong_lifecycle = replace(records.lifecycle, closed_intent=wrong_intent)
    wrong_lifecycle_head = replace(records.lifecycle_head, closure_index=2)
    wrong_closed_head = replace(records.closed_head, lifecycle_head_digest="0" * 64)

    for field, value, message in (
        ("lifecycle", wrong_lifecycle, "does not close"),
        ("lifecycle_head", wrong_lifecycle_head, "does not identify"),
        ("closed_head", wrong_closed_head, "does not identify"),
    ):
        with pytest.raises(ValueError, match=message):
            replace(records, **{field: value})


@pytest.mark.parametrize(
    "field",
    [
        "intent_key",
        "intent_head_key",
        "closure_key",
        "lifecycle_head_key",
    ],
)
def test_initial_closure_records_require_canonical_keys(field):
    records = _records()

    with pytest.raises(ValueError, match="canonical"):
        replace(records, **{field: "other"})


def test_initial_closure_records_hide_plaintext_identity():
    records = _records()

    assert RUN_ID not in records.closure_key
    assert "intent-a" not in records.closure_key
