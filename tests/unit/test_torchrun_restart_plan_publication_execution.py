"""Contract tests for guarded restart-plan publication execution."""

from __future__ import annotations

import hashlib
import threading
from collections.abc import Collection, Mapping
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreEntry,
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_authority import (
    RestartPlanPublicationAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_execution import (
    RestartPlanPublicationExecutionClockError,
    RestartPlanPublicationExecutionConflict,
    RestartPlanPublicationExecutionCorrupt,
    RestartPlanPublicationExecutionDeadlineElapsed,
    RestartPlanPublicationExecutionLeaseLost,
    RestartPlanPublicationExecutionRegistrationLost,
    RestartPlanPublicationExecutor,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_state import (
    PreparedRestartPlanPublication,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 900) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms

    def set(self, now_unix_ms: int) -> None:
        with self._lock:
            self.now_unix_ms = now_unix_ms


class TamperedPublicationResultStore(InMemoryControlStore):
    def __init__(self, *, clock: ManualClock, tamper: str) -> None:
        super().__init__(clock=clock)
        self._tamper = tamper

    def compare_set_many_guarded(
        self,
        writes: Mapping[str, ControlStoreWrite],
        *,
        guard_key: str,
        expected_guard_revision: int,
        not_before_unix_ms: int,
        deadline_unix_ms: int,
        conditions: Mapping[str, int | None] | None = None,
        never_created_conditions: Collection[str] | None = None,
    ) -> Mapping[str, ControlStoreEntry]:
        committed = dict(
            super().compare_set_many_guarded(
                writes,
                guard_key=guard_key,
                expected_guard_revision=expected_guard_revision,
                not_before_unix_ms=not_before_unix_ms,
                deadline_unix_ms=deadline_unix_ms,
                conditions=conditions,
                never_created_conditions=never_created_conditions,
            )
        )
        plan_keys = [key for key in committed if "/restart-plans/" in key]
        if not plan_keys:
            return committed
        plan_key = next(key for key in plan_keys if not key.endswith("/recovery-manifest"))
        generation_head_key = next(key for key in committed if key.endswith("/generation-head"))
        immutable_key = next(key for key in committed if key not in {plan_key, generation_head_key})
        if self._tamper == "missing":
            committed.pop(plan_key)
        elif self._tamper == "unexpected":
            committed[f"{plan_key}/extra"] = committed[plan_key]
        elif self._tamper == "value":
            committed[plan_key] = replace(committed[plan_key], value=b"{}")
        elif self._tamper == "transaction":
            committed[plan_key] = replace(
                committed[plan_key],
                transaction_sequence=committed[plan_key].transaction_sequence + 1,
            )
        elif self._tamper == "guard":
            committed[plan_key] = replace(
                committed[plan_key],
                guard_value_digest="0" * 64,
            )
        elif self._tamper == "head_lineage":
            committed[generation_head_key] = replace(
                committed[generation_head_key],
                mutation_sequence=4,
                value_sequence=4,
            )
        elif self._tamper == "head_revision":
            committed[generation_head_key] = replace(
                committed[generation_head_key],
                revision=2,
            )
        elif self._tamper == "immutable_lineage":
            committed[immutable_key] = replace(
                committed[immutable_key],
                mutation_sequence=2,
                value_sequence=2,
            )
        elif self._tamper == "time":
            committed[plan_key] = replace(
                committed[plan_key],
                committed_at_unix_ms=None,
            )
        elif self._tamper == "order":
            for key, entry in tuple(committed.items()):
                committed[key] = replace(entry, transaction_sequence=1)
        else:
            raise AssertionError(f"unsupported tamper {self._tamper!r}")
        return committed


def _seed_value(
    store: InMemoryControlStore,
    key: str,
    *,
    revisions: int = 1,
) -> ControlStoreEntry:
    entry = None
    expected_revision = None
    for revision in range(revisions):
        entry = store.compare_set(
            key,
            expected_revision=expected_revision,
            value=f"{key}:{revision}".encode(),
        )
        expected_revision = entry.revision
    assert entry is not None
    return entry


def _state(
    *,
    tamper: str | None = None,
) -> tuple[ManualClock, InMemoryControlStore, PreparedRestartPlanPublication]:
    clock = ManualClock()
    store = (
        InMemoryControlStore(clock=clock)
        if tamper is None
        else TamperedPublicationResultStore(clock=clock, tamper=tamper)
    )
    run_digest = hashlib.sha256(RUN_ID.encode()).hexdigest()
    run_prefix = f"lm_resiliency/torchrun/v1/runs/{run_digest}"
    guard_key = f"{run_prefix}/coordinator-lease"
    lease_record = CoordinatorLeaseRecord(
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        lease_duration_ms=500,
    )
    guard_entry = None
    expected_guard_revision = None
    for _ in range(4):
        guard_entry = store.compare_set_in_window(
            guard_key,
            expected_revision=expected_guard_revision,
            not_before_unix_ms=900,
            deadline_unix_ms=1_400,
            value=lease_record.to_json(),
        )
        expected_guard_revision = guard_entry.revision
    assert guard_entry is not None
    authority = CoordinatorLeaseAuthority(
        lease=HeldCoordinatorLease(
            record=lease_record,
            fencing_token=guard_entry.revision,
            granted_at_unix_ms=900,
        ),
        transaction_sequence=guard_entry.transaction_sequence,
        mutation_sequence=guard_entry.mutation_sequence,
        value_sequence=guard_entry.value_sequence,
        lifetime_sequence=guard_entry.lifetime_sequence,
    )

    generation_head_key = f"{run_prefix}/generation-head"
    generation_head_entry = _seed_value(store, generation_head_key, revisions=2)
    source_key = f"{run_prefix}/generations/1"
    registration_keys = {
        "node-a": f"{run_prefix}/agents/node-a",
        "node-c": f"{run_prefix}/agents/node-c",
    }
    lifecycle_conditions = {
        f"{run_prefix}/restart-intents/intent-a": 1,
        f"{run_prefix}/restart-intent-head": 1,
        f"{run_prefix}/restart-intent-closures/1": 1,
        f"{run_prefix}/restart-intent-lifecycle-head": 1,
    }
    _seed_value(store, source_key)
    for key in registration_keys.values():
        _seed_value(store, key)
    for key in lifecycle_conditions:
        _seed_value(store, key)

    writes = {
        generation_head_key: ControlStoreWrite(
            expected_revision=generation_head_entry.revision,
            value=b"generation-head-2",
        ),
        f"{run_prefix}/generations/2": ControlStoreWrite(
            expected_revision=None,
            value=b"generation-snapshot-2",
            require_never_created=True,
        ),
        f"{run_prefix}/restart-plans/2/recovery-manifest": ControlStoreWrite(
            expected_revision=None,
            value=b"recovery-manifest",
            require_never_created=True,
        ),
        f"{run_prefix}/restart-plans/2": ControlStoreWrite(
            expected_revision=None,
            value=b"restart-plan",
            require_never_created=True,
        ),
        f"{run_prefix}/quarantines/node-b": ControlStoreWrite(
            expected_revision=None,
            value=b"quarantine-node-b",
            require_never_created=True,
        ),
    }
    intent_record = object()
    lifecycle_record = object()
    generation_record = object()
    current_snapshot = SimpleNamespace(
        record=generation_record,
        transaction_sequence=5,
    )
    registration_histories = {
        node_id: SimpleNamespace(
            current=SimpleNamespace(
                expires_at_unix_ms=1_300,
            ),
            authorities=(SimpleNamespace(transaction_sequence=7),),
        )
        for node_id in registration_keys
    }
    generation_state = SimpleNamespace(
        intent_record=intent_record,
        lifecycle_record=lifecycle_record,
        from_snapshot=generation_record,
    )
    candidate = SimpleNamespace(
        plan=SimpleNamespace(
            run_id=RUN_ID,
            to_generation=2,
        ),
        placement_state=SimpleNamespace(
            generation_state=generation_state,
            registration_histories=registration_histories,
        ),
        recovery_state=SimpleNamespace(
            copy_state=SimpleNamespace(
                inventory_state=SimpleNamespace(
                    quarantine_state=SimpleNamespace(
                        manifest_state=SimpleNamespace(
                            resolved_manifest=SimpleNamespace(
                                source_snapshot=SimpleNamespace(
                                    transaction_sequence=5,
                                )
                            )
                        )
                    )
                )
            )
        ),
    )
    records = SimpleNamespace(
        candidate=candidate,
        current=SimpleNamespace(
            snapshot=current_snapshot,
            head_revision=generation_head_entry.revision,
        ),
        run_prefix=run_prefix,
        generation_head_key=generation_head_key,
        successor_generation_snapshot_key=f"{run_prefix}/generations/2",
        registration_keys=registration_keys,
        writes=writes,
        conditions={
            source_key: 1,
            **{key: 1 for key in registration_keys.values()},
        },
    )
    publication_authority = Mock(spec=RestartPlanPublicationAuthority)
    publication_authority.records = records
    publication_authority.coordinator_authority = authority
    publication_authority.observed_at_unix_ms = 1_000
    publication_authority.not_before_unix_ms = 1_000
    publication_authority.deadline_unix_ms = 1_300

    lifecycle_fence = Mock(spec=RestartPlanPublicationLifecycleFence)
    lifecycle_fence.closure = SimpleNamespace(
        intent=intent_record,
        lifecycle=lifecycle_record,
        generation_snapshot=current_snapshot,
        closing_authority=authority,
        lease_history=(authority,),
        closed_at_unix_ms=1_000,
    )
    lifecycle_fence.conditions = lifecycle_conditions
    lifecycle_fence.transaction_sequence = 8
    prepared = PreparedRestartPlanPublication(
        authority=publication_authority,
        lifecycle_fence=lifecycle_fence,
    )
    clock.set(1_100)
    return clock, store, prepared


def test_executor_commits_and_verifies_publication():
    _, store, prepared = _state()

    committed = RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)

    assert set(committed.entries) == set(prepared.writes)
    assert committed.generation_head_entry == store.get(
        prepared.authority.records.generation_head_key
    )
    assert committed.successor_snapshot_entry == store.get(
        prepared.authority.records.successor_generation_snapshot_key
    )
    assert committed.committed_at_unix_ms == 1_100
    assert committed.transaction_sequence > prepared.lifecycle_fence.transaction_sequence
    with pytest.raises(TypeError):
        committed.entries["other"] = committed.generation_head_entry


def test_executor_rejects_changed_state_and_duplicate_publication():
    _, store, prepared = _state()
    condition_key = next(iter(prepared.lifecycle_fence.conditions))
    condition_entry = store.get(condition_key)
    assert condition_entry is not None
    store.compare_set(
        condition_key,
        expected_revision=condition_entry.revision,
        value=condition_entry.value,
    )

    with pytest.raises(RestartPlanPublicationExecutionConflict, match="state changed"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)

    _, store, prepared = _state()
    executor = RestartPlanPublicationExecutor(store, run_id=RUN_ID)
    executor.execute(prepared)
    with pytest.raises(RestartPlanPublicationExecutionConflict):
        executor.execute(prepared)


def test_executor_rejects_changed_registration():
    _, store, prepared = _state()
    registration_key = next(iter(prepared.authority.records.registration_keys.values()))
    registration_entry = store.get(registration_key)
    assert registration_entry is not None
    store.compare_set(
        registration_key,
        expected_revision=registration_entry.revision,
        value=registration_entry.value,
    )

    with pytest.raises(
        RestartPlanPublicationExecutionRegistrationLost,
        match="registration changed",
    ):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)


def test_executor_rejects_changed_or_expired_lease():
    clock, store, prepared = _state()
    guard_entry = store.get(prepared.guard_key)
    assert guard_entry is not None
    clock.set(1_150)
    store.compare_set_in_window(
        prepared.guard_key,
        expected_revision=guard_entry.revision,
        not_before_unix_ms=1_150,
        deadline_unix_ms=1_400,
        value=guard_entry.value,
    )

    with pytest.raises(RestartPlanPublicationExecutionLeaseLost, match="changed"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)

    clock, store, prepared = _state()
    prepared.authority.deadline_unix_ms = 1_400
    for (
        history
    ) in prepared.authority.records.candidate.placement_state.registration_histories.values():
        history.current.expires_at_unix_ms = 1_500
    clock.set(1_400)
    with pytest.raises(RestartPlanPublicationExecutionLeaseLost, match="expired"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)


def test_executor_classifies_registration_and_plan_deadlines():
    clock, store, prepared = _state()
    clock.set(1_300)
    with pytest.raises(RestartPlanPublicationExecutionRegistrationLost, match="expired"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)

    clock, store, prepared = _state()
    prepared.authority.deadline_unix_ms = 1_250
    clock.set(1_250)
    with pytest.raises(RestartPlanPublicationExecutionDeadlineElapsed, match="deadline"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)


def test_executor_rejects_store_time_before_preparation():
    clock, store, prepared = _state()
    clock.set(999)

    with pytest.raises(RestartPlanPublicationExecutionClockError, match="contradicts"):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)


@pytest.mark.parametrize(
    "tamper",
    [
        "missing",
        "unexpected",
        "value",
        "transaction",
        "guard",
        "head_lineage",
        "head_revision",
        "immutable_lineage",
        "time",
        "order",
    ],
)
def test_executor_rejects_tampered_results(tamper: str):
    _, store, prepared = _state(tamper=tamper)

    with pytest.raises(RestartPlanPublicationExecutionCorrupt):
        RestartPlanPublicationExecutor(store, run_id=RUN_ID).execute(prepared)


def test_executor_requires_exact_inputs():
    _, store, prepared = _state()
    executor = RestartPlanPublicationExecutor(store, run_id=RUN_ID)

    with pytest.raises(TypeError, match="PreparedRestartPlanPublication"):
        executor.execute(prepared.authority)
    with pytest.raises(ValueError, match="another run"):
        RestartPlanPublicationExecutor(store, run_id="other-run").execute(prepared)
    with pytest.raises(ValueError, match="non-empty"):
        RestartPlanPublicationExecutor(store, run_id="")
