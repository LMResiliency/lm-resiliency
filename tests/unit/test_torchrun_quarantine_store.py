"""Contract tests for fail-closed torchrun node-quarantine storage."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import (
    ControlStoreWrite,
    InMemoryControlStore,
)
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._protocol import (
    RestartIntent,
    RestartPlan,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._quarantine_records import (
    NodeQuarantineRecord,
)
from lm_resiliency.integrations.torchrun._quarantine_store import (
    NodeQuarantineRepository,
    QuarantineLeaseLost,
    QuarantineStateCorrupt,
    node_quarantine_key,
)

RUN_ID = "training-run"


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self.now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self.now_unix_ms


def _intent(
    *,
    run_id: str = RUN_ID,
    suspected_node_ids: tuple[str, ...] = ("node-b",),
) -> RestartIntent:
    return RestartIntent(
        intent_id="intent-0",
        run_id=run_id,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=suspected_node_ids,
        prepare_deadline_unix_ms=2_000,
    )


def _plan(
    *,
    run_id: str = RUN_ID,
    quarantined_node_ids: tuple[str, ...] = ("node-b",),
) -> RestartPlan:
    return RestartPlan(
        plan_id="plan-1",
        intent_id="intent-0",
        run_id=run_id,
        from_generation=0,
        to_generation=1,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        recovery_mode="recovery_verified",
        checkpoint_source="durable",
        checkpoint_step=10,
        checkpoint_id="checkpoint-10",
        checkpoint_manifest_id="manifest-10",
        slot_assignments=(
            SlotAssignment(
                logical_node_slot=0,
                node_id="node-a",
                first_global_rank=0,
                local_world_size=2,
            ),
            SlotAssignment(
                logical_node_slot=1,
                node_id="node-c",
                first_global_rank=2,
                local_world_size=2,
            ),
        ),
        quarantined_node_ids=quarantined_node_ids,
        expected_world_size=4,
        topology_digest="topology-v1",
        restart_deadline_unix_ms=3_000,
    )


def _repository(
    store: InMemoryControlStore,
    *,
    run_id: str = RUN_ID,
) -> NodeQuarantineRepository:
    return NodeQuarantineRepository(store, run_id=run_id)


def _state(
    *,
    run_id: str = RUN_ID,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    NodeQuarantineRepository,
    HeldCoordinatorLease,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    repository = _repository(store, run_id=run_id)
    lease = CoordinatorLeaseManager(
        store,
        run_id=run_id,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    ).acquire()
    return clock, store, repository, lease


def _writes(
    repository: NodeQuarantineRepository,
    lease: HeldCoordinatorLease,
    *,
    plan: RestartPlan | None = None,
    intent: RestartIntent | None = None,
):
    return repository.prepare_plan_writes(
        plan or _plan(),
        intent or _intent(),
        lease,
        authorized_resource_ids_by_node={"node-b": ("gpu-b0",)},
        resource_to_node_id={"gpu-b0": "node-b"},
    )


def test_plan_writes_are_create_once_and_bind_plan_fields():
    _, _, repository, lease = _state()

    writes = _writes(repository, lease)

    assert list(writes) == [repository.quarantine_key("node-b")]
    write = writes[repository.quarantine_key("node-b")]
    assert write.expected_revision is None
    assert write.require_never_created
    assert NodeQuarantineRecord.from_json(write.value) == NodeQuarantineRecord(
        run_id=RUN_ID,
        node_id="node-b",
        plan_id="plan-1",
        intent_id="intent-0",
        from_generation=0,
        effective_generation=1,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        resource_ids=("gpu-b0",),
        coordinator_id="coordinator-a",
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=100,
        coordinator_fencing_token=lease.fencing_token,
    )
    with pytest.raises(TypeError):
        writes["other"] = write


def test_plan_writes_allow_no_quarantine():
    _, _, repository, lease = _state()

    writes = repository.prepare_plan_writes(
        _plan(quarantined_node_ids=()),
        _intent(suspected_node_ids=()),
        lease,
        authorized_resource_ids_by_node={},
        resource_to_node_id={},
    )

    assert not writes


@pytest.mark.parametrize(
    ("plan", "intent", "message"),
    [
        (_plan(run_id="other-run"), _intent(), "repository run"),
        (_plan(), _intent(run_id="other-run"), "repository run"),
        (replace(_plan(), intent_id="other-intent"), _intent(), "supplied intent"),
        (replace(_plan(), from_generation=1, to_generation=2), _intent(), "generation"),
        (replace(_plan(), incident_ids=("other-incident",)), _intent(), "incidents"),
        (replace(_plan(), reason_code="other-reason"), _intent(), "reason"),
        (_plan(), _intent(suspected_node_ids=("node-a",)), "outside the intent"),
        (
            replace(_plan(), recovery_mode="latest"),
            _intent(),
            "weaker",
        ),
    ],
)
def test_plan_writes_reject_mismatched_plan_and_intent(plan, intent, message):
    _, _, repository, lease = _state()

    with pytest.raises(ValueError, match=message):
        _writes(repository, lease, plan=plan, intent=intent)


def test_plan_writes_require_exact_quarantined_node_resource_keys():
    _, _, repository, lease = _state()

    with pytest.raises(ValueError, match="exactly match"):
        repository.prepare_plan_writes(
            _plan(),
            _intent(),
            lease,
            authorized_resource_ids_by_node={},
            resource_to_node_id={"gpu-b0": "node-b"},
        )


@pytest.mark.parametrize(
    "resource_to_node_id",
    [
        {},
        {"gpu-b0": "node-a"},
    ],
)
def test_plan_writes_require_trusted_resource_ownership(resource_to_node_id):
    _, _, repository, lease = _state()

    with pytest.raises(ValueError, match="trusted as owned"):
        repository.prepare_plan_writes(
            _plan(),
            _intent(),
            lease,
            authorized_resource_ids_by_node={"node-b": ("gpu-b0",)},
            resource_to_node_id=resource_to_node_id,
        )


def test_plan_writes_allow_node_level_evidence_without_resource_ids():
    _, _, repository, lease = _state()

    writes = repository.prepare_plan_writes(
        _plan(),
        _intent(),
        lease,
        authorized_resource_ids_by_node={"node-b": ()},
        resource_to_node_id={},
    )

    record = NodeQuarantineRecord.from_json(writes[repository.quarantine_key("node-b")].value)
    assert record.resource_ids == ()


def test_plan_writes_reject_stale_coordinator_lease():
    clock, store, repository, lease = _state()
    store.compare_set_in_window(
        repository.coordinator_lease_key,
        expected_revision=lease.fencing_token,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
        value=lease.record.to_json(),
    )

    with pytest.raises(QuarantineLeaseLost, match="changed"):
        _writes(repository, lease)


def test_repository_reads_guarded_quarantine():
    _, store, repository, lease = _state()

    committed = store.compare_set_many_guarded(
        _writes(repository, lease),
        guard_key=repository.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )
    stored = repository.get("node-b")

    assert stored is not None
    assert stored.record.node_id == "node-b"
    assert stored.entry == committed[repository.quarantine_key("node-b")]


def test_repository_returns_none_only_for_never_created_quarantine():
    repository = _repository(InMemoryControlStore())

    assert repository.get("node-b") is None


def test_repository_rejects_deleted_quarantine():
    _, store, repository, lease = _state()
    committed = store.compare_set_many_guarded(
        _writes(repository, lease),
        guard_key=repository.coordinator_lease_key,
        expected_guard_revision=lease.fencing_token,
        not_before_unix_ms=lease.granted_at_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )
    entry = committed[repository.quarantine_key("node-b")]
    store.compare_delete(
        repository.quarantine_key("node-b"),
        expected_revision=entry.revision,
    )

    with pytest.raises(QuarantineStateCorrupt, match="deleted"):
        repository.get("node-b")


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (b"{}", "malformed"),
        (
            NodeQuarantineRecord(
                run_id="other-run",
                node_id="node-b",
                plan_id="plan-1",
                intent_id="intent-0",
                from_generation=0,
                effective_generation=1,
                incident_ids=("incident-a",),
                reason_code="attributed_sdc",
                resource_ids=(),
                coordinator_id="coordinator-a",
                lease_id="lease-a",
                coordinator_lease_duration_ms=100,
                coordinator_fencing_token=1,
            ).to_json(),
            "another run or node",
        ),
    ],
)
def test_repository_rejects_malformed_or_foreign_records(value, message):
    store = InMemoryControlStore()
    repository = _repository(store)
    store.compare_set(
        repository.quarantine_key("node-b"),
        expected_revision=None,
        value=value,
    )

    with pytest.raises(QuarantineStateCorrupt, match=message):
        repository.get("node-b")


def test_repository_requires_guarded_create_provenance():
    _, store, repository, lease = _state()
    write = _writes(repository, lease)[repository.quarantine_key("node-b")]
    store.compare_set(
        repository.quarantine_key("node-b"),
        expected_revision=None,
        value=write.value,
    )

    with pytest.raises(QuarantineStateCorrupt, match="commit time"):
        repository.get("node-b")


def test_repository_rejects_quarantine_guarded_by_another_key():
    clock, store, repository, lease = _state()
    wrong_guard = store.compare_set_in_window(
        "other/lease",
        expected_revision=None,
        not_before_unix_ms=1_000,
        deadline_unix_ms=None,
        value=b"lease",
    )
    store.compare_set_many_guarded(
        _writes(repository, lease),
        guard_key="other/lease",
        expected_guard_revision=wrong_guard.revision,
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_100,
    )

    with pytest.raises(QuarantineStateCorrupt, match="run coordinator lease"):
        repository.get("node-b")


def test_repository_authenticates_opaque_coordinator_lease_bytes():
    clock, store, repository, lease = _state()
    malformed_guard = store.compare_set_in_window(
        repository.coordinator_lease_key,
        expected_revision=lease.fencing_token,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
        value=b"malformed-lease",
    )
    record = NodeQuarantineRecord(
        run_id=RUN_ID,
        node_id="node-b",
        plan_id="plan-1",
        intent_id="intent-0",
        from_generation=0,
        effective_generation=1,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        resource_ids=("gpu-b0",),
        coordinator_id=lease.record.coordinator_id,
        lease_id=lease.record.lease_id,
        coordinator_lease_duration_ms=lease.record.lease_duration_ms,
        coordinator_fencing_token=malformed_guard.revision,
    )
    store.compare_set_many_guarded(
        {
            repository.quarantine_key("node-b"): ControlStoreWrite(
                expected_revision=None,
                value=record.to_json(),
                require_never_created=True,
            ),
        },
        guard_key=repository.coordinator_lease_key,
        expected_guard_revision=malformed_guard.revision,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
    )

    with pytest.raises(QuarantineStateCorrupt, match="lease identity"):
        repository.get("node-b")


def test_repository_rejects_recreated_quarantine():
    _, store, repository, lease = _state()
    key = repository.quarantine_key("node-b")
    write = _writes(repository, lease)[key]
    original = store.compare_set(key, expected_revision=None, value=write.value)
    store.compare_delete(key, expected_revision=original.revision)
    store.compare_set(key, expected_revision=None, value=write.value)

    with pytest.raises(QuarantineStateCorrupt, match="noninitial"):
        repository.get("node-b")


def test_quarantine_keys_are_run_and_node_scoped_without_plaintext_identity():
    first = node_quarantine_key("run-a", "node-a")
    second = node_quarantine_key("run-a", "node-b")
    third = node_quarantine_key("run-b", "node-a")

    assert len({first, second, third}) == 3
    assert "run-a" not in first
    assert "node-a" not in first
