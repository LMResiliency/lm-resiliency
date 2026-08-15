"""Contract tests for authenticated torchrun node-quarantine writes."""

from __future__ import annotations

import threading
from dataclasses import replace

import pytest

from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
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
    NodeQuarantineWriteRepository,
    QuarantineLeaseLost,
    QuarantineWriteCorrupt,
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


def _state(
    *,
    run_id: str = RUN_ID,
) -> tuple[
    ManualClock,
    InMemoryControlStore,
    NodeQuarantineWriteRepository,
    HeldCoordinatorLease,
]:
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    repository = NodeQuarantineWriteRepository(store, run_id=run_id)
    lease = CoordinatorLeaseManager(
        store,
        run_id=run_id,
        coordinator_id="coordinator-a",
        lease_duration_ms=100,
        clock=clock,
    ).acquire()
    return clock, store, repository, lease


def _writes(
    repository: NodeQuarantineWriteRepository,
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


def test_plan_writes_are_create_once_and_bind_authority():
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
        (replace(_plan(), recovery_mode="latest"), _intent(), "weaker"),
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


def test_plan_writes_reject_malformed_persisted_lease():
    clock, store, repository, lease = _state()
    malformed = store.compare_set_in_window(
        repository.coordinator_lease_key,
        expected_revision=lease.fencing_token,
        not_before_unix_ms=clock.now_unix_ms,
        deadline_unix_ms=lease.expires_at_unix_ms,
        value=b"malformed",
    )
    fabricated = HeldCoordinatorLease(
        record=lease.record,
        fencing_token=malformed.revision,
        granted_at_unix_ms=malformed.committed_at_unix_ms,
    )

    with pytest.raises(QuarantineWriteCorrupt, match="malformed"):
        _writes(repository, fabricated)


def test_quarantine_keys_are_run_and_node_scoped_without_plaintext_identity():
    first = node_quarantine_key("run-a", "node-a")
    second = node_quarantine_key("run-a", "node-b")
    third = node_quarantine_key("run-b", "node-a")

    assert len({first, second, third}) == 3
    assert "run-a" not in first
    assert "node-a" not in first
