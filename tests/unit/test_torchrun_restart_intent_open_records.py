"""Contract tests for prepared initial restart-intent transaction inputs."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Any, cast

import pytest

from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._generation_reader import (
    CurrentGeneration,
    StoredGenerationSnapshot,
)
from lm_resiliency.integrations.torchrun._generation_records import (
    GenerationSnapshotRecord,
)
from lm_resiliency.integrations.torchrun._protocol import (
    RankAssignment,
    RestartIntent,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._restart_intent_open_records import (
    PreparedInitialRestartIntentOpen,
)
from lm_resiliency.integrations.torchrun._restart_intent_records import (
    RestartIntentHeadRecord,
    RestartIntentRecord,
)

RUN_ID = "training-run"
RUN_DIGEST = hashlib.sha256(RUN_ID.encode("utf-8")).hexdigest()
RUN_PREFIX = f"lm_resiliency/torchrun/v1/runs/{RUN_DIGEST}"


def _intent(
    *,
    suspected_node_ids: tuple[str, ...] = ("node-b",),
) -> RestartIntent:
    return RestartIntent(
        intent_id="intent-a",
        run_id=RUN_ID,
        generation=0,
        incident_ids=("incident-a",),
        reason_code="attributed_sdc",
        minimum_recovery_mode="recovery_verified",
        suspected_node_ids=suspected_node_ids,
        prepare_deadline_unix_ms=1_050,
    )


def _current() -> CurrentGeneration:
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=0,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )
    record = GenerationSnapshotRecord(
        assignment=assignment,
        previous_snapshot_digest=None,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=100,
        coordinator_fencing_token=7,
    )
    return CurrentGeneration(
        snapshot=StoredGenerationSnapshot(
            record=record,
            revision=11,
            committed_at_unix_ms=1_000,
            transaction_sequence=4,
            guard_mutation_sequence=1,
            guard_value_sequence=1,
            guard_lifetime_sequence=1,
            guard_committed_at_unix_ms=1_000,
        ),
        head_revision=12,
    )


def _prepared() -> PreparedInitialRestartIntentOpen:
    current = _current()
    record = RestartIntentRecord(
        intent=_intent(),
        generation_snapshot_digest=current.snapshot.record.digest,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        coordinator_lease_duration_ms=100,
        coordinator_fencing_token=7,
    )
    head = RestartIntentHeadRecord(
        run_id=RUN_ID,
        generation=0,
        intent_id="intent-a",
        intent_digest=record.digest,
    )
    intent_digest = hashlib.sha256(b"intent-a").hexdigest()
    return PreparedInitialRestartIntentOpen(
        record=record,
        head=head,
        current=current,
        lease=HeldCoordinatorLease(
            record=CoordinatorLeaseRecord(
                run_id=RUN_ID,
                coordinator_id="coordinator-a",
                lease_id="lease-a",
                lease_duration_ms=100,
            ),
            fencing_token=7,
            granted_at_unix_ms=1_000,
        ),
        intent_key=f"{RUN_PREFIX}/restart-intents/{intent_digest}",
        intent_head_key=f"{RUN_PREFIX}/restart-intent-head",
        lifecycle_head_key=f"{RUN_PREFIX}/restart-intent-lifecycle-head",
        coordinator_lease_key=f"{RUN_PREFIX}/coordinator-lease",
        generation_head_key=f"{RUN_PREFIX}/generation-head",
        generation_snapshot_key=f"{RUN_PREFIX}/generations/0",
        not_before_unix_ms=1_000,
        deadline_unix_ms=1_050,
    )


def test_prepared_initial_open_builds_immutable_create_once_transaction_inputs():
    prepared = _prepared()

    assert set(prepared.writes) == {prepared.intent_head_key, prepared.intent_key}
    assert all(write.expected_revision is None for write in prepared.writes.values())
    assert all(write.require_never_created for write in prepared.writes.values())
    assert prepared.writes[prepared.intent_head_key].value == prepared.head.to_json()
    assert prepared.writes[prepared.intent_key].value == prepared.record.to_json()
    assert prepared.never_created_conditions == frozenset({prepared.lifecycle_head_key})
    assert prepared.conditions == {
        prepared.generation_head_key: prepared.current.head_revision,
        prepared.generation_snapshot_key: prepared.current.snapshot.revision,
    }
    with pytest.raises(TypeError):
        cast(Any, prepared.writes)["other"] = next(iter(prepared.writes.values()))
    with pytest.raises(TypeError):
        cast(Any, prepared.conditions)["other"] = 1


def test_prepared_initial_open_is_immutable():
    prepared = _prepared()

    with pytest.raises(AttributeError):
        prepared.intent_key = "other"


@pytest.mark.parametrize(
    "changes",
    [
        {"head": replace(_prepared().head, intent_id="other")},
        {"lease": replace(_prepared().lease, fencing_token=8)},
        {"intent_key": "other"},
        {"intent_head_key": "other"},
        {"lifecycle_head_key": "other"},
        {"coordinator_lease_key": "other"},
        {"generation_head_key": "other"},
        {"generation_snapshot_key": "other"},
        {"lease": replace(_prepared().lease, granted_at_unix_ms=1_001)},
        {"not_before_unix_ms": 999},
        {"deadline_unix_ms": 1_051},
    ],
)
def test_prepared_initial_open_rejects_inconsistent_authority(changes):
    with pytest.raises(ValueError):
        replace(_prepared(), **changes)


def test_prepared_initial_open_requires_expected_record_types():
    prepared = _prepared()

    with pytest.raises(TypeError, match="RestartIntentRecord"):
        replace(prepared, record={})
    with pytest.raises(TypeError, match="RestartIntentHeadRecord"):
        replace(prepared, head={})
    with pytest.raises(TypeError, match="CurrentGeneration"):
        replace(prepared, current={})
    with pytest.raises(TypeError, match="HeldCoordinatorLease"):
        replace(prepared, lease={})


def test_prepared_initial_open_binds_exact_generation_snapshot():
    prepared = _prepared()
    changed_snapshot = replace(
        prepared.current.snapshot,
        record=replace(
            prepared.current.snapshot.record,
            coordinator_id="other-coordinator",
        ),
    )

    with pytest.raises(ValueError, match="generation"):
        replace(
            prepared,
            current=replace(prepared.current, snapshot=changed_snapshot),
        )


def test_prepared_initial_open_rejects_suspects_outside_generation():
    prepared = _prepared()
    record = replace(
        prepared.record,
        intent=_intent(suspected_node_ids=("node-c",)),
    )
    head = replace(
        prepared.head,
        intent_digest=record.digest,
    )

    with pytest.raises(ValueError, match="outside"):
        replace(prepared, record=record, head=head)


def test_prepared_initial_open_binds_complete_held_lease():
    prepared = _prepared()
    longer_lease = replace(
        prepared.lease,
        record=replace(
            prepared.lease.record,
            lease_duration_ms=1_000,
        ),
    )

    with pytest.raises(ValueError, match="lease does not authorize"):
        replace(prepared, lease=longer_lease)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("not_before_unix_ms", 0),
        ("deadline_unix_ms", 0),
    ],
)
def test_prepared_initial_open_rejects_invalid_integer_fields(field, value):
    with pytest.raises(ValueError, match="positive integer"):
        replace(_prepared(), **{field: value})


def test_prepared_initial_open_rejects_invalid_generation_revisions():
    prepared = _prepared()

    with pytest.raises(ValueError, match="positive integer"):
        replace(
            prepared,
            current=replace(prepared.current, head_revision=False),
        )
    with pytest.raises(ValueError, match="positive integer"):
        replace(
            prepared,
            current=replace(
                prepared.current,
                snapshot=replace(prepared.current.snapshot, revision=0),
            ),
        )


def test_prepared_initial_open_rejects_invalid_time_ordering():
    prepared = _prepared()

    with pytest.raises(ValueError, match="generation snapshot"):
        replace(
            prepared,
            current=replace(
                prepared.current,
                snapshot=replace(
                    prepared.current.snapshot,
                    committed_at_unix_ms=1_001,
                ),
            ),
        )
    with pytest.raises(ValueError, match="precede its deadline"):
        replace(
            prepared,
            not_before_unix_ms=prepared.deadline_unix_ms,
        )


def test_prepared_initial_open_uses_canonical_hashed_keys():
    prepared = _prepared()

    assert RUN_ID not in prepared.intent_key
    assert "intent-a" not in prepared.intent_key
    assert prepared.intent_key.startswith(f"{RUN_PREFIX}/restart-intents/")
