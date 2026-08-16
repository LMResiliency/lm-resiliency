"""Contract tests for pure restart-plan publication state."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import pytest

from lm_resiliency.integrations.torchrun._control_store import ControlStoreWrite
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication import (
    RestartPlanPublicationClockError,
    RestartPlanPublicationConflict,
    RestartPlanPublicationCorrupt,
    RestartPlanPublicationError,
    RestartPlanPublicationLeaseLost,
    RestartPlanPublicationPreparer,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_authority import (
    RestartPlanPublicationAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle import (
    RestartPlanPublicationLifecycleFence,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_lifecycle_reader import (
    RestartPlanPublicationLifecycleConflict,
    RestartPlanPublicationLifecycleCorrupt,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_preparation import (
    RestartPlanPublicationPreparationClockError,
    RestartPlanPublicationPreparationConflict,
    RestartPlanPublicationPreparationCorrupt,
    RestartPlanPublicationPreparationLeaseLost,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_state import (
    PreparedRestartPlanPublication,
)

RUN_ID = "training-run"


class FailingReader:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self._value = value
        self._error = error

    def read(self):
        if self._error is not None:
            raise self._error
        return self._value


class FailingPreparer:
    def __init__(self, value=None, error: Exception | None = None) -> None:
        self._value = value
        self._error = error

    def prepare(self, records):
        if self._error is not None:
            raise self._error
        return self._value


def _lease_authority(
    *,
    lease_id: str,
    fencing_token: int,
    transaction_sequence: int,
    mutation_sequence: int,
) -> CoordinatorLeaseAuthority:
    return CoordinatorLeaseAuthority(
        lease=HeldCoordinatorLease(
            record=CoordinatorLeaseRecord(
                run_id=RUN_ID,
                coordinator_id="coordinator-a",
                lease_id=lease_id,
                lease_duration_ms=500,
            ),
            fencing_token=fencing_token,
            granted_at_unix_ms=900 + mutation_sequence,
        ),
        transaction_sequence=transaction_sequence,
        mutation_sequence=mutation_sequence,
        value_sequence=mutation_sequence,
        lifetime_sequence=1,
    )


def _prepared() -> PreparedRestartPlanPublication:
    closing_authority = _lease_authority(
        lease_id="lease-a",
        fencing_token=3,
        transaction_sequence=3,
        mutation_sequence=3,
    )
    publication_authority = _lease_authority(
        lease_id="lease-a",
        fencing_token=4,
        transaction_sequence=6,
        mutation_sequence=4,
    )
    intent_record = object()
    lifecycle_record = object()
    generation_record = object()
    generation_snapshot = SimpleNamespace(record=generation_record)

    authority = Mock(spec=RestartPlanPublicationAuthority)
    authority.records = SimpleNamespace(
        candidate=SimpleNamespace(
            placement_state=SimpleNamespace(
                generation_state=SimpleNamespace(
                    intent_record=intent_record,
                    lifecycle_record=lifecycle_record,
                    from_snapshot=generation_record,
                )
            )
        ),
        current=SimpleNamespace(snapshot=generation_snapshot),
        run_prefix="lm_resiliency/torchrun/v1/runs/run-digest",
        writes={
            "generation-head": ControlStoreWrite(
                expected_revision=7,
                value=b"head",
            )
        },
        conditions={
            "source-generation": 11,
            "agent-registration": 13,
        },
    )
    authority.coordinator_authority = publication_authority
    authority.observed_at_unix_ms = 1_100
    authority.not_before_unix_ms = 1_100
    authority.deadline_unix_ms = 1_300

    lifecycle_fence = Mock(spec=RestartPlanPublicationLifecycleFence)
    lifecycle_fence.closure = SimpleNamespace(
        intent=intent_record,
        lifecycle=lifecycle_record,
        generation_snapshot=generation_snapshot,
        closing_authority=closing_authority,
        lease_history=(closing_authority, publication_authority),
        closed_at_unix_ms=1_050,
    )
    lifecycle_fence.conditions = {
        "restart-intent": 17,
        "restart-intent-head": 19,
    }
    return PreparedRestartPlanPublication(
        authority=authority,
        lifecycle_fence=lifecycle_fence,
    )


def test_prepared_publication_exposes_atomic_inputs():
    prepared = _prepared()

    assert prepared.writes == {
        "generation-head": ControlStoreWrite(
            expected_revision=7,
            value=b"head",
        )
    }
    assert prepared.conditions == {
        "source-generation": 11,
        "agent-registration": 13,
        "restart-intent": 17,
        "restart-intent-head": 19,
    }
    assert prepared.guard_key.endswith("/coordinator-lease")
    assert prepared.expected_guard_revision == 4
    assert prepared.not_before_unix_ms == 1_100
    assert prepared.deadline_unix_ms == 1_300


def test_prepared_publication_is_immutable():
    prepared = _prepared()

    with pytest.raises(AttributeError):
        prepared.authority = prepared.authority
    with pytest.raises(TypeError):
        prepared.conditions["other"] = 1


def test_prepared_publication_requires_exact_types():
    prepared = _prepared()

    with pytest.raises(TypeError, match="authority must be"):
        replace(prepared, authority=cast(Any, prepared.lifecycle_fence))
    with pytest.raises(TypeError, match="lifecycle_fence must be"):
        replace(prepared, lifecycle_fence=cast(Any, prepared.authority))


@pytest.mark.parametrize(
    "path",
    ["intent_record", "lifecycle_record", "from_snapshot", "current_snapshot"],
)
def test_prepared_publication_requires_exact_closed_lifecycle(path: str):
    prepared = _prepared()
    authority = prepared.authority
    generation_state = authority.records.candidate.placement_state.generation_state
    if path == "intent_record":
        generation_state.intent_record = object()
    elif path == "lifecycle_record":
        generation_state.lifecycle_record = object()
    elif path == "from_snapshot":
        generation_state.from_snapshot = object()
    else:
        authority.records.current.snapshot = object()

    with pytest.raises(ValueError, match="does not match its closed lifecycle"):
        PreparedRestartPlanPublication(
            authority=authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def test_prepared_publication_rejects_observation_before_closure():
    prepared = _prepared()
    prepared.authority.observed_at_unix_ms = 1_049

    with pytest.raises(ValueError, match="observation precedes closure"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def test_prepared_publication_requires_authority_in_durable_history():
    prepared = _prepared()
    prepared.lifecycle_fence.closure.lease_history = (
        prepared.lifecycle_fence.closure.closing_authority,
    )

    with pytest.raises(ValueError, match="absent from durable lease history"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def test_prepared_publication_accepts_closing_authority_for_publication():
    prepared = _prepared()
    closing_authority = prepared.lifecycle_fence.closure.closing_authority
    prepared.authority.coordinator_authority = closing_authority
    prepared.lifecycle_fence.closure.lease_history = (closing_authority,)

    recomposed = PreparedRestartPlanPublication(
        authority=prepared.authority,
        lifecycle_fence=prepared.lifecycle_fence,
    )

    assert recomposed.expected_guard_revision == closing_authority.lease.fencing_token


def test_prepared_publication_rejects_authority_before_closure_authority():
    prepared = _prepared()
    prepared.lifecycle_fence.closure.lease_history = tuple(
        reversed(prepared.lifecycle_fence.closure.lease_history)
    )

    with pytest.raises(ValueError, match="predates closure authority"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def test_prepared_publication_rejects_conflicting_condition_revision():
    prepared = _prepared()
    prepared.lifecycle_fence.conditions = {"source-generation": 12}

    with pytest.raises(ValueError, match="condition revisions disagree"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def test_prepared_publication_rejects_condition_target_overlap():
    prepared = _prepared()
    prepared.lifecycle_fence.conditions = {"generation-head": 7}

    with pytest.raises(ValueError, match="must not also be transaction targets"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


@pytest.mark.parametrize("location", ["conditions", "writes"])
def test_prepared_publication_rejects_guard_key_reuse(location: str):
    prepared = _prepared()
    guard_key = prepared.guard_key
    if location == "conditions":
        prepared.authority.records.conditions[guard_key] = 23
    else:
        prepared.authority.records.writes[guard_key] = ControlStoreWrite(
            expected_revision=None,
            value=b"guard",
        )

    with pytest.raises(ValueError, match="guard key must not"):
        PreparedRestartPlanPublication(
            authority=prepared.authority,
            lifecycle_fence=prepared.lifecycle_fence,
        )


def _preparer_for(prepared: PreparedRestartPlanPublication) -> RestartPlanPublicationPreparer:
    preparer = RestartPlanPublicationPreparer(
        cast(Any, object()),
        run_id=RUN_ID,
        clock=lambda: 1_100,
    )
    preparer._authority_preparer = cast(
        Any,
        FailingPreparer(value=prepared.authority),
    )
    preparer._lifecycle_reader = cast(
        Any,
        FailingReader(value=prepared.lifecycle_fence),
    )
    return preparer


def test_publication_preparer_composes_authenticated_inputs_without_mutation():
    expected = _prepared()
    preparer = _preparer_for(expected)

    prepared = preparer.prepare(expected.authority.records)

    assert prepared == expected


@pytest.mark.parametrize(
    ("dependency_error", "expected_error"),
    [
        (
            RestartPlanPublicationPreparationConflict("authority changed"),
            RestartPlanPublicationConflict,
        ),
        (
            RestartPlanPublicationPreparationLeaseLost("lease lost"),
            RestartPlanPublicationLeaseLost,
        ),
        (
            RestartPlanPublicationPreparationClockError("clock moved"),
            RestartPlanPublicationClockError,
        ),
        (
            RestartPlanPublicationPreparationCorrupt("lease corrupt"),
            RestartPlanPublicationCorrupt,
        ),
    ],
)
def test_publication_preparer_translates_authority_failures(
    dependency_error: RuntimeError,
    expected_error: type[RestartPlanPublicationError],
):
    expected = _prepared()
    preparer = _preparer_for(expected)
    preparer._authority_preparer = cast(Any, FailingPreparer(error=dependency_error))

    with pytest.raises(expected_error):
        preparer.prepare(expected.authority.records)


@pytest.mark.parametrize(
    ("dependency_error", "expected_error"),
    [
        (
            RestartPlanPublicationLifecycleConflict("lifecycle changed"),
            RestartPlanPublicationConflict,
        ),
        (
            RestartPlanPublicationLifecycleCorrupt("lifecycle corrupt"),
            RestartPlanPublicationCorrupt,
        ),
    ],
)
def test_publication_preparer_translates_lifecycle_failures(
    dependency_error: RuntimeError,
    expected_error: type[RestartPlanPublicationError],
):
    expected = _prepared()
    preparer = _preparer_for(expected)
    preparer._lifecycle_reader = cast(Any, FailingReader(error=dependency_error))

    with pytest.raises(expected_error):
        preparer.prepare(expected.authority.records)


def test_publication_preparer_classifies_cross_read_mismatch_as_conflict():
    expected = _prepared()
    preparer = _preparer_for(expected)
    expected.authority.observed_at_unix_ms = 1_049

    with pytest.raises(RestartPlanPublicationConflict, match="changed during preparation"):
        preparer.prepare(expected.authority.records)


def test_publication_preparer_requires_valid_constructor_inputs():
    with pytest.raises(ValueError, match="non-empty"):
        RestartPlanPublicationPreparer(
            cast(Any, object()),
            run_id="",
            clock=lambda: 1,
        )
    with pytest.raises(TypeError, match="callable"):
        RestartPlanPublicationPreparer(
            cast(Any, object()),
            run_id=RUN_ID,
            clock=cast(Any, None),
        )
