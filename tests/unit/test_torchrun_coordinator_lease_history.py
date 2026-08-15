"""Contract tests for torchrun coordinator lease authority values."""

from __future__ import annotations

import dataclasses

import pytest

from lm_resiliency.integrations.torchrun._control_store import ControlStoreEntry
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseRecord,
    HeldCoordinatorLease,
)
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseAuthorityCorrupt,
)


def _record(*, run_id: str = "training-run") -> CoordinatorLeaseRecord:
    return CoordinatorLeaseRecord(
        run_id=run_id,
        coordinator_id="coordinator-a",
        lease_id="lease-a",
        lease_duration_ms=100,
    )


def _entry(
    *,
    record: CoordinatorLeaseRecord | None = None,
    committed_at_unix_ms: int | None = 1_000,
    mutation_sequence: int = 1,
    value_sequence: int = 1,
    lifetime_sequence: int = 1,
) -> ControlStoreEntry:
    lease_record = record or _record()
    return ControlStoreEntry(
        value=lease_record.to_json(),
        revision=7,
        committed_at_unix_ms=committed_at_unix_ms,
        transaction_sequence=11,
        mutation_sequence=mutation_sequence,
        value_sequence=value_sequence,
        lifetime_sequence=lifetime_sequence,
    )


def test_coordinator_lease_authority_decodes_canonical_entry():
    entry = _entry()

    authority = CoordinatorLeaseAuthority.from_entry(
        entry,
        run_id="training-run",
    )

    assert authority == CoordinatorLeaseAuthority(
        lease=HeldCoordinatorLease(
            record=_record(),
            fencing_token=7,
            granted_at_unix_ms=1_000,
        ),
        transaction_sequence=11,
        mutation_sequence=1,
        value_sequence=1,
        lifetime_sequence=1,
    )


def test_coordinator_lease_authority_is_immutable():
    authority = CoordinatorLeaseAuthority.from_entry(
        _entry(),
        run_id="training-run",
    )

    with pytest.raises(dataclasses.FrozenInstanceError):
        authority.mutation_sequence = 2


@pytest.mark.parametrize(
    ("mutation_sequence", "value_sequence", "lifetime_sequence", "message"),
    [
        (2, 1, 2, "mutation_sequence is too small"),
        (3, 1, 2, "value_sequence is too small"),
        (3, 3, 2, "value_sequence is too large"),
    ],
)
def test_coordinator_lease_authority_rejects_impossible_sequences(
    mutation_sequence: int,
    value_sequence: int,
    lifetime_sequence: int,
    message: str,
):
    with pytest.raises(ValueError, match=message):
        CoordinatorLeaseAuthority.from_entry(
            _entry(
                mutation_sequence=mutation_sequence,
                value_sequence=value_sequence,
                lifetime_sequence=lifetime_sequence,
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_transaction_before_mutation():
    with pytest.raises(ValueError, match="transaction_sequence is too small"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(
                _entry(mutation_sequence=2, value_sequence=1),
                transaction_sequence=1,
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_malformed_or_wrong_run():
    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="malformed"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(_entry(), value=b"not-json"),
            run_id="training-run",
        )

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="another run"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(record=_record(run_id="other-run")),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_noncanonical_bytes():
    entry = _entry()

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="noncanonical"):
        CoordinatorLeaseAuthority.from_entry(
            dataclasses.replace(
                entry,
                value=entry.value.replace(b",", b", "),
            ),
            run_id="training-run",
        )


def test_coordinator_lease_authority_requires_authoritative_commit_time():
    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="commit time"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(committed_at_unix_ms=None),
            run_id="training-run",
        )


def test_coordinator_lease_authority_rejects_guarded_entry():
    entry = dataclasses.replace(
        _entry(),
        guard_key="guard",
        guard_revision=1,
        guard_value_digest="a" * 64,
        guard_mutation_sequence=1,
        guard_value_sequence=1,
        guard_lifetime_sequence=1,
        guard_committed_at_unix_ms=1_000,
    )

    with pytest.raises(CoordinatorLeaseAuthorityCorrupt, match="guard provenance"):
        CoordinatorLeaseAuthority.from_entry(
            entry,
            run_id="training-run",
        )


@pytest.mark.parametrize("run_id", ("", " ", None, 1))
def test_coordinator_lease_authority_rejects_invalid_expected_run_id(run_id: object):
    with pytest.raises(ValueError, match="run_id"):
        CoordinatorLeaseAuthority.from_entry(
            _entry(),
            run_id=run_id,  # type: ignore[arg-type]
        )


def test_coordinator_lease_authority_rejects_wrong_entry_type():
    with pytest.raises(TypeError, match="ControlStoreEntry"):
        CoordinatorLeaseAuthority.from_entry(
            object(),
            run_id="training-run",
        )
