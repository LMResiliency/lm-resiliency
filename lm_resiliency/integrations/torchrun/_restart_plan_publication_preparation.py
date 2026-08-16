"""Authenticated, non-mutating preparation of restart-plan publication authority."""

from __future__ import annotations

import threading
from collections.abc import Callable

from lm_resiliency.integrations.torchrun._control_store import ControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease_history import (
    CoordinatorLeaseAuthority,
    CoordinatorLeaseHistoryCorrupt,
    CoordinatorLeaseHistoryError,
    CoordinatorLeaseHistoryReader,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_authority import (
    RestartPlanPublicationAuthority,
)
from lm_resiliency.integrations.torchrun._restart_plan_publication_records import (
    RestartPlanPublicationRecords,
)

_MAX_READ_ATTEMPTS = 8


class RestartPlanPublicationPreparationError(RuntimeError):
    """Base error for authenticating restart-plan publication authority."""


class RestartPlanPublicationPreparationConflict(RestartPlanPublicationPreparationError):
    """Raised when coordinator authority changes repeatedly during preparation."""


class RestartPlanPublicationPreparationLeaseLost(RestartPlanPublicationPreparationError):
    """Raised when the plan's coordinator authority is absent, stale, or expired."""


class RestartPlanPublicationPreparationCorrupt(RestartPlanPublicationPreparationError):
    """Raised when durable coordinator authority is contradictory."""


class RestartPlanPublicationPreparationClockError(RestartPlanPublicationPreparationError):
    """Raised when the coordinator preparation clock is invalid."""


class RestartPlanPublicationAuthorityPreparer:
    """Authenticate publication records against the live coordinator lease."""

    def __init__(
        self,
        store: ControlStore,
        *,
        run_id: str,
        clock: Callable[[], int],
    ) -> None:
        self._run_id = _nonempty_string(run_id, "run_id")
        if not callable(clock):
            raise TypeError("clock must be callable")
        self._clock = clock
        self._clock_lock = threading.Lock()
        self._last_now_unix_ms = 0
        self._lease_history_reader = CoordinatorLeaseHistoryReader(
            store,
            run_id=self._run_id,
        )

    def prepare(
        self,
        records: RestartPlanPublicationRecords,
    ) -> RestartPlanPublicationAuthority:
        """Return authenticated, non-mutating publication authority."""

        if not isinstance(records, RestartPlanPublicationRecords):
            raise TypeError("records must be RestartPlanPublicationRecords")
        if records.candidate.plan.run_id != self._run_id:
            raise ValueError("restart-plan publication records belong to another run")
        authority, now_unix_ms = self._read_stable_authority()
        self._require_exact_authority(records, authority)
        required_observation = max(
            authority.lease.granted_at_unix_ms,
            records.current.snapshot.committed_at_unix_ms,
            records.candidate.placement_state.observed_at_unix_ms,
            records.candidate.recovery_state.copy_state.inventory_state.quarantine_state.manifest_state.resolved_manifest.source_snapshot.committed_at_unix_ms,
        )
        if now_unix_ms < required_observation:
            raise RestartPlanPublicationPreparationClockError(
                "restart-plan publication clock precedes authoritative inputs"
            )
        deadline_unix_ms = min(
            records.deadline_unix_ms,
            authority.lease.expires_at_unix_ms,
        )
        if now_unix_ms >= deadline_unix_ms:
            raise RestartPlanPublicationPreparationLeaseLost(
                "restart-plan publication authority window elapsed before preparation"
            )
        try:
            return RestartPlanPublicationAuthority(
                records=records,
                coordinator_authority=authority,
                observed_at_unix_ms=now_unix_ms,
            )
        except (TypeError, ValueError) as error:
            raise RestartPlanPublicationPreparationCorrupt(
                "durable coordinator authority cannot authorize restart-plan publication"
            ) from error

    def _read_stable_authority(
        self,
    ) -> tuple[CoordinatorLeaseAuthority, int]:
        for _ in range(_MAX_READ_ATTEMPTS):
            history = self._read_history()
            now_unix_ms = self._now_unix_ms()
            if history != self._read_history():
                continue
            if not history:
                raise RestartPlanPublicationPreparationLeaseLost(
                    "no live coordinator lease can authorize restart-plan publication"
                )
            return history[-1], now_unix_ms
        raise RestartPlanPublicationPreparationConflict(
            "coordinator lease history changed repeatedly during publication preparation"
        )

    def _read_history(self) -> tuple[CoordinatorLeaseAuthority, ...]:
        try:
            return self._lease_history_reader.read()
        except CoordinatorLeaseHistoryCorrupt as error:
            raise RestartPlanPublicationPreparationCorrupt(
                "coordinator lease history is corrupt"
            ) from error
        except CoordinatorLeaseHistoryError as error:
            raise RestartPlanPublicationPreparationConflict(
                "coordinator lease history changed repeatedly during preparation"
            ) from error

    def _require_exact_authority(
        self,
        records: RestartPlanPublicationRecords,
        authority: CoordinatorLeaseAuthority,
    ) -> None:
        plan_record = records.candidate.placement_state.generation_state.record
        lease = authority.lease
        if (
            lease.record.run_id != self._run_id
            or lease.record.coordinator_id != plan_record.coordinator_id
            or lease.record.lease_id != plan_record.lease_id
            or lease.record.lease_duration_ms != plan_record.coordinator_lease_duration_ms
            or lease.fencing_token != plan_record.coordinator_fencing_token
        ):
            raise RestartPlanPublicationPreparationLeaseLost(
                "restart plan is not authorized by the live coordinator lease"
            )

    def _now_unix_ms(self) -> int:
        with self._clock_lock:
            try:
                now_unix_ms = _positive_integer(
                    self._clock(),
                    "restart-plan publication preparation clock",
                )
            except (TypeError, ValueError) as error:
                raise RestartPlanPublicationPreparationClockError(
                    "restart-plan publication preparation clock is invalid"
                ) from error
            if now_unix_ms < self._last_now_unix_ms:
                raise RestartPlanPublicationPreparationClockError(
                    "restart-plan publication preparation clock moved backward"
                )
            self._last_now_unix_ms = now_unix_ms
            return now_unix_ms


def _nonempty_string(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{path} must be a non-empty string")
    return value


def _positive_integer(value: object, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{path} must be a positive integer")
    return value


__all__ = [
    "RestartPlanPublicationAuthorityPreparer",
    "RestartPlanPublicationPreparationClockError",
    "RestartPlanPublicationPreparationConflict",
    "RestartPlanPublicationPreparationCorrupt",
    "RestartPlanPublicationPreparationError",
    "RestartPlanPublicationPreparationLeaseLost",
]
