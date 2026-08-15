"""Automatic incident scheduler and framework-neutral evaluation runtime."""

from __future__ import annotations

import hashlib
import heapq
import json
import random
import threading
import time
from bisect import bisect_left
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import torch.distributed as dist

from lm_resiliency.fault_injection._json import freeze_json_mapping, thaw_json
from lm_resiliency.fault_injection.config import (
    ClockOrigin,
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultSpec,
    IncidentLifetime,
    RetriggerPolicy,
    SafetyClass,
)
from lm_resiliency.fault_injection.executors import (
    FaultExecutionRequest,
    FaultExecutionResult,
    FaultExecutor,
)
from lm_resiliency.fault_injection.frameworks import (
    TrainingContext,
    resolve_training_context,
)
from lm_resiliency.fault_injection.local import (
    LocalFaultEffect,
    LocalFaultExecutor,
)
from lm_resiliency.fault_injection.reports import (
    CampaignReport,
    FaultEvaluation,
    FaultInjectionRecord,
    InjectionStatus,
    LocalizationResult,
)
from lm_resiliency.fault_injection.state import (
    CampaignJournal,
    CampaignStateStore,
    MemoryCampaignStateStore,
)
from lm_resiliency.lifecycle import (
    register_automatic_cleanup,
    unregister_automatic_cleanup,
)


class UnsupportedFaultError(ValueError):
    """Raised before training when no configured executor supports a fault."""


@dataclass(frozen=True, slots=True)
class _StagedIncident:
    incident: FaultIncident
    attempt: int
    occurrence_id: str
    selected: bool


@dataclass(slots=True)
class _ScheduleCursor:
    incidents: tuple[FaultIncident, ...]
    expiration: bool = False
    _heap: list[tuple[int, int, int]] = field(init=False, default_factory=list)
    _cache: dict[int, tuple[tuple[FaultIncident, int], ...]] = field(
        init=False,
        default_factory=dict,
    )
    _high_water: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        for order, incident in enumerate(self.incidents):
            offset = int(incident.lifetime.iterations or 0) - 1 if self.expiration else 0
            if self.expiration and incident.lifetime.iterations is None:
                continue
            first = _next_trigger_at_or_after(incident, 1)
            if first is not None:
                heapq.heappush(self._heap, (first + offset, order, offset))

    def at(self, iteration: int) -> tuple[tuple[FaultIncident, int], ...]:
        cached = self._cache.get(iteration)
        if cached is not None:
            return cached
        if iteration < self._high_water:
            result: list[tuple[FaultIncident, int]] = []
            for incident in self.incidents:
                if self.expiration and incident.lifetime.iterations is None:
                    continue
                offset = int(incident.lifetime.iterations or 0) - 1 if self.expiration else 0
                start = iteration - offset
                if start > 0 and _trigger_contains_iteration(incident, start):
                    result.append((incident, start))
            return tuple(result)

        while self._heap and self._heap[0][0] < iteration:
            _scheduled, order, offset = heapq.heappop(self._heap)
            incident = self.incidents[order]
            next_start = _next_trigger_at_or_after(incident, iteration - offset)
            if next_start is not None:
                heapq.heappush(self._heap, (next_start + offset, order, offset))

        matched: list[tuple[int, int, int]] = []
        while self._heap and self._heap[0][0] == iteration:
            matched.append(heapq.heappop(self._heap))
        for item in matched:
            heapq.heappush(self._heap, item)
        result = tuple(
            (self.incidents[order], iteration - offset) for _scheduled, order, offset in matched
        )
        self._high_water = iteration
        self._cache[iteration] = result
        while len(self._cache) > 8:
            self._cache.pop(next(iter(self._cache)))
        return result


@dataclass(slots=True)
class _CampaignSchedule:
    incidents: tuple[FaultIncident, ...]
    _candidates: _ScheduleCursor = field(init=False)
    _expirations: _ScheduleCursor = field(init=False)

    def __post_init__(self) -> None:
        self._candidates = _ScheduleCursor(self.incidents)
        self._expirations = _ScheduleCursor(self.incidents, expiration=True)

    def candidates(self, iteration: int) -> tuple[FaultIncident, ...]:
        return tuple(incident for incident, _start in self._candidates.at(iteration))

    def expirations(self, iteration: int) -> tuple[tuple[FaultIncident, int], ...]:
        return self._expirations.at(iteration)


@dataclass(slots=True)
class _ExternalEffect:
    request: FaultExecutionRequest
    record: FaultInjectionRecord
    executor: FaultExecutor
    result: FaultExecutionResult
    done: bool = False
    lock: threading.RLock = field(default_factory=threading.RLock, repr=False)

    def complete(self, *, cancelled: bool = False) -> None:
        with self.lock:
            if self.done:
                return
            try:
                evidence = self.executor.deactivate(self.request, self.result)
                normalized_evidence = _json_mapping(
                    evidence,
                    "fault executor deactivation evidence",
                )
            except BaseException as error:
                with self.record._lock:
                    self.record.status = InjectionStatus.FAILED
                    self.record.error = f"fault deactivation failed: {error}"
                    self.record.completed_at_ns = time.monotonic_ns()
                self.done = True
                raise
            with self.record._lock:
                if normalized_evidence:
                    merged = dict(self.record.evidence)
                    merged.update(normalized_evidence)
                    self.record.evidence = merged
                self.record.status = (
                    InjectionStatus.COMPLETED
                    if self.record.verified and not cancelled
                    else InjectionStatus.CANCELLED
                )
                self.record.completed_at_ns = time.monotonic_ns()
            self.done = True

    def rollback(self, error: BaseException) -> None:
        with self.lock:
            if self.done:
                return
            cleanup_error: BaseException | None = None
            if self.result.active:
                try:
                    self.executor.deactivate(self.request, self.result)
                except BaseException as caught:
                    cleanup_error = caught
            with self.record._lock:
                self.record.status = InjectionStatus.FAILED
                self.record.error = f"fault activation rolled back: {error}"
                if cleanup_error is not None:
                    self.record.error += f"; rollback cleanup also failed: {cleanup_error}"
                self.record.completed_at_ns = time.monotonic_ns()
            self.done = True
            if cleanup_error is not None:
                raise RuntimeError(
                    f"fault rollback cleanup failed: {cleanup_error}"
                ) from cleanup_error


@dataclass(slots=True)
class _ActiveFault:
    incident: FaultIncident
    start_iteration: int
    effect: LocalFaultEffect | _ExternalEffect

    @property
    def done(self) -> bool:
        return self.effect.done

    def complete(
        self,
        *,
        cancelled: bool = False,
        preserve_replaced_state: bool = False,
    ) -> None:
        if isinstance(self.effect, LocalFaultEffect):
            self.effect.complete(
                cancelled=cancelled,
                preserve_replaced_state=preserve_replaced_state,
            )
        else:
            self.effect.complete(cancelled=cancelled)

    def rollback(self, error: BaseException) -> None:
        if isinstance(self.effect, LocalFaultEffect):
            self.effect.fail(
                RuntimeError(f"fault activation rolled back: {error}"),
                propagate_cleanup_error=True,
            )
        else:
            self.effect.rollback(error)


class FaultInjectionSession:
    """Automatically scheduled rank-local campaign runtime."""

    def __init__(
        self,
        target: Any,
        optimizer: Any | None,
        *,
        campaign: FaultCampaign,
        completed_iterations: int | None = None,
        state_store: CampaignStateStore | None = None,
        executors: Sequence[FaultExecutor] = (),
        rank: int | None = None,
        _defer_activation: bool = False,
    ) -> None:
        if not isinstance(campaign, FaultCampaign):
            raise TypeError("campaign must be a FaultCampaign")
        self.campaign = campaign
        self._context: TrainingContext = resolve_training_context(target, optimizer)
        self.framework = self._context.framework
        distributed_rank = _distributed_rank()
        self.rank = distributed_rank if rank is None else int(rank)
        if self.rank < 0:
            raise ValueError("fault injection rank must be non-negative")
        if dist.is_available() and dist.is_initialized() and self.rank != distributed_rank:
            raise ValueError(
                f"fault injection rank override {self.rank} does not match "
                f"distributed rank {distributed_rank}"
            )
        if campaign.clock.origin is ClockOrigin.CAMPAIGN_START:
            completed = 0
        elif completed_iterations is None:
            completed = self._context.inferred_completed_iterations
        else:
            if isinstance(completed_iterations, bool) or not isinstance(
                completed_iterations,
                int,
            ):
                raise TypeError("completed_iterations must be an integer")
            completed = completed_iterations
        if completed < 0:
            raise ValueError("completed_iterations must be non-negative")
        self._completed_iterations = completed
        self._current_iteration = completed + 1
        self._state_store = MemoryCampaignStateStore() if state_store is None else state_store
        loaded_journal = self._state_store.load(campaign.name)
        self._journal_base = CampaignJournal.from_dict(loaded_journal.to_dict())
        self._journal = CampaignJournal.from_dict(loaded_journal.to_dict())
        self._journal.bind_manifest(campaign.manifest_identity)
        self._journal_committed = False
        self._executors = tuple(executors)
        self._local = LocalFaultExecutor(self._context, self.rank)
        self._records: list[FaultInjectionRecord] = []
        self._records_lock = threading.RLock()
        self._lifecycle_lock = threading.RLock()
        self._active: list[_ActiveFault] = []
        self._closed = False
        self._started = False
        self._faults = tuple(fault for incident in campaign.incidents for fault in incident.faults)
        self._schedule = _CampaignSchedule(campaign.incidents)

        try:
            self._validate_capabilities()
            self._local.validate_targets(self._faults)
            self._local.validate_schedule(self.campaign.incidents)
            self._local.sync_history(self._history_faults_for(self._current_iteration))
            self._preflight_current_iteration()
            self._context.register_step_callback(self._on_step_complete)
            self._context.register_state_replacement_callback(self._mark_state_replaced)
            register_automatic_cleanup(self)
            if not _defer_activation:
                self._commit_journal_binding()
                self._start()
        except BaseException as error:
            cleanup_error = self._cleanup()
            if cleanup_error is not None:
                _add_exception_note(
                    error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise

    @property
    def completed_iterations(self) -> int:
        return self._completed_iterations

    @property
    def current_iteration(self) -> int:
        return self._current_iteration

    @property
    def records(self) -> tuple[FaultInjectionRecord, ...]:
        with self._records_lock:
            return tuple(self._records)

    @property
    def journal_attempts_identity(self) -> str:
        """Return a stable digest of restart-sensitive occurrence attempts."""
        encoded = json.dumps(
            self._journal.attempts,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    @property
    def supported_failure_types(self) -> frozenset[FailureType]:
        supported: set[FailureType] = set()
        for failure_type in FailureType:
            for fault in self._faults:
                if fault.type is not failure_type:
                    continue
                if self._local.supports(fault) or any(
                    executor.supports(fault) for executor in self._executors
                ):
                    supported.add(failure_type)
                    break
        return frozenset(supported)

    def notify_recovery(self) -> None:
        """End permanent effects configured to last until recovery."""
        with self._lifecycle_lock:
            self._complete_until("recovery")

    def notify_replacement(self) -> None:
        """End permanent effects configured to last until replacement."""
        with self._lifecycle_lock:
            self._complete_until("replacement")

    def evaluate(
        self,
        results: Sequence[LocalizationResult | Mapping[str, Any]] = (),
    ) -> CampaignReport:
        """Compare neutral localization results with verified incident ground truth."""
        normalized = tuple(
            LocalizationResult.from_dict(result) if isinstance(result, Mapping) else result
            for result in results
        )
        if not all(isinstance(result, LocalizationResult) for result in normalized):
            raise TypeError("localization results must be LocalizationResult instances or mappings")
        by_occurrence: dict[str, list[LocalizationResult]] = {}
        for result in normalized:
            by_occurrence.setdefault(result.occurrence_id, []).append(result)
        with self._lifecycle_lock:
            with self._records_lock:
                records = tuple(record.snapshot() for record in self._records)
            completed_iterations = self.completed_iterations
        grouped = _group_records(records)
        unknown = sorted(set(by_occurrence) - set(grouped))
        if unknown:
            raise ValueError(f"localization results reference unknown occurrences: {unknown}")

        evaluations = tuple(
            _evaluate_occurrence(
                occurrence_id,
                occurrence_records,
                tuple(by_occurrence.get(occurrence_id, ())),
                next(
                    incident
                    for incident in self.campaign.incidents
                    if incident.incident_id == occurrence_records[0].incident_id
                ),
            )
            for occurrence_id, occurrence_records in grouped.items()
        )
        return CampaignReport(
            campaign=self.campaign.name,
            manifest_identity=self.campaign.manifest_identity,
            manifest=self.campaign.to_dict(),
            framework=self.framework,
            rank=self.rank,
            completed_iterations=completed_iterations,
            injections=records,
            localizations=normalized,
            evaluations=evaluations,
            metadata=self.campaign.metadata,
        )

    def close(self) -> None:
        if self._closed:
            return
        first_error = self._cleanup(complete_campaign_end=True)
        if first_error is not None:
            raise RuntimeError("fault injection cleanup failed") from first_error

    def _start(self) -> None:
        """Arm the current iteration after preparation has succeeded."""
        if self._closed:
            raise RuntimeError("cannot start a closed fault injection session")
        if self._started:
            return
        if not self._journal_committed:
            raise RuntimeError("campaign journal binding has not been committed")
        self._enter_iteration_consistently(self._current_iteration)
        self._started = True

    def _commit_journal_binding(self) -> None:
        if self._journal_committed:
            return
        if not self._state_store.compare_and_swap(self._journal_base, self._journal):
            current = self._state_store.load(self.campaign.name)
            if current.to_dict() != self._journal.to_dict():
                raise RuntimeError("campaign state changed concurrently during manifest binding")
        self._journal_base = CampaignJournal.from_dict(self._journal.to_dict())
        self._journal_committed = True

    def _cleanup(
        self,
        *,
        complete_campaign_end: bool = False,
    ) -> BaseException | None:
        """Best-effort cleanup shared by failed construction and close()."""
        with self._lifecycle_lock:
            return self._cleanup_locked(complete_campaign_end=complete_campaign_end)

    def _cleanup_locked(
        self,
        *,
        complete_campaign_end: bool,
    ) -> BaseException | None:
        if self._closed:
            return None
        unregister_automatic_cleanup(self)
        first_error: BaseException | None = None
        for active in reversed(self._active):
            if not active.done:
                try:
                    active.complete(
                        cancelled=(
                            not complete_campaign_end
                            or active.incident.lifetime.until != "campaign_end"
                        ),
                    )
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        self._active.clear()
        try:
            self._local.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        try:
            self._context.close()
        except BaseException as error:
            if first_error is None:
                first_error = error
        self._closed = True
        return first_error

    def __enter__(self) -> "FaultInjectionSession":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        try:
            if _exc is None:
                self.close()
            else:
                cleanup_error = self._cleanup()
                if cleanup_error is not None:
                    raise RuntimeError("fault injection cleanup failed") from cleanup_error
        except Exception as cleanup_error:
            if _exc is None:
                raise
            _add_exception_note(
                _exc,
                f"fault injection cleanup also failed: {cleanup_error}",
            )

    def _validate_capabilities(self) -> None:
        unsupported: list[str] = []
        for incident in self.campaign.incidents:
            for fault in incident.faults:
                if fault.target.execution_rank != self.rank:
                    continue
                if self._local.supports(fault):
                    continue
                if any(executor.supports(fault) for executor in self._executors):
                    continue
                unsupported.append(
                    f"{incident.incident_id}/{fault.fault_id}: "
                    f"{fault.type.value} on {fault.target.surface.value}"
                )
        if unsupported:
            joined = "; ".join(unsupported)
            raise UnsupportedFaultError(f"campaign has no configured executor for: {joined}")

    def _preflight_current_iteration(self) -> None:
        self._preflight_iteration(self._current_iteration)

    def _preflight_iteration(self, iteration: int) -> None:
        requests = self._requests_for_iteration(iteration)
        required_history_occurrences = frozenset(
            _occurrence_id(incident, iteration, attempt)
            for incident, attempt in self._selected_incidents_for_iteration(iteration)
            if incident.safety is not SafetyClass.SAFE_IN_PROCESS
        )
        self._local.validate_activations(
            requests,
            required_history_occurrences=required_history_occurrences,
        )
        for request in requests:
            if self._local.supports(request.fault):
                continue
            executor = self._executor_for(request.fault)
            self._validate_executor_lifecycle(executor)
            if request.lifetime.matching_calls is not None and not getattr(
                executor,
                "completes_inline",
                False,
            ):
                raise ValueError(
                    f"fault executor {executor.name!r} must declare "
                    "completes_inline=True for matching_calls incidents"
                )
            if _requires_active_external_effect(request.lifetime) and getattr(
                executor,
                "one_shot",
                False,
            ):
                raise ValueError(
                    f"fault executor {executor.name!r} cannot use a one-shot result "
                    "for a multi-iteration or until lifetime"
                )
            validate = getattr(executor, "validate", None)
            if callable(validate):
                validate(request)

    @staticmethod
    def _validate_executor_lifecycle(executor: FaultExecutor) -> None:
        if getattr(executor, "can_deactivate", True) is False and not getattr(
            executor,
            "one_shot",
            False,
        ):
            raise ValueError(
                f"fault executor {executor.name!r} without deactivation must declare one_shot=True"
            )

    def _requests_for_iteration(self, iteration: int) -> tuple[FaultExecutionRequest, ...]:
        requests: list[FaultExecutionRequest] = []
        for incident, attempt in self._selected_incidents_for_iteration(iteration):
            occurrence_id = _occurrence_id(incident, iteration, attempt)
            for fault in incident.faults:
                if fault.target.execution_rank != self.rank:
                    continue
                requests.append(
                    self._build_request(
                        incident,
                        fault,
                        occurrence_id,
                        iteration,
                        attempt,
                    )
                )
        return tuple(requests)

    def _has_safe_current_activation(self) -> bool:
        return self._has_safe_activation(self._current_iteration)

    def _has_safe_activation(self, iteration: int) -> bool:
        selected = self._selected_incidents_for_iteration(iteration)
        faults = tuple(fault for incident, _attempt in selected for fault in incident.faults)
        return bool(faults) and all(fault.safety is SafetyClass.SAFE_IN_PROCESS for fault in faults)

    def _selected_incidents_for_iteration(
        self,
        iteration: int,
    ) -> tuple[tuple[FaultIncident, int], ...]:
        return tuple(
            (incident, attempt)
            for incident, attempt in self._candidate_incidents_for_iteration(iteration)
            if _probability_selected(
                self.campaign.seed,
                incident.incident_id,
                iteration,
                incident.trigger.probability,
            )
        )

    def _candidate_incidents_for_iteration(
        self,
        iteration: int,
    ) -> tuple[tuple[FaultIncident, int], ...]:
        candidates: list[tuple[FaultIncident, int]] = []
        for incident in self._schedule.candidates(iteration):
            attempt_count = self._journal.attempt_count(incident.incident_id, iteration)
            if incident.retrigger is RetriggerPolicy.ONCE and attempt_count >= 1:
                continue
            if incident.retrigger is RetriggerPolicy.MAX_OCCURRENCES and attempt_count >= int(
                incident.max_occurrences or 0
            ):
                continue
            candidates.append((incident, attempt_count + 1))
        return tuple(candidates)

    def _on_step_complete(self, optimizer_error: BaseException | None = None) -> None:
        with self._lifecycle_lock:
            self._on_step_complete_locked(optimizer_error)

    def _on_step_complete_locked(
        self,
        optimizer_error: BaseException | None = None,
    ) -> None:
        if self._closed:
            if optimizer_error is not None:
                raise optimizer_error
            return
        finished = self._current_iteration
        next_iteration = finished + 1
        needs_consensus = _distributed_world_size() > 1 and self._requires_boundary_consensus(
            finished, next_iteration
        )
        preparation_error = optimizer_error
        if optimizer_error is None:
            try:
                expiration_error: BaseException | None = None
                for active in self._active:
                    lifetime = active.incident.lifetime
                    if (
                        not active.done
                        and lifetime.iterations is not None
                        and finished >= active.start_iteration + lifetime.iterations - 1
                    ):
                        try:
                            active.complete()
                        except BaseException as error:
                            if expiration_error is None:
                                expiration_error = error
                self._discard_completed()
                if expiration_error is not None:
                    raise expiration_error
                self._completed_iterations = finished
                self._current_iteration = next_iteration
                self._local.sync_history(self._history_faults_for(self._current_iteration))
            except BaseException as error:
                preparation_error = error
        if needs_consensus:
            stage = "optimizer step" if optimizer_error is not None else "iteration preparation"
            failures = self._gather_runtime_rank_errors(
                preparation_error,
                stage,
            )
            if failures:
                cleanup_error = self._cleanup()
                boundary_error = RuntimeError(
                    f"fault injection {stage} failed; " + "; ".join(failures)
                )
                if cleanup_error is not None:
                    _add_exception_note(
                        boundary_error,
                        f"fault injection cleanup also failed: {cleanup_error}",
                    )
                if optimizer_error is not None:
                    _add_exception_note(optimizer_error, str(boundary_error))
                    if cleanup_error is not None:
                        _add_exception_note(
                            optimizer_error,
                            f"fault injection cleanup also failed: {cleanup_error}",
                        )
                    raise optimizer_error
                if preparation_error is not None and not isinstance(preparation_error, Exception):
                    _add_exception_note(preparation_error, str(boundary_error))
                    if cleanup_error is not None:
                        _add_exception_note(
                            preparation_error,
                            f"fault injection cleanup also failed: {cleanup_error}",
                        )
                    raise preparation_error
                raise boundary_error from preparation_error
        elif preparation_error is not None:
            cleanup_error = self._cleanup()
            if cleanup_error is not None:
                _add_exception_note(
                    preparation_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise preparation_error
        self._enter_iteration_consistently(self._current_iteration)

    def _requires_boundary_consensus(
        self,
        finished_iteration: int,
        next_iteration: int,
    ) -> bool:
        if self._candidate_incidents_for_iteration(next_iteration):
            return True
        if self._history_faults_for(next_iteration):
            return True
        for incident, start_iteration in self._schedule.expirations(finished_iteration):
            if any(fault.safety is not SafetyClass.SAFE_IN_PROCESS for fault in incident.faults):
                continue
            if not _probability_selected(
                self.campaign.seed,
                incident.incident_id,
                start_iteration,
                incident.trigger.probability,
            ):
                continue
            if self._journal.attempt_count(incident.incident_id, start_iteration) > 0:
                return True
        return False

    def _enter_iteration_consistently(self, iteration: int) -> None:
        candidates = self._candidate_incidents_for_iteration(iteration)
        if not candidates:
            return
        requires_activation_consensus = self._has_safe_activation(iteration)
        if _distributed_world_size() <= 1:
            try:
                self._preflight_iteration(iteration)
                staged = self._stage_iteration_attempts(iteration, candidates)
                self._activate_staged_iteration(iteration, staged)
            except BaseException as error:
                cleanup_error = self._cleanup()
                if cleanup_error is not None:
                    _add_exception_note(
                        error,
                        f"fault injection cleanup also failed: {cleanup_error}",
                    )
                raise
            return

        preflight_error: BaseException | None = None
        try:
            self._preflight_iteration(iteration)
        except BaseException as error:
            preflight_error = error
        failures = self._gather_runtime_rank_errors(
            preflight_error,
            "iteration preflight",
        )
        if failures:
            cleanup_error = self._cleanup()
            boundary_error = RuntimeError(
                "fault injection iteration preflight failed; " + "; ".join(failures)
            )
            if cleanup_error is not None:
                _add_exception_note(
                    boundary_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            if preflight_error is not None and not isinstance(preflight_error, Exception):
                _add_exception_note(preflight_error, str(boundary_error))
                if cleanup_error is not None:
                    _add_exception_note(
                        preflight_error,
                        f"fault injection cleanup also failed: {cleanup_error}",
                    )
                raise preflight_error
            raise boundary_error from preflight_error

        staged: tuple[_StagedIncident, ...] = ()
        previous_attempts = dict(self._journal.attempts)
        persistence_error: BaseException | None = None
        try:
            staged = self._stage_iteration_attempts(iteration, candidates)
        except BaseException as error:
            persistence_error = error
        failures = self._gather_runtime_rank_errors(
            persistence_error,
            "attempt persistence",
        )
        if failures:
            rollback_error: BaseException | None = None
            try:
                self._restore_journal_attempts(previous_attempts)
            except BaseException as error:
                rollback_error = error
            rollback_failures = self._gather_runtime_rank_errors(
                rollback_error,
                "attempt rollback",
            )
            cleanup_error = self._cleanup()
            boundary_error = RuntimeError(
                "fault injection attempt persistence failed; " + "; ".join(failures)
            )
            if rollback_failures:
                _add_exception_note(
                    boundary_error,
                    "fault injection attempt rollback also failed; " + "; ".join(rollback_failures),
                )
            if cleanup_error is not None:
                _add_exception_note(
                    boundary_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            if persistence_error is not None and not isinstance(
                persistence_error,
                Exception,
            ):
                _add_exception_note(persistence_error, str(boundary_error))
                if rollback_failures:
                    _add_exception_note(
                        persistence_error,
                        "fault injection attempt rollback also failed; "
                        + "; ".join(rollback_failures),
                    )
                if cleanup_error is not None:
                    _add_exception_note(
                        persistence_error,
                        f"fault injection cleanup also failed: {cleanup_error}",
                    )
                raise persistence_error
            raise boundary_error from persistence_error

        active_start = len(self._active)
        record_start = len(self._records)
        activation_error: BaseException | None = None
        try:
            self._activate_staged_iteration(iteration, staged)
        except BaseException as error:
            activation_error = error

        failures = (
            self._gather_runtime_rank_errors(
                activation_error,
                "iteration arming",
            )
            if requires_activation_consensus
            else []
        )
        if failures:
            boundary_error = RuntimeError(
                "fault injection iteration arming failed; " + "; ".join(failures)
            )
            rollback_error = self._rollback_new_activations(
                active_start,
                record_start,
                boundary_error,
            )
            cleanup_error = self._cleanup()
            if rollback_error is not None:
                _add_exception_note(
                    boundary_error,
                    f"fault activation rollback also failed: {rollback_error}",
                )
            if cleanup_error is not None:
                _add_exception_note(
                    boundary_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            if activation_error is not None and not isinstance(activation_error, Exception):
                _add_exception_note(activation_error, str(boundary_error))
                if rollback_error is not None:
                    _add_exception_note(
                        activation_error,
                        f"fault activation rollback also failed: {rollback_error}",
                    )
                if cleanup_error is not None:
                    _add_exception_note(
                        activation_error,
                        f"fault injection cleanup also failed: {cleanup_error}",
                    )
                raise activation_error
            raise boundary_error from activation_error
        if activation_error is not None:
            cleanup_error = self._cleanup()
            if cleanup_error is not None:
                _add_exception_note(
                    activation_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise activation_error

    def _stage_iteration_attempts(
        self,
        iteration: int,
        candidates: tuple[tuple[FaultIncident, int], ...],
    ) -> tuple[_StagedIncident, ...]:
        selections = tuple(
            (
                incident,
                expected_attempt,
                _probability_selected(
                    self.campaign.seed,
                    incident.incident_id,
                    iteration,
                    incident.trigger.probability,
                ),
            )
            for incident, expected_attempt in candidates
        )
        if not any(selected for _incident, _attempt, selected in selections):
            return tuple(
                _StagedIncident(
                    incident=incident,
                    attempt=expected_attempt,
                    occurrence_id=_occurrence_id(
                        incident,
                        iteration,
                        expected_attempt,
                    ),
                    selected=False,
                )
                for incident, expected_attempt, _selected in selections
            )

        previous = CampaignJournal.from_dict(self._journal.to_dict())
        updated = CampaignJournal.from_dict(self._journal.to_dict())
        staged: list[_StagedIncident] = []
        for incident, expected_attempt, selected in selections:
            if selected:
                attempt = updated.record_attempt(incident.incident_id, iteration)
                if attempt != expected_attempt:
                    raise RuntimeError(
                        f"campaign journal attempt for {incident.incident_id!r} changed "
                        f"from {expected_attempt} to {attempt} during staging"
                    )
            else:
                attempt = expected_attempt
            staged.append(
                _StagedIncident(
                    incident=incident,
                    attempt=attempt,
                    occurrence_id=_occurrence_id(incident, iteration, attempt),
                    selected=selected,
                )
            )
        if not self._state_store.compare_and_swap(previous, updated):
            raise RuntimeError("campaign state changed concurrently while claiming occurrences")
        self._journal = updated
        return tuple(staged)

    def _gather_runtime_rank_errors(
        self,
        error: BaseException | None,
        stage: str,
    ) -> list[str]:
        try:
            return _gather_rank_errors(error)
        except BaseException as collective_error:
            cleanup_error = self._cleanup()
            _add_exception_note(
                collective_error,
                f"fault injection {stage} consensus failed",
            )
            if cleanup_error is not None:
                _add_exception_note(
                    collective_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise

    def _restore_journal_attempts(self, attempts: dict[str, int]) -> None:
        current = CampaignJournal.from_dict(self._journal.to_dict())
        restored = CampaignJournal.from_dict(self._journal.to_dict())
        restored.attempts.clear()
        restored.attempts.update(attempts)
        if not self._state_store.compare_and_swap(current, restored):
            raise RuntimeError("campaign state changed concurrently during attempt rollback")
        self._journal = restored

    def _activate_staged_iteration(
        self,
        iteration: int,
        staged: tuple[_StagedIncident, ...],
    ) -> None:
        active_start = len(self._active)
        record_start = len(self._records)
        try:
            for item in staged:
                incident = item.incident
                if not item.selected:
                    self._record_probability_skip(
                        incident,
                        item.occurrence_id,
                        iteration,
                        item.attempt,
                    )
                    continue
                self._activate_incident(
                    incident,
                    item.occurrence_id,
                    iteration,
                    item.attempt,
                )
        except BaseException as error:
            cleanup_error = self._rollback_new_activations(
                active_start,
                record_start,
                error,
            )
            if cleanup_error is not None:
                _add_exception_note(error, f"fault rollback also failed: {cleanup_error}")
            raise

    def _rollback_new_activations(
        self,
        active_start: int,
        record_start: int,
        error: BaseException,
    ) -> BaseException | None:
        cleanup_error: BaseException | None = None
        for active in reversed(self._active[active_start:]):
            if not active.done:
                try:
                    active.rollback(error)
                except BaseException as caught:
                    if cleanup_error is None:
                        cleanup_error = caught
        del self._active[active_start:]
        with self._records_lock:
            for record in self._records[record_start:]:
                with record._lock:
                    if record.status in {
                        InjectionStatus.SKIPPED_PROBABILITY,
                        InjectionStatus.FAILED,
                    }:
                        continue
                    record.status = InjectionStatus.FAILED
                    record.error = f"fault activation rolled back: {error}"
                    record.completed_at_ns = time.monotonic_ns()
        return cleanup_error

    def _activate_incident(
        self,
        incident: FaultIncident,
        occurrence_id: str,
        iteration: int,
        attempt: int,
    ) -> None:
        activated: list[_ActiveFault] = []
        try:
            with self._records_lock:
                for fault in incident.faults:
                    if fault.target.execution_rank != self.rank:
                        continue
                    request = self._build_request(
                        incident,
                        fault,
                        occurrence_id,
                        iteration,
                        attempt,
                    )
                    record = self._new_record(request, incident)
                    self._records.append(record)
                    if self._local.supports(fault):
                        effect: LocalFaultEffect | _ExternalEffect = self._local.activate(
                            request,
                            record,
                        )
                    else:
                        effect = self._activate_external(request, record)
                    active = _ActiveFault(incident, iteration, effect)
                    activated.append(active)
                    if not active.done:
                        self._active.append(active)
        except BaseException as error:
            cleanup_error: BaseException | None = None
            for active in reversed(activated):
                if not active.done:
                    try:
                        active.rollback(error)
                    except BaseException as caught:
                        if cleanup_error is None:
                            cleanup_error = caught
            self._discard_completed()
            if cleanup_error is not None:
                _add_exception_note(error, f"fault rollback also failed: {cleanup_error}")
            raise

    def _activate_external(
        self,
        request: FaultExecutionRequest,
        record: FaultInjectionRecord,
    ) -> _ExternalEffect:
        executor = self._executor_for(request.fault)
        with record._lock:
            record.executor = executor.name
        self._validate_executor_lifecycle(executor)
        try:
            result = executor.activate(request)
        except Exception as error:
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = str(error)
                record.completed_at_ns = time.monotonic_ns()
            raise
        if not isinstance(result, FaultExecutionResult):
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = "fault executor activate must return FaultExecutionResult"
                record.completed_at_ns = time.monotonic_ns()
            raise TypeError(record.error)
        if result.active and getattr(executor, "one_shot", False):
            deactivation_error: Exception | None = None
            if getattr(executor, "can_deactivate", True):
                try:
                    executor.deactivate(request, result)
                except Exception as caught:
                    deactivation_error = caught
            with record._lock:
                record.verified = result.verified
                record.status = InjectionStatus.FAILED
                record.error = "one-shot fault executor returned an active effect"
                if deactivation_error is not None:
                    record.error += f"; deactivation also failed: {deactivation_error}"
                record.activated_at_ns = time.monotonic_ns()
                record.completed_at_ns = time.monotonic_ns()
            raise ValueError(record.error)
        if result.active and getattr(executor, "can_deactivate", True) is False:
            with record._lock:
                record.verified = result.verified
                record.status = InjectionStatus.FAILED
                record.error = (
                    "fault executor returned an active effect without a deactivation callback"
                )
                record.activated_at_ns = time.monotonic_ns()
                record.completed_at_ns = time.monotonic_ns()
            raise ValueError(record.error)
        try:
            evidence = _json_mapping(result.evidence, "fault executor activation evidence")
        except Exception as error:
            deactivation_error: Exception | None = None
            if result.active:
                try:
                    executor.deactivate(request, result)
                except Exception as caught:
                    deactivation_error = caught
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = str(error)
                if deactivation_error is not None:
                    record.error += f"; deactivation also failed: {deactivation_error}"
                record.completed_at_ns = time.monotonic_ns()
            raise ValueError(record.error) from error
        with record._lock:
            record.verified = result.verified
            record.evidence = evidence
            record.activated_at_ns = time.monotonic_ns()
        if not result.verified:
            deactivation_error: Exception | None = None
            if result.active:
                try:
                    executor.deactivate(request, result)
                except Exception as error:
                    deactivation_error = error
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = "fault executor could not verify activation"
                if deactivation_error is not None:
                    record.error += f"; deactivation also failed: {deactivation_error}"
                record.completed_at_ns = time.monotonic_ns()
            raise RuntimeError(record.error)
        if _requires_active_external_effect(request.lifetime) and not result.active:
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = (
                    "external executor completed inline before its configured "
                    "multi-iteration or until lifetime"
                )
                record.completed_at_ns = time.monotonic_ns()
            raise ValueError(record.error)
        if request.lifetime.matching_calls is not None and result.active:
            try:
                executor.deactivate(request, result)
            except Exception as error:
                with record._lock:
                    record.status = InjectionStatus.FAILED
                    record.error = (
                        "external executor returned an active matching_calls fault "
                        f"and deactivation failed: {error}"
                    )
                    record.completed_at_ns = time.monotonic_ns()
                raise RuntimeError(record.error) from error
            with record._lock:
                record.status = InjectionStatus.FAILED
                record.error = (
                    "external executors must complete matching_calls faults during activation"
                )
                record.completed_at_ns = time.monotonic_ns()
            raise ValueError(
                "external executors must complete matching_calls faults during activation"
            )
        with record._lock:
            record.status = InjectionStatus.ACTIVE if result.active else InjectionStatus.COMPLETED
            if not result.active:
                record.completed_at_ns = time.monotonic_ns()
        return _ExternalEffect(
            request=request,
            record=record,
            executor=executor,
            result=result,
            done=not result.active,
        )

    def _executor_for(self, fault: FaultSpec) -> FaultExecutor:
        executor = next(
            (candidate for candidate in self._executors if candidate.supports(fault)),
            None,
        )
        if executor is None:
            raise UnsupportedFaultError(f"no executor supports {fault.type.value}")
        return executor

    def _build_request(
        self,
        incident: FaultIncident,
        fault: FaultSpec,
        occurrence_id: str,
        iteration: int,
        attempt: int,
    ) -> FaultExecutionRequest:
        return FaultExecutionRequest(
            campaign=self.campaign.name,
            seed=_fault_seed(
                self.campaign.seed,
                incident.incident_id,
                fault.fault_id,
                iteration,
            ),
            occurrence_id=occurrence_id,
            incident_id=incident.incident_id,
            iteration=iteration,
            attempt=attempt,
            temporal_behavior=incident.temporal_behavior,
            lifetime=incident.lifetime,
            fault=fault,
        )

    def _new_record(
        self,
        request: FaultExecutionRequest,
        incident: FaultIncident,
    ) -> FaultInjectionRecord:
        return FaultInjectionRecord(
            injection_id=f"{request.occurrence_id}/{request.fault.fault_id}",
            occurrence_id=request.occurrence_id,
            incident_id=request.incident_id,
            fault_id=request.fault.fault_id,
            iteration=request.iteration,
            attempt=request.attempt,
            temporal_behavior=request.temporal_behavior,
            failure_type=request.fault.type.value,
            expected_kind=request.fault.expected_kind,
            safety=request.fault.safety.value,
            framework=self.framework,
            executor=self._local.name,
            execution_rank=request.fault.target.execution_rank,
            target=request.fault.target.to_dict(),
            parameters=request.fault.parameters,
        )

    def _record_probability_skip(
        self,
        incident: FaultIncident,
        occurrence_id: str,
        iteration: int,
        attempt: int,
    ) -> None:
        with self._records_lock:
            for fault in incident.faults:
                if fault.target.execution_rank != self.rank:
                    continue
                record = FaultInjectionRecord(
                    injection_id=f"{occurrence_id}/{fault.fault_id}",
                    occurrence_id=occurrence_id,
                    incident_id=incident.incident_id,
                    fault_id=fault.fault_id,
                    iteration=iteration,
                    attempt=attempt,
                    temporal_behavior=incident.temporal_behavior,
                    failure_type=fault.type.value,
                    expected_kind=fault.expected_kind,
                    safety=fault.safety.value,
                    framework=self.framework,
                    executor="none",
                    execution_rank=fault.target.execution_rank,
                    target=fault.target.to_dict(),
                    parameters=fault.parameters,
                    status=InjectionStatus.SKIPPED_PROBABILITY,
                    completed_at_ns=time.monotonic_ns(),
                )
                self._records.append(record)

    def _complete_until(self, boundary: str) -> None:
        first_error: BaseException | None = None
        for active in self._active:
            if not active.done and active.incident.lifetime.until == boundary:
                try:
                    active.complete(preserve_replaced_state=boundary == "recovery")
                except BaseException as error:
                    if first_error is None:
                        first_error = error
        self._discard_completed()
        if first_error is not None:
            if not isinstance(first_error, Exception):
                raise first_error
            raise RuntimeError(f"fault {boundary} cleanup failed") from first_error

    def _discard_completed(self) -> None:
        self._active = [active for active in self._active if not active.done]

    def _mark_state_replaced(
        self,
        state_family: str,
        identities: frozenset[int],
    ) -> None:
        with self._lifecycle_lock:
            if state_family == "model":
                surfaces = {"weight", "bias"}
            elif state_family == "optimizer":
                surfaces = {"optimizer_state"}
            elif state_family == "checkpoint":
                surfaces = {"weight", "bias", "optimizer_state"}
            else:
                raise ValueError(f"unsupported state replacement family {state_family!r}")
            for active in self._active:
                if (
                    isinstance(active.effect, LocalFaultEffect)
                    and not active.done
                    and active.effect.record.target.get("surface") in surfaces
                    and (
                        state_family == "checkpoint"
                        or active.effect.replacement_identity in identities
                    )
                ):
                    active.effect.mark_state_replaced()
                    lifetime = active.incident.lifetime
                    if (
                        lifetime.iterations is not None
                        and self._completed_iterations
                        < active.start_iteration + lifetime.iterations - 1
                    ):
                        active.effect.fail(
                            RuntimeError(
                                "bounded fault state was replaced before its "
                                "configured lifetime completed"
                            ),
                            preserve_replaced_state=True,
                        )
            self._discard_completed()

    def _history_faults_for(self, iteration: int) -> tuple[FaultSpec, ...]:
        return tuple(
            fault
            for scheduled_iteration in (iteration, iteration + 1)
            for incident, _attempt in self._selected_incidents_for_iteration(scheduled_iteration)
            for fault in incident.faults
            if fault.type in {FailureType.STALE_STATE, FailureType.DUPLICATE}
        )


def enable_fault_injection(
    target: Any,
    optimizer: Any | None = None,
    *,
    campaign: FaultCampaign,
    completed_iterations: int | None = None,
    state_store: CampaignStateStore | None = None,
    executors: Sequence[FaultExecutor] = (),
    rank: int | None = None,
) -> FaultInjectionSession:
    """Enable an automatically scheduled campaign on initialized training objects."""
    arguments = {
        "campaign": campaign,
        "completed_iterations": completed_iterations,
        "state_store": state_store,
        "executors": executors,
        "rank": rank,
    }
    if not dist.is_available() or not dist.is_initialized() or dist.get_world_size() <= 1:
        return FaultInjectionSession(target, optimizer, **arguments)

    world_size = dist.get_world_size()
    arguments["_defer_activation"] = True
    session: FaultInjectionSession | None = None
    local_error: BaseException | None = None
    campaign_identity: str | None = None
    try:
        _validate_distributed_target_ranks(campaign, world_size)
        campaign_identity = _campaign_identity(campaign)
        session = FaultInjectionSession(target, optimizer, **arguments)
    except BaseException as error:
        local_error = error

    error_summary = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    preparation = {
        "error": error_summary,
        "campaign_identity": campaign_identity,
        "current_iteration": None if session is None else session.current_iteration,
        "journal_attempts_identity": (
            None if session is None else session.journal_attempts_identity
        ),
    }
    gathered_preparations: list[dict[str, Any] | None] = [None] * world_size
    try:
        dist.all_gather_object(gathered_preparations, preparation)
    except BaseException as error:
        cleanup_error: BaseException | None = None
        if session is not None:
            try:
                session.close()
            except BaseException as caught:
                cleanup_error = caught
        if cleanup_error is not None:
            _add_exception_note(
                error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        raise
    failures: list[str] = []
    for prepared_rank, prepared in enumerate(gathered_preparations):
        if not isinstance(prepared, Mapping):
            failures.append(
                f"rank {prepared_rank} returned invalid fault injection preparation state"
            )
        elif prepared["error"] is not None:
            failures.append(f"rank {prepared_rank}: {prepared['error']}")
    if not failures:
        reference = gathered_preparations[0]
        assert isinstance(reference, Mapping)
        for prepared_rank, prepared in enumerate(gathered_preparations[1:], start=1):
            assert isinstance(prepared, Mapping)
            if prepared["campaign_identity"] != reference["campaign_identity"]:
                failures.append(f"rank {prepared_rank} campaign manifest differs from rank 0")
            if prepared["current_iteration"] != reference["current_iteration"]:
                failures.append(
                    f"rank {prepared_rank} current_iteration "
                    f"{prepared['current_iteration']} differs from rank 0 "
                    f"{reference['current_iteration']}"
                )
            if prepared["journal_attempts_identity"] != reference["journal_attempts_identity"]:
                failures.append(
                    f"rank {prepared_rank} campaign journal attempts differ from rank 0"
                )
    if failures:
        cleanup_error: BaseException | None = None
        if session is not None:
            try:
                session.close()
            except BaseException as error:
                cleanup_error = error
        message = "fault injection enablement failed; " + "; ".join(failures)
        enablement_error = RuntimeError(message)
        if cleanup_error is not None:
            _add_exception_note(
                enablement_error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        if local_error is not None and not isinstance(local_error, Exception):
            _add_exception_note(local_error, str(enablement_error))
            if cleanup_error is not None:
                _add_exception_note(
                    local_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise local_error
        raise enablement_error from local_error
    if session is None:
        raise AssertionError("distributed fault injection enablement returned no session")
    journal_error: BaseException | None = None
    try:
        session._commit_journal_binding()
    except BaseException as error:
        journal_error = error
    try:
        journal_failures = _gather_rank_errors(journal_error)
    except BaseException as error:
        cleanup_error = session._cleanup()
        if cleanup_error is not None:
            _add_exception_note(
                error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        raise
    if journal_failures:
        cleanup_error = session._cleanup()
        enablement_error = RuntimeError(
            "fault injection journal binding failed; " + "; ".join(journal_failures)
        )
        if cleanup_error is not None:
            _add_exception_note(
                enablement_error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        if journal_error is not None and not isinstance(journal_error, Exception):
            _add_exception_note(journal_error, str(enablement_error))
            if cleanup_error is not None:
                _add_exception_note(
                    journal_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise journal_error
        raise enablement_error from journal_error
    try:
        session._start()
    except BaseException as error:
        cleanup_error = session._cleanup()
        if cleanup_error is not None:
            _add_exception_note(
                error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        raise
    return session


def _add_exception_note(error: BaseException, note: str) -> None:
    """Attach cleanup context while supporting the Python 3.10 runtime floor."""
    add_note = getattr(error, "add_note", None)
    if callable(add_note):
        add_note(note)
        return
    notes = getattr(error, "__notes__", None)
    if notes is None:
        error.__notes__ = [note]
    else:
        notes.append(note)


def _campaign_identity(campaign: FaultCampaign) -> str:
    return campaign.manifest_identity


def _gather_rank_errors(error: BaseException | None) -> list[str]:
    summary = None if error is None else f"{type(error).__name__}: {error}"
    world_size = dist.get_world_size()
    gathered: list[str | None] = [None] * world_size
    dist.all_gather_object(gathered, summary)
    return [
        f"rank {failed_rank}: {failure}"
        for failed_rank, failure in enumerate(gathered)
        if failure is not None
    ]


def _validate_distributed_target_ranks(campaign: FaultCampaign, world_size: int) -> None:
    invalid = sorted(
        {
            fault.target.execution_rank
            for incident in campaign.incidents
            for fault in incident.faults
            if fault.target.execution_rank >= world_size
        }
    )
    if invalid:
        raise ValueError(
            f"campaign targets global ranks outside world size {world_size}: {invalid}"
        )


def _distributed_rank() -> int:
    return dist.get_rank() if dist.is_available() and dist.is_initialized() else 0


def _distributed_world_size() -> int:
    if not dist.is_available() or not dist.is_initialized():
        return 1
    return dist.get_world_size()


def _json_mapping(value: Mapping[str, Any] | None, label: str) -> dict[str, Any]:
    normalized = {} if value is None else dict(value)
    try:
        frozen = freeze_json_mapping(normalized, label)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be strictly JSON-serializable: {error}") from error
    thawed = thaw_json(frozen)
    if not isinstance(thawed, dict):
        raise AssertionError("JSON mapping thaw did not produce a dictionary")
    return thawed


def _requires_active_external_effect(lifetime: IncidentLifetime) -> bool:
    return lifetime.until is not None or (
        lifetime.iterations is not None and lifetime.iterations > 1
    )


def _probability_selected(
    campaign_seed: int,
    incident_id: str,
    iteration: int,
    probability: float,
) -> bool:
    if probability >= 1.0:
        return True
    if probability <= 0.0:
        return False
    seed = _stable_seed(campaign_seed, incident_id, str(iteration))
    return random.Random(seed).random() < probability


def _fault_seed(
    campaign_seed: int,
    incident_id: str,
    fault_id: str,
    iteration: int,
) -> int:
    return _stable_seed(campaign_seed, incident_id, fault_id, str(iteration))


def _stable_seed(campaign_seed: int, *parts: str) -> int:
    digest = hashlib.blake2b(digest_size=8)
    digest.update(int(campaign_seed).to_bytes(16, "little", signed=True))
    for part in parts:
        digest.update(b"\0")
        digest.update(part.encode("utf-8"))
    return int.from_bytes(digest.digest(), "little")


def _trigger_contains_iteration(incident: FaultIncident, iteration: int) -> bool:
    trigger = incident.trigger
    if trigger.range is not None:
        return trigger.range.matches(iteration)
    position = bisect_left(trigger.at, iteration)
    return position < len(trigger.at) and trigger.at[position] == iteration


def _next_trigger_at_or_after(
    incident: FaultIncident,
    iteration: int,
) -> int | None:
    trigger = incident.trigger
    if trigger.range is None:
        position = bisect_left(trigger.at, iteration)
        return None if position == len(trigger.at) else trigger.at[position]
    if iteration <= trigger.range.start:
        return trigger.range.start
    steps = (iteration - trigger.range.start + trigger.range.every - 1) // trigger.range.every
    candidate = trigger.range.start + steps * trigger.range.every
    return candidate if candidate <= trigger.range.end else None


def _occurrence_id(
    incident: FaultIncident,
    iteration: int,
    attempt: int,
) -> str:
    base = f"{incident.incident_id}@{iteration}"
    return base if attempt == 1 else f"{base}#{attempt}"


def _group_records(
    records: Sequence[FaultInjectionRecord],
) -> dict[str, tuple[FaultInjectionRecord, ...]]:
    grouped: dict[str, list[FaultInjectionRecord]] = {}
    for record in records:
        grouped.setdefault(record.occurrence_id, []).append(record)
    return {
        occurrence_id: tuple(occurrence_records)
        for occurrence_id, occurrence_records in grouped.items()
    }


def _evaluate_occurrence(
    occurrence_id: str,
    records: tuple[FaultInjectionRecord, ...],
    results: tuple[LocalizationResult, ...],
    incident: FaultIncident,
) -> FaultEvaluation:
    expected_ranks = tuple(
        sorted(
            {
                expected_rank
                for fault in incident.faults
                if (expected_rank := _expected_rank(fault)) is not None
            }
        )
    )
    expected_resources = tuple(
        sorted(
            {
                fault.target.resource
                for fault in incident.faults
                if fault.target.resource is not None
            }
        )
    )
    expected_components = {
        component
        for fault in incident.faults
        if (
            component := (
                fault.target.component
                if fault.target.component is not None
                else fault.target.module_path
            )
        )
        is not None
    }
    expected_fault_ids = {fault.fault_id for fault in incident.faults}
    recorded_fault_ids = [record.fault_id for record in records]
    complete_action_set = (
        len(recorded_fault_ids) == len(expected_fault_ids)
        and set(recorded_fault_ids) == expected_fault_ids
    )
    injection_succeeded = (
        complete_action_set
        and bool(records)
        and all(record.injection_succeeded for record in records)
    )
    detected_results = tuple(result for result in results if result.detected)
    localizing_results = tuple(
        result for result in detected_results if result.scope != "peer_group"
    )
    detected = bool(detected_results)
    reported_ranks = tuple(
        sorted({rank for result in localizing_results for rank in result.failed_ranks})
    )
    reported_resources = tuple(
        sorted({resource for result in localizing_results for resource in result.failed_resources})
    )
    expected_rank_resource_pairs = {
        (fault.target.rank, fault.target.resource)
        for fault in incident.faults
        if fault.target.rank is not None and fault.target.resource is not None
    }
    reported_rank_resource_pairs = {
        (rank, resource)
        for result in localizing_results
        for rank in result.failed_ranks
        for resource in result.failed_resources
    }
    ranks_match = set(expected_ranks) == set(reported_ranks)
    resources_match = set(expected_resources) == set(reported_resources)
    rank_resource_pairs_match = (
        not expected_rank_resource_pairs
        or reported_rank_resource_pairs == expected_rank_resource_pairs
    )
    unexpected_ranks = tuple(rank for rank in reported_ranks if rank not in expected_ranks)
    unexpected_resources = tuple(
        resource for resource in reported_resources if resource not in expected_resources
    )
    expected_kinds = {fault.expected_kind for fault in incident.faults}
    reported_kinds = {result.kind for result in localizing_results if result.kind is not None}
    if not reported_kinds and len(expected_kinds) == 1:
        kind_matches: bool | None = None
    else:
        expected_targets_by_kind = {
            kind: _expected_targets_for_kind(incident, kind) for kind in expected_kinds
        }
        reported_targets_by_kind = {
            kind: _reported_targets_for_kind(localizing_results, kind) for kind in reported_kinds
        }
        kind_matches = (
            reported_kinds == expected_kinds
            and reported_targets_by_kind == expected_targets_by_kind
        )
    reported_components = {
        component for result in localizing_results for component in result.components
    }
    if not reported_components:
        component_matches: bool | None = None
    else:
        expected_targets_by_component = {
            component: _expected_targets_for_component(incident, component)
            for component in expected_components
        }
        reported_targets_by_component = {
            component: _reported_targets_for_component(localizing_results, component)
            for component in reported_components
        }
        expected_targets_by_kind_component = {
            (kind, component): _expected_targets_for_kind_component(
                incident,
                kind,
                component,
            )
            for kind in expected_kinds
            for component in expected_components
            if any(
                fault.expected_kind == kind
                and (fault.target.component or fault.target.module_path) == component
                for fault in incident.faults
            )
        }
        reported_targets_by_kind_component = {
            (kind, component): _reported_targets_for_kind_component(
                localizing_results,
                kind,
                component,
            )
            for kind in reported_kinds
            for component in reported_components
            if any(
                result.kind == kind and component in result.components
                for result in localizing_results
            )
        }
        component_matches = (
            bool(expected_components)
            and reported_components == expected_components
            and reported_targets_by_component == expected_targets_by_component
            and (
                not reported_kinds
                or reported_targets_by_kind_component == expected_targets_by_kind_component
            )
        )
    localized = (
        injection_succeeded
        and detected
        and ranks_match
        and resources_match
        and rank_resource_pairs_match
        and kind_matches is not False
        and component_matches is not False
    )
    return FaultEvaluation(
        occurrence_id=occurrence_id,
        injection_succeeded=injection_succeeded,
        detected=detected,
        localized=localized,
        expected_ranks=expected_ranks,
        reported_ranks=reported_ranks,
        unexpected_ranks=unexpected_ranks,
        expected_resources=expected_resources,
        reported_resources=reported_resources,
        unexpected_resources=unexpected_resources,
        kind_matches=kind_matches,
        component_matches=component_matches,
        latency_ms=max(
            (result.latency_ms for result in detected_results if result.latency_ms is not None),
            default=None,
        ),
    )


def _expected_targets_for_kind(
    incident: FaultIncident,
    kind: str,
) -> frozenset[tuple[int | None, str | None]]:
    faults = tuple(fault for fault in incident.faults if fault.expected_kind == kind)
    return _expected_target_associations(faults)


def _expected_rank(fault: FaultSpec) -> int | None:
    if fault.target.rank is not None:
        return fault.target.rank
    if fault.target.resource is not None:
        return None
    return fault.target.execution_rank


def _reported_targets_for_kind(
    results: tuple[LocalizationResult, ...],
    kind: str,
) -> frozenset[tuple[int | None, str | None]]:
    matching = tuple(result for result in results if result.kind == kind)
    return _reported_target_associations(matching)


def _expected_targets_for_component(
    incident: FaultIncident,
    component: str,
) -> frozenset[tuple[int | None, str | None]]:
    faults = tuple(
        fault
        for fault in incident.faults
        if (fault.target.component or fault.target.module_path) == component
    )
    return _expected_target_associations(faults)


def _reported_targets_for_component(
    results: tuple[LocalizationResult, ...],
    component: str,
) -> frozenset[tuple[int | None, str | None]]:
    matching = tuple(result for result in results if component in result.components)
    return _reported_target_associations(matching)


def _expected_targets_for_kind_component(
    incident: FaultIncident,
    kind: str,
    component: str,
) -> frozenset[tuple[int | None, str | None]]:
    faults = tuple(
        fault
        for fault in incident.faults
        if fault.expected_kind == kind
        and (fault.target.component or fault.target.module_path) == component
    )
    return _expected_target_associations(faults)


def _reported_targets_for_kind_component(
    results: tuple[LocalizationResult, ...],
    kind: str,
    component: str,
) -> frozenset[tuple[int | None, str | None]]:
    matching = tuple(
        result for result in results if result.kind == kind and component in result.components
    )
    return _reported_target_associations(matching)


def _expected_target_associations(
    faults: tuple[FaultSpec, ...],
) -> frozenset[tuple[int | None, str | None]]:
    return frozenset((_expected_rank(fault), fault.target.resource) for fault in faults)


def _reported_target_associations(
    results: tuple[LocalizationResult, ...],
) -> frozenset[tuple[int | None, str | None]]:
    associations: set[tuple[int | None, str | None]] = set()
    for result in results:
        ranks: tuple[int | None, ...] = result.failed_ranks or (None,)
        resources: tuple[str | None, ...] = result.failed_resources or (None,)
        associations.update((rank, resource) for rank in ranks for resource in resources)
    return frozenset(associations)


__all__ = [
    "FaultInjectionSession",
    "UnsupportedFaultError",
    "enable_fault_injection",
]
