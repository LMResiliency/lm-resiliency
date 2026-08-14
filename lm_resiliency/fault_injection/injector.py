"""Automatic incident scheduler and framework-neutral evaluation runtime."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch.distributed as dist

from lm_resiliency.fault_injection.config import (
    ClockOrigin,
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultSpec,
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


@dataclass(slots=True)
class _ExternalEffect:
    request: FaultExecutionRequest
    record: FaultInjectionRecord
    executor: FaultExecutor
    result: FaultExecutionResult
    done: bool = False

    def complete(self) -> None:
        if self.done:
            return
        try:
            evidence = self.executor.deactivate(self.request, self.result)
        except Exception as error:
            self.record.status = InjectionStatus.FAILED
            self.record.error = f"fault deactivation failed: {error}"
            self.record.completed_at_ns = time.monotonic_ns()
            self.done = True
            raise
        if evidence:
            merged = dict(self.record.evidence)
            merged.update(evidence)
            self.record.evidence = merged
        self.record.status = (
            InjectionStatus.COMPLETED if self.record.verified else InjectionStatus.CANCELLED
        )
        self.record.completed_at_ns = time.monotonic_ns()
        self.done = True


@dataclass(slots=True)
class _ActiveFault:
    incident: FaultIncident
    start_iteration: int
    effect: LocalFaultEffect | _ExternalEffect

    @property
    def done(self) -> bool:
        return self.effect.done

    def complete(self) -> None:
        self.effect.complete()


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
        self.rank = _distributed_rank() if rank is None else int(rank)
        if self.rank < 0:
            raise ValueError("fault injection rank must be non-negative")
        if campaign.clock.origin is ClockOrigin.CAMPAIGN_START:
            completed = 0
        elif completed_iterations is None:
            completed = self._context.inferred_completed_iterations
        else:
            completed = int(completed_iterations)
        if completed < 0:
            raise ValueError("completed_iterations must be non-negative")
        self._completed_iterations = completed
        self._current_iteration = completed + 1
        self._state_store = state_store or MemoryCampaignStateStore()
        self._journal: CampaignJournal = self._state_store.load(campaign.name)
        self._journal.bind_manifest(campaign.manifest_identity)
        self._state_store.save(self._journal)
        self._executors = tuple(executors)
        self._local = LocalFaultExecutor(self._context, self.rank)
        self._records: list[FaultInjectionRecord] = []
        self._active: list[_ActiveFault] = []
        self._closed = False
        self._started = False
        self._faults = tuple(fault for incident in campaign.incidents for fault in incident.faults)

        try:
            self._validate_capabilities()
            self._local.validate_targets(self._faults)
            self._local.sync_history(self._history_faults_for(self._current_iteration))
            self._preflight_current_iteration()
            self._context.register_step_callback(self._on_step_complete)
            register_automatic_cleanup(self)
            if not _defer_activation:
                self._start()
        except Exception as error:
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
        return tuple(self._records)

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
        self._complete_until("recovery")

    def notify_replacement(self) -> None:
        """End permanent effects configured to last until replacement."""
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
        by_occurrence: dict[str, LocalizationResult] = {}
        for result in normalized:
            if result.occurrence_id in by_occurrence:
                raise ValueError(f"duplicate localization result for {result.occurrence_id!r}")
            by_occurrence[result.occurrence_id] = result
        grouped = _group_records(self._records)
        unknown = sorted(set(by_occurrence) - set(grouped))
        if unknown:
            raise ValueError(f"localization results reference unknown occurrences: {unknown}")

        evaluations = tuple(
            _evaluate_occurrence(occurrence_id, records, by_occurrence.get(occurrence_id))
            for occurrence_id, records in grouped.items()
        )
        return CampaignReport(
            campaign=self.campaign.name,
            manifest=self.campaign.to_dict(),
            framework=self.framework,
            rank=self.rank,
            completed_iterations=self.completed_iterations,
            injections=tuple(self._records),
            localizations=normalized,
            evaluations=evaluations,
            metadata=self.campaign.metadata,
        )

    def close(self) -> None:
        if self._closed:
            return
        first_error = self._cleanup()
        if first_error is not None:
            raise RuntimeError("fault injection cleanup failed") from first_error

    def _start(self) -> None:
        """Arm the current iteration after preparation has succeeded."""
        if self._closed:
            raise RuntimeError("cannot start a closed fault injection session")
        if self._started:
            return
        self._enter_iteration(self._current_iteration)
        self._started = True

    def _cleanup(self) -> Exception | None:
        """Best-effort cleanup shared by failed construction and close()."""
        if self._closed:
            return None
        unregister_automatic_cleanup(self)
        first_error: Exception | None = None
        for active in reversed(self._active):
            if not active.done:
                try:
                    active.complete()
                except Exception as error:
                    if first_error is None:
                        first_error = error
        self._active.clear()
        try:
            self._local.close()
        except Exception as error:
            if first_error is None:
                first_error = error
        try:
            self._context.close()
        except Exception as error:
            if first_error is None:
                first_error = error
        self._closed = True
        return first_error

    def __enter__(self) -> "FaultInjectionSession":
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.close()

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
        requests = self._requests_for_iteration(self._current_iteration)
        self._local.validate_activations(requests)
        for request in requests:
            if self._local.supports(request.fault):
                continue
            executor = self._executor_for(request.fault)
            validate = getattr(executor, "validate", None)
            if callable(validate):
                validate(request)

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
        selected = self._selected_incidents_for_iteration(self._current_iteration)
        faults = tuple(fault for incident, _attempt in selected for fault in incident.faults)
        return bool(faults) and all(fault.safety is SafetyClass.SAFE_IN_PROCESS for fault in faults)

    def _selected_incidents_for_iteration(
        self,
        iteration: int,
    ) -> tuple[tuple[FaultIncident, int], ...]:
        selected: list[tuple[FaultIncident, int]] = []
        for incident in self.campaign.incidents:
            if not incident.trigger.matches(iteration):
                continue
            attempt_count = self._journal.attempt_count(incident.incident_id, iteration)
            if incident.retrigger is RetriggerPolicy.ONCE and attempt_count >= 1:
                continue
            if incident.retrigger is RetriggerPolicy.MAX_OCCURRENCES and attempt_count >= int(
                incident.max_occurrences or 0
            ):
                continue
            if not _probability_selected(
                self.campaign.seed,
                incident.incident_id,
                iteration,
                incident.trigger.probability,
            ):
                continue
            selected.append((incident, attempt_count + 1))
        return tuple(selected)

    def _on_step_complete(self) -> None:
        if self._closed:
            return
        finished = self._current_iteration
        for active in self._active:
            lifetime = active.incident.lifetime
            if (
                not active.done
                and lifetime.iterations is not None
                and finished >= active.start_iteration + lifetime.iterations - 1
            ):
                active.complete()
        self._discard_completed()
        self._completed_iterations = finished
        self._current_iteration = finished + 1
        self._local.sync_history(self._history_faults_for(self._current_iteration))
        self._enter_iteration(self._current_iteration)

    def _enter_iteration(self, iteration: int) -> None:
        active_start = len(self._active)
        try:
            for incident in self.campaign.incidents:
                if not incident.trigger.matches(iteration):
                    continue
                attempt_count = self._journal.attempt_count(
                    incident.incident_id,
                    iteration,
                )
                if incident.retrigger is RetriggerPolicy.ONCE and attempt_count >= 1:
                    continue
                if incident.retrigger is RetriggerPolicy.MAX_OCCURRENCES and attempt_count >= int(
                    incident.max_occurrences or 0
                ):
                    continue
                attempt = self._journal.record_attempt(incident.incident_id, iteration)
                self._state_store.save(self._journal)
                occurrence_id = _occurrence_id(incident, iteration, attempt)
                if not _probability_selected(
                    self.campaign.seed,
                    incident.incident_id,
                    iteration,
                    incident.trigger.probability,
                ):
                    self._record_probability_skip(
                        incident,
                        occurrence_id,
                        iteration,
                        attempt,
                    )
                    continue
                self._activate_incident(
                    incident,
                    occurrence_id,
                    iteration,
                    attempt,
                )
        except Exception as error:
            cleanup_error: Exception | None = None
            for active in reversed(self._active[active_start:]):
                if not active.done:
                    try:
                        active.complete()
                    except Exception as caught:
                        if cleanup_error is None:
                            cleanup_error = caught
            del self._active[active_start:]
            if cleanup_error is not None:
                _add_exception_note(error, f"fault rollback also failed: {cleanup_error}")
            raise

    def _activate_incident(
        self,
        incident: FaultIncident,
        occurrence_id: str,
        iteration: int,
        attempt: int,
    ) -> None:
        activated: list[_ActiveFault] = []
        try:
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
        except Exception as error:
            cleanup_error: Exception | None = None
            for active in reversed(activated):
                if not active.done:
                    try:
                        active.complete()
                    except Exception as caught:
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
        try:
            result = executor.activate(request)
        except Exception as error:
            record.status = InjectionStatus.FAILED
            record.error = str(error)
            record.completed_at_ns = time.monotonic_ns()
            raise
        if not isinstance(result, FaultExecutionResult):
            record.status = InjectionStatus.FAILED
            record.error = "fault executor activate must return FaultExecutionResult"
            record.completed_at_ns = time.monotonic_ns()
            raise TypeError(record.error)
        if result.active and getattr(executor, "can_deactivate", True) is False:
            record.executor = executor.name
            record.verified = result.verified
            record.evidence = dict(result.evidence)
            record.status = InjectionStatus.FAILED
            record.error = (
                "fault executor returned an active effect without a deactivation callback"
            )
            record.activated_at_ns = time.monotonic_ns()
            record.completed_at_ns = time.monotonic_ns()
            raise ValueError(record.error)
        record.executor = executor.name
        record.verified = result.verified
        record.evidence = dict(result.evidence)
        record.activated_at_ns = time.monotonic_ns()
        if not result.verified:
            deactivation_error: Exception | None = None
            if result.active:
                try:
                    executor.deactivate(request, result)
                except Exception as error:
                    deactivation_error = error
            record.status = InjectionStatus.FAILED
            record.error = "fault executor could not verify activation"
            if deactivation_error is not None:
                record.error += f"; deactivation also failed: {deactivation_error}"
            record.completed_at_ns = time.monotonic_ns()
            raise RuntimeError(record.error)
        if request.lifetime.matching_calls is not None and result.active:
            try:
                executor.deactivate(request, result)
            except Exception as error:
                record.status = InjectionStatus.FAILED
                record.error = (
                    "external executor returned an active matching_calls fault "
                    f"and deactivation failed: {error}"
                )
                record.completed_at_ns = time.monotonic_ns()
                raise RuntimeError(record.error) from error
            record.status = InjectionStatus.FAILED
            record.error = (
                "external executors must complete matching_calls faults during activation"
            )
            record.completed_at_ns = time.monotonic_ns()
            raise ValueError(
                "external executors must complete matching_calls faults during activation"
            )
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
        for active in self._active:
            if not active.done and active.incident.lifetime.until == boundary:
                active.complete()
        self._discard_completed()

    def _discard_completed(self) -> None:
        self._active = [active for active in self._active if not active.done]

    def _history_faults_for(self, iteration: int) -> tuple[FaultSpec, ...]:
        return tuple(
            fault
            for incident in self.campaign.incidents
            if incident.trigger.matches(iteration) or incident.trigger.matches(iteration + 1)
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
    local_error: Exception | None = None
    campaign_identity: str | None = None
    try:
        _validate_distributed_target_ranks(campaign, world_size)
        campaign_identity = _campaign_identity(campaign)
        session = FaultInjectionSession(target, optimizer, **arguments)
    except Exception as error:
        local_error = error

    error_summary = None if local_error is None else f"{type(local_error).__name__}: {local_error}"
    preparation = {
        "error": error_summary,
        "campaign_identity": campaign_identity,
        "current_iteration": None if session is None else session.current_iteration,
    }
    gathered_preparations: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered_preparations, preparation)
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
    if failures:
        cleanup_error: Exception | None = None
        if session is not None:
            try:
                session.close()
            except Exception as error:
                cleanup_error = error
        message = "fault injection enablement failed; " + "; ".join(failures)
        enablement_error = RuntimeError(message)
        if cleanup_error is not None:
            _add_exception_note(
                enablement_error,
                f"fault injection cleanup also failed: {cleanup_error}",
            )
        raise enablement_error from local_error
    if session is None:
        raise AssertionError("distributed fault injection enablement returned no session")
    if session._has_safe_current_activation():
        activation_error: Exception | None = None
        try:
            session._start()
        except Exception as error:
            activation_error = error
        activation_summary = (
            None
            if activation_error is None
            else f"{type(activation_error).__name__}: {activation_error}"
        )
        gathered_activations: list[str | None] = [None] * world_size
        dist.all_gather_object(gathered_activations, activation_summary)
        activation_failures = [
            f"rank {failed_rank}: {summary}"
            for failed_rank, summary in enumerate(gathered_activations)
            if summary is not None
        ]
        if activation_failures:
            cleanup_error: Exception | None = None
            try:
                session.close()
            except Exception as error:
                cleanup_error = error
            enablement_error = RuntimeError(
                "fault injection arming failed; " + "; ".join(activation_failures)
            )
            if cleanup_error is not None:
                _add_exception_note(
                    enablement_error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise enablement_error from activation_error
    else:
        try:
            session._start()
        except Exception as error:
            try:
                session.close()
            except Exception as cleanup_error:
                _add_exception_note(
                    error,
                    f"fault injection cleanup also failed: {cleanup_error}",
                )
            raise
    return session


def _add_exception_note(error: Exception, note: str) -> None:
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
    result: LocalizationResult | None,
) -> FaultEvaluation:
    expected_ranks = tuple(
        sorted({rank for record in records if (rank := record.expected_rank) is not None})
    )
    expected_resources = tuple(
        sorted(
            {resource for record in records if (resource := record.expected_resource) is not None}
        )
    )
    expected_components = {
        component for record in records if (component := record.expected_component) is not None
    }
    injection_succeeded = bool(records) and all(record.injection_succeeded for record in records)
    detected = bool(result is not None and result.detected)
    reported_ranks = () if result is None else result.failed_ranks
    reported_resources = () if result is None else result.failed_resources
    ranks_match = set(expected_ranks) == set(reported_ranks)
    resources_match = set(expected_resources) == set(reported_resources)
    unexpected_ranks = tuple(rank for rank in reported_ranks if rank not in expected_ranks)
    unexpected_resources = tuple(
        resource for resource in reported_resources if resource not in expected_resources
    )
    expected_kinds = {record.expected_kind for record in records}
    kind_matches = None if result is None or result.kind is None else result.kind in expected_kinds
    component_matches = (
        None
        if result is None or not result.components
        else expected_components.issubset(result.components)
    )
    localized = (
        injection_succeeded
        and detected
        and ranks_match
        and resources_match
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
        latency_ms=None if result is None else result.latency_ms,
    )


__all__ = [
    "FaultInjectionSession",
    "UnsupportedFaultError",
    "enable_fault_injection",
]
