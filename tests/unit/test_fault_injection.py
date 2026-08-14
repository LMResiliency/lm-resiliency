"""Tests for incident-oriented, automatically scheduled fault campaigns."""

from __future__ import annotations

import copy
import inspect
import json
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency import (
    CallbackFaultExecutor,
    ClockOrigin,
    ClockSpec,
    CorruptionOperation,
    FailureType,
    FaultCampaign,
    FaultExecutionResult,
    FaultIncident,
    FaultInjectionSession,
    FaultMagnitude,
    FaultScope,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
    InjectionStatus,
    IterationRange,
    JsonCampaignStateStore,
    LocalizationResult,
    MemoryCampaignStateStore,
    RetriggerPolicy,
    SafetyClass,
    UnsupportedFaultError,
    enable_fault_injection,
)
from lm_resiliency.fault_injection.state import CampaignJournal


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        with torch.no_grad():
            for index, layer in enumerate(self.layers):
                layer.weight.copy_(torch.linspace(-0.75 + index, 0.75 + index, 16).reshape(4, 4))
                layer.bias.fill_(0.25 + index)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class OutputResolverModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.attention = nn.Module()
        self.attention.output = nn.Linear(4, 4)
        self.lm_head = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.lm_head(value)


class Wrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


class DTensor(torch.Tensor):
    @staticmethod
    def __new__(cls, local: torch.Tensor):
        return torch.Tensor._make_subclass(cls, local, local.requires_grad)

    def to_local(self) -> torch.Tensor:
        return self.as_subclass(torch.Tensor)

    @classmethod
    def from_local(cls, local: torch.Tensor, **metadata):
        value = cls(local)
        value.device_mesh = metadata.get("device_mesh")
        value.placements = metadata.get("placements")
        return value


class FakeDeepSpeedEngine:
    def __init__(self, module: nn.Module, *, global_steps: int = 0) -> None:
        self.module = module
        self.optimizer = torch.optim.SGD(module.parameters(), lr=0.0)
        self.global_steps = global_steps

    def step(self) -> None:
        self.optimizer.step()
        self.global_steps += 1

    def zero_optimization_stage(self) -> int:
        return 2


class AccumulatingDeepSpeedEngine(FakeDeepSpeedEngine):
    def __init__(self, module: nn.Module, *, accumulation_steps: int = 2) -> None:
        super().__init__(module)
        self.accumulation_steps = accumulation_steps
        self.micro_steps = 0

    def step(self) -> None:
        self.micro_steps += 1
        if self.micro_steps % self.accumulation_steps == 0:
            self.optimizer.step()
            self.global_steps += 1


class OptimizerStep:
    pass


class PipelineEngine(FakeDeepSpeedEngine):
    def _exec_optimizer_step(self) -> None:
        self.optimizer.step()
        self.global_steps += 1

    def step(self) -> None:
        raise RuntimeError("PipelineEngine.step() is disabled")

    _INSTRUCTION_MAP = {OptimizerStep: _exec_optimizer_step}


class FakeTorchTitanTrainer:
    def __init__(self, model_parts: list[nn.Module], *, step: int = 0) -> None:
        self.model_parts = model_parts
        self.optimizers = torch.optim.SGD(
            [parameter for model in model_parts for parameter in model.parameters()],
            lr=0.0,
        )
        self.lr_schedulers = object()
        self.parallel_dims = object()
        self.checkpointer = object()
        self.step = step

    def train(self) -> None:
        pass


def _target(
    *,
    surface: FaultSurface = FaultSurface.OUTPUT,
    rank: int | None = 0,
    module_path: str | None = "layers.0",
    component: str | None = None,
    index: int | None = None,
    resource: str | None = None,
) -> FaultTarget:
    return FaultTarget(
        rank=rank,
        surface=surface,
        module_path=module_path,
        component=component,
        index=index,
        resource=resource,
    )


def _corruption(
    *,
    fault_id: str = "fault",
    target: FaultTarget | None = None,
    operation: CorruptionOperation = CorruptionOperation.SIGN_FLIP,
    scope: FaultScope = FaultScope.FULL,
    **parameters,
) -> FaultSpec:
    return FaultSpec(
        fault_id=fault_id,
        type=FailureType.TENSOR_CORRUPTION,
        target=target or _target(),
        parameters={
            "operation": operation.value,
            "scope": scope.value,
            **parameters,
        },
    )


def _incident(
    *,
    incident_id: str = "incident",
    at: tuple[int, ...] = (2,),
    trigger_range: IterationRange | None = None,
    probability: float = 1.0,
    lifetime: IncidentLifetime | None = None,
    faults: tuple[FaultSpec, ...] | None = None,
    retrigger: RetriggerPolicy = RetriggerPolicy.ONCE,
    max_occurrences: int | None = None,
) -> FaultIncident:
    return FaultIncident(
        incident_id=incident_id,
        trigger=IncidentTrigger(
            at=() if trigger_range is not None else at,
            range=trigger_range,
            probability=probability,
        ),
        lifetime=lifetime or IncidentLifetime(matching_calls=1),
        faults=faults or (_corruption(),),
        retrigger=retrigger,
        max_occurrences=max_occurrences,
    )


def _campaign(
    *incidents: FaultIncident,
    name: str = "unit-campaign",
    seed: int = 17,
    origin: ClockOrigin = ClockOrigin.TRAINING_RUN,
) -> FaultCampaign:
    return FaultCampaign(
        name=name,
        seed=seed,
        clock=ClockSpec(origin=origin),
        incidents=incidents or (_incident(),),
        metadata={"suite": "unit"},
    )


def _optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(model.parameters(), lr=0.0)


def _step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    value: torch.Tensor | None = None,
) -> torch.Tensor:
    optimizer.zero_grad()
    output = model(torch.ones(2, 4) if value is None else value)
    output.sum().backward()
    optimizer.step()
    return output.detach()


def _external_fault(
    failure_type: FailureType,
    *,
    fault_id: str | None = None,
    resource: str = "node-0",
) -> FaultSpec:
    parameters: dict[str, object] = {}
    if failure_type is FailureType.TENSOR_CORRUPTION:
        parameters["operation"] = CorruptionOperation.SIGN_FLIP.value
    if failure_type is FailureType.DELAY:
        parameters["delay_ms"] = 10.0
    return FaultSpec(
        fault_id=fault_id or failure_type.value,
        type=failure_type,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.RESOURCE,
            resource=resource,
        ),
        parameters=parameters,
    )


def _recording_executor(
    supported_types: set[FailureType],
    events: list[tuple[str, str]],
    *,
    active: bool = False,
    max_safety: SafetyClass = SafetyClass.CLUSTER_DESTRUCTIVE,
) -> CallbackFaultExecutor:
    def activate(request):
        events.append(("activate", request.fault.fault_id))
        return FaultExecutionResult(
            verified=True,
            active=active,
            token=request.occurrence_id,
            evidence={"executor": "recording"},
        )

    def deactivate(request, _result):
        events.append(("deactivate", request.fault.fault_id))
        return {"deactivated": True}

    return CallbackFaultExecutor(
        name="recording",
        supported_types=supported_types,
        activate=activate,
        deactivate=deactivate,
        max_safety=max_safety,
    )


def test_incident_campaign_json_round_trip(tmp_path) -> None:
    campaign = _campaign(
        _incident(
            incident_id="exact",
            at=(3, 11, 27),
            probability=0.5,
        ),
        _incident(
            incident_id="range",
            trigger_range=IterationRange(start=100, end=200, every=20),
            lifetime=IncidentLifetime(iterations=3),
        ),
    )
    path = tmp_path / "campaign.json"

    campaign.to_json(path)
    restored = FaultCampaign.from_json(path)

    assert restored == campaign
    assert json.loads(path.read_text()) == campaign.to_dict()
    assert "framework" not in campaign.to_dict()
    assert campaign.clock.type.value == "training_iteration"


@pytest.mark.parametrize(
    ("path", "field"),
    [
        ((), "unknown_campaign"),
        (("clock",), "unknown_clock"),
        (("incidents", 0), "unknown_incident"),
        (("incidents", 0, "trigger"), "probablity"),
        (("incidents", 0, "lifetime"), "unknown_lifetime"),
        (("incidents", 0, "faults", 0), "unknown_fault"),
        (("incidents", 0, "faults", 0, "target"), "unknown_target"),
    ],
)
def test_campaign_parser_rejects_unknown_fields(
    path: tuple[object, ...],
    field: str,
) -> None:
    value = copy.deepcopy(_campaign().to_dict())
    current: object = value
    for part in path:
        current = current[part]  # type: ignore[index]
    assert isinstance(current, dict)
    current[field] = 0

    with pytest.raises(ValueError, match="unknown fields"):
        FaultCampaign.from_dict(value)


def test_campaign_parser_rejects_unknown_range_fields() -> None:
    value = _campaign(_incident(trigger_range=IterationRange(start=2, end=4))).to_dict()
    value["incidents"][0]["trigger"]["range"]["evry"] = 1

    with pytest.raises(ValueError, match="unknown fields"):
        FaultCampaign.from_dict(value)


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("incidents", 0, "trigger", "at", 0), True),
        (("incidents", 0, "trigger", "at", 0), 2.5),
        (("incidents", 0, "trigger", "range", "start"), 2.5),
        (("incidents", 0, "lifetime", "iterations"), True),
        (("incidents", 0, "faults", 0, "target", "rank"), 0.5),
        (("seed",), True),
    ],
)
def test_campaign_parser_rejects_non_integer_integer_fields(
    path: tuple[object, ...],
    value: object,
) -> None:
    trigger_range = IterationRange(start=2, end=4) if "range" in path else None
    campaign = _campaign(
        _incident(
            at=(2,) if trigger_range is None else (),
            trigger_range=trigger_range,
            lifetime=IncidentLifetime(iterations=1),
        )
    ).to_dict()
    current: object = campaign
    for part in path[:-1]:
        current = current[part]  # type: ignore[index]
    current[path[-1]] = value  # type: ignore[index]

    with pytest.raises(TypeError, match="must be an integer"):
        FaultCampaign.from_dict(campaign)


def test_campaign_mappings_are_deeply_immutable_snapshots() -> None:
    metadata = {"labels": ["nightly"], "nested": {"owner": "resiliency"}}
    parameters = {
        "operation": "sign_flip",
        "nested": {"values": [1, 2]},
    }
    campaign = FaultCampaign(
        name="immutable",
        incidents=(
            _incident(
                faults=(
                    FaultSpec(
                        fault_id="immutable-fault",
                        type=FailureType.TENSOR_CORRUPTION,
                        target=FaultTarget(
                            rank=0,
                            surface=FaultSurface.OUTPUT,
                            module_path="layers.0",
                            metadata=metadata,
                        ),
                        parameters=parameters,
                    ),
                ),
            ),
        ),
        metadata=metadata,
    )
    identity = campaign.manifest_identity
    metadata["labels"].append("mutated")
    parameters["nested"]["values"].append(3)

    with pytest.raises(TypeError):
        campaign.metadata["new"] = "value"  # type: ignore[index]
    with pytest.raises(TypeError):
        campaign.incidents[0].faults[0].parameters["operation"] = "noise"  # type: ignore[index]

    assert campaign.manifest_identity == identity
    assert campaign.to_dict()["metadata"]["labels"] == ["nightly"]
    assert campaign.to_dict()["incidents"][0]["faults"][0]["parameters"]["nested"] == {
        "values": [1, 2]
    }


def test_temporal_behavior_is_derived_from_schedule() -> None:
    transient = _incident(at=(3,))
    sparse_range = _incident(
        at=(),
        trigger_range=IterationRange(start=3, end=4, every=10),
    )
    intermittent = _incident(at=(3, 8))
    probabilistic = _incident(at=(3,), probability=0.5)
    permanent = _incident(
        at=(3,),
        lifetime=IncidentLifetime(until="replacement"),
    )

    assert transient.temporal_behavior == "transient"
    assert sparse_range.temporal_behavior == "transient"
    assert intermittent.temporal_behavior == "intermittent"
    assert probabilistic.temporal_behavior == "intermittent"
    assert permanent.temporal_behavior == "permanent"


def test_invalid_schedule_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        IncidentTrigger()
    with pytest.raises(ValueError, match="exactly one"):
        IncidentTrigger(at=(1,), range=IterationRange(1, 2))
    with pytest.raises(ValueError, match="sorted"):
        IncidentTrigger(at=(2, 1))
    with pytest.raises(ValueError, match="single trigger"):
        _incident(
            at=(2, 3),
            lifetime=IncidentLifetime(until="recovery"),
        )
    with pytest.raises(ValueError, match="positive max_occurrences"):
        _incident(
            retrigger=RetriggerPolicy.MAX_OCCURRENCES,
            max_occurrences=0,
        )
    with pytest.raises(ValueError, match="signed 128-bit"):
        _campaign(seed=2**127)


@pytest.mark.parametrize(
    "surface",
    [FaultSurface.WEIGHT, FaultSurface.BIAS, FaultSurface.OPTIMIZER_STATE],
)
def test_state_tensor_faults_reject_matching_call_lifetimes(
    surface: FaultSurface,
) -> None:
    parameters = {"parameter": surface.value}
    if surface is FaultSurface.OPTIMIZER_STATE:
        parameters = {"parameter": "weight", "state_key": "exp_avg"}

    with pytest.raises(ValueError, match="do not support matching_calls"):
        _incident(
            lifetime=IncidentLifetime(matching_calls=1),
            faults=(
                _corruption(
                    target=_target(surface=surface),
                    **parameters,
                ),
            ),
        )


def test_fault_parameter_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="parameters.operation"):
        FaultSpec(
            fault_id="corruption",
            type=FailureType.TENSOR_CORRUPTION,
            target=_target(),
        )
    with pytest.raises(ValueError, match="delay_ms"):
        FaultSpec(
            fault_id="delay",
            type=FailureType.DELAY,
            target=_target(surface=FaultSurface.COMPUTE),
        )
    with pytest.raises(ValueError, match="parameters.value"):
        _corruption(operation=CorruptionOperation.SET_VALUE)
    with pytest.raises(ValueError, match="factor must change"):
        _corruption(
            operation=CorruptionOperation.SCALE,
            factor=1.0,
        )


@pytest.mark.parametrize(
    ("operation", "parameters", "error_type", "message"),
    [
        (
            CorruptionOperation.SCALE,
            {"factor": "large"},
            TypeError,
            "parameters.factor must be a number",
        ),
        (
            CorruptionOperation.NOISE,
            {"std": "small"},
            TypeError,
            "parameters.std must be a number",
        ),
        (
            CorruptionOperation.NOISE,
            {"std": 0.0},
            ValueError,
            "parameters.std must be greater than zero",
        ),
        (
            CorruptionOperation.SET_VALUE,
            {"value": "invalid"},
            ValueError,
            "parameters.value must be numeric",
        ),
    ],
)
def test_tensor_corruption_rejects_malformed_operation_parameters(
    operation: CorruptionOperation,
    parameters: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _corruption(operation=operation, **parameters)


@pytest.mark.parametrize(
    "failure_type",
    [
        FailureType.STALE_STATE,
        FailureType.DROP,
        FailureType.DUPLICATE,
        FailureType.REORDER,
    ],
)
def test_state_flow_faults_validate_scope_during_manifest_construction(
    failure_type: FailureType,
) -> None:
    with pytest.raises(ValueError, match="not a valid FaultScope"):
        FaultSpec(
            fault_id=failure_type.value,
            type=failure_type,
            target=_target(),
            parameters={"scope": "invalid"},
        )


@pytest.mark.parametrize("delay_ms", ["nan", "inf", True])
def test_delay_rejects_non_numeric_parameters(delay_ms: object) -> None:
    with pytest.raises(TypeError, match="delay_ms must be a number"):
        FaultSpec(
            fault_id="delay",
            type=FailureType.DELAY,
            target=_target(surface=FaultSurface.COMPUTE),
            parameters={"delay_ms": delay_ms},
        )


@pytest.mark.parametrize("delay_ms", [float("nan"), float("inf")])
def test_delay_rejects_non_finite_parameters(delay_ms: float) -> None:
    with pytest.raises(ValueError, match="non-finite|finite parameters.delay_ms"):
        FaultSpec(
            fault_id="delay",
            type=FailureType.DELAY,
            target=_target(surface=FaultSurface.COMPUTE),
            parameters={"delay_ms": delay_ms},
        )


def test_enablement_has_no_trigger_or_framework_argument() -> None:
    signature = inspect.signature(enable_fault_injection)

    assert "framework" not in signature.parameters
    assert "trigger" not in dir(enable_fault_injection)
    assert "campaign" in signature.parameters


def test_distributed_enablement_propagates_remote_rank_failure() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()

    def gather(preparations, local_preparation) -> None:
        remote = dict(local_preparation)
        remote["error"] = "LookupError: invalid gradient target"
        preparations[:] = [local_preparation, remote]

    with (
        patch(
            "lm_resiliency.fault_injection.injector.FaultInjectionSession",
            return_value=session,
        ),
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match="rank 1: LookupError"):
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(),
            )

    session.close.assert_called_once()
    session._start.assert_not_called()


def test_distributed_enablement_preserves_rank_failure_when_cleanup_fails() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()
    session.close.side_effect = RuntimeError("backend cleanup failed")

    def gather(preparations, local_preparation) -> None:
        remote = dict(local_preparation)
        remote["error"] = "LookupError: invalid gradient target"
        preparations[:] = [local_preparation, remote]

    with (
        patch(
            "lm_resiliency.fault_injection.injector.FaultInjectionSession",
            return_value=session,
        ),
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match="rank 1: LookupError") as caught:
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(),
            )

    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    session._start.assert_not_called()


def test_distributed_enablement_arms_iteration_one_after_consensus() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.PROCESS_TERMINATION}, events)
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(
        _incident(
            at=(1,),
            faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
        )
    )

    def gather(preparations, local_preparation) -> None:
        assert events == []
        if isinstance(local_preparation, dict):
            assert local_preparation["error"] is None
            preparations[:] = [local_preparation, dict(local_preparation)]
        else:
            preparations[:] = [local_preparation, local_preparation]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        session = enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
            executors=(executor,),
            rank=0,
        )

    assert events == [("activate", "process_termination")]
    session.close()


def test_distributed_attempt_persistence_precedes_destructive_activation() -> None:
    class FailingAttemptStore(MemoryCampaignStateStore):
        def __init__(self) -> None:
            super().__init__()
            self.saves = 0

        def save(self, journal) -> None:
            self.saves += 1
            if self.saves == 2:
                raise OSError("campaign state disk is unavailable")
            super().save(journal)

    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.PROCESS_TERMINATION}, events)
    model = TinyModel()
    store = FailingAttemptStore()

    def gather(values, local_value) -> None:
        if isinstance(local_value, dict):
            values[:] = [local_value, dict(local_value)]
        else:
            values[:] = [local_value, local_value]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(RuntimeError, match="attempt persistence failed"),
    ):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
                )
            ),
            state_store=store,
            executors=(executor,),
        )

    assert events == []


def test_distributed_safe_arming_failure_is_propagated_before_return() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()
    session.current_iteration = 1
    session._has_safe_current_activation.return_value = True
    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if gathers == 1:
            values[:] = [local_value, dict(local_value)]
        elif gathers == 2:
            values[:] = [local_value, local_value]
        else:
            values[:] = [local_value, "LookupError: optimizer state is unavailable"]

    with (
        patch(
            "lm_resiliency.fault_injection.injector.FaultInjectionSession",
            return_value=session,
        ),
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match="rank 1: LookupError"):
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(_incident(at=(1,))),
            )

    session._start.assert_called_once()
    session.close.assert_called_once()


def test_safe_arming_consensus_decision_includes_remote_rank_faults() -> None:
    model = TinyModel()
    fault = _corruption(target=_target(rank=1))
    with patch.object(FaultInjectionSession, "_start"):
        session = enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(_incident(at=(1,), faults=(fault,))),
            rank=0,
        )

    assert session._has_safe_current_activation()
    assert session.records == ()
    session.close()


def test_distributed_rank_override_must_match_process_rank() -> None:
    model = TinyModel()

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=1),
    ):
        with pytest.raises(ValueError, match="does not match distributed rank 1"):
            FaultInjectionSession(
                model,
                _optimizer(model),
                campaign=_campaign(),
                rank=0,
            )


def test_future_iteration_preflight_failure_is_propagated_to_all_ranks() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter="weight",
        state_key="missing",
    )
    campaign = _campaign(
        _incident(
            at=(2,),
            lifetime=IncidentLifetime(iterations=1),
            faults=(fault,),
        )
    )

    def gather(values, local_value) -> None:
        if isinstance(local_value, dict):
            values[:] = [local_value, dict(local_value)]
        else:
            values[:] = [local_value, None]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        session = enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
        )
        with pytest.raises(RuntimeError, match="iteration preflight failed.*rank 0"):
            _step(model, optimizer)

    assert session._closed


def test_single_rank_future_preflight_failure_does_not_consume_attempt() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    store = MemoryCampaignStateStore()
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter="weight",
        state_key="missing",
    )
    campaign = _campaign(
        _incident(
            at=(2,),
            lifetime=IncidentLifetime(iterations=1),
            faults=(fault,),
        )
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=campaign,
        state_store=store,
    )

    with pytest.raises(LookupError, match="missing"):
        _step(model, optimizer)

    assert store.load(campaign.name).attempts == {}
    session.close()


def test_future_boundary_preparation_failure_is_propagated_to_all_ranks() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)

    def gather(values, local_value) -> None:
        if isinstance(local_value, dict):
            values[:] = [local_value, dict(local_value)]
        else:
            values[:] = [local_value, None]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        session = enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(_incident(at=(2,))),
        )
        session._local.sync_history = MagicMock(
            side_effect=RuntimeError("history observer setup failed")
        )
        with pytest.raises(RuntimeError, match="iteration preparation failed.*rank 0"):
            _step(model, optimizer)

    assert session._closed


@pytest.mark.parametrize(
    ("field", "remote_value", "message"),
    [
        ("campaign_identity", "different-manifest", "campaign manifest differs"),
        ("current_iteration", 2, "current_iteration 2 differs"),
        (
            "journal_attempts_identity",
            "different-journal",
            "campaign journal attempts differ",
        ),
    ],
)
def test_distributed_enablement_requires_consistent_campaign_state(
    field: str,
    remote_value: object,
    message: str,
) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()
    session.current_iteration = 1

    def gather(preparations, local_preparation) -> None:
        remote = dict(local_preparation)
        remote[field] = remote_value
        preparations[:] = [local_preparation, remote]

    with (
        patch(
            "lm_resiliency.fault_injection.injector.FaultInjectionSession",
            return_value=session,
        ),
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match=message):
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(),
            )

    session.close.assert_called_once()
    session._start.assert_not_called()


def test_failed_distributed_preparation_does_not_persist_manifest_binding() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    store = MemoryCampaignStateStore()
    campaign = _campaign(_incident(at=(2,)), name="distributed-transaction")

    def gather(preparations, local_preparation) -> None:
        remote = dict(local_preparation)
        remote["campaign_identity"] = "different-manifest"
        preparations[:] = [local_preparation, remote]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match="campaign manifest differs"):
            enable_fault_injection(
                model,
                optimizer,
                campaign=campaign,
                state_store=store,
            )

    assert store.load(campaign.name).manifest_identity is None


def test_distributed_enablement_rejects_targets_outside_world_size() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = _corruption(target=_target(rank=2))

    def gather(preparations, local_preparation) -> None:
        preparations[:] = [local_preparation, dict(local_preparation)]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
    ):
        with pytest.raises(RuntimeError, match="outside world size 2"):
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(_incident(faults=(fault,))),
            )


def test_exact_iterations_activate_automatically() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(_incident(at=(2, 4)))
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    first = _step(model, optimizer)
    second = _step(model, optimizer)
    third = _step(model, optimizer)
    fourth = _step(model, optimizer)

    assert not torch.equal(first, second)
    assert not torch.equal(third, fourth)
    assert [record.iteration for record in session.records] == [2, 4]
    assert all(record.status is InjectionStatus.COMPLETED for record in session.records)
    assert session.completed_iterations == 4
    assert session.current_iteration == 5
    session.close()


def test_training_iteration_does_not_count_gradient_accumulation_forwards() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    value = torch.ones(2, 4)
    baseline = model(value).detach()
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )

    optimizer.zero_grad()
    model(value).sum().backward()
    model(value).sum().backward()
    assert session.records == ()
    optimizer.step()

    optimizer.zero_grad()
    injected = model(value)
    clean = model(value)
    injected.sum().backward()
    clean.sum().backward()
    optimizer.step()

    assert not torch.equal(injected, baseline)
    torch.testing.assert_close(clean, baseline)
    assert [record.iteration for record in session.records] == [2]
    assert session.completed_iterations == 2
    session.close()


def test_range_schedule_and_probability_zero() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(
        _incident(
            trigger_range=IterationRange(2, 6, every=2),
            probability=0.0,
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    for _ in range(6):
        _step(model, optimizer)

    assert [record.iteration for record in session.records] == [2, 4, 6]
    assert all(record.status is InjectionStatus.SKIPPED_PROBABILITY for record in session.records)
    session.close()


def test_completed_iterations_aligns_absolute_training_clock() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(_incident(at=(6,)))
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=campaign,
        completed_iterations=5,
        rank=0,
    )

    output = _step(model, optimizer)

    assert output.shape == (2, 4)
    assert session.records[0].iteration == 6
    assert session.completed_iterations == 6
    session.close()


@pytest.mark.parametrize("completed_iterations", [True, 1.5, "1"])
def test_completed_iterations_rejects_coercion(completed_iterations: object) -> None:
    model = TinyModel()

    with pytest.raises(TypeError, match="must be an integer"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(_incident(at=(2,))),
            completed_iterations=completed_iterations,  # type: ignore[arg-type]
            rank=0,
        )


@pytest.mark.parametrize(
    "surface",
    [FaultSurface.INPUT, FaultSurface.OUTPUT, FaultSurface.GRADIENT],
)
def test_noop_flow_fault_fails_record_without_aborting_training(
    surface: FaultSurface,
) -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    with torch.no_grad():
        model[0].weight.zero_()
        model[0].bias.zero_()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id=f"drop-{surface.value}",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=surface,
            module_path="0",
        ),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    optimizer.zero_grad()
    output = model(torch.zeros(2, 4))
    loss = output.sum() if surface is not FaultSurface.GRADIENT else (output * 0).sum()
    loss.backward()
    optimizer.step()

    assert session.records[0].status is InjectionStatus.FAILED
    assert not session.records[0].verified
    assert "did not change" in (session.records[0].error or "")
    session.close()


def test_campaign_start_origin_ignores_framework_progress() -> None:
    engine = FakeDeepSpeedEngine(TinyModel(), global_steps=40)
    campaign = _campaign(
        _incident(at=(1,)),
        origin=ClockOrigin.CAMPAIGN_START,
    )
    session = enable_fault_injection(engine, campaign=campaign, rank=0)

    assert session.current_iteration == 1
    assert session.records[0].iteration == 1
    session.close()


def test_permanent_parameter_fault_restores_on_recovery() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="recovery"),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    assert not torch.equal(model.layers[0].weight, baseline)
    assert session.records[0].status is InjectionStatus.ACTIVE

    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_iteration_lifetime_restores_after_window() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(iterations=2),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    assert not torch.equal(model.layers[0].weight, baseline)
    _step(model, optimizer)
    assert not torch.equal(model.layers[0].weight, baseline)
    _step(model, optimizer)

    torch.testing.assert_close(model.layers[0].weight, baseline)
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_bounded_state_fault_retirement_preserves_optimizer_update() -> None:
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(iterations=1),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    operation=CorruptionOperation.SCALE,
                    scope=FaultScope.FULL,
                    factor=2.0,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    optimizer.step()

    torch.testing.assert_close(model.layers[0].weight, baseline - 0.1)
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_bounded_state_fault_retires_noncontiguous_parameter() -> None:
    model = TinyModel()
    parameter = nn.Parameter(model.layers[0].weight.detach().clone().transpose(0, 1))
    assert not parameter.is_contiguous()
    model.layers[0].weight = parameter
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    baseline = parameter.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(iterations=1),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    operation=CorruptionOperation.SCALE,
                    scope=FaultScope.FULL,
                    factor=2.0,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    for current in model.parameters():
        current.grad = torch.ones_like(current)
    optimizer.step()

    torch.testing.assert_close(parameter, baseline - 0.1)
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_state_retirement_is_idempotent_after_external_restore() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()

    def restore_before_injector_callback(*_args) -> None:
        with torch.no_grad():
            model.layers[0].weight.copy_(baseline)

    reset_handle = optimizer.register_step_post_hook(restore_before_injector_callback)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(
                    _corruption(
                        target=_target(surface=FaultSurface.WEIGHT),
                        operation=CorruptionOperation.SCALE,
                        scope=FaultScope.FULL,
                        factor=2.0,
                    ),
                ),
            )
        ),
        rank=0,
    )

    optimizer.step()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()
    reset_handle.remove()


def test_bounded_nonfinite_state_fault_is_rejected_before_mutation() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        operation=CorruptionOperation.SET_VALUE,
        scope=FaultScope.SINGLE,
        value="nan",
    )

    with pytest.raises(ValueError, match="finite retirement delta"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(fault,),
                )
            ),
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)


@pytest.mark.parametrize(
    ("operation", "parameters"),
    [
        (CorruptionOperation.SINGLE_BITFLIP, {}),
        (CorruptionOperation.MULTI_BITFLIP, {}),
        (CorruptionOperation.SET_VALUE, {"value": 0.0}),
        (CorruptionOperation.SET_VALUE, {"value": 1.0}),
        (CorruptionOperation.SET_VALUE, {"value": "nan"}),
        (CorruptionOperation.SET_VALUE, {"value": "inf"}),
        (CorruptionOperation.SCALE, {"factor": 3.0}),
        (CorruptionOperation.NOISE, {"std": 1.0}),
        (CorruptionOperation.SIGN_FLIP, {}),
    ],
)
def test_numerical_corruption_operations(
    operation: CorruptionOperation,
    parameters: dict[str, object],
) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="recovery"),
            faults=(
                _corruption(
                    operation=operation,
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                    magnitude=FaultMagnitude.MEDIUM.value,
                    **parameters,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    assert not torch.equal(model.layers[0].weight, baseline)
    assert session.records[0].verified
    session.notify_recovery()
    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_recovery_cleanup_preserves_checkpoint_loaded_nonfinite_state() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = _corruption(
        operation=CorruptionOperation.SET_VALUE,
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
        value="nan",
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )
    recovered = torch.full_like(model.layers[0].weight, 7.0)

    with torch.no_grad():
        model.layers[0].weight.copy_(recovered)
    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, recovered)
    session.close()


def test_set_nan_rejects_an_existing_nan_as_an_unverified_noop() -> None:
    model = TinyModel()
    with torch.no_grad():
        model.layers[0].weight.view(-1)[8] = torch.nan
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="recovery"),
            faults=(
                _corruption(
                    operation=CorruptionOperation.SET_VALUE,
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                    value="nan",
                ),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="did not change"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=campaign,
            rank=0,
        )


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_bit_flips_support_documented_dtypes(dtype: torch.dtype) -> None:
    model = TinyModel().to(dtype=dtype)
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="recovery"),
            faults=(
                _corruption(
                    operation=CorruptionOperation.SINGLE_BITFLIP,
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    assert not torch.equal(model.layers[0].weight, baseline)
    session.notify_recovery()
    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_input_drop_and_output_reorder() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    drop = FaultSpec(
        fault_id="drop-input",
        type=FailureType.DROP,
        target=_target(surface=FaultSurface.INPUT),
        parameters={"scope": FaultScope.FULL.value},
    )
    reorder = FaultSpec(
        fault_id="reorder-output",
        type=FailureType.REORDER,
        target=_target(surface=FaultSurface.OUTPUT, module_path="layers.1"),
    )
    campaign = _campaign(
        _incident(incident_id="drop", at=(1,), faults=(drop,)),
        _incident(incident_id="reorder", at=(2,), faults=(reorder,)),
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)
    value = torch.stack((torch.ones(4), torch.full((4,), 2.0)))

    dropped = _step(model, optimizer, value)
    clean = model(value).detach()
    reordered = _step(model, optimizer, value)

    assert not torch.equal(dropped, clean)
    torch.testing.assert_close(reordered, torch.flip(clean, dims=(0,)))
    session.close()


@pytest.mark.parametrize("failure_type", [FailureType.STALE_STATE, FailureType.DUPLICATE])
def test_stale_and_duplicate_output_use_prior_observation(
    failure_type: FailureType,
) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="prior-output",
        type=failure_type,
        target=_target(surface=FaultSurface.OUTPUT, module_path="layers.1"),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,), faults=(fault,))),
        rank=0,
    )
    first_input = torch.ones(2, 4)
    second_input = torch.full((2, 4), 3.0)

    first = _step(model, optimizer, first_input)
    second = _step(model, optimizer, second_input)

    torch.testing.assert_close(second, first)
    session.close()


def test_stale_history_collection_is_bounded_to_the_scheduled_window() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="scheduled-stale-output",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OUTPUT, module_path="layers.1"),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(4,), faults=(fault,))),
        rank=0,
    )

    assert not session._local._history
    _step(model, optimizer, torch.ones(2, 4))
    assert not session._local._history
    _step(model, optimizer, torch.full((2, 4), 2.0))
    assert session._local._history
    previous = _step(model, optimizer, torch.full((2, 4), 3.0))
    stale = _step(model, optimizer, torch.full((2, 4), 4.0))

    torch.testing.assert_close(stale, previous)
    assert not session._local._history
    assert not session._local._observer_handles
    session.close()


def test_future_stale_target_is_validated_before_history_window() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="invalid-future-stale-output",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OUTPUT, module_path="missing.layer"),
        parameters={"scope": FaultScope.FULL.value},
    )

    with pytest.raises(LookupError, match="missing.layer"):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(_incident(at=(20,), faults=(fault,))),
            rank=0,
        )

    _step(model, optimizer)


def test_runtime_injection_failure_is_recorded_and_hook_is_removed() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="stale-without-history",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OUTPUT, module_path="layers.1"),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="no prior observed value"):
        model(torch.ones(2, 4))

    assert session.records[0].status is InjectionStatus.FAILED
    assert "no prior observed value" in str(session.records[0].error)
    _step(model, optimizer)
    session.close()


def test_output_fault_preserves_original_forward_exception() -> None:
    class RaisingLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(1))

        def forward(self, _value: torch.Tensor) -> torch.Tensor:
            raise RuntimeError("model forward failed")

    model = TinyModel()
    model.layers[0] = RaisingLayer()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="model forward failed"):
        model(torch.ones(2, 4))

    assert session.records[0].status is InjectionStatus.PENDING
    session.close()
    assert session.records[0].status is InjectionStatus.CANCELLED


def test_gradient_drop_is_applied_once() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="drop-gradient",
        type=FailureType.DROP,
        target=_target(
            surface=FaultSurface.GRADIENT,
            module_path="layers.0",
        ),
        parameters={"parameter": "weight", "scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    optimizer.zero_grad()
    model(torch.ones(2, 4)).sum().backward()

    assert torch.count_nonzero(model.layers[0].weight.grad) == 0
    assert session.records[0].status is InjectionStatus.COMPLETED
    optimizer.step()
    session.close()


def test_deepspeed_zero3_weight_uses_local_partition_storage() -> None:
    model = TinyModel()
    placeholder = nn.Parameter(torch.empty(0))
    partition = torch.linspace(-1.0, 1.0, 16)
    placeholder.ds_tensor = partition
    model.layers[0].weight = placeholder
    engine = FakeDeepSpeedEngine(model)
    engine.zero_optimization_stage = lambda: 3
    baseline = partition.clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        engine,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert placeholder.numel() == 0
    assert not torch.equal(partition, baseline)
    session.notify_recovery()
    torch.testing.assert_close(partition, baseline)
    session.close()


def test_deepspeed_zero3_gradient_hook_uses_original_parameter() -> None:
    model = TinyModel()
    placeholder = nn.Parameter(torch.empty(0))
    placeholder.ds_tensor = torch.ones(16)
    model.layers[0].weight = placeholder
    engine = FakeDeepSpeedEngine(model)
    engine.zero_optimization_stage = lambda: 3
    fault = FaultSpec(
        fault_id="zero3-gradient",
        type=FailureType.DROP,
        target=_target(surface=FaultSurface.GRADIENT),
        parameters={"parameter": "weight", "scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        engine,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    assert session.records[0].status is InjectionStatus.PENDING
    assert (
        session._context.resolve_gradient_parameter(
            fault.target,
            parameter_name="weight",
        )
        is placeholder
    )
    session.close()


def test_deepspeed_zero12_optimizer_state_uses_master_fragment() -> None:
    model = TinyModel()
    parameter = model.layers[0].weight
    fragment = torch.linspace(1.0, 2.0, parameter.numel())

    class Mapping:
        optim_fragment = {"exp_avg": fragment}

        def get_optim_state_keys(self):
            return ("exp_avg",)

        def get_optim_state_fragment(self, key):
            return self.optim_fragment[key]

    parameter._hp_mapping = Mapping()
    engine = FakeDeepSpeedEngine(model)
    baseline = fragment.clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        scope=FaultScope.SINGLE,
        parameter="weight",
        state_key="exp_avg",
    )
    session = enable_fault_injection(
        engine,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert not torch.equal(fragment, baseline)
    session.notify_recovery()
    torch.testing.assert_close(fragment, baseline)
    session.close()


def test_deepspeed_zero3_optimizer_state_uses_local_master_partition() -> None:
    model = TinyModel()
    parameter = nn.Parameter(torch.empty(0))
    parameter.ds_tensor = torch.ones(4)
    parameter.ds_id = 7
    model.layers[0].weight = parameter
    master = nn.Parameter(torch.ones(8))
    state = torch.linspace(1.0, 2.0, 8)
    base_optimizer = SimpleNamespace(state={master: {"exp_avg": state}})

    class Zero3:
        fp32_partitioned_groups_flat = [master]
        optimizer = base_optimizer
        grad_position = {7: (0, 2, 4)}

        def get_param_id(self, _parameter):
            return 7

        def _swappable_optimizer_subgroup(self, _group_idx):
            return False

        def _get_fp32_opt_state_partition(
            self,
            _parameter,
            *,
            release_swap_buffers,
            optim_state_key,
        ):
            assert not release_swap_buffers
            assert optim_state_key == "exp_avg"
            return state.narrow(0, 2, 4), 0

    parameter._z3_optimizer = Zero3()
    engine = FakeDeepSpeedEngine(model)
    engine.optimizer.optimizer = base_optimizer
    baseline = state.clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        scope=FaultScope.SINGLE,
        parameter="weight",
        state_key="exp_avg",
    )
    session = enable_fault_injection(
        engine,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert torch.equal(state[:2], baseline[:2])
    assert not torch.equal(state[2:6], baseline[2:6])
    assert torch.equal(state[6:], baseline[6:])
    session.notify_recovery()
    torch.testing.assert_close(state, baseline)
    session.close()


def test_optimizer_state_corruption_and_restoration() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    _step(model, optimizer)
    state = optimizer.state[model.layers[0].weight]["exp_avg"]
    baseline = state.detach().clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        scope=FaultScope.SINGLE,
        parameter="weight",
        state_key="exp_avg",
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(2,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        completed_iterations=1,
        rank=0,
    )

    assert not torch.equal(state, baseline)
    session.notify_recovery()
    torch.testing.assert_close(state, baseline)
    session.close()


def test_bounded_state_fault_preserves_checkpoint_loaded_replacement() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    replacement = {key: torch.full_like(value, 7.0) for key, value in model.state_dict().items()}

    def load_checkpoint(*_args, **_kwargs) -> None:
        model.load_state_dict(replacement)

    handle = optimizer.register_step_post_hook(load_checkpoint)
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    _step(model, optimizer)

    torch.testing.assert_close(
        model.layers[0].weight,
        torch.full_like(model.layers[0].weight, 7.0),
    )
    handle.remove()
    session.close()


def test_partial_model_state_load_does_not_preserve_omitted_fault_target() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    replacement = torch.full_like(model.layers[1].weight, 7.0)
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    model.load_state_dict({"layers.1.weight": replacement}, strict=False)
    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    torch.testing.assert_close(model.layers[1].weight, replacement)
    session.close()


def test_optimizer_state_load_does_not_preserve_active_model_fault() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    optimizer.load_state_dict(optimizer.state_dict())
    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_model_state_load_does_not_preserve_active_optimizer_fault() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    _step(model, optimizer)
    state = optimizer.state[model.layers[0].weight]["exp_avg"]
    baseline = state.detach().clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        scope=FaultScope.SINGLE,
        parameter="weight",
        state_key="exp_avg",
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(2,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        completed_iterations=1,
        rank=0,
    )

    model.load_state_dict(model.state_dict())
    session.notify_recovery()

    torch.testing.assert_close(state, baseline)
    session.close()


def test_sparse_state_fault_does_not_use_full_tensor_transform() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )

    with patch(
        "lm_resiliency.fault_injection.local._transform_tensor",
        side_effect=AssertionError("state path must not clone the full tensor"),
    ):
        session = enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(fault,),
                )
            ),
            rank=0,
        )
        session.notify_recovery()
        session.close()


def test_immediate_optimizer_state_fault_is_preflighted() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    fault = _corruption(
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter="weight",
        state_key="exp_avg",
    )

    with pytest.raises(LookupError, match="optimizer state tensor"):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(fault,),
                )
            ),
            rank=0,
        )


def test_dtensor_state_fault_mutates_and_retires_local_shard() -> None:
    model = TinyModel()
    parameter = nn.Parameter(DTensor(model.layers[0].weight.detach().clone()))
    parameter.device_mesh = object()
    parameter.placements = ("shard",)
    model.layers[0].weight = parameter
    optimizer = _optimizer(model)
    baseline = parameter.to_local().detach().clone()
    fault = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert not torch.equal(parameter.to_local(), baseline)
    session.close()
    torch.testing.assert_close(parameter.to_local(), baseline)


def test_delay_fault_uses_module_hook_without_manual_trigger() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="compute-delay",
        type=FailureType.DELAY,
        target=_target(surface=FaultSurface.COMPUTE),
        parameters={"delay_ms": 25.0},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    with patch("lm_resiliency.fault_injection.local.time.sleep") as sleep:
        _step(model, optimizer)

    sleep.assert_called_once_with(0.025)
    assert session.records[0].expected_kind == "straggler"
    session.close()


def test_logical_transformer_block_resolution() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[1].weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="transformer_block",
            index=1,
        ),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert not torch.equal(model.layers[1].weight, baseline)
    session.close()
    torch.testing.assert_close(model.layers[1].weight, baseline)


def test_logical_embedding_prefers_token_embeddings() -> None:
    class EmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.position_embeddings = nn.Embedding(8, 4)
            self.token_embeddings = nn.Embedding(8, 4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.token_embeddings(value).sum(dim=1)

    model = EmbeddingModel()
    position = model.position_embeddings.weight.detach().clone()
    tokens = model.token_embeddings.weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="embedding",
        ),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    torch.testing.assert_close(model.position_embeddings.weight, position)
    assert not torch.equal(model.token_embeddings.weight, tokens)
    session.close()


def test_logical_embedding_rejects_ambiguous_generic_matches() -> None:
    class EmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_a = nn.Embedding(8, 4)
            self.embed_b = nn.Embedding(8, 4)

    model = EmbeddingModel()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="embedding",
        ),
    )

    with pytest.raises(ValueError, match="logical embedding target is ambiguous"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(fault,),
                )
            ),
            rank=0,
        )


@pytest.mark.parametrize("component", ["output", "lm_head"])
def test_logical_output_prefers_terminal_head_over_nested_output(
    component: str,
) -> None:
    model = OutputResolverModel()
    attention = model.attention.output.weight.detach().clone()
    head = model.lm_head.weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component=component,
        ),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    torch.testing.assert_close(model.attention.output.weight, attention)
    assert not torch.equal(model.lm_head.weight, head)
    session.close()


def test_explicit_module_path_precedes_logical_fallback_across_wrappers() -> None:
    model = TinyModel()
    wrapped = Wrapper(model)
    optimizer = _optimizer(wrapped)
    first = model.layers[0].weight.detach().clone()
    second = model.layers[1].weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path="layers.1",
            component="transformer_block",
            index=0,
        ),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        wrapped,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    torch.testing.assert_close(model.layers[0].weight, first)
    assert not torch.equal(model.layers[1].weight, second)
    session.close()
    torch.testing.assert_close(model.layers[1].weight, second)


def test_framework_inference_and_automatic_step_boundaries() -> None:
    pytorch_model = TinyModel()
    pytorch_optimizer = _optimizer(pytorch_model)
    deepspeed = FakeDeepSpeedEngine(TinyModel())
    torchtitan = FakeTorchTitanTrainer([TinyModel()])
    megatron_model = TinyModel()
    megatron_optimizer = _optimizer(megatron_model)
    cases = (
        (
            "pytorch",
            enable_fault_injection(
                pytorch_model,
                pytorch_optimizer,
                campaign=_campaign(_incident(at=(1,))),
                rank=0,
            ),
            lambda: _step(pytorch_model, pytorch_optimizer),
        ),
        (
            "deepspeed",
            enable_fault_injection(
                deepspeed,
                campaign=_campaign(_incident(at=(1,))),
                rank=0,
            ),
            lambda: (
                deepspeed.module(torch.ones(2, 4)).sum().backward(),
                deepspeed.step(),
            ),
        ),
        (
            "torchtitan",
            enable_fault_injection(
                torchtitan,
                campaign=_campaign(_incident(at=(1,))),
                rank=0,
            ),
            lambda: _step(torchtitan.model_parts[0], torchtitan.optimizers),
        ),
        (
            "megatron",
            enable_fault_injection(
                [Wrapper(Wrapper(megatron_model))],
                megatron_optimizer,
                campaign=_campaign(_incident(at=(1,))),
                rank=0,
            ),
            lambda: _step(megatron_model, megatron_optimizer),
        ),
    )

    for expected_framework, session, run in cases:
        run()
        assert session.framework == expected_framework
        assert session.completed_iterations == 1
        assert session.records[0].verified
        session.close()


def test_deepspeed_pipeline_uses_optimizer_instruction_map() -> None:
    engine = PipelineEngine(TinyModel())
    original_map = engine._INSTRUCTION_MAP
    session = enable_fault_injection(
        engine,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )

    assert engine._INSTRUCTION_MAP is not original_map
    MethodType(engine._INSTRUCTION_MAP[OptimizerStep], engine)()
    assert session.completed_iterations == 1
    assert session.records[0].status is InjectionStatus.PENDING
    output = engine.module(torch.ones(2, 4))
    assert session.records[0].iteration == 2
    output.sum().backward()
    MethodType(engine._INSTRUCTION_MAP[OptimizerStep], engine)()
    assert session.completed_iterations == 2

    session.close()
    assert engine._INSTRUCTION_MAP is original_map


def test_deepspeed_pipeline_composes_with_existing_instruction_wrapper() -> None:
    engine = PipelineEngine(TinyModel())
    calls: list[str] = []

    def scout_instruction(bound_engine) -> None:
        calls.append("scout")
        PipelineEngine._exec_optimizer_step(bound_engine)

    scout_map = {OptimizerStep: scout_instruction}
    engine._INSTRUCTION_MAP = scout_map
    session = enable_fault_injection(
        engine,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )

    MethodType(engine._INSTRUCTION_MAP[OptimizerStep], engine)()

    assert calls == ["scout"]
    assert session.completed_iterations == 1
    session.close()
    assert engine._INSTRUCTION_MAP is scout_map


def test_deepspeed_clock_advances_only_when_global_steps_changes() -> None:
    engine = AccumulatingDeepSpeedEngine(TinyModel(), accumulation_steps=2)
    session = enable_fault_injection(
        engine,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )

    engine.step()
    assert session.completed_iterations == 0
    assert session.records == ()
    engine.step()
    engine.step()
    assert session.completed_iterations == 1
    assert session.records[0].status is InjectionStatus.PENDING
    engine.step()

    assert session.completed_iterations == 2
    assert session.records[0].iteration == 2
    session.close()


def test_pipeline_global_layer_index_uses_layer_number_metadata() -> None:
    class PipelineChunk(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([nn.Linear(4, 4)])
            self.decoder.layers[0].layer_number = 13

    chunk = PipelineChunk()
    optimizer = _optimizer(chunk)
    baseline = chunk.decoder.layers[0].weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="transformer_block",
            index=12,
        ),
        scope=FaultScope.SINGLE,
    )
    session = enable_fault_injection(
        [chunk],
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(fault,),
            )
        ),
        rank=0,
    )

    assert not torch.equal(chunk.decoder.layers[0].weight, baseline)
    session.close()


def test_pipeline_layer_number_requires_an_unambiguous_base() -> None:
    class PipelineModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
            self.layers[0].layer_number = 12
            self.layers[1].layer_number = 13

    model = PipelineModule()
    engine = PipelineEngine(model)
    target = FaultTarget(
        rank=0,
        surface=FaultSurface.WEIGHT,
        component="transformer_block",
        index=12,
    )
    with pytest.raises(LookupError, match="global layer metadata"):
        enable_fault_injection(
            engine,
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(_corruption(target=target),),
                )
            ),
            rank=0,
        )

    explicit = FaultTarget(
        rank=0,
        surface=FaultSurface.WEIGHT,
        component="transformer_block",
        index=12,
        metadata={"layer_number_base": 0},
    )
    session = enable_fault_injection(
        engine,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(_corruption(target=explicit),),
            )
        ),
        rank=0,
    )
    session.close()


def test_global_expert_index_requires_topology_metadata() -> None:
    class Experts(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    model = Experts()
    ambiguous = FaultTarget(
        rank=0,
        surface=FaultSurface.WEIGHT,
        component="expert",
        index=2,
    )
    with pytest.raises(LookupError, match="expert"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(_corruption(target=ambiguous),),
                )
            ),
            rank=0,
        )

    resolved = FaultTarget(
        rank=0,
        surface=FaultSurface.WEIGHT,
        component="expert",
        index=2,
        metadata={"expert_parallel_rank": 1, "num_local_experts": 2},
    )
    baseline = model.experts[0].weight.detach().clone()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(_corruption(target=resolved),),
            )
        ),
        rank=0,
    )

    assert not torch.equal(model.experts[0].weight, baseline)
    session.close()


def test_logical_block_gradient_uses_a_deterministic_nested_parameter() -> None:
    class Block(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.mlp = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))

        def forward(self, value):
            return self.mlp(value)

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([Block()])

        def forward(self, value):
            return self.layers[0](value)

    model = Model()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="block-gradient",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.GRADIENT,
            component="transformer_block",
            index=0,
        ),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )
    model(torch.ones(2, 4)).sum().backward()

    assert torch.count_nonzero(model.layers[0].mlp[0].bias.grad) == 0
    session.close()


def test_pipeline_global_layer_index_rejects_stage_local_suffixes() -> None:
    chunk = TinyModel()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="transformer_block",
            index=0,
        )
    )

    with pytest.raises(LookupError, match="global layer metadata"):
        enable_fault_injection(
            [chunk],
            _optimizer(chunk),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(fault,),
                )
            ),
            rank=0,
        )


def test_once_retrigger_uses_restart_stable_journal() -> None:
    store = MemoryCampaignStateStore()
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.PROCESS_TERMINATION}, events)
    campaign = _campaign(
        _incident(
            at=(1,),
            faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
        )
    )
    first_model = TinyModel()
    first = enable_fault_injection(
        first_model,
        _optimizer(first_model),
        campaign=campaign,
        state_store=store,
        executors=(executor,),
        rank=0,
    )
    first.close()
    second_model = TinyModel()
    second = enable_fault_injection(
        second_model,
        _optimizer(second_model),
        campaign=campaign,
        state_store=store,
        executors=(executor,),
        rank=0,
    )

    assert len(first.records) == 1
    assert second.records == ()
    assert events == [("activate", "process_termination")]
    second.close()


@pytest.mark.parametrize("store_kind", ["memory", "json"])
def test_restart_journal_rejects_changed_manifest_with_same_name(
    store_kind: str,
    tmp_path,
) -> None:
    store = (
        MemoryCampaignStateStore()
        if store_kind == "memory"
        else JsonCampaignStateStore(tmp_path / "campaign-state.json")
    )
    first_campaign = _campaign(_incident(at=(2,)), name="stable-name", seed=1)
    first_model = TinyModel()
    first = enable_fault_injection(
        first_model,
        _optimizer(first_model),
        campaign=first_campaign,
        state_store=store,
        rank=0,
    )
    first.close()
    changed_campaign = _campaign(_incident(at=(2,)), name="stable-name", seed=2)
    second_model = TinyModel()

    with pytest.raises(ValueError, match="manifest identity"):
        enable_fault_injection(
            second_model,
            _optimizer(second_model),
            campaign=changed_campaign,
            state_store=store,
            rank=0,
        )


def test_every_attempt_and_max_occurrences_retrigger() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.IO_ERROR}, events)
    store = MemoryCampaignStateStore()
    every = _campaign(
        _incident(
            at=(1,),
            retrigger=RetriggerPolicy.EVERY_ATTEMPT,
            faults=(_external_fault(FailureType.IO_ERROR),),
        ),
        name="every",
    )

    occurrence_ids = []
    for _ in range(2):
        model = TinyModel()
        session = enable_fault_injection(
            model,
            _optimizer(model),
            campaign=every,
            state_store=store,
            executors=(executor,),
            rank=0,
        )
        occurrence_ids.append(session.records[0].occurrence_id)
        session.close()

    assert occurrence_ids == ["incident@1", "incident@1#2"]

    limited = _campaign(
        _incident(
            at=(1,),
            retrigger=RetriggerPolicy.MAX_OCCURRENCES,
            max_occurrences=2,
            faults=(_external_fault(FailureType.IO_ERROR),),
        ),
        name="limited",
    )
    counts = []
    for _ in range(3):
        model = TinyModel()
        session = enable_fault_injection(
            model,
            _optimizer(model),
            campaign=limited,
            state_store=store,
            executors=(executor,),
            rank=0,
        )
        counts.append(len(session.records))
        session.close()

    assert counts == [1, 1, 0]


def test_json_state_store_is_atomic_and_campaign_scoped(tmp_path) -> None:
    path = tmp_path / "campaign-state.json"
    store = JsonCampaignStateStore(path)
    journal = store.load("campaign-a")
    journal.record_attempt("incident", 10)
    store.save(journal)

    restored = store.load("campaign-a")

    assert restored.attempt_count("incident", 10) == 1
    with pytest.raises(ValueError, match="belongs to"):
        store.load("campaign-b")


@pytest.mark.parametrize(
    "value",
    [
        {"campaign": "campaign", "manifest_identity": "identity"},
        {
            "campaign": "campaign",
            "manifest_identity": "identity",
            "attempts": {},
            "attemps": {},
        },
    ],
)
def test_campaign_journal_requires_exact_schema(value: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="required fields|unknown fields"):
        CampaignJournal.from_dict(value)


@pytest.mark.parametrize("count", [-1, 0])
def test_campaign_journal_rejects_non_positive_attempt_counts(count: int) -> None:
    with pytest.raises(ValueError, match="attempt counts must be positive"):
        CampaignJournal.from_dict(
            {
                "campaign": "campaign",
                "manifest_identity": "identity",
                "attempts": {"incident@1": count},
            }
        )


@pytest.mark.parametrize("count", [True, 1.5, "1"])
def test_campaign_journal_rejects_non_integer_attempt_counts(count: object) -> None:
    with pytest.raises(TypeError, match="attempt counts must be integers"):
        CampaignJournal.from_dict(
            {
                "campaign": "campaign",
                "manifest_identity": "identity",
                "attempts": {"incident@1": count},
            }
        )


@pytest.mark.parametrize("key", ["incident", "@1", "incident@0", "incident@-1"])
def test_campaign_journal_rejects_invalid_attempt_keys(key: str) -> None:
    with pytest.raises(ValueError, match="attempt keys"):
        CampaignJournal.from_dict(
            {
                "campaign": "campaign",
                "manifest_identity": "identity",
                "attempts": {key: 1},
            }
        )


def test_all_canonical_failure_types_can_use_capability_executor() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(set(FailureType), events)
    incidents = tuple(
        _incident(
            incident_id=f"incident-{failure_type.value}",
            at=(1,),
            faults=(_external_fault(failure_type),),
        )
        for failure_type in FailureType
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(*incidents),
        executors=(executor,),
        rank=0,
    )

    assert len(session.records) == len(FailureType)
    assert session.supported_failure_types == frozenset(FailureType)
    assert all(record.verified for record in session.records)
    assert all(record.status is InjectionStatus.COMPLETED for record in session.records)
    session.close()


def test_unsupported_and_unsafe_faults_fail_before_training() -> None:
    model = TinyModel()
    campaign = _campaign(
        _incident(
            at=(1,),
            faults=(_external_fault(FailureType.NETWORK_PARTITION),),
        )
    )
    with pytest.raises(UnsupportedFaultError, match="network_partition"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=campaign,
            rank=0,
        )

    executor = _recording_executor(
        {FailureType.PROCESS_TERMINATION},
        [],
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )
    process_campaign = _campaign(
        _incident(
            at=(1,),
            faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
        )
    )
    with pytest.raises(UnsupportedFaultError, match="process_termination"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=process_campaign,
            executors=(executor,),
            rank=0,
        )


def test_unverified_external_activation_is_rejected_and_deactivated() -> None:
    events: list[str] = []

    def activate(_request):
        events.append("activate")
        return FaultExecutionResult(verified=False, active=True)

    def deactivate(_request, _result):
        events.append("deactivate")
        return None

    executor = CallbackFaultExecutor(
        name="unverified",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
        )
    )

    with pytest.raises(RuntimeError, match="could not verify activation"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=campaign,
            executors=(executor,),
            rank=0,
        )

    assert events == ["activate", "deactivate"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("verified", "false"),
        ("active", 1),
    ],
)
def test_external_result_flags_must_be_booleans(field: str, value: object) -> None:
    arguments = {"verified": True, "active": False}
    arguments[field] = value

    with pytest.raises(TypeError, match=f"fault execution {field} must be a boolean"):
        FaultExecutionResult(**arguments)


def test_active_external_effect_requires_deactivation_callback() -> None:
    with pytest.raises(ValueError, match="must declare one_shot=True"):
        CallbackFaultExecutor(
            name="leaky",
            supported_types={FailureType.PROCESS_TERMINATION},
            activate=lambda _request: FaultExecutionResult(verified=True, active=True),
            max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
        )

    one_shot_executor = CallbackFaultExecutor(
        name="one-shot",
        supported_types={FailureType.EXCEPTION},
        activate=lambda _request: FaultExecutionResult(verified=True, active=False),
        one_shot=True,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    one_shot_model = TinyModel()
    session = enable_fault_injection(
        one_shot_model,
        _optimizer(one_shot_model),
        campaign=_campaign(_incident(at=(1,), faults=(_external_fault(FailureType.EXCEPTION),))),
        executors=(one_shot_executor,),
        rank=0,
    )
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_external_activation_failure_records_selected_executor() -> None:
    def activate(_request):
        raise RuntimeError("activation failed")

    executor = CallbackFaultExecutor(
        name="failing-executor",
        supported_types={FailureType.EXCEPTION},
        activate=activate,
        deactivate=lambda _request, _result: None,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(2,),
                faults=(_external_fault(FailureType.EXCEPTION),),
            )
        ),
        executors=(executor,),
    )

    with pytest.raises(RuntimeError, match="activation failed"):
        _step(model, optimizer)

    assert session.records[0].executor == "failing-executor"
    assert session.records[0].status is InjectionStatus.FAILED
    session.close()


def test_external_executor_evidence_must_be_json_serializable() -> None:
    events: list[str] = []

    def activate(_request):
        events.append("activate")
        return FaultExecutionResult(
            verified=True,
            active=True,
            evidence={"tensor": torch.tensor(1.0)},
        )

    def deactivate(_request, _result):
        events.append("deactivate")
        return None

    executor = CallbackFaultExecutor(
        name="invalid-evidence",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()

    with pytest.raises(ValueError, match="strictly JSON-serializable"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="campaign_end"),
                    faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == ["activate", "deactivate"]


def test_external_executor_evidence_is_a_deep_snapshot() -> None:
    nested_values = [1]
    executor = CallbackFaultExecutor(
        name="snapshot-evidence",
        supported_types={FailureType.EXCEPTION},
        activate=lambda _request: FaultExecutionResult(
            verified=True,
            active=False,
            evidence={"nested": {"values": nested_values}},
        ),
        one_shot=True,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(_external_fault(FailureType.EXCEPTION),),
            )
        ),
        executors=(executor,),
    )
    nested_values.append(2)

    assert session.records[0].evidence == {"nested": {"values": [1]}}
    json.dumps(session.evaluate().to_dict(), allow_nan=False)
    session.close()


def test_external_permanent_fault_deactivates_on_replacement() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.RESOURCE_UNAVAILABLE},
        events,
        active=True,
    )
    model = TinyModel()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="replacement"),
            faults=(_external_fault(FailureType.RESOURCE_UNAVAILABLE),),
        )
    )
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )

    assert session.records[0].status is InjectionStatus.ACTIVE
    session.notify_replacement()

    assert events == [
        ("activate", "resource_unavailable"),
        ("deactivate", "resource_unavailable"),
    ]
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_close_cleans_optimizer_hook_when_external_deactivation_fails() -> None:
    def activate(_request):
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(_request, _result):
        raise RuntimeError("backend cleanup failed")

    executor = CallbackFaultExecutor(
        name="failing-cleanup",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
        )
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="cleanup failed"):
        session.close()

    assert session.records[0].status is InjectionStatus.FAILED
    optimizer.step()
    assert session.completed_iterations == 0
    session.close()


def test_context_manager_preserves_body_error_when_cleanup_fails() -> None:
    def activate(_request):
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(_request, _result):
        raise RuntimeError("backend cleanup failed")

    executor = CallbackFaultExecutor(
        name="failing-cleanup",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="campaign_end"),
                faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(ValueError, match="training failed") as caught:
        with session:
            raise ValueError("training failed")

    assert any("cleanup also failed" in note for note in caught.value.__notes__)


def test_overlapping_local_faults_roll_back_partial_activation() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    target = _target(surface=FaultSurface.WEIGHT)
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(
                _corruption(fault_id="first", target=target, scope=FaultScope.SINGLE),
                _corruption(fault_id="second", target=target, scope=FaultScope.SINGLE),
            ),
        )
    )

    with pytest.raises(RuntimeError, match="already active"):
        enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)
    _step(model, optimizer)


def test_failed_later_incident_rolls_back_earlier_active_incidents() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            incident_id="active-state",
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                ),
            ),
        ),
        _incident(
            incident_id="invalid-target",
            at=(1,),
            faults=(
                _corruption(
                    target=_target(module_path="missing.layer"),
                    scope=FaultScope.SINGLE,
                ),
            ),
        ),
    )

    with pytest.raises(LookupError, match="missing.layer"):
        enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)
    _step(model, optimizer)


def test_correlated_faults_share_one_occurrence_and_evaluation() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.PROCESS_TERMINATION, FailureType.RESOURCE_UNAVAILABLE},
        events,
    )
    faults = (
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="process",
            resource="process-3",
        ),
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="gpu",
            resource="gpu-3",
        ),
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,), faults=faults)),
        executors=(executor,),
        rank=0,
    )

    report = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-3", "process-3"),
                kind="process_failure",
            )
        ]
    )

    assert {record.occurrence_id for record in session.records} == {"incident@1"}
    assert report.evaluations[0].localized
    assert report.evaluations[0].kind_matches
    session.close()


def test_correlated_multi_rank_incident_uses_job_wide_expected_targets() -> None:
    rank_zero = _corruption(
        fault_id="rank-zero",
        target=_target(rank=0),
    )
    rank_one = _corruption(
        fault_id="rank-one",
        target=_target(rank=1),
    )
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(rank_zero, rank_one))),
        rank=0,
    )
    _step(model, optimizer)

    evaluation = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0, 1),
                kind="sdc",
            )
        ]
    ).evaluations[0]

    assert evaluation.injection_succeeded
    assert evaluation.expected_ranks == (0, 1)
    assert evaluation.localized
    session.close()


def test_mixed_kind_correlated_faults_require_per_kind_target_evidence() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.PROCESS_TERMINATION, FailureType.TIMEOUT},
        events,
    )
    faults = (
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="process",
            resource="process-0",
        ),
        _external_fault(
            FailureType.TIMEOUT,
            fault_id="timeout",
            resource="worker-0",
        ),
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,), faults=faults)),
        executors=(executor,),
        rank=0,
    )

    partial = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("process-0", "worker-0"),
                kind="process_failure",
            )
        ]
    ).evaluations[0]
    complete = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("process-0",),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("worker-0",),
                kind="straggler",
            ),
        ]
    ).evaluations[0]

    assert not partial.localized
    assert partial.kind_matches is False
    assert complete.localized
    assert complete.kind_matches
    session.close()


@pytest.mark.parametrize(
    "result",
    [
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            failed_ranks=(0,),
            kind="straggler",
        ),
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            failed_ranks=(0,),
            components=("layers.1",),
        ),
    ],
)
def test_supplied_kind_and_component_mismatches_are_not_localized(
    result: LocalizationResult,
) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    _step(model, optimizer)

    evaluation = session.evaluate([result]).evaluations[0]

    assert not evaluation.localized
    assert evaluation.kind_matches is False or evaluation.component_matches is False
    session.close()


def test_unexpected_localization_targets_are_not_localized() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    _step(model, optimizer)

    evaluation = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0, 1),
                failed_resources=("unexpected-gpu",),
            )
        ]
    ).evaluations[0]

    assert not evaluation.localized
    assert evaluation.unexpected_ranks == (1,)
    assert evaluation.unexpected_resources == ("unexpected-gpu",)
    session.close()


def test_component_evidence_is_an_overclaim_when_target_has_no_component() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.PROCESS_TERMINATION}, events)
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
            )
        ),
        executors=(executor,),
    )

    evaluation = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("node-0",),
                kind="process_failure",
                components=("transformer_block",),
            )
        ]
    ).evaluations[0]

    assert not evaluation.localized
    assert evaluation.component_matches is False
    session.close()


def test_unspecified_target_rank_scores_the_execution_rank() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = _corruption(target=_target(rank=None))
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )
    _step(model, optimizer)

    report = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
            )
        ]
    )

    assert report.evaluations[0].expected_ranks == (0,)
    assert report.evaluations[0].localized
    session.close()


def test_report_json_contains_manifest_ground_truth_and_scoring(tmp_path) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    campaign = _campaign(_incident(at=(1,)))
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)
    _step(model, optimizer)
    report = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                kind="sdc",
                components=("layers.0",),
                latency_ms=4.5,
            )
        ]
    )
    path = tmp_path / "report.json"

    report.to_json(path)
    value = json.loads(path.read_text())

    assert value["manifest_identity"] == campaign.manifest_identity
    assert value["manifest"] == campaign.to_dict()
    assert value["completed_iterations"] == 1
    assert value["injections"][0]["injection_succeeded"]
    assert value["evaluations"][0]["localized"]
    assert value["evaluations"][0]["component_matches"]
    session.close()


def test_campaign_report_snapshots_mutable_injection_records() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )

    report = session.evaluate()
    assert report.injections[0].status is InjectionStatus.PENDING
    assert not report.injections[0].injection_succeeded
    assert not report.evaluations[0].injection_succeeded

    _step(model, optimizer)

    assert session.records[0].injection_succeeded
    assert report.injections[0].status is InjectionStatus.PENDING
    assert not report.injections[0].injection_succeeded
    assert not report.evaluations[0].injection_succeeded
    session.close()


def test_evaluation_rejects_unknown_and_inconsistent_results() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    _step(model, optimizer)

    with pytest.raises(ValueError, match="unknown occurrences"):
        session.evaluate([LocalizationResult(occurrence_id="unknown@1", detected=False)])
    with pytest.raises(ValueError, match="cannot report failed targets"):
        LocalizationResult(
            occurrence_id="incident@1",
            detected=False,
            failed_ranks=(0,),
        )
    with pytest.raises(TypeError, match="detected must be a boolean"):
        LocalizationResult.from_dict(
            {
                "occurrence_id": "incident@1",
                "detected": "false",
            }
        )
    session.close()


@pytest.mark.parametrize("kind", ["", "   "])
def test_localization_rejects_empty_kind(kind: str) -> None:
    with pytest.raises(ValueError, match="kind must be non-empty"):
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            kind=kind,
        )


def test_localization_rejects_non_string_kind() -> None:
    with pytest.raises(TypeError, match="kind must be a string"):
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            kind=Path("sdc"),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("latency", [float("nan"), float("inf"), -1.0])
def test_localization_rejects_invalid_latency(latency: float) -> None:
    with pytest.raises(ValueError, match="finite and non-negative"):
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            latency_ms=latency,
        )


def test_localization_rejects_non_json_metadata() -> None:
    with pytest.raises(TypeError, match="only JSON values"):
        LocalizationResult(
            occurrence_id="incident@1",
            detected=True,
            metadata={"path": Path("report.json")},
        )


@pytest.mark.parametrize("field", ["failed_resources", "components"])
@pytest.mark.parametrize("value", ["node-0", [1], [""]])
def test_localization_rejects_invalid_string_sequences(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match="sequence of strings|contain"):
        LocalizationResult.from_dict(
            {
                "occurrence_id": "incident@1",
                "detected": True,
                field: value,
            }
        )


def test_localization_rejects_coerced_numeric_fields() -> None:
    with pytest.raises(TypeError, match="latency_ms must be a number"):
        LocalizationResult.from_dict(
            {
                "occurrence_id": "incident@1",
                "detected": True,
                "latency_ms": "4.5",
            }
        )
    with pytest.raises(TypeError, match="failed_ranks must contain integers"):
        LocalizationResult.from_dict(
            {
                "occurrence_id": "incident@1",
                "detected": True,
                "failed_ranks": ["0"],
            }
        )


def test_close_restores_active_faults_and_optimizer_hook() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    campaign = _campaign(
        _incident(
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.WEIGHT),
                    scope=FaultScope.SINGLE,
                ),
            ),
        )
    )
    session = enable_fault_injection(model, optimizer, campaign=campaign, rank=0)

    session.close()
    torch.testing.assert_close(model.layers[0].weight, baseline)
    optimizer.step()
    assert session.completed_iterations == 0
