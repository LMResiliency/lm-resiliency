"""Tests for incident-oriented, automatically scheduled fault campaigns."""

from __future__ import annotations

import inspect
import json
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


class Wrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


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
    with pytest.raises(ValueError, match="optimizer_state faults do not support matching_calls"):
        _incident(
            lifetime=IncidentLifetime(matching_calls=1),
            faults=(
                _corruption(
                    target=_target(surface=FaultSurface.OPTIMIZER_STATE),
                    parameter="weight",
                    state_key="exp_avg",
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


def test_enablement_has_no_trigger_or_framework_argument() -> None:
    signature = inspect.signature(enable_fault_injection)

    assert "framework" not in signature.parameters
    assert "trigger" not in dir(enable_fault_injection)
    assert "campaign" in signature.parameters


def test_distributed_enablement_propagates_remote_rank_failure() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()

    def gather(errors, local_error) -> None:
        errors[:] = [local_error, "LookupError: invalid gradient target"]

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

    def gather(errors, local_error) -> None:
        errors[:] = [local_error, "LookupError: invalid gradient target"]

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

    def gather(errors, local_error) -> None:
        assert local_error is None
        assert events == []
        errors[:] = [None, None]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
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

    assert value["manifest"] == campaign.to_dict()
    assert value["completed_iterations"] == 1
    assert value["injections"][0]["injection_succeeded"]
    assert value["evaluations"][0]["localized"]
    assert value["evaluations"][0]["component_matches"]
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
    session.close()


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
