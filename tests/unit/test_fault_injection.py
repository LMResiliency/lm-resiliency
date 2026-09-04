"""Tests for incident-oriented, automatically scheduled fault campaigns."""

from __future__ import annotations

import copy
import inspect
import json
import os
import stat
import threading
from pathlib import Path
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency import (
    SCHEMA_VERSION,
    CallbackFaultExecutor,
    ClockOrigin,
    ClockSpec,
    CorruptionOperation,
    FailureType,
    FaultCampaign,
    FaultExecutionResult,
    FaultIncident,
    FaultInjectionRecord,
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
    SystemFailureType,
    UnsupportedFaultError,
    enable_fault_injection,
)
from lm_resiliency.fault_injection.frameworks import (
    _base_optimizers,
    resolve_training_context,
)
from lm_resiliency.fault_injection.injector import _ActiveFault, _CampaignSchedule
from lm_resiliency.fault_injection.local import (
    LocalFaultEffect,
    LocalFaultExecutor,
    _History,
    _observe_history,
    _write_linear,
)
from lm_resiliency.fault_injection.state import CampaignJournal
from lm_resiliency.integrations._common import notify_checkpoint_tensor_load


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
    schema_version: int = 1,
) -> FaultCampaign:
    return FaultCampaign(
        name=name,
        seed=seed,
        clock=ClockSpec(origin=origin),
        incidents=incidents or (_incident(),),
        metadata={"suite": "unit"},
        schema_version=schema_version,
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
    rank: int | None = 0,
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
            rank=rank,
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
        completes_inline=not active,
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
    assert campaign.schema_version == 1
    assert SCHEMA_VERSION == 2


def test_default_campaign_identity_matches_explicit_legacy_schema() -> None:
    default_campaign = _campaign()
    explicit_legacy = _campaign(schema_version=1)
    journal = CampaignJournal(
        campaign=default_campaign.name,
        manifest_identity=explicit_legacy.manifest_identity,
    )

    journal.bind_manifest(default_campaign.manifest_identity)

    assert default_campaign.to_dict()["schema_version"] == 1


def test_campaign_parser_retains_legacy_schema_version_one() -> None:
    value = _campaign().to_dict()
    value["schema_version"] = 1

    restored = FaultCampaign.from_dict(value)

    assert restored.schema_version == 1
    assert restored.to_dict()["schema_version"] == 1


def test_campaign_parser_treats_missing_schema_version_as_legacy() -> None:
    value = _campaign().to_dict()
    del value["schema_version"]

    assert FaultCampaign.from_dict(value).schema_version == 1


@pytest.mark.parametrize(
    ("system_failure_type", "effect"),
    [
        (SystemFailureType.HOST_MEMORY_EXHAUSTION, FailureType.RESOURCE_EXHAUSTION),
        (SystemFailureType.HOST_RESOURCE_EXHAUSTION, FailureType.RESOURCE_EXHAUSTION),
        (SystemFailureType.CUDA_OUT_OF_MEMORY, FailureType.RESOURCE_EXHAUSTION),
        (SystemFailureType.CUDA_RUNTIME_FAILURE, FailureType.EXCEPTION),
        (SystemFailureType.DURABLE_STORAGE_EXHAUSTION, FailureType.IO_ERROR),
        (SystemFailureType.DURABLE_STORAGE_FAILURE, FailureType.CHECKPOINT_CORRUPTION),
        (SystemFailureType.PCIE_LINK_FAILURE, FailureType.RESOURCE_UNAVAILABLE),
        (SystemFailureType.PCIE_LINK_DEGRADATION, FailureType.DELAY),
        (SystemFailureType.FABRIC_LINK_FAILURE, FailureType.NETWORK_PARTITION),
        (SystemFailureType.FABRIC_CONGESTION, FailureType.DELAY),
        (SystemFailureType.DATA_SAMPLE_CORRUPTION, FailureType.PAYLOAD_CORRUPTION),
        (SystemFailureType.DATA_SHARD_UNAVAILABLE, FailureType.RESOURCE_UNAVAILABLE),
        (SystemFailureType.INPUT_POSITION_DIVERGENCE, FailureType.STALE_STATE),
        (SystemFailureType.SOFTWARE_ENVIRONMENT_DRIFT, FailureType.CONFIG_DRIFT),
        (SystemFailureType.HOST_PERFORMANCE_DEGRADATION, FailureType.DELAY),
        (SystemFailureType.GPU_THROTTLING, FailureType.DELAY),
        (SystemFailureType.TRAINING_RUNTIME_FAILURE, FailureType.EXCEPTION),
        (SystemFailureType.CONTROL_PLANE_FAILURE, FailureType.RESOURCE_UNAVAILABLE),
        (SystemFailureType.TRANSIENT_COMPUTE_CORRUPTION, FailureType.TENSOR_CORRUPTION),
        (SystemFailureType.COMMON_MODE_CORRUPTION, FailureType.TENSOR_CORRUPTION),
        (SystemFailureType.SINGLE_OWNER_STATE_CORRUPTION, FailureType.STALE_STATE),
    ],
)
def test_pretraining_system_failure_types_accept_compatible_effects(
    system_failure_type: SystemFailureType,
    effect: FailureType,
) -> None:
    parameters: dict[str, object] = {}
    if effect is FailureType.TENSOR_CORRUPTION:
        parameters["operation"] = CorruptionOperation.SIGN_FLIP.value
    if effect is FailureType.DELAY:
        parameters["delay_ms"] = 1.0

    fault = FaultSpec(
        fault_id=system_failure_type.value,
        type=effect,
        system_failure_type=system_failure_type,
        target=FaultTarget(
            surface=FaultSurface.RESOURCE,
            resource="system-under-test",
        ),
        parameters=parameters,
    )

    restored = FaultSpec.from_dict(fault.to_dict())
    assert restored == fault
    assert restored.to_dict()["system_failure_type"] == system_failure_type.value


def test_system_failure_type_rejects_incompatible_observable_effect() -> None:
    with pytest.raises(
        ValueError,
        match="cuda_out_of_memory cannot produce observable effect network_partition",
    ):
        FaultSpec(
            fault_id="invalid-cuda-oom",
            type=FailureType.NETWORK_PARTITION,
            system_failure_type=SystemFailureType.CUDA_OUT_OF_MEMORY,
            target=FaultTarget(
                surface=FaultSurface.RESOURCE,
                resource="gpu-0",
            ),
        )


def test_system_failure_type_requires_schema_version_two() -> None:
    fault = FaultSpec(
        fault_id="host-oom",
        type=FailureType.RESOURCE_EXHAUSTION,
        system_failure_type=SystemFailureType.HOST_MEMORY_EXHAUSTION,
        target=FaultTarget(
            surface=FaultSurface.RESOURCE,
            resource="host-0",
        ),
    )
    value = _campaign(
        _incident(faults=(fault,)),
        schema_version=SCHEMA_VERSION,
    ).to_dict()
    value["schema_version"] = 1

    with pytest.raises(
        ValueError,
        match="system_failure_type requires campaign schema_version 2",
    ):
        FaultCampaign.from_dict(value)


def test_schema_version_one_rejects_null_system_failure_type_field() -> None:
    value = _campaign().to_dict()
    value["incidents"][0]["faults"][0]["system_failure_type"] = None

    with pytest.raises(
        ValueError,
        match="system_failure_type requires campaign schema_version 2",
    ):
        FaultCampaign.from_dict(value)


def test_unspecified_system_failure_type_preserves_existing_manifest_shape() -> None:
    value = _corruption().to_dict()

    assert "system_failure_type" not in value


def test_system_failure_type_is_preserved_in_ground_truth_records() -> None:
    model = TinyModel()
    events: list[tuple[str, str]] = []
    fault = FaultSpec(
        fault_id="host-oom",
        type=FailureType.RESOURCE_EXHAUSTION,
        system_failure_type=SystemFailureType.HOST_MEMORY_EXHAUSTION,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.RESOURCE,
            resource="host-0",
        ),
    )
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(at=(1,), faults=(fault,)),
            schema_version=SCHEMA_VERSION,
        ),
        executors=(
            _recording_executor(
                {FailureType.RESOURCE_EXHAUSTION},
                events,
            ),
        ),
    )

    record = session.records[0]
    assert record.system_failure_type == SystemFailureType.HOST_MEMORY_EXHAUSTION.value
    assert record.to_dict()["system_failure_type"] == "host_memory_exhaustion"
    session.close()


def test_fault_injection_record_preserves_legacy_positional_constructor() -> None:
    record = FaultInjectionRecord(
        "injection",
        "occurrence",
        "incident",
        "fault",
        1,
        2,
        "transient",
        "delay",
        "straggler",
        "safe",
        "pytorch",
        "executor",
        0,
        {},
        {},
        InjectionStatus.COMPLETED,
        True,
        10,
        20,
        {"duration_ms": 1.0},
        None,
    )

    assert record.status is InjectionStatus.COMPLETED
    assert record.verified
    assert record.activated_at_ns == 10
    assert record.completed_at_ns == 20
    assert record.evidence == {"duration_ms": 1.0}
    assert record.error is None
    assert record.system_failure_type is None


def test_bounded_incident_rejects_overlapping_candidates() -> None:
    with pytest.raises(ValueError, match="must not overlap"):
        _incident(
            at=(1, 2),
            lifetime=IncidentLifetime(iterations=2),
        )

    incident = _incident(
        at=(1, 3),
        lifetime=IncidentLifetime(iterations=2),
    )
    assert incident.trigger.at == (1, 3)


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
                            surface=FaultSurface.RESOURCE,
                            resource="gpu-0",
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


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("campaign", None, "campaign name must be a string"),
        ("incident", True, "incident_id must be a string"),
        ("fault", 7, "fault_id must be a string"),
    ],
)
def test_campaign_parser_rejects_non_string_identifiers(
    field: str,
    value: object,
    message: str,
) -> None:
    payload = _campaign().to_dict()
    if field == "campaign":
        payload["name"] = value
    elif field == "incident":
        payload["incidents"][0]["incident_id"] = value
    else:
        payload["incidents"][0]["faults"][0]["fault_id"] = value

    with pytest.raises(TypeError, match=message):
        FaultCampaign.from_dict(payload)


@pytest.mark.parametrize("rank", [True, 0.5, "0"])
def test_session_rejects_coercible_non_integer_rank_overrides(rank: object) -> None:
    model = TinyModel()

    with pytest.raises(TypeError, match="rank must be an integer"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(),
            rank=rank,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("field", ["module_path", "operation", "path"])
def test_fault_target_rejects_mutable_string_selectors(field: str) -> None:
    target = {
        "surface": "resource",
        "resource": "gpu-0",
        field: ["mutable"],
    }

    with pytest.raises(TypeError, match=f"{field} must be a string"):
        FaultTarget.from_dict(target)


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
    with pytest.raises(ValueError, match="unsupported built-in local parameters.*scoep"):
        _corruption(scoep=FaultScope.FULL.value)
    with pytest.raises(ValueError, match="unsupported built-in local parameters.*factor"):
        _corruption(factor=2.0)


@pytest.mark.parametrize(
    ("surface", "parameters", "error_type", "message"),
    [
        (FaultSurface.WEIGHT, {"parameter": 0}, TypeError, "parameters.parameter must be a string"),
        (
            FaultSurface.OPTIMIZER_STATE,
            {"parameter": "weight", "state_key": True},
            TypeError,
            "parameters.state_key must be a string",
        ),
        (
            FaultSurface.OPTIMIZER_STATE,
            {"parameter": "weight", "state_key": "  "},
            ValueError,
            "parameters.state_key must be non-empty",
        ),
    ],
)
def test_state_selectors_require_non_empty_strings(
    surface: FaultSurface,
    parameters: dict[str, object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        _corruption(target=_target(surface=surface), **parameters)


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


def test_reorder_rejects_ignored_scope_during_manifest_construction() -> None:
    with pytest.raises(ValueError, match="unsupported built-in local parameters.*scope"):
        FaultSpec(
            fault_id="reorder",
            type=FailureType.REORDER,
            target=_target(),
            parameters={"scope": FaultScope.FULL.value},
        )


@pytest.mark.parametrize("component", [None, "embedding", "output"])
def test_non_indexed_logical_targets_reject_index(component: str | None) -> None:
    with pytest.raises(ValueError, match="index is supported only for layer or expert"):
        FaultTarget(
            rank=0,
            surface=FaultSurface.OUTPUT,
            module_path="layers.0" if component is None else None,
            component=component,
            index=1,
        )


@pytest.mark.parametrize("component", [1, True, [], {}])
def test_fault_target_rejects_non_string_components(component: object) -> None:
    with pytest.raises(TypeError, match="component must be a string"):
        FaultTarget(
            surface=FaultSurface.OUTPUT,
            component=component,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("component", ["", "   "])
def test_fault_target_rejects_empty_components(component: str) -> None:
    with pytest.raises(ValueError, match="component must be non-empty"):
        FaultTarget(
            surface=FaultSurface.OUTPUT,
            component=component,
        )


@pytest.mark.parametrize("resource", [[], {}, 1, True])
def test_fault_target_rejects_non_string_resources(resource: object) -> None:
    with pytest.raises(TypeError, match="resource must be a string"):
        FaultTarget(
            surface=FaultSurface.RESOURCE,
            resource=resource,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("resource", ["", "   "])
def test_fault_target_rejects_empty_resources(resource: str) -> None:
    with pytest.raises(ValueError, match="resource must be non-empty"):
        FaultTarget(
            surface=FaultSurface.RESOURCE,
            resource=resource,
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


def test_delay_rejects_values_above_the_platform_timer_limit() -> None:
    with pytest.raises(ValueError, match="platform timer limit"):
        FaultSpec(
            fault_id="delay",
            type=FailureType.DELAY,
            target=_target(surface=FaultSurface.COMPUTE),
            parameters={"delay_ms": threading.TIMEOUT_MAX * 2000.0},
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


def test_distributed_enablement_interrupt_reaches_preparation_consensus() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    gathered: list[dict[str, object]] = []

    def gather(preparations, local_preparation) -> None:
        gathered.append(local_preparation)
        remote = dict(local_preparation)
        remote["error"] = None
        preparations[:] = [local_preparation, remote]

    with (
        patch(
            "lm_resiliency.fault_injection.injector.FaultInjectionSession",
            side_effect=KeyboardInterrupt("stop preparation"),
        ),
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(KeyboardInterrupt, match="stop preparation") as caught,
    ):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(),
        )

    assert gathered[0]["error"] == "KeyboardInterrupt: stop preparation"
    assert any("rank 0: KeyboardInterrupt" in note for note in caught.value.__notes__)


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

    gathers = 0

    def gather(preparations, local_preparation) -> None:
        nonlocal gathers
        gathers += 1
        if gathers < 4:
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
    assert gathers == 4
    session.close()


def test_journal_consensus_failure_closes_deferred_session() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = MagicMock()
    session.current_iteration = 1
    session.journal_attempts_identity = "journal"
    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if gathers == 1:
            values[:] = [local_value, dict(local_value)]
            return
        raise RuntimeError("journal consensus timed out")

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
        pytest.raises(RuntimeError, match="journal consensus timed out"),
    ):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(),
        )

    session._cleanup.assert_called_once()
    session._start.assert_not_called()


def test_distributed_destructive_activation_failure_raises_without_post_consensus() -> None:
    def activate(_request):
        raise RuntimeError("executor refused activation")

    executor = CallbackFaultExecutor(
        name="failing-destructive",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=lambda _request, _result: None,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()

    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if gathers > 4:
            raise AssertionError("destructive activation must not enter post-activation consensus")
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
        pytest.raises(RuntimeError, match="executor refused activation"),
    ):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="campaign_end"),
                    faults=(
                        _external_fault(
                            FailureType.PROCESS_TERMINATION,
                            rank=0,
                        ),
                    ),
                )
            ),
            executors=(executor,),
        )
    assert gathers == 4


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


def test_distributed_preparation_collective_failure_closes_deferred_session() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    original_close = FaultInjectionSession.close
    closed: list[FaultInjectionSession] = []

    def close(session: FaultInjectionSession) -> None:
        closed.append(session)
        original_close(session)

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch("lm_resiliency.fault_injection.injector.dist.get_rank", return_value=0),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=RuntimeError("preparation timed out"),
        ),
        patch.object(FaultInjectionSession, "close", close),
        pytest.raises(RuntimeError, match="preparation timed out"),
    ):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(_incident(at=(2,))),
        )

    assert len(closed) == 1
    assert closed[0]._closed


def test_external_effect_finalization_is_serialized() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.PROCESS_TERMINATION},
        events,
        active=True,
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="recovery"),
                faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
            )
        ),
        executors=(executor,),
        rank=0,
    )
    effect = session._active[0].effect
    first = threading.Thread(target=effect.complete)
    second = threading.Thread(target=effect.complete)

    first.start()
    second.start()
    first.join(timeout=5)
    second.join(timeout=5)

    assert events.count(("deactivate", "process_termination")) == 1
    assert effect.done
    session.close()


def test_distributed_safe_arming_failure_is_propagated_before_return() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if isinstance(local_value, dict):
            values[:] = [local_value, dict(local_value)]
        elif gathers == 5:
            values[:] = [local_value, "LookupError: optimizer state is unavailable"]
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
    ):
        with pytest.raises(RuntimeError, match="rank 1: LookupError"):
            enable_fault_injection(
                model,
                optimizer,
                campaign=_campaign(
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
                ),
            )

    assert gathers == 5
    torch.testing.assert_close(model.layers[0].weight, baseline)


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
    assert session._closed


def test_single_rank_future_preflight_failure_restores_active_faults() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()

    def validate(_request) -> None:
        raise RuntimeError("future executor unavailable")

    executor = CallbackFaultExecutor(
        name="future-failure",
        supported_types={FailureType.EXCEPTION},
        activate=lambda _request: FaultExecutionResult(verified=True, active=False),
        validate=validate,
        one_shot=True,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    persistent = _corruption(
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
    )
    future = _external_fault(FailureType.EXCEPTION)
    campaign = _campaign(
        _incident(
            incident_id="persistent",
            at=(1,),
            lifetime=IncidentLifetime(until="campaign_end"),
            faults=(persistent,),
        ),
        _incident(
            incident_id="future",
            at=(2,),
            faults=(future,),
        ),
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )

    assert not torch.equal(model.layers[0].weight, baseline)
    with pytest.raises(RuntimeError, match="future executor unavailable"):
        _step(model, optimizer)

    assert session._closed
    torch.testing.assert_close(model.layers[0].weight, baseline)


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
    store = MemoryCampaignStateStore()
    campaign = _campaign(
        _incident(
            trigger_range=IterationRange(2, 6, every=2),
            probability=0.0,
        )
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=campaign,
        state_store=store,
        rank=0,
    )

    for _ in range(6):
        _step(model, optimizer)

    assert [record.iteration for record in session.records] == [2, 4, 6]
    assert all(record.status is InjectionStatus.SKIPPED_PROBABILITY for record in session.records)
    assert store.load(campaign.name).attempts == {}
    session.close()


def test_probability_skip_does_not_clone_the_attempt_journal() -> None:
    incident = _incident(at=(2,), probability=0.0)
    model = TinyModel()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(incident),
        rank=0,
        _defer_activation=True,
    )

    with patch.object(
        CampaignJournal,
        "from_dict",
        wraps=CampaignJournal.from_dict,
    ) as from_dict:
        staged = session._stage_iteration_attempts(2, ((incident, 1),))

    assert len(staged) == 1
    assert not staged[0].selected
    assert from_dict.call_count == 0
    session.close()


def test_large_range_schedule_remains_lazy() -> None:
    incident = _incident(
        trigger_range=IterationRange(start=1, end=1_000_000_000, every=3),
    )
    schedule = _CampaignSchedule((incident,))

    assert schedule.incidents == (incident,)
    assert schedule.candidates(1) == (incident,)
    assert schedule.candidates(999_999_997) == (incident,)
    assert schedule.candidates(999_999_999) == ()
    assert schedule.candidates(1_000_000_000) == (incident,)


def test_many_incidents_do_not_scan_on_healthy_iterations() -> None:
    incidents = tuple(
        _incident(
            incident_id=f"future-{index}",
            at=(1_000_000 + index,),
        )
        for index in range(1_000)
    )
    schedule = _CampaignSchedule(incidents)

    with patch(
        "lm_resiliency.fault_injection.injector._trigger_contains_iteration",
        side_effect=AssertionError("healthy-path incident scan"),
    ):
        for iteration in range(1, 101):
            assert schedule.candidates(iteration) == ()
            assert schedule.expirations(iteration) == ()


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


def test_invalid_reorder_shape_fails_record_without_aborting_training() -> None:
    model = nn.Sequential(nn.Linear(4, 4))
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="reorder-output",
        type=FailureType.REORDER,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.OUTPUT,
            module_path="0",
        ),
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    optimizer.zero_grad()
    output = model(torch.ones(1, 4))
    output.sum().backward()
    optimizer.step()

    assert output.shape == (1, 4)
    assert session.records[0].status is InjectionStatus.FAILED
    assert "leading dimension of at least two" in (session.records[0].error or "")
    session.close()


def test_scalar_reorder_fails_record_without_aborting_training() -> None:
    class ScalarLayer(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum()

    model = nn.Sequential(ScalarLayer())
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=0.0)
    fault = FaultSpec(
        fault_id="reorder-scalar-output",
        type=FailureType.REORDER,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.OUTPUT,
            module_path="0",
        ),
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    output = model(torch.ones(1, 4))

    assert output.shape == ()
    assert session.records[0].status is InjectionStatus.FAILED
    assert "leading dimension of at least two" in (session.records[0].error or "")
    session.close()


def test_unsupported_gradient_reorder_fails_without_aborting_backward() -> None:
    class ScalarParameterLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum() * self.weight

    model = nn.Sequential(ScalarParameterLayer())
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="reorder-scalar-gradient",
        type=FailureType.REORDER,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.GRADIENT,
            module_path="0",
        ),
        parameters={"parameter": "weight"},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    optimizer.zero_grad()
    model(torch.ones(1, 4)).backward()
    optimizer.step()

    assert session.records[0].status is InjectionStatus.FAILED
    assert "leading dimension of at least two" in (session.records[0].error or "")
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


def test_unscheduled_steps_do_not_rescan_incident_triggers() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(100,))),
        rank=0,
    )

    with patch.object(
        IncidentTrigger,
        "matches",
        side_effect=AssertionError("runtime trigger scan"),
    ):
        _step(model, optimizer)
        assert not session._requires_boundary_consensus(1, 2)

    session.close()


def test_close_waits_for_inflight_step_callback_and_removes_rearmed_faults() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )
    callback_entered = threading.Event()
    release_callback = threading.Event()
    close_done = threading.Event()
    original_sync_history = session._local.sync_history

    def blocking_sync_history(faults) -> None:
        callback_entered.set()
        release_callback.wait(timeout=5)
        original_sync_history(faults)

    session._local.sync_history = blocking_sync_history
    step_thread = threading.Thread(target=_step, args=(model, optimizer))
    step_thread.start()
    assert callback_entered.wait(timeout=5)

    def close() -> None:
        session.close()
        close_done.set()

    close_thread = threading.Thread(target=close)
    close_thread.start()
    assert not close_done.wait(timeout=0.05)
    release_callback.set()
    step_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert close_done.is_set()
    assert session._closed
    assert session._active == []
    assert session._local._observer_handles == {}


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


def test_bounded_float16_state_restores_unchanged_injected_value_exactly() -> None:
    model = TinyModel().half()
    with torch.no_grad():
        model.layers[0].weight.fill_(0.1)
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
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
                        operation=CorruptionOperation.SET_VALUE,
                        scope=FaultScope.SINGLE,
                        value=60_000.0,
                    ),
                ),
            )
        ),
        rank=0,
    )

    assert not torch.equal(model.layers[0].weight, baseline)
    optimizer.step()

    assert torch.equal(model.layers[0].weight, baseline)
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
                    **(
                        {"magnitude": FaultMagnitude.MEDIUM.value}
                        if operation
                        in {
                            CorruptionOperation.SINGLE_BITFLIP,
                            CorruptionOperation.MULTI_BITFLIP,
                            CorruptionOperation.SCALE,
                            CorruptionOperation.NOISE,
                        }
                        else {}
                    ),
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


def test_empty_output_tensor_fails_record_without_aborting_training() -> None:
    class EmptyOutput(nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value[:0]

    model = nn.Sequential(EmptyOutput())
    optimizer = torch.optim.SGD([nn.Parameter(torch.ones(()))], lr=0.0)
    fault = FaultSpec(
        fault_id="drop-empty-output",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.OUTPUT,
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

    output = model(torch.ones(2, 4))

    assert output.shape == (0, 4)
    assert session.records[0].status is InjectionStatus.FAILED
    assert "non-empty" in (session.records[0].error or "")
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

    output = model(torch.ones(2, 4))

    assert output.shape == (2, 4)
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


def test_close_waits_for_in_flight_output_hook() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    entered = threading.Event()
    release = threading.Event()
    close_done = threading.Event()
    forward_errors: list[Exception] = []

    def blocked_transform(value, _request, _history):
        entered.set()
        if not release.wait(timeout=5):
            raise TimeoutError("test did not release the output transform")
        if not isinstance(value, torch.Tensor):
            raise TypeError("expected tensor output")
        return value.neg(), value.numel()

    def run_forward() -> None:
        try:
            model(torch.ones(2, 4))
        except Exception as error:
            forward_errors.append(error)

    def close_session() -> None:
        session.close()
        close_done.set()

    with patch(
        "lm_resiliency.fault_injection.local._transform_tree",
        side_effect=blocked_transform,
    ):
        forward_thread = threading.Thread(target=run_forward)
        forward_thread.start()
        assert entered.wait(timeout=5)
        close_thread = threading.Thread(target=close_session)
        close_thread.start()
        assert not close_done.wait(timeout=0.05)
        release.set()
        forward_thread.join(timeout=5)
        close_thread.join(timeout=5)

    assert not forward_thread.is_alive()
    assert not close_thread.is_alive()
    assert forward_errors == []
    assert close_done.is_set()
    assert session.records[0].status is InjectionStatus.COMPLETED


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


def test_deepspeed_zero3_master_resync_is_not_retired_twice() -> None:
    model = TinyModel()
    placeholder = nn.Parameter(torch.empty(0))
    partition = torch.linspace(-1.0, 1.0, 16)
    master = partition.clone()
    placeholder.ds_tensor = partition
    placeholder._z3_optimizer = object()
    model.layers[0].weight = placeholder
    engine = FakeDeepSpeedEngine(model)
    engine.zero_optimization_stage = lambda: 3

    def resync_step() -> None:
        with torch.no_grad():
            master.sub_(0.125)
            partition.copy_(master)
        engine.global_steps += 1

    engine.step = resync_step
    session = enable_fault_injection(
        engine,
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

    assert not torch.equal(partition, master)
    engine.step()

    torch.testing.assert_close(partition, master)
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


def test_deepspeed_direct_checkpoint_tensor_load_preserves_recovered_state() -> None:
    model = TinyModel()
    engine = FakeDeepSpeedEngine(model)
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
    recovered = model.layers[0].weight.detach().clone()

    notify_checkpoint_tensor_load(SimpleNamespace(_engine=engine))
    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, recovered)
    assert session.records[0].status is InjectionStatus.COMPLETED
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


def test_tied_parameter_alias_load_preserves_replaced_fault_target() -> None:
    class TiedModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.embedding.weight

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.lm_head(self.embedding(value))

    model = TiedModel()
    optimizer = _optimizer(model)
    replacement = torch.full_like(model.embedding.weight, 7.0)
    fault = _corruption(
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.WEIGHT,
            module_path="embedding",
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

    model.load_state_dict({"lm_head.weight": replacement}, strict=False)
    session.notify_recovery()

    torch.testing.assert_close(model.embedding.weight, replacement)
    torch.testing.assert_close(model.lm_head.weight, replacement)
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
    effect = session._active[0].effect

    with patch.object(effect.cancel_event, "wait", return_value=False) as wait:
        _step(model, optimizer)

    wait.assert_called_once_with(0.025)
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


def test_logical_output_bypasses_root_module_with_generic_output_name() -> None:
    class RootOutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.output = nn.Linear(4, 4)
            self.lm_head = nn.Linear(4, 4)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.lm_head(value)

    model = RootOutputModel()
    generic_output = model.output.weight.detach().clone()
    head = model.lm_head.weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="output",
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

    torch.testing.assert_close(model.output.weight, generic_output)
    assert not torch.equal(model.lm_head.weight, head)
    session.close()


def test_explicit_module_path_must_match_logical_target_across_wrappers() -> None:
    model = TinyModel()
    wrapped = Wrapper(model)
    optimizer = _optimizer(wrapped)
    second = model.layers[1].weight.detach().clone()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path="layers.1",
            component="transformer_block",
            index=1,
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

    assert not torch.equal(model.layers[1].weight, second)
    session.close()
    torch.testing.assert_close(model.layers[1].weight, second)


def test_explicit_module_path_rejects_contradictory_logical_target() -> None:
    model = TinyModel()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path="layers.1",
            component="transformer_block",
            index=0,
        ),
        scope=FaultScope.SINGLE,
    )

    with pytest.raises(ValueError, match="does not match or refine logical target"):
        enable_fault_injection(
            Wrapper(model),
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


def test_combined_target_requires_both_selectors_to_resolve() -> None:
    model = TinyModel()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path="layers.0",
            component="missing-component",
        ),
    )

    with pytest.raises(LookupError, match="requires both selectors to resolve"):
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


def test_explicit_path_is_resolved_in_wrapped_user_model_namespace() -> None:
    class UserModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.module = nn.Linear(4, 4, bias=False)
            with torch.no_grad():
                self.module.weight.copy_(torch.eye(4))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.module(value) + 1.0

    user_model = UserModel()
    model = Wrapper(user_model)
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="drop-user-child-output",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.OUTPUT,
            module_path="module",
        ),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    output = model(torch.ones(2, 4))

    torch.testing.assert_close(output, torch.ones_like(output))
    assert session.records[0].injection_succeeded
    session.close()


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


def test_failed_pytorch_optimizer_step_reaches_boundary_consensus() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)

    def fail_step(*_args, **_kwargs):
        raise RuntimeError("optimizer update failed")

    optimizer.step = fail_step  # type: ignore[method-assign]
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )
    stages: list[tuple[str, BaseException | None]] = []

    def gather(error: BaseException | None, stage: str) -> list[str]:
        stages.append((stage, error))
        return ["rank 0: RuntimeError: optimizer update failed"]

    with (
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch.object(session, "_gather_runtime_rank_errors", side_effect=gather),
        pytest.raises(RuntimeError, match="optimizer update failed") as caught,
    ):
        optimizer.step()

    assert stages == [("optimizer step", caught.value)]
    assert session.completed_iterations == 0
    assert session._closed
    assert any("fault injection optimizer step failed" in note for note in caught.value.__notes__)


def test_megatron_clock_ignores_skipped_optimizer_updates() -> None:
    class MegatronOptimizer(torch.optim.SGD):
        update_succeeded = False

        def step(self, closure=None):
            if self.update_succeeded:
                super().step(closure)
            return (self.update_succeeded, None, None)

    model = TinyModel()
    optimizer = MegatronOptimizer(model.parameters(), lr=0.0)
    session = enable_fault_injection(
        [Wrapper(model)],
        optimizer,
        campaign=_campaign(_incident(at=(2,))),
        rank=0,
    )

    optimizer.step()

    assert session.completed_iterations == 0
    assert session.records == ()

    optimizer.update_succeeded = True
    optimizer.step()

    assert session.completed_iterations == 1
    assert session.records[0].status is InjectionStatus.PENDING
    session.close()


def test_megatron_optimizer_state_resolves_chained_master_parameter() -> None:
    model = TinyModel().half()
    target_parameter = model.layers[0].weight
    unrelated_parameter = nn.Parameter(torch.zeros_like(target_parameter, dtype=torch.float32))
    target_main = nn.Parameter(target_parameter.detach().float().clone())
    unrelated_base = torch.optim.Adam([unrelated_parameter])
    target_base = torch.optim.Adam([target_main])
    expected = torch.full_like(target_main, 0.25)
    target_base.state[target_main]["exp_avg"] = expected

    class MixedPrecisionOptimizer:
        def __init__(self, model_parameter, main_parameter, base):
            self.float16_groups = [[model_parameter]]
            self.fp32_from_float16_groups = [[main_parameter]]
            self.optimizer = base

    class ChainedOptimizer:
        def __init__(self):
            self.chained_optimizers = [
                MixedPrecisionOptimizer(
                    nn.Parameter(torch.zeros_like(target_parameter)),
                    unrelated_parameter,
                    unrelated_base,
                ),
                MixedPrecisionOptimizer(target_parameter, target_main, target_base),
            ]

        def step(self):
            return True, None, None

    context = resolve_training_context([Wrapper(model)], ChainedOptimizer())

    resolved = context.resolve_optimizer_state(
        _target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter_name="weight",
        state_key="exp_avg",
    )

    assert resolved is expected


def test_megatron_master_resync_is_not_retired_twice() -> None:
    model = TinyModel().half()
    model_parameter = model.layers[0].weight
    main_parameter = nn.Parameter(model_parameter.detach().float().clone())
    base_optimizer = torch.optim.SGD([main_parameter], lr=0.0)

    class MixedPrecisionOptimizer:
        def __init__(self):
            self.float16_groups = [[model_parameter]]
            self.fp32_from_float16_groups = [[main_parameter]]
            self.optimizer = base_optimizer

        def step(self):
            with torch.no_grad():
                main_parameter.sub_(0.125)
                model_parameter.copy_(main_parameter)
            return True, None, None

    optimizer = MixedPrecisionOptimizer()
    session = enable_fault_injection(
        [Wrapper(model)],
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

    assert not torch.equal(model_parameter, main_parameter.to(model_parameter.dtype))
    optimizer.step()

    torch.testing.assert_close(
        model_parameter,
        main_parameter.to(model_parameter.dtype),
    )
    assert session.records[0].status is InjectionStatus.COMPLETED
    session.close()


@pytest.mark.parametrize(
    "lifetime",
    [IncidentLifetime(iterations=2), IncidentLifetime(until="recovery")],
)
def test_megatron_master_resync_rejects_multi_boundary_state_faults(
    lifetime: IncidentLifetime,
) -> None:
    model = TinyModel().half()
    model_parameter = model.layers[0].weight
    main_parameter = nn.Parameter(model_parameter.detach().float().clone())
    base_optimizer = torch.optim.SGD([main_parameter], lr=0.0)

    class MixedPrecisionOptimizer:
        def __init__(self):
            self.float16_groups = [[model_parameter]]
            self.fp32_from_float16_groups = [[main_parameter]]
            self.optimizer = base_optimizer

        def step(self):
            with torch.no_grad():
                model_parameter.copy_(main_parameter)
            return True, None, None

    baseline = model_parameter.detach().clone()
    with pytest.raises(ValueError, match="optimizer-resynchronized.*iterations=1"):
        enable_fault_injection(
            [Wrapper(model)],
            MixedPrecisionOptimizer(),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=lifetime,
                    faults=(
                        _corruption(
                            target=_target(surface=FaultSurface.WEIGHT),
                            scope=FaultScope.FULL,
                        ),
                    ),
                )
            ),
            rank=0,
        )

    torch.testing.assert_close(model_parameter, baseline)


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


def test_json_state_store_syncs_parent_directory_before_save_returns(tmp_path) -> None:
    path = tmp_path / "campaign-state.json"
    store = JsonCampaignStateStore(path)
    synced_directories: list[bool] = []
    real_fsync = os.fsync

    def track_fsync(descriptor: int) -> None:
        synced_directories.append(stat.S_ISDIR(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    with patch(
        "lm_resiliency.fault_injection.state.os.fsync",
        side_effect=track_fsync,
    ):
        store.save(CampaignJournal("campaign"))

    assert synced_directories == [False, True]


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


@pytest.mark.parametrize(
    "key",
    ["incident", "@1", "incident@0", "incident@-1", "incident@01"],
)
def test_campaign_journal_rejects_invalid_attempt_keys(key: str) -> None:
    with pytest.raises(ValueError, match="attempt keys"):
        CampaignJournal.from_dict(
            {
                "campaign": "campaign",
                "manifest_identity": "identity",
                "attempts": {key: 1},
            }
        )


def test_overlapping_sessions_cannot_claim_the_same_once_occurrence(tmp_path: Path) -> None:
    path = tmp_path / "campaign-state.json"
    campaign = _campaign(_incident(at=(1,)))
    first_model = TinyModel()
    second_model = TinyModel()
    first = FaultInjectionSession(
        first_model,
        _optimizer(first_model),
        campaign=campaign,
        state_store=JsonCampaignStateStore(path),
        rank=0,
        _defer_activation=True,
    )
    second = FaultInjectionSession(
        second_model,
        _optimizer(second_model),
        campaign=campaign,
        state_store=JsonCampaignStateStore(path),
        rank=0,
        _defer_activation=True,
    )
    first._commit_journal_binding()
    second._commit_journal_binding()

    first._start()
    with pytest.raises(RuntimeError, match="changed concurrently"):
        second._start()

    assert len(first.records) == 1
    assert second.records == ()
    first.close()
    second.close()


def test_exact_trigger_lookup_uses_binary_search() -> None:
    incident = _incident(at=tuple(range(1, 100_001)))
    schedule = _CampaignSchedule((incident,))

    with patch(
        "lm_resiliency.fault_injection.injector.bisect_left",
        wraps=__import__("bisect").bisect_left,
    ) as lookup:
        assert schedule.candidates(99_999) == (incident,)
        assert schedule.candidates(100_001) == ()

    assert lookup.call_count == 2


def test_evaluation_snapshots_record_transitions_atomically() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    record = session.records[0]
    transition_started = threading.Event()
    release_transition = threading.Event()
    evaluation_done = threading.Event()
    reports = []

    def transition() -> None:
        with record._lock:
            record.verified = True
            transition_started.set()
            release_transition.wait(timeout=5)
            record.status = InjectionStatus.ACTIVE

    def evaluate() -> None:
        reports.append(session.evaluate())
        evaluation_done.set()

    transition_thread = threading.Thread(target=transition)
    transition_thread.start()
    assert transition_started.wait(timeout=5)
    evaluation_thread = threading.Thread(target=evaluate)
    evaluation_thread.start()
    assert not evaluation_done.wait(timeout=0.05)
    release_transition.set()
    transition_thread.join(timeout=5)
    evaluation_thread.join(timeout=5)

    assert reports[0].injections[0].injection_succeeded
    assert reports[0].injections[0].status is InjectionStatus.ACTIVE
    session.close()


def test_evaluation_waits_for_correlated_lifecycle_transition() -> None:
    model = TinyModel()
    faults = (
        _corruption(
            fault_id="layer-zero",
            target=_target(surface=FaultSurface.OUTPUT, module_path="layers.0"),
        ),
        _corruption(
            fault_id="layer-one",
            target=_target(surface=FaultSurface.OUTPUT, module_path="layers.1"),
        ),
    )
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,), faults=faults)),
        rank=0,
    )
    records = session.records
    transition_started = threading.Event()
    release_transition = threading.Event()
    evaluation_done = threading.Event()
    reports = []

    def transition() -> None:
        with session._lifecycle_lock:
            with records[0]._lock:
                records[0].verified = True
                records[0].status = InjectionStatus.COMPLETED
            transition_started.set()
            release_transition.wait(timeout=5)
            with records[1]._lock:
                records[1].verified = True
                records[1].status = InjectionStatus.COMPLETED

    def evaluate() -> None:
        reports.append(session.evaluate())
        evaluation_done.set()

    transition_thread = threading.Thread(target=transition)
    transition_thread.start()
    assert transition_started.wait(timeout=5)
    evaluation_thread = threading.Thread(target=evaluate)
    evaluation_thread.start()
    assert not evaluation_done.wait(timeout=0.05)
    release_transition.set()
    transition_thread.join(timeout=5)
    evaluation_thread.join(timeout=5)

    assert [record.status for record in reports[0].injections] == [
        InjectionStatus.COMPLETED,
        InjectionStatus.COMPLETED,
    ]
    session.close()


def test_live_record_serialization_waits_for_atomic_transition() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
    )
    record = session.records[0]
    transition_started = threading.Event()
    release_transition = threading.Event()
    serialization_done = threading.Event()
    payloads: list[dict[str, object]] = []

    def transition() -> None:
        with record._lock:
            record.verified = True
            transition_started.set()
            release_transition.wait(timeout=5)
            record.status = InjectionStatus.ACTIVE

    def serialize() -> None:
        payloads.append(record.to_dict())
        serialization_done.set()

    transition_thread = threading.Thread(target=transition)
    transition_thread.start()
    assert transition_started.wait(timeout=5)
    serialization_thread = threading.Thread(target=serialize)
    serialization_thread.start()
    assert not serialization_done.wait(timeout=0.05)
    release_transition.set()
    transition_thread.join(timeout=5)
    serialization_thread.join(timeout=5)

    assert payloads[0]["verified"] is True
    assert payloads[0]["status"] == InjectionStatus.ACTIVE.value
    assert payloads[0]["injection_succeeded"] is True
    session.close()


def test_bit_flip_on_unsupported_float_dtype_fails_without_escaping_hook() -> None:
    if not hasattr(torch, "float8_e4m3fn"):
        pytest.skip("PyTorch does not expose float8")

    class Float8OutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])
            self.anchor = nn.Parameter(torch.ones(()))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layers[0](value)

    model = Float8OutputModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(
                    _corruption(
                        operation=CorruptionOperation.SINGLE_BITFLIP,
                        scope=FaultScope.SINGLE,
                    ),
                ),
            )
        ),
        rank=0,
    )

    output = model(torch.ones(2, 4, dtype=torch.float8_e4m3fn))

    assert output.dtype is torch.float8_e4m3fn
    assert session.records[0].status is InjectionStatus.FAILED
    assert "bit flips do not support dtype" in str(session.records[0].error)
    session.close()


def test_set_value_overflow_fails_without_escaping_output_hook() -> None:
    class HalfOutputModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = nn.ModuleList([nn.Identity()])
            self.anchor = nn.Parameter(torch.ones(()))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layers[0](value)

    model = HalfOutputModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(
                    _corruption(
                        operation=CorruptionOperation.SET_VALUE,
                        scope=FaultScope.SINGLE,
                        value=1e100,
                    ),
                ),
            )
        ),
        rank=0,
    )

    output = model(torch.ones(2, 4, dtype=torch.float16))

    assert output.dtype is torch.float16
    assert session.records[0].status is InjectionStatus.FAILED
    assert "outside dtype torch.float16 range" in str(session.records[0].error)
    session.close()


def test_sparse_gradient_fails_without_escaping_hook() -> None:
    class SparseEmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4, sparse=True)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.embedding(tokens).sum()

    model = SparseEmbeddingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(
                    _corruption(
                        target=_target(
                            surface=FaultSurface.GRADIENT,
                            module_path="embedding",
                        ),
                    ),
                ),
            )
        ),
        rank=0,
    )

    model(torch.tensor([1, 3])).backward()

    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.is_sparse
    assert session.records[0].status is InjectionStatus.FAILED
    assert "layout torch.sparse_coo is not supported" in str(session.records[0].error)
    session.close()


def test_sparse_gradient_history_fails_without_escaping_hook() -> None:
    class SparseEmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4, sparse=True)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.embedding(tokens).sum()

    model = SparseEmbeddingModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stale_gradient = FaultSpec(
        fault_id="stale-gradient",
        type=FailureType.STALE_STATE,
        target=_target(
            surface=FaultSurface.GRADIENT,
            module_path="embedding",
        ),
        parameters={"scope": FaultScope.SINGLE.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,), faults=(stale_gradient,))),
        rank=0,
    )

    _step(model, optimizer, value=torch.tensor([1, 3]))
    assert session.records[0].status is InjectionStatus.PENDING
    _step(model, optimizer, value=torch.tensor([2, 4]))

    assert model.embedding.weight.grad is not None
    assert model.embedding.weight.grad.is_sparse
    assert session.records[0].status is InjectionStatus.FAILED
    assert "layout torch.sparse_coo is not supported" in str(session.records[0].error)
    session.close()


def test_empty_gradient_history_fails_without_escaping_hook() -> None:
    class EmptyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.empty(0))

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.sum() + self.weight.sum()

    class EmptyParameterModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = EmptyLayer()

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layer(value)

    model = EmptyParameterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stale_gradient = FaultSpec(
        fault_id="stale-empty-gradient",
        type=FailureType.STALE_STATE,
        target=_target(
            surface=FaultSurface.GRADIENT,
            module_path="layer",
        ),
        parameters={"scope": FaultScope.SINGLE.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,), faults=(stale_gradient,))),
        rank=0,
    )

    _step(model, optimizer)
    _step(model, optimizer)

    assert session.records[0].status is InjectionStatus.FAILED
    assert "must be non-empty" in str(session.records[0].error)
    session.close()


def test_empty_module_history_invalidates_the_observation_slot() -> None:
    class BatchModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layer = nn.Linear(1, 1)

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.layer(value)

    model = BatchModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    stale_output = FaultSpec(
        fault_id="stale-output",
        type=FailureType.STALE_STATE,
        target=_target(
            surface=FaultSurface.OUTPUT,
            module_path="layer",
        ),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(5, 6), faults=(stale_output,))),
        rank=0,
    )

    for value in (
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.ones(1, 1),
        torch.empty(0, 1),
        torch.full((1, 1), 2.0),
    ):
        optimizer.zero_grad()
        model(value).sum().backward()
        optimizer.step()

    assert len(session.records) == 2
    assert session.records[1].status is InjectionStatus.FAILED
    assert "no prior observed value" in str(session.records[1].error)
    session.close()


def test_rejected_history_observation_invalidates_the_latest_slot() -> None:
    history = _History()
    _observe_history(history, torch.tensor([1.0]), FaultScope.SINGLE)
    sparse = torch.sparse_coo_tensor(
        torch.tensor([[0]]),
        torch.tensor([2.0]),
        (1,),
        check_invariants=False,
    )
    _observe_history(history, sparse, FaultScope.SINGLE)
    _observe_history(history, torch.tensor([3.0]), FaultScope.SINGLE)

    assert history.observation_error is None
    assert history.previous is None
    torch.testing.assert_close(history.latest, torch.tensor([3.0]))


def test_close_interrupts_in_flight_delay() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                faults=(
                    FaultSpec(
                        fault_id="long-delay",
                        type=FailureType.DELAY,
                        target=_target(surface=FaultSurface.COMPUTE),
                        parameters={"delay_ms": 10_000.0},
                    ),
                ),
            )
        ),
        rank=0,
    )
    effect = session._active[0].effect
    entered = threading.Event()
    original_event = effect.cancel_event

    class NotifyingEvent:
        def set(self) -> None:
            original_event.set()

        def is_set(self) -> bool:
            return original_event.is_set()

        def wait(self, timeout: float | None = None) -> bool:
            entered.set()
            return original_event.wait(timeout)

    effect.cancel_event = NotifyingEvent()
    forward = threading.Thread(target=lambda: model(torch.ones(2, 4)))
    forward.start()
    assert entered.wait(timeout=5)

    session.close()
    forward.join(timeout=1)

    assert not forward.is_alive()
    assert session.records[0].status is InjectionStatus.CANCELLED


def test_close_cancels_verified_effect_before_matching_call_lifetime_finishes() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(matching_calls=10),
            )
        ),
        rank=0,
    )

    model(torch.ones(2, 4))
    assert session.records[0].status is InjectionStatus.ACTIVE

    session.close()

    assert session.records[0].status is InjectionStatus.CANCELLED
    assert not session.records[0].injection_succeeded


def test_history_observer_cleanup_attempts_every_handle() -> None:
    removed: list[str] = []

    class Handle:
        def __init__(self, name: str, *, fail: bool = False) -> None:
            self.name = name
            self.fail = fail

        def remove(self) -> None:
            removed.append(self.name)
            if self.fail:
                raise RuntimeError(f"{self.name} failed")

    executor = LocalFaultExecutor(MagicMock(), rank=0)
    executor._observer_handles = {
        ("first",): [Handle("first", fail=True), Handle("second")],
        ("third",): [Handle("third")],
    }

    with pytest.raises(RuntimeError, match="first failed"):
        executor.close()

    assert removed == ["first", "second", "third"]
    assert executor._observer_handles == {}


@pytest.mark.parametrize(
    "surface",
    [FaultSurface.INPUT, FaultSurface.OUTPUT, FaultSurface.GRADIENT],
)
def test_flow_history_stores_only_the_selected_scope(surface: FaultSurface) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id=f"stale-{surface.value}",
        type=FailureType.STALE_STATE,
        target=_target(surface=surface),
        parameters={"scope": FaultScope.SINGLE.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(2,), faults=(fault,))),
        rank=0,
    )

    optimizer.zero_grad()
    model(torch.ones(64, 4)).sum().backward()
    history = next(iter(session._local._history.values()))

    assert history.latest is not None
    assert history.latest.numel() == 1
    assert history.latest_shape is not None
    assert history.latest_shape.numel() > history.latest.numel()
    session.close()


def test_global_expert_resolution_ignores_numeric_descendants() -> None:
    class ExpertModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.experts = nn.ModuleList([nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))])

        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return self.experts[0](value)

    model = ExpertModel()
    context = resolve_training_context(model, _optimizer(model))
    target = FaultTarget(
        rank=0,
        surface=FaultSurface.OUTPUT,
        component="expert",
        index=0,
        metadata={"expert_parallel_rank": 0, "num_local_experts": 1},
    )

    assert context.resolve_module(target) is model.experts[0]
    context.close()


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


def test_one_shot_executor_rejects_active_results_even_with_cleanup() -> None:
    events: list[str] = []

    def activate(_request):
        events.append("activate")
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(_request, _result):
        events.append("deactivate")
        return {"deactivated": True}

    executor = CallbackFaultExecutor(
        name="contradictory-one-shot",
        supported_types={FailureType.EXCEPTION},
        activate=activate,
        deactivate=deactivate,
        one_shot=True,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()

    with pytest.raises(ValueError, match="one-shot fault executor returned an active effect"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    faults=(_external_fault(FailureType.EXCEPTION),),
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == ["activate", "deactivate"]


def test_external_matching_calls_requires_inline_capability_before_activation() -> None:
    events: list[str] = []

    def activate(_request):
        events.append("activate")
        return FaultExecutionResult(verified=True, active=True)

    executor = CallbackFaultExecutor(
        name="not-inline",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=lambda _request, _result: events.append("deactivate"),
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()

    with pytest.raises(ValueError, match="completes_inline=True"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(matching_calls=1),
                    faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == []


def test_external_persistent_lifetime_requires_an_active_result() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.EXCEPTION},
        events,
        active=False,
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
                lifetime=IncidentLifetime(iterations=2),
                faults=(_external_fault(FailureType.EXCEPTION),),
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(ValueError, match="completed inline before"):
        _step(model, optimizer)

    assert events == [("activate", "exception")]
    assert session.records[0].status is InjectionStatus.FAILED
    session.close()


def test_external_activation_failure_records_selected_executor() -> None:
    def activate(_request):
        raise RuntimeError("activation failed")

    executor = CallbackFaultExecutor(
        name="failing-executor",
        supported_types={FailureType.EXCEPTION},
        activate=activate,
        deactivate=lambda _request, _result: None,
        completes_inline=True,
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


def test_invalid_external_evidence_retains_cleanup_ownership_across_interrupt() -> None:
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
        if events.count("deactivate") == 1:
            raise KeyboardInterrupt("stop evidence cleanup")
        return None

    executor = CallbackFaultExecutor(
        name="interrupting-invalid-evidence",
        supported_types={FailureType.PROCESS_TERMINATION},
        activate=activate,
        deactivate=deactivate,
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
                lifetime=IncidentLifetime(until="campaign_end"),
                faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(KeyboardInterrupt, match="stop evidence cleanup"):
        _step(model, optimizer)

    assert events == ["activate", "deactivate", "deactivate"]
    assert session.records[0].status is InjectionStatus.FAILED
    session.close()


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


def test_replacement_attempts_all_matching_effect_cleanup_after_failure() -> None:
    events: list[tuple[str, str]] = []

    def activate(request):
        events.append(("activate", request.fault.fault_id))
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(request, _result):
        events.append(("deactivate", request.fault.fault_id))
        if request.fault.fault_id == "first":
            raise RuntimeError("first cleanup failed")
        return None

    executor = CallbackFaultExecutor(
        name="partial-cleanup-failure",
        supported_types={
            FailureType.RESOURCE_UNAVAILABLE,
            FailureType.PROCESS_TERMINATION,
        },
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="first",
            resource="gpu-0",
        ),
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="second",
            resource="process-0",
        ),
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="replacement"),
                faults=faults,
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="replacement cleanup failed"):
        session.notify_replacement()

    assert events[-2:] == [("deactivate", "first"), ("deactivate", "second")]
    assert session.records[0].status is InjectionStatus.FAILED
    assert session.records[1].status is InjectionStatus.COMPLETED
    session.close()


def test_replacement_attempts_all_cleanup_after_interrupt() -> None:
    deactivated: list[str] = []

    def activate(request):
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(request, _result):
        deactivated.append(request.fault.fault_id)
        if request.fault.fault_id == "first":
            raise KeyboardInterrupt("stop replacement cleanup")
        return None

    executor = CallbackFaultExecutor(
        name="interrupting-replacement-cleanup",
        supported_types={
            FailureType.RESOURCE_UNAVAILABLE,
            FailureType.PROCESS_TERMINATION,
        },
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="first",
            resource="gpu-0",
        ),
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="second",
            resource="process-0",
        ),
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="replacement"),
                faults=faults,
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(KeyboardInterrupt, match="stop replacement cleanup"):
        session.notify_replacement()

    assert deactivated == ["first", "second"]
    assert session.records[0].status is InjectionStatus.FAILED
    assert session.records[1].status is InjectionStatus.COMPLETED
    assert session._active == []
    session.close()


def test_expiration_attempts_all_matching_effect_cleanup_after_failure() -> None:
    events: list[tuple[str, str]] = []

    def activate(request):
        events.append(("activate", request.fault.fault_id))
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(request, _result):
        events.append(("deactivate", request.fault.fault_id))
        if request.fault.fault_id == "first":
            raise RuntimeError("first expiration cleanup failed")
        return None

    executor = CallbackFaultExecutor(
        name="partial-expiration-failure",
        supported_types={
            FailureType.RESOURCE_UNAVAILABLE,
            FailureType.PROCESS_TERMINATION,
        },
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="first",
            resource="gpu-0",
        ),
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="second",
            resource="process-0",
        ),
    )
    model = TinyModel()
    optimizer = _optimizer(model)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=1),
                faults=faults,
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="first expiration cleanup failed"):
        _step(model, optimizer)

    assert events[-2:] == [("deactivate", "first"), ("deactivate", "second")]
    assert session.records[0].status is InjectionStatus.FAILED
    assert session.records[1].status is InjectionStatus.COMPLETED
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


def test_framework_context_cleanup_attempts_every_restoration() -> None:
    model = TinyModel()
    context = resolve_training_context(model, _optimizer(model))
    events: list[str] = []

    def first() -> None:
        events.append("first")

    def failing() -> None:
        events.append("failing")
        raise RuntimeError("optimizer hook removal failed")

    def last() -> None:
        events.append("last")

    context._cleanups.extend((first, failing, last))

    with pytest.raises(RuntimeError, match="optimizer hook removal failed"):
        context.close()

    assert events == ["last", "failing", "first"]
    assert context._cleanups == []


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

    with pytest.raises(ValueError, match="same resolved target"):
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


def test_cross_incident_target_overlap_is_rejected_before_training() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    target = _target(surface=FaultSurface.WEIGHT)
    campaign = _campaign(
        _incident(
            incident_id="long-window",
            at=(1,),
            lifetime=IncidentLifetime(iterations=2),
            faults=(
                _corruption(
                    fault_id="long-weight",
                    target=target,
                    scope=FaultScope.SINGLE,
                ),
            ),
        ),
        _incident(
            incident_id="overlapping-window",
            at=(2,),
            lifetime=IncidentLifetime(iterations=1),
            faults=(
                _corruption(
                    fault_id="overlap-weight",
                    target=target,
                    scope=FaultScope.SINGLE,
                ),
            ),
        ),
    )

    with pytest.raises(ValueError, match="may not overlap on the same resolved target"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=campaign,
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)


def test_repeated_local_matching_call_candidates_are_rejected() -> None:
    model = TinyModel()

    with pytest.raises(ValueError, match="matching_calls incidents require a single"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1, 2),
                    lifetime=IncidentLifetime(matching_calls=2),
                )
            ),
            rank=0,
        )


def test_unmatched_single_call_retires_before_later_incident() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    shared_target = _target(surface=FaultSurface.OUTPUT)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                incident_id="first-call",
                at=(1,),
                lifetime=IncidentLifetime(matching_calls=1),
                faults=(_corruption(fault_id="first", target=shared_target),),
            ),
            _incident(
                incident_id="later-call",
                at=(2,),
                lifetime=IncidentLifetime(matching_calls=1),
                faults=(_corruption(fault_id="later", target=shared_target),),
            ),
        ),
        rank=0,
    )

    optimizer.step()
    assert session.records[0].status is InjectionStatus.CANCELLED

    model(torch.ones(2, 4))
    assert session.records[1].status is InjectionStatus.COMPLETED
    session.close()


def test_equivalent_target_selectors_collide_by_resolved_parameter() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    explicit = _corruption(
        fault_id="explicit",
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path="layers.0",
        ),
        scope=FaultScope.SINGLE,
    )
    logical = _corruption(
        fault_id="logical",
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="transformer_block",
            index=0,
        ),
        scope=FaultScope.SINGLE,
        parameter="weight",
    )

    with pytest.raises(ValueError, match="same resolved target"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(explicit, logical),
                )
            ),
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)


def test_weight_and_bias_surface_aliases_collide_by_parameter_storage() -> None:
    model = TinyModel()
    baseline = model.layers[0].bias.detach().clone()
    weight_alias = _corruption(
        fault_id="weight-alias",
        target=_target(surface=FaultSurface.WEIGHT),
        scope=FaultScope.SINGLE,
        parameter="bias",
    )
    bias = _corruption(
        fault_id="bias",
        target=_target(surface=FaultSurface.BIAS),
        scope=FaultScope.SINGLE,
    )

    with pytest.raises(ValueError, match="same resolved target"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(weight_alias, bias),
                )
            ),
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].bias, baseline)


def test_state_history_is_separate_for_different_scopes() -> None:
    model = TinyModel()
    stale_single = FaultSpec(
        fault_id="stale-single",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.WEIGHT),
        parameters={"scope": FaultScope.SINGLE.value},
    )
    stale_full = FaultSpec(
        fault_id="stale-full",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.WEIGHT),
        parameters={"scope": FaultScope.FULL.value},
    )
    context = resolve_training_context(model, _optimizer(model))
    executor = LocalFaultExecutor(context, rank=0)
    executor.sync_history((stale_single, stale_full))

    assert len(executor._history) == 2
    executor.close()
    context.close()


def test_optimizer_state_history_survives_state_dict_tensor_replacement() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    _step(model, optimizer)
    stale = FaultSpec(
        fault_id="stale-optimizer-state",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameters={
            "scope": FaultScope.SINGLE.value,
            "parameter": "weight",
            "state_key": "exp_avg",
        },
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(3,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(stale,),
            )
        ),
        completed_iterations=1,
        rank=0,
    )
    state = copy.deepcopy(optimizer.state_dict())
    previous_tensor = optimizer.state[model.layers[0].weight]["exp_avg"]

    optimizer.load_state_dict(state)
    replacement_tensor = optimizer.state[model.layers[0].weight]["exp_avg"]
    assert replacement_tensor is not previous_tensor
    _step(model, optimizer)
    _step(model, optimizer)

    assert session.records[0].injection_succeeded
    session.close()


def test_stale_state_uses_one_update_old_snapshot() -> None:
    model = nn.Sequential(nn.Linear(1, 1, bias=False))
    with torch.no_grad():
        model[0].weight.fill_(1.0)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    baseline = model[0].weight.detach().clone()
    stale = FaultSpec(
        fault_id="stale-weight",
        type=FailureType.STALE_STATE,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.WEIGHT,
            module_path="0",
        ),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(2,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(stale,),
            )
        ),
        rank=0,
    )

    optimizer.zero_grad()
    model(torch.ones(1, 1)).sum().backward()
    optimizer.step()

    torch.testing.assert_close(model[0].weight, baseline)
    assert session.records[0].injection_succeeded
    session.close()


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
            rank=None,
        ),
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="gpu",
            resource="gpu-3",
            rank=None,
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
                failed_resources=("gpu-3", "process-3"),
                kind="process_failure",
            )
        ]
    )

    assert {record.occurrence_id for record in session.records} == {"incident@1"}
    assert all(record.expected_rank is None for record in session.records)
    assert report.evaluations[0].localized
    assert report.evaluations[0].kind_matches
    assert report.evaluations[0].expected_ranks == ()
    assert report.evaluations[0].reported_ranks == ()
    overclaimed = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-3", "process-3"),
                kind="process_failure",
            )
        ]
    ).evaluations[0]
    assert not overclaimed.localized
    assert overclaimed.unexpected_ranks == (0,)
    session.close()


def test_resource_target_with_explicit_rank_requires_both_attribution_dimensions() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.RESOURCE_UNAVAILABLE}, events)
    fault = _external_fault(
        FailureType.RESOURCE_UNAVAILABLE,
        fault_id="ranked-gpu",
        resource="gpu-3",
        rank=3,
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        executors=(executor,),
        rank=3,
    )

    complete = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(3,),
                failed_resources=("gpu-3",),
                kind="process_failure",
            )
        ]
    ).evaluations[0]
    missing_rank = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=("gpu-3",),
                kind="process_failure",
            )
        ]
    ).evaluations[0]

    assert complete.expected_ranks == (3,)
    assert complete.expected_resources == ("gpu-3",)
    assert complete.localized
    assert not missing_rank.localized
    session.close()


def test_rank_resource_pairs_must_preserve_their_association() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.RESOURCE_UNAVAILABLE}, events)
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="rank-zero-gpu",
            resource="gpu-0",
            rank=0,
        ),
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="rank-one-gpu",
            resource="gpu-1",
            rank=1,
        ),
    )
    campaign = _campaign(_incident(at=(1,), faults=faults))
    model = TinyModel()
    rank_zero = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )
    rank_one = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=campaign,
        executors=(executor,),
        rank=1,
    )
    rank_zero._records.extend(rank_one.records)

    swapped = rank_zero.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-1",),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(1,),
                failed_resources=("gpu-0",),
                kind="process_failure",
            ),
        ]
    ).evaluations[0]

    assert swapped.expected_ranks == (0, 1)
    assert swapped.expected_resources == ("gpu-0", "gpu-1")
    assert not swapped.localized
    rank_zero.close()
    rank_one.close()


def test_correlated_multi_rank_incident_requires_all_action_records() -> None:
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

    assert not evaluation.injection_succeeded
    assert evaluation.expected_ranks == (0, 1)
    assert not evaluation.localized
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
            rank=None,
        ),
        _external_fault(
            FailureType.TIMEOUT,
            fault_id="timeout",
            resource="worker-0",
            rank=None,
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
                failed_resources=("process-0",),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=("worker-0",),
                kind="straggler",
            ),
        ]
    ).evaluations[0]
    overclaimed = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=("process-0", "worker-0"),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=("process-0", "worker-0"),
                kind="straggler",
            ),
        ]
    ).evaluations[0]

    assert not partial.localized
    assert partial.kind_matches is False
    assert complete.localized
    assert complete.kind_matches
    assert not overclaimed.localized
    assert overclaimed.kind_matches is False
    session.close()


def test_mixed_kinds_preserve_rank_resource_pairs_within_each_kind() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.RESOURCE_UNAVAILABLE, FailureType.TIMEOUT},
        events,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="process-rank-zero",
            resource="gpu-a",
            rank=0,
        ),
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="process-rank-one",
            resource="gpu-b",
            rank=1,
        ),
        _external_fault(
            FailureType.TIMEOUT,
            fault_id="timeout-rank-zero",
            resource="gpu-b",
            rank=0,
        ),
        _external_fault(
            FailureType.TIMEOUT,
            fault_id="timeout-rank-one",
            resource="gpu-a",
            rank=1,
        ),
    )
    campaign = _campaign(_incident(at=(1,), faults=faults))
    rank_zero_model = TinyModel()
    rank_one_model = TinyModel()
    rank_zero = enable_fault_injection(
        rank_zero_model,
        _optimizer(rank_zero_model),
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )
    rank_one = enable_fault_injection(
        rank_one_model,
        _optimizer(rank_one_model),
        campaign=campaign,
        executors=(executor,),
        rank=1,
    )
    rank_zero._records.extend(rank_one.records)

    swapped = rank_zero.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-b",),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(1,),
                failed_resources=("gpu-a",),
                kind="process_failure",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-a",),
                kind="straggler",
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(1,),
                failed_resources=("gpu-b",),
                kind="straggler",
            ),
        ]
    ).evaluations[0]

    assert swapped.expected_ranks == (0, 1)
    assert swapped.expected_resources == ("gpu-a", "gpu-b")
    assert swapped.kind_matches is False
    assert not swapped.localized
    rank_zero.close()
    rank_one.close()


def test_mixed_kind_components_require_kind_component_association() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.PROCESS_TERMINATION, FailureType.TIMEOUT},
        events,
    )
    shared_resource = "worker-0"
    faults = (
        FaultSpec(
            fault_id="hidden-process",
            type=FailureType.PROCESS_TERMINATION,
            target=FaultTarget(
                rank=None,
                surface=FaultSurface.RESOURCE,
                resource=shared_resource,
                component="hidden",
            ),
        ),
        FaultSpec(
            fault_id="output-timeout",
            type=FailureType.TIMEOUT,
            target=FaultTarget(
                rank=None,
                surface=FaultSurface.RESOURCE,
                resource=shared_resource,
                component="output",
            ),
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

    swapped = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=(shared_resource,),
                kind="process_failure",
                components=("output",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_resources=(shared_resource,),
                kind="straggler",
                components=("hidden",),
            ),
        ]
    ).evaluations[0]

    assert swapped.kind_matches
    assert swapped.component_matches is False
    assert not swapped.localized
    session.close()


def test_correlated_components_require_per_target_evidence() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.RESOURCE_UNAVAILABLE}, events)
    faults = (
        FaultSpec(
            fault_id="layer-resource",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="gpu-layer",
                component="layers.0",
            ),
        ),
        FaultSpec(
            fault_id="embedding-resource",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="gpu-embedding",
                component="embedding",
            ),
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

    complete = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-layer",),
                kind="process_failure",
                components=("layers.0",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-embedding",),
                kind="process_failure",
                components=("embedding",),
            ),
        ]
    ).evaluations[0]
    swapped = session.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-layer",),
                kind="process_failure",
                components=("embedding",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-embedding",),
                kind="process_failure",
                components=("layers.0",),
            ),
        ]
    ).evaluations[0]

    assert complete.localized
    assert complete.component_matches
    assert not swapped.localized
    assert swapped.component_matches is False
    session.close()


def test_components_preserve_rank_resource_target_associations() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor({FailureType.RESOURCE_UNAVAILABLE}, events)
    faults = (
        FaultSpec(
            fault_id="hidden-rank-zero",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="gpu-0",
                component="hidden",
            ),
        ),
        FaultSpec(
            fault_id="hidden-rank-one",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=1,
                surface=FaultSurface.RESOURCE,
                resource="gpu-1",
                component="hidden",
            ),
        ),
        FaultSpec(
            fault_id="output-rank-zero",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="gpu-1",
                component="output",
            ),
        ),
        FaultSpec(
            fault_id="output-rank-one",
            type=FailureType.RESOURCE_UNAVAILABLE,
            target=FaultTarget(
                rank=1,
                surface=FaultSurface.RESOURCE,
                resource="gpu-0",
                component="output",
            ),
        ),
    )
    campaign = _campaign(_incident(at=(1,), faults=faults))
    rank_zero_model = TinyModel()
    rank_one_model = TinyModel()
    rank_zero = enable_fault_injection(
        rank_zero_model,
        _optimizer(rank_zero_model),
        campaign=campaign,
        executors=(executor,),
        rank=0,
    )
    rank_one = enable_fault_injection(
        rank_one_model,
        _optimizer(rank_one_model),
        campaign=campaign,
        executors=(executor,),
        rank=1,
    )
    rank_zero._records.extend(rank_one.records)

    swapped = rank_zero.evaluate(
        [
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-1",),
                kind="process_failure",
                components=("hidden",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(1,),
                failed_resources=("gpu-0",),
                kind="process_failure",
                components=("hidden",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(0,),
                failed_resources=("gpu-0",),
                kind="process_failure",
                components=("output",),
            ),
            LocalizationResult(
                occurrence_id="incident@1",
                detected=True,
                failed_ranks=(1,),
                failed_resources=("gpu-1",),
                kind="process_failure",
                components=("output",),
            ),
        ]
    ).evaluations[0]

    assert swapped.kind_matches
    assert swapped.component_matches is False
    assert not swapped.localized
    rank_zero.close()
    rank_one.close()


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


def test_extra_component_evidence_is_not_localized() -> None:
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
                failed_ranks=(0,),
                components=("layers.0", "embedding"),
            )
        ]
    ).evaluations[0]

    assert not evaluation.localized
    assert evaluation.component_matches is False
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


def test_peer_group_localization_is_detection_without_attribution_credit() -> None:
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
            {
                "occurrence_id": "incident@1",
                "detected": True,
                "failed_ranks": [0],
                "kind": "sdc",
                "scope": "peer_group",
            }
        ]
    ).evaluations[0]

    assert evaluation.detected
    assert not evaluation.localized
    assert evaluation.reported_ranks == ()
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


def test_state_fault_waits_for_async_device_reads_before_mutation_and_restore() -> None:
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

    with patch("lm_resiliency.fault_injection.local._synchronize_state_mutation") as synchronize:
        session = enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
            rank=0,
        )
        assert synchronize.call_count == 1
        session.notify_recovery()
        assert synchronize.call_count == 2

    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


@pytest.mark.parametrize(
    "surface",
    [FaultSurface.INPUT, FaultSurface.OUTPUT, FaultSurface.GRADIENT],
)
def test_missing_flow_history_fails_without_aborting_training(
    surface: FaultSurface,
) -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id=f"stale-{surface.value}",
        type=FailureType.STALE_STATE,
        target=_target(surface=surface),
        parameters={"scope": FaultScope.FULL.value},
    )
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(_incident(at=(1,), faults=(fault,))),
        rank=0,
    )

    _step(model, optimizer)

    assert session.records[0].status is InjectionStatus.FAILED
    assert "no prior observed value" in (session.records[0].error or "")
    session.close()


def test_first_iteration_state_history_fails_without_aborting_enablement() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    fault = FaultSpec(
        fault_id="stale-weight",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.WEIGHT),
        parameters={"scope": FaultScope.FULL.value},
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

    assert session.records[0].status is InjectionStatus.FAILED
    assert "no prior observed value" in (session.records[0].error or "")
    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_optimizer_state_history_waits_for_lazy_state_initialization() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    fault = FaultSpec(
        fault_id="stale-exp-avg",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameters={
            "scope": FaultScope.FULL.value,
            "parameter": "weight",
            "state_key": "exp_avg",
        },
    )

    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(2,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(fault,),
            )
        ),
        rank=0,
    )
    _step(model, optimizer)
    _step(model, optimizer)

    assert session.records[0].status is InjectionStatus.FAILED
    assert "no prior observed value" in (session.records[0].error or "")
    session.close()


def test_logical_layer_rejects_ambiguous_suffix_matches() -> None:
    class EncoderDecoderModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Module()
            self.encoder.layers = nn.ModuleList([nn.Linear(4, 4)])
            self.decoder = nn.Module()
            self.decoder.layers = nn.ModuleList([nn.Linear(4, 4)])

    model = EncoderDecoderModel()
    fault = _corruption(
        target=_target(
            surface=FaultSurface.WEIGHT,
            module_path=None,
            component="transformer_block",
            index=0,
        ),
    )

    with pytest.raises(LookupError, match="resolves to multiple modules"):
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


def test_staged_activation_rollback_marks_verified_records_failed() -> None:
    def activate(_request):
        raise RuntimeError("later activation failed")

    executor = CallbackFaultExecutor(
        name="failing-executor",
        supported_types={FailureType.EXCEPTION},
        activate=activate,
        deactivate=lambda _request, _result: None,
        completes_inline=True,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                incident_id="local-first",
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
                incident_id="external-second",
                at=(1,),
                faults=(_external_fault(FailureType.EXCEPTION),),
            ),
        ),
        executors=(executor,),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()

    with pytest.raises(RuntimeError, match="later activation failed"):
        session._start()

    local_record = next(record for record in session.records if record.incident_id == "local-first")
    assert local_record.status is InjectionStatus.FAILED
    assert not local_record.injection_succeeded
    assert "rolled back" in (local_record.error or "")
    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_safe_activation_interrupt_rolls_back_and_reaches_consensus() -> None:
    def activate(request):
        if request.fault.fault_id == "first":
            return FaultExecutionResult(verified=True, active=False)
        raise KeyboardInterrupt("stop activation")

    executor = CallbackFaultExecutor(
        name="interrupting-safe-executor",
        supported_types={FailureType.DELAY},
        activate=activate,
        one_shot=True,
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )
    faults = (
        FaultSpec(
            fault_id="first",
            type=FailureType.DELAY,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="first",
            ),
            parameters={"delay_ms": 1.0},
        ),
        FaultSpec(
            fault_id="interrupt",
            type=FailureType.DELAY,
            target=FaultTarget(
                rank=0,
                surface=FaultSurface.RESOURCE,
                resource="interrupt",
            ),
            parameters={"delay_ms": 1.0},
        ),
    )
    model = TinyModel()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,), faults=faults)),
        executors=(executor,),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    stages: list[tuple[str, BaseException | None]] = []

    def gather(error: BaseException | None, stage: str) -> list[str]:
        stages.append((stage, error))
        if stage == "iteration arming":
            return ["rank 0: KeyboardInterrupt: stop activation"]
        return []

    with (
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch.object(session, "_gather_runtime_rank_errors", side_effect=gather),
        pytest.raises(KeyboardInterrupt, match="stop activation"),
    ):
        session._enter_iteration_consistently(1)

    assert [stage for stage, _error in stages] == [
        "iteration preflight",
        "attempt persistence",
        "iteration arming",
    ]
    assert isinstance(stages[-1][1], KeyboardInterrupt)
    assert all(record.status is InjectionStatus.FAILED for record in session.records)
    assert all(not record.injection_succeeded for record in session.records)


def test_attempt_persistence_interrupt_reaches_rank_consensus() -> None:
    model = TinyModel()
    store = MemoryCampaignStateStore()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        state_store=store,
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    original_compare_and_swap = store.compare_and_swap
    compare_calls = 0

    def compare_and_swap(expected, updated):
        nonlocal compare_calls
        compare_calls += 1
        if compare_calls == 1:
            raise KeyboardInterrupt("stop persistence")
        return original_compare_and_swap(expected, updated)

    stages: list[tuple[str, BaseException | None]] = []

    def gather(error: BaseException | None, stage: str) -> list[str]:
        stages.append((stage, error))
        if stage == "attempt persistence":
            return ["rank 0: KeyboardInterrupt: stop persistence"]
        return []

    with (
        patch.object(store, "compare_and_swap", side_effect=compare_and_swap),
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch.object(session, "_gather_runtime_rank_errors", side_effect=gather),
        pytest.raises(KeyboardInterrupt, match="stop persistence") as caught,
    ):
        session._start()

    assert [stage for stage, _error in stages] == [
        "iteration preflight",
        "attempt persistence",
        "attempt rollback",
    ]
    assert isinstance(stages[1][1], KeyboardInterrupt)
    assert any("attempt persistence failed" in note for note in caught.value.__notes__)
    assert store.load(session.campaign.name).attempts == {}
    assert session._closed


def test_runtime_preflight_interrupt_reaches_rank_consensus() -> None:
    model = TinyModel()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    stages: list[tuple[str, BaseException | None]] = []

    def gather(error: BaseException | None, stage: str) -> list[str]:
        stages.append((stage, error))
        return ["rank 0: KeyboardInterrupt: stop preflight"]

    with (
        patch.object(
            session,
            "_preflight_iteration",
            side_effect=KeyboardInterrupt("stop preflight"),
        ),
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch.object(session, "_gather_runtime_rank_errors", side_effect=gather),
        pytest.raises(KeyboardInterrupt, match="stop preflight") as caught,
    ):
        session._start()

    assert len(stages) == 1
    assert stages[0][0] == "iteration preflight"
    assert isinstance(stages[0][1], KeyboardInterrupt)
    assert any("iteration preflight failed" in note for note in caught.value.__notes__)
    assert session._closed


def test_local_activation_interrupt_restores_partial_state_effect() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
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
        ),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    original_activate_state = session._local._activate_state

    def activate_then_interrupt(request, effect) -> None:
        original_activate_state(request, effect)
        raise KeyboardInterrupt("stop local activation")

    with (
        patch.object(
            session._local,
            "_activate_state",
            side_effect=activate_then_interrupt,
        ),
        pytest.raises(KeyboardInterrupt, match="stop local activation"),
    ):
        session._start()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    assert session.records[0].status is InjectionStatus.FAILED
    assert not session.records[0].injection_succeeded
    assert session._closed


def test_local_activation_rollback_propagates_cleanup_failure() -> None:
    record = MagicMock()
    effect = LocalFaultEffect(
        record=record,
        target_key=("parameter", 0),
        on_done=lambda _key: None,
        remaining_calls=None,
    )

    def fail_cleanup(_preserve_replaced_state: bool, _replacement_confirmed: bool) -> None:
        raise RuntimeError("local restoration failed")

    effect.cleanup_callbacks.append(fail_cleanup)
    active = _ActiveFault(_incident(), 1, effect)

    with pytest.raises(RuntimeError, match="local restoration failed"):
        active.rollback(RuntimeError("later activation failed"))

    assert effect.done
    assert record.status is InjectionStatus.FAILED
    assert "cleanup also failed: local restoration failed" in record.error


def test_staged_activation_surfaces_external_rollback_cleanup_failure() -> None:
    events: list[tuple[str, str]] = []

    def activate(request):
        events.append(("activate", request.fault.fault_id))
        if request.fault.fault_id == "second":
            raise RuntimeError("later activation failed")
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(request, _result):
        events.append(("deactivate", request.fault.fault_id))
        raise RuntimeError("rollback cleanup failed")

    executor = CallbackFaultExecutor(
        name="rollback-cleanup-failure",
        supported_types={
            FailureType.PROCESS_TERMINATION,
            FailureType.RESOURCE_UNAVAILABLE,
        },
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="first",
            resource="gpu-0",
        ),
        _external_fault(
            FailureType.PROCESS_TERMINATION,
            fault_id="second",
            resource="process-0",
        ),
    )
    model = TinyModel()

    with pytest.raises(RuntimeError, match="later activation failed") as caught:
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="campaign_end"),
                    faults=faults,
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == [
        ("activate", "first"),
        ("activate", "second"),
        ("deactivate", "first"),
    ]
    assert any("rollback cleanup failed" in note for note in caught.value.__notes__)


def test_staged_activation_continues_rollback_after_interrupt() -> None:
    events: list[tuple[str, str]] = []

    def activate(request):
        events.append(("activate", request.fault.fault_id))
        if request.fault.fault_id == "fourth":
            raise RuntimeError("later activation failed")
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(request, _result):
        events.append(("deactivate", request.fault.fault_id))
        if request.fault.fault_id == "third":
            raise KeyboardInterrupt("stop rollback")
        return None

    executor = CallbackFaultExecutor(
        name="interrupting-rollback",
        supported_types={FailureType.RESOURCE_UNAVAILABLE},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = tuple(
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id=fault_id,
            resource=f"gpu-{index}",
        )
        for index, fault_id in enumerate(("first", "second", "third", "fourth"))
    )
    model = TinyModel()

    with pytest.raises(RuntimeError, match="later activation failed") as caught:
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(until="campaign_end"),
                    faults=faults,
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == [
        ("activate", "first"),
        ("activate", "second"),
        ("activate", "third"),
        ("activate", "fourth"),
        ("deactivate", "third"),
        ("deactivate", "second"),
        ("deactivate", "first"),
    ]
    assert any("stop rollback" in note for note in caught.value.__notes__)


def test_close_continues_after_interrupt_class_deactivation_failure() -> None:
    deactivated: list[str] = []

    def activate(request):
        return FaultExecutionResult(
            verified=True,
            active=True,
            token=request.fault.fault_id,
        )

    def deactivate(request, _result):
        deactivated.append(request.fault.fault_id)
        if request.fault.fault_id == "second":
            raise KeyboardInterrupt("stop cleanup")
        return None

    executor = CallbackFaultExecutor(
        name="interrupting-cleanup",
        supported_types={FailureType.RESOURCE_UNAVAILABLE},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )
    faults = (
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="first",
            resource="gpu-0",
        ),
        _external_fault(
            FailureType.RESOURCE_UNAVAILABLE,
            fault_id="second",
            resource="gpu-1",
        ),
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(until="campaign_end"),
                faults=faults,
            )
        ),
        executors=(executor,),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="fault injection cleanup failed") as caught:
        session.close()

    assert isinstance(caught.value.__cause__, KeyboardInterrupt)
    assert deactivated == ["second", "first"]
    assert session._closed


def test_probability_skip_does_not_require_expiration_consensus() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                probability=0.0,
                lifetime=IncidentLifetime(iterations=2),
            )
        ),
        rank=0,
    )

    assert session.records[0].status is InjectionStatus.SKIPPED_PROBABILITY
    assert not session._requires_boundary_consensus(1, 2)
    session.close()


def test_selected_bounded_occurrence_requires_consensus_on_non_target_ranks() -> None:
    model = TinyModel()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=2),
            )
        ),
        rank=1,
        _defer_activation=True,
    )
    session._journal.record_attempt("incident", 1)

    assert session._requires_boundary_consensus(2, 3)
    session.close()


def test_destructive_expiration_does_not_require_in_band_consensus() -> None:
    model = TinyModel()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=2),
                faults=(_external_fault(FailureType.PROCESS_TERMINATION),),
            )
        ),
        executors=(
            _recording_executor(
                {FailureType.PROCESS_TERMINATION},
                [],
            ),
        ),
        rank=1,
        _defer_activation=True,
    )
    session._journal.record_attempt("incident", 1)

    assert not session._requires_boundary_consensus(2, 3)
    session.close()


def test_safe_expiration_interrupt_reaches_boundary_consensus() -> None:
    stages: list[tuple[str, BaseException | None]] = []

    def activate(_request):
        return FaultExecutionResult(verified=True, active=True)

    def deactivate(_request, _result):
        raise KeyboardInterrupt("stop expiration")

    executor = CallbackFaultExecutor(
        name="interrupting-expiration",
        supported_types={FailureType.DELAY},
        activate=activate,
        deactivate=deactivate,
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )
    fault = FaultSpec(
        fault_id="safe-delay",
        type=FailureType.DELAY,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.RESOURCE,
            resource="worker-0",
        ),
        parameters={"delay_ms": 1.0},
    )
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _optimizer(model),
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(fault,),
            )
        ),
        executors=(executor,),
        rank=0,
    )

    def gather(error: BaseException | None, stage: str) -> list[str]:
        stages.append((stage, error))
        return ["rank 0: KeyboardInterrupt: stop expiration"]

    with (
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch.object(session, "_gather_runtime_rank_errors", side_effect=gather),
        pytest.raises(KeyboardInterrupt, match="stop expiration") as caught,
    ):
        session._on_step_complete()

    assert len(stages) == 1
    assert stages[0][0] == "iteration preparation"
    assert isinstance(stages[0][1], KeyboardInterrupt)
    assert any("iteration preparation failed" in note for note in caught.value.__notes__)
    assert session._closed


def test_optimizer_container_discovers_child_optimizers_first() -> None:
    parameter = nn.Parameter(torch.ones(1))
    child = torch.optim.Adam([parameter])

    class OptimizerContainer(torch.optim.Optimizer):
        def __init__(self) -> None:
            super().__init__([nn.Parameter(torch.zeros(1))], {})
            self.optimizers = [child]

        def step(self, closure=None):
            return None

    container = OptimizerContainer()

    assert _base_optimizers(container) == (child,)


def test_integer_only_input_fault_fails_without_aborting_training() -> None:
    class EmbeddingModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(8, 4)

        def forward(self, tokens: torch.Tensor) -> torch.Tensor:
            return self.embedding(tokens).sum()

    model = EmbeddingModel()
    optimizer = _optimizer(model)
    fault = FaultSpec(
        fault_id="drop-token-ids",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.INPUT,
            module_path="embedding",
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
    model(torch.tensor([1, 2], dtype=torch.long)).backward()
    optimizer.step()

    assert session.records[0].status is InjectionStatus.FAILED
    assert "floating-point tensor" in (session.records[0].error or "")
    session.close()


def test_failed_model_state_load_does_not_mark_active_fault_replaced() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
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
        ),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(
            {"layers.0.weight": torch.ones(1)},
            strict=False,
        )
    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, baseline)
    session.close()


def test_partial_failed_model_state_load_marks_copied_fault_target_replaced() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    replacement = torch.full_like(model.layers[0].weight, 3.0)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
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
        ),
        rank=0,
    )

    with pytest.raises(RuntimeError, match="size mismatch"):
        model.load_state_dict(
            {
                "layers.0.weight": replacement,
                "layers.1.weight": torch.ones(1),
            },
            strict=False,
        )
    torch.testing.assert_close(model.layers[0].weight, replacement)

    session.notify_recovery()

    torch.testing.assert_close(model.layers[0].weight, replacement)
    session.close()


def test_bounded_state_replacement_before_expiration_fails_occurrence() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    replacement = torch.full_like(model.layers[0].weight, 5.0)
    session = enable_fault_injection(
        model,
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=3),
                faults=(
                    _corruption(
                        target=_target(surface=FaultSurface.WEIGHT),
                        scope=FaultScope.SINGLE,
                    ),
                ),
            )
        ),
        rank=0,
    )

    model.load_state_dict({"layers.0.weight": replacement}, strict=False)

    record = session.records[0]
    assert record.status is InjectionStatus.FAILED
    assert "replaced before its configured lifetime completed" in (record.error or "")
    torch.testing.assert_close(model.layers[0].weight, replacement)
    session.close()


def test_one_iteration_state_replacement_before_optimizer_boundary_fails_occurrence() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    replacement = torch.full_like(model.layers[0].weight, 7.0)
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
                        scope=FaultScope.SINGLE,
                    ),
                ),
            )
        ),
        rank=0,
    )

    model.load_state_dict({"layers.0.weight": replacement}, strict=False)

    record = session.records[0]
    assert record.status is InjectionStatus.FAILED
    assert "replaced before its configured lifetime completed" in (record.error or "")
    torch.testing.assert_close(model.layers[0].weight, replacement)
    session.close()


def test_implicit_and_explicit_optimizer_state_keys_collide() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    _step(model, optimizer)
    state = optimizer.state[model.layers[0].weight]["exp_avg"]
    baseline = state.detach().clone()
    implicit = _corruption(
        fault_id="implicit-state",
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter="weight",
    )
    explicit = _corruption(
        fault_id="explicit-state",
        target=_target(surface=FaultSurface.OPTIMIZER_STATE),
        parameter="weight",
        state_key="exp_avg",
    )

    with pytest.raises(ValueError, match="same resolved target"):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(
                _incident(
                    at=(2,),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(implicit, explicit),
                )
            ),
            completed_iterations=1,
            rank=0,
        )

    torch.testing.assert_close(state, baseline)


def test_duplicate_resolved_targets_in_one_incident_fail_enablement() -> None:
    model = TinyModel()
    by_weight_alias = _corruption(
        fault_id="weight-alias",
        target=_target(surface=FaultSurface.WEIGHT),
        parameter="bias",
    )
    by_bias_surface = _corruption(
        fault_id="bias-surface",
        target=_target(surface=FaultSurface.BIAS),
    )

    with pytest.raises(ValueError, match="same resolved target"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(2,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(by_weight_alias, by_bias_surface),
                )
            ),
            rank=0,
        )


def test_frozen_gradient_target_fails_before_training() -> None:
    model = TinyModel()
    model.layers[0].weight.requires_grad_(False)
    fault = _corruption(
        target=_target(surface=FaultSurface.GRADIENT),
    )

    with pytest.raises(LookupError, match="does not require gradients"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(_incident(at=(2,), faults=(fault,))),
            rank=0,
        )


def test_parameter_state_fault_rejects_registered_buffer_alias() -> None:
    model = nn.Sequential(nn.BatchNorm1d(4))
    fault = _corruption(
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.WEIGHT,
            module_path="0",
        ),
        parameter="running_mean",
    )

    with pytest.raises(LookupError, match="no tensor parameter"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(2,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(fault,),
                )
            ),
            rank=0,
        )


def test_collective_desync_expects_hang_localization() -> None:
    fault = FaultSpec(
        fault_id="collective-desync",
        type=FailureType.COLLECTIVE_DESYNC,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.COLLECTIVE,
        ),
    )

    assert fault.expected_kind == "hang"


def test_collective_drop_requires_cluster_destructive_executor() -> None:
    fault = FaultSpec(
        fault_id="collective-drop",
        type=FailureType.DROP,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.COLLECTIVE,
        ),
    )
    safe_executor = CallbackFaultExecutor(
        name="safe-drop",
        supported_types={FailureType.DROP},
        activate=lambda _request: FaultExecutionResult(verified=True, active=False),
        one_shot=True,
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )

    assert fault.safety is SafetyClass.CLUSTER_DESTRUCTIVE
    assert fault.expected_kind == "hang"
    assert not safe_executor.supports(fault)


def test_collective_duplicate_requires_cluster_destructive_executor() -> None:
    fault = FaultSpec(
        fault_id="collective-duplicate",
        type=FailureType.DUPLICATE,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.COLLECTIVE,
        ),
    )
    safe_executor = CallbackFaultExecutor(
        name="safe-duplicate",
        supported_types={FailureType.DUPLICATE},
        activate=lambda _request: FaultExecutionResult(verified=True, active=False),
        one_shot=True,
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )

    assert fault.safety is SafetyClass.CLUSTER_DESTRUCTIVE
    assert fault.expected_kind == "hang"
    assert not safe_executor.supports(fault)


def test_collective_reorder_requires_cluster_destructive_executor() -> None:
    fault = FaultSpec(
        fault_id="collective-reorder",
        type=FailureType.REORDER,
        target=FaultTarget(
            rank=0,
            surface=FaultSurface.COLLECTIVE,
        ),
    )
    safe_executor = CallbackFaultExecutor(
        name="safe-reorder",
        supported_types={FailureType.REORDER},
        activate=lambda _request: FaultExecutionResult(verified=True, active=False),
        one_shot=True,
        max_safety=SafetyClass.SAFE_IN_PROCESS,
    )

    assert fault.safety is SafetyClass.CLUSTER_DESTRUCTIVE
    assert fault.expected_kind == "hang"
    assert not safe_executor.supports(fault)


def test_memory_state_store_rejects_cross_campaign_compare_and_swap() -> None:
    store = MemoryCampaignStateStore()
    expected = CampaignJournal(campaign="campaign-a")
    updated = CampaignJournal(campaign="campaign-b")

    with pytest.raises(ValueError, match="requires one campaign"):
        store.compare_and_swap(expected, updated)

    assert store.load("campaign-a").to_dict() == expected.to_dict()
    assert store.load("campaign-b").to_dict() == updated.to_dict()


def test_falsey_state_store_is_preserved() -> None:
    class FalseyStore(MemoryCampaignStateStore):
        def __bool__(self) -> bool:
            return False

    model = TinyModel()
    store = FalseyStore()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(2,))),
        state_store=store,
        rank=0,
        _defer_activation=True,
    )

    assert session._state_store is store
    session.close()


def test_distributed_partial_attempt_save_restores_successful_rank_journal() -> None:
    model = TinyModel()
    store = MemoryCampaignStateStore()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(_incident(at=(1,))),
        state_store=store,
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if gathers == 2:
            values[:] = [local_value, "OSError: remote campaign state disk failed"]
        else:
            values[:] = [local_value, local_value]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(RuntimeError, match="attempt persistence failed"),
    ):
        session._start()

    assert store.load(session.campaign.name).attempts == {}


def test_runtime_consensus_exception_cleans_active_effects() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
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
        ),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    session._start()
    assert not torch.equal(model.layers[0].weight, baseline)

    with (
        patch.object(session, "_requires_boundary_consensus", return_value=True),
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch(
            "lm_resiliency.fault_injection.injector._gather_rank_errors",
            side_effect=RuntimeError("collective timed out"),
        ),
        pytest.raises(RuntimeError, match="collective timed out") as caught,
    ):
        session._on_step_complete()

    assert session._closed
    assert session.records[0].status is InjectionStatus.CANCELLED
    assert any("iteration preparation consensus failed" in note for note in caught.value.__notes__)
    torch.testing.assert_close(model.layers[0].weight, baseline)


def test_runtime_consensus_interrupt_cleans_active_effects() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
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
        ),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    session._start()
    assert not torch.equal(model.layers[0].weight, baseline)

    with (
        patch.object(session, "_requires_boundary_consensus", return_value=True),
        patch(
            "lm_resiliency.fault_injection.injector._distributed_world_size",
            return_value=2,
        ),
        patch(
            "lm_resiliency.fault_injection.injector._gather_rank_errors",
            side_effect=KeyboardInterrupt("collective interrupted"),
        ),
        pytest.raises(KeyboardInterrupt, match="collective interrupted") as caught,
    ):
        session._on_step_complete()

    assert session._closed
    assert session.records[0].status is InjectionStatus.CANCELLED
    assert any("iteration preparation consensus failed" in note for note in caught.value.__notes__)
    torch.testing.assert_close(model.layers[0].weight, baseline)


def test_non_consensus_preparation_failure_cleans_active_effects() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
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
        ),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    session._start()
    assert not torch.equal(model.layers[0].weight, baseline)

    with (
        patch.object(
            session._local,
            "sync_history",
            side_effect=RuntimeError("history preparation failed"),
        ),
        pytest.raises(RuntimeError, match="history preparation failed"),
    ):
        session._on_step_complete()

    assert session._closed
    assert session.records[0].status is InjectionStatus.CANCELLED
    torch.testing.assert_close(model.layers[0].weight, baseline)


def test_remote_safe_arming_failure_rolls_back_local_record() -> None:
    model = TinyModel()
    baseline = model.layers[0].weight.detach().clone()
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=_campaign(
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
        ),
        rank=0,
        _defer_activation=True,
    )
    session._commit_journal_binding()
    gathers = 0

    def gather(values, local_value) -> None:
        nonlocal gathers
        gathers += 1
        if gathers == 3:
            values[:] = [local_value, "RuntimeError: remote arming failed"]
        else:
            values[:] = [local_value, local_value]

    with (
        patch("lm_resiliency.fault_injection.injector.dist.is_available", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.is_initialized", return_value=True),
        patch("lm_resiliency.fault_injection.injector.dist.get_world_size", return_value=2),
        patch(
            "lm_resiliency.fault_injection.injector.dist.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(RuntimeError, match="remote arming failed"),
    ):
        session._start()

    assert session.records[0].status is InjectionStatus.FAILED
    assert not session.records[0].injection_succeeded
    torch.testing.assert_close(model.layers[0].weight, baseline)


@pytest.mark.parametrize(
    ("probability", "attempts"),
    [
        (0.0, {}),
        (1.0, {"incident@2": 1}),
    ],
)
def test_unselected_stale_candidates_do_not_install_history(
    probability: float,
    attempts: dict[str, int],
) -> None:
    model = TinyModel()
    fault = FaultSpec(
        fault_id="stale-output",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.OUTPUT),
        parameters={"scope": FaultScope.FULL.value},
    )
    campaign = _campaign(
        _incident(
            at=(2,),
            probability=probability,
            faults=(fault,),
        )
    )
    store = MemoryCampaignStateStore()
    store.save(
        CampaignJournal(
            campaign=campaign.name,
            manifest_identity=campaign.manifest_identity,
            attempts=attempts,
        )
    )
    session = FaultInjectionSession(
        model,
        _optimizer(model),
        campaign=campaign,
        state_store=store,
        rank=0,
        _defer_activation=True,
    )

    assert session._history_faults_for(1) == ()
    assert session._local._history == {}
    session.close()


def test_equal_valued_megatron_master_resynchronization_is_preserved() -> None:
    model = TinyModel().bfloat16()
    model_parameter = model.layers[0].weight
    main_parameter = nn.Parameter(model_parameter.detach().float().clone())
    base_optimizer = torch.optim.SGD([main_parameter], lr=0.0)

    class MixedPrecisionOptimizer:
        def __init__(self):
            self.float16_groups = [[model_parameter]]
            self.fp32_from_float16_groups = [[main_parameter]]
            self.optimizer = base_optimizer

        def step(self):
            with torch.no_grad():
                main_parameter.fill_(0.125)
                model_parameter.copy_(main_parameter)
            return True, None, None

    optimizer = MixedPrecisionOptimizer()
    session = enable_fault_injection(
        [Wrapper(model)],
        optimizer,
        campaign=_campaign(
            _incident(
                at=(1,),
                lifetime=IncidentLifetime(iterations=1),
                faults=(
                    _corruption(
                        target=_target(surface=FaultSurface.WEIGHT),
                        operation=CorruptionOperation.SET_VALUE,
                        scope=FaultScope.FULL,
                        value=0.125,
                    ),
                ),
            )
        ),
        rank=0,
    )

    optimizer.step()

    torch.testing.assert_close(
        model_parameter,
        main_parameter.to(model_parameter.dtype),
    )
    assert torch.all(model_parameter == torch.tensor(0.125, dtype=model_parameter.dtype))
    session.close()


def test_optimizer_state_schedule_distinguishes_explicit_state_entries() -> None:
    model = TinyModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0)
    _step(model, optimizer)
    faults = tuple(
        _corruption(
            fault_id=f"corrupt-{state_key}",
            target=_target(surface=FaultSurface.OPTIMIZER_STATE),
            parameter="weight",
            state_key=state_key,
        )
        for state_key in ("exp_avg", "exp_avg_sq")
    )
    context = resolve_training_context(model, optimizer)
    executor = LocalFaultExecutor(context, rank=0)

    executor.validate_schedule(
        (
            _incident(
                at=(2,),
                lifetime=IncidentLifetime(iterations=1),
                faults=faults,
            ),
        )
    )

    executor.close()
    context.close()


def test_large_disjoint_periodic_schedules_do_not_collide() -> None:
    model = TinyModel()
    context = resolve_training_context(model, _optimizer(model))
    executor = LocalFaultExecutor(context, rank=0)
    target = _target(surface=FaultSurface.WEIGHT)

    executor.validate_schedule(
        (
            _incident(
                incident_id="odd",
                trigger_range=IterationRange(start=1, end=100_001, every=2),
                lifetime=IncidentLifetime(iterations=1),
                faults=(_corruption(fault_id="odd-weight", target=target),),
            ),
            _incident(
                incident_id="even",
                trigger_range=IterationRange(start=2, end=100_000, every=2),
                lifetime=IncidentLifetime(iterations=1),
                faults=(_corruption(fault_id="even-weight", target=target),),
            ),
        )
    )

    executor.close()
    context.close()


def test_destructive_correlated_incident_requires_local_history_before_activation() -> None:
    events: list[tuple[str, str]] = []
    executor = _recording_executor(
        {FailureType.EXCEPTION},
        events,
        max_safety=SafetyClass.ISOLATED_DESTRUCTIVE,
    )
    stale = FaultSpec(
        fault_id="stale-weight",
        type=FailureType.STALE_STATE,
        target=_target(surface=FaultSurface.WEIGHT),
        parameters={"scope": FaultScope.FULL.value},
    )
    model = TinyModel()

    with pytest.raises(RuntimeError, match="no prior observed value"):
        enable_fault_injection(
            model,
            _optimizer(model),
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(stale, _external_fault(FailureType.EXCEPTION)),
                )
            ),
            executors=(executor,),
            rank=0,
        )

    assert events == []


def test_state_mutation_interrupt_restores_after_the_write_started() -> None:
    model = TinyModel()
    optimizer = _optimizer(model)
    baseline = model.layers[0].weight.detach().clone()
    interrupted = False

    def write_then_interrupt(tensor, indices, values) -> None:
        nonlocal interrupted
        _write_linear(tensor, indices, values)
        if not interrupted:
            interrupted = True
            raise KeyboardInterrupt("stop during state write")

    with (
        patch(
            "lm_resiliency.fault_injection.local._write_linear",
            side_effect=write_then_interrupt,
        ),
        pytest.raises(KeyboardInterrupt, match="stop during state write"),
    ):
        enable_fault_injection(
            model,
            optimizer,
            campaign=_campaign(
                _incident(
                    at=(1,),
                    lifetime=IncidentLifetime(iterations=1),
                    faults=(
                        _corruption(
                            target=_target(surface=FaultSurface.WEIGHT),
                            scope=FaultScope.FULL,
                        ),
                    ),
                )
            ),
            rank=0,
        )

    torch.testing.assert_close(model.layers[0].weight, baseline)
