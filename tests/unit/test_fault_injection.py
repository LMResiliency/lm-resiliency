"""Tests for the framework-aware fault injection evaluation kit."""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency import (
    FaultCampaign,
    FaultLocation,
    FaultMagnitude,
    FaultPersistence,
    FaultScope,
    FaultSpec,
    FaultTarget,
    FaultType,
    InjectionStatus,
    LocalizationResult,
    enable_fault_injection,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=True)
        with torch.no_grad():
            self.layer.weight.fill_(1.0)
            self.layer.bias.fill_(0.5)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.layer(value)


class FakeDeepSpeedEngine:
    def __init__(self, module: nn.Module) -> None:
        self.module = module
        self.optimizer = object()

    def step(self) -> None:
        pass

    def zero_optimization_stage(self) -> int:
        return 2


class FakeTorchTitanTrainer:
    def __init__(self, model_parts: list[nn.Module]) -> None:
        self.model_parts = model_parts
        self.optimizers = object()
        self.lr_schedulers = object()
        self.parallel_dims = object()
        self.checkpointer = object()

    def train(self) -> None:
        pass


class Wrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.module(value)


def _campaign(
    *,
    fault_type: FaultType = FaultType.SIGN_FLIP,
    location: FaultLocation = FaultLocation.WEIGHT,
    persistence: FaultPersistence = FaultPersistence.PERSISTENT,
    steps: tuple[int, ...] = (2,),
    probability: float = 1.0,
    call_index: int = 1,
    delay_ms: float = 0.0,
    model_part: int = 0,
) -> FaultCampaign:
    return FaultCampaign(
        name="unit-campaign",
        faults=(
            FaultSpec(
                fault_id="fault-a",
                fault_type=fault_type,
                target=FaultTarget(
                    rank=0,
                    module="layer",
                    location=location,
                    model_part=model_part,
                ),
                steps=steps,
                magnitude=FaultMagnitude.MEDIUM,
                scope=FaultScope.SINGLE,
                persistence=persistence,
                probability=probability,
                seed=17,
                call_index=call_index,
                delay_ms=delay_ms,
            ),
        ),
        metadata={"suite": "unit"},
    )


def test_campaign_json_round_trip(tmp_path) -> None:
    campaign = _campaign(steps=(2, 4), persistence=FaultPersistence.TRANSIENT)
    path = tmp_path / "campaign.json"

    campaign.to_json(path)
    restored = FaultCampaign.from_json(path)

    assert restored == campaign
    assert json.loads(path.read_text()) == campaign.to_dict()


def test_persistent_parameter_fault_records_ground_truth_and_restores() -> None:
    model = TinyModel()
    baseline = model.layer.weight.detach().clone()
    session = enable_fault_injection(model, _campaign(), rank=0)

    assert session.trigger(1) == ()
    records = session.trigger(2)

    assert len(records) == 1
    assert records[0].status is InjectionStatus.INJECTED
    assert records[0].injection_id == "fault-a@2"
    assert records[0].affected_elements == 1
    assert not torch.equal(model.layer.weight, baseline)
    assert session.trigger(2) == ()

    report = session.evaluate(
        [
            LocalizationResult(
                injection_id="fault-a@2",
                detected=True,
                failed_ranks=(0, 3),
                kind="sdc",
                component="layer",
                latency_ms=4.5,
            )
        ]
    )
    evaluation = report.evaluations[0]
    assert evaluation.localized is True
    assert evaluation.unexpected_ranks == (3,)
    assert evaluation.kind_matches is True
    assert evaluation.component_matches is True
    assert report.metadata == {"suite": "unit"}

    session.close()
    torch.testing.assert_close(model.layer.weight, baseline)


def test_transient_output_fault_targets_requested_call_only() -> None:
    model = TinyModel()
    campaign = _campaign(
        location=FaultLocation.OUTPUT,
        persistence=FaultPersistence.TRANSIENT,
        call_index=2,
    )
    session = enable_fault_injection(model, campaign, rank=0)
    value = torch.ones(2, 4)
    expected = model(value)

    record = session.trigger(2)[0]
    torch.testing.assert_close(model(value), expected)
    corrupted = model(value)
    assert not torch.equal(corrupted, expected)
    torch.testing.assert_close(model(value), expected)
    assert record.status is InjectionStatus.INJECTED

    session.close()


def test_transient_parameter_fault_restores_after_forward() -> None:
    model = TinyModel()
    campaign = _campaign(
        fault_type=FaultType.SCALE_UP,
        persistence=FaultPersistence.TRANSIENT,
    )
    session = enable_fault_injection(model, campaign, rank=0)
    value = torch.ones(1, 4)
    baseline_weight = model.layer.weight.detach().clone()
    expected = model(value)

    record = session.trigger(2)[0]
    corrupted = model(value)

    assert record.status is InjectionStatus.INJECTED
    assert not torch.equal(corrupted, expected)
    torch.testing.assert_close(model.layer.weight, baseline_weight)
    torch.testing.assert_close(model(value), expected)
    session.close()


def test_delay_fault_records_straggler_ground_truth() -> None:
    model = TinyModel()
    campaign = _campaign(
        fault_type=FaultType.DELAY,
        location=FaultLocation.OUTPUT,
        persistence=FaultPersistence.TRANSIENT,
        delay_ms=25.0,
    )
    session = enable_fault_injection(model, campaign, rank=0)

    record = session.trigger(2)[0]
    with patch("lm_resiliency.fault_injection.injector.time.sleep") as sleep:
        model(torch.ones(1, 4))

    sleep.assert_called_once_with(0.025)
    assert record.status is InjectionStatus.INJECTED
    assert record.expected_kind == "straggler"
    session.close()


def test_probability_zero_is_reproducibly_skipped() -> None:
    model = TinyModel()
    baseline = model.layer.weight.detach().clone()
    session = enable_fault_injection(
        model,
        _campaign(probability=0.0),
        rank=0,
    )

    record = session.trigger(2)[0]

    assert record.status is InjectionStatus.SKIPPED_PROBABILITY
    assert record.injection_succeeded is False
    torch.testing.assert_close(model.layer.weight, baseline)
    assert session.evaluate().evaluations[0].localized is False
    session.close()


@pytest.mark.parametrize(
    ("fault_type", "magnitude"),
    [
        (FaultType.SINGLE_BITFLIP, FaultMagnitude.NEAR_INVISIBLE),
        (FaultType.MULTI_BITFLIP, FaultMagnitude.MEDIUM),
        (FaultType.STUCK_AT_ZERO, FaultMagnitude.MEDIUM),
        (FaultType.STUCK_AT_ONE, FaultMagnitude.MEDIUM),
        (FaultType.SCALE_UP, FaultMagnitude.SUBTLE),
        (FaultType.SCALE_DOWN, FaultMagnitude.SUBTLE),
        (FaultType.GAUSSIAN_NOISE, FaultMagnitude.MEDIUM),
        (FaultType.SIGN_FLIP, FaultMagnitude.MEDIUM),
        (FaultType.SET_NAN, FaultMagnitude.MEDIUM),
        (FaultType.SET_INF, FaultMagnitude.MEDIUM),
    ],
)
def test_supported_numerical_faults_change_selected_values(
    fault_type: FaultType,
    magnitude: FaultMagnitude,
) -> None:
    model = TinyModel()
    with torch.no_grad():
        model.layer.weight.copy_(torch.linspace(-0.75, 0.75, 16).reshape(4, 4))
    baseline = model.layer.weight.detach().clone()
    campaign = FaultCampaign(
        name="numerical-matrix",
        faults=(
            FaultSpec(
                fault_id="matrix",
                fault_type=fault_type,
                target=FaultTarget(rank=0, module="layer"),
                steps=(1,),
                magnitude=magnitude,
                persistence=FaultPersistence.PERSISTENT,
            ),
        ),
    )
    session = enable_fault_injection(model, campaign, rank=0)

    record = session.trigger(1)[0]

    assert record.status is InjectionStatus.INJECTED
    assert not torch.equal(model.layer.weight, baseline)
    session.close()
    torch.testing.assert_close(model.layer.weight, baseline)


@pytest.mark.parametrize(
    ("framework", "target"),
    [
        ("pytorch", TinyModel()),
        ("deepspeed", FakeDeepSpeedEngine(TinyModel())),
        ("torchtitan", FakeTorchTitanTrainer([TinyModel()])),
        ("megatron", [Wrapper(Wrapper(TinyModel()))]),
    ],
)
def test_framework_model_resolution(framework: str, target: object) -> None:
    session = enable_fault_injection(
        target,
        _campaign(),
        framework=framework,
        rank=0,
    )

    record = session.trigger(2)[0]

    assert session.framework == framework
    assert record.status is InjectionStatus.INJECTED
    session.close()


def test_torchtitan_model_part_selection() -> None:
    first = TinyModel()
    second = TinyModel()
    trainer = FakeTorchTitanTrainer([first, second])
    first_baseline = first.layer.weight.detach().clone()
    second_baseline = second.layer.weight.detach().clone()
    session = enable_fault_injection(
        trainer,
        _campaign(model_part=1),
        rank=0,
    )

    session.trigger(2)

    torch.testing.assert_close(first.layer.weight, first_baseline)
    assert not torch.equal(second.layer.weight, second_baseline)
    session.close()
    torch.testing.assert_close(second.layer.weight, second_baseline)


def test_report_json_contains_ground_truth_and_localization(tmp_path) -> None:
    model = TinyModel()
    session = enable_fault_injection(model, _campaign(), rank=0)
    session.trigger(2)
    report = session.evaluate(
        [
            {
                "injection_id": "fault-a@2",
                "detected": True,
                "failed_ranks": [0],
                "kind": "sdc",
            }
        ]
    )
    path = tmp_path / "report.json"

    report.to_json(path)
    value = json.loads(path.read_text())

    assert value["campaign"] == "unit-campaign"
    assert value["manifest"] == _campaign().to_dict()
    assert value["injections"][0]["injection_succeeded"] is True
    assert value["injections"][0]["scope"] == "single"
    assert value["injections"][0]["persistence"] == "persistent"
    assert value["injections"][0]["seed"] == 17
    assert value["evaluations"][0]["localized"] is True
    session.close()


def test_invalid_fault_contracts_are_rejected() -> None:
    with pytest.raises(ValueError, match="delay faults must target"):
        _campaign(fault_type=FaultType.DELAY, delay_ms=1.0)
    with pytest.raises(ValueError, match="persistent faults support exactly one"):
        _campaign(steps=(1, 2))
    with pytest.raises(ValueError, match="call_index is only configurable"):
        _campaign(call_index=2)
    with pytest.raises(ValueError, match="duplicates"):
        FaultSpec(
            fault_id="duplicate-step",
            fault_type=FaultType.SIGN_FLIP,
            target=FaultTarget(rank=0),
            steps=(1, 1),
        )
    with pytest.raises(ValueError, match="overlap"):
        FaultCampaign(
            name="overlap",
            faults=(
                FaultSpec(
                    fault_id="first",
                    fault_type=FaultType.SIGN_FLIP,
                    target=FaultTarget(rank=0, module="layer"),
                ),
                FaultSpec(
                    fault_id="second",
                    fault_type=FaultType.SET_NAN,
                    target=FaultTarget(rank=0, module="layer"),
                ),
            ),
        )


def test_evaluation_rejects_unknown_or_inconsistent_results() -> None:
    session = enable_fault_injection(TinyModel(), _campaign(), rank=0)
    session.trigger(2)

    with pytest.raises(ValueError, match="unknown injections"):
        session.evaluate([LocalizationResult(injection_id="unknown@2", detected=False)])
    with pytest.raises(ValueError, match="cannot report failed ranks"):
        LocalizationResult(
            injection_id="fault-a@2",
            detected=False,
            failed_ranks=(0,),
        )

    session.close()


@pytest.mark.parametrize(
    "dtype",
    [torch.float16, torch.bfloat16, torch.float32, torch.float64],
)
def test_bit_flips_support_documented_floating_point_dtypes(dtype: torch.dtype) -> None:
    model = TinyModel().to(dtype=dtype)
    baseline = model.layer.weight.detach().clone()
    campaign = FaultCampaign(
        name="bitflip-dtypes",
        faults=(
            FaultSpec(
                fault_id="bitflip",
                fault_type=FaultType.SINGLE_BITFLIP,
                target=FaultTarget(rank=0, module="layer"),
                magnitude=FaultMagnitude.SUBTLE,
                persistence=FaultPersistence.PERSISTENT,
            ),
        ),
    )
    session = enable_fault_injection(model, campaign, rank=0)

    session.trigger(1)

    assert not torch.equal(model.layer.weight, baseline)
    session.close()
    torch.testing.assert_close(model.layer.weight, baseline)


def test_session_cancels_pending_fault_on_close() -> None:
    model = TinyModel()
    session = enable_fault_injection(
        model,
        _campaign(
            location=FaultLocation.OUTPUT,
            persistence=FaultPersistence.TRANSIENT,
        ),
        rank=0,
    )
    record = session.trigger(2)[0]

    session.close()

    assert record.status is InjectionStatus.CANCELLED


def test_non_target_rank_does_not_create_local_ground_truth() -> None:
    model = TinyModel()
    campaign = FaultCampaign(
        name="remote-rank",
        faults=(
            FaultSpec(
                fault_id="remote",
                fault_type=FaultType.SIGN_FLIP,
                target=FaultTarget(rank=3, module="layer"),
                steps=(1,),
            ),
        ),
    )
    session = enable_fault_injection(model, campaign, rank=0)

    assert session.trigger(1) == ()
    assert session.records == ()
    session.close()
