import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from examples.fault_injection.compare import compare_artifacts, compare_payloads
from examples.fault_injection.generate_campaign import build_campaign
from examples.fault_injection.pytorch import (
    EvaluationStateReset,
    _last_scheduled_iteration,
    _state_reset_iterations,
    _validate_run,
    _validate_target_ranks,
)
from lm_resiliency import FaultCampaign

CAMPAIGN_PATH = Path("examples/fault_injection/campaign.json")


def _injection_payload() -> dict:
    return {
        "campaign": "pytorch-production-loop-sdc",
        "injections": [
            {
                "occurrence_id": "hidden-output-sdc@4",
                "iteration": 4,
                "execution_rank": 0,
                "expected_kind": "sdc",
                "injection_succeeded": True,
                "target": {
                    "rank": 0,
                    "component": "transformer_block",
                    "index": 0,
                    "surface": "output",
                },
            }
        ],
    }


def _localization_payload(*, failed_rank: int = 0, layer_id: int = -1) -> dict:
    return {
        "campaign": "pytorch-production-loop-sdc",
        "reports": [
            {
                "training_iteration": 4,
                "failed_ranks": [failed_rank],
                "kind": "sdc",
                "scope": "rank",
                "layer_id": layer_id,
                "sources": ["hidden.output"],
            }
        ],
    }


def test_checked_in_campaign_targets_the_scout_replay_layer() -> None:
    campaign = FaultCampaign.from_json(CAMPAIGN_PATH)

    _validate_run(campaign, steps=68)
    _validate_target_ranks(campaign, world_size=8)
    faults = [fault for incident in campaign.incidents for fault in incident.faults]
    candidate_occurrences = sum(len(incident.trigger.at) for incident in campaign.incidents)
    fault_records = sum(
        len(incident.trigger.at) * len(incident.faults) for incident in campaign.incidents
    )
    type_surface_pairs = {(fault.type.value, fault.target.surface.value) for fault in faults}

    assert campaign.to_dict() == build_campaign().to_dict()
    assert len(campaign.incidents) == 46
    assert candidate_occurrences == 48
    assert fault_records == 53
    assert _last_scheduled_iteration(campaign) == 67
    assert {fault.target.execution_rank for fault in faults} == set(range(8))
    assert type_surface_pairs == {
        ("tensor_corruption", surface)
        for surface in ("input", "output", "weight", "bias", "gradient", "optimizer_state")
    } | {
        (failure_type, surface)
        for failure_type in ("stale_state", "duplicate")
        for surface in ("input", "output", "weight", "bias", "gradient", "optimizer_state")
    } | {
        (failure_type, surface)
        for failure_type in ("drop", "reorder")
        for surface in ("input", "output", "gradient")
    } | {("delay", surface) for surface in ("input", "output", "compute")}
    assert all(fault.target.component == "transformer_block" for fault in faults)
    assert all(fault.target.index == 0 for fault in faults)
    assert {incident.temporal_behavior for incident in campaign.incidents} == {
        "transient",
        "intermittent",
        "permanent",
    }
    assert _state_reset_iterations(campaign) == set(range(41, 68, 2))


def test_evaluation_state_reset_restores_the_last_clean_optimizer_boundary() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reset = EvaluationStateReset(
        SimpleNamespace(module=model),
        optimizer,
        {2},
    )

    def step() -> None:
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    step()
    clean = model.weight.detach().clone()
    step()

    torch.testing.assert_close(model.weight, clean)
    assert reset.restored_iterations == [2]
    reset.close()


def test_comparison_accepts_matching_scout_localization() -> None:
    evaluation = compare_payloads(_injection_payload(), _localization_payload())

    assert evaluation["summary"] == {
        "injected_occurrences": 1,
        "detected_occurrences": 1,
        "localized_occurrences": 1,
        "injected_actions": 1,
        "detected_actions": 1,
        "localized_actions": 1,
        "passed": True,
    }
    assert evaluation["evaluations"][0]["action_count"] == 1
    assert evaluation["evaluations"][0]["rank_match"]
    assert evaluation["evaluations"][0]["kind_match"]
    assert evaluation["evaluations"][0]["layer_match"]
    assert evaluation["evaluations"][0]["layer_evidence"] == "aggregate_component_source"
    assert evaluation["evaluations"][0]["source_match"]


def test_comparison_rejects_wrong_rank(tmp_path: Path) -> None:
    injection_path = tmp_path / "injection.json"
    localization_path = tmp_path / "localization.json"
    evaluation_path = tmp_path / "evaluation.json"
    injection_path.write_text(json.dumps(_injection_payload()), encoding="utf-8")
    localization_path.write_text(
        json.dumps(_localization_payload(failed_rank=1)),
        encoding="utf-8",
    )

    evaluation = compare_artifacts(
        injection_path,
        localization_path,
        evaluation_path,
    )

    assert not evaluation["summary"]["passed"]
    assert not evaluation["evaluations"][0]["rank_match"]
    assert json.loads(evaluation_path.read_text()) == evaluation


def test_comparison_counts_correlated_rank_local_actions() -> None:
    injection = _injection_payload()
    second = {
        **injection["injections"][0],
        "execution_rank": 1,
        "target": {
            **injection["injections"][0]["target"],
            "rank": 1,
        },
    }
    injection["injections"].append(second)
    localization = _localization_payload()
    localization["reports"][0]["failed_ranks"] = [0, 1]

    evaluation = compare_payloads(injection, localization)

    assert evaluation["summary"] == {
        "injected_occurrences": 1,
        "detected_occurrences": 1,
        "localized_occurrences": 1,
        "injected_actions": 2,
        "detected_actions": 2,
        "localized_actions": 2,
        "passed": True,
    }
    assert evaluation["evaluations"][0]["action_count"] == 2


@pytest.mark.parametrize(
    ("changes", "mismatch"),
    [
        ({"kind": "straggler"}, "kind_match"),
        ({"sources": ["output.output"]}, "source_match"),
        ({"layer_id": 1}, "layer_match"),
    ],
)
def test_comparison_rejects_wrong_failure_evidence(
    changes: dict,
    mismatch: str,
) -> None:
    localization = _localization_payload()
    localization["reports"][0].update(changes)

    evaluation = compare_payloads(_injection_payload(), localization)

    assert not evaluation["summary"]["passed"]
    assert not evaluation["evaluations"][0][mismatch]


def test_example_rejects_missing_post_fault_iteration_and_rank() -> None:
    campaign = FaultCampaign.from_json(CAMPAIGN_PATH)

    with pytest.raises(ValueError, match="clean post-fault"):
        _validate_run(campaign, steps=67)

    with pytest.raises(ValueError, match="unavailable global ranks"):
        _validate_target_ranks(campaign, world_size=7)


def test_comparison_rejects_mismatched_campaigns() -> None:
    localization = _localization_payload()
    localization["campaign"] = "another-campaign"

    with pytest.raises(ValueError, match="different campaigns"):
        compare_payloads(_injection_payload(), localization)
