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
MANIFEST_IDENTITY = "0123456789abcdef"


def _injection_payload() -> dict:
    return {
        "campaign": "pytorch-production-loop-sdc",
        "manifest_identity": MANIFEST_IDENTITY,
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
        "manifest_identity": MANIFEST_IDENTITY,
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
    assert _state_reset_iterations(campaign) == (
        set(range(4, 28)) | {31, 32, 34, 36, 38, 39} | set(range(41, 68, 2))
    )


def test_range_schedule_helpers_remain_lazy_for_large_campaigns() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "large-range",
            "incidents": [
                {
                    "id": "range",
                    "trigger": {
                        "range": {
                            "start": 1,
                            "end": 1_000_000_000,
                            "every": 3,
                        }
                    },
                    "lifetime": {"matching_calls": 1},
                    "faults": [
                        {
                            "id": "output",
                            "type": "tensor_corruption",
                            "target": {
                                "rank": 0,
                                "module_path": "layers.0",
                                "surface": "output",
                            },
                            "parameters": {
                                "operation": "sign_flip",
                                "scope": "single",
                            },
                        }
                    ],
                }
            ],
        }
    )

    reset_iterations = _state_reset_iterations(campaign)

    assert _last_scheduled_iteration(campaign) == 1_000_000_000
    assert 1 in reset_iterations
    assert 4 in reset_iterations
    assert 2 not in reset_iterations
    assert 1_000_000_000 in reset_iterations


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

    assert evaluation["manifest_identity"] == MANIFEST_IDENTITY
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


def test_comparison_correlates_failure_kind_with_rank() -> None:
    injection = _injection_payload()
    injection["injections"].append(
        {
            **injection["injections"][0],
            "fault_id": "rank-1-delay",
            "execution_rank": 1,
            "expected_kind": "straggler",
            "target": {
                **injection["injections"][0]["target"],
                "rank": 1,
                "surface": "compute",
            },
        }
    )
    localization = _localization_payload(failed_rank=1)
    localization["reports"].append(
        {
            "training_iteration": 4,
            "failed_ranks": [0],
            "kind": "straggler",
            "scope": "rank",
        }
    )

    evaluation = compare_payloads(injection, localization)
    occurrence = evaluation["evaluations"][0]

    assert occurrence["rank_match"]
    assert occurrence["kind_match"]
    assert not occurrence["kind_rank_match"]
    assert occurrence["detected_action_count"] == 0
    assert not occurrence["localized"]
    assert not evaluation["summary"]["passed"]


@pytest.mark.parametrize("status", ["pending", "failed", "cancelled"])
def test_comparison_fails_selected_unsuccessful_injections(status: str) -> None:
    injection = _injection_payload()
    injection["injections"][0].update(
        {
            "status": status,
            "injection_succeeded": False,
        }
    )

    evaluation = compare_payloads(injection, _localization_payload())

    assert not evaluation["summary"]["passed"]
    assert evaluation["summary"]["injected_occurrences"] == 0
    assert evaluation["summary"]["injected_actions"] == 0
    assert not evaluation["evaluations"][0]["injection_succeeded"]
    assert not evaluation["evaluations"][0]["localized"]


def test_comparison_ignores_explicit_probability_skips() -> None:
    injection = _injection_payload()
    injection["injections"].append(
        {
            **injection["injections"][0],
            "occurrence_id": "skipped@5",
            "iteration": 5,
            "status": "skipped_probability",
            "injection_succeeded": False,
        }
    )

    evaluation = compare_payloads(injection, _localization_payload())

    assert evaluation["summary"]["passed"]
    assert len(evaluation["evaluations"]) == 1


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


def test_comparison_rejects_extra_positive_layer_attribution() -> None:
    injection = _injection_payload()
    injection["injections"][0]["target"]["index"] = 0
    localization = _localization_payload()
    localization["reports"].extend(
        [
            {
                **localization["reports"][0],
                "layer_id": 0,
            },
            {
                **localization["reports"][0],
                "layer_id": 1,
            },
        ]
    )

    evaluation = compare_payloads(injection, localization)

    assert not evaluation["evaluations"][0]["layer_match"]
    assert not evaluation["summary"]["passed"]


def test_wrong_kind_at_the_injected_iteration_is_not_counted_as_detection() -> None:
    injection = _injection_payload()
    localization = _localization_payload()
    localization["reports"][0]["kind"] = "straggler"

    evaluation = compare_payloads(injection, localization)

    assert not evaluation["evaluations"][0]["detected"]
    assert evaluation["summary"]["detected_occurrences"] == 0
    assert evaluation["summary"]["detected_actions"] == 0


@pytest.mark.parametrize("training_iteration", [4.5, True, "4"])
def test_comparison_rejects_coerced_localization_iteration(
    training_iteration: object,
) -> None:
    localization = _localization_payload()
    localization["reports"][0]["training_iteration"] = training_iteration

    with pytest.raises(TypeError, match="training_iteration must be an integer"):
        compare_payloads(_injection_payload(), localization)


@pytest.mark.parametrize("failed_rank", [True, 0.5, "0"])
def test_comparison_rejects_coerced_localization_rank(failed_rank: object) -> None:
    localization = _localization_payload()
    localization["reports"][0]["failed_ranks"] = [failed_rank]

    with pytest.raises(TypeError, match="failed rank must be an integer"):
        compare_payloads(_injection_payload(), localization)


@pytest.mark.parametrize("layer_id", [True, 0.9, "0"])
def test_comparison_rejects_coerced_localization_layer_id(layer_id: object) -> None:
    localization = _localization_payload()
    localization["reports"][0]["layer_id"] = layer_id

    with pytest.raises(TypeError, match="layer_id must be an integer"):
        compare_payloads(_injection_payload(), localization)


def test_comparison_rejects_ambiguous_same_iteration_occurrences() -> None:
    injection = _injection_payload()
    injection["injections"].append(
        {
            **injection["injections"][0],
            "occurrence_id": "another-sdc@4",
            "fault_id": "another-sdc",
        }
    )

    with pytest.raises(ValueError, match="cannot be correlated uniquely"):
        compare_payloads(injection, _localization_payload())


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


def test_comparison_requires_matching_manifest_identity() -> None:
    localization = _localization_payload()
    localization["manifest_identity"] = "different-manifest"

    with pytest.raises(ValueError, match="different campaign manifests"):
        compare_payloads(_injection_payload(), localization)

    del localization["manifest_identity"]
    with pytest.raises(ValueError, match="requires a non-empty manifest_identity"):
        compare_payloads(_injection_payload(), localization)
