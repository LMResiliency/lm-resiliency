import copy
import hashlib
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
    _state_hold_iterations,
    _state_reset_iterations,
    _teardown,
    _validate_run,
    _validate_target_ranks,
)
from lm_resiliency import FaultCampaign

CAMPAIGN_PATH = Path("examples/fault_injection/campaign.json")
BASE_MANIFEST = {
    "incidents": [
        {
            "incident_id": "hidden-output-sdc",
            "trigger": {"at": [4], "probability": 1.0},
            "faults": [{"fault_id": "hidden-output"}],
        }
    ]
}


def _manifest_identity(manifest: dict) -> str:
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


MANIFEST_IDENTITY = _manifest_identity(BASE_MANIFEST)


def _injection_payload() -> dict:
    return {
        "campaign": "pytorch-production-loop-sdc",
        "manifest_identity": MANIFEST_IDENTITY,
        "completed_iterations": 4,
        "manifest": copy.deepcopy(BASE_MANIFEST),
        "injections": [
            {
                "occurrence_id": "hidden-output-sdc@4",
                "incident_id": "hidden-output-sdc",
                "fault_id": "hidden-output",
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


def _refresh_manifest_identity(
    injection: dict,
    localization: dict | None = None,
) -> None:
    identity = _manifest_identity(injection["manifest"])
    injection["manifest_identity"] = identity
    if localization is not None:
        localization["manifest_identity"] = identity


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


def test_evaluation_state_reset_freezes_snapshot_during_bounded_window() -> None:
    model = torch.nn.Linear(2, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    reset = EvaluationStateReset(
        SimpleNamespace(module=model),
        optimizer,
        {3},
        {1, 2},
    )
    clean = model.weight.detach().clone()

    def step() -> None:
        optimizer.zero_grad()
        model(torch.ones(1, 2)).sum().backward()
        optimizer.step()

    step()
    first_faulty = model.weight.detach().clone()
    step()
    second_faulty = model.weight.detach().clone()
    step()

    assert not torch.equal(first_faulty, clean)
    assert not torch.equal(second_faulty, first_faulty)
    torch.testing.assert_close(model.weight, clean)
    assert reset.restored_iterations == [3]
    reset.close()


def test_teardown_attempts_every_cleanup_and_preserves_active_error() -> None:
    events: list[str] = []

    class Cleanup:
        def __init__(self, name: str, *, error: Exception | None = None) -> None:
            self.name = name
            self.error = error

        def close(self) -> None:
            events.append(self.name)
            if self.error is not None:
                raise self.error

    active_error = RuntimeError("training failed")
    _teardown(
        Cleanup("faults", error=RuntimeError("fault cleanup failed")),
        Cleanup("state-reset", error=RuntimeError("state cleanup failed")),
        Cleanup("resiliency"),
        active_error=active_error,
        destroy_process_group=lambda: events.append("process-group"),
    )

    assert events == ["faults", "state-reset", "resiliency", "process-group"]
    assert getattr(active_error, "__notes__", []) == [
        "example teardown also failed: fault cleanup failed",
        "example teardown also failed: state cleanup failed",
    ]


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
        "fault_id": "rank-one-output",
        "execution_rank": 1,
        "target": {
            **injection["injections"][0]["target"],
            "rank": 1,
        },
    }
    injection["manifest"]["incidents"][0]["faults"].append({"fault_id": "rank-one-output"})
    injection["injections"].append(second)
    localization = _localization_payload()
    localization["reports"][0]["failed_ranks"] = [0, 1]
    _refresh_manifest_identity(injection, localization)

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


def test_comparison_scores_resource_targets_without_executor_rank_blame() -> None:
    injection = _injection_payload()
    injection["injections"][0].update(
        {
            "execution_rank": 0,
            "expected_kind": "process_failure",
            "target": {
                "surface": "resource",
                "resource": "node-5",
            },
        }
    )
    localization = _localization_payload()
    localization["reports"][0] = {
        "training_iteration": 4,
        "failed_ranks": [],
        "failed_resources": ["node-5"],
        "kind": "process_failure",
        "scope": "resource",
    }

    evaluation = compare_payloads(injection, localization)
    occurrence = evaluation["evaluations"][0]

    assert occurrence["localized"]
    assert occurrence["expected"]["ranks"] == []
    assert occurrence["expected"]["resources"] == ["node-5"]
    assert occurrence["observed"]["ranks"] == []
    assert occurrence["observed"]["resources"] == ["node-5"]
    assert occurrence["resource_match"]
    assert occurrence["kind_resource_match"]
    assert evaluation["summary"]["passed"]


def test_comparison_correlates_failure_kind_with_rank() -> None:
    injection = _injection_payload()
    injection["manifest"]["incidents"][0]["faults"].append({"fault_id": "rank-1-delay"})
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
    _refresh_manifest_identity(injection, localization)

    evaluation = compare_payloads(injection, localization)
    occurrence = evaluation["evaluations"][0]

    assert occurrence["rank_match"]
    assert occurrence["kind_match"]
    assert not occurrence["kind_rank_match"]
    assert occurrence["detected_action_count"] == 0
    assert not occurrence["localized"]
    assert not evaluation["summary"]["passed"]


def test_comparison_requires_every_manifest_action_record() -> None:
    injection = _injection_payload()
    injection["manifest"]["incidents"][0]["faults"].append({"fault_id": "missing-rank-one-action"})
    localization = _localization_payload()
    _refresh_manifest_identity(injection, localization)

    evaluation = compare_payloads(injection, localization)

    occurrence = evaluation["evaluations"][0]
    assert occurrence["action_count"] == 1
    assert occurrence["expected_action_count"] == 2
    assert not occurrence["injection_succeeded"]
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
    injection["completed_iterations"] = 5
    injection["manifest"]["incidents"].append(
        {
            "incident_id": "skipped",
            "trigger": {"at": [5], "probability": 0.0},
            "faults": [{"fault_id": "skipped-output"}],
        }
    )
    injection["injections"].append(
        {
            **injection["injections"][0],
            "occurrence_id": "skipped@5",
            "incident_id": "skipped",
            "fault_id": "skipped-output",
            "iteration": 5,
            "status": "skipped_probability",
            "injection_succeeded": False,
        }
    )
    localization = _localization_payload()
    _refresh_manifest_identity(injection, localization)

    evaluation = compare_payloads(injection, localization)

    assert evaluation["summary"]["passed"]
    assert len(evaluation["evaluations"]) == 1


def test_comparison_rejects_a_wholly_missing_scheduled_occurrence() -> None:
    injection = _injection_payload()
    injection["completed_iterations"] = 5
    injection["manifest"]["incidents"].append(
        {
            "incident_id": "missing",
            "trigger": {"at": [5], "probability": 1.0},
            "faults": [{"fault_id": "missing-output"}],
        }
    )
    localization = _localization_payload()
    _refresh_manifest_identity(injection, localization)

    with pytest.raises(ValueError, match="occurrence coverage mismatch"):
        compare_payloads(injection, localization)


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


def test_example_clean_boundary_includes_bounded_incident_lifetime() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "bounded-lifetime",
            "incidents": [
                {
                    "id": "long-window",
                    "trigger": {"at": [5]},
                    "lifetime": {"iterations": 10},
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

    assert _last_scheduled_iteration(campaign) == 14
    with pytest.raises(ValueError, match="clean post-fault"):
        _validate_run(campaign, steps=14)
    _validate_run(campaign, steps=15)


def test_state_reset_waits_for_bounded_state_fault_expiration() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "bounded-state-reset",
            "incidents": [
                {
                    "id": "weight-window",
                    "trigger": {"at": [5]},
                    "lifetime": {"iterations": 3},
                    "faults": [
                        {
                            "id": "weight",
                            "type": "tensor_corruption",
                            "target": {
                                "rank": 0,
                                "module_path": "layers.0",
                                "surface": "weight",
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
    hold_iterations = _state_hold_iterations(campaign)

    assert 5 not in reset_iterations
    assert 6 not in reset_iterations
    assert 7 in reset_iterations
    assert 5 in hold_iterations
    assert 6 in hold_iterations
    assert 7 not in hold_iterations


def test_state_hold_schedule_remains_lazy_for_long_exact_lifetime() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "long-state-window",
            "incidents": [
                {
                    "id": "weight-window",
                    "trigger": {"at": [5]},
                    "lifetime": {"iterations": 1_000_000_000},
                    "faults": [
                        {
                            "id": "weight",
                            "type": "tensor_corruption",
                            "target": {
                                "rank": 0,
                                "module_path": "layers.0",
                                "surface": "weight",
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

    hold_iterations = _state_hold_iterations(campaign)

    assert 5 in hold_iterations
    assert 1_000_000_003 in hold_iterations
    assert 1_000_000_004 not in hold_iterations


def test_example_rejects_campaign_end_state_reset() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "campaign-end-state",
            "incidents": [
                {
                    "id": "weight-window",
                    "trigger": {"at": [5]},
                    "lifetime": {"until": "campaign_end"},
                    "faults": [
                        {
                            "id": "weight",
                            "type": "tensor_corruption",
                            "target": {
                                "rank": 0,
                                "module_path": "layers.0",
                                "surface": "weight",
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

    with pytest.raises(ValueError, match="does not support campaign_end"):
        _state_reset_iterations(campaign)


def test_example_rejects_multi_call_gradient_affecting_reset() -> None:
    campaign = FaultCampaign.from_dict(
        {
            "schema_version": 1,
            "name": "multi-call-reset",
            "incidents": [
                {
                    "id": "output-window",
                    "trigger": {"at": [5]},
                    "lifetime": {"matching_calls": 2},
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

    with pytest.raises(ValueError, match="supports matching_calls=1"):
        _state_reset_iterations(campaign)


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


def test_comparison_rejects_tampered_embedded_manifest() -> None:
    injection = _injection_payload()
    injection["manifest"]["incidents"].clear()

    with pytest.raises(ValueError, match="does not match its manifest_identity"):
        compare_payloads(injection, _localization_payload())
