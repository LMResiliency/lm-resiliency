"""Compare injected-fault ground truth with normalized SCOUT JSON reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def compare_payloads(
    injection: Mapping[str, Any],
    localization: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a JSON-ready evaluation of injection and localization artifacts."""
    if injection.get("campaign") != localization.get("campaign"):
        raise ValueError("injection and localization artifacts belong to different campaigns")
    injection_identity = _required_manifest_identity(injection, "injection")
    localization_identity = _required_manifest_identity(localization, "localization")
    if injection_identity != localization_identity:
        raise ValueError(
            "injection and localization artifacts belong to different campaign manifests"
        )
    records = [
        dict(record)
        for record in injection.get("injections", ())
        if record.get("status") != "skipped_probability"
    ]
    reports = [dict(report) for report in localization.get("reports", ())]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["occurrence_id"]), []).append(record)

    evaluations = [
        _evaluate_occurrence(occurrence_id, occurrence_records, reports)
        for occurrence_id, occurrence_records in grouped.items()
    ]
    localized = sum(bool(evaluation["localized"]) for evaluation in evaluations)
    detected_actions = sum(int(evaluation["detected_action_count"]) for evaluation in evaluations)
    localized_actions = sum(
        int(evaluation["action_count"]) for evaluation in evaluations if evaluation["localized"]
    )
    summary = {
        "injected_occurrences": sum(
            bool(evaluation["injection_succeeded"]) for evaluation in evaluations
        ),
        "detected_occurrences": sum(bool(item["detected"]) for item in evaluations),
        "localized_occurrences": localized,
        "injected_actions": sum(bool(record.get("injection_succeeded")) for record in records),
        "detected_actions": detected_actions,
        "localized_actions": localized_actions,
        "passed": bool(evaluations) and localized == len(evaluations),
    }
    return {
        "schema_version": 1,
        "campaign": injection.get("campaign"),
        "manifest_identity": injection_identity,
        "summary": summary,
        "evaluations": evaluations,
    }


def compare_artifacts(
    injection_path: str | Path,
    localization_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load two JSON artifacts, compare them, and optionally write the result."""
    injection = _load_object(injection_path)
    localization = _load_object(localization_path)
    evaluation = compare_payloads(injection, localization)
    if output_path is not None:
        _write_json(Path(output_path), evaluation)
    return evaluation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()
    evaluation = compare_artifacts(
        args.artifact_dir / "injection.json",
        args.artifact_dir / "localization.json",
        args.artifact_dir / "evaluation.json",
    )
    print(json.dumps(evaluation, indent=2, sort_keys=True))
    if not evaluation["summary"]["passed"]:
        raise SystemExit(1)


def _evaluate_occurrence(
    occurrence_id: str,
    records: Sequence[Mapping[str, Any]],
    reports: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    injection_succeeded = bool(records) and all(
        bool(record.get("injection_succeeded")) for record in records
    )
    iterations = {int(record["iteration"]) for record in records}
    if len(iterations) != 1:
        raise ValueError(f"occurrence {occurrence_id!r} spans multiple training iterations")
    iteration = iterations.pop()
    expected_ranks = sorted({int(record["execution_rank"]) for record in records})
    expected_kinds = sorted({str(record["expected_kind"]) for record in records})
    expected_ranks_by_kind = {
        kind: sorted(
            {
                int(record["execution_rank"])
                for record in records
                if str(record["expected_kind"]) == kind
            }
        )
        for kind in expected_kinds
    }
    expected_layers = sorted(
        {
            int(target["index"])
            for record in records
            if (target := dict(record.get("target", {}))).get("component")
            in {"transformer_block", "transformer_layer", "layer"}
            and target.get("index") is not None
        }
    )
    expected_source_prefixes = sorted(
        {prefix for record in records if (prefix := _expected_source_prefix(record)) is not None}
    )
    at_iteration = [
        dict(report) for report in reports if int(report.get("training_iteration", -1)) == iteration
    ]
    matching = [report for report in at_iteration if str(report.get("kind")) in expected_kinds]
    observed_kinds = sorted({str(report.get("kind")) for report in at_iteration})
    observed_ranks = sorted(
        {int(rank) for report in matching for rank in report.get("failed_ranks", ())}
    )
    observed_ranks_by_kind = {
        kind: sorted(
            {
                int(rank)
                for report in at_iteration
                if str(report.get("kind")) == kind
                for rank in report.get("failed_ranks", ())
            }
        )
        for kind in expected_kinds
    }
    reported_layer_ids = {
        int(report["layer_id"]) for report in matching if report.get("layer_id") is not None
    }
    observed_layers = sorted(layer_id for layer_id in reported_layer_ids if layer_id >= 0)
    aggregate_layer_report = -1 in reported_layer_ids
    observed_sources = sorted(
        {str(source) for report in matching for source in report.get("sources", ())}
    )
    kind_match = set(expected_kinds).issubset(observed_kinds)
    rank_match = observed_ranks == expected_ranks
    kind_rank_match = observed_ranks_by_kind == expected_ranks_by_kind
    detected_action_count = sum(
        int(record["execution_rank"])
        in observed_ranks_by_kind.get(str(record["expected_kind"]), ())
        for record in records
    )
    source_match = not expected_source_prefixes or all(
        any(source.startswith(prefix) for source in observed_sources)
        for prefix in expected_source_prefixes
    )
    if not expected_layers:
        layer_match = True
        layer_evidence = "not_required"
    elif set(expected_layers).issubset(observed_layers):
        layer_match = True
        layer_evidence = "layer_id"
    elif aggregate_layer_report and not expected_source_prefixes:
        layer_match = True
        layer_evidence = "aggregate_replay_catalog"
    elif aggregate_layer_report and source_match:
        layer_match = True
        layer_evidence = "aggregate_component_source"
    else:
        layer_match = False
        layer_evidence = "missing"
    detected = bool(matching)
    localized = (
        injection_succeeded
        and detected
        and kind_match
        and rank_match
        and kind_rank_match
        and layer_match
        and source_match
    )
    return {
        "occurrence_id": occurrence_id,
        "iteration": iteration,
        "action_count": len(records),
        "detected_action_count": detected_action_count,
        "injection_succeeded": injection_succeeded,
        "detected": detected,
        "localized": localized,
        "expected": {
            "ranks": expected_ranks,
            "kinds": expected_kinds,
            "ranks_by_kind": expected_ranks_by_kind,
            "layers": expected_layers,
            "source_prefixes": expected_source_prefixes,
        },
        "observed": {
            "ranks": observed_ranks,
            "kinds": observed_kinds,
            "ranks_by_kind": observed_ranks_by_kind,
            "layers": observed_layers,
            "aggregate_layer_report": aggregate_layer_report,
            "sources": observed_sources,
        },
        "kind_match": kind_match,
        "rank_match": rank_match,
        "kind_rank_match": kind_rank_match,
        "layer_match": layer_match,
        "layer_evidence": layer_evidence,
        "source_match": source_match,
        "matched_reports": matching,
    }


def _expected_source_prefix(record: Mapping[str, Any]) -> str | None:
    if record.get("expected_kind") != "sdc":
        return None
    target = dict(record.get("target", {}))
    component = str(target.get("component", "")).lower().replace("-", "_")
    if component in {"transformer_block", "transformer_layer", "layer"}:
        return "hidden."
    if component in {"embedding", "token_embedding"}:
        return "embedding."
    if component in {"output", "lm_head"}:
        return "output."
    module_path = str(target.get("module_path", ""))
    if ".layers." in f".{module_path}.":
        return "hidden."
    return None


def _load_object(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _required_manifest_identity(payload: Mapping[str, Any], label: str) -> str:
    value = payload.get("manifest_identity")
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} artifact requires a non-empty manifest_identity")
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
