"""Compare injected-fault ground truth with normalized SCOUT JSON reports."""

from __future__ import annotations

import argparse
import hashlib
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
    _validate_embedded_manifest_identity(injection, injection_identity)
    all_records = [dict(record) for record in injection.get("injections", ())]
    _validate_scheduled_occurrence_coverage(injection, all_records)
    records = [
        dict(record) for record in all_records if record.get("status") != "skipped_probability"
    ]
    expected_fault_ids = _manifest_fault_ids(injection)
    reports = [dict(report) for report in localization.get("reports", ())]
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(str(record["occurrence_id"]), []).append(record)
    _reject_ambiguous_occurrence_iterations(grouped)

    evaluations = [
        _evaluate_occurrence(
            occurrence_id,
            occurrence_records,
            reports,
            expected_fault_ids,
        )
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
    manifest_fault_ids: Mapping[str, frozenset[str]],
) -> dict[str, Any]:
    incident_ids = {str(record.get("incident_id", "")) for record in records}
    if len(incident_ids) != 1 or "" in incident_ids:
        raise ValueError(f"occurrence {occurrence_id!r} requires one non-empty incident_id")
    incident_id = incident_ids.pop()
    expected_fault_ids = manifest_fault_ids.get(incident_id)
    if expected_fault_ids is None:
        raise ValueError(
            f"occurrence {occurrence_id!r} references unknown incident {incident_id!r}"
        )
    recorded_fault_ids = [str(record.get("fault_id", "")) for record in records]
    complete_action_set = (
        len(recorded_fault_ids) == len(expected_fault_ids)
        and set(recorded_fault_ids) == expected_fault_ids
    )
    injection_succeeded = (
        complete_action_set
        and bool(records)
        and all(bool(record.get("injection_succeeded")) for record in records)
    )
    iterations = {int(record["iteration"]) for record in records}
    if len(iterations) != 1:
        raise ValueError(f"occurrence {occurrence_id!r} spans multiple training iterations")
    iteration = iterations.pop()
    expected_ranks = sorted(
        {
            expected_rank
            for record in records
            if (expected_rank := _expected_rank(record)) is not None
        }
    )
    expected_resources = sorted(
        {resource for record in records if (resource := _expected_resource(record)) is not None}
    )
    expected_kinds = sorted({str(record["expected_kind"]) for record in records})
    expected_ranks_by_kind = {
        kind: sorted(
            {
                expected_rank
                for record in records
                if str(record["expected_kind"]) == kind
                and (expected_rank := _expected_rank(record)) is not None
            }
        )
        for kind in expected_kinds
    }
    expected_resources_by_kind = {
        kind: sorted(
            {
                resource
                for record in records
                if str(record["expected_kind"]) == kind
                and (resource := _expected_resource(record)) is not None
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
        dict(report)
        for report in reports
        if _required_integer(
            report.get("training_iteration"),
            "localization training_iteration",
        )
        == iteration
    ]
    matching = [report for report in at_iteration if str(report.get("kind")) in expected_kinds]
    observed_kinds = sorted({str(report.get("kind")) for report in at_iteration})
    observed_ranks = sorted(
        {
            _required_integer(rank, "localization failed rank")
            for report in matching
            for rank in report.get("failed_ranks", ())
        }
    )
    observed_ranks_by_kind = {
        kind: sorted(
            {
                _required_integer(rank, "localization failed rank")
                for report in at_iteration
                if str(report.get("kind")) == kind
                for rank in report.get("failed_ranks", ())
            }
        )
        for kind in expected_kinds
    }
    observed_resources = sorted(
        {str(resource) for report in matching for resource in report.get("failed_resources", ())}
    )
    observed_resources_by_kind = {
        kind: sorted(
            {
                str(resource)
                for report in at_iteration
                if str(report.get("kind")) == kind
                for resource in report.get("failed_resources", ())
            }
        )
        for kind in expected_kinds
    }
    reported_layer_ids = {
        _required_integer(report["layer_id"], "localization layer_id")
        for report in matching
        if report.get("layer_id") is not None
    }
    observed_layers = sorted(layer_id for layer_id in reported_layer_ids if layer_id >= 0)
    aggregate_layer_report = -1 in reported_layer_ids
    observed_sources = sorted(
        {str(source) for report in matching for source in report.get("sources", ())}
    )
    kind_match = set(expected_kinds).issubset(observed_kinds)
    rank_match = observed_ranks == expected_ranks
    resource_match = observed_resources == expected_resources
    kind_rank_match = observed_ranks_by_kind == expected_ranks_by_kind
    kind_resource_match = observed_resources_by_kind == expected_resources_by_kind
    detected_action_count = sum(
        (
            (expected_rank := _expected_rank(record)) is None
            or expected_rank in observed_ranks_by_kind.get(str(record["expected_kind"]), ())
        )
        and (
            (expected_resource := _expected_resource(record)) is None
            or expected_resource in observed_resources_by_kind.get(str(record["expected_kind"]), ())
        )
        for record in records
    )
    source_match = not expected_source_prefixes or all(
        any(source.startswith(prefix) for source in observed_sources)
        for prefix in expected_source_prefixes
    )
    if not expected_layers:
        layer_match = True
        layer_evidence = "not_required"
    elif set(expected_layers) == set(observed_layers):
        layer_match = True
        layer_evidence = "layer_id"
    elif not observed_layers and aggregate_layer_report and not expected_source_prefixes:
        layer_match = True
        layer_evidence = "aggregate_replay_catalog"
    elif not observed_layers and aggregate_layer_report and source_match:
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
        and resource_match
        and kind_rank_match
        and kind_resource_match
        and layer_match
        and source_match
    )
    return {
        "occurrence_id": occurrence_id,
        "iteration": iteration,
        "action_count": len(records),
        "expected_action_count": len(expected_fault_ids),
        "detected_action_count": detected_action_count,
        "injection_succeeded": injection_succeeded,
        "detected": detected,
        "localized": localized,
        "expected": {
            "ranks": expected_ranks,
            "resources": expected_resources,
            "kinds": expected_kinds,
            "ranks_by_kind": expected_ranks_by_kind,
            "resources_by_kind": expected_resources_by_kind,
            "layers": expected_layers,
            "source_prefixes": expected_source_prefixes,
        },
        "observed": {
            "ranks": observed_ranks,
            "resources": observed_resources,
            "kinds": observed_kinds,
            "ranks_by_kind": observed_ranks_by_kind,
            "resources_by_kind": observed_resources_by_kind,
            "layers": observed_layers,
            "aggregate_layer_report": aggregate_layer_report,
            "sources": observed_sources,
        },
        "kind_match": kind_match,
        "rank_match": rank_match,
        "resource_match": resource_match,
        "kind_rank_match": kind_rank_match,
        "kind_resource_match": kind_resource_match,
        "layer_match": layer_match,
        "layer_evidence": layer_evidence,
        "source_match": source_match,
        "matched_reports": matching,
    }


def _expected_rank(record: Mapping[str, Any]) -> int | None:
    target = record.get("target", {})
    if not isinstance(target, Mapping):
        raise TypeError("injection target must be an object")
    rank = target.get("rank")
    if rank is not None:
        return _required_integer(rank, "injection target rank")
    if target.get("resource") is not None:
        return None
    return _required_integer(record.get("execution_rank"), "injection execution_rank")


def _expected_resource(record: Mapping[str, Any]) -> str | None:
    target = record.get("target", {})
    if not isinstance(target, Mapping):
        raise TypeError("injection target must be an object")
    resource = target.get("resource")
    if resource is None:
        return None
    if not isinstance(resource, str):
        raise TypeError("injection target resource must be a string")
    if not resource:
        raise ValueError("injection target resource must be non-empty")
    return resource


def _manifest_fault_ids(injection: Mapping[str, Any]) -> dict[str, frozenset[str]]:
    manifest = injection.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("injection artifact requires an embedded manifest")
    incidents = manifest.get("incidents")
    if isinstance(incidents, (str, bytes)) or not isinstance(incidents, Sequence):
        raise TypeError("injection manifest incidents must be an array")
    expected: dict[str, frozenset[str]] = {}
    for incident in incidents:
        if not isinstance(incident, Mapping):
            raise TypeError("injection manifest incidents must contain objects")
        incident_id = str(incident.get("incident_id", incident.get("id", "")))
        faults = incident.get("faults")
        if not incident_id:
            raise ValueError("injection manifest incident_id must be non-empty")
        if isinstance(faults, (str, bytes)) or not isinstance(faults, Sequence):
            raise TypeError("injection manifest faults must be an array")
        fault_ids = [
            str(fault.get("fault_id", fault.get("id", "")))
            for fault in faults
            if isinstance(fault, Mapping)
        ]
        if len(fault_ids) != len(faults) or not fault_ids or any(not item for item in fault_ids):
            raise ValueError("injection manifest faults require non-empty fault_id values")
        if len(set(fault_ids)) != len(fault_ids):
            raise ValueError("injection manifest fault_id values must be unique per incident")
        expected[incident_id] = frozenset(fault_ids)
    return expected


def _validate_embedded_manifest_identity(
    injection: Mapping[str, Any],
    expected_identity: str,
) -> None:
    manifest = injection.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("injection artifact requires an embedded manifest")
    encoded = json.dumps(
        manifest,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    actual_identity = hashlib.sha256(encoded).hexdigest()
    if actual_identity != expected_identity:
        raise ValueError("injection manifest does not match its manifest_identity")


def _validate_scheduled_occurrence_coverage(
    injection: Mapping[str, Any],
    records: Sequence[Mapping[str, Any]],
) -> None:
    completed_iterations = _required_integer(
        injection.get("completed_iterations"),
        "injection completed_iterations",
    )
    if completed_iterations < 0:
        raise ValueError("injection completed_iterations must be non-negative")
    manifest = injection.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("injection artifact requires an embedded manifest")
    incidents = manifest.get("incidents")
    if isinstance(incidents, (str, bytes)) or not isinstance(incidents, Sequence):
        raise TypeError("injection manifest incidents must be an array")

    observed: dict[str, set[int]] = {}
    for record in records:
        incident_id = record.get("incident_id")
        if not isinstance(incident_id, str) or not incident_id:
            raise ValueError("injection records require a non-empty incident_id")
        iteration = _required_integer(record.get("iteration"), "injection iteration")
        observed.setdefault(incident_id, set()).add(iteration)

    manifest_ids: set[str] = set()
    for incident in incidents:
        if not isinstance(incident, Mapping):
            raise TypeError("injection manifest incidents must contain objects")
        incident_id = incident.get("incident_id", incident.get("id"))
        if not isinstance(incident_id, str) or not incident_id:
            raise ValueError("injection manifest incident_id must be non-empty")
        manifest_ids.add(incident_id)
        trigger = incident.get("trigger")
        if not isinstance(trigger, Mapping):
            raise TypeError("injection manifest trigger must be an object")
        observed_iterations = observed.get(incident_id, set())
        if "at" in trigger:
            at = trigger["at"]
            if isinstance(at, (str, bytes)) or not isinstance(at, Sequence):
                raise TypeError("injection manifest trigger at must be an array")
            scheduled = {
                _required_integer(value, "injection manifest trigger iteration") for value in at
            }
            expected_iterations = {
                iteration for iteration in scheduled if iteration <= completed_iterations
            }
            if observed_iterations != expected_iterations:
                missing = sorted(expected_iterations - observed_iterations)
                unexpected = sorted(observed_iterations - expected_iterations)
                raise ValueError(
                    f"incident {incident_id!r} occurrence coverage mismatch: "
                    f"missing={missing}, unexpected={unexpected}"
                )
            continue

        trigger_range = trigger.get("range")
        if not isinstance(trigger_range, Mapping):
            raise TypeError("injection manifest trigger requires at or range")
        start = _required_integer(trigger_range.get("start"), "trigger range start")
        end = _required_integer(trigger_range.get("end"), "trigger range end")
        every = _required_integer(trigger_range.get("every", 1), "trigger range every")
        if start <= 0 or end < start or every <= 0:
            raise ValueError("injection manifest trigger range is invalid")
        last = min(end, completed_iterations)
        expected_count = 0 if last < start else (last - start) // every + 1
        unexpected = sorted(
            iteration
            for iteration in observed_iterations
            if iteration < start or iteration > last or (iteration - start) % every != 0
        )
        if unexpected or len(observed_iterations) != expected_count:
            raise ValueError(
                f"incident {incident_id!r} occurrence coverage mismatch: "
                f"expected_count={expected_count}, observed_count={len(observed_iterations)}, "
                f"unexpected={unexpected}"
            )

    unknown = sorted(set(observed) - manifest_ids)
    if unknown:
        raise ValueError(f"injection records reference unknown incidents: {unknown}")


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


def _required_integer(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    return value


def _reject_ambiguous_occurrence_iterations(
    grouped: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    owners: dict[int, str] = {}
    for occurrence_id, records in grouped.items():
        iterations = {
            _required_integer(record.get("iteration"), "injection iteration") for record in records
        }
        if len(iterations) != 1:
            continue
        iteration = next(iter(iterations))
        previous = owners.setdefault(iteration, occurrence_id)
        if previous != occurrence_id:
            raise ValueError(
                "localization reports cannot be correlated uniquely because "
                f"occurrences {previous!r} and {occurrence_id!r} share training "
                f"iteration {iteration}"
            )


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
