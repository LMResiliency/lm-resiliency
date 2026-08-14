"""Compare injected-fault ground truth with normalized SCOUT JSON reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from lm_resiliency.fault_injection.config import FailureType, expected_failure_kind


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
    for record in all_records:
        _record_injection_succeeded(record)
    _validate_scheduled_occurrence_coverage(injection, all_records)
    records = [
        dict(record) for record in all_records if record.get("status") != "skipped_probability"
    ]
    expected_actions = _manifest_actions(injection)
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
            expected_actions,
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
        "injected_actions": sum(_record_injection_succeeded(record) for record in records),
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
    manifest_actions: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    incident_ids = {str(record.get("incident_id", "")) for record in records}
    if len(incident_ids) != 1 or "" in incident_ids:
        raise ValueError(f"occurrence {occurrence_id!r} requires one non-empty incident_id")
    incident_id = incident_ids.pop()
    expected_by_fault_id = manifest_actions.get(incident_id)
    if expected_by_fault_id is None:
        raise ValueError(
            f"occurrence {occurrence_id!r} references unknown incident {incident_id!r}"
        )
    recorded_fault_ids = [
        _required_string(record.get("fault_id"), "injection fault_id") for record in records
    ]
    for record, fault_id in zip(records, recorded_fault_ids, strict=True):
        action = expected_by_fault_id.get(fault_id)
        if action is None:
            raise ValueError(f"occurrence {occurrence_id!r} references unknown fault {fault_id!r}")
        _validate_record_against_manifest(record, action)
    expected_fault_ids = frozenset(expected_by_fault_id)
    complete_action_set = (
        len(recorded_fault_ids) == len(expected_fault_ids)
        and set(recorded_fault_ids) == expected_fault_ids
    )
    injection_succeeded = (
        complete_action_set
        and bool(records)
        and all(_record_injection_succeeded(record) for record in records)
    )
    iterations = {
        _required_integer(record.get("iteration"), "injection iteration") for record in records
    }
    if len(iterations) != 1:
        raise ValueError(f"occurrence {occurrence_id!r} spans multiple training iterations")
    iteration = iterations.pop()
    expected_ranks = sorted(
        {
            expected_rank
            for action in expected_by_fault_id.values()
            if (expected_rank := _expected_action_rank(action)) is not None
        }
    )
    expected_resources = sorted(
        {
            resource
            for action in expected_by_fault_id.values()
            if (resource := _expected_action_resource(action)) is not None
        }
    )
    expected_kinds = sorted(
        {_expected_action_kind(action) for action in expected_by_fault_id.values()}
    )
    expected_ranks_by_kind = {
        kind: sorted(
            {
                expected_rank
                for action in expected_by_fault_id.values()
                if _expected_action_kind(action) == kind
                and (expected_rank := _expected_action_rank(action)) is not None
            }
        )
        for kind in expected_kinds
    }
    expected_resources_by_kind = {
        kind: sorted(
            {
                resource
                for action in expected_by_fault_id.values()
                if _expected_action_kind(action) == kind
                and (resource := _expected_action_resource(action)) is not None
            }
        )
        for kind in expected_kinds
    }
    expected_layers = sorted(
        {
            int(target["index"])
            for action in expected_by_fault_id.values()
            if (target := _action_target(action)).get("component")
            in {"transformer_block", "transformer_layer", "layer"}
            and target.get("index") is not None
        }
    )
    expected_source_prefixes = sorted(
        {
            prefix
            for action in expected_by_fault_id.values()
            if (prefix := _expected_action_source_prefix(action)) is not None
        }
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
        {
            resource
            for report in matching
            for resource in _required_string_sequence(
                report.get("failed_resources", ()),
                "localization failed_resources",
            )
        }
    )
    observed_resources_by_kind = {
        kind: sorted(
            {
                resource
                for report in at_iteration
                if str(report.get("kind")) == kind
                for resource in _required_string_sequence(
                    report.get("failed_resources", ()),
                    "localization failed_resources",
                )
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
            (expected_rank := _expected_action_rank(expected_by_fault_id[str(record["fault_id"])]))
            is None
            or expected_rank
            in observed_ranks_by_kind.get(
                _expected_action_kind(expected_by_fault_id[str(record["fault_id"])]),
                (),
            )
        )
        and (
            (
                expected_resource := _expected_action_resource(
                    expected_by_fault_id[str(record["fault_id"])]
                )
            )
            is None
            or expected_resource
            in observed_resources_by_kind.get(
                _expected_action_kind(expected_by_fault_id[str(record["fault_id"])]),
                (),
            )
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
        "expected_action_count": len(expected_by_fault_id),
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


def _expected_action_rank(action: Mapping[str, Any]) -> int | None:
    target = _action_target(action)
    rank = target.get("rank")
    if rank is not None:
        return _required_integer(rank, "manifest target rank")
    if target.get("resource") is not None:
        return None
    return 0


def _expected_action_resource(action: Mapping[str, Any]) -> str | None:
    target = _action_target(action)
    resource = target.get("resource")
    if resource is None:
        return None
    if not isinstance(resource, str):
        raise TypeError("manifest target resource must be a string")
    if not resource:
        raise ValueError("manifest target resource must be non-empty")
    return resource


def _expected_action_kind(action: Mapping[str, Any]) -> str:
    failure_type = _required_string(action.get("type"), "manifest fault type")
    try:
        return expected_failure_kind(FailureType(failure_type))
    except ValueError as error:
        raise ValueError(f"manifest fault type {failure_type!r} is unsupported") from error


def _action_target(action: Mapping[str, Any]) -> Mapping[str, Any]:
    target = action.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("manifest fault target must be an object")
    return target


def _manifest_actions(
    injection: Mapping[str, Any],
) -> dict[str, dict[str, Mapping[str, Any]]]:
    manifest = injection.get("manifest")
    if not isinstance(manifest, Mapping):
        raise ValueError("injection artifact requires an embedded manifest")
    incidents = manifest.get("incidents")
    if isinstance(incidents, (str, bytes)) or not isinstance(incidents, Sequence):
        raise TypeError("injection manifest incidents must be an array")
    expected: dict[str, dict[str, Mapping[str, Any]]] = {}
    for incident in incidents:
        if not isinstance(incident, Mapping):
            raise TypeError("injection manifest incidents must contain objects")
        incident_id = _required_string(
            incident.get("incident_id", incident.get("id")),
            "injection manifest incident_id",
        )
        faults = incident.get("faults")
        if isinstance(faults, (str, bytes)) or not isinstance(faults, Sequence):
            raise TypeError("injection manifest faults must be an array")
        actions: dict[str, Mapping[str, Any]] = {}
        for fault in faults:
            if not isinstance(fault, Mapping):
                raise TypeError("injection manifest faults must contain objects")
            fault_id = _required_string(
                fault.get("fault_id", fault.get("id")),
                "injection manifest fault_id",
            )
            if fault_id in actions:
                raise ValueError("injection manifest fault_id values must be unique per incident")
            _action_target(fault)
            _expected_action_kind(fault)
            actions[fault_id] = fault
        if not actions:
            raise ValueError("injection manifest faults require non-empty fault_id values")
        expected[incident_id] = actions
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


def _validate_record_against_manifest(
    record: Mapping[str, Any],
    action: Mapping[str, Any],
) -> None:
    expected_target = dict(_action_target(action))
    target = record.get("target")
    if not isinstance(target, Mapping):
        raise TypeError("injection record target must be an object")
    if dict(target) != expected_target:
        raise ValueError("injection record target does not match the authenticated manifest")
    expected_type = _required_string(action.get("type"), "manifest fault type")
    if _required_string(record.get("failure_type"), "injection failure_type") != expected_type:
        raise ValueError("injection record failure_type does not match the authenticated manifest")
    expected_kind = _expected_action_kind(action)
    if _required_string(record.get("expected_kind"), "injection expected_kind") != expected_kind:
        raise ValueError("injection record expected_kind does not match the authenticated manifest")
    parameters = record.get("parameters")
    if not isinstance(parameters, Mapping):
        raise TypeError("injection record parameters must be an object")
    expected_parameters = action.get("parameters", {})
    if not isinstance(expected_parameters, Mapping):
        raise TypeError("manifest fault parameters must be an object")
    if dict(parameters) != dict(expected_parameters):
        raise ValueError("injection record parameters do not match the authenticated manifest")
    expected_execution_rank = expected_target.get("rank", 0)
    if expected_execution_rank is None:
        expected_execution_rank = 0
    if _required_integer(
        record.get("execution_rank"),
        "injection execution_rank",
    ) != _required_integer(expected_execution_rank, "manifest execution rank"):
        raise ValueError("injection execution_rank does not match the authenticated manifest")


def _record_injection_succeeded(record: Mapping[str, Any]) -> bool:
    succeeded = record.get("injection_succeeded")
    if not isinstance(succeeded, bool):
        raise TypeError("injection_succeeded must be a boolean")
    verified = record.get("verified")
    if not isinstance(verified, bool):
        raise TypeError("injection verified must be a boolean")
    status = _required_string(record.get("status"), "injection status")
    if status not in {
        "pending",
        "active",
        "completed",
        "skipped_probability",
        "failed",
        "cancelled",
    }:
        raise ValueError(f"injection status {status!r} is unsupported")
    expected = verified and status in {"active", "completed"}
    if succeeded != expected:
        raise ValueError(
            "injection_succeeded disagrees with the record status and verification state"
        )
    return succeeded


def _expected_action_source_prefix(action: Mapping[str, Any]) -> str | None:
    if _expected_action_kind(action) != "sdc":
        return None
    target = dict(_action_target(action))
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


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    if not value:
        raise ValueError(f"{label} must be non-empty")
    return value


def _required_string_sequence(value: object, label: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"{label} must be an array of strings")
    return tuple(_required_string(item, f"{label} item") for item in value)


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
