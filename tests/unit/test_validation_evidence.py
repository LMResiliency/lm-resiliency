"""Tests for revision-bound validation evidence bundles."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import validation_evidence

_COMMIT = "0123456789abcdef" * 2 + "01234567"


def _write_payload(bundle: Path) -> None:
    bundle.mkdir()
    summary = {
        "schema_version": 1,
        "campaign_id": "test-campaign",
        "status": "passed",
        "started_at": "2026-08-14T00:00:00+00:00",
        "completed_at": "2026-08-14T00:01:00+00:00",
        "counts": {"total": 1, "passed": 1, "failed": 0, "skipped": 0, "errors": 0},
        "metrics": {"duration_seconds": 60.0},
        "results": [{"command_id": "smoke", "status": "passed"}],
        "topology": {"hosts": 1, "world_size": 1, "gpu_count": 0},
        "seed": 7,
        "configuration": {"mode": "test"},
    }
    environment = {
        "schema_version": 1,
        "captured_at": "2026-08-14T00:00:00+00:00",
        "hardware": {
            "host": "test",
            "hosts": 1,
            "world_size": 1,
            "gpu_count": 0,
            "devices": [],
        },
        "software": {
            "python": "3.12.11",
            "frameworks": {"pytorch": "2.13.0"},
        },
        "runner": {"name": "test"},
    }
    (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (bundle / "environment.json").write_text(json.dumps(environment), encoding="utf-8")
    (bundle / "commands.txt").write_text("[smoke] python -m pytest -q\n", encoding="utf-8")
    (bundle / "smoke.log").write_text("passed\n", encoding="utf-8")


def _seal(bundle: Path) -> dict:
    return validation_evidence.seal_bundle(
        bundle,
        campaign_id="test-campaign",
        tier="cpu",
        repository="LMResiliency/lm-resiliency",
        commit=_COMMIT,
        ref="refs/heads/main",
        artifact_name=f"test-campaign-{_COMMIT}",
        artifact_url="https://github.com/LMResiliency/lm-resiliency/actions/runs/1",
        frameworks=["pytorch", "scout"],
        boundaries=["CPU only."],
    )


def test_seal_and_verify_revision_bound_bundle(tmp_path):
    bundle = tmp_path / "evidence"
    _write_payload(bundle)

    manifest = _seal(bundle)
    verified = validation_evidence.verify_bundle(bundle, expected_commit=_COMMIT)

    assert verified == manifest
    assert manifest["project"]["version"] == "0.1.0"
    assert manifest["campaign"]["command_ids"] == ["smoke"]
    assert manifest["qualification"]["boundaries"] == ["CPU only."]
    assert {record["path"] for record in manifest["files"]} == {
        "commands.txt",
        "environment.json",
        "smoke.log",
        "summary.json",
    }


def test_verify_rejects_stale_revision(tmp_path):
    bundle = tmp_path / "evidence"
    _write_payload(bundle)
    _seal(bundle)

    with pytest.raises(ValueError, match="validation evidence is stale"):
        validation_evidence.verify_bundle(bundle, expected_commit="f" * 40)


def test_verify_rejects_payload_tampering(tmp_path):
    bundle = tmp_path / "evidence"
    _write_payload(bundle)
    _seal(bundle)
    (bundle / "smoke.log").write_text("changed\n", encoding="utf-8")

    with pytest.raises(ValueError, match="payload digest mismatch"):
        validation_evidence.verify_bundle(bundle)


def test_seal_rejects_unidentified_commands(tmp_path):
    bundle = tmp_path / "evidence"
    _write_payload(bundle)
    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    summary["results"][0]["command_id"] = "unknown"
    (bundle / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with pytest.raises(ValueError, match="unknown command ID"):
        _seal(bundle)


def test_create_pytest_bundle_records_counts_environment_and_log(tmp_path):
    junit = tmp_path / "results.xml"
    junit.write_text(
        '<testsuites><testsuite tests="3" failures="0" errors="0" skipped="1" '
        'time="1.25" /></testsuites>',
        encoding="utf-8",
    )
    bundle = tmp_path / "evidence"

    manifest = validation_evidence.create_pytest_bundle(
        bundle,
        junit=junit,
        action_status="success",
        command="python -m pytest -q --junitxml=/tmp/results.xml",
        configuration={"python": "3.12", "torch": "2.13.0"},
        seal_options={
            "campaign_id": "cpu-unit-primary",
            "tier": "cpu",
            "repository": "LMResiliency/lm-resiliency",
            "commit": _COMMIT,
            "ref": "refs/heads/main",
            "artifact_name": f"cpu-primary-{_COMMIT}",
            "artifact_url": "https://github.com/LMResiliency/lm-resiliency/actions/runs/1",
            "frameworks": ["pytorch"],
            "boundaries": ["CPU only."],
        },
    )

    summary = json.loads((bundle / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"] == {
        "total": 3,
        "passed": 2,
        "failed": 0,
        "skipped": 1,
        "errors": 0,
    }
    assert summary["results"][0]["log_sha256"]
    assert manifest["campaign"]["status"] == "passed"
    validation_evidence.verify_bundle(bundle, expected_commit=_COMMIT)


def test_schema_documents_the_executable_manifest_contract():
    schema_path = validation_evidence._REPOSITORY_ROOT / "validation/evidence-schema-v1.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    assert schema["properties"]["schema_version"] == {"const": 1}
    assert set(schema["required"]) == {
        "schema_version",
        "project",
        "campaign",
        "artifact",
        "qualification",
        "files",
    }
