"""Create and verify revision-bound validation evidence bundles."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = 1
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_PAYLOAD_NAMES = {"summary.json", "environment.json", "commands.txt"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_COMMIT = re.compile(r"^[0-9a-f]{40}$")
_COMMAND = re.compile(r"^\[([a-z0-9][a-z0-9._-]*)\] (.+)$")


def _timestamp() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing {path}") from error


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _project() -> tuple[str, str]:
    values = {}
    in_project = False
    for line in (_REPOSITORY_ROOT / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("["):
            in_project = stripped == "[project]"
            continue
        if not in_project:
            continue
        match = re.fullmatch(r'(name|version)\s*=\s*"([^"]+)"', stripped)
        if match:
            values[match.group(1)] = match.group(2)
    if set(values) != {"name", "version"}:
        raise ValueError("pyproject.toml must define literal project name and version strings")
    return values["name"], values["version"]


def _command_ids(path: Path) -> list[str]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"missing {path}") from error
    if not lines:
        raise ValueError("commands.txt must contain at least one command")
    ids = []
    for line in lines:
        match = _COMMAND.fullmatch(line)
        if match is None:
            raise ValueError(f"invalid commands.txt line: {line!r}")
        ids.append(match.group(1))
    if len(ids) != len(set(ids)):
        raise ValueError("commands.txt contains duplicate command IDs")
    return ids


def _require_string(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _validate_summary(summary: Any, campaign_id: str, command_ids: list[str]) -> None:
    required = {
        "schema_version",
        "campaign_id",
        "status",
        "started_at",
        "completed_at",
        "counts",
        "metrics",
        "results",
        "topology",
        "seed",
        "configuration",
    }
    if not isinstance(summary, dict) or set(summary) != required:
        raise ValueError("summary.json has unexpected or missing fields")
    if summary["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("summary.json has an unsupported schema version")
    if summary["campaign_id"] != campaign_id:
        raise ValueError("summary campaign_id does not match the manifest")
    if summary["status"] not in {"passed", "failed", "cancelled"}:
        raise ValueError("summary status is invalid")
    timestamps = []
    for field in ("started_at", "completed_at"):
        timestamp = _require_string(summary[field], f"summary {field}")
        try:
            parsed = dt.datetime.fromisoformat(timestamp)
        except ValueError as error:
            raise ValueError(f"summary {field} must be an ISO-8601 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError(f"summary {field} must include a timezone")
        timestamps.append(parsed)
    if timestamps[1] < timestamps[0]:
        raise ValueError("summary completed_at precedes started_at")
    counts = summary["counts"]
    count_keys = {"total", "passed", "failed", "skipped", "errors"}
    if (
        not isinstance(counts, dict)
        or set(counts) != count_keys
        or any(type(counts[key]) is not int or counts[key] < 0 for key in count_keys)
        or counts["total"]
        != counts["passed"] + counts["failed"] + counts["skipped"] + counts["errors"]
    ):
        raise ValueError("summary counts are inconsistent")
    if not isinstance(summary["metrics"], dict):
        raise ValueError("summary metrics must be an object")
    if not isinstance(summary["results"], list):
        raise ValueError("summary results must be a list")
    if summary["status"] == "passed" and (
        counts["passed"] < 1 or counts["failed"] or counts["errors"]
    ):
        raise ValueError("a passed summary must contain successful results and no failures")
    result_ids = []
    for result in summary["results"]:
        if not isinstance(result, dict):
            raise ValueError("summary result entries must be objects")
        result_ids.append(_require_string(result.get("command_id"), "result command_id"))
        if result.get("status") not in {"passed", "failed", "skipped", "error"}:
            raise ValueError("summary result status is invalid")
    if not set(result_ids).issubset(command_ids):
        raise ValueError("summary refers to an unknown command ID")
    topology = summary["topology"]
    if not isinstance(topology, dict):
        raise ValueError("summary topology must be an object")
    for field in ("hosts", "world_size", "gpu_count"):
        if type(topology.get(field)) is not int or topology[field] < 0:
            raise ValueError(f"summary topology {field} must be a non-negative integer")
    if not isinstance(summary["configuration"], dict):
        raise ValueError("summary configuration must be an object")
    if summary["seed"] is not None and type(summary["seed"]) is not int:
        raise ValueError("summary seed must be an integer or null")


def _validate_result_logs(bundle: Path, summary: dict[str, Any]) -> None:
    for result in summary["results"]:
        log = result.get("log")
        digest = result.get("log_sha256")
        if log is None and digest is None:
            continue
        if not isinstance(log, str) or not log or _SHA256.fullmatch(str(digest)) is None:
            raise ValueError("summary result log identity is invalid")
        path = bundle / log
        try:
            path.resolve().relative_to(bundle)
        except ValueError as error:
            raise ValueError("summary result log escapes the evidence bundle") from error
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(f"summary result log digest mismatch for {log}")


def _validate_environment(environment: Any) -> None:
    required = {"schema_version", "captured_at", "hardware", "software", "runner"}
    if not isinstance(environment, dict) or set(environment) != required:
        raise ValueError("environment.json has unexpected or missing fields")
    if environment["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("environment.json has an unsupported schema version")
    _require_string(environment["captured_at"], "environment captured_at")
    if not isinstance(environment["hardware"], dict):
        raise ValueError("environment hardware must be an object")
    for field in ("hosts", "world_size", "gpu_count"):
        value = environment["hardware"].get(field)
        if type(value) is not int or value < 0:
            raise ValueError(f"environment hardware {field} must be a non-negative integer")
    if not isinstance(environment["software"], dict):
        raise ValueError("environment software must be an object")
    frameworks = environment["software"].get("frameworks")
    if not isinstance(frameworks, dict) or not frameworks:
        raise ValueError("environment software frameworks must be a non-empty object")
    if not isinstance(environment["runner"], dict):
        raise ValueError("environment runner must be an object")


def _payload_files(bundle: Path) -> list[Path]:
    paths = sorted(
        path
        for path in bundle.rglob("*")
        if path.is_file() and path.name not in {"manifest.json", "checksums.txt"}
    )
    if any(path.is_symlink() for path in paths):
        raise ValueError("evidence payloads must not be symbolic links")
    return paths


def seal_bundle(
    bundle: Path,
    *,
    campaign_id: str,
    tier: str,
    repository: str,
    commit: str,
    ref: str,
    artifact_name: str,
    artifact_url: str,
    frameworks: list[str],
    boundaries: list[str],
) -> dict[str, Any]:
    """Validate payload records and seal them with an exact identity and digests."""
    bundle = bundle.resolve()
    missing = _PAYLOAD_NAMES - {path.name for path in bundle.iterdir() if path.is_file()}
    if missing:
        raise ValueError(f"evidence bundle is missing required files: {sorted(missing)}")
    if _COMMIT.fullmatch(commit) is None:
        raise ValueError("commit must be a full 40-character lowercase Git SHA")
    campaign_id = _require_string(campaign_id, "campaign_id")
    tier = _require_string(tier, "tier")
    repository = _require_string(repository, "repository")
    ref = _require_string(ref, "ref")
    artifact_name = _require_string(artifact_name, "artifact_name")
    artifact_url = _require_string(artifact_url, "artifact_url")
    if not frameworks or any(not isinstance(item, str) or not item for item in frameworks):
        raise ValueError("at least one framework must be recorded")
    if not boundaries or any(not isinstance(item, str) or not item for item in boundaries):
        raise ValueError("at least one qualification boundary must be recorded")

    command_ids = _command_ids(bundle / "commands.txt")
    summary = _read_json(bundle / "summary.json")
    environment = _read_json(bundle / "environment.json")
    _validate_summary(summary, campaign_id, command_ids)
    _validate_environment(environment)
    for field in ("hosts", "world_size", "gpu_count"):
        if summary["topology"][field] != environment["hardware"][field]:
            raise ValueError(f"summary and environment topology {field} differ")
    _validate_result_logs(bundle, summary)
    project_name, version = _project()
    payloads = [
        {
            "path": str(path.relative_to(bundle)),
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _payload_files(bundle)
    ]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "project": {
            "name": project_name,
            "version": version,
            "repository": repository,
            "commit_sha": commit,
            "ref": ref,
        },
        "campaign": {
            "id": campaign_id,
            "tier": tier,
            "status": summary["status"],
            "started_at": summary["started_at"],
            "completed_at": summary["completed_at"],
            "command_ids": command_ids,
            "seed": summary["seed"],
            "configuration": summary["configuration"],
        },
        "artifact": {"name": artifact_name, "url": artifact_url},
        "qualification": {
            "frameworks": sorted(set(frameworks)),
            "topology": summary["topology"],
            "boundaries": boundaries,
        },
        "files": payloads,
    }
    _write_json(bundle / "manifest.json", manifest)
    checksum_paths = [bundle / "manifest.json", *_payload_files(bundle)]
    (bundle / "checksums.txt").write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(bundle)}\n" for path in checksum_paths),
        encoding="utf-8",
    )
    return manifest


def verify_bundle(bundle: Path, *, expected_commit: str | None = None) -> dict[str, Any]:
    """Reject schema, identity, membership, size, or digest drift in a bundle."""
    bundle = bundle.resolve()
    manifest = _read_json(bundle / "manifest.json")
    required = {"schema_version", "project", "campaign", "artifact", "qualification", "files"}
    if not isinstance(manifest, dict) or set(manifest) != required:
        raise ValueError("manifest.json has unexpected or missing fields")
    if manifest["schema_version"] != _SCHEMA_VERSION:
        raise ValueError("manifest.json has an unsupported schema version")
    project = manifest["project"]
    project_keys = {"name", "version", "repository", "commit_sha", "ref"}
    if not isinstance(project, dict) or set(project) != project_keys:
        raise ValueError("manifest project identity is invalid")
    project_name, version = _project()
    if project["name"] != project_name or project["version"] != version:
        raise ValueError("manifest package identity does not match pyproject.toml")
    if _COMMIT.fullmatch(str(project["commit_sha"])) is None:
        raise ValueError("manifest commit_sha is not a full lowercase Git SHA")
    if expected_commit is not None and project["commit_sha"] != expected_commit:
        raise ValueError(
            f"validation evidence is stale: covers {project['commit_sha']}, expected {expected_commit}"
        )
    for field in ("repository", "ref"):
        _require_string(project[field], f"manifest project {field}")

    campaign = manifest["campaign"]
    campaign_keys = {
        "id",
        "tier",
        "status",
        "started_at",
        "completed_at",
        "command_ids",
        "seed",
        "configuration",
    }
    if not isinstance(campaign, dict) or set(campaign) != campaign_keys:
        raise ValueError("manifest campaign is invalid")
    for field in ("id", "tier"):
        _require_string(campaign[field], f"manifest campaign {field}")
    command_ids = _command_ids(bundle / "commands.txt")
    if campaign["command_ids"] != command_ids:
        raise ValueError("manifest command IDs do not match commands.txt")
    summary = _read_json(bundle / "summary.json")
    environment = _read_json(bundle / "environment.json")
    _validate_summary(summary, campaign["id"], command_ids)
    _validate_environment(environment)
    for field in ("hosts", "world_size", "gpu_count"):
        if summary["topology"][field] != environment["hardware"][field]:
            raise ValueError(f"summary and environment topology {field} differ")
    _validate_result_logs(bundle, summary)
    if campaign["status"] != summary["status"]:
        raise ValueError("manifest and summary status differ")
    for field in ("seed", "configuration", "started_at", "completed_at"):
        if campaign[field] != summary[field]:
            raise ValueError(f"manifest and summary {field} differ")

    artifact = manifest["artifact"]
    if not isinstance(artifact, dict) or set(artifact) != {"name", "url"}:
        raise ValueError("manifest artifact is invalid")
    _require_string(artifact["name"], "manifest artifact name")
    _require_string(artifact["url"], "manifest artifact URL")
    qualification = manifest["qualification"]
    if not isinstance(qualification, dict) or set(qualification) != {
        "frameworks",
        "topology",
        "boundaries",
    }:
        raise ValueError("manifest qualification is invalid")
    if qualification["topology"] != summary["topology"]:
        raise ValueError("manifest and summary topology differ")
    frameworks = qualification["frameworks"]
    if (
        not isinstance(frameworks, list)
        or not frameworks
        or len(frameworks) != len(set(frameworks))
        or any(not isinstance(item, str) or not item for item in frameworks)
    ):
        raise ValueError("manifest frameworks must be a non-empty list")
    boundaries = qualification["boundaries"]
    if (
        not isinstance(boundaries, list)
        or not boundaries
        or any(not isinstance(item, str) or not item for item in boundaries)
    ):
        raise ValueError("manifest boundaries must be a non-empty list")

    records = manifest["files"]
    actual = _payload_files(bundle)
    if not isinstance(records, list) or len(records) != len(actual):
        raise ValueError("manifest payload membership does not match the bundle")
    expected_by_path: dict[str, dict[str, Any]] = {}
    for record in records:
        if (
            not isinstance(record, dict)
            or set(record) != {"path", "bytes", "sha256"}
            or not isinstance(record["path"], str)
            or type(record["bytes"]) is not int
            or not isinstance(record["sha256"], str)
            or _SHA256.fullmatch(record["sha256"]) is None
            or record["path"] in expected_by_path
        ):
            raise ValueError("manifest payload record is invalid")
        expected_by_path[record["path"]] = record
    if set(expected_by_path) != {str(path.relative_to(bundle)) for path in actual}:
        raise ValueError("manifest payload membership does not match the bundle")
    for path in actual:
        record = expected_by_path[str(path.relative_to(bundle))]
        if path.stat().st_size != record["bytes"] or _sha256(path) != record["sha256"]:
            raise ValueError(f"evidence payload digest mismatch for {record['path']}")

    checksum_paths = [bundle / "manifest.json", *actual]
    expected_checksums = "".join(
        f"{_sha256(path)}  {path.relative_to(bundle)}\n" for path in checksum_paths
    )
    try:
        checksums = (bundle / "checksums.txt").read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise ValueError("missing checksums.txt") from error
    if checksums != expected_checksums:
        raise ValueError("checksums.txt does not match the evidence bundle")
    return manifest


def _framework_versions() -> dict[str, str | None]:
    try:
        import torch
    except ImportError:
        torch_version = None
        cuda_version = None
        nccl_version = None
    else:
        torch_version = torch.__version__
        cuda_version = torch.version.cuda
        nccl = torch.cuda.nccl.version() if torch.cuda.is_available() else None
        nccl_version = ".".join(map(str, nccl)) if isinstance(nccl, tuple) else nccl
    return {"pytorch": torch_version, "cuda": cuda_version, "nccl": nccl_version}


def _parse_assignments(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        key, separator, item = value.partition("=")
        if not separator or not key or not item:
            raise ValueError(f"configuration must use NAME=VALUE: {value!r}")
        result[key] = item
    return result


def create_pytest_bundle(
    bundle: Path,
    *,
    junit: Path,
    action_status: str,
    command: str,
    configuration: dict[str, str],
    seal_options: dict[str, Any],
) -> dict[str, Any]:
    """Create a canonical CPU evidence payload from pytest JUnit output."""
    bundle.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any]
    if junit.is_file():
        destination = bundle / "junit.xml"
        shutil.copyfile(junit, destination)
        root = ET.parse(destination).getroot()
        suites = [root] if root.tag == "testsuite" else list(root)
        total = sum(int(suite.attrib.get("tests", 0)) for suite in suites)
        failures = sum(int(suite.attrib.get("failures", 0)) for suite in suites)
        errors = sum(int(suite.attrib.get("errors", 0)) for suite in suites)
        skipped = sum(int(suite.attrib.get("skipped", 0)) for suite in suites)
        passed = total - failures - errors - skipped
        duration = sum(float(suite.attrib.get("time", 0.0)) for suite in suites)
        result = {
            "command_id": "cpu-unit-suite",
            "status": "passed"
            if failures == errors == 0 and action_status == "success"
            else "failed",
            "duration_seconds": duration,
            "log": destination.name,
            "log_sha256": _sha256(destination),
        }
    else:
        total, passed, failures, skipped, errors, duration = 1, 0, 0, 0, 1, 0.0
        result = {
            "command_id": "cpu-unit-suite",
            "status": "error",
            "duration_seconds": duration,
            "error": f"pytest did not produce {junit}",
        }
    status = "passed" if result["status"] == "passed" else "failed"
    completed_at = dt.datetime.now(dt.timezone.utc)
    started_at = completed_at - dt.timedelta(seconds=duration)
    summary = {
        "schema_version": _SCHEMA_VERSION,
        "campaign_id": seal_options["campaign_id"],
        "status": status,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "counts": {
            "total": total,
            "passed": passed,
            "failed": failures,
            "skipped": skipped,
            "errors": errors,
        },
        "metrics": {"duration_seconds": duration},
        "results": [result],
        "topology": {"hosts": 1, "world_size": 1, "gpu_count": 0},
        "seed": None,
        "configuration": configuration,
    }
    environment = {
        "schema_version": _SCHEMA_VERSION,
        "captured_at": _timestamp(),
        "hardware": {
            "host": platform.node(),
            "platform": platform.platform(),
            "hosts": 1,
            "world_size": 1,
            "gpu_count": 0,
            "devices": [],
        },
        "software": {
            "python": platform.python_version(),
            "frameworks": _framework_versions(),
        },
        "runner": {
            "name": os.environ.get("RUNNER_NAME"),
            "os": os.environ.get("RUNNER_OS"),
            "arch": os.environ.get("RUNNER_ARCH"),
        },
    }
    _write_json(bundle / "summary.json", summary)
    _write_json(bundle / "environment.json", environment)
    (bundle / "commands.txt").write_text(f"[cpu-unit-suite] {command}\n", encoding="utf-8")
    return seal_bundle(bundle, **seal_options)


def _add_seal_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--campaign-id", required=True)
    parser.add_argument("--tier", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ref", required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--artifact-url", required=True)
    parser.add_argument("--framework", action="append", dest="frameworks", required=True)
    parser.add_argument("--boundary", action="append", dest="boundaries", required=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    _add_seal_arguments(seal_parser)
    verify_parser = subparsers.add_parser("verify")
    verify_parser.add_argument("--bundle", type=Path, required=True)
    verify_parser.add_argument("--expected-commit")
    pytest_parser = subparsers.add_parser("create-pytest")
    _add_seal_arguments(pytest_parser)
    pytest_parser.add_argument("--junit", type=Path, required=True)
    pytest_parser.add_argument("--action-status", required=True)
    pytest_parser.add_argument("--recorded-command", required=True)
    pytest_parser.add_argument("--configuration", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "verify":
            value = verify_bundle(args.bundle, expected_commit=args.expected_commit)
        else:
            seal_options = {
                "campaign_id": args.campaign_id,
                "tier": args.tier,
                "repository": args.repository,
                "commit": args.commit,
                "ref": args.ref,
                "artifact_name": args.artifact_name,
                "artifact_url": args.artifact_url,
                "frameworks": args.frameworks,
                "boundaries": args.boundaries,
            }
            if args.command == "seal":
                value = seal_bundle(args.bundle, **seal_options)
            else:
                value = create_pytest_bundle(
                    args.bundle,
                    junit=args.junit,
                    action_status=args.action_status,
                    command=args.recorded_command,
                    configuration=_parse_assignments(args.configuration),
                    seal_options=seal_options,
                )
    except (OSError, TypeError, ValueError, ET.ParseError) as error:
        print(f"validation evidence failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
