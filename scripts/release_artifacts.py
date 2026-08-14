"""Create and verify the digest manifest reused by every release job."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import tomllib

_SCHEMA_VERSION = 1
_ARTIFACT_PATTERNS = ("*.whl", "*.tar.gz")
_MANIFEST_NAME = "release-manifest.json"
_CHECKSUMS_NAME = "SHA256SUMS"
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project() -> tuple[str, str]:
    with (_REPOSITORY_ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]
    return project["name"], project["version"]


def _artifacts(dist: Path) -> list[Path]:
    artifacts = sorted({path for pattern in _ARTIFACT_PATTERNS for path in dist.glob(pattern)})
    if not artifacts:
        raise ValueError(f"no wheel or source distribution found in {dist}")
    return artifacts


def create_manifest(
    dist: Path,
    *,
    repository: str,
    commit: str,
    ref: str,
) -> dict[str, Any]:
    """Write one manifest and checksum list for the built distributions."""
    dist = dist.resolve()
    project_name, version = _project()
    artifacts = [
        {
            "name": path.name,
            "size": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in _artifacts(dist)
    ]
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "project": project_name,
        "version": version,
        "repository": repository,
        "commit_sha": commit,
        "ref": ref,
        "artifacts": artifacts,
    }
    (dist / _MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (dist / _CHECKSUMS_NAME).write_text(
        "".join(f"{artifact['sha256']}  {artifact['name']}\n" for artifact in artifacts),
        encoding="utf-8",
    )
    return manifest


def verify_manifest(
    dist: Path,
    *,
    repository: str,
    commit: str,
    ref: str,
) -> dict[str, Any]:
    """Reject any identity, membership, size, or digest drift in downloaded artifacts."""
    dist = dist.resolve()
    manifest_path = dist / _MANIFEST_NAME
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid or missing {manifest_path}") from error

    expected_keys = {
        "schema_version",
        "project",
        "version",
        "repository",
        "commit_sha",
        "ref",
        "artifacts",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_keys:
        raise ValueError("release manifest has unexpected or missing fields")
    project_name, version = _project()
    expected_identity = {
        "schema_version": _SCHEMA_VERSION,
        "project": project_name,
        "version": version,
        "repository": repository,
        "commit_sha": commit,
        "ref": ref,
    }
    for key, expected in expected_identity.items():
        if manifest[key] != expected:
            raise ValueError(f"release manifest {key} is {manifest[key]!r}, expected {expected!r}")

    records = manifest["artifacts"]
    if not isinstance(records, list) or not records:
        raise ValueError("release manifest artifacts must be a non-empty list")
    if any(
        not isinstance(record, dict)
        or set(record) != {"name", "size", "sha256"}
        or not isinstance(record["name"], str)
        or not isinstance(record["size"], int)
        or not isinstance(record["sha256"], str)
        for record in records
    ):
        raise ValueError("release artifact record has invalid fields")
    expected_names = {path.name for path in _artifacts(dist)}
    record_names = {record["name"] for record in records}
    if record_names != expected_names or len(records) != len(expected_names):
        raise ValueError("release artifact membership does not match the manifest")

    checksum_lines = []
    for record in records:
        path = dist / record["name"]
        if path.stat().st_size != record["size"]:
            raise ValueError(f"release artifact size mismatch for {path.name}")
        digest = _sha256(path)
        if digest != record["sha256"]:
            raise ValueError(f"release artifact digest mismatch for {path.name}")
        checksum_lines.append(f"{digest}  {path.name}\n")

    checksums = dist / _CHECKSUMS_NAME
    if checksums.read_text(encoding="utf-8") != "".join(checksum_lines):
        raise ValueError(f"{_CHECKSUMS_NAME} does not match the release manifest")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("create", "verify"))
    parser.add_argument("--dist", type=Path, required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--ref", required=True)
    args = parser.parse_args()

    kwargs = {
        "repository": args.repository,
        "commit": args.commit,
        "ref": args.ref,
    }
    try:
        if args.command == "create":
            value = create_manifest(args.dist, **kwargs)
        else:
            value = verify_manifest(args.dist, **kwargs)
    except (OSError, TypeError, ValueError) as error:
        print(f"release artifact validation failed: {error}", file=sys.stderr)
        return 1
    print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
