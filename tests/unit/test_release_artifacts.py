"""Tests for release artifact identity manifests."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def release_artifacts():
    if sys.version_info < (3, 11):
        pytest.skip("release tooling runs on Python 3.12 and uses standard-library tomllib")
    return importlib.import_module("scripts.release_artifacts")


def test_release_manifest_round_trip(tmp_path, release_artifacts):
    (tmp_path / "lm_resiliency-0.1.0-py3-none-any.whl").write_bytes(b"wheel")
    (tmp_path / "lm_resiliency-0.1.0.tar.gz").write_bytes(b"sdist")

    created = release_artifacts.create_manifest(
        tmp_path,
        repository="LMResiliency/lm-resiliency",
        commit="a" * 40,
        ref="refs/tags/v0.1.0",
    )
    verified = release_artifacts.verify_manifest(
        tmp_path,
        repository="LMResiliency/lm-resiliency",
        commit="a" * 40,
        ref="refs/tags/v0.1.0",
    )

    assert verified == created
    assert [record["name"] for record in verified["artifacts"]] == [
        "lm_resiliency-0.1.0-py3-none-any.whl",
        "lm_resiliency-0.1.0.tar.gz",
    ]
    assert (tmp_path / "SHA256SUMS").read_text().count("\n") == 2


def test_release_manifest_rejects_digest_drift(tmp_path, release_artifacts):
    wheel = tmp_path / "lm_resiliency-0.1.0-py3-none-any.whl"
    wheel.write_bytes(b"wheel")
    release_artifacts.create_manifest(
        tmp_path,
        repository="LMResiliency/lm-resiliency",
        commit="a" * 40,
        ref="refs/tags/v0.1.0",
    )
    wheel.write_bytes(b"tampered")

    with pytest.raises(ValueError, match="size mismatch"):
        release_artifacts.verify_manifest(
            tmp_path,
            repository="LMResiliency/lm-resiliency",
            commit="a" * 40,
            ref="refs/tags/v0.1.0",
        )


def test_release_manifest_rejects_identity_drift(tmp_path, release_artifacts):
    (tmp_path / "lm_resiliency-0.1.0.tar.gz").write_bytes(b"sdist")
    release_artifacts.create_manifest(
        tmp_path,
        repository="LMResiliency/lm-resiliency",
        commit="a" * 40,
        ref="refs/tags/v0.1.0",
    )
    manifest_path = tmp_path / "release-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["commit_sha"] = "b" * 40
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="commit_sha"):
        release_artifacts.verify_manifest(
            tmp_path,
            repository="LMResiliency/lm-resiliency",
            commit="a" * 40,
            ref="refs/tags/v0.1.0",
        )


def test_production_release_jobs_require_tag_push_and_recheck_target():
    workflow = (_REPOSITORY_ROOT / ".github" / "workflows" / "release.yml").read_text()

    production_condition = "if: github.event_name == 'push' && startsWith(github.ref, 'refs/tags/')"
    assert workflow.count(production_condition) == 2
    assert workflow.count("- name: Recheck release tag target") == 2
    assert '"repos/$GITHUB_REPOSITORY/releases?per_page=100"' in workflow
    assert "select(.tag_name == env.GITHUB_REF_NAME)" in workflow
    assert 'gh release upload "$GITHUB_REF_NAME" dist/*' in workflow
    assert workflow.count("for attempt in $(seq 1 12)") >= 3
    assert "Release attestation was not available" in workflow
    assert "Asset attestation was not available" in workflow
