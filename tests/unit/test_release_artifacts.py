"""Tests for release artifact identity manifests."""

from __future__ import annotations

import json

import pytest

from scripts import release_artifacts


def test_release_manifest_round_trip(tmp_path):
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


def test_release_manifest_rejects_digest_drift(tmp_path):
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


def test_release_manifest_rejects_identity_drift(tmp_path):
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
