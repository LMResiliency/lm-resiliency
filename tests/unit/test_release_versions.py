from __future__ import annotations

import pytest
from packaging.version import InvalidVersion

from scripts.release_versions import is_prerelease, latest_stable_tag


@pytest.mark.parametrize(
    "version",
    ["1.0a0", "1.0-a", "1.0-RC1", "1.0.dev0", "1.0-preview2"],
)
def test_pep440_prereleases_are_classified(version: str):
    assert is_prerelease(version)


@pytest.mark.parametrize("version", ["1.0", "1.0.post1", "1!1.0", "1.0+cuda"])
def test_stable_versions_are_not_classified_as_prereleases(version: str):
    assert not is_prerelease(version)


def test_invalid_version_is_rejected():
    with pytest.raises(InvalidVersion):
        is_prerelease("release-candidate")


def test_latest_stable_tag_uses_pep440_ordering():
    assert latest_stable_tag(
        ["v1.0-RC1", "v1.0.post1", "release-notes", "v1.0.0", "v1.0"]
    ) == "v1.0.post1"


def test_latest_stable_tag_requires_a_valid_candidate():
    with pytest.raises(ValueError, match="no stable PEP 440"):
        latest_stable_tag(["v1.0rc1", "nightly"])
