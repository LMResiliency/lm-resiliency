"""README Quick Start contract tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"


def test_release_quickstart_clones_installed_version_tag():
    readme = README.read_text()

    assert 'version("lm-resiliency")' in readme
    assert '--branch "v${LM_RESILIENCY_VERSION}"' in readme
    assert "git clone --depth 1 https://github.com/LMResiliency/lm-resiliency.git" not in readme
