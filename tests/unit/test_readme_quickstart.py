"""README Quick Start contract tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"


def test_quickstart_uses_example_shipped_with_installed_package():
    readme = README.read_text()
    quickstart = readme.split("## Quick Start", maxsplit=1)[1].split(
        "### Add resiliency to Megatron Core", maxsplit=1
    )[0]

    assert "python -m pip install lm-resiliency" in quickstart
    assert "lm-resiliency-quickstart" in quickstart
    assert "git clone" not in quickstart
    assert "python examples/quickstart.py" not in quickstart
