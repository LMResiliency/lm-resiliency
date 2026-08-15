"""README Quick Start contract tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"


def test_quickstart_keeps_source_and_example_on_same_revision():
    readme = README.read_text()
    quickstart = readme.split("## Quick Start", maxsplit=1)[1].split(
        "### Add resiliency to Megatron Core", maxsplit=1
    )[0]

    assert "git clone https://github.com/LMResiliency/lm-resiliency.git" in quickstart
    assert "python -m pip install -e ." in quickstart
    assert "python examples/quickstart.py" in quickstart
    assert "lm-resiliency-quickstart" not in quickstart
