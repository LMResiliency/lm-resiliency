"""README installation and native torchrun example contract tests."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"


def test_installation_matches_source_and_stable_release_paths():
    readme = README.read_text()
    installation = readme.split("## Installation", maxsplit=1)[1].split("## Examples", maxsplit=1)[
        0
    ]

    assert readme.index("## Installation") < readme.index("## Examples")
    assert "### From source" in installation
    assert "git clone https://github.com/LMResiliency/lm-resiliency.git" in installation
    assert "python -m pip install -e ." in installation
    assert "### Stable releases" in installation
    assert "python -m pip install lm-resiliency" in installation
    assert "torchrun \\" not in installation


def test_examples_use_native_torchrun_integration():
    readme = README.read_text()
    examples = readme.split("## Examples", maxsplit=1)[1].split("## Framework Support", maxsplit=1)[
        0
    ]

    assert "torchrun \\" in examples
    assert "Keep one eight-GPU node active" in examples
    assert "park a second eight-GPU node as standby" in examples
    assert "restart its worker group up to four times" in examples
    assert "active/standby admission and recovery coordination" in examples
    assert "--nnodes=1:2" in examples
    assert "--rdzv-backend=lm_resiliency" in examples
    assert '--rdzv-endpoint="${RDZV_HOST}:29400"' in examples
    assert "store_type=tcp" in examples
    assert "lm_resiliency_restart_context_path=${RESTART_CONTEXT}" in examples
    assert "lm_resiliency_worker_config=" in examples
    assert "examples.production_loops.${framework}" in examples
    assert "enable_resiliency(" not in examples


def test_examples_include_torchrun_resiliency_cycle():
    readme = README.read_text()
    examples = readme.split("## Examples", maxsplit=1)[1].split("## Framework Support", maxsplit=1)[
        0
    ]

    assert "### Resiliency cycle" in examples
    assert "all 21 canonical failure types" in examples
    assert "17 incidents exercise same-node restart" in examples
    assert "four incidents" in examples
    assert "training world size of four" in examples
    assert "other four GPU-agents remain parked as standbys" in examples
    assert "all eight allocated GPUs" in examples
    assert "SCOUT replay localization" in examples
    assert "single_node_pressure.json" in examples
    assert "examples.torchrun.resiliency_cycle.pressure orchestrate" in examples
    assert "--gpus 0,1,2,3,4,5,6,7" in examples
    assert "--remote-gpus" not in examples


def test_readme_presents_torchrun_instead_of_external_manager_integration():
    readme = README.read_text()

    assert "Bring your own launcher" not in readme
    assert "## Quick Start" not in readme
    assert "## Manager Integration" not in readme
    assert "## Native Torchrun Recovery" not in readme
    assert "[Torchrun Resiliency](docs/torchrun_resiliency.md)" in readme
