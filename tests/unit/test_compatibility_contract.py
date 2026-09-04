import re
from pathlib import Path

ROOT = Path(__file__).parents[2]


def _declared_minor_series(text: str, pattern: str) -> set[str]:
    match = re.search(pattern, text)
    assert match is not None
    lower_major, lower_minor, upper_major, upper_minor = map(int, match.groups())
    assert lower_major == upper_major
    return {f"{lower_major}.{minor}" for minor in range(lower_minor, upper_minor)}


def _unit_matrix() -> list[tuple[str, str, str, str]]:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text()
    matrix = workflow.split("        include:\n", maxsplit=1)[1]
    matrix = matrix.split("\n\n    env:", maxsplit=1)[0]
    rows = re.findall(
        r'- name: ([\w.-]+)\n\s+python: "([\d.]+)"\n'
        r'\s+numpy: "([\d.]+)"\n\s+torch: "([\d.]+)"',
        matrix,
    )
    assert rows
    return rows


def test_unit_matrix_covers_every_declared_minor_series():
    pyproject = (ROOT / "pyproject.toml").read_text()
    rows = _unit_matrix()

    declared_python = _declared_minor_series(
        pyproject,
        r'requires-python = ">=(\d+)\.(\d+),<(\d+)\.(\d+)"',
    )
    declared_torch = _declared_minor_series(
        pyproject,
        r'"torch>=(\d+)\.(\d+),<(\d+)\.(\d+)"',
    )

    assert {python for _, python, _, _ in rows} == declared_python
    assert {torch.rsplit(".", maxsplit=1)[0] for _, _, _, torch in rows} == declared_torch


def test_python_classifiers_match_declared_range():
    pyproject = (ROOT / "pyproject.toml").read_text()
    declared_python = _declared_minor_series(
        pyproject,
        r'requires-python = ">=(\d+)\.(\d+),<(\d+)\.(\d+)"',
    )
    classifiers = set(re.findall(r"Programming Language :: Python :: (\d+\.\d+)", pyproject))

    assert classifiers == declared_python


def test_torchrun_plugin_uses_zero_argument_handler_factory():
    pyproject = (ROOT / "pyproject.toml").read_text()

    assert (
        'lm_resiliency = "lm_resiliency.integrations.torchrun:get_rendezvous_handler_creator"'
        in pyproject
    )


def test_compatibility_guide_lists_each_ci_combination():
    guide = (ROOT / "docs" / "compatibility.md").read_text()

    for name, python, numpy, torch in _unit_matrix():
        assert f"| `{name}` | {python} | {numpy} | {torch} |" in guide
