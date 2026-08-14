import re
from pathlib import Path

ROOT = Path(__file__).parents[2]
CONSTRAINTS = ROOT / "requirements" / "tool-versions.txt"


def _tool_versions() -> dict[str, str]:
    versions = {}
    for line in CONSTRAINTS.read_text().splitlines():
        if line and not line.startswith("#"):
            name, version = line.split("==", maxsplit=1)
            versions[name] = version
    return versions


def _dev_extra() -> dict[str, str]:
    pyproject = (ROOT / "pyproject.toml").read_text()
    body = pyproject.split("dev = [", maxsplit=1)[1].split("]", maxsplit=1)[0]
    return dict(re.findall(r'"([\w-]+)==([\d.]+)"', body))


def test_dev_extra_matches_tool_constraints():
    assert _dev_extra() == _tool_versions()


def test_pre_commit_ruff_matches_tool_constraints():
    config = (ROOT / ".pre-commit-config.yaml").read_text()
    revision = re.search(r"rev: v([\d.]+)", config)

    assert revision is not None
    assert revision.group(1) == _tool_versions()["ruff"]


def test_workflow_tool_installs_use_shared_constraints():
    tools = _tool_versions()

    for workflow in (ROOT / ".github" / "workflows").glob("*.yml"):
        normalized = workflow.read_text().replace("\\\n", " ")
        for line in normalized.splitlines():
            if "-m pip install" not in line:
                continue
            referenced = {
                tool
                for tool in tools
                if re.search(rf"(?<![\w-]){re.escape(tool)}(?:[\s<=>]|$)", line)
            }
            if referenced:
                assert "-c requirements/tool-versions.txt" in line, (
                    f"{workflow.name} installs {sorted(referenced)} without the shared constraints"
                )
