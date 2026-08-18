import ast
import json
from pathlib import Path

import pytest

from examples.production_loops._common import (
    parse_run_arguments,
    torchrun_checkpoint_step,
    training_step_range,
    write_validation_summary,
)
from lm_resiliency.integrations.torchrun.worker_adapter import (
    TorchrunWorkerContext,
    _feature_options,
    _load_config,
)


def test_shared_run_arguments_prepare_validation_output_dir(tmp_path: Path) -> None:
    output_dir = tmp_path / "validation"

    arguments = parse_run_arguments(["--validation-output-dir", str(output_dir), "--steps", "3"])

    assert arguments.validation_output_dir == output_dir
    assert arguments.steps == 3
    assert output_dir.is_dir()


def test_shared_summary_writer_publishes_framework_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    summary = {"framework": "pytorch", "steps": 3}

    write_validation_summary(tmp_path, summary, writer=True)

    summary_path = tmp_path / "pytorch-production-loop.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == summary
    assert json.loads(capsys.readouterr().out) == summary


def test_shared_summary_writer_skips_non_writer_rank(tmp_path: Path) -> None:
    write_validation_summary(
        tmp_path,
        {"framework": "pytorch", "steps": 3},
        writer=False,
    )

    assert not list(tmp_path.iterdir())


def test_shared_training_range_resumes_from_torchrun_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP", "4")

    assert torchrun_checkpoint_step() == 4
    assert list(training_step_range(torchrun_checkpoint_step(), 7)) == [4, 5, 6]


@pytest.mark.parametrize(
    ("current_step", "target_step"),
    [
        (4, 4),
        (5, 4),
        (-1, 4),
    ],
)
def test_shared_training_range_rejects_invalid_resume_target(
    current_step: int,
    target_step: int,
) -> None:
    with pytest.raises(ValueError):
        training_step_range(current_step, target_step)


@pytest.mark.parametrize(
    "module_name",
    [
        "deepspeed.py",
        "megatron.py",
        "pytorch.py",
        "torchtitan.py",
    ],
)
def test_user_training_examples_have_no_lm_resiliency_imports(
    module_name: str,
) -> None:
    source_path = Path(__file__).parents[2] / "examples" / "production_loops" / module_name
    source = source_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(source_path))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not any(
        module == "lm_resiliency" or module.startswith("lm_resiliency.")
        for module in imported_modules
    )
    assert "LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED" not in source
    assert "assert_torchrun_adapter_attached" not in source
    assert "--max-restarts=4" in source


def test_checked_in_production_policy_is_valid(tmp_path: Path) -> None:
    policy = (
        Path(__file__).parents[2] / "examples" / "production_loops" / "policies" / "resiliency.toml"
    )
    context = TorchrunWorkerContext(
        run_id="production-policy-test",
        node_id="node-a",
        local_world_size=1,
        restart_context_path=(tmp_path / "restart-context.json").resolve(),
    )

    checkpoint, replay, options = _feature_options(_load_config(policy), context)

    assert options == {
        "enable_checkpoint": True,
        "enable_detection": True,
        "interval": 1,
    }
    assert checkpoint is not None
    assert checkpoint.replication_jump == 4
    assert checkpoint.disk_flush_interval == 0
    assert replay is not None
    assert replay.rotate_layers is False
