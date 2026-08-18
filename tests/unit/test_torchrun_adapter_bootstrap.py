"""Tests for the torchrun adapter-bootstrap workflow."""

import os
import time
from pathlib import Path

import pytest

from examples.torchrun.adapter_bootstrap._validation import (
    ObserveTorchrunAdapterAttachment,
    assert_torchrun_adapter_attached,
)
from examples.torchrun.adapter_bootstrap.framework_worker import (
    FRAMEWORK_IMPORT_ROOTS,
)
from examples.torchrun.adapter_bootstrap.matrix import _write_worker_policy
from lm_resiliency.integrations.torchrun.worker_adapter import (
    TorchrunWorkerContext,
    _feature_options,
    _load_config,
)


def test_validation_asserts_torchrun_adapter_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED", "1")

    assert_torchrun_adapter_attached()


def test_validation_rejects_missing_torchrun_adapter_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED", raising=False)

    with pytest.raises(RuntimeError, match="worker adapter did not attach"):
        assert_torchrun_adapter_attached()


def test_pytorch_framework_import_uses_torch_package() -> None:
    assert FRAMEWORK_IMPORT_ROOTS["pytorch"] == "torch"


def test_validation_observer_latches_attachment_across_cleanup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED", raising=False)

    with ObserveTorchrunAdapterAttachment():
        monkeypatch.setenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED", "1")
        time.sleep(0.02)
        monkeypatch.delenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED")


def test_validation_observer_rejects_missing_attachment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED", raising=False)

    with pytest.raises(RuntimeError, match="worker adapter did not attach"):
        with ObserveTorchrunAdapterAttachment():
            assert "LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED" not in os.environ


def test_checked_in_smoke_policy_is_disabled(tmp_path: Path) -> None:
    policy = (
        Path(__file__).parents[2]
        / "examples"
        / "torchrun"
        / "adapter_bootstrap"
        / "policies"
        / "smoke.toml"
    )
    context = TorchrunWorkerContext(
        run_id="smoke-policy-test",
        node_id="node-a",
        local_world_size=1,
        restart_context_path=(tmp_path / "restart-context.json").resolve(),
    )

    checkpoint, replay, options = _feature_options(_load_config(policy), context)

    assert options["enable_checkpoint"] is False
    assert options["enable_detection"] is False
    assert checkpoint is None
    assert replay is None


def test_matrix_policy_scopes_checkpoints_to_output_directory(tmp_path: Path) -> None:
    policy = tmp_path / "worker.toml"
    _write_worker_policy(policy, replication_jump=1)
    context = TorchrunWorkerContext(
        run_id="matrix-policy-test",
        node_id="node-a",
        local_world_size=2,
        restart_context_path=(tmp_path / "restart-context.json").resolve(),
    )

    checkpoint, replay, _options = _feature_options(_load_config(policy), context)

    assert checkpoint is not None
    assert checkpoint.disk_folder == str((tmp_path / "checkpoints").resolve())
    assert replay is not None
