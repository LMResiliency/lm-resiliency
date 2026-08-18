"""Tests for the torchrun adapter-bootstrap workflow."""

from pathlib import Path

import pytest

from examples.torchrun.adapter_bootstrap._validation import (
    assert_torchrun_adapter_attached,
)
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
