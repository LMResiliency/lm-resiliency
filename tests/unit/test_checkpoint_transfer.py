"""Checkpoint transfer contract and timeout tests."""

from __future__ import annotations

import socket
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest
import torch

from lm_resiliency.checkpointing.transfer import (
    NixlCheckpointTransfer,
    TorchDistTransfer,
    _payload_manifest,
    _transfer_tag,
    _validate_manifest,
)


def test_manifest_binds_key_endpoints_host_and_tensor_layout():
    tensor = torch.ones(4, dtype=torch.float32)
    manifest = _payload_manifest(
        [tensor.unsqueeze(0)],
        key="step-7",
        source_rank=3,
        destination_rank=9,
        destination_host=socket.gethostname(),
    )

    _validate_manifest(
        manifest,
        [tensor.unsqueeze(0)],
        key="step-7",
        source_rank=3,
        destination_rank=9,
        peer_host=socket.gethostname(),
    )
    with pytest.raises(RuntimeError, match="key mismatch"):
        _validate_manifest(
            manifest,
            [tensor.unsqueeze(0)],
            key="step-8",
            source_rank=3,
            destination_rank=9,
            peer_host=socket.gethostname(),
        )
    with pytest.raises(RuntimeError, match="tensor shape"):
        _validate_manifest(
            manifest,
            [torch.ones(3)],
            key="step-7",
            source_rank=3,
            destination_rank=9,
            peer_host=socket.gethostname(),
        )


def test_transfer_tags_are_stable_and_keyed():
    assert _transfer_tag("owner/step-7") == _transfer_tag("owner/step-7")
    assert _transfer_tag("owner/step-7") != _transfer_tag("peer/step-7")


def _nixl_transfer(state: str) -> tuple[NixlCheckpointTransfer, MagicMock]:
    buffer = torch.zeros(4)
    manifest = _payload_manifest(
        [buffer],
        key="step-7",
        source_rank=1,
        destination_rank=0,
        destination_host=socket.gethostname(),
    )
    control = MagicMock()
    control.get_transfer_meta.return_value = {
        "transport": "nixl",
        "manifest": manifest,
        "agent_meta": "agent",
        "descs": "descs",
    }
    agent = MagicMock()
    agent.check_xfer_state.return_value = state
    transfer = object.__new__(NixlCheckpointTransfer)
    transfer._control = control
    transfer._agent = agent
    transfer._rank = 0
    transfer._timeout_s = 1.0
    transfer._poll_interval_s = 0.001
    return transfer, agent


def test_nixl_terminal_failure_is_not_polled_forever():
    transfer, agent = _nixl_transfer("FAILED")

    with pytest.raises(RuntimeError, match="failed"):
        transfer.fetch([torch.zeros(4)], "step-7", peer_rank=1, peer_host=socket.gethostname())

    agent.deregister_memory.assert_called_once()


def test_nixl_timeout_cancels_transfer():
    transfer, agent = _nixl_transfer("PENDING")

    with (
        patch(
            "lm_resiliency.checkpointing.transfer.time.monotonic",
            side_effect=[10.0, 11.1],
        ),
        pytest.raises(TimeoutError, match="exceeded"),
    ):
        transfer.fetch([torch.zeros(4)], "step-7", peer_rank=1, peer_host=socket.gethostname())

    agent.cancel_xfer.assert_called_once()
    agent.deregister_memory.assert_called_once()


def test_torch_dist_wait_has_a_bounded_failure():
    transfer = object.__new__(TorchDistTransfer)
    transfer._timeout_s = 2.0
    transfer._wait_timeout = timedelta(seconds=2.0)
    work = MagicMock()
    work.wait.return_value = False

    with pytest.raises(TimeoutError, match="exceeded 2.0s"):
        transfer._wait(work, operation="manifest")
