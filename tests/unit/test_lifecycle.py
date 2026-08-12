"""Tests for process-level resiliency cleanup."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import torch
import torch.nn as nn

from lm_resiliency.integrations.pytorch import enable_resiliency
from lm_resiliency.lifecycle import (
    _close_registered_handles,
    _discard_inherited_handles,
    register_automatic_cleanup,
    unregister_automatic_cleanup,
)


@pytest.fixture(autouse=True)
def isolate_cleanup_registry():
    _close_registered_handles()
    yield
    _close_registered_handles()


def test_registered_handles_close_in_reverse_order():
    events = []
    first = MagicMock()
    first.close.side_effect = lambda: events.append("first")
    second = MagicMock()
    second.close.side_effect = lambda: events.append("second")

    register_automatic_cleanup(first)
    register_automatic_cleanup(second)
    _close_registered_handles()

    assert events == ["second", "first"]


def test_duplicate_registration_closes_handle_once():
    handle = MagicMock()

    register_automatic_cleanup(handle)
    register_automatic_cleanup(handle)
    _close_registered_handles()

    handle.close.assert_called_once_with()


def test_explicit_unregister_prevents_automatic_close():
    handle = MagicMock()

    register_automatic_cleanup(handle)
    unregister_automatic_cleanup(handle)
    _close_registered_handles()

    handle.close.assert_not_called()


def test_cleanup_continues_after_one_handle_fails(caplog):
    healthy = MagicMock()
    failing = MagicMock()
    failing.close.side_effect = RuntimeError("cleanup failed")
    register_automatic_cleanup(healthy)
    register_automatic_cleanup(failing)

    _close_registered_handles()

    failing.close.assert_called_once_with()
    healthy.close.assert_called_once_with()
    assert "Automatic resiliency cleanup failed" in caplog.text


def test_fork_reset_discards_parent_handles_and_accepts_child_handles():
    parent_handle = MagicMock()
    child_handle = MagicMock()
    register_automatic_cleanup(parent_handle)

    _discard_inherited_handles()
    register_automatic_cleanup(child_handle)
    _close_registered_handles()

    parent_handle.close.assert_not_called()
    child_handle.close.assert_called_once_with()


def test_pytorch_handle_closes_without_explicit_user_call():
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    handle = enable_resiliency(
        model,
        optimizer,
        enable_checkpoint=False,
        enable_detection=False,
    )

    _close_registered_handles()
    optimizer.step()

    assert handle.closed
    assert handle.step_count == 0


def test_registered_handle_closes_at_real_interpreter_exit(tmp_path):
    marker = tmp_path / "closed"
    script = f"""
from pathlib import Path
from lm_resiliency.lifecycle import register_automatic_cleanup

class Handle:
    def close(self):
        Path({str(marker)!r}).write_text("closed")

register_automatic_cleanup(Handle())
"""

    subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).parents[2],
        check=True,
    )

    assert marker.read_text() == "closed"
