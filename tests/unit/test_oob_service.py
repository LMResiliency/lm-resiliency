"""Unit tests for OOB daemon rendezvous setup."""

from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
import torch.distributed as dist

from lm_resiliency.detection.oob_service import (
    OOBHangConfig,
    OOBHangService,
    _init_daemon_process_group,
    _rendezvous_method,
)


def _unstarted_service() -> OOBHangService:
    return OOBHangService(global_rank=0, peer_ranks=[0], config=OOBHangConfig())


def test_readiness_wait_uses_child_signal():
    service = _unstarted_service()
    service._process = Mock(is_alive=Mock(return_value=True), exitcode=None)
    service._ready_event.set()

    service.wait_until_ready(timeout_s=0.01)


def test_readiness_wait_reports_live_child_timeout():
    service = _unstarted_service()
    service._process = Mock(is_alive=Mock(return_value=True), exitcode=None)

    with pytest.raises(TimeoutError, match="not ready within"):
        service.wait_until_ready(timeout_s=0.0)


def test_readiness_wait_reports_early_child_exit():
    service = _unstarted_service()
    service._process = Mock(is_alive=Mock(return_value=False), exitcode=17)

    with pytest.raises(RuntimeError, match="exited before readiness with code 17"):
        service.wait_until_ready(timeout_s=0.0)


def test_tcp_rendezvous_owns_a_store_instead_of_reusing_torchrun_agent_store():
    timeout = timedelta(seconds=10)
    store = object()
    with (
        patch.object(dist, "TCPStore", return_value=store) as tcp_store,
        patch.object(dist, "init_process_group") as init_process_group,
    ):
        result = _init_daemon_process_group(
            init_method="tcp://127.0.0.1:30100",
            address="127.0.0.1",
            port=30100,
            local_rank=0,
            world_size=4,
            timeout=timeout,
        )

    assert result is store
    tcp_store.assert_called_once_with("127.0.0.1", 30100, 4, True, timeout)
    init_process_group.assert_called_once_with(
        backend="gloo",
        store=store,
        rank=0,
        world_size=4,
        timeout=timeout,
    )


def test_file_rendezvous_keeps_init_method_path():
    timeout = timedelta(seconds=10)
    with patch.object(dist, "init_process_group") as init_process_group:
        result = _init_daemon_process_group(
            init_method="file:///tmp/scout-oob",
            address="ignored",
            port=0,
            local_rank=1,
            world_size=2,
            timeout=timeout,
        )

    assert result is None
    init_process_group.assert_called_once_with(
        backend="gloo",
        init_method="file:///tmp/scout-oob",
        rank=1,
        world_size=2,
        timeout=timeout,
    )


def test_explicit_tcp_endpoint_takes_precedence_over_state_directory():
    config = OOBHangConfig(
        state_dir="/tmp/scout-oob",
        master_addr="10.0.0.1",
        master_port=29600,
    )

    assert _rendezvous_method(config, [0, 1, 2, 3], "10.0.0.1", 29600) == "tcp://10.0.0.1:29600"


def test_state_directory_remains_file_rendezvous_fallback():
    config = OOBHangConfig(state_dir="/tmp/scout-oob")

    method = _rendezvous_method(config, [0, 1, 2, 3], "127.0.0.1", 29600)

    assert method.startswith("file:///tmp/scout-oob/oob_rendezvous/")
