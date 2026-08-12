"""Unit tests for OOB daemon rendezvous setup."""

from datetime import timedelta
from unittest.mock import patch

import torch.distributed as dist

from lm_resiliency.detection.oob_service import (
    OOBHangConfig,
    _init_daemon_process_group,
    _rendezvous_method,
)


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
