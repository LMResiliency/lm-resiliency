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


def test_health_check_reports_post_start_child_exit():
    service = _unstarted_service()
    service._ready_event.set()
    service._process = Mock(is_alive=Mock(return_value=False), exitcode=23)

    with pytest.raises(RuntimeError, match="exited unexpectedly with code 23"):
        service.ensure_healthy()


def test_supervisor_publishes_child_failure_to_callback_queue():
    callback = Mock()
    service = OOBHangService(
        global_rank=3,
        peer_ranks=[2, 3],
        config=OOBHangConfig(),
        report_callback=callback,
    )
    process = Mock(exitcode=9)

    service._supervise_child(process)

    report = service._report_queue.get(timeout=1.0)
    assert report["kind"] == "oob_daemon_failure"
    assert report["failed_ranks"] == [3]
    assert report["exit_code"] == 9
    service._report_queue.close()


def test_tracker_channel_is_namespaced_by_run_generation_process_and_rank(monkeypatch):
    monkeypatch.setenv("LM_RUN_ID", "job-a")
    monkeypatch.setenv("TORCHELASTIC_RESTART_COUNT", "4")

    rank_zero = OOBHangService(global_rank=0, peer_ranks=[0, 1], config=OOBHangConfig())
    rank_one = OOBHangService(global_rank=1, peer_ranks=[0, 1], config=OOBHangConfig())
    assert rank_zero.tracker_name != rank_one.tracker_name
    assert rank_zero.tracker_name.startswith("scout_op_")
    assert "job-a" not in rank_zero.tracker_name
    assert rank_zero.tracker_token != rank_one.tracker_token


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
