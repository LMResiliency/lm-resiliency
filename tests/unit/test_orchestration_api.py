"""Public GEMINI and SCOUT contracts consumed by external orchestrators."""

import inspect
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency import (
    InMemoryCkptConfig,
    ReplayHarnessConfig,
    ResiliencyHandle,
    enable_resiliency,
)
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import ReplayResult, StragglerDetail
from lm_resiliency.integrations.deepspeed import DeepSpeedResiliency
from lm_resiliency.integrations.deepspeed import (
    enable_resiliency as enable_deepspeed_resiliency,
)
from lm_resiliency.integrations.megatron import MegatronResiliency
from lm_resiliency.integrations.megatron import (
    enable_resiliency as enable_megatron_resiliency,
)
from lm_resiliency.integrations.torchtitan import (
    enable_resiliency as enable_torchtitan_resiliency,
)
from lm_resiliency.integrations.torchtitan.training import TorchTitanResiliency
from lm_resiliency.manager_api import (
    CheckpointTransfer,
    OrchestrationHooks,
    RecoveryDecision,
    RecoveryDecisionCallback,
    SCOUTFaultReport,
    TransferMetadataStore,
    make_transfer,
    replay_fault_reports,
)


def test_manager_api_contract_is_explicit():
    import lm_resiliency.manager_api as manager_api

    expected = {
        "CheckpointTransfer",
        "HardwareHealthMonitor",
        "HealthConfig",
        "HealthEvent",
        "HealthReading",
        "HealthSeverity",
        "HealthSource",
        "NvmlSource",
        "OrchestrationHooks",
        "RecoveryDecision",
        "RecoveryDecisionCallback",
        "RestartDestinationResolver",
        "SCOUTFaultCallback",
        "SCOUTFaultReport",
        "TransferMetadataStore",
        "dispatch_replay_faults",
        "find_drift",
        "format_drift",
        "local_fingerprint",
        "make_transfer",
        "replay_fault_reports",
    }

    assert set(manager_api.__all__) == expected
    for symbol in expected:
        assert getattr(manager_api, symbol) is not None


def test_public_orchestration_symbols_import_from_manager_api():
    assert CheckpointTransfer is not None
    assert OrchestrationHooks is not None
    assert RecoveryDecision is not None
    assert RecoveryDecisionCallback is not None
    assert TransferMetadataStore is not None
    assert SCOUTFaultReport is not None
    for handle_type in (
        ResiliencyHandle,
        DeepSpeedResiliency,
        MegatronResiliency,
        TorchTitanResiliency,
    ):
        assert callable(getattr(handle_type, "checkpoint_io"))
        assert callable(getattr(handle_type, "flush_for_restart"))
        assert callable(getattr(handle_type, "set_restart_destination"))
        assert callable(getattr(handle_type, "copy_checkpoint_to"))


def test_all_public_enable_functions_accept_orchestration_hooks():
    for enable in (
        enable_resiliency,
        enable_torchtitan_resiliency,
        enable_deepspeed_resiliency,
        enable_megatron_resiliency,
    ):
        assert "orchestration" in inspect.signature(enable).parameters


def test_replay_fault_reports_are_global_ranked_and_json_ready():
    result = ReplayResult(
        sdc_bitmap=[0, 1],
        straggler_bitmap=[0, 0],
        replay_time_ms=1.0,
        layer_id=7,
        peer_ranks=[4, 12],
        sdc_sources=["optimizer_updated_weight"],
        temporal_group_slowdown=True,
        straggler_confirmations=2,
        straggler_detail=StragglerDetail(
            straggler_rank=None,
            straggler_type="shared_compute",
            compute_times_ms=[],
            comm_times_ms=[],
            compute_bitmap=[0, 0],
        ),
    )

    reports = {report["kind"]: report for report in replay_fault_reports(result)}

    assert reports["sdc"] == {
        "failed_ranks": [12],
        "kind": "sdc",
        "scope": "rank",
        "layer_id": 7,
        "sources": ["optimizer_updated_weight"],
    }
    assert reports["straggler"]["failed_ranks"] == [4, 12]
    assert reports["straggler"]["scope"] == "peer_group"
    assert reports["straggler"]["confirmations"] == 2


def test_inconclusive_replay_precondition_reports_the_peer_group():
    result = ReplayResult(
        sdc_bitmap=[0, 0],
        straggler_bitmap=[0, 0],
        replay_time_ms=1.0,
        layer_id=4,
        peer_ranks=[6, 10],
        c3_results={
            "replay_input": C3Result(
                C3Status.INCONCLUSIVE,
                [0, 0],
                [101, 202],
            )
        },
    )

    assert replay_fault_reports(result) == [
        {
            "failed_ranks": [6, 10],
            "kind": "sdc",
            "scope": "peer_group",
            "layer_id": 4,
            "sources": ["replay_input"],
        }
    ]


def test_orchestration_hooks_normalize_replay_reports_and_bind_restart():
    reports = []

    def restart_destination():
        return "/restart"

    hooks = OrchestrationHooks(
        report_fault=reports.append,
        restart_destination=restart_destination,
    )
    hooks.replay_fault_callback(
        ReplayResult(
            sdc_bitmap=[0, 1],
            straggler_bitmap=[0, 0],
            replay_time_ms=1.0,
            layer_id=3,
            peer_ranks=[8, 9],
        )
    )
    handle = MagicMock()
    hooks.bind(handle)

    assert reports == [
        {
            "failed_ranks": [9],
            "kind": "sdc",
            "scope": "rank",
            "layer_id": 3,
            "sources": [],
        }
    ]
    handle.set_restart_destination.assert_called_once_with(restart_destination)


@patch("lm_resiliency._feature_wiring.ModelReplayHarness")
def test_unified_api_forwards_replay_and_oob_callbacks(harness_cls):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    replay_callback = MagicMock()
    oob_callback = MagicMock()

    handle = enable_resiliency(
        model,
        optimizer,
        enable_checkpoint=False,
        replay=ReplayHarnessConfig(check_interval=1),
        fault_callback=replay_callback,
        oob_fault_callback=oob_callback,
    )

    assert harness_cls.call_args.kwargs["callback"] is replay_callback
    assert harness_cls.call_args.kwargs["oob_fault_callback"] is oob_callback
    handle.close()


@patch("lm_resiliency._feature_wiring.ModelReplayHarness")
def test_unified_api_wires_one_orchestration_object(harness_cls):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report_fault = MagicMock()
    hooks = OrchestrationHooks(report_fault=report_fault)

    handle = enable_resiliency(
        model,
        optimizer,
        enable_checkpoint=False,
        replay=ReplayHarnessConfig(check_interval=1),
        orchestration=hooks,
    )

    replay_callback = harness_cls.call_args.kwargs["callback"]
    replay_callback(
        ReplayResult(
            sdc_bitmap=[1],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=2,
            peer_ranks=[6],
        )
    )
    report_fault.assert_called_once_with(
        {
            "failed_ranks": [6],
            "kind": "sdc",
            "scope": "rank",
            "layer_id": 2,
            "sources": [],
        }
    )
    assert harness_cls.call_args.kwargs["oob_fault_callback"] is report_fault
    handle.close()


@patch("lm_resiliency._feature_wiring.ModelReplayHarness")
def test_unified_api_wires_recovery_only_orchestration(harness_cls):
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    report_recovery = MagicMock()
    hooks = OrchestrationHooks(report_recovery=report_recovery)

    handle = enable_resiliency(
        model,
        optimizer,
        enable_checkpoint=False,
        replay=ReplayHarnessConfig(check_interval=1),
        orchestration=hooks,
    )

    replay_callback = harness_cls.call_args.kwargs["callback"]
    replay_callback(
        ReplayResult(
            sdc_bitmap=[1],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=2,
            peer_ranks=[6],
        )
    )

    decision = report_recovery.call_args.args[0]
    assert decision["failure_kind"] == "sdc"
    assert decision["recovery_mode"] == "recovery_verified"
    assert handle.last_recovery_decision == decision
    assert callable(harness_cls.call_args.kwargs["oob_fault_callback"])
    handle.close()


def test_unified_api_rejects_duplicate_orchestration_callbacks():
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    with pytest.raises(ValueError, match="orchestration.report_fault"):
        enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            enable_detection=False,
            fault_callback=lambda result: None,
            orchestration=OrchestrationHooks(report_fault=lambda report: None),
        )


def test_handle_exposes_restart_destination_and_explicit_copy():
    checkpoint = MagicMock()
    checkpoint.copy_to.return_value = 9
    handle = ResiliencyHandle()
    handle.ckpt_manager = checkpoint

    def resolver():
        return "/restart"

    handle.set_restart_destination(resolver)

    checkpoint.set_restart_destination.assert_called_once_with(resolver)
    assert handle.copy_checkpoint_to("/durable") == 9
    checkpoint.copy_to.assert_called_once_with("/durable")


def test_transfer_factory_validates_public_contract():
    with pytest.raises(ValueError, match="unsupported"):
        make_transfer("unknown", rank=0)
    with pytest.raises(ValueError, match="metadata_store"):
        make_transfer("nixl", rank=0)

    with patch("lm_resiliency.checkpointing.transfer.TorchDistTransfer") as transfer_cls:
        transfer = make_transfer("torch_dist", rank=3, chunk_size=1024)

    assert transfer is transfer_cls.return_value
    transfer_cls.assert_called_once_with(
        3,
        1024,
        timeout_s=120.0,
        process_group=None,
    )


def test_nixl_fallback_requires_explicit_two_sided_opt_in():
    store = MagicMock()
    with patch(
        "lm_resiliency.checkpointing.transfer.NixlCheckpointTransfer",
        side_effect=RuntimeError("missing nixl"),
    ):
        with pytest.raises(RuntimeError, match="fallback was not enabled"):
            make_transfer("nixl", rank=0, metadata_store=store)

    with (
        patch(
            "lm_resiliency.checkpointing.transfer.NixlCheckpointTransfer",
            side_effect=RuntimeError("missing nixl"),
        ),
        patch("lm_resiliency.checkpointing.transfer.TorchDistTransfer") as transfer_cls,
    ):
        transfer = make_transfer(
            "nixl",
            rank=0,
            metadata_store=store,
            allow_backend_fallback=True,
        )

    assert transfer is transfer_cls.return_value


def test_checkpoint_only_handle_has_no_platform_dependency(tmp_path):
    model = nn.Linear(4, 4)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    handle = enable_resiliency(
        model,
        optimizer,
        interval=1,
        enable_detection=False,
        checkpoint=InMemoryCkptConfig(
            interval=1,
            disk_flush_interval=0,
            disk_folder=str(tmp_path),
        ),
    )

    assert handle.ckpt_manager is not None
    handle.close()
