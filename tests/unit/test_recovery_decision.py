"""Tests for manager-facing checkpoint recovery decisions."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock

from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.handle import ResiliencyHandle
from lm_resiliency.orchestration import OrchestrationHooks
from lm_resiliency.recovery import build_recovery_decision


def test_prepare_recovery_emits_exact_gemini_checkpoint_selection():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 24
    decisions = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(report_recovery=decisions.append)
    hooks.bind(handle)

    mode = handle.prepare_recovery("straggler", all_ranks_accessible=True)

    assert mode is RecoveryMode.LATEST_GEMINI
    assert decisions == [
        {
            "failure_kind": "straggler",
            "recovery_mode": "latest",
            "checkpoint_source": "gemini",
            "checkpoint_step": 24,
            "checkpoint_id": None,
            "topology_digest": None,
            "all_ranks_accessible": True,
            "available": True,
            "reason": "accessible_straggler",
        }
    ]
    assert handle.last_recovery_decision == decisions[0]
    assert json.loads(json.dumps(decisions[0])) == decisions[0]
    manager.find_latest.assert_not_called()


def test_recovery_decision_reports_checkpoint_topology_digest():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 24
    manager.topology_id = "checkpoint-topology"

    decision = build_recovery_decision(
        failure_kind="straggler",
        recovery_mode=RecoveryMode.LATEST_GEMINI,
        all_ranks_accessible=True,
        reason="accessible_straggler",
        checkpoint_manager=manager,
        durable_checkpoint=None,
        allow_collective=False,
    )

    assert decision["topology_digest"] == "checkpoint-topology"


def test_replay_sdc_emits_recovery_decision_before_fault_report():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 20
    events = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(
        report_fault=lambda report: events.append(("fault", report)),
        report_recovery=lambda decision: events.append(("recovery", decision)),
    )
    hooks.bind(handle)

    hooks.replay_fault_callback(
        ReplayResult(
            sdc_bitmap=[0, 1],
            straggler_bitmap=[0, 0],
            replay_time_ms=1.0,
            layer_id=2,
            peer_ranks=[4, 7],
        )
    )

    assert [kind for kind, _ in events] == ["recovery", "fault"]
    decision = events[0][1]
    assert decision["failure_kind"] == "sdc"
    assert decision["recovery_mode"] == "recovery_verified"
    assert decision["checkpoint_step"] == 20
    assert handle.last_recovery_decision == decision
    assert events[1][1]["kind"] == "sdc"
    manager.find_latest.assert_not_called()


def test_replay_straggler_emits_latest_checkpoint_decision():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 21
    decisions = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(report_recovery=decisions.append)
    hooks.bind(handle)

    hooks.replay_fault_callback(
        ReplayResult(
            sdc_bitmap=[0, 0],
            straggler_bitmap=[1, 0],
            replay_time_ms=2.0,
            layer_id=3,
            peer_ranks=[4, 7],
        )
    )

    assert decisions == [
        {
            "failure_kind": "straggler",
            "recovery_mode": "latest",
            "checkpoint_source": "gemini",
            "checkpoint_step": 21,
            "checkpoint_id": None,
            "topology_digest": None,
            "all_ranks_accessible": True,
            "available": True,
            "reason": "replay_straggler_allows_latest_recovery",
        }
    ]
    assert handle.last_recovery_decision == decisions[0]


def test_oob_hang_emits_conservative_noncollective_decision():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 18
    faults = []
    decisions = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(
        report_fault=faults.append,
        report_recovery=decisions.append,
    )
    hooks.bind(handle)
    report = {"kind": "hang", "failed_ranks": [3], "scope": "rank"}

    hooks.oob_fault_callback(report)

    assert faults == [report]
    assert decisions == [
        {
            "failure_kind": "hang",
            "recovery_mode": "recovery_verified",
            "checkpoint_source": "gemini",
            "checkpoint_step": 18,
            "checkpoint_id": None,
            "topology_digest": None,
            "all_ranks_accessible": False,
            "available": True,
            "reason": "oob_hang_requires_conservative_recovery",
        }
    ]
    assert handle.last_recovery_decision == decisions[0]
    manager.find_latest.assert_not_called()
    manager.set_recovery_mode.assert_not_called()


def test_dataloader_stall_emits_latest_checkpoint_decision():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 19
    faults = []
    decisions = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(
        report_fault=faults.append,
        report_recovery=decisions.append,
    )
    hooks.bind(handle)
    report = {"kind": "data_stall", "failed_ranks": [2], "scope": "rank"}

    hooks.oob_fault_callback(report)

    assert faults == [report]
    assert decisions == [
        {
            "failure_kind": "data_stall",
            "recovery_mode": "latest",
            "checkpoint_source": "gemini",
            "checkpoint_step": 19,
            "checkpoint_id": None,
            "topology_digest": None,
            "all_ranks_accessible": True,
            "available": True,
            "reason": "dataloader_stall_allows_latest_recovery",
        }
    ]
    assert handle.last_recovery_decision == decisions[0]


def test_checkpoint_stall_emits_latest_checkpoint_decision():
    manager = MagicMock()
    manager.local_recovery_step.return_value = 21
    faults = []
    decisions = []
    handle = ResiliencyHandle()
    handle.ckpt_manager = manager
    hooks = OrchestrationHooks(
        report_fault=faults.append,
        report_recovery=decisions.append,
    )
    hooks.bind(handle)
    report = {
        "kind": "checkpoint_stall",
        "failed_ranks": [3],
        "scope": "rank",
        "stage_kind": "checkpoint_write",
    }

    hooks.oob_fault_callback(report)

    assert faults == [report]
    assert decisions == [
        {
            "failure_kind": "checkpoint_stall",
            "recovery_mode": "latest",
            "checkpoint_source": "gemini",
            "checkpoint_step": 21,
            "checkpoint_id": None,
            "topology_digest": None,
            "all_ranks_accessible": True,
            "available": True,
            "reason": "checkpoint_io_stall_allows_latest_recovery",
        }
    ]
    assert handle.last_recovery_decision == decisions[0]


def test_decision_falls_back_to_durable_checkpoint_identity():
    record = SimpleNamespace(step=11, checkpoint_id="scout-step-11-epoch-2-plan")
    durable = SimpleNamespace(latest_validated=record)

    decision = build_recovery_decision(
        failure_kind="sdc",
        recovery_mode=RecoveryMode.RECOVERY_VERIFIED,
        all_ranks_accessible=False,
        reason="sdc_detected",
        checkpoint_manager=None,
        durable_checkpoint=durable,
        allow_collective=False,
    )

    assert decision["checkpoint_source"] == "durable"
    assert decision["checkpoint_step"] == 11
    assert decision["checkpoint_id"] == record.checkpoint_id
    assert decision["available"] is True


def test_decision_explicitly_reports_when_no_checkpoint_is_available():
    decision = build_recovery_decision(
        failure_kind="machine_unavailable",
        recovery_mode=RecoveryMode.RECOVERY_VERIFIED,
        all_ranks_accessible=False,
        reason="required_machine_unavailable",
        checkpoint_manager=None,
        durable_checkpoint=None,
        allow_collective=False,
    )

    assert decision["checkpoint_source"] == "none"
    assert decision["checkpoint_step"] == -1
    assert decision["checkpoint_id"] is None
    assert decision["available"] is False
