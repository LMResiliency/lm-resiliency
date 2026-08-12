"""Tests for the framework-shared checkpoint validation workflow."""

import logging
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointConfig,
    DurableCheckpointCoordinator,
)
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.integrations._checkpoint_certification import (
    CheckpointCertificationCoordinator,
)


def _result(
    shape_id: str = "captured",
    *,
    sdc: bool = False,
    straggler: bool = False,
    cycle_complete: bool = False,
    full_cycle: bool = False,
    cycle_size: int = 1,
    dense: bool = True,
) -> ReplayResult:
    return ReplayResult(
        sdc_bitmap=[int(sdc)],
        straggler_bitmap=[int(straggler)],
        replay_time_ms=1.0,
        layer_id=0,
        checked_shape_ids=[shape_id],
        completed_shape_cycle=full_cycle,
        completed_scheduled_cycle=cycle_complete,
        shape_cycle_size=cycle_size,
        dense_replay=dense,
    )


def _coordinator(
    manager,
    harness,
    *,
    cadence: ResiliencyCadence,
    durable=None,
) -> CheckpointCertificationCoordinator:
    return CheckpointCertificationCoordinator(
        checkpoint_manager=manager,
        replay_harness=harness,
        durable_checkpoint=durable,
        cadence=cadence,
        checkpoint_tensors=lambda: [torch.tensor([1.0])],
        checkpoint_extra=lambda: {"cursor": 7},
        fault_callback=None,
        logger=logging.getLogger(__name__),
    )


def _durable(tmp_path, shape_ids):
    adapter = SimpleNamespace(
        save_candidate=MagicMock(return_value={"path": "candidate"}),
        load_checkpoint=MagicMock(),
        commit_candidate=MagicMock(),
        quarantine_candidate=MagicMock(),
    )
    durable = DurableCheckpointCoordinator(
        DurableCheckpointConfig(
            manifest_dir=str(tmp_path),
            environment_id="test",
            adapter=adapter,
        ),
        shape_plan_id="test-plan",
        shape_ids=shape_ids,
    )
    return durable, adapter


def test_scheduled_check_saves_without_running_full_shape_cycle():
    manager = MagicMock()
    harness = MagicMock()
    harness.step.side_effect = [None, _result(cycle_complete=True)]
    harness.current_replay_shape.shape_id = "captured"
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=2,
            detection_interval=2,
        ),
    )

    coordinator.post_step(1)
    coordinator.post_step(2)

    manager.save_tensors.assert_called_once()
    manager.persist_verified_boundary.assert_called_once_with(2)
    manager.persist_cycle_boundary.assert_not_called()
    assert "complete_shape_cycle" not in harness.step.call_args_list[1].kwargs


def test_sdc_skips_capture_and_rejects_candidate():
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result(sdc=True)
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
    )

    coordinator.post_step(1)

    manager.save_tensors.assert_not_called()
    manager.reject_candidate.assert_called_once()


def test_dense_recipe_result_between_base_intervals_is_not_ignored():
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result(sdc=True)
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=10,
            detection_interval=10,
        ),
    )

    coordinator.post_step(2)

    manager.reject_candidate.assert_called_once()
    manager.save_tensors.assert_not_called()


def test_dense_recipe_result_does_not_create_durable_checkpoint_off_cadence(tmp_path):
    durable, adapter = _durable(tmp_path, ["captured"])
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result(cycle_complete=True)
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=10,
            detection_interval=10,
        ),
        durable=durable,
    )

    coordinator.post_step(2)

    adapter.save_candidate.assert_not_called()
    manager.save_tensors.assert_not_called()
    manager.persist_verified_boundary.assert_not_called()


@patch(
    "lm_resiliency.integrations._checkpoint_certification._global_replay_consensus",
    return_value=SimpleNamespace(
        all_results_available=True,
        any_sdc=True,
        scheduled_cycles_agree=True,
        scheduled_cycle_complete=False,
        all_shape_cycles_complete=True,
    ),
)
def test_sdc_in_another_peer_group_rejects_candidate_globally(_consensus):
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result()
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
    )

    coordinator.post_step(1)

    manager.save_tensors.assert_not_called()
    manager.reject_candidate.assert_called_once()


@patch(
    "lm_resiliency.integrations._checkpoint_certification._global_replay_consensus",
    return_value=SimpleNamespace(
        all_results_available=True,
        any_sdc=False,
        scheduled_cycles_agree=False,
        scheduled_cycle_complete=False,
        all_shape_cycles_complete=False,
    ),
)
def test_recipe_cycle_disagreement_fails_before_checkpoint_transition(_consensus):
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result(cycle_complete=True)
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
    )

    with pytest.raises(RuntimeError, match="recipe schedulers disagree"):
        coordinator.post_step(1)

    manager.save_tensors.assert_not_called()
    manager.persist_cycle_boundary.assert_not_called()


def test_completed_straggler_still_saves_checkpoint():
    manager = MagicMock()
    harness = MagicMock()
    harness.step.return_value = _result(straggler=True, cycle_complete=True)
    harness.current_replay_shape.shape_id = "captured"
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
    )

    coordinator.post_step(1)

    manager.save_tensors.assert_called_once()
    manager.persist_verified_boundary.assert_called_once_with(1)


def test_cross_pg_localization_runs_before_fault_reporting():
    manager = MagicMock()
    harness = MagicMock()
    result = _result(straggler=True, cycle_complete=True)
    harness.step.return_value = result
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
    )

    coordinator.post_step(1)

    harness.finalize_communication_localization.assert_called_once_with(result)


def test_dense_checkpoint_is_verified_after_each_accepted_check(tmp_path):
    durable, adapter = _durable(tmp_path, ["captured"])
    manager = MagicMock()
    harness = MagicMock()
    harness.current_replay_shape.shape_id = "captured"
    harness.step.side_effect = [
        _result(cycle_complete=True),
        _result(cycle_complete=True),
    ]
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
        durable=durable,
    )

    coordinator.post_step(1)

    assert durable.pending is None
    assert durable.latest_validated is not None
    assert durable.latest_validated.step == 1
    adapter.commit_candidate.assert_called_once()

    coordinator.post_step(2)

    assert durable.latest_validated is not None
    assert durable.latest_validated.step == 2
    assert durable.pending is None
    assert manager.save_tensors.call_count == 2
    assert [call.args[0] for call in manager.persist_verified_boundary.call_args_list] == [1, 2]
    manager.persist_cycle_boundary.assert_not_called()
    assert adapter.commit_candidate.call_count == 2


def test_moe_checkpoint_rotates_one_recipe_and_promotes_after_2k(tmp_path):
    durable, adapter = _durable(tmp_path, ["small", "large"])
    manager = MagicMock()
    harness = MagicMock()
    harness.current_replay_shape.shape_id = "small"
    harness.step.side_effect = [
        _result("small", cycle_size=2, dense=False),
        _result("large", cycle_complete=True, cycle_size=2, dense=False),
        _result("small", cycle_size=2, dense=False),
        _result("large", cycle_complete=True, cycle_size=2, dense=False),
    ]
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
        durable=durable,
    )

    for step in range(1, 5):
        coordinator.post_step(step)

    assert manager.save_tensors.call_count == 4
    assert [call.args[0] for call in manager.persist_cycle_boundary.call_args_list] == [2, 4]
    assert len(adapter.save_candidate.call_args_list) == 2
    assert adapter.save_candidate.call_args_list[0].args[0].step == 2
    assert adapter.save_candidate.call_args_list[1].args[0].step == 4
    assert durable.latest_validated is not None
    assert durable.latest_validated.step == 2
    assert durable.pending is not None
    assert durable.pending.step == 4


def test_one_recipe_dynamic_catalog_still_requires_followup_cycle(tmp_path):
    durable, adapter = _durable(tmp_path, ["only"])
    manager = MagicMock()
    harness = MagicMock()
    harness.current_replay_shape.shape_id = "only"
    harness.step.side_effect = [
        _result(
            "only",
            cycle_complete=True,
            full_cycle=True,
            cycle_size=1,
            dense=False,
        ),
        _result(
            "only",
            cycle_complete=True,
            full_cycle=True,
            cycle_size=1,
            dense=False,
        ),
    ]
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
        durable=durable,
    )

    coordinator.post_step(1)

    assert durable.pending is not None
    assert durable.pending.step == 1
    assert durable.latest_validated is None
    adapter.commit_candidate.assert_not_called()

    coordinator.post_step(2)

    assert durable.latest_validated is not None
    assert durable.latest_validated.step == 1
    assert durable.pending is not None
    assert durable.pending.step == 2
    assert [call.args[0] for call in manager.persist_cycle_boundary.call_args_list] == [1, 2]
    manager.persist_verified_boundary.assert_not_called()
    adapter.commit_candidate.assert_called_once()


def test_failure_recovery_modes_follow_fault_classification():
    manager = MagicMock()
    harness = MagicMock()
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence(interval=1, checkpoint_enabled=True, detection_enabled=True),
    )
    check_all = MagicMock(return_value=_result(full_cycle=True))

    hang_mode = coordinator.prepare_recovery(
        failure_kind="hang",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )
    straggler_mode = coordinator.prepare_recovery(
        failure_kind="straggler",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )
    disconnected_mode = coordinator.prepare_recovery(
        failure_kind="uncertain",
        all_ranks_accessible=False,
        check_all_recipes=check_all,
        step=24,
    )
    confirmed_sdc_mode = coordinator.prepare_recovery(
        failure_kind="sdc",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )
    clean_mode = coordinator.prepare_recovery(
        failure_kind="uncertain",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )
    check_all.return_value = _result(full_cycle=False)
    incomplete_mode = coordinator.prepare_recovery(
        failure_kind="uncertain",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )
    check_all.return_value = _result(sdc=True, full_cycle=False)
    sdc_mode = coordinator.prepare_recovery(
        failure_kind="uncertain",
        all_ranks_accessible=True,
        check_all_recipes=check_all,
        step=24,
    )

    assert hang_mode is RecoveryMode.LATEST_GEMINI
    assert straggler_mode is RecoveryMode.LATEST_GEMINI
    assert disconnected_mode is RecoveryMode.RECOVERY_VERIFIED
    assert confirmed_sdc_mode is RecoveryMode.RECOVERY_VERIFIED
    assert clean_mode is RecoveryMode.LATEST_GEMINI
    assert incomplete_mode is RecoveryMode.RECOVERY_VERIFIED
    assert sdc_mode is RecoveryMode.RECOVERY_VERIFIED
    assert check_all.call_count == 4
    assert manager.set_recovery_mode.call_count == 3
    assert manager.reject_candidate.call_count == 4


def test_durable_candidate_write_failure_clears_gemini_candidate(tmp_path):
    durable, adapter = _durable(tmp_path, ["captured"])
    adapter.save_candidate.side_effect = RuntimeError("durable write failed")
    manager = MagicMock()
    harness = MagicMock()
    harness.current_replay_shape.shape_id = "captured"
    harness.step.return_value = _result(cycle_complete=True)
    coordinator = _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
        durable=durable,
    )

    with pytest.raises(RuntimeError, match="durable write failed"):
        coordinator.post_step(1)

    manager.persist_verified_boundary.assert_called_once_with(1)
    manager.clear_candidate.assert_called_once()


def test_restarted_durable_window_clears_unmatched_gemini_candidate(tmp_path):
    durable, _ = _durable(tmp_path, ["captured"])
    manager = MagicMock()
    manager.checkpoint_status = SimpleNamespace(candidate_step=10)
    harness = MagicMock()

    _coordinator(
        manager,
        harness,
        cadence=ResiliencyCadence.from_component_intervals(
            checkpoint_interval=1,
            detection_interval=1,
        ),
        durable=durable,
    )

    manager.clear_candidate.assert_called_once()
