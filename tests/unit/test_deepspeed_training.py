"""Unit tests for DeepSpeed training integration (GEMINI hooks)."""

from __future__ import annotations

import tempfile
from types import MethodType
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.manager import RecoveryMode
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.optimizer_step import (
    OPTIMIZER_STATUS_OK,
    OptimizerReplayBatch,
)
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig
from lm_resiliency.integrations._common import prepare_checkpoint_tensor_load
from lm_resiliency.integrations.deepspeed.training import (
    DeepSpeedResiliency,
    enable_deepspeed_resiliency,
)


class FakeBaseOptimizer:
    def __init__(self, params: list[torch.Tensor]):
        self.param_groups = [{"params": params}]
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for p in params:
            self.state[p] = {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
            }


class FakeZeROOptimizer:
    def __init__(self, hidden_size: int = 32):
        self.bit16_groups_flat = [torch.randn(hidden_size, dtype=torch.bfloat16)]
        self.single_partition_of_fp32_groups = [torch.randn(hidden_size // 2, dtype=torch.float32)]
        self.optimizer = FakeBaseOptimizer(self.single_partition_of_fp32_groups)


class FakeDeepSpeedEngine:
    def __init__(self, hidden_size: int = 32):
        self.module = nn.Linear(hidden_size, hidden_size)
        self.optimizer = FakeZeROOptimizer(hidden_size)
        self.global_rank = 0
        self.world_size = 1
        self.dp_world_size = 1
        self.data_parallel_group = None
        self.global_steps = 0
        self.lr_scheduler = None
        self._step_called = False

    def zero_optimization_stage(self) -> int:
        return 2

    def step(self, lr_kwargs=None):
        self._step_called = True
        self.global_steps += 1


class OptimizerStep:
    pass


class PipelineEngine(FakeDeepSpeedEngine):
    def __init__(self, hidden_size: int = 32):
        super().__init__(hidden_size)
        self.optimizer_steps = 0

    def _exec_optimizer_step(self, lr_kwargs=None):
        del lr_kwargs
        self.optimizer_steps += 1
        self.global_steps += 1

    def step(self, *args, **kwargs):
        raise RuntimeError("PipelineEngine.step() is disabled")

    _INSTRUCTION_MAP = {OptimizerStep: _exec_optimizer_step}


def test_checkpoint_load_materializes_only_saved_optimizer_state():
    adapter = MagicMock()
    adapter.collect_checkpoint_tensors.return_value = [torch.zeros(1), torch.zeros(1)]

    prepare_checkpoint_tensor_load(adapter, [torch.zeros(1), torch.zeros(1)])
    adapter.materialize_optimizer_state.assert_not_called()

    prepare_checkpoint_tensor_load(adapter, [torch.zeros(1)] * 5)
    adapter.materialize_optimizer_state.assert_called_once()


class TestDeepSpeedResiliencyInit:
    def test_wraps_engine_step(self):
        engine = FakeDeepSpeedEngine()
        original_step = engine.step

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
            )

        # step should be replaced
        assert engine.step != original_step
        resiliency.close()
        # restored after close
        assert engine.step == original_step

    def test_step_increments_count(self):
        engine = FakeDeepSpeedEngine()

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
            )

        engine.step()
        assert resiliency.step_count == 1
        engine.step()
        assert resiliency.step_count == 2
        resiliency.close()

    def test_prepare_recovery_reports_selected_checkpoint(self):
        engine = FakeDeepSpeedEngine()
        decisions = []

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
            )

        manager = MagicMock()
        manager.local_recovery_step.return_value = 14
        resiliency._ckpt_manager = manager
        resiliency._certification = MagicMock()
        resiliency._certification.prepare_recovery.return_value = RecoveryMode.LATEST_GEMINI
        resiliency.set_recovery_decision_callback(decisions.append)

        mode = resiliency.prepare_recovery("straggler")

        assert mode is RecoveryMode.LATEST_GEMINI
        assert decisions[0]["checkpoint_step"] == 14
        assert decisions[0]["checkpoint_source"] == "gemini"
        assert resiliency.last_recovery_decision == decisions[0]
        resiliency.close()

    def test_optimizer_replay_is_captured_only_at_detection_interval(self):
        engine = FakeDeepSpeedEngine()
        engine.module = nn.Module()
        engine.module.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        parameter = nn.Parameter(torch.randn(16))
        base_optimizer = torch.optim.AdamW([parameter], lr=0.01)
        engine.optimizer.single_partition_of_fp32_groups = [parameter]
        engine.optimizer.optimizer = base_optimizer

        def step(lr_kwargs=None):
            del lr_kwargs
            parameter.grad = torch.randn_like(parameter)
            base_optimizer.step()
            engine.global_steps += 1

        engine.step = step
        harness = MagicMock()
        harness.has_capture = True
        harness.step.return_value = None
        harness.optimizer_replay_due.side_effect = lambda step: step % 2 == 0

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "lm_resiliency.integrations.deepspeed.training.ModelReplayHarness",
                return_value=harness,
            ),
        ):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
                detection_config=ReplayHarnessConfig(check_interval=2),
            )

        engine.step()
        first_tensors = harness.step.call_args.kwargs["optimizer_step_tensors"]
        assert first_tensors is None

        engine.step()
        second_tensors = harness.step.call_args.kwargs["optimizer_step_tensors"]
        assert isinstance(second_tensors, OptimizerReplayBatch)
        assert len(second_tensors.recipes) == 1
        assert second_tensors.recipes[0].status == OPTIMIZER_STATUS_OK
        assert second_tensors.recipes[0].capture is not None
        resiliency.close()

    def test_unsupported_zero_base_optimizer_disables_recipe(self):
        engine = FakeDeepSpeedEngine()
        engine.module = nn.Module()
        engine.module.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        harness = MagicMock()
        harness.has_capture = True
        harness.step.return_value = None

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "lm_resiliency.integrations.deepspeed.training.ModelReplayHarness",
                return_value=harness,
            ),
        ):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
                detection_config=ReplayHarnessConfig(check_interval=1),
            )

        engine.step()
        assert harness.step.call_args.kwargs["optimizer_step_tensors"] is None
        resiliency.close()

    def test_zero3_releases_direct_replay_backward_leftovers(self):
        engine = FakeDeepSpeedEngine()
        engine.zero_optimization_stage = lambda: 3
        engine.optimizer.parameter_offload = MagicMock()
        engine.module = nn.Module()
        engine.module.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        harness = MagicMock()
        harness.has_capture = True
        harness.step.return_value = None

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "lm_resiliency.integrations.deepspeed.training.ModelReplayHarness",
                return_value=harness,
            ),
        ):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
                detection_config=ReplayHarnessConfig(check_interval=1),
            )

        engine.step()
        engine.optimizer.parameter_offload.release_backward_leftovers.assert_called_once()
        resiliency.close()

    def test_pipeline_engine_hooks_internal_optimizer_boundary(self):
        engine = PipelineEngine()

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
            )

        MethodType(engine._INSTRUCTION_MAP[OptimizerStep], engine)()
        assert engine.optimizer_steps == 1
        assert resiliency.step_count == 1
        resiliency.close()
        assert engine._INSTRUCTION_MAP is PipelineEngine._INSTRUCTION_MAP

    def test_zero3_disables_replicated_parameter_comparison(self):
        engine = FakeDeepSpeedEngine()
        engine.dp_world_size = 4
        engine.world_size = 4
        engine.zero_optimization_stage = lambda: 3
        engine.module = nn.Module()
        engine.module.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        harness = MagicMock()

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "lm_resiliency.integrations.deepspeed.training.ModelReplayHarness",
                return_value=harness,
            ) as harness_cls,
        ):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(enable=False),
                detection_config=ReplayHarnessConfig(check_interval=2),
            )

        assert harness_cls.call_args.kwargs["config"].compare_parameter_state is False
        resiliency.close()


class TestDeepSpeedResiliencyCheckpoint:
    def test_save_tensors_called_at_interval(self):
        engine = FakeDeepSpeedEngine()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            config = InMemoryCkptConfig(
                enable=True,
                interval=2,
                skip_replication_if_hsdp=True,
                disk_flush_interval=0,
                disk_folder=tempfile.mkdtemp(),
            )
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=config,
            )

        # Step 1: no checkpoint
        engine.step()
        assert resiliency._ckpt_manager._save_count == 0

        # Step 2: checkpoint triggered
        engine.step()
        assert resiliency._ckpt_manager._save_count == 1

        # Step 3: no checkpoint
        engine.step()
        assert resiliency._ckpt_manager._save_count == 1

        # Step 4: checkpoint triggered
        engine.step()
        assert resiliency._ckpt_manager._save_count == 2

        resiliency.close()

    def test_tensor_list_cached(self):
        """collect_checkpoint_tensors is called once, then reused."""
        engine = FakeDeepSpeedEngine()

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
        ):
            config = InMemoryCkptConfig(
                enable=True,
                interval=1,
                skip_replication_if_hsdp=True,
                disk_flush_interval=0,
                disk_folder=tempfile.mkdtemp(),
            )
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=config,
            )

        engine.step()
        tensors_first = resiliency._ckpt_tensors
        assert tensors_first is not None

        engine.step()
        assert resiliency._ckpt_tensors is tensors_first  # same object

        resiliency.close()

    def test_sdc_skips_checkpoint_capture(self):
        engine = FakeDeepSpeedEngine()
        engine.module = nn.Module()
        engine.module.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])
        harness = MagicMock()
        harness.step.return_value = ReplayResult(
            sdc_bitmap=[1],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=0,
        )

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_rank", return_value=0),
            patch("torch.distributed.get_world_size", return_value=1),
            patch(
                "lm_resiliency.integrations.deepspeed.training.ModelReplayHarness",
                return_value=harness,
            ),
        ):
            resiliency = DeepSpeedResiliency(
                engine=engine,
                ckpt_config=InMemoryCkptConfig(
                    enable=True,
                    interval=1,
                    skip_replication_if_hsdp=True,
                    disk_flush_interval=0,
                    disk_folder=tempfile.mkdtemp(),
                ),
                detection_config=ReplayHarnessConfig(check_interval=1),
            )

        engine.step()

        assert resiliency._ckpt_manager._save_count == 0
        resiliency.close()


class TestEnableDeepSpeedResiliency:
    def test_convenience_function(self):
        engine = FakeDeepSpeedEngine()

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = enable_deepspeed_resiliency(
                engine=engine,
                interval=5,
                enable_detection=False,
            )

        assert resiliency._ckpt_interval == 5
        assert resiliency._replay_harness is None
        resiliency.close()

    def test_registers_automatic_cleanup(self):
        engine = FakeDeepSpeedEngine()

        with (
            patch("torch.distributed.is_initialized", return_value=False),
            patch(
                "lm_resiliency.integrations.deepspeed.training.register_automatic_cleanup"
            ) as register_cleanup,
        ):
            resiliency = enable_deepspeed_resiliency(
                engine=engine,
                enable_checkpoint=False,
                enable_detection=False,
            )

        register_cleanup.assert_called_once_with(resiliency)
        resiliency.close()

    def test_disabled_when_interval_zero(self):
        engine = FakeDeepSpeedEngine()

        with patch("torch.distributed.is_initialized", return_value=False):
            resiliency = enable_deepspeed_resiliency(
                engine=engine,
                enable_checkpoint=False,
                enable_detection=False,
            )

        assert resiliency._ckpt_manager is None
        resiliency.close()
