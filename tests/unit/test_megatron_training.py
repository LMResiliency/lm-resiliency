"""Unit tests for Megatron training integration (MegatronResiliency).

Tests optimizer wrapping, post-step hooks, checkpoint triggering,
detection triggering, recovery flow, and the convenience API.
All tests use mocks to avoid requiring megatron-core or distributed init.
"""

import sys
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Mock megatron.core.mpu
mock_mpu = MagicMock()
mock_mpu.get_tensor_model_parallel_world_size.return_value = 1
mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1
mock_mpu.get_data_parallel_group.return_value = None
sys.modules.setdefault("megatron", MagicMock())
sys.modules.setdefault("megatron.core", MagicMock(mpu=mock_mpu))
sys.modules.setdefault("megatron.core.mpu", mock_mpu)

from lm_resiliency.cadence import ResiliencyCadence  # noqa: E402
from lm_resiliency.checkpointing.config import InMemoryCkptConfig  # noqa: E402
from lm_resiliency.checkpointing.manager import RecoveryMode  # noqa: E402
from lm_resiliency.detection.layer_replay import ReplayResult  # noqa: E402
from lm_resiliency.detection.optimizer_step import (  # noqa: E402
    OPTIMIZER_STATUS_OK,
    OptimizerReplayBatch,
)
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig  # noqa: E402
from lm_resiliency.integrations.megatron.training import (  # noqa: E402
    MegatronResiliency,
    enable_megatron_resiliency,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fake model/optimizer
# ──────────────────────────────────────────────────────────────────────────────


class FakeModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(num_layers)])

    @property
    def decoder(self):
        return self

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FakeOptimizer:
    def __init__(self):
        self.step_count = 0
        self.step_results = []

    def step(self):
        self.step_count += 1
        return True, 1.0, 0

    def state_dict(self):
        return {"step": self.step_count}

    def load_state_dict(self, state_dict):
        self.step_count = state_dict.get("step", 0)


class FakeScheduler:
    def __init__(self):
        self._step_count = 0

    def step(self, increment=1):
        self._step_count += increment
        return self._step_count

    def state_dict(self):
        return {"step_count": self._step_count}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict.get("step_count", 0)


# ──────────────────────────────────────────────────────────────────────────────
# Optimizer wrapping tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOptimizerWrapping:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_wraps_optimizer_step(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        # Wrapped step calls post_step, which increments count
        optimizer.step()
        assert resiliency.step_count == 1
        # Original step was saved
        assert resiliency._original_step is not None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_close_restores_original_step(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )
        resiliency.close()

        # After close, step no longer increments resiliency count
        optimizer.step()
        assert resiliency.step_count == 0

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_scheduler_is_completed_step_boundary(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )
        certification = MagicMock()
        resiliency._certification = certification

        optimizer.step()
        assert resiliency.step_count == 0
        certification.post_step.assert_not_called()

        assert scheduler.step(increment=4) == 4
        assert resiliency.step_count == 1
        certification.post_step.assert_called_once_with(
            1,
            optimizer_step_tensors=None,
        )
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_close_restores_scheduler_step(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()
        original_step = scheduler.step
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        resiliency.close()

        assert scheduler.step == original_step

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_close_preserves_later_scheduler_wrapper(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )
        later_wrapper = MagicMock(return_value=17)
        scheduler.step = later_wrapper

        resiliency.close()

        assert scheduler.step is later_wrapper

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_wrapped_step_returns_original_result(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        result = optimizer.step()
        assert result == (True, 1.0, 0)
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_wrapped_step_passes_args_kwargs(self, *mocks):
        model = FakeModel()
        received = []

        class ArgCapturingOptimizer:
            def step(self, *args, **kwargs):
                received.append((args, kwargs))
                return True, 0.0, 0

            def state_dict(self):
                return {}

            def load_state_dict(self, sd):
                pass

        optimizer = ArgCapturingOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        optimizer.step("arg1", key="val")
        assert received[-1] == (("arg1",), {"key": "val"})
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# Step counting tests
# ──────────────────────────────────────────────────────────────────────────────


class TestStepCounting:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_step_count_increments(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        assert resiliency.step_count == 0
        optimizer.step()
        assert resiliency.step_count == 1
        optimizer.step()
        optimizer.step()
        assert resiliency.step_count == 3
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_step_count_settable_for_recovery(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        resiliency.step_count = 500
        assert resiliency.step_count == 500
        optimizer.step()
        assert resiliency.step_count == 501
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# Checkpoint trigger tests
# ──────────────────────────────────────────────────────────────────────────────


class TestCheckpointTrigger:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_checkpoint_not_triggered_when_disabled(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        assert resiliency._ckpt_manager is None
        for _ in range(20):
            optimizer.step()
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_gemini_checkpoint_engine_created_when_enabled(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=True, interval=5),
        )

        assert resiliency._ckpt_manager is not None
        assert resiliency._ckpt_interval == 5
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_save_now_without_gemini_checkpointing(self, *mocks):
        """save_now is a no-op when checkpointing is disabled."""
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        resiliency.save_now(step=10)  # Should not raise
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_recovery_restores_caller_owned_training_state(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        restored = []
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            load_extra_state_fn=restored.append,
        )
        resiliency._ckpt_manager = MagicMock()
        resiliency._ckpt_manager.load_tensors.return_value = (
            [],
            7,
            {"iteration": 7},
        )
        resiliency._adapter.collect_checkpoint_tensors = MagicMock(return_value=[])
        resiliency._adapter.load_checkpoint_tensors = MagicMock()

        assert resiliency.try_recover() == 7
        assert restored == [{"iteration": 7}]
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_save_now_requires_scout_result_when_detection_is_active(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )
        manager = MagicMock()
        harness = MagicMock()
        harness.has_capture = False
        harness.has_replay_capture = False
        resiliency._ckpt_manager = manager
        resiliency._replay_harness = harness
        resiliency._certification.checkpoint_manager = manager
        resiliency._certification.replay_harness = harness

        resiliency.save_now(step=10)

        manager.save_tensors.assert_not_called()
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# Detection tests
# ──────────────────────────────────────────────────────────────────────────────


class TestDetection:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_detection_disabled_when_no_config(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=None,
        )

        assert resiliency._replay_harness is None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_sdc_skips_checkpoint_capture(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )
        manager = MagicMock()
        resiliency._ckpt_manager = manager
        resiliency._cadence = ResiliencyCadence(
            interval=1,
            checkpoint_enabled=True,
            detection_enabled=True,
        )
        harness = MagicMock()
        harness.step.return_value = ReplayResult(
            sdc_bitmap=[1],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=0,
        )
        resiliency._replay_harness = harness
        resiliency._certification.checkpoint_manager = manager
        resiliency._certification.replay_harness = harness
        resiliency._certification.cadence = resiliency._cadence

        optimizer.step()

        manager.save_tensors.assert_not_called()
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_check_now_without_detection(self, *mocks):
        """check_now returns None when detection is disabled."""
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=None,
        )

        result = resiliency.check_now()
        assert result is None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_detection_warns_when_no_layers(self, *mocks):
        """Model without repeated layers → warning, harness not created."""
        model = nn.Linear(16, 16)  # No repeated layers
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=10),
        )

        assert resiliency._replay_harness is None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_optimizer_replay_is_captured_only_at_detection_interval(self, *mocks):
        model = FakeModel()
        parameter = nn.Parameter(torch.randn(16))
        base_optimizer = torch.optim.AdamW([parameter], lr=0.01)

        class DistributedOptimizer(FakeOptimizer):
            def __init__(self):
                super().__init__()
                self.optimizer = base_optimizer

            def step(self):
                parameter.grad = torch.randn_like(parameter)
                self.optimizer.step()
                return super().step()

        optimizer = DistributedOptimizer()
        harness = MagicMock()
        harness.has_capture = True
        harness.step.return_value = None
        harness.optimizer_replay_due.side_effect = lambda step: step % 2 == 0
        with patch(
            "lm_resiliency.integrations.megatron.training.ModelReplayHarness",
            return_value=harness,
        ):
            resiliency = MegatronResiliency(
                model=[model],
                optimizer=optimizer,
                ckpt_config=InMemoryCkptConfig(enable=False),
                detection_config=ReplayHarnessConfig(check_interval=2),
            )

        optimizer.step()
        first_tensors = harness.step.call_args.kwargs["optimizer_step_tensors"]
        assert first_tensors is None

        optimizer.step()
        second_tensors = harness.step.call_args.kwargs["optimizer_step_tensors"]
        assert isinstance(second_tensors, OptimizerReplayBatch)
        assert len(second_tensors.recipes) == 1
        assert second_tensors.recipes[0].status == OPTIMIZER_STATUS_OK
        assert second_tensors.recipes[0].capture is not None
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# Recovery tests
# ──────────────────────────────────────────────────────────────────────────────


class TestRecovery:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_try_recover_returns_negative_when_disabled(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
        )

        assert resiliency.try_recover() == -1
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_try_recover_returns_negative_when_no_data(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = MegatronResiliency(
            model=[model],
            optimizer=optimizer,
            ckpt_config=InMemoryCkptConfig(enable=True, interval=5),
        )

        # No saves done yet → nothing to recover
        assert resiliency.try_recover() == -1
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# enable_megatron_resiliency convenience API tests
# ──────────────────────────────────────────────────────────────────────────────


class TestEnableMegatronResiliency:
    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_returns_megatron_resiliency(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        assert isinstance(resiliency, MegatronResiliency)
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_registers_automatic_cleanup(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        with patch(
            "lm_resiliency.integrations.megatron.training.register_automatic_cleanup"
        ) as register_cleanup:
            resiliency = enable_megatron_resiliency(
                model=[model],
                optimizer=optimizer,
                enable_checkpoint=False,
                enable_detection=False,
            )

        register_cleanup.assert_called_once_with(resiliency)
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_checkpoint_switch_disables_gemini(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        assert resiliency._ckpt_manager is None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_detection_switch_disables_scout(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        assert resiliency._replay_harness is None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_interval_sets_checkpoint_config(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            interval=7,
            enable_detection=False,
        )

        assert resiliency._ckpt_interval == 7
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_extra_state_fn_stored(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        iteration = [0]

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
            extra_state_fn=lambda: {"iteration": iteration[0]},
        )

        assert resiliency._extra_state_fn is not None
        iteration[0] = 42
        assert resiliency._extra_state_fn()["iteration"] == 42
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_fault_callback_stored(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        faults = []

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
            fault_callback=lambda r: faults.append(r),
        )

        assert resiliency._fault_callback is not None
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_interval_overrides_explicit_checkpoint_config_cadence(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            interval=7,
            enable_detection=False,
            ckpt_config=InMemoryCkptConfig(enable=True, interval=3),
        )

        assert resiliency._ckpt_interval == 7
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_with_scheduler(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()
        scheduler = FakeScheduler()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
            enable_checkpoint=False,
            enable_detection=False,
        )

        assert resiliency._opt_param_scheduler is scheduler
        resiliency.close()


# ──────────────────────────────────────────────────────────────────────────────
# End-to-end single-rank integration
# ──────────────────────────────────────────────────────────────────────────────


class TestSingleRankEndToEnd:
    """End-to-end tests running the full flow on a single rank (no dist)."""

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_training_loop_simulation(self, *mocks):
        """Simulate a training loop: model → forward → backward → step."""
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        # Simulate 10 training steps
        for i in range(10):
            x = torch.randn(4, 16)
            y = model(x)
            loss = y.sum()
            loss.backward()
            optimizer.step()

        assert resiliency.step_count == 10
        assert optimizer.step_count == 10
        resiliency.close()

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.distributed.get_world_size", return_value=1)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_close_is_idempotent(self, *mocks):
        model = FakeModel()
        optimizer = FakeOptimizer()

        resiliency = enable_megatron_resiliency(
            model=[model],
            optimizer=optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        resiliency.close()
        resiliency.close()  # Should not raise


@patch("torch.distributed.is_initialized", return_value=False)
@patch("torch.distributed.get_world_size", return_value=1)
@patch("torch.distributed.get_rank", return_value=0)
def test_prepare_recovery_reports_selected_megatron_checkpoint(*mocks):
    model = FakeModel()
    optimizer = FakeOptimizer()
    decisions = []
    resiliency = MegatronResiliency(
        model=[model],
        optimizer=optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
    )
    manager = MagicMock()
    manager.local_recovery_step.return_value = 16
    resiliency._ckpt_manager = manager
    resiliency._certification = MagicMock()
    resiliency._certification.prepare_recovery.return_value = RecoveryMode.LATEST_GEMINI
    resiliency.set_recovery_decision_callback(decisions.append)

    mode = resiliency.prepare_recovery("straggler")

    assert mode is RecoveryMode.LATEST_GEMINI
    assert decisions[0]["checkpoint_step"] == 16
    assert decisions[0]["checkpoint_source"] == "gemini"
    assert resiliency.last_recovery_decision == decisions[0]
    resiliency.close()
