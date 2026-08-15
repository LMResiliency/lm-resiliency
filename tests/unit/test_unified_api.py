"""Unit tests for the unified enable_resiliency() API."""

from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency.api import enable_resiliency
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointConfig,
    DurableCheckpointCoordinator,
    DurableCheckpointRecord,
)
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig
from lm_resiliency.detection.replay_shapes import ReplayShape


@dataclass
class RecordingDurableAdapter:
    saved: list[DurableCheckpointRecord] = field(default_factory=list)
    loaded: list[DurableCheckpointRecord] = field(default_factory=list)
    committed: list[DurableCheckpointRecord] = field(default_factory=list)
    quarantined: list[DurableCheckpointRecord] = field(default_factory=list)

    def save_candidate(self, candidate):
        self.saved.append(candidate)

    def load_checkpoint(self, checkpoint):
        self.loaded.append(checkpoint)
        return checkpoint.step

    def commit_candidate(self, checkpoint, previous):
        del previous
        self.committed.append(checkpoint)

    def quarantine_candidate(self, checkpoint, reason):
        del reason
        self.quarantined.append(checkpoint)


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


class SimpleModel(nn.Module):
    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.embed = nn.Embedding(100, hidden_dim)
        self.layers = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.head = nn.Linear(hidden_dim, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.embed(x)
        for layer in self.layers:
            h = layer(h)
        return self.head(h)


class TestUnifiedAPIReplayOnly:
    def test_replay_enabled(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            replay=replay_cfg,
        )

        assert state.replay_harness is not None
        assert state.ckpt_manager is None

    def test_replay_captures_on_forward(self):
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            replay=replay_cfg,
        )

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)
        assert state.replay_harness.has_capture

    def test_step_counter_increments_via_optimizer(self):
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            replay=replay_cfg,
        )

        for _ in range(5):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        assert state.step_count == 5
        assert state.replay_harness.step_count == 5

    def test_optimizer_step_check_receives_live_optimizer(self):
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            enable_checkpoint=False,
            replay=ReplayHarnessConfig(
                check_interval=1,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
            ),
        )
        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0])
        detector.replay_invocation.return_value = ReplayResult(
            sdc_bitmap=[0],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=0,
            peer_ranks=[0],
            replay_times_ms=[1.0],
            sdc_source_bitmaps={"output": [0]},
            spatial_straggler_bitmap=[0],
        )
        detector.compare_tensor_groups.return_value = {
            "optimizer_updated_weight": C3Result(C3Status.AGREE, [0], [0])
        }
        state.replay_harness._detector = detector

        model(torch.randint(0, 100, (2, 8))).sum().backward()
        optimizer.step()

        compared = detector.compare_tensor_groups.call_args.args[0]
        assert list(compared) == ["optimizer_updated_weight"]
        state.close()


class TestUnifiedAPICheckpointOnly:
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_checkpoint_enabled(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=5)

        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            enable_detection=False,
            checkpoint=ckpt_cfg,
        )

        assert state.ckpt_manager is not None
        assert state.replay_harness is None
        mock_mgr_cls.assert_called_once()

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_checkpoint_save_triggered_at_interval(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=3)

        state = enable_resiliency(
            model,
            optimizer,
            interval=3,
            enable_detection=False,
            checkpoint=ckpt_cfg,
        )

        for _ in range(6):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        assert mock_mgr.save.call_count == 2
        # First save at step 3, second at step 6
        call_args = mock_mgr.save.call_args_list
        assert call_args[0][0][1] == 3  # step=3
        assert call_args[1][0][1] == 6  # step=6

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_checkpoint_not_triggered_before_interval(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=10)

        state = enable_resiliency(
            model,
            optimizer,
            interval=10,
            enable_detection=False,
            checkpoint=ckpt_cfg,
        )

        for _ in range(4):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        assert mock_mgr.save.call_count == 0

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_checkpoint_save_includes_model_and_optimizer(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=1)

        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            enable_detection=False,
            checkpoint=ckpt_cfg,
        )

        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        sd = mock_mgr.save.call_args[0][0]
        assert "model" in sd
        assert "optimizer" in sd


class TestUnifiedAPIBothFeatures:
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_both_enabled(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=5)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            checkpoint=ckpt_cfg,
            replay=replay_cfg,
        )

        assert state.ckpt_manager is not None
        assert state.replay_harness is not None
        assert state.replay_harness._config.check_interval == ckpt_cfg.interval

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_both_features_run_correctly(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=2)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            interval=2,
            checkpoint=ckpt_cfg,
            replay=replay_cfg,
        )
        harness = state.replay_harness

        def clean_step(**kwargs):
            del kwargs
            harness._step_count += 1
            if harness._step_count % ckpt_cfg.interval:
                return None
            return ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
            )

        harness.step = MagicMock(side_effect=clean_step)

        for _ in range(4):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        # Detection and GEMINI capture run at the same two-step boundary.
        assert mock_mgr.save.call_count == 2
        assert [call.args[1] for call in mock_mgr.save.call_args_list] == [2, 4]
        assert all("validated" not in call.kwargs for call in mock_mgr.save.call_args_list)
        # Replay harness tracked all steps
        assert harness.step_count == 4
        assert state.step_count == 4

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_sdc_skips_checkpoint_capture(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            checkpoint=InMemoryCkptConfig(interval=1),
            replay=ReplayHarnessConfig(check_interval=50),
        )
        state.replay_harness.step = MagicMock(
            return_value=ReplayResult(
                sdc_bitmap=[1],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
            )
        )

        optimizer.step()

        mock_mgr.save.assert_not_called()
        state.close()

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_straggler_does_not_skip_checkpoint_capture(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            checkpoint=InMemoryCkptConfig(interval=1),
            replay=ReplayHarnessConfig(check_interval=50),
        )
        state.replay_harness.step = MagicMock(
            return_value=ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[1],
                replay_time_ms=1.0,
                layer_id=0,
            )
        )

        optimizer.step()

        mock_mgr.save.assert_called_once()
        state.close()

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_unavailable_boundary_check_skips_checkpoint(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = enable_resiliency(
            model,
            optimizer,
            interval=2,
            checkpoint=InMemoryCkptConfig(interval=2),
            replay=ReplayHarnessConfig(check_interval=2),
        )
        state.replay_harness.step = MagicMock(return_value=None)

        optimizer.step()
        optimizer.step()

        mock_mgr.save.assert_not_called()
        state.close()


class TestUnifiedAPIDurableCheckpoint:
    @patch("lm_resiliency._feature_wiring.ModelReplayHarness")
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_moe_candidate_promotes_after_two_rotating_shape_cycles(
        self,
        mock_mgr_cls,
        mock_harness_cls,
        tmp_path,
    ):
        manager = MagicMock()
        manager.load.return_value = None
        mock_mgr_cls.return_value = manager
        small = ReplayShape("small", (8,))
        large = ReplayShape("large", (64,))
        harness = MagicMock()
        harness.replay_shape_plan_id = "qualified-moe-plan"
        harness.replay_shapes = (small, large)
        harness.current_replay_shape = small
        harness.temporal_state_dict.return_value = {}
        harness.step.side_effect = [
            ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
                checked_shape_ids=["small"],
                completed_shape_cycle=False,
                completed_scheduled_cycle=False,
                shape_cycle_size=2,
            ),
            ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
                checked_shape_ids=["large"],
                completed_shape_cycle=False,
                completed_scheduled_cycle=True,
                shape_cycle_size=2,
            ),
            ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
                checked_shape_ids=["small"],
                completed_shape_cycle=False,
                completed_scheduled_cycle=False,
                shape_cycle_size=2,
            ),
            ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
                checked_shape_ids=["large"],
                completed_shape_cycle=False,
                completed_scheduled_cycle=True,
                shape_cycle_size=2,
            ),
        ]
        mock_harness_cls.return_value = harness
        adapter = RecordingDurableAdapter()
        durable = DurableCheckpointConfig(
            manifest_dir=str(tmp_path),
            environment_id="a100-production",
            adapter=adapter,
        )
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            checkpoint=InMemoryCkptConfig(interval=1),
            replay=ReplayHarnessConfig(check_interval=1),
            durable_checkpoint=durable,
        )
        for _ in range(4):
            optimizer.step()

        assert manager.save.call_count == 4
        assert all("validated" not in call.kwargs for call in manager.save.call_args_list)
        assert manager.persist_cycle_boundary.call_count == 2
        assert len(adapter.saved) == 2
        assert len(adapter.committed) == 1
        assert adapter.committed[0].step == 2
        assert adapter.committed[0].checked_shape_ids == ("small", "large")
        assert "complete_shape_cycle" not in harness.step.call_args.kwargs
        state.close()

    @patch("lm_resiliency._feature_wiring.ModelReplayHarness")
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_sdc_skips_durable_and_gemini_candidates(
        self,
        mock_mgr_cls,
        mock_harness_cls,
        tmp_path,
    ):
        manager = MagicMock()
        manager.load.return_value = None
        mock_mgr_cls.return_value = manager
        shape = ReplayShape("captured")
        harness = MagicMock()
        harness.replay_shape_plan_id = "dense-plan"
        harness.replay_shapes = (shape,)
        harness.current_replay_shape = shape
        harness.temporal_state_dict.return_value = {}
        harness.step.return_value = ReplayResult(
            sdc_bitmap=[1],
            straggler_bitmap=[0],
            replay_time_ms=1.0,
            layer_id=0,
            checked_shape_ids=["captured"],
        )
        mock_harness_cls.return_value = harness
        adapter = RecordingDurableAdapter()
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            checkpoint=InMemoryCkptConfig(interval=1),
            replay=ReplayHarnessConfig(check_interval=1),
            durable_checkpoint=DurableCheckpointConfig(
                manifest_dir=str(tmp_path),
                environment_id="a100-production",
                adapter=adapter,
            ),
        )

        optimizer.step()

        manager.save.assert_not_called()
        assert len(adapter.saved) == 0
        assert len(adapter.quarantined) == 0
        assert not (tmp_path / "LATEST_SCOUT_VALIDATED").exists()
        state.close()

    @patch("lm_resiliency._feature_wiring.ModelReplayHarness")
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_recovery_uses_validated_manifest_instead_of_unconstrained_fallback(
        self,
        mock_mgr_cls,
        mock_harness_cls,
        tmp_path,
    ):
        adapter = RecordingDurableAdapter()
        config = DurableCheckpointConfig(
            manifest_dir=str(tmp_path),
            environment_id="a100-production",
            adapter=adapter,
        )
        seed = DurableCheckpointCoordinator(
            config,
            shape_plan_id="dense-plan",
            shape_ids=["captured"],
        )
        seed.begin_candidate(step=42, first_shape_id="captured")
        seed.observe(
            ReplayResult(
                sdc_bitmap=[0],
                straggler_bitmap=[0],
                replay_time_ms=1.0,
                layer_id=0,
                checked_shape_ids=["captured"],
            ),
            step=42,
        )

        manager = MagicMock()
        manager.load.return_value = None
        mock_mgr_cls.return_value = manager
        shape = ReplayShape("captured")
        harness = MagicMock()
        harness.replay_shape_plan_id = "dense-plan"
        harness.replay_shapes = (shape,)
        harness.current_replay_shape = shape
        mock_harness_cls.return_value = harness
        fallback = MagicMock()
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        state = enable_resiliency(
            model,
            optimizer,
            interval=1,
            checkpoint=InMemoryCkptConfig(interval=1),
            replay=ReplayHarnessConfig(check_interval=1),
            durable_checkpoint=config,
            load_fallback=fallback,
        )

        assert state.recovered_step == 42
        assert state.step_count == 42
        assert adapter.loaded[-1].checkpoint_id == adapter.committed[-1].checkpoint_id
        fallback.assert_not_called()
        state.close()


class TestUnifiedAPIDisabled:
    def test_neither_enabled_returns_empty_state(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )

        assert state.ckpt_manager is None
        assert state.replay_harness is None

    def test_checkpoint_disabled_flag(self):
        model = SimpleModel()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(enable=False)

        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            enable_detection=False,
            checkpoint=ckpt_cfg,
        )

        assert state.ckpt_manager is None

    def test_nonpositive_interval_is_rejected(self):
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        with pytest.raises(ValueError, match="interval"):
            enable_resiliency(model, optimizer, interval=0)


class TestUnifiedAPIClose:
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_close_cleans_up(self, mock_mgr_cls):
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        ckpt_cfg = InMemoryCkptConfig(interval=5)
        replay_cfg = ReplayHarnessConfig(check_interval=0)

        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            checkpoint=ckpt_cfg,
            replay=replay_cfg,
        )

        state.close()

        mock_mgr.close.assert_called_once()
        # After close, hooks no longer fire
        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        out.sum().backward()
        optimizer.step()
        optimizer.zero_grad()

        # Step count should not increment after close
        assert state.step_count == 0

    def test_public_handle_runs_close_callbacks_once(self):
        from lm_resiliency import ResiliencyHandle

        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        handle = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            enable_detection=False,
        )
        callback = MagicMock()
        handle.add_close_callback(callback)

        assert isinstance(handle, ResiliencyHandle)
        handle.close()
        handle.close()

        assert handle.closed
        callback.assert_called_once()
        with pytest.raises(RuntimeError, match="closed"):
            handle.add_close_callback(callback)


class TestUnifiedAPIRecovery:
    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_recovery_from_in_memory(self, mock_mgr_cls):
        """When in-memory checkpoint exists, model/optimizer are restored."""
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        saved_sd = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = (saved_sd, 42)
        mock_mgr_cls.return_value = mock_mgr

        fallback_called = []
        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            enable_detection=False,
            checkpoint=InMemoryCkptConfig(interval=5),
            load_fallback=lambda: fallback_called.append(True),
        )

        assert state.recovered_step == 42
        assert state.step_count == 42
        assert len(fallback_called) == 0

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_fallback_called_when_no_in_memory(self, mock_mgr_cls):
        """When no in-memory checkpoint, load_fallback is called."""
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        fallback_called = []
        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            enable_detection=False,
            checkpoint=InMemoryCkptConfig(interval=5),
            load_fallback=lambda: fallback_called.append(True),
        )

        assert state.recovered_step == -1
        assert len(fallback_called) == 1

    def test_fallback_called_when_no_checkpoint_config(self):
        """When checkpoint is disabled, load_fallback is still called."""
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        fallback_called = []
        state = enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            enable_detection=False,
            load_fallback=lambda: fallback_called.append(True),
        )

        assert len(fallback_called) == 1

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_no_fallback_no_crash(self, mock_mgr_cls):
        """No fallback and no in-memory checkpoint — starts fresh without error."""
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = None
        mock_mgr_cls.return_value = mock_mgr

        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            enable_detection=False,
            checkpoint=InMemoryCkptConfig(interval=5),
        )

        assert state.recovered_step == -1

    @patch("lm_resiliency._feature_wiring.InMemoryCheckpointManager")
    def test_step_count_resumes_after_recovery(self, mock_mgr_cls):
        """After recovery from step 42, next save happens at step 42+interval."""
        model = SimpleModel(hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

        saved_sd = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        mock_mgr = MagicMock()
        mock_mgr.load.return_value = (saved_sd, 8)
        mock_mgr_cls.return_value = mock_mgr

        state = enable_resiliency(
            model,
            optimizer,
            interval=5,
            enable_detection=False,
            checkpoint=InMemoryCkptConfig(interval=5),
        )

        assert state.step_count == 8

        # Run 2 steps: step_count becomes 9, 10 — save at 10 (divisible by 5)
        for _ in range(2):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            out.sum().backward()
            optimizer.step()
            optimizer.zero_grad()

        assert mock_mgr.save.call_count == 1
        assert mock_mgr.save.call_args[0][1] == 10


class TestUnifiedAPIImport:
    def test_stable_package_root_contract(self):
        import lm_resiliency

        expected = {
            "AllToAllCapture",
            "AllToAllReplayPolicy",
            "AllToAllTrafficMatrix",
            "BalancedAndPermutationPolicy",
            "CallbackDurableCheckpointAdapter",
            "DurableCheckpointConfig",
            "FrameworkName",
            "GroupedExpertMaterializer",
            "InMemoryCkptConfig",
            "LeadingDimensionMaterializer",
            "OrchestrationHooks",
            "ReplayHarnessConfig",
            "ReplayWorkload",
            "RecoveryDecision",
            "RecoveryDecisionCallback",
            "RecoveryMode",
            "ResiliencyHandle",
            "ResiliencySession",
            "SCOUTFaultCallback",
            "SCOUTFaultReport",
            "enable_resiliency",
            "estimate_chunk_size",
            "replay_fault_reports",
        }
        assert set(lm_resiliency.__all__) == expected
        for symbol in expected:
            assert getattr(lm_resiliency, symbol) is not None
        for symbol in {
            "CheckpointTransfer",
            "InMemoryCheckpointManager",
            "LayerReplayDetector",
            "ModelReplayHarness",
            "NixlCheckpointTransfer",
            "ReplayShape",
            "TorchDistTransfer",
            "make_transfer",
        }:
            assert not hasattr(lm_resiliency, symbol)

    def test_low_level_types_are_explicitly_experimental(self):
        from lm_resiliency.experimental import (
            InMemoryCheckpointManager,
            ModelReplayHarness,
            ParallelismInfo,
            ReplayShape,
            ReplayShapeMaterializer,
            ReplayShapePlan,
        )

        assert InMemoryCheckpointManager is not None
        assert ModelReplayHarness is not None
        assert ParallelismInfo is not None
        assert ReplayShape is not None
        assert ReplayShapeMaterializer is not None
        assert ReplayShapePlan is not None
