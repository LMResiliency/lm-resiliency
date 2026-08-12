"""Unit tests for ModelReplayHarness — layer auto-detection, hook capture, replay."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import (
    ReplayResult,
    StragglerDetail,
    _slow_outlier_bitmap,
)
from lm_resiliency.detection.optimizer_step import OptimizerReplayBatch
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    ReplayHarnessConfig,
    find_embedding_layer,
    find_output_layer,
    find_repeated_layers,
)


def _c3_result(bitmap: list[int]) -> C3Result:
    return C3Result(
        C3Status.ATTRIBUTED if any(bitmap) else C3Status.AGREE,
        bitmap,
        list(range(len(bitmap))),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Mock model architectures
# ──────────────────────────────────────────────────────────────────────────────


class TransformerBlock(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.linear(self.norm(x))


class LlamaLikeModel(nn.Module):
    """model.layers pattern (Llama, Mistral)."""

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


class TorchTitanLikeModel(nn.Module):
    """TorchTitan 0.2 stores transformer blocks in a string-keyed ModuleDict."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleDict(
            {str(index): TransformerBlock(hidden_dim) for index in range(num_layers)}
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers.values():
            x = layer(x)
        return x


class GPT2LikeModel(nn.Module):
    """model.transformer.h pattern (GPT-2)."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.transformer = nn.Module()
        self.transformer.wte = nn.Embedding(100, hidden_dim)
        self.transformer.h = nn.ModuleList(
            [TransformerBlock(hidden_dim) for _ in range(num_layers)]
        )
        self.lm_head = nn.Linear(hidden_dim, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.transformer.wte(x)
        for block in self.transformer.h:
            h = block(h)
        return self.lm_head(h)


class HFWrappedModel(nn.Module):
    """model.model.layers pattern (HuggingFace wrappers)."""

    def __init__(self, num_layers: int = 4, hidden_dim: int = 64):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, hidden_dim)
        self.model.layers = nn.ModuleList([TransformerBlock(hidden_dim) for _ in range(num_layers)])
        self.lm_head = nn.Linear(hidden_dim, 100)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.model.embed_tokens(x)
        for layer in self.model.layers:
            h = layer(h)
        return self.lm_head(h)


class FlatModel(nn.Module):
    """No repeated layers — should fail auto-detection."""

    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(64, 128)
        self.linear2 = nn.Linear(128, 64)

    def forward(self, x):
        return self.linear2(torch.relu(self.linear1(x)))


class StructuredBlock(nn.Module):
    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, x, freqs_cis, attention_masks, positions=None):
        del freqs_cis
        output = self.linear(x)
        if attention_masks is not None:
            output = output + attention_masks
        if positions is not None:
            output = output + positions.unsqueeze(-1)
        return output


class StructuredModel(nn.Module):
    def __init__(self, num_layers: int = 2, hidden_dim: int = 32):
        super().__init__()
        self.layers = nn.ModuleList([StructuredBlock(hidden_dim) for _ in range(num_layers)])

    def forward(self, x, freqs_cis, attention_masks, positions):
        for layer in self.layers:
            x = layer(
                x,
                freqs_cis,
                attention_masks,
                positions=positions,
            )
        return x


class RoutedExpertModel(nn.Module):
    """Toy post-dispatch boundary whose input buffer is reused after the expert."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.expert = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor, *, use_expert: bool = True) -> torch.Tensor:
        if not use_expert:
            return x
        routed_tokens = x + 1
        output = self.expert(routed_tokens)
        routed_tokens.add_(100)  # model a dispatcher reusing its temporary buffer
        return output


# ──────────────────────────────────────────────────────────────────────────────
# Tests: find_repeated_layers
# ──────────────────────────────────────────────────────────────────────────────


class TestFindRepeatedLayers:
    def test_llama_pattern(self):
        model = LlamaLikeModel(num_layers=6)
        layers = find_repeated_layers(model)
        assert layers is not None
        assert len(layers) == 6
        assert isinstance(layers[0], TransformerBlock)

    def test_gpt2_pattern(self):
        model = GPT2LikeModel(num_layers=4)
        layers = find_repeated_layers(model)
        assert layers is not None
        assert len(layers) == 4

    def test_torchtitan_moduledict_pattern(self):
        model = TorchTitanLikeModel(num_layers=5)
        layers = find_repeated_layers(model)
        assert layers is not None
        assert len(layers) == 5
        assert isinstance(layers[0], TransformerBlock)

    def test_hf_wrapper_pattern(self):
        model = HFWrappedModel(num_layers=8)
        layers = find_repeated_layers(model)
        assert layers is not None
        assert len(layers) == 8

    def test_flat_model_returns_none(self):
        model = FlatModel()
        layers = find_repeated_layers(model)
        assert layers is None

    def test_explicit_layers_bypass(self):
        model = FlatModel()
        explicit = nn.ModuleList([TransformerBlock() for _ in range(3)])
        layers = find_repeated_layers(model)
        assert layers is None
        # But passing explicitly to harness should work (tested below)


@pytest.mark.parametrize(
    ("model", "embedding_path", "output_path"),
    [
        (LlamaLikeModel(), "embed", "head"),
        (GPT2LikeModel(), "transformer.wte", "lm_head"),
        (HFWrappedModel(), "model.embed_tokens", "lm_head"),
    ],
)
def test_dense_boundary_discovery(model, embedding_path, output_path):
    def resolve(path):
        current = model
        for name in path.split("."):
            current = getattr(current, name)
        return current

    assert find_embedding_layer(model) is resolve(embedding_path)
    assert find_output_layer(model) is resolve(output_path)


def test_dense_boundary_discovery_unwraps_ddp_style_module():
    model = LlamaLikeModel()

    class Wrapper(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

    wrapped = Wrapper(Wrapper(model))

    assert find_embedding_layer(wrapped) is model.embed
    assert find_output_layer(wrapped) is model.head


def test_straggler_filter_requires_ratio_and_absolute_excess():
    bitmap = [1, 0, 0, 0]

    assert _slow_outlier_bitmap(bitmap, [7.5, 6.6, 6.7, 6.8], 1.1, 2.0) == [0, 0, 0, 0]
    assert _slow_outlier_bitmap(bitmap, [27.0, 6.6, 6.7, 6.8], 1.1, 2.0) == [1, 0, 0, 0]


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Hook capture
# ──────────────────────────────────────────────────────────────────────────────


class TestHookCapture:
    def test_forward_capture(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        assert not harness.has_capture

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        assert harness.has_capture
        assert harness._activation is not None
        assert harness._activation.shape == (2, 8, 32)

    def test_backward_capture(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x = torch.randint(0, 100, (2, 8))
        out = model(x)
        loss = out.sum()
        loss.backward()

        assert harness.has_grad
        assert harness._grad_output is not None
        assert harness._grad_output.shape == (2, 8, 32)
        assert not model.layers[0]._backward_hooks

    def test_captures_update_each_step(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x1 = torch.randint(0, 100, (2, 8))
        _ = model(x1)
        act1 = harness._activation.clone()

        x2 = torch.randint(0, 100, (2, 8))
        _ = model(x2)
        act2 = harness._activation.clone()

        assert not torch.equal(act1, act2)

    def test_different_layer_index(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=2, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        assert harness.has_capture
        # Layer 2's input should differ from layer 0's input
        # (transformed by layers 0 and 1)

    def test_captures_complete_args_and_kwargs(self):
        model = StructuredModel()
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(check_interval=0),
            layers=model.layers,
        )
        x = torch.randn(2, 8, 32, requires_grad=True)
        freqs_cis = torch.randn(8, 4)
        attention_masks = torch.randn(2, 8, 32)
        positions = torch.arange(8).expand(2, -1)

        model(x, freqs_cis, attention_masks, positions).sum().backward()

        invocation = harness._invocation
        assert invocation is not None
        assert len(invocation.args) == 3
        assert torch.equal(invocation.args[1], freqs_cis)
        assert torch.equal(invocation.args[2], attention_masks)
        assert torch.equal(invocation.kwargs["positions"], positions)
        assert invocation.input_requires_grad[0] is True
        assert invocation.grad_output is not None

    def test_explicit_moe_boundary_owns_routed_expert_inputs(self):
        model = RoutedExpertModel()
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                capture_inputs_by_value=True,
            ),
            replay_modules=[model.expert],
        )
        tokens = torch.randn(3, 8)

        model(tokens)

        assert harness.target_layer is model.expert
        assert harness._activation is not None
        assert torch.equal(harness._activation, tokens + 1)

    def test_scheduled_moe_replay_skips_stale_expert_capture(self):
        model = RoutedExpertModel()
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=2,
                capture_inputs_by_value=True,
            ),
            replay_modules=[model.expert],
        )
        tokens = torch.randn(3, 8)

        model(tokens, use_expert=True)
        assert harness.step() is None
        model(tokens, use_expert=False)

        # The selected expert received no tokens on the scheduled step. SCOUT must
        # not replay the previous step's expert invocation as though it were current.
        assert harness.step() is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Configuration validation
# ──────────────────────────────────────────────────────────────────────────────


class TestConfigValidation:
    def test_layers_and_replay_modules_are_mutually_exclusive(self):
        model = LlamaLikeModel(num_layers=2)
        with pytest.raises(ValueError, match="only one"):
            ModelReplayHarness(
                model,
                layers=model.layers,
                replay_modules=model.layers,
            )

    def test_invalid_layer_index(self):
        model = LlamaLikeModel(num_layers=4)
        config = ReplayHarnessConfig(layer_index=10)
        with pytest.raises(ValueError, match="layer_index=10"):
            ModelReplayHarness(model, config=config, layers=model.layers)

    def test_auto_detect_failure(self):
        model = FlatModel()
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            ModelReplayHarness(model)

    def test_explicit_layers_override(self):
        model = FlatModel()
        explicit = nn.ModuleList([TransformerBlock(64) for _ in range(3)])
        config = ReplayHarnessConfig(layer_index=1)
        harness = ModelReplayHarness(model, config=config, layers=explicit)
        assert harness._target_layer is explicit[1]

    def test_runtime_options_are_forwarded_to_detector(self):
        model = LlamaLikeModel(num_layers=2)
        gradient_communication = MagicMock()
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                synchronize_rng=True,
                compare_parameter_state=False,
            ),
            layers=model.layers,
            gradient_communication=gradient_communication,
        )

        with patch("lm_resiliency.detection.replay_harness.LayerReplayDetector") as detector_cls:
            harness._get_detector()

        assert detector_cls.call_args.kwargs["synchronize_rng"] is True
        assert detector_cls.call_args.kwargs["compare_parameter_state"] is False
        assert detector_cls.call_args.kwargs["gradient_communication"] is gradient_communication


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Step counting and interval
# ──────────────────────────────────────────────────────────────────────────────


class TestStepInterval:
    def test_step_counting(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        harness.step()
        harness.step()
        harness.step()
        assert harness.step_count == 3

    def test_check_not_triggered_before_interval(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=5)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        # Steps 1-4 should return None (not yet at interval)
        for _ in range(4):
            result = harness.step()
            assert result is None

    def test_check_interval_zero_means_manual(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        for _ in range(100):
            result = harness.step()
            assert result is None

    def test_check_raises_without_capture(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        with pytest.raises(RuntimeError, match="No activation captured"):
            harness.check()


class TestDenseRecipes:
    @staticmethod
    def _replay_result(
        *,
        layer_id: int = 0,
        bitmap: list[int] | None = None,
    ) -> ReplayResult:
        bitmap = bitmap or [0, 0]
        return ReplayResult(
            sdc_bitmap=bitmap.copy(),
            straggler_bitmap=[0] * len(bitmap),
            replay_time_ms=1.0,
            layer_id=layer_id,
            peer_ranks=list(range(len(bitmap))),
            replay_times_ms=[1.0] * len(bitmap),
            sdc_source_bitmaps={"output": bitmap.copy()},
            spatial_straggler_bitmap=[0] * len(bitmap),
        )

    def _detector(self) -> MagicMock:
        detector = MagicMock()
        detector.peer_ranks = [0, 1]
        detector.replay_shape_consensus.return_value = (True, [0, 0])
        detector.replay_invocation.side_effect = lambda **kwargs: self._replay_result(
            layer_id=kwargs["layer_id"]
        )
        detector.compare_tensor_groups.return_value = {
            "optimizer_updated_weight": _c3_result([0, 0])
        }
        return detector

    def test_manual_dense_check_covers_all_four_recipe_classes(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8))).sum().backward()
        optimizer.step()
        detector = self._detector()
        harness._detector = detector

        result = harness.check(optimizer=optimizer)

        assert result.checked_recipe_ids == [
            "embedding",
            "hidden",
            "output",
            "optimizer",
        ]
        assert [call.kwargs["layer"] for call in detector.replay_invocation.call_args_list] == [
            model.embed,
            model.layers[0],
            model.head,
        ]
        detector.compare_tensor_groups.assert_called_once()
        assert result.completed_shape_cycle
        assert result.shape_cycle_size == 1

    def test_dense_fault_source_names_include_recipe_class(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
                optimizer_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8)))
        detector = self._detector()

        def replay(**kwargs):
            bitmap = [0, 1] if kwargs["layer"] is model.head else [0, 0]
            return self._replay_result(layer_id=kwargs["layer_id"], bitmap=bitmap)

        detector.replay_invocation.side_effect = replay
        harness._detector = detector

        result = harness.check()

        assert result.sdc_bitmap == [0, 1]
        assert result.sdc_sources == ["output.output"]
        assert set(result.sdc_source_bitmaps) == {
            "embedding.output",
            "hidden.output",
            "output.output",
        }

    def test_dense_recipe_intervals_are_independent(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=11,
                embedding_check_interval=2,
                hidden_check_interval=3,
                output_check_interval=5,
                optimizer_check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
            ),
            layers=model.layers,
        )
        detector = self._detector()
        harness._detector = detector
        checked: list[list[str] | None] = []

        for _ in range(6):
            model(torch.randint(0, 100, (2, 8)))
            result = harness.step()
            checked.append(result.checked_recipe_ids if result is not None else None)

        assert checked == [
            None,
            ["embedding"],
            ["hidden"],
            ["embedding"],
            ["output"],
            ["embedding", "hidden"],
        ]
        assert detector.replay_invocation.call_count == 6

    def test_zero_interval_disables_recipe_for_manual_and_scheduled_checks(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=2,
                embedding_check_interval=0,
                hidden_check_interval=2,
                output_check_interval=0,
                optimizer_check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8)))
        detector = self._detector()
        harness._detector = detector

        manual = harness.check()
        assert manual.checked_recipe_ids == ["hidden"]

        assert harness.step() is None
        model(torch.randint(0, 100, (2, 8)))
        scheduled = harness.step()
        assert scheduled is not None
        assert scheduled.checked_recipe_ids == ["hidden"]

    def test_scheduled_dense_check_without_current_evidence_is_skipped(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=1,
                embedding_check_interval=0,
                output_check_interval=0,
                optimizer_check_interval=0,
            ),
            layers=model.layers,
        )

        assert harness.step() is None


# ──────────────────────────────────────────────────────────────────────────────
# Tests: Hook removal
# ──────────────────────────────────────────────────────────────────────────────


class TestOptimizerHook:
    def test_auto_step_via_optimizer(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        # check_interval=0 means manual-only, so step() never calls check()
        # (avoids needing dist.init_process_group in unit tests)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, optimizer=optimizer, config=config, layers=model.layers)

        # Run 3 training steps via optimizer.step()
        for i in range(3):
            x = torch.randint(0, 100, (2, 8))
            out = model(x)
            loss = out.sum()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # After 3 optimizer steps, harness should have auto-incremented
        assert harness.step_count == 3

    def test_callback_on_fault(self):
        """Callback is invoked when a fault is detected (simulated via manual step)."""
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)

        faults = []
        harness = ModelReplayHarness(
            model, config=config, layers=model.layers, callback=lambda r: faults.append(r)
        )

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)

        # step() won't trigger check with interval=0
        harness.step()
        assert len(faults) == 0

    def test_last_result_property(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        assert harness.last_result is None

    def test_check_rotates_to_next_layer(self):
        model = LlamaLikeModel(num_layers=3, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
                optimizer_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8)))
        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0])
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
        harness._detector = detector

        harness.check()

        assert harness.target_layer is model.layers[1]
        assert not harness.has_capture
        model(torch.randint(0, 100, (2, 8)))
        assert harness.has_capture

    def test_optimizer_step_check_compares_only_updated_layer_weights(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8))).sum().backward()
        optimizer.step()

        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0, 0, 0])
        detector.replay_invocation.return_value = ReplayResult(
            sdc_bitmap=[0, 0, 0, 0],
            straggler_bitmap=[0, 0, 0, 0],
            replay_time_ms=1.0,
            layer_id=0,
            peer_ranks=[0, 1, 2, 3],
            replay_times_ms=[1.0] * 4,
            sdc_source_bitmaps={"output": [0, 0, 0, 0]},
            spatial_straggler_bitmap=[0, 0, 0, 0],
        )
        detector.compare_tensor_groups.return_value = {
            "optimizer_updated_weight": _c3_result([0, 1, 0, 0])
        }
        harness._detector = detector

        result = harness.check(optimizer=optimizer)

        compared = detector.compare_tensor_groups.call_args.args[0]
        assert list(compared) == ["optimizer_updated_weight"]
        assert len(compared["optimizer_updated_weight"]) == len(list(model.layers[0].parameters()))
        assert result.sdc_bitmap == [0, 1, 0, 0]
        assert result.sdc_sources == ["optimizer.optimizer_updated_weight"]

    def test_precomputed_optimizer_transition_is_merged_into_replay_result(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8))).sum().backward()

        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0, 0, 0])
        detector.replay_invocation.return_value = ReplayResult(
            sdc_bitmap=[0, 0, 0, 0],
            straggler_bitmap=[0, 0, 0, 0],
            replay_time_ms=1.0,
            layer_id=0,
            peer_ranks=[0, 1, 2, 3],
            replay_times_ms=[1.0] * 4,
            sdc_source_bitmaps={"output": [0, 0, 0, 0]},
            spatial_straggler_bitmap=[0, 0, 0, 0],
        )
        detector.compare_tensor_groups.return_value = {
            "optimizer_updated_weight": _c3_result([0, 0, 1, 0])
        }
        harness._detector = detector
        optimizer_step_tensors = {
            "optimizer_updated_weight": [torch.tensor([0]), torch.randn(8), torch.randn(8)]
        }

        result = harness.check(optimizer_step_tensors=optimizer_step_tensors)

        detector.compare_tensor_groups.assert_called_once_with(optimizer_step_tensors)
        assert result.sdc_bitmap == [0, 0, 1, 0]
        assert result.sdc_sources == ["optimizer.optimizer_updated_weight"]

    def test_source_broadcast_optimizer_batch_is_merged_into_replay_result(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8))).sum().backward()

        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0, 0, 0])
        detector.replay_invocation.return_value = ReplayResult(
            sdc_bitmap=[0, 0, 0, 0],
            straggler_bitmap=[0, 0, 0, 0],
            replay_time_ms=1.0,
            layer_id=0,
            peer_ranks=[0, 1, 2, 3],
            replay_times_ms=[1.0] * 4,
            sdc_source_bitmaps={"output": [0, 0, 0, 0]},
            spatial_straggler_bitmap=[0, 0, 0, 0],
        )
        detector.replay_optimizer_batch.return_value = {
            "optimizer_replay_input": _c3_result([0, 0, 0, 0]),
            "optimizer_updated_weight": _c3_result([0, 1, 0, 0]),
        }
        harness._detector = detector
        batch = OptimizerReplayBatch((MagicMock(),))

        result = harness.check(optimizer_step_tensors=batch)

        detector.replay_optimizer_batch.assert_called_once_with(batch)
        assert result.sdc_bitmap == [0, 1, 0, 0]
        assert result.sdc_sources == ["optimizer.optimizer_updated_weight"]

    def test_straggler_requires_confirmation_and_is_decomposed(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                enable_temporal=False,
                straggler_confirmation_rounds=2,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
                optimizer_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8)))
        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0, 0, 0])
        detector.replay_invocation.side_effect = [
            ReplayResult(
                sdc_bitmap=[0, 0, 0, 0],
                straggler_bitmap=[0, 0, 1, 0],
                replay_time_ms=10.0,
                layer_id=0,
                peer_ranks=[0, 1, 2, 3],
                replay_times_ms=[10.0, 10.0, 40.0, 10.0],
                sdc_source_bitmaps={"output": [0, 0, 0, 0]},
                spatial_straggler_bitmap=[0, 0, 1, 0],
            ),
            ReplayResult(
                sdc_bitmap=[0, 0, 0, 0],
                straggler_bitmap=[0, 0, 1, 0],
                replay_time_ms=11.0,
                layer_id=0,
                peer_ranks=[0, 1, 2, 3],
                replay_times_ms=[10.0, 11.0, 41.0, 10.0],
                sdc_source_bitmaps={"output": [0, 0, 0, 0]},
                spatial_straggler_bitmap=[0, 0, 1, 0],
            ),
        ]
        detector.localize_invocation_straggler.return_value = StragglerDetail(
            straggler_rank=2,
            straggler_type="compute",
            compute_times_ms=[5.0, 5.0, 35.0, 5.0],
            comm_times_ms=[1.0, 1.0, 1.0, 1.0],
            compute_bitmap=[0, 0, 1, 0],
        )
        harness._detector = detector

        result = harness.check()

        assert result.straggler_bitmap == [0, 0, 1, 0]
        assert result.straggler_confirmations == 2
        assert result.straggler_detail is not None
        assert result.straggler_detail.straggler_type == "compute"
        assert detector.replay_invocation.call_count == 2
        detector.localize_invocation_straggler.assert_called_once()

    def test_temporal_group_slowdown_uses_clean_history(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        harness = ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(
                check_interval=0,
                rotate_layers=False,
                temporal_min_samples=2,
                temporal_slowdown_ratio=1.2,
                straggler_confirmation_rounds=2,
                scale_factors=[],
                embedding_check_interval=0,
                output_check_interval=0,
                optimizer_check_interval=0,
            ),
            layers=model.layers,
        )
        model(torch.randint(0, 100, (2, 8)))
        detector = MagicMock()
        detector.replay_shape_consensus.return_value = (True, [0, 0, 0, 0])

        def replay_at(value):
            return ReplayResult(
                sdc_bitmap=[0, 0, 0, 0],
                straggler_bitmap=[0, 0, 0, 0],
                replay_time_ms=value,
                layer_id=0,
                peer_ranks=[0, 1, 2, 3],
                replay_times_ms=[value] * 4,
                sdc_source_bitmaps={"output": [0, 0, 0, 0]},
                spatial_straggler_bitmap=[0, 0, 0, 0],
            )

        detector.replay_invocation.side_effect = [
            replay_at(10.0),
            replay_at(10.0),
            replay_at(15.0),
            replay_at(15.0),
        ]
        detector.localize_invocation_straggler.return_value = StragglerDetail(
            straggler_rank=None,
            straggler_type="none",
            compute_times_ms=[12.0] * 4,
            comm_times_ms=[3.0] * 4,
            compute_bitmap=[0, 0, 0, 0],
        )
        harness._detector = detector

        assert harness.check().temporal_group_slowdown is False
        assert harness.check().temporal_group_slowdown is False
        result = harness.check()

        assert result.temporal_group_slowdown is True
        assert result.temporal_straggler_bitmap == [1, 1, 1, 1]
        assert result.straggler_confirmations == 2
        assert result.straggler_detail is not None
        assert result.straggler_detail.straggler_type == "shared_compute"


class TestHookRemoval:
    def test_distributed_harness_starts_and_stops_oob_daemon_automatically(self):
        model = LlamaLikeModel(num_layers=2, hidden_dim=32)
        oob_callback = MagicMock()

        with (
            patch.object(dist, "is_initialized", return_value=True),
            patch.object(dist, "get_world_size", return_value=4),
            patch.object(dist, "get_rank", return_value=1),
            patch(
                "lm_resiliency.detection.replay_harness.HangInstrumentation"
            ) as instrumentation_cls,
            patch("lm_resiliency.detection.replay_harness.OOBHangService") as service_cls,
        ):
            harness = ModelReplayHarness(
                model,
                config=ReplayHarnessConfig(check_interval=7),
                layers=model.layers,
                oob_fault_callback=oob_callback,
            )

            assert service_cls.call_args.kwargs["report_callback"] is oob_callback
            service_cls.return_value.start.assert_called_once()
            instrumentation_cls.assert_called_once_with(
                model,
                model.layers,
                1,
                progress_event=service_cls.return_value.progress_event,
            )
            dataloader = harness.instrument_dataloader([1, 2])
            assert dataloader._scout_detection_interval == 7
            harness.remove_hooks()

        service_cls.return_value.close.assert_called_once()
        instrumentation_cls.return_value.close.assert_called_once()

    def test_remove_hooks(self):
        model = LlamaLikeModel(num_layers=4, hidden_dim=32)
        config = ReplayHarnessConfig(layer_index=0, check_interval=0)
        harness = ModelReplayHarness(model, config=config, layers=model.layers)

        x = torch.randint(0, 100, (2, 8))
        _ = model(x)
        assert harness.has_capture

        harness.remove_hooks()
        harness._activation = None

        _ = model(x)
        # After removal, hooks no longer fire
        assert not harness.has_capture
