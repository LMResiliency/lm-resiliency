"""Unit tests for DeepSpeed adapter — verifies tensor collection and restoration."""

from __future__ import annotations

import torch
import torch.nn as nn

from lm_resiliency.integrations.deepspeed.adapter import (
    DeepSpeedAdapter,
    _find_transformer_layers,
)


class FakeBaseOptimizer:
    """Simulates torch.optim.Adam with param_groups and state."""

    def __init__(self, params: list[torch.Tensor]):
        self.param_groups = [{"params": params}]
        self.state: dict[torch.Tensor, dict[str, torch.Tensor]] = {}
        for p in params:
            self.state[p] = {
                "exp_avg": torch.zeros_like(p),
                "exp_avg_sq": torch.zeros_like(p),
                "step": torch.tensor(1),
            }


class FakeZeROStage2Optimizer:
    """Simulates DeepSpeedZeroOptimizer (Stage 1/2) data structures."""

    def __init__(self, hidden_size: int = 64, num_groups: int = 2):
        self.bit16_groups_flat = []
        self.single_partition_of_fp32_groups = []

        for _ in range(num_groups):
            flat_bf16 = torch.randn(hidden_size, dtype=torch.bfloat16)
            fp32_partition = torch.randn(hidden_size // 2, dtype=torch.float32)
            self.bit16_groups_flat.append(flat_bf16)
            self.single_partition_of_fp32_groups.append(fp32_partition)

        self.optimizer = FakeBaseOptimizer(self.single_partition_of_fp32_groups)


class FakeZeROStage3Optimizer:
    """Simulates DeepSpeedZeroOptimizer_Stage3 data structures."""

    def __init__(self, hidden_size: int = 64, num_subgroups: int = 2):
        self.fp16_partitioned_groups_flat = []
        self.fp32_partitioned_groups_flat = []

        for _ in range(num_subgroups):
            fp16_partition = torch.randn(hidden_size // 4, dtype=torch.float16)
            fp32_partition = torch.randn(hidden_size // 4, dtype=torch.float32)
            self.fp16_partitioned_groups_flat.append(fp16_partition)
            self.fp32_partitioned_groups_flat.append(fp32_partition)

        self.optimizer = FakeBaseOptimizer(self.fp32_partitioned_groups_flat)


class FakeDeepSpeedEngine:
    """Simulates DeepSpeed engine."""

    def __init__(self, stage: int = 2, hidden_size: int = 64):
        self._stage = stage
        self.module = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.global_rank = 0
        self.world_size = 8
        self.dp_world_size = 8
        self.data_parallel_group = None
        self.global_steps = 0
        self.lr_scheduler = None

        if stage <= 2:
            self.optimizer = FakeZeROStage2Optimizer(hidden_size)
        else:
            self.optimizer = FakeZeROStage3Optimizer(hidden_size)

    def zero_optimization_stage(self) -> int:
        return self._stage


class TestDeepSpeedAdapterStage2:
    """Tests for ZeRO Stage 1/2 tensor collection."""

    def setup_method(self):
        self.engine = FakeDeepSpeedEngine(stage=2, hidden_size=64)
        self.adapter = DeepSpeedAdapter(self.engine)

    def test_collect_returns_tensors(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        assert len(tensors) > 0
        assert all(isinstance(t, torch.Tensor) for t in tensors)

    def test_collect_includes_bf16_flat_groups(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        opt = self.engine.optimizer
        for flat_group in opt.bit16_groups_flat:
            assert any(t.data_ptr() == flat_group.data_ptr() for t in tensors)

    def test_collect_includes_fp32_partitions(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        opt = self.engine.optimizer
        for fp32_part in opt.single_partition_of_fp32_groups:
            assert any(t.data_ptr() == fp32_part.data_ptr() for t in tensors)

    def test_collect_includes_optimizer_state(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        opt = self.engine.optimizer
        base_opt = opt.optimizer
        for fp32_part in opt.single_partition_of_fp32_groups:
            state = base_opt.state[fp32_part]
            for v in state.values():
                if isinstance(v, torch.Tensor):
                    assert any(t.data_ptr() == v.data_ptr() for t in tensors)

    def test_collect_deterministic_order(self):
        t1 = self.adapter.collect_checkpoint_tensors()
        t2 = self.adapter.collect_checkpoint_tensors()
        assert len(t1) == len(t2)
        for a, b in zip(t1, t2):
            assert a.data_ptr() == b.data_ptr()

    def test_exposes_base_optimizer_for_transition_replay(self):
        assert self.adapter.get_base_optimizers() == [self.engine.optimizer.optimizer]

    def test_exposes_partial_offload_backup_optimizer(self):
        backup = torch.optim.SGD([torch.zeros(4)], lr=0.1)
        self.engine.optimizer.backup_optimizer = backup

        assert self.adapter.get_base_optimizers() == [
            self.engine.optimizer.optimizer,
            backup,
        ]

    def test_load_checkpoint_tensors_restores(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        saved = [t.clone() for t in tensors]

        # Corrupt live tensors
        for t in tensors:
            t.fill_(999.0)

        self.adapter.load_checkpoint_tensors(saved)

        live = self.adapter.collect_checkpoint_tensors()
        for live_t, saved_t in zip(live, saved):
            assert torch.allclose(live_t.float(), saved_t.float())

    def test_tensor_count_stage2(self):
        """Stage 2: 2 bf16_flat + 2 fp32_partitions + optimizer state tensors."""
        tensors = self.adapter.collect_checkpoint_tensors()
        # 2 bf16 flat + 2 fp32 partitions + 2×3 state tensors (exp_avg, exp_avg_sq, step)
        assert len(tensors) == 2 + 2 + 6


class TestDeepSpeedAdapterStage3:
    """Tests for ZeRO Stage 3 tensor collection."""

    def setup_method(self):
        self.engine = FakeDeepSpeedEngine(stage=3, hidden_size=64)
        self.adapter = DeepSpeedAdapter(self.engine)

    def test_collect_returns_tensors(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        assert len(tensors) > 0

    def test_collect_includes_fp16_partitions(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        opt = self.engine.optimizer
        for fp16_part in opt.fp16_partitioned_groups_flat:
            assert any(t.data_ptr() == fp16_part.data_ptr() for t in tensors)

    def test_collect_includes_fp32_partitions(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        opt = self.engine.optimizer
        for fp32_part in opt.fp32_partitioned_groups_flat:
            assert any(t.data_ptr() == fp32_part.data_ptr() for t in tensors)

    def test_load_checkpoint_tensors_restores(self):
        tensors = self.adapter.collect_checkpoint_tensors()
        saved = [t.clone() for t in tensors]

        for t in tensors:
            t.fill_(0.0)

        self.adapter.load_checkpoint_tensors(saved)

        live = self.adapter.collect_checkpoint_tensors()
        for live_t, saved_t in zip(live, saved):
            assert torch.allclose(live_t.float(), saved_t.float())

    def test_tensor_count_stage3(self):
        """Stage 3: 2 fp16_partitions + 2 fp32_partitions + optimizer state."""
        tensors = self.adapter.collect_checkpoint_tensors()
        # 2 fp16 + 2 fp32 + 2×3 state tensors
        assert len(tensors) == 2 + 2 + 6


class TestParallelismInfo:
    def test_stage2_has_replicas(self):
        engine = FakeDeepSpeedEngine(stage=2)
        adapter = DeepSpeedAdapter(engine)
        info = adapter.get_parallelism_info()
        assert info.dp_replicate == 8
        assert info.dp_shard == 1
        assert info.has_natural_replicas

    def test_stage3_no_replicas(self):
        engine = FakeDeepSpeedEngine(stage=3)
        adapter = DeepSpeedAdapter(engine)
        info = adapter.get_parallelism_info()
        assert info.dp_replicate == 1
        assert info.dp_shard == 8
        assert not info.has_natural_replicas


class TestFindTransformerLayers:
    """Tests for _find_transformer_layers helper."""

    def test_finds_model_layers(self):
        """HuggingFace pattern: model.model.layers."""

        class FakeModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = nn.Module()
                self.model.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(12)])

        result = _find_transformer_layers(FakeModel())
        assert result is not None
        assert len(result) == 12

    def test_finds_transformer_h(self):
        """GPT-2 pattern: model.transformer.h."""

        class FakeGPT2(nn.Module):
            def __init__(self):
                super().__init__()
                self.transformer = nn.Module()
                self.transformer.h = nn.ModuleList([nn.Linear(4, 4) for _ in range(6)])

        result = _find_transformer_layers(FakeGPT2())
        assert result is not None
        assert len(result) == 6

    def test_finds_direct_layers(self):
        """Direct pattern: model.layers."""

        class FakeLlama(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(8)])

        result = _find_transformer_layers(FakeLlama())
        assert result is not None
        assert len(result) == 8

    def test_finds_encoder_layers(self):
        """Encoder pattern: model.encoder.layers."""

        class FakeEncoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Module()
                self.encoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(4)])

        result = _find_transformer_layers(FakeEncoder())
        assert result is not None
        assert len(result) == 4

    def test_fallback_largest_module_list(self):
        """Fallback: finds largest ModuleList."""

        class FakeCustom(nn.Module):
            def __init__(self):
                super().__init__()
                self.blocks = nn.ModuleList([nn.Linear(4, 4) for _ in range(10)])
                self.heads = nn.ModuleList([nn.Linear(4, 4) for _ in range(2)])

        result = _find_transformer_layers(FakeCustom())
        assert result is not None
        assert len(result) == 10

    def test_returns_none_for_no_layers(self):
        """Returns None if no suitable ModuleList found."""

        class FakeFlat(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(4, 4)

        result = _find_transformer_layers(FakeFlat())
        assert result is None
