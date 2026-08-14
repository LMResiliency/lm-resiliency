"""Unit tests for Megatron-core adapter.

Tests model unwrapping, layer discovery, state dict extraction/loading,
parallelism info computation, and DP-sharded optimizer handling.
All tests use mocks to avoid requiring megatron-core installation.
"""

import sys
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

# Mock megatron.core.mpu before importing adapter
mock_mpu = MagicMock()
mock_mpu.get_tensor_model_parallel_world_size.return_value = 2
mock_mpu.get_pipeline_model_parallel_world_size.return_value = 2
mock_mpu.get_data_parallel_group.return_value = None
sys.modules["megatron"] = MagicMock()
sys.modules["megatron.core"] = MagicMock(mpu=mock_mpu)
sys.modules["megatron.core.mpu"] = mock_mpu

from lm_resiliency.integrations.megatron.adapter import (  # noqa: E402
    MegatronAdapter,
    _find_transformer_layers,
    _get_base_optimizers,
    _optimizer_is_distributed,
    _unwrap_model_chunk,
)

# ──────────────────────────────────────────────────────────────────────────────
# Fake model/optimizer classes
# ──────────────────────────────────────────────────────────────────────────────


class FakeTransformerBlock(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(32, 32) for _ in range(num_layers)])


class FakeGPTModel(nn.Module):
    def __init__(self, num_layers=4):
        super().__init__()
        self.decoder = FakeTransformerBlock(num_layers)


class FakeEncoderModel(nn.Module):
    def __init__(self, num_layers=6):
        super().__init__()
        self.encoder = FakeTransformerBlock(num_layers)


class FakeDDPWrapper:
    """Simulates Megatron's DistributedDataParallel wrapper."""

    def __init__(self, module):
        self.module = module


class FakeFloat16Module:
    """Simulates Megatron's Float16Module → DDP → MegatronModule nesting."""

    def __init__(self, module):
        self.module = FakeDDPWrapper(module)


class FakeDistributedOptimizer:
    """Simulates Megatron's DistributedOptimizer."""

    def __init__(self):
        self._state = {"param_groups": [{"lr": 0.001}]}
        self._param_state = {"shard_0": torch.zeros(10)}

    def state_dict(self):
        return self._state.copy()

    def load_state_dict(self, state_dict):
        self._state.update(state_dict)

    def save_parameter_state(self, dest: dict):
        dest.update(self._param_state)

    def load_parameter_state(self, state: dict):
        self._param_state.update(state)

    def step(self):
        return True, 1.0, 0


class FakeRegularOptimizer:
    """Non-distributed optimizer (no save_parameter_state)."""

    def state_dict(self):
        return {"lr": 0.01}

    def load_state_dict(self, state_dict):
        pass


class FakeLRScheduler:
    def __init__(self):
        self._step_count = 0
        self._lr = 0.001

    def state_dict(self):
        return {"step_count": self._step_count, "lr": self._lr}

    def load_state_dict(self, state_dict):
        self._step_count = state_dict.get("step_count", 0)
        self._lr = state_dict.get("lr", 0.001)


# ──────────────────────────────────────────────────────────────────────────────
# _unwrap_model_chunk tests
# ──────────────────────────────────────────────────────────────────────────────


class TestUnwrapModelChunk:
    def test_no_wrapper(self):
        model = FakeGPTModel()
        assert _unwrap_model_chunk(model) is model

    def test_single_wrapper(self):
        model = FakeGPTModel()
        wrapped = FakeDDPWrapper(model)
        assert _unwrap_model_chunk(wrapped) is model

    def test_double_wrapper(self):
        """Float16Module → DDP → base model."""
        model = FakeGPTModel()
        wrapped = FakeFloat16Module(model)
        assert _unwrap_model_chunk(wrapped) is model

    def test_preserves_module_type(self):
        model = FakeGPTModel(num_layers=3)
        wrapped = FakeDDPWrapper(model)
        result = _unwrap_model_chunk(wrapped)
        assert isinstance(result, FakeGPTModel)
        assert hasattr(result, "decoder")


# ──────────────────────────────────────────────────────────────────────────────
# _find_transformer_layers tests
# ──────────────────────────────────────────────────────────────────────────────


class TestFindTransformerLayers:
    def test_gpt_model_decoder_pattern(self):
        model = FakeGPTModel(num_layers=6)
        layers = _find_transformer_layers(model)
        assert layers is not None
        assert len(layers) == 6
        assert isinstance(layers, nn.ModuleList)

    def test_encoder_pattern(self):
        model = FakeEncoderModel(num_layers=8)
        layers = _find_transformer_layers(model)
        assert layers is not None
        assert len(layers) == 8

    def test_decoder_takes_priority_over_encoder(self):
        """If both decoder and encoder exist, decoder is found first."""
        model = nn.Module()
        model.decoder = FakeTransformerBlock(4)
        model.encoder = FakeTransformerBlock(6)
        layers = _find_transformer_layers(model)
        assert len(layers) == 4

    def test_fallback_transformer_block_by_classname(self):
        """Falls back to searching for class named TransformerBlock."""

        class TransformerBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList([nn.Linear(16, 16) for _ in range(3)])

        model = nn.Module()
        model.sub = nn.Module()
        model.sub.block = TransformerBlock()
        layers = _find_transformer_layers(model)
        assert layers is not None
        assert len(layers) == 3

    def test_no_layers_found(self):
        model = nn.Linear(10, 10)
        assert _find_transformer_layers(model) is None

    def test_empty_module(self):
        model = nn.Module()
        assert _find_transformer_layers(model) is None

    def test_non_modulelist_decoder_layers(self):
        """Decoder.layers is not a ModuleList — should not match."""
        model = nn.Module()
        model.decoder = nn.Module()
        model.decoder.layers = [nn.Linear(10, 10)]  # plain list, not ModuleList
        assert _find_transformer_layers(model) is None


# ──────────────────────────────────────────────────────────────────────────────
# _optimizer_is_distributed tests
# ──────────────────────────────────────────────────────────────────────────────


class TestOptimizerIsDistributed:
    def test_distributed_optimizer(self):
        class DistributedOptimizer:
            pass

        assert _optimizer_is_distributed(DistributedOptimizer()) is True

    def test_mixed_precision_distributed_optimizer(self):
        class MixedPrecisionDistributedOptimizer:
            pass

        assert _optimizer_is_distributed(MixedPrecisionDistributedOptimizer()) is True

    def test_regular_optimizer(self):
        opt = torch.optim.SGD([torch.zeros(1)], lr=0.01)
        assert _optimizer_is_distributed(opt) is False

    def test_chained_distributed_optimizer(self):
        class DistributedOptimizer:
            pass

        class ChainedOptimizer:
            chained_optimizers = [DistributedOptimizer()]

        assert _optimizer_is_distributed(ChainedOptimizer()) is True

    def test_adam_optimizer(self):
        opt = torch.optim.Adam([torch.zeros(1)], lr=0.001)
        assert _optimizer_is_distributed(opt) is False


class TestBaseOptimizers:
    def test_unwraps_distributed_optimizer(self):
        base = torch.optim.AdamW([torch.zeros(4)], lr=0.01)

        class WrappedOptimizer:
            optimizer = base

        assert _get_base_optimizers(WrappedOptimizer()) == [base]

    def test_unwraps_every_chained_optimizer(self):
        first = torch.optim.AdamW([torch.zeros(4)], lr=0.01)
        second = torch.optim.SGD([torch.zeros(4)], lr=0.1)

        class WrappedOptimizer:
            def __init__(self, optimizer):
                self.optimizer = optimizer

        class ChainedOptimizer:
            chained_optimizers = [WrappedOptimizer(first), WrappedOptimizer(second)]

        assert _get_base_optimizers(ChainedOptimizer()) == [first, second]


# ──────────────────────────────────────────────────────────────────────────────
# MegatronAdapter tests
# ──────────────────────────────────────────────────────────────────────────────


class TestMegatronAdapterStateDict:
    def setup_method(self):
        self.model = FakeGPTModel(num_layers=4)
        self.wrapped_model = FakeDDPWrapper(self.model)
        self.optimizer = FakeDistributedOptimizer()
        self.scheduler = FakeLRScheduler()

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_get_state_dict_contains_all_keys(self, *mocks):
        adapter = MegatronAdapter(
            model=[self.wrapped_model],
            optimizer=self.optimizer,
            opt_param_scheduler=self.scheduler,
        )
        state = adapter.get_state_dict()

        assert "model_0" in state
        assert "optimizer" in state
        assert "optimizer_param_state" in state
        assert "opt_param_scheduler" in state

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_model_state_contains_parameters(self, *mocks):
        adapter = MegatronAdapter(model=[self.wrapped_model], optimizer=self.optimizer)
        state = adapter.get_state_dict()

        model_state = state["model_0"]
        assert "decoder.layers.0.weight" in model_state
        assert "decoder.layers.0.bias" in model_state
        assert "decoder.layers.3.weight" in model_state

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_optimizer_param_state_saved(self, *mocks):
        adapter = MegatronAdapter(model=[self.wrapped_model], optimizer=self.optimizer)
        state = adapter.get_state_dict()

        assert "optimizer_param_state" in state
        assert "shard_0" in state["optimizer_param_state"]
        assert torch.is_tensor(state["optimizer_param_state"]["shard_0"])

    def test_uses_inter_optimizer_instance_group_as_replica_oracle(self):
        adapter = MegatronAdapter(model=[self.wrapped_model], optimizer=self.optimizer)
        replica_group = object()
        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch.object(
                mock_mpu,
                "get_inter_distributed_optimizer_instance_group",
                return_value=replica_group,
                create=True,
            ),
        ):
            assert adapter.get_optimizer_replica_group() is replica_group

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_no_param_state_for_regular_optimizer(self, *mocks):
        """Regular optimizer without save_parameter_state skips that key."""
        adapter = MegatronAdapter(model=[self.wrapped_model], optimizer=FakeRegularOptimizer())
        state = adapter.get_state_dict()

        assert "optimizer_param_state" not in state

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_scheduler_state_roundtrip(self, *mocks):
        self.scheduler._step_count = 42
        self.scheduler._lr = 0.0001

        adapter = MegatronAdapter(
            model=[self.wrapped_model],
            optimizer=self.optimizer,
            opt_param_scheduler=self.scheduler,
        )
        state = adapter.get_state_dict()

        # Reset and reload
        self.scheduler._step_count = 0
        self.scheduler._lr = 0.001
        adapter.load_state_dict(state)

        assert self.scheduler._step_count == 42
        assert self.scheduler._lr == 0.0001

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_state_dict_updates_model(self, *mocks):
        adapter = MegatronAdapter(model=[self.wrapped_model], optimizer=self.optimizer)

        # Save known weights, cloning to decouple from in-place ops
        with torch.no_grad():
            self.model.decoder.layers[0].weight.fill_(99.0)

        state = adapter.get_state_dict()
        # Clone model state to decouple from the live parameter tensors
        state["model_0"] = {k: v.clone() for k, v in state["model_0"].items()}

        # Reset weights
        with torch.no_grad():
            self.model.decoder.layers[0].weight.fill_(0.0)

        adapter.load_state_dict(state)
        assert self.model.decoder.layers[0].weight[0, 0].item() == 99.0

    @patch("lm_resiliency.integrations.megatron.adapter.notify_checkpoint_tensor_load")
    def test_load_checkpoint_tensors_notifies_replacement_observers(
        self,
        notify_checkpoint_tensor_load,
    ):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        adapter = MegatronAdapter(
            model=[self.wrapped_model],
            optimizer=optimizer,
        )
        tensors = adapter.collect_checkpoint_tensors()
        saved = [tensor.clone() for tensor in tensors]
        for tensor in tensors:
            tensor.zero_()

        adapter.load_checkpoint_tensors(saved)

        for live, expected in zip(adapter.collect_checkpoint_tensors(), saved):
            torch.testing.assert_close(live, expected)
        notify_checkpoint_tensor_load.assert_called_once_with(adapter)

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_extra_state_included(self, *mocks):
        adapter = MegatronAdapter(
            model=[self.wrapped_model],
            optimizer=self.optimizer,
            extra_state={"iteration": 100, "tokens_seen": 50000, "rng_state": [1, 2, 3]},
        )
        state = adapter.get_state_dict()

        assert state["iteration"] == 100
        assert state["tokens_seen"] == 50000
        assert state["rng_state"] == [1, 2, 3]

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_no_scheduler_skips_scheduler_key(self, *mocks):
        adapter = MegatronAdapter(
            model=[self.wrapped_model],
            optimizer=self.optimizer,
            opt_param_scheduler=None,
        )
        state = adapter.get_state_dict()
        assert "opt_param_scheduler" not in state


class TestMegatronAdapterVirtualPipeline:
    """Tests for virtual pipeline parallelism (multiple model chunks)."""

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_multiple_model_chunks(self, *mocks):
        chunk_0 = FakeDDPWrapper(FakeGPTModel(2))
        chunk_1 = FakeDDPWrapper(FakeGPTModel(3))
        adapter = MegatronAdapter(model=[chunk_0, chunk_1], optimizer=FakeDistributedOptimizer())

        state = adapter.get_state_dict()
        assert "model_0" in state
        assert "model_1" in state
        # Chunk 0 has 2 layers, chunk 1 has 3
        assert "decoder.layers.0.weight" in state["model_0"]
        assert "decoder.layers.1.weight" in state["model_0"]
        assert "decoder.layers.2.weight" in state["model_1"]

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_load_multiple_chunks(self, *mocks):
        model_0 = FakeGPTModel(2)
        model_1 = FakeGPTModel(2)
        chunk_0 = FakeDDPWrapper(model_0)
        chunk_1 = FakeDDPWrapper(model_1)
        adapter = MegatronAdapter(model=[chunk_0, chunk_1], optimizer=FakeDistributedOptimizer())

        # Set known weights
        with torch.no_grad():
            model_0.decoder.layers[0].weight.fill_(1.0)
            model_1.decoder.layers[0].weight.fill_(2.0)

        state = adapter.get_state_dict()
        # Clone to decouple from live parameter tensors
        state["model_0"] = {k: v.clone() for k, v in state["model_0"].items()}
        state["model_1"] = {k: v.clone() for k, v in state["model_1"].items()}

        # Zero out and reload
        with torch.no_grad():
            model_0.decoder.layers[0].weight.fill_(0.0)
            model_1.decoder.layers[0].weight.fill_(0.0)

        adapter.load_state_dict(state)
        assert model_0.decoder.layers[0].weight[0, 0].item() == 1.0
        assert model_1.decoder.layers[0].weight[0, 0].item() == 2.0

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_get_repeated_layers_from_all_virtual_pipeline_chunks(self, *mocks):
        chunk_0 = FakeDDPWrapper(FakeGPTModel(4))
        chunk_1 = FakeDDPWrapper(FakeGPTModel(4))
        adapter = MegatronAdapter(model=[chunk_0, chunk_1], optimizer=FakeDistributedOptimizer())

        layers = adapter.get_repeated_layers()
        assert layers is not None
        assert len(layers) == 8
        assert layers[:4] == list(chunk_0.module.decoder.layers)
        assert layers[4:] == list(chunk_1.module.decoder.layers)


class TestMegatronAdapterParallelismInfo:
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_tp2_pp2_dp2_distributed_optimizer(self, *mocks):
        """TP=2, PP=2, world=8 → DP=2, all sharded."""
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 2
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 2

        model = FakeDDPWrapper(FakeGPTModel())
        adapter = MegatronAdapter(model=[model], optimizer=FakeDistributedOptimizer())
        info = adapter.get_parallelism_info()

        assert info.tp == 2
        assert info.pp == 2
        assert info.world_size == 8
        assert info.dp_shard == 2
        assert info.dp_replicate == 1
        assert info.has_natural_replicas is False

    @patch("torch.distributed.get_world_size", return_value=16)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_tp4_pp1_dp4_regular_optimizer(self, *mocks):
        """TP=4, PP=1, world=16 → DP=4, fully replicated (no distributed opt)."""
        mock_mpu.get_tensor_model_parallel_world_size.return_value = 4
        mock_mpu.get_pipeline_model_parallel_world_size.return_value = 1

        model = FakeDDPWrapper(FakeGPTModel())
        adapter = MegatronAdapter(model=[model], optimizer=FakeRegularOptimizer())
        info = adapter.get_parallelism_info()

        assert info.tp == 4
        assert info.pp == 1
        assert info.dp_replicate == 4
        assert info.dp_shard == 1
        assert info.has_natural_replicas is True

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=5)
    def test_rank_and_world_size(self, *mocks):
        model = FakeDDPWrapper(FakeGPTModel())
        adapter = MegatronAdapter(model=[model], optimizer=FakeDistributedOptimizer())
        assert adapter.rank == 5
        assert adapter.world_size == 8


class TestMegatronAdapterDPGroup:
    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_get_dp_group_returns_mpu_group(self, *mocks):
        mock_group = MagicMock()
        mock_mpu.get_data_parallel_group.return_value = mock_group

        model = FakeDDPWrapper(FakeGPTModel())
        adapter = MegatronAdapter(model=[model], optimizer=FakeDistributedOptimizer())
        assert adapter.get_dp_group() is mock_group

        mock_mpu.get_data_parallel_group.return_value = None

    @patch("torch.distributed.get_world_size", return_value=8)
    @patch("torch.distributed.get_rank", return_value=0)
    def test_get_repeated_layers_none_when_no_layers(self, *mocks):
        """Model without recognizable layer structure returns None."""
        plain_model = nn.Linear(10, 10)
        model = FakeDDPWrapper(plain_model)
        adapter = MegatronAdapter(model=[model], optimizer=FakeDistributedOptimizer())
        assert adapter.get_repeated_layers() is None
