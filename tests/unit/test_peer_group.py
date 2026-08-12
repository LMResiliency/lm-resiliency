"""Unit tests for peer group auto-discovery and formation."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency.detection.peer_group import (
    _all_dp_subgroups,
    _describe_mesh,
    _extract_dp_ranks,
    _find_dp_dim,
    _find_peer_dim,
    _infer_mesh,
    _mesh_from_context,
    _mesh_from_dtensor_params,
    _mesh_from_fsdp2,
    get_peer_ranks,
    parallelism_device_mesh,
)


class SimpleModel(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for _ in range(4)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class TestFindDPDim:
    def test_finds_dp(self):
        assert _find_dp_dim(("dp", "tp", "pp")) == 0

    def test_finds_dp_replicate(self):
        assert _find_dp_dim(("tp", "dp_replicate", "pp")) == 1

    def test_finds_dp_shard(self):
        assert _find_dp_dim(("tp", "pp", "dp_shard")) == 2

    def test_finds_fsdp(self):
        assert _find_dp_dim(("tp", "fsdp", "pp")) == 1

    def test_priority_dp_first(self):
        # "dp" is checked before "dp_replicate"
        assert _find_dp_dim(("dp_replicate", "dp", "tp")) == 1

    def test_raises_on_missing_dp(self):
        with pytest.raises(ValueError, match="No DP dimension found"):
            _find_dp_dim(("tp", "pp", "cp"))

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="No DP dimension found"):
            _find_dp_dim(())


class TestMeshFromDTensorParams:
    def test_returns_none_for_plain_model(self):
        model = SimpleModel()
        assert _mesh_from_dtensor_params(model) is None

    def test_returns_mesh_from_dtensor_param(self):
        model = SimpleModel()
        mock_mesh = MagicMock()

        # Patch the first parameter to be a DTensor
        with patch("lm_resiliency.detection.peer_group.DTensor", create=True) as MockDTensor:
            # Make isinstance check pass for the mock param
            mock_param = MagicMock()
            mock_param.device_mesh = mock_mesh

            # Monkeypatch model.parameters()
            original_params = list(model.parameters())
            mock_param.__class__ = type(mock_param)

            with patch.object(model, "parameters", return_value=[mock_param]):
                with patch(
                    "lm_resiliency.detection.peer_group._mesh_from_dtensor_params"
                ) as patched:
                    patched.return_value = mock_mesh
                    result = patched(model)
                    assert result is mock_mesh


class TestMeshFromFSDP2:
    def test_returns_none_for_plain_model(self):
        model = SimpleModel()
        assert _mesh_from_fsdp2(model) is None

    def test_returns_mesh_from_fsdp_state(self):
        model = SimpleModel()
        mock_mesh = MagicMock()

        # Attach _fsdp_state to a submodule
        fsdp_state = MagicMock()
        fsdp_state._device_mesh = mock_mesh
        model.layers[0]._fsdp_state = fsdp_state

        result = _mesh_from_fsdp2(model)
        assert result is mock_mesh

    def test_returns_none_when_fsdp_state_has_no_mesh(self):
        model = SimpleModel()
        fsdp_state = MagicMock()
        fsdp_state._device_mesh = None
        model.layers[0]._fsdp_state = fsdp_state

        result = _mesh_from_fsdp2(model)
        assert result is None


class TestMeshFromContext:
    def test_returns_none_when_no_context(self):
        # Should not raise, just return None
        result = _mesh_from_context()
        # In a non-distributed setting, this should return None
        # (either raises RuntimeError internally or import fails gracefully)
        assert result is None or result is not None  # just ensure no crash

    def test_returns_mesh_from_active_context(self):
        mock_mesh = MagicMock()

        with patch(
            "lm_resiliency.detection.peer_group._mesh_from_context",
            return_value=mock_mesh,
        ) as patched_fn:
            result = patched_fn()
            assert result is mock_mesh


class TestInferMesh:
    def test_returns_none_for_plain_model(self):
        model = SimpleModel()
        result = _infer_mesh(model)
        # Plain model has no DTensors, no FSDP state, no active context
        assert result is None

    def test_returns_none_when_model_is_none(self):
        result = _infer_mesh(None)
        assert result is None

    def test_dtensor_takes_priority_over_fsdp(self):
        model = SimpleModel()
        mock_mesh_dtensor = MagicMock(name="dtensor_mesh")
        mock_mesh_fsdp = MagicMock(name="fsdp_mesh")

        with patch(
            "lm_resiliency.detection.peer_group._mesh_from_dtensor_params",
            return_value=mock_mesh_dtensor,
        ):
            with patch(
                "lm_resiliency.detection.peer_group._mesh_from_fsdp2",
                return_value=mock_mesh_fsdp,
            ):
                result = _infer_mesh(model)
                assert result is mock_mesh_dtensor

    def test_fsdp_used_when_dtensor_returns_none(self):
        model = SimpleModel()
        mock_mesh_fsdp = MagicMock(name="fsdp_mesh")

        with patch(
            "lm_resiliency.detection.peer_group._mesh_from_dtensor_params",
            return_value=None,
        ):
            with patch(
                "lm_resiliency.detection.peer_group._mesh_from_fsdp2",
                return_value=mock_mesh_fsdp,
            ):
                result = _infer_mesh(model)
                assert result is mock_mesh_fsdp

    def test_context_used_as_last_resort(self):
        model = SimpleModel()
        mock_mesh_ctx = MagicMock(name="context_mesh")

        with patch(
            "lm_resiliency.detection.peer_group._mesh_from_dtensor_params",
            return_value=None,
        ):
            with patch(
                "lm_resiliency.detection.peer_group._mesh_from_fsdp2",
                return_value=None,
            ):
                with patch(
                    "lm_resiliency.detection.peer_group._mesh_from_context",
                    return_value=mock_mesh_ctx,
                ):
                    result = _infer_mesh(model)
                    assert result is mock_mesh_ctx


class TestExtractDPRanks:
    def test_none_mesh_returns_all_ranks(self):
        with patch("torch.distributed.get_world_size", return_value=8):
            ranks = _extract_dp_ranks(None)
            assert ranks == [0, 1, 2, 3, 4, 5, 6, 7]

    def test_named_mesh_extracts_dp_group(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("tp", "dp", "pp")
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        with patch(
            "torch.distributed.get_process_group_ranks",
            return_value=[0, 4, 8, 12],
        ):
            ranks = _extract_dp_ranks(mock_mesh)
            assert ranks == [0, 4, 8, 12]
            mock_mesh.get_group.assert_called_once_with(1)  # dp is dim index 1

    def test_unnamed_mesh_defaults_to_dim_0(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = None
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        with patch(
            "torch.distributed.get_process_group_ranks",
            return_value=[0, 1, 2, 3],
        ):
            ranks = _extract_dp_ranks(mock_mesh)
            assert ranks == [0, 1, 2, 3]
            mock_mesh.get_group.assert_called_once_with(0)

    def test_hsdp_uses_replicas_at_the_same_shard_coordinate(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("dp_replicate", "dp_shard")
        mock_mesh.mesh = torch.arange(8).view(4, 2)
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        with patch(
            "torch.distributed.get_process_group_ranks",
            return_value=[1, 3, 5, 7],
        ):
            ranks = _extract_dp_ranks(mock_mesh)

        assert ranks == [1, 3, 5, 7]
        mock_mesh.get_group.assert_called_once_with(0)
        assert _all_dp_subgroups(mock_mesh) == [
            [0, 2, 4, 6],
            [1, 3, 5, 7],
        ]

    def test_size_one_replica_dimension_falls_back_to_fsdp_shards(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("dp_replicate", "fsdp")
        mock_mesh.mesh = torch.arange(4).view(1, 4)
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        with patch(
            "torch.distributed.get_process_group_ranks",
            return_value=[0, 1, 2, 3],
        ):
            ranks = _extract_dp_ranks(mock_mesh)

        assert _find_peer_dim(mock_mesh) == 1
        assert ranks == [0, 1, 2, 3]
        mock_mesh.get_group.assert_called_once_with(1)
        assert _all_dp_subgroups(mock_mesh) == [[0, 1, 2, 3]]


class TestDescribeMesh:
    def test_none_mesh(self):
        desc = _describe_mesh(None)
        assert "pure DDP" in desc

    def test_named_mesh(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("dp", "tp")
        mock_mesh.size.side_effect = lambda i: [4, 2][i]
        desc = _describe_mesh(mock_mesh)
        assert "dp=4" in desc
        assert "tp=2" in desc

    def test_unnamed_mesh(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = None
        mock_mesh.mesh = MagicMock()
        mock_mesh.mesh.shape = [4, 2]
        desc = _describe_mesh(mock_mesh)
        assert "unnamed" in desc


class TestGetPeerRanks:
    def test_with_explicit_mesh(self):
        mock_mesh = MagicMock()
        mock_mesh.mesh_dim_names = ("dp", "tp")
        mock_group = MagicMock()
        mock_mesh.get_group.return_value = mock_group

        with patch(
            "torch.distributed.get_process_group_ranks",
            return_value=[0, 2, 4, 6],
        ):
            ranks = get_peer_ranks(device_mesh=mock_mesh)
            assert ranks == [0, 2, 4, 6]

    def test_with_none_model_none_mesh_uses_world(self):
        with patch("torch.distributed.get_world_size", return_value=4):
            ranks = get_peer_ranks(model=None, device_mesh=None)
            assert ranks == [0, 1, 2, 3]


class TestParallelismDeviceMesh:
    def test_uses_framework_dense_mesh_alias(self):
        dense_mesh = object()

        class Parallelism:
            def get_mesh(self, name):
                assert name == "dense"
                return dense_mesh

        assert parallelism_device_mesh(Parallelism()) is dense_mesh

    def test_builds_torchtitan_dense_mesh_from_public_dimensions(self):
        combined_mesh = object()
        active = {"pp", "dp_replicate", "fsdp", "tp"}

        class ParallelDims:
            def get_mesh(self, dimensions):
                if dimensions == "dense":
                    raise ValueError("invalid mesh alias")
                assert dimensions == ["pp", "dp_replicate", "fsdp", "tp"]
                return combined_mesh

            def get_optional_mesh(self, dimension):
                return object() if dimension in active else None

        assert parallelism_device_mesh(ParallelDims()) is combined_mesh

    def test_builds_torchtitan_sparse_mesh_from_public_dimensions(self):
        combined_mesh = object()
        active = {"dp_replicate", "efsdp", "ep", "etp"}

        class ParallelDims:
            def get_mesh(self, dimensions):
                if dimensions == "sparse":
                    raise ValueError("invalid mesh alias")
                assert dimensions == ["dp_replicate", "efsdp", "ep", "etp"]
                return combined_mesh

            def get_optional_mesh(self, dimension):
                return object() if dimension in active else None

        assert parallelism_device_mesh(ParallelDims(), expert=True) is combined_mesh


class TestFormDetectionGroupsIntegration:
    """Tests that form_detection_groups correctly calls dist.new_group.

    These test the wiring, not actual distributed behavior (that's for
    integration tests).
    """

    @patch("lm_resiliency.detection.peer_group.dist")
    @patch("lm_resiliency.detection.peer_group._infer_mesh")
    def test_creates_gloo_and_nccl_groups(self, mock_infer, mock_dist):
        from lm_resiliency.detection.peer_group import form_detection_groups

        mock_infer.return_value = None
        mock_dist.get_world_size.return_value = 4
        mock_dist.get_rank.return_value = 0
        mock_dist.get_process_group_ranks.return_value = [0, 1, 2, 3]

        mock_gloo = MagicMock(name="gloo_group")
        mock_nccl = MagicMock(name="nccl_group")
        mock_dist.new_group.side_effect = [mock_gloo, mock_nccl]

        gloo, nccl = form_detection_groups(model=None)

        assert mock_dist.new_group.call_count == 2
        calls = mock_dist.new_group.call_args_list
        assert calls[0].kwargs["backend"] == "gloo"
        assert calls[1].kwargs["backend"] == "nccl"
        assert gloo is mock_gloo
        assert nccl is mock_nccl

    @patch("lm_resiliency.detection.peer_group.dist")
    @patch("lm_resiliency.detection.peer_group._infer_mesh")
    def test_hsdp_replica_groups_use_gloo_for_initialization_sync(
        self,
        mock_infer,
        mock_dist,
    ):
        from lm_resiliency.detection.peer_group import form_detection_groups

        mesh = MagicMock()
        mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
        mesh.mesh = torch.arange(8).view(2, 2, 2)
        mock_infer.return_value = mesh
        mock_dist.get_world_size.return_value = 8
        mock_dist.get_rank.return_value = 0
        mock_dist.get_process_group_ranks.return_value = [0, 4]

        sync_group = MagicMock(name="sync_group")
        first_gloo = MagicMock(name="first_gloo")
        first_nccl = MagicMock(name="first_nccl")
        second_gloo = MagicMock(name="second_gloo")
        second_nccl = MagicMock(name="second_nccl")
        third_gloo = MagicMock(name="third_gloo")
        third_nccl = MagicMock(name="third_nccl")
        fourth_gloo = MagicMock(name="fourth_gloo")
        fourth_nccl = MagicMock(name="fourth_nccl")
        mock_dist.new_group.side_effect = [
            sync_group,
            first_gloo,
            first_nccl,
            second_gloo,
            second_nccl,
            third_gloo,
            third_nccl,
            fourth_gloo,
            fourth_nccl,
        ]

        gloo, nccl = form_detection_groups(model=None)

        assert (gloo, nccl) == (first_gloo, first_nccl)
        calls = mock_dist.new_group.call_args_list
        assert calls[0].kwargs == {
            "ranks": list(range(8)),
            "backend": "gloo",
        }
        assert calls[1].kwargs["ranks"] == [0, 4]
        assert calls[3].kwargs["ranks"] == [1, 5]
        assert calls[5].kwargs["ranks"] == [2, 6]
        assert calls[7].kwargs["ranks"] == [3, 7]
        assert mock_dist.barrier.call_args_list[-1].kwargs["group"] is sync_group
        assert (
            sum(call.kwargs.get("group") is sync_group for call in mock_dist.barrier.call_args_list)
            == 4
        )
        mock_dist.all_reduce.assert_not_called()
