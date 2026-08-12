"""Unit tests for checkpoint replication backend setup."""

from unittest.mock import patch

import pytest
import torch
import torch.distributed as dist

from lm_resiliency.checkpointing.replication import (
    ChunkedGlooBackend,
    GlooBackend,
    _prepare_receive_buffers,
)


@pytest.mark.parametrize("backend_type", [GlooBackend, ChunkedGlooBackend])
def test_gloo_backends_share_segment_pairing_and_group_creation(backend_type):
    groups = [object(), object()]
    with (
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "new_group", side_effect=groups) as new_group,
    ):
        backend = backend_type(replication_jump=2, world_size=4, rank=3)

    assert backend.peer_rank == 1
    assert backend._group is groups[1]
    assert [call.kwargs["ranks"] for call in new_group.call_args_list] == [[0, 2], [1, 3]]


def test_chunked_backend_requires_distributed_before_validating_chunks():
    with patch.object(dist, "is_initialized", return_value=False):
        with pytest.raises(RuntimeError, match="ChunkedGlooBackend"):
            ChunkedGlooBackend(replication_jump=1, chunk_size=0)


def test_receive_buffers_follow_peer_tensor_layout():
    buffers = [torch.empty(2, dtype=torch.float32)]
    original_list = buffers

    _prepare_receive_buffers(
        buffers,
        [
            {"shape": [3, 4], "dtype": "float16"},
            {"shape": [5], "dtype": "int64"},
        ],
    )

    assert buffers is original_list
    assert [(tensor.shape, tensor.dtype) for tensor in buffers] == [
        (torch.Size([3, 4]), torch.float16),
        (torch.Size([5]), torch.int64),
    ]
