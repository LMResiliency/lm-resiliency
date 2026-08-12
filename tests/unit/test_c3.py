"""Unit tests for C³ outlier detection logic (no GPU, no distributed required)."""

from __future__ import annotations

import enum
from unittest.mock import patch

import pytest
import torch

from lm_resiliency.detection.c3 import (
    C3,
    C3Mode,
    C3Status,
    _compute_structure_signature,
    _compute_tensor_signature,
)


class TestTensorSignature:
    def test_deterministic(self):
        t = torch.randn(100)
        sig1 = _compute_tensor_signature(t)
        sig2 = _compute_tensor_signature(t)
        assert sig1 == sig2

    def test_different_tensors_different_sigs(self):
        t1 = torch.zeros(100)
        t2 = torch.ones(100)
        assert _compute_tensor_signature(t1) != _compute_tensor_signature(t2)

    def test_position_sensitive(self):
        t1 = torch.tensor([1.0, 2.0, 3.0, 4.0])
        t2 = torch.tensor([4.0, 3.0, 2.0, 1.0])
        assert _compute_tensor_signature(t1) != _compute_tensor_signature(t2)

    def test_single_element_difference(self):
        t1 = torch.zeros(1000)
        t2 = torch.zeros(1000)
        t2[500] = 1.0
        assert _compute_tensor_signature(t1) != _compute_tensor_signature(t2)

    def test_odd_fold_width_does_not_drop_elements(self):
        for tensor_length, changed_index in ((3072, 4), (5120, 8), (7168, 12)):
            baseline = torch.zeros(tensor_length)
            changed = baseline.clone()
            changed[changed_index] = 1.0
            assert _compute_tensor_signature(baseline) != _compute_tensor_signature(changed)

    def test_padding_does_not_hide_trailing_zero_length(self):
        assert _compute_tensor_signature(torch.zeros(1024)) != _compute_tensor_signature(
            torch.zeros(1025)
        )

    def test_shape_is_part_of_signature(self):
        flat = torch.arange(12, dtype=torch.float32)
        assert _compute_tensor_signature(flat) != _compute_tensor_signature(flat.reshape(3, 4))

    def test_dtype_is_part_of_signature(self):
        raw = torch.arange(8, dtype=torch.int32)
        assert _compute_tensor_signature(raw) != _compute_tensor_signature(raw.view(torch.float32))

    def test_large_tensor(self):
        t = torch.randn(100000)
        sig = _compute_tensor_signature(t)
        assert isinstance(sig, int)
        assert -(1 << 63) <= sig <= (1 << 63) - 1


class TestStructuredSignature:
    def test_nested_tensor_and_metadata_are_deterministic(self):
        value = {
            "activation": torch.arange(8).reshape(2, 4),
            "kwargs": {"scale": 0.5, "causal": True},
        }

        assert _compute_structure_signature(value) == _compute_structure_signature(value)

    def test_tensor_value_and_scalar_metadata_affect_signature(self):
        baseline = (torch.zeros(4), {"scale": 1.0})
        changed_tensor = (torch.tensor([0.0, 0.0, 1.0, 0.0]), {"scale": 1.0})
        changed_metadata = (torch.zeros(4), {"scale": 2.0})

        assert _compute_structure_signature(baseline) != _compute_structure_signature(
            changed_tensor
        )
        assert _compute_structure_signature(baseline) != _compute_structure_signature(
            changed_metadata
        )

    def test_mapping_order_does_not_affect_signature(self):
        first = {"torch_cpu": torch.arange(4), "python": (3, (1, 2), None)}
        second = {"python": (3, (1, 2), None), "torch_cpu": torch.arange(4)}

        assert _compute_structure_signature(first) == _compute_structure_signature(second)

    def test_optimizer_configuration_types_are_supported(self):
        class Mode(enum.Enum):
            FUSED = "fused"

        signature = _compute_structure_signature(
            {
                "dtype": torch.bfloat16,
                "device": torch.device("cpu"),
                "mode": Mode.FUSED,
                "optimizer": torch.optim.AdamW,
            }
        )

        assert isinstance(signature, int)


class TestOutlierDetection:
    """Test the outlier identification logic directly (no distributed)."""

    def test_exact_all_same(self):
        result = C3.classify_evidence([17, 17, 17, 17], C3Mode.EXACT)
        assert result.status is C3Status.AGREE
        assert result.bitmap == [0, 0, 0, 0]
        assert result.evidence == [17, 17, 17, 17]

    def test_exact_one_outlier(self):
        result = C3.classify_evidence([17, 17, 15, 17], C3Mode.EXACT)
        assert result.status is C3Status.ATTRIBUTED
        assert result.bitmap == [0, 0, 1, 0]

    def test_exact_two_outliers(self):
        result = C3.classify_evidence([17, 23, 17, 23, 17, 17], C3Mode.EXACT)
        assert result.status is C3Status.ATTRIBUTED
        assert result.bitmap == [0, 1, 0, 1, 0, 0]

    def test_exact_tie_has_no_culprit(self):
        result = C3.classify_evidence([17, 23, 17, 23], C3Mode.EXACT)
        assert result.status is C3Status.INCONCLUSIVE
        assert result.bitmap == [0, 0, 0, 0]
        assert result.evidence == [17, 23, 17, 23]

    def test_exact_diverged_rank(self):
        result = C3.classify_evidence([17, 17, 23, 17], C3Mode.EXACT)
        assert result.status is C3Status.ATTRIBUTED
        assert result.bitmap == [0, 0, 1, 0]

    def test_exact_even_group_does_not_cancel_duplicate_minority(self):
        result = C3.classify_evidence([17, 17, 17, 17, 17, 17, 23, 23], C3Mode.EXACT)
        assert result.status is C3Status.ATTRIBUTED
        assert result.bitmap == [0, 0, 0, 0, 0, 0, 1, 1]

    def test_statistical_all_similar(self):
        values = [10.0, 10.1, 9.9, 10.0]
        result = C3.classify_evidence(values, C3Mode.STATISTICAL)
        assert result.status is C3Status.AGREE
        assert result.bitmap == [0, 0, 0, 0]

    def test_statistical_one_slow(self):
        values = [10.0, 10.1, 9.9, 200.0]
        result = C3.classify_evidence(values, C3Mode.STATISTICAL)
        assert result.status is C3Status.ATTRIBUTED
        assert result.bitmap == [0, 0, 0, 1]
        assert result.evidence == values

    def test_statistical_identical(self):
        values = [10.0, 10.0, 10.0, 10.0]
        result = C3.classify_evidence(values, C3Mode.STATISTICAL)
        assert result.status is C3Status.AGREE
        assert result.bitmap == [0, 0, 0, 0]

    def test_exact_single_rank(self):
        result = C3.classify_evidence([17], C3Mode.EXACT)
        assert result.status is C3Status.AGREE
        assert result.bitmap == [0]

    def test_statistical_single_rank(self):
        result = C3.classify_evidence([10.0], C3Mode.STATISTICAL)
        assert result.status is C3Status.AGREE
        assert result.bitmap == [0]

    def test_empty_evidence_is_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            C3.classify_evidence([], C3Mode.EXACT)

    def test_nonpositive_statistical_threshold_is_rejected(self):
        with pytest.raises(ValueError, match="positive"):
            C3.classify_evidence([1.0, 2.0], C3Mode.STATISTICAL, 0.0)


def test_cpu_tensor_path_allgathers_signatures_without_xor_allreduce():
    c3 = C3.__new__(C3)
    c3._group = object()
    c3._nccl_group = None
    c3._world_size = 4
    local_signature = _compute_tensor_signature(torch.arange(8))

    def gather_signatures(gathered, local, group=None):
        assert group is c3._group
        for output, value in zip(
            gathered,
            [local_signature, local_signature, local_signature + 1, local_signature],
        ):
            output.fill_(value)

    with (
        patch("lm_resiliency.detection.c3.dist.all_gather", side_effect=gather_signatures),
        patch("lm_resiliency.detection.c3.dist.all_reduce") as all_reduce,
    ):
        result = c3.run_tensor(torch.arange(8))

    all_reduce.assert_not_called()
    assert result.status is C3Status.ATTRIBUTED
    assert result.bitmap == [0, 0, 1, 0]
