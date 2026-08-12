"""Tests for state_dict flatten/unflatten."""

import pytest
import torch

from lm_resiliency.checkpointing.state_dict import flatten, unflatten


def test_flatten_simple_dict():
    state = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(3),
    }
    metadata, tensors = flatten(state)
    assert len(tensors) == 2
    assert len(metadata.tensor_entries) == 2
    assert metadata.tensor_entries[0].key_path == ("weight",)
    assert metadata.tensor_entries[1].key_path == ("bias",)


def test_unflatten_roundtrip():
    state = {
        "weight": torch.randn(3, 4),
        "bias": torch.randn(3),
        "step": 42,
    }
    metadata, tensors = flatten(state)
    restored = unflatten(metadata, tensors)
    assert torch.equal(restored["weight"], state["weight"])
    assert torch.equal(restored["bias"], state["bias"])
    assert restored["step"] == 42


def test_nested_dict():
    state = {
        "model": {
            "layers.0.weight": torch.randn(10, 10),
            "layers.0.bias": torch.randn(10),
        },
        "optimizer": {
            "state": {
                "0": {"momentum": torch.zeros(10, 10)},
            },
        },
        "step": 100,
    }
    metadata, tensors = flatten(state)
    assert len(tensors) == 3
    restored = unflatten(metadata, tensors)
    assert torch.equal(restored["model"]["layers.0.weight"], state["model"]["layers.0.weight"])
    assert torch.equal(
        restored["optimizer"]["state"]["0"]["momentum"],
        state["optimizer"]["state"]["0"]["momentum"],
    )
    assert restored["step"] == 100


def test_list_in_state_dict():
    state = {
        "params": [torch.randn(2, 2), torch.randn(3)],
        "config": {"lr": 0.01},
    }
    metadata, tensors = flatten(state)
    assert len(tensors) == 2
    restored = unflatten(metadata, tensors)
    assert torch.equal(restored["params"][0], state["params"][0])
    assert torch.equal(restored["params"][1], state["params"][1])
    assert restored["config"]["lr"] == 0.01


def test_empty_state_dict():
    state = {"info": "no tensors here", "count": 5}
    metadata, tensors = flatten(state)
    assert len(tensors) == 0
    restored = unflatten(metadata, tensors)
    assert restored == state


def test_unflatten_length_mismatch():
    state = {"w": torch.randn(2, 2)}
    metadata, tensors = flatten(state)
    with pytest.raises(ValueError, match="Expected 1 tensors, got 2"):
        unflatten(metadata, [torch.randn(2, 2), torch.randn(3)])


def test_tensor_metadata_preserved():
    state = {"w": torch.randn(5, 3, dtype=torch.float16)}
    metadata, tensors = flatten(state)
    entry = metadata.tensor_entries[0]
    assert entry.shape == torch.Size([5, 3])
    assert entry.dtype == torch.float16


def test_large_nested_structure():
    """Verify flatten/unflatten with a realistic model-like state dict."""
    state = {
        "model": {
            f"layer_{i}": {
                "weight": torch.randn(64, 64),
                "bias": torch.randn(64),
            }
            for i in range(10)
        },
        "optimizer": {
            "param_groups": [{"lr": 0.001, "weight_decay": 0.01}],
            "state": {
                str(i): {
                    "exp_avg": torch.zeros(64, 64),
                    "exp_avg_sq": torch.zeros(64, 64),
                }
                for i in range(10)
            },
        },
    }
    metadata, tensors = flatten(state)
    # 10 layers * 2 (weight+bias) + 10 optimizer states * 2 (exp_avg + exp_avg_sq)
    assert len(tensors) == 40
    restored = unflatten(metadata, tensors)
    assert torch.equal(restored["model"]["layer_5"]["weight"], state["model"]["layer_5"]["weight"])
    assert restored["optimizer"]["param_groups"][0]["lr"] == 0.001
