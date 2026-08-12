"""Flatten and unflatten nested state dicts for efficient bulk tensor operations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(slots=True)
class TensorEntry:
    """Metadata for a single tensor extracted from the state dict."""

    key_path: tuple[str | int, ...]
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device


@dataclass(slots=True)
class FlatStateDictMetadata:
    """Stores structure metadata needed to reconstruct the original state dict."""

    tensor_entries: list[TensorEntry] = field(default_factory=list)
    non_tensor_data: dict[str, Any] = field(default_factory=dict)


def flatten(state_dict: dict[str, Any]) -> tuple[FlatStateDictMetadata, list[torch.Tensor]]:
    """Extract all tensors from a nested state dict into a flat list.

    Returns metadata for reassembly and the ordered list of tensors.
    The original state_dict is not modified.
    """
    metadata = FlatStateDictMetadata()
    tensors: list[torch.Tensor] = []

    def _recurse(obj: Any, path: tuple[str | int, ...]) -> Any:
        if isinstance(obj, torch.Tensor):
            metadata.tensor_entries.append(
                TensorEntry(
                    key_path=path,
                    shape=obj.shape,
                    dtype=obj.dtype,
                    device=obj.device,
                )
            )
            tensors.append(obj)
            return None  # placeholder
        elif isinstance(obj, dict):
            return {k: _recurse(v, path + (k,)) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return type(obj)(_recurse(v, path + (i,)) for i, v in enumerate(obj))
        else:
            return obj

    skeleton = _recurse(state_dict, ())
    metadata.non_tensor_data = skeleton
    return metadata, tensors


def unflatten(metadata: FlatStateDictMetadata, tensors: list[torch.Tensor]) -> dict[str, Any]:
    """Reconstruct the original state dict from metadata and a flat tensor list.

    Tensors are inserted back into the nested structure at their recorded positions.
    """
    if len(tensors) != len(metadata.tensor_entries):
        raise ValueError(f"Expected {len(metadata.tensor_entries)} tensors, got {len(tensors)}")

    import copy

    result = copy.deepcopy(metadata.non_tensor_data)

    for entry, tensor in zip(metadata.tensor_entries, tensors):
        _set_nested(result, entry.key_path, tensor)

    return result


def _set_nested(obj: Any, path: tuple[str | int, ...], value: Any) -> None:
    """Set a value at a nested key path in a dict/list structure."""
    for key in path[:-1]:
        if isinstance(obj, dict):
            obj = obj[key]
        else:
            obj = obj[key]
    final_key = path[-1]
    if isinstance(obj, dict):
        obj[final_key] = value
    else:
        obj[final_key] = value
