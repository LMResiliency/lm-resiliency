"""Schema-constrained metadata encoding for GEMINI node-local checkpoints."""

from __future__ import annotations

import base64
import json
import math
from typing import Any

import numpy as np
import torch

from lm_resiliency.checkpointing.state_dict import FlatStateDictMetadata, TensorEntry

FORMAT_NAME = "lm-resiliency.gemini.node-local"
FORMAT_VERSION = 3

_METADATA_SCHEMA_VERSION = 1
_MAX_METADATA_BYTES = 64 * 1024 * 1024
_MAX_METADATA_DEPTH = 100
_MAX_CONTAINER_ITEMS = 1_000_000


class CheckpointFormatError(ValueError):
    """Raised when a node-local checkpoint does not match the safe schema."""


def encode_metadata(metadata: FlatStateDictMetadata) -> str:
    """Encode reconstruction metadata as schema-constrained JSON."""
    if not isinstance(metadata, FlatStateDictMetadata):
        raise TypeError("checkpoint metadata must be FlatStateDictMetadata")

    entries: list[dict[str, object]] = []
    for index, entry in enumerate(metadata.tensor_entries):
        if not isinstance(entry, TensorEntry):
            raise TypeError(f"tensor_entries[{index}] must be TensorEntry")
        key_path = list(entry.key_path)
        if any(type(key) not in (str, int) for key in key_path):
            raise TypeError(f"tensor_entries[{index}].key_path must contain only str or int")
        shape = list(entry.shape)
        if any(type(dimension) is not int or dimension < 0 for dimension in shape):
            raise TypeError(f"tensor_entries[{index}].shape must contain non-negative integers")
        entries.append(
            {
                "key_path": key_path,
                "shape": shape,
                "dtype": _encode_dtype(entry.dtype),
                "device": str(entry.device),
            }
        )

    document = {
        "schema_version": _METADATA_SCHEMA_VERSION,
        "tensor_entries": entries,
        "non_tensor_data": _encode_value(metadata.non_tensor_data, depth=0),
    }
    encoded = json.dumps(document, separators=(",", ":"), sort_keys=True)
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise TypeError(f"checkpoint metadata exceeds {_MAX_METADATA_BYTES} bytes")
    return encoded


def decode_metadata(encoded: object) -> FlatStateDictMetadata:
    """Decode and validate reconstruction metadata from the safe JSON schema."""
    if not isinstance(encoded, str):
        raise CheckpointFormatError("checkpoint metadata_json must be a string")
    if len(encoded.encode("utf-8")) > _MAX_METADATA_BYTES:
        raise CheckpointFormatError(f"checkpoint metadata exceeds {_MAX_METADATA_BYTES} bytes")
    try:
        document = json.loads(encoded)
    except (json.JSONDecodeError, UnicodeError) as error:
        raise CheckpointFormatError("checkpoint metadata_json is not valid JSON") from error

    _require_mapping(document, "checkpoint metadata")
    _require_exact_keys(
        document,
        {"schema_version", "tensor_entries", "non_tensor_data"},
        "checkpoint metadata",
    )
    if document["schema_version"] != _METADATA_SCHEMA_VERSION:
        raise CheckpointFormatError(
            f"unsupported checkpoint metadata schema {document['schema_version']!r}"
        )

    raw_entries = document["tensor_entries"]
    if not isinstance(raw_entries, list):
        raise CheckpointFormatError("checkpoint tensor_entries must be a list")
    _check_container_size(raw_entries, "checkpoint tensor_entries")

    entries: list[TensorEntry] = []
    seen_paths: set[tuple[str | int, ...]] = set()
    for index, raw_entry in enumerate(raw_entries):
        label = f"checkpoint tensor_entries[{index}]"
        _require_mapping(raw_entry, label)
        _require_exact_keys(raw_entry, {"key_path", "shape", "dtype", "device"}, label)

        raw_path = raw_entry["key_path"]
        if not isinstance(raw_path, list) or any(type(key) not in (str, int) for key in raw_path):
            raise CheckpointFormatError(f"{label}.key_path must contain only str or int")
        key_path = tuple(raw_path)
        if key_path in seen_paths:
            raise CheckpointFormatError(f"{label}.key_path is duplicated")
        seen_paths.add(key_path)

        raw_shape = raw_entry["shape"]
        if not isinstance(raw_shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in raw_shape
        ):
            raise CheckpointFormatError(f"{label}.shape must contain non-negative integers")
        if not isinstance(raw_entry["device"], str):
            raise CheckpointFormatError(f"{label}.device must be a string")
        try:
            device = torch.device(raw_entry["device"])
        except (RuntimeError, ValueError) as error:
            raise CheckpointFormatError(f"{label}.device is invalid") from error

        entries.append(
            TensorEntry(
                key_path=key_path,
                shape=torch.Size(raw_shape),
                dtype=_decode_dtype(raw_entry["dtype"], f"{label}.dtype"),
                device=device,
            )
        )

    non_tensor_data = _decode_value(document["non_tensor_data"], depth=0)
    if type(non_tensor_data) is not dict:
        raise CheckpointFormatError("checkpoint non_tensor_data must be a dictionary")
    _validate_tensor_paths(non_tensor_data, entries)
    return FlatStateDictMetadata(
        tensor_entries=entries,
        non_tensor_data=non_tensor_data,
    )


def validate_payload(
    payload: object,
) -> tuple[FlatStateDictMetadata, list[torch.Tensor], list[int] | None, dict[str, object]]:
    """Validate the weights-only payload before recovery can apply its contents."""
    _require_mapping(payload, "checkpoint payload")
    _require_exact_keys(
        payload,
        {"format", "version", "identity", "metadata_json", "tensors", "checksums"},
        "checkpoint payload",
    )
    if payload["format"] != FORMAT_NAME:
        raise CheckpointFormatError("unrecognized or legacy checkpoint format")
    if payload["version"] != FORMAT_VERSION:
        raise CheckpointFormatError(f"unsupported checkpoint format version {payload['version']!r}")

    identity = payload["identity"]
    _require_mapping(identity, "checkpoint identity")
    _require_exact_keys(
        identity,
        {"run_id", "topology_id", "owner_rank", "step"},
        "checkpoint identity",
    )
    if not isinstance(identity["run_id"], str) or not identity["run_id"].strip():
        raise CheckpointFormatError("checkpoint identity run_id must be a non-empty string")
    if not isinstance(identity["topology_id"], str) or not identity["topology_id"].strip():
        raise CheckpointFormatError("checkpoint identity topology_id must be a non-empty string")
    for field in ("owner_rank", "step"):
        if type(identity[field]) is not int or identity[field] < 0:
            raise CheckpointFormatError(
                f"checkpoint identity {field} must be a non-negative integer"
            )

    metadata = decode_metadata(payload["metadata_json"])
    tensors = payload["tensors"]
    if not isinstance(tensors, list) or any(
        not isinstance(tensor, torch.Tensor) for tensor in tensors
    ):
        raise CheckpointFormatError("checkpoint tensors must be a list of tensors")
    if len(tensors) != len(metadata.tensor_entries):
        raise CheckpointFormatError(
            f"checkpoint has {len(tensors)} tensors but metadata describes "
            f"{len(metadata.tensor_entries)}"
        )
    for index, (tensor, entry) in enumerate(zip(tensors, metadata.tensor_entries)):
        if tensor.shape != entry.shape:
            raise CheckpointFormatError(
                f"checkpoint tensor {index} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(entry.shape)}"
            )
        if tensor.dtype != entry.dtype:
            raise CheckpointFormatError(
                f"checkpoint tensor {index} has dtype {tensor.dtype}, expected {entry.dtype}"
            )

    checksums = payload["checksums"]
    if checksums is not None:
        if not isinstance(checksums, list) or any(
            type(checksum) is not int or not 0 <= checksum <= 0xFFFFFFFF for checksum in checksums
        ):
            raise CheckpointFormatError(
                "checkpoint checksums must be null or a list of unsigned CRC-32 values"
            )
    return metadata, tensors, checksums, dict(identity)


def _encode_value(value: Any, *, depth: int) -> dict[str, Any]:
    _check_depth(depth)
    if value is None:
        return {"kind": "none"}
    if type(value) is bool:
        return {"kind": "bool", "value": value}
    if type(value) is int:
        return {"kind": "int", "value": value}
    if type(value) is float:
        return {"kind": "float", "value": value}
    if type(value) is complex:
        return {"kind": "complex", "value": [value.real, value.imag]}
    if type(value) is str:
        return {"kind": "str", "value": value}
    if type(value) is bytes:
        return {"kind": "bytes", "value": base64.b64encode(value).decode("ascii")}
    if type(value) is bytearray:
        return {"kind": "bytearray", "value": base64.b64encode(value).decode("ascii")}
    if isinstance(value, torch.Size):
        if any(dimension < 0 for dimension in value):
            raise TypeError("torch.Size checkpoint metadata cannot contain negative dimensions")
        return {"kind": "torch_size", "value": list(value)}
    if isinstance(value, torch.dtype):
        return {"kind": "torch_dtype", "value": _encode_dtype(value)}
    if isinstance(value, torch.device):
        return {"kind": "torch_device", "value": str(value)}
    if isinstance(value, torch.Tensor):
        if value.device.type != "cpu":
            raise TypeError("only CPU tensors are supported in checkpoint metadata")
        if value.requires_grad:
            raise TypeError("tensors requiring gradients are not supported in checkpoint metadata")
        if value.layout is not torch.strided or value.is_quantized:
            raise TypeError(
                "only dense, non-quantized tensors are supported in checkpoint metadata"
            )
        contiguous = value.detach().contiguous()
        return {
            "kind": "torch_tensor",
            "dtype": _encode_dtype(contiguous.dtype),
            "shape": list(contiguous.shape),
            "data": base64.b64encode(
                contiguous.flatten().view(torch.uint8).numpy().tobytes()
            ).decode("ascii"),
        }
    if type(value) is np.ndarray:
        if value.dtype.hasobject:
            raise TypeError("object-dtype NumPy arrays are not supported in checkpoint metadata")
        _require_plain_numpy_dtype(value.dtype)
        contiguous = np.ascontiguousarray(value)
        return {
            "kind": "numpy_array",
            "dtype": contiguous.dtype.str,
            "shape": list(contiguous.shape),
            "data": base64.b64encode(contiguous.tobytes()).decode("ascii"),
        }
    if isinstance(value, np.generic):
        if value.dtype.hasobject:
            raise TypeError("object-dtype NumPy scalars are not supported in checkpoint metadata")
        _require_plain_numpy_dtype(value.dtype)
        return {
            "kind": "numpy_scalar",
            "dtype": value.dtype.str,
            "data": base64.b64encode(value.tobytes()).decode("ascii"),
        }
    if type(value) is dict:
        _check_container_size(value, "checkpoint metadata mapping")
        return {
            "kind": "dict",
            "items": [
                [_encode_value(key, depth=depth + 1), _encode_value(item, depth=depth + 1)]
                for key, item in value.items()
            ],
        }
    if type(value) in (list, tuple, set, frozenset):
        _check_container_size(value, "checkpoint metadata container")
        kind = {
            list: "list",
            tuple: "tuple",
            set: "set",
            frozenset: "frozenset",
        }[type(value)]
        return {
            "kind": kind,
            "items": [_encode_value(item, depth=depth + 1) for item in value],
        }
    raise TypeError(f"unsupported checkpoint metadata type: {type(value).__name__}")


def _decode_value(value: object, *, depth: int) -> Any:
    _check_depth(depth, error_type=CheckpointFormatError)
    _require_mapping(value, "checkpoint metadata value")
    kind = value.get("kind")
    if not isinstance(kind, str):
        raise CheckpointFormatError("checkpoint metadata value kind must be a string")

    if kind == "none":
        _require_exact_keys(value, {"kind"}, "checkpoint metadata none")
        return None
    if kind in {"bool", "int", "float", "str"}:
        _require_exact_keys(value, {"kind", "value"}, f"checkpoint metadata {kind}")
        expected = {"bool": bool, "int": int, "float": float, "str": str}[kind]
        raw = value["value"]
        if type(raw) is not expected:
            raise CheckpointFormatError(f"checkpoint metadata {kind} has an invalid value")
        return raw
    if kind == "complex":
        _require_exact_keys(value, {"kind", "value"}, "checkpoint metadata complex")
        raw = value["value"]
        if (
            not isinstance(raw, list)
            or len(raw) != 2
            or any(type(component) not in (int, float) for component in raw)
        ):
            raise CheckpointFormatError("checkpoint metadata complex has an invalid value")
        return complex(raw[0], raw[1])
    if kind in {"bytes", "bytearray"}:
        _require_exact_keys(value, {"kind", "value"}, f"checkpoint metadata {kind}")
        raw = _decode_base64(value["value"], f"checkpoint metadata {kind}")
        return raw if kind == "bytes" else bytearray(raw)
    if kind == "torch_size":
        _require_exact_keys(value, {"kind", "value"}, "checkpoint metadata torch_size")
        shape = value["value"]
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise CheckpointFormatError("checkpoint metadata torch_size has an invalid value")
        return torch.Size(shape)
    if kind == "torch_dtype":
        _require_exact_keys(value, {"kind", "value"}, "checkpoint metadata torch_dtype")
        return _decode_dtype(value["value"], "checkpoint metadata torch_dtype")
    if kind == "torch_device":
        _require_exact_keys(value, {"kind", "value"}, "checkpoint metadata torch_device")
        if not isinstance(value["value"], str):
            raise CheckpointFormatError("checkpoint metadata torch_device has an invalid value")
        try:
            return torch.device(value["value"])
        except (RuntimeError, ValueError) as error:
            raise CheckpointFormatError(
                "checkpoint metadata torch_device has an invalid value"
            ) from error
    if kind == "torch_tensor":
        _require_exact_keys(
            value,
            {"kind", "dtype", "shape", "data"},
            "checkpoint metadata torch_tensor",
        )
        dtype = _decode_dtype(value["dtype"], "checkpoint metadata torch_tensor dtype")
        shape = value["shape"]
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise CheckpointFormatError("checkpoint metadata torch_tensor shape is invalid")
        data = _decode_base64(value["data"], "checkpoint metadata torch_tensor")
        try:
            element_size = torch.empty((), dtype=dtype).element_size()
        except RuntimeError as error:
            raise CheckpointFormatError(
                "checkpoint metadata torch_tensor dtype is unsupported"
            ) from error
        element_count = math.prod(shape)
        if len(data) != element_count * element_size:
            raise CheckpointFormatError("checkpoint metadata torch_tensor byte length is invalid")
        if not data:
            return torch.empty(shape, dtype=dtype)
        return torch.frombuffer(bytearray(data), dtype=dtype).clone().reshape(shape)
    if kind in {"numpy_array", "numpy_scalar"}:
        required = {"kind", "dtype", "data"}
        if kind == "numpy_array":
            required.add("shape")
        _require_exact_keys(value, required, f"checkpoint metadata {kind}")
        dtype = _decode_numpy_dtype(value["dtype"])
        data = _decode_base64(value["data"], f"checkpoint metadata {kind}")
        if kind == "numpy_scalar":
            if len(data) != dtype.itemsize:
                raise CheckpointFormatError("checkpoint NumPy scalar byte length is invalid")
            return np.frombuffer(data, dtype=dtype, count=1)[0]
        shape = value["shape"]
        if not isinstance(shape, list) or any(
            type(dimension) is not int or dimension < 0 for dimension in shape
        ):
            raise CheckpointFormatError("checkpoint NumPy array shape is invalid")
        expected_bytes = math.prod(shape) * dtype.itemsize
        if len(data) != expected_bytes:
            raise CheckpointFormatError("checkpoint NumPy array byte length is invalid")
        return np.frombuffer(data, dtype=dtype).copy().reshape(shape)
    if kind == "dict":
        _require_exact_keys(value, {"kind", "items"}, "checkpoint metadata dict")
        items = value["items"]
        if not isinstance(items, list):
            raise CheckpointFormatError("checkpoint metadata dict items must be a list")
        _check_container_size(items, "checkpoint metadata dict")
        result: dict[Any, Any] = {}
        for pair in items:
            if not isinstance(pair, list) or len(pair) != 2:
                raise CheckpointFormatError("checkpoint metadata dict item must be a pair")
            key = _decode_value(pair[0], depth=depth + 1)
            try:
                duplicate = key in result
            except TypeError as error:
                raise CheckpointFormatError(
                    "checkpoint metadata dict key is not hashable"
                ) from error
            if duplicate:
                raise CheckpointFormatError("checkpoint metadata dict contains a duplicate key")
            result[key] = _decode_value(pair[1], depth=depth + 1)
        return result
    if kind in {"list", "tuple", "set", "frozenset"}:
        _require_exact_keys(value, {"kind", "items"}, f"checkpoint metadata {kind}")
        items = value["items"]
        if not isinstance(items, list):
            raise CheckpointFormatError(f"checkpoint metadata {kind} items must be a list")
        _check_container_size(items, f"checkpoint metadata {kind}")
        decoded = [_decode_value(item, depth=depth + 1) for item in items]
        if kind == "list":
            return decoded
        if kind == "tuple":
            return tuple(decoded)
        try:
            return set(decoded) if kind == "set" else frozenset(decoded)
        except TypeError as error:
            raise CheckpointFormatError(
                f"checkpoint metadata {kind} item is not hashable"
            ) from error
    raise CheckpointFormatError(f"unsupported checkpoint metadata kind {kind!r}")


def _encode_dtype(dtype: object) -> str:
    if not isinstance(dtype, torch.dtype):
        raise TypeError("checkpoint tensor dtype must be torch.dtype")
    name = str(dtype)
    if not name.startswith("torch."):
        raise TypeError(f"unsupported checkpoint tensor dtype {dtype!r}")
    return name.removeprefix("torch.")


def _decode_dtype(value: object, label: str) -> torch.dtype:
    if not isinstance(value, str) or not value or "." in value:
        raise CheckpointFormatError(f"{label} is invalid")
    dtype = getattr(torch, value, None)
    if not isinstance(dtype, torch.dtype):
        raise CheckpointFormatError(f"{label} is unsupported")
    return dtype


def _decode_numpy_dtype(value: object) -> np.dtype:
    if not isinstance(value, str) or len(value) > 64:
        raise CheckpointFormatError("checkpoint NumPy dtype is invalid")
    try:
        dtype = np.dtype(value)
    except TypeError as error:
        raise CheckpointFormatError("checkpoint NumPy dtype is invalid") from error
    if dtype.hasobject:
        raise CheckpointFormatError("object-dtype NumPy values are not allowed")
    return dtype


def _require_plain_numpy_dtype(dtype: np.dtype) -> None:
    if dtype.fields is not None or dtype.subdtype is not None:
        raise TypeError(
            "structured and subarray NumPy dtypes are not supported in checkpoint metadata"
        )


def _validate_tensor_paths(
    skeleton: dict[str, Any],
    entries: list[TensorEntry],
) -> None:
    """Ensure every tensor path identifies an assignable placeholder."""
    for index, entry in enumerate(entries):
        label = f"checkpoint tensor_entries[{index}].key_path"
        if not entry.key_path:
            raise CheckpointFormatError(f"{label} cannot be empty")

        current: Any = skeleton
        for offset, key in enumerate(entry.key_path):
            final = offset == len(entry.key_path) - 1
            if type(current) is dict:
                if key not in current:
                    raise CheckpointFormatError(f"{label} does not exist in the skeleton")
                child = current[key]
            elif type(current) in (list, tuple):
                if type(key) is not int or not 0 <= key < len(current):
                    raise CheckpointFormatError(f"{label} is invalid for its sequence")
                if final and type(current) is tuple:
                    raise CheckpointFormatError(f"{label} targets an immutable tuple slot")
                child = current[key]
            else:
                raise CheckpointFormatError(f"{label} traverses a non-container value")

            if final:
                if child is not None:
                    raise CheckpointFormatError(f"{label} does not identify a tensor placeholder")
                break
            current = child


def _decode_base64(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise CheckpointFormatError(f"{label} data must be a string")
    try:
        return base64.b64decode(value, validate=True)
    except (ValueError, TypeError) as error:
        raise CheckpointFormatError(f"{label} data is not valid base64") from error


def _require_mapping(value: object, label: str) -> None:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise CheckpointFormatError(f"{label} must be a string-keyed mapping")


def _require_exact_keys(value: dict[str, object], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise CheckpointFormatError(f"{label} has unexpected or missing fields")


def _check_depth(depth: int, error_type: type[Exception] = TypeError) -> None:
    if depth > _MAX_METADATA_DEPTH:
        raise error_type(f"checkpoint metadata exceeds nesting depth {_MAX_METADATA_DEPTH}")


def _check_container_size(value: object, label: str) -> None:
    if len(value) > _MAX_CONTAINER_ITEMS:  # type: ignore[arg-type]
        raise CheckpointFormatError(f"{label} exceeds {_MAX_CONTAINER_ITEMS} items")
