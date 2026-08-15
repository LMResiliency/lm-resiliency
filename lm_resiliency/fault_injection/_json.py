"""Strict immutable JSON values used by public fault-injection contracts."""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping
from typing import Any


class FrozenMapping(Mapping[str, Any]):
    """An immutable mapping whose nested values are also immutable."""

    __slots__ = ("_data",)

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._data = dict(values)

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def __len__(self) -> int:
        return len(self._data)

    def __repr__(self) -> str:
        return f"FrozenMapping({self._data!r})"

    def __deepcopy__(self, _memo: dict[int, Any]) -> "FrozenMapping":
        return self


def freeze_json(value: Any, label: str) -> Any:
    """Validate one strict JSON value and return an immutable snapshot."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain non-finite numbers")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise TypeError(f"{label} object keys must be strings")
            frozen[key] = freeze_json(item, f"{label}.{key}")
        return FrozenMapping(frozen)
    if isinstance(value, (list, tuple)):
        return tuple(freeze_json(item, f"{label}[{index}]") for index, item in enumerate(value))
    raise TypeError(f"{label} must contain only JSON values, not {type(value).__name__}")


def freeze_json_mapping(value: Mapping[str, Any], label: str) -> FrozenMapping:
    """Validate and freeze a JSON object."""
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    frozen = freeze_json(value, label)
    if not isinstance(frozen, FrozenMapping):
        raise AssertionError("mapping freeze did not produce FrozenMapping")
    return frozen


def thaw_json(value: Any) -> Any:
    """Return mutable JSON containers suitable for serialization."""
    if isinstance(value, Mapping):
        return {str(key): thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


__all__ = ["FrozenMapping", "freeze_json", "freeze_json_mapping", "thaw_json"]
