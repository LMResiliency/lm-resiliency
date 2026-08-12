"""Fault injection for testing SCOUT detection boundaries.

Provides a parameterized fault injector that corrupts model layers in
controlled ways — varying type, magnitude, scope, and location.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn as nn


class FaultType(Enum):
    SINGLE_BITFLIP = "single_bitflip"
    MULTI_BITFLIP = "multi_bitflip"
    STUCK_AT_ZERO = "stuck_at_zero"
    STUCK_AT_ONE = "stuck_at_one"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    GAUSSIAN_NOISE = "gaussian_noise"
    SIGN_FLIP = "sign_flip"
    SET_NAN = "set_nan"
    SET_INF = "set_inf"
    TRANSIENT_FORWARD = "transient_forward"


class Magnitude(Enum):
    CATASTROPHIC = "catastrophic"
    LARGE = "large"
    MEDIUM = "medium"
    SUBTLE = "subtle"
    NEAR_INVISIBLE = "near_invisible"


class Scope(Enum):
    SINGLE = "single"
    ROW = "row"
    PERCENT_1 = "1%"
    PERCENT_10 = "10%"
    FULL = "100%"


class Location(Enum):
    WEIGHT = "weight"
    BIAS = "bias"


MAGNITUDE_VALUES = {
    Magnitude.CATASTROPHIC: 1e38,
    Magnitude.LARGE: 1e4,
    Magnitude.MEDIUM: 1e1,
    Magnitude.SUBTLE: 1e-3,
    Magnitude.NEAR_INVISIBLE: 1e-7,
}


@dataclass
class FaultConfig:
    fault_type: FaultType
    magnitude: Magnitude
    scope: Scope
    location: Location

    @property
    def key(self) -> str:
        return f"{self.fault_type.value}|{self.magnitude.value}|{self.scope.value}|{self.location.value}"


def inject_fault(layer: nn.Module, config: FaultConfig) -> object | None:
    """Inject a fault into a layer's parameters.

    Args:
        layer: Module with .weight and optionally .bias attributes.
        config: Fault configuration specifying type, magnitude, scope, location.

    Returns:
        For transient faults, returns the hook handle (auto-removed after one forward).
        For persistent faults, returns None.
        Returns None without injecting if the target parameter doesn't exist
        (e.g., bias on a layer with bias=False).
    """
    if config.location == Location.WEIGHT:
        target = layer.weight
    elif config.location == Location.BIAS:
        if layer.bias is None:
            return None
        target = layer.bias
    else:
        return None

    if config.fault_type == FaultType.TRANSIENT_FORWARD:
        return _inject_transient(layer, target, config)

    _inject_persistent(target, config)
    return None


def _inject_transient(layer: nn.Module, target: torch.Tensor, config: FaultConfig) -> object:
    """Inject a fault that only manifests during one forward pass."""
    mag = MAGNITUDE_VALUES[config.magnitude]

    def _hook(module, input, output):
        indices = _get_target_indices(output, config.scope)
        flat = output.flatten()
        flat[indices] += mag
        module._transient_hook.remove()
        return output

    handle = layer.register_forward_hook(_hook)
    layer._transient_hook = handle
    return handle


def _inject_persistent(target: torch.Tensor, config: FaultConfig):
    """Inject a persistent fault into a parameter tensor."""
    indices = _get_target_indices(target, config.scope)

    with torch.no_grad():
        flat = target.flatten()

        if config.fault_type == FaultType.SINGLE_BITFLIP:
            _flip_bits(target, num_bits=1, indices=indices, magnitude=config.magnitude)

        elif config.fault_type == FaultType.MULTI_BITFLIP:
            _flip_bits(target, num_bits=4, indices=indices, magnitude=config.magnitude)

        elif config.fault_type == FaultType.STUCK_AT_ZERO:
            flat[indices] = 0.0

        elif config.fault_type == FaultType.STUCK_AT_ONE:
            flat[indices] = 1.0

        elif config.fault_type == FaultType.SCALE_UP:
            if config.magnitude == Magnitude.CATASTROPHIC:
                flat[indices] *= 1e6
            elif config.magnitude == Magnitude.LARGE:
                flat[indices] *= 100.0
            elif config.magnitude == Magnitude.MEDIUM:
                flat[indices] *= 10.0
            elif config.magnitude == Magnitude.SUBTLE:
                flat[indices] *= 2.0
            elif config.magnitude == Magnitude.NEAR_INVISIBLE:
                flat[indices] *= 1.0001

        elif config.fault_type == FaultType.SCALE_DOWN:
            if config.magnitude == Magnitude.CATASTROPHIC:
                flat[indices] *= 1e-6
            elif config.magnitude == Magnitude.LARGE:
                flat[indices] *= 0.01
            elif config.magnitude == Magnitude.MEDIUM:
                flat[indices] *= 0.1
            elif config.magnitude == Magnitude.SUBTLE:
                flat[indices] *= 0.5
            elif config.magnitude == Magnitude.NEAR_INVISIBLE:
                flat[indices] *= 0.9999

        elif config.fault_type == FaultType.GAUSSIAN_NOISE:
            noise_std = MAGNITUDE_VALUES[config.magnitude]
            noise = torch.randn(indices.numel(), device=target.device) * noise_std
            flat[indices] += noise

        elif config.fault_type == FaultType.SIGN_FLIP:
            flat[indices] *= -1.0

        elif config.fault_type == FaultType.SET_NAN:
            flat[indices] = float("nan")

        elif config.fault_type == FaultType.SET_INF:
            flat[indices] = float("inf")


def _flip_bits(
    tensor: torch.Tensor,
    num_bits: int,
    indices: torch.Tensor,
    magnitude: Magnitude = Magnitude.MEDIUM,
):
    """Flip bits in the IEEE 754 representation of selected elements.

    Bit position depends on magnitude: catastrophic flips exponent/sign bits,
    near-invisible flips lowest mantissa bits.
    """
    bit_starts = {
        Magnitude.CATASTROPHIC: 28,
        Magnitude.LARGE: 23,
        Magnitude.MEDIUM: 16,
        Magnitude.SUBTLE: 8,
        Magnitude.NEAR_INVISIBLE: 0,
    }
    start = bit_starts[magnitude]

    flat = tensor.flatten()
    for i, idx in enumerate(indices):
        val = flat[idx].item()
        bits = struct.unpack("I", struct.pack("f", val))[0]
        for b in range(num_bits):
            bit_pos = start + b
            bits ^= 1 << bit_pos
        new_val = struct.unpack("f", struct.pack("I", bits))[0]
        flat[idx] = new_val


def _get_target_indices(tensor: torch.Tensor, scope: Scope) -> torch.Tensor:
    """Select which elements to corrupt based on scope."""
    numel = tensor.numel()
    if scope == Scope.SINGLE:
        return torch.tensor([numel // 2])
    elif scope == Scope.ROW:
        row_size = tensor.shape[-1] if tensor.ndim >= 2 else min(256, numel)
        return torch.arange(row_size)
    elif scope == Scope.PERCENT_1:
        n = max(1, numel // 100)
        step = numel // n
        return torch.arange(0, numel, step)[:n]
    elif scope == Scope.PERCENT_10:
        n = max(1, numel // 10)
        step = numel // n
        return torch.arange(0, numel, step)[:n]
    elif scope == Scope.FULL:
        return torch.arange(numel)
    return torch.tensor([0])
