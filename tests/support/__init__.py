"""Test support for fault injection and detection validation."""

from tests.support.fault_injector import (
    FaultConfig,
    FaultType,
    Location,
    Magnitude,
    Scope,
    inject_fault,
)

__all__ = [
    "FaultConfig",
    "FaultType",
    "Location",
    "Magnitude",
    "Scope",
    "inject_fault",
]
