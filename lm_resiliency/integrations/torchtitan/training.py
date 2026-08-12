"""Compatibility imports for the native PyTorch FSDP2/HSDP runtime."""

from lm_resiliency.integrations.pytorch.fsdp import (
    PyTorchFSDPResiliency,
    _is_hsdp_model,
    enable_fsdp2_resiliency,
    has_dtensor_params,
)

TorchTitanResiliency = PyTorchFSDPResiliency

__all__ = [
    "TorchTitanResiliency",
    "_is_hsdp_model",
    "enable_fsdp2_resiliency",
    "has_dtensor_params",
]
