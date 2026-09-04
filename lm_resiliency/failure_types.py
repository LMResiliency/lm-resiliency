"""Pre-training system failure classifications shared across LM Resiliency."""

from __future__ import annotations

from enum import Enum


class SystemFailureType(str, Enum):
    """System root causes that a pre-training resiliency campaign can evaluate.

    These values classify why a failure happened. ``FailureType`` continues to
    describe the observable effect that an executor injects and SCOUT or another
    system attempts to detect.
    """

    HOST_MEMORY_EXHAUSTION = "host_memory_exhaustion"
    HOST_RESOURCE_EXHAUSTION = "host_resource_exhaustion"
    CUDA_OUT_OF_MEMORY = "cuda_out_of_memory"
    CUDA_RUNTIME_FAILURE = "cuda_runtime_failure"
    DURABLE_STORAGE_EXHAUSTION = "durable_storage_exhaustion"
    DURABLE_STORAGE_FAILURE = "durable_storage_failure"
    PCIE_LINK_FAILURE = "pcie_link_failure"
    PCIE_LINK_DEGRADATION = "pcie_link_degradation"
    FABRIC_LINK_FAILURE = "fabric_link_failure"
    FABRIC_CONGESTION = "fabric_congestion"
    DATA_SAMPLE_CORRUPTION = "data_sample_corruption"
    DATA_SHARD_UNAVAILABLE = "data_shard_unavailable"
    INPUT_POSITION_DIVERGENCE = "input_position_divergence"
    SOFTWARE_ENVIRONMENT_DRIFT = "software_environment_drift"
    HOST_PERFORMANCE_DEGRADATION = "host_performance_degradation"
    GPU_THROTTLING = "gpu_throttling"
    TRAINING_RUNTIME_FAILURE = "training_runtime_failure"
    CONTROL_PLANE_FAILURE = "control_plane_failure"
    TRANSIENT_COMPUTE_CORRUPTION = "transient_compute_corruption"
    COMMON_MODE_CORRUPTION = "common_mode_corruption"
    SINGLE_OWNER_STATE_CORRUPTION = "single_owner_state_corruption"


__all__ = ["SystemFailureType"]
