"""Hardware health monitoring — detection of easy-to-detect failures.

Extends SCOUT's scope. SCOUT infers *hard-to-detect* faults (SDC, straggler,
hang) from the training workload via replay + consensus. This module reads the
*easy-to-detect* faults the platform reports directly — uncorrectable ECC, fatal
XIDs, row-remap failure, NVLink errors, thermal shutdown, or a GPU falling off the
bus — and emits them through a caller-owned callback. No cross-rank consensus is
needed: a device's own counters are ground truth for that device.

See docs/scout.md#hardware-telemetry for the operational contract.
"""

from lm_resiliency.detection.health.config import HealthConfig
from lm_resiliency.detection.health.monitor import HardwareHealthMonitor, HealthEvent
from lm_resiliency.detection.health.sources import (
    FakeSource,
    HealthReading,
    HealthSeverity,
    HealthSource,
    NvmlSource,
)

__all__ = [
    "HealthConfig",
    "HardwareHealthMonitor",
    "HealthEvent",
    "HealthReading",
    "HealthSeverity",
    "HealthSource",
    "NvmlSource",
    "FakeSource",
]
