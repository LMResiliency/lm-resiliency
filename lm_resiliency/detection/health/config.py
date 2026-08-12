"""Configuration for hardware health monitoring (easy-to-detect failures).

Complements SCOUT: SCOUT infers *hard-to-detect* faults (SDC, straggler, hang)
from the training workload; this module reads *easy-to-detect* faults the driver
and fabric already report directly (ECC, XID, NVLink, thermal, PCIe) and routes
them into the same recovery pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class HealthConfig:
    """Thresholds and cadence for the hardware health monitor.

    Args:
        enable: Master switch.
        poll_interval_s: How often to read telemetry (out-of-band, low frequency).

        fatal_on_uncorrectable_ecc: An uncorrectable (double-bit) ECC error means
            corrupted memory — treat as fatal (drain + recover).
        correctable_ecc_warn_rate: Correctable ECC errors per poll above which to
            warn (rising rate often precedes an uncorrectable event).
        fatal_on_remap_failure: A row-remap *failure* (no spare rows left) is fatal.

        nvlink_error_warn / nvlink_error_fatal: Per-poll increase in NVLink
            recovery/CRC errors above which to warn / treat as fatal.

        temp_warn_c: Warn above this GPU temperature (°C).
        temp_fatal_margin_c: Fatal when within this many °C of the GPU's HW slowdown
            /shutdown limit (imminent thermal shutdown).

        fatal_xids: NVIDIA XID codes that indicate a broken GPU needing a swap.
            Defaults cover double-bit ECC (48), row-remap (63/64), NVLink (74),
            GPU fell off the bus (79), and contained/uncontained ECC (94/95).
        sources: Which telemetry sources to enable ("nvml"; extensible).
    """

    enable: bool = True
    poll_interval_s: float = 5.0

    # ECC / memory
    fatal_on_uncorrectable_ecc: bool = True
    correctable_ecc_warn_rate: int = 100
    fatal_on_remap_failure: bool = True

    # NVLink
    nvlink_error_warn: int = 100
    nvlink_error_fatal: int = 10000

    # Thermal
    temp_warn_c: int = 88
    temp_fatal_margin_c: int = 2

    # XID (needs DCGM or dmesg; NVML exposes only some)
    fatal_xids: tuple[int, ...] = (48, 63, 64, 74, 79, 94, 95)

    sources: tuple[str, ...] = ("nvml",)
