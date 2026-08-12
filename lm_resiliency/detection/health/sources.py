"""Telemetry sources for hardware health monitoring.

A source reads raw counters from the platform and returns HealthReadings; the
monitor (monitor.py) classifies them into events. Sources are read-only and
side-effect-free so they can be polled from an out-of-band thread.
"""

from __future__ import annotations

import enum
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class HealthSeverity(enum.IntEnum):
    OK = 0
    WARN = 1  # log + track; does not trigger recovery
    FATAL = 2  # the device is broken/imminently gone → report a fault


@dataclass(slots=True)
class HealthReading:
    """One telemetry sample. ``metric`` is a stable key the monitor classifies."""

    source: str
    device: int
    metric: str
    value: float
    detail: str = ""


class HealthSource(ABC):
    @abstractmethod
    def read(self) -> list[HealthReading]:
        """Return the current readings (cheap; called every poll)."""

    def close(self) -> None:  # noqa: B027 - optional
        ...


class NvmlSource(HealthSource):
    """Reads GPU health for one device via NVML (pynvml).

    Emits readings for: uncorrectable / correctable ECC (volatile), row-remap
    pending / failure, per-GPU summed NVLink recovery+CRC errors, temperature and
    the HW shutdown threshold. If a device query raises, that itself is emitted as
    a ``device_lost`` reading (a GPU that fell off the bus).

    device_index is the NVML (physical) index. With CUDA_VISIBLE_DEVICES remapping
    CUDA ordinals, pass the physical index (or match by UUID) — see
    docs/scout.md#hardware-telemetry.
    """

    def __init__(self, device_index: int = 0) -> None:
        self._index = device_index
        self._nvml = None
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        except Exception as e:  # pragma: no cover - depends on platform
            logger.warning(f"NVML unavailable ({e}); GPU health monitoring disabled")

    @property
    def available(self) -> bool:
        return self._nvml is not None

    def read(self) -> list[HealthReading]:
        if self._nvml is None:
            return []
        n, h, dev = self._nvml, self._handle, self._index
        out: list[HealthReading] = []
        try:
            out.append(
                HealthReading(
                    "nvml",
                    dev,
                    "ecc_uncorrectable",
                    float(
                        n.nvmlDeviceGetTotalEccErrors(
                            h, n.NVML_MEMORY_ERROR_TYPE_UNCORRECTED, n.NVML_VOLATILE_ECC
                        )
                    ),
                )
            )
            out.append(
                HealthReading(
                    "nvml",
                    dev,
                    "ecc_correctable",
                    float(
                        n.nvmlDeviceGetTotalEccErrors(
                            h, n.NVML_MEMORY_ERROR_TYPE_CORRECTED, n.NVML_VOLATILE_ECC
                        )
                    ),
                )
            )
        except Exception:
            pass
        try:
            rows = n.nvmlDeviceGetRemappedRows(h)  # (corr, unc, pending, failure)
            out.append(HealthReading("nvml", dev, "remap_pending", float(bool(rows[2]))))
            out.append(HealthReading("nvml", dev, "remap_failure", float(bool(rows[3]))))
        except Exception:
            pass
        try:
            nvlink = 0
            for link in range(getattr(n, "NVML_NVLINK_MAX_LINKS", 18)):
                try:
                    nvlink += n.nvmlDeviceGetNvLinkErrorCounter(
                        h, link, n.NVML_NVLINK_ERROR_DL_RECOVERY
                    )
                    nvlink += n.nvmlDeviceGetNvLinkErrorCounter(
                        h, link, n.NVML_NVLINK_ERROR_DL_CRC_DATA
                    )
                except Exception:
                    break
            out.append(HealthReading("nvml", dev, "nvlink_errors", float(nvlink)))
        except Exception:
            pass
        try:
            temp = n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU)
            shutdown = n.nvmlDeviceGetTemperatureThreshold(h, n.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
            out.append(
                HealthReading(
                    "nvml", dev, "temperature", float(temp), detail=f"shutdown={shutdown}C"
                )
            )
            out.append(HealthReading("nvml", dev, "temp_shutdown_limit", float(shutdown)))
        except Exception:
            pass

        if not out:  # every query failed → the device is likely gone
            return [HealthReading("nvml", dev, "device_lost", 1.0, detail="NVML queries failed")]
        return out

    def close(self) -> None:
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # pragma: no cover
                pass


class FakeSource(HealthSource):
    """Injectable source for tests: returns whatever readings it's given."""

    def __init__(self, readings: list[HealthReading] | None = None) -> None:
        self.readings = readings or []

    def read(self) -> list[HealthReading]:
        return list(self.readings)
