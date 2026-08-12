"""HardwareHealthMonitor: poll telemetry, classify, and report fatal faults.

Runs a low-frequency out-of-band poll loop per worker. Driver and fabric counters
are ground truth for the device, so unlike SCOUT, no cross-rank consensus is needed.
Fatal events are passed to the caller through ``on_event``.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import Callable

from lm_resiliency.detection.health.config import HealthConfig
from lm_resiliency.detection.health.sources import HealthReading, HealthSeverity, HealthSource

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class HealthEvent:
    device: int
    metric: str
    value: float
    severity: HealthSeverity
    message: str


class HardwareHealthMonitor:
    """Classifies telemetry readings into events and reports fatal ones.

    Args:
        config: thresholds + cadence.
        sources: telemetry sources (e.g. one NvmlSource for this rank's GPU).
        on_event: called with each FATAL HealthEvent (once per device+metric).
    """

    def __init__(
        self,
        config: HealthConfig,
        sources: list[HealthSource],
        on_event: Callable[[HealthEvent], None] | None = None,
    ) -> None:
        self._cfg = config
        self._sources = sources
        self._on_event = on_event
        self._prev: dict[tuple[int, str], float] = {}  # last value for delta metrics
        self._fired: set[tuple[int, str]] = set()  # fatal already reported
        self._warned: set[tuple[int, str]] = set()  # warn already logged
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # ── classification ────────────────────────────────────────────────────────
    def poll_once(self) -> list[HealthEvent]:
        """Read all sources once; return the *new* FATAL events (deduped)."""
        readings: list[HealthReading] = []
        for src in self._sources:
            try:
                readings.extend(src.read())
            except Exception as e:  # a source failing shouldn't kill the monitor
                logger.warning(f"health source {type(src).__name__} read failed: {e}")

        limits = {r.device: r.value for r in readings if r.metric == "temp_shutdown_limit"}
        new_fatal: list[HealthEvent] = []

        for r in readings:
            sev, msg = self._classify(r, limits)
            if sev == HealthSeverity.OK:
                continue
            key = (r.device, r.metric)
            if sev == HealthSeverity.FATAL:
                if key in self._fired:
                    continue
                self._fired.add(key)
                ev = HealthEvent(r.device, r.metric, r.value, sev, msg)
                logger.error(f"HEALTH FATAL: {msg}")
                new_fatal.append(ev)
                if self._on_event is not None:
                    self._on_event(ev)
            elif key not in self._warned:
                self._warned.add(key)
                logger.warning(f"HEALTH WARN: {msg}")

        # Update deltas after classification so the first sample isn't a spike.
        for r in readings:
            if r.metric in ("ecc_correctable", "nvlink_errors"):
                self._prev[(r.device, r.metric)] = r.value
        return new_fatal

    def _classify(self, r: HealthReading, limits: dict[int, float]) -> tuple[HealthSeverity, str]:
        c = self._cfg
        d, v = r.device, r.value
        m = r.metric
        if m == "ecc_uncorrectable" and v > 0 and c.fatal_on_uncorrectable_ecc:
            return HealthSeverity.FATAL, f"GPU{d}: {int(v)} uncorrectable ECC error(s)"
        if m == "remap_failure" and v > 0 and c.fatal_on_remap_failure:
            return HealthSeverity.FATAL, f"GPU{d}: row-remap failure (no spare rows)"
        if m == "device_lost":
            return HealthSeverity.FATAL, f"GPU{d}: device lost ({r.detail})"
        if m == "xid":
            if int(v) in c.fatal_xids:
                return HealthSeverity.FATAL, f"GPU{d}: fatal XID {int(v)}"
            return HealthSeverity.WARN, f"GPU{d}: XID {int(v)}"
        if m == "remap_pending" and v > 0:
            return HealthSeverity.WARN, f"GPU{d}: row remap pending"
        if m == "nvlink_errors":
            delta = v - self._prev.get((d, m), v)
            if delta >= c.nvlink_error_fatal:
                return HealthSeverity.FATAL, f"GPU{d}: +{int(delta)} NVLink errors this poll"
            if delta >= c.nvlink_error_warn:
                return HealthSeverity.WARN, f"GPU{d}: +{int(delta)} NVLink errors this poll"
        if m == "ecc_correctable":
            delta = v - self._prev.get((d, m), v)
            if delta >= c.correctable_ecc_warn_rate:
                return HealthSeverity.WARN, f"GPU{d}: +{int(delta)} correctable ECC this poll"
        if m == "temperature":
            limit = limits.get(d)
            if limit and v >= limit - c.temp_fatal_margin_c:
                return HealthSeverity.FATAL, f"GPU{d}: {int(v)}C near shutdown ({int(limit)}C)"
            if v >= c.temp_warn_c:
                return HealthSeverity.WARN, f"GPU{d}: {int(v)}C hot"
        return HealthSeverity.OK, ""

    # ── lifecycle ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        if not self._cfg.enable or self._thread is not None:
            return

        def _loop() -> None:
            while not self._stop.wait(self._cfg.poll_interval_s):
                try:
                    self.poll_once()
                except Exception as e:  # never let the monitor thread die silently
                    logger.warning(f"health poll failed: {e}")

        self._thread = threading.Thread(target=_loop, name="lm-health-monitor", daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for src in self._sources:
            src.close()
