"""Unified GEMINI checkpoint and SCOUT detection scheduling."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ResiliencyCadence:
    """Use one shared interval for SCOUT certification and GEMINI capture."""

    interval: int = 0
    checkpoint_enabled: bool = False
    detection_enabled: bool = False

    def __post_init__(self) -> None:
        if (self.checkpoint_enabled or self.detection_enabled) and self.interval <= 0:
            raise ValueError("interval must be greater than zero when resiliency is enabled")

    @classmethod
    def from_component_intervals(
        cls,
        *,
        checkpoint_interval: int = 0,
        detection_interval: int = 0,
    ) -> ResiliencyCadence:
        """Build a unified cadence and reject independently configured intervals."""
        enabled_intervals = {
            interval for interval in (checkpoint_interval, detection_interval) if interval > 0
        }
        if len(enabled_intervals) > 1:
            raise ValueError(
                "GEMINI and SCOUT must use one interval; configure cadence through "
                "enable_resiliency(interval=...)"
            )
        interval = next(iter(enabled_intervals), 0)
        return cls(
            interval=interval,
            checkpoint_enabled=checkpoint_interval > 0,
            detection_enabled=detection_interval > 0,
        )

    def detection_due(self, step: int) -> bool:
        return self.detection_enabled and step % self.interval == 0

    def checkpoint_due(self, step: int) -> bool:
        if not self.checkpoint_enabled:
            return False
        return step % self.interval == 0
