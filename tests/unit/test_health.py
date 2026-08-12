"""Unit tests for hardware health monitoring (health/).

No GPU required — drives the monitor with a FakeSource and checks classification,
the fatal→callback path, WARN-does-not-report, and dedup.
"""

from __future__ import annotations

from lm_resiliency.detection.health import (
    FakeSource,
    HardwareHealthMonitor,
    HealthConfig,
    HealthReading,
    HealthSeverity,
)


def _monitor(readings):
    events = []
    src = FakeSource(
        [HealthReading(*r) if not isinstance(r, HealthReading) else r for r in readings]
    )
    mon = HardwareHealthMonitor(HealthConfig(), [src], on_event=events.append)
    return mon, src, events


def test_uncorrectable_ecc_is_fatal():
    mon, _, events = _monitor([("nvml", 0, "ecc_uncorrectable", 1.0, "")])
    fatal = mon.poll_once()
    assert len(fatal) == 1 and fatal[0].metric == "ecc_uncorrectable"
    assert fatal[0].severity == HealthSeverity.FATAL
    assert len(events) == 1  # callback fired


def test_remap_failure_and_device_lost_are_fatal():
    mon, _, events = _monitor(
        [
            ("nvml", 0, "remap_failure", 1.0, ""),
            ("nvml", 1, "device_lost", 1.0, "queries failed"),
        ]
    )
    metrics = {e.metric for e in mon.poll_once()}
    assert metrics == {"remap_failure", "device_lost"}
    assert len(events) == 2


def test_xid_fatal_vs_warn():
    mon_f, _, ev_f = _monitor([("nvml", 0, "xid", 79.0, "")])  # off-bus → fatal
    assert len(mon_f.poll_once()) == 1 and len(ev_f) == 1
    mon_w, _, ev_w = _monitor([("nvml", 0, "xid", 13.0, "")])  # graphics exception → warn only
    assert mon_w.poll_once() == [] and ev_w == []


def test_temperature_warn_vs_fatal_near_shutdown():
    # 90C with warn=88, no shutdown limit present → WARN (no fault)
    mon_w, _, ev_w = _monitor([("nvml", 0, "temperature", 90.0, "")])
    assert mon_w.poll_once() == [] and ev_w == []
    # 91C with shutdown limit 92 and margin 2 → within margin → FATAL
    mon_f, _, ev_f = _monitor(
        [
            ("nvml", 0, "temp_shutdown_limit", 92.0, ""),
            ("nvml", 0, "temperature", 91.0, ""),
        ]
    )
    fatal = mon_f.poll_once()
    assert len(fatal) == 1 and fatal[0].metric == "temperature"
    assert len(ev_f) == 1


def test_nvlink_delta_thresholds():
    src = FakeSource([HealthReading("nvml", 0, "nvlink_errors", 0.0)])
    events = []
    mon = HardwareHealthMonitor(HealthConfig(), [src], on_event=events.append)
    mon.poll_once()  # baseline 0
    src.readings = [HealthReading("nvml", 0, "nvlink_errors", 20000.0)]  # jump > fatal (10000)
    fatal = mon.poll_once()
    assert len(fatal) == 1 and fatal[0].metric == "nvlink_errors"


def test_fatal_is_deduped():
    mon, _, events = _monitor([("nvml", 0, "ecc_uncorrectable", 1.0, "")])
    assert len(mon.poll_once()) == 1
    assert mon.poll_once() == []  # persistent fault does not re-fire
    assert len(events) == 1


def test_disabled_config_does_not_report():
    src = FakeSource([HealthReading("nvml", 0, "ecc_uncorrectable", 1.0)])
    cfg = HealthConfig(fatal_on_uncorrectable_ecc=False)
    mon = HardwareHealthMonitor(cfg, [src], on_event=lambda e: None)
    assert mon.poll_once() == []
