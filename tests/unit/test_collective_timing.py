"""Unit tests for the per-group collective-timing collector (SwitchLocalizer feed).

Pure buffering/aggregation — the CUDA timing hook + cross-rank gather are seams
(injected)."""

from __future__ import annotations

import pytest

from lm_resiliency.detection.collective_timing import (
    CollectiveTimingCollector,
    GroupSpec,
    run_detection_round,
)
from lm_resiliency.detection.switch_localizer import FabricTopology, SwitchLocalizer

NODE_LEAF = {f"m{i}": leaf for i, leaf in enumerate("AABBCCDD")}


def test_group_spec_of_maps_ranks_to_nodes():
    spec = GroupSpec.of("ar", [2, 4], node_of_rank=lambda r: f"m{r}")
    assert spec == GroupSpec("ar", ("m2", "m4"))


def test_record_and_snapshot_median_aggregation():
    groups = {"g1": GroupSpec("ar", ("m2", "m4")), "g2": GroupSpec("ar", ("m0", "m1"))}
    c = CollectiveTimingCollector(groups)
    for lat in (5.0, 7.0, 6.0):  # window of samples → median 6.0
        c.record("g1", lat)
    c.record("g2", 1.0)
    snap = {m.group_id: m for m in c.snapshot()}
    assert snap["g1"].latency_s == 6.0 and snap["g1"].members == ("m2", "m4")
    assert snap["g2"].latency_s == 1.0
    assert not snap["g1"].failed
    assert c.snapshot() == []  # window cleared


def test_timed_records_elapsed_and_failure():
    # clock is read start/stop per `timed`: (100.0→100.5)=0.5 ok, (200.0→200.5)=0.5 failed
    clock = iter([100.0, 100.5, 200.0, 200.5])
    c = CollectiveTimingCollector({"g1": GroupSpec("ar", ("m2", "m4"))}, clock=lambda: next(clock))
    with c.timed("g1"):
        pass  # elapsed 0.5, ok
    with pytest.raises(RuntimeError):
        with c.timed("g1"):
            raise RuntimeError("collective aborted")  # elapsed 0.5, failed
    snap = c.snapshot()[0]
    assert snap.latency_s == 0.5  # median([0.5, 0.5])
    assert snap.failed  # an aborted collective marks the group failed


def test_mark_failed_hang_with_no_latency():
    c = CollectiveTimingCollector({"g1": GroupSpec("ar", ("m2", "m4"))})
    c.mark_failed("g1")  # watchdog abort, no timing
    snap = c.snapshot()[0]
    assert snap.failed and snap.latency_s == float("inf")


def test_unknown_group_rejected():
    c = CollectiveTimingCollector({"g1": GroupSpec("ar", ("m2", "m4"))})
    with pytest.raises(KeyError):
        c.record("nope", 1.0)


def test_run_detection_round_localizes_switch():
    """End-to-end: buffered timings → snapshot → (identity gather) → localizer names
    the faulty leaf B."""
    topo = FabricTopology(NODE_LEAF)
    loc = SwitchLocalizer(topo, persistence=1)
    groups = {
        "g1": GroupSpec("ar", ("m2", "m4")),  # slow (m2 behind leaf B)
        "g2": GroupSpec("ar", ("m3", "m6")),  # slow (m3 behind leaf B), disjoint from g1
        "g5": GroupSpec("ar", ("m0", "m4")),  # fast cross-leaf → exonerates the spine
    }
    c = CollectiveTimingCollector(groups)
    c.record("g1", 5.0)
    c.record("g2", 5.0)
    c.record("g5", 1.0)
    v = run_detection_round(c, loc)
    assert v.kind == "fabric:switch" and v.suspect == "B" and v.confirmed
