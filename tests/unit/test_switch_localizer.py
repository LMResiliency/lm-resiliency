"""Unit tests for SCOUT switch-fault localization (cross-group tomography).

Pure algorithm — no GPU / fabric. A 2-tier fat tree: m0,m1→leaf A; m2,m3→B;
m4,m5→C; m6,m7→D; cross-leaf traffic crosses the default spine `spine0`."""

from __future__ import annotations

from lm_resiliency.detection.switch_localizer import (
    FabricTopology,
    GroupMeasurement,
    SwitchLocalizer,
)

NODE_LEAF = {f"m{i}": leaf for i, leaf in enumerate("AABBCCDD")}


def _topo():
    return FabricTopology(NODE_LEAF)


def _m(gid, members, latency, failed=False, collective="ar"):
    return GroupMeasurement(gid, tuple(members), collective, latency, failed)


# ── topology ──────────────────────────────────────────────────────────────────
def test_switches_on_path():
    t = _topo()
    assert t.switches_on_path(["m0", "m1"]) == {"A"}  # intra-leaf
    assert t.switches_on_path(["m0", "m2"]) == {"A", "B", "spine0"}  # cross-leaf → spine
    assert t.nodes_behind("B") == {"m2", "m3"}


def test_from_dict_with_spine_route():
    t = FabricTopology.from_dict(
        {"node_leaf": NODE_LEAF, "spine_route": [["A", "B", "spineX"]], "default_spine": "spine0"}
    )
    assert t.switches_on_path(["m0", "m2"]) == {"A", "B", "spineX"}  # A↔B routed via spineX
    assert t.switches_on_path(["m0", "m4"]) == {"A", "C", "spine0"}  # others default


# ── localization ──────────────────────────────────────────────────────────────
def test_localize_leaf_switch_fault():
    """Leaf B degraded: groups with a member behind B are slow (and share B but no
    common node); the unique switch on all slow + no fast group is B."""
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [
        _m("g1", ["m2", "m4"], 5.0),  # slow (m2 behind B)
        _m("g2", ["m3", "m6"], 5.0),  # slow (m3 behind B) — no common node with g1
        _m("g3", ["m4", "m5"], 1.0),  # fast (C)
        _m("g4", ["m6", "m7"], 1.0),  # fast (D)
        _m("g5", ["m0", "m4"], 1.0),  # fast cross-leaf (A,C,spine0) → exonerates spine0
    ]
    v = loc.observe(ms)
    assert v.kind == "fabric:switch" and v.suspect == "B" and v.confirmed


def test_localize_spine_fault():
    """Spine degraded: every cross-leaf group slow, every intra-leaf group fast → the
    spine is the unique suspect (leaves are cleared by the intra-leaf fast groups)."""
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [
        _m("g1", ["m0", "m2"], 5.0),  # slow cross-leaf
        _m("g2", ["m4", "m6"], 5.0),  # slow cross-leaf
        _m("g3", ["m0", "m1"], 1.0),  # fast A
        _m("g4", ["m2", "m3"], 1.0),  # fast B
        _m("g5", ["m4", "m5"], 1.0),  # fast C
        _m("g6", ["m6", "m7"], 1.0),  # fast D
    ]
    v = loc.observe(ms)
    assert v.kind == "fabric:switch" and v.suspect == "spine0" and v.confirmed


def test_node_fault_is_not_a_switch():
    """All slow groups share a common node → node/NIC fault, deferred to the node path."""
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [_m("g1", ["m2", "m4"], 5.0), _m("g2", ["m2", "m6"], 5.0), _m("g3", ["m4", "m5"], 1.0)]
    v = loc.observe(ms)
    assert v.kind == "node" and v.suspect == "m2"


def test_compute_straggler_not_blamed_on_fabric():
    """A slow group explained by a known compute straggler is dropped, not attributed
    to a switch."""
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [_m("g1", ["m2", "m4"], 5.0), _m("g2", ["m2", "m6"], 5.0), _m("g3", ["m4", "m5"], 1.0)]
    v = loc.observe(ms, compute_stragglers=frozenset({"m2"}))
    assert v.kind is None and v.slow_groups == 0


def test_hang_counts_as_slow():
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [
        _m("g1", ["m2", "m4"], 1.0, failed=True),  # hang (failed) despite low latency
        _m("g2", ["m3", "m6"], 1.0, failed=True),
        _m("g3", ["m4", "m5"], 1.0),
        _m("g4", ["m0", "m4"], 1.0),
    ]
    v = loc.observe(ms)
    assert v.kind == "fabric:switch" and v.suspect == "B"


def test_ambiguous_needs_targeted_probe():
    """When every slow group spans exactly B↔C, both B and C explain the slow set →
    ambiguous, not confirmed; the caller must probe (different nodes, same switch)."""
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [
        _m("g1", ["m2", "m4"], 5.0),  # B,C
        _m("g2", ["m3", "m5"], 5.0),  # B,C (no common node with g1)
        _m("g3", ["m0", "m6"], 1.0),  # fast A,D → clears spine0
    ]
    v = loc.observe(ms)
    assert v.kind is None and not v.confirmed
    assert v.candidates == ("B", "C")


def test_persistence_gates_confirmation_and_resets():
    """A suspect must hold `persistence` consecutive rounds; a clean round resets it
    (rejecting transient congestion)."""
    loc = SwitchLocalizer(_topo(), persistence=3)
    slow = [
        _m("g1", ["m2", "m4"], 5.0),
        _m("g2", ["m3", "m6"], 5.0),
        _m("g5", ["m0", "m4"], 1.0),
    ]
    clean = [_m("g1", ["m2", "m4"], 1.0), _m("g5", ["m0", "m4"], 1.0)]

    assert loc.observe(slow).confirmed is False  # round 1
    assert loc.observe(slow).confirmed is False  # round 2
    assert loc.observe(slow).confirmed is True  # round 3 → confirmed
    loc.observe(clean)  # transient clear → streak reset
    assert loc.observe(slow).confirmed is False  # back to round 1


def test_report_only_on_confirmed_switch():
    loc = SwitchLocalizer(_topo(), persistence=1)
    ms = [
        _m("g1", ["m2", "m4"], 5.0),
        _m("g2", ["m3", "m6"], 5.0),
        _m("g5", ["m0", "m4"], 1.0),
    ]
    v = loc.observe(ms)
    rep = loc.report(v)
    assert rep is not None
    assert rep["kind"] == "fabric:switch" and rep["switch"] == "B"
    assert rep["failed_ranks"] == []  # report-only: provider fixes the switch
    assert rep["blast_radius"] == ["m2", "m3"]
    # a node verdict or an unconfirmed one yields no switch report
    assert (
        loc.report(loc.observe([_m("g1", ["m2", "m4"], 5.0), _m("g2", ["m2", "m6"], 5.0)])) is None
    )
