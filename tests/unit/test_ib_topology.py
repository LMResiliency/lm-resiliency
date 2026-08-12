"""Unit tests for the IB-SM topology adapter (ibnetdiscover → FabricTopology).

No fabric needed — parses a representative `ibnetdiscover` fixture. (Real IB tool
output varies by version; this validates the parser + route derivation, not a live
fabric.)"""

from __future__ import annotations

from lm_resiliency.detection.ib_topology import build_fabric_topology, parse_ibnetdiscover
from lm_resiliency.detection.switch_localizer import GroupMeasurement, SwitchLocalizer

# 2 leaves + 1 spine, 4 hosts: n0,n1 → S-leaf1; n2,n3 → S-leaf2; cross-leaf via S-spine1.
IBND_1SPINE = """
Switch  40 "S-leaf1"    # "leaf1" enhanced port 0 lid 1
[1]     "H-n0"[1](0x10)         # "n0 mlx5_0" lid 10 4xHDR
[2]     "H-n1"[1](0x11)         # "n1 mlx5_0" lid 11 4xHDR
[3]     "S-spine1"[1]           # "spine1" lid 3 4xHDR

Switch  40 "S-leaf2"    # "leaf2" enhanced port 0 lid 2
[1]     "H-n2"[1](0x12)         # "n2 mlx5_0" lid 12 4xHDR
[2]     "H-n3"[1](0x13)         # "n3 mlx5_0" lid 13 4xHDR
[3]     "H-n4"[1](0x14)         # "n4 mlx5_0" lid 14 4xHDR
[4]     "S-spine1"[2]           # "spine1" lid 3 4xHDR

Switch  36 "S-spine1"   # "spine1" enhanced port 0 lid 3
[1]     "S-leaf1"[3]            # "leaf1" lid 1 4xHDR
[2]     "S-leaf2"[4]            # "leaf2" lid 2 4xHDR

Ca      1 "H-n0"        # "n0 mlx5_0"
[1](0x10)   "S-leaf1"[1]            # lid 10
Ca      1 "H-n1"        # "n1 mlx5_0"
[1](0x11)   "S-leaf1"[2]            # lid 11
Ca      1 "H-n2"        # "n2 mlx5_0"
[1](0x12)   "S-leaf2"[1]            # lid 12
Ca      1 "H-n3"        # "n3 mlx5_0"
[1](0x13)   "S-leaf2"[2]            # lid 13
Ca      1 "H-n4"        # "n4 mlx5_0"
[1](0x14)   "S-leaf2"[3]            # lid 14
"""


def test_parse_classifies_leaves_spines_and_hosts():
    fab = parse_ibnetdiscover(IBND_1SPINE)
    assert fab.switch_guids == {"S-leaf1", "S-leaf2", "S-spine1"}
    assert fab.leaf_guids == {"S-leaf1", "S-leaf2"}
    assert fab.spine_guids == {"S-spine1"}
    assert fab.node_leaf == {
        "n0": "S-leaf1",
        "n1": "S-leaf1",
        "n2": "S-leaf2",
        "n3": "S-leaf2",
        "n4": "S-leaf2",
    }
    assert fab.adjacency["S-leaf1"] == {"S-spine1"} and fab.adjacency["S-spine1"] == {
        "S-leaf1",
        "S-leaf2",
    }


def test_topology_paths_from_ibnetdiscover():
    t = build_fabric_topology(IBND_1SPINE)
    assert t.switches_on_path(["n0", "n1"]) == {"S-leaf1"}  # intra-leaf
    assert t.switches_on_path(["n2", "n3"]) == {"S-leaf2"}
    assert t.switches_on_path(["n0", "n2"]) == {
        "S-leaf1",
        "S-leaf2",
        "S-spine1",
    }  # cross-leaf → spine
    assert t.nodes_behind("S-leaf2") == {"n2", "n3", "n4"}


def test_end_to_end_localize_from_parsed_topology():
    """Parsed topology + synthetic per-group timings → SwitchLocalizer names the leaf.

    Two cross-leaf slow groups (disjoint endpoints) plus an intra-leaf2 slow group —
    the intra-leaf2 one crosses leaf2 but NOT the spine, which drops the spine's
    slow-cover below 1 and leaves leaf2 as the unique culprit."""
    loc = SwitchLocalizer(build_fabric_topology(IBND_1SPINE), persistence=1)
    ms = [
        GroupMeasurement("g1", ("n2", "n0"), "ar", 5.0),  # slow cross-leaf (leaf1,leaf2,spine1)
        GroupMeasurement("g2", ("n3", "n1"), "ar", 5.0),  # slow cross-leaf, disjoint from g1
        GroupMeasurement("g3", ("n3", "n4"), "ar", 5.0),  # slow INTRA-leaf2 → exonerates the spine
        GroupMeasurement("g4", ("n0", "n1"), "ar", 1.0),  # fast leaf1
    ]
    v = loc.observe(ms)
    assert v.kind == "fabric:switch" and v.suspect == "S-leaf2" and v.confirmed


# 2 leaves + 2 spines (ECMP): each leaf uplinks to both spines.
IBND_2SPINE = """
Switch  40 "S-leaf1"    # "leaf1"
[1]     "H-n0"[1](0x10)         # "n0 mlx5_0" lid 10
[3]     "S-spine1"[1]           # "spine1" lid 3
[4]     "S-spine2"[1]           # "spine2" lid 4

Switch  40 "S-leaf2"    # "leaf2"
[1]     "H-n1"[1](0x11)         # "n1 mlx5_0" lid 11
[3]     "S-spine1"[2]           # "spine1" lid 3
[4]     "S-spine2"[2]           # "spine2" lid 4

Switch  36 "S-spine1"   # "spine1"
[1]     "S-leaf1"[3]            # "leaf1"
[2]     "S-leaf2"[3]            # "leaf2"

Switch  36 "S-spine2"   # "spine2"
[1]     "S-leaf1"[4]            # "leaf1"
[2]     "S-leaf2"[4]            # "leaf2"

Ca      1 "H-n0"        # "n0 mlx5_0"
[1](0x10)   "S-leaf1"[1]            # lid 10
Ca      1 "H-n1"        # "n1 mlx5_0"
[1](0x11)   "S-leaf2"[1]            # lid 11
"""


def test_multi_spine_classification_and_deterministic_route():
    fab = parse_ibnetdiscover(IBND_2SPINE)
    assert fab.leaf_guids == {"S-leaf1", "S-leaf2"}
    assert fab.spine_guids == {"S-spine1", "S-spine2"}
    t = fab.to_topology()
    path = t.switches_on_path(["n0", "n1"])  # cross-leaf
    assert {"S-leaf1", "S-leaf2"} <= path
    spines_used = path & {"S-spine1", "S-spine2"}
    assert len(spines_used) == 1  # one deterministic spine chosen (LFT gives the exact one)
