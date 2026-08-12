"""IB subnet-manager topology adapter → `FabricTopology` (the live seam for
switch-fault localization, see switch_localizer.py and
docs/scout.md#stragglers-and-communication-localization).

Parses `ibnetdiscover` output — the standard InfiniBand fabric dump — into the
`node_leaf` + `spine_route` map SCOUT needs to compute π(g). Switches are keyed by
their **GUID** (stable, and the id you report to the provider); each host is keyed by
the **hostname** in its HCA description, so the ids line up with the training node ids
used in `GroupMeasurement`s.

Scope: a 2-tier leaf/spine fat tree with static routing. `spine_route` is derived by
shortest path over the switch graph — exact for a single spine and symmetric trees;
for multi-spine ECMP the precise per-destination route lives in the switches' linear
forwarding tables (`ibroute` / `dump_lfts`), a further refinement.

IMPORTANT: `ibnetdiscover` formatting varies across IB tool/OFED versions. This parser
targets the common documented format and is tolerant (it skips lines it doesn't
recognize); **validate it against your cluster's real output**, or supply the map
directly via `FabricTopology.from_dict`.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field

from lm_resiliency.detection.switch_localizer import FabricTopology

# `Switch 40 "S-…guid…"  # "name…" …`  /  `Ca 2 "H-…guid…"  # "hostname mlx5_0"`
_HEADER = re.compile(r'^(Switch|Ca)\s+\d+\s+"([^"]+)"(?:.*?#\s*"([^"]*)")?')
# a port/link line inside a block: `[3]  "S-…peer…"[1]` or `[1](0x…)  "S-…"[3]  # …`
_LINK = re.compile(r'^\[\d+\](?:\([^)]*\))?\s+"([^"]+)"')


@dataclass
class Fabric:
    """Parsed IB fabric: switch GUIDs, host→leaf attachment, switch adjacency."""

    switch_guids: set[str] = field(default_factory=set)
    ca_host: dict[str, str] = field(default_factory=dict)  # CA guid → hostname
    node_leaf: dict[str, str] = field(default_factory=dict)  # hostname → leaf switch guid
    adjacency: dict[str, set[str]] = field(
        default_factory=dict
    )  # switch guid → neighbor switch guids
    leaf_guids: set[str] = field(default_factory=set)  # switches with ≥1 host attached
    switch_name: dict[str, str] = field(default_factory=dict)  # guid → admin description (cosmetic)

    @property
    def spine_guids(self) -> set[str]:
        return self.switch_guids - self.leaf_guids

    def _shortest_path(self, src: str, dst: str) -> list[str]:
        """BFS over the switch graph (src, dst are leaf guids)."""
        if src == dst:
            return [src]
        prev = {src: None}
        q = deque([src])
        while q:
            u = q.popleft()
            for v in sorted(self.adjacency.get(u, ())):  # sorted → deterministic route
                if v not in prev:
                    prev[v] = u
                    if v == dst:
                        path, cur = [], v
                        while cur is not None:
                            path.append(cur)
                            cur = prev[cur]
                        return list(reversed(path))
                    q.append(v)
        return []  # disconnected (shouldn't happen in a healthy fabric)

    def to_topology(self) -> FabricTopology:
        """Build `FabricTopology`: node_leaf + spine_route (spine on each leaf pair's path)."""
        leaves = sorted(self.leaf_guids)
        spine_route: dict[frozenset[str], str] = {}
        for i in range(len(leaves)):
            for j in range(i + 1, len(leaves)):
                path = self._shortest_path(leaves[i], leaves[j])
                spines = [s for s in path if s in self.spine_guids]
                if spines:  # first spine on the (deterministic) path carries the pair
                    spine_route[frozenset((leaves[i], leaves[j]))] = spines[0]
        default_spine = sorted(self.spine_guids)[0] if self.spine_guids else ""
        return FabricTopology(self.node_leaf, spine_route, default_spine=default_spine)


def parse_ibnetdiscover(text: str) -> Fabric:
    """Parse `ibnetdiscover` output into a `Fabric`."""
    lines = text.splitlines()
    fab = Fabric()

    # Pass 1: collect every Switch/Ca header (guids + names) so link peers resolve.
    for line in lines:
        h = _HEADER.match(line.strip())
        if not h:
            continue
        kind, guid, desc = h.group(1), h.group(2), h.group(3) or ""
        if kind == "Switch":
            fab.switch_guids.add(guid)
            fab.switch_name[guid] = desc.strip()
            fab.adjacency.setdefault(guid, set())
        else:  # Ca — hostname is the first token of the description
            fab.ca_host[guid] = desc.split()[0] if desc.split() else guid

    # Pass 2: walk link lines under their block; wire host→leaf and switch↔switch.
    cur_kind = cur_guid = None
    for line in lines:
        s = line.strip()
        h = _HEADER.match(s)
        if h:
            cur_kind, cur_guid = h.group(1), h.group(2)
            continue
        link = _LINK.match(s)
        if not (link and cur_guid):
            continue
        peer = link.group(1)
        if cur_kind == "Switch":
            if peer in fab.ca_host:  # a host hangs off this switch → it's a leaf
                fab.leaf_guids.add(cur_guid)
                fab.node_leaf[fab.ca_host[peer]] = cur_guid
            elif peer in fab.switch_guids:  # switch↔switch link
                fab.adjacency[cur_guid].add(peer)
                fab.adjacency.setdefault(peer, set()).add(cur_guid)
    return fab


def build_fabric_topology(ibnetdiscover_text: str) -> FabricTopology:
    """Convenience: `ibnetdiscover` text → `FabricTopology` for the SwitchLocalizer."""
    return parse_ibnetdiscover(ibnetdiscover_text).to_topology()
