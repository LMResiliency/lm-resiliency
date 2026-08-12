"""SCOUT switch-fault localization via cross-group network tomography.

Detects a faulty/degraded **network switch** on an HPC InfiniBand + RDMA fabric with
static (destination-based) routing, from **application-level collective performance** —
the completion times of the training's own per-group collectives (DP AllReduce, FSDP
AllGather, ...). A switch is *shared*, so it degrades **every group whose collective
crosses it and no group that avoids it**; intersecting the slow groups' paths against
the fabric topology pinpoints the switch. Design:
docs/scout.md#stragglers-and-communication-localization.

Scope: switch granularity (not link/port); recovery is **report-only** (the provider
owns the switch). The two live seams are (a) building `FabricTopology` from the IB
subnet manager's routing (deterministic under static routing) and (b) feeding per-group
collective timings in; both are injected here so the localization logic is pure and
unit-tested.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class GroupMeasurement:
    """One collective's completion time on one peer group, this round.

    `collective` labels the *like-for-like* comparison class (same op + message size +
    topology role), so only parallel instances of the same collective are compared.
    `failed=True` marks a hang/timeout (a hard failure), always treated as slow.
    """

    group_id: str
    members: tuple[str, ...]  # node ids in the group
    collective: str
    latency_s: float
    failed: bool = False


@dataclass
class SwitchVerdict:
    kind: str | None  # "fabric:switch" | "node" | None
    suspect: str | None  # switch id, or the common node, or None
    confirmed: bool  # persisted `persistence` rounds → safe to report
    slow_groups: int
    streak: int = 0  # consecutive rounds this suspect has held
    candidates: tuple[str, ...] = ()  # >1 ⇒ ambiguous, needs a targeted probe
    detail: str = ""


class FabricTopology:
    """Static-routing fabric model: which switches a group's collective traverses.

    A 2-tier fat tree: every node hangs off a **leaf** switch, and cross-leaf traffic
    crosses a **spine** determined by the (deterministic) route. `switches_on_path`
    returns π(g) — the set of switches a collective among `members` touches: their
    leaves, plus the spine(s) linking distinct leaves.
    """

    def __init__(
        self,
        node_leaf: dict[str, str],
        spine_route: dict[frozenset[str], str] | None = None,
        default_spine: str = "spine0",
    ) -> None:
        self._node_leaf = dict(node_leaf)
        self._spine_route = dict(spine_route or {})
        self._default_spine = default_spine

    @classmethod
    def from_dict(cls, d: dict) -> FabricTopology:
        """Load from the map an IB-SM adapter emits: {node_leaf, spine_route?,
        default_spine?}. JSON can't key on sets, so `spine_route` is a list of
        [leaf_a, leaf_b, spine] triples."""
        route = {frozenset((a, b)): s for a, b, s in d.get("spine_route", [])}
        return cls(d["node_leaf"], route, d.get("default_spine", "spine0"))

    def leaf_of(self, node: str) -> str:
        return self._node_leaf[node]

    def _spine(self, leaf_a: str, leaf_b: str) -> str:
        return self._spine_route.get(frozenset((leaf_a, leaf_b)), self._default_spine)

    def switches_on_path(self, members) -> set[str]:
        leaves = {self._node_leaf[n] for n in members}
        switches: set[str] = set(leaves)
        leaf_list = sorted(leaves)
        for i in range(len(leaf_list)):
            for j in range(i + 1, len(leaf_list)):
                switches.add(self._spine(leaf_list[i], leaf_list[j]))
        return switches

    def nodes_behind(self, switch: str) -> set[str]:
        """Blast radius: nodes whose leaf is `switch` (for report context)."""
        return {n for n, leaf in self._node_leaf.items() if leaf == switch}


class SwitchLocalizer:
    """Cross-group tomography over per-round collective timings → suspect switch.

    Per round: flag each group slow vs. its like-for-like siblings (or `failed`); if the
    slow groups share a common *node*, it's a node fault (defer to the node path); else
    the switch on every slow group and no fast one is the suspect. A suspect must persist
    `persistence` rounds before it is `confirmed` (rejects transient congestion).
    """

    def __init__(
        self,
        topology: FabricTopology,
        slow_factor: float = 1.5,
        persistence: int = 3,
        cover_threshold: float = 0.999,
        clean_threshold: float = 1e-9,
    ) -> None:
        self._topo = topology
        self._slow_factor = slow_factor
        self._persistence = persistence
        self._cover = cover_threshold
        self._clean = clean_threshold
        self._last_suspect: str | None = None
        self._streak = 0

    def observe(
        self,
        measurements: list[GroupMeasurement],
        compute_stragglers: frozenset[str] = frozenset(),
    ) -> SwitchVerdict:
        """One detection round. `compute_stragglers` are nodes SCOUT's replay already
        blamed for compute slowness — groups slow *only* because of them are excluded so
        a slow GPU isn't misattributed to the fabric."""
        slow, fast = self._partition(measurements, compute_stragglers)

        if not slow:
            self._last_suspect, self._streak = None, 0
            return SwitchVerdict(None, None, False, 0, detail="no slow groups")

        # Discriminator: a node in *every* slow group is a node/NIC fault, not a switch.
        common_nodes = set(slow[0].members).intersection(*(set(m.members) for m in slow[1:]))
        if common_nodes:
            self._last_suspect, self._streak = None, 0
            node = sorted(common_nodes)[0]
            return SwitchVerdict(
                "node",
                node,
                False,
                len(slow),
                detail=f"common endpoint {node} — node fault, not a switch",
            )

        candidates = self._hitting_set(slow, fast)
        if len(candidates) != 1:
            self._last_suspect, self._streak = None, 0
            return SwitchVerdict(
                None,
                None,
                False,
                len(slow),
                candidates=tuple(sorted(candidates)),
                detail=(
                    "ambiguous — needs a targeted probe"
                    if candidates
                    else "no switch explains the slow set"
                ),
            )

        suspect = next(iter(candidates))
        self._streak = self._streak + 1 if suspect == self._last_suspect else 1
        self._last_suspect = suspect
        confirmed = self._streak >= self._persistence
        return SwitchVerdict(
            "fabric:switch",
            suspect,
            confirmed,
            len(slow),
            streak=self._streak,
            detail=f"switch {suspect}: on all {len(slow)} slow groups, no healthy group "
            f"({self._streak}/{self._persistence} rounds)",
        )

    # ── internals ───────────────────────────────────────────────────────────
    def _partition(self, measurements, compute_stragglers):
        """Split groups into slow / fast, like-for-like within each collective class."""
        by_coll: dict[str, list[GroupMeasurement]] = defaultdict(list)
        for m in measurements:
            by_coll[m.collective].append(m)

        slow, fast = [], []
        for siblings in by_coll.values():
            baseline = min(m.latency_s for m in siblings)  # fastest sibling = healthy norm
            for m in siblings:
                is_slow = m.failed or (baseline > 0 and m.latency_s > self._slow_factor * baseline)
                # drop a slow group fully explained by a known compute straggler member
                if is_slow and compute_stragglers and set(m.members) & compute_stragglers:
                    continue
                (slow if is_slow else fast).append(m)
        return slow, fast

    def _hitting_set(self, slow, fast) -> set[str]:
        """Switches on *every* slow group's path and on *no* healthy group's path."""
        n_slow, n_fast = len(slow), len(fast)
        slow_paths = [self._topo.switches_on_path(m.members) for m in slow]
        fast_paths = [self._topo.switches_on_path(m.members) for m in fast]
        candidates: set[str] = set().union(*slow_paths) if slow_paths else set()
        out = set()
        for s in candidates:
            slow_cover = sum(s in p for p in slow_paths) / n_slow
            clean = (sum(s in p for p in fast_paths) / n_fast) if n_fast else 0.0
            if slow_cover >= self._cover and clean <= self._clean:
                out.add(s)
        return out

    def report(self, verdict: SwitchVerdict) -> dict | None:
        """Report-only recovery: a confirmed switch → a `fabric:switch` event for the
        provider (health/ticket API) + the quarantine DB. No ranks to swap."""
        if not (verdict.confirmed and verdict.kind == "fabric:switch"):
            return None
        return {
            "kind": "fabric:switch",
            "switch": verdict.suspect,
            "failed_ranks": [],  # report-only: the provider fixes/replaces the switch
            "blast_radius": sorted(self._topo.nodes_behind(verdict.suspect)),
            "detail": verdict.detail,
        }
