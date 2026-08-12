"""Per-group collective-timing collector — the live seam feeding the SwitchLocalizer.

SCOUT localizes a faulty switch from **application-level** collective performance
(docs/scout.md#stragglers-and-communication-localization). This collects that signal: it times the training's
own collectives per process group, buffers per-group latencies over a detection window,
and `snapshot()`s them into `GroupMeasurement`s for `SwitchLocalizer.observe`.

Two pieces are runtime seams (need a live distributed run, so they're injected here and
the buffering/aggregation is kept pure + unit-tested):

  1. **The timing hook.** `timed()` wraps a collective with a wall clock; for NCCL —
     which is async on the CUDA stream — wrap with CUDA events (record before/after,
     `event.elapsed_time` after a later sync) or `torch.cuda.synchronize()` around the
     call, so the measurement is the *collective's* duration, not the enqueue time.
  2. **The cross-rank gather.** Each rank only times its *own* groups; the tomography
     needs every group. `run_detection_round(gather=...)` gathers all ranks' snapshots
     to the correlator (e.g. an all-gather of the small measurement list) before
     `observe`. The default gather is identity (single-rank / tests).
"""

from __future__ import annotations

import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
from statistics import median
from typing import Callable

from lm_resiliency.detection.switch_localizer import (
    GroupMeasurement,
    SwitchLocalizer,
    SwitchVerdict,
)


@dataclass(frozen=True)
class GroupSpec:
    """A peer group's identity for tomography: its like-for-like collective label
    (op + message size + topology role) and its member **node** ids."""

    collective: str
    members: tuple[str, ...]

    @classmethod
    def of(cls, collective: str, member_ranks, node_of_rank: Callable[[int], str]) -> GroupSpec:
        """Build from a process group's member ranks + a rank→node map (peer_group.py
        gives the ranks; the roster gives the nodes)."""
        return cls(collective, tuple(node_of_rank(r) for r in member_ranks))


class CollectiveTimingCollector:
    """Buffers per-group collective latencies over a detection window (one rank's view).

    `record`/`timed` accumulate samples; `snapshot` aggregates each group to a robust
    (median) latency + a failed flag and clears the window.
    """

    def __init__(
        self, groups: dict[str, GroupSpec], clock: Callable[[], float] = time.perf_counter
    ):
        self._groups = dict(groups)
        self._clock = clock
        self._samples: dict[str, list[float]] = defaultdict(list)
        self._failed: set[str] = set()

    def record(self, group_id: str, latency_s: float, failed: bool = False) -> None:
        if group_id not in self._groups:
            raise KeyError(f"unknown group {group_id!r}")
        self._samples[group_id].append(latency_s)
        if failed:
            self._failed.add(group_id)

    @contextmanager
    def timed(self, group_id: str):
        """Time a collective and record it. Wall-clock by default — for NCCL, wrap the
        body so the collective has actually completed (CUDA events / synchronize)."""
        t0 = self._clock()
        ok = True
        try:
            yield
        except BaseException:  # a hang/timeout/abort on this collective = failed
            ok = False
            raise
        finally:
            self.record(group_id, self._clock() - t0, failed=not ok)

    def mark_failed(self, group_id: str) -> None:
        """Record a hang/timeout with no usable latency (e.g. a watchdog abort)."""
        if group_id not in self._groups:
            raise KeyError(f"unknown group {group_id!r}")
        self._samples.setdefault(group_id, [])
        self._failed.add(group_id)

    def snapshot(self, reset: bool = True) -> list[GroupMeasurement]:
        """Aggregate the window into one `GroupMeasurement` per group with samples."""
        out = []
        for gid, samples in self._samples.items():
            spec = self._groups[gid]
            latency = median(samples) if samples else float("inf")
            out.append(
                GroupMeasurement(
                    gid, spec.members, spec.collective, latency, failed=gid in self._failed
                )
            )
        if reset:
            self._samples.clear()
            self._failed.clear()
        return out


def run_detection_round(
    collector: CollectiveTimingCollector,
    localizer: SwitchLocalizer,
    gather: Callable[[list[GroupMeasurement]], list[GroupMeasurement]] = lambda m: m,
    compute_stragglers: frozenset[str] = frozenset(),
) -> SwitchVerdict:
    """One round: snapshot this rank's timings → gather all ranks' → localize.

    `gather` collects every rank's measurements to the correlator (all-gather in
    production; identity for a single rank / tests). `compute_stragglers` are nodes
    SCOUT's replay already blamed, so a slow GPU isn't misread as a switch.
    """
    measurements = gather(collector.snapshot())
    return localizer.observe(measurements, compute_stragglers)
