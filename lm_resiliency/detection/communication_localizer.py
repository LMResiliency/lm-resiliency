"""Endpoint localization from overlapping process-group timing observations.

This module is deliberately framework agnostic. Training adapters provide one
``CollectiveObservation`` per process group and operation window; the localizer
normalizes each observation against that group's own like-for-like baseline and
correlates slow groups that exercise the same communication resource class.

This is endpoint inference, not fabric-path inference. It can implicate a rank, node,
HCA, or NIC represented in the job roster, but it never attributes a switch or link.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Mapping

_RESOURCE_KINDS = frozenset({"rank", "node", "hca", "nic"})


@dataclass(frozen=True)
class EndpointMetadata:
    """Physical communication identities associated with one training rank."""

    node: str | None = None
    hca: str | None = None
    nic: str | None = None


@dataclass(frozen=True)
class CollectiveObservation:
    """One process group's aggregate collective latency for a detection window.

    ``baseline_latency_s`` must come from the same group and comparison key:
    collective type, message bytes, group size, and topology role. A failed
    collective may omit its baseline because failure itself is a slow signal.

    ``resource_class`` separates communication domains such as ``inter_node`` and
    ``intra_node``. Healthy evidence only contradicts a suspect within the same
    resource class, so a healthy TP/NVLink group cannot exonerate an inter-node NIC.
    ``member_resources`` optionally overrides the rank roster with the resource(s)
    used by each member for this PG, which is required for accurate multi-rail
    inference.
    """

    group_id: str
    members: tuple[str, ...]
    collective: str
    message_bytes: int
    topology_role: str
    resource_class: str
    latency_s: float
    baseline_latency_s: float | None
    resource_kind: str = "rank"
    failed: bool = False
    member_resources: tuple[str | tuple[str, ...], ...] | None = None

    def __post_init__(self) -> None:
        if not self.group_id or not self.members:
            raise ValueError("group_id and members must be non-empty")
        if len(set(self.members)) != len(self.members):
            raise ValueError(f"group {self.group_id!r} contains duplicate members")
        if not self.collective or not self.topology_role or not self.resource_class:
            raise ValueError("collective, topology_role, and resource_class must be non-empty")
        if self.message_bytes < 0:
            raise ValueError("message_bytes must be non-negative")
        if self.resource_kind not in _RESOURCE_KINDS:
            raise ValueError(
                f"resource_kind must be one of {sorted(_RESOURCE_KINDS)}, "
                f"got {self.resource_kind!r}"
            )
        if self.member_resources is not None:
            if len(self.member_resources) != len(self.members):
                raise ValueError("member_resources must align one-to-one with members")
            for resources in self.member_resources:
                values = (resources,) if isinstance(resources, str) else resources
                if not values or any(not resource for resource in values):
                    raise ValueError("member_resources cannot contain empty resource ids")
        if not self.failed and (not math.isfinite(self.latency_s) or self.latency_s < 0):
            raise ValueError("a completed collective requires a finite, non-negative latency")
        if self.baseline_latency_s is None:
            if not self.failed:
                raise ValueError("a completed collective requires a baseline")
        elif not math.isfinite(self.baseline_latency_s) or self.baseline_latency_s <= 0:
            raise ValueError("baseline_latency_s must be finite and positive")

    @property
    def comparison_key(self) -> tuple[str, int, int, str]:
        """Dimensions that must match when constructing this group's baseline."""

        return (self.collective, self.message_bytes, len(self.members), self.topology_role)

    @property
    def normalized_latency(self) -> float:
        """Current latency divided by this process group's own clean baseline."""

        if self.failed:
            return math.inf
        assert self.baseline_latency_s is not None
        return self.latency_s / self.baseline_latency_s


@dataclass(frozen=True)
class EndpointCandidate:
    """Evidence for one physical endpoint candidate."""

    resource_class: str
    resource_kind: str
    resource_id: str
    ranks: tuple[str, ...]
    supporting_groups: tuple[str, ...]
    support_ratio: float


@dataclass(frozen=True)
class CommunicationVerdict:
    """Result of one overlapping-process-group cross-validation round."""

    status: str  # "normal" | "localized" | "ambiguous" | "inconclusive"
    candidates: tuple[EndpointCandidate, ...] = ()
    confirmed: bool = False
    streak: int = 0
    confidence: float = 0.0
    slow_groups: tuple[str, ...] = ()
    healthy_groups: tuple[str, ...] = ()
    excluded_compute_groups: tuple[str, ...] = ()
    contradicted_resources: tuple[str, ...] = ()
    detail: str = ""

    @property
    def suspect(self) -> EndpointCandidate | None:
        """The unique endpoint when this round is localizable."""

        return self.candidates[0] if self.status == "localized" else None


class CommunicationLocalizer:
    """Cross-validate communication slowdowns across overlapping process groups.

    A unique endpoint must occur in at least two slow groups with different endpoint
    incidence, survive healthy-group contradiction checks in the same resource class,
    and persist for ``persistence`` rounds before it is confirmed. When no endpoint
    covers every slow group, candidates supported by multiple groups are returned as a
    weighted multi-fault/ambiguous list rather than over-attributing one culprit.
    """

    def __init__(
        self,
        endpoint_metadata: Mapping[str, EndpointMetadata] | None = None,
        *,
        slow_factor: float = 1.5,
        healthy_factor: float = 1.15,
        persistence: int = 3,
        min_slow_groups: int = 2,
        max_candidates: int = 8,
    ) -> None:
        if not 1.0 <= healthy_factor < slow_factor:
            raise ValueError("require 1 <= healthy_factor < slow_factor")
        if persistence < 1 or min_slow_groups < 2 or max_candidates < 1:
            raise ValueError(
                "persistence and max_candidates must be positive; min_slow_groups >= 2"
            )
        self._metadata = dict(endpoint_metadata or {})
        self._slow_factor = slow_factor
        self._healthy_factor = healthy_factor
        self._persistence = persistence
        self._min_slow_groups = min_slow_groups
        self._max_candidates = max_candidates
        self._last_candidate: tuple[str, str, str] | None = None
        self._streak = 0

    def observe(
        self,
        observations: list[CollectiveObservation],
        compute_stragglers: frozenset[str] = frozenset(),
    ) -> CommunicationVerdict:
        """Correlate one round of process-group observations.

        Slow groups containing a known compute straggler are excluded conservatively:
        their latency already has a non-network explanation. Observations between the
        healthy and slow thresholds are neutral and neither implicate nor exonerate an
        endpoint.
        """

        slow: list[CollectiveObservation] = []
        healthy: list[CollectiveObservation] = []
        excluded: list[CollectiveObservation] = []
        for observation in observations:
            score = observation.normalized_latency
            if score >= self._slow_factor:
                if set(observation.members) & compute_stragglers:
                    excluded.append(observation)
                else:
                    slow.append(observation)
            elif score <= self._healthy_factor:
                healthy.append(observation)

        slow_ids = tuple(sorted({observation.group_id for observation in slow}))
        healthy_ids = tuple(sorted({observation.group_id for observation in healthy}))
        excluded_ids = tuple(sorted({observation.group_id for observation in excluded}))

        if not slow:
            self._reset_confirmation()
            detail = (
                "all slow groups were excluded by known compute stragglers"
                if excluded
                else "no slow communication groups"
            )
            return CommunicationVerdict(
                "normal",
                slow_groups=slow_ids,
                healthy_groups=healthy_ids,
                excluded_compute_groups=excluded_ids,
                detail=detail,
            )

        slow_by_class = self._by_resource_class(slow)
        healthy_by_class = self._by_resource_class(healthy)
        exact_candidates: list[EndpointCandidate] = []
        partial_candidates: list[EndpointCandidate] = []
        contradicted: set[str] = set()
        insufficient: list[str] = []

        for class_key, class_slow in slow_by_class.items():
            resource_class, resource_kind = class_key
            group_ids = {observation.group_id for observation in class_slow}
            resource_sets = [self._resources(observation) for observation in class_slow]
            incidence_patterns = {frozenset(resources) for resources in resource_sets}
            class_label = f"{resource_class}/{resource_kind}"

            if len(group_ids) < self._min_slow_groups:
                insufficient.append(
                    f"{class_label}: {len(group_ids)} slow group(s), need {self._min_slow_groups}"
                )
                continue
            if len(incidence_patterns) < 2:
                insufficient.append(f"{class_label}: slow groups lack independent incidence")
                continue

            class_healthy = healthy_by_class.get(class_key, [])
            healthy_resources: set[str] = set()
            for observation in class_healthy:
                healthy_resources.update(self._resources(observation))

            common = set.intersection(*resource_sets)
            contradicted.update(
                f"{resource_class}:{resource_kind}:{resource}"
                for resource in common & healthy_resources
            )
            common.difference_update(healthy_resources)

            if common:
                exact_candidates.extend(
                    self._candidate(
                        resource,
                        resource_class,
                        resource_kind,
                        class_slow,
                        resource_sets,
                    )
                    for resource in common
                )
                continue

            # No single endpoint explains every slow group. Rank resources by how
            # many independent slow groups they hit, while treating healthy incidence
            # as a hard contradiction.
            support: dict[str, set[str]] = defaultdict(set)
            for observation, resources in zip(class_slow, resource_sets):
                for resource in resources - healthy_resources:
                    support[resource].add(observation.group_id)
            ranked = sorted(support, key=lambda resource: (-len(support[resource]), resource))
            for resource in ranked:
                if len(support[resource]) < self._min_slow_groups:
                    continue
                partial_candidates.append(
                    self._candidate(
                        resource,
                        resource_class,
                        resource_kind,
                        class_slow,
                        resource_sets,
                    )
                )

        candidates = exact_candidates + partial_candidates
        candidates.sort(
            key=lambda candidate: (
                -candidate.support_ratio,
                candidate.resource_class,
                candidate.resource_kind,
                candidate.resource_id,
            )
        )
        candidates = candidates[: self._max_candidates]

        if len(exact_candidates) == 1 and not partial_candidates and not insufficient:
            candidate = exact_candidates[0]
            fingerprint = (
                candidate.resource_class,
                candidate.resource_kind,
                candidate.resource_id,
            )
            self._streak = self._streak + 1 if fingerprint == self._last_candidate else 1
            self._last_candidate = fingerprint
            confirmed = self._streak >= self._persistence
            return CommunicationVerdict(
                "localized",
                candidates=(candidate,),
                confirmed=confirmed,
                streak=self._streak,
                confidence=min(1.0, self._streak / self._persistence),
                slow_groups=slow_ids,
                healthy_groups=healthy_ids,
                excluded_compute_groups=excluded_ids,
                contradicted_resources=tuple(sorted(contradicted)),
                detail=(
                    f"{candidate.resource_class} {candidate.resource_kind} "
                    f"{candidate.resource_id} is shared by "
                    f"{len(candidate.supporting_groups)} independent slow groups and "
                    f"no healthy same-resource group "
                    f"({self._streak}/{self._persistence} rounds)"
                ),
            )

        self._reset_confirmation()
        if candidates:
            return CommunicationVerdict(
                "ambiguous",
                candidates=tuple(candidates),
                slow_groups=slow_ids,
                healthy_groups=healthy_ids,
                excluded_compute_groups=excluded_ids,
                contradicted_resources=tuple(sorted(contradicted)),
                detail="multiple or partial endpoint explanations; candidate list is not attribution",
            )

        reasons = "; ".join(insufficient)
        if contradicted:
            reasons = "; ".join(
                filter(None, [reasons, "common endpoint contradicted by healthy group"])
            )
        return CommunicationVerdict(
            "inconclusive",
            slow_groups=slow_ids,
            healthy_groups=healthy_ids,
            excluded_compute_groups=excluded_ids,
            contradicted_resources=tuple(sorted(contradicted)),
            detail=reasons or "no endpoint is supported by enough independent slow groups",
        )

    @staticmethod
    def _by_resource_class(
        observations: list[CollectiveObservation],
    ) -> dict[tuple[str, str], list[CollectiveObservation]]:
        grouped: dict[tuple[str, str], list[CollectiveObservation]] = defaultdict(list)
        for observation in observations:
            grouped[(observation.resource_class, observation.resource_kind)].append(observation)
        return grouped

    def _resources(self, observation: CollectiveObservation) -> set[str]:
        if observation.member_resources is not None:
            return {
                resource
                for index in range(len(observation.members))
                for resource in self._resources_for_member(observation, index)
            }
        if observation.resource_kind == "rank":
            return set(observation.members)

        resources = set()
        for rank in observation.members:
            metadata = self._metadata.get(rank)
            resource = (
                getattr(metadata, observation.resource_kind, None) if metadata is not None else None
            )
            if not resource:
                raise ValueError(f"missing {observation.resource_kind} metadata for rank {rank!r}")
            resources.add(resource)
        return resources

    def _candidate(
        self,
        resource: str,
        resource_class: str,
        resource_kind: str,
        observations: list[CollectiveObservation],
        resource_sets: list[set[str]],
    ) -> EndpointCandidate:
        supporting_groups = tuple(
            sorted(
                {
                    observation.group_id
                    for observation, resources in zip(observations, resource_sets)
                    if resource in resources
                }
            )
        )
        ranks = tuple(
            sorted(
                {
                    rank
                    for observation in observations
                    for index, rank in enumerate(observation.members)
                    if resource in self._resources_for_member(observation, index)
                }
            )
        )
        return EndpointCandidate(
            resource_class=resource_class,
            resource_kind=resource_kind,
            resource_id=resource,
            ranks=ranks,
            supporting_groups=supporting_groups,
            support_ratio=len(supporting_groups)
            / len({observation.group_id for observation in observations}),
        )

    def _resources_for_member(
        self, observation: CollectiveObservation, index: int
    ) -> tuple[str, ...]:
        if observation.member_resources is not None:
            resources = observation.member_resources[index]
            return (resources,) if isinstance(resources, str) else resources
        rank = observation.members[index]
        if observation.resource_kind == "rank":
            return (rank,)
        metadata = self._metadata.get(rank)
        resource = (
            getattr(metadata, observation.resource_kind, None) if metadata is not None else None
        )
        if not resource:
            raise ValueError(f"missing {observation.resource_kind} metadata for rank {rank!r}")
        return (resource,)

    def _reset_confirmation(self) -> None:
        self._last_candidate = None
        self._streak = 0
