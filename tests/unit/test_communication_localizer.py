"""Tests for overlapping-process-group communication endpoint inference."""

from __future__ import annotations

import pytest

from lm_resiliency.detection.communication_localizer import (
    CollectiveObservation,
    CommunicationLocalizer,
    EndpointMetadata,
)


def _o(
    group_id,
    members,
    latency=2.0,
    baseline=1.0,
    *,
    role,
    resource_class="inter_node",
    resource_kind="rank",
    collective="all_reduce",
    message_bytes=1024,
    member_resources=None,
):
    return CollectiveObservation(
        group_id=group_id,
        members=tuple(members),
        collective=collective,
        message_bytes=message_bytes,
        topology_role=role,
        resource_class=resource_class,
        latency_s=latency,
        baseline_latency_s=baseline,
        resource_kind=resource_kind,
        member_resources=member_resources,
    )


def test_overlapping_dp_and_ep_groups_localize_rank():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp-0", ["r0", "r4", "r8", "r12"], role="dp"),
        _o("ep-0", ["r0", "r1", "r2", "r3"], role="ep", collective="all_to_all"),
        _o("dp-1", ["r5", "r9", "r13", "r17"], latency=1.0, role="dp"),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.confirmed
    assert verdict.suspect is not None
    assert verdict.suspect.resource_id == "r0"
    assert verdict.suspect.supporting_groups == ("dp-0", "ep-0")


def test_cross_node_hsdp_shard_and_replica_groups_localize_rank():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o(
            "hsdp-shard-0",
            ["r0", "r1", "r2", "r3"],
            role="hsdp_shard",
            collective="reduce_scatter",
        ),
        _o(
            "hsdp-replicate-0",
            ["r0", "r4", "r8", "r12"],
            role="hsdp_replicate",
        ),
        _o(
            "hsdp-shard-1",
            ["r4", "r5", "r6", "r7"],
            latency=1.0,
            role="hsdp_shard",
            collective="reduce_scatter",
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.suspect is not None and verdict.suspect.resource_id == "r0"


def test_node_local_hsdp_shard_does_not_exonerate_or_localize_inter_node_endpoint():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o(
            "hsdp-replicate-0",
            ["r0", "r4", "r8", "r12"],
            role="hsdp_replicate",
        ),
        _o(
            "hsdp-shard-0",
            ["r0", "r1", "r2", "r3"],
            latency=1.0,
            role="hsdp_shard",
            resource_class="intra_node",
            collective="reduce_scatter",
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "inconclusive"
    assert verdict.candidates == ()


def test_rank_membership_aggregates_to_shared_nic():
    metadata = {
        "r0": EndpointMetadata(node="n0", hca="h0", nic="nic0"),
        "r1": EndpointMetadata(node="n0", hca="h0", nic="nic0"),
        "r2": EndpointMetadata(node="n1", hca="h1", nic="nic1"),
        "r3": EndpointMetadata(node="n2", hca="h2", nic="nic2"),
    }
    localizer = CommunicationLocalizer(metadata, persistence=1)
    observations = [
        _o("dp", ["r0", "r2"], role="dp", resource_kind="nic"),
        _o(
            "ep",
            ["r1", "r3"],
            role="ep",
            resource_kind="nic",
            collective="all_to_all",
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.suspect is not None
    assert verdict.suspect.resource_kind == "nic"
    assert verdict.suspect.resource_id == "nic0"
    assert verdict.suspect.ranks == ("r0", "r1")


def test_per_group_resource_map_handles_multi_rail_endpoints():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o(
            "dp",
            ["r0", "r2"],
            role="dp",
            resource_kind="nic",
            member_resources=("nic-a", "nic-c"),
        ),
        _o(
            "ep",
            ["r0", "r3"],
            role="ep",
            resource_kind="nic",
            collective="all_to_all",
            member_resources=("nic-a", "nic-d"),
        ),
        _o(
            "tp",
            ["r0", "r1"],
            latency=1.0,
            role="tp",
            resource_kind="nic",
            member_resources=("nic-b", "nic-b"),
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.suspect is not None and verdict.suspect.resource_id == "nic-a"


def test_one_member_can_use_multiple_rails_in_a_group():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o(
            "dp",
            ["r0", "r2"],
            role="dp",
            resource_kind="nic",
            member_resources=(("nic-a", "nic-b"), "nic-c"),
        ),
        _o(
            "ep",
            ["r0", "r3"],
            role="ep",
            resource_kind="nic",
            collective="all_to_all",
            member_resources=(("nic-a", "nic-d"), "nic-e"),
        ),
        _o(
            "cp",
            ["r0", "r4"],
            latency=1.0,
            role="cp",
            resource_kind="nic",
            collective="all_gather",
            member_resources=(("nic-b", "nic-d"), "nic-f"),
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.suspect is not None and verdict.suspect.resource_id == "nic-a"


def test_healthy_same_resource_group_contradicts_candidate():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp", ["r0", "r4"], role="dp"),
        _o("ep", ["r0", "r1"], role="ep", collective="all_to_all"),
        _o("cp", ["r0", "r7"], latency=1.0, role="cp", collective="all_gather"),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "inconclusive"
    assert verdict.candidates == ()
    assert verdict.contradicted_resources == ("inter_node:rank:r0",)


def test_healthy_different_resource_class_does_not_exonerate_nic():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp", ["r0", "r4"], role="dp"),
        _o("ep", ["r0", "r1"], role="ep", collective="all_to_all"),
        _o(
            "tp",
            ["r0", "r2"],
            latency=1.0,
            role="tp",
            resource_class="intra_node",
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "localized"
    assert verdict.suspect is not None and verdict.suspect.resource_id == "r0"


@pytest.mark.parametrize(
    "observations",
    [
        [_o("dp", ["r0", "r4"], role="dp")],
        [
            _o("dp", ["r0", "r4"], role="dp"),
            _o("ep", ["r0", "r4"], role="ep", collective="all_to_all"),
        ],
    ],
)
def test_requires_two_groups_with_independent_incidence(observations):
    verdict = CommunicationLocalizer(persistence=1).observe(observations)

    assert verdict.status == "inconclusive"
    assert not verdict.confirmed


def test_persistence_confirms_and_clean_round_resets():
    localizer = CommunicationLocalizer(persistence=2)
    slow = [
        _o("dp", ["r0", "r4"], role="dp"),
        _o("ep", ["r0", "r1"], role="ep", collective="all_to_all"),
    ]
    clean = [
        _o("dp", ["r0", "r4"], latency=1.0, role="dp"),
        _o("ep", ["r0", "r1"], latency=1.0, role="ep", collective="all_to_all"),
    ]

    assert localizer.observe(slow).confirmed is False
    assert localizer.observe(slow).confirmed is True
    assert localizer.observe(clean).status == "normal"
    assert localizer.observe(slow).streak == 1


def test_known_compute_straggler_excludes_affected_slow_group():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp", ["r0", "r4"], role="dp"),
        _o("ep", ["r0", "r1"], role="ep", collective="all_to_all"),
    ]

    verdict = localizer.observe(observations, compute_stragglers=frozenset({"r4"}))

    assert verdict.status == "inconclusive"
    assert verdict.excluded_compute_groups == ("dp",)
    assert verdict.slow_groups == ("ep",)


def test_multiple_fault_pattern_returns_weighted_candidates():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp-0", ["r0", "x0"], role="dp"),
        _o("ep-0", ["r0", "x1"], role="ep", collective="all_to_all"),
        _o("dp-1", ["r8", "x2"], role="dp"),
        _o("ep-1", ["r8", "x3"], role="ep", collective="all_to_all"),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "ambiguous"
    assert [candidate.resource_id for candidate in verdict.candidates] == ["r0", "r8"]
    assert all(candidate.support_ratio == 0.5 for candidate in verdict.candidates)
    assert not verdict.confirmed


def test_unresolved_second_resource_class_prevents_unique_attribution():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp", ["r0", "r4"], role="dp"),
        _o("ep", ["r0", "r1"], role="ep", collective="all_to_all"),
        _o(
            "tp",
            ["r8", "r9"],
            role="tp",
            resource_class="intra_node",
        ),
    ]

    verdict = localizer.observe(observations)

    assert verdict.status == "ambiguous"
    assert [candidate.resource_id for candidate in verdict.candidates] == ["r0"]
    assert not verdict.confirmed


def test_each_group_is_normalized_against_its_own_comparison_baseline():
    localizer = CommunicationLocalizer(persistence=1)
    observations = [
        _o("dp", ["r0", "r4"], latency=10.0, baseline=5.0, role="dp", message_bytes=4096),
        _o(
            "ep",
            ["r0", "r1"],
            latency=6.0,
            baseline=3.0,
            role="ep",
            collective="all_to_all",
            message_bytes=2048,
        ),
        _o("healthy", ["r2", "r3"], latency=6.0, baseline=6.0, role="dp"),
    ]

    verdict = localizer.observe(observations)

    assert observations[0].comparison_key == ("all_reduce", 4096, 2, "dp")
    assert verdict.status == "localized"
    assert verdict.suspect is not None and verdict.suspect.resource_id == "r0"


def test_missing_requested_physical_metadata_is_rejected():
    localizer = CommunicationLocalizer(
        {"r0": EndpointMetadata(nic="nic0")},
        persistence=1,
    )
    observations = [
        _o("dp", ["r0", "r1"], role="dp", resource_kind="nic"),
        _o("ep", ["r0", "r2"], role="ep", resource_kind="nic"),
    ]

    with pytest.raises(ValueError, match="missing nic metadata for rank 'r1'"):
        localizer.observe(observations)
