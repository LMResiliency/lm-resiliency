"""Tests for policy-driven representative AllToAll replay."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from lm_resiliency.detection.all_to_all_replay import (
    AllToAllCapture,
    AllToAllReplayOutcome,
    AllToAllReplayPolicy,
    AllToAllTrafficMatrix,
    BalancedAndPermutationPolicy,
    _generate_policy_matrices,
)
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import LayerReplayDetector, ReplayResult
from lm_resiliency.detection.replay_harness import ModelReplayHarness


def _capture(*, sequence: int = 0) -> AllToAllCapture:
    return AllToAllCapture(
        sequence=sequence,
        collective="all_to_all_single",
        group_ranks=(0, 1, 2, 3),
        dtype=torch.float16,
        trailing_shape=(8,),
        element_size=2,
        observed_splits=(
            (5, 1, 0, 2),
            (0, 3, 4, 1),
            (2, 2, 2, 2),
            (1, 0, 3, 4),
        ),
    )


def test_default_policy_is_bounded_deterministic_and_replica_invariant():
    policy = BalancedAndPermutationPolicy(max_payload_bytes_per_rank=256)

    first = list(policy.generate(_capture()))
    second = list(
        policy.generate(
            replace(
                _capture(),
                observed_splits=(
                    (8, 0, 0, 0),
                    (0, 8, 0, 0),
                    (0, 0, 8, 0),
                    (0, 0, 0, 8),
                ),
            )
        )
    )

    assert first == second
    assert [matrix.name for matrix in first] == [
        "balanced",
        "cyclic_permutation_1",
    ]
    for matrix in first:
        assert all(sum(row) == 16 for row in matrix.splits)
        assert all(sum(row[destination] for row in matrix.splits) == 16 for destination in range(4))


def test_default_policy_rotates_sparse_route_by_sequence():
    policy = BalancedAndPermutationPolicy(max_payload_bytes_per_rank=256)

    matrices = list(policy.generate(_capture(sequence=2)))

    assert matrices[1].name == "cyclic_permutation_3"
    assert matrices[1].splits[0] == (0, 0, 0, 16)
    assert matrices[1].splits[1] == (16, 0, 0, 0)


def test_custom_policy_can_replace_the_default_matrix_selection():
    class DiagonalPolicy(AllToAllReplayPolicy):
        def generate(self, capture):
            return [
                AllToAllTrafficMatrix(
                    "diagonal",
                    tuple(
                        tuple(
                            2 if source == destination else 0
                            for destination in range(capture.group_size)
                        )
                        for source in range(capture.group_size)
                    ),
                )
            ]

    (matrix,) = DiagonalPolicy().generate(_capture())

    assert matrix.name == "diagonal"
    assert matrix.splits[2] == (0, 0, 2, 0)


def test_expected_boolean_requires_a_healthy_majority():
    detector = object.__new__(LayerReplayDetector)
    detector._c3 = MagicMock()
    detector._c3.run_scalar.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [0, 0, 0],
    )

    result = detector.compare_expected_boolean(False)

    assert result.status is C3Status.INCONCLUSIVE
    assert result.bitmap == [0, 0, 0]


def test_executor_rejects_rank_dependent_policy_matrices_before_collective():
    policy = BalancedAndPermutationPolicy(max_payload_bytes_per_rank=256)

    def gather(output, local, group):
        assert group == "ep"
        output[:] = [local] * 4
        changed = list(output[2][2])
        changed[0] = ("different", changed[0][1])
        output[2] = (True, "", tuple(changed))

    with (
        patch(
            "lm_resiliency.detection.all_to_all_replay.dist.all_gather_object",
            side_effect=gather,
        ),
        pytest.raises(RuntimeError, match="different matrices"),
    ):
        _generate_policy_matrices(_capture(), policy, group="ep")


def test_harness_attaches_correctness_and_timing_evidence():
    agree = C3Result(C3Status.AGREE, [0, 0, 0], [1, 1, 1])
    detector = MagicMock()
    detector.compare_structure.return_value = agree
    detector.compare_expected_boolean.return_value = agree
    policy = BalancedAndPermutationPolicy(max_payload_bytes_per_rank=256)
    matrix = AllToAllTrafficMatrix(
        "balanced",
        ((1, 1), (1, 1)),
    )
    outcome = AllToAllReplayOutcome(
        matrix=matrix,
        sequence=0,
        group_ranks=(0, 1),
        dtype=torch.float32,
        trailing_shape=(4,),
        latency_ms=1.5,
        input_bytes=32,
        output_bytes=32,
        correct=True,
    )
    harness = object.__new__(ModelReplayHarness)
    harness._config = SimpleNamespace(all_to_all_policy=policy)
    harness._all_to_all_replay_recipes = (object(),)
    harness._all_to_all_executor = MagicMock()
    harness._all_to_all_executor.replay.return_value = [outcome]
    harness._hang_instrumentation = None
    harness._get_detector = lambda: detector
    result = ReplayResult(
        sdc_bitmap=[0, 0, 0],
        straggler_bitmap=[0, 0, 0],
        replay_time_ms=1.0,
        layer_id=0,
        peer_ranks=[0, 1, 2],
        spatial_straggler_bitmap=[0, 0, 0],
    )

    harness._attach_all_to_all_replay(result)
    harness._attach_all_to_all_replay(result)

    assert result.checked_recipe_ids == ["all_to_all"]
    assert {
        "all_to_all.recipe_count",
        "all_to_all.0.execution",
        "all_to_all.0.matrix_count",
        "all_to_all.0.0.contract",
        "all_to_all.0.0.output",
    } <= result.c3_results.keys()
    harness._all_to_all_executor.replay.assert_called_once()
    detector.add_communication_timing.assert_called_once_with(
        result,
        name="all_to_all_replay.balanced",
        elapsed_ms=1.5,
        group_ranks=(0, 1),
        topology_role="ep",
        message_bytes=32,
        sequence=0,
    )
