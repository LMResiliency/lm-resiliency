"""Tests for grouped-kernel expert-count validation domains."""

from __future__ import annotations

import pytest

from tests.support.grouped_kernel_expert_matrix import (
    ACTUAL_EXPERT_COUNTS,
    heterogeneous_count_vectors,
    scalar_n_exec_values,
)


def test_scalar_domain_is_independent_of_actual_expert_count():
    domains = {experts: scalar_n_exec_values() for experts in ACTUAL_EXPERT_COUNTS}

    assert len(set(domains.values())) == 1
    assert next(iter(domains.values())) == tuple(range(1, 513))


@pytest.mark.parametrize("num_experts", (2, 4, 8, 16))
def test_heterogeneous_vectors_are_deterministic_and_valid(num_experts):
    first = heterogeneous_count_vectors(num_experts)
    second = heterogeneous_count_vectors(num_experts)

    assert first == second
    assert len(first) >= 128
    assert all(len(vector) == num_experts for vector in first)
    assert all(0 <= count <= 512 for vector in first for count in vector)
    assert all(any(vector) and len(set(vector)) > 1 for vector in first)


def test_single_expert_has_no_heterogeneous_vectors():
    assert heterogeneous_count_vectors(1) == ()


@pytest.mark.parametrize(
    ("kwargs", "message"),
    (
        ({"num_experts": 0}, "num_experts"),
        ({"num_experts": 2, "max_n_exec": 0}, "max_n_exec"),
        ({"num_experts": 2, "num_sms": 0}, "num_sms"),
        ({"num_experts": 2, "random_samples": -1}, "random_samples"),
    ),
)
def test_heterogeneous_vector_domain_rejects_invalid_arguments(kwargs, message):
    with pytest.raises(ValueError, match=message):
        heterogeneous_count_vectors(**kwargs)
