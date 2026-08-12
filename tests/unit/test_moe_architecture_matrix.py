"""Tests for architecture-oriented scalar MoE profiling domains."""

from __future__ import annotations

import pytest

from tests.support.moe_architecture_matrix import (
    ARCHITECTURE_PRESETS,
    MoEArchitecturePreset,
    per_expert_n_exec_values,
)


def test_architecture_presets_have_consistent_expert_parallelism():
    assert len(ARCHITECTURE_PRESETS) == 6
    for preset in ARCHITECTURE_PRESETS.values():
        assert preset.global_experts // preset.expert_parallel == preset.local_experts


def test_each_preset_profiles_the_same_exhaustive_scalar_range():
    preset = ARCHITECTURE_PRESETS["medium-4-local"]

    values = per_expert_n_exec_values(preset)

    assert values == tuple(range(1, 513))
    assert all(
        per_expert_n_exec_values(candidate) == values for candidate in ARCHITECTURE_PRESETS.values()
    )


def test_local_expert_count_is_metadata_not_a_recipe_dimension():
    values_by_count = {
        preset.local_experts: per_expert_n_exec_values(preset)
        for preset in ARCHITECTURE_PRESETS.values()
    }
    assert len(set(values_by_count.values())) == 1


def test_scalar_range_can_override_the_preset_default():
    preset = ARCHITECTURE_PRESETS["medium-4-local"]

    values = per_expert_n_exec_values(preset, min_n_exec=513, max_n_exec=1024)

    assert values == tuple(range(513, 1025))


def test_scalar_range_rejects_nonpositive_override():
    preset = ARCHITECTURE_PRESETS["medium-4-local"]

    with pytest.raises(ValueError, match="must be positive"):
        per_expert_n_exec_values(preset, max_n_exec=0)


def test_scalar_range_rejects_reversed_bounds():
    preset = ARCHITECTURE_PRESETS["medium-4-local"]

    with pytest.raises(ValueError, match="cannot exceed"):
        per_expert_n_exec_values(preset, min_n_exec=1025, max_n_exec=1024)


def test_architecture_preset_rejects_inconsistent_local_expert_count():
    with pytest.raises(ValueError, match="local experts"):
        MoEArchitecturePreset(
            name="invalid",
            local_experts=3,
            hidden=128,
            expert_output=128,
            max_n_exec=128,
            global_experts=8,
            top_k=2,
            expert_parallel=4,
            description="invalid",
        )
