"""Deterministic grouped-kernel expert-count validation domains."""

from __future__ import annotations

import random

ACTUAL_EXPERT_COUNTS = (1, 2, 4, 8, 16)
MAX_N_EXEC = 512


def scalar_n_exec_values(max_n_exec: int = MAX_N_EXEC) -> tuple[int, ...]:
    """Return the exhaustive scalar per-expert row-count domain."""
    if max_n_exec < 1:
        raise ValueError("max_n_exec must be positive")
    return tuple(range(1, max_n_exec + 1))


def heterogeneous_count_vectors(
    num_experts: int,
    *,
    max_n_exec: int = MAX_N_EXEC,
    num_sms: int = 108,
    random_samples: int = 128,
) -> tuple[tuple[int, ...], ...]:
    """Build boundary-focused and random nonuniform grouped-expert counts."""
    if num_experts < 1:
        raise ValueError("num_experts must be positive")
    if max_n_exec < 1:
        raise ValueError("max_n_exec must be positive")
    if num_sms < 1:
        raise ValueError("num_sms must be positive")
    if random_samples < 0:
        raise ValueError("random_samples cannot be negative")
    if num_experts == 1:
        return ()

    boundaries = {
        0,
        1,
        min(31, max_n_exec),
        min(32, max_n_exec),
        min(33, max_n_exec),
        min(63, max_n_exec),
        min(64, max_n_exec),
        min(65, max_n_exec),
        min(127, max_n_exec),
        min(128, max_n_exec),
        min(129, max_n_exec),
        min(255, max_n_exec),
        min(256, max_n_exec),
        min(257, max_n_exec),
        max_n_exec - 1,
        max_n_exec,
    }
    for work_per_expert, tile_rows in ((2, 64), (4, 32)):
        for pressure_multiple in (1, 2):
            tiles = max(1, (num_sms * pressure_multiple) // (num_experts * work_per_expert))
            transition = tiles * tile_rows
            boundaries.update((transition - 1, transition, transition + 1))
    ordered = tuple(sorted(value for value in boundaries if 0 <= value <= max_n_exec))

    vectors: set[tuple[int, ...]] = set()
    positions = {0, num_experts // 2, num_experts - 1}
    for value in ordered:
        for position in positions:
            one_active = [0] * num_experts
            one_active[position] = value
            vectors.add(tuple(one_active))

            one_distinct = [1] * num_experts
            one_distinct[position] = value
            vectors.add(tuple(one_distinct))

            one_small = [max_n_exec] * num_experts
            one_small[position] = value
            vectors.add(tuple(one_small))

        next_value = ordered[(ordered.index(value) + 1) % len(ordered)]
        vectors.add(
            tuple(value if expert % 2 == 0 else next_value for expert in range(num_experts))
        )

    for shift in range(min(num_experts, len(ordered))):
        vectors.add(
            tuple(ordered[(expert + shift) % len(ordered)] for expert in range(num_experts))
        )

    generator = random.Random(1729 + num_experts * 1009 + max_n_exec)
    for _ in range(random_samples):
        vectors.add(tuple(generator.randint(0, max_n_exec) for _expert in range(num_experts)))

    return tuple(sorted(vector for vector in vectors if any(vector) and len(set(vector)) > 1))
