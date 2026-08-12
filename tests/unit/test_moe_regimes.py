"""Tests for offline MoE execution-regime discovery and scheduling."""

from __future__ import annotations

import json
import subprocess
import sysconfig
from dataclasses import replace
from pathlib import Path

import pytest

from lm_resiliency.detection.moe_regimes import (
    CatalogEnvironmentMismatch,
    ConflictingObservationError,
    CTASemantics,
    ExecutionFingerprint,
    ExecutionObservation,
    KernelLaunch,
    MoEExecutionEnvironment,
    MoERegimeCatalog,
    MoEReplayScheduler,
    ProfileLocation,
    ProfileRequest,
    RegimeDiscoveryError,
    build_profile_requests,
    current_moe_environment,
    discover_execution_regimes,
    kernel_launches_from_kineto_trace,
    load_observations,
    load_profile_requests,
    profile_requests,
    save_observations,
    save_profile_requests,
)

_DERIVATION_DIGEST = "a" * 64
_QUALIFICATION_DIGEST = "b" * 64


def _environment(**updates):
    values = {
        "backend": {
            "name": "transformer-engine-grouped-gemm",
            "version": "2.3.0",
        },
        "container_digest": "sha256:test-container",
        "cublas": "12.8.4",
        "cuda_build": "12.8",
        "cuda_graphs": True,
        "driver": "570.86.15",
        "gpu_capability": [9, 0],
        "gpu_model": "H100 SXM",
        "model": {
            "expert_hidden": 7168,
            "expert_intermediate": 2048,
            "alignment": 128,
        },
        "model_commit": "0123456789abcdef",
        "nccl": "2.25.1",
        "overlap": "expert-only",
        "parallelism": {"ep": 8, "tp": 1, "dp": 2},
        "precision": "fp8",
        "precision_recipe": "delayed-scaling",
        "sm_count": 132,
        "torch": "2.7.0+cu128",
        "workspace_policy": "fixed-4MiB",
    }
    values.update(updates)
    return MoEExecutionEnvironment.from_mapping(values)


def _fingerprint(
    kernel="gemm_128x128",
    *,
    grid=(132, 1, 1),
    tail="aligned",
    workspace=4096,
    pressure="full-sm",
    persistent_work=0,
    roles=("gemm-tile",),
    cta_mapping="tile-grid",
    cta_role_counts=None,
    qualification_role_counts=None,
    derivation_source="generated_metadata",
    derivation_digest=_DERIVATION_DIGEST,
    qualification_source="instrumented",
    qualification_digest=_QUALIFICATION_DIGEST,
):
    work_items = persistent_work or grid[0] * grid[1] * grid[2]
    role_counts = (
        _partition_role_counts(work_items, roles) if cta_role_counts is None else cta_role_counts
    )
    qualified_counts = (
        role_counts if qualification_role_counts is None else qualification_role_counts
    )
    return ExecutionFingerprint.create(
        kernels=[
            KernelLaunch(
                kernel,
                grid=grid,
                block=(256, 1, 1),
                shared_memory_bytes=65536,
                registers_per_thread=128,
            )
        ],
        algorithm_ids=["cublaslt-17"],
        tile_shapes=["128x128x64"],
        tail_path=tail,
        workspace_bytes=workspace,
        pressure_class=pressure,
        overlap_class="expert-only",
        persistent_work_items=(persistent_work,),
        cta_semantics=(
            _cta_semantics(
                work_items,
                mapping_class=cta_mapping,
                role_counts=role_counts,
                qualification_role_counts=qualified_counts,
                derivation_source=derivation_source,
                derivation_digest=derivation_digest,
                qualification_source=qualification_source,
                qualification_digest=qualification_digest,
            ),
        ),
    )


def _partition_role_counts(work_items, roles):
    labels = tuple(sorted(set(roles)))
    if not labels or work_items < len(labels):
        raise ValueError("semantic roles must form a non-empty partition")
    counts = {role: 1 for role in labels}
    repeated_role = "interior" if "interior" in counts else labels[0]
    counts[repeated_role] += work_items - len(labels)
    return counts


def _cta_semantics(
    work_items,
    *,
    mapping_class="tile-grid",
    role_counts=None,
    qualification_role_counts=None,
    derivation_source="generated_metadata",
    derivation_digest=_DERIVATION_DIGEST,
    qualification_source="instrumented",
    qualification_digest=_QUALIFICATION_DIGEST,
):
    declared = {"gemm-tile": work_items} if role_counts is None else role_counts
    qualified = declared if qualification_role_counts is None else qualification_role_counts
    return CTASemantics.create(
        mapping_class=mapping_class,
        role_counts=declared,
        qualification_role_counts=qualified,
        derivation_source=derivation_source,
        derivation_digest=derivation_digest,
        qualification_source=qualification_source,
        qualification_digest=qualification_digest,
    )


def _observation(
    n_exec,
    fingerprint,
    *,
    layer=0,
    expert=0,
    ep_position=2,
    execution_class="default",
):
    return ExecutionObservation(
        request=ProfileRequest(
            n_exec=n_exec,
            location=ProfileLocation(
                layer_id=str(layer),
                expert_id=expert,
                ep_position=ep_position,
                execution_class=execution_class,
            ),
        ),
        fingerprint=fingerprint,
    )


def _discover(observations, **kwargs):
    repeated = tuple(observation for observation in observations for _ in range(3))
    return discover_execution_regimes(repeated, **kwargs)


def test_environment_identity_is_canonical_and_detects_staleness():
    first = MoEExecutionEnvironment.from_mapping({"cuda": "12.8", "gpu": "H100"})
    second = MoEExecutionEnvironment.from_mapping({"gpu": "H100", "cuda": "12.8"})
    assert first.identifier == second.identifier
    assert first.difference(second) == {}

    changed = MoEExecutionEnvironment.from_mapping({"gpu": "H100", "cuda": "12.9"})
    assert first.difference(changed) == {"cuda": ("12.8", "12.9")}


def test_current_environment_records_backend_model_and_cpu_fallback(monkeypatch):
    monkeypatch.setattr("torch.cuda.is_available", lambda: False)
    environment = current_moe_environment(
        backend="grouped-gemm",
        backend_version="1.2.3",
        model={"hidden": 4096, "intermediate": 14336},
        precision="bf16",
        parallelism={"ep": 8},
    ).to_dict()

    assert environment["backend"] == {"name": "grouped-gemm", "version": "1.2.3"}
    assert environment["model"]["hidden"] == 4096
    assert environment["gpu_model"] == "cpu-only-discovery"


def test_discovery_groups_only_identical_physical_fingerprints():
    aligned = _fingerprint()
    tail = _fingerprint(kernel="gemm_128x128_tail", grid=(133, 1, 1), tail="tail-1")
    observations = [
        _observation(128, aligned, layer=0, expert=0),
        _observation(128, aligned, layer=4, expert=3),
        _observation(256, aligned, layer=9, expert=1),
        _observation(257, tail, layer=2, expert=2),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2
    assert catalog.ep_position == 2
    assert catalog.cycle_size == 2
    aligned_regime = next(regime for regime in catalog.regimes if regime.fingerprint == aligned)
    assert aligned_regime.n_exec_values == (128, 256)
    assert aligned_regime.representatives == (256,)
    assert len(aligned_regime.locations) == 3
    assert catalog.missing_shapes([128, 256, 257]) == ()
    assert catalog.missing_shapes([127, 128]) == (127,)


def test_multiple_representatives_cover_regime_boundaries_and_interior():
    fingerprint = _fingerprint()
    observations = [_observation(value, fingerprint) for value in [64, 128, 192, 256, 320]]

    catalog = _discover(
        observations,
        environment=_environment(),
        representatives_per_regime=3,
    )

    assert catalog.regimes[0].representatives == (64, 192, 320)
    assert catalog.cycle_size == 3


def test_plan_and_pressure_groups_scalable_cta_counts():
    observations = [
        _observation(128, _fingerprint(grid=(132, 1, 1))),
        _observation(256, _fingerprint(grid=(264, 1, 1))),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].n_exec_values == (128, 256)
    assert catalog.regimes[0].representatives == (256,)
    assert catalog.cycle_size == 1

    exact = _discover(
        observations,
        environment=_environment(),
        equivalence_policy="exact_launch",
    )
    assert len(exact.regimes) == 2


def test_cta_count_pressure_transition_splits_regimes():
    observations = [
        _observation(
            128,
            _fingerprint(grid=(64, 1, 1), pressure="underfilled"),
        ),
        _observation(
            256,
            _fingerprint(grid=(264, 1, 1), pressure="saturated"),
        ),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2
    assert catalog.cycle_size == 2


def test_thousands_of_cta_counts_compress_to_one_role_covering_representative():
    observations = [
        _observation(
            cta_count * 128,
            _fingerprint(grid=(cta_count, 1, 1)),
        )
        for cta_count in range(1, 2049)
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (2048 * 128,)
    assert catalog.cycle_size == 1


def test_non_nested_cta_grids_do_not_inflate_catalog_when_mapping_is_same():
    observations = [
        _observation(128, _fingerprint(grid=(8, 1, 1))),
        _observation(256, _fingerprint(grid=(4, 2, 1))),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (256,)
    assert catalog.cycle_size == 1
    assert replace(catalog.regimes[0], representatives=(128,)).representatives == (128,)


def test_different_cta_mapping_classes_split_regimes():
    observations = [
        _observation(
            128,
            _fingerprint(grid=(8, 1, 1), cta_mapping="x=output-tile"),
        ),
        _observation(
            256,
            _fingerprint(grid=(4, 2, 1), cta_mapping="x=expert,y=output-tile"),
        ),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2
    assert catalog.cycle_size == 2


def test_per_role_counts_select_a_dominating_representative():
    observations = [
        _observation(128, _fingerprint(grid=(8, 1, 1), roles=("interior",))),
        _observation(
            256,
            _fingerprint(grid=(16, 1, 1), roles=("boundary", "interior")),
        ),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (256,)
    assert all(
        catalog.regimes[0].representatives_cover_shape(n_exec)
        for n_exec in catalog.regimes[0].n_exec_values
    )


def _replay_signature_with_role_fault(
    independent_role_counts,
    kernel_count,
    kernel_index,
    injected_role,
    injected_occurrence=1,
):
    """Inject corruption from an oracle independent of adapter declarations."""
    signature = [0] * kernel_count
    if independent_role_counts.get(injected_role, 0) >= injected_occurrence:
        signature[kernel_index] ^= 1
    return tuple(signature)


def test_role_specific_counts_prevent_false_total_cta_dominance():
    declared_role_counts = {
        128: {"boundary": 4, "interior": 4},
        256: {"boundary": 1, "interior": 15},
    }
    qualification_role_counts = {
        128: {"boundary": 4, "interior": 4},
        256: {"boundary": 1, "interior": 15},
    }
    independent_fault_oracle = {
        128: {"boundary": 4, "interior": 4},
        256: {"boundary": 1, "interior": 15},
    }
    observations = [
        _observation(
            128,
            _fingerprint(
                grid=(8, 1, 1),
                cta_role_counts=declared_role_counts[128],
                qualification_role_counts=qualification_role_counts[128],
            ),
        ),
        _observation(
            256,
            _fingerprint(
                grid=(16, 1, 1),
                cta_role_counts=declared_role_counts[256],
                qualification_role_counts=qualification_role_counts[256],
            ),
        ),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (128, 256)
    assert catalog.cycle_size == 2
    assert _replay_signature_with_role_fault(
        independent_fault_oracle[256],
        1,
        0,
        "boundary",
        4,
    ) == (0,)
    assert _replay_signature_with_role_fault(
        independent_fault_oracle[128],
        1,
        0,
        "boundary",
        4,
    ) == (1,)


def test_persistent_work_count_scales_within_one_regime():
    observations = [
        _observation(128, _fingerprint(grid=(108, 1, 1), persistent_work=128)),
        _observation(256, _fingerprint(grid=(108, 1, 1), persistent_work=256)),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (256,)
    assert catalog.cycle_size == 1


def test_direct_and_persistent_scheduling_are_separate_regimes():
    observations = [
        _observation(128, _fingerprint(grid=(108, 1, 1), persistent_work=0)),
        _observation(256, _fingerprint(grid=(108, 1, 1), persistent_work=256)),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2


def test_role_fault_injections_are_covered_by_selected_representatives():
    declared_role_counts = {
        64: {"interior": 7, "x-edge": 1},
        128: {"interior": 7, "y-edge": 1},
        256: {"boundary": 1, "interior": 14, "x-edge": 1},
    }
    qualification_role_counts = {
        64: {"interior": 7, "x-edge": 1},
        128: {"interior": 7, "y-edge": 1},
        256: {"boundary": 1, "interior": 14, "x-edge": 1},
    }
    independent_fault_oracle = {
        64: {"interior": 7, "x-edge": 1},
        128: {"interior": 7, "y-edge": 1},
        256: {"boundary": 1, "interior": 14, "x-edge": 1},
    }
    observations = [
        _observation(
            64,
            _fingerprint(
                grid=(4, 2, 1),
                cta_role_counts=declared_role_counts[64],
                qualification_role_counts=qualification_role_counts[64],
            ),
        ),
        _observation(
            128,
            _fingerprint(
                grid=(2, 4, 1),
                cta_role_counts=declared_role_counts[128],
                qualification_role_counts=qualification_role_counts[128],
            ),
        ),
        _observation(
            256,
            _fingerprint(
                grid=(8, 2, 1),
                cta_role_counts=declared_role_counts[256],
                qualification_role_counts=qualification_role_counts[256],
            ),
        ),
    ]

    regime = _discover(observations, environment=_environment()).regimes[0]

    assert regime.representatives == (128, 256)
    for n_exec in regime.n_exec_values:
        target = regime.fingerprint_for_shape(n_exec)
        for kernel_index in range(len(target.kernels)):
            for injected_role, injected_occurrence in independent_fault_oracle[n_exec].items():
                assert any(
                    _replay_signature_with_role_fault(
                        independent_fault_oracle[representative],
                        len(target.kernels),
                        kernel_index,
                        injected_role,
                        injected_occurrence,
                    )
                    != (0,) * len(target.kernels)
                    for representative in regime.representatives
                )

    assert all(
        _replay_signature_with_role_fault(
            independent_fault_oracle[representative],
            1,
            0,
            "unmodeled-role",
        )
        == (0,)
        for representative in regime.representatives
    )

    with pytest.raises(ValueError, match="do not cover per-role work counts"):
        replace(regime, representatives=(128,))


def test_independent_qualification_catches_omitted_role():
    valid = _fingerprint(grid=(8, 1, 1), roles=("interior",))
    omitted_boundary = _fingerprint(
        grid=(16, 1, 1),
        cta_role_counts={"interior": 16},
        qualification_role_counts={"boundary": 1, "interior": 15},
    )

    catalog = _discover(
        [
            _observation(128, valid),
            _observation(256, omitted_boundary),
        ],
        environment=_environment(),
    )

    assert len(catalog.regimes) == 2
    semantics = omitted_boundary.cta_semantics[0]
    assert "boundary" not in dict(semantics.role_counts)
    assert "boundary" in dict(semantics.qualification_role_counts)


@pytest.mark.parametrize(
    "updates",
    [
        {"derivation_source": "adapter_guess"},
        {"qualification_source": "generated_metadata"},
        {"qualification_digest": _DERIVATION_DIGEST},
    ],
)
def test_semantic_evidence_must_be_independent(updates):
    observations = [
        _observation(128, _fingerprint(grid=(8, 1, 1), **updates)),
        _observation(256, _fingerprint(grid=(16, 1, 1), **updates)),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2


def test_semantic_evidence_provenance_does_not_split_execution_regimes():
    first = _fingerprint(grid=(8, 1, 1))
    second = _fingerprint(
        grid=(16, 1, 1),
        derivation_source="backend_api",
        derivation_digest="c" * 64,
        qualification_source="source_audit",
        qualification_digest="d" * 64,
    )

    catalog = _discover(
        [
            _observation(128, first),
            _observation(256, second),
        ],
        environment=_environment(),
    )

    assert first.identifier != second.identifier
    assert len(catalog.regimes) == 1
    assert catalog.regimes[0].representatives == (256,)


def test_underfilled_grid_does_not_inflate_cycle_and_allows_sm_qualification():
    catalog = _discover(
        [_observation(128, _fingerprint(grid=(2, 1, 1)))],
        environment=_environment(sm_count=4),
    )
    regime = catalog.regimes[0]
    recipes = catalog.replay_recipes

    assert len(recipes) == 1
    scheduler = MoEReplayScheduler(catalog, replay_interval=5)
    assert scheduler.detection_bound_steps == 5
    assert scheduler.next_recipe().recipe == recipes[0]
    assert scheduler.completed_cycles == 1

    coverage_key = (regime.regime_id, regime.representatives[0], 0)
    with pytest.raises(RegimeDiscoveryError, match="SM coverage is incomplete"):
        catalog.validate_representative_sm_coverage({coverage_key: {0, 1, 2}})
    catalog.validate_representative_sm_coverage({coverage_key: {0, 1, 2, 3}})


@pytest.mark.parametrize(
    "missing_field",
    [
        "kernel",
        "algorithm",
        "tile",
        "tail",
        "workspace",
        "pressure",
        "overlap",
        "persistent_work",
        "cta_semantics",
        "cta_role_counts",
        "cta_partition_total",
        "cta_qualification",
        "cta_mapping",
        "cta_derivation",
        "cta_qualification_source",
        "grid",
        "block",
    ],
)
def test_incomplete_equivalence_metadata_falls_back_to_exact_launches(missing_field):
    def incomplete_fingerprint(grid, *, remove_grid=False):
        work_items = grid[0] * grid[1] * grid[2]
        role_counts = None
        if missing_field == "cta_role_counts":
            role_counts = {}
        elif missing_field == "cta_partition_total":
            role_counts = {"gemm-tile": work_items - 1}
        semantics = _cta_semantics(
            work_items,
            mapping_class="unknown" if missing_field == "cta_mapping" else "tile-grid",
            role_counts=role_counts,
            qualification_role_counts={} if missing_field == "cta_qualification" else None,
            derivation_source=(
                "unknown" if missing_field == "cta_derivation" else "generated_metadata"
            ),
            qualification_source=(
                "unknown" if missing_field == "cta_qualification_source" else "instrumented"
            ),
        )
        return ExecutionFingerprint.create(
            kernels=[
                KernelLaunch(
                    "" if missing_field == "kernel" else "gemm",
                    grid=None if missing_field == "grid" and remove_grid else grid,
                    block=None if missing_field == "block" else (256, 1, 1),
                )
            ],
            algorithm_ids=() if missing_field == "algorithm" else ("algo-1",),
            tile_shapes=() if missing_field == "tile" else ("128x128x64",),
            tail_path="unknown" if missing_field == "tail" else "aligned",
            workspace_bytes=None if missing_field == "workspace" else 4096,
            pressure_class="unknown" if missing_field == "pressure" else "full-sm",
            overlap_class="unknown" if missing_field == "overlap" else "expert-only",
            persistent_work_items=() if missing_field == "persistent_work" else (0,),
            cta_semantics=() if missing_field == "cta_semantics" else (semantics,),
        )

    observations = [
        _observation(128, incomplete_fingerprint((132, 1, 1), remove_grid=True)),
        _observation(256, incomplete_fingerprint((132, 1, 1))),
    ]

    catalog = _discover(observations, environment=_environment())

    assert len(catalog.regimes) == 2


@pytest.mark.parametrize(
    ("missing_field", "missing_value"),
    [
        ("algorithm", ("",)),
        ("algorithm", (None,)),
        ("tile", ("unknown",)),
        ("tail", ""),
        ("tail", None),
        ("pressure", "UNKNOWN"),
        ("overlap", "unset"),
        ("overlap", None),
        ("cta_mapping", "unknown"),
        ("cta_derivation_digest", "unknown"),
        ("cta_qualification_digest", "unknown"),
    ],
)
def test_placeholder_equivalence_metadata_falls_back_to_exact_launches(
    missing_field, missing_value
):
    def incomplete_fingerprint(grid):
        semantics_values = {
            "mapping_class": "tile-grid",
            "role_counts": {"gemm-tile": 132},
            "qualification_role_counts": {"gemm-tile": 132},
            "derivation_source": "generated_metadata",
            "derivation_digest": _DERIVATION_DIGEST,
            "qualification_source": "instrumented",
            "qualification_digest": _QUALIFICATION_DIGEST,
        }
        semantic_field = {
            "cta_mapping": "mapping_class",
            "cta_derivation_digest": "derivation_digest",
            "cta_qualification_digest": "qualification_digest",
        }.get(missing_field)
        if semantic_field is not None:
            semantics_values[semantic_field] = missing_value
        values = {
            "algorithm_ids": ("algo-1",),
            "tile_shapes": ("128x128x64",),
            "tail_path": "aligned",
            "workspace_bytes": 4096,
            "pressure_class": "full-sm",
            "overlap_class": "expert-only",
            "persistent_work_items": (0,),
            "cta_semantics": (CTASemantics.create(**semantics_values),),
        }
        field_name = {
            "algorithm": "algorithm_ids",
            "tile": "tile_shapes",
            "tail": "tail_path",
            "pressure": "pressure_class",
            "overlap": "overlap_class",
        }.get(missing_field)
        if field_name is not None:
            values[field_name] = missing_value
        return ExecutionFingerprint.create(
            kernels=[KernelLaunch("gemm", grid=grid, block=(256, 1, 1))],
            **values,
        )

    catalog = _discover(
        [
            _observation(128, incomplete_fingerprint((132, 1, 1))),
            _observation(256, incomplete_fingerprint((132, 1, 1))),
        ],
        environment=_environment(),
    )

    assert len(catalog.regimes) == 2


def test_placeholder_custom_metadata_falls_back_to_exact_launches():
    def fingerprint(grid):
        return ExecutionFingerprint.create(
            kernels=[KernelLaunch("gemm", grid=grid, block=(256, 1, 1))],
            algorithm_ids=("algo-1",),
            tile_shapes=("128x128x64",),
            tail_path="aligned",
            workspace_bytes=4096,
            pressure_class="full-sm",
            overlap_class="expert-only",
            persistent_work_items=(0,),
            cta_semantics=(_cta_semantics(grid[0] * grid[1] * grid[2]),),
            extra={"backend_hint": None},
        )

    catalog = _discover(
        [
            _observation(128, fingerprint((132, 1, 1))),
            _observation(256, fingerprint((132, 1, 1))),
        ],
        environment=_environment(),
    )

    assert len(catalog.regimes) == 2


def test_conflicting_homogeneous_shape_is_rejected():
    observations = [
        _observation(128, _fingerprint(), layer=0),
        _observation(128, _fingerprint(tail="unexpected-tail"), layer=1),
    ]

    with pytest.raises(ConflictingObservationError, match="conflicting"):
        _discover(observations, environment=_environment())


def test_explicit_execution_classes_preserve_heterogeneous_layers():
    observations = [
        _observation(128, _fingerprint(), execution_class="standard"),
        _observation(
            128,
            _fingerprint(kernel="large-expert"),
            execution_class="large-expert",
        ),
    ]

    catalog = _discover(observations, environment=_environment())

    assert {regime.execution_class for regime in catalog.regimes} == {
        "standard",
        "large-expert",
    }
    assert len(catalog.regimes_for_shape(128, execution_class="standard")) == 1


def test_catalog_requires_one_ep_position():
    observations = [
        _observation(128, _fingerprint(), ep_position=0),
        _observation(256, _fingerprint(), ep_position=1),
    ]
    with pytest.raises(RegimeDiscoveryError, match="separate"):
        _discover(observations, environment=_environment())


def test_discovery_proves_expected_layer_expert_shape_coverage():
    observed = _observation(128, _fingerprint(), layer=0, expert=0)
    missing = ProfileRequest(
        128,
        ProfileLocation(layer_id="7", expert_id=3, ep_position=2),
    )

    with pytest.raises(RegimeDiscoveryError, match="missed 1 required"):
        _discover(
            [observed],
            environment=_environment(),
            expected_requests=[observed.request, missing],
        )

    catalog = _discover([observed], environment=_environment())
    assert catalog.missing_profile_requests([observed.request, missing]) == (missing,)
    assert catalog.observation_count(observed.request) == 3


def test_discovery_requires_three_observations_per_request_by_default():
    observed = _observation(128, _fingerprint())

    with pytest.raises(RegimeDiscoveryError, match="fewer than 3 observations"):
        discover_execution_regimes([observed], environment=_environment())

    catalog = discover_execution_regimes([observed] * 3, environment=_environment())
    assert catalog.observation_count(observed.request) == 3


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("driver", None),
        ("model_commit", "<model commit>"),
        ("precision_recipe", "unknown"),
        ("extra", {"custom_execution_flag": None}),
    ],
)
def test_catalog_rejects_incomplete_or_placeholder_environment(field, value):
    attributes = _environment().to_dict()
    if value is None:
        attributes.pop(field)
    else:
        attributes[field] = value

    with pytest.raises(RegimeDiscoveryError, match=field):
        _discover(
            [_observation(128, _fingerprint())],
            environment=MoEExecutionEnvironment.from_mapping(attributes),
        )


def test_catalog_accepts_adapter_environment_fields_under_extra():
    attributes = _environment().to_dict()
    adapter_fields = {
        name: attributes.pop(name)
        for name in (
            "container_digest",
            "cublas",
            "cuda_graphs",
            "model_commit",
            "overlap",
            "precision_recipe",
            "workspace_policy",
        )
    }
    attributes["extra"] = adapter_fields

    catalog = _discover(
        [_observation(128, _fingerprint())],
        environment=MoEExecutionEnvironment.from_mapping(attributes),
    )

    assert catalog.environment.to_dict()["extra"] == adapter_fields


def test_catalog_and_raw_observation_round_trip(tmp_path):
    observations = (
        _observation(128, _fingerprint()),
        _observation(129, _fingerprint(tail="tail-1")),
    )
    observations_path = tmp_path / "observations.jsonl"
    save_observations(observations, observations_path)
    assert load_observations(observations_path) == observations

    catalog = _discover(observations, environment=_environment())
    catalog_path = tmp_path / "catalog.json"
    catalog.save(catalog_path)
    loaded = MoERegimeCatalog.load(catalog_path)
    assert loaded == catalog
    assert loaded.identifier == catalog.identifier
    assert loaded.observation_count(observations[0].request) == 3

    payload = json.loads(catalog_path.read_text())
    payload["environment"]["cuda"] = "changed-without-checksum-update"
    catalog_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum"):
        MoERegimeCatalog.load(catalog_path)


def test_catalog_refuses_a_changed_training_environment():
    catalog = _discover([_observation(128, _fingerprint())], environment=_environment())
    catalog.validate_environment(_environment())
    with pytest.raises(CatalogEnvironmentMismatch, match="precision"):
        catalog.validate_environment(_environment(precision="bf16"))


def test_discovery_cli_builds_a_catalog(tmp_path):
    observations_path = tmp_path / "observations.jsonl"
    environment_path = tmp_path / "environment.json"
    manifest_path = tmp_path / "manifest.json"
    catalog_path = tmp_path / "catalog.json"
    observation = _observation(128, _fingerprint())
    save_observations([observation] * 3, observations_path)
    save_profile_requests([observation.request], manifest_path)
    environment_path.write_text(json.dumps(_environment().to_dict()))
    command = Path(sysconfig.get_path("scripts")) / "lm-resiliency-discover-moe-regimes"
    assert command.is_file()

    result = subprocess.run(
        [
            command,
            "--observations",
            str(observations_path),
            "--environment",
            str(environment_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(catalog_path),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "1 regimes" in result.stdout
    assert MoERegimeCatalog.load(catalog_path).cycle_size == 1


def test_profile_requests_preserves_every_layer_expert_shape():
    requests = [
        ProfileRequest(64, ProfileLocation("0", 1, 3)),
        ProfileRequest(128, ProfileLocation("7", 5, 3)),
    ]
    seen = []

    def profiler(request):
        seen.append(request)
        return _fingerprint(tail=f"n-{request.n_exec}")

    observations = profile_requests(requests, profiler)
    assert seen == [requests[0]] * 3 + [requests[1]] * 3
    assert [observation.n_exec for observation in observations] == [64] * 3 + [128] * 3

    with pytest.raises(ValueError, match="repetitions"):
        profile_requests(requests, profiler, repetitions=0)


def test_profile_request_manifest_round_trip(tmp_path):
    requests = [
        ProfileRequest(64, ProfileLocation("0", 1, 3)),
        ProfileRequest(128, ProfileLocation("7", 5, 3)),
    ]
    path = tmp_path / "manifest.json"

    save_profile_requests(requests, path)

    assert load_profile_requests(path) == tuple(requests)
    with pytest.raises(ValueError, match="duplicate"):
        save_profile_requests([requests[0], requests[0]], path)


def test_build_profile_requests_covers_all_layers_experts_and_shapes():
    requests = build_profile_requests(
        layer_ids=[0, 1],
        expert_ids=[4, 5],
        ep_position=3,
        n_exec_values=[0, 128, 256],
    )

    assert len(requests) == 12
    assert {request.location.layer_id for request in requests} == {"0", "1"}
    assert {request.location.expert_id for request in requests} == {4, 5}
    assert {request.n_exec for request in requests} == {0, 128, 256}
    assert {request.location.ep_position for request in requests} == {3}


def test_kineto_parser_retains_kernel_order_and_launch_details():
    trace = {
        "traceEvents": [
            {"cat": "cpu_op", "name": "aten::mm", "args": {}},
            {
                "cat": "kernel",
                "name": "cutlass_gemm",
                "args": {
                    "grid": [264, 1, 1],
                    "block": "256,1,1",
                    "shared memory": 65536,
                    "registers per thread": 128,
                },
            },
            {
                "cat": "Kernel",
                "name": "silu_mul",
                "args": {
                    "grid x": 32,
                    "grid y": 2,
                    "grid z": 1,
                    "block x": 256,
                    "block y": 1,
                    "block z": 1,
                },
            },
        ]
    }

    launches = kernel_launches_from_kineto_trace(trace)

    assert [launch.name for launch in launches] == ["cutlass_gemm", "silu_mul"]
    assert launches[0].grid == (264, 1, 1)
    assert launches[0].block == (256, 1, 1)
    assert launches[0].shared_memory_bytes == 65536
    assert launches[1].grid == (32, 2, 1)


def test_scheduler_rotates_complete_catalog_and_reports_general_bound():
    observations = [
        _observation(64, _fingerprint(tail="aligned")),
        _observation(65, _fingerprint(tail="tail-1")),
        _observation(66, _fingerprint(tail="tail-2")),
    ]
    catalog = _discover(observations, environment=_environment())
    scheduler = MoEReplayScheduler(catalog, replay_interval=10)

    assert scheduler.detection_bound_steps == 30
    assert scheduler.recipe_for_step(9) is None
    first = scheduler.recipe_for_step(10)
    assert first is not None and first.position == 0 and first.cycle == 0
    assert scheduler.recipe_for_step(11) is None
    second = scheduler.recipe_for_step(20)
    third = scheduler.recipe_for_step(30)
    assert second is not None and second.position == 1
    assert third is not None and third.completes_cycle
    assert scheduler.completed_cycles == 1
    assert scheduler.recipe_for_step(40).position == 0


def test_catalog_converts_representatives_to_unified_replay_shape_plan():
    catalog = _discover(
        [
            _observation(64, _fingerprint(tail="aligned")),
            _observation(65, _fingerprint(tail="tail")),
        ],
        environment=_environment(),
    )

    plan = catalog.to_replay_shape_plan()

    assert {shape.dimensions for shape in plan.shapes} == {(64,), (65,)}
    assert len(plan.shapes) == catalog.cycle_size
    assert plan.source_id == f"moe-catalog:{catalog.identifier}"


def test_scheduler_state_is_bound_to_catalog():
    catalog = _discover(
        [_observation(64, _fingerprint()), _observation(65, _fingerprint(tail="tail"))],
        environment=_environment(),
    )
    first = MoEReplayScheduler(catalog, replay_interval=5)
    first.next_recipe()
    state = first.state_dict()

    restored = MoEReplayScheduler(catalog, replay_interval=5)
    restored.load_state_dict(state)
    assert restored.next_recipe().position == 1

    changed = _discover([_observation(64, _fingerprint())], environment=_environment())
    with pytest.raises(CatalogEnvironmentMismatch, match="different catalog"):
        MoEReplayScheduler(changed, replay_interval=5).load_state_dict(state)


def test_discovery_rejects_catalog_above_replay_budget():
    observations = [
        _observation(64, _fingerprint(tail="aligned")),
        _observation(65, _fingerprint(tail="tail-1")),
        _observation(66, _fingerprint(tail="tail-2")),
    ]

    with pytest.raises(RegimeDiscoveryError, match="requires 3 replay recipes"):
        _discover(
            observations,
            environment=_environment(),
            max_replay_recipes=2,
        )

    catalog = _discover(
        observations,
        environment=_environment(),
        max_replay_recipes=3,
    )
    assert catalog.cycle_size == 3
