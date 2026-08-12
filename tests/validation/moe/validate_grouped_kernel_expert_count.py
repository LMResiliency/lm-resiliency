"""Validate scalar SCOUT catalogs with actual multi-expert grouped kernels.

Each invocation evaluates one actual grouped-kernel expert count:

    python tests/validation/moe/validate_grouped_kernel_expert_count.py \
        --actual-experts 4 \
        --artifact-dir /tmp/scout-grouped-kernel-experts/e4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).parents[3]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

import torch
import triton

from lm_resiliency.detection.moe_regimes import (
    ExecutionFingerprint,
    ExecutionObservation,
    KernelLaunch,
    MoERegimeCatalog,
    ProfileLocation,
    ProfileRequest,
    TorchCudaExecutionProfiler,
    _fingerprint_covers,
    _regime_equivalence_key,
    current_moe_environment,
    discover_execution_regimes,
    save_observations,
    save_profile_requests,
)
from tests.support.grouped_kernel_expert_matrix import (
    ACTUAL_EXPERT_COUNTS,
    heterogeneous_count_vectors,
    scalar_n_exec_values,
)
from tests.support.triton_grouped_expert import InstrumentedGroupedExperts

_HIDDEN = 128
_OUTPUT = 128
_EXECUTION_CLASS = "triton-persistent-grouped-forward-backward"


def _source_identity() -> str:
    digest = hashlib.sha256()
    for relative in (
        "lm_resiliency/detection/moe_regimes.py",
        "tests/support/triton_grouped_expert.py",
        "tests/support/grouped_kernel_expert_matrix.py",
        "tests/validation/moe/validate_grouped_kernel_expert_count.py",
    ):
        digest.update((_REPOSITORY / relative).read_bytes())
    return f"grouped-kernel-validation-sources-{digest.hexdigest()}"


def _inputs(
    count_values: Sequence[int],
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    packed_rows = sum(int(value) for value in count_values)
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randn(
        packed_rows,
        _HIDDEN,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    counts = torch.tensor(tuple(count_values), device=device, dtype=torch.int32)
    grad_output = torch.randn(
        packed_rows,
        _OUTPUT,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return tokens, counts, grad_output


def _reference(
    backend: InstrumentedGroupedExperts,
    tokens: torch.Tensor,
    count_values: Sequence[int],
    grad_output: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expected_output = torch.empty_like(grad_output)
    expected_grad_input = torch.empty_like(tokens)
    expected_grad_weight = torch.empty_like(backend.weight)
    offset = 0
    for expert, count in enumerate(count_values):
        extent = int(count)
        expert_tokens = tokens[offset : offset + extent].float()
        expert_grad_output = grad_output[offset : offset + extent].float()
        weight = backend.weight[expert].float()
        expected_output[offset : offset + extent] = (expert_tokens @ weight).bfloat16()
        expected_grad_input[offset : offset + extent] = (expert_grad_output @ weight.T).bfloat16()
        expected_grad_weight[expert] = (expert_tokens.T @ expert_grad_output).bfloat16()
        offset += extent
    return expected_output, expected_grad_input, expected_grad_weight


def _validate_numerics(
    backend: InstrumentedGroupedExperts,
    *,
    max_n_exec: int,
    device: torch.device,
) -> int:
    experts = backend.num_experts
    vectors = [
        (1,) * experts,
        (max(1, max_n_exec // 2),) * experts,
        (max_n_exec,) * experts,
    ]
    if experts > 1:
        vectors.extend(
            (
                tuple(1 if expert % 2 == 0 else max_n_exec for expert in range(experts)),
                tuple(
                    0 if expert == 0 else min(max_n_exec, 32 + expert) for expert in range(experts)
                ),
            )
        )

    checked = 0
    for index, count_values in enumerate(vectors):
        tokens, counts, grad_output = _inputs(
            count_values,
            device=device,
            seed=200 + index,
        )
        backend.clear_fault()
        output, grad_input, grad_weight = backend.run_forward_backward(
            tokens,
            counts,
            grad_output,
        )
        torch.cuda.synchronize(device)
        expected = _reference(backend, tokens, count_values, grad_output)
        actual = (output, grad_input, grad_weight)
        labels = ("forward", "input-gradient", "weight-gradient")
        tolerances = ((0.01, 0.01), (0.01, 0.01), (0.01, 0.02))
        for label, observed, reference, (rtol, atol) in zip(
            labels,
            actual,
            expected,
            tolerances,
            strict=True,
        ):
            if not torch.allclose(observed, reference, rtol=rtol, atol=atol):
                difference = (observed.float() - reference.float()).abs().max().item()
                raise RuntimeError(
                    f"{label} numerical mismatch for counts={count_values}; "
                    f"maximum absolute difference={difference}"
                )
        checked += 1
    return checked


def _fingerprint(
    launches: tuple[KernelLaunch, ...],
    backend: InstrumentedGroupedExperts,
    count_values: Sequence[int],
) -> ExecutionFingerprint:
    counts = tuple(int(value) for value in count_values)
    hints = backend.execution_hints(counts, kernel_count=len(launches))
    return ExecutionFingerprint.create(
        kernels=launches,
        algorithm_ids=hints.algorithm_ids,
        tile_shapes=hints.tile_shapes,
        tail_path=hints.tail_path,
        workspace_bytes=hints.workspace_bytes,
        pressure_class=hints.pressure_class,
        overlap_class=hints.overlap_class,
        persistent_work_items=hints.persistent_work_items,
        cta_semantics=hints.cta_semantics,
        extra=hints.extra,
    )


def _kineto_launches(
    backend: InstrumentedGroupedExperts,
    *,
    device: torch.device,
) -> tuple[KernelLaunch, ...]:
    n_exec = 64
    count_values = (n_exec,) * backend.num_experts
    tokens, counts, grad_output = _inputs(count_values, device=device, seed=101)
    request = ProfileRequest(
        n_exec,
        ProfileLocation("kineto", 0, 0, _EXECUTION_CLASS),
    )

    def workload(_request: ProfileRequest) -> None:
        backend.run_forward_backward(tokens, counts, grad_output)

    def hints(_request: ProfileRequest, launches: tuple[KernelLaunch, ...]):
        return backend.execution_hints(count_values, kernel_count=len(launches))

    fingerprint = TorchCudaExecutionProfiler(
        workload,
        hints=hints,
        warmup=1,
        device=device,
    )(request)
    if len(fingerprint.kernels) != 3:
        raise RuntimeError(f"expected three Triton kernels, got {len(fingerprint.kernels)}")
    return fingerprint.kernels


def _collect_scalar_observations(
    backend: InstrumentedGroupedExperts,
    n_exec_values: Sequence[int],
    launches: tuple[KernelLaunch, ...],
    *,
    repetitions: int,
    device: torch.device,
) -> tuple[tuple[ProfileRequest, ...], tuple[ExecutionObservation, ...]]:
    location = ProfileLocation("actual-expert-matrix", 0, 0, _EXECUTION_CLASS)
    requests = tuple(ProfileRequest(n_exec, location) for n_exec in n_exec_values)
    observations = []
    for request_index, request in enumerate(requests):
        count_values = (request.n_exec,) * backend.num_experts
        tokens, counts, grad_output = _inputs(
            count_values,
            device=device,
            seed=1000 + request_index,
        )
        identifiers = set()
        for _ in range(repetitions):
            backend.run_forward_backward(tokens, counts, grad_output)
            torch.cuda.synchronize(device)
            fingerprint = _fingerprint(launches, backend, count_values)
            identifiers.add(fingerprint.identifier)
            observations.append(ExecutionObservation(request=request, fingerprint=fingerprint))
        if len(identifiers) != 1:
            raise RuntimeError(
                f"unstable fingerprint for E={backend.num_experts}/"
                f"n_exec={request.n_exec}: {identifiers}"
            )
    return requests, tuple(observations)


def _environment(
    backend: InstrumentedGroupedExperts,
    *,
    max_n_exec: int,
    manifest_size: int,
    device: torch.device,
) -> Any:
    return current_moe_environment(
        backend="triton-persistent-grouped-gemm",
        backend_version=triton.__version__,
        model={
            "hidden_size": backend.hidden,
            "expert_output_size": backend.output,
            "actual_local_experts": backend.num_experts,
            "profiled_experts": backend.num_experts,
            "minimum_per_expert_n_exec": 1,
            "maximum_per_expert_n_exec": max_n_exec,
        },
        precision="bf16",
        parallelism={"expert_parallel": 1, "expert_tp": 1},
        extra={
            "container_digest": "aws-a100-cuda13-validation-environment",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "manifest": "uniform-scalar-per-expert-n-exec-v1",
            "manifest_size": manifest_size,
            "model_commit": _source_identity(),
            "overlap": "none",
            "precision_recipe": "native-bf16",
            "workspace_policy": "zero-workspace",
        },
        device=device,
    )


def _fault_oracle(
    backend: InstrumentedGroupedExperts,
    catalog: MoERegimeCatalog,
    *,
    mode: str,
    device: torch.device,
) -> dict[str, int]:
    if mode == "none":
        return {"representatives": 0, "roles": 0, "injections": 0}
    representatives = 0
    roles_checked = 0
    injections = 0
    for recipe_index, recipe in enumerate(catalog.replay_recipes):
        count_values = (recipe.n_exec,) * backend.num_experts
        tokens, counts, grad_output = _inputs(
            count_values,
            device=device,
            seed=5000 + recipe_index,
        )
        backend.clear_fault()
        healthy_output, healthy_input_grad, healthy_weight_grad = backend.run_forward_backward(
            tokens, counts, grad_output
        )
        torch.cuda.synchronize(device)
        healthy = {
            "forward": healthy_output.clone(),
            "input-gradient": healthy_input_grad.clone(),
            "weight-gradient": healthy_weight_grad.clone(),
        }
        occurrences = backend.role_occurrences(count_values)
        representatives += 1
        for kernel, by_role in occurrences.items():
            for work_items in by_role.values():
                roles_checked += 1
                selected: Iterable[int] = work_items if mode == "all" else work_items[:1]
                for work_item in selected:
                    backend.set_fault(kernel, work_item)
                    if kernel == "forward":
                        faulty = backend.launch_forward(tokens, counts)
                    else:
                        faulty_input, faulty_weight = backend.launch_backward(
                            tokens,
                            counts,
                            backend.weight,
                            grad_output,
                        )
                        faulty = faulty_input if kernel == "input-gradient" else faulty_weight
                    torch.cuda.synchronize(device)
                    if torch.equal(faulty, healthy[kernel]):
                        raise RuntimeError(
                            f"fault oracle did not trigger for E={backend.num_experts}/"
                            f"n_exec={recipe.n_exec}/{kernel}/{work_item}"
                        )
                    injections += 1
        backend.clear_fault()
    return {
        "representatives": representatives,
        "roles": roles_checked,
        "injections": injections,
    }


def _representative_candidates(
    catalog: MoERegimeCatalog,
) -> tuple[tuple[int, str, ExecutionFingerprint], ...]:
    candidates = []
    for regime in catalog.regimes:
        for n_exec in regime.representatives:
            fingerprint = regime.fingerprint_for_shape(n_exec)
            key = _regime_equivalence_key(
                fingerprint,
                policy=catalog.equivalence_policy,
                n_exec=n_exec,
            )
            candidates.append((n_exec, key, fingerprint))
    return tuple(candidates)


def _uncovered_role_obligations(
    candidates: Sequence[ExecutionFingerprint],
    target: ExecutionFingerprint,
) -> tuple[str, ...]:
    if not candidates:
        return ("hard-regime:missing",)
    uncovered = []
    for kernel_index, target_semantics in enumerate(target.cta_semantics):
        for role, target_count in target_semantics.role_counts:
            candidate_count = max(
                dict(candidate.cta_semantics[kernel_index].role_counts).get(role, 0)
                for candidate in candidates
            )
            if candidate_count < target_count:
                uncovered.append(
                    f"kernel={kernel_index}/role={role}/"
                    f"target={target_count}/cycle-max={candidate_count}"
                )
    return tuple(uncovered)


def _validate_heterogeneous_vectors(
    backend: InstrumentedGroupedExperts,
    catalog: MoERegimeCatalog,
    launches: tuple[KernelLaunch, ...],
    *,
    max_n_exec: int,
    random_samples: int,
    repetitions: int,
    device: torch.device,
) -> tuple[dict[str, Any], tuple[dict[str, Any], ...]]:
    vectors = heterogeneous_count_vectors(
        backend.num_experts,
        max_n_exec=max_n_exec,
        num_sms=backend.num_sms,
        random_samples=random_samples,
    )
    candidates = _representative_candidates(catalog)
    records = []
    for vector_index, count_values in enumerate(vectors):
        tokens, counts, grad_output = _inputs(
            count_values,
            device=device,
            seed=100_000 + vector_index,
        )
        fingerprints = []
        for _ in range(repetitions):
            backend.run_forward_backward(tokens, counts, grad_output)
            torch.cuda.synchronize(device)
            fingerprints.append(_fingerprint(launches, backend, count_values))
        identifiers = {fingerprint.identifier for fingerprint in fingerprints}
        if len(identifiers) != 1:
            raise RuntimeError(
                f"unstable heterogeneous fingerprint for E={backend.num_experts}/"
                f"counts={count_values}: {identifiers}"
            )
        target = fingerprints[0]
        target_key = _regime_equivalence_key(
            target,
            policy=catalog.equivalence_policy,
            n_exec=max(count_values),
        )
        same_regime = tuple(
            fingerprint for _n_exec, key, fingerprint in candidates if key == target_key
        )
        single_recipe_covered = any(
            _fingerprint_covers(candidate, target) for candidate in same_regime
        )
        uncovered = _uncovered_role_obligations(same_regime, target)
        records.append(
            {
                "counts": list(count_values),
                "hard_regime_id": target_key,
                "hard_regime_covered": bool(same_regime),
                "single_recipe_covered": single_recipe_covered,
                "cycle_role_envelope_covered": not uncovered,
                "uncovered_role_obligations": list(uncovered),
            }
        )

    hard_regime_covered = sum(record["hard_regime_covered"] for record in records)
    single_recipe_covered = sum(record["single_recipe_covered"] for record in records)
    cycle_covered = sum(record["cycle_role_envelope_covered"] for record in records)
    summary = {
        "vectors": len(records),
        "repetitions_per_vector": repetitions,
        "hard_regime_covered": hard_regime_covered,
        "single_recipe_covered": single_recipe_covered,
        "cycle_role_envelope_covered": cycle_covered,
        "all_hard_regimes_covered": hard_regime_covered == len(records),
        "all_cycle_role_envelopes_covered": cycle_covered == len(records),
    }
    return summary, tuple(records)


def _regime_breakdown(catalog: MoERegimeCatalog) -> list[dict[str, Any]]:
    return [
        {
            "regime_id": regime.regime_id,
            "pressure_class": regime.fingerprint.pressure_class,
            "member_n_exec_values": len(regime.n_exec_values),
            "representatives": list(regime.representatives),
        }
        for regime in catalog.regimes
    ]


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1729)
    args.artifact_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    backend = InstrumentedGroupedExperts(
        num_experts=args.actual_experts,
        hidden=_HIDDEN,
        output=_OUTPUT,
        device=device,
    )
    n_exec_values = scalar_n_exec_values(args.max_n_exec)
    numerical_vectors = _validate_numerics(
        backend,
        max_n_exec=args.max_n_exec,
        device=device,
    )
    launches = _kineto_launches(backend, device=device)
    requests, observations = _collect_scalar_observations(
        backend,
        n_exec_values,
        launches,
        repetitions=args.repetitions,
        device=device,
    )
    environment = _environment(
        backend,
        max_n_exec=args.max_n_exec,
        manifest_size=len(requests),
        device=device,
    )
    exact = discover_execution_regimes(
        observations,
        environment=environment,
        equivalence_policy="exact_launch",
        expected_requests=requests,
        minimum_observations_per_request=args.repetitions,
    )
    compressed = discover_execution_regimes(
        observations,
        environment=environment,
        equivalence_policy="plan_and_pressure",
        expected_requests=requests,
        minimum_observations_per_request=args.repetitions,
        max_replay_recipes=args.recipe_budget,
    )

    save_profile_requests(requests, args.artifact_dir / "manifest.json")
    save_observations(observations, args.artifact_dir / "observations.jsonl")
    exact.save(args.artifact_dir / "catalog-exact.json")
    compressed.save(args.artifact_dir / "catalog-compressed.json")
    loaded = MoERegimeCatalog.load(args.artifact_dir / "catalog-compressed.json")
    if loaded.identifier != compressed.identifier:
        raise RuntimeError("compressed catalog checksum changed after serialization")

    fault = _fault_oracle(
        backend,
        compressed,
        mode=args.fault_occurrences,
        device=device,
    )
    heterogeneous, heterogeneous_records = _validate_heterogeneous_vectors(
        backend,
        compressed,
        launches,
        max_n_exec=args.max_n_exec,
        random_samples=args.heterogeneous_random_samples,
        repetitions=args.heterogeneous_repetitions,
        device=device,
    )
    (args.artifact_dir / "heterogeneous-coverage.jsonl").write_text(
        "".join(json.dumps(record, sort_keys=True) + "\n" for record in heterogeneous_records)
    )

    exact_count = exact.cycle_size
    compressed_count = compressed.cycle_size
    fault_oracle_passed = args.fault_occurrences == "none" or (
        fault["representatives"] == compressed_count
        and fault["roles"] > 0
        and fault["injections"] >= fault["roles"]
    )
    passed = (
        exact_count == len(n_exec_values)
        and compressed_count <= args.recipe_budget
        and fault_oracle_passed
        and heterogeneous["all_hard_regimes_covered"]
        and heterogeneous["all_cycle_role_envelopes_covered"]
    )
    summary = {
        "result": "PASS" if passed else "FAIL",
        "gpu": torch.cuda.get_device_name(device),
        "sm_count": backend.num_sms,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "actual_grouped_kernel_experts": backend.num_experts,
        "projection": {"hidden": backend.hidden, "output": backend.output},
        "scalar_manifest": {
            "minimum_n_exec": n_exec_values[0],
            "maximum_n_exec": n_exec_values[-1],
            "physical_shapes": len(n_exec_values),
            "repetitions_per_shape": args.repetitions,
            "observations": len(observations),
        },
        "numerical_vectors": numerical_vectors,
        "exact_recipes": exact_count,
        "compressed_recipes": compressed_count,
        "recipe_reduction_fraction": 1.0 - compressed_count / exact_count,
        "compression_ratio": exact_count / compressed_count,
        "regimes": len(compressed.regimes),
        "regime_breakdown": _regime_breakdown(compressed),
        "recipe_budget": args.recipe_budget,
        "within_recipe_budget": compressed_count <= args.recipe_budget,
        "fault_oracle": fault,
        "fault_oracle_passed": fault_oracle_passed,
        "heterogeneous_coverage": heterogeneous,
        "catalog_id": compressed.identifier,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise RuntimeError(f"grouped-kernel expert validation failed for E={backend.num_experts}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--actual-experts",
        type=int,
        choices=ACTUAL_EXPERT_COUNTS,
        required=True,
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max-n-exec", type=int, default=512)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--recipe-budget", type=int, default=50)
    parser.add_argument("--heterogeneous-random-samples", type=int, default=128)
    parser.add_argument("--heterogeneous-repetitions", type=int, default=2)
    parser.add_argument(
        "--fault-occurrences",
        choices=("roles", "all", "none"),
        default="roles",
    )
    args = parser.parse_args()
    if args.max_n_exec < 1:
        parser.error("--max-n-exec must be positive")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.recipe_budget < 1:
        parser.error("--recipe-budget must be positive")
    if args.heterogeneous_random_samples < 0:
        parser.error("--heterogeneous-random-samples cannot be negative")
    if args.heterogeneous_repetitions < 1:
        parser.error("--heterogeneous-repetitions must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
