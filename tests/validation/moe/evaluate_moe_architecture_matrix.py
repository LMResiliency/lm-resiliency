"""Evaluate scalar SCOUT MoE recipe counts across expert projections.

Each invocation profiles one representative expert GEMM over the exhaustive
per-expert ``n_exec`` range declared by the preset or an explicit upper bound:

    python tests/validation/moe/evaluate_moe_architecture_matrix.py \
        --preset large-1-local \
        --max-n-exec 2048 \
        --artifact-dir /tmp/scout-moe-architectures/large-1-local
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from collections import Counter
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
    current_moe_environment,
    discover_execution_regimes,
    load_observations,
    load_profile_requests,
    save_observations,
    save_profile_requests,
)
from tests.support.moe_architecture_matrix import (
    ARCHITECTURE_PRESETS,
    MoEArchitecturePreset,
    per_expert_n_exec_values,
)
from tests.support.triton_grouped_expert import InstrumentedGroupedExperts

_EXECUTION_CLASS = "triton-persistent-expert-forward-backward"


def _source_identity() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256()
    for relative in (
        "lm_resiliency/detection/moe_regimes.py",
        "tests/support/triton_grouped_expert.py",
        "tests/support/moe_architecture_matrix.py",
        "tests/validation/moe/evaluate_moe_architecture_matrix.py",
    ):
        digest.update((_REPOSITORY / relative).read_bytes())
    return f"{commit}-sources-{digest.hexdigest()[:16]}"


def _inputs(
    n_exec: int,
    preset: MoEArchitecturePreset,
    *,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    tokens = torch.randn(
        n_exec,
        preset.hidden,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    counts = torch.tensor((n_exec,), device=device, dtype=torch.int32)
    grad_output = torch.randn(
        n_exec,
        preset.expert_output,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return tokens, counts, grad_output


def _fingerprint(
    launches: tuple[KernelLaunch, ...],
    backend: InstrumentedGroupedExperts,
    n_exec: int,
) -> ExecutionFingerprint:
    hints = backend.execution_hints((n_exec,), kernel_count=len(launches))
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
    preset: MoEArchitecturePreset,
    n_exec_values: tuple[int, ...],
    device: torch.device,
) -> tuple[KernelLaunch, ...]:
    n_exec = next(
        (value for value in n_exec_values if value >= 64),
        n_exec_values[-1],
    )
    tokens, counts, grad_output = _inputs(n_exec, preset, device=device, seed=101)
    request = ProfileRequest(
        n_exec,
        ProfileLocation("kineto", 0, 0, _EXECUTION_CLASS),
    )

    def workload(_request: ProfileRequest) -> None:
        backend.run_forward_backward(tokens, counts, grad_output)

    def hints(profile_request, launches):
        return backend.execution_hints(
            (profile_request.n_exec,),
            kernel_count=len(launches),
        )

    fingerprint = TorchCudaExecutionProfiler(
        workload,
        hints=hints,
        warmup=1,
        device=device,
    )(request)
    if len(fingerprint.kernels) != 3:
        raise RuntimeError(f"expected three Triton kernels, got {len(fingerprint.kernels)}")
    return fingerprint.kernels


def _validate_numerics(
    backend: InstrumentedGroupedExperts,
    preset: MoEArchitecturePreset,
    n_exec_values: tuple[int, ...],
    device: torch.device,
) -> int:
    checked = 0
    candidates = (n_exec_values[0], n_exec_values[len(n_exec_values) // 2], n_exec_values[-1])
    for index, n_exec in enumerate(dict.fromkeys(candidates)):
        tokens, counts, grad_output = _inputs(
            n_exec,
            preset,
            device=device,
            seed=200 + index,
        )
        output, grad_input, grad_weight = backend.run_forward_backward(
            tokens,
            counts,
            grad_output,
        )
        torch.cuda.synchronize(device)
        weight = backend.weight[0].float()
        expected_output = (tokens.float() @ weight).bfloat16()
        expected_grad_input = (grad_output.float() @ weight.T).bfloat16()
        expected_grad_weight = (tokens.float().T @ grad_output.float()).bfloat16().unsqueeze(0)
        if not torch.allclose(output, expected_output, rtol=0.01, atol=0.01):
            raise RuntimeError(f"forward numerical mismatch for n_exec={n_exec}")
        if not torch.allclose(grad_input, expected_grad_input, rtol=0.01, atol=0.01):
            raise RuntimeError(f"input-gradient numerical mismatch for n_exec={n_exec}")
        if not torch.allclose(grad_weight, expected_grad_weight, rtol=0.01, atol=0.02):
            raise RuntimeError(f"weight-gradient numerical mismatch for n_exec={n_exec}")
        checked += 1
    return checked


def _collect_observations(
    backend: InstrumentedGroupedExperts,
    preset: MoEArchitecturePreset,
    n_exec_values: tuple[int, ...],
    device: torch.device,
    launches: tuple[KernelLaunch, ...],
    *,
    repetitions: int,
) -> tuple[tuple[ProfileRequest, ...], tuple[ExecutionObservation, ...]]:
    location = ProfileLocation("architecture-matrix", 0, 0, _EXECUTION_CLASS)
    requests = tuple(ProfileRequest(n_exec, location) for n_exec in n_exec_values)
    observations = []
    for request_index, request in enumerate(requests):
        tokens, counts, grad_output = _inputs(
            request.n_exec,
            preset,
            device=device,
            seed=1000 + request_index,
        )
        identifiers = set()
        for _ in range(repetitions):
            backend.run_forward_backward(tokens, counts, grad_output)
            torch.cuda.synchronize(device)
            fingerprint = _fingerprint(launches, backend, request.n_exec)
            identifiers.add(fingerprint.identifier)
            observations.append(ExecutionObservation(request=request, fingerprint=fingerprint))
        if len(identifiers) != 1:
            raise RuntimeError(f"unstable fingerprint for n_exec={request.n_exec}: {identifiers}")
    return requests, tuple(observations)


def _fault_oracle_by_role(
    backend: InstrumentedGroupedExperts,
    preset: MoEArchitecturePreset,
    catalog: MoERegimeCatalog,
    device: torch.device,
) -> dict[str, int]:
    roles_checked = 0
    injections = 0
    for recipe_index, recipe in enumerate(catalog.replay_recipes):
        n_exec = recipe.n_exec
        tokens, counts, grad_output = _inputs(
            n_exec,
            preset,
            device=device,
            seed=5000 + recipe_index,
        )
        backend.clear_fault()
        healthy_output, healthy_input_grad, healthy_weight_grad = backend.run_forward_backward(
            tokens,
            counts,
            grad_output,
        )
        torch.cuda.synchronize(device)
        healthy = {
            "forward": healthy_output.clone(),
            "input-gradient": healthy_input_grad.clone(),
            "weight-gradient": healthy_weight_grad.clone(),
        }
        for kernel, by_role in backend.role_occurrences((n_exec,)).items():
            for work_items in by_role.values():
                roles_checked += 1
                work_item = work_items[0]
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
                        f"fault oracle did not trigger for n_exec={n_exec}/{kernel}/{work_item}"
                    )
                injections += 1
        backend.clear_fault()
    return {"roles": roles_checked, "injections": injections}


def _regime_breakdown(catalog: MoERegimeCatalog) -> list[dict[str, Any]]:
    return [
        {
            "regime_id": regime.regime_id,
            "pressure_class": regime.fingerprint.pressure_class,
            "member_n_exec_values": len(regime.n_exec_values),
            "representatives": len(regime.representatives),
        }
        for regime in catalog.regimes
    ]


def _environment(
    preset: MoEArchitecturePreset,
    *,
    device: torch.device,
    manifest_size: int,
    max_n_exec: int,
) -> Any:
    return current_moe_environment(
        backend="triton-persistent-grouped-gemm",
        backend_version=triton.__version__,
        model={
            "preset": preset.name,
            "hidden_size": preset.hidden,
            "expert_output_size": preset.expert_output,
            "global_experts": preset.global_experts,
            "local_experts_metadata_only": preset.local_experts,
            "top_k": preset.top_k,
            "max_per_expert_n_exec": max_n_exec,
            "profiled_experts": 1,
        },
        precision="bf16",
        parallelism={
            "expert_parallel": preset.expert_parallel,
            "expert_tp": 1,
        },
        extra={
            "container_digest": "aws-a100-cuda13-validation-environment",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "manifest": "exhaustive-per-expert-n-exec-v1",
            "manifest_size": manifest_size,
            "model_commit": _source_identity(),
            "overlap": "none",
            "preset_default_max_per_expert_n_exec": preset.max_n_exec,
            "precision_recipe": "native-bf16",
            "workspace_policy": "zero-workspace",
        },
        device=device,
    )


def _load_profile_shards(
    paths: tuple[Path, ...],
    *,
    preset: MoEArchitecturePreset,
    expected_n_exec_values: tuple[int, ...],
    repetitions: int,
) -> tuple[
    tuple[ProfileRequest, ...],
    tuple[ExecutionObservation, ...],
    int,
    list[dict[str, Any]],
]:
    requests = []
    observations = []
    numerical_shapes = 0
    shard_records = []
    for path in paths:
        summary = json.loads((path / "summary.json").read_text())
        if summary.get("result") != "PASS":
            raise RuntimeError(f"profile shard did not pass: {path}")
        if summary.get("preset", {}).get("name") != preset.name:
            raise RuntimeError(f"profile shard uses the wrong preset: {path}")
        if summary.get("repetitions_per_shape") != repetitions:
            raise RuntimeError(f"profile shard uses the wrong repetition count: {path}")
        shard_requests = load_profile_requests(path / "manifest.json")
        requests.extend(shard_requests)
        observations.extend(load_observations(path / "observations.jsonl"))
        numerical_shapes += int(summary["numerical_shapes"])
        shard_records.append(
            {
                "path": str(path),
                "minimum_n_exec": summary["manifest"]["minimum_n_exec"],
                "maximum_n_exec": summary["manifest"]["maximum_n_exec"],
                "physical_shapes": summary["manifest"]["physical_shapes"],
                "elapsed_seconds": summary["elapsed_seconds"],
                "catalog_id": summary["catalog_id"],
            }
        )

    expected_location = ProfileLocation("architecture-matrix", 0, 0, _EXECUTION_CLASS)
    expected_requests = tuple(
        ProfileRequest(n_exec, expected_location) for n_exec in expected_n_exec_values
    )
    if len(requests) != len(set(requests)):
        raise RuntimeError("profile shard manifests overlap")
    if set(requests) != set(expected_requests):
        missing = sorted(set(expected_requests) - set(requests))
        extra = sorted(set(requests) - set(expected_requests))
        raise RuntimeError(
            f"profile shards do not exactly cover the requested domain: "
            f"missing={len(missing)}, extra={len(extra)}"
        )

    observation_counts = Counter(observation.request for observation in observations)
    invalid_counts = {
        request: observation_counts[request]
        for request in expected_requests
        if observation_counts[request] != repetitions
    }
    if invalid_counts:
        raise RuntimeError(
            f"profile shards have incorrect observation counts for {len(invalid_counts)} requests"
        )
    return (
        expected_requests,
        tuple(observations),
        numerical_shapes,
        shard_records,
    )


def evaluate(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preset = ARCHITECTURE_PRESETS[args.preset]
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1729)
    n_exec_values = per_expert_n_exec_values(
        preset,
        min_n_exec=args.min_n_exec,
        max_n_exec=args.max_n_exec,
    )
    args.artifact_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    backend = InstrumentedGroupedExperts(
        num_experts=1,
        hidden=preset.hidden,
        output=preset.expert_output,
        device=device,
    )
    if args.profile_shards:
        requests, observations, numerical_shapes, profile_shards = _load_profile_shards(
            tuple(args.profile_shards),
            preset=preset,
            expected_n_exec_values=n_exec_values,
            repetitions=args.repetitions,
        )
    else:
        numerical_shapes = _validate_numerics(backend, preset, n_exec_values, device)
        launches = _kineto_launches(backend, preset, n_exec_values, device)
        requests, observations = _collect_observations(
            backend,
            preset,
            n_exec_values,
            device,
            launches,
            repetitions=args.repetitions,
        )
        profile_shards = []
    environment = _environment(
        preset,
        device=device,
        manifest_size=len(n_exec_values),
        max_n_exec=n_exec_values[-1],
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
    )

    save_profile_requests(requests, args.artifact_dir / "manifest.json")
    save_observations(observations, args.artifact_dir / "observations.jsonl")
    exact.save(args.artifact_dir / "catalog-exact.json")
    compressed.save(args.artifact_dir / "catalog-compressed.json")
    loaded = MoERegimeCatalog.load(args.artifact_dir / "catalog-compressed.json")
    if loaded.identifier != compressed.identifier:
        raise RuntimeError("compressed catalog checksum changed after serialization")

    fault = (
        _fault_oracle_by_role(backend, preset, compressed, device)
        if args.fault_occurrences == "roles"
        else {"roles": 0, "injections": 0}
    )
    exact_count = exact.cycle_size
    compressed_count = compressed.cycle_size
    summary = {
        "result": "PASS",
        "preset": {
            "name": preset.name,
            "description": preset.description,
            "local_experts_metadata_only": preset.local_experts,
            "global_experts_metadata_only": preset.global_experts,
            "top_k_metadata_only": preset.top_k,
            "expert_parallel_metadata_only": preset.expert_parallel,
            "hidden": preset.hidden,
            "expert_output": preset.expert_output,
            "preset_default_max_per_expert_n_exec": preset.max_n_exec,
            "qualified_max_per_expert_n_exec": n_exec_values[-1],
            "profiled_experts": 1,
        },
        "manifest": {
            "kind": "exhaustive-per-expert-n-exec-v1",
            "minimum_n_exec": n_exec_values[0],
            "maximum_n_exec": n_exec_values[-1],
            "physical_shapes": len(n_exec_values),
            "exhaustive": True,
        },
        "gpu": torch.cuda.get_device_name(device),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "repetitions_per_shape": args.repetitions,
        "observations": len(observations),
        "numerical_shapes": numerical_shapes,
        "profile_shards": profile_shards,
        "exact_recipes": exact_count,
        "compressed_recipes": compressed_count,
        "recipe_reduction_fraction": 1.0 - compressed_count / exact_count,
        "compression_ratio": exact_count / compressed_count,
        "regimes": len(compressed.regimes),
        "regime_breakdown": _regime_breakdown(compressed),
        "recipe_budget": args.recipe_budget,
        "within_recipe_budget": compressed_count <= args.recipe_budget,
        "fault_oracle": fault,
        "catalog_id": compressed.identifier,
        "elapsed_seconds": time.monotonic() - started,
    }
    (args.artifact_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=sorted(ARCHITECTURE_PRESETS), required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--min-n-exec",
        type=int,
        default=1,
        help="inclusive scalar n_exec lower bound",
    )
    parser.add_argument(
        "--max-n-exec",
        type=int,
        help="override the preset's inclusive scalar n_exec upper bound",
    )
    parser.add_argument(
        "--profile-shards",
        type=Path,
        nargs="+",
        help="merge raw observations from disjoint completed artifact directories",
    )
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--recipe-budget", type=int, default=50)
    parser.add_argument(
        "--fault-occurrences",
        choices=("roles", "none"),
        default="none",
    )
    args = parser.parse_args()
    if args.min_n_exec < 1:
        parser.error("--min-n-exec must be positive")
    if args.max_n_exec is not None and args.max_n_exec < 1:
        parser.error("--max-n-exec must be positive")
    if args.max_n_exec is not None and args.min_n_exec > args.max_n_exec:
        parser.error("--min-n-exec cannot exceed --max-n-exec")
    if args.repetitions < 1:
        parser.error("--repetitions must be positive")
    if args.recipe_budget < 1:
        parser.error("--recipe-budget must be positive")
    evaluate(args)


if __name__ == "__main__":
    main()
