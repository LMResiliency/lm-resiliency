"""Qualify scalar SCOUT MoE regime compression on one or two A100 hosts.

Profile mode catalogs one representative expert GEMM over an exhaustive scalar
``n_exec`` range:

    python tests/validation/moe/validate_moe_regime_compression.py profile \
        --artifact-dir /tmp/scout-moe-compression

Distributed mode expands each scalar recipe uniformly across local experts:

    torchrun --nproc_per_node=8 tests/validation/moe/validate_moe_regime_compression.py \
        distributed --catalog /tmp/scout-moe-compression/catalog-compressed.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import Any

_REPOSITORY = Path(__file__).parents[3]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

import torch
import torch.distributed as dist
import triton

from lm_resiliency import (
    GroupedExpertMaterializer,
    ReplayHarnessConfig,
    ReplayWorkload,
)
from lm_resiliency.detection.moe_regimes import (
    CatalogEnvironmentMismatch,
    ExecutionFingerprint,
    ExecutionObservation,
    KernelLaunch,
    MoEExecutionEnvironment,
    MoERegimeCatalog,
    ProfileLocation,
    ProfileRequest,
    TorchCudaExecutionProfiler,
    current_moe_environment,
    discover_execution_regimes,
    save_observations,
    save_profile_requests,
)
from lm_resiliency.experimental import ModelReplayHarness
from tests.support.triton_grouped_expert import InstrumentedGroupedExperts

_LOCAL_EXPERTS = 4
_HIDDEN = 128
_OUTPUT = 128
_MAX_N_EXEC = 3457
_EXECUTION_CLASS = "triton-persistent-expert-forward-backward"


def _qualification_n_exec_values() -> tuple[int, ...]:
    """Return the exhaustive admitted per-expert physical row-count range."""
    return tuple(range(1, _MAX_N_EXEC + 1))


def _source_identity() -> str:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=_REPOSITORY,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    import hashlib

    digest = hashlib.sha256()
    for relative in (
        "lm_resiliency/detection/moe_regimes.py",
        "lm_resiliency/detection/replay_shapes.py",
        "tests/support/triton_grouped_expert.py",
        "tests/validation/moe/validate_moe_regime_compression.py",
    ):
        digest.update((_REPOSITORY / relative).read_bytes())
    return f"{commit}-sources-{digest.hexdigest()[:16]}"


def _environment() -> MoEExecutionEnvironment:
    return current_moe_environment(
        backend="triton-persistent-grouped-gemm",
        backend_version=triton.__version__,
        model={
            "hidden_size": _HIDDEN,
            "expert_output_size": _OUTPUT,
            "local_experts_metadata_only": _LOCAL_EXPERTS,
            "profiled_experts": 1,
            "minimum_per_expert_n_exec": 1,
            "maximum_per_expert_n_exec": _MAX_N_EXEC,
        },
        precision="bf16",
        parallelism={"dp_replicas": 16, "ep": 1, "expert_tp": 1},
        extra={
            "container_digest": "aws-a100-cuda13-validation-environment",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "model_commit": _source_identity(),
            "overlap": "none",
            "precision_recipe": "native-bf16",
            "workspace_policy": "zero-workspace",
        },
    )


def _inputs(
    n_exec: int,
    *,
    num_experts: int,
    device: torch.device,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    rows = n_exec * num_experts
    tokens = torch.randn(
        rows,
        _HIDDEN,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    counts = torch.full(
        (num_experts,),
        n_exec,
        device=device,
        dtype=torch.int32,
    )
    grad_output = torch.randn(
        rows,
        _OUTPUT,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    return tokens, counts, grad_output


def _fingerprint(
    launches: tuple[KernelLaunch, ...],
    backend: InstrumentedGroupedExperts,
    n_exec: int,
    *,
    role_count_updates: dict[str, dict[str, int]] | None = None,
) -> ExecutionFingerprint:
    shape = (n_exec,) * backend.num_experts
    hints = backend.execution_hints(
        shape,
        kernel_count=len(launches),
        role_count_updates=role_count_updates,
    )
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


def _kineto_launch_template(
    backend: InstrumentedGroupedExperts,
    device: torch.device,
) -> tuple[KernelLaunch, ...]:
    n_exec = 65
    tokens, counts, grad_output = _inputs(
        n_exec,
        num_experts=backend.num_experts,
        device=device,
        seed=101,
    )
    request = ProfileRequest(
        n_exec,
        ProfileLocation("qualification", 0, 0, _EXECUTION_CLASS),
    )

    def workload(_request: ProfileRequest) -> None:
        backend.run_forward_backward(tokens, counts, grad_output)

    def hints(profile_request, launches):
        return backend.execution_hints(
            (profile_request.n_exec,) * backend.num_experts,
            kernel_count=len(launches),
        )

    fingerprint = TorchCudaExecutionProfiler(
        workload,
        hints=hints,
        warmup=3,
        device=device,
    )(request)
    if len(fingerprint.kernels) != 3:
        raise RuntimeError(f"expected three Triton kernels, got {len(fingerprint.kernels)}")
    if any(kernel.grid != (backend.num_sms, 1, 1) for kernel in fingerprint.kernels):
        raise RuntimeError(f"unexpected persistent launch grids: {fingerprint.kernels}")
    return fingerprint.kernels


def _validate_numerics(
    backend: InstrumentedGroupedExperts,
    device: torch.device,
) -> int:
    checked = 0
    for index, n_exec in enumerate((1, 31, 32, 33, 128, 129, 512, _MAX_N_EXEC)):
        tokens, counts, grad_output = _inputs(
            n_exec,
            num_experts=backend.num_experts,
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
    device: torch.device,
    launches: tuple[KernelLaunch, ...],
    *,
    repetitions: int,
) -> tuple[tuple[ProfileRequest, ...], tuple[ExecutionObservation, ...]]:
    location = ProfileLocation("qualification", 0, 0, _EXECUTION_CLASS)
    requests = tuple(ProfileRequest(n_exec, location) for n_exec in _qualification_n_exec_values())
    observations = []
    for request_index, request in enumerate(requests):
        tokens, counts, grad_output = _inputs(
            request.n_exec,
            num_experts=backend.num_experts,
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


def _negative_role_challenges(
    backend: InstrumentedGroupedExperts,
    device: torch.device,
    launches: tuple[KernelLaunch, ...],
    environment: MoEExecutionEnvironment,
) -> dict[str, bool]:
    location = ProfileLocation("negative", 0, 0, "negative-role-challenge")
    n_exec_values = (32, 64)
    observations = []
    requests = []
    for index, n_exec in enumerate(n_exec_values):
        request = ProfileRequest(n_exec, location)
        requests.append(request)
        tokens, counts, grad_output = _inputs(
            n_exec,
            num_experts=backend.num_experts,
            device=device,
            seed=3000 + index,
        )
        backend.run_forward_backward(tokens, counts, grad_output)
        torch.cuda.synchronize(device)
        updates = None
        if index == 1:
            role_counts = backend.evidence((n_exec,))[0].declared_role_counts
            roles = sorted(role_counts)
            if len(roles) < 2:
                raise RuntimeError("negative challenge requires at least two forward roles")
            updates = {"forward": dict(role_counts)}
            omitted = roles[0]
            replacement = roles[1]
            updates["forward"][replacement] += updates["forward"].pop(omitted)
        fingerprint = _fingerprint(
            launches,
            backend,
            n_exec,
            role_count_updates=updates,
        )
        observations.extend(
            ExecutionObservation(request=request, fingerprint=fingerprint) for _ in range(3)
        )
    catalog = discover_execution_regimes(
        observations,
        environment=environment,
        equivalence_policy="plan_and_pressure",
        expected_requests=requests,
    )
    omitted_role_fell_back = catalog.cycle_size == len(n_exec_values)

    n_exec = n_exec_values[1]
    tokens, counts, grad_output = _inputs(
        n_exec,
        num_experts=backend.num_experts,
        device=device,
        seed=3100,
    )
    backend.run_forward_backward(tokens, counts, grad_output)
    torch.cuda.synchronize(device)
    role_counts = backend.evidence((n_exec,))[0].declared_role_counts
    wrong_counts = dict(role_counts)
    wrong_counts[next(iter(wrong_counts))] -= 1
    wrong = _fingerprint(
        launches,
        backend,
        n_exec,
        role_count_updates={"forward": wrong_counts},
    )
    wrong_count_rejected_compression = not wrong.cta_semantics[0].is_compression_ready(
        wrong.kernels[0],
        wrong.persistent_work_items[0],
    )
    return {
        "omitted_role_fell_back_to_exact": omitted_role_fell_back,
        "wrong_count_rejected_compression": wrong_count_rejected_compression,
    }


def _coverage_obligations(catalog: MoERegimeCatalog) -> int:
    obligations = 0
    for regime in catalog.regimes:
        for n_exec in regime.n_exec_values:
            target = regime.fingerprint_for_shape(n_exec)
            if not regime.representatives_cover_shape(n_exec):
                raise RuntimeError(f"no representative covers n_exec={n_exec}")
            obligations += sum(
                sum(count for _role, count in semantics.role_counts)
                for semantics in target.cta_semantics
            )
    return obligations


def _fault_oracle(
    backend: InstrumentedGroupedExperts,
    catalog: MoERegimeCatalog,
    device: torch.device,
    *,
    mode: str,
) -> dict[str, int]:
    if mode == "none":
        return {"representatives": 0, "roles": 0, "injections": 0}
    representative_count = 0
    roles_checked = 0
    injections = 0
    for recipe_index, recipe in enumerate(catalog.replay_recipes):
        n_exec = recipe.n_exec
        tokens, counts, grad_output = _inputs(
            n_exec,
            num_experts=backend.num_experts,
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
        representative_count += 1
        for kernel, by_role in backend.role_occurrences((n_exec,)).items():
            for work_items in by_role.values():
                roles_checked += 1
                selected_items: Iterable[int] = work_items[:1] if mode == "roles" else work_items
                for work_item in selected_items:
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
    return {
        "representatives": representative_count,
        "roles": roles_checked,
        "injections": injections,
    }


def profile(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    torch.manual_seed(1729)
    artifact_dir = args.artifact_dir
    artifact_dir.mkdir(parents=True, exist_ok=True)

    backend = InstrumentedGroupedExperts(
        num_experts=1,
        hidden=_HIDDEN,
        output=_OUTPUT,
        device=device,
    )
    numerical_shapes = _validate_numerics(backend, device)
    launches = _kineto_launch_template(backend, device)
    requests, observations = _collect_observations(
        backend,
        device,
        launches,
        repetitions=args.repetitions,
    )
    environment = _environment()
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
        max_replay_recipes=args.max_replay_recipes,
    )

    (artifact_dir / "environment.json").write_text(
        json.dumps(environment.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    save_profile_requests(requests, artifact_dir / "manifest.json")
    save_observations(observations, artifact_dir / "observations.jsonl")
    exact.save(artifact_dir / "catalog-exact.json")
    compressed.save(artifact_dir / "catalog-compressed.json")

    loaded = MoERegimeCatalog.load(artifact_dir / "catalog-compressed.json")
    if loaded.identifier != compressed.identifier:
        raise RuntimeError("compressed catalog checksum changed after serialization")
    drift_rejected = False
    drifted = MoEExecutionEnvironment.from_mapping({**environment.to_dict(), "precision": "fp16"})
    try:
        loaded.validate_environment(drifted)
    except CatalogEnvironmentMismatch:
        drift_rejected = True
    if not drift_rejected:
        raise RuntimeError("catalog did not reject precision drift")

    negative = _negative_role_challenges(backend, device, launches, environment)
    if not all(negative.values()):
        raise RuntimeError(f"negative role challenge failed: {negative}")
    fault = _fault_oracle(backend, compressed, device, mode=args.fault_occurrences)
    exact_count = exact.cycle_size
    compressed_count = compressed.cycle_size
    summary: dict[str, Any] = {
        "result": "PASS",
        "gpu": torch.cuda.get_device_name(device),
        "sm_count": backend.num_sms,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "triton": triton.__version__,
        "profiled_experts": backend.num_experts,
        "local_experts_metadata_only": _LOCAL_EXPERTS,
        "minimum_per_expert_n_exec": 1,
        "maximum_per_expert_n_exec": _MAX_N_EXEC,
        "physical_shapes": len(requests),
        "manifest_exhaustive": True,
        "repetitions_per_shape": args.repetitions,
        "observations": len(observations),
        "numerical_shapes": numerical_shapes,
        "exact_recipes": exact_count,
        "compressed_recipes": compressed_count,
        "recipe_reduction_fraction": 1.0 - compressed_count / exact_count,
        "compression_ratio": exact_count / compressed_count,
        "regimes": len(compressed.regimes),
        "coverage_obligations": _coverage_obligations(compressed),
        "fault_oracle": fault,
        "negative_challenges": negative,
        "environment_drift_rejected": drift_rejected,
        "kineto_kernels": [
            {"name": kernel.name, "grid": kernel.grid, "block": kernel.block} for kernel in launches
        ],
        "catalog_id": compressed.identifier,
    }
    (artifact_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _assert_all(condition: bool, message: str) -> None:
    value = torch.tensor(int(condition), device="cuda", dtype=torch.int64)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if value.item() != 1:
        raise AssertionError(message)


def distributed(args: argparse.Namespace) -> None:
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    catalog = MoERegimeCatalog.load(args.catalog)

    torch.manual_seed(1729)
    backend = InstrumentedGroupedExperts(
        num_experts=_LOCAL_EXPERTS,
        hidden=_HIDDEN,
        output=_OUTPUT,
        device=device,
    )
    with torch.no_grad():
        for parameter in backend.parameters():
            dist.broadcast(parameter, src=0)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    workload = ReplayWorkload.from_moe_catalog(
        catalog,
        replay_modules=[backend],
        materializer=GroupedExpertMaterializer(),
    )
    harness = ModelReplayHarness(
        backend,
        group=gloo_group,
        nccl_group=nccl_group,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            capture_inputs_by_value=True,
            workload=workload,
            compare_parameter_state=False,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
        ),
    )

    source_n_exec = 64
    tokens, counts, grad_output = _inputs(
        source_n_exec,
        num_experts=_LOCAL_EXPERTS,
        device=device,
        seed=9000,
    )
    output = backend(tokens.requires_grad_(True), counts)
    output.backward(grad_output)
    healthy = harness.check_shape_cycle()
    _assert_all(healthy.completed_shape_cycle, str(healthy.checked_shapes))
    _assert_all(
        len(healthy.checked_shapes) == catalog.cycle_size,
        str(healthy.checked_shapes),
    )
    _assert_all(not any(healthy.sdc_bitmap), str(healthy.sdc_source_bitmaps))

    victim = world_size - 1
    if rank == victim:
        backend.set_fault("forward", 0)
    corrupted = harness.check_shape_cycle()
    expected = [0] * world_size
    expected[victim] = 1
    _assert_all(
        not corrupted.completed_shape_cycle and len(corrupted.checked_shapes) == 1,
        str(corrupted.checked_shapes),
    )
    _assert_all(corrupted.sdc_bitmap == expected, str(corrupted.sdc_source_bitmaps))

    harness.remove_hooks()
    dist.barrier()
    if rank == 0:
        print(
            "PASS: scalar MoE catalog replayed all "
            f"{catalog.cycle_size} representatives as "
            f"[n_exec] x {_LOCAL_EXPERTS} across {world_size} A100s and "
            f"localized rank {victim} on the next scheduled representative",
            flush=True,
        )
    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile_parser = subparsers.add_parser("profile")
    profile_parser.add_argument("--artifact-dir", type=Path, required=True)
    profile_parser.add_argument("--device", type=int, default=0)
    profile_parser.add_argument("--repetitions", type=int, default=3)
    profile_parser.add_argument("--max-replay-recipes", type=int, default=50)
    profile_parser.add_argument(
        "--fault-occurrences",
        choices=("all", "roles", "none"),
        default="all",
    )
    distributed_parser = subparsers.add_parser("distributed")
    distributed_parser.add_argument("--catalog", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "profile":
        if args.repetitions < 1:
            parser.error("--repetitions must be positive")
        profile(args)
    else:
        distributed(args)


if __name__ == "__main__":
    main()
