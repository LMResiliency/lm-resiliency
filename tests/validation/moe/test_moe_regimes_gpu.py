"""CUDA smoke tests for MoE execution-regime profiling."""

from __future__ import annotations

import pytest
import torch

from lm_resiliency.detection.moe_regimes import (
    CTASemantics,
    ExecutionFingerprint,
    ExecutionHints,
    ExecutionObservation,
    KernelLaunch,
    MoEExecutionEnvironment,
    ProfileLocation,
    ProfileRequest,
    TorchCudaExecutionProfiler,
    current_moe_environment,
    discover_execution_regimes,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]

_DERIVATION_DIGEST = "a" * 64
_QUALIFICATION_DIGEST = "b" * 64


def _smoke_environment() -> MoEExecutionEnvironment:
    return current_moe_environment(
        backend="torch-mm-smoke",
        backend_version=torch.__version__,
        model={"hidden_size": 512, "intermediate_size": 512},
        precision="fp16",
        parallelism={"dp": 1, "ep": 1, "tp": 1},
        extra={
            "container_digest": "gpu-smoke-test",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "model_commit": "gpu-smoke-test",
            "overlap": "none",
            "precision_recipe": "native-fp16",
            "workspace_policy": "PyTorch default",
        },
    )


def _semantics(mapping_class, role_counts):
    return CTASemantics.create(
        mapping_class=mapping_class,
        role_counts=role_counts,
        qualification_role_counts=role_counts,
        derivation_source="generated_metadata",
        derivation_digest=_DERIVATION_DIGEST,
        qualification_source="instrumented",
        qualification_digest=_QUALIFICATION_DIGEST,
    )


def _observations(request, fingerprint):
    return tuple(ExecutionObservation(request=request, fingerprint=fingerprint) for _ in range(3))


def test_cuda_profiler_uses_both_real_shape_fingerprints():
    torch.manual_seed(17)
    device = torch.device("cuda:0")
    inputs = {
        n_exec: (
            torch.randn((n_exec, 512), device=device, dtype=torch.float16),
            torch.randn((512, 512), device=device, dtype=torch.float16),
        )
        for n_exec in (512, 1536)
    }
    outputs = {}

    def workload(request):
        left, right = inputs[request.n_exec]
        outputs[request.n_exec] = torch.mm(left, right)

    def hints(_request, launches):
        return ExecutionHints(
            algorithm_ids=("torch-mm-smoke",),
            tile_shapes=("backend-opaque-smoke",),
            tail_path="aligned",
            workspace_bytes=0,
            pressure_class="smoke-only",
            overlap_class="none",
            persistent_work_items=(0,) * len(launches),
            cta_semantics=tuple(
                _semantics(
                    "opaque-gemm-tile-grid",
                    {"gemm-tile": launch.grid[0] * launch.grid[1] * launch.grid[2]},
                )
                for launch in launches
                if launch.grid is not None
            ),
        )

    profiler = TorchCudaExecutionProfiler(workload, hints=hints, warmup=3, device=device)
    location = ProfileLocation("smoke", 0, 0)
    first_request = ProfileRequest(512, location)
    second_request = ProfileRequest(1536, location)

    first_runs = tuple(profiler(first_request) for _ in range(5))
    second_runs = tuple(profiler(second_request) for _ in range(3))
    assert len({fingerprint.identifier for fingerprint in first_runs}) == 1
    assert len({fingerprint.identifier for fingerprint in second_runs}) == 1
    first = first_runs[0]
    second = second_runs[0]
    assert first.kernels and second.kernels
    assert all(
        kernel.name and kernel.grid and kernel.block
        for fingerprint in (first, second)
        for kernel in fingerprint.kernels
    )
    assert second.identifier != first.identifier

    observations = _observations(first_request, first) + _observations(second_request, second)
    exact = discover_execution_regimes(
        observations,
        environment=_smoke_environment(),
        equivalence_policy="exact_launch",
        expected_requests=(first_request, second_request),
    )
    compressed = discover_execution_regimes(
        observations,
        environment=_smoke_environment(),
        equivalence_policy="plan_and_pressure",
        expected_requests=(first_request, second_request),
    )

    assert len(exact.regimes) == 2
    for request, fingerprint in ((first_request, first), (second_request, second)):
        regimes = compressed.regimes_for_shape(request.n_exec)
        assert len(regimes) == 1
        assert regimes[0].fingerprint_for_shape(request.n_exec) == fingerprint


def test_synthetic_cta_role_dominance_compresses_to_qualified_counts():
    location = ProfileLocation("synthetic", 0, 0)
    first_request = ProfileRequest(512, location)
    second_request = ProfileRequest(1536, location)

    def fingerprint(grid_x, role_counts):
        return ExecutionFingerprint.create(
            kernels=[KernelLaunch("synthetic-gemm", grid=(grid_x, 1, 1), block=(256, 1, 1))],
            algorithm_ids=("synthetic-algo",),
            tile_shapes=("128x128x64",),
            tail_path="aligned",
            workspace_bytes=0,
            pressure_class="saturated",
            overlap_class="none",
            persistent_work_items=(0,),
            cta_semantics=(_semantics("x=output-tile", role_counts),),
        )

    first = fingerprint(8, {"interior": 8})
    second = fingerprint(16, {"boundary": 1, "interior": 15})
    observations = _observations(first_request, first) + _observations(second_request, second)

    compressed = discover_execution_regimes(
        observations,
        environment=_smoke_environment(),
        equivalence_policy="plan_and_pressure",
        expected_requests=(first_request, second_request),
    )

    assert len(compressed.regimes) == 1
    assert compressed.regimes[0].representatives == (1536,)


def test_native_grouped_gemm_profiles_exact_forward_backward_catalog():
    """Profile a jagged MoE expert stage without claiming opaque CTA semantics."""
    torch.manual_seed(29)
    device = torch.device("cuda:0")
    hidden = 128
    intermediate = 256
    experts = 4
    inputs = {}
    for n_exec in (512, 1536):
        counts = [n_exec // experts] * experts
        counts[-1] += n_exec - sum(counts)
        offsets = torch.tensor(
            [sum(counts[: index + 1]) for index in range(experts)],
            dtype=torch.int32,
            device=device,
        )
        inputs[n_exec] = {
            # grouped_mm requires offs[-1] to be below the physical row count.
            "activation": torch.randn(
                n_exec + 1,
                hidden,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            "first_weight": torch.randn(
                experts,
                hidden,
                intermediate,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            "second_weight": torch.randn(
                experts,
                intermediate,
                hidden,
                device=device,
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            "offsets": offsets,
        }

    def workload(request):
        tensors = inputs[request.n_exec]
        for name in ("activation", "first_weight", "second_weight"):
            tensors[name].grad = None
        first = torch.nn.functional.grouped_mm(
            tensors["activation"],
            tensors["first_weight"],
            offs=tensors["offsets"],
        )
        activated = torch.nn.functional.gelu(first)
        output = torch.nn.functional.grouped_mm(
            activated,
            tensors["second_weight"],
            offs=tensors["offsets"],
        )
        output[: request.n_exec].float().square().mean().backward()

    def hints(_request, launches):
        return ExecutionHints(
            algorithm_ids=("torch-native-grouped-mm-opaque",) * len(launches),
            tile_shapes=("backend-opaque",) * len(launches),
            tail_path="backend-opaque",
            workspace_bytes=0,
            pressure_class="unqualified",
            overlap_class="none",
            persistent_work_items=(),
            cta_semantics=(),
        )

    profiler = TorchCudaExecutionProfiler(workload, hints=hints, warmup=3, device=device)
    location = ProfileLocation("native-grouped-forward-backward", 0, 0)
    requests = tuple(ProfileRequest(n_exec, location) for n_exec in inputs)
    observations = []
    for request in requests:
        # Prime autograd's reusable gradient buffers before comparing trace order.
        profiler(request)
        fingerprints = tuple(profiler(request) for _ in range(5))
        assert len({fingerprint.identifier for fingerprint in fingerprints}) == 1
        assert fingerprints[0].kernels
        assert sum("cutlass" in kernel.name.lower() for kernel in fingerprints[0].kernels) >= 2
        observations.extend(_observations(request, fingerprints[0]))

    environment = current_moe_environment(
        backend="torch-native-grouped-mm",
        backend_version=torch.__version__,
        model={
            "hidden_size": hidden,
            "intermediate_size": intermediate,
            "local_experts": experts,
        },
        precision="bf16",
        parallelism={"dp": 1, "ep": 1, "tp": 1},
        extra={
            "container_digest": "host-validation-environment",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "model_commit": "validation-only-random-weights",
            "overlap": "none",
            "precision_recipe": "native-bf16",
            "workspace_policy": "PyTorch default",
        },
    )
    exact = discover_execution_regimes(
        observations,
        environment=environment,
        equivalence_policy="exact_launch",
        expected_requests=requests,
    )
    conservative = discover_execution_regimes(
        observations,
        environment=environment,
        equivalence_policy="plan_and_pressure",
        expected_requests=requests,
    )

    assert exact.cycle_size == len(requests)
    assert conservative.cycle_size == exact.cycle_size
    assert len(conservative.regimes) == len(exact.regimes)
