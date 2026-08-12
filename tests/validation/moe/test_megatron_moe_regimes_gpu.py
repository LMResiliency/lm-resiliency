"""Production-backend regime discovery for Megatron Core TEGroupedMLP."""

from __future__ import annotations

import os

import pytest
import torch
import torch.distributed as dist

from lm_resiliency.detection.moe_regimes import (
    ExecutionHints,
    ExecutionObservation,
    ProfileLocation,
    ProfileRequest,
    TorchCudaExecutionProfiler,
    current_moe_environment,
    discover_execution_regimes,
)
from tests.validation.moe.test_megatron_moe_replay import (
    _build_grouped_experts,
    _singleton_nccl_group,
)

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.megatron,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
]


@pytest.fixture(scope="module")
def grouped_experts():
    if dist.is_initialized():
        assert dist.get_world_size() == 1
        initialized_here = False
    else:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29651")
        dist.init_process_group("nccl", rank=0, world_size=1)
        initialized_here = True
    torch.cuda.set_device(0)
    torch.manual_seed(1729)
    experts = _build_grouped_experts(_singleton_nccl_group(0, 1))
    yield experts
    if initialized_here:
        dist.destroy_process_group()


def test_megatron_te_grouped_mlp_builds_stable_exact_catalog(grouped_experts):
    import transformer_engine

    inputs = {}
    for n_exec in (128, 512):
        inputs[n_exec] = (
            torch.randn(
                n_exec,
                128,
                device="cuda",
                dtype=torch.bfloat16,
                requires_grad=True,
            ),
            torch.full((4,), n_exec // 4, device="cuda", dtype=torch.int64),
            torch.rand(n_exec, device="cuda", dtype=torch.bfloat16),
        )

    def workload(request):
        tokens, counts, probabilities = inputs[request.n_exec]
        tokens.grad = None
        for parameter in grouped_experts.parameters():
            parameter.grad = None
        output, _ = grouped_experts(tokens, counts, probabilities)
        output.float().square().mean().backward()

    def hints(_request, launches):
        return ExecutionHints(
            algorithm_ids=("transformer-engine-opaque",) * len(launches),
            tile_shapes=("backend-opaque",) * len(launches),
            tail_path="backend-opaque",
            workspace_bytes=0,
            pressure_class="unqualified",
            overlap_class="none",
        )

    profiler = TorchCudaExecutionProfiler(
        workload,
        hints=hints,
        warmup=3,
        device=torch.device("cuda"),
    )
    location = ProfileLocation("megatron-te-grouped-forward-backward", 0, 0)
    requests = tuple(ProfileRequest(n_exec, location) for n_exec in inputs)
    observations = []
    fingerprints = {}
    for request in requests:
        profiler(request)
        repeated = tuple(profiler(request) for _ in range(5))
        assert len({fingerprint.identifier for fingerprint in repeated}) == 1
        fingerprints[request.n_exec] = repeated[0]
        assert len(repeated[0].kernels) >= 8
        assert (
            sum(
                "gemm" in kernel.name.lower() or "cutlass" in kernel.name.lower()
                for kernel in repeated[0].kernels
            )
            >= 8
        )
        observations.extend(
            ExecutionObservation(request=request, fingerprint=repeated[0]) for _ in range(3)
        )

    assert fingerprints[128].identifier != fingerprints[512].identifier
    environment = current_moe_environment(
        backend="megatron-core-transformer-engine-grouped-mlp",
        backend_version=transformer_engine.__version__,
        model={
            "hidden_size": 128,
            "intermediate_size": 256,
            "local_experts": 4,
            "top_k": 2,
        },
        precision="bf16",
        parallelism={"dp": 1, "ep": 1, "tp": 1},
        extra={
            "container_digest": "aws-pytorch-cuda13-vendor-environment",
            "cublas": f"PyTorch CUDA {torch.version.cuda}",
            "cuda_graphs": False,
            "model_commit": "validation-only-random-weights",
            "overlap": "none",
            "precision_recipe": "native-bf16",
            "workspace_policy": "Transformer Engine default",
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

    assert exact.cycle_size == 2
    assert conservative.cycle_size == exact.cycle_size
    assert len(conservative.regimes) == len(exact.regimes) == 2
