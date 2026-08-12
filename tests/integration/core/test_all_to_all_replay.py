"""Four-GPU integration test for representative AllToAll replay.

Run:
    torchrun --standalone --nproc_per_node=4 \
        tests/integration/core/test_all_to_all_replay.py
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency import ReplayHarnessConfig
from lm_resiliency.detection.all_to_all_replay import (
    AllToAllReplayExecutor,
    AllToAllReplayRecipe,
    BalancedAndPermutationPolicy,
    TensorReplaySpec,
)
from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.experimental import ModelReplayHarness

_OBSERVED = (
    (2, 1, 0, 1),
    (0, 2, 1, 1),
    (1, 0, 2, 1),
    (1, 1, 0, 2),
)


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("AllToAll replay integration requires CUDA")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dtype = torch.bfloat16
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 4

    input_splits = _OBSERVED[rank]
    output_splits = tuple(row[rank] for row in _OBSERVED)
    element_size = torch.empty((), dtype=dtype).element_size()
    input_spec = TensorReplaySpec(
        shape=(sum(input_splits), 3),
        dtype=dtype,
        numel=sum(input_splits) * 3,
        element_size=element_size,
    )
    output_spec = TensorReplaySpec(
        shape=(sum(output_splits), 3),
        dtype=dtype,
        numel=sum(output_splits) * 3,
        element_size=element_size,
    )
    recipe = AllToAllReplayRecipe(
        sequence=0,
        collective="all_to_all_single",
        group_ranks=tuple(range(world_size)),
        inputs=(input_spec,),
        outputs=(output_spec,),
        input_split_sizes=input_splits,
        output_split_sizes=output_splits,
        async_op=False,
        group=dist.group.WORLD,
    )
    payload_bytes = 4 * 3 * element_size
    outcomes = AllToAllReplayExecutor(device).replay(
        recipe,
        BalancedAndPermutationPolicy(
            max_payload_bytes_per_rank=payload_bytes,
        ),
    )

    assert [outcome.matrix.name for outcome in outcomes] == [
        "balanced",
        "cyclic_permutation_1",
    ]
    assert all(outcome.correct for outcome in outcomes)
    assert all(outcome.input_bytes == payload_bytes for outcome in outcomes)
    assert all(outcome.output_bytes == payload_bytes for outcome in outcomes)

    layer = nn.Linear(3, 3, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        layer.weight.copy_(torch.eye(3, device=device, dtype=dtype))
    model = nn.Sequential(layer)
    harness = ModelReplayHarness(
        model,
        group=dist.group.WORLD,
        nccl_group=dist.group.WORLD,
        device=device,
        config=ReplayHarnessConfig(
            check_interval=0,
            rotate_layers=False,
            scale_factors=[],
            enable_temporal=False,
            all_to_all_policy=BalancedAndPermutationPolicy(
                max_payload_bytes_per_rank=payload_bytes,
            ),
        ),
        layers=[layer],
    )
    model(torch.arange(12, device=device, dtype=dtype).reshape(4, 3))
    training_input = torch.arange(
        sum(input_splits) * 3,
        device=device,
        dtype=dtype,
    ).reshape(sum(input_splits), 3)
    training_output = torch.empty(
        (sum(output_splits), 3),
        device=device,
        dtype=dtype,
    )
    dist.all_to_all_single(
        training_output,
        training_input,
        output_split_sizes=list(output_splits),
        input_split_sizes=list(input_splits),
    )
    result = harness.check()
    all_to_all_results = {
        name: evidence
        for name, evidence in result.c3_results.items()
        if name.startswith("all_to_all.")
    }
    assert "all_to_all" in result.checked_recipe_ids
    assert all_to_all_results
    assert all(evidence.status is C3Status.AGREE for evidence in all_to_all_results.values())
    assert any(
        sample.collective.startswith("all_to_all_replay.") for sample in result.collective_timings
    )
    harness.remove_hooks()

    dist.barrier()
    if rank == 0:
        print(
            "SCOUT ALLTOALL REPLAY OK: automatic capture reconstructed uneven "
            "metadata and verified balanced and cyclic-permutation traffic on "
            f"{device.type}."
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
