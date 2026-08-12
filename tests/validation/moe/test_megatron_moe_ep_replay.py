"""Validate SCOUT replay with Megatron Core EP All-to-All across two hosts.

Run on two 8-GPU hosts with matching torchrun rendezvous settings.
Transformer Engine and Megatron Core are optional runtime dependencies.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPOSITORY = Path(__file__).resolve().parents[3]
if str(_REPOSITORY) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY))

import torch
import torch.distributed as dist

from lm_resiliency import (
    GroupedExpertMaterializer,
    ReplayHarnessConfig,
    ReplayWorkload,
)
from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.experimental import ModelReplayHarness
from tests.validation.moe.test_megatron_moe_replay import (
    _build_grouped_experts,
    _build_routed_moe,
    _singleton_nccl_group,
    _synchronize_parameters,
)


def _new_selected_group(
    rank: int,
    memberships: list[list[int]],
    *,
    backend: str,
) -> tuple[dist.ProcessGroup, list[int]]:
    selected_group = None
    selected_members = None
    for members in memberships:
        group = dist.new_group(members, backend=backend)
        if rank in members:
            selected_group = group
            selected_members = members
    assert selected_group is not None
    assert selected_members is not None
    return selected_group, selected_members


def _assert_world(condition: bool, message: str) -> None:
    passed = torch.tensor(int(condition), device="cuda", dtype=torch.int64)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    assert passed.item() == 1, message


def _harness(
    module,
    workload: ReplayWorkload,
    *,
    gloo_group: dist.ProcessGroup,
    nccl_group: dist.ProcessGroup,
    local_rank: int,
) -> ModelReplayHarness:
    return ModelReplayHarness(
        module,
        group=gloo_group,
        nccl_group=nccl_group,
        device=torch.device("cuda", local_rank),
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


def main() -> None:
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    assert world_size == 16, "this topology requires two 8-GPU hosts"

    # Each four-rank EP replica spans both hosts, so dispatch exercises the
    # inter-node transport. Four matching expert partitions form SCOUT peers.
    ep_memberships = [
        [0, 1, 8, 9],
        [2, 3, 10, 11],
        [4, 5, 12, 13],
        [6, 7, 14, 15],
    ]
    peer_memberships = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [8, 10, 12, 14],
        [9, 11, 13, 15],
    ]
    ep_group, ep_members = _new_selected_group(rank, ep_memberships, backend="nccl")
    gloo_group, peer_members = _new_selected_group(rank, peer_memberships, backend="gloo")
    nccl_group, _ = _new_selected_group(rank, peer_memberships, backend="nccl")
    singleton_group = _singleton_nccl_group(rank, world_size)

    torch.manual_seed(1701)
    routed_moe = _build_routed_moe(
        singleton_group,
        ep_group=ep_group,
        expert_model_parallel_size=4,
        num_moe_experts=8,
        token_dispatcher="alltoall",
    )
    _synchronize_parameters(routed_moe, nccl_group, src=peer_members[0])
    routed_harness = _harness(
        routed_moe,
        ReplayWorkload.dense([routed_moe]),
        gloo_group=gloo_group,
        nccl_group=nccl_group,
        local_rank=local_rank,
    )
    hidden_states = torch.randn(
        32,
        2,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    routed_output, _ = routed_moe(hidden_states)
    routed_output.float().square().mean().backward()
    routed_result = routed_harness.check()
    _assert_world(not any(routed_result.sdc_bitmap), str(routed_result.sdc_source_bitmaps))
    _assert_world(
        {"output", "input_gradient", "parameter_gradient"}.issubset(
            routed_result.sdc_source_bitmaps
        ),
        str(routed_result.sdc_source_bitmaps),
    )
    all_to_all_results = {
        name: result
        for name, result in routed_result.c3_results.items()
        if name.startswith("all_to_all.")
    }
    _assert_world("all_to_all" in routed_result.checked_recipe_ids, "AllToAll replay missing")
    _assert_world(bool(all_to_all_results), "AllToAll C3 evidence missing")
    _assert_world(
        all(result.status is C3Status.AGREE for result in all_to_all_results.values()),
        str(all_to_all_results),
    )
    _assert_world(
        any(
            sample.collective.startswith("all_to_all_replay.")
            for sample in routed_result.collective_timings
        ),
        str(routed_result.collective_timings),
    )
    routed_harness.remove_hooks()

    torch.manual_seed(1729)
    experts = _build_grouped_experts(
        singleton_group,
        ep_group=ep_group,
        expert_model_parallel_size=4,
        num_moe_experts=8,
    )
    _synchronize_parameters(experts, nccl_group, src=peer_members[0])
    workload = ReplayWorkload.from_shapes(
        [(128,), (512,)],
        replay_modules=[experts],
        materializer=GroupedExpertMaterializer(),
        source_id="megatron-te-ep4-grouped-mlp",
    )
    harness = _harness(
        experts,
        workload,
        gloo_group=gloo_group,
        nccl_group=nccl_group,
        local_rank=local_rank,
    )

    counts = torch.tensor([112, 144], device="cuda", dtype=torch.int64)
    tokens = torch.randn(
        256,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    probabilities = torch.rand(256, device="cuda", dtype=torch.bfloat16)
    output, _ = experts(tokens, counts, probabilities)
    output.float().square().mean().backward()

    healthy = harness.check_shape_cycle()
    _assert_world(healthy.completed_shape_cycle, str(healthy.checked_shapes))
    _assert_world(healthy.checked_shapes == [(128,), (512,)], str(healthy.checked_shapes))
    _assert_world(not any(healthy.sdc_bitmap), str(healthy.sdc_source_bitmaps))

    victim = world_size - 1
    fault_parameter = next(experts.parameters())
    if rank == victim:
        with torch.no_grad():
            fault_parameter.add_(2)

    parameter_sum = fault_parameter.detach().float().sum()
    peer_parameter_sums = [torch.zeros_like(parameter_sum) for _ in peer_members]
    dist.all_gather(peer_parameter_sums, parameter_sum, group=nccl_group)
    parameter_differs = any(
        not torch.equal(peer_parameter_sums[0], peer_sum) for peer_sum in peer_parameter_sums[1:]
    )
    _assert_world(
        parameter_differs == (victim in peer_members),
        f"peers={peer_members}: parameter sums={peer_parameter_sums}",
    )

    expected = [0] * len(peer_members)
    if victim in peer_members:
        expected[peer_members.index(victim)] = 1
    corrupted_small = harness.check()
    _assert_world(corrupted_small.replay_shape == (128,), str(corrupted_small.replay_shape))
    _assert_world(
        corrupted_small.sdc_bitmap == expected,
        f"peers={peer_members}: {corrupted_small.sdc_source_bitmaps}",
    )

    corrupted_large = harness.check()
    _assert_world(corrupted_large.replay_shape == (512,), str(corrupted_large.replay_shape))
    _assert_world(
        corrupted_large.sdc_bitmap == expected,
        f"peers={peer_members}: {corrupted_large.sdc_source_bitmaps}",
    )

    harness.remove_hooks()
    dist.barrier()
    if rank == 0:
        print(
            "PASS: four inter-node EP=4 All-to-All replicas stayed clean, "
            "completed the grouped-expert shape cycle, and localized rank 15",
            flush=True,
        )
        print(f"EP replica 0 ranks: {ep_members}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
