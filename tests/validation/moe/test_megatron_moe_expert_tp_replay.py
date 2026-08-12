"""Validate SCOUT replay with Megatron EP and expert tensor parallelism.

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

    # Rank order within each model group is [ep0/tp0, ep0/tp1,
    # ep1/tp0, ep1/tp1]. EP communication crosses the two hosts.
    model_memberships = [
        [0, 1, 8, 9],
        [2, 3, 10, 11],
        [4, 5, 12, 13],
        [6, 7, 14, 15],
    ]
    ep_memberships = [
        [0, 8],
        [1, 9],
        [2, 10],
        [3, 11],
        [4, 12],
        [5, 13],
        [6, 14],
        [7, 15],
    ]
    expert_tp_memberships = [
        [0, 1],
        [8, 9],
        [2, 3],
        [10, 11],
        [4, 5],
        [12, 13],
        [6, 7],
        [14, 15],
    ]
    peer_memberships = [
        [0, 2, 4, 6],
        [1, 3, 5, 7],
        [8, 10, 12, 14],
        [9, 11, 13, 15],
    ]
    tp_ep_group, model_members = _new_selected_group(rank, model_memberships, backend="nccl")
    ep_group, _ = _new_selected_group(rank, ep_memberships, backend="nccl")
    expert_tp_group, _ = _new_selected_group(rank, expert_tp_memberships, backend="nccl")
    gloo_group, peer_members = _new_selected_group(rank, peer_memberships, backend="gloo")
    nccl_group, _ = _new_selected_group(rank, peer_memberships, backend="nccl")
    singleton_group = _singleton_nccl_group(rank, world_size)

    torch.manual_seed(1801)
    routed_moe = _build_routed_moe(
        singleton_group,
        ep_group=ep_group,
        expt_tp_group=expert_tp_group,
        tp_ep_group=tp_ep_group,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
        num_moe_experts=4,
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
    routed_harness.remove_hooks()

    torch.manual_seed(1829)
    experts = _build_grouped_experts(
        singleton_group,
        ep_group=ep_group,
        expt_tp_group=expert_tp_group,
        expert_model_parallel_size=2,
        expert_tensor_parallel_size=2,
        num_moe_experts=4,
    )
    _synchronize_parameters(experts, nccl_group, src=peer_members[0])
    workload = ReplayWorkload.from_shapes(
        [(128,), (512,)],
        replay_modules=[experts],
        materializer=GroupedExpertMaterializer(),
        source_id="megatron-te-ep2-expert-tp2-grouped-mlp",
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
    if rank == victim:
        with torch.no_grad():
            next(experts.parameters()).add_(2)

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
            "PASS: EP=2 and expert-TP=2 routed MoE stayed clean, completed "
            "the grouped-expert shape cycle, and localized rank 15",
            flush=True,
        )
        print(f"Model replica 0 ranks: {model_members}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
