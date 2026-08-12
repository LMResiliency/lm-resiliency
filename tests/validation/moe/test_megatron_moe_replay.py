"""Validate SCOUT shape replay against Megatron Core's production TEGroupedMLP.

Run on one node:
    torchrun --nproc_per_node=8 tests/validation/moe/test_megatron_moe_replay.py

Run on two nodes with the same command and rendezvous settings on both hosts.
Transformer Engine and Megatron Core are optional runtime dependencies.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.nn.functional as F

from lm_resiliency import (
    GroupedExpertMaterializer,
    ReplayHarnessConfig,
    ReplayWorkload,
)
from lm_resiliency.experimental import ModelReplayHarness


def _assert_all(condition: bool, message: str) -> None:
    passed = torch.tensor(int(condition), device="cuda", dtype=torch.int64)
    dist.all_reduce(passed, op=dist.ReduceOp.MIN)
    assert passed.item() == 1, message


def _singleton_nccl_group(rank: int, world_size: int) -> dist.ProcessGroup:
    local_group = None
    for member in range(world_size):
        group = dist.new_group([member], backend="nccl")
        if member == rank:
            local_group = group
    assert local_group is not None
    return local_group


def _build_grouped_experts(
    singleton_group: dist.ProcessGroup,
    *,
    ep_group: dist.ProcessGroup | None = None,
    expt_tp_group: dist.ProcessGroup | None = None,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    num_moe_experts: int = 4,
):
    try:
        from megatron.core.extensions.transformer_engine import HAVE_TE
        from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.moe.experts import TEGroupedMLP
        from megatron.core.transformer.transformer_config import TransformerConfig
    except ImportError as error:
        raise RuntimeError(
            "Megatron Core and Transformer Engine are required for this validation"
        ) from error

    assert HAVE_TE, "Megatron Core did not detect Transformer Engine"
    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_moe_experts=num_moe_experts,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=F.gelu,
        moe_grouped_gemm=True,
        transformer_impl="transformer_engine",
    )
    ep_group = ep_group or singleton_group
    expt_tp_group = expt_tp_group or singleton_group
    groups = ProcessGroupCollection(ep=ep_group, expt_tp=expt_tp_group)
    experts = (
        TESpecProvider()
        .grouped_mlp_modules(True)(
            num_moe_experts // expert_model_parallel_size,
            config,
            pg_collection=groups,
        )
        .cuda()
    )
    assert isinstance(experts, TEGroupedMLP)
    return experts


def _build_routed_moe(
    singleton_group: dist.ProcessGroup,
    *,
    ep_group: dist.ProcessGroup | None = None,
    expt_tp_group: dist.ProcessGroup | None = None,
    tp_ep_group: dist.ProcessGroup | None = None,
    expert_model_parallel_size: int = 1,
    expert_tensor_parallel_size: int = 1,
    num_moe_experts: int = 4,
    token_dispatcher: str = "allgather",
):
    from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec
    from megatron.core.process_groups_config import ProcessGroupCollection
    from megatron.core.transformer.moe.experts import TEGroupedMLP
    from megatron.core.transformer.transformer_config import TransformerConfig

    config = TransformerConfig(
        num_layers=1,
        hidden_size=128,
        num_attention_heads=8,
        ffn_hidden_size=256,
        moe_ffn_hidden_size=256,
        num_moe_experts=num_moe_experts,
        expert_model_parallel_size=expert_model_parallel_size,
        expert_tensor_parallel_size=expert_tensor_parallel_size,
        params_dtype=torch.bfloat16,
        bf16=True,
        add_bias_linear=False,
        gated_linear_unit=False,
        activation_func=F.gelu,
        moe_grouped_gemm=True,
        moe_router_topk=2,
        moe_router_load_balancing_type="none",
        moe_token_dispatcher_type=token_dispatcher,
        moe_permute_fusion=False,
        transformer_impl="transformer_engine",
    )
    ep_group = ep_group or singleton_group
    expt_tp_group = expt_tp_group or singleton_group
    tp_ep_group = tp_ep_group or ep_group
    groups = ProcessGroupCollection(
        tp=singleton_group,
        cp=singleton_group,
        tp_cp=singleton_group,
        tp_dp_cp=singleton_group,
        ep=ep_group,
        expt_tp=expt_tp_group,
        tp_ep=tp_ep_group,
    )
    moe = get_moe_module_spec(
        use_te=True,
        num_experts=num_moe_experts,
        moe_grouped_gemm=True,
    )(config=config, pg_collection=groups).cuda()
    assert isinstance(moe.experts, TEGroupedMLP)
    return moe


def _synchronize_parameters(
    module,
    group: dist.ProcessGroup,
    *,
    src: int = 0,
) -> None:
    with torch.no_grad():
        for parameter in module.parameters():
            dist.broadcast(parameter, src=src, group=group)


def main() -> None:
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)

    gloo_group = dist.new_group(backend="gloo")
    nccl_group = dist.new_group(backend="nccl")
    singleton_group = _singleton_nccl_group(rank, world_size)

    torch.manual_seed(1701)
    routed_moe = _build_routed_moe(singleton_group)
    _synchronize_parameters(routed_moe, nccl_group)
    routed_harness = ModelReplayHarness(
        routed_moe,
        group=gloo_group,
        nccl_group=nccl_group,
        device=torch.device("cuda", local_rank),
        config=ReplayHarnessConfig(
            check_interval=0,
            workload=ReplayWorkload.dense([routed_moe]),
            compare_parameter_state=False,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
        ),
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
    _assert_all(not any(routed_result.sdc_bitmap), str(routed_result.sdc_source_bitmaps))
    _assert_all(
        {"output", "input_gradient", "parameter_gradient"}.issubset(
            routed_result.sdc_source_bitmaps
        ),
        str(routed_result.sdc_source_bitmaps),
    )
    routed_harness.remove_hooks()

    torch.manual_seed(1729)
    experts = _build_grouped_experts(singleton_group)
    _synchronize_parameters(experts, nccl_group)

    workload = ReplayWorkload.from_shapes(
        [(128,), (512,)],
        replay_modules=[experts],
        materializer=GroupedExpertMaterializer(),
        source_id="megatron-te-grouped-mlp",
    )
    harness = ModelReplayHarness(
        experts,
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

    counts = torch.tensor([48, 80, 32, 96], device="cuda", dtype=torch.int64)
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
    _assert_all(healthy.completed_shape_cycle, str(healthy.checked_shapes))
    _assert_all(healthy.checked_shapes == [(128,), (512,)], str(healthy.checked_shapes))
    _assert_all(not any(healthy.sdc_bitmap), str(healthy.sdc_source_bitmaps))
    _assert_all(
        {"output", "input_gradient", "parameter_gradient"}.issubset(healthy.sdc_source_bitmaps),
        str(healthy.sdc_source_bitmaps),
    )

    if world_size == 1:
        harness.remove_hooks()
        print(
            "PASS: Megatron routed MoE and TEGroupedMLP replay passed with "
            "forward/backward coverage",
            flush=True,
        )
        dist.destroy_process_group()
        return

    victim = world_size - 1
    if rank == victim:
        with torch.no_grad():
            next(experts.parameters()).view(-1)[0].add_(2)

    corrupted_small = harness.check()
    expected = [0] * world_size
    expected[victim] = 1
    _assert_all(corrupted_small.replay_shape == (128,), str(corrupted_small.replay_shape))
    _assert_all(
        corrupted_small.sdc_bitmap == expected,
        str(corrupted_small.sdc_source_bitmaps),
    )
    _assert_all(
        any(
            corrupted_small.sdc_source_bitmaps.get(source) == expected
            for source in ("output", "input_gradient", "parameter_gradient")
        ),
        str(corrupted_small.sdc_source_bitmaps),
    )

    corrupted_large = harness.check()
    _assert_all(corrupted_large.replay_shape == (512,), str(corrupted_large.replay_shape))
    _assert_all(
        corrupted_large.sdc_bitmap == expected,
        str(corrupted_large.sdc_source_bitmaps),
    )
    _assert_all(
        any(
            corrupted_large.sdc_source_bitmaps.get(source) == expected
            for source in ("output", "input_gradient", "parameter_gradient")
        ),
        str(corrupted_large.sdc_source_bitmaps),
    )

    harness.remove_hooks()
    dist.barrier()
    if rank == 0:
        print(
            "PASS: Megatron routed MoE was clean, TEGroupedMLP replayed "
            "and certified 128/512-token shapes, and SCOUT "
            f"localized rank {victim} across {world_size} ranks",
            flush=True,
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
