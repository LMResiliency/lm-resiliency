"""Run Megatron Core's production training loop with injected resiliency.

The job uses a real Megatron GPTModel, Megatron DDP, distributed Adam, the
framework forward/backward schedule, ``train_step()``, and ``train()``.
Only token generation and external logging/checkpoint services are synthetic.

Run on one eight-GPU host:

    torchrun --rdzv-backend=lm_resiliency \
      --rdzv-endpoint=/tmp/lm-resiliency-megatron-rdzv \
      --rdzv-id=megatron-production \
      --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-megatron-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/production_loops/policies/resiliency.toml" \
      --nnodes=1:1 --nproc-per-node=8 --module \
      examples.production_loops.megatron \
      --artifact-dir /tmp/megatron-production-loop
"""

from __future__ import annotations

import argparse
import json
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator

import torch
import torch.distributed as dist

from examples.production_loops._common import add_run_arguments, require_resiliency_adapter

VOCABULARY_SIZE = 128
SEQUENCE_LENGTH = 16
MICRO_BATCH_SIZE = 2


class _DefaultArgs(SimpleNamespace):
    """Return ``None`` for disabled Megatron options not used by this job."""

    def __getattr__(self, name: str) -> Any:
        del name
        return None


class _Timer:
    def start(self, *args: Any, **kwargs: Any) -> "_Timer":
        del args, kwargs
        return self

    def stop(self, *args: Any, **kwargs: Any) -> "_Timer":
        del args, kwargs
        return self

    def elapsed(self, *args: Any, **kwargs: Any) -> float:
        del args, kwargs
        return 0.0

    def active_time(self) -> float:
        return 0.0


class _Timers:
    def __call__(self, *args: Any, **kwargs: Any) -> _Timer:
        del args, kwargs
        return _Timer()

    def log(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs


class _EnergyMonitor:
    def setup(self) -> None:
        pass

    def resume(self) -> None:
        pass

    def pause(self) -> None:
        pass

    def lap(self) -> None:
        pass

    def get_total(self) -> float:
        return 0.0

    def shutdown(self) -> None:
        pass


class _RerunStateMachine:
    def __init__(self) -> None:
        self.current_iteration = 0
        self._run_forward_backward = True

    def should_run_forward_backward(self, data_iterator: Any) -> bool:
        del data_iterator
        result = self._run_forward_backward
        self._run_forward_backward = not self._run_forward_backward
        return result

    def should_checkpoint_and_exit(self) -> tuple[bool, bool, int]:
        return False, False, 0


class _FTIntegration:
    def __getattr__(self, name: str):
        del name
        return lambda *args, **kwargs: None


def _arguments(rank: int, world_size: int, train_steps: int) -> _DefaultArgs:
    return _DefaultArgs(
        rank=rank,
        world_size=world_size,
        data_parallel_size=world_size,
        iteration=0,
        curr_iteration=0,
        train_iters=train_steps,
        train_samples=train_steps * MICRO_BATCH_SIZE * world_size,
        consumed_train_samples=0,
        skipped_train_samples=0,
        num_floating_point_operations_so_far=0.0,
        global_batch_size=MICRO_BATCH_SIZE * world_size,
        micro_batch_size=MICRO_BATCH_SIZE,
        seq_length=SEQUENCE_LENGTH,
        decoder_seq_length=None,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        iterations_to_skip=[],
        perform_rl_step=False,
        skip_train=False,
        hybrid_context_parallel=False,
        run_workload_inspector_server=False,
        save=None,
        async_save=False,
        log_throughput=False,
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        align_grad_reduce=False,
        align_param_gather=False,
        log_energy=False,
        manual_gc=False,
        log_straggler=False,
        cuda_graph_impl=None,
        optimizer_cuda_graph=False,
        profile=False,
        gpu_sniff_test_interval=None,
        distributed_timeout_seconds_after_init=None,
        reuse_grad_buf_for_mxfp8_param_ag=False,
        moe_expert_rank_capacity_factor=None,
        empty_unused_memory_level=0,
        vision_pretraining=False,
        vision_pretraining_type=None,
        save_params_interval=None,
        save_activations_interval=None,
        save_tokens_per_expert_interval=None,
        save_wgrads_interval=None,
        save_dgrads_interval=None,
        barrier_with_L1_time=False,
        qk_clip=False,
        log_max_attention_logit=False,
        log_num_zeros_in_grad=False,
        eval_interval=0,
        do_valid=False,
        start_eval_at_iter=None,
        num_experts=None,
        log_params_norm=False,
        check_weight_hash_across_dp_replicas_interval=None,
    )


def _install_training_services(training: Any, args: _DefaultArgs) -> None:
    rerun = _RerunStateMachine()
    timers = _Timers()
    training.get_args = lambda: args
    training.get_timers = lambda: timers
    training.get_energy_monitor = lambda: _EnergyMonitor()
    training.get_one_logger = lambda: None
    training.get_tensorboard_writer = lambda: None
    training.get_wandb_writer = lambda: None
    training.get_rerun_state_machine = lambda: rerun
    training.write_args_to_tensorboard = lambda: None
    training.print_datetime = lambda *args, **kwargs: None
    training.training_log = lambda *args, **kwargs: False
    training.maybe_finalize_async_save = lambda *args, **kwargs: None
    training.checkpoint_and_decide_exit = lambda *args, **kwargs: False
    training.num_floating_point_operations = lambda *args, **kwargs: 0.0
    training.post_training_step_callbacks = (
        lambda model, optimizer, scheduler, iteration, profiler, flops, context: flops
    )
    training.ft_integration = _FTIntegration()
    training.one_logger_utils.on_train_start = lambda *args, **kwargs: None
    training.one_logger_utils.track_e2e_metrics = lambda *args, **kwargs: None
    training.one_logger_utils.finish = lambda *args, **kwargs: None


def _tokens(rank: int, step: int, device: torch.device) -> tuple[torch.Tensor, ...]:
    generator = torch.Generator().manual_seed(1_000_003 * rank + step)
    values = torch.randint(
        0,
        VOCABULARY_SIZE,
        (MICRO_BATCH_SIZE, SEQUENCE_LENGTH + 1),
        generator=generator,
    ).to(device)
    tokens = values[:, :-1]
    labels = values[:, 1:]
    positions = torch.arange(SEQUENCE_LENGTH, device=device).unsqueeze(0).expand_as(tokens)
    return tokens, labels, positions


def _data_iterator(
    rank: int,
    device: torch.device,
) -> Iterator[tuple[torch.Tensor, ...]]:
    step = 0
    while True:
        yield _tokens(rank, step, device)
        step += 1


def _loss(loss_mask: torch.Tensor, output: torch.Tensor):
    losses = output.float().view(-1)
    mask = loss_mask.view(-1).float()
    loss = torch.sum(losses * mask)
    token_count = mask.sum().to(dtype=torch.int)
    return loss, token_count, {"lm loss": torch.stack((loss.detach(), token_count.float()))}


def _forward_step(data_iterator: Iterator, model: Any):
    tokens, labels, positions = next(data_iterator)
    output = model(
        input_ids=tokens,
        position_ids=positions,
        attention_mask=None,
        labels=labels,
    )
    return output, partial(_loss, torch.ones_like(labels, dtype=torch.float32))


def _build_model_optimizer_scheduler(device: torch.device):
    from megatron.core import parallel_state as mpu
    from megatron.core.distributed import DistributedDataParallel as DDP
    from megatron.core.distributed import DistributedDataParallelConfig
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import get_gpt_layer_local_spec
    from megatron.core.optimizer import OptimizerConfig, get_megatron_optimizer
    from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed
    from megatron.core.transformer.transformer_config import TransformerConfig

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
    )
    model_parallel_cuda_manual_seed(123)
    torch.manual_seed(123)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        use_cpu_initialization=True,
        pipeline_dtype=torch.float32,
        bf16=False,
        fp16=False,
        hidden_dropout=0.1,
        attention_dropout=0.1,
    )
    gpt = GPTModel(
        config=config,
        transformer_layer_spec=get_gpt_layer_local_spec(),
        vocab_size=VOCABULARY_SIZE,
        max_sequence_length=SEQUENCE_LENGTH,
        position_embedding_type="rope",
    ).to(device)
    model = DDP(
        config=config,
        ddp_config=DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            overlap_grad_reduce=False,
            use_distributed_optimizer=True,
        ),
        module=gpt,
    )
    optimizer = get_megatron_optimizer(
        OptimizerConfig(
            optimizer="adam",
            lr=1e-3,
            min_lr=1e-4,
            use_distributed_optimizer=True,
            bf16=False,
            fp16=False,
        ),
        [model],
    )
    scheduler = OptimizerParamScheduler(
        optimizer,
        init_lr=1e-3,
        max_lr=1e-3,
        min_lr=1e-4,
        lr_warmup_steps=0,
        lr_decay_steps=100,
        lr_decay_style="linear",
        start_wd=0.0,
        end_wd=0.0,
        wd_incr_steps=100,
        wd_incr_style="constant",
    )
    return model, optimizer, scheduler, config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    add_run_arguments(parser)
    parsed = parser.parse_args()
    parsed.artifact_dir.mkdir(parents=True, exist_ok=True)

    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(__import__("os").environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)

    from megatron.core.num_microbatches_calculator import (
        init_num_microbatches_calculator,
    )
    from megatron.training import training

    args = _arguments(rank, world_size, parsed.steps)
    init_num_microbatches_calculator(
        rank=rank,
        global_batch_size=args.global_batch_size,
        micro_batch_size=args.micro_batch_size,
        data_parallel_size=args.data_parallel_size,
        decrease_batch_size_if_needed=False,
        step_batch_size_schedule=None,
        seq_length=args.seq_length,
    )
    _install_training_services(training, args)
    model, optimizer, scheduler, config = _build_model_optimizer_scheduler(device)

    try:
        iteration, _ = training.train(
            _forward_step,
            [model],
            optimizer,
            scheduler,
            _data_iterator(rank, device),
            None,
            None,
            config,
            {},
            None,
        )
        require_resiliency_adapter()
        if iteration != parsed.steps:
            raise AssertionError(f"step mismatch: megatron={iteration}")

        summary = {
            "framework": "megatron",
            "framework_loop": "megatron.training.training.train",
            "model": "megatron.core.models.gpt.GPTModel",
            "world_size": world_size,
            "steps": iteration,
            "consumed_train_samples": args.consumed_train_samples,
            "resiliency_adapter_attached": True,
        }
        if rank == 0:
            (parsed.artifact_dir / "megatron-production-loop.json").write_text(
                json.dumps(summary, indent=2) + "\n",
                encoding="utf-8",
            )
            print(json.dumps(summary, sort_keys=True), flush=True)
    finally:
        failed = sys.exc_info()[0] is not None
        if not failed:
            dist.barrier()
            from megatron.core import parallel_state as mpu

            mpu.destroy_model_parallel()
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
