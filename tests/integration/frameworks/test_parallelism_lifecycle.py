"""Validate public SCOUT integration lifecycles under model parallelism.

Run one case at a time on two eight-GPU hosts:

    torchrun --nnodes=2 --nproc-per-node=8 \
      --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR \
      tests/integration/frameworks/test_parallelism_lifecycle.py \
      --case pytorch-cp
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from lm_resiliency import (
    CallbackDurableCheckpointAdapter,
    DurableCheckpointConfig,
    InMemoryCkptConfig,
    ReplayHarnessConfig,
    ReplayWorkload,
    enable_resiliency,
)
from lm_resiliency.detection.c3 import C3Status

_WORLD_SIZE = 16
_FAULT_RANK = 15


def _trace(message: str) -> None:
    if os.environ.get("SCOUT_VALIDATION_TRACE"):
        print(f"TRACE rank={dist.get_rank()} {message}", flush=True)


class ReplayBlock(nn.Module):
    def __init__(self, groups: list[dist.ProcessGroup]) -> None:
        super().__init__()
        self.linear = nn.Linear(32, 32, bias=False, device="cuda")
        with torch.no_grad():
            self.linear.weight.copy_(torch.eye(32, device="cuda"))
        self.groups = groups
        self.inject_replay_sdc = False

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        for group in self.groups:
            dist.all_reduce(outputs, group=group)
            outputs = outputs / dist.get_world_size(group)
        if self.inject_replay_sdc:
            outputs = outputs.clone()
            outputs.flatten()[-1] += 1
        return torch.nn.functional.gelu(outputs)


class Decoder(nn.Module):
    def __init__(self, layers: list[ReplayBlock]) -> None:
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            inputs = layer(inputs)
        return inputs


class Model(nn.Module):
    def __init__(self, groups: list[dist.ProcessGroup], *, layers: int = 2) -> None:
        super().__init__()
        self.decoder = Decoder([ReplayBlock(groups) for _ in range(layers)])

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(inputs)


def _replay_config(
    target: ReplayBlock,
    *,
    expert: bool,
    detect_stragglers: bool = False,
) -> ReplayHarnessConfig:
    workload = (
        ReplayWorkload(replay_modules=(target,), peer_role="expert")
        if expert
        else ReplayWorkload.dense([target])
    )
    return ReplayHarnessConfig(
        check_interval=1,
        workload=workload,
        rotate_layers=False,
        enable_temporal=False,
        scale_factors=[],
        compare_parameter_state=False,
        straggler_min_slowdown_ratio=1.5 if detect_stragglers else 100.0,
        straggler_min_slowdown_ms=20.0 if detect_stragglers else 10_000.0,
    )


def _device_mesh_case(case: str) -> tuple[DeviceMesh, list[dist.ProcessGroup], bool]:
    if case == "pytorch-comm-straggler":
        shape = (8, 2)
        names = ("dp", "tp")
        collective_names = ("tp",)
        expert = False
    elif case == "pytorch-cp":
        shape = (4, 2, 2)
        names = ("dp", "cp", "tp")
        collective_names = ("cp", "tp")
        expert = False
    elif case == "pytorch-pp":
        shape = (4, 2, 2)
        names = ("dp", "pp", "tp")
        collective_names = ("tp",)
        expert = False
    elif case == "pytorch-expert":
        shape = (4, 2, 2)
        names = ("dp", "ep", "etp")
        collective_names = ("ep", "etp")
        expert = True
    else:
        raise ValueError(f"unsupported PyTorch lifecycle case: {case}")
    mesh = DeviceMesh(
        "cuda",
        torch.arange(_WORLD_SIZE).reshape(shape),
        mesh_dim_names=names,
    )
    return mesh, [mesh.get_group(name) for name in collective_names], expert


def _torchtitan_case(case: str) -> tuple[Any, list[dist.ProcessGroup], bool, bool]:
    from torchtitan.distributed import ParallelDims

    if case == "torchtitan-dense":
        parallel_dims = ParallelDims(
            dp_replicate=2,
            dp_shard=1,
            cp=2,
            tp=2,
            pp=2,
            ep=1,
            etp=1,
            world_size=_WORLD_SIZE,
        )
        collective_names = ("cp", "tp")
        expert = False
        apply_fsdp = True
    elif case == "torchtitan-expert":
        parallel_dims = ParallelDims(
            dp_replicate=4,
            dp_shard=2,
            cp=1,
            tp=2,
            pp=1,
            ep=2,
            etp=2,
            world_size=_WORLD_SIZE,
        )
        collective_names = ("ep", "etp")
        expert = True
        apply_fsdp = False
    else:
        raise ValueError(f"unsupported TorchTitan lifecycle case: {case}")
    parallel_dims.build_mesh()
    groups = [parallel_dims.get_mesh(name).get_group() for name in collective_names]
    return parallel_dims, groups, expert, apply_fsdp


def _megatron_case(case: str) -> tuple[list[Model], list[dist.ProcessGroup], bool]:
    from megatron.core import parallel_state as mpu

    if case == "megatron-dense":
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            context_parallel_size=2,
            create_gloo_process_groups=True,
        )
        groups = [
            mpu.get_tensor_model_parallel_group(),
            mpu.get_context_parallel_group(),
        ]
        expert = False
        chunks = 1
    elif case == "megatron-pipeline":
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            create_gloo_process_groups=True,
        )
        groups = [mpu.get_tensor_model_parallel_group()]
        expert = False
        chunks = 2
    elif case == "megatron-expert":
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=2,
            create_gloo_process_groups=True,
        )
        groups = [
            mpu.get_expert_model_parallel_group(),
            mpu.get_expert_tensor_parallel_group(),
        ]
        expert = True
        chunks = 1
    else:
        raise ValueError(f"unsupported Megatron lifecycle case: {case}")
    return [Model(groups) for _ in range(chunks)], groups, expert


def _harness(handle: Any):
    harness = getattr(handle, "replay_harness", None)
    if harness is None:
        harness = getattr(handle, "_replay_harness", None)
    if harness is None:
        raise AssertionError("public integration did not create a replay harness")
    return harness


def _train_step(
    models: list[Model],
    optimizer: torch.optim.Optimizer,
    *,
    inject_fault: bool,
) -> None:
    _trace("train-step zero-grad")
    optimizer.zero_grad(set_to_none=True)
    inputs = torch.linspace(-1.0, 1.0, 4 * 32, device="cuda").reshape(4, 32)
    outputs = inputs
    _trace("train-step forward")
    for model in models:
        outputs = model(outputs)
    _trace("train-step backward")
    outputs.float().square().mean().backward()
    target = models[0].decoder.layers[0]
    if inject_fault and dist.get_rank() == _FAULT_RANK:
        target.inject_replay_sdc = True
    _trace("train-step optimizer")
    optimizer.step()
    target.inject_replay_sdc = False
    _trace("train-step complete")


def _run_training(
    case: str,
    models: list[Model],
    optimizer: torch.optim.Optimizer,
    handle: Any,
) -> dict[str, Any]:
    harness = _harness(handle)
    _train_step(models, optimizer, inject_fault=False)
    if harness.last_result is None or any(harness.last_result.sdc_bitmap):
        raise AssertionError(f"{case}: healthy control was not clean")

    _train_step(models, optimizer, inject_fault=True)
    fault_result = harness.last_result
    if fault_result is None:
        raise AssertionError(f"{case}: fault step produced no replay result")
    expected = [int(rank == _FAULT_RANK) for rank in fault_result.peer_ranks]
    if fault_result.sdc_bitmap != expected:
        raise AssertionError(
            f"{case}: localized {fault_result.sdc_bitmap} for peers "
            f"{fault_result.peer_ranks}, expected {expected}"
        )

    _train_step(models, optimizer, inject_fault=False)
    if harness.last_result is None or any(harness.last_result.sdc_bitmap):
        raise AssertionError(f"{case}: post-fault replay was not clean")

    localized = [
        rank
        for rank, failed in zip(
            fault_result.peer_ranks,
            fault_result.sdc_bitmap,
            strict=True,
        )
        if failed
    ]
    observation = {
        "rank": dist.get_rank(),
        "peer_ranks": fault_result.peer_ranks,
        "localized": localized,
    }
    observations: list[dict[str, Any] | None] = [None] * _WORLD_SIZE
    dist.all_gather_object(observations, observation)
    fault_observers = [
        item["rank"]
        for item in observations
        if item is not None and item["localized"] == [_FAULT_RANK]
    ]
    expected_observers = next(
        len(item["peer_ranks"])
        for item in observations
        if item is not None and _FAULT_RANK in item["peer_ranks"]
    )
    if len(fault_observers) != expected_observers:
        raise AssertionError(
            f"{case}: fault observers {fault_observers}, expected "
            f"{expected_observers} equivalent peers"
        )
    return {
        "case": case,
        "world_size": _WORLD_SIZE,
        "peer_group_size": len(fault_result.peer_ranks),
        "fault_rank": _FAULT_RANK,
        "fault_observers": fault_observers,
        "exact_fault_localization": True,
        "clean_control": True,
        "clean_post_fault_replay": True,
        "steps": handle.step_count,
    }


def _run_pytorch(case: str) -> dict[str, Any]:
    mesh, groups, expert = _device_mesh_case(case)
    model = Model(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = model.decoder.layers[0]
    handle = enable_resiliency(
        model,
        optimizer,
        framework="pytorch",
        interval=1,
        enable_checkpoint=False,
        replay=_replay_config(target, expert=expert),
        parallelism_info=mesh,
    )
    try:
        result = _run_training(case, [model], optimizer, handle)
        result["topology_source"] = "torch.distributed.DeviceMesh"
        return result
    finally:
        handle.close()


def _run_pytorch_communication_straggler(case: str) -> dict[str, Any]:
    mesh, groups, _ = _device_mesh_case(case)
    model = Model(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = model.decoder.layers[0]
    handle = enable_resiliency(
        model,
        optimizer,
        framework="pytorch",
        interval=1,
        enable_checkpoint=False,
        replay=_replay_config(
            target,
            expert=False,
            detect_stragglers=True,
        ),
        parallelism_info=mesh,
    )
    try:
        _train_step([model], optimizer, inject_fault=False)
        harness = _harness(handle)
        detector = harness._get_detector()
        activation = torch.linspace(
            -1.0,
            1.0,
            4 * 32,
            device="cuda",
        ).reshape(4, 32)
        healthy = detector.localize_straggler(
            target,
            activation,
            threshold_sigma=5.0,
        )
        if healthy.straggler_type != "none":
            raise AssertionError(f"{case}: healthy control classified as {healthy.straggler_type}")

        original_all_reduce = dist.all_reduce

        def delayed_all_reduce(*args, **kwargs):
            if dist.get_rank() == _FAULT_RANK:
                torch.cuda.synchronize()
                time.sleep(0.2)
            return original_all_reduce(*args, **kwargs)

        dist.all_reduce = delayed_all_reduce
        try:
            detail = detector.localize_straggler(
                target,
                activation,
                threshold_sigma=3.0,
            )
        finally:
            dist.all_reduce = original_all_reduce

        expected_rank = 14 if dist.get_rank() % 2 == 0 else 15
        expected_bitmap = [int(peer == expected_rank) for peer in detector._peer_ranks]
        if detail.straggler_type != "communication":
            raise AssertionError(
                f"{case}: expected communication classification, got {detail.straggler_type}"
            )
        if any(detail.compute_bitmap):
            raise AssertionError(
                f"{case}: collective delay was misclassified as compute {detail.compute_bitmap}"
            )
        if detail.communication_bitmap != expected_bitmap:
            raise AssertionError(
                f"{case}: communication bitmap {detail.communication_bitmap} "
                f"for peers {detector._peer_ranks}, expected {expected_bitmap}"
            )
        if detail.straggler_rank != expected_rank:
            raise AssertionError(
                f"{case}: localized {detail.straggler_rank}, expected {expected_rank}"
            )

        observation = {
            "rank": dist.get_rank(),
            "peer_ranks": detector._peer_ranks,
            "affected_rank": detail.straggler_rank,
            "classification": detail.straggler_type,
            "compute_bitmap": detail.compute_bitmap,
            "communication_bitmap": detail.communication_bitmap,
            "communication_times_ms": detail.comm_times_ms,
        }
        observations: list[dict[str, Any] | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(observations, observation)

        _train_step([model], optimizer, inject_fault=False)
        if harness.last_result is None or any(harness.last_result.sdc_bitmap):
            raise AssertionError(f"{case}: post-fault training step was not clean")

        return {
            "case": case,
            "world_size": _WORLD_SIZE,
            "tensor_parallel_size": 2,
            "detection_peer_group_size": 8,
            "delayed_collective_rank": _FAULT_RANK,
            "affected_tp_pair": [14, 15],
            "classifications": sorted(
                {item["classification"] for item in observations if item is not None}
            ),
            "compute_false_positive": False,
            "clean_control": True,
            "clean_post_fault_step": True,
        }
    finally:
        handle.close()


def _run_torchtitan(case: str) -> dict[str, Any]:
    _trace("torchtitan build topology")
    parallel_dims, groups, expert, apply_fsdp = _torchtitan_case(case)
    _trace("torchtitan build model")
    model = Model(groups)
    if apply_fsdp:
        from torch.distributed.fsdp import fully_shard

        hsdp_mesh = parallel_dims.get_mesh(["dp_replicate", "fsdp"])
        _trace(
            "torchtitan HSDP mesh "
            f"names={hsdp_mesh.mesh_dim_names} ranks={hsdp_mesh.mesh.flatten().tolist()}"
        )
        for layer in model.decoder.layers:
            fully_shard(layer, mesh=hsdp_mesh)
        fully_shard(model, mesh=hsdp_mesh)
        _trace("torchtitan FSDP wrapping complete")
        if os.environ.get("SCOUT_VALIDATION_TRACE"):
            from lm_resiliency.detection.peer_group import _infer_mesh

            inferred_mesh = _infer_mesh(model)
            _trace(
                "torchtitan inferred FSDP mesh "
                f"names={inferred_mesh.mesh_dim_names} "
                f"ranks={inferred_mesh.mesh.flatten().tolist()}"
            )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = model.decoder.layers[0]
    _trace("torchtitan enable public integration")
    handle = enable_resiliency(
        model,
        optimizer,
        framework="torchtitan",
        interval=1,
        enable_checkpoint=False,
        replay=_replay_config(target, expert=expert),
        parallelism_info=parallel_dims,
    )
    _trace("torchtitan public integration enabled")
    try:
        result = _run_training(case, [model], optimizer, handle)
        result["topology_source"] = "torchtitan.distributed.ParallelDims"
        result["fsdp_runtime"] = apply_fsdp
        return result
    finally:
        handle.close()


def _run_megatron(case: str) -> dict[str, Any]:
    from megatron.core import parallel_state as mpu

    models, _, expert = _megatron_case(case)
    parameters = [parameter for model in models for parameter in model.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4)
    target = models[0].decoder.layers[0]
    handle = enable_resiliency(
        models,
        optimizer,
        interval=1,
        enable_checkpoint=False,
        replay=_replay_config(target, expert=expert),
    )
    try:
        result = _run_training(case, models, optimizer, handle)
        optimizer_input = _harness(handle).last_result.c3_results.get(
            "optimizer.optimizer_replay_input"
        )
        if optimizer_input is None or optimizer_input.status is not C3Status.AGREE:
            raise AssertionError(f"{case}: optimizer source-broadcast replay was not verified")
        result["topology_source"] = "megatron.core.parallel_state"
        result["model_chunks"] = len(models)
        result["optimizer_replay_input"] = "agree"
        return result
    finally:
        handle.close()
        mpu.destroy_model_parallel()


def _run_durable(case: str) -> dict[str, Any]:
    topology_case = "pytorch-expert" if case == "pytorch-durable-expert" else "pytorch-pp"
    mesh, groups, expert = _device_mesh_case(topology_case)
    model = Model(groups)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    target = model.decoder.layers[0]
    committed: list[int] = []
    rejected: list[int] = []
    manifest_dir = tempfile.mkdtemp(prefix="scout-durable-")
    adapter = CallbackDurableCheckpointAdapter(
        save_candidate_fn=lambda candidate: {"step": candidate.step},
        load_checkpoint_fn=lambda checkpoint: checkpoint.step,
        commit_candidate_fn=lambda checkpoint, previous: committed.append(checkpoint.step),
        quarantine_candidate_fn=lambda checkpoint, reason: rejected.append(checkpoint.step),
    )
    handle = enable_resiliency(
        model,
        optimizer,
        framework="pytorch",
        interval=1,
        checkpoint=InMemoryCkptConfig(
            enable=True,
            interval=1,
            disk_flush_interval=0,
            disk_folder=manifest_dir,
        ),
        replay=_replay_config(target, expert=expert),
        parallelism_info=mesh,
        durable_checkpoint=DurableCheckpointConfig(
            manifest_dir=manifest_dir,
            environment_id="scout-parallelism-lifecycle",
            adapter=adapter,
        ),
    )
    try:
        _train_step([model], optimizer, inject_fault=False)
        if not handle.durable_checkpoint.has_pending:
            raise AssertionError("first clean cycle did not create a durable candidate")
        if handle.durable_checkpoint.latest_validated is not None:
            raise AssertionError("candidate was verified after only one cycle")

        _train_step([model], optimizer, inject_fault=False)
        if handle.durable_checkpoint.latest_validated is None:
            raise AssertionError("following clean cycle did not verify checkpoint 1")
        if handle.durable_checkpoint.latest_validated.step != 1:
            raise AssertionError("unexpected recovery-verified checkpoint step")
        if handle.ckpt_manager is None:
            raise AssertionError("GEMINI checkpoint manager was not created")
        handle.ckpt_manager.maybe_wait()
        status = handle.ckpt_manager.checkpoint_status
        if status.recovery_verified_step != 1 or status.candidate_step != 2:
            raise AssertionError("GEMINI did not retain verified step 1 and candidate step 2")

        _train_step([model], optimizer, inject_fault=True)
        if handle.durable_checkpoint.has_pending:
            raise AssertionError("corrupt durable candidate remained pending")
        if handle.durable_checkpoint.latest_validated.step != 1:
            raise AssertionError("corrupt boundary replaced the clean durable checkpoint")
        recovered_memory_checkpoint = handle.ckpt_manager.load()
        if recovered_memory_checkpoint is None or recovered_memory_checkpoint[1] != 1:
            raise AssertionError("corrupt boundary replaced the last GEMINI checkpoint")

        local_quarantine = int(rejected == [2])
        quarantine_writers: list[int | None] = [None] * _WORLD_SIZE
        dist.all_gather_object(quarantine_writers, local_quarantine)
        if sum(int(value or 0) for value in quarantine_writers) != 1:
            raise AssertionError(
                "exactly one durable manifest writer should quarantine candidate 2; "
                f"observed {quarantine_writers}"
            )
        return {
            "case": case,
            "world_size": _WORLD_SIZE,
            "topology": "expert" if expert else "dense",
            "candidate_step": 2,
            "gemini_recovered_step": recovered_memory_checkpoint[1],
            "corrupt_checkpoint_step": 3,
            "corrupt_checkpoint_skipped_globally": True,
            "durable_candidate_quarantined_once": True,
            "recovery_verified_step": handle.durable_checkpoint.latest_validated.step,
        }
    finally:
        handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=(
            "pytorch-cp",
            "pytorch-pp",
            "pytorch-expert",
            "pytorch-comm-straggler",
            "torchtitan-dense",
            "torchtitan-expert",
            "megatron-dense",
            "megatron-pipeline",
            "megatron-expert",
            "pytorch-durable-dense",
            "pytorch-durable-expert",
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    if dist.get_world_size() != _WORLD_SIZE:
        raise RuntimeError(f"this validation requires {_WORLD_SIZE} ranks")
    try:
        if args.case == "pytorch-comm-straggler":
            result = _run_pytorch_communication_straggler(args.case)
        elif args.case.startswith("pytorch-") and not args.case.startswith("pytorch-durable-"):
            result = _run_pytorch(args.case)
        elif args.case.startswith("torchtitan-"):
            result = _run_torchtitan(args.case)
        elif args.case.startswith("megatron-"):
            result = _run_megatron(args.case)
        else:
            result = _run_durable(args.case)
        if dist.get_rank() == 0:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            output = args.artifact_dir / f"{args.case}.json"
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"PASS {args.case}: {json.dumps(result, sort_keys=True)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
