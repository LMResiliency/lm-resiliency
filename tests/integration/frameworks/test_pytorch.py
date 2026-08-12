"""Two-node native PyTorch validation for GEMINI and SCOUT.

Run the same command on two eight-GPU hosts, changing only ``--node-rank``:

    torchrun --nnodes=2 --nproc-per-node=8 --node-rank=0 \
      --master-addr=<host-0-private-ip> --master-port=29600 \
      tests/integration/frameworks/test_pytorch.py \
      --architecture=ddp --scenario=gemini

Architectures: ``ddp``, ``fsdp2``, and ``hsdp``.
Scenarios: ``gemini``, ``sdc``, and ``straggler``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import time
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

WORLD_SIZE = 16
REPLICATION_JUMP = 8
FAULT_RANK = 9


class Block(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.up = nn.Linear(width, width * 2)
        self.down = nn.Linear(width * 2, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        hidden = torch.nn.functional.gelu(self.up(self.norm(value)))
        return value + self.down(hidden)


class Model(nn.Module):
    def __init__(self, width: int = 128, layers: int = 3) -> None:
        super().__init__()
        self.embed = nn.Linear(width, width)
        self.layers = nn.ModuleList([Block(width) for _ in range(layers)])
        self.output = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = self.embed(value)
        for layer in self.layers:
            value = layer(value)
        return self.output(value)


@dataclass
class TrainingObjects:
    model: nn.Module
    optimizer: torch.optim.Optimizer
    mesh: Any | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--architecture",
        choices=("ddp", "fsdp2", "hsdp"),
        required=True,
    )
    parser.add_argument(
        "--scenario",
        choices=("gemini", "sdc", "straggler"),
        required=True,
    )
    return parser.parse_args()


def setup() -> torch.device:
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="cpu:gloo,cuda:nccl",
        timeout=timedelta(minutes=5),
    )
    if dist.get_world_size() != WORLD_SIZE:
        raise RuntimeError(
            f"this validation requires {WORLD_SIZE} ranks, got {dist.get_world_size()}"
        )
    return torch.device("cuda", local_rank)


def build_training_objects(
    architecture: str,
    device: torch.device,
) -> TrainingObjects:
    torch.manual_seed(20260808)
    model = Model().to(device)
    mesh = None

    if architecture == "ddp":
        model = DistributedDataParallel(model, device_ids=[device.index])
    else:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        if architecture == "fsdp2":
            mesh = init_device_mesh(
                "cuda",
                (WORLD_SIZE,),
                mesh_dim_names=("dp_shard",),
            )
        else:
            mesh = init_device_mesh(
                "cuda",
                (4, 4),
                mesh_dim_names=("dp_replicate", "dp_shard"),
            )
        for layer in model.layers:
            fully_shard(layer, mesh=mesh)
        fully_shard(model, mesh=mesh)

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    return TrainingObjects(model=model, optimizer=optimizer, mesh=mesh)


def train_step(
    objects: TrainingObjects,
    device: torch.device,
    step: int,
    *,
    before_optimizer_step=None,
) -> float:
    torch.manual_seed(4100 + step)
    value = torch.randn(4, 8, 128, device=device)
    objects.optimizer.zero_grad(set_to_none=True)
    output = objects.model(value)
    loss = output.float().square().mean()
    loss.backward()
    if before_optimizer_step is not None:
        before_optimizer_step()
    objects.optimizer.step()
    return float(loss.detach())


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def checkpoint_tensor_state(objects: TrainingObjects) -> list[torch.Tensor]:
    tensors = [
        local_tensor(parameter).detach().cpu().clone() for parameter in objects.model.parameters()
    ]
    for parameter in objects.model.parameters():
        state = objects.optimizer.state.get(parameter, {})
        for key in sorted(state):
            value = state[key]
            if isinstance(value, torch.Tensor):
                tensors.append(local_tensor(value).detach().cpu().clone())
    return tensors


def tensors_match(
    expected: list[torch.Tensor],
    actual: list[torch.Tensor],
) -> bool:
    return len(expected) == len(actual) and all(
        left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
        for left, right in zip(expected, actual)
    )


def tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        value = tensor.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(str(value.dtype).encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def assert_all(condition: bool, message: str, device: torch.device) -> None:
    value = torch.tensor([int(condition)], dtype=torch.int32, device=device)
    dist.all_reduce(value, op=dist.ReduceOp.MIN)
    if not value.item():
        raise AssertionError(message)


def checkpoint_config(folder: str):
    from lm_resiliency import InMemoryCkptConfig

    return InMemoryCkptConfig(
        interval=1,
        replication_jump=REPLICATION_JUMP,
        replication_chunk_size=256 * 1024,
        disk_flush_interval=0,
        disk_folder=folder,
    )


def replay_config():
    from lm_resiliency import ReplayHarnessConfig

    return ReplayHarnessConfig(
        check_interval=1,
        rotate_layers=False,
        scale_factors=[],
        enable_temporal=False,
        straggler_confirmation_rounds=2,
        straggler_min_slowdown_ratio=1.5,
        straggler_min_slowdown_ms=20.0,
    )


def peer_replica_is_exact(handle, device: torch.device) -> bool:
    manager = handle.ckpt_manager
    assert manager is not None
    manager.maybe_wait()
    manager.finalize_replication()

    if manager._skip_replication:
        return not manager._replicator.enabled

    slots = [
        manager._buffer_pool.peer_current,
        manager._buffer_pool.peer_previous,
    ]
    peer_slot = max(slots, key=lambda slot: slot.step)
    if peer_slot.step <= 0:
        return False

    own_slot = manager._buffer_pool.get_slot_by_step(peer_slot.step)
    if own_slot is None:
        return False
    own_digest = tensor_digest(own_slot.tensors)
    all_digests: list[str | None] = [None] * WORLD_SIZE
    dist.all_gather_object(all_digests, own_digest)

    peer_rank = manager._replicator.peer_rank
    peer_digest = tensor_digest(peer_slot.tensors)
    return (
        peer_rank == (dist.get_rank() + REPLICATION_JUMP) % WORLD_SIZE
        and peer_digest == all_digests[peer_rank]
    )


def expected_rng_values(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    cpu_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state(device)
    expected_cpu = torch.rand(8)
    expected_cuda = torch.rand(8, device=device).cpu()
    torch.set_rng_state(cpu_state)
    torch.cuda.set_rng_state(cuda_state, device)
    return expected_cpu, expected_cuda


def verify_rng_values(
    expected: tuple[torch.Tensor, torch.Tensor],
    device: torch.device,
) -> bool:
    actual_cpu = torch.rand(8)
    actual_cuda = torch.rand(8, device=device).cpu()
    return torch.equal(actual_cpu, expected[0]) and torch.equal(
        actual_cuda,
        expected[1],
    )


def recover_fresh_training_objects(
    architecture: str,
    device: torch.device,
    folder: str,
    restored_extra: dict[str, Any],
):
    from lm_resiliency import enable_resiliency

    objects = build_training_objects(architecture, device)
    handle = enable_resiliency(
        objects.model,
        objects.optimizer,
        interval=1,
        enable_detection=False,
        checkpoint=checkpoint_config(folder),
        device=device,
        load_extra_state_fn=restored_extra.update,
    )
    return objects, handle


def run_gemini(
    architecture: str,
    device: torch.device,
) -> dict[str, Any]:
    from lm_resiliency import enable_resiliency

    rank = dist.get_rank()
    external_state: dict[str, Any] = {"marker": -1}
    with tempfile.TemporaryDirectory(
        prefix=f"pytorch-two-node-{architecture}-gemini-rank-{rank}-"
    ) as folder:
        objects = build_training_objects(architecture, device)
        handle = enable_resiliency(
            objects.model,
            objects.optimizer,
            interval=1,
            enable_detection=False,
            checkpoint=checkpoint_config(folder),
            device=device,
            extra_state_fn=lambda: dict(external_state),
        )

        losses = []
        for step in range(1, 4):
            external_state["marker"] = rank * 100 + step
            losses.append(train_step(objects, device, step))

        expected_state = checkpoint_tensor_state(objects)
        expected_rng = expected_rng_values(device)
        replica_exact = peer_replica_is_exact(handle, device)
        assert_all(replica_exact, "cross-host checkpoint replica mismatch", device)

        flushed_step = handle.flush_for_restart()
        assert_all(flushed_step == 3, "GEMINI did not flush step 3", device)
        handle.close()
        del objects
        torch.cuda.empty_cache()
        dist.barrier()

        restored_extra: dict[str, Any] = {}
        recovered_objects, recovered = recover_fresh_training_objects(
            architecture,
            device,
            folder,
            restored_extra,
        )
        recovered_state = checkpoint_tensor_state(recovered_objects)
        state_exact = tensors_match(expected_state, recovered_state)
        extra_exact = restored_extra == {"marker": rank * 100 + 3}
        rng_exact = verify_rng_values(expected_rng, device)
        assert_all(
            recovered.recovered_step == 3
            and recovered.step_count == 3
            and state_exact
            and extra_exact
            and rng_exact,
            "fresh-process GEMINI recovery was not bitwise equivalent",
            device,
        )
        recovered.close()

    return {
        "architecture": architecture,
        "scenario": "gemini",
        "world_size": WORLD_SIZE,
        "losses": losses,
        "checkpoint_step": 3,
        "cross_host_replica": "exact" if architecture != "hsdp" else "natural HSDP replica",
        "model_optimizer_state": "bitwise exact",
        "extra_state": "exact",
        "rng_state": "exact",
    }


def inject_sdc(
    architecture: str,
    handle,
    rank: int,
):
    target = handle.replay_harness.target_layer
    if architecture == "fsdp2":

        def corrupt_output(_module, _inputs, output):
            return output + 1.0 if rank == FAULT_RANK else output

        return target.register_forward_hook(corrupt_output)

    with torch.no_grad():
        parameter = next(target.parameters())
        if architecture == "hsdp":
            if rank == FAULT_RANK:
                if parameter.grad is None:
                    raise RuntimeError("HSDP fault injection requires a materialized gradient")
                local_tensor(parameter.grad).view(-1)[0].add_(10.0)
        elif rank == FAULT_RANK:
            parameter.view(-1)[0].add_(1.0)
    return None


def run_sdc(
    architecture: str,
    device: torch.device,
) -> dict[str, Any]:
    from lm_resiliency import OrchestrationHooks, enable_resiliency

    rank = dist.get_rank()
    faults = []
    recovery_decisions = []
    external_state: dict[str, Any] = {"marker": -1}
    with tempfile.TemporaryDirectory(
        prefix=f"pytorch-two-node-{architecture}-sdc-rank-{rank}-"
    ) as folder:
        objects = build_training_objects(architecture, device)
        handle = enable_resiliency(
            objects.model,
            objects.optimizer,
            interval=1,
            checkpoint=checkpoint_config(folder),
            replay=replay_config(),
            device=device,
            fault_callback=faults.append,
            extra_state_fn=lambda: dict(external_state),
            orchestration=OrchestrationHooks(
                report_recovery=recovery_decisions.append,
            ),
        )

        external_state["marker"] = rank * 100 + 1
        clean_loss = train_step(objects, device, 1)
        assert handle.ckpt_manager is not None
        handle.ckpt_manager.maybe_wait()
        clean_result = handle.replay_harness.last_result
        module_recipes = ("embedding", "hidden", "output")
        precondition_surfaces = (
            ("replay_input", "replay_rng_state")
            if architecture == "fsdp2"
            else ("replay_input", "replay_rng_state", "parameter_state")
        )
        preconditions_clean = (
            clean_result is not None
            and all(recipe in clean_result.checked_recipe_ids for recipe in module_recipes)
            and all(
                clean_result.c3_results[f"{recipe}.{surface}"].status.value == "agree"
                for recipe in module_recipes
                for surface in precondition_surfaces
            )
        )
        clean = (
            not faults
            and handle.ckpt_manager._last_saved_step == 1
            and handle.ckpt_manager.checkpoint_status.recovery_verified_step == 1
            and preconditions_clean
        )
        if not clean:
            print(
                f"rank {rank} healthy-control diagnostics: "
                f"faults={len(faults)}, "
                f"saved={handle.ckpt_manager._last_saved_step}, "
                f"verified={handle.ckpt_manager.checkpoint_status.recovery_verified_step}, "
                f"recipes={clean_result.checked_recipe_ids if clean_result else None}, "
                f"statuses="
                f"{ ({name: result.status.value for name, result in clean_result.c3_results.items()} if clean_result else None) }, "
                f"sdc={clean_result.sdc_bitmap if clean_result else None}, "
                f"straggler={clean_result.straggler_bitmap if clean_result else None}",
                flush=True,
            )
        assert_all(clean, "healthy SCOUT control failed", device)

        if architecture != "ddp":
            assert clean_result is not None
            timing_name = "fsdp_parameter_all_gather.timing"
            timing_samples = [
                sample
                for sample in clean_result.collective_timings
                if sample.collective == "fsdp_parameter_all_gather"
            ]
            if architecture == "hsdp":
                start = (rank // 4) * 4
                expected_materialization_group = tuple(range(start, start + 4))
            else:
                expected_materialization_group = tuple(range(WORLD_SIZE))
            timing_clean = (
                timing_name in clean_result.c3_results
                and len(timing_samples) == 1
                and timing_samples[0].group_ranks == expected_materialization_group
            )
            assert_all(
                timing_clean,
                "FSDP parameter materialization timing was not retained",
                device,
            )

        external_state["marker"] = rank * 100 + 2
        train_step(objects, device, 2)
        handle.ckpt_manager.maybe_wait()
        expected_state = checkpoint_tensor_state(objects)
        expected_rng = expected_rng_values(device)
        verified = handle.ckpt_manager.checkpoint_status.recovery_verified_step == 2
        assert_all(verified, "dense checkpoint 2 was not verified immediately", device)

        hook = None

        def before_optimizer_step():
            nonlocal hook
            hook = inject_sdc(architecture, handle, rank)

        external_state["marker"] = rank * 100 + 3
        fault_loss = train_step(
            objects,
            device,
            3,
            before_optimizer_step=before_optimizer_step,
        )
        if hook is not None:
            hook.remove()

        if architecture == "hsdp":
            expected_peers = list(range(rank % 4, WORLD_SIZE, 4))
            fault_replica_index = FAULT_RANK // 4
            expected_bitmap = [
                int(index == fault_replica_index) for index, _candidate in enumerate(expected_peers)
            ]
            expects_local_shard_evidence = rank % 4 == FAULT_RANK % 4
        else:
            expected_peers = list(range(WORLD_SIZE))
            expected_bitmap = [int(candidate == FAULT_RANK) for candidate in expected_peers]
            expects_local_shard_evidence = False
        localized = (
            len(faults) == 1
            and faults[0].peer_ranks == expected_peers
            and faults[0].sdc_bitmap == expected_bitmap
            and not any(faults[0].straggler_bitmap)
            and (("local_parameter_shard" in faults[0].sdc_sources) == expects_local_shard_evidence)
        )
        recovery_handoff = (
            len(recovery_decisions) == 1
            and recovery_decisions[0]["failure_kind"] == "sdc"
            and recovery_decisions[0]["recovery_mode"] == "recovery_verified"
            and recovery_decisions[0]["checkpoint_source"] == "gemini"
            and recovery_decisions[0]["checkpoint_step"] == 2
            and recovery_decisions[0]["available"]
        )
        checkpoint_skipped = (
            handle.ckpt_manager._last_saved_step == 2 and handle.ckpt_manager.find_latest() == 2
        )
        if not localized:
            print(
                f"rank {rank} localization diagnostics: "
                f"expected_peers={expected_peers}, "
                f"expected_bitmap={expected_bitmap}, "
                f"faults="
                f"{[(fault.peer_ranks, fault.sdc_bitmap, fault.sdc_sources) for fault in faults]}",
                flush=True,
            )
        assert_all(localized, "SCOUT did not localize the injected SDC exactly", device)
        assert_all(
            recovery_handoff,
            "SCOUT did not hand the recovery-verified checkpoint to the manager",
            device,
        )
        assert_all(
            checkpoint_skipped,
            "GEMINI exposed the SDC-contaminated checkpoint",
            device,
        )

        flushed_step = handle.flush_for_restart()
        assert_all(
            flushed_step == 2,
            "GEMINI did not persist the latest completed candidate",
            device,
        )
        handle.close()
        del objects
        torch.cuda.empty_cache()
        dist.barrier()

        restored_extra: dict[str, Any] = {}
        recovered_objects, recovered = recover_fresh_training_objects(
            architecture,
            device,
            folder,
            restored_extra,
        )
        recovered_state = checkpoint_tensor_state(recovered_objects)
        recovered_exact = (
            recovered.recovered_step == 2
            and tensors_match(expected_state, recovered_state)
            and restored_extra == {"marker": rank * 100 + 2}
            and verify_rng_values(expected_rng, device)
        )
        assert_all(
            recovered_exact,
            "recovery did not select the recovery-verified checkpoint",
            device,
        )
        recovered.close()

    return {
        "architecture": architecture,
        "scenario": "sdc",
        "world_size": WORLD_SIZE,
        "clean_loss": clean_loss,
        "fault_loss": fault_loss,
        "fault_rank": FAULT_RANK,
        "sdc_bitmap": expected_bitmap,
        "peer_group": expected_peers,
        "recovery_decision": "recovery_verified step 2",
        "contaminated_checkpoint": "not captured",
        "recovered_step": 2,
        "recovered_state": "bitwise exact",
    }


def run_straggler(
    architecture: str,
    device: torch.device,
) -> dict[str, Any]:
    from lm_resiliency import enable_resiliency

    rank = dist.get_rank()
    faults = []
    replay_results = []
    objects = build_training_objects(architecture, device)
    handle = enable_resiliency(
        objects.model,
        objects.optimizer,
        interval=1,
        enable_checkpoint=False,
        replay=replay_config(),
        device=device,
        fault_callback=faults.append,
    )
    assert handle.replay_harness is not None
    original_step = handle.replay_harness.step

    def record_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        replay_results.append(result)
        return result

    handle.replay_harness.step = record_step

    clean_loss = train_step(objects, device, 1)
    assert_all(not faults, "healthy SCOUT straggler control failed", device)
    replay_results.clear()

    hook = None

    def before_optimizer_step():
        nonlocal hook
        target = handle.replay_harness.target_layer

        def delay_replay(_module, _inputs):
            if rank == FAULT_RANK:
                torch.cuda.synchronize(device)
                time.sleep(0.25)

        hook = target.register_forward_pre_hook(delay_replay)

    fault_loss = train_step(
        objects,
        device,
        2,
        before_optimizer_step=before_optimizer_step,
    )
    assert hook is not None
    hook.remove()

    expected_bitmap = [int(candidate == FAULT_RANK) for candidate in range(WORLD_SIZE)]
    localized = (
        len(faults) == 1
        and faults[0].straggler_bitmap == expected_bitmap
        and not any(faults[0].sdc_bitmap)
        and faults[0].straggler_confirmations >= 2
        and faults[0].straggler_detail is not None
        and faults[0].straggler_detail.straggler_rank == FAULT_RANK
    )
    if rank == 0:
        replay = replay_results[-1] if replay_results else None
        diagnostic = {
            "fault_count": len(faults),
            "replay_result": replay is not None,
            "straggler_bitmap": (replay.straggler_bitmap if replay is not None else None),
            "sdc_bitmap": replay.sdc_bitmap if replay is not None else None,
            "confirmations": (replay.straggler_confirmations if replay is not None else None),
            "replay_times_ms": replay.replay_times_ms if replay is not None else None,
            "detail_rank": (
                replay.straggler_detail.straggler_rank
                if replay is not None and replay.straggler_detail is not None
                else None
            ),
            "detail_type": (
                replay.straggler_detail.straggler_type
                if replay is not None and replay.straggler_detail is not None
                else None
            ),
        }
        print("PYTORCH STRAGGLER DIAGNOSTIC " + json.dumps(diagnostic), flush=True)
    assert_all(
        localized,
        "SCOUT did not confirm and localize the replay straggler",
        device,
    )
    detail = faults[0].straggler_detail
    handle.close()

    return {
        "architecture": architecture,
        "scenario": "straggler",
        "world_size": WORLD_SIZE,
        "clean_loss": clean_loss,
        "fault_loss": fault_loss,
        "fault_rank": FAULT_RANK,
        "straggler_bitmap": expected_bitmap,
        "confirmations": faults[0].straggler_confirmations,
        "classification": detail.straggler_type,
    }


def main() -> None:
    args = parse_args()
    device = setup()
    try:
        if args.scenario == "gemini":
            result = run_gemini(args.architecture, device)
        elif args.scenario == "sdc":
            result = run_sdc(args.architecture, device)
        else:
            result = run_straggler(args.architecture, device)
        dist.barrier()
        if dist.get_rank() == 0:
            print("PYTORCH TWO-NODE PASS " + json.dumps(result, sort_keys=True), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
