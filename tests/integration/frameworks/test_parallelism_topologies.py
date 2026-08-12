"""Two-node SCOUT validation across framework parallelism topologies.

Each invocation runs one topology in a fresh 16-rank job. The test verifies:

1. The framework adapter selects only semantically equivalent replay peers.
2. Replay remains clean while model-parallel NCCL collectives are active.
3. A deterministic output fault on global rank 15 is localized exactly.

Example:

    torchrun --nnodes=2 --nproc-per-node=8 \
      --node-rank=$NODE_RANK --master-addr=$MASTER_ADDR \
      tests/integration/frameworks/test_parallelism_topologies.py \
      --case pytorch-tp-pp --artifact-dir /tmp/scout-parallelisms
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from lm_resiliency.detection.c3 import C3Status
from lm_resiliency.detection.layer_replay import LayerReplayDetector
from lm_resiliency.detection.peer_group import (
    form_detection_groups,
    parallelism_device_mesh,
)
from lm_resiliency.detection.topology import ReplayPeerGroup, ReplayPeerRole

_WORLD_SIZE = 16
_FAULT_RANK = 15


@dataclass(frozen=True)
class MeshCase:
    shape: tuple[int, ...]
    names: tuple[str, ...]
    collective_dims: tuple[str, ...]
    role: ReplayPeerRole
    framework: str
    coverage: tuple[str, ...]


_MESH_CASES = {
    "pytorch-tp-pp": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "pp", "tp"),
        collective_dims=("tp",),
        role=ReplayPeerRole.DENSE,
        framework="PyTorch",
        coverage=("DDP", "TP", "SP", "PP"),
    ),
    "pytorch-cp": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "cp", "tp"),
        collective_dims=("cp", "tp"),
        role=ReplayPeerRole.DENSE,
        framework="PyTorch",
        coverage=("DDP", "TP", "SP", "CP"),
    ),
    "pytorch-expert": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "ep", "etp"),
        collective_dims=("ep", "etp"),
        role=ReplayPeerRole.EXPERT,
        framework="PyTorch",
        coverage=("DP", "EP", "expert TP"),
    ),
    "torchtitan-dense": MeshCase(
        shape=(2, 2, 2, 2),
        names=("pp", "dp_replicate", "fsdp", "tp"),
        collective_dims=("fsdp", "tp"),
        role=ReplayPeerRole.DENSE,
        framework="TorchTitan",
        coverage=("PP", "HSDP", "FSDP2", "CP", "TP", "SP"),
    ),
    "torchtitan-expert": MeshCase(
        shape=(2, 2, 2, 2),
        names=("dp_replicate", "efsdp", "ep", "etp"),
        collective_dims=("ep", "etp"),
        role=ReplayPeerRole.EXPERT,
        framework="TorchTitan",
        coverage=("HSDP", "FSDP2", "EP", "expert TP"),
    ),
    "deepspeed-tp-pp": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "pp", "tp"),
        collective_dims=("tp",),
        role=ReplayPeerRole.DENSE,
        framework="DeepSpeed",
        coverage=("DP", "ZeRO", "TP", "PP"),
    ),
    "deepspeed-sp": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "sp", "tp"),
        collective_dims=("sp", "tp"),
        role=ReplayPeerRole.DENSE,
        framework="DeepSpeed",
        coverage=("DP", "ZeRO", "TP", "Ulysses SP"),
    ),
    "deepspeed-expert": MeshCase(
        shape=(4, 2, 2),
        names=("dp", "ep", "etp"),
        collective_dims=("ep", "etp"),
        role=ReplayPeerRole.EXPERT,
        framework="DeepSpeed",
        coverage=("DP", "ZeRO", "EP", "expert TP"),
    ),
}


class CollectiveReplayLayer(nn.Module):
    """Small deterministic layer that executes selected model-parallel collectives."""

    def __init__(self, groups: list[dist.ProcessGroup], *, expert: bool) -> None:
        super().__init__()
        self.projection = nn.Linear(32, 32, bias=False, device="cuda", dtype=torch.float32)
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(32, device="cuda"))
        if expert:
            self.projection.weight.group_name = "ep_size_2"
        self._groups = groups
        self.fault_rank: int | None = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        output = self.projection(inputs)
        for group in self._groups:
            dist.all_reduce(output, group=group)
            output = output / dist.get_world_size(group)
        if dist.get_rank() == self.fault_rank:
            output = output + 1.0
        return output


class _DeepSpeedTopologyEngine:
    def __init__(
        self,
        module: nn.Module,
        dense_group: dist.ProcessGroup,
        expert_group: dist.ProcessGroup | None,
    ) -> None:
        self.module = module
        self.data_parallel_group = dense_group
        self.expert_data_parallel_group = (
            {"ep_size_2": expert_group} if expert_group is not None else {}
        )


class _MegatronOptimizer:
    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        del state


def _mesh_case(case_name: str, case: MeshCase) -> dict[str, Any]:
    if case.framework == "TorchTitan":
        return _torchtitan_case(case_name, case)

    rank = dist.get_rank()
    mesh_tensor = torch.arange(_WORLD_SIZE).reshape(case.shape)
    mesh = DeviceMesh(
        "cuda",
        mesh_tensor,
        mesh_dim_names=case.names,
    )
    collective_groups = [mesh.get_group(name) for name in case.collective_dims]
    layer = CollectiveReplayLayer(
        collective_groups,
        expert=case.role is ReplayPeerRole.EXPERT,
    )

    if case.framework == "DeepSpeed":
        from lm_resiliency.integrations.deepspeed.adapter import DeepSpeedAdapter

        dense_group = mesh.get_group("dp")
        expert_group = dense_group if case.role is ReplayPeerRole.EXPERT else None
        engine = _DeepSpeedTopologyEngine(layer, dense_group, expert_group)
        peer_group = DeepSpeedAdapter(engine).get_replay_peer_group(
            case.role,
            (layer,),
        )
    else:
        gloo_group, nccl_group = form_detection_groups(device_mesh=mesh)
        peer_group = ReplayPeerGroup(case.role, gloo_group, nccl_group)

    expected_peer_dims = _peer_dimension_names(mesh)
    expected_peers = _mesh_coordinate_peers(mesh, expected_peer_dims)
    if peer_group.peer_ranks != expected_peers:
        raise AssertionError(
            f"{case_name}: rank {rank} selected peers {peer_group.peer_ranks}, "
            f"expected {expected_peers}"
        )
    return _run_replay_checks(case_name, case.framework, case.coverage, layer, peer_group)


def _torchtitan_case(case_name: str, case: MeshCase) -> dict[str, Any]:
    from torchtitan.distributed import ParallelDims

    if case.role is ReplayPeerRole.EXPERT:
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
    else:
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

    parallel_dims.build_mesh()
    replay_mesh = parallelism_device_mesh(
        parallel_dims,
        expert=case.role is ReplayPeerRole.EXPERT,
    )
    if replay_mesh is None:
        raise AssertionError(f"{case_name}: TorchTitan replay mesh was not resolved")
    layer = CollectiveReplayLayer(
        [parallel_dims.get_mesh(name).get_group() for name in collective_names],
        expert=case.role is ReplayPeerRole.EXPERT,
    )
    gloo_group, nccl_group = form_detection_groups(device_mesh=replay_mesh)
    peer_group = ReplayPeerGroup(case.role, gloo_group, nccl_group)
    expected_peers = _mesh_coordinate_peers(
        replay_mesh,
        _peer_dimension_names(replay_mesh),
    )
    if peer_group.peer_ranks != expected_peers:
        raise AssertionError(
            f"{case_name}: rank {dist.get_rank()} selected peers "
            f"{peer_group.peer_ranks}, expected {expected_peers}"
        )
    result = _run_replay_checks(
        case_name,
        case.framework,
        case.coverage,
        layer,
        peer_group,
    )
    result["topology_source"] = "torchtitan.distributed.ParallelDims"
    return result


def _megatron_case(case_name: str) -> dict[str, Any]:
    from megatron.core import parallel_state as mpu

    if case_name == "megatron-dense":
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            create_gloo_process_groups=True,
        )
        role = ReplayPeerRole.DENSE
        groups = [
            mpu.get_tensor_model_parallel_group(),
            mpu.get_context_parallel_group(),
        ]
        coverage = ("DP", "TP", "SP", "CP")
        expected_size = 4
    elif case_name == "megatron-pipeline":
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=2,
            create_gloo_process_groups=True,
        )
        role = ReplayPeerRole.DENSE
        groups = [mpu.get_tensor_model_parallel_group()]
        coverage = ("DP", "TP", "SP", "PP", "virtual PP")
        expected_size = 4
    else:
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=2,
            expert_tensor_parallel_size=2,
            create_gloo_process_groups=True,
        )
        role = ReplayPeerRole.EXPERT
        groups = [
            mpu.get_expert_model_parallel_group(),
            mpu.get_expert_tensor_parallel_group(),
        ]
        coverage = ("DP", "TP", "SP", "EP", "expert TP")
        expected_size = 4

    try:
        from lm_resiliency.integrations.megatron.adapter import MegatronAdapter

        layer = CollectiveReplayLayer(groups, expert=role is ReplayPeerRole.EXPERT)
        model = nn.Module()
        model.layers = nn.ModuleList([layer])
        peer_group = MegatronAdapter(
            [model],
            _MegatronOptimizer(),
        ).get_replay_peer_group(role)
        if len(peer_group.peer_ranks) != expected_size:
            raise AssertionError(
                f"{case_name}: expected {expected_size} peers, got {peer_group.peer_ranks}"
            )
        return _run_replay_checks(
            case_name,
            "Megatron Core",
            coverage,
            layer,
            peer_group,
        )
    finally:
        mpu.destroy_model_parallel()


def _mesh_coordinate_peers(
    mesh: DeviceMesh,
    peer_dim_names: set[str],
) -> list[int]:
    names = tuple(mesh.mesh_dim_names or ())
    peer_dims = [index for index, name in enumerate(names) if name in peer_dim_names]
    rank_position = (mesh.mesh == dist.get_rank()).nonzero(as_tuple=False).flatten().tolist()
    index: list[int | slice] = []
    for dimension in range(mesh.mesh.ndim):
        index.append(slice(None) if dimension in peer_dims else rank_position[dimension])
    return sorted(mesh.mesh[tuple(index)].flatten().tolist())


def _peer_dimension_names(mesh: DeviceMesh) -> set[str]:
    names = tuple(mesh.mesh_dim_names or ())
    if "dp" in names:
        return {"dp"}
    if "dp_replicate" in names and mesh.mesh.shape[names.index("dp_replicate")] > 1:
        return {"dp_replicate"}
    for name in ("dp_shard", "fsdp", "efsdp"):
        if name in names:
            return {name}
    return {"dp_replicate"} if "dp_replicate" in names else set()


def _run_replay_checks(
    case_name: str,
    framework: str,
    coverage: tuple[str, ...],
    layer: CollectiveReplayLayer,
    peer_group: ReplayPeerGroup,
) -> dict[str, Any]:
    detector = LayerReplayDetector(
        group=peer_group.group,
        nccl_group=peer_group.nccl_group,
        device=torch.device("cuda", int(os.environ["LOCAL_RANK"])),
        straggler_min_slowdown_ratio=100.0,
        straggler_min_slowdown_ms=10_000.0,
    )
    activation = torch.arange(
        128,
        device="cuda",
        dtype=torch.float32,
    ).reshape(4, 32)

    clean = detector.replay_forward(layer, activation)
    if any(clean.sdc_bitmap):
        raise AssertionError(f"{case_name}: clean replay reported SDC {clean.sdc_bitmap}")
    required_clean_evidence = {
        "replay_input",
        "replay_rng_state",
        "parameter_state",
        "output",
    }
    missing = required_clean_evidence.difference(clean.c3_results)
    if missing:
        raise AssertionError(f"{case_name}: missing replay evidence {sorted(missing)}")
    non_agreeing = {
        name: clean.c3_results[name].status.value
        for name in required_clean_evidence
        if clean.c3_results[name].status is not C3Status.AGREE
    }
    if non_agreeing:
        raise AssertionError(f"{case_name}: clean replay preconditions disagreed {non_agreeing}")

    layer.fault_rank = _FAULT_RANK
    fault = detector.replay_forward(layer, activation)
    localized = [
        peer for peer, failed in zip(fault.peer_ranks, fault.sdc_bitmap, strict=True) if failed
    ]
    fault_in_group = _FAULT_RANK in fault.peer_ranks
    can_attribute = len(fault.peer_ranks) >= 3
    expected = [_FAULT_RANK] if fault_in_group and can_attribute else []
    if localized != expected:
        raise AssertionError(
            f"{case_name}: rank {dist.get_rank()} localized {localized}, expected {expected}"
        )
    output_result = fault.c3_results["output"]
    mismatch_observed = fault_in_group and (
        output_result.status is C3Status.ATTRIBUTED
        or (output_result.status is C3Status.INCONCLUSIVE and len(set(output_result.evidence)) > 1)
    )
    if fault_in_group and not mismatch_observed:
        raise AssertionError(
            f"{case_name}: peer group {fault.peer_ranks} did not observe rank "
            f"{_FAULT_RANK}'s disagreement"
        )

    local = {
        "rank": dist.get_rank(),
        "peer_ranks": fault.peer_ranks,
        "localized": localized,
        "mismatch_observed": mismatch_observed,
    }
    gathered: list[dict[str, Any] | None] = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered, local)
    fault_observers = [
        item["rank"] for item in gathered if item is not None and item["localized"] == [_FAULT_RANK]
    ]
    mismatch_observers = [
        item["rank"] for item in gathered if item is not None and item["mismatch_observed"]
    ]
    if dist.get_rank() == 0:
        expected_observers = len(peer_group.peer_ranks)
        observed = fault_observers if can_attribute else mismatch_observers
        if len(observed) != expected_observers:
            verdict = "localize" if can_attribute else "observe disagreement from"
            raise AssertionError(
                f"{case_name}: expected one peer group to {verdict} rank "
                f"{_FAULT_RANK}, got observers {observed}"
            )
    dist.barrier()
    return {
        "case": case_name,
        "framework": framework,
        "coverage": list(coverage),
        "world_size": _WORLD_SIZE,
        "peer_group_size": len(peer_group.peer_ranks),
        "fault_rank": _FAULT_RANK,
        "fault_observers": fault_observers,
        "mismatch_observers": mismatch_observers,
        "clean_replay": True,
        "replay_preconditions": "agree",
        "exact_fault_localization": can_attribute,
        "fault_disagreement_detected": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--case",
        required=True,
        choices=sorted(
            [
                *_MESH_CASES,
                "megatron-dense",
                "megatron-pipeline",
                "megatron-expert",
            ]
        ),
    )
    parser.add_argument("--artifact-dir", type=Path, required=True)
    args = parser.parse_args()

    if int(os.environ["WORLD_SIZE"]) != _WORLD_SIZE:
        raise RuntimeError(f"this validation requires exactly {_WORLD_SIZE} ranks")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    try:
        if args.case in _MESH_CASES:
            result = _mesh_case(args.case, _MESH_CASES[args.case])
        else:
            result = _megatron_case(args.case)
        if dist.get_rank() == 0:
            args.artifact_dir.mkdir(parents=True, exist_ok=True)
            output = args.artifact_dir / f"{args.case}.json"
            output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
            print(f"PASS {args.case}: {json.dumps(result, sort_keys=True)}", flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
