"""Validate fault injection against FSDP2 DTensor local shards.

Run:

    torchrun --standalone --nproc-per-node=2 \
      tests/integration/core/test_fault_injection_dtensor.py
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency import (
    CorruptionOperation,
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultScope,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
    enable_fault_injection,
)


def main() -> None:
    if not dist.is_initialized():
        dist.init_process_group(backend="cpu:gloo,cuda:nccl")
    rank = dist.get_rank()
    if dist.get_world_size() != 2:
        raise RuntimeError("fault-injection DTensor validation requires exactly two ranks")
    torch.cuda.set_device(rank)

    from torch.distributed.fsdp import fully_shard

    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Linear(64, 64)).cuda()
    fully_shard(model[0])
    fully_shard(model[2])
    fully_shard(model)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0)
    parameter = model[0].weight
    if type(parameter).__name__ != "DTensor":
        raise AssertionError("FSDP2 did not expose a DTensor parameter")
    for target_rank in range(dist.get_world_size()):
        baseline = parameter.to_local().detach().clone()
        fault = FaultSpec(
            fault_id=f"rank-{target_rank}-weight-sign-flip",
            type=FailureType.TENSOR_CORRUPTION,
            target=FaultTarget(
                rank=target_rank,
                surface=FaultSurface.WEIGHT,
                module_path="0",
            ),
            parameters={
                "operation": CorruptionOperation.SIGN_FLIP.value,
                "scope": FaultScope.SINGLE.value,
            },
        )
        campaign = FaultCampaign(
            name=f"fsdp2-local-shard-state-rank-{target_rank}",
            incidents=(
                FaultIncident(
                    incident_id="dtensor-weight-corruption",
                    trigger=IncidentTrigger(at=(1,)),
                    lifetime=IncidentLifetime(until="recovery"),
                    faults=(fault,),
                ),
            ),
        )

        session = enable_fault_injection(
            model,
            optimizer,
            campaign=campaign,
        )
        changed = not torch.equal(parameter.to_local(), baseline)
        expected_change = rank == target_rank
        record_ok = (
            len(session.records) == 1 and session.records[0].verified
            if expected_change
            else session.records == ()
        )
        session.notify_recovery()
        restored = torch.equal(parameter.to_local(), baseline)
        session.close()

        status = torch.tensor(
            [int(changed == expected_change and restored and record_ok)],
            device="cuda",
        )
        dist.all_reduce(status, op=dist.ReduceOp.MIN)
        if status.item() != 1:
            raise AssertionError(
                f"rank {rank}, target {target_rank}: DTensor fault changed={changed}, "
                f"expected_change={expected_change}, restored={restored}, "
                f"records={[record.to_dict() for record in session.records]}"
            )
    if rank == 0:
        print("PASS fault-injection-dtensor-local-shard", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
