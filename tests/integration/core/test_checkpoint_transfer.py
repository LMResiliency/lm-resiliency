"""Two-rank keyed TorchDistTransfer validation.

Run:
    torchrun --nproc-per-node=2 tests/integration/core/test_checkpoint_transfer.py
"""

from __future__ import annotations

import socket
import threading
from unittest.mock import patch

import torch
import torch.distributed as dist

from lm_resiliency.checkpointing.transfer import TorchDistTransfer


def _run_threads(functions) -> list[BaseException]:
    errors: list[BaseException] = []

    def run(function) -> None:
        try:
            function()
        except BaseException as error:  # noqa: BLE001 - surfaced after every thread joins
            errors.append(error)

    threads = [threading.Thread(target=run, args=(function,)) for function in functions]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
    if any(thread.is_alive() for thread in threads):
        errors.append(TimeoutError("checkpoint transfer thread did not finish"))
    return errors


def main() -> None:
    dist.init_process_group("gloo")
    rank = dist.get_rank()
    peer = 1 - rank
    host = socket.gethostname()
    transfer = TorchDistTransfer(rank, chunk_size=8, timeout_s=5.0)

    if rank == 0:
        tensors = [torch.arange(8, dtype=torch.float32), torch.arange(5, dtype=torch.int64)]
        errors = _run_threads(
            [
                lambda: transfer.serve([tensors[0]], "owner/step-7", peer, host),
                lambda: transfer.serve([tensors[1]], "peer/step-7", peer, host),
            ]
        )
    else:
        first = torch.empty(8, dtype=torch.float32)
        second = torch.empty(5, dtype=torch.int64)
        errors = _run_threads(
            [
                lambda: transfer.fetch([first], "owner/step-7", peer, host),
                lambda: transfer.fetch([second], "peer/step-7", peer, host),
            ]
        )
        if not errors:
            assert torch.equal(first, torch.arange(8, dtype=torch.float32))
            assert torch.equal(second, torch.arange(5, dtype=torch.int64))

    ok = torch.tensor([int(not errors)], dtype=torch.int32)
    dist.all_reduce(ok, op=dist.ReduceOp.MIN)
    assert ok.item() == 1, errors

    mismatch_rejected = False
    try:
        if rank == 0:
            transfer.serve([torch.ones(4)], "shape-mismatch", peer, host)
        else:
            transfer.fetch([torch.empty(3)], "shape-mismatch", peer, host)
    except RuntimeError:
        mismatch_rejected = True
    rejected = torch.tensor([int(mismatch_rejected)], dtype=torch.int32)
    dist.all_reduce(rejected, op=dist.ReduceOp.MIN)
    assert rejected.item() == 1

    checksum_rejected = False
    try:
        if rank == 0:
            with patch(
                "lm_resiliency.checkpointing.transfer.shard_checksums",
                return_value=[0],
            ):
                transfer.serve([torch.ones(4)], "checksum-mismatch", peer, host)
        else:
            transfer.fetch([torch.empty(4)], "checksum-mismatch", peer, host)
    except RuntimeError:
        checksum_rejected = True
    rejected = torch.tensor([int(checksum_rejected)], dtype=torch.int32)
    dist.all_reduce(rejected, op=dist.ReduceOp.MIN)
    assert rejected.item() == 1

    transfer.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
