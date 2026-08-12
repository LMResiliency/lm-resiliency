from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from examples.production_loops._common import ReplayFaultCampaign


class _CheckpointManager:
    def __init__(self, step: int) -> None:
        self._last_saved_step = step
        self.checkpoint_status = SimpleNamespace(recovery_verified_step=step)


def _result(*, fault: bool) -> SimpleNamespace:
    return SimpleNamespace(
        peer_ranks=[0, 1],
        sdc_bitmap=[1, 0] if fault else [0, 0],
        straggler_bitmap=[0, 0],
        sdc_sources=["hidden.output"] if fault else [],
        checked_recipe_ids={"embedding", "hidden", "output", "optimizer"},
    )


def test_fault_campaign_corrupts_replay_but_not_training_forward() -> None:
    target = nn.Identity()
    manager = _CheckpointManager(step=1)
    handle = SimpleNamespace(
        replay_harness=SimpleNamespace(target_layer=target),
        ckpt_manager=manager,
    )
    campaign = ReplayFaultCampaign(
        steps=5,
        rank=0,
        world_size=2,
        inject_fault=True,
        fault_step=2,
        fault_rank=0,
    )
    campaign.bind(handle)
    campaign.start_step(2)

    value = torch.zeros(2)
    torch.testing.assert_close(target(value), value)
    torch.testing.assert_close(target(value), value + 1.0)

    campaign.record_fault(_result(fault=True))
    campaign.recovery_decisions.append(
        {
            "failure_kind": "sdc",
            "recovery_mode": "recovery_verified",
            "checkpoint_source": "gemini",
            "checkpoint_step": 1,
            "available": True,
        }
    )
    manager._last_saved_step = 5
    manager.checkpoint_status.recovery_verified_step = 5

    campaign.validate(
        handle,
        _result(fault=False),
        {"embedding", "hidden", "output", "optimizer"},
    )
    campaign.close()


def test_fault_campaign_requires_two_clean_post_fault_steps() -> None:
    args = SimpleNamespace(
        steps=5,
        inject_fault=True,
        fault_step=4,
        fault_rank=-1,
    )

    with pytest.raises(ValueError, match="post-fault"):
        ReplayFaultCampaign.from_args(args, rank=0, world_size=2)
