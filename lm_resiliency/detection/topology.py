"""Equivalent-peer topology for SCOUT replay."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch.distributed as dist


class ReplayPeerRole(str, Enum):
    """Model-state role that determines which ranks are equivalent replay peers."""

    DENSE = "dense"
    EXPERT = "expert"


@dataclass(frozen=True)
class ReplayPeerGroup:
    """Matching control- and tensor-plane groups for one replay role."""

    role: ReplayPeerRole
    group: dist.ProcessGroup | None
    nccl_group: dist.ProcessGroup | None

    def __post_init__(self) -> None:
        if (self.group is None) != (self.nccl_group is None):
            raise ValueError("replay peer group requires both Gloo and NCCL groups")
        if self.group is None or not dist.is_available() or not dist.is_initialized():
            return
        control_ranks = dist.get_process_group_ranks(self.group)
        tensor_ranks = dist.get_process_group_ranks(self.nccl_group)
        if control_ranks != tensor_ranks:
            raise ValueError("replay control and tensor groups must have identical rank membership")

    @property
    def peer_ranks(self) -> list[int]:
        """Global ranks in this equivalent-peer group."""
        if self.group is None:
            if not dist.is_available() or not dist.is_initialized():
                return [0]
            return list(range(dist.get_world_size()))
        return list(dist.get_process_group_ranks(self.group))


def normalize_replay_peer_role(role: ReplayPeerRole | str) -> ReplayPeerRole:
    """Validate and normalize a public replay-role value."""
    try:
        return ReplayPeerRole(role)
    except ValueError as exc:
        values = ", ".join(item.value for item in ReplayPeerRole)
        raise ValueError(
            f"unsupported replay peer role {role!r}; expected one of: {values}"
        ) from exc
