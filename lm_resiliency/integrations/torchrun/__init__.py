"""Torchrun rendezvous entry point for manager-owned recovery plans.

The implementation remains private; torchrun discovers the supported
integration through the ``torchrun.handlers`` entry-point group.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed.elastic.rendezvous import (
        RendezvousHandler,
        RendezvousParameters,
    )


def get_rendezvous_handler_creator() -> Callable[["RendezvousParameters"], "RendezvousHandler"]:
    """Return a creator that loads the runtime after entry-point discovery."""

    def create(params: "RendezvousParameters") -> "RendezvousHandler":
        from ._simple_runtime import _create_rendezvous_handler

        return _create_rendezvous_handler(params)

    return create
