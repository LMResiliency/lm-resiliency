"""Public torchrun integration and rendezvous entry point.

Torchrun discovers the backend through the ``torchrun.handlers`` entry-point
group. Unlisted submodules remain internal.
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


from .coordinator import (  # noqa: E402
    TorchrunInitialPlacement,
    TorchrunRecoveryCoordinator,
    TorchrunRecoveryRequest,
    TorchrunSuccessorPlacement,
    derive_torchrun_node_id,
)
from .launch import TorchrunLaunchConfig  # noqa: E402
from .worker_adapter import (  # noqa: E402
    DeepSpeedWorkerAdapter,
    MegatronWorkerAdapter,
    NativePyTorchAdapter,
    NativePyTorchDDPAdapter,
    TorchrunWorkerAdapter,
    TorchrunWorkerAdapterError,
    TorchrunWorkerContext,
    TorchTitanWorkerAdapter,
    get_torchrun_worker_context,
)

__all__ = [
    "DeepSpeedWorkerAdapter",
    "MegatronWorkerAdapter",
    "NativePyTorchAdapter",
    "NativePyTorchDDPAdapter",
    "TorchTitanWorkerAdapter",
    "TorchrunInitialPlacement",
    "TorchrunLaunchConfig",
    "TorchrunRecoveryCoordinator",
    "TorchrunRecoveryRequest",
    "TorchrunSuccessorPlacement",
    "TorchrunWorkerAdapter",
    "TorchrunWorkerAdapterError",
    "TorchrunWorkerContext",
    "derive_torchrun_node_id",
    "get_torchrun_worker_context",
    "get_rendezvous_handler_creator",
]
