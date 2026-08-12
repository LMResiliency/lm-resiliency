"""Framework adapter interface for lm_resiliency.

Each supported training framework implements this interface to bridge
framework-specific state management with lm_resiliency's core checkpointing.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Iterable

    import torch


def materialize_adam_state(optimizer: Any, params: Iterable[torch.Tensor]) -> None:
    """Create zeroed Adam state (``exp_avg``, ``exp_avg_sq``, ``step``) for any ``params``
    whose optimizer state is not yet allocated.

    On a freshly-restarted process the optimizer's per-parameter momentum buffers do not
    exist until the first ``.step()`` — so the ``save_tensors`` fast-reload has no live
    tensors to copy the checkpointed moments into (the count mismatches, or the moments
    are silently dropped and training resumes with the wrong bias correction). This
    reproduces exactly the entries ``torch.optim.Adam`` lazily creates on its first step,
    **without** running a step (params are left untouched); the reloaded values then
    overwrite these zeros in place. Idempotent — params that already have state are skipped.
    """
    import torch

    for p in params:
        st = optimizer.state.setdefault(p, {})
        if "exp_avg" not in st:
            st["step"] = torch.zeros((), dtype=torch.float32)
            st["exp_avg"] = torch.zeros_like(p)
            st["exp_avg_sq"] = torch.zeros_like(p)


class FrameworkAdapter(ABC):
    """Abstract interface between a training framework and lm_resiliency.

    Implementations handle:
    - Extracting full training state (model, optimizer, scheduler, etc.)
    - Applying a loaded checkpoint back to the framework's components
    - Detecting parallelism configuration (for HSDP optimization, etc.)
    """

    @abstractmethod
    def get_state_dict(self) -> dict[str, Any]:
        """Extract the full training state as a flat dict of tensors/scalars.

        Should include model parameters, optimizer state, LR scheduler state,
        dataloader state, and any training metadata (step, tokens seen, etc.).
        """

    @abstractmethod
    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Apply a checkpoint state dict to all framework components."""

    def materialize_optimizer_state(self) -> None:
        """Allocate the optimizer-state tensors on a freshly-restarted engine so the
        ``save_tensors`` fast-reload has live targets to copy into.

        Default is a no-op (for adapters whose reload uses ``state_dict``, which
        re-creates state on load). ``save_tensors``-based adapters override this with the
        framework-native materializer. Called by ``try_recover`` **before**
        ``load_checkpoint_tensors``. See ``materialize_adam_state``."""

    @abstractmethod
    def get_parallelism_info(self) -> ParallelismInfo:
        """Return parallelism configuration for optimization decisions."""

    @property
    @abstractmethod
    def rank(self) -> int:
        """Current process rank."""

    @property
    @abstractmethod
    def world_size(self) -> int:
        """Total number of processes."""


class ParallelismInfo:
    """Describes the parallelism configuration of the training job.

    Used by lm_resiliency to make optimization decisions, e.g.:
    - Skip P2P replication when natural replicas exist (HSDP/ZeRO with replication)
    - Choose replication_jump based on failure domain boundaries
    """

    def __init__(
        self,
        dp_replicate: int = 1,
        dp_shard: int = 1,
        tp: int = 1,
        pp: int = 1,
        world_size: int = 1,
    ) -> None:
        self.dp_replicate = dp_replicate
        self.dp_shard = dp_shard
        self.tp = tp
        self.pp = pp
        self.world_size = world_size

    @property
    def has_natural_replicas(self) -> bool:
        """True if the parallelism strategy already replicates checkpoint shards."""
        return self.dp_replicate > 1
