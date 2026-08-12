"""Unstable low-level APIs for research and integration development.

Objects in this module may change within the ``0.x`` release series.
"""

from lm_resiliency.adapters import FrameworkAdapter, ParallelismInfo
from lm_resiliency.checkpointing.durable import (
    DurableCheckpointAdapter,
    DurableCheckpointCoordinator,
    DurableCheckpointRecord,
)
from lm_resiliency.checkpointing.manager import InMemoryCheckpointManager, RecoveryMode
from lm_resiliency.checkpointing.transfer import (
    NixlCheckpointTransfer,
    TorchDistTransfer,
)
from lm_resiliency.detection.c3 import C3, C3Mode, C3Result, C3Status
from lm_resiliency.detection.layer_replay import LayerReplayDetector
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    enable_replay_detection,
)
from lm_resiliency.detection.replay_shapes import (
    ReplayShape,
    ReplayShapeMaterializer,
    ReplayShapePlan,
)
from lm_resiliency.detection.stage_instrumentation import InstrumentedDataLoader
from lm_resiliency.detection.topology import ReplayPeerGroup, ReplayPeerRole

__all__ = [
    "C3",
    "C3Mode",
    "C3Result",
    "C3Status",
    "DurableCheckpointAdapter",
    "DurableCheckpointCoordinator",
    "DurableCheckpointRecord",
    "FrameworkAdapter",
    "InMemoryCheckpointManager",
    "InstrumentedDataLoader",
    "LayerReplayDetector",
    "ModelReplayHarness",
    "NixlCheckpointTransfer",
    "ParallelismInfo",
    "ReplayPeerGroup",
    "ReplayPeerRole",
    "ReplayShape",
    "ReplayShapeMaterializer",
    "ReplayShapePlan",
    "RecoveryMode",
    "TorchDistTransfer",
    "enable_replay_detection",
]
