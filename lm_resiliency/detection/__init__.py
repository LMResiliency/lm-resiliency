"""SCOUT: Failure localization for distributed LLM training.

Modules:
    c3              — Consensus Collective Communication primitive
    peer_group      — Auto-discovery and formation of detection peer groups
    layer_replay    — SDC + straggler detection via deterministic replay
    replay_harness  — One-line integration for model-level replay detection
    hang_detector   — OOB hang detection daemon (op-ID consensus)
    op_tracker      — Training-side shared memory progress publisher
    communication_localizer — overlapping-PG endpoint cross-validation
    switch_localizer — network-switch fault localization via cross-group tomography
"""

from lm_resiliency.detection.all_to_all_replay import (
    AllToAllCapture,
    AllToAllReplayExecutor,
    AllToAllReplayOutcome,
    AllToAllReplayPolicy,
    AllToAllReplayRecipe,
    AllToAllTrafficMatrix,
    BalancedAndPermutationPolicy,
)
from lm_resiliency.detection.c3 import C3, C3Mode, C3Result, C3Status
from lm_resiliency.detection.collective_timing import (
    CollectiveTimingCollector,
    GroupSpec,
    run_detection_round,
)
from lm_resiliency.detection.communication_localizer import (
    CollectiveObservation,
    CommunicationLocalizer,
    CommunicationVerdict,
    EndpointCandidate,
    EndpointMetadata,
)
from lm_resiliency.detection.cross_pg import (
    CollectiveTimingSample,
    CrossPGCoordinator,
    CrossPGResult,
)
from lm_resiliency.detection.hang_detector import HangDetectionDaemon, HangLocalizationResult
from lm_resiliency.detection.hang_instrumentation import HangInstrumentation
from lm_resiliency.detection.ib_topology import (
    Fabric,
    build_fabric_topology,
    parse_ibnetdiscover,
)
from lm_resiliency.detection.layer_replay import LayerReplayDetector, ReplayResult
from lm_resiliency.detection.moe_regimes import (
    CTASemantics,
    ExecutionFingerprint,
    ExecutionHints,
    ExecutionObservation,
    KernelLaunch,
    MoEExecutionEnvironment,
    MoERegimeCatalog,
    MoEReplayScheduler,
    ProfileLocation,
    ProfileRequest,
    TorchCudaExecutionProfiler,
    build_profile_requests,
    current_moe_environment,
    discover_execution_regimes,
    load_profile_requests,
    profile_requests,
    save_profile_requests,
)
from lm_resiliency.detection.oob_service import OOBHangConfig, OOBHangService
from lm_resiliency.detection.peer_group import form_detection_groups, get_peer_ranks
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    ReplayHarnessConfig,
    enable_replay_detection,
)
from lm_resiliency.detection.replay_shapes import (
    GroupedExpertMaterializer,
    LeadingDimensionMaterializer,
    ReplayShape,
    ReplayShapeMaterializer,
    ReplayShapePlan,
    ReplayShapePlanMismatch,
    ReplayShapeScheduler,
    ReplayWorkload,
)
from lm_resiliency.detection.reports import (
    SCOUTFaultCallback,
    SCOUTFaultReport,
    dispatch_replay_faults,
    replay_fault_reports,
)
from lm_resiliency.detection.stage_instrumentation import InstrumentedDataLoader
from lm_resiliency.detection.switch_localizer import (
    FabricTopology,
    GroupMeasurement,
    SwitchLocalizer,
    SwitchVerdict,
)

__all__ = [
    "AllToAllCapture",
    "AllToAllReplayExecutor",
    "AllToAllReplayOutcome",
    "AllToAllReplayPolicy",
    "AllToAllReplayRecipe",
    "AllToAllTrafficMatrix",
    "BalancedAndPermutationPolicy",
    "Fabric",
    "build_fabric_topology",
    "parse_ibnetdiscover",
    "CollectiveTimingCollector",
    "CollectiveTimingSample",
    "CollectiveObservation",
    "CommunicationLocalizer",
    "CommunicationVerdict",
    "CrossPGCoordinator",
    "CrossPGResult",
    "EndpointCandidate",
    "EndpointMetadata",
    "GroupSpec",
    "run_detection_round",
    "C3",
    "C3Mode",
    "C3Result",
    "C3Status",
    "HangDetectionDaemon",
    "HangInstrumentation",
    "HangLocalizationResult",
    "InstrumentedDataLoader",
    "OOBHangConfig",
    "OOBHangService",
    "LayerReplayDetector",
    "CTASemantics",
    "ExecutionFingerprint",
    "ExecutionHints",
    "ExecutionObservation",
    "KernelLaunch",
    "MoEExecutionEnvironment",
    "MoERegimeCatalog",
    "MoEReplayScheduler",
    "ProfileLocation",
    "ProfileRequest",
    "TorchCudaExecutionProfiler",
    "build_profile_requests",
    "current_moe_environment",
    "discover_execution_regimes",
    "load_profile_requests",
    "profile_requests",
    "save_profile_requests",
    "ModelReplayHarness",
    "ReplayHarnessConfig",
    "ReplayShape",
    "ReplayShapeMaterializer",
    "ReplayShapePlan",
    "ReplayShapePlanMismatch",
    "ReplayShapeScheduler",
    "ReplayWorkload",
    "GroupedExpertMaterializer",
    "LeadingDimensionMaterializer",
    "ReplayResult",
    "SCOUTFaultCallback",
    "SCOUTFaultReport",
    "dispatch_replay_faults",
    "enable_replay_detection",
    "form_detection_groups",
    "get_peer_ranks",
    "replay_fault_reports",
    "FabricTopology",
    "GroupMeasurement",
    "SwitchLocalizer",
    "SwitchVerdict",
]
