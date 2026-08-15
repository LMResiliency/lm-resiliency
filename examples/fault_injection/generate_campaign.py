"""Generate the systematic eight-GPU fault-injection campaign."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Iterable

from lm_resiliency import (
    ClockSpec,
    CorruptionOperation,
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
)

_BLOCK_PATH = "layers.0.mlp.0"


def build_campaign() -> FaultCampaign:
    """Build a matrix covering every built-in local failure/surface pair."""
    incidents: list[FaultIncident] = []
    next_iteration = 4

    def add(
        incident_id: str,
        faults: Iterable[FaultSpec],
        *,
        gap_after: int = 0,
        lifetime: IncidentLifetime | None = None,
        at: tuple[int, ...] | None = None,
    ) -> None:
        nonlocal next_iteration
        scheduled = at or (next_iteration,)
        incidents.append(
            FaultIncident(
                incident_id=incident_id,
                trigger=IncidentTrigger(at=scheduled),
                lifetime=lifetime or IncidentLifetime(iterations=1),
                faults=tuple(faults),
            )
        )
        next_iteration = scheduled[-1] + 1 + gap_after

    output_operations = (
        ("sign-flip", CorruptionOperation.SIGN_FLIP, {"scope": "100%"}),
        (
            "set-zero",
            CorruptionOperation.SET_VALUE,
            {"value": 0.0, "scope": "100%"},
        ),
        (
            "set-one",
            CorruptionOperation.SET_VALUE,
            {"value": 1.0, "scope": "10%"},
        ),
        ("scale-up", CorruptionOperation.SCALE, {"factor": 10.0, "scope": "10%"}),
        ("scale-down", CorruptionOperation.SCALE, {"factor": 0.1, "scope": "10%"}),
        ("noise", CorruptionOperation.NOISE, {"std": 1.0, "scope": "10%"}),
        (
            "single-bitflip",
            CorruptionOperation.SINGLE_BITFLIP,
            {"magnitude": "large", "scope": "single"},
        ),
        (
            "multi-bitflip",
            CorruptionOperation.MULTI_BITFLIP,
            {"magnitude": "large", "scope": "row"},
        ),
    )
    input_operations = (
        ("sign-flip", CorruptionOperation.SIGN_FLIP, {"scope": "100%"}),
        (
            "set-zero",
            CorruptionOperation.SET_VALUE,
            {"value": 0.0, "scope": "1%"},
        ),
        (
            "set-one",
            CorruptionOperation.SET_VALUE,
            {"value": 1.0, "scope": "1%"},
        ),
        ("scale-up", CorruptionOperation.SCALE, {"factor": 2.0, "scope": "10%"}),
        ("scale-down", CorruptionOperation.SCALE, {"factor": 0.5, "scope": "10%"}),
        ("noise", CorruptionOperation.NOISE, {"std": 1.0, "scope": "row"}),
        (
            "single-bitflip",
            CorruptionOperation.SINGLE_BITFLIP,
            {"magnitude": "medium", "scope": "single"},
        ),
        (
            "multi-bitflip",
            CorruptionOperation.MULTI_BITFLIP,
            {"magnitude": "medium", "scope": "row"},
        ),
    )
    for surface, operations in (
        (FaultSurface.OUTPUT, output_operations),
        (FaultSurface.INPUT, input_operations),
    ):
        for rank, (name, operation, parameters) in enumerate(operations):
            incident_id = f"{surface.value}-{name}"
            add(
                incident_id,
                (
                    _corruption(
                        f"rank-{rank}-{incident_id}",
                        rank,
                        surface,
                        operation,
                        **parameters,
                    ),
                ),
            )

    flow_rank = 0
    for surface in (FaultSurface.OUTPUT, FaultSurface.INPUT):
        for failure_type in (
            FailureType.STALE_STATE,
            FailureType.DUPLICATE,
            FailureType.DROP,
            FailureType.REORDER,
        ):
            incident_id = f"{surface.value}-{failure_type.value}"
            add(
                incident_id,
                (
                    _fault(
                        f"rank-{flow_rank}-{incident_id}",
                        failure_type,
                        flow_rank,
                        surface,
                        parameters=(
                            {} if failure_type is FailureType.REORDER else {"scope": "100%"}
                        ),
                    ),
                ),
            )
            flow_rank = (flow_rank + 1) % 8

    for rank, surface in enumerate((FaultSurface.COMPUTE, FaultSurface.INPUT, FaultSurface.OUTPUT)):
        add(
            f"{surface.value}-delay",
            (
                _fault(
                    f"rank-{rank}-{surface.value}-delay",
                    FailureType.DELAY,
                    rank,
                    surface,
                    parameters={"delay_ms": 1000.0},
                ),
            ),
        )

    add(
        "correlated-two-rank-output-corruption",
        (
            _corruption(
                "rank-0-correlated-output-sign-flip",
                0,
                FaultSurface.OUTPUT,
                CorruptionOperation.SIGN_FLIP,
                scope="100%",
            ),
            _corruption(
                "rank-1-correlated-output-sign-flip",
                1,
                FaultSurface.OUTPUT,
                CorruptionOperation.SIGN_FLIP,
                scope="100%",
            ),
        ),
    )
    add(
        "correlated-three-rank-input-drop",
        tuple(
            _fault(
                f"rank-{rank}-correlated-input-drop",
                FailureType.DROP,
                rank,
                FaultSurface.INPUT,
                parameters={"scope": "100%"},
            )
            for rank in (2, 3, 4)
        ),
    )
    add(
        "correlated-three-rank-compute-delay",
        tuple(
            _fault(
                f"rank-{rank}-correlated-compute-delay",
                FailureType.DELAY,
                rank,
                FaultSurface.COMPUTE,
                parameters={"delay_ms": 1000.0},
            )
            for rank in (5, 6, 7)
        ),
    )

    intermittent_at = (
        next_iteration,
        next_iteration + 2,
        next_iteration + 4,
    )
    add(
        "intermittent-output-corruption",
        (
            _corruption(
                "rank-5-intermittent-output-sign-flip",
                5,
                FaultSurface.OUTPUT,
                CorruptionOperation.SIGN_FLIP,
                scope="100%",
            ),
        ),
        at=intermittent_at,
    )
    add(
        "permanent-output-corruption-until-recovery",
        (
            _corruption(
                "rank-6-permanent-output-scale",
                6,
                FaultSurface.OUTPUT,
                CorruptionOperation.SCALE,
                factor=10.0,
                scope="100%",
            ),
        ),
        gap_after=1,
        lifetime=IncidentLifetime(until="recovery"),
    )

    state_cases = (
        (
            "weight-corruption",
            _corruption(
                "rank-0-weight-sign-flip",
                0,
                FaultSurface.WEIGHT,
                CorruptionOperation.SIGN_FLIP,
                module_path=_BLOCK_PATH,
                parameter="weight",
                scope="10%",
            ),
        ),
        (
            "bias-corruption",
            _corruption(
                "rank-1-bias-scale",
                1,
                FaultSurface.BIAS,
                CorruptionOperation.SCALE,
                module_path=_BLOCK_PATH,
                parameter="bias",
                factor=10.0,
                scope="100%",
            ),
        ),
        (
            "gradient-corruption",
            _corruption(
                "rank-2-gradient-noise",
                2,
                FaultSurface.GRADIENT,
                CorruptionOperation.NOISE,
                module_path=_BLOCK_PATH,
                parameter="weight",
                std=1.0,
                scope="10%",
            ),
        ),
        (
            "optimizer-state-corruption",
            _corruption(
                "rank-3-optimizer-state-sign-flip",
                3,
                FaultSurface.OPTIMIZER_STATE,
                CorruptionOperation.SIGN_FLIP,
                module_path=_BLOCK_PATH,
                parameter="weight",
                state_key="exp_avg",
                scope="10%",
            ),
        ),
    )
    for incident_id, fault in state_cases:
        add(incident_id, (fault,), gap_after=1)

    rank = 4
    for failure_type in (FailureType.STALE_STATE, FailureType.DUPLICATE):
        for surface, parameter in (
            (FaultSurface.WEIGHT, "weight"),
            (FaultSurface.BIAS, "bias"),
            (FaultSurface.GRADIENT, "weight"),
            (FaultSurface.OPTIMIZER_STATE, "weight"),
        ):
            incident_id = f"{surface.value}-{failure_type.value}"
            parameters: dict[str, Any] = {
                "parameter": parameter,
                "scope": "100%",
            }
            if surface is FaultSurface.OPTIMIZER_STATE:
                parameters["state_key"] = "exp_avg"
            add(
                incident_id,
                (
                    _fault(
                        f"rank-{rank}-{incident_id}",
                        failure_type,
                        rank,
                        surface,
                        module_path=_BLOCK_PATH,
                        parameters=parameters,
                    ),
                ),
                gap_after=1,
            )
            rank = (rank + 1) % 8

    for failure_type, rank in (
        (FailureType.DROP, 4),
        (FailureType.REORDER, 5),
    ):
        incident_id = f"gradient-{failure_type.value}"
        add(
            incident_id,
            (
                _fault(
                    f"rank-{rank}-{incident_id}",
                    failure_type,
                    rank,
                    FaultSurface.GRADIENT,
                    module_path=_BLOCK_PATH,
                    parameters=(
                        {"parameter": "weight"}
                        if failure_type is FailureType.REORDER
                        else {"parameter": "weight", "scope": "100%"}
                    ),
                ),
            ),
            gap_after=1,
        )

    return FaultCampaign(
        name="pytorch-8gpu-systematic-faults",
        seed=17,
        clock=ClockSpec(),
        incidents=tuple(incidents),
        metadata={
            "world_size": 8,
            "workload": "examples.fault_injection.pytorch",
            "purpose": "Systematic built-in local fault and SCOUT localization matrix",
        },
    )


def _fault(
    fault_id: str,
    failure_type: FailureType,
    rank: int,
    surface: FaultSurface,
    *,
    module_path: str | None = None,
    parameters: dict[str, Any] | None = None,
) -> FaultSpec:
    return FaultSpec(
        fault_id=fault_id,
        type=failure_type,
        target=FaultTarget(
            rank=rank,
            component="transformer_block",
            index=0,
            module_path=module_path,
            surface=surface,
        ),
        parameters=parameters or {},
    )


def _corruption(
    fault_id: str,
    rank: int,
    surface: FaultSurface,
    operation: CorruptionOperation,
    *,
    module_path: str | None = None,
    **parameters: Any,
) -> FaultSpec:
    return _fault(
        fault_id,
        FailureType.TENSOR_CORRUPTION,
        rank,
        surface,
        module_path=module_path,
        parameters={"operation": operation.value, **parameters},
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).with_name("campaign.json"),
    )
    args = parser.parse_args()
    build_campaign().to_json(args.output)


if __name__ == "__main__":
    main()
