"""Run a framework-neutral fault-injection localization evaluation on CPU."""

from __future__ import annotations

import json

import torch
import torch.nn as nn

from lm_resiliency import (
    FaultCampaign,
    FaultLocation,
    FaultPersistence,
    FaultSpec,
    FaultTarget,
    FaultType,
    LocalizationResult,
    enable_fault_injection,
)


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(4, 4)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.hidden(value)


def main() -> None:
    torch.manual_seed(7)
    model = TinyModel()
    campaign = FaultCampaign(
        name="cpu-output-sign-flip",
        framework="pytorch",
        faults=(
            FaultSpec(
                fault_id="hidden-output",
                fault_type=FaultType.SIGN_FLIP,
                target=FaultTarget(
                    rank=0,
                    module="hidden",
                    location=FaultLocation.OUTPUT,
                ),
                steps=(2,),
                persistence=FaultPersistence.TRANSIENT,
            ),
        ),
        metadata={"workload": "examples/fault_injection.py"},
    )

    with enable_fault_injection(model, campaign) as session:
        for step in range(1, 4):
            session.trigger(step)
            model(torch.ones(1, 4))

        record = session.records[0]
        report = session.evaluate(
            [
                LocalizationResult(
                    injection_id=record.injection_id,
                    detected=True,
                    failed_ranks=(0,),
                    kind="sdc",
                    component="hidden",
                )
            ]
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
