"""Framework-selecting worker for torchrun fault-injection campaigns."""

from __future__ import annotations

import argparse
import os
from collections.abc import Callable
from pathlib import Path

from lm_resiliency import FaultCampaign
from lm_resiliency.integrations.torchrun import (
    TorchrunWorkerContext,
    get_torchrun_worker_context,
)

from .campaign import CAMPAIGN_FILENAME, pressure_events
from .replay_fault import ReplayFaultCampaign
from .runtime import (
    CampaignRuntime,
    DriverConfig,
    FrameworkDriver,
    checkpoint_config,
    replay_config,
)

FRAMEWORKS = ("pytorch", "deepspeed", "megatron", "torchtitan")


def _validate_worker_context(context: TorchrunWorkerContext | None) -> None:
    if context is None:
        return
    if context.local_world_size != 1:
        raise RuntimeError("fault-injection validation requires one worker per torchrun agent")
    if context.generation > 0 and context.checkpoint_source != "gemini":
        raise RuntimeError("resiliency-cycle successors require GEMINI recovery contexts")


def _driver_factory(framework: str) -> Callable[[DriverConfig], FrameworkDriver]:
    if framework == "pytorch":
        from .frameworks.pytorch import PyTorchDriver

        return PyTorchDriver
    if framework == "deepspeed":
        from .frameworks.deepspeed import DeepSpeedDriver

        return DeepSpeedDriver
    if framework == "megatron":
        from .frameworks.megatron import MegatronDriver

        return MegatronDriver
    if framework == "torchtitan":
        from .frameworks.torchtitan import TorchTitanDriver

        return TorchTitanDriver
    raise ValueError(f"unsupported framework {framework!r}")


def run_worker(
    *,
    framework: str,
    mode: str,
    fault_campaign_dir: Path,
) -> None:
    if mode not in {"baseline", "campaign"}:
        raise ValueError("mode must be 'baseline' or 'campaign'")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    context = get_torchrun_worker_context() if mode == "campaign" else None
    _validate_worker_context(context)
    generation = 0 if context is None else context.generation
    node_id = f"baseline-rank-{rank}" if context is None else context.node_id

    manifest = FaultCampaign.from_json(fault_campaign_dir / CAMPAIGN_FILENAME)
    events = pressure_events(manifest)
    expected_world_size = int(manifest.metadata["active_nodes"])
    if world_size != expected_world_size:
        raise RuntimeError(f"expected {expected_world_size} ranks, got {world_size}")
    if world_size < 4 or world_size % 2:
        raise RuntimeError("fault-injection validation requires an even world of at least four")
    total_steps = int(manifest.metadata["total_steps"])
    event = None if mode == "baseline" or generation >= len(events) else events[generation]
    replacement_event = event is not None and event.kind == "replacement"
    replay_campaign = ReplayFaultCampaign(
        steps=total_steps,
        rank=rank,
        world_size=world_size,
        inject_fault=replacement_event,
        fault_step=event.step if replacement_event else total_steps,
        fault_rank=event.fault_rank if replacement_event else 0,
    )
    runtime = CampaignRuntime(
        campaign_dir=fault_campaign_dir,
        context=context,
        event=event,
        framework=framework,
        generation=generation,
        mode=mode,
        node_id=node_id,
        rank=rank,
        replay_campaign=replay_campaign,
        total_steps=total_steps,
    )
    driver = _driver_factory(framework)(
        DriverConfig(
            campaign_dir=fault_campaign_dir,
            checkpoint=checkpoint_config(
                campaign_dir=fault_campaign_dir,
                framework=framework,
                mode=mode,
                replication_jump=world_size // 2,
            ),
            replay=replay_config(),
            recovery_mode=context.recovery_mode if context is not None else None,
            recovery_step=context.checkpoint_step if context is not None else None,
            expected_topology_id=context.topology_digest if context is not None else None,
            fault_callback=runtime.record_fault,
            orchestration=runtime.orchestration,
            total_steps=total_steps,
        )
    )
    try:
        runtime.initialize(driver)
        driver.run(
            before_step=runtime.before_step,
            after_step=lambda step, loss: runtime.after_step(driver, step, loss),
            total_steps=total_steps,
        )
        runtime.finish(driver)
    finally:
        try:
            runtime.close()
        finally:
            driver.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    parser.add_argument("--mode", choices=("baseline", "campaign"), required=True)
    parser.add_argument("--fault-campaign-dir", type=Path, required=True)
    args = parser.parse_args()
    run_worker(
        framework=args.framework,
        mode=args.mode,
        fault_campaign_dir=args.fault_campaign_dir,
    )


if __name__ == "__main__":
    main()
