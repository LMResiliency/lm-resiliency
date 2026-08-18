"""Replay-only SDC controls for torchrun fault-injection campaigns."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Any

import torch
from torch.utils._pytree import tree_flatten, tree_unflatten

from lm_resiliency import OrchestrationHooks


@dataclass
class ReplayFaultCampaign:
    """Inject one replay-only fault and validate checkpoint trust transitions."""

    steps: int
    rank: int
    world_size: int
    inject_fault: bool
    fault_step: int
    fault_rank: int
    faults: list[Any] = field(default_factory=list)
    recovery_decisions: list[dict[str, Any]] = field(default_factory=list)
    checkpoint_steps_at_fault: list[int] = field(default_factory=list)
    _current_step: int = 0
    _target_calls: int = 0
    _injected_calls: int = 0
    _handle: Any | None = None
    _fault_hook: Any | None = None

    @classmethod
    def from_args(
        cls,
        args: argparse.Namespace,
        *,
        rank: int,
        world_size: int,
    ) -> ReplayFaultCampaign:
        fault_rank = world_size - 1 if args.fault_rank < 0 else args.fault_rank
        if args.steps <= 0:
            raise ValueError("--steps must be positive")
        if args.inject_fault:
            if args.steps < args.fault_step + 2:
                raise ValueError("--steps must include at least two post-fault steps")
            if args.fault_step < 2:
                raise ValueError("--fault-step must follow at least one clean step")
            if fault_rank < 0 or fault_rank >= world_size:
                raise ValueError("--fault-rank must identify a global training rank")
        return cls(
            steps=args.steps,
            rank=rank,
            world_size=world_size,
            inject_fault=bool(args.inject_fault),
            fault_step=args.fault_step,
            fault_rank=fault_rank,
        )

    @property
    def orchestration(self) -> OrchestrationHooks:
        """Return manager hooks used to capture automatic recovery decisions."""
        return OrchestrationHooks(report_recovery=self.recovery_decisions.append)

    def start_step(self, step: int) -> None:
        """Mark the framework iteration before its model forward begins."""
        self._current_step = step
        self._target_calls = 0

    def bind(self, handle: Any) -> None:
        """Attach the replay-only fault after SCOUT has installed its hooks."""
        self._handle = handle
        if not self.inject_fault:
            return
        harness = _replay_harness(handle)
        if harness is None:
            raise AssertionError("SCOUT did not create a replay harness")
        self._fault_hook = harness.target_layer.register_forward_hook(
            self._inject_after_training_forward,
            with_kwargs=True,
        )

    def record_fault(self, result: Any) -> None:
        """Retain replay evidence and the checkpoint boundary visible at detection."""
        self.faults.append(result)
        manager = _checkpoint_manager(self._handle)
        if manager is not None:
            self.checkpoint_steps_at_fault.append(int(manager._last_saved_step))

    def validate(self, handle: Any, final_result: Any, expected_recipes: set[str]) -> None:
        """Validate clean completion or the complete transient-fault campaign."""
        manager = _checkpoint_manager(handle)
        if manager is None:
            raise AssertionError("GEMINI did not create a checkpoint manager")
        if final_result is None or any(final_result.sdc_bitmap):
            raise AssertionError("the final production-loop replay was not clean")
        if not expected_recipes.issubset(final_result.checked_recipe_ids):
            raise AssertionError(f"missing replay recipes: {final_result.checked_recipe_ids}")
        if int(manager._last_saved_step) != self.steps:
            raise AssertionError("GEMINI did not capture the final clean optimizer step")
        if int(manager.checkpoint_status.recovery_verified_step) != self.steps:
            raise AssertionError("the final clean checkpoint was not recovery-verified")

        if not self.inject_fault:
            if self.faults or self.recovery_decisions:
                raise AssertionError("the clean example reported an unexpected fault")
            return

        if len(self.faults) != 1:
            raise AssertionError(f"expected one injected fault, observed {len(self.faults)}")
        fault = self.faults[0]
        expected_bitmap = [int(peer_rank == self.fault_rank) for peer_rank in fault.peer_ranks]
        if fault.sdc_bitmap != expected_bitmap or any(fault.straggler_bitmap):
            raise AssertionError(
                f"fault localization mismatch: peers={fault.peer_ranks}, "
                f"sdc={fault.sdc_bitmap}, expected={expected_bitmap}"
            )
        if not any(source.startswith("hidden.") for source in fault.sdc_sources):
            raise AssertionError(f"hidden replay did not report the fault: {fault.sdc_sources}")
        if self.checkpoint_steps_at_fault != [self.fault_step - 1]:
            raise AssertionError(
                "GEMINI exposed the fault-step checkpoint before SCOUT certification"
            )
        if len(self.recovery_decisions) != 1:
            raise AssertionError(
                f"expected one recovery decision, observed {self.recovery_decisions}"
            )
        decision = self.recovery_decisions[0]
        expected_step = self.fault_step - 1
        if (
            decision["failure_kind"] != "sdc"
            or decision["recovery_mode"] != "recovery_verified"
            or decision["checkpoint_source"] != "gemini"
            or decision["checkpoint_step"] != expected_step
            or not decision["available"]
        ):
            raise AssertionError(f"invalid recovery decision: {decision}")
        if self.rank == self.fault_rank:
            if self._injected_calls == 0:
                raise AssertionError("the selected fault rank did not inject a replay fault")
        elif self._injected_calls != 0:
            raise AssertionError("a healthy rank injected a replay fault")

    def summary(self) -> dict[str, Any]:
        """Return JSON-ready fault campaign evidence."""
        if not self.inject_fault:
            return {"fault_injection": "disabled"}
        decision = self.recovery_decisions[0]
        return {
            "fault_injection": "transient hidden replay",
            "fault_rank": self.fault_rank,
            "fault_step": self.fault_step,
            "localized_bitmap": list(self.faults[0].sdc_bitmap),
            "recovery_mode": decision["recovery_mode"],
            "recovery_checkpoint_step": decision["checkpoint_step"],
            "post_fault_clean_steps": self.steps - self.fault_step,
        }

    def close(self) -> None:
        """Remove the example-owned injection hook."""
        if self._fault_hook is not None:
            self._fault_hook.remove()
            self._fault_hook = None

    def _inject_after_training_forward(
        self,
        _module: Any,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any | None:
        self._target_calls += 1
        if (
            self._current_step != self.fault_step
            or self.rank != self.fault_rank
            or self._target_calls == 1
        ):
            return None
        corrupted, changed = _add_to_first_tensor(output)
        if not changed:
            raise RuntimeError("the replay target produced no tensor output")
        self._injected_calls += 1
        return corrupted


def _add_to_first_tensor(value: Any) -> tuple[Any, bool]:
    leaves, spec = tree_flatten(value)
    changed = False
    for index, leaf in enumerate(leaves):
        if isinstance(leaf, torch.Tensor):
            leaves[index] = leaf + 1.0
            changed = True
            break
    return tree_unflatten(leaves, spec), changed


def _checkpoint_manager(handle: Any | None) -> Any | None:
    if handle is None:
        return None
    manager = getattr(handle, "ckpt_manager", None)
    return manager if manager is not None else getattr(handle, "_ckpt_manager", None)


def _replay_harness(handle: Any) -> Any | None:
    harness = getattr(handle, "replay_harness", None)
    return harness if harness is not None else getattr(handle, "_replay_harness", None)
