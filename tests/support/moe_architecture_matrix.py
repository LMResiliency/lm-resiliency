"""Architecture presets and deterministic routing manifests for MoE qualification."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MoEArchitecturePreset:
    """One architecture-inspired local grouped-expert projection."""

    name: str
    local_experts: int
    hidden: int
    expert_output: int
    max_n_exec: int
    global_experts: int
    top_k: int
    expert_parallel: int
    description: str

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("architecture preset name cannot be empty")
        if (
            min(
                self.local_experts,
                self.hidden,
                self.expert_output,
                self.max_n_exec,
                self.global_experts,
                self.top_k,
                self.expert_parallel,
            )
            < 1
        ):
            raise ValueError("architecture preset dimensions must be positive")
        if self.top_k > self.global_experts:
            raise ValueError("top_k cannot exceed the global expert count")
        if self.global_experts % self.expert_parallel:
            raise ValueError("global experts must be divisible by expert parallelism")
        if self.global_experts // self.expert_parallel != self.local_experts:
            raise ValueError("local experts must equal global_experts / expert_parallel")


ARCHITECTURE_PRESETS = {
    preset.name: preset
    for preset in (
        MoEArchitecturePreset(
            name="large-1-local",
            local_experts=1,
            hidden=4096,
            expert_output=14336,
            max_n_exec=512,
            global_experts=8,
            top_k=2,
            expert_parallel=8,
            description="Large experts with one local expert per EP rank.",
        ),
        MoEArchitecturePreset(
            name="large-2-local",
            local_experts=2,
            hidden=4096,
            expert_output=14336,
            max_n_exec=512,
            global_experts=8,
            top_k=2,
            expert_parallel=4,
            description="Large experts with two local experts per EP rank.",
        ),
        MoEArchitecturePreset(
            name="wide-2-local",
            local_experts=2,
            hidden=6144,
            expert_output=10752,
            max_n_exec=512,
            global_experts=16,
            top_k=4,
            expert_parallel=8,
            description="Wide hidden state with two local experts per EP rank.",
        ),
        MoEArchitecturePreset(
            name="medium-4-local",
            local_experts=4,
            hidden=4096,
            expert_output=4096,
            max_n_exec=512,
            global_experts=16,
            top_k=2,
            expert_parallel=4,
            description="Medium-width experts with four local experts.",
        ),
        MoEArchitecturePreset(
            name="narrow-8-local",
            local_experts=8,
            hidden=5120,
            expert_output=1536,
            max_n_exec=512,
            global_experts=64,
            top_k=6,
            expert_parallel=8,
            description="Fine-grained narrow experts with eight local experts.",
        ),
        MoEArchitecturePreset(
            name="fine-16-local",
            local_experts=16,
            hidden=2048,
            expert_output=1024,
            max_n_exec=512,
            global_experts=64,
            top_k=8,
            expert_parallel=4,
            description="Many fine-grained local experts.",
        ),
    )
}


def per_expert_n_exec_values(
    preset: MoEArchitecturePreset,
    *,
    min_n_exec: int = 1,
    max_n_exec: int | None = None,
) -> tuple[int, ...]:
    """Enumerate the scalar physical row-count range for one expert GEMM."""
    maximum = preset.max_n_exec if max_n_exec is None else max_n_exec
    if min_n_exec < 1 or maximum < 1:
        raise ValueError("n_exec bounds must be positive")
    if min_n_exec > maximum:
        raise ValueError("minimum n_exec cannot exceed maximum n_exec")
    return tuple(range(min_n_exec, maximum + 1))
