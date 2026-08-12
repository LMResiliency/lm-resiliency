"""Build a SCOUT MoE execution-regime catalog from GPU profiler observations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lm_resiliency.detection.moe_regimes import (
    MoEExecutionEnvironment,
    discover_execution_regimes,
    load_observations,
    load_profile_requests,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Partition physical MoE profiler fingerprints into trigger-equivalent "
            "execution regimes."
        )
    )
    parser.add_argument(
        "--observations",
        required=True,
        type=Path,
        help="JSON Lines produced by save_observations() on the target GPU",
    )
    parser.add_argument(
        "--environment",
        required=True,
        type=Path,
        help="JSON object describing the exact hardware/software/model configuration",
    )
    parser.add_argument(
        "--manifest",
        required=True,
        type=Path,
        help="complete request manifest JSON produced by save_profile_requests()",
    )
    parser.add_argument("--output", required=True, type=Path, help="catalog JSON path")
    parser.add_argument(
        "--representatives-per-regime",
        type=int,
        default=1,
        help="minimum shapes retained per regime; per-role count coverage may require more",
    )
    parser.add_argument(
        "--equivalence-policy",
        choices=("plan_and_pressure", "exact_launch"),
        default="plan_and_pressure",
        help=(
            "plan_and_pressure groups scalable CTA counts when kernel plans, mapping, "
            "and pressure agree, then covers semantic roles; exact_launch separates shapes"
        ),
    )
    parser.add_argument(
        "--minimum-observations-per-request",
        type=int,
        default=3,
        help="minimum stable repetitions required for every manifest request",
    )
    parser.add_argument(
        "--max-replay-recipes",
        type=int,
        default=32,
        help="reject a compressed catalog larger than this online replay budget",
    )
    args = parser.parse_args()
    if args.minimum_observations_per_request < 3:
        parser.error("--minimum-observations-per-request must be at least 3")
    if args.max_replay_recipes < 1:
        parser.error("--max-replay-recipes must be positive")

    environment = MoEExecutionEnvironment.from_mapping(json.loads(args.environment.read_text()))
    observations = load_observations(args.observations)
    manifest = load_profile_requests(args.manifest)
    catalog = discover_execution_regimes(
        observations,
        environment=environment,
        representatives_per_regime=args.representatives_per_regime,
        equivalence_policy=args.equivalence_policy,
        expected_requests=manifest,
        minimum_observations_per_request=args.minimum_observations_per_request,
        max_replay_recipes=args.max_replay_recipes,
    )
    catalog.save(args.output)

    print(
        f"wrote {args.output}: {len(catalog.regimes)} regimes, "
        f"{catalog.cycle_size} replay recipes, EP position {catalog.ep_position}, "
        f"catalog {catalog.identifier[:12]}"
    )


if __name__ == "__main__":
    main()
