# MoE Execution Regimes

SCOUT maps variable expert-GEMM row counts to a bounded replay catalog.
Catalog discovery is offline and bound to one GPU, software stack, model configuration, precision recipe, backend, and parallelism configuration.

The scalar catalog domain is `n_exec`, the physical row count received by one expert GEMM after dispatch, capacity handling, token dropping, padding, and alignment.
Local and global expert counts are environment metadata unless the production kernel physically executes several expert problems together.
Changing the number of jointly executed experts can change regime boundaries and requires a separately qualified catalog.

The implementation is in [`lm_resiliency/detection/moe_regimes.py`](../lm_resiliency/detection/moe_regimes.py).

## Coverage Contract

Every observation records:

- ordered kernel names and launch geometry;
- algorithm, tile, block shape, tail path, and alignment;
- workspace, pressure, and overlap class;
- direct or persistent scheduling mode;
- logical-work count and CTA-to-work mapping;
- disjoint semantic-role counts; and
- declaration and independent qualification evidence.

`exact_launch` retains every distinct physical shape.
`plan_and_pressure` may compress scalable work only when every hard execution property agrees.

Within one regime, representative `A` covers target `B` only when `A` executes at least as many instances of every modeled semantic role.
Discovery retains the smallest observed frontier that covers all admitted shapes.
The recipe count is therefore an objective result, not a configured target.

`max_replay_recipes` is an acceptance ceiling.
Discovery never merges hard boundaries to satisfy it.

Semantic roles require two independent sources, such as generated kernel metadata and an instrumented logical-work trace.
Missing requests, unstable fingerprints, unknown execution fields, mismatched role partitions, or non-independent evidence force exact fallback or reject the catalog.

This contract covers permanent or reproducible faults tied to modeled roles and execution regimes.

## Discovery Workflow

### Enumerate Physical Shapes

Enumerate every admitted `n_exec` for each homogeneous layer, expert, and EP position:

```python
from lm_resiliency.detection.moe_regimes import build_profile_requests

requests = build_profile_requests(
    layer_ids=range(num_hidden_layers),
    expert_ids=experts_at_ep_position,
    ep_position=0,
    n_exec_values=all_admitted_physical_lengths,
)
```

Build separate execution classes when projections, precision recipes, kernels, or other execution-defining settings differ.
Do not enumerate combinations of complete `tokens_per_expert` vectors for a scalar catalog.

### Record the Environment

Use `current_moe_environment()` to record the backend and version, precision, model dimensions, routing and alignment policy, parallelism, GPU, CUDA, PyTorch, NCCL, driver, and deployment-specific execution settings.
Placeholder values such as `"unknown"` and `"<version>"` are rejected.
Loading under a different identity raises `CatalogEnvironmentMismatch`.

### Profile the Production Expert Stage

Profile the same fused or grouped expert path used by training:

```python
from lm_resiliency.detection.moe_regimes import (
    TorchCudaExecutionProfiler,
    profile_requests,
    save_observations,
    save_profile_requests,
)

profiler = TorchCudaExecutionProfiler(
    run_production_expert_stage,
    hints=qualified_backend_hints,
    warmup=3,
)
observations = profile_requests(requests, profiler)

save_observations(observations, "moe-observations.jsonl")
save_profile_requests(requests, "moe-manifest.json")
```

Profiling repeats each request three times by default.
Discovery rejects missing observations or conflicting fingerprints for the same shape.
Profiler launch geometry alone is insufficient because it does not identify logical-work roles.

### Build and Audit the Catalog

```python
from lm_resiliency.detection.moe_regimes import discover_execution_regimes

catalog = discover_execution_regimes(
    observations,
    environment=environment,
    equivalence_policy="plan_and_pressure",
    expected_requests=requests,
    max_replay_recipes=50,
)
catalog.save("moe-regimes-ep0.json")
```

The installed CLI builds the same catalog from saved observations:

```bash
lm-resiliency-discover-moe-regimes \
  --observations moe-observations.jsonl \
  --environment moe-environment.json \
  --manifest moe-manifest.json \
  --output moe-regimes-ep0.json \
  --max-replay-recipes 50
```

Audit exact and compressed catalogs from the same observations, complete request coverage, fingerprint stability, hard boundaries, representative role dominance, serialization, environment mismatch rejection, omitted-role challenges, wrong-count challenges, and independent fault injections.

### Configure Online Replay

```python
from lm_resiliency import (
    GroupedExpertMaterializer,
    ReplayHarnessConfig,
    ReplayWorkload,
    enable_resiliency,
)

catalog.validate_environment(current_environment)

workload = ReplayWorkload.from_moe_catalog(
    catalog,
    replay_modules=[model.post_dispatch_expert_stage],
    materializer=GroupedExpertMaterializer(),
)

resiliency = enable_resiliency(
    model,
    optimizer,
    interval=10,
    replay=ReplayHarnessConfig(
        workload=workload,
        capture_inputs_by_value=True,
        rotate_layers=False,
    ),
)
```

`GroupedExpertMaterializer` converts scalar `n_exec` into `[n_exec] * local_expert_count` and resizes packed token-aligned inputs.
Use a backend-specific materializer when physical shape is not represented by packed rows and one count vector.

The replay harness compares schedule signatures across peers, advances only after a successful replay, and checkpoints its catalog position.
For `K` recipes at interval `I`, one complete cycle takes `K * I` optimizer steps.

## Validation Method

The validation campaigns used an instrumented Triton persistent grouped-GEMM with BF16 on NVIDIA A100-SXM4-40GB GPUs.
The environment used NVIDIA driver 580.126.16, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.

Each admitted scalar shape was executed three times.
The campaigns recorded launch geometry, declared semantic roles from generated metadata, independently qualified roles from device traces, built exact and compressed catalogs, injected every selected work-item occurrence, and challenged omitted roles, incorrect counts, and environment drift.

The primary scalar range was `n_exec=1..3457`.
Controlled projection and grouped-expert matrices exhaustively covered 512, 1,024, and 2,048 scalar shapes.

## Scalar Compression Results

| Metric | Result |
|---|---:|
| Scalar physical shapes | 3,457 |
| Stable observations | 10,371 |
| Exact recipes | 3,457 |
| Execution regimes | 4 |
| Compressed recipes | 16 |
| Recipe reduction | 99.54% |
| Compression ratio | 216.06x |
| Target role-occurrence obligations | 1,134,224 |
| Representative work-item injections | 5,960/5,960 |
| Representative kernel-role classes | 237 |
| Healthy distributed replay | Passed on 8 and 16 A100s |
| Fault localization | Rank 7 on 8 GPUs; rank 15 on 16 GPUs |
| Omitted-role challenge | Fell back to exact |
| Wrong-count challenge | Rejected compression |
| Environment-drift challenge | Rejected stale catalog |

The objective 16-recipe catalog comprised:

| Forward / input-gradient / weight-gradient pressure | Scalar members | Representatives |
|---|---:|---|
| `underfilled / underfilled / underfilled` | 832 | 31, 32, 63, 800, 831, 832 |
| `underfilled / underfilled / one-repeat` | 896 | 1696, 1727, 1728 |
| `underfilled / underfilled / multi-repeat` | 1,664 | 3360, 3391, 3392 |
| `one-repeat / one-repeat / multi-repeat` | 65 | 3424, 3455, 3456, 3457 |

Representatives clustered at row-tail, reduction-tail, and persistent-pressure transitions.

## Projection and Range Results

The projection matrix profiled one physical expert across `n_exec=1..512`, `1..1024`, and `1..2048`.
Each domain column reports `execution regimes / compressed recipes`.

| Preset | Local-expert metadata | Projection | 512 | 1,024 | 2,048 | 2,048 reduction |
|---|---:|---:|---:|---:|---:|---:|
| `large-1-local` | 1 | 4096 x 14336 | 3 / 12 | 3 / 12 | 3 / 12 | 99.41% |
| `large-2-local` | 2 | 4096 x 14336 | 3 / 12 | 3 / 12 | 3 / 12 | 99.41% |
| `wide-2-local` | 2 | 6144 x 10752 | 3 / 10 | 3 / 10 | 3 / 10 | 99.51% |
| `medium-4-local` | 4 | 4096 x 4096 | 3 / 12 | 3 / 12 | 3 / 12 | 99.41% |
| `narrow-8-local` | 8 | 5120 x 1536 | 4 / 15 | 5 / 18 | 5 / 18 | 99.12% |
| `fine-16-local` | 16 | 2048 x 1024 | 3 / 16 | 4 / 19 | 4 / 19 | 99.07% |

| Domain | Exact recipes per preset | Observations per preset | Shards per preset | Maximum compressed recipes | Result |
|---|---:|---:|---:|---:|---|
| `1..512` | 512 | 1,536 | 1 | 16 | Passed |
| `1..1024` | 1,024 | 3,072 | 8 | 19 | Passed |
| `1..2048` | 2,048 | 6,144 | 8 | 19 | Passed |

All 96 profile shards passed numerical and fingerprint-stability checks and contributed 55,296 stable observations.
Increasing the exact domain from 512 to 2,048 did not increase the recipe count for the large, wide, or medium projections.
The narrow and fine projections crossed one additional pressure boundary between 512 and 1,024 and added no further recipes between 1,024 and 2,048.

The identical `large-1-local` and `large-2-local` results confirm that expert-count metadata alone does not create a recipe axis.
Projection dimensions changed the objective count only when they changed execution geometry or crossed a hard pressure boundary.

## Grouped-Kernel Results

The grouped campaign physically allocated and executed `E={1,2,4,8,16}` distinct experts:

```text
tokens_per_expert = [n_exec] * E
packed_rows = n_exec * E
```

Each domain column reports `execution regimes / compressed recipes`.

| Actual experts | 512 | 1,024 | 2,048 | 2,048 reduction | 2,048 role injections |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 / 6 | 2 / 9 | 3 / 12 | 99.41% | 171/171 |
| 2 | 2 / 9 | 3 / 12 | 4 / 18 | 99.12% | 285/285 |
| 4 | 3 / 12 | 4 / 15 | 5 / 27 | 98.68% | 520/520 |
| 8 | 4 / 18 | 5 / 30 | 5 / 42 | 97.95% | 870/870 |
| 16 | 5 / 24 | 5 / 36 | 5 / 48 | 97.66% | 1,040/1,040 |

| Domain | Exact recipes per expert count | Observations per expert count | Total observations | Maximum compressed recipes |
|---|---:|---:|---:|---:|
| `1..512` | 512 | 1,536 | 7,680 | 24 |
| `1..1024` | 1,024 | 3,072 | 15,360 | 36 |
| `1..2048` | 2,048 | 6,144 | 30,720 | 48 |

All numerical checks, repeated fingerprints, catalog round trips, and 4,757 selected-role injections passed in the extended campaigns.
Repeating the campaigns with the two hosts exchanged reproduced catalog identities, recipe counts, injection counts, and heterogeneous coverage counts.

The exact count remained equal to the scalar-domain size and independent of `E`.
The compressed count grew when physical grouped execution introduced new pressure regimes or non-dominated role-count frontiers.
At `E=16`, the 2,048-shape catalog required 48 recipes, so the tested 50-recipe ceiling must not be extrapolated to larger domains.

### Heterogeneous Routing

Uniform catalogs were challenged with tile and pressure boundaries, one-active-expert cases, alternating and cyclic patterns, and 128 seeded random vectors.
Cycle role-envelope coverage required selected recipes in a target's hard regime to dominate every target role count collectively.

| Domain | Actual experts | Vectors | Hard regime represented | Single-recipe dominance | Cycle role-envelope coverage | Result |
|---:|---:|---:|---:|---:|---:|---|
| 512 | 2 | 245 | 245/245 | 135/245 | 245/245 | Pass |
| 512 | 4 | 343 | 343/343 | 135/343 | 271/343 | Fail |
| 512 | 8 | 377 | 377/377 | 96/377 | 240/377 | Fail |
| 512 | 16 | 385 | 385/385 | 35/385 | 111/385 | Fail |
| 1,024 | 2 | 266 | 266/266 | 157/266 | 239/266 | Fail |
| 1,024 | 4 | 373 | 373/373 | 183/373 | 302/373 | Fail |
| 1,024 | 8 | 407 | 407/407 | 110/407 | 284/407 | Fail |
| 1,024 | 16 | 385 | 385/385 | 54/385 | 138/385 | Fail |
| 2,048 | 2 | 287 | 287/287 | 167/287 | 254/287 | Fail |
| 2,048 | 4 | 403 | 403/403 | 168/403 | 286/403 | Fail |
| 2,048 | 8 | 407 | 407/407 | 142/407 | 304/407 | Fail |
| 2,048 | 16 | 385 | 385/385 | 69/385 | 185/385 | Fail |

Every heterogeneous target belonged to a represented hard regime, but uniform scalar cycles did not cover arbitrary heterogeneous routing for `E>=2` in the extended domains.
For example, `[1, 1024]` combines a queue-first tail role with high pressure in a way no selected uniform representative covers.

Uniform domains of up to 2,048 shapes therefore compressed to 12-48 recipes across 1-16 jointly executed experts without exponential growth.
Broader heterogeneous coverage requires qualified vector templates, a different materializer, or a narrower backend-specific fault model.

## Reproduction

Run the scalar campaign:

```bash
python tests/validation/moe/validate_moe_regime_compression.py profile \
  --artifact-dir /tmp/scout-moe-scalar-compression
```

Run one projection preset:

```bash
python tests/validation/moe/evaluate_moe_architecture_matrix.py \
  --preset medium-4-local \
  --max-n-exec 2048 \
  --artifact-dir /tmp/scout-moe-scalar-matrix/medium-4-local
```

Run one grouped-expert matrix member:

```bash
python tests/validation/moe/validate_grouped_kernel_expert_count.py \
  --actual-experts 4 \
  --max-n-exec 2048 \
  --artifact-dir /tmp/scout-grouped-kernel-experts/e4 \
  --fault-occurrences roles
```

Run distributed replay:

```bash
torchrun --nproc-per-node=8 \
  tests/validation/moe/validate_moe_regime_compression.py distributed \
  --catalog /tmp/scout-moe-scalar-compression/catalog-compressed.json
```

For parallel acquisition, profile disjoint scalar intervals and pass each completed artifact directory to a final invocation with `--profile-shards`.
Catalog discovery runs once over the merged exhaustive observation set.

## Qualification Checklist

Before production use:

1. Profile the exact target GPU and software stack.
2. Cover every admitted physical shape and homogeneous execution class.
3. Compare forward outputs, input gradients, and parameter gradients with trusted references.
4. Verify stable repeated fingerprints and every hard regime boundary.
5. Qualify semantic roles from independent evidence.
6. Verify representative coverage of every target obligation.
7. Run omitted-role, wrong-count, and environment-drift challenges.
8. Inject every selected role occurrence with an independent software oracle.
9. Run one complete healthy catalog cycle on the target topology.
10. Inject a rank-specific reproducible fault and require localization within one `K * I` cycle.
11. Retain exact recipes when the backend cannot expose sufficient semantics.

Report exact and compressed recipe counts, compression ratio, role obligations, injected occurrences, topology, environment identity, and claim boundaries.

## Boundaries

- Results qualify the instrumented Triton backend, A100, BF16, tested projections, grouped expert counts, and stated scalar domains.
- Transformer Engine used exact recipes because independent CTA-role semantics were unavailable.
- Other backends, dimensions, precisions, GPUs, compilers, and scheduling policies require separate catalogs.
- Software injection covers modeled semantic roles for permanent or reproducible triggers.
- One-shot transients and common-mode defects remain outside the result.
- Uniform scalar grouped-expert catalogs are not qualified for arbitrary heterogeneous routing vectors.
