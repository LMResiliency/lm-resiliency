# Compatibility

`lm-resiliency` supports the version ranges below.
The lower bounds and primary development versions pass the complete CPU unit suite.
Distributed GPU validation covers the exact framework versions listed as tested.

## Version Matrix

| Component | Supported | Tested |
|---|---|---|
| Python | `>=3.10,<3.13` | 3.10, 3.11, and 3.12 |
| NumPy | `>=1.26,<3` | 1.26.4 and 2.5.1 |
| PyTorch | `>=2.10,<2.14` | 2.10.0, 2.11.0, 2.12.0, and 2.13.0 |
| TorchTitan | `>=0.2.2,<0.3` | 0.2.2 |
| Megatron Core | `>=0.18.2,<0.19` | 0.18.2 |
| DeepSpeed | `>=0.19.4,<0.20` | 0.19.4 |
| Transformer Engine grouped experts | Integration-provided | 2.10.0 |

The complete CPU unit suite qualifies these representative combinations:

| CI target | Python | NumPy | PyTorch |
|---|---|---|---|
| `minimum` | 3.10 | 1.26.4 | 2.10.0 |
| `python-3.11-torch-2.11` | 3.11 | 1.26.4 | 2.11.0 |
| `torch-2.12` | 3.12 | 2.5.1 | 2.12.0 |
| `primary` | 3.12 | 2.5.1 | 2.13.0 |

The primary combination also passes the distributed framework campaign.
Production Megatron grouped-expert validation used PyTorch 2.10.0 and Transformer Engine 2.10.0.

CUDA, NCCL, GPU, and topology details for distributed runs are recorded in [Validation](validation.md).

## Compatibility Policy

Patch releases inside the declared ranges are supported unless an upstream regression is documented.
New Python or framework minor series are unsupported until the CPU suite and applicable distributed integration tests pass.
Each newly qualified minor series requires an explicit metadata update instead of an open-ended upper bound.
Framework adapters may use framework-specific topology and optimizer APIs, so successful dependency resolution alone is not treated as compatibility evidence.
Compatibility defects within a declared range are handled as release bugs and may require narrowing the range in a patch release.
Contract tests require every declared Python and PyTorch minor series to appear in the CI matrix and require every CI combination to appear in this guide.

## API Compatibility

Stable exports from `lm_resiliency`, `lm_resiliency.manager_api`, and explicit framework integration entry points remain backward compatible across `0.1.x` patch releases.
The documented `lm-resiliency-quickstart` and `lm-resiliency-discover-moe-regimes` commands and their existing options follow the same patch-release compatibility policy.
Removing or changing a stable interface requires a new minor release and migration notes.
Objects under `lm_resiliency.experimental` and unlisted module paths may change in any `0.x` release.
The exact stable exports are listed in the [API guide](api.md#public-api-stability) and enforced by contract tests.

## Manager Compatibility

The separately distributed manager must depend on the same `0.x` minor series, for example `lm-resiliency>=0.1,<0.2`.
The manager repository owns wheel-to-wheel compatibility testing because this repository does not include manager code or manager tests.
That test must build both wheels, install them into a clean environment, and exercise `OrchestrationHooks`, normalized SCOUT reports, restart flushing, and checkpoint transfer through public imports.
