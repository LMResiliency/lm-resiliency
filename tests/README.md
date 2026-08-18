# Tests

The test tree is organized by execution contract.

| Directory | Purpose | Default CI |
|---|---|---|
| `unit/` | Fast CPU contracts for public APIs and internal components | Yes |
| `integration/core/` | Focused distributed GEMINI and SCOUT component tests | No |
| `integration/frameworks/` | Real framework engines and distributed topology lifecycles | No |
| `validation/fault_injection/` | Eight-GPU fault-injection and SCOUT localization qualification | No |
| `validation/moe/` | Expensive GPU qualification and MoE regime campaigns | No |
| `support/` | Shared models and validation helpers | N/A |

Run the default CPU suite:

```bash
python -m pytest -q
```

Distributed programs document their required `torchrun` command in the module docstring.
Production-loop integration is exercised through `examples/production_loops/`.
Systematic fault injection and localization are covered by
`validation/fault_injection/`.
The public fault injection evaluation kit is covered by `unit/test_fault_injection.py`.
MoE validation is manual because it depends on specific GPU, Triton, Megatron Core, and Transformer Engine environments.
Reproducible healthy-path performance commands and regression thresholds are documented in [benchmarks](../benchmarks/README.md).

## Automated GPU Qualification

The scheduled and maintainer-dispatched [GPU Qualification workflow](../.github/workflows/gpu-qualification.yml) runs a two-GPU, one-host trusted tier on a self-hosted runner. It covers trajectory-equivalent checkpoint recovery, Gloo/NCCL replay, synchronized dropout RNG, structured invocation replay, FSDP2 local-shard checkpoint recovery, and automatic exit cleanup.

The runner must be an isolated Linux x64 host with at least two visible CUDA GPUs and the labels `self-hosted`, `linux`, `x64`, `gpu`, and `lm-resiliency`. The workflow is deliberately not triggered by pull requests, so untrusted fork code cannot execute on the privileged runner. Prefer an ephemeral runner image and do not attach production credentials or writable production storage.

Every run uploads `environment.json`, `commands.txt`, per-command logs, `summary.json`, `summary.md`, and SHA-256 checksums under an artifact named for the exact commit. Run the same tier manually with:

```bash
python tests/validation/run_gpu_qualification.py \
  --artifact-dir /tmp/lm-resiliency-gpu-qualification \
  --minimum-gpus 2
```

Eight/sixteen-GPU, multi-node, optional-framework, and MoE campaigns remain release qualification rather than part of this frequent tier.
