# Tests

The test tree is organized by execution contract.

| Directory | Purpose | Default CI |
|---|---|---|
| `unit/` | Fast CPU contracts for public APIs and internal components | Yes |
| `integration/core/` | Focused distributed GEMINI and SCOUT component tests | No |
| `integration/frameworks/` | Real framework engines and distributed topology lifecycles | No |
| `validation/moe/` | Expensive GPU qualification and MoE regime campaigns | No |
| `support/` | Shared models, fault injection, and validation helpers | N/A |

Run the default CPU suite:

```bash
python -m pytest -q
```

Distributed programs document their required `torchrun` command in the module docstring.
Production-loop integration is exercised directly through `examples/production_loops/` with `--inject-fault`.
MoE validation is manual because it depends on specific GPU, Triton, Megatron Core, and Transformer Engine environments.
