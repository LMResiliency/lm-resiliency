# Examples

All distributed training examples use the native `lm_resiliency` torchrun
backend. User training modules do not import `lm_resiliency`; the rendezvous
plugin validates the worker policy and installs the inferred framework adapter
before user code runs.

The examples are organized by use case:

| Path | Purpose |
|---|---|
| [`production_loops/`](production_loops/) | Framework-native PyTorch, DeepSpeed, Megatron Core, and TorchTitan training loops |
| [`torchrun/`](torchrun/README.md) | Native torchrun adapter bootstrap, restart, and standby replacement |

## Production Loops

The production-loop examples train tiny causal language models with
deterministic synthetic tokens while preserving each framework's real training
lifecycle. The four framework modules do not import `lm_resiliency`; the
`lm_resiliency` rendezvous backend infers and installs the framework adapter
from the user module's imports.

| Framework | Example | Framework-owned path |
|---|---|---|
| PyTorch | [pytorch.py](production_loops/pytorch.py) | DDP forward, backward, and `AdamW.step()` |
| TorchTitan | [torchtitan.py](production_loops/torchtitan.py) | `Trainer.train()` |
| Megatron Core | [megatron.py](production_loops/megatron.py) | `training.train()` and `train_step()` |
| DeepSpeed | [deepspeed.py](production_loops/deepspeed.py) | `DeepSpeedEngine.backward()` and `DeepSpeedEngine.step()` |

See the [torchrun resiliency guide](../docs/torchrun_resiliency.md) for the
integrated launch command and the [torchrun workflows](torchrun/README.md) for
runnable bootstrap, framework-matrix, and recovery campaigns. Select the
corresponding module under `examples.production_loops`; the worker infers the
framework from its imports.

Each framework example runs ten steps by default; set `--steps` to change the
duration. Rank zero writes a framework summary under
`--validation-output-dir`; the directory is example validation output and is
not used for restart contexts or GEMINI checkpoints.

## Torchrun Workflows

See the [torchrun guide](torchrun/README.md) for:

- a CPU adapter-bootstrap smoke test;
- clean automatic-adapter checks across all four frameworks; and
- coordinator-driven same-node restart and SCOUT-localized standby replacement.
