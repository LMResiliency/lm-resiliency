# Examples

The examples are organized by use case:

| Path | Purpose |
|---|---|
| [`quickstart.py`](quickstart.py) | Single-process CPU training and GEMINI recovery |
| [`production_loops/`](production_loops/) | Framework-native PyTorch, DeepSpeed, Megatron Core, and TorchTitan training loops |
| [`torchrun/`](torchrun/README.md) | Native torchrun adapter bootstrap, restart, and standby replacement |

## Quick Start

After installing `lm-resiliency`, run its packaged command to train a tiny
causal LM on CPU. The command executes the forward, backward, optimizer, and
GEMINI checkpoint lifecycle without requiring a GPU:

```bash
lm-resiliency-quickstart \
  --checkpoint-dir /tmp/lm-resiliency-quickstart/checkpoints \
  --run-id my-quickstart
```

Run it again with a larger step target to resume from the saved GEMINI checkpoint:

```bash
lm-resiliency-quickstart \
  --steps 6 \
  --checkpoint-dir /tmp/lm-resiliency-quickstart/checkpoints \
  --run-id my-quickstart
```

The installed command and library come from the same wheel. Developers working
from a source checkout can use the equivalent
[`quickstart.py`](quickstart.py) wrapper.

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

Run one example on a single eight-GPU host from the repository root:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=8 \
  --max-restarts=4 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint=/tmp/lm-resiliency-torchtitan-rdzv \
  --rdzv-id=torchtitan-production \
  --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-torchtitan-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/production_loops/policies/resiliency.toml" \
  --module \
  examples.production_loops.torchtitan \
  --validation-output-dir /tmp/torchtitan-production-loop
```

Replace the module, rendezvous paths, and validation output directory for
PyTorch, Megatron Core, or DeepSpeed. The worker infers the framework from
imports. The checked-in worker policy uses
`replication_jump=4` for this eight-rank topology. Other world layouts must use
a policy with a valid deployment-specific GEMINI pairing.

Each framework example runs ten steps by default; set `--steps` to change the
duration. Rank zero writes a framework summary under
`--validation-output-dir`; the directory is example validation output and is
not used for restart contexts or GEMINI checkpoints.

## Torchrun Workflows

See the [torchrun guide](torchrun/README.md) for:

- a CPU adapter-bootstrap smoke test;
- clean automatic-adapter checks across all four frameworks; and
- manager-driven same-node restart and SCOUT-localized standby replacement.
