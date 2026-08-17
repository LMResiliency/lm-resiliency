# Examples

## Quick Start

After installing `lm-resiliency`, run its packaged command to train a tiny causal LM on CPU.
The example executes the complete forward, backward, optimizer, and GEMINI checkpoint lifecycle without requiring a GPU.

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

The installed command and library come from the same wheel.
Developers working from a source checkout can use the equivalent [quickstart.py](quickstart.py) wrapper.
The single-process example validates training-loop integration and recovery.
Use the distributed examples below to exercise SCOUT replay and multi-rank localization.

## Fault Injection Evaluation

Run the [fault injection and SCOUT localization](fault_injection/README.md)
example on eight GPUs to exercise the systematic evaluation path:

```bash
torchrun --standalone --nproc-per-node=8 --module \
  examples.fault_injection.pytorch \
  --artifact-dir /tmp/lm-resiliency-fault-evaluation
```

The example injects 48 scheduled occurrences and 53 rank-local actions into a
real eight-GPU DDP production loop, collects the normalized JSON localization
emitted by `enable_resiliency()`, and compares it with verified injection
ground truth.
See the [fault injection guide](../docs/fault_injection.md) for campaign
manifests, target selection, supported faults, and safety boundaries.

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
