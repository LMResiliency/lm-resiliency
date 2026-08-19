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

Run one example on two eight-GPU hosts from the repository root. Start the same
command on both hosts and set `RDZV_HOST` to a hostname or IP address reachable
from both:

```bash
RDZV_HOST=node-a

# --nnodes=1:2 keeps one node active and parks a second node as standby.
# --nproc-per-node=8 launches eight training workers only on the active node.
# --max-restarts=4 allows each torchrun agent to restart its workers four times.
# --rdzv-backend=lm_resiliency enables active/standby admission and recovery.
torchrun \
  --nnodes=1:2 \
  --nproc-per-node=8 \
  --max-restarts=4 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint="${RDZV_HOST}:29400" \
  --rdzv-id=torchtitan-production \
  --rdzv-conf="store_type=tcp,read_timeout=120,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-torchtitan-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/production_loops/policies/resiliency.toml" \
  --module \
  examples.production_loops.torchtitan \
  --validation-output-dir /tmp/torchtitan-production-loop
```

The first registered host receives the eight logical training ranks. The other
host remains parked as a standby until torchrun selects it to replace a faulty
host. Use `--nnodes=1:1` only for a single-host run without standby
replacement.

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
- coordinator-driven same-node restart and SCOUT-localized standby replacement.
