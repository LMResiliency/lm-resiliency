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
| Torchrun pressure validation | [pressure.py](torchrun_resiliency/pressure.py) | Repeated job restart, SCOUT localization, GEMINI recovery, standby replacement, and stable logical ranks |
| TorchTitan | [torchtitan.py](production_loops/torchtitan.py) | `Trainer.train()` |
| Megatron Core | [megatron.py](production_loops/megatron.py) | `training.train()` and `train_step()` |
| DeepSpeed | [deepspeed.py](production_loops/deepspeed.py) | `DeepSpeedEngine.backward()` and `DeepSpeedEngine.step()` |

Run one example on a single eight-GPU host from the repository root:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=8 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint=/tmp/lm-resiliency-torchtitan-rdzv \
  --rdzv-id=torchtitan-production \
  --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-torchtitan-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/production_loops/policies/resiliency.toml" \
  --module \
  examples.production_loops.torchtitan \
  --artifact-dir /tmp/torchtitan-production-loop
```

Replace the module, rendezvous paths, and artifact directory for PyTorch,
Megatron Core, or DeepSpeed. The worker infers the framework from imports. The
checked-in worker policy uses
`replication_jump=4` for this eight-rank topology. Other world layouts must use
a policy with a valid deployment-specific GEMINI pairing.

Run the pressure campaign across two eight-GPU hosts with every GPU modeled as
an independent node. The fault-campaign directory is the single bundle for the
generated manifest, restart-stable state, artifacts, logs, checkpoints, and
summary:

```bash
python -m examples.torchrun_resiliency.pressure orchestrate \
  --fault-campaign-dir /shared/lm-resiliency-torchrun-pressure \
  --gpus 0,1,2,3,4,5,6,7 \
  --remote-host "$REMOTE_HOST" \
  --remote-gpus 0,1,2,3,4,5,6,7 \
  --remote-python "$REMOTE_PYTHON" \
  --remote-source-dir /tmp/lm-resiliency-torchrun-source \
  --rdzv-host "$LOCAL_HOST"
```

The command runs an uninterrupted eight-rank baseline, starts eight active
one-GPU agents and eight parked one-GPU standbys, and executes 24 manager
generations. Sixteen process-stall incidents restart the same assigned nodes;
eight replay-only SDC incidents target distinct logical ranks, quarantine the
localized GPU-node, and consume one standby. Every recovery must restore the
manager-selected GEMINI checkpoint bitwise before training resumes.

Normal deployments derive node identity from `/etc/machine-id` and commit the
first `min_nodes` registrations as the initial group. This campaign launches
multiple agents per physical host to model every GPU as a node, so the harness
supplies distinct synthetic machine-ID files through
`LM_RESILIENCY_MACHINE_ID_PATH`. Synthetic identities are test-only.

Each framework example runs ten steps by default; set `--steps` to change the
duration. The dedicated torchrun replacement campaign above exercises exact
SCOUT localization, fault-step checkpoint exclusion, manager recovery
selection, and clean post-replacement completion. The broader declarative fault
campaign is documented under [Fault Injection Evaluation](#fault-injection-evaluation).
