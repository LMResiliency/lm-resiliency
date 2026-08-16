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

The production-loop examples train tiny causal language models with deterministic synthetic tokens while preserving each framework's real training lifecycle.
They support one or two hosts through standard `torchrun` arguments.

| Framework | Example | Framework-owned path |
|---|---|---|
| PyTorch | [pytorch.py](production_loops/pytorch.py) | DDP forward, backward, and `AdamW.step()` |
| PyTorch with torchrun replacement | [torchrun.py](production_loops/torchrun.py) | DDP, SCOUT localization, GEMINI recovery, standby replacement, and stable logical ranks |
| TorchTitan | [torchtitan.py](production_loops/torchtitan.py) | `Trainer.train()` |
| Megatron Core | [megatron.py](production_loops/megatron.py) | `training.train()` and `train_step()` |
| DeepSpeed | [deepspeed.py](production_loops/deepspeed.py) | `DeepSpeedEngine.backward()` and `DeepSpeedEngine.step()` |

Run one example on a single eight-GPU host:

```bash
torchrun --standalone --nproc-per-node=8 --module \
  examples.production_loops.torchtitan \
  --artifact-dir /tmp/torchtitan-production-loop
```

Run it on two eight-GPU hosts:

```bash
torchrun --nnodes=2 --nproc-per-node=8 --module \
  --node-rank="$NODE_RANK" \
  --master-addr="$MASTER_ADDR" --master-port=29800 \
  examples.production_loops.torchtitan \
  --artifact-dir /tmp/torchtitan-production-loop
```

Replace the module and artifact directory for PyTorch, Megatron Core, or DeepSpeed.
Use a different rendezvous port for concurrent jobs.

Run the complete torchrun replacement campaign on one host with six GPUs:

```bash
python -m examples.production_loops.torchrun orchestrate \
  --workspace /tmp/lm-resiliency-torchrun \
  --gpus 0,1,2,3,4,5
```

Run the same campaign across two hosts with three GPUs per host and a shared
workspace:

```bash
python -m examples.production_loops.torchrun orchestrate \
  --workspace /shared/lm-resiliency-torchrun \
  --gpus 0,1,2 \
  --remote-host "$REMOTE_HOST" \
  --remote-gpus 0,1,2 \
  --remote-python "$REMOTE_PYTHON" \
  --remote-source-dir /tmp/lm-resiliency-torchrun-source \
  --rdzv-host "$LOCAL_HOST"
```

The orchestrator runs an uninterrupted baseline, starts four active agents and
two parked standbys, injects SCOUT replay faults at optimizer steps 3 and 6,
publishes manager-owned recovery plans, and requires GEMINI to restore verified
steps 2 and 5 exactly before training resumes. It also verifies slot
inheritance, clean agent shutdown, exact losses and RNG state, and strict final
tensor error bounds against the baseline.

Each example runs ten steps by default.
Set `--steps` to change the duration.
Add `--inject-fault` to introduce one transient hidden-layer replay fault at step 4 on the last global rank.
The fault campaign requires exact SCOUT localization, exclusion of the fault-step checkpoint, a recovery-verified manager decision, and clean post-fault certification.
