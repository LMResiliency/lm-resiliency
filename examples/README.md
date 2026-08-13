# Examples

## Quick Start

Run [quickstart.py](quickstart.py) with plain Python to train a tiny causal LM on CPU.
The example executes the complete forward, backward, optimizer, and GEMINI checkpoint lifecycle without requiring a GPU.

```bash
python examples/quickstart.py \
  --checkpoint-dir /tmp/lm-resiliency-quickstart/checkpoints
```

Run it again with a larger step target to resume from the saved GEMINI checkpoint:

```bash
python examples/quickstart.py \
  --steps 6 \
  --checkpoint-dir /tmp/lm-resiliency-quickstart/checkpoints
```

The single-process example validates training-loop integration and recovery.
Use the distributed examples below to exercise SCOUT replay and multi-rank localization.

## Fault Injection Evaluation

Run [fault_injection.py](fault_injection.py) with plain Python to inject one
transient output corruption, record its ground truth, and score a neutral
localization result:

```bash
python examples/fault_injection.py
```

The example uses the public framework-aware evaluation API without enabling
SCOUT or GEMINI.
See the [fault injection guide](../docs/fault_injection.md) for campaign
manifests, target selection, supported faults, and safety boundaries.

## Production Loops

The production-loop examples train tiny causal language models with deterministic synthetic tokens while preserving each framework's real training lifecycle.
They support one or two hosts through standard `torchrun` arguments.

| Framework | Example | Framework-owned path |
|---|---|---|
| PyTorch | [pytorch.py](production_loops/pytorch.py) | DDP forward, backward, and `AdamW.step()` |
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

Each example runs ten steps by default.
Set `--steps` to change the duration.
Add `--inject-fault` to introduce one transient hidden-layer replay fault at step 4 on the last global rank.
The fault campaign requires exact SCOUT localization, exclusion of the fault-step checkpoint, a recovery-verified manager decision, and clean post-fault certification.
