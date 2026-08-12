# Examples

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
