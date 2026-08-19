# Closed-Loop Recovery for Torchrun Training

Distributed training can fail without producing a clean process crash. A rank may stop making progress, run slowly, desynchronize from its peers, or continue with corrupted state. Restarting workers is only part of the solution. The system must first decide what failed, which training state is safe, and whether the current machines should restart or a standby should take over.

Torchrun already provides the worker lifecycle. LM Resiliency adds the evidence and recovery decisions around that lifecycle:

- **SCOUT** detects and localizes failures.
- **GEMINI** maintains recent in-memory checkpoints and identifies trusted recovery state.
- The **LM Resiliency torchrun integration** turns that decision into a coordinated restart or standby replacement.

Together they create closed-loop recovery: protect the running job, detect a failure, select safe state, recover the worker group, and continue training.

## Architecture at a Glance

The design has three layers:

```text
+-------------------------------------------------------------+
|                           torchrun                          |
|  +-------------------------------------------------------+  |
|  |                active + standby agents                |  |
|  |               launch and restart workers              |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
               ^                                |
               |                                |
    committed recovery plan         launch + monitor workers
               |                                |
               |                                v
+-------------------------------------------------------------+
|                        LM Resiliency                        |
|  +-------------------------------------------------------+  |
|  |        rendezvous backend + framework adapters        |  |
|  |              GEMINI in-memory checkpoints             |  |
|  |      SCOUT detection, localization, certification     |  |
|  |                committed recovery plan                |  |
|  +-------------------------------------------------------+  |
+-------------------------------------------------------------+
               ^                                |
               |                                |
   state + failure evidence             bootstrap + hooks
               |                                |
               |                                v
+-------------------------------------------------------------+
|         PyTorch / DeepSpeed / Megatron / TorchTitan         |
+-------------------------------------------------------------+
```

| Layer | Responsibility |
|---|---|
| torchrun | Run active and standby agents; launch, monitor, stop, and restart workers |
| LM Resiliency | Protect training state, detect failures, select recovery state, and coordinate restart or replacement |
| Training frameworks | Execute the model and optimizer loop through their normal framework APIs |

The scheduler or infrastructure layer sits outside the diagram. It allocates the fleet and repairs or retires physical hardware.

## The Recovery Cycle

The interaction is easiest to understand by following one failure.

**1. torchrun starts active and standby agents.**

Suppose an allocation contains six nodes and the job needs four:

```text
active:   node-1, node-2, node-3, node-4
standby:  node-5, node-6
```

Every allocated node runs a torchrun agent. Four nodes launch the training job; the other two remain parked until a replacement is needed.

**2. LM Resiliency attaches to the training framework.**

The active agents start the user module normally. LM Resiliency detects whether the job uses PyTorch, DeepSpeed, Megatron Core, or TorchTitan and attaches at the framework's initialization boundary.

**3. The training framework runs its normal training loop.**

The framework continues to execute forward, backward, optimizer, and scheduler steps through its existing APIs. LM Resiliency observes the framework-owned step boundaries without taking control of the loop.

**4. LM Resiliency protects the healthy training loop.**

GEMINI captures frequent in-memory checkpoints while SCOUT observes replay, progress, and health evidence. Their combined evidence distinguishes recent state from recent state that is safe to recover.

**5. LM Resiliency commits one recovery decision.**

Assume SCOUT localizes a persistent fault to `node-4`. LM Resiliency combines that result with GEMINI's trusted state. torchrun contributes worker-lifecycle and membership signals, while LM Resiliency combines the available evidence and publishes a committed recovery plan for torchrun to execute:

```text
faulty node:          node-4
recovery checkpoint: step 1240
replacement node:    node-5
```

The decision is committed before torchrun changes the worker group. It names the recovery checkpoint, the machines that may continue, and the standby that may join.

**6. torchrun restarts or replaces the workers.**

Torchrun stops the current worker group and launches the approved successor:

```text
active:       node-1, node-2, node-3, node-5
quarantined:  node-4
standby:      node-6
```

A process-only failure can restart the existing machines without consuming a standby. A persistent machine failure replaces only the affected machine.

**7. LM Resiliency completes closed-loop recovery.**

The framework adapter restores the selected model, optimizer, scheduler, RNG, and framework-owned progress state. After all workers agree on that state, the framework returns to its normal training loop and GEMINI and SCOUT resume protection. A later failure enters the same recovery cycle again.

## Running the Integrated Path

Without LM Resiliency, a single-node eight-GPU job can use torchrun's standard `c10d` rendezvous backend:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=8 \
  --max-restarts=4 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${RDZV_HOST}:29400" \
  --rdzv-id="${RUN_ID}" \
  --module your_training.module
```

This command can restart workers, but it does not add SCOUT detection, GEMINI checkpoint recovery, or standby replacement.

To enable LM Resiliency, select its rendezvous backend, allocate a standby, and supply a worker policy:

```bash
torchrun \
  --nnodes=1:2 \
  --nproc-per-node=8 \
  --max-restarts=4 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint="${RDZV_HOST}:29400" \
  --rdzv-id="${RUN_ID}" \
  --rdzv-conf="store_type=tcp,\
lm_resiliency_restart_context_path=${RESTART_CONTEXT},\
lm_resiliency_worker_config=${WORKER_CONFIG}" \
  --module your_training.module
```

In this example, one eight-GPU node trains while a second eight-GPU node waits as standby. Both hosts run the same command. The training loop in `your_training.module` does not need any changes.

See the [torchrun examples](../examples/torchrun/README.md) for runnable bootstrap and recovery campaigns. See the [API guide](api.md#enable-through-torchrun) for configuration and extension contracts.

## Validation and Boundaries

The integration has been exercised with native PyTorch, DeepSpeed, Megatron Core, and TorchTitan. Multi-generation validation covered same-node restarts, SCOUT-localized standby replacement, and exact checkpoint restoration.

The current design intentionally targets fixed-size recovery:

- standby machines are allocated before failure;
- active world size remains stable during recovery;
- one shared rendezvous store coordinates the agents;
- built-in adapters restore GEMINI-managed state; and
- replacing a worker does not repair hardware or allocate a new host.

Within those boundaries, LM Resiliency gives torchrun enough information to do more than relaunch processes: it recovers the training job from state that has already been checked for safety.
