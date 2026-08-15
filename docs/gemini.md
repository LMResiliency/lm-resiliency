# GEMINI

GEMINI captures training state into host memory, replicates it to a peer, and optionally persists it to node-local storage.
It provides a fast recovery tier for PyTorch, TorchTitan, Megatron Core, and DeepSpeed while leaving durable global checkpointing to the training framework.

For the system design and evaluation, see [GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints](https://doi.org/10.1145/3600006.3613145).
This guide defines the operational contract of the current implementation.

## Architecture

Checkpoint work is pipelined after each capture:

1. GEMINI copies the current model, optimizer, and caller-owned state from GPU to pinned CPU memory.
2. A background completion worker waits for the host copy and immediately starts replication to a paired rank.
3. Node-local serialization runs independently at the configured flush cadence.

The asynchronous GPU-to-CPU copy records a CUDA event.
`maybe_wait()` waits for the completion worker and therefore guarantees that peer replication has been launched.
This boundary prevents a checkpoint from mixing state from two optimizer steps.

Replication can span several training steps, but it must complete before the source buffer is reused at the next capture.
Transfers are divided into fixed-size chunks to bound head-of-line interference with training communication.

## Buffer Layout

Replicated mode allocates four host slots lazily on the first capture:

| Slot | Role |
|---|---|
| `own_current` | Receives the current GPU-to-CPU copy and becomes the replication source |
| `own_previous` | Retains the prior completed local recovery copy |
| `peer_current` | Receives the peer's current replica |
| `peer_previous` | Retains the prior completed peer recovery copy |

Each slot owns its tensor buffers, structural metadata, step, and non-tensor state.
Non-tensor state includes scheduler, sampler, RNG, and training-position data supplied by the framework adapter or `extra_state_fn`.
The metadata travels with peer replication so recovery reconstructs the complete captured state.

HSDP can skip explicit peer replication because the replica dimension already holds corresponding shards.
In that mode GEMINI uses two own slots.

## Peer Replication

`replication_jump` pairs ranks separated by a fixed world-rank distance.
The default selects one visible node width so paired ranks normally reside on different hosts.

For a world size of 16 and `replication_jump=8`, rank `0` pairs with rank `8`, rank `1` with rank `9`, and so on.
Validate this assumption when rank placement is not contiguous by node.

The built-in replication path uses a dedicated Gloo process group.
It is validated for correctness over TCP and is not a line-rate RDMA implementation.
Production deployments can use manager-driven Torch Distributed or NIXL transfer APIs for replacement workflows, but automatic checkpoint replication currently uses Gloo.
Manager-driven transfers bind a key to endpoint and tensor metadata, verify per-chunk checksums, and use bounded waits.
The torch-distributed backend communicates on a dedicated Gloo group and requires both endpoints; fallback from one-sided NIXL is therefore explicit rather than automatic.

`replication_chunk_size` limits the largest in-flight transfer unit.
Choose it from measured training communication slack rather than assuming a fixed network rate:

```python
from lm_resiliency import estimate_chunk_size

chunk_size = estimate_chunk_size(
    nic_bandwidth_gbps=400,
    layer_compute_ms=9.4,
    ag_time_ms=1.0,
    max_ag_delay_fraction=0.05,
)
```

The default is 16 MiB.
The estimator returns a bound derived from the supplied bandwidth and prefetch timing; it is not an automatic network profiler.

## Configuration

```python
from lm_resiliency import InMemoryCkptConfig

config = InMemoryCkptConfig(
    enable=True,
    interval=10,
    replication_jump=-1,
    replication_chunk_size=16 * 1024 * 1024,
    disk_flush_interval=100,
    disk_folder="./checkpoints",
    verify_integrity=False,
    skip_replication_if_hsdp=True,
    pin_memory=True,
)
```

| Field | Meaning |
|---|---|
| `interval` | Capture cadence in optimizer steps when GEMINI runs independently |
| `replication_jump` | Peer rank spacing; `-1` uses the visible GPU count |
| `replication_chunk_size` | Bytes per replication send |
| `disk_flush_interval` | Node-local flush cadence; `0` disables periodic flush |
| `disk_folder` | Node-local checkpoint directory |
| `verify_integrity` | Store and verify CRC-32 for serialized shards |
| `skip_replication_if_hsdp` | Use natural HSDP replicas instead of explicit peer transfer |
| `pin_memory` | Allocate page-locked host buffers for asynchronous copies |

The unified `enable_resiliency(..., interval=N)` call overrides the component interval.

## Checkpoint Validation

When GEMINI and SCOUT are enabled together, `interval=N` is the GEMINI capture frequency and the default interval for each SCOUT recipe class.
SCOUT runs first.
A healthy result, or a straggler result without SDC, permits GEMINI to capture that step.
An SDC result prevents the capture.
Dense recipe classes can override the default interval independently, but a checkpoint boundary still requires an accepted SCOUT result at that step.

For `N=10`:

| Step | Action |
|---:|---|
| 1-9 | Training |
| 10, Healthy or Straggler without SDC | Capture step 10 and replicate it after D2H completion |
| 10, SDC | Skip step 10 capture and require verified recovery |

For dense replay, the shape catalog has `K=1`.
The checkpoint captured after an accepted check becomes `RECOVERY_VERIFIED` immediately and does not enter a second validation cycle.
Dynamic catalogs, including a MoE catalog that compresses to one recipe, use the candidate protocol below.

This policy assumes unsafe state remains observable at the validation boundary through persistent corruption or recurring failure.
One-shot transient corruption is outside the SCOUT guarantee.

## Durable MoE Certification

Normal MoE training checks one recipe every `N` steps rather than running all
`K` recipes at one boundary. After the first complete accepted cycle:

1. GEMINI persists the latest CPU checkpoint and received peer replica as
   `CANDIDATE`.
2. The framework writes the corresponding durable candidate.
3. The candidate remains excluded from conservative SDC recovery.

After the following complete accepted cycle, the prior candidate becomes
`RECOVERY_VERIFIED`, and the current boundary becomes the next candidate.
Verification therefore covers two consecutive `K`-recipe cycles.

An SDC result rejects the current candidate and preserves the preceding
`RECOVERY_VERIFIED` generation.
See [SCOUT checkpoint certification](scout.md#checkpoint-certification-and-recovery) for checkpoint states, global trust decisions, and failure recovery.

SCOUT comparison groups remain topology-specific, but checkpoint acceptance is
job-wide. Before any save, promotion, or rejection, all checkpoint ranks agree
whether the scheduled cycle completed and whether any comparison group found
SDC. An SDC in one group therefore rejects the candidate on every rank.

## Persistence and Recovery

GEMINI serializes only completed slots.
Each shard is written to a same-directory temporary file, flushed, and atomically renamed before it becomes visible to recovery.
Restart mirrors use the same publication rule.
Temporary files left by a terminated writer are ignored and removed before the next write to that shard, while latest-mode recovery walks older generations until it finds the newest shard set that every rank can load and validate.
At a dense accepted boundary, it persists the latest local CPU checkpoint and received peer replica as recovery-verified.
At a dynamic recipe-cycle boundary, it persists them as the new candidate and records candidate and recovery-verified steps separately.
Each rank owns its status sidecar; several workers on one node never update the same mutable trust record.
A persisted peer shard carries the corresponding peer status so recovery can reconstruct the verified generation after node loss.
If newer status cannot be established, recovery selects the common verified step conservatively.
With `verify_integrity=True`, it stores a CRC-32 for each shard and treats a checksum failure or missing checksum metadata as an unavailable shard.
Checksums detect stored-byte corruption; they cannot prove that the source GPU state was numerically correct.
Enabling integrity verification is therefore a fail-closed configuration change:
node-local checkpoints written earlier with `verify_integrity=False` are not eligible for recovery until a new checksummed generation is persisted.

Node-local files use checkpoint format version 2. Tensors are loaded through PyTorch's `weights_only=True` path, while reconstruction metadata is represented by a schema-constrained JSON document. The metadata schema supports the built-in containers and scalar types used by framework state, NumPy RNG values, and dense CPU tensors such as PyTorch RNG state. GEMINI validates the payload fields, metadata types, tensor count, shapes, dtypes, reconstruction paths, and optional CRC values before recovery can apply state. Unsupported caller-owned metadata fails the checkpoint write instead of falling back to unrestricted pickle deserialization.

The original `0.1.0` node-local format used unrestricted pickle metadata and is intentionally not loadable by newer versions. Complete an in-progress `0.1.0` recovery before upgrading, or use the framework-owned durable checkpoint as the upgrade boundary. Node-local GEMINI files remain a fast restart tier rather than a cross-version interchange format.

Safe deserialization prevents a replaced file from invoking arbitrary pickle globals, but it does not establish who produced the tensor values. Treat the checkpoint directory as training-state storage: restrict filesystem ownership and write permissions to the training job, avoid following untrusted symlinks or copying files from untrusted sources, and use framework-owned durable checkpoints when stronger provenance is required. CRC-32 detects accidental byte corruption only; an attacker able to replace a shard can also replace its checksum.

`SIGTERM` and `SIGINT` handlers synchronously flush the newest recoverable state during the scheduler grace period and then chain to the previous handler.
`SIGKILL`, power loss, and complete node loss cannot be handled locally.
Peer replicas and framework durable checkpoints cover those cases.

For an accessible straggler, recovery selects the newest rank-consistent GEMINI step available from:

1. complete local GEMINI state;
2. a complete peer replica;
3. node-local serialized state; or
4. the caller's framework fallback.

Every rank participates in recovery-step selection.
A newer step present on only part of the job is not selected.

Detected SDC or an inaccessible machine selects the persisted `RECOVERY_VERIFIED` step.
For a hang or uncertain failure with all ranks accessible, SCOUT replays the full enabled catalog.
A complete clean sweep selects the latest GEMINI checkpoint.
Detected SDC, missing evidence, or an incomplete sweep selects `RECOVERY_VERIFIED`.

## Public Integration

Use the unified entry point for normal training:

```python
from lm_resiliency import enable_resiliency

resiliency = enable_resiliency(
    model,
    optimizer,
    interval=10,
    load_fallback=load_durable_checkpoint,
)
```

The lower-level `InMemoryCheckpointManager` is available for research and custom adapter development, but is not part of the stable API:

```python
from lm_resiliency import InMemoryCkptConfig
from lm_resiliency.experimental import InMemoryCheckpointManager

manager = InMemoryCheckpointManager(InMemoryCkptConfig(interval=10))

for step in range(1, max_steps + 1):
    loss = train_step()
    loss.backward()

    # Complete the prior capture before parameters are modified again.
    manager.maybe_wait()
    optimizer.step()
    optimizer.zero_grad()

    if step % manager.config.interval == 0:
        manager.save(state_dict(), step)

manager.close()
```

On recovery, call collective `find_latest()` before `load()`.
Framework integrations call the manager at safe optimizer boundaries and include their complete recovery state.

## Measured Overhead

The following measurements used eight A100-SXM4-80GB GPUs and approximately 1 billion parameters per rank after sharding:

| Workload | Checkpoint size | Blocking per capture | Amortized overhead |
|---|---:|---:|---:|
| TorchTitan/Megatron-style FSDP, 8.855 s step, interval 5 | 12.05 GB/rank | 7.7 ms | 1.5 ms/step, or 0.017% |
| DeepSpeed ZeRO-2, interval 5 | 14.00 GB/rank | 2.36 ms | 0.015% of the measured step |

These values measure the synchronization-visible portion of asynchronous capture.
They do not infer overhead from small differences between independent end-to-end step-time runs.
Hardware, state layout, capture cadence, and available copy or network overlap can change the result.
The reproducible [healthy-path benchmark](../benchmarks/README.md) records current-package baseline, GEMINI, SCOUT, and combined measurements and applies explicit scheduled-run thresholds.
The table above remains historical evidence for its stated A100 workload and is not a baseline for different benchmark shapes or hardware.

## Boundaries

- Built-in peer replication uses Gloo over TCP; RDMA and line-rate transport performance are not claimed.
- Node-local storage is a fast restart tier, not a replacement for framework-owned durable checkpoints.
- `replication_jump` assumes a compatible rank placement and must be validated for the deployment.
- Integrity checks detect corruption after capture, not corruption already present in source state.
- Safe weights-only loading prevents arbitrary pickle execution but does not authenticate tensor values or their source.
- Node-local checkpoint format version 2 does not load the unrestricted-pickle format written by `0.1.0`.
- Signal-triggered flush requires a catchable signal and sufficient termination grace time.
- Manager policy, relaunch, placement, and physical replacement remain external.
