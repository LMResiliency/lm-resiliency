# SCOUT

SCOUT localizes latent failures such as silent data corruption (SDC), stragglers, collective desynchronization (hangs), and process stalls by comparing equivalent training peers.
It also localizes telemetry-visible permanent GPU and NVLink endpoint failures through direct hardware health sources.
SCOUT also certifies which checkpoints are safe to use for recovery.

For the system design, algorithms, and evaluation rationale, see [SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training](https://arxiv.org/abs/2608.11034).
This guide defines the operational contract of the current implementation.

## Coverage Contract

All rank attribution requires equivalent peers and a strict healthy majority.
Coverage is sampled rather than continuous.

| Failure | Evidence | Localization requirement |
|---|---|---|
| Compute, kernel, or HBM corruption | Replay output or gradient signatures | The failure recurs during sampled replay. |
| Parameter or optimizer corruption | Parameter-state or updated-weight signatures | Peers own corresponding replicated state. |
| Compute straggler | Confirmed replay-time deviation | Equivalent replay remains slow across confirmation rounds. |
| Communication straggler | Collective timing and overlapping process groups | Independent slow groups implicate the same endpoint without a healthy contradiction. |
| Collective desynchronization | Out-of-band progress or collective fingerprints | A strict majority agrees on expected progress or metadata. |
| Process stall | Out-of-band stall timer | Visible progress has stopped. |
| DataLoader stall | Instrumented `next()` latency | The delay exceeds absolute, relative, and persistence thresholds. |
| Checkpoint I/O stall | Instrumented read or write latency | The delay exceeds absolute, relative, and persistence thresholds. |
| Permanent GPU memory or device failure | Uncorrectable ECC, row-remap failure, fatal XID, or device loss | A health source reports the physical device and fatal metric. |
| NVLink endpoint failure | Per-GPU NVLink recovery and CRC error growth | The counter delta exceeds the configured fatal threshold. |
| Imminent thermal shutdown | GPU temperature and hardware shutdown limit | Temperature reaches the configured fatal margin below shutdown. |

Replay and out-of-band progress localization require equivalent peers and consensus.
Direct health telemetry identifies the reporting physical device without cross-rank consensus.
Correctable ECC growth, pending row remaps, nonfatal XIDs, and elevated temperature produce warnings but do not trigger recovery.

SCOUT does not provide an oracle for common-mode failures.
Two disagreeing peers can detect a mismatch but cannot identify the faulty peer without an independent reference.

Pure single-owner FSDP or ZeRO-3 state also lacks an independent live copy.
Use HSDP replication, a shadow copy, parity, trusted recomputation, or another independent reference when owner-shard corruption must be detected.

## Runtime Paths

SCOUT uses three complementary paths:

- In-process replay checks numerical results, optimizer transitions, and timing at optimizer boundaries.
- An out-of-band Gloo process compares progress and metadata while the training process may be blocked.
- A low-frequency health-monitor thread polls direct device or fabric telemetry independently of training progress.

Replay uses the training GPU and cannot run while that process is blocked.
The health monitor can report telemetry-visible failures without replaying the workload.
The out-of-band process exits with its parent.
Worker termination, relaunch, placement, and quarantine remain manager responsibilities.

## Peer Groups and Consensus

Dense replay peers occupy the same TP, PP, CP, and EP coordinates and vary only along a replicated data-parallel dimension.
HSDP uses fixed-shard `dp_replicate` groups, so every peer owns the corresponding shard coordinate.
Pure FSDP falls back to shard peers after materializing globally comparable replay evidence.

SCOUT uses NCCL for GPU replay data and Gloo for small signatures, progress, and out-of-band coordination.
Equivalent peers must use compatible hardware, software, model state, replay shapes, and deterministic behavior.

C3 returns a status, peer-local bitmap, and ordered evidence:

| Status | Meaning |
|---|---|
| `Agree` | Evidence satisfies the comparison rule. |
| `Attributed` | A strict majority identifies one or more divergent peers. |
| `Inconclusive` | Evidence disagrees without enough support for attribution. |

An inconclusive exact result blocks checkpoint certification even when no culprit can be assigned.
Bitmap positions map to global ranks through `ReplayResult.peer_ranks`.

## Replay Surfaces

Dense replay covers four independently scheduled recipe classes:

- embedding;
- rotating hidden layers;
- language-model output; and
- optimizer updates.

Before module execution, exact C3 verifies the broadcast invocation and synchronized RNG state.
Replicated and HSDP paths also compare corresponding materialized parameter state.
Forward-backward replay compares outputs, input gradients, parameter gradients, and adapter-provided updated state without mutating live gradients.

DeepSpeed and Megatron optimizer replay use a bounded rotating slice.
One source peer broadcasts copied pre-update parameters, effective gradients, optimizer state, and update configuration.
Every peer verifies the same recipe, executes the optimizer kernel on isolated state, and compares the updated result.

MoE replay rotates through a qualified execution-regime catalog.
See [MoE execution regimes](moe_execution_regimes.md) for catalog construction, validation, and qualification boundaries.

SCOUT also captures Python-visible AllToAll layouts.
The default replay policy executes bounded balanced and cyclic-permutation matrices with deterministic routing verification.
Custom `AllToAllReplayPolicy` implementations may generate other representative matrices, but no bounded policy covers every possible traffic matrix.

## Stragglers and Communication Localization

A replay-time candidate must exceed configured absolute and relative thresholds for the required confirmation rounds.
Detailed replay separates visible collective time from residual compute time.
FSDP parameter materialization is reported independently as `fsdp_parameter_all_gather` communication evidence.

SCOUT correlates complete observations from independently slow process groups.
When their membership intersects at one rank and no healthy group contradicts the diagnosis, SCOUT maps that rank to its host and reports the host as the manager replacement scope.

Framework-internal or compiled collectives that bypass public `torch.distributed` boundaries require adapter-visible timing.
NIC, HCA, cable, link, port, and switch attribution requires deployment resource maps or an external fabric diagnostic source.

## Out-of-Band Detection

The training process publishes progress around repeated layers, public Python collectives, sampled DataLoader calls, and instrumented checkpoint I/O.
The observer starts consensus only after progress stops for the configured threshold.
Normal progress does not produce periodic out-of-band network traffic.

Public collective fingerprints include operation type, process-group membership, fixed tensor metadata, reduction operation or root, and sequence position.
Rank-dependent values such as valid expert AllToAll splits are excluded from equality checks.

DataLoader and checkpoint-I/O localization require explicit instrumentation.
SCOUT wraps certified durable checkpoint callbacks and framework fallback loads automatically.
Use `instrument_dataloader(...)` and `checkpoint_io(...)` for other boundaries.

An explicit `hang_master_addr` or `hang_master_port` selects TCP rendezvous.
`hang_state_dir` remains available for status files and is used for file rendezvous only when no TCP endpoint is configured.

## Checkpoint Certification and Recovery

SCOUT runs before checkpoint capture at the same optimizer boundary.

| SCOUT result | Checkpoint action |
|---|---|
| Healthy | Capture and certify according to the recipe catalog. |
| Straggler without SDC | Capture and report the performance fault. |
| SDC or inconclusive exact evidence | Reject the current state and preserve the prior trusted checkpoint. |

The implementation uses three trust states:

| State | Recovery use |
|---|---|
| Latest GEMINI | Accessible straggler or a complete clean emergency replay. |
| `CANDIDATE` | Persisted dynamic-catalog boundary that is not yet trusted for conservative recovery. |
| `RECOVERY_VERIFIED` | Dense accepted boundary or dynamic candidate validated by the following complete catalog cycle. |

An accepted dense check certifies the current checkpoint immediately.
For a dynamic catalog, the first complete accepted cycle creates a candidate and the following complete accepted cycle promotes it to recovery-verified.
Any comparison group reporting SDC rejects the candidate job-wide.

`DurableCheckpointConfig` connects this trust state to framework-owned checkpoint bytes.
The adapter callbacks have the following execution contract:

| Callback | Called on | Collectives |
|---|---|---|
| `save_candidate` | Every checkpoint rank | Allowed |
| `load_checkpoint` | Every checkpoint rank | Allowed |
| `commit_candidate` | Manifest writer only | Not allowed |
| `quarantine_candidate` | Manifest writer only | Not allowed |

The manifest directory must have durability equivalent to the framework checkpoint store.
Checkpoint bytes remain the framework's responsibility and must include model, optimizer, scheduler, input position, RNG state, and training step.

Recovery selection follows this policy:

| Failure condition | Selected checkpoint |
|---|---|
| Detected SDC | `RECOVERY_VERIFIED` |
| Required rank or machine inaccessible | `RECOVERY_VERIFIED` |
| Accessible straggler | Latest complete GEMINI checkpoint |
| Hang or uncertain failure with all ranks accessible | Latest GEMINI only after a complete clean catalog replay |
| Emergency replay is incomplete, unavailable, or detects SDC | `RECOVERY_VERIFIED` |

## Reports and Manager Handoff

In-process checks return `ReplayResult`.
Important fields include:

| Field | Meaning |
|---|---|
| `sdc_bitmap` | Peers with numerical disagreement |
| `sdc_sources` | Independent evidence surfaces |
| `c3_results` | Per-surface status, bitmap, and evidence |
| `checked_recipe_ids` | Dense recipes executed by the check |
| `straggler_bitmap` | Confirmed timing outliers |
| `collective_timings` | Per-collective timing with process-group membership |
| `cross_pg_result` | Correlated endpoint and host replacement scope |

Out-of-band failures use JSON-ready `SCOUTFaultReport` values.
Checkpoint selection uses `RecoveryDecision`, including recovery mode, source, checkpoint step or durable ID, availability, and reason.

`OrchestrationHooks` delivers both contracts to an external manager.
SCOUT and the framework recommend a checkpoint; the manager owns restart, and the relaunched framework performs the rank-consistent load.

## Configuration

`ReplayHarnessConfig` controls recipe intervals, deterministic replay, parameter comparison, AllToAll policy, straggler thresholds, temporal windows, and out-of-band thresholds.
Set `check_interval=0` for manual replay.
Explicit embedding, hidden, output, and optimizer intervals override the base interval, and zero disables that recipe.

See the [API guide](api.md#configure-protection) for configuration examples and the [manager integration API](api.md#manager-integration) for orchestration callbacks.

## Hardware Telemetry

SCOUT complements workload-based inference with direct telemetry for permanent or imminent hardware failures.
`HardwareHealthMonitor` polls caller-supplied `HealthSource` implementations in a low-frequency daemon thread and emits each fatal device-and-metric event once through `on_event`.

The built-in `NvmlSource` reports:

- uncorrectable and correctable ECC state;
- pending or failed row remapping;
- aggregate per-GPU NVLink recovery and CRC errors;
- temperature and hardware shutdown limit; and
- device loss when all NVML queries fail.

The monitor treats uncorrectable ECC, row-remap exhaustion, device loss, a configured fatal XID, severe NVLink error growth, and temperature near shutdown as fatal.
Warnings record precursor conditions without requesting recovery.

```python
from lm_resiliency.manager_api import (
    HardwareHealthMonitor,
    HealthConfig,
    NvmlSource,
)

health_monitor = HardwareHealthMonitor(
    HealthConfig(),
    [NvmlSource(device_index=physical_gpu_index)],
    on_event=manager_health_callback,
)
health_monitor.start()
```

The caller resolves `physical_gpu_index` and supplies `manager_health_callback`.
`device_index` is the physical NVML index, not necessarily the CUDA ordinal after `CUDA_VISIBLE_DEVICES` remapping.
The built-in NVLink reading localizes a failing GPU endpoint, not an individual link, cable, or switch.
Complete XID, InfiniBand, PCIe, NIC, HCA, or fabric coverage requires DCGM, system-log, or deployment-specific `HealthSource` implementations.

Fatal health events are direct localization inputs to the manager.
SCOUT does not terminate workers or quarantine hardware.

## Guarantee Boundary

- SCOUT covers telemetry-visible permanent failures directly and active permanent or intermittent computational failures that recur under a qualified replay.
- Health localization is limited to signals exposed by configured sources and their device or endpoint granularity.
- One-shot transients, common-mode failures, and finite-signature collisions remain outside the guarantee.
- Replay samples computation and is not continuous duplication.
- Pure single-owner shards require an independent oracle.
- AllToAll replay covers policy-selected representative matrices.
- Automatic communication attribution requires complete observations from independently slow groups and no healthy contradiction.
- Framework-internal collectives require adapter-visible timing before they can participate in communication attribution.
- Process supervision, relaunch, replacement, quarantine, and fabric management remain external.

See [validation](validation.md) for measured coverage and [compatibility](compatibility.md) for supported software versions.
