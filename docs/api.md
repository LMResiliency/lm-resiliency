# API Guide

`lm-resiliency` exposes one entry point for enabling GEMINI checkpointing and SCOUT failure localization.
This guide covers the public integration contract; component design and validation details are linked at the end.

## Public API Stability

The package separates supported interfaces from low-level research components:

| Namespace | Contract |
|---|---|
| `lm_resiliency` | Stable training integration API |
| `lm_resiliency.manager_api` | Stable platform-neutral manager API |
| `lm_resiliency.integrations.<framework>` | Stable explicit framework entry points |
| `lm_resiliency.experimental` | Unstable low-level components |

The stable package-root exports are:

| Area | Exports |
|---|---|
| Activation and lifecycle | `enable_resiliency`, `FrameworkName`, `ResiliencySession`, `ResiliencyHandle` |
| Configuration | `InMemoryCkptConfig`, `ReplayHarnessConfig` |
| Dynamic and MoE replay | `ReplayWorkload`, `GroupedExpertMaterializer`, `LeadingDimensionMaterializer` |
| AllToAll replay | `AllToAllReplayPolicy`, `AllToAllCapture`, `AllToAllTrafficMatrix`, `BalancedAndPermutationPolicy` |
| Durable certification | `DurableCheckpointConfig`, `CallbackDurableCheckpointAdapter` |
| Fault reporting | `SCOUTFaultReport`, `SCOUTFaultCallback`, `OrchestrationHooks`, `replay_fault_reports` |
| Recovery handoff | `RecoveryDecision`, `RecoveryDecisionCallback` |
| Checkpoint tuning | `estimate_chunk_size` |
| Fault injection evaluation | `enable_fault_injection`, `FaultCampaign`, `FaultIncident`, `IncidentTrigger`, `IncidentLifetime`, `FaultSpec`, `FaultTarget`, `FailureType`, `FaultSurface`, `CorruptionOperation`, `CallbackFaultExecutor`, `FaultInjectionSession`, `LocalizationResult`, `CampaignReport` |

The stable manager API exports are:

| Area | Exports |
|---|---|
| Coordination | `OrchestrationHooks`, `RestartDestinationResolver`, `RecoveryDecision`, `RecoveryDecisionCallback` |
| Fault reports | `SCOUTFaultReport`, `SCOUTFaultCallback`, `replay_fault_reports`, `dispatch_replay_faults` |
| Checkpoint transfer | `CheckpointTransfer`, `TransferMetadataStore`, `make_transfer` |
| Hardware health | `HealthConfig`, `HardwareHealthMonitor`, `HealthEvent`, `HealthReading`, `HealthSeverity`, `HealthSource`, `NvmlSource` |
| Configuration drift | `local_fingerprint`, `find_drift`, `format_drift` |

Other module paths are internal unless this guide identifies them as supported.
Objects under `lm_resiliency.experimental` may change within the `0.x` release series.

## Enable Resiliency

Initialize `torch.distributed`, create the framework training objects, and call `enable_resiliency` once on every rank:

```python
from lm_resiliency import enable_resiliency

resiliency = enable_resiliency(
    model,
    optimizer,
    interval=10,
)

train()
```

Call `enable_resiliency` after framework initialization and before the training loop.
The returned handle owns hooks, background workers, and process groups created by the integration.
The integration releases these resources automatically at normal process exit.

`interval` is the shared optimizer-step cadence.
When GEMINI and SCOUT are both enabled, checkpoint capture and validation use the coordinated cadence described in [GEMINI checkpointing](gemini.md#checkpoint-validation).

## Framework Support

The package infers the framework from the first training object:

| Framework | First argument | SCOUT parallelism |
|---|---|---|
| PyTorch | `torch.nn.Module` | DDP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| TorchTitan | Initialized `Trainer` | DP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| Megatron Core | Model chunks in a list or tuple | DP, TP, SP, CP, PP, virtual PP, EP, expert TP |
| DeepSpeed | DeepSpeed engine | DP, ZeRO 1-3, TP, PP, Ulysses SP, EP, expert TP |

SCOUT derives equivalent-peer groups from each framework's topology.
Dense replay uses data-parallel or FSDP peers while preserving the model-parallel coordinates required by the replayed layer.
Expert replay uses expert-data-parallel peers while preserving EP and expert-TP coordinates.
Exact localization requires at least three equivalent peers.

Use the package-root API for every framework.
Pass `framework="pytorch"`, `"torchtitan"`, `"megatron"`, or `"deepspeed"` only when a custom wrapper makes automatic dispatch ambiguous.

TorchTitan accepts its initialized trainer directly:

```python
resiliency = enable_resiliency(trainer, interval=10)
trainer.train()
```

The adapter checkpoints and restores trainer, scheduler, dataloader, optimizer, and model state.
It also coordinates TorchTitan's durable loader with an earlier GEMINI recovery.

Megatron accepts the model chunks, optimizer, and scheduler created before its production `train()` call:

```python
from megatron.training import get_args

from lm_resiliency import enable_resiliency


def attach_resiliency(model_chunks, optimizer, scheduler):
    resiliency = enable_resiliency(
        model_chunks,
        optimizer,
        opt_param_scheduler=scheduler,
        interval=10,
    )
    get_args().iteration = resiliency.step_count
    return resiliency
```

Call this helper immediately after Megatron creates the three training objects, then continue into its existing `train()` call.
Capture and restore caller-owned sample or dataset state with `extra_state_fn` and `load_extra_state_fn`.

DeepSpeed owns its optimizer, so pass only the engine:

```python
resiliency = enable_resiliency(engine, interval=10)
```

Unsupported arguments for the selected adapter raise `TypeError`.

## Evaluate Fault Localization

The fault injection evaluation kit binds a declarative campaign to PyTorch,
TorchTitan, Megatron Core, or DeepSpeed training objects:

```python
from lm_resiliency import (
    CorruptionOperation,
    FailureType,
    FaultCampaign,
    FaultIncident,
    FaultScope,
    FaultSpec,
    FaultSurface,
    FaultTarget,
    IncidentLifetime,
    IncidentTrigger,
    enable_fault_injection,
)

campaign = FaultCampaign(
    name="output-sdc",
    incidents=(
        FaultIncident(
            incident_id="hidden-sdc",
            trigger=IncidentTrigger(at=(20,)),
            lifetime=IncidentLifetime(matching_calls=1),
            faults=(
                FaultSpec(
                    fault_id="hidden-sign-flip",
                    type=FailureType.TENSOR_CORRUPTION,
                    target=FaultTarget(
                        rank=0,
                        module_path="layers.2",
                        surface=FaultSurface.OUTPUT,
                    ),
                    parameters={
                        "operation": CorruptionOperation.SIGN_FLIP.value,
                        "scope": FaultScope.FULL.value,
                    },
                ),
            ),
        ),
    ),
)
faults = enable_fault_injection(model, optimizer, campaign=campaign)
```

The framework optimizer boundary advances the `training_iteration` clock
automatically; the training loop remains unchanged.
The injector records verified ground truth independently of SCOUT and GEMINI,
then `faults.evaluate(...)` compares neutral localization results with expected
ranks, resources, and components.
See [Fault injection evaluation](fault_injection.md) for manifests, framework
targets, permanent and intermittent schedules, destructive executors, scoring,
and safety boundaries.

## Native PyTorch

Call the package-root API after constructing the model and optimizer.
Replicated models and `DistributedDataParallel` use the standard PyTorch state-dict path.

FSDP2 and HSDP models created with `torch.distributed.fsdp.fully_shard()` use a local-shard path automatically:

```python
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard

from lm_resiliency import enable_resiliency


mesh = init_device_mesh(
    "cuda",
    (dp_replicate, dp_shard),
    mesh_dim_names=("dp_replicate", "dp_shard"),
)
for layer in model.layers:
    fully_shard(layer, mesh=mesh)
fully_shard(model, mesh=mesh)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
resiliency = enable_resiliency(model, optimizer, interval=10)
```

Use a one-dimensional mesh for pure FSDP2.
The integration infers FSDP2 or HSDP from the parameter DeviceMesh, checkpoints local DTensor shards, and unshards only the sampled module during SCOUT replay.
For HSDP, SCOUT compares ranks at the same shard coordinate across `dp_replicate`.
Pure FSDP without natural replicas uses shard peers.
For a nonstandard mesh, pass an object with `dp_replicate` and `dp_shard` attributes as `parallelism_info`.
When an initialized TorchTitan `Trainer` is passed, the adapter reads its `ParallelDims` object automatically.
For lower-level TorchTitan model-and-optimizer integration, pass `parallelism_info` explicitly.

SCOUT creates independent Gloo and NCCL detection groups from the model topology.
To provide custom groups, pass both `group` and `nccl_group`; supplying only one raises `ValueError`.

## Configure Protection

GEMINI and SCOUT are enabled by default.
Disable either component when only one is required:

```python
# GEMINI only.
resiliency = enable_resiliency(
    model,
    optimizer,
    enable_detection=False,
)

# SCOUT only.
resiliency = enable_resiliency(
    model,
    optimizer,
    enable_checkpoint=False,
)
```

Use component configuration objects for nondefault behavior:

```python
from lm_resiliency import (
    InMemoryCkptConfig,
    ReplayHarnessConfig,
    enable_resiliency,
)

resiliency = enable_resiliency(
    model,
    optimizer,
    interval=10,
    checkpoint=InMemoryCkptConfig(
        replication_jump=8,
        disk_folder="/local_nvme/gemini",
        verify_integrity=True,
    ),
    replay=ReplayHarnessConfig(
        rotate_layers=True,
        embedding_check_interval=10,
        hidden_check_interval=10,
        output_check_interval=20,
        optimizer_check_interval=10,
        straggler_confirmation_rounds=2,
        hang_stall_threshold_s=30.0,
    ),
)
```

The top-level `interval` overrides `InMemoryCkptConfig.interval` and the base `ReplayHarnessConfig.check_interval`.
Explicit embedding, hidden, output, and optimizer check intervals continue to override that base replay cadence.
See [GEMINI configuration](gemini.md#configuration) and [SCOUT configuration](scout.md#configuration) for configuration guidance.

## Instrument a DataLoader

SCOUT can measure sampled `DataLoader.next()` stalls from its out-of-band monitor:

```python
train_dataloader = resiliency.instrument_dataloader(
    train_dataloader,
    name="train",
)
```

Instrument the DataLoader before creating its long-lived iterator.
The original DataLoader is returned unchanged when SCOUT is disabled.

## Instrument Checkpoint I/O

Wrap framework-owned checkpoint calls so the out-of-band observer can localize blocked or slow reads and writes:

```python
with resiliency.checkpoint_io("write", name="periodic"):
    save_checkpoint()

with resiliency.checkpoint_io("read", name="recovery"):
    load_checkpoint()
```

SCOUT-certified durable checkpoint callbacks and fallback loads are instrumented automatically.
Use the context manager for other framework checkpoint paths.

## Add Caller-Owned State

The PyTorch adapter can include scheduler, sampler, or other caller-owned state:

```python
def capture_extra_state():
    return {
        "scheduler": scheduler.state_dict(),
        "sampler": sampler.state_dict(),
    }


def restore_extra_state(state):
    scheduler.load_state_dict(state["scheduler"])
    sampler.load_state_dict(state["sampler"])


resiliency = enable_resiliency(
    model,
    optimizer,
    extra_state_fn=capture_extra_state,
    load_extra_state_fn=restore_extra_state,
)
```

Do not include model or optimizer state.
The framework adapter captures those objects.

## Configure Recovery

Provide the framework's durable loader as the fallback:

```python
resiliency = enable_resiliency(
    model,
    optimizer,
    load_fallback=load_durable_checkpoint,
)
```

GEMINI first attempts its available memory and node-local tiers.
It calls `load_fallback` only when no recoverable GEMINI checkpoint exists.

Use `resiliency.step_count` when the training loop needs an explicit resume point:

```python
for step in range(resiliency.step_count, max_steps):
    train_step()
```

## Configure MoE Replay

Bind a qualified execution-regime catalog to the post-dispatch expert stage:

```python
from lm_resiliency import (
    GroupedExpertMaterializer,
    ReplayHarnessConfig,
    ReplayWorkload,
    enable_resiliency,
)

workload = ReplayWorkload.from_moe_catalog(
    catalog,
    replay_modules=[model.post_dispatch_expert_stage],
    materializer=GroupedExpertMaterializer(),
)

resiliency = enable_resiliency(
    model,
    optimizer,
    replay=ReplayHarnessConfig(
        workload=workload,
        capture_inputs_by_value=True,
        rotate_layers=False,
    ),
)
```

`GroupedExpertMaterializer` supports the common `(tokens, tokens_per_expert, probabilities)` contract.
For scalar recipe `n_exec`, it sets every local expert count to `n_exec` and resizes token-aligned inputs to `n_exec * local_expert_count` rows.
Use `counts_input` for a different count argument and `alignment` for backend row alignment.

Use `LeadingDimensionMaterializer` only when the expert boundary has no separate count or offset metadata.
`ReplayWorkload.from_moe_catalog()` marks the workload as expert replay, so Megatron, DeepSpeed, and mesh-based adapters select expert-data-parallel peers automatically.
For a custom expert workload built with `ReplayWorkload.from_shapes()`, pass `peer_role="expert"`.
See [MoE execution regimes](moe_execution_regimes.md) for catalog qualification and exact fallback.

## Configure AllToAll Replay

SCOUT captures Python-visible AllToAll tensor layouts, split matrices, sequence positions, and process-group membership.
At each replay check, the configured policy generates bounded representative traffic matrices.
The shared executor materializes deterministic rank-tagged payloads, executes the matrices on the captured process group, verifies routing exactly, and reports timing through C3 and Cross-PG localization.

The default `BalancedAndPermutationPolicy` executes at most two matrices:

- a balanced matrix for concurrent peer traffic; and
- a cyclic permutation matrix that places each sender's payload on one route while keeping per-rank send and receive volume equal.

The default payload budget is 4 MiB per rank and one tensor row is the minimum allocation unit.
The matrices depend on process-group size, tensor row width, and sequence position rather than observed token routing, so equivalent data-parallel replicas remain comparable.
This is representative coverage, not exhaustive coverage of every possible routing matrix.

Change the payload bound through `ReplayHarnessConfig`:

```python
from lm_resiliency import (
    BalancedAndPermutationPolicy,
    ReplayHarnessConfig,
    enable_resiliency,
)

resiliency = enable_resiliency(
    model,
    optimizer,
    replay=ReplayHarnessConfig(
        all_to_all_policy=BalancedAndPermutationPolicy(
            max_payload_bytes_per_rank=2 * 1024 * 1024,
        ),
    ),
)
```

To define another policy, implement `AllToAllReplayPolicy.generate(capture)` and return one or more square `AllToAllTrafficMatrix` values.
`capture.observed_splits` exposes the reconstructed training matrix for trace-based policies.
Return matrices in deterministic order on equivalent replicas.
Set `all_to_all_policy=None` to disable this replay surface.

## Report Faults

Direct integrations have two callback paths:

| Callback | Payload | Signals |
|---|---|---|
| `fault_callback` | `ReplayResult` | SDC and replay-localized stragglers |
| `oob_fault_callback` | `SCOUTFaultReport` | Hangs, DataLoader stalls, and checkpoint I/O stalls |

`replay_fault_reports(result)` converts a `ReplayResult` into JSON-ready `SCOUTFaultReport` records.
For confirmed communication stragglers, SCOUT automatically intersects slow process groups and reports the shared rank's host as `scope="node"`.
The report includes the endpoint rank and host, supporting and contradictory group evidence, and confidence.
Use `OrchestrationHooks` when an external manager needs one normalized callback.

## Certify Durable Checkpoints

`DurableCheckpointConfig` connects SCOUT evidence to framework-owned checkpoint bytes:

```python
from lm_resiliency import (
    CallbackDurableCheckpointAdapter,
    DurableCheckpointConfig,
    enable_resiliency,
)

adapter = CallbackDurableCheckpointAdapter(
    save_candidate_fn=save_candidate,
    load_checkpoint_fn=load_checkpoint,
    commit_candidate_fn=commit_candidate,
    quarantine_candidate_fn=quarantine_candidate,
)

resiliency = enable_resiliency(
    model,
    optimizer,
    durable_checkpoint=DurableCheckpointConfig(
        manifest_dir="/shared/checkpoints/scout",
        environment_id=job_environment_id,
        adapter=adapter,
    ),
)
```

Certification requires SCOUT.
An accepted dense check certifies its checkpoint immediately because the dense shape catalog has one entry.
Dynamic MoE training runs one configured replay-shape recipe per scheduled check.
After the first complete MoE catalog cycle, the framework persists the boundary as `CANDIDATE`.
The following complete clean cycle promotes that checkpoint to `RECOVERY_VERIFIED` and creates the next candidate.
The manifest directory must have durability equivalent to the framework checkpoint store.
Detected SDC and inaccessible-machine recovery select only the
recovery-verified manifest and do not call an unconstrained `load_fallback`.
See [SCOUT checkpoint certification](scout.md#checkpoint-certification-and-recovery) for the callback contract.

## Select Recovery Trust

The launcher can classify a failure before recovery:

```python
mode = resiliency.prepare_recovery(
    "hang",
    all_ranks_accessible=True,
)
```

The high-level choices are:

| Failure condition | Selected checkpoint |
|---|---|
| Accessible straggler | Latest complete GEMINI checkpoint |
| Hang or uncertain failure, all ranks accessible, full-catalog replay complete and clean | Latest complete GEMINI checkpoint |
| Full-catalog replay detects SDC, is incomplete, or returns missing evidence | `RECOVERY_VERIFIED` |
| SDC already detected | `RECOVERY_VERIFIED` |
| Required rank or machine inaccessible | `RECOVERY_VERIFIED` |

On relaunch, pass the selected mode when it is known externally:

```python
from lm_resiliency import RecoveryMode, enable_resiliency

resiliency = enable_resiliency(
    model,
    optimizer,
    recovery_mode=RecoveryMode.RECOVERY_VERIFIED,
)
```

An SDC decision is also persisted beside GEMINI checkpoint metadata so a
same-storage restart cannot silently choose a newer candidate.

## Manager Integration

`OrchestrationHooks` provides a platform-neutral boundary for fault reports, checkpoint selection, and restart destinations:

```python
from lm_resiliency import enable_resiliency
from lm_resiliency.manager_api import OrchestrationHooks

hooks = OrchestrationHooks(
    report_fault=lambda report: control.report_fault(rank, report),
    report_recovery=lambda decision: control.report_recovery(rank, decision),
    restart_destination=control.restart_destination,
)

resiliency = enable_resiliency(
    model,
    optimizer,
    orchestration=hooks,
)
```

`report_recovery` receives a JSON-ready `RecoveryDecision` before the corresponding automatic fault report.
The decision identifies the failure class, selected trust mode, checkpoint source, step, optional durable checkpoint ID, availability, and selection reason.
Checkpoint lookup in the reporting process is rank-local and noncollective because a fault may have already made the training process group unusable.
The manager can conservatively combine worker decisions, include the selected decision in its restart command, and pass `recovery_mode` to the relaunched integration.
The restarted GEMINI loader retains responsibility for selecting a rank-consistent generation collectively.

In-process SDC automatically emits a recovery-verified decision, while a replay-localized straggler emits a latest-GEMINI decision.
An OOB hang emits a conservative recovery-verified decision without entering a blocked training process or invoking replay.
An instrumented DataLoader stall emits a latest-GEMINI decision because the reporting worker remains accessible.
An instrumented checkpoint I/O stall also emits a latest-GEMINI decision.
Explicit calls to `prepare_recovery(...)` emit the resulting decision and retain it as `last_recovery_decision`.

Before replacing or migrating a worker, preserve its latest recoverable state:

```python
step = resiliency.flush_for_restart()
destination = control.restart_destination()

if step >= 0 and destination is not None:
    resiliency.copy_checkpoint_to(destination)
```

`copy_checkpoint_to` copies flushed local and peer shards.
Import `make_transfer` from `lm_resiliency.manager_api` when endpoint-addressed manager transfer is required.
The `"torch_dist"` and `"nixl"` backends share the `CheckpointTransfer` contract.

Launcher retry, placement, replacement, and quarantine policy remain outside this repository.

## Monitor Hardware Health

The optional hardware-health API complements replay and out-of-band consensus with direct telemetry for permanent or imminent device failures.
It is not enabled automatically by `enable_resiliency()`.

```python
from lm_resiliency.manager_api import (
    HardwareHealthMonitor,
    HealthConfig,
    NvmlSource,
)

monitor = HardwareHealthMonitor(
    HealthConfig(poll_interval_s=5.0),
    [NvmlSource(device_index=physical_gpu_index)],
    on_event=manager_health_callback,
)
monitor.start()
```

The caller resolves `physical_gpu_index` and supplies `manager_health_callback`.
`on_event` receives one deduplicated `HealthEvent` for each fatal device-and-metric pair.
The built-in NVML source covers uncorrectable ECC, row-remap failure, NVLink error growth, temperature near shutdown, and device loss.
Custom `HealthSource` implementations can supply XID, PCIe, InfiniBand, NIC, HCA, or fabric telemetry.
Warnings such as correctable ECC growth, pending row remaps, nonfatal XIDs, and elevated temperature are logged but do not invoke `on_event`.

Call `monitor.close()` when the manager or worker lifecycle ends.
See [SCOUT hardware telemetry](scout.md#hardware-telemetry) for thresholds, localization granularity, and manager ownership.

## Lifecycle

Every framework adapter returns the same lifecycle surface:

| API | Behavior |
|---|---|
| `step_count` | Completed optimizer steps, including the recovered resume point |
| `instrument_dataloader(...)` | Adds sampled DataLoader-stall instrumentation |
| `checkpoint_io(...)` | Marks a framework checkpoint read or write for OOB localization |
| `flush_for_restart()` | Flushes recoverable GEMINI state and returns its step, or `-1` |
| `set_restart_destination(...)` | Sets an optional restart-mirror resolver |
| `copy_checkpoint_to(...)` | Copies flushed checkpoint shards |
| `prepare_recovery(...)` | Runs any required emergency recipe cycle and selects latest or recovery-verified trust |
| `last_recovery_decision` | Most recent JSON-ready checkpoint decision emitted to orchestration |
| `close()` | Optionally releases resources before process exit |

Normal training scripts do not need to call `close()`.
Call it only when a long-lived process must tear down one training session before starting another.

## Advanced Integration

Framework-specific entry points remain available for explicit devices, process groups, and nonstandard topology metadata:

- `lm_resiliency.integrations.pytorch.enable_resiliency`
- `lm_resiliency.integrations.torchtitan.enable_resiliency`
- `lm_resiliency.integrations.megatron.enable_resiliency`
- `lm_resiliency.integrations.deepspeed.enable_resiliency`

Prefer the package-root API unless the automatic adapter cannot represent the topology.
Import low-level checkpoint managers, replay detectors, topology objects, and concrete transfer implementations from `lm_resiliency.experimental` only when accepting an unstable API.

## Command-Line Interface

The package installs `lm-resiliency-discover-moe-regimes` for building a SCOUT replay catalog from saved profiler observations, an environment description, and a complete request manifest.
See [MoE execution regimes](moe_execution_regimes.md#build-and-audit-the-catalog) for the command and catalog-qualification requirements.

## Related Guides

- [Production-loop examples](../examples/README.md)
- [Fault injection evaluation](fault_injection.md)
- [GEMINI](gemini.md)
- [SCOUT](scout.md)
- [MoE execution regimes](moe_execution_regimes.md)
- [Validation summary](validation.md)
