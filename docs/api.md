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
| `lm_resiliency.integrations.torchrun` | Stable torchrun worker-adapter API |
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
| Fault injection evaluation | `enable_fault_injection`, `SCHEMA_VERSION`, `FaultCampaign`, `FaultIncident`, `IncidentTrigger`, `IncidentLifetime`, `IterationRange`, `ClockSpec`, `ClockType`, `ClockOrigin`, `FaultSpec`, `FaultTarget`, `FailureType`, `SystemFailureType`, `FaultSurface`, `FaultScope`, `FaultMagnitude`, `CorruptionOperation`, `SafetyClass`, `RetriggerPolicy`, `FaultExecutor`, `CallbackFaultExecutor`, `FaultExecutionRequest`, `FaultExecutionResult`, `UnsupportedFaultError`, `FaultInjectionSession`, `InjectionStatus`, `FaultInjectionRecord`, `LocalizationResult`, `FaultEvaluation`, `CampaignReport`, `CampaignJournal`, `CampaignStateStore`, `MemoryCampaignStateStore`, `JsonCampaignStateStore` |

The stable manager API exports are:

| Area | Exports |
|---|---|
| Coordination | `OrchestrationHooks`, `RestartDestinationResolver`, `RecoveryDecision`, `RecoveryDecisionCallback` |
| Fault reports | `SCOUTFaultReport`, `SCOUTFaultCallback`, `replay_fault_reports`, `dispatch_replay_faults` |
| Checkpoint transfer | `CheckpointTransfer`, `TransferMetadataStore`, `make_transfer` |
| Hardware health | `HealthConfig`, `HardwareHealthMonitor`, `HealthEvent`, `HealthReading`, `HealthSeverity`, `HealthSource`, `NvmlSource` |
| Configuration drift | `local_fingerprint`, `find_drift`, `format_drift` |

The stable torchrun integration exports are:

| Area | Exports |
|---|---|
| Worker contract | `TorchrunWorkerAdapter`, `TorchrunWorkerContext`, `TorchrunWorkerAdapterError`, `get_torchrun_worker_context` |
| Built-in adapters | `NativePyTorchAdapter`, `NativePyTorchDDPAdapter`, `TorchTitanWorkerAdapter`, `MegatronWorkerAdapter`, `DeepSpeedWorkerAdapter` |
| Manager coordination | `TorchrunRecoveryCoordinator`, `TorchrunRecoveryRequest`, `TorchrunInitialPlacement`, `TorchrunSuccessorPlacement` |
| Launcher configuration | `TorchrunLaunchConfig` |
| Node identity | `derive_torchrun_node_id` |
| Rendezvous registration | `create_rendezvous_handler`, `get_rendezvous_handler_creator` |

Other module paths are internal unless this guide identifies them as supported.
Objects under `lm_resiliency.experimental` may change within the `0.x` release series.

## Enable Through Torchrun

The `lm_resiliency` rendezvous backend installs framework-import monitoring
before the user module starts, then selects the corresponding worker adapter as
the training stack is imported. The user training module does not import
`lm_resiliency`:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=1 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint=/tmp/lm-resiliency-rdzv \
  --rdzv-id=my-run \
  --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-context/context.json,\
lm_resiliency_worker_config=/absolute/path/worker.toml" \
  --module your_training.module \
  [application arguments...]
```

Set the context path with the
`lm_resiliency_restart_context_path` rendezvous option. The runtime creates the
parent with owner-only permissions while constructing the rendezvous handler.
If the directory already exists with different ownership or group/other access,
startup fails closed instead of changing administrator-provisioned permissions.

The rendezvous handler derives a stable node identity from `/etc/machine-id`
and commits the first `min_nodes` unique registrations as the initial training
group. Additional registered nodes remain parked as standbys. Test harnesses
and container deployments may point `LM_RESILIENCY_MACHINE_ID_PATH` at another
absolute file containing a valid machine ID; duplicate identities fail closed.

Built-in adapter selection is inferred from framework imports:

| Imported framework | Objects passed to the existing framework integration |
|---|---|
| Native PyTorch | One unambiguous root `torch.nn.Module` and optimizer |
| TorchTitan | The initialized `torchtitan.train.Trainer` |
| Megatron Core | The model chunks, optimizer, and scheduler returned by `setup_model_and_optimizer()` |
| DeepSpeed | The engine returned by `deepspeed.initialize()` |

PyTorch is treated as tentative because every higher-level framework imports
it. Importing TorchTitan, Megatron Core, or DeepSpeed before attachment selects
that more specific adapter. Importing multiple higher-level supported
frameworks fails closed instead of selecting one arbitrarily. Distributed
workers also agree on the inferred framework when the default process group
initializes, before any built-in adapter enters attachment collectives.
When the default backend is NCCL, this one-time agreement uses a temporary
Gloo group so it does not depend on the application's CUDA device-binding
order.

Worker adapters do **not** accept a parallelism strategy. They pass the same
framework objects to the existing `enable_resiliency()` API that explicit user
code would pass. The existing framework integration remains the single owner of
DDP, FSDP2/HSDP, TP, SP, CP, PP, EP, expert-TP, ZeRO, and framework-specific
process-group discovery.

The distributed PyTorch adapter attaches only after a DDP or FSDP construction
boundary that every participating rank must cross. At the first recognized
distributed forward, all ranks agree that each owns exactly one valid
model/optimizer pair before any rank starts LM Resiliency attachment
collectives. The optimizer must own all trainable parameters of that model and
no parameters from another model. Multiple optimizers, foreign parameters, or
a distributed forward without a recognized all-rank construction boundary fail
closed before attachment. Single-process jobs retain
root-module-forward discovery. Bottom-up FSDP2/HSDP wrapping is reduced to the
outermost sharded root, and optimizer subclasses defined after bootstrap are
instrumented when the class is created.
The other built-ins attach at their framework-owned initialization boundary and
likewise fail closed if the expected complete object bundle is unavailable.
On a replacement generation, built-ins accept GEMINI restart contexts and load
only the exact manager-selected step. The live checkpoint-topology digest must
match the manager plan before checkpoint state is applied. Missing, corrupt,
unverified, newer, older, or topology-incompatible state fails closed. A
durable-source restart requires a custom worker adapter that owns the
framework's durable loader.
Before the user module starts, bootstrap publishes the manager-selected resume
position as `LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP` (`0` for generation zero).
Zero-import applications can use that value to restore deterministic sampler or
input position while the adapter independently verifies the recovered
framework state.
After manager-selected GEMINI recovery, the built-in DeepSpeed adapter rejects
a later `engine.load_checkpoint()` call because it could overwrite the selected
model and optimizer state. DeepSpeed applications that also restore
framework-owned client state must provide a custom worker adapter that
coordinates both recovery mechanisms.
For generation zero, a successful DeepSpeed framework load is allowed before
the first resiliency-managed training step and seeds the resiliency counter
from `engine.global_steps`.
Built-in adapters also close their resiliency handle at the framework's normal
distributed teardown boundary, before process groups are destroyed. This
completes asynchronous checkpoint and replay work before user code reports a
clean shutdown.

Worker policy is a strict versioned TOML file:

```toml
schema_version = 1
interval = 10
enable_checkpoint = true
enable_detection = true

[checkpoint]
disk_folder = "/shared/checkpoints"
disk_flush_interval = 100
verify_integrity = true

[replay]
rotate_layers = true
```

Unknown fields, missing or unsupported schema versions, and invalid values fail
closed. Assigned nodes must agree on the exact policy contents at rendezvous,
and each worker revalidates that digest before parsing the policy. Disabled
feature sections are still validated. `replication_jump` must
be valid for the deployed checkpoint group; it is not inferred from the
launcher node count. `checkpoint.disk_flush_interval` must be nonnegative; zero
disables periodic disk persistence. `checkpoint.replication_jump` must be `-1`
or positive, and `checkpoint.replication_chunk_size` must be positive. Supplying
`lm_resiliency_worker_config` enables automatic worker
instrumentation; omitting it leaves explicit `enable_resiliency()` integrations
unchanged. Explicit integrations can call `get_torchrun_worker_context()` to
obtain run identity, hashed node identity, worker width, generation, logical
slot, topology digest, and the manager-selected recovery decision without
duplicating them as application arguments. Workers obtain their local width from torchrun's standard
`LOCAL_WORLD_SIZE` environment and verify replacement plans against it.

Managers use `TorchrunRecoveryCoordinator` with the same c10d store used by the
rendezvous backend. It exposes committed initial placement, publishes immutable
same-node or replacement successor generations, and closes the run to wake
parked standbys. After successor publication, the coordinator renews a
generation-specific store lease until the manager deadline. Agents combine its
sequence and remaining duration with host-local monotonic time, so restart
eligibility does not depend on synchronized wall clocks. Keep the coordinator
alive through successor admission and poll `check_health()` while waiting.
Transient lease-store failures are retried; persistent renewal failure is
latched and raised by `check_health()` and subsequent coordinator operations.
`TorchrunLaunchConfig` constructs the common LM Resiliency torchrun arguments
without owning subprocess, scheduler, SSH, or GPU placement behavior.

Custom stacks set `adapter = "package.module:factory"` in the worker TOML. The
factory receives `TorchrunWorkerContext` and returns an object implementing
`install(context)`. A stack adapter owns framework-specific discovery and loop
state. Successor contexts include the manager-selected checkpoint source, step,
durable checkpoint ID when applicable, checkpoint manifest ID, topology digest,
and recovery deadline. Torchrun itself cannot infer scheduler, dataloader
position, or arbitrary caller-owned iteration state safely.

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
    SCHEMA_VERSION,
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
    SystemFailureType,
    enable_fault_injection,
)

campaign = FaultCampaign(
    name="output-sdc",
    schema_version=SCHEMA_VERSION,
    incidents=(
        FaultIncident(
            incident_id="hidden-sdc",
            trigger=IncidentTrigger(at=(20,)),
            lifetime=IncidentLifetime(matching_calls=1),
            faults=(
                FaultSpec(
                    fault_id="hidden-sign-flip",
                    type=FailureType.TENSOR_CORRUPTION,
                    system_failure_type=SystemFailureType.TRANSIENT_COMPUTE_CORRUPTION,
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
Custom `CampaignStateStore` implementations must provide atomic
`compare_and_swap(expected, updated)` in addition to `load()` and `save()`, so
overlapping old and replacement workers cannot both claim a restart-stable
occurrence.
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
        run_id="training-run-2026-08-15",
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
The decision identifies the failure class, selected trust mode, checkpoint
source, step, optional durable checkpoint ID, checkpoint-topology digest,
availability, and selection reason.
Checkpoint lookup in the reporting process is rank-local and noncollective because a fault may have already made the training process group unusable.
The manager can conservatively combine worker decisions, include the selected decision in its restart command, and pass `recovery_mode` to the relaunched integration.
The restarted GEMINI loader retains responsibility for collectively validating
and loading that exact manager-selected generation.

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
Every transfer is keyed, bounded by `timeout_s`, and validates endpoint, tensor layout, byte count, and per-chunk checksums before applying received buffers.
`TorchDistTransfer` uses a dedicated Gloo pair group even when the training world uses NCCL; both endpoints create the same pair in coordinated order, or the manager may pass an already-created dedicated Gloo `process_group`.
NIXL is one-sided while the torch-distributed backend requires both endpoints to participate, so `make_transfer("nixl", ...)` fails if NIXL is unavailable unless the manager explicitly sets `allow_backend_fallback=True` after arranging the two-sided protocol.

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
