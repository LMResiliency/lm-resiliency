# Fault Injection Evaluation Kit

The fault injection evaluation kit runs reproducible failure campaigns against
PyTorch, TorchTitan, Megatron Core, and DeepSpeed pre-training workloads.
It records verified ground truth and accepts neutral localization results, without
depending on SCOUT, GEMINI, or another resiliency system.

Campaigns describe observable failure effects.
Built-in hooks execute safe model and optimizer faults; callback executors connect
the same schema to isolated process, storage, communication, and cluster controls.
See the [fault injection and SCOUT localization example](../examples/fault_injection/README.md)
for a production DDP loop that compares injection and localization JSON artifacts.

## Enable a Campaign

Enable the campaign once after framework initialization and before training:

```python
from lm_resiliency import enable_fault_injection

faults = enable_fault_injection(
    model,
    optimizer,
    campaign=campaign,
)

train()
```

No per-iteration trigger is required.
The integration observes framework optimizer boundaries and arms incidents at the
arbitrary training iterations specified by the campaign.
In a distributed job, every rank completes campaign preparation and capability
validation before any current-iteration incident is armed. Preparation rejects
target ranks outside the process-group world size and requires every rank to
report the same canonical manifest and `current_iteration`. This prevents an
iteration-one process, communication, or cluster fault from interrupting
enablement consensus or running with divergent schedules.
Later scheduled safe incidents use the same rank-consistent preflight and arming
consensus at their optimizer boundary. Iterations without a scheduled campaign
candidate add no distributed synchronization. The optional `rank=` override is
reserved for non-distributed testing; in an initialized distributed job it must
match the process's global rank.

The first arguments match the framework lifecycle:

| Framework | Enablement |
|---|---|
| PyTorch | `enable_fault_injection(model, optimizer, campaign=...)` |
| TorchTitan | `enable_fault_injection(trainer, campaign=...)` |
| Megatron Core | `enable_fault_injection(model_chunks, optimizer, campaign=...)` |
| DeepSpeed | `enable_fault_injection(engine, campaign=...)` |

Framework selection is automatic and is not stored in the campaign.

## Training Iterations

A `training_iteration` is one logical global training step that normally ends at
an optimizer update.
It may contain multiple microbatches, forward calls, backward calls, pipeline
stages, or gradient-accumulation passes.

An incident at iteration 100 is armed after iteration 99 completes and applies to
the training work that produces optimizer update 100.
Forward-call counts and SCOUT check cadence do not affect this clock.

The default clock uses absolute training-run iterations:

```json
{
  "type": "training_iteration",
  "origin": "training_run"
}
```

TorchTitan and DeepSpeed progress is discovered from their training objects.
Pass `completed_iterations=` when native PyTorch or Megatron is enabled after an
externally managed resume.
Use `"origin": "campaign_start"` when iteration numbers should be relative to
enablement instead.

## Incident Manifest

The versioned JSON format separates the incident schedule from its correlated
failure effects:

```json
{
  "schema_version": 1,
  "name": "mixed-production-failures",
  "seed": 17,
  "clock": {
    "type": "training_iteration",
    "origin": "training_run"
  },
  "incidents": [
    {
      "incident_id": "sporadic-sdc",
      "trigger": {
        "at": [37, 203, 911],
        "probability": 1.0
      },
      "lifetime": {
        "matching_calls": 1
      },
      "retrigger": "once",
      "faults": [
        {
          "fault_id": "hidden-output",
          "type": "tensor_corruption",
          "target": {
            "rank": 3,
            "component": "transformer_block",
            "index": 12,
            "surface": "output"
          },
          "parameters": {
            "operation": "sign_flip",
            "scope": "1%"
          }
        }
      ]
    },
    {
      "incident_id": "intermittent-straggler",
      "trigger": {
        "range": {
          "start": 300,
          "end": 500,
          "every": 20
        },
        "probability": 0.25
      },
      "lifetime": {
        "iterations": 3
      },
      "retrigger": "once",
      "faults": [
        {
          "fault_id": "collective-delay",
          "type": "delay",
          "target": {
            "rank": 5,
            "surface": "collective",
            "operation": "all_reduce"
          },
          "parameters": {
            "delay_ms": 500
          }
        }
      ]
    }
  ],
  "metadata": {
    "workload": "llama-pretraining",
    "topology": "dp=8,tp=2"
  }
}
```

Load and write manifests with `FaultCampaign.from_json(...)` and
`FaultCampaign.to_json(...)`.
The versioned parser rejects unknown fields at every executable schema level,
including misspelled trigger fields. Arbitrary user data remains allowed only
inside `metadata` and executor-defined `parameters`.

## Campaign Field Reference

A campaign has four nested levels:

```text
campaign
  incidents[]
    faults[]
      target
      parameters
```

One campaign contains scheduled incidents. An incident may contain multiple
correlated fault actions, and every action has its own target and executor
parameters.

### Campaign

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `schema_version` | No | `1` | Manifest compatibility version. Serialized campaigns always include it; unsupported versions are rejected. |
| `name` | Yes | - | Non-empty campaign identifier included in reports and state-store records. |
| `seed` | No | `0` | Signed 128-bit seed used for deterministic probability selection and fault randomness. |
| `clock` | No | `{"type": "training_iteration", "origin": "training_run"}` | Defines how incident trigger positions are interpreted. |
| `incidents` | Yes | - | Non-empty array of incident objects. `incident_id` values must be unique in the campaign. |
| `metadata` | No | `{}` | User-defined experiment metadata such as model, workload, topology, or ticket ID. It does not change execution. |

### Clock

| Field | Values | Meaning |
|---|---|---|
| `type` | `training_iteration` | Schedule incidents at logical training iterations, not forward calls or microbatches. |
| `origin` | `training_run`, `campaign_start` | Use absolute training-run progress or count relative to campaign enablement. |

With `training_run`, an incident at iteration 100 still refers to iteration 100
after a restart. With `campaign_start`, iteration 1 is the first iteration after
`enable_fault_injection()`.

### Incident

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `incident_id` | Yes | - | Non-empty identifier unique within the campaign. |
| `trigger` | Yes | - | Candidate training iterations and optional selection probability. |
| `lifetime` | Yes | - | How long each selected occurrence remains active. |
| `retrigger` | No | `once` | Whether the same candidate may fire again after rollback or restart. |
| `max_occurrences` | Conditional | - | Positive retry limit; valid only when `retrigger` is `max_occurrences`. |
| `faults` | Yes | - | Non-empty array of correlated fault actions. `fault_id` values must be unique within the incident. |

All actions in `faults` share the incident trigger, lifetime, occurrence ID, and
temporal classification. For example, one incident can delay three ranks at the
same iteration. That is one occurrence containing three fault actions.

The runtime generates an occurrence ID as `<incident_id>@<iteration>`. A repeated
attempt is suffixed with `#<attempt>`, for example `network-delay@500#2`.

### Trigger

Use exactly one of `at` or `range`.

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `at` | One of `at`/`range` | - | Sorted, unique array of positive iteration numbers. |
| `range.start` | For `range` | - | First candidate iteration; must be positive. |
| `range.end` | For `range` | - | Last candidate iteration, inclusive. |
| `range.every` | No | `1` | Positive spacing between range candidates. |
| `probability` | No | `1.0` | Selection probability for each candidate, from `0.0` through `1.0`. |

Probability does not make a campaign irreproducible. Selection is derived from
`seed`, `incident_id`, and the candidate iteration, so the same manifest selects
the same occurrences.

### Lifetime

Use exactly one lifetime field:

| Field | Value | Meaning |
|---|---|---|
| `matching_calls` | Positive integer | Apply the fault to that many matching target operations, then remove it. Not valid for `weight`, `bias`, or `optimizer_state`; use `iterations` or `until` so state mutations remain active through backward and the optimizer boundary. |
| `iterations` | Positive integer | Keep the fault active for that many training iterations, including its trigger iteration. |
| `until` | `recovery` | Keep it active until `FaultInjectionSession.notify_recovery()`. |
| `until` | `replacement` | Keep it active until `FaultInjectionSession.notify_replacement()`. |
| `until` | `campaign_end` | Keep it active until the session closes. |

An `until` lifetime is permanent and therefore requires a single trigger
candidate. Multiple bounded candidates or probabilistic bounded candidates are
classified as intermittent; one bounded candidate is transient.

### Retrigger Policy

| Value | Behavior |
|---|---|
| `once` | Fire a candidate at most once. This is restart-stable when the campaign state store survives worker replacement. |
| `every_attempt` | Fire whenever execution reaches the candidate again after rollback. |
| `max_occurrences` | Fire at most `max_occurrences` times for the candidate. |

### Fault Action

| Field | Required | Meaning |
|---|---:|---|
| `fault_id` | Yes | Non-empty action identifier unique within its incident. |
| `type` | Yes | Canonical observable failure effect, such as `tensor_corruption`, `delay`, or `process_termination`. |
| `target` | Yes | Framework-neutral location where the effect is applied. |
| `parameters` | No | Type- and executor-specific settings. Defaults to `{}`. |

The runtime generates an injection ID as
`<occurrence_id>/<fault_id>`. Correlated actions therefore have different
injection IDs but the same occurrence ID.

### Target

| Field | Required | Default | Meaning |
|---|---:|---|---|
| `surface` | Yes | - | Training surface: `input`, `output`, `weight`, `bias`, `gradient`, `optimizer_state`, `rng_state`, `sampler_state`, `data`, `checkpoint`, `compute`, `collective`, `process`, `resource`, or `config`. |
| `rank` | No | `0` | Global rank that executes the action. |
| `model_part` | No | `0` | TorchTitan model-part or Megatron model-chunk index. |
| `component` | For logical module targets | - | Logical component such as `transformer_block`, `embedding`, `output`, or `expert`. |
| `index` | For indexed components | - | Global layer or expert index. |
| `module_path` | For explicit module targets | - | Exact path from `named_modules()`, such as `model.layers.12.mlp`. It is attempted before logical component resolution. |
| `operation` | Executor-specific | - | Runtime operation such as `all_reduce` for a collective fault. |
| `resource` | Executor-specific | - | Resource selector such as a GPU, NIC, worker, or node identifier. |
| `path` | Executor-specific | - | Checkpoint or storage path. |
| `metadata` | No | `{}` | Additional selector data consumed by a custom executor. |

Module surfaces (`input`, `output`, `weight`, `bias`, `gradient`,
`optimizer_state`, and `compute`) require either `module_path` or `component`.
Supplying both is useful when the exact framework path is known but a logical
component should remain available as a fallback and for localization comparison.
For pipeline-sharded Megatron, TorchTitan, or DeepSpeed models, logical
transformer-layer indices are resolved through global module metadata such as
`layer_number` or `global_layer_index`. The injector rejects ambiguous
stage-local numbering instead of binding the same local suffix on every stage.

### Built-In Parameters

| Parameter | Applies to | Default | Meaning |
|---|---|---|---|
| `operation` | `tensor_corruption` | Required | `single_bitflip`, `multi_bitflip`, `set_value`, `scale`, `noise`, or `sign_flip`. |
| `scope` | Local tensor and state-flow faults | `single` | Elements selected by the effect: `single`, `row`, `1%`, `10%`, or `100%`. `reorder` always reorders the full leading dimension. |
| `magnitude` | Tensor corruption | `medium` | `catastrophic`, `large`, `medium`, `subtle`, or `near_invisible`; selects bit position or the default scale/noise strength. |
| `value` | `set_value` | Required | Numeric value, or `nan`, `inf`, or `-inf`. |
| `factor` | `scale` | Derived from `magnitude` | Explicit multiplication factor. |
| `std` | `noise` | Derived from `magnitude` | Gaussian-noise standard deviation. |
| `parameter` | Parameter, gradient, or optimizer-state target | Surface-dependent | Exact parameter attribute, commonly `weight` or `bias`. |
| `state_key` | `optimizer_state` | First suitable tensor | Exact optimizer-state entry, such as `exp_avg` or `exp_avg_sq`. |
| `delay_ms` | `delay` | Required | Positive delay in milliseconds. |

Stale and duplicate faults retain two observed values only during their
scheduled collection window. Module and gradient observation starts one
training iteration before a candidate and is removed at the optimizer boundary
after the occurrence. State surfaces are snapshotted over the equivalent
two-boundary window. A stale or duplicate incident scheduled for the first
iteration fails verification because no prior value exists.

Weight, bias, and optimizer-state faults with bounded iteration lifetimes are
retired by removing the injected finite delta from the current tensor. This
preserves optimizer updates made during the active window. A bounded state fault
whose injected delta is non-finite is rejected before mutation; represent such
corruption with an `until` lifetime and recover the workload afterward.
FSDP2/HSDP `DTensor` state is mutated and verified through the rank-local shard.

Custom executors receive the full `target` and `parameters` objects unchanged.
They may define additional fields, but should validate those fields before the
training loop begins.

## Schedule Examples

Exact arbitrary iterations:

```json
{"at": [37, 203, 911], "probability": 1.0}
```

Inclusive periodic range:

```json
{
  "range": {"start": 100, "end": 500, "every": 20},
  "probability": 0.25
}
```

Permanent fault beginning at one exact iteration:

```json
{
  "trigger": {"at": [1200]},
  "lifetime": {"until": "replacement"}
}
```

## Failure Model

The canonical types cover common observable LLM pre-training failures:

| Family | Types |
|---|---|
| Numerical and state | `tensor_corruption`, `stale_state`, `drop`, `duplicate`, `reorder`, `config_drift` |
| Performance and liveness | `delay`, `hang`, `timeout` |
| Runtime and availability | `exception`, `resource_exhaustion`, `process_termination`, `resource_unavailable` |
| Checkpoint and storage | `checkpoint_corruption`, `checkpoint_truncation`, `checkpoint_missing`, `io_error` |
| Communication | `payload_corruption`, `collective_desync`, `message_drop`, `network_partition` |

The built-in local executor supports:

- numerical corruption of module inputs, outputs, weights, biases, gradients,
  and optimizer-state tensors;
- stale or duplicated inputs, outputs, parameters, gradients, and optimizer state;
- dropped or reordered inputs, outputs, and gradients; and
- delays on logical module or compute targets.

Tensor corruption operations are `single_bitflip`, `multi_bitflip`, `set_value`,
`scale`, `noise`, and `sign_flip`.
Scopes are `single`, `row`, `1%`, `10%`, and `100%`.
Bit flips support `float16`, `bfloat16`, `float32`, and `float64`.

## Logical Targets

Prefer framework-neutral targets:

```json
{
  "rank": 3,
  "component": "transformer_block",
  "index": 12,
  "surface": "gradient"
}
```

The built-in resolver recognizes transformer blocks, embeddings, output heads,
and experts.
Use `module_path` as an explicit escape hatch:

```json
{
  "rank": 3,
  "module_path": "model.layers.12.mlp",
  "surface": "output"
}
```

Other target fields identify an operation, resource, or storage path.
The execution rank applies the action and records rank-local ground truth.

## Destructive and Environment-Specific Failures

Process, checkpoint, storage, communication, node, and network effects require an
executor with explicit capabilities:

```python
from lm_resiliency import (
    CallbackFaultExecutor,
    FailureType,
    FaultExecutionResult,
    SafetyClass,
)

executor = CallbackFaultExecutor(
    name="cluster-manager",
    supported_types={
        FailureType.PROCESS_TERMINATION,
        FailureType.RESOURCE_UNAVAILABLE,
        FailureType.NETWORK_PARTITION,
    },
    max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    activate=lambda request: FaultExecutionResult(
        verified=manager.apply_fault(request.to_dict()),
        active=True,
        token=request.occurrence_id,
    ),
    deactivate=lambda request, result: manager.clear_fault(result.token),
)

faults = enable_fault_injection(
    model,
    optimizer,
    campaign=campaign,
    executors=(executor,),
)
```

Safety classes are:

- `safe_in_process`;
- `isolated_destructive`; and
- `cluster_destructive`.

Campaign enablement fails before training when no configured executor supports a
local fault or when an executor's safety ceiling is insufficient.
Unsupported effects are never silently skipped.
At execution time, the selected executor must return verified evidence.
An unverifiable activation is marked failed, deactivated when necessary, and is
not accepted as campaign ground truth.
Activation and deactivation evidence must be strictly JSON-serializable; invalid
evidence fails the action immediately and active effects are deactivated.
An executor that returns an active effect must provide a deactivation callback;
callback-free executors are valid only for one-shot `active=false` results.

The kit injects observable effects, not physical defects.
For example, an ECC event is represented as tensor corruption or resource loss;
a failed cable is represented as delay, message loss, or network partition.

## Restart and Retrigger Behavior

Each candidate occurrence is journaled before activation.
The default `retrigger: "once"` prevents the same occurrence from firing again
after rollback when a persistent state store is used.

```python
from lm_resiliency import JsonCampaignStateStore

faults = enable_fault_injection(
    model,
    optimizer,
    campaign=campaign,
    state_store=JsonCampaignStateStore(f"/manager-state/campaign-rank-{rank}.json"),
)
```

Supported policies are:

- `once`;
- `every_attempt`; and
- `max_occurrences`, with a positive `max_occurrences` field.

The state file must survive worker replacement for restart-stable behavior.
Use one JSON state file per rank, or provide a manager-backed
`CampaignStateStore` that serializes concurrent updates across workers.
For a distributed enablement, the initial manifest binding is persisted only
after every rank agrees on the manifest and training iteration.
The journal stores the canonical manifest identity. Reusing a campaign name with
changed triggers, targets, parameters, or metadata requires a new state file;
stale attempts are never applied to an edited manifest.

## Ground Truth and Evaluation

An occurrence ID identifies one incident candidate, such as
`sporadic-sdc@203`.
Correlated faults share the same occurrence ID and have individual injection IDs.

```python
from lm_resiliency import LocalizationResult

report = faults.evaluate(
    [
        LocalizationResult(
            occurrence_id="sporadic-sdc@203",
            detected=True,
            failed_ranks=(3,),
            kind="sdc",
            components=("transformer_block",),
            latency_ms=8.4,
        )
    ]
)
report.to_json("campaign-report.json")
```

Reports contain:

- the canonical manifest identity;
- the complete versioned manifest;
- actual iteration and attempt numbers;
- temporal behavior and safety class;
- executor and verification evidence;
- expected ranks, resources, and components;
- submitted neutral localization results; and
- attribution accuracy and latency.

An occurrence is `localized` only when injection succeeded, detection was
reported, and the reported rank and resource sets exactly match the expected
sets. Extra targets are attribution errors, not successful localization. If a
localization result also supplies `kind` or `components`, that evidence must
match the injected fault; omit optional evidence when the resiliency system does
not report it.
Example detection counts also require a report with the expected failure kind;
an unrelated report at the same iteration is retained as evidence but does not
count as detecting the injected occurrence.

Reports are rank-local.
A distributed campaign runner or training manager should collect reports from all
ranks and preserve the software, hardware, topology, model, and workload metadata.

## Current Boundaries

The schema and executor contract cover all canonical families above.
The repository does not directly terminate the current process, damage arbitrary
storage, alter a production network, or remove cluster resources.
Those effects require an explicitly configured isolated or cluster executor.

DeepSpeed clocks advance only when `global_steps` advances, so gradient
accumulation microbatches do not consume campaign iterations. PipelineEngine
uses the active `OptimizerStep` instruction-map entry, composing with an already
enabled SCOUT wrapper.

Permanent non-finite mutation of live parameter or optimizer state requires
checkpoint recovery because the affected values cannot be retired while
preserving subsequent updates. Use those state faults in isolated campaigns that
recover or terminate after localization.
