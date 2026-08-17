# Torchrun Native Resiliency

Status: implemented and validated

Last updated: 2026-08-17

## Summary

The integration uses fixed-size training with replacement:

- the scheduler starts torchrun agents on an allocated fleet;
- exactly `min_nodes` logical slots run trainers;
- excess agents remain parked as standbys;
- SCOUT localizes the faulty logical rank;
- GEMINI selects and flushes one recovery-verified checkpoint;
- the LM Resiliency manager publishes one immutable `RestartPlan`; and
- the torchrun rendezvous handler consumes that plan, preserves logical slots,
  restarts the healthy group, and admits only the selected standby.

Torchrun is a lifecycle mechanism, not the recovery decision-maker. It does not
re-evaluate SCOUT evidence, GEMINI trust, or placement policy.

## Why `--nnodes=min:max` Is Not a Standby Contract

Stock `--nnodes=min_nodes:max_nodes` permits any active rendezvous size in that
range. It does not mean "`min_nodes` trainers plus reserved standby agents."

For example, with `--nnodes=4:6`, late nodes are normally scale-up candidates.
The simplified LM Resiliency handler instead keeps exactly four logical slots:

```text
generation 0:  node-a node-b node-c node-d
standby:       node-e node-f

failure:       node-d
generation 1:  node-a node-b node-c node-e
standby:       node-f
```

`node-e` inherits `node-d`'s logical slot and global-rank range. World size,
local worker count, and framework topology do not change.

## Ownership

| Concern | Owner |
|---|---|
| Fault detection and localization | SCOUT |
| Checkpoint capture, trust, and recovery step | GEMINI |
| Quarantine, replacement choice, and final plan | LM Resiliency manager |
| Standby parking and stable slot admission | torchrun rendezvous handler |
| Worker-group stop and relaunch | stock torchrun elastic agent |
| Model, optimizer, RNG, scheduler, and input state | framework integration |
| Host allocation and physical repair | scheduler/infrastructure |

The manager publishes only after it has one safe decision. The torchrun-facing
contract is therefore one canonical `RestartPlan`, not a second evidence or
consensus protocol.

## Recovery Plan

The plan contains the information torchrun needs:

- run, plan, intent, and generation identities;
- the exact node-to-slot assignment for the successor generation;
- quarantined node IDs;
- recovery mode and checkpoint source;
- checkpoint step and optional durable checkpoint ID;
- expected world size and topology digest; and
- an exclusive restart deadline.

The runtime stores one create-once plan per successor generation. Publication
advances the current generation by compare-and-set, so conflicting plans fail
closed.

For GEMINI recovery the plan carries:

```text
recovery_mode = recovery_verified
checkpoint_source = gemini
checkpoint_step = <verified optimizer step>
checkpoint_id = null
```

The shared GEMINI folder and stable checkpoint run ID remain deployment
configuration. They are not duplicated in every restart plan.

## Rendezvous Lifecycle

### Initial generation

Each agent reads `/etc/machine-id`, validates the standard 32-hexadecimal
machine identity, and registers a domain-separated SHA-256 identifier in the
rendezvous store. The raw machine ID is not published. The first `min_nodes`
unique registrations are committed once as generation zero and logical slot
order. Later registrations remain standbys. Every agent then heartbeats under
its committed machine identity.

- assigned nodes publish readiness and form the worker group;
- unassigned nodes remain blocked in `next_rendezvous()`; and
- parked standbys are hidden from `num_nodes_waiting()`.

Two live agents resolving to the same machine identity fail closed. This
prevents two torchrun agents on one physical node from claiming separate
failure domains. The validation campaign intentionally simulates multiple
nodes on one host by setting `LM_RESILIENCY_MACHINE_ID_PATH` to a different
synthetic machine-ID file for each agent.

### Replacement

After the manager publishes generation `N + 1`:

1. healthy active agents observe one live plan-selected replacement through
   `num_nodes_waiting()`;
2. stock torchrun performs its normal full-worker-group restart;
3. the selected standby is admitted at the failed node's logical slot;
4. every admitted node receives the same generation and world size;
5. the handler writes a node-local canonical `RestartContext`; and
6. workers enable LM Resiliency with the plan's recovery mode and restore the
   selected GEMINI or durable checkpoint.

Random late agents and unselected standbys never create a restart signal.

### Completion

The manager closes the run after training completes. Parked agents exit cleanly
without starting trainers.

## Restart Context

The handler writes one owner-only JSON file per node. The file contains:

- run, plan, and generation identity;
- logical node slot and global-rank range;
- recovery mode and checkpoint selection;
- world size and topology digest; and
- restart deadline.

Publication uses same-directory atomic replacement. Workers reject malformed,
noncanonical, oversized, nonregular, or non-owner-only files.
The handler creates a missing parent directory with owner-only permissions.
An existing directory with different ownership or group/other access is
rejected rather than modified.

The framework consumes the context before training resumes. GEMINI then
restores model, optimizer, caller-owned state, and RNG from the selected step.

## Configuration

The integration is registered through the supported `torchrun.handlers`
entry-point group as rendezvous backend `lm_resiliency`.

Required per-agent rendezvous configuration:

| Key | Meaning |
|---|---|
| `lm_resiliency_restart_context_path` | Absolute node-local context path |

Optional settings:

| Key | Default | Meaning |
|---|---:|---|
| `lm_resiliency_join_timeout_ms` | `300000` | Bounded formation/bootstrap window |
| `lm_resiliency_poll_interval_ms` | `250` | Store polling interval |
| `lm_resiliency_heartbeat_timeout_ms` | `10000` | Standby/agent liveness window |
| `lm_resiliency_worker_config` | unset | Absolute path to the strict worker policy; enables automatic framework instrumentation |

The values are supplied through `--rdzv-conf`. The `lm_resiliency_` namespace
prevents collisions with c10d and future torchrun rendezvous options.

Physical node identity and initial admission are automatic. Production agents
read `/etc/machine-id`. `LM_RESILIENCY_MACHINE_ID_PATH` can select another
absolute machine-ID file for containerized deployments or test harnesses.

Example agent command for a user module with no LM Resiliency imports:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=1 \
  --max-restarts=4 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint="$STORE_ENDPOINT" \
  --rdzv-id="$RUN_ID" \
  --rdzv-conf="store_type=tcp,is_host=false,read_timeout=120,\
lm_resiliency_restart_context_path=$CONTEXT_PATH,\
lm_resiliency_worker_config=/absolute/path/worker.toml" \
  --module examples.torchrun_resiliency.smoke \
  --artifact-dir /tmp/lm-resiliency-user-loop
```

For a file-backed single-host run, set `store_type=file` and use a private
local rendezvous path. Multi-host validation uses an orchestrator-owned
`TCPStore`.

When `lm_resiliency_worker_config` is set, the rendezvous plugin installs framework-import
monitoring before Python executes the user module. Native PyTorch is tentative;
an import of TorchTitan, Megatron Core, or DeepSpeed selects the corresponding
more specific adapter before attachment. Multiple higher-level supported
frameworks fail closed. The user module still owns the ordinary framework
training loop.

Torchrun supplies `LOCAL_WORLD_SIZE` to every worker from
`--nproc-per-node`. Replacement workers compare that value with the
manager-selected rank range before framework initialization, so the same
topology invariant is preserved without duplicating worker width in
`--rdzv-conf`.

Adapters select a framework, not a parallelism strategy. Each adapter passes the
same objects accepted by that framework's existing `enable_resiliency()` API,
and the existing integration discovers supported DDP/FSDP/HSDP/TP/PP/EP/ZeRO
topology exactly as it does for explicit activation. Framework-object discovery
is intentionally fail-closed; stacks with custom trainers, multiple optimizers,
sharded optimizers, or caller-owned loop state provide a custom adapter rather
than relying on process-wide object scanning.

The worker policy is schema-versioned and strict even for disabled features.
Custom stacks set `adapter = "package.module:factory"` in that policy; built-in
frameworks omit `adapter` and use import inference.
GEMINI topology settings remain deployment-owned: for example,
`replication_jump` must divide the checkpoint world according to the pairing
contract documented in [GEMINI](gemini.md#peer-replication).

## Manager Interface

The manager must provide two operations:

```python
plans.publish(restart_plan)
plans.close_run()
```

`publish()` is called only after SCOUT localization, GEMINI checkpoint
selection/flush, quarantine, and standby placement succeed. `close_run()` wakes
parked standbys after successful job completion.

The implementation currently exposes this store internally as
`SimpleRecoveryPlanStore`; the production-loop example demonstrates the full
manager flow.

## Safety Properties

- active world size remains fixed;
- generation-zero placement is immutable after the first `min_nodes`
  registrations are committed;
- duplicate physical machine identities fail closed;
- logical slots and global-rank ranges remain stable;
- a plan is immutable and generation-fenced;
- only manager-selected replacements become visible to healthy agents;
- an expired plan cannot admit a worker;
- unselected and quarantined nodes are absent from the successor assignment;
- restart context is written before replacement workers train;
- malformed plan/context state fails closed;
- heartbeat loss suppresses restart signaling; and
- transient control-store contention does not create a false restart edge.

## Validation

The minimal zero-import user loop is
[`examples/torchrun_resiliency/smoke.py`](../examples/torchrun_resiliency/smoke.py).
Its
[`smoke.toml`](../examples/torchrun_resiliency/policies/smoke.toml)
policy disables GEMINI and SCOUT so a CPU run can validate only pre-module
adapter activation. The framework production loops use
[`resiliency.toml`](../examples/production_loops/policies/resiliency.toml)
to exercise both mechanisms on the documented eight-GPU topology.

The executable validation campaign remains
[`examples/torchrun_resiliency/pressure.py`](../examples/torchrun_resiliency/pressure.py).
It includes manager, fault-injection, baseline-comparison, and two-host
orchestration logic that is not part of a normal user training module.
It performs:

1. an uninterrupted eight-rank baseline;
2. eight active one-GPU agents plus eight parked one-GPU standbys;
3. sixteen same-node restart-only incidents;
4. eight replay-only SDC incidents targeting distinct logical ranks;
5. exact SCOUT localization and recovery-verified checkpoint selection;
6. manager plan publication for every incident;
7. stable-slot replacement until every initial GPU-node is quarantined; and
8. bounded numerical comparison against the uninterrupted baseline.

The `--fault-campaign-dir` bundle owns `campaign.json`, restart-stable state,
artifacts, logs, checkpoints, and the final summary. It is a validation-only
surface and is not required by normal torchrun training.

See [Validation](validation.md#torchrun-standby-replacement) and
[Examples](../examples/README.md#production-loops) for commands and numerical
acceptance bounds.

## Current Boundaries

- fixed-size replacement only; no scale-up or scale-down;
- stable topology and local worker count across generations;
- standbys must already be allocated and running;
- one torchrun agent per unique `/etc/machine-id` is required outside the
  synthetic validation harness;
- the manager, not torchrun, owns evidence evaluation and quarantine policy;
- one shared control-store endpoint is required;
- built-in injected adapters support exact manager-selected GEMINI recovery;
  durable-source replacement requires a custom adapter with a framework loader;
- replacement does not repair hardware or allocate new hosts; and
- the production campaign uses DDP and tiny deterministic workloads, not
  publication-scale convergence.
