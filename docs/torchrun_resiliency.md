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
- checkpoint step, checkpoint manifest ID, and optional durable checkpoint ID;
- expected world size and topology digest; and
- an exclusive restart deadline.

The runtime stores one create-once plan per successor generation. Publication
advances the current generation by compare-and-set, so conflicting plans fail
closed. The coordinator converts the manager's absolute deadline into a
renewing store lease containing a sequence and remaining duration. Agents
require locally observed lease progress and translate the remaining duration
to their own monotonic clocks; they never compare wall-clock timestamps from
different hosts. Lease freshness remains bounded by the heartbeat cadence, but
the restart context carries the full manager window so worker startup is not
limited to one heartbeat timeout.

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

1. the coordinator renews the generation lease until the manager deadline;
2. healthy active agents observe one live plan-selected replacement through
   `num_nodes_waiting()`;
3. stock torchrun performs its normal full-worker-group restart;
4. the selected standby is admitted at the failed node's logical slot;
5. every admitted node receives the same generation and world size;
6. the handler writes a node-local canonical `RestartContext`; and
7. workers enable LM Resiliency with the plan's recovery mode and restore the
   selected GEMINI or durable checkpoint.

Random late agents, unselected standbys, and plans without a progressing
manager lease never create a restart signal.

### Completion

The manager closes the run after training completes. Parked agents exit cleanly
without starting trainers.

## Restart Context

The handler writes one owner-only JSON file per node. The file contains:

- run, plan, and generation identity;
- logical node slot and global-rank range;
- recovery mode and checkpoint selection;
- world size and topology digest; and
- a node-local monotonic lease deadline.

Publication uses same-directory atomic replacement. Workers reject malformed,
noncanonical, oversized, nonregular, non-owner-only, or locally expired files.
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
  --module your_training.module \
  [application arguments...]
```

For a file-backed single-host run, set `store_type=file` and use a private
local rendezvous path. Multi-host validation uses an orchestrator-owned
`TCPStore`.

When `lm_resiliency_worker_config` is set, the rendezvous plugin installs framework-import
monitoring before Python executes the user module. Native PyTorch is tentative;
an import of TorchTitan, Megatron Core, or DeepSpeed selects the corresponding
more specific adapter before attachment. Multiple higher-level supported
frameworks fail closed. After the default process group initializes, every rank
agrees on the inferred framework before a built-in adapter can enter GEMINI or
SCOUT collectives. Missing or conflicting framework evidence fails all ranks
before attachment. The user module still owns the ordinary framework training
loop.
If the default process group uses NCCL, framework agreement runs on a temporary
Gloo group and destroys that group immediately afterward. Agreement therefore
does not require user code to bind `LOCAL_RANK` to a CUDA device before
`init_process_group()`.

Torchrun supplies `LOCAL_WORLD_SIZE` to every worker from
`--nproc-per-node`. Replacement workers compare that value with the
manager-selected rank range before framework initialization, so the same
topology invariant is preserved without duplicating worker width in
`--rdzv-conf`.
Bootstrap exposes the manager-selected resume position as
`LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP`; generation-zero workers receive `0`.
Applications can use it for deterministic sampler or token positioning without
importing LM Resiliency.

Adapters select a framework, not a parallelism strategy. Each adapter passes the
same objects accepted by that framework's existing `enable_resiliency()` API,
and the existing integration discovers supported DDP/FSDP/HSDP/TP/PP/EP/ZeRO
topology exactly as it does for explicit activation. Framework-object discovery
is intentionally fail-closed; stacks with custom trainers, multiple optimizers,
sharded optimizers, or caller-owned loop state provide a custom adapter rather
than relying on process-wide object scanning.

Distributed native PyTorch attachment requires a recognized DDP or FSDP
construction boundary that all participating ranks cross. At the first
recognized distributed forward, every rank collectively proves that it has
exactly one valid model/optimizer pair before any rank starts LM Resiliency
attachment collectives. Rank-local warmup forwards cannot initiate those
collectives.

After attachment, built-in adapters bind cleanup to the framework's normal
distributed teardown boundary. The resiliency handle closes before process
groups are destroyed, so asynchronous checkpoint and replay work completes
before the application reports a clean exit.
On generation zero, DeepSpeed may call `engine.load_checkpoint()` before its
first resiliency-managed training step; a successful load seeds the resiliency
counter from `engine.global_steps`. Successor generations reject framework
loads after the manager-selected GEMINI state has been restored.

The worker policy is schema-versioned and strict even for disabled features.
Every assigned node publishes a digest of the exact policy bytes, rendezvous
requires one cohort-wide value, and worker bootstrap revalidates the digest
before parsing or installing an adapter.
`checkpoint.disk_flush_interval` must be nonnegative; zero disables periodic
disk persistence.
Custom stacks set `adapter = "package.module:factory"` in that policy; built-in
frameworks omit `adapter` and use import inference.
GEMINI topology settings remain deployment-owned: for example,
`replication_jump` must divide the checkpoint world according to the pairing
contract documented in [GEMINI](gemini.md#peer-replication).

## Manager Interface

The public `TorchrunRecoveryCoordinator` wraps the canonical store protocol:

```python
from lm_resiliency.integrations.torchrun import (
    TorchrunRecoveryCoordinator,
    TorchrunRecoveryRequest,
)

coordinator = TorchrunRecoveryCoordinator(store, run_id=run_id)
initial = coordinator.initial_placement(
    active_node_count=8,
    allocated_node_count=16,
)
assert initial is not None

successor = coordinator.publish_successor(
    generation=0,
    active_node_ids=initial.active_node_ids,
    quarantined_node_ids=(),
    request=TorchrunRecoveryRequest(...),
    local_world_size=1,
    replacement=(faulty_slot, selected_standby),
)

coordinator.close()
```

`publish_successor()` is called only after SCOUT localization, GEMINI checkpoint
selection/flush, quarantine, and standby placement succeed. Omitting
`replacement` restarts the same assignments. Supplying `(logical_slot,
standby_node_id)` replaces exactly one physical node while preserving its
logical rank range. The coordinator must remain alive through successor
admission so it can renew the store lease. Managers call `check_health()` while
waiting for admission. Transient store errors are retried, while persistent
renewal failure is latched and raised by health checks and later coordinator
operations. `close()` stops renewal and wakes parked standbys after successful
completion.

`TorchrunLaunchConfig` constructs the framework-neutral torchrun argument list.
It deliberately does not launch subprocesses, select GPUs, invoke SSH, or
implement scheduler policy.

## Safety Properties

- active world size remains fixed;
- generation-zero placement is immutable after the first `min_nodes`
  registrations are committed;
- duplicate physical machine identities fail closed;
- logical slots and global-rank ranges remain stable;
- a plan is immutable and generation-fenced;
- plan eligibility is proven through a manager-renewed store lease and
  host-local monotonic time rather than cross-host wall clocks;
- only manager-selected replacements become visible to healthy agents;
- an expired plan cannot admit a worker;
- unselected and quarantined nodes are absent from the successor assignment;
- restart context is written before replacement workers train;
- worker generation, rank ranges, world size, and local lease deadline must match
  the rendezvous decision before framework initialization;
- malformed plan/context state fails closed;
- heartbeat loss suppresses restart signaling; and
- transient control-store contention does not create a false restart edge.

## Validation

Source-owned tests cover handler construction, generation-zero admission,
duplicate-identity rejection, standby parking, same-node successor plans,
slot-preserving replacement, context publication, worker-width fencing,
zero-import bootstrap, framework inference, exact recovery-step verification,
and bounded shutdown.

Distributed validation used one torchrun agent per A100 and exercised eight
active agents with eight standbys. The manager published sixteen same-node
successor generations and eight SCOUT-localized replacements while preserving
the fixed eight-rank world. Every successor rank restored the selected
recovery-verified GEMINI checkpoint before training resumed. Native PyTorch,
DeepSpeed, Megatron Core, and TorchTitan also completed a focused
restart-and-replacement campaign through their framework-owned training
boundaries.

## Current Boundaries

- fixed-size replacement only; no scale-up or scale-down;
- stable topology and local worker count across generations;
- standbys must already be allocated and running;
- one torchrun agent per unique `/etc/machine-id` is required outside the
  synthetic validation harness;
- the manager, not torchrun, owns evidence evaluation and quarantine policy;
- one shared control-store endpoint is required;
- built-in injected adapters support exact manager-selected GEMINI recovery;
  the selected step and checkpoint-topology digest are validated before load;
  durable-source replacement requires a custom adapter with a framework loader;
- replacement does not repair hardware or allocate new hosts; and
- the production campaign uses DDP and tiny deterministic workloads, not
  publication-scale convergence.
