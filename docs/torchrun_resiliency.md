# Torchrun Native Resiliency Integration

Status: draft for discussion

Last updated: 2026-08-14

## Decision Summary

The recommended first design is **fixed-size training with replacement**, not
general elastic training:

- allocate a fleet of `max_nodes` homogeneous hosts;
- run the training job on exactly `min_nodes` active hosts;
- keep `max_nodes - min_nodes` hosts as agent-only standbys;
- preserve a stable logical node slot and rank range across restart;
- let SCOUT report fault evidence and GEMINI report recoverable checkpoint
  state;
- let a torchrun-owned coordinator make the final quarantine, placement, and
  restart decision; and
- pass one fenced `RestartContext` to every relaunched worker so
  `lm-resiliency` and the framework recover the same checkpoint generation.

The existing meaning of `torchrun --nnodes=min_nodes:max_nodes` is not sufficient
for this design. It permits the active worker-group membership and world size to
change. A late node is admitted through re-rendezvous and all workers restart.
Ranks are not stable across those restarts. Treating the extra nodes as standbys
therefore requires an explicit admission policy and stable logical slots rather
than relying on the range alone.

The design should be implemented in two stages:

1. an out-of-tree reference launcher using a custom elastic agent, control
   coordinator, and rendezvous handler; then
2. an upstreamable torchrun resiliency-policy interface that removes the need
   to subclass internal agent methods.

## Scope

### Goals

- Replace a localized bad node with a healthy standby without changing the
  training topology.
- Prevent quarantined nodes from rejoining later rendezvous generations.
- Select recovery state conservatively from SCOUT and GEMINI evidence.
- Preserve the complete framework state for one coherent optimizer step.
- Keep TorchTitan, native PyTorch, Megatron Core, and DeepSpeed integrations
  independent of torchrun process-management details.
- Fence stale fault reports and restart commands from earlier generations.
- Fail closed when there are too few healthy nodes or no complete trusted
  checkpoint.

### Initial assumptions

- `min_nodes` is the exact active training fleet size.
- `max_nodes - min_nodes` is the maximum standby capacity.
- Every admitted node has the same local worker count and compatible hardware,
  software, model configuration, and framework version.
- A replacement node inherits the failed node's logical slot and global-rank
  range.
- Version 1 does not change DP, TP, PP, CP, EP, expert-TP, FSDP/HSDP, or ZeRO
  degrees during recovery.
- Standby agents are running and reachable, but standby trainer processes are
  not.
- The control store is strongly consistent enough to publish a single restart
  plan by compare-and-swap.

### Non-goals for version 1

- Elastic scale-up or scale-down of the active training world.
- Recovery with a different model-parallel layout or global batch size.
- Partial restart of one rank while the rest of the worker group continues.
- Reusing a node that SCOUT could not safely clear.
- Physical hardware repair.
- Turning inconclusive peer evidence into confident node attribution.

## Existing Behavior and Missing Hooks

Stock torchrun already provides useful foundations:

- an agent process on every participating node;
- local worker lifecycle and failure monitoring;
- worker-group restart after process failure;
- dynamic rendezvous with `min_nodes` and `max_nodes`;
- custom rendezvous handlers through the `torchrun.handlers` entry-point group;
- restart-count and run-ID environment variables; and
- full worker-group restart on failure or membership change.

It does not currently provide the complete contract required here:

1. `min_nodes:max_nodes` describes elastic active membership, not
   `min_nodes` active nodes plus replacement-only standbys.
2. A healthy late node is considered a scale-up candidate, which triggers a
   worker-group restart and may increase `WORLD_SIZE`.
3. Rank assignment is not stable across re-rendezvous.
4. Worker failure handling does not consume SCOUT fault localization or
   checkpoint trust.
5. Rendezvous admission has no standard quarantine database or logical-slot
   inheritance contract.
6. The worker-to-agent control plane has no stable fault, checkpoint, or
   graceful-restart API.
7. A custom rendezvous handler alone cannot force all healthy agents to stop
   their workers for a SCOUT-requested replacement.

PyTorch 2.13 includes an experimental worker control-plane server, but it is a
generic handler server and not a stable restart or quarantine protocol. The
reference integration should not make its safety contract depend on that
experimental surface.

The integration must also avoid depending on `--max-restarts` as its only
replacement budget. Failure retries and membership changes have had different
accounting behavior across torchrun versions. The coordinator should own an
explicit `max_replacement_generations` policy.

## Three-Layer Architecture

```text
                    control store / coordinator
              restart plans, slot leases, quarantine
                              |
                              v
+------------------------------------------------------------------+
| Layer 1: torchrun                                                 |
|                                                                  |
| ResilientElasticAgent  SlotAwareRendezvous  RestartCoordinator   |
| - owns active/standby state                                      |
| - maps stable node IDs to logical slots                          |
| - commits one fenced RestartPlan                                 |
| - stops all active workers                                       |
| - excludes quarantined nodes                                     |
| - launches exactly min_nodes slots                               |
+-----------------------------+------------------------------------+
                              |
           local control socket| restart-context file/environment
                              v
+------------------------------------------------------------------+
| Layer 2: lm-resiliency                                           |
|                                                                  |
| TorchrunOrchestrationClient  SCOUT  GEMINI  manager_api          |
| - reports normalized fault evidence                              |
| - proposes checkpoint trust and locally available state          |
| - flushes and transfers eligible checkpoint shards               |
| - validates and consumes the committed RestartContext            |
| - performs rank-consistent checkpoint selection after relaunch   |
+-----------------------------+------------------------------------+
                              |
             framework-neutral| recovery request and adapter
                              v
+------------------------------------------------------------------+
| Layer 3: TorchTitan or another torchrun framework                 |
|                                                                  |
| FrameworkAdapter / Trainer                                       |
| - owns model, optimizer, scheduler, RNG, sampler/data position    |
| - exposes safe optimizer boundaries                              |
| - loads framework durable state when GEMINI is unavailable        |
| - resumes at the step selected by lm-resiliency                  |
+------------------------------------------------------------------+
```

### Ownership boundary

| Concern | Owner |
|---|---|
| Detection, localization, evidence confidence | SCOUT |
| Checkpoint capture, trust, inventory, transfer | GEMINI |
| Final node quarantine and replacement policy | torchrun layer |
| Active/standby admission and logical-slot assignment | torchrun layer |
| Final restart generation and worker termination | torchrun layer |
| Model and optimizer state semantics | framework adapter |
| Physical repair or scheduler allocation | infrastructure manager |

`lm-resiliency` may recommend a replacement scope, but it must not directly
evict a physical host. A rank-local report is evidence, not a cluster-wide
decision.

## Identity Model

Ranks cannot be used as durable resource identities. Every event and plan must
carry both the current rank mapping and stable infrastructure identities.

```python
class AgentIdentity(TypedDict):
    run_id: str
    node_id: str              # Stable scheduler or infrastructure identity.
    agent_id: str             # Unique torchrun agent incarnation.
    hostname: str
    local_world_size: int
    resource_ids: list[str]   # GPU UUIDs, NICs, HCAs, or deployment IDs.
    environment_digest: str   # Software, configuration, and capability identity.


class WorkerIdentity(TypedDict):
    run_id: str
    generation: int
    node_id: str
    agent_id: str
    logical_node_slot: int    # Stable for the lifetime of the training job.
    global_rank: int
    local_rank: int
    local_world_size: int
    hostname: str
    gpu_uuid: str | None
    topology_digest: str
```

The coordinator records an immutable `RankAssignment` for each generation:

```python
@dataclass(frozen=True)
class RankAssignment:
    run_id: str
    generation: int
    active_nodes: int
    local_world_size: int
    slot_to_node_id: dict[int, str]
    slot_to_rank_range: dict[int, tuple[int, int]]
    topology_digest: str
```

When node `host-a` in slot `3` is replaced by `host-spare-1`, the new node
inherits slot `3` and the same rank range. This keeps GEMINI shard ownership and
framework parallel coordinates stable.

`GROUP_RANK`, hostname, PID, and global rank alone are not valid quarantine
keys. A quarantine record uses `node_id` and may also contain GPU, NIC, HCA, or
other resource IDs.

`node_id` must come from a trusted scheduler, cloud instance identity, or
deployment inventory. A worker-provided hostname is diagnostic metadata and
must not be allowed to choose its own quarantine identity. If no stable node
identity is available, replacement mode should refuse to start.

The topology digest covers at least the active world size, local worker count,
framework parallel dimensions, logical rank-to-parallel-coordinate mapping,
checkpoint schema, and model configuration needed to interpret rank-local
state. It is separate from `environment_digest`, which describes whether a node
is eligible to run the same software and hardware workload.

## API: lm-resiliency to torchrun

The current `OrchestrationHooks` callbacks are the starting point. The torchrun
integration wraps them in a versioned, incident-correlated envelope and sends
them to the local agent over a Unix-domain socket.

### Fault event

```python
class HardwareFaultReport(TypedDict):
    kind: Literal["hardware"]
    resource_kind: Literal["gpu", "node", "nic", "hca", "link"]
    resource_id: str
    metric: str
    value: float
    severity: Literal["fatal"]
    message: str


class FaultEvent(TypedDict):
    schema_version: int
    event_id: str
    incident_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    optimizer_step: int
    report: SCOUTFaultReport | HardwareFaultReport
```

Requirements:

- `event_id` is idempotent.
- `incident_id` correlates the fault with recovery and checkpoint events.
- The coordinator rejects a stale `generation`.
- The original `SCOUTFaultReport` is preserved without upgrading its
  attribution. A direct `HealthEvent` is normalized to
  `HardwareFaultReport` without upgrading its resource granularity.
- The torchrun layer resolves reported ranks through that generation's
  `RankAssignment`.
- The orchestration dispatcher allocates `incident_id` once and uses it for
  both the recovery and fault callbacks. The client must not infer correlation
  from callback timing.

### Recovery proposal

The existing `RecoveryDecision` remains a rank-local, noncollective result. At
the torchrun boundary it is explicitly called a proposal:

```python
class RecoveryProposalEvent(TypedDict):
    schema_version: int
    event_id: str
    incident_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    decision: RecoveryDecision
```

The coordinator combines proposals with a conservative lattice:

```text
RECOVERY_VERIFIED dominates LATEST_GEMINI
unavailable required shard dominates locally available state
missing or stale evidence blocks promotion to a less conservative mode
```

A worker must never interpret its local `checkpoint_step` as the final global
step. The restarted job still calls GEMINI's collective `find_latest()` before
loading.

### Checkpoint inventory

`RecoveryDecision` says what one worker recommends. Replacement also needs to
know where every logical-rank shard can be obtained.

```python
class CheckpointCopy(TypedDict):
    owner_global_rank: int
    holder_node_id: str
    holder_kind: Literal["owner", "peer", "durable"]
    storage_kind: Literal["memory", "node_local", "shared", "remote"]
    location_token: str
    complete: bool
    checksums_available: bool


class CheckpointInventoryEvent(TypedDict):
    schema_version: int
    event_id: str
    run_id: str
    generation: int
    reporter: WorkerIdentity
    step: int
    trust: Literal["latest", "candidate", "recovery_verified"]
    topology_digest: str
    copies: list[CheckpointCopy]
```

Only completed checkpoint slots may appear with `complete=True`. A
`CANDIDATE` inventory can be retained for diagnosis but is never selected for
conservative recovery.

`location_token` is an opaque control-plane reference, not an unchecked path
that another worker is allowed to open.

### Local client protocol

```python
class ResiliencyEventSink(Protocol):
    def publish_fault(self, event: FaultEvent) -> None: ...
    def publish_recovery(self, event: RecoveryProposalEvent) -> None: ...
    def publish_checkpoint(self, event: CheckpointInventoryEvent) -> None: ...
    def acknowledge_restart(self, ack: RestartAck) -> None: ...
```

The callbacks must be bounded and must not use the training process group.
Delivery failure is visible and causes the coordinator to use conservative
recovery; it must not be silently treated as a healthy report.

## API: torchrun to lm-resiliency

Restart is a two-phase protocol:

1. a `RestartIntent` quiesces accessible workers and gathers final checkpoint
   preparation results without advancing the generation; then
2. an immutable `RestartPlan` selects the actual checkpoint and next placement
   from those results.

This prevents a failed flush or transfer from invalidating a plan that already
promised `LATEST_GEMINI`.

### Restart intent

```python
class RestartIntent(TypedDict):
    schema_version: int
    intent_id: str
    run_id: str
    generation: int
    incident_ids: list[str]
    reason_code: str
    minimum_recovery_mode: Literal["latest", "recovery_verified"]
    suspected_node_ids: list[str]
    prepare_deadline_unix_ms: int
```

The intent is a single-writer, compare-and-swap record for the current
generation. It does not authorize a new worker group. Agents receiving it
quiesce or terminate local training according to the incident, prepare eligible
checkpoint state, and wait for either a committed plan or an explicit abort.

An abort is allowed only before a plan is committed and only when resuming the
same generation is safe. An SDC, inaccessible node, corrupted checkpoint, or
lost worker group cannot be aborted back to normal training.

### Committed restart plan

Only the coordinator writes a `RestartPlan`. It is published atomically after
checkpoint preparation finishes or times out and advances the monotonically
increasing generation.

```python
class SlotAssignment(TypedDict):
    logical_node_slot: int
    node_id: str
    first_global_rank: int
    local_world_size: int


class RestartPlan(TypedDict):
    schema_version: int
    plan_id: str
    intent_id: str
    run_id: str
    from_generation: int
    to_generation: int
    incident_ids: list[str]
    reason_code: str
    recovery_mode: Literal["latest", "recovery_verified"]
    checkpoint_source: Literal["gemini", "durable"]
    checkpoint_step: int
    checkpoint_id: str | None
    checkpoint_manifest_id: str
    slot_assignments: list[SlotAssignment]
    quarantined_node_ids: list[str]
    expected_world_size: int
    topology_digest: str
    stop_deadline_unix_ms: int
```

Before commit, the coordinator validates:

- exactly `min_nodes` active logical slots are assigned;
- every assigned node is healthy, compatible, and not quarantined;
- each logical slot is assigned once;
- the topology digest matches the prior generation;
- the checkpoint is complete for every required logical rank;
- the checkpoint trust satisfies the selected recovery mode;
- the plan never selects `CANDIDATE`;
- the target step is coherent across all selected shards; and
- the plan generation is the successor of the current committed generation.

### Restart context passed to workers

The agent derives a rank-local context from the committed plan and writes it to
an atomically replaced file. The worker receives only the file path:

```text
LM_RESILIENCY_RESTART_CONTEXT=/run/torchrun/<run-id>/restart-context.json
```

```python
class RestartContext(TypedDict):
    schema_version: int
    plan_id: str
    run_id: str
    generation: int
    node_id: str
    logical_node_slot: int
    global_rank: int
    local_rank: int
    expected_world_size: int
    topology_digest: str
    recovery_mode: Literal["latest", "recovery_verified"]
    checkpoint_source: Literal["gemini", "durable"]
    checkpoint_step: int
    checkpoint_id: str | None
    checkpoint_manifest_id: str
    reason_code: str
```

`lm-resiliency` rejects startup if the context conflicts with torchrun's
`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `TORCHELASTIC_RUN_ID`, or the framework
topology.

The context must not contain a newer, less trusted local checkpoint merely
because it exists on the replacement host.

The committed `checkpoint_step` is a pin, not a hint. On a managed restart,
GEMINI must load exactly that step from `checkpoint_manifest_id` on every rank
or fail. Its existing collective `find_latest()` remains useful for validating
rank-wide availability during unmanaged recovery, but it must not silently
substitute a different step after the coordinator commits a plan. This requires
an additive exact-step checkpoint-manager surface used by
`enable_resiliency(..., restart_context=restart)`:

```python
checkpoint_manager.load_exact(
    step=restart.checkpoint_step,
    manifest_id=restart.checkpoint_manifest_id,
    recovery_mode=restart.recovery_mode,
)
```

This is an implementation boundary, not an additional call required in the
training script. A durable source is similarly pinned by `checkpoint_id` and
step.

### Prepare-for-restart command

For an accessible worker group, the agent derives a local command from the
intent:

```python
class PrepareRestart(TypedDict):
    intent_id: str
    generation: int
    minimum_recovery_mode: Literal["latest", "recovery_verified"]
    destination_token: str | None
    deadline_unix_ms: int
```

The `lm-resiliency` client:

1. stops accepting new unsafe checkpoint candidates;
2. waits boundedly for the eligible GEMINI capture and replication state;
3. flushes completed owner and peer slots;
4. transfers or mirrors only checkpoint state allowed by the plan; and
5. writes a `RestartAck`.

```python
class RestartAck(TypedDict):
    intent_id: str
    node_id: str
    generation: int
    flushed_step: int
    inventory_event_ids: list[str]
    transferred_owner_ranks: list[int]
    transferred_peer_ranks: list[int]
    success: bool
    reason: str
```

The socket listener only validates and stages the command. It must not call
CUDA, framework, or checkpoint operations from a background listener thread.
The agent then requests a framework safe point or sends the configured
catchable termination signal. GEMINI's bounded main-thread/signal path performs
the flush and writes the acknowledgement.

The agent waits only until the intent deadline and reports missing
acknowledgements. Missing or failed preparation cannot promote
`LATEST_GEMINI`; the coordinator must commit a plan using a complete
recovery-verified source or fail the restart.

For a crashed or unreachable node, no preparation is assumed.

## API: lm-resiliency to the Framework

The framework receives a platform-neutral recovery request. It does not receive
quarantine or placement policy.

```python
@dataclass(frozen=True)
class FrameworkRecoveryRequest:
    run_id: str
    generation: int
    recovery_mode: RecoveryMode
    checkpoint_source: Literal["gemini", "durable"]
    checkpoint_step: int
    checkpoint_id: str | None
    expected_world_size: int
    topology_digest: str


class FrameworkRecoveryAdapter(Protocol):
    def validate_restart(self, request: FrameworkRecoveryRequest) -> None: ...
    def load_durable(self, request: FrameworkRecoveryRequest) -> int | None: ...
```

The proposed activation API is additive:

```python
from lm_resiliency import enable_resiliency
from lm_resiliency.integrations.torchrun import (
    RestartContext,
    TorchrunOrchestrationClient,
)

restart = RestartContext.from_env()
control = TorchrunOrchestrationClient.connect(restart)

resiliency = enable_resiliency(
    trainer,
    orchestration=control.hooks(),
    restart_context=restart,
)
trainer.train()
```

`restart_context=None` preserves the current behavior. Existing
`load_fallback: Callable[[], int | None]` remains supported. A new
context-aware durable loader can be added without changing the old callback.

For TorchTitan, the existing adapter already captures:

- model state;
- optimizer state;
- LR scheduler state;
- trainer state;
- dataloader position; and
- the recovered training step.

The adapter must additionally validate that the relaunched
`ParallelDims/world_size` produces the committed topology digest before any
checkpoint is loaded.

## Torchrun Agent and Rendezvous Design

### Node states

```text
STANDBY -> ADMITTED -> ACTIVE -> DRAINING -> QUARANTINED
     \          \         \          \-> FAILED
      \----------> REJECTED
```

- `STANDBY`: agent is registered; no trainer process is running.
- `ADMITTED`: coordinator assigned a logical slot for the next generation.
- `ACTIVE`: local workers are running for the committed generation.
- `DRAINING`: restart is committed; bounded checkpoint preparation is running.
- `QUARANTINED`: node cannot enter another generation for this run.
- `FAILED`: agent or node is unreachable; its slot may be reassigned.

### Job states

```text
RUNNING
  -> DECIDING
  -> PREPARING
  -> COMMITTING
  -> STOPPING
  -> RENDEZVOUS
  -> RECOVERING
  -> RUNNING
```

Any state may transition to `FAILED` when there are too few eligible nodes, no
complete trusted checkpoint, a conflicting plan, or an unrecoverable topology
mismatch.

### Coordinator

The `RestartCoordinator` is the single writer for:

- current generation;
- current restart intent;
- active slot leases;
- standby eligibility;
- quarantine records;
- checkpoint manifest selection; and
- committed restart plans.

The writer may be lease-elected rather than a fixed process. All authoritative
state must be in the strongly consistent store so another coordinator can take
over after the lease expires without changing or duplicating an already
committed plan.

It may be implemented over the rendezvous store for a prototype, but control
keys must be namespaced separately from worker-group bootstrap keys. Production
deployments should use a durable external service if the rendezvous endpoint is
not sufficiently available or persistent.

### Slot-aware rendezvous

`SlotAwareRendezvous` admits only nodes named by the committed plan:

- standby agents block without creating trainers;
- quarantined nodes receive a terminal rejection;
- a replacement node receives the failed node's logical slot;
- exactly `min_nodes` slots complete the rendezvous; and
- returned group rank is derived from logical slot, not arrival order.

This is stricter than the normal dynamic-rendezvous wait list. A random late
node must not trigger scale-up.

### Resilient elastic agent

The prototype `ResilientElasticAgent` extends the local agent loop to poll the
intent and committed generation. It:

1. asks accessible workers to prepare when it observes an intent;
2. waits in a quiesced state after acknowledgement or timeout;
3. stops all local workers when it observes the committed plan;
4. reports the local stop result;
5. enters slot-aware rendezvous if admitted; and
6. injects the rank-local `RestartContext` before starting workers.

All active agents act on the same committed plan. A node-local SCOUT report
cannot directly force only its own workers to restart.

Current torchrun does not expose this policy as a stable public hook. The
prototype will likely subclass protected agent behavior and must therefore be
pinned and tested for each supported PyTorch minor version.

The upstream target is a public interface similar to:

```python
class ElasticResiliencyPolicy(Protocol):
    def register_agent(self, agent: AgentIdentity) -> AgentAdmission: ...
    def poll_action(self, state: AgentState) -> AgentAction: ...
    def prepare_worker_env(
        self,
        assignment: RankAssignment,
        local_rank: int,
    ) -> dict[str, str]: ...
    def report_action_result(self, result: AgentActionResult) -> None: ...
```

This hook should run above process monitoring and below the CLI, so a policy can
request a coordinated restart without pretending a healthy worker crashed.

## Control-Plane Integrity

- Create the local control directory with owner-only permissions and verify Unix
  peer credentials where supported.
- Write intent, context, and acknowledgement files with atomic replace.
- Provision `node_id` and agent credentials outside the training worker.
- Namespace every store key by run ID and schema version.
- Protect the external store with deployment authentication and authorization.
- Include the plan ID and generation in every mutable operation.
- Reject path traversal and never expand `location_token` as a raw filesystem
  path without resolving it through the trusted control client.
- Treat malformed, unauthenticated, or conflicting input as unavailable
  evidence, not as permission to continue.

## Healthy-Path Cost

The integration must not add a global synchronization to normal optimizer
steps.

- Fault and recovery events are emitted only when SCOUT produces actionable
  evidence.
- Checkpoint inventory contains metadata only and is published asynchronously
  after a completed checkpoint transition.
- Agent plan polling is out of process and bounded.
- Standby agents do not allocate model state or join training collectives.
- Checkpoint flush and transfer happen only after a restart intent.
- Control delivery backpressure must not block the training thread
  indefinitely.

## Recovery Policy

| Incident | Node action | Recovery mode |
|---|---|---|
| Attributed SDC on one node/GPU | Quarantine affected node conservatively | `RECOVERY_VERIFIED` |
| Inconclusive exact replay | Do not over-attribute; operator policy may replace a broader scope | `RECOVERY_VERIFIED` |
| Required node inaccessible | Mark failed and replace its slot | `RECOVERY_VERIFIED` |
| Confirmed accessible compute or communication straggler | Drain and replace when policy threshold is met | `LATEST_GEMINI` if a complete generation is prepared; otherwise verified |
| Hang with complete clean emergency replay and all ranks accessible | Restart or replace implicated scope | `LATEST_GEMINI` |
| Hang with missing, stale, incomplete, or contradictory evidence | Restart conservatively; avoid unsupported attribution | `RECOVERY_VERIFIED` |
| Operator-initiated healthy migration | Drain source and replace slot | `LATEST_GEMINI` if preparation succeeds |

### Aggregation rules

The coordinator uses all available events for the incident:

- any SDC or inconclusive exact evidence forces verified recovery;
- any inaccessible required rank forces verified recovery;
- a latest proposal is accepted only if the checkpoint manifest proves every
  required logical-rank shard is complete at one step;
- conflicting steps are not merged;
- after plan commit, workers load the exact selected step rather than
  independently choosing a newer local generation;
- missing reporter events do not count as agreement;
- a checksum failure makes that copy unavailable;
- checksum success does not prove the original GPU state was numerically
  correct; and
- one comparison group cannot certify the job if another group found SDC.

## Checkpoint Placement and Transfer

Stable logical slots are required for the fast path. GEMINI checkpoints are
rank-sharded and peer replicas preserve the original owner rank. If torchrun
arbitrarily renumbers ranks after replacement, a new framework topology may not
match those shards even when `WORLD_SIZE` is unchanged.

For each selected checkpoint, the coordinator builds one immutable manifest:

```python
class RankCheckpointCopies(TypedDict):
    owner_global_rank: int
    copies: list[CheckpointCopy]


class RecoveryManifest(TypedDict):
    manifest_id: str
    run_id: str
    source_generation: int
    step: int
    trust: Literal["latest", "recovery_verified"]
    topology_digest: str
    rank_copies: list[RankCheckpointCopies]
```

The manifest is usable only when every required rank has at least one eligible
copy.

Preferred source order:

1. completed owner copy on an accessible healthy node;
2. completed peer replica on a different healthy node;
3. previously mirrored restart storage; or
4. framework durable recovery-verified checkpoint.

For an SDC incident, the selected generation must already be
`RECOVERY_VERIFIED`. The current suspect state is never promoted. A stored
verified copy on a suspect machine should be transferred only when policy
allows it and integrity verification succeeds; a healthy peer or durable copy
is preferred.

The current `CheckpointTransfer` interface can move shards after the
coordinator chooses source and destination endpoints. The transport never
chooses the checkpoint or replacement node.

If a failed node and every holder of its peer replicas are unavailable, GEMINI
cannot reconstruct that rank's fast checkpoint. Recovery falls back to the
framework durable checkpoint.

## End-to-End Flows

### Initial launch

1. All `max_nodes` agents register stable node identities.
2. The coordinator validates compatibility and selects exactly `min_nodes`.
3. Selected nodes receive logical slots; the rest enter `STANDBY`.
4. Slot-aware rendezvous assigns deterministic rank ranges.
5. Agents start workers with generation `0`.
6. `lm-resiliency` registers the worker identity and framework topology digest.

### Accessible straggler replacement

1. SCOUT emits a confirmed straggler report and latest-checkpoint proposal.
2. The coordinator maps the reported rank to its generation and node.
3. The coordinator opens a restart intent and provisionally reserves a
   compatible standby for the affected logical slot.
4. Every active agent enters `DRAINING`.
5. Workers flush eligible GEMINI state and report transfer results.
6. The coordinator commits a restart plan using the newest complete allowed
   generation, or falls back to verified recovery.
7. Agents stop all workers.
8. The source node becomes quarantined or standby according to policy.
9. The replacement and surviving nodes rendezvous with the same slots/ranks.
10. Relaunched workers validate the restart context and load exactly the
    committed rank-consistent step.
11. TorchTitan restores the complete state and resumes.

### Inaccessible node

1. The agent or infrastructure marks one node unavailable.
2. The coordinator opens an intent so surviving nodes can expose their
   completed peer replicas.
3. The coordinator selects `RECOVERY_VERIFIED`.
4. The committed manifest uses peer replicas or the durable verified
   checkpoint.
5. A standby inherits the failed logical slot.
6. Surviving workers are stopped and the fixed-size group is relaunched.
7. If any required shard is unavailable, the job fails rather than mixing
   checkpoint generations.

### SDC

1. SCOUT rejects the current candidate and emits attributed or inconclusive
   evidence.
2. The coordinator selects only `RECOVERY_VERIFIED`.
3. The affected node is quarantined only to the scope supported by evidence and
   deployment policy.
4. No current or candidate checkpoint is copied into the restart manifest.
5. The fixed-size group relaunches and loads the verified generation.

## Failure Handling

- **Coordinator unavailable:** do not autonomously form a new generation.
- **Coordinator unavailable with an open intent:** remain quiesced and fail
  after the control-plane deadline; do not resume training without an explicit
  safe abort.
- **Conflicting committed plans:** fail the run; never choose locally.
- **Stale event:** ignore it and record the rejection.
- **Missing preparation acknowledgement:** treat the node's latest state as
  unavailable.
- **Standby fails admission health checks:** select another standby or fail.
- **Too few eligible nodes:** wait within policy or fail; do not silently shrink
  the training world.
- **Restart context mismatch:** exit before framework checkpoint loading.
- **Checkpoint manifest incomplete:** use a complete verified durable
  checkpoint or fail.
- **Transfer timeout or checksum failure:** mark that copy unavailable and
  re-plan only from the same or a more conservative trust state.
- **Agent crash after plan commit:** another agent cannot rewrite the plan; the
  same generation may be retried idempotently.

## Packaging and Compatibility

- Keep the public neutral payloads in `lm_resiliency.manager_api`.
- Put torchrun-specific code under `lm_resiliency.integrations.torchrun`.
- Keep TorchTitan imports lazy.
- Importing `lm_resiliency` must not import TorchTitan or CUDA-only packages.
- Version every wire payload independently from the Python package version.
- Reject unknown required fields or unsupported schema versions.
- Pin the prototype agent integration to qualified PyTorch minor versions
  because protected agent methods are not a stable API.
- Preserve the existing `OrchestrationHooks`, `RecoveryDecision`,
  `SCOUTFaultReport`, and framework entry points.
- Use PyTorch 2.13 as the initial prototype baseline, then separately qualify
  the other PyTorch minors in the supported compatibility range.

### Proposed prototype command

An out-of-tree command avoids changing torchrun's existing CLI semantics:

```bash
python -m lm_resiliency.integrations.torchrun.launch \
  --active-nnodes=8 \
  --fleet-nnodes=10 \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --rdzv-id="${JOB_ID}" \
  --max-replacement-generations=2 \
  --module torchtitan.train ...
```

### Upstream CLI target

If PyTorch accepts a resiliency-policy hook, torchrun can preserve its current
default and add explicit replacement-only semantics:

```bash
torchrun \
  --nnodes=8:10 \
  --active-nnodes=8 \
  --standby-policy=replace-only \
  --resiliency-policy=lm_resiliency \
  --nproc-per-node=8 \
  --rdzv-backend=c10d \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --rdzv-id="${JOB_ID}" \
  --module torchtitan.train ...
```

The names are placeholders for upstream discussion. The important requirement
is that replacement-only mode is opt-in and does not silently change the
meaning of existing `--nnodes=min:max` jobs.

## Validation Plan

### Contract tests

- JSON round-trip and schema-version rejection for every event and plan.
- Stale generation, duplicate event, and conflicting-plan rejection.
- Conservative recovery lattice across missing and conflicting proposals.
- Rank-to-node mapping through immutable generation snapshots.
- Quarantine by stable node ID, never by rank alone.
- Plan validation for exact active size, unique slots, topology digest, and
  complete checkpoint manifest.

### CPU multi-process tests

- Standby agents do not spawn trainer workers.
- A new standby does not cause scale-up.
- A committed plan stops every active local worker group.
- A replacement inherits the same logical slot and rank range.
- A quarantined node cannot rejoin.
- Restart context mismatch aborts before user code initializes recovery.
- Agent/coordinator restart preserves idempotent generation state.

### Focused GPU and multi-node tests

Start with `min_nodes=2`, `max_nodes=3`, and the smallest topology that
exercises peer replication:

1. kill one active node and recover its rank shards from peer replicas;
2. replace an accessible straggler after bounded flush and transfer;
3. inject SDC and verify rollback to the prior recovery-verified generation;
4. make one rank's newest shard unavailable and verify global step selection
   falls back consistently;
5. verify bitwise model, optimizer, scheduler, RNG, and data-position recovery;
6. repeat with TorchTitan and native PyTorch;
7. exercise one HSDP or FSDP2 topology with stable shard coordinates; and
8. document the remaining hardware-dependent coverage.

The test must assert that no quarantined node runs a trainer process in the new
generation and that the active world size remains exactly `min_nodes`.

## Alternatives Considered

### Use stock `--nnodes=min:max` and callbacks only

Insufficient. It cannot keep extra nodes standby-only, preserve stable rank
slots, or exclude a localized bad node from later rendezvous.

### Implement only a custom rendezvous handler

Insufficient. It can control admission, but it cannot safely ask all currently
healthy agents to prepare and stop their workers after a SCOUT decision.

### Allow the active world to shrink after a failure

Deferred. It changes framework parallelism, checkpoint sharding, batch size,
optimizer semantics, and potentially the training trajectory. It requires a
separate elastic-topology and resharding contract.

### Put placement policy inside lm-resiliency

Rejected. SCOUT and GEMINI do not own scheduler state, leases, or physical
resource lifecycle. They should provide evidence and recovery state to a
torchrun or infrastructure manager.

### Delegate everything to an external scheduler

Viable for production, and the neutral manager API should continue supporting
it. The reference design still adds value by defining the torchrun-native
worker/agent contract and enabling local-agent standby replacement.

## Phased Implementation

### Phase 0: agree on contracts

- Decide stable node identity and logical-slot semantics.
- Agree on the event, inventory, manifest, plan, and restart-context schemas.
- Decide where the single-writer coordinator runs.

### Phase 1: manager-neutral data model

- Add versioned event envelopes and `RestartContext`.
- Extend checkpoint inventory without changing current callback behavior.
- Add contract tests and conservative aggregation tests.

### Phase 2: reference torchrun integration

- Add the local orchestration client.
- Add the coordinator and quarantine store.
- Add standby-only admission and stable slot rendezvous.
- Add a version-pinned resilient elastic agent and prototype launcher.

### Phase 3: framework validation

- Validate native PyTorch and TorchTitan end to end.
- Add Megatron Core and DeepSpeed after the neutral contract is stable.

### Phase 4: upstream torchrun proposal

- Propose a public coordinated-restart policy hook.
- Propose explicit replacement-only standby semantics.
- Remove protected-method subclassing when the public hook is available.

## Open Questions

Recommended defaults for the first prototype:

- pre-launch standby agents in the same allocation;
- require stable logical slots and exact rank-range inheritance;
- quarantine the whole node when one full-node torchrun local worker group is
  the scheduling unit;
- prefer healthy peer or durable verified copies over a suspect node;
- keep framework durable checkpoint IDs opaque;
- use a separate replacement-generation budget; and
- propose the smallest upstream hooks for coordinated restart, worker
  environment injection, and admission before proposing a broad policy API.

1. **Coordinator placement:** should the first prototype use a namespaced c10d
   rendezvous store, etcd, or a separate manager service?
2. **Stable slots:** can all target deployments guarantee that a standby can
   inherit the failed node's exact rank range and local device count?
3. **Quarantine scope:** when SCOUT identifies one GPU but torchrun launches a
   homogeneous full-node local worker group, should policy always quarantine
   the whole node in version 1?
4. **Verified copies on suspect hardware:** may a previously certified
   checkpoint be transferred from a now-suspect node when its checksum passes,
   or must recovery use only a healthy peer/durable copy?
5. **Replacement budget:** should repeated replacement generations be bounded
   separately by incident type, node, and whole job?
6. **Standby lifecycle:** are standby agents pre-launched in the same scheduler
   allocation, or must the coordinator request a new host on demand?
7. **Durable checkpoint handoff:** should `checkpoint_id` remain framework
   opaque, or should the restart manifest expose a common durable-checkpoint
   URI contract?
8. **Upstream surface:** is a general `ElasticResiliencyPolicy` acceptable to
   torchrun, or should the first upstream change be a smaller coordinated
   restart and worker-environment hook?

## References

- [GEMINI operational contract](gemini.md)
- [SCOUT operational contract](scout.md)
- [LM Resiliency API guide](api.md)
- [Compatibility policy](compatibility.md)
- [PyTorch torchrun elastic launch documentation](https://docs.pytorch.org/docs/2.13/elastic/run.html)
- [PyTorch elastic agent documentation](https://docs.pytorch.org/docs/2.13/elastic/agent.html)
- [PyTorch rendezvous documentation](https://docs.pytorch.org/docs/2.13/elastic/rendezvous.html)
