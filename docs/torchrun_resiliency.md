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
- let an LM Resiliency coordinator make the final quarantine, placement, and
  restart-plan decision;
- use a custom `RendezvousHandler` to park standbys and expose only a
  plan-selected replacement to torchrun's existing membership-change path; and
- pass one fenced `RestartContext` to every relaunched worker so
  `lm-resiliency` and the framework recover the same checkpoint generation.

### Why `--nnodes=min:max` is not a standby contract

In stock torchrun, `--nnodes=min_nodes:max_nodes` bounds the number of nodes
that may participate in an elastic rendezvous. It does not divide an allocated
fleet into active nodes and replacement-only standbys:

- `min_nodes` is the threshold at which a rendezvous is allowed to complete,
  not the exact number of nodes that must run trainers;
- `max_nodes` is the largest active membership for a rendezvous round, not a
  fleet size whose excess capacity is reserved; and
- every admitted node is an active worker-group member. There is no standard
  state meaning "keep this agent registered, but do not start trainers unless a
  particular active node is replaced."

For example, `--nnodes=8:10` can initially form an active group of 8, 9, or 10
nodes depending on which agents arrive before the rendezvous completes. If it
starts with 8 and a ninth node arrives later, that node is a candidate for the
next rendezvous round. The resulting membership change restarts the worker
group and may increase `WORLD_SIZE`; it is not treated as passive reserve
capacity. Conversely, after a departure, a new round may form with any allowed
membership rather than waiting for a policy-selected replacement that restores
exactly 8 active nodes.

The semantic differences are:

| Property | Stock `--nnodes=min:max` | Replacement-only design |
|---|---|---|
| Active size | Any rendezvous size in the configured range | Exactly `min_nodes` |
| Extra nodes | Elastic scale-up candidates | Agent-only standbys |
| Restart trigger | Worker failure or rendezvous membership change | Worker failure or a plan-selected membership change; admission is gated by `RestartPlan` |
| Placement | Participants selected by rendezvous; ranks reassigned per round | Coordinator selects a node for a specific logical slot |
| Rank identity | Global and group ranks may change after re-rendezvous | Logical slot and rank range remain stable |
| Fault exclusion | No job-specific quarantine contract in the range | Quarantined node and resource IDs cannot be readmitted |
| Recovery state | Restart is not coupled to checkpoint trust or completeness | Relaunch is fenced to one complete trusted checkpoint manifest |

Even if a stock re-rendezvous happens to return to 8 nodes, it does not
guarantee that the replacement inherits the departed node's rank range.
Arbitrary renumbering can change framework parallel coordinates and disconnect
GEMINI's rank-owned shards and peer replicas from their intended owners.
`--max-restarts` only bounds retries; it does not add standby admission,
quarantine, slot inheritance, or checkpoint-selection semantics. The design
therefore needs an explicit replacement-only admission policy and stable
logical slots in addition to a node-count range.

The design uses existing torchrun extension points rather than adding a new
resiliency-policy interface:

1. register an out-of-tree `SlotAwareRendezvousHandler` through the
   `torchrun.handlers` entry-point group;
2. keep the stock `LocalElasticAgent` and its normal full-worker-group restart
   behavior; and
3. keep fault evaluation, checkpoint preparation, quarantine, and immutable
   restart-plan selection in the LM Resiliency control plane.

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
- A planned restart that preserves the identical physical node membership;
  version 1's healthy-group restart edge is admission of at least one standby.
- Reusing a node that SCOUT could not safely clear.
- Physical hardware repair.
- Turning inconclusive peer evidence into confident node attribution.

## Existing Interfaces and Required Integration

Stock torchrun already provides useful foundations:

- an agent process on every participating node;
- local worker lifecycle and failure monitoring;
- worker-group restart after process failure;
- dynamic rendezvous with `min_nodes` and `max_nodes`;
- custom rendezvous handlers through the `torchrun.handlers` entry-point group;
- `RendezvousHandler.next_rendezvous()` for admission and rank assignment;
- `RendezvousHandler.num_nodes_waiting()` for requesting re-rendezvous;
- restart-count and run-ID environment variables; and
- full worker-group restart on failure or membership change.

The existing interfaces are sufficient for the torchrun lifecycle mechanics:

- a standby agent remains blocked in `next_rendezvous()`, so its trainers are
  not started;
- the handler can return a deterministic group rank derived from a logical
  slot;
- parked standbys are kept outside the handler's reported wait count; and
- after a restart plan commits, the handler reports only the selected
  replacement through `num_nodes_waiting()`. The stock elastic agent then
  performs its normal full-worker-group membership restart.

The handler is not the recovery decision-maker. The integration still must
provide contracts that stock torchrun does not:

1. `min_nodes:max_nodes` describes elastic active membership, not
   `min_nodes` active nodes plus replacement-only standbys.
2. The stock rendezvous handlers do not distinguish passive standbys from
   scale-up candidates.
3. Stock rank assignment is not stable across re-rendezvous.
4. Torchrun failure handling does not consume SCOUT fault localization or
   checkpoint trust.
5. The rendezvous API does not define a quarantine database or logical-slot
   inheritance contract.
6. Checkpoint preparation and recovery-plan consensus remain application and
   manager responsibilities.

PyTorch 2.13 includes an experimental worker control-plane server, but it is a
generic handler server and not a stable restart or quarantine protocol. The
reference integration does not require that experimental surface.

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
| Stock LocalElasticAgent     RendezvousHandler plugin              |
| - monitors and restarts local workers                            |
| - calls next_rendezvous() before worker launch                    |
| - polls num_nodes_waiting() while workers are healthy             |
| - performs the standard full-group membership restart            |
+-----------------------------+------------------------------------+
                              |
        control client / store | restart-context file/environment
                              v
+------------------------------------------------------------------+
| Layer 2: lm-resiliency                                           |
|                                                                  |
| SlotAwareRendezvousHandler  RestartCoordinator  SCOUT  GEMINI    |
| TorchrunOrchestrationClient  manager_api                          |
| - owns active/standby state and logical-slot assignments         |
| - commits one fenced RestartPlan                                 |
| - exposes only the selected replacement as waiting               |
| - excludes quarantined nodes                                     |
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
| Final node quarantine and replacement policy | LM Resiliency coordinator |
| Active/standby admission and logical-slot assignment | `SlotAwareRendezvousHandler` and coordinator |
| Worker-group stop and relaunch | Stock torchrun elastic agent |
| Final restart generation and checkpoint manifest | LM Resiliency coordinator |
| Model and optimizer state semantics | framework adapter |
| Physical repair or scheduler allocation | infrastructure manager |

SCOUT and GEMINI may recommend a replacement scope and recovery state, but they
must not directly evict a physical host. A rank-local report is evidence, not a
cluster-wide decision.

## Identity Model

Ranks cannot be used as durable resource identities. Every event and plan must
carry both the current rank mapping and stable infrastructure identities.

```python
class AgentIdentity(TypedDict):
    run_id: str
    node_id: str  # Stable scheduler or infrastructure identity.
    agent_id: str  # Unique torchrun agent incarnation.
    hostname: str
    local_world_size: int
    resource_ids: list[str]  # GPU UUIDs, NICs, HCAs, or deployment IDs.
    environment_digest: str  # Software, configuration, and capability identity.


class WorkerIdentity(TypedDict):
    run_id: str
    generation: int
    node_id: str
    agent_id: str
    logical_node_slot: int  # Stable for the lifetime of the training job.
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

Version 1 behavior persists an immutable whole-node `NodeQuarantineRecord` for
every excluded node. Record schema version 2 binds the stable run and node
identities to the committed plan and intent, the failed and effective successor
generations, the incident set, policy reason, affected resource evidence, and
the complete coordinator lease identity, duration, and fencing token that
authorized it. Schema version 1 is rejected rather than interpreted as this
expanded layout. The resource list may be empty when evidence supports only
node-level exclusion; it does not narrow the effect of the record. A node
quarantine is permanent for the run and must be create-only under the
coordinator lease.
The quarantine write repository does not expose a standalone commit or read
operation. It authenticates the held coordinator lease, validates plan/intent
linkage and trusted resource ownership, then returns create-once writes that the
coordinator must compose into the same guarded transaction as restart-plan
publication and generation advancement. No quarantine is authoritative until a
later combined reader verifies the matching persisted plan, successor
generation, quarantine record, commit timestamp, and guard provenance from that
transaction. Resource evidence passed to the repository must already be
authorized from validated fault evidence; ownership validation prevents that
evidence from naming another node's resource.

`node_id` must come from a trusted scheduler, cloud instance identity, or
deployment inventory. A worker-provided hostname is diagnostic metadata and
must not be allowed to choose its own quarantine identity. If no stable node
identity is available, replacement mode should refuse to start.

The trusted resource inventory maps every resource ID to both its stable node
owner, resource kind (`gpu`, `nic`, `hca`, `link`, or `node`), and the global
rank for rank-bound endpoints. Worker registration proves that an agent may
report a resource; it does not allow the worker to relabel a GPU as a NIC or
attribute another local worker's GPU to its own rank.

The topology digest covers at least the active world size, local worker count,
framework parallel dimensions, logical rank-to-parallel-coordinate mapping,
checkpoint schema, and model configuration needed to interpret rank-local
state. It is separate from `environment_digest`, which describes whether a node
is eligible to run the same software and hardware workload.

Each live agent persists a strict, versioned `AgentRegistrationRecord` that
contains its complete immutable `AgentIdentity`, a unique registration lifetime
ID, and the registration lease duration. The control-store revision is the
opaque fencing token, and the store-stamped commit time plus duration determines
expiry. Registration proves that one agent incarnation is live on a trusted
node identity; it does not by itself make the node standby-eligible, admit it to
a generation, or authorize resources absent from the trusted infrastructure
inventory.

The registration manager stores one live record per run and stable node ID.
Retrying registration for the same immutable agent identity renews the existing
registration ID; a different agent or changed environment remains blocked until
the current registration expires. Renewal and release authenticate the complete
held record, fencing revision, and authoritative grant time before issuing a
store-time-guarded mutation. Expiry takeover creates a new registration ID and
fencing token. Registration keys are independently scoped by hashed run and node
identity, so agents on different nodes do not contend.

Equivalent heartbeat renewals use a store refresh operation that compacts
unreferenced intermediate values. The retained history keeps the initial and
current values of each registration lifetime plus every revision consumed by a
successful guarded transaction condition. Acknowledgements and plan commits
therefore retain the exact registration authority they used, while long-running
active and standby agents do not accumulate one durable history entry per
heartbeat. History readers accept compacted same-registration mutation gaps only
when the skipped renewals could all have completed before lease expiry.

The coordinator does not enumerate arbitrary control-store keys. A registration
reader receives the trusted scheduler node-ID set explicitly, derives the same
hashed keys as the agents, and reads only those registrations. After completing
the reads, it samples the observer clock once and classifies each trusted node
as live, expired, or missing. This observation is conservative rather than an
atomic multi-key snapshot: a concurrent renewal may appear expired or missing,
but a selected registration must be revalidated before admission or plan
commit. A store-stamped grant time later than the observer clock is a clock
error, not a live registration.

## API: lm-resiliency to the coordinator

The current `OrchestrationHooks` callbacks are the starting point. The torchrun
integration wraps them in a versioned, incident-correlated envelope and sends
them to the coordinator through the configured manager transport. A local
Unix-domain socket or sidecar may provide that transport, but the stock
torchrun agent is not part of the protocol.

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
- Event admission validates the reporter's node, logical slot, local worker
  count, rank range, and topology digest against that generation's immutable
  `RankAssignment`; worker-supplied rank arithmetic alone is insufficient.
- Event admission also binds the worker to its registered `AgentIdentity`.
  Hardware reports are accepted only when the reported node or resource belongs
  to that agent and both its node owner and resource kind match the trusted
  scheduler or infrastructure inventory.
- The original `SCOUTFaultReport` is preserved without upgrading its
  attribution. A direct `HealthEvent` is normalized to
  `HardwareFaultReport` without upgrading its resource granularity.
- The coordinator resolves reported ranks through that generation's
  `RankAssignment` and rejects any reported rank outside its committed ranges.
- `failed_ranks`, `endpoint_rank`, `dataloader_culprit_ranks`, and
  `stage_culprit_ranks` are validated together. Culprit and endpoint ranks must
  be included in `failed_ranks`; node or resource endpoint IDs must resolve to
  the endpoint rank's node through the trusted assignment and resource map.
  Rank-bound resources must also match the trusted resource-to-rank binding;
  node-shared NIC, HCA, and link resources need no artificial rank binding.
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

A schema-v1 proposal accepts only the known failure kinds `straggler`,
`data_stall`, `checkpoint_stall`, `hang`, `uncertain`, `sdc`, and
`machine_unavailable`. Unknown or newly introduced kinds are rejected until a
new schema defines their recovery semantics.

A worker must never interpret its local `checkpoint_step` as the final global
step. The coordinator selects one step from the complete job-wide inventory,
and the restarted job collectively validates that exact step before loading.

### Checkpoint inventory

`RecoveryDecision` says what one worker recommends. Replacement also needs to
know where every logical-rank shard can be obtained.

```python
class CheckpointCopy(TypedDict):
    owner_global_rank: int
    checkpoint_step: int
    inventory_event_id: str
    checkpoint_id: str | None
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

Only completed checkpoint slots may appear with `complete=True`.
`checkpoint_step` must equal the enclosing inventory event's positive `step`;
`inventory_event_id` binds the copy to that source event, and the coordinator
validates the complete copy record against the referenced event when assembling
the recovery manifest. A `CANDIDATE` inventory can be retained for diagnosis
but is never selected for conservative recovery. Durable copies must use shared
or remote storage and carry the opaque durable `checkpoint_id`; that ID must
match the committed plan. Owner and peer copies do not carry a durable
checkpoint ID. Node-local and in-memory copies remain subject to holder health
and quarantine, and are eligible only when the inventory event was reported by
that holder. Shared or remote copies may instead be referenced by an
authenticated control-plane inventory.

`location_token` is an opaque control-plane reference, not an unchecked path
that another worker is allowed to open.

Worker inventory is not checkpoint certification. The coordinator consumes a
separate record from the trusted SCOUT/GEMINI catalog store:

```python
class CheckpointCertification(TypedDict):
    schema_version: int
    certification_id: str
    run_id: str
    source_generation: int
    step: int
    topology_digest: str
    checkpoint_source: Literal["gemini", "durable"]
    checkpoint_id: str | None
    expected_world_size: int
    certification_kind: Literal[
        "dense_consensus",
        "dynamic_candidate_promotion",
    ]
    inventory_event_digests: dict[str, str]  # event ID to canonical SHA-256
```

This record is written only after job-wide dense acceptance or the documented
two-cycle dynamic-catalog promotion. It is not accepted through the worker
event sink. A worker-declared `recovery_verified` inventory is eligible only
when a trusted certification matches its run, generation, step, topology,
source, durable checkpoint ID, world size, and canonical inventory-event
digest. Reusing an event ID with different copy contents therefore cannot reuse
the earlier certification.

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

## API: coordinator to lm-resiliency

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
generation. It does not authorize a new worker group. Accessible workers
observe it through `TorchrunOrchestrationClient`, quiesce at a framework-safe
boundary, prepare eligible checkpoint state, and wait for either a committed
plan or an explicit abort. The stock torchrun agent still considers the worker
group healthy during this preparation phase.

The persisted form is an immutable, strict `RestartIntentRecord`. It embeds the
canonical intent and binds it to the exact committed generation-snapshot digest
plus the complete coordinator lease identity, duration, and opaque fencing
token that authorized it. Duplicate or unknown fields, unsupported schema
versions, malformed nested intents, and noncanonical SHA-256 digests fail
closed. A strict `RestartIntentHeadRecord` identifies the single current intent
by run, generation, intent ID, and canonical intent-record digest. The head
payload is immutable; its future store key is compare-and-swap managed so
opening an intent can atomically create both the create-once intent record and
the current pointer. The strict `RestartIntentLifecycleRecord` separately binds
the last closed intent head to the coordinator lease that performed closure;
that lease may be a renewal of the lease that opened the intent. The record
schemas do not define a store mutation API. Lease-fenced intent creation,
lifecycle observation and closure, and other lifecycle transitions are
separate control-plane components. Each closure record is immutable and has a
canonical digest. A strict `RestartIntentLifecycleHeadRecord` identifies the
latest closure by run, positive closure index, generation, intent ID, and
closure-record digest. The head is the only mutable lifecycle pointer; later
readers require its store mutation and value sequences to equal the closure
index so replaying an older A record after B cannot appear to roll lifecycle
state back. A strict `RestartIntentClosedHeadRecord` is the durable closed value
for the current-intent key. It binds the closure index, generation, and intent
identity to the canonical lifecycle-head digest. A future closure transaction
atomically replaces the open current-intent head with this marker while
advancing the lifecycle head and creating the immutable closure record. The
next intent replaces the marker instead of relying on deletion, so a restarted
reader can prove closure from persisted state. Atomic opening and closure
transactions are separate control-plane components. A frozen
`InitialRestartIntentClosureRecords` value links the first committed intent to
its immutable closure record, first lifecycle head, and durable closed marker.
It exposes create-once closure/lifecycle writes, a revision-guarded
current-head update, and an exact immutable-intent condition, but does not
authenticate a closing lease or mutate the store. A frozen
`PreparedInitialRestartIntentClosure` descriptor binds those records to a
contiguous coordinator-lease authority slice beginning at the exact lease that
opened the intent. It accepts same-lease renewal and nonoverlapping in-place
replacement while rejecting skipped mutations, expired renewals, overlapping
leases, recurrence of any generation or opening lease identity or fencing
token after replacement, and delete/recreate transitions that lack a durable
deletion tombstone. It bounds the future transaction to the final live lease
but performs no store reads or mutations. A later preparer selects the slice
from verified durable lease history. A frozen
`PreparedInitialRestartIntentOpen` descriptor binds the first intent record,
current-intent head, exact generation revisions, coordinator fencing token,
canonical run-scoped keys, and lease/intent commit window. It exposes immutable
create-once writes and side-effect-free generation conditions but performs no
store reads or mutations. The descriptor carries the complete held coordinator
lease and requires the intent record to match its identity, duration, fencing
token, and authoritative grant time. A later preparer authenticates that handle
against persisted ownership before constructing the descriptor. The initial
preparer also revalidates the exact generation, intent scope, never-opened
lifecycle state, and remaining lease/intent window against a monotonic
coordinator clock sampled after those reads. The observation becomes the
transaction's lower time bound; the store remains authoritative at execution.
The prepared transaction also carries a never-created lifecycle-head
condition. The store checks both current absence and durable key history at the
same linearization point as the lease, generation conditions, and intent
writes. The preparer performs no mutation. The initial-open executor submits
that guarded transaction, translates lease, deadline, clock, generation, and
lifecycle conflicts, and verifies that both returned entries share the expected
bytes, commit time, transaction sequence, generation order, and lease
provenance. Preparation preserves the authenticated lease entry's transaction,
mutation, value, and lifetime sequences. Those sequences must equal the
generation snapshot's authority for the same fencing token or advance
consistently for a nonexpired renewal or nonoverlapping replacement. Execution
requires both committed entries to match that exact lease provenance and to
follow both the generation snapshot and the authenticated lease transaction.
The lease mutation count cannot exceed the store-global transaction gap, a
post-generation lease grant cannot predate the generation commit, and a lease
ID or fencing token cannot reappear after a different acquisition in the
verified generation history. Preparation obtains that history through one
stable reader traversal rather than rereading the full predecessor chain for
each generation.

A non-mutating `RestartIntentClosurePreparer` reconstructs the current
committed opening, double-collects verified coordinator-lease history, requires
the supplied closing lease to be the live durable tail, selects the contiguous
authority slice beginning at the opening lease, and samples a monotonic commit
lower bound after those reads. It returns
`PreparedInitialRestartIntentClosure`; a separate executor owns the guarded
store transaction and committed-result verification. Lease-key lifetime
changes remain rejected until the store retains authoritative deletion
evidence.

`RestartIntentClosureExecutor` submits that prepared transaction and returns a
frozen `CommittedInitialRestartIntentClosure` only after verifying the exact
three-key result, immutable closure/lifecycle creation, one in-place
current-head replacement, shared transaction identity, closing-lease
provenance, prepared time window, and causal order after both opening and lease
authority. Store revision, history, deadline, and clock failures are translated
into closure-specific fail-closed errors.

A read-only `RestartIntentOpenStateReader` reconstructs the same
`CommittedInitialRestartIntentOpen` contract after coordinator failover. It
stably reads the current-intent head and immutable intent, requires the
intent's generation to remain current, locates the exact opening authority in
verified durable coordinator-lease history, and reuses the existing
prepared/committed validators. Missing records, noncanonical bytes,
contradictory lifecycle state, deleted heads, and lease provenance absent from
durable history fail closed. When the current head is a closed marker, the
reader raises `RestartIntentOpenStateClosed` instead of treating the opening as
absent. That signal does not authenticate the closure; the lifecycle reader
must verify the linked closure records separately. The open-state reader
performs no mutation.

A frozen `PersistedInitialRestartIntentClosure` value decodes the first
closure's immutable intent, retained open head, durable closed marker,
immutable closure record, and lifecycle head from their authoritative store
entries. It requires canonical bytes, one linked initial closure, immutable
create-once records, exactly one current-head replacement, canonical
coordinator-lease guard keys, shared opening and closure transactions, and
causal transaction order. Generation and durable lease-history authentication
remain the responsibility of the lifecycle reader that constructs this value.

A frozen `AuthenticatedInitialRestartIntentClosure` value binds that decoded
closure to the referenced immutable generation and verified durable
coordinator-lease history. It resolves the exact generation, opening, and
closing authorities, requires them to occur in order, requires generation and
intent opening to precede the immediate successor generation in both
transaction and store time, and verifies that generation creation, intent
opening, and closure each committed inside their authoritative lease windows
and before the next durable lease mutation. When an immediate successor
exists, its exact guard must also resolve to durable lease history and its
generation commit must fall inside that authority's lease window. Construction
is fail closed and performs no control-store reads or mutations. Because
retained value history does not identify the intervening delete transaction,
an authority followed by a delete-and-recreate lifetime cannot authenticate an
earlier commit; that case remains rejected until the store provides a durable
deletion tombstone.

A read-only `InitialRestartIntentLifecycleReader` observes that first closure
separately. It takes stable snapshots of the current-intent head, its retained
open predecessor, immutable intent, immutable closure record, and lifecycle
head; requires durable history for every observed key; obtains the referenced
generation and verified coordinator-lease history through their existing
readers; then constructs `PersistedInitialRestartIntentClosure` and
`AuthenticatedInitialRestartIntentClosure`. Missing, rewritten, split,
noncanonical, or contradictory state fails closed. Retryable dependency-read
contention remains retryable, and the reader performs no mutation.
Before the first closure, the same stable observation requires the current
open head and its linked immutable intent to remain canonical create-once
records from one guarded transaction; it also requires the lifecycle head and
initial closure key to have no durable history. The guard must be the run's
canonical coordinator lease with the fencing token and lease digest carried by
the immutable intent. Both opening entries require authoritative commit times,
must follow the current generation, must bind its exact snapshot digest, and
must name one exact authority in durable lease history whose grant, expiry, and
fencing sequences bound the opening before the next lease mutation. The
generation's own guard must resolve to one durable authority and satisfy the
same causal lease bounds. Every suspected node must belong to that generation.
Both pre-closure and closure reads double-collect verified generation and
coordinator-lease histories around the persisted state so failover cannot
combine old lease authority with a newer generation.

`suspected_node_ids` is the policy-approved replacement scope for the
incident. Every listed node must belong to the committed generation and must be
absent from the next assignment. Policy may additionally quarantine a removed
node when the evidence supports doing so.

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
    restart_deadline_unix_ms: int
```

The persisted `RestartPlanRecord` contains the canonical plan; digests for the
selected recovery-manifest record, checkpoint-evidence record, closed-intent
lifecycle record, source generation, and successor generation; the exact
node-to-quarantine-record digest map; and the coordinator lease identity that
authorized publication. Its quarantine keys must exactly match
`quarantined_node_ids`. The envelope is strict and immutable, but it is not a
substitute for `validate_restart_plan()` or for the later atomic publication
transaction.
`RestartPlanEvidenceRecord` preserves the canonical inventory events and
trusted certification records used to admit that plan. The record is bound to
one run, plan ID, and manifest ID and exposes its own digest for the plan
envelope. Construction proves only canonical durable evidence identity; later
state validation must still match the events and certifications to the plan,
manifest, acknowledgements, source generation, and copy eligibility.
The publication bundle writes that evidence record create-once beside the
manifest, plan, quarantine records, successor snapshot, and generation-head
update. Its digest must already match the signed plan envelope.
`RestartPlanGenerationState` binds that envelope to the exact closed intent,
current generation, successor generation, successor assignment, and shared
coordinator publication authority. Manifest and quarantine resolution remain
separate cross-record checks.
`RestartPlanManifestState` then binds the exact resolved manifest record and
checks plan/manifest metadata, source world size, exact current-generation
assignment when applicable, and trust compatibility. It still does not prove
per-rank copy completeness or inventory certification.
`RestartPlanQuarantineState` binds the exact node quarantine records and
requires their digests, plan metadata, and coordinator authority to match the
manifest-bound plan. It does not validate resource evidence or prove atomic
publication.
`RestartPlanInventoryState` then binds every manifest copy to its exact
inventory event, source-generation reporter identity, local holder, and
compatible trust state. It does not prove copy completeness, holder
availability, acknowledgement authorization, or trusted certification.
For `recovery_verified` manifests, `RestartPlanCertificationState` requires
matching trusted catalog records whose canonical event digests cover every
referenced inventory event. It still does not prove copy completeness or
holder availability.
For `latest` manifests, `RestartPlanLatestEvidenceState` requires the stable
restart-acknowledgement collection to answer the exact closed intent and
current generation, and every referenced inventory event must be authorized by
the reporting agent's successful acknowledgement, flushed step, and canonical
event digest. It still does not prove per-rank copy completeness, holder
availability, or fallback eligibility.
`RestartPlanCopyEligibilityState` independently requires exact rank coverage
and rejects every incomplete, wrong-source, wrong-role, process-memory, or
unavailable node-local copy advertised by the immutable manifest. Shared and
remote copies remain independent of holder-node admission. This value does not
establish latest acknowledgement or recovery-verified certification evidence.
`RestartPlanRecoveryEvidenceState` composes that copy eligibility with exactly
one matching trust path: latest restart-acknowledgement evidence or
recovery-verified checkpoint certification. Both values must describe the same
immutable inventory state. Placement, quarantine admission, and atomic
publication remain separate checks.
`RestartPlanPlacementState` performs the registration-backed part of placement
admission without reading or mutating the control store. It requires the
closed intent's suspected nodes to be removed, surviving nodes to retain their
logical slots, the active node count to remain fixed, at least one genuinely
new replacement node, and exact current registration histories for every
successor node. Each selected registration must be live at one observation time
and must match the planned local worker count and the coordinator's expected
`environment_digest`. Trusted hardware inventory, previously committed
quarantine state, recovery evidence, and atomic publication remain separate
admission boundaries.
`RestartPlanCandidateState` composes one placement state with one recovery
evidence state only when both describe the exact same lifecycle and generation
records. The registration observation must also precede the plan's exclusive
restart deadline. This value is still only a publication candidate: trusted
prior quarantine and the atomic store transaction remain separate gates.
`RestartPlanPublicationRecords` derives the canonical immutable plan, manifest,
quarantine, successor-generation, and generation-head writes from one admitted
candidate. It CAS-updates the generation head and conditions on the exact
current and manifest-source snapshot revisions plus every exact successor
registration fencing revision. When the current and manifest-source snapshots
share one generation key, their immutable records and revisions must both
match. Its exclusive deadline is the earliest selected registration expiry or
plan restart deadline. It performs no store access and does not authenticate
the live coordinator lease, lifecycle revision, quarantine resource evidence,
transaction result, or prior committed quarantine state.
`RestartPlanPublicationAuthority` binds that bundle to the exact coordinator
lease identity already embedded in its plan, successor, and quarantine
records. Its observation must follow the candidate's generation,
manifest-source, placement, and lease inputs, and its exclusive deadline is
additionally capped by coordinator-lease expiry. This pure value performs no
store access and does not prove that the supplied authority is still live.
`RestartPlanPublicationAuthorityPreparer` reads a stable durable coordinator
lease history around one monotonic clock sample, requires the live history tail
to match the plan's exact coordinator identity and fencing token, and returns
that pure authority value without mutating the store. Repeated lease changes
remain retryable conflicts; missing, stale, expired, or contradictory authority
fails closed. Lifecycle fencing, quarantine-resource authorization, execution,
and committed-result verification remain separate publication layers.
`RestartPlanPublicationLifecycleFence` derives immutable revision conditions
for the exact closed intent, durable closed head, lifecycle record, and
lifecycle head from one authenticated closure. It rejects any closure whose
successor generation is already committed. Candidate identity, live-store
observation, transaction execution, and committed-result verification remain
separate layers.
`RestartPlanPublicationLifecycleReader` converts the existing stable,
authenticated lifecycle read into that fence without mutating the store. An
open or missing intent, repeated read contention, or an already-committed
successor remains a retryable publication conflict; contradictory persisted
lifecycle state fails closed as corruption.
`PreparedRestartPlanPublication` then composes the authenticated coordinator
authority with that exact lifecycle fence. It requires the candidate's intent,
closure, and source generation to match the authenticated lifecycle; requires
the publication authority to be at or after the closing authority in the same
durable lease history; and merges the lifecycle and candidate revision
conditions into immutable transaction inputs. It still performs no store
access, quarantine-resource authorization, execution, or result verification.
`RestartPlanPublicationPreparer` sequences the existing live-authority
preparer before the stable lifecycle reader and returns that composed value
without mutating the store. It exposes one error boundary for retryable
authority/lifecycle contention, lost coordinator authority, unsafe clocks, and
durable corruption. A closure committed after the authority observation or
other cross-read mismatch remains retryable contention.
`RestartPlanPublicationExecutor` then submits those exact writes, guard,
revision conditions, and store-time window in one atomic transaction. Its
committed result verifies the exact key/value set, generation-head and
immutable-record lineage, common transaction identity, coordinator-lease
provenance, input ordering, and commit window before the publication is
accepted. Lifecycle, generation, or registration churn remains a classified
conflict; substituted store responses fail closed as corruption.
`PersistedRestartPlanPublication` reconstructs the publication transaction
from canonical plan, manifest, recovery-evidence, quarantine,
successor-snapshot, and generation-head entries for one requested run. It
requires exact envelope digests, the requested successor generation, an exact
plan-derived successor assignment and predecessor digest, immutable record
lineage, one transaction and commit time, canonical coordinator-lease guard
provenance, a transaction sequence after the generation-head and lease
mutations, and a commit inside both the lease and restart-deadline windows.
Decoding also rejects manifest metadata or trust that conflicts with the plan,
evidence metadata that conflicts with the plan or manifest, and quarantine
records that conflict with the plan lifecycle or publication authority.
This transaction decoder does not by itself resolve the source generation,
closed lifecycle, lease history, inventory evidence, or currently committed
generation; the stable publication reader performs those checks before
rendezvous may expose the plan.
`RestartPlanPublicationReader` double-collects the current generation and
every atomic publication entry, decodes the stable bundle through
`PersistedRestartPlanPublication`, and requires the successor snapshot and
generation-head revisions plus store metadata to equal the authoritative
current generation. Missing, malformed, mixed, or substituted state fails
closed; repeated head movement remains a retryable read conflict. This first
readback boundary still does not reauthorize lifecycle or checkpoint evidence.
`RestartPlanPersistedRecoveryState` reauthorizes the checkpoint side of one
persisted publication without reading or mutating the store. It resolves the
exact manifest source generation, quarantine records, inventory events, copy
eligibility, and exactly one trust path: persisted trusted certifications for
`recovery_verified`, or supplied restart-acknowledgement evidence for `latest`.
`RestartPlanPublicationReader.read_recovery_state()` double-collects that
publication, the authenticated closed restart-intent lifecycle, and the exact
manifest source snapshot. It requires the lifecycle's source snapshot and
immediate successor, including authoritative store metadata, to equal the
signed publication, and requires the publication transaction and commit time
to follow lifecycle closure, before returning the reauthorized recovery state.
Repeated lifecycle, generation, or lease-history movement remains a retryable
conflict; missing, mixed, causally impossible, or unsafe recovery evidence
fails closed. For `latest`, `read_recovery_state()` automatically reconstructs
the exact stable restart-acknowledgement evidence from the same authenticated
closed intent when the caller does not supply evidence explicitly. Collector
contention remains retryable; missing, corrupted, or contradictory historical
receipts fail closed.

Before commit, the coordinator validates:

- the plan is fenced to the same intent ID, run, generation, incidents, and
  reason;
- the recovery mode is at least as conservative as the intent's minimum;
- every node in the intent's suspected scope is removed from the next
  assignment;
- every new quarantine entry is both in that suspected scope and removed from
  the current assignment; unrelated standbys require separate trusted fault
  evidence before quarantine;
- exactly `min_nodes` active logical slots are assigned;
- at least one assigned node is new to the active membership, because version
  1 uses standby admission as its healthy-group restart edge;
- every surviving node retains its committed logical slot and rank range;
- every assigned node is healthy, compatible, and not quarantined;
- each logical slot is assigned once;
- active size, local world size, total world size, and topology digest match the
  committed generation;
- the checkpoint is complete for every required logical rank;
- every copy included in the committed manifest belongs to the selected
  positive step and selected checkpoint source; a rank entry with one eligible
  copy and any additional ineligible copy is rejected rather than left for the
  loader to choose;
- every copy exactly matches its referenced inventory event, and the source
  inventory trust satisfies the manifest trust;
- the checkpoint trust satisfies the selected recovery mode;
- the plan never selects `CANDIDATE`;
- the target step is coherent across all selected shards;
- the plan generation is the successor of the current committed generation;
  and
- the restart deadline has not elapsed when the plan is committed or exposed
  through rendezvous.

### Restart context passed to workers

Before returning from `next_rendezvous()`, `SlotAwareRendezvousHandler` derives
a node-local context from the committed plan and writes it to an atomically
replaced file. The deployment places the fixed file path in the torchrun
agent's environment before launch, so every local worker receives:

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
    first_global_rank: int
    local_world_size: int
    expected_world_size: int
    topology_digest: str
    recovery_mode: Literal["latest", "recovery_verified"]
    checkpoint_source: Literal["gemini", "durable"]
    checkpoint_step: int
    checkpoint_id: str | None
    checkpoint_manifest_id: str
    reason_code: str
```

Each worker derives its expected rank as
`first_global_rank + LOCAL_RANK`. `lm-resiliency` rejects startup if the
context conflicts with torchrun's `RANK`, `LOCAL_RANK`, `LOCAL_WORLD_SIZE`,
`WORLD_SIZE`, `TORCHELASTIC_RUN_ID`, or the framework topology.
`expected_world_size` must also be divisible by `local_world_size`, so a
context cannot describe a partial logical node slot.

Before inspecting checkpoint fields, the worker validates the complete context
against the currently committed `RestartPlan` for its node. The plan ID,
successor generation, assignment, topology, recovery mode, checkpoint pin, and
reason must match exactly. A leftover context from an earlier plan is rejected
even when stable ranks make every torchrun environment value identical.
This acceptance check receives trusted current time and rejects the context
when the committed plan's restart deadline has elapsed. Rendezvous performs
the same deadline check immediately before exposing the plan.

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

For an accessible worker group, `TorchrunOrchestrationClient` derives a local
command from the intent:

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
    run_id: str
    node_id: str
    agent_id: str
    generation: int
    flushed_step: int
    inventory_event_digests: dict[str, str]  # event ID to canonical SHA-256
    transferred_owner_ranks: list[int]
    transferred_peer_ranks: list[int]
    success: bool
    reason: str
```

The listener only validates and stages the command. It must not call CUDA,
framework, or checkpoint operations from a background listener thread. The
framework adapter consumes the command at a safe optimizer boundary. GEMINI's
bounded main-thread path performs the flush and writes the acknowledgement,
after which the worker remains quiesced.

The coordinator waits only until the intent deadline. Missing or failed
acknowledgements make the associated latest inventory unavailable. A latest
manifest may use an inventory event only when the reporting node returned a
successful acknowledgement for the selected step and included that event ID.
Previously certified recovery-verified inventory remains eligible without
promoting unprepared latest state. Missing or failed preparation cannot promote
`LATEST_GEMINI`; the coordinator must commit a plan using a complete
recovery-verified source or fail the restart.

The manager transport authenticates the acknowledgement sender. The
coordinator checks the acknowledgement's run, node, and agent incarnation
against that authenticated identity and requires the agent to match the
inventory reporter. The coordinator also records receipt time outside the
worker payload and rejects acknowledgements received at or after
`prepare_deadline_unix_ms`. Payload identity fields and timestamps alone are
not authentication. For latest recovery, the acknowledgement's canonical
inventory digest must match the exact event used by the manifest; event-ID
reuse cannot substitute different copies after preparation.

The durable receipt envelope stores the worker acknowledgement together with
the exact immutable restart-intent record, the authenticated agent-registration
lifetime, its registration fencing token and authoritative grant time, and the
coordinator-recorded receipt time. The envelope rejects a sender identity that
does not match the acknowledgement, a receipt outside the authenticated
registration lifetime, or a receipt at or after the intent deadline. The
deadline is exclusive so every accepted receipt has a representable guarded
commit window. The record is not self-authenticating: the acknowledgement
persistence layer must still verify the registration and intent against
authoritative control-store state before committing it.

Each intent/node acknowledgement key is create-once. Its guarded transaction
conditions on the exact immutable intent revision, the still-open current
intent-head revision, and the authenticated agent-registration revision. A
concurrent intent closure, agent renewal/replacement, or duplicate
acknowledgement therefore prevents the receipt from being committed. A receipt
must also be no earlier than the authoritative intent-opening commit, so
pre-intent transport data cannot be replayed as preparation evidence. The
coordinator-lease authority and commit-time window are added by a separate
prepared-write layer before this transaction can execute.

The prepared-write authority is one verified durable coordinator-lease value.
Its commit lower bound must follow the receipt, intent opening, and lease grant;
its exclusive deadline is bounded by both the restart-intent deadline and the
lease expiry. This immutable value still performs no store reads or writes.
The authentication value binds the receipt to the exact committed intent, the
complete store-stamped agent-registration authority, active generation
membership, and one coordinator-lease authority. It also rejects a receipt that
predates the authoritative intent-opening commit. This value performs no store
reads or writes. A following read-only layer constructs it from stable durable
state before the preparation layer selects a bounded commit window.

Each retained agent-registration value is first decoded as one immutable
authority. The strict decoder binds canonical registration bytes to the
expected run and node, requires an authoritative commit time, rejects guarded
registration writes, and preserves the store transaction, mutation, value, and
lifetime sequences. It also rejects impossible per-entry sequence lineage. The
following stable reader validates that the retained history begins at the
initial sequences and contains every replacement and recreated-key transition,
plus the compacted same-registration renewal boundaries needed to prove lease
continuity. It rejects overlapping registrations, expired renewal spans,
recurrent registration identities or fencing tokens, and a current value that
is not the retained tail. A released registration remains visible in the
immutable history while the current value is absent.

The restart-acknowledgement state reader double-collects the current open
intent, verified agent-registration history, and durable coordinator-lease
history. It binds a receipt to the exact current registration and live
coordinator authority without sampling a clock or mutating the store. Durable
contradictions fail closed, while repeated-read contention remains retryable.
The following non-mutating preparer samples a monotonic coordinator clock only
after authentication, rejects an elapsed registration, coordinator lease, or
intent window, preserves the exact registration authority used by the atomic
registration-revision condition, and chooses the earliest exclusive deadline
for the guarded receipt transaction.

The guarded acknowledgement executor publishes the create-once receipt while
the intent, current-intent head, agent registration, and coordinator lease
remain unchanged. Its committed result must match the exact receipt bytes,
carry the prepared coordinator-lease provenance, remain inside the prepared
time window, and follow the durable intent opening, registration authority, and
coordinator authority in store transaction order. Conflicts distinguish
changed intent state, lost registration, lost coordinator ownership, elapsed
intent time, and contradictory store time.

After coordinator failover, one strict persisted-receipt value reconstructs a
committed acknowledgement from its canonical store entry, committed intent
opening, complete agent-registration authority, and coordinator authority. It
requires immutable key lineage, exact guard provenance, active generation
membership, causal transaction order, and a commit inside all three authority
windows. It performs no store reads; a following stable reader supplies the
durable dependencies before acknowledgement collection or quorum logic.

The per-node receipt reader double-collects the restart-intent opening, the
create-once acknowledgement key, verified retained agent-registration history,
and complete coordinator-lease history. For the active preparation path it
reads the current opening. After closure it can instead consume the durable
opening retained by the authenticated lifecycle record, without reconstructing
lost preparation revisions or time bounds. A never-created receipt key returns
no acknowledgement. Deleted, rewritten, malformed, or orphaned receipts fail
closed. Renewed registrations and coordinator leases do not invalidate an
already committed receipt because the reader resolves the exact historical
authorities stamped into that receipt.

An immutable collection value binds one receipt-or-absence observation to every
active node in the committed generation. Its keys must exactly match the
generation assignment, every receipt must answer the same intent opening and
match its node key, and the mapping is frozen in active-slot order. The value
separately exposes received, missing, successful, and explicitly failed node
sets without making a quorum or restart-policy decision.

A read-only collector obtains two identical full active-node observations while
the durable restart-intent opening remains unchanged. Before closure it also
stabilizes the current opening. After closure it accepts an authenticated
lifecycle value and reconstructs the same durable opening from the retained
immutable intent and open-head entries. Because acknowledgement keys are
immutable create-once records, this double collect yields one stable
receipt-or-absence snapshot without requiring a store-wide read transaction.
Repeated changes are retryable conflicts; contradictory per-node state fails
closed as corruption. The collector still makes no quorum or restart decision.

A pure evidence value authorizes a `latest` checkpoint inventory event only
when its reporter is valid for the committed rank assignment and the reporter's
persisted acknowledgement succeeded for the same agent incarnation and flushed
step while naming the event's exact canonical digest. Missing or failed
acknowledgements, foreign topology or rank identity, candidate/verified events,
and reused event IDs with different bytes are not latest-preparation evidence.
This value does not select a manifest or recovery fallback.

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

## Torchrun Rendezvous Integration

### Node states

```text
STANDBY -> ADMITTED -> ACTIVE -> PREPARING -> RESTARTING
     \          \         \          \-> FAILED
      \----------> REJECTED             \-> QUARANTINED
```

- `STANDBY`: agent is registered; no trainer process is running.
- `ADMITTED`: coordinator assigned a logical slot for the next generation.
- `ACTIVE`: local workers are running for the committed generation.
- `PREPARING`: workers observed an intent, quiesced, and are performing bounded
  checkpoint preparation; the stock agent still sees healthy worker processes.
- `RESTARTING`: the plan is committed and torchrun is stopping, rendezvousing,
  or relaunching the worker group.
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

The internal control-store boundary exposes opaque byte values with per-key
compare-and-set revisions. Creation requires the key to be absent; replacement
and deletion require the exact current revision. Revisions never repeat for a
key, including after deletion and recreation, so a delayed coordinator cannot
mistake a recreated record for an older value with identical bytes. Higher
coordinator records add lease fencing and run/schema namespaces in subsequent
layers rather than weakening this storage primitive. The store also retains a
durable `has_history(key)` bit after deletion, allowing readers to distinguish a
never-created authoritative key from one that was removed. `get_history(key)`
returns every committed value entry in mutation order, including overwritten
values and values from earlier key lifetimes. Deletes do not invent a payload;
their revision, mutation, transaction, and lifetime effects remain visible as
gaps between adjacent immutable entries. Failed operations append nothing.
Each retained coordinator-lease value is decoded into an immutable authority
record only after validating canonical run-scoped lease bytes, authoritative
commit time, unguarded lease storage, and possible mutation/value/lifetime
sequence lineage. A lease key's mutation count also cannot exceed the
store-global transaction sequence. Complete history traversal and
adjacent-transition validation build on this strict value boundary.
The history reader obtains a stable immutable snapshot, requires the live value
to be its final retained entry, requires empty/nonempty values to agree with the
durable history marker, and validates every adjacent renewal, replacement, and
delete/recreate transition. A never-created lease has empty history; retained
history with no live lease fails closed because the current store contract has
no authoritative tombstone proving which retained value was deleted last.
Renewals must commit before expiry, in-place replacements cannot overlap, no
value or lifetime may be skipped, and lease identities and fencing tokens
cannot recur after replacement. Higher layers can therefore reconstruct live
coordinator authority after process loss without trusting process-local
observations.

Coordinator state that spans multiple keys uses one guarded store transaction.
The transaction first checks the coordinator lease revision, then any
side-effect-free revision conditions, every target revision, and one
authoritative store-time window under the same lock. Condition keys cannot also
be the guard or transaction targets. It publishes all target values with the
same commit timestamp and store-stamped guard key, guard revision, guard-value
digest, ordered guard mutation and value sequences, guard-key lifetime sequence,
authoritative guard commit time, and one store-global transaction sequence, or
publishes none. All values in one atomic transaction share that sequence.
Successful single-key mutations receive their own sequence, deletes consume a
sequence, and rejected operations consume none. Readers use this ordering to
prove cross-key causality when multiple commits have the same millisecond
timestamp. The value sequence changes whenever the guarded bytes change or the
key is recreated, but remains stable across same-value renewals. Store entries
also reject mutation, value, and lifetime sequence combinations that cannot
arise from create, update, delete, and recreate operations. This is the
primitive used to create an immutable generation snapshot and advance the
generation head without allowing a stale or expired coordinator to commit
either half alone. Revision conditions also let later components prove that the
generation head did not advance while publishing an intent, without rewriting
the head or weakening its immutable history checks.
Create-once transaction targets may additionally require that the key has no
prior committed history, not merely that it is currently absent. This prevents
a deleted immutable quarantine or snapshot key from being recreated inside an
otherwise valid atomic commit.

Persisted generation state uses strict versioned records.
`GenerationSnapshotRecord` schema version 2 contains the immutable
`RankAssignment`, the previous snapshot digest, and the complete coordinator
lease identity, duration, and opaque fencing token that committed it. Version 1
is rejected explicitly rather than being interpreted as the expanded layout.
Generation zero must not name a predecessor, while every later generation must
carry a lowercase SHA-256 predecessor digest. A `GenerationHeadRecord` contains
only the run, generation, and canonical snapshot digest. Both records reject
missing, unknown, duplicate, or unsupported fields.

The generation-state reader derives snapshot, head, and coordinator-lease keys
from one run-scoped namespace. It requires the head and snapshot to agree on
generation, digest, authoritative commit time, and store-stamped guard
provenance, then verifies the complete predecessor digest, timestamp,
guard-mutation, and lease-identity chain back to generation zero. Lease
identities cannot reappear after a different acquisition, and one acquisition
cannot change coordinator identity, lease duration, or store-stamped guard-key
lifetime. Every snapshot entry must retain store-stamped mutation, value, and
lifetime sequence `1`, while the head mutation and value sequences must equal
`generation + 1`; this rejects replacement, recreation, extra rewrites, and
rollback. The mutable head and referenced snapshot must share one store-global
transaction sequence, and snapshot transaction sequences must advance strictly
through the predecessor chain. Every generation commit must fall within the
stamped lease grant and duration. The head must remain in its first store-key
lifetime, and a stable immediate successor snapshot beyond the head is
contradictory state for every read API. Replacing a lease in place must not
overlap the prior lease's stamped
expiry, and one fencing token cannot identify multiple guard mutations. Fencing
revisions are compared only for equality; ordering comes from the store-stamped
guard mutation sequence. Each guard-key lifetime increase requires at least one
delete and one create mutation, including before the first persisted generation.
Because deletion advances mutation sequence without advancing value sequence,
every entry and adjacent transition must also satisfy the corresponding
value-sequence upper bound. When generations omit intervening renewals of one
lease, the mutation distance bounds the latest grant time that a valid renewal
chain could reach, so skipped valid renewals are accepted while resurrection
after expiry is rejected. A same-key takeover with unobserved intervening
mutations is ambiguous and rejected. An older generation commit cannot postdate
the next guard mutation that fences it. Opaque fencing tokens cannot reappear
after another guard mutation. One lease identity must retain one guard value
sequence, so changing A to B and back to A cannot be misread as skipped
renewals. The durable history of the generation-head key is the run
initialization marker, so deleting both the head and generation zero cannot make
an initialized run appear empty. Grant times, guard mutation/value sequences,
and key-lifetime sequences cannot move backward. Missing or contradictory
history is corruption, not an empty or partially usable assignment.

The generation-state manager initializes generation zero only after the reader
proves the run has no committed or deleted head history. It commits the mutable
head and one create-only immutable snapshot in the same lease-guarded
transaction. A successor must name the exact current head, advance by one
generation, preserve active-node count, local world size, logical rank ranges,
topology digest, and the logical slot of every surviving node, and link to the
predecessor's canonical snapshot digest. Concurrent initializers or successor
writers therefore produce one committed winner; stale leases, stale heads,
expired store-time windows, partial target conflicts, and unexpected
transaction results fail closed.

Coordinator ownership is stored under a schema-versioned, run-scoped lease key.
The lease record binds the run, a unique coordinator-process incarnation, a
unique lease acquisition, and the lease duration. The control store attaches
its authoritative commit time to every timed mutation. A held lease expires at
that commit time plus the persisted duration, so network delay before the CAS
cannot consume the granted lifetime. The record's opaque control-store revision
is the fencing token; acquisition, renewal, expiry takeover, release, and
reacquisition therefore produce distinct tokens, while the store's mutation
sequence supplies ordering. Every later authoritative coordinator mutation must
carry the currently held token and fail closed when it is stale.

Lease time comes from one authoritative, nondecreasing control-plane clock
shared by all contenders. A clock that moves backward aborts lease operations.
Every lease mutation atomically requires store time to be no earlier than the
coordinator's observation, preventing a client-side timestamp from committing
future-dated ownership after a backward clock step. The client also rejects an
acquisition or renewal response if the commit-time-derived lease expired while
the response was in flight.
Expiry is inclusive: at `expires_at_unix_ms` the old holder may no longer renew,
and a contender may take over. Renewal uses a deadline-guarded compare-and-set
whose time predicate is evaluated atomically by that authoritative store; a
client-side precheck alone cannot resurrect a lease after network delay.
Takeover uses one store-time sample and is permitted only after the old lease's
derived expiry; the replacement then receives a full duration from that commit
time. Release uses the same guarded store-time window and cannot delete an
expired lease or mutate ownership after a clock regression. Retrying acquisition
with the same active `coordinator_id` revalidates and renews the existing record
at store time while preserving its lease ID, so the coordinator ID must uniquely
identify one live process incarnation and must not be reused after process
restart.

### Slot-aware rendezvous

`SlotAwareRendezvousHandler` implements the existing `RendezvousHandler`
interface and is registered through `torchrun.handlers`. It admits only nodes
named by the committed plan:

- `next_rendezvous()` blocks standby agents without creating trainers;
- blocked standbys maintain a registration lease or heartbeat;
- quarantined nodes remain blocked or receive a terminal rejection according to
  deployment cleanup policy, but are never admitted;
- a replacement node receives the failed node's logical slot;
- exactly `min_nodes` slots complete the rendezvous; and
- returned group rank is derived from logical slot, not arrival order.

The first runtime slice admits only the immutable generation-zero assignment.
It registers every agent, renews that registration while the handler is alive,
returns active nodes by logical slot, keeps unassigned agents blocked without
reporting them as waiting, and propagates one immutable run-scoped closure that
wakes parked standbys and releases registrations. It also clears any stale
node-local restart context before the initial generation starts. The join
timeout bounds initial generation formation and worker bootstrap only; after
generation zero exists, passive standbys remain parked until assignment,
explicit shared closure, local shutdown, cancellation, or registration failure.
Local `shutdown()` releases only that handler's resources; only `set_closed()`
publishes the run-scoped terminal closure. Registration heartbeats schedule
each renewal from the returned lease's remaining lifetime rather than from a
fixed interval. An assigned node ID is admitted only when its retained
registration history has the same local worker count and environment digest as
the current handler, so an incompatible process cannot reuse an expired node
registration. The first assigned node commits one immutable run-scoped workload
compatibility identity, and every other assigned node must match it; unassigned
standbys cannot create or replace that record. Initial registration backend I/O
does not hold the handler's ownership lock, so local shutdown remains bounded
while registration is stalled. A registration response arriving after local
shutdown is not installed or heartbeated; a daemon performs an explicit
best-effort release, with lease expiry as the conservative fallback. Initial
registration itself is bounded by the formation deadline.

Before publishing readiness, each assigned handler clears its stale node-local
restart context. Slot zero owns one durable, generation-scoped attempt head,
while all assigned slots publish attempt-scoped arrivals tied to their live
registration. Slot zero validates the complete registration set once and
atomically publishes one immutable completion proof conditioned on the shared
attempt head plus every arrival and registration revision. Other slots wait on
that proof instead of rereading every registration history, keeping control
store work linear in the active node count. Slot zero reloads its current
renewed registration immediately before each guarded completion attempt, so
heartbeat renewal cannot strand an otherwise complete barrier on a stale
fencing token. An incomplete attempt remains usable when a handler incarnation
is replaced; a completed attempt advances the shared head before the next
worker launch. Missing or unprepared assigned nodes therefore fail at the
formation deadline instead of launching a partial worker group. After
bootstrap completes, every handler revalidates the generation head before
returning its slot.

Registration release is best effort and bounded during local shutdown. If the
backend remains unavailable, cleanup returns without waiting indefinitely and
the registration expires under its existing lease. Global closure publication
also runs through a bounded daemon operation; local heartbeat and registration
cleanup still proceeds when the control store is unavailable.
Replacement generation admission, restart-context publication, and the positive
`num_nodes_waiting()` restart edge are enabled only after authoritative
restart-plan readback is integrated in the following slice.

All handler incarnations use one immutable generation-scoped `PrefixStore` for
PyTorch's worker bootstrap keys. Slot zero publishes one address/port pair and
later retries or replacement agent incarnations reuse it, so no process-local
attempt counter can split the worker group across namespaces. Bootstrap reads
are bounded by the remaining join deadline, so a missing slot-zero publisher
cannot strand other agents on the underlying store timeout. When the deployment
supplies an already running agent-owned bootstrap endpoint, the handler reports
`use_agent_store=True`; a generated endpoint reports `False` so the stock agent
creates the worker-side TCP store. After bounded bootstrap reads complete, every
returned store is restored to the shared configured join timeout so later stock
agent coordination does not inherit rank-specific remaining-deadline values.

Passive standbys are not reported by `num_nodes_waiting()`. After a plan
commits, the selected replacement becomes the only newly waiting node visible
to active agents. This is the restart edge: a random late node must not trigger
scale-up. This mechanism is used only when the next plan admits at least one
standby; it is not a generic restart signal for unchanged membership.

Before returning an admitted node from `next_rendezvous()`, the handler:

1. validates that the assignment belongs to the latest committed generation;
2. writes the node-local `RestartContext`;
3. returns the logical slot as the rendezvous group rank; and
4. returns exactly `min_nodes` as the group world size.

### Stock elastic agent behavior

No custom elastic agent is required. The existing agent behavior is used in two
ways:

1. a worker failure follows torchrun's normal failure-restart path; and
2. while workers are healthy, a positive `num_nodes_waiting()` result causes
   the agent to stop and re-rendezvous its local worker group.

For an accessible replacement, workers first quiesce and acknowledge the
`RestartIntent`. Only after the coordinator commits the plan does the handler
expose the selected replacement as waiting. Each active agent then observes the
same membership change and enters its standard restart path. A worker that
missed the preparation deadline is treated as unavailable and cannot influence
the committed checkpoint manifest. Correctness does not depend on agents
stopping in the same polling instant because the plan is already immutable and
the new rendezvous cannot complete without its assignments.

For an inaccessible node, surviving workers eventually fail or their agents
observe the normal failure path.
`SlotAwareRendezvousHandler.next_rendezvous()` blocks the survivors until the
coordinator commits a conservative plan and assigns a standby. A node-local
SCOUT report never directly manipulates `num_nodes_waiting()` or chooses the
next membership.

The handler cannot customize torchrun's worker-stop signal or timeout for an
individual incident. Version 1 therefore requires workers to quiesce before a
planned membership restart, relies on the agent's configured bounded shutdown,
and treats failure to complete the next rendezvous before the plan deadline as
a failed restart. If a supported torchrun version cannot satisfy those
constraints, that version requires an out-of-tree agent fallback.

On successful job completion or terminal failure, the coordinator closes the
handler generation. Blocked standbys observe closure, release their leases, and
exit without starting trainers.

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
- Control-client intent polling is bounded.
- The stock agent's existing rendezvous wait-count polling is unchanged.
- Standby agents do not allocate model state or join training collectives.
- Checkpoint flush and transfer happen only after a restart intent.
- Control delivery backpressure must not block the training thread
  indefinitely.

## Recovery Policy

| Incident | Node action | Recovery mode |
|---|---|---|
| Attributed SDC on one node/GPU | Quarantine affected node conservatively | `RECOVERY_VERIFIED` |
| Inconclusive exact replay | Do not over-attribute; replace only a policy-defined conservative scope or fail closed | `RECOVERY_VERIFIED` |
| Required node inaccessible | Mark failed and replace its slot | `RECOVERY_VERIFIED` |
| Confirmed accessible compute or communication straggler | Drain and replace when policy threshold is met | `LATEST_GEMINI` if a complete generation is prepared; otherwise verified |
| Hang with complete clean emergency replay and all ranks accessible | Replace a policy-selected scope | `LATEST_GEMINI` |
| Hang with missing, stale, incomplete, or contradictory evidence | Replace a conservative policy-defined scope or fail closed | `RECOVERY_VERIFIED` |
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

The persisted `RecoveryManifestRecord` contains the canonical manifest and the
digest of its immutable source-generation snapshot. This record is create-once
and exposes its own canonical digest for the later restart-plan envelope.
Resolving the source snapshot and validating inventory, certification,
acknowledgement, assignment, and copy eligibility remain separate admission
steps; constructing the record alone does not make a manifest safe to commit.
`ResolvedRecoveryManifest` binds the record to one verified immutable
generation snapshot and requires exact run, source-generation, topology, and
snapshot-digest agreement. It still does not establish completeness or trust.

The manifest is usable only when every required rank has at least one copy and
every included copy is eligible for the manifest's exact positive step and the
plan's selected source. The coordinator must omit rejected alternatives before
commit; an immutable committed manifest cannot advertise an incomplete,
wrong-source, or unproven fallback. GEMINI plans use owner or peer copies;
durable plans use durable copies. Only shared or remote storage is independent
of holder-node availability. The manifest's trust label is not
self-authenticating: every selected copy must match its referenced inventory
record, and `RECOVERY_VERIFIED` manifests may use only recovery-verified
inventory evidence. Durable recovery always requires a recovery-verified
manifest and recovery-verified inventory, even if the originating intent
allowed latest recovery.

An in-memory or node-local copy is selectable only when its holder remains in
the next assignment. Process-memory copies are never selectable because stock
elastic restart destroys every worker address space, including workers on
retained nodes. Successful preparation must publish new node-local inventory
for the flushed bytes. A departing node's transfer counters in `RestartAck` do
not by themselves prove that a destination has the bytes. Version 1 requires
such transfers to materialize as authenticated shared or remote copy
provenance before plan commit; a future local-destination transfer record may
extend this without weakening the holder rule.

The coordinator resolves every rank through the immutable `RankAssignment` for
the manifest's `source_generation`. An `owner` copy must be held by the node
that owned that rank in that generation; a `peer` copy must be held elsewhere.
The source assignment's run, generation, topology digest, and world size must
match the manifest and plan. When source and current generations are equal,
their assignments must be identical rather than merely share metadata.

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
4. Slot-aware rendezvous assigns deterministic rank ranges; parked standbys
   remain blocked in `next_rendezvous()`.
5. Agents start workers with generation `0`.
6. `lm-resiliency` registers the worker identity and framework topology digest.

### Accessible straggler replacement

1. SCOUT emits a confirmed straggler report and latest-checkpoint proposal.
2. The coordinator maps the reported rank to its generation and node.
3. The coordinator opens a restart intent and provisionally reserves a
   compatible standby for the affected logical slot.
4. Every accessible worker enters `PREPARING`, quiesces at a safe boundary, and
   acknowledges the intent.
5. Workers flush eligible GEMINI state and report transfer results.
6. The coordinator commits a restart plan using the newest complete allowed
   generation, or falls back to verified recovery.
7. `SlotAwareRendezvousHandler` exposes the selected standby through
   `num_nodes_waiting()`.
8. Stock elastic agents observe the membership change and restart their local
   worker groups.
9. The source node becomes quarantined or standby according to policy.
10. The replacement and surviving nodes rendezvous with the same slots/ranks.
11. Relaunched workers validate the restart context and load exactly the
    committed rank-consistent step.
12. TorchTitan restores the complete state and resumes.

### Inaccessible node

1. The agent or infrastructure marks one node unavailable.
2. The coordinator opens an intent so surviving nodes can expose their
   completed peer replicas.
3. The coordinator selects `RECOVERY_VERIFIED`.
4. The committed manifest uses peer replicas or the durable verified
   checkpoint.
5. A standby inherits the failed logical slot.
6. Surviving agents follow torchrun's failure-restart path and block in
   slot-aware rendezvous until the plan is committed.
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
- **Job closes while standbys are parked:** close the rendezvous generation and
  wake blocked standby agents so they exit without spawning workers.

## Packaging and Compatibility

- Keep the public neutral payloads in `lm_resiliency.manager_api`.
- Put torchrun-specific code under `lm_resiliency.integrations.torchrun`.
- Keep TorchTitan imports lazy.
- Importing `lm_resiliency` must not import TorchTitan or CUDA-only packages.
- Version every wire payload independently from the Python package version.
- Reject unknown required fields or unsupported schema versions.
- Implement standby admission through the documented `RendezvousHandler`
  interface and `torchrun.handlers` registration mechanism.
- Do not subclass protected elastic-agent methods.
- Qualify the handler against every supported PyTorch minor because the stock
  agent's membership-restart behavior remains a runtime compatibility
  dependency.
- Preserve the existing `OrchestrationHooks`, `RecoveryDecision`,
  `SCOUTFaultReport`, and framework entry points.
- Use PyTorch 2.13 as the initial prototype baseline, then separately qualify
  the other PyTorch minors in the supported compatibility range.

### Proposed command

The custom rendezvous backend makes replacement-only semantics explicit without
changing torchrun's default behavior:

```toml
# /shared/lm-resiliency/torchrun.toml
schema_version = 1
control_endpoint = "control.internal:443"
replacement_only = true
max_replacement_generations = 2
registration_lease_duration_ms = 30000
poll_interval_ms = 1000
join_timeout_ms = 300000
```

The shared policy file contains no node identity or credentials. Every agent
loads the file, applies any explicitly supplied shared `--rdzv-conf`
overrides, and registers the canonical runtime digest over the resolved policy,
run ID, rendezvous endpoint, and `min_nodes`/`max_nodes` range. Drift in
whitespace or node-local settings does not matter, while drift in effective
shared settings or fleet-size semantics fails closed.
Node-specific `node_id` and `restart_context_path` values come from
`--rdzv-conf` or `LM_RESILIENCY_NODE_ID` and
`LM_RESILIENCY_RESTART_CONTEXT`; conflicting sources are rejected. Credentials
and other secrets remain in deployment-provided environment or credential
providers and are not included in the policy file or digest.
Because PyTorch does not include `nproc_per_node` in
`RendezvousParameters`, the agent also requires `local_world_size` and the
deployment-generated workload `environment_digest` through `--rdzv-conf` or
`LM_RESILIENCY_LOCAL_WORLD_SIZE` and
`LM_RESILIENCY_ENVIRONMENT_DIGEST`. The scheduler or deployment integration
must also provide the node's trusted GPU, NIC, HCA, and link identifiers as a
semicolon-delimited `resource_ids` value or `LM_RESILIENCY_RESOURCE_IDS`; use
`[]` for an explicitly empty inventory. These per-node IDs are stored in the
immutable `AgentIdentity` used to authenticate SCOUT hardware reports, but are
excluded from the shared compatibility digest because each node owns different
resources. The registered runtime digest includes the local worker count. The
immutable agent environment digest combines that runtime digest with the
deployment workload digest, so either runtime or software/capability drift
produces a different agent identity.

```bash
export LM_RESILIENCY_NODE_ID="${SCHEDULER_NODE_ID}"
export LM_RESILIENCY_LOCAL_WORLD_SIZE=8
export LM_RESILIENCY_ENVIRONMENT_DIGEST="${WORKLOAD_ENVIRONMENT_DIGEST}"
export LM_RESILIENCY_RESOURCE_IDS="${GPU_UUIDS_SEMICOLON_DELIMITED}"
export LM_RESILIENCY_RESTART_CONTEXT="/run/lm-resiliency/${JOB_ID}/restart-context.json"

torchrun \
  --nnodes=8:10 \
  --nproc-per-node=8 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint="${RDZV_ENDPOINT}" \
  --rdzv-id="${JOB_ID}" \
  --rdzv-conf="config=/shared/lm-resiliency/torchrun.toml" \
  --module torchtitan.train ...
```

The command runs on all `max_nodes` machines. The backend interprets
`min_nodes` as the exact active-slot count and `max_nodes` as the registered
fleet limit. Selecting `--rdzv-backend=lm_resiliency` is the explicit opt-in;
the built-in rendezvous backends retain their existing elastic semantics.

## Validation Plan

### Contract tests

- JSON round-trip and schema-version rejection for every event and plan.
- Stale generation, duplicate event, and conflicting-plan rejection.
- Conservative recovery lattice across missing and conflicting proposals.
- Rank-to-node mapping through immutable generation snapshots.
- Quarantine by stable node ID, never by rank alone.
- Plan validation for intent fencing, unexpired deadlines, exact active and
  worker counts, unique slots, topology digest, source-compatible copies, and a
  complete single-step checkpoint manifest.

### CPU multi-process tests

- Standby agents do not spawn trainer workers.
- A new standby does not cause scale-up.
- Parked standbys do not contribute to `num_nodes_waiting()`.
- A committed plan exposes only the selected replacement as waiting.
- The stock agent membership path restarts every active local worker group.
- A replacement inherits the same logical slot and rank range.
- A quarantined node cannot rejoin.
- Closing the job wakes parked standbys and starts no trainer processes.
- The node-local `RestartContext` is written before `next_rendezvous()` returns.
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

### Use a custom rendezvous handler without a recovery coordinator

Insufficient. The handler provides standby admission, stable ranks, and the
membership-restart edge, but it does not evaluate SCOUT evidence, prepare
GEMINI state, select a complete checkpoint manifest, or decide quarantine.

### Add a custom elastic agent

Not selected for version 1. PyTorch exposes agent extension points, but the
existing membership-change path already provides the required full-group
restart after the rendezvous handler reveals a selected replacement. A custom
agent would add version coupling without removing the need for the coordinator
and worker preparation protocol.

### Add a new upstream torchrun resiliency-policy interface

Not required. The existing `RendezvousHandler` interface and stock elastic
agent cover admission and restart mechanics. Reconsider an upstream change only
if focused validation finds a correctness requirement that those interfaces
cannot satisfy.

### Allow the active world to shrink after a failure

Deferred. It changes framework parallelism, checkpoint sharding, batch size,
optimizer semantics, and potentially the training trajectory. It requires a
separate elastic-topology and resharding contract.

### Put placement policy inside SCOUT or GEMINI

Rejected. SCOUT and GEMINI do not own scheduler state, leases, or physical
resource lifecycle. They should provide evidence and recovery state to a
coordinator or infrastructure manager.

### Delegate replacement decisions to an external scheduler

Viable for production, and the neutral manager API should continue supporting
it. The reference design still adds value by defining the rendezvous and
recovery contracts needed for local standby replacement.

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
- Add the `torchrun.handlers` rendezvous plugin.
- Add standby-only admission, stable slot assignment, and restart-context
  publication.
- Use the stock `LocalElasticAgent` membership-restart path.

### Phase 3: framework validation

- Validate native PyTorch and TorchTitan end to end.
- Add Megatron Core and DeepSpeed after the neutral contract is stable.

### Phase 4: production hardening

- Validate handler and agent behavior across every supported PyTorch minor.
- Add durable coordinator failover and scheduler integration.
- Exercise repeated replacement generations and bounded shutdown failures.

## Open Questions

Recommended defaults for the first prototype:

- pre-launch standby agents in the same allocation;
- require stable logical slots and exact rank-range inheritance;
- quarantine the whole node when one full-node torchrun local worker group is
  the scheduling unit;
- prefer healthy peer or durable verified copies over a suspect node;
- keep framework durable checkpoint IDs opaque;
- use a separate replacement-generation budget; and
- use the existing rendezvous plugin and stock agent before considering any
  torchrun change.

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
8. **Compatibility fallback:** if a supported PyTorch minor changes the stock
   membership-restart behavior, should that minor be excluded or supported by
   a version-specific out-of-tree `SimpleElasticAgent` launcher?

## References

- [GEMINI operational contract](gemini.md)
- [SCOUT operational contract](scout.md)
- [LM Resiliency API guide](api.md)
- [Compatibility policy](compatibility.md)
- [PyTorch torchrun elastic launch documentation](https://docs.pytorch.org/docs/2.13/elastic/run.html)
- [PyTorch elastic agent documentation](https://docs.pytorch.org/docs/2.13/elastic/agent.html)
- [PyTorch rendezvous documentation](https://docs.pytorch.org/docs/2.13/elastic/rendezvous.html)
