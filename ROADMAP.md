# Roadmap

This roadmap describes the intended direction of `lm-resiliency`.
It is a planning document, not a commitment to specific release dates.
Priorities may change based on evaluation results, contributor interest, upstream
framework changes, and access to representative hardware.

## Mission

Extend the project's core capabilities across the model lifecycle:

- detect silent data corruption, stalls, hangs, and performance degradation;
- localize failures to actionable components;
- recover from trusted state with minimal lost work; and
- evaluate resiliency systems with reproducible, comparable fault campaigns.

The current release focuses on distributed LLM pre-training.
It provides SCOUT replay-based fault localization, GEMINI in-memory checkpointing,
framework integrations, and platform-neutral manager handoff.
See [SCOUT](docs/scout.md), [GEMINI](docs/gemini.md), and the current
[validation report](docs/validation.md) for the implemented contract and its
boundaries.

## Guiding Principles

- **Measurable guarantees:** Define the fault model, coverage boundary, recovery
  contract, and validation evidence for every feature.
- **Framework neutrality:** Keep fault reports, recovery decisions, and evaluation
  results portable across training and serving stacks.
- **Workload-aware protection:** Use model structure, execution topology, and
  framework state instead of treating an LLM workload as an opaque process.
- **Incremental adoption:** Preserve existing training and serving loops wherever
  possible and avoid requiring a framework fork.
- **Explicit trust:** Distinguish recent state from state that has been independently
  verified as safe for recovery.
- **Reproducibility:** Publish campaign definitions, environment manifests, raw
  results, and known limitations.

## Priority Overview

| Horizon | Initiative | Intended outcome |
|---|---|---|
| Near term | Training manager integration | Automated launch, restart, quarantine, replacement, and recovery coordination |
| Near term | Fault injection evaluation kit | Framework-aware fault campaigns with localization ground truth |
| Near term | Large-scale evaluation | Evidence beyond the current 16-GPU, two-host validation boundary |
| Mid term | Post-training resiliency | Trusted recovery for SFT, preference optimization, and distributed RL |
| Mid term | Inference fleet resiliency | Detection, quarantine, rerouting, and request-state recovery for serving fleets |
| Mid term | Expanded SCOUT coverage | Independent evidence for single-owner shards and additional execution backends |

## Training Manager Integration

Integrate worker-local detection and recovery state with an external training
manager that can act on faults across the job and cluster.

### Planned Capabilities

- Add high-performance checkpoint replication and replacement transfer backends.
- Provide a reference training manager integration built on `OrchestrationHooks`.
- Launch and restart distributed jobs with an explicit recovery decision.
- Map logical ranks and fault reports to GPUs, hosts, and other cluster resources.
- Demonstrate worker drain, quarantine, placement, and replacement flows.
- Coordinate checkpoint transfer before restart or replacement when the source
  remains accessible.
- Integrate normalized events with common metrics, tracing, and alerting systems.
- Add stable fault and recovery reason codes for dashboards and automation.
- Correlate SCOUT evidence with GPU telemetry, NCCL diagnostics, and deployment
  resource maps.
- Validate complete framework durable fallback after fail-stop recovery.

SCOUT remains responsible for detecting and localizing suspected faults.
GEMINI remains responsible for exposing recoverable checkpoint state.
The training manager owns job launch, restart, placement, resource quarantine,
and replacement coordination. Physical hardware repair or replacement remains an
infrastructure operator responsibility.

## Fault Injection Evaluation Kit

Create an independently enabled feature within `lm-resiliency` for injecting
faults through supported training frameworks. The kit records the exact injected
fault as ground truth, then a campaign evaluates whether the resiliency system
under test detects and localizes that fault correctly.

The injector should not depend on SCOUT or GEMINI being enabled, and it should
not require adapters for individual resiliency systems.

### Planned Capabilities

- Extract and generalize the current test-only fault injection helpers.
- Support injection through PyTorch, TorchTitan, Megatron Core, and DeepSpeed
  framework integrations.
- Provide a declarative campaign format for targets, triggers, duration,
  persistence, probability, seeds, and expected outcomes.
- Support deterministic, intermittent, probabilistic, and persistent faults.
- Cover several fault classes:
  - numerical corruption, including bit flips, NaN/Inf values, scaling, noise,
    and sign changes;
  - stale, dropped, duplicated, or corrupted gradients and optimizer state;
  - model, checkpoint, RNG, sampler, and input-pipeline corruption;
  - compute, collective, DataLoader, and checkpoint-I/O delays;
  - process hangs, termination, worker loss, and node loss; and
  - communication faults that can be injected safely in a controlled environment.
- Separate safe in-process simulation from destructive cluster-level injection.
- Verify that an injection took effect before scoring the system under test.
- Record the expected faulty rank, device, node, layer, operation, or endpoint as
  campaign ground truth.
- Accept a neutral localization result from the resiliency system under test and
  compare it with the injected-fault ground truth.
- Emit a common machine-readable result format.
- Record the software, hardware, topology, model, workload, seed, and injection
  parameters needed to reproduce a campaign.

### Common Evaluation Results

Each campaign should report:

- whether the fault was injected successfully;
- the injected fault type, target, trigger, and duration;
- whether it was detected;
- whether the faulty rank, device, node, or endpoint was localized correctly;
- any incorrect or additional fault attribution; and
- the observed detection and localization latency.

### Initial Success Criteria

- Run the same campaign manifest across at least two supported training
  frameworks.
- Evaluate SCOUT localization without coupling the injector to SCOUT internals.
- Allow another resiliency system to consume the same injected workload and
  submit its localization result without adding a system-specific injector
  adapter.
- Reproduce a campaign from a checked-in manifest and seed.
- Produce a comparable JSON summary containing injection ground truth and the
  submitted localization result.
- Include explicit safety controls for destructive injections.

## Large-Scale Evaluation

Extend validation beyond tiny models, synthetic token data, 16 GPUs, and
replay-only production-loop injection.

### Scale and Workload Coverage

- Maintain the current 16-GPU campaign as a frequent qualification tier.
- Add reproducible 32- to 64-GPU release qualification.
- Run periodic 128-GPU and larger campaigns when suitable infrastructure is
  available.
- Evaluate representative dense and MoE workloads with realistic model state,
  datasets, sequence lengths, and checkpoint sizes.
- Include longer clean runs to measure false-positive behavior and performance
  stability.
- Exercise larger tensor, pipeline, context, data, and expert parallel topologies.

### Failure and Recovery Coverage

- Exercise actual worker termination, process-group failure, scheduler relaunch,
  node replacement, and framework durable fallback.
- Validate recovery after simultaneous compute and storage failures.
- Evaluate recurring intermittent faults, not only permanent or deterministic
  faults.
- Verify end-to-end convergence or training-trajectory equivalence after recovery.
- Test multi-rack placement and failures that cross network or failure domains.

### Runtime and Transport Coverage

- Qualify RDMA-capable checkpoint replication and transfer implementations,
  including deployment-specific EFA or NIXL paths where available.
- Evaluate FP8, Transformer Engine, CUDA Graphs, compilation, communication
  overlap, and production attention or MoE kernels.
- Measure interference between training collectives, checkpoint replication,
  replay, and recovery traffic.

### Published Metrics

- detection recall and false-positive rate;
- localization accuracy and confidence;
- time to detect, localize, select recovery, and resume;
- rollback distance and lost accelerator time;
- recovery goodput and steady-state overhead;
- checkpoint capture and transfer latency;
- state equivalence and post-recovery convergence; and
- infrastructure, software, and topology details.

## Post-Training Resiliency

Extend protection from pre-training to supervised fine-tuning, preference
optimization, distillation, quantization-aware training, and reinforcement
learning.

### Stage 1: Training-Like Post-Training

- Support supervised fine-tuning, distillation, DPO-style preference optimization,
  and quantization-aware training.
- Capture and restore model, optimizer, scheduler, sampler, RNG, and dataset
  position with the same trust rules used for pre-training.
- Qualify parameter-efficient fine-tuning and mixed frozen/trainable state.
- Validate recovery equivalence for packed sequences and multi-dataset sampling.

### Stage 2: Distributed RL and RLVR

- Support PPO, GRPO, RLVR, and asynchronous actor-rollout architectures.
- Define consistent recovery state across trainers, rollout workers, reference
  models, reward models, and inference engines.
- Protect generated trajectories, replay buffers, advantages, reward-normalization
  state, KL controllers, task queues, and sampler positions.
- Track trainer-to-rollout model versions and prevent stale trajectories from
  entering an update after recovery.
- Detect duplicated, omitted, or corrupted prompts and trajectories.
- Introduce a cross-component recovery transaction so all roles resume from a
  mutually consistent generation.

Initial framework candidates include TorchTune, TorchTitan post-training or RL
components, and distributed RL stacks such as verl. Selection should favor the
smallest integration that exercises the required consistency model.

### Success Criteria

- Resume without duplicating or omitting optimization samples.
- Preserve model, optimizer, scheduler, data, and algorithm-specific state.
- Reject stale rollout data after trainer recovery.
- Demonstrate fault injection, detection, restart, and continued learning in a
  real post-training workflow.

## Inference Fleet Resiliency

Adapt SCOUT and GEMINI ideas to online and batch inference without assuming that
serving replicas behave like synchronous training peers.

### Stage 1: Detection and Quarantine

- Run deterministic canary or sampled shadow requests across equivalent replicas.
- Compare logits, token distributions, or stable signatures rather than sampled
  text when decoding is nondeterministic.
- Detect silent output divergence, stale weights, corrupted KV-cache blocks,
  compute stragglers, collective faults, and worker hangs.
- Localize faults within tensor-, pipeline-, data-, or expert-parallel serving
  groups.
- Drain and quarantine suspect replicas through a manager-neutral API.

### Stage 2: Traffic Recovery

- Reroute or retry affected requests on healthy replicas.
- Preserve request identity and prevent duplicate externally visible results.
- Support safe worker reload and canary-based reintegration.
- Coordinate model-version and adapter-version consistency across the fleet.

### Stage 3: Stateful Request Continuation

- Resume long-running generation from token or block boundaries.
- Evaluate replication or compact checkpointing of KV-cache and request progress.
- Protect reusable prefix caches against stale or corrupted entries.
- Recover pipeline- or tensor-parallel request state after worker replacement.
- Bound the memory, bandwidth, latency, and privacy costs of request-state
  protection.

### Serving Metrics

- incorrect-output and detected-divergence rate;
- request success and retry rate;
- time to first token and inter-token latency;
- wasted or regenerated tokens;
- request recovery latency;
- fleet capacity lost during quarantine; and
- steady-state protection overhead.

An initial integration should target one serving engine before adding multiple
orchestration environments.

## Expanded SCOUT Coverage

Address current attribution and execution-coverage boundaries.

### Independent Evidence for Single-Owner State

Pure FSDP and ZeRO-3 owner shards do not have an independent live copy that can
identify the faulty owner. Investigate:

- sampled shadow recomputation;
- parity or erasure-coded state;
- low-rate redundant parameter or optimizer fragments;
- trusted checkpoint comparisons; and
- hardware-assisted or cross-step integrity evidence.

Any approach must state its detection latency, storage cost, compute overhead,
and common-mode failure boundary.

### Additional Replay Surfaces

- FP8 and quantized training paths;
- production attention and fused optimizer kernels;
- compiled and CUDA Graph execution;
- additional Transformer Engine kernels;
- heterogeneous MoE routing and larger expert-parallel topologies; and
- framework-internal collectives that bypass public PyTorch boundaries.

### Fabric-Aware Localization

- Map ranks to GPUs, NICs, HCAs, links, switches, racks, and hosts.
- Combine replay and collective evidence with DCGM, NCCL, system-log, and fabric
  telemetry.
- Distinguish compute, local-link, host-network, and shared-switch failures.
- Preserve contradictory evidence and confidence in manager-facing reports.

## Additional Research Directions

- Automatically derive compact replay catalogs for attention, MoE, optimizer, and
  compiler-generated execution regimes.
- Use coded computation or parity to reduce the cost of independent references.
- Support elastic restart with a changed world size or parallelism layout.
- Verify checkpoint and recovery invariants across framework version upgrades.
- Develop resilience service-level objectives that jointly optimize overhead,
  detection latency, rollback distance, and recovery time.
- Create anonymized, portable fault traces for repeatable research comparisons.

## Scope Boundaries

- Fault injection results apply only to the documented fault model and campaign
  environment.
- Sampled replay is not continuous duplication and cannot guarantee detection of
  every one-shot transient.
- Strict peer consensus cannot identify a common-mode failure without an
  independent reference.
- Cluster scheduling, process replacement, traffic routing, and hardware repair
  remain external responsibilities, even when the project supplies reference
  integrations.
- Initial inference work should prioritize detection and safe quarantine before
  claiming transparent request continuation.
- Large-scale targets depend on access to suitable infrastructure and should not
  weaken the reproducibility requirements for smaller qualification tiers.

## Contributing

Roadmap contributions should define:

1. the failure model and intended guarantee;
2. the supported workload and topology;
3. the recovery or manager contract;
4. measurable success criteria;
5. a reproducible validation plan; and
6. known limitations and unsupported cases.

See [CONTRIBUTING.md](CONTRIBUTING.md) for development and pull-request
requirements.
