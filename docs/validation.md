# Validation

Date: 2026-08-13 UTC.

This report summarizes the release-candidate validation evidence for GEMINI and SCOUT.
Focused integration programs use deterministic training workloads and fault injection to verify checkpoint equivalence, exact fault localization, candidate exclusion, and framework topology handling.

## Automated Qualification Status

The [GPU Qualification workflow](https://github.com/LMResiliency/lm-resiliency/actions/workflows/gpu-qualification.yml) runs weekly and on trusted maintainer dispatch using a two-GPU self-hosted runner. The workflow badge in the README links to the last run, including its exact revision and completion date. Each run also uploads a machine-readable summary, environment and topology inventory, exact commands, per-command logs, and SHA-256 checksums.

This frequent tier exercises single-GPU trajectory-equivalent recovery plus two-rank Gloo/NCCL replay, synchronized RNG, FSDP2 checkpoint recovery, and process-exit cleanup. It does not replace the larger release-qualification campaigns documented below. Self-hosted runner provisioning and isolation requirements are documented in the [test guide](../tests/README.md#automated-gpu-qualification).

## Release Baseline

| Check | Result |
|---|---|
| CPU unit tests | 634 passed |
| Clean PyPI quick start | Python 3.12 environment passed `pip check`; tiny CPU causal LM trained through step 4, recovered step 4, and continued through step 6 |
| Ruff | Passed |
| Python bytecode compilation | Passed |
| 4-rank two-host OOB checkpoint I/O | Delayed checkpoint write localized to rank 2 after two confirmations |
| 16-rank CUDA optimizer replay | Different rank-local Adam transitions converged under one source recipe; injected rank-2 replay corruption localized exactly |
| 4-rank two-host NCCL AllToAll replay | Uneven captured split metadata reconstructed; bounded balanced and cyclic-permutation matrices executed and verified on every rank |
| 4-rank two-host Cross-PG localization | Slow overlapping NCCL groups intersected at rank 0 and selected its mapped host |
| 16-rank native PyTorch current lifecycle | 3/3 passed across DDP, FSDP2, and HSDP |
| 16-rank DeepSpeed current lifecycle | 2/2 passed across ZeRO-1 and ZeRO-2 |
| 16-rank production training loops | PyTorch, TorchTitan, Megatron Core, and DeepSpeed completed ten-step real framework loops with exact remote-rank fault localization and recovery handoff |
| 16-rank framework topology matrix | 11/11 passed across PyTorch, TorchTitan, Megatron Core, and DeepSpeed |
| 16-rank Megatron optimizer lifecycle | 3/3 passed across dense, pipeline, and expert topologies |
| 8-GPU two-recipe checkpoint lifecycle | Candidate at step 2, recovery-verified at step 4, injected SDC at step 5 recovered step 2 |
| 4-GPU native PyTorch DDP, superseded checkpoint protocol | Dense two-cycle promotion and exact rank-1 SDC localization passed historically |
| 8-GPU multi-GPU recovery suite | 7/7 passed |
| 8-GPU save/load and signal-flush suite | 6/6 passed |
| 8-GPU TorchTitan Llama debug-model suite | 4/4 passed |
| 8-GPU Megatron integration suite | 11/11 passed, including exact two-cycle SDC recovery |
| 8-GPU DeepSpeed ZeRO-2 integration suite | 5/5 passed, including exact two-cycle SDC recovery |
| 16-rank native PyTorch matrix | 9/9 passed across DDP, FSDP2, and HSDP |
| 16-rank DeepSpeed matrix | 2/2 passed across ZeRO-1 and ZeRO-2 |
| 16-rank framework lifecycle matrix | 11/11 passed across PyTorch, TorchTitan, Megatron, and durable dense and expert cases |

The CPU suite covers checkpoint capture and recovery, replay preconditions, peer metadata, local persistence, integrity failures, recipe-cycle accounting, emergency scheduler preservation, recovery-mode selection, durable promotion, checkpoint I/O boundaries, hardware-health classification and callback delivery, representative AllToAll policy generation, FSDP materialization timing, framework adapters, and public lifecycle behavior.

The final distributed campaign used two hosts with eight A100 GPUs per host and 16 ranks over TCP on Amazon ENA.
Every current-lifecycle and focused feature job completed successfully.
The campaign directly validated structured C3 results, exact replay preconditions, immediate dense certification, embedding, hidden, output, and optimizer replay, source-broadcast optimizer recipes, representative AllToAll replay, fixed-shard HSDP groups, FSDP materialization timing, checkpoint-I/O localization, Cross-PG host selection, recovery-verified checkpoint handoff, global candidate rejection, and bitwise restart.

The campaign identified and corrected three distributed-only integration defects.
Explicit OOB TCP endpoints now take precedence over file rendezvous when a status directory is also configured.
FSDP parameter preconditions now run before post-forward reshard invalidates materialized storage, and pure-FSDP replay evidence is materialized globally before C3.
DDP boundary discovery now unwraps framework wrappers so embedding and output recipes remain visible.

The production-loop campaign identified and corrected three additional framework-integration defects.
Replay now restores the captured autocast context for mixed-precision framework loops.
SCOUT captures output gradients with tensor hooks instead of module backward hooks, avoiding conflicts with Megatron's custom autograd functions.
Replay transport now normalizes strided source tensors to owned contiguous storage before NCCL broadcast.

## Production Training Loops

The production-loop programs use real framework-owned training lifecycles and tiny causal language models.
Only token generation, logging, and external storage services are synthetic.
Each program completed ten optimizer steps on one eight-GPU host and on two eight-GPU hosts.
PyTorch, TorchTitan, and DeepSpeed used BF16, while Megatron Core used FP32.

At step 4, each campaign injected a transient hidden-layer fault only during SCOUT replay on the last global rank.
The live training forward remained unchanged so the same process could validate rejection and clean post-fault certification without a relaunch.
On two hosts, the injected rank was global rank 15 on the second host.

| Framework | Production path | Two-host result |
|---|---|---|
| PyTorch | Tiny causal LM through DDP forward, backward, and `AdamW.step()` | Exact rank-15 localization at step 4; checkpoint remained at step 3; application step, GEMINI, and all four SCOUT recipes reached clean step 10 |
| TorchTitan | Llama 3 debug model through `Trainer.train()` | Exact rank-15 localization at step 4; checkpoint remained at step 3; trainer, scheduler, dataloader, GEMINI, and all four SCOUT recipes reached clean step 10 |
| Megatron Core | `GPTModel`, Megatron DDP, distributed Adam, forward/backward schedule, `train_step()`, and `training.train()` | Exact rank-15 localization at step 4; checkpoint remained at step 3; iteration, consumed samples, GEMINI, and all four SCOUT recipes reached clean step 10 |
| DeepSpeed | Tiny causal LM through `DeepSpeedEngine.backward()` and `DeepSpeedEngine.step()` with ZeRO-2 | Exact rank-15 localization at step 4; checkpoint remained at step 3; engine global step, GEMINI, and all four SCOUT recipes reached clean step 10 |

Every manager-facing decision selected recovery-verified GEMINI step 3.
Each campaign completed six clean post-fault steps and certified step 10.
The same protocol passed on one host with rank 7 as the injected rank.

Native PyTorch does not prescribe a trainer.
The production example validates its application-owned DDP loop.
Dedicated lifecycle programs additionally validate FSDP2 and HSDP on the same two hosts.

The executable one-host and two-host `torchrun` commands are documented in [Production-loop examples](../examples/README.md).
The same examples enable the integration campaign directly with `--inject-fault`.

## Native PyTorch

The package-root `enable_resiliency()` API was validated with a deterministic three-block transformer and AdamW.

| Architecture | Topology | GEMINI recovery | SCOUT SDC | SCOUT straggler |
|---|---|---|---|---|
| DDP | One replica per rank across 16 ranks | Passed | Passed | Passed |
| FSDP2 | One-dimensional `dp_shard=16` mesh | Passed | Passed | Passed |
| HSDP | `dp_replicate=4`, `dp_shard=4` mesh | Passed | Passed | Passed |

GEMINI restored model, AdamW, caller-owned, CPU RNG, and CUDA RNG state bitwise in every topology.
DDP and FSDP2 peer replicas were byte-exact across hosts.
HSDP recovered local shards using its natural replicas.

SCOUT localized injected rank 9 exactly in every SDC and compute-straggler case.
Healthy controls remained clean.
Each SDC-contaminated step was excluded, candidate trust was rejected job-wide, and restart selected the exact prior recovery-verified state.
All three architectures exercised embedding, hidden, output, and optimizer recipes.
Replay-input and RNG preconditions agreed in every architecture.
HSDP additionally verified corresponding materialized parameter state within fixed-shard replica groups.
FSDP2 and HSDP retained parameter-AllGather timing as communication evidence.
The manager-facing RecoveryDecision selected GEMINI step 2 before restart.

Pure FSDP2 parameter-shard corruption without an independent replica remains outside the majority-comparison contract.
The FSDP2 campaign therefore injected an independently observable rank-local replay-output fault.

## Framework Parallelism

SCOUT compared equivalent peers while preserving each framework's model-parallel coordinates.
SDC cases required a clean control, exact injected-rank localization in each affected peer group, global candidate rejection, and a clean post-fault replay.
GEMINI cases required bitwise recovery of model, optimizer, caller-owned, and RNG state.

| Integration | Validated configurations | Result |
|---|---|---|
| PyTorch `DeviceMesh` | TP with PP, CP, and EP with expert TP | 3/3 passed with replay-precondition agreement and exact rank-15 localization |
| TorchTitan | HSDP with TP, CP, and PP; sparse EP with expert TP | 2/2 detected injected disagreement; the two-peer dense group correctly did not claim a unique culprit |
| Megatron Core | Dense TP and CP, PP with virtual model chunks, and EP with expert TP | 3/3 passed using native parallel-state groups and source-broadcast optimizer replay |
| DeepSpeed topology | TP with PP, Ulysses SP, and EP with expert TP | 3/3 passed with replay-precondition agreement and exact rank-15 localization |
| DeepSpeed ZeRO | ZeRO stages 1 and 2 | 2/2 passed source-broadcast optimizer replay, recovery handoff, and bitwise step-2 restart |

The current topology cases used live NCCL groups and each framework's native topology discovery.
The current environment did not include Transformer Engine, so the earlier production Megatron MoE campaign was not repeated.
That historical campaign remains documented below.

A delayed TP collective on rank 15 was classified as a communication straggler for ranks 14 and 15 without a compute-straggler or SDC false positive.

## Detection and Recovery Campaigns

The following campaigns predate the current structured C3 and immediate dense-certification contracts.
Their topology, detection, replay, and recovery results remain evidence, but they do not independently validate the current protocol.

The framework campaign used 16 A100-SXM4-40GB GPUs, PyTorch 2.13.0+cu130, NCCL 2.29.7, TorchTitan 0.2.2, Megatron Core 0.18.2, and DeepSpeed 0.19.4.
Inter-node traffic used TCP over Amazon ENA.

| Area | Result |
|---|---|
| 8-GPU peer replication | Exact peer contents, unequal rank layouts, metadata, disk reload, and 256 KiB and 16 MiB chunking passed |
| 8-GPU DeepSpeed | Capture, replay, corrupt/recover equivalence, and overhead smoke checks passed, 4/4 |
| Automatic exit cleanup | Native PyTorch DDP, TorchTitan, Megatron Core, and DeepSpeed ZeRO-2 closed GEMINI, removed SCOUT hooks, restored framework hooks, and exited before process-group teardown across 8/8 rank processes |
| 16-GPU dense training | Baseline, GEMINI, SCOUT, and combined runs matched loss and parameter trajectories in 20/20 trials with no false detections |
| 16-GPU replication matrix | 135/135 TCP cells passed across three checkpoint sizes, intervals, chunk sizes, and five seeds |
| SCOUT numerical faults | 344/344 forward and parameter faults were localized, including 64 near-invisible cases; 30/30 healthy controls stayed clean |
| SCOUT hangs and stalls | 120/120 primary launches and 30/30 focused out-of-band, DataLoader, and process-stall launches passed |
| Durable certification | 46 candidate, commit, rejection, and recovery tests passed |
| Production MoE replay | Routed BF16 Megatron MoE and Transformer Engine `TEGroupedMLP` forward and backward replay passed on 16 A100s |
| Production MoE faults | A clean 128/512-row cycle certified, then localized a persistent expert-weight fault on rank 15 at both shapes |
| EP All-to-All | Four EP=4 replicas spanning both hosts passed routed forward and backward, clean replay, and rank-15 localization |
| Expert tensor parallelism | Four EP=2, expert-TP=2 replicas passed routed forward and backward, clean replay, and rank-15 localization |
| Production MoE catalog | Five post-prime profiles per shape were stable and selected distinct physical plans; missing role evidence retained exact recipes |
| Environment soak | A 16-GPU, two-hour run reported no numerical, ECC, XID, kernel, or link errors |

The soak is an environment health check, not a fault-tolerance guarantee.

Production MoE validation used Megatron Core 0.18.2, Transformer Engine 2.10.0, PyTorch 2.10.0, CUDA 13.0, BF16, and top-2 routing.
The baseline and expert-TP campaigns used four experts; the EP=4 campaign used eight.
The routed layer exercised the router, dispatcher, `TEGroupedMLP`, backward path, inter-node EP All-to-All, and expert-TP AllGather and ReduceScatter.

## MoE Execution Regimes

The compression campaign used an instrumented Triton persistent grouped-GEMM with BF16 on A100-SXM4-40GB GPUs.
The environment used NVIDIA driver 580.126.16, PyTorch 2.13.0+cu130, CUDA 13.0, and Triton 3.7.1.

| Evidence | Result |
|---|---|
| Scalar compression | 3,457 exact scalar shapes compressed to 16 recipes across four regimes, a 99.54% reduction |
| Role coverage | 1,134,224 target obligations covered; 5,960/5,960 representative work-item injections passed |
| Distributed replay | All 16 representatives replayed cleanly on 8 and 16 A100s; persistent faults localized rank 7 and rank 15 |
| Negative challenges | Omitted roles forced exact fallback, wrong counts rejected compression, and environment drift rejected stale catalogs |
| Projection sensitivity | Six projections over 512, 1,024, and 2,048 shapes required 10-19 recipes |
| Grouped expert count | Physical execution of 1, 2, 4, 8, and 16 experts over 2,048 shapes required 12, 18, 27, 42, and 48 recipes |
| Grouped role injections | All 4,757 selected-role injections passed in each extended campaign and host-swapped repeat |
| Heterogeneous routing | Uniform cycles covered all tested `E=2` vectors at 512 shapes, but not arbitrary vectors for `E>=4`; at 1,024 and 2,048 shapes they did not cover arbitrary vectors for any `E>=2` |

See [MoE execution regimes](moe_execution_regimes.md) for the complete methodology, matrices, reproduction commands, and qualification boundary.

## Boundaries

- Multi-node transport used TCP over ENA; RDMA, EFA, NIXL, and line-rate performance were not validated.
- Scale beyond 16 GPUs was not tested.
- Production-loop validation uses tiny models, deterministic synthetic token data, and replay-only transient fault injection; it validates lifecycle integration, localization, candidate exclusion, recovery handoff, and post-fault certification, not publication-scale convergence or process relaunch.
- Production MoE coverage includes BF16 Megatron and Transformer Engine, EP up to 4, and expert TP up to 2.
- FP8, CUDA Graphs, communication overlap, and publication-scale repetitions were not validated.
- Transformer Engine recipes remained exact because independent CTA-role semantics were unavailable.
- Compression results apply only to the instrumented Triton backend and qualified environments.
- Uniform scalar catalogs are not qualified for arbitrary heterogeneous routing vectors.
- AllToAll validation covers the default bounded balanced and cyclic-permutation policy, not every possible traffic matrix.
- Software fault injection covers modeled semantic roles for permanent or reproducible triggers; one-shot transients and common-mode faults remain outside the result.
- Framework durable fallback after a fail-stop was not exercised.
- Scheduler policy, relaunch, placement, physical replacement, quarantine, and cluster observability remain outside this repository.
