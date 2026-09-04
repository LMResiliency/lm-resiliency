# Fault Injection Validation, 2026-09-04

This report records a two-stage manual qualification of the 21 canonical
`FailureType` effects. It distinguishes verified injection ground truth from
independent SCOUT localization.

## Result

| Question | Result |
|---|---:|
| Canonical failure effects in the repository | 21 |
| Effects exercised by the all-types pressure campaign | 21/21 |
| Effects with exact SCOUT localization across both stages | 6/21 |
| Stage 1 injected occurrences/actions | 48/53 |
| Stage 1 localized occurrences/actions | 48/53 |
| Stage 2 restart/replacement generations | 21 |
| Stage 2 exact recoveries | 21/21 |

The six effects independently localized by SCOUT were
`tensor_corruption`, `stale_state`, `drop`, `duplicate`, `reorder`, and
`delay`.

Stage 2 verified all 21 effects through the pressure harness, but only its
`tensor_corruption` case requested and produced exact SCOUT rank localization.
For the other 20 cases, the target came from the authenticated campaign
manifest and the isolated executor verified that the requested semantic effect
occurred. That is injection ground truth, not independent localization.

## Environment

The runs used source based on commit
`49e4b3f2f51d4a8f902f007896b1eba0a021aebd` with uncommitted feature work.
Stage 2 also included the local torchrun entry-point compatibility fix described
below. These are current manual results, not a sealed release evidence bundle.

| Item | Value |
|---|---|
| Host 0 | `ip-172-31-10-187`, 8 x NVIDIA A100-SXM4-40GB |
| Host 1 | `ip-172-31-2-86`, 8 x NVIDIA A100-SXM4-40GB |
| Driver | 580.126.16 |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| CUDA runtime reported by PyTorch | 13.0 |
| Inter-host transport | TCP over interface `ens32`; InfiniBand disabled |

Stage 1 used four GPUs on each host. Stage 2 used one host as four active
single-GPU torchrun agents plus four standby agents, matching the checked-in
`single_node_pressure.json` topology. The second host was not needed for that
single-host replacement profile.

## Stage 1: systematic SCOUT localization

The systematic DDP campaign ran 68 iterations across eight ranks. It covered 46
incident definitions, 48 selected occurrences, and 53 rank-local actions.
SCOUT ran on every optimizer boundary.

Representative launch on each host:

```bash
NCCL_SOCKET_IFNAME=ens32 GLOO_SOCKET_IFNAME=ens32 NCCL_IB_DISABLE=1 \
torchrun \
  --nnodes=2 \
  --nproc-per-node=4 \
  --node-rank=<0-or-1> \
  --master-addr=172.31.10.187 \
  --master-port=29604 \
  --module tests.validation.fault_injection.pytorch \
  --artifact-dir /tmp/lm-resiliency-stage1-20260904.nZGzgc
```

Result:

```json
{
  "detected_actions": 53,
  "detected_occurrences": 48,
  "injected_actions": 53,
  "injected_occurrences": 48,
  "localized_actions": 53,
  "localized_occurrences": 48,
  "passed": true
}
```

Every selected occurrence matched the expected failure kind, rank, component
source, and layer evidence. The final fault-free iteration was clean.

Stage 1 artifact digests:

| Artifact | SHA-256 |
|---|---|
| `evaluation.json` | `e578d5848ae1c0809b726e20edec0106e58a4da9eb6fe37f9bd10293ab3613e7` |
| `injection.json` | `1aeadefd8cc69f5659286685022f8b41a42d87a8ec292b822f1844a68c617ea2` |
| `localization.json` | `ff96c976ed4d3aeac5c33fb00ea7a2536dd7aac18d008addad04adc7dc2a3df4` |

The authenticated campaign manifest identity was
`189246f61de1f47cb634ee70e60135a167fd6ad4b0dd3e12aea8804a78ca0617`.

## Stage 2: all-types restart and replacement pressure

The checked-in all-types campaign ran one incident per generation with a clean
checkpoint step between incidents:

```bash
campaign_dir=/tmp/lm-resiliency-stage2-pressure-20260904.Ezo9mq
cp examples/torchrun/resiliency_cycle/campaigns/single_node_pressure.json \
  "$campaign_dir/campaign.json"

NCCL_SOCKET_IFNAME=ens32 GLOO_SOCKET_IFNAME=ens32 NCCL_IB_DISABLE=1 \
python -m examples.torchrun.resiliency_cycle.pressure orchestrate \
  --framework pytorch \
  --fault-campaign-dir "$campaign_dir" \
  --gpus 0,1,2,3,4,5,6,7
```

All 21 incidents completed. Seventeen used same-node restart and four used a
standby replacement. Every successor generation restored the selected GEMINI
checkpoint exactly. The final step was 42.

The final campaign-to-baseline comparison reported:

| State | Difference |
|---|---:|
| Loss | 0.0 |
| Model | 5.960464477539063e-08 |
| Optimizer | 4.656612873077393e-10 |
| RNG | Bitwise exact |

The campaign summary reports `recoveries: "bitwise exact"`. Its manifest
identity was
`4c9177d8f739f2e0fe01608f66777e661966a8368db34a238aed58b4bbe9bf81`.

Stage 2 artifact digests:

| Artifact | SHA-256 |
|---|---|
| `summary.json` | `3052f9a48bcff0ca80ab9913fd53355677927d6d4a34a9861ad43ce78e640bdd` |
| `campaign.json` | `36781f2b285876e29c9a134e99a2f943fe7fef60429aa91189acb3c8557b004e` |
| `state.json` | `038cea5a2ef868639c9500a5b28502decd2f5d7d964c5c0ab5b825e26358ea0a` |

## Per-effect evidence and localization

“SCOUT exact” means SCOUT independently named the injected rank in Stage 1 or
Stage 2. “Executor only” means the pressure harness verified its own injected
effect and the manager recovered the job, but no independent SCOUT localization
claim was made for that case.

| Failure type | Stage 2 injection evidence | Localization result |
|---|---|---|
| `tensor_corruption` | Sign-flipped payload | SCOUT exact |
| `stale_state` | Selected step 1 while latest was step 2 | SCOUT exact in Stage 1; executor only in Stage 2 |
| `drop` | Output count 2 from input count 3 | SCOUT exact in Stage 1; executor only in Stage 2 |
| `duplicate` | Output count 3 from input count 2 | SCOUT exact in Stage 1; executor only in Stage 2 |
| `reorder` | Observed order differed from expected order | SCOUT exact in Stage 1; executor only in Stage 2 |
| `delay` | 5.059 ms observed for a 5.0 ms request | SCOUT exact in Stage 1; executor only in Stage 2 |
| `hang` | Bounded watchdog observed a non-returning child | Executor only |
| `timeout` | Bounded child operation timed out | Executor only |
| `exception` | Expected runtime exception caught | Executor only |
| `resource_exhaustion` | Requested 2,048 bytes against a 1,024-byte sandbox quota | Executor only |
| `process_termination` | Disposable child exited with signal return code -15 | Executor only |
| `resource_unavailable` | Disposable resource was absent | Executor only |
| `checkpoint_corruption` | Disposable checkpoint bytes changed | Executor only |
| `checkpoint_truncation` | Disposable checkpoint reduced from 18 bytes to 4 | Executor only |
| `checkpoint_missing` | Disposable checkpoint removed | Executor only |
| `io_error` | Directory read produced `IsADirectoryError` | Executor only |
| `payload_corruption` | Payload digest changed | Executor only |
| `collective_desync` | Invocation sequence differed | Executor only |
| `message_drop` | One sent message, zero received | Executor only |
| `network_partition` | Closed socketpair blocked communication | Executor only |
| `config_drift` | Configuration digests differed | Executor only |

## Interpretation and boundary

All 21 canonical effects can be exercised without placing destructive effects
inside one continuous training process. The restart/replacement harness is the
correct isolation boundary for that coverage.

The result does not establish that SCOUT localizes all 21 types. Exact SCOUT
localization currently has direct evidence for six types in this qualification.
The remaining effects need an appropriate independent evidence source, such as
out-of-band progress, health telemetry, checkpoint-integrity validation, or
communication endpoint observations. Some cases, especially collective
desynchronization with only two endpoints, must remain collective-scope or
inconclusive rather than naming an unsupported culprit.

Stage 2 injects semantic effects into disposable sandbox resources. It does not
claim physical GPU OOM, device loss, filesystem-device failure, or host-fabric
partition. Those stronger campaigns require dedicated cluster controls,
bounded cleanup, and detector-specific acceptance criteria.

## Harness issue found during validation

PyTorch 2.13 loads `torchrun.handlers` entry points as zero-argument factories.
The repository advertised the direct one-argument handler creator, so the
all-types campaign initially failed before admission. The entry point now
targets `get_rendezvous_handler_creator`, while
`create_rendezvous_handler(params)` remains available as a public direct API.
A focused unit test covers both call forms.
