# Torchrun Workflows

This directory contains runnable qualification workflows for the native
torchrun integration:

| Directory | Purpose |
|---|---|
| [`adapter_bootstrap/`](adapter_bootstrap/) | Zero-import bootstrap and clean automatic-adapter checks |
| [`resiliency_cycle/`](resiliency_cycle/) | Detection, recovery decision, restart, checkpoint restoration, and standby replacement |

The framework-owned training loops remain under
[`examples/production_loops`](../production_loops/). The bootstrap matrix and
resiliency-cycle drivers reuse those framework components rather than defining
another set of toy training loops.

These programs require explicit hardware and framework environments and are not
part of default pull-request CI.

## Prerequisites

Run commands from the repository root in a supported Python environment with
the repository installed:

```bash
python -m pip install -e ".[dev]"
```

Install the relevant optional framework extra before selecting DeepSpeed,
Megatron Core, or TorchTitan. The framework matrix requires an even world size
of at least two; resiliency-cycle campaigns require an even active world size
of at least four. The controller treats each supplied GPU as an independent
synthetic torchrun node and starts one torchrun agent per GPU.

## Adapter Bootstrap Smoke

The smoke worker deliberately does not import `lm_resiliency`. Its disabled
policy proves that the rendezvous backend installs an inferred framework
adapter before user code runs, without enabling GEMINI or SCOUT or requiring
CUDA:

```bash
torchrun \
  --nnodes=1:1 \
  --nproc-per-node=1 \
  --max-restarts=0 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint=/tmp/lm-resiliency-smoke-rdzv \
  --rdzv-id=lm-resiliency-smoke \
  --rdzv-conf="store_type=file,\
lm_resiliency_restart_context_path=/tmp/lm-resiliency-smoke-context/context.json,\
lm_resiliency_worker_config=$PWD/examples/torchrun/adapter_bootstrap/policies/smoke.toml" \
  --module \
  examples.torchrun.adapter_bootstrap.smoke \
  --validation-output-dir /tmp/lm-resiliency-smoke
```

The run succeeds only when the result JSON contains
`"resiliency_adapter_attached": true`.

## Framework Matrix

The clean matrix launches the unchanged production loop for PyTorch,
DeepSpeed, Megatron Core, or TorchTitan through the automatic worker adapter:

```bash
python -m examples.torchrun.adapter_bootstrap.matrix \
  --framework all \
  --nproc-per-node 8 \
  --validation-output-dir /tmp/lm-resiliency-framework-matrix
```

`--nproc-per-node` must be an even integer of at least two. Repeat
`--framework` to select a subset. The controller writes one framework summary
per run and a combined `summary.json` under `--validation-output-dir`.

## Resiliency Cycle

The package keeps the user-facing controller and campaign manifests separate
from implementation details:

```text
resiliency_cycle/
  pressure.py
  campaigns/
  harness/
    artifacts.py
    campaign.py
    launch.py
    replay_fault.py
    runtime.py
    verify.py
    worker.py
    frameworks/
```

The outer `pressure.py` controller:

1. runs an uninterrupted baseline;
2. launches one torchrun agent per supplied GPU;
3. waits for automatic active/standby admission;
4. injects the scheduled failure type through the public fault-injection API;
5. publishes manager-selected successor generations; and
6. compares the final managed state with the baseline.

The controller is not a training worker and must not itself be launched through
torchrun. Select one framework with `--framework`:

```text
pytorch
deepspeed
megatron
torchtitan
```

### Campaign Bundle

`--fault-campaign-dir` is the run bundle. It contains:

```text
campaign.json
state.json
summary.json
baseline-artifacts/
campaign-artifacts/
contexts/
machine-ids/
baseline-*.log
*.log
```

Use a fresh directory for every run. Before launching any worker, the
controller rejects a bundle containing anything other than `campaign.json`;
this prevents reports or final artifacts from an earlier run satisfying the
current campaign's waits.

The checked-in short campaign uses four active GPU-nodes and one standby. It
schedules one same-node restart followed by one SCOUT-localized replacement:

```bash
campaign_dir=/shared/lm-resiliency-framework-recovery
mkdir -p "$campaign_dir"
cp examples/torchrun/resiliency_cycle/campaigns/framework_restart_replacement.json \
  "$campaign_dir/campaign.json"

python -m examples.torchrun.resiliency_cycle.pressure orchestrate \
  --framework pytorch \
  --fault-campaign-dir "$campaign_dir" \
  --gpus 0,1,2,3,4
```

If `campaign.json` is absent, the controller creates the default pressure
profile: eight active GPU-nodes, eight standbys, sixteen same-node restarts,
and eight replacements.

Fault injection is configured only through the campaign bundle. It is not a
worker-policy or `--rdzv-conf` setting. Replacement incidents must run at step
2 or later so a clean recovery-verified checkpoint exists before corruption.

The checked-in `single_node_pressure.json` profile uses one eight-GPU host:
four active GPU-agents form a training world size of four and four GPU-agents
remain parked as standbys. It schedules every canonical fault-injection type
once. Four replacement-class incidents consume the standbys; the remaining
seventeen incidents use same-node restart and exact recovery. Destructive
process, storage, resource, and communication effects are executed against
disposable rank-local sandbox resources. Only replay tensor corruption claims
real SCOUT rank localization in this single-host profile.

### Multi-Host Run

The campaign directory and its parent must be visible at the same path on both
hosts. Supply local and remote GPU lists of the sizes required by the campaign,
plus the remote Python environment, remote source directory, and a
controller-reachable rendezvous address:

```bash
python -m examples.torchrun.resiliency_cycle.pressure orchestrate \
  --framework pytorch \
  --fault-campaign-dir /shared/lm-resiliency-pressure \
  --gpus 0,1,2,3,4,5,6,7 \
  --remote-gpus 0,1,2,3,4,5,6,7 \
  --remote-host worker-b \
  --remote-python /opt/lm-resiliency/bin/python \
  --remote-source-dir /shared/lm-resiliency-source \
  --rdzv-host 10.0.0.10
```

SSH must work in batch mode. The controller copies the current source tree to
`--remote-source-dir` before launching remote agents and installs it with the
interpreter selected by `--remote-python`. Gloo and NCCL choose their normal
host interfaces; set their standard environment variables in the launch
environment only when the deployment requires an explicit interface.

### Acceptance Criteria

The controller exits successfully only when:

- every scheduled incident produces reports from the full active world;
- SCOUT localizes each replay-only SDC to the scheduled logical rank;
- the selected standby inherits the failed rank and logical slot;
- every successor rank restores the manager-selected checkpoint step and
  job-wide GEMINI topology exactly;
- RNG and framework-owned recovery state match; and
- final loss, model, and optimizer state remain within the
  framework-specific baseline tolerances.

The final evidence is recorded in `<fault-campaign-dir>/summary.json`.
