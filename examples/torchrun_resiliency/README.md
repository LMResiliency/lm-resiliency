# Torchrun Resiliency Validation

This package contains validation workloads for the native torchrun integration.
It is separate from `examples.production_loops`, which contains only ordinary
framework-owned training loops and their shared resiliency policy.

## Smoke

`smoke.py` is the smallest user module used to prove that the rendezvous backend
installs the inferred framework adapter before user code runs. Its deliberately
disabled policy lives at `policies/smoke.toml`; it validates bootstrap without
requiring CUDA or enabling GEMINI or SCOUT.

## Pressure Campaign

`pressure.py` owns destructive orchestration and fault injection. The only
user-facing campaign argument is:

```bash
--fault-campaign-dir /shared/lm-resiliency-pressure
```

The directory is a campaign bundle. The harness creates `campaign.json` when it
is absent and stores restart-stable state, checkpoints, logs, artifacts, and
the final summary beside it. Fault injection is configured by `campaign.json`,
not by the production worker policy or rendezvous configuration.

The default campaign models each of 16 GPUs as an independent node:

- eight active one-GPU torchrun agents;
- eight parked one-GPU standbys;
- 16 process-stall incidents that restart the same active assignments; and
- eight replay-only SDC incidents that quarantine one active GPU-node and admit
  one standby.

Synthetic machine-ID files are used only to model multiple nodes on each
physical validation host. Production agents derive their identity from
`/etc/machine-id`.
