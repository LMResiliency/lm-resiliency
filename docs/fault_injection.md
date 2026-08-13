# Fault Injection Evaluation Kit

The fault injection evaluation kit runs reproducible, framework-aware campaigns
inside `lm-resiliency`.
It is independent of SCOUT and GEMINI: the injector records verified fault ground
truth, and any resiliency system can submit a neutral localization result for
comparison.

The current safe in-process surface supports numerical corruption and module
delays in PyTorch, TorchTitan, Megatron Core, and DeepSpeed models.

## Quick Start

Define a campaign, bind it to the initialized training object, and trigger the
campaign before each target training step:

```python
import torch

from lm_resiliency import (
    FaultCampaign,
    FaultLocation,
    FaultPersistence,
    FaultSpec,
    FaultTarget,
    FaultType,
    LocalizationResult,
    enable_fault_injection,
)

campaign = FaultCampaign(
    name="hidden-output-sdc",
    framework="pytorch",
    faults=(
        FaultSpec(
            fault_id="rank-0-layer-2",
            fault_type=FaultType.SIGN_FLIP,
            target=FaultTarget(
                rank=0,
                module="layers.2",
                location=FaultLocation.OUTPUT,
            ),
            steps=(20,),
            persistence=FaultPersistence.TRANSIENT,
        ),
    ),
)

session = enable_fault_injection(model, campaign)

for step, batch in enumerate(train_loader, start=1):
    session.trigger(step)
    loss = model(batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

`trigger(step)` creates ground truth only on the target rank.
An output or transient parameter fault is armed until the target module executes.
Use `call_index=2` to target the second invocation after the trigger, such as a
replay invocation rather than the normal forward pass.

After the resiliency system reports its decision, evaluate it without importing
that system into the injector:

```python
record = session.records[0]
report = session.evaluate(
    [
        LocalizationResult(
            injection_id=record.injection_id,
            detected=True,
            failed_ranks=(0,),
            kind="sdc",
            component="layers.2",
            latency_ms=8.4,
        )
    ]
)
report.to_json("campaign-report.json")
session.close()
```

`localized` is true only when the injection succeeded, the submitted result
declares detection, and the expected rank is present in `failed_ranks`.
Optional kind and component matches are reported separately.

## Framework Targets

Pass the same initialized training object used by the corresponding
`enable_resiliency` integration:

| Framework | Fault-injection target |
|---|---|
| PyTorch | `torch.nn.Module` |
| TorchTitan | Initialized trainer exposing `model_parts` |
| Megatron Core | Model chunks in a list or tuple |
| DeepSpeed | Engine exposing `module` |

Automatic framework discovery is the default.
Set `framework` on the campaign or pass it to `enable_fault_injection` when a
custom wrapper makes discovery ambiguous.

`FaultTarget.model_part` selects a TorchTitan model part or Megatron model chunk.
`FaultTarget.module` is the dot-separated path returned by `named_modules()`;
an empty path targets the selected model root.

## Campaign Manifest

Campaigns can be checked into source control and loaded from JSON:

```json
{
  "name": "probabilistic-output-sdc",
  "framework": "pytorch",
  "metadata": {
    "model": "tiny-gpt",
    "topology": "dp=4"
  },
  "faults": [
    {
      "fault_id": "hidden-sign-flip",
      "fault_type": "sign_flip",
      "target": {
        "rank": 3,
        "model_part": 0,
        "module": "layers.2",
        "location": "output"
      },
      "steps": [20, 40],
      "magnitude": "medium",
      "scope": "single",
      "persistence": "transient",
      "probability": 0.5,
      "seed": 17,
      "call_index": 1,
      "delay_ms": 0.0
    }
  ]
}
```

Load and write manifests with `FaultCampaign.from_json(...)` and
`FaultCampaign.to_json(...)`.
Probability selection is deterministic for the campaign seed, step, and target
rank.

## Supported Faults

Numerical faults can target a module's `weight`, `bias`, or floating-point
`output`:

| Fault type | Behavior |
|---|---|
| `single_bitflip` | Flip one bit selected by magnitude |
| `multi_bitflip` | Flip four adjacent bits selected by magnitude |
| `stuck_at_zero` | Set selected values to zero |
| `stuck_at_one` | Set selected values to one |
| `scale_up` / `scale_down` | Multiply selected values by a magnitude-dependent factor |
| `gaussian_noise` | Add seeded magnitude-dependent noise |
| `sign_flip` | Negate selected values |
| `set_nan` / `set_inf` | Replace selected values with NaN or infinity |
| `delay` | Sleep after the selected module invocation |

Scopes select one value, one row, 1 percent, 10 percent, or the full tensor.
Bit flips support `float16`, `bfloat16`, `float32`, and `float64`.
A numerical injection that does not change the selected values is rejected
instead of being counted as successful ground truth.

Transient parameter faults are restored after the target forward call.
Transient output faults return a corrupted copy without modifying the original
output tensor.
Persistent faults modify a parameter at their single trigger step and remain
active until `restore()` or `close()`.

## Ground Truth and Evaluation

Every attempted target-rank occurrence has an injection ID in the form
`<fault_id>@<step>` and an `InjectionStatus`:

- `injected`: the effect was verified;
- `skipped_probability`: deterministic probability selection skipped it;
- `failed`: the target could not be resolved or the effect was not verified; or
- `cancelled`: the session closed before an armed transient fault executed.

`CampaignReport` includes:

- the full campaign manifest and reproduction metadata;
- the complete injection ground truth;
- the neutral localization results supplied by the caller;
- correct and unexpected rank attribution;
- optional fault-kind and component matches; and
- detection or localization latency supplied by the evaluated system.

Reports are rank-local. A distributed campaign runner should collect one report
per rank and preserve the framework, topology, workload, software, and hardware
metadata required to reproduce the run.

## Safety and Current Boundaries

The current API intentionally implements only safe in-process numerical
corruption and delays.
It does not yet inject process termination, node loss, communication failure,
checkpoint corruption, input-pipeline corruption, gradient or optimizer-state
corruption, or destructive cluster-level faults.

Persistent parameter restoration writes the values captured at injection time.
Do not continue optimizer updates and later restore those values in a production
training run, because that would overwrite intervening updates.
Use persistent faults in isolated evaluation jobs that recover or terminate
after localization, or restore them before the next optimizer step.

The injector verifies the local effect, but the campaign runner remains
responsible for synchronization, collecting rank-local reports, and applying
cluster safety policy.
