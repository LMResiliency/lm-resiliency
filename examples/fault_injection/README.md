# Fault Injection and SCOUT Localization

This example runs a real PyTorch DDP training loop, injects a systematic failure
campaign through `enable_fault_injection()`, and uses `enable_resiliency()` to
detect and localize every occurrence with SCOUT.

Run the systematic campaign on eight GPUs:

```bash
torchrun --standalone --nproc-per-node=8 --module \
  examples.fault_injection.pytorch \
  --artifact-dir /tmp/lm-resiliency-fault-evaluation
```

The checked-in [campaign.json](campaign.json) distributes 46 incident
definitions across all eight global ranks. They produce 48 scheduled
occurrences and 53 rank-local fault actions.
See the
[campaign field reference](../../docs/fault_injection.md#campaign-field-reference)
for every manifest field, allowed value, default, and validation rule.

The campaign covers every failure-type/surface pair supported by the built-in
local executor:

| Family | Coverage |
|---|---|
| Numerical corruption | Sign flip, set zero/one, scale up/down, noise, single-bit flip, and multi-bit flip |
| Tensor surfaces | Input, output, weight, bias, gradient, and Adam optimizer state |
| State flow | Stale and duplicate on all six tensor surfaces; drop and reorder on input, output, and gradient |
| Performance | Delay on compute, input, and output surfaces |
| Correlation | Simultaneous faults on two and three ranks while retaining a healthy majority |
| Temporal behavior | Transient, three-occurrence intermittent, and permanent-until-recovery incidents |

SCOUT runs every iteration and emits normalized JSON reports through
`OrchestrationHooks.report_fault`.
Iteration 68 is fault-free and verifies that detection returns to a clean state.
The default run length is derived from each incident's final trigger and bounded
lifetime, then adds one clean iteration; it never treats a still-active
multi-iteration effect as the post-fault certification step.

Weight, bias, gradient, and optimizer-state faults can contaminate later cases.
For those cases, an evaluation-only optimizer hook restores the last clean
model and optimizer snapshot after SCOUT reports the fault and at the configured
effect-expiration boundary. The clean snapshot is frozen throughout a bounded
fault window, so contaminated intermediate iterations cannot become the next
reset point. Reset and hold windows use the campaign's deterministic
probability selection, so an explicitly skipped occurrence never restores or
holds evaluation state. If one incident expires while another state-affecting
window remains active, the hold window takes precedence and defers the full
model/optimizer reset until the later window also expires. This isolates each
case; it is not a replacement for production checkpoint recovery. The example
rejects `campaign_end` lifetimes for gradient-affecting incidents because they
cannot produce a clean certification iteration before shutdown. It requires
`matching_calls=1` for every incident because framework call multiplicity
cannot be converted portably into optimizer-iteration run length or
certification boundaries. These reset-policy constraints are validated before
the example initializes distributed process groups, GEMINI, or SCOUT, so an
unsupported manifest cannot leave runtime resources outside the teardown
boundary.

Teardown attempts fault-injection cleanup, evaluation-state cleanup, resiliency
cleanup, and process-group destruction independently. A cleanup failure is
reported without skipping later cleanup, and an active training exception
remains the primary error.

The example writes:

| Artifact | Contents |
|---|---|
| `injection.json` | Executed campaign manifest and verified injection ground truth |
| `localization.json` | JSON-ready failures reported by `enable_resiliency()` |
| `evaluation.json` | Occurrence- and action-level detection/localization counts plus rank, failure-kind, and SCOUT component-source comparison |

All three artifacts carry the same canonical `manifest_identity`. The comparator
recomputes that identity from the embedded injection manifest and rejects
tampered, missing, or mismatched identities, so localization output from an
earlier campaign revision cannot satisfy new injection ground truth with the
same name.
Expected fault kinds, ranks, resources, layers, components, and parameters are
derived from that authenticated manifest rather than trusted from individual
injection records. Record targets and lifecycle fields must match the manifest
and use strict JSON types before they can contribute to a passing result.
Probability selection is recomputed from the manifest seed, incident ID, and
iteration. A skipped occurrence must contain every manifest action and match
the deterministic selection result.
SCOUT reports currently carry a training iteration but no campaign occurrence
ID, so this example also rejects two distinct occurrences scheduled at the same
iteration instead of crediting one report to both.

The process exits unsuccessfully unless every selected action is injected
successfully and every resulting occurrence is detected and localized.
Explicit probability skips are excluded; pending, failed, or cancelled records
remain in the evaluation and fail the campaign. Correlated incidents remain one
detected occurrence, while `injected_actions`, `detected_actions`, and
`localized_actions` account for every successful rank-local fault action inside
those incidents.
Dense catalog reports use `layer_id: -1` when several replay recipes are
aggregated; in that case the comparison uses the reported `hidden.*`,
`embedding.*`, or `output.*` source as component-localization evidence.
That evidence remains associated with the ranks or resources in the individual
SCOUT report, so valid source names swapped between targets fail localization.
Aggregate straggler reports are tied to the configured replay catalog.
Re-run the comparison independently with:

```bash
python -m examples.fault_injection.compare \
  --artifact-dir /tmp/lm-resiliency-fault-evaluation
```

`enable_resiliency()` is intentionally called before
`enable_fault_injection()`.
That registration order lets SCOUT inspect the active one-iteration fault before
the fault scheduler retires it at the optimizer boundary.

Regenerate the manifest after changing the matrix definition:

```bash
python -m examples.fault_injection.generate_campaign
```

The campaign intentionally excludes destructive process, storage, collective,
resource, and network faults. Those require an isolated job, an explicit
capability executor, and usually a training manager to relaunch or replace
workers. They cannot be combined safely with a continuous-loop correctness
matrix.
