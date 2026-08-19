<div align="center">

# LM Resiliency

### Fast fault localization and checkpoint recovery for distributed LLM pre-training

[![PyTorch 2.10–2.13](https://img.shields.io/badge/PyTorch-2.10--2.13-EE4C2C.svg)](https://pytorch.org/)
[![torchrun native](https://img.shields.io/badge/torchrun-native-EE4C2C.svg)](docs/torchrun_resiliency.md)
[![TorchTitan 0.2.2](https://img.shields.io/badge/TorchTitan-0.2.2-EE4C2C.svg)](https://github.com/pytorch/torchtitan)
[![Megatron Core 0.18.2](https://img.shields.io/badge/Megatron_Core-0.18.2-76B900.svg)](https://github.com/NVIDIA/Megatron-LM)
[![DeepSpeed 0.19.4](https://img.shields.io/badge/DeepSpeed-0.19.4-0078D4.svg)](https://github.com/deepspeedai/DeepSpeed)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![arXiv](https://img.shields.io/badge/arXiv-2608.11034-b31b1b.svg)](https://arxiv.org/abs/2608.11034)
[![SOSP](https://img.shields.io/badge/SOSP-2023-violet.svg)](https://doi.org/10.1145/3600006.3613145)

[Installation](#installation) · [Examples](#examples) · [API Guide](docs/api.md) · [Frameworks](#framework-support) · [Validation](docs/validation.md) · [Roadmap](ROADMAP.md)

</div>

`lm-resiliency` includes two runtime safeguards to an existing training stack:

| | Protects against | What you gain |
|---|---|---|
| **SCOUT** | Latent and recurring SDC, contaminated recovery state, compute/input/communication stragglers, collective hangs, process stalls, and supported hardware failures | Pinpoint faulty ranks, GPUs, nodes, or communication endpoints; certify safe recovery state; and trigger automatic restart or standby replacement through torchrun |
| **GEMINI** | Slow, infrequent durable checkpoints | Frequent asynchronous in-memory checkpoints, peer replication, and fast recovery from nearby state |

## Why LM Resiliency

- **SDC-safe recovery-verified checkpoint:** SCOUT certifies recovery checkpoints and excludes candidates affected by recurring SDC.
- **Localize latent and permanent failures at runtime:** SCOUT identifies affected ranks, GPUs, nodes, communication endpoints, peer groups, or telemetry-reported physical devices.
- **Native `torchrun` integration:** `lm_resiliency` works as a `torchrun` rendezvous backend to automatically detect and localize failures, replace faulty nodes with standbys, and restart training from recovery-verified checkpoint.
- **Minimize rollback and checkpoint retrieve:** GEMINI saves training states to CPU memory at high frequency, reducing lost computation and checkpoint retrieval overhead after a failure.
- **No training-loop rewrite:** `lm-resiliency` attaches hooks at framework initialization and leaves the existing training loop unchanged.
- **Keep protection lightweight:** `lm_resiliency` incurs less than 1% amortized overhead to the training throughput.

## Installation

LM Resiliency can be installed from source or from a stable release.

### From source

```bash
git clone https://github.com/LMResiliency/lm-resiliency.git
cd lm-resiliency
python -m pip install -e .
```

### Stable releases

```bash
python -m pip install lm-resiliency
```

Append `[deepspeed]`, `[megatron]`, `[torchtitan]`, or `[all]` to either
installation command when the corresponding optional framework integration is
needed.

## Examples

### Production loop

From a source checkout, select a framework and launch its unchanged training
loop on two eight-GPU hosts. Run the same command on both hosts and set
`RDZV_HOST` to a hostname or IP address reachable from both:

```bash
framework=pytorch  # pytorch, deepspeed, megatron, or torchtitan
RDZV_HOST=node-a
WORKER_CONFIG="examples/production_loops/policies/resiliency.toml"
RESTART_CONTEXT="/tmp/lm-resiliency-${framework}-context/context.json"

# Keep one eight-GPU node active and park a second eight-GPU node as standby.
# Allow each torchrun agent to restart its worker group up to four times.
# Use LM Resiliency for active/standby admission and recovery coordination.
torchrun \
  --nnodes=1:2 \
  --nproc-per-node=8 \
  --max-restarts=4 \
  --rdzv-backend=lm_resiliency \
  --rdzv-endpoint="${RDZV_HOST}:29400" \
  --rdzv-id="${framework}-example" \
  --rdzv-conf="store_type=tcp,read_timeout=120,\
lm_resiliency_restart_context_path=${RESTART_CONTEXT},\
lm_resiliency_worker_config=${WORKER_CONFIG}" \
  --module \
  "examples.production_loops.${framework}" \
  --validation-output-dir "/tmp/lm-resiliency-${framework}"
```

See [Torchrun Resiliency](docs/torchrun_resiliency.md) to understand how LM Resiliency provides closed-loop recovery for the training loop.

### Resiliency cycle

The resiliency-cycle example pressure-tests the complete torchrun recovery
workflow under repeated failures. It compares a managed run with an
uninterrupted baseline and verifies that every restart and replacement restores
the selected checkpoint without changing the final training state.

This campaign uses one eight-GPU host and treats each GPU as one synthetic
torchrun node. Four active GPU-agents form a training world size of four, while
the other four GPU-agents remain parked as standbys. The `--gpus` option lists
all eight allocated GPUs. The campaign injects all 21 canonical failure types:
17 incidents exercise same-node restart and exact recovery, while four incidents
consume the four standbys and replace one active GPU-agent each. Tensor
corruption uses SCOUT replay localization. Process-, storage-, resource-, and
network-destructive effects run in disposable rank-local sandboxes so the
single-host example does not damage the host or production network:

```bash
campaign_dir=$(mktemp -d /tmp/lm-resiliency-cycle.XXXXXX)
cp examples/torchrun/resiliency_cycle/campaigns/single_node_pressure.json \
  "$campaign_dir/campaign.json"

python -m examples.torchrun.resiliency_cycle.pressure orchestrate \
  --framework pytorch \
  --fault-campaign-dir "$campaign_dir" \
  --gpus 0,1,2,3,4,5,6,7
```

The command succeeds only after all 21 incidents complete, every successor
generation restores the coordinator-selected checkpoint step and topology, and
the final managed state matches the uninterrupted baseline. See the
[torchrun workflow guide](examples/torchrun/README.md) for multi-host and
campaign configuration details.

## Framework Support

| Framework | SCOUT parallelism |
|---|---|
| PyTorch | DDP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| TorchTitan | DP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| Megatron Core | DP, TP, SP, CP, PP, virtual PP, EP, expert TP |
| DeepSpeed | DP, ZeRO 1-3, TP, PP, Ulysses SP, EP, expert TP |

The package-root `enable_resiliency` entry point selects dense or expert replay peers from framework topology metadata.
See the [API guide](docs/api.md) for framework invocation, configuration, recovery, callbacks, and lifecycle management.
See the [compatibility policy](docs/compatibility.md) for supported and tested versions.

## Documentation

| Topic | Guide |
|---|---|
| Public APIs and automatic framework adapters | [API guide](docs/api.md) |
| Native torchrun restart and replacement | [Torchrun Resiliency](docs/torchrun_resiliency.md) |
| Runnable torchrun framework integrations | [Examples](examples/README.md) |
| Reproducible fault campaigns and localization scoring | [Fault injection evaluation](docs/fault_injection.md) |
| Supported Python and framework versions | [Compatibility](docs/compatibility.md) |
| GEMINI checkpoint tiers, recovery, and cadence | [GEMINI guide](docs/gemini.md) |
| SCOUT coverage, replay, fault reports, and checkpoint certification | [SCOUT guide](docs/scout.md) |
| MoE regime discovery, qualification, and measured results | [MoE execution regimes](docs/moe_execution_regimes.md) |
| Revision-bound evidence format, results, and limitations | [Validation report](docs/validation.md) |
| Planned project direction and priorities | [Roadmap](ROADMAP.md) |

## Contributing

Contributions to code, tests, documentation, and framework integrations are welcome.
See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, required checks, GPU validation, and pull-request expectations.
Report security vulnerabilities privately according to [SECURITY.md](SECURITY.md).

## Citation

If you use `lm-resiliency` in your research, please cite the relevant papers.

### GEMINI

[GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints](https://doi.org/10.1145/3600006.3613145)

```bibtex
@inproceedings{gemini-sosp23,
  title = {{GEMINI}: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints},
  author = {Wang, Zhuang and Jia, Zhen and Zheng, Shuai and Zhang, Zhen and Fu, Xinwei and Ng, T. S. Eugene and Wang, Yida},
  booktitle = {Proceedings of the 29th ACM Symposium on Operating Systems Principles},
  year = {2023},
}
```

### SCOUT

[SCOUT: Symmetric Consensus Outlier Detection for Failure Localization in LLM Pre-Training](https://arxiv.org/abs/2608.11034)

```bibtex
@misc{wang2026scout,
  title = {{SCOUT}: Symmetric Consensus Outlier Detection for Failure Localization in {LLM} Pre-Training},
  author = {Wang, Zhuang},
  year = {2026},
  url = {https://arxiv.org/abs/2608.11034}
}
```

## License

Licensed under the [BSD-3-Clause License](LICENSE).
