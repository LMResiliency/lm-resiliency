<div align="center">

# LLM Resiliency

### Fast checkpoint recovery and fault localization for distributed LLM pre-training

[![arXiv](https://img.shields.io/badge/arXiv-2608.11034-b31b1b.svg)](https://arxiv.org/abs/2608.11034)
[![SOSP](https://img.shields.io/badge/SOSP-2023-violet.svg)](https://doi.org/10.1145/3600006.3613145)
[![PyPI](https://img.shields.io/pypi/v/lm-resiliency.svg)](https://pypi.org/project/lm-resiliency/)
[![Python 3.10–3.12](https://img.shields.io/badge/Python-3.10--3.12-3776AB.svg)](https://www.python.org/)
[![PyTorch 2.10–2.13](https://img.shields.io/badge/PyTorch-2.10--2.13-EE4C2C.svg)](https://pytorch.org/)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)

[Quick Start](#quick-start) · [API Guide](docs/api.md) · [Frameworks](#framework-support) · [Validation](docs/validation.md)

</div>

`lm-resiliency` includes two runtime safeguards to an existing training stack:

| | Protects against | What you gain |
|---|---|---|
| **SCOUT** | Silent data corruption (SDC); compute, input-pipeline, and communication stragglers; collective desynchronization; process stalls | Pinpoint faulty ranks, GPUs, nodes, or communication endpoints during training, and exclude checkpoints affected by recurring SDC from recovery; see the [coverage contract](docs/scout.md#coverage-contract) |
| **GEMINI** | Slow, infrequent durable checkpoints | Frequent asynchronous in-memory checkpoints, peer replication, and fast recovery from nearby state |

## Why LLM Resiliency

- **SDC-safe recovery-verified checkpoint:** SCOUT certifies recovery checkpoints and excludes candidates affected by recurring SDC, preventing corrupted state from being selected during recovery.
- **Localize faulty components at runtime:** SCOUT identifies affected ranks, GPUs, nodes, communication endpoints, or peer groups, including communication hangs while the training job is blocked.
- **Minimize rollback:** GEMINI saves complete training state to CPU memory at high frequency with no measurable training-throughput loss, reducing lost computation after a failure.
- **Retrieve checkpoints quickly:** GEMINI restores nearby state from memory, a surviving peer, or node-local storage, minimizing checkpoint retrieval time and global-storage reads.
- **Keep protection lightweight:** SCOUT incurs less than 1% amortized overhead during training for runtime failure localization.
- **No framework fork:** `lm-resiliency` integrates with PyTorch, TorchTitan, Megatron Core, and DeepSpeed through automatic adapters and one public entry point.
- **No training-loop rewrite:** `lm-resiliency` attaches hooks at framework initialization and leaves the existing training loop unchanged.
- **Bring your own launcher:** Users can integrate `lm-resiliency` with `torchrun`, Slurm, Kubernetes, or custom managers through platform-neutral APIs.

## Quick Start

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install "lm-resiliency[megatron]"
```

Use `[torchtitan]`, `[deepspeed]`, `[all]`, or the core package without an extra for other environments.

### Add resiliency to Megatron Core

Attach resiliency after Megatron creates its model chunks, optimizer, and scheduler.
Then resume the existing `train()` loop from the recovered iteration.

```python
from megatron.training import get_args

from lm_resiliency import enable_resiliency


def attach_resiliency(model_chunks, optimizer, scheduler):
    resiliency = enable_resiliency(
        model_chunks,
        optimizer,
        opt_param_scheduler=scheduler,
        interval=10,
    )
    get_args().iteration = resiliency.step_count
    return resiliency


# Call attach_resiliency(...) after Megatron setup, then enter train() unchanged.
```

See the [Megatron Core production-loop example](examples/production_loops/megatron.py) for a complete tiny-GPT job.

### Add resiliency to TorchTitan

Pass the initialized TorchTitan `Trainer` before entering its existing training loop.
The adapter discovers the model, optimizer, scheduler, dataloader, topology, and checkpoint state.

```python
from torchtitan.train import Trainer
from lm_resiliency import enable_resiliency


class MyTrainer(Trainer):
    def train(self):
        enable_resiliency(self, interval=10)
        super().train()
```

See the [TorchTitan production-loop example](examples/production_loops/torchtitan.py) for a complete Llama debug-model job.

### Framework support

| Framework | SCOUT parallelism |
|---|---|
| PyTorch | DDP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| TorchTitan | DP, FSDP2, HSDP, TP, SP, CP, PP, EP, expert TP |
| Megatron Core | DP, TP, SP, CP, PP, virtual PP, EP, expert TP |
| DeepSpeed | DP, ZeRO 1-3, TP, PP, Ulysses SP, EP, expert TP |

The package-root `enable_resiliency` entry point selects dense or expert replay peers from framework topology metadata.
See the [API guide](docs/api.md) for framework invocation, configuration, recovery, callbacks, and lifecycle management.
See the [compatibility policy](docs/compatibility.md) for supported and tested versions.
See the [PyTorch production-loop example](examples/production_loops/pytorch.py) for a complete DDP causal-LM job.
See the [DeepSpeed production-loop example](examples/production_loops/deepspeed.py) for a complete ZeRO-2 causal-LM job.

## Manager Integration

External launchers and cluster managers can consume normalized SCOUT reports and checkpoint `RecoveryDecision` records, preserve GEMINI state before worker replacement, and coordinate checkpoint transfer through platform-neutral APIs.
Launcher-specific retry, placement, and replacement policy remains external. See [Manager Integration](docs/api.md#manager-integration) in the API guide.

## Documentation

| Topic | Guide |
|---|---|
| Public APIs and manager integration | [API guide](docs/api.md) |
| Runnable framework integrations | [Production-loop examples](examples/README.md) |
| Supported Python and framework versions | [Compatibility](docs/compatibility.md) |
| GEMINI checkpoint tiers, recovery, and cadence | [GEMINI guide](docs/gemini.md) |
| SCOUT coverage, replay, fault reports, and checkpoint certification | [SCOUT guide](docs/scout.md) |
| MoE regime discovery, qualification, and measured results | [MoE execution regimes](docs/moe_execution_regimes.md) |
| Complete test evidence and limitations | [Validation report](docs/validation.md) |

## Development

```bash
git clone https://github.com/LMResiliency/lm-resiliency.git
cd lm-resiliency
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
CUDA_VISIBLE_DEVICES="" python -m pytest -q
```

GPU and distributed tests are opt-in and document their required `torchrun` command in each test file.
See the [test guide](tests/README.md) for the integration and validation layout.

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
