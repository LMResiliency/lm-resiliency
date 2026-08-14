# Contributing

Contributions to code, tests, documentation, and framework integrations are welcome.
Open an issue before substantial changes so the design and validation scope can be agreed before implementation.
Security reports must follow [SECURITY.md](SECURITY.md), not the public issue tracker.

## Development Environment

Use a supported Python version from the [compatibility policy](docs/compatibility.md).
The following commands create a standard `pip` environment:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

The `dev` extra is synchronized with `requirements/tool-versions.txt`, which is also used as the automation constraints file.
Update both files together when changing a development or CI tool version.

Framework-specific development can install `.[torchtitan]`, `.[megatron]`, `.[deepspeed]`, or `.[all]` in the same environment.

## Required Checks

Run the CPU suite with CUDA hidden, even on a GPU host:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q
ruff check .
ruff format --check .
pre-commit run --all-files
python -m build
```

Add focused tests for changed behavior.
The complete CPU suite must pass without optional framework packages being imported by the package root.

## Optional GPU Checks

GPU, distributed, and multi-node tests are opt-in.
Run the integration scripts relevant to the change using the `torchrun` command documented in the test file or its linked validation guide.
Use the smallest topology that exercises the affected behavior, and record the framework versions, GPU model, world size, command, and result in the pull request.

Changes to framework adapters, distributed collectives, checkpoint replication, replay peer selection, or MoE execution regimes require focused GPU validation when suitable hardware is available.
If the required hardware is unavailable, state exactly which validation remains outstanding.

## Pull Requests

Keep each pull request focused on one coherent change.
Use `[Type] Concise imperative summary` for the pull-request title.
Supported type tags are `[Bugfix]`, `[Feature]`, `[Docs]`, `[Refactor]`, `[Perf]`, `[Test]`, and `[CI/Release]`.
An optional subsystem tag may follow the type, such as `[Bugfix][SCOUT] Reject an inconclusive replay precondition`.
Describe the problem, implementation, compatibility impact, and validation performed.
Update public documentation and contract tests when changing supported behavior or import paths.
Preserve backward compatibility for stable interfaces as defined in [Compatibility](docs/compatibility.md).
Identify copied or adapted code and include its license, copyright notice, source URL, and source revision.
Do not commit credentials, private keys, customer data, model data, generated environments, build products, or local validation artifacts.

By submitting a contribution, you represent that you have the authority to license it under the repository's [BSD-3-Clause License](LICENSE).
You retain copyright in your contribution.

All contributors must follow the [Code of Conduct](CODE_OF_CONDUCT.md).
