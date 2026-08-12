# Open-Source Release Checklist

Date: 2026-08-12 UTC.

This checklist applies to a new public repository created from the `release/open-source` snapshot.
The private repository's earlier commits and other branches are outside this checklist because they will not be copied.

## Current Baseline

| Check | Result |
|---|---|
| CPU unit suite with CUDA hidden | 632 passed |
| Ruff lint | Passed |
| Ruff format check | Passed |
| GitHub Actions workflow validation | `actionlint` and `zizmor` passed |
| Source distribution and wheel build | Passed |
| Core import without optional framework imports | Passed |
| Tracked build products, environments, and validation artifacts | None found |
| Obvious credentials, private keys, internal IP addresses, and local paths in the tracked tree | None found |
| Dedicated secret scan | `detect-secrets` 1.5.0 passed |
| Primary runtime dependency audit | `pip-audit` 2.9.0 found no known vulnerabilities |
| License | BSD-3-Clause |
| Distributed GPU validation | Documented in [Validation Summary](validation.md) |

## Required Before Publication

### 1. Add Continuous Integration

- [x] Add a GitHub Actions workflow that runs the CPU unit suite with CUDA hidden.
- [x] Run `ruff check` and `ruff format --check` in the workflow.
- [x] Build both the source distribution and wheel from a clean checkout.
- [x] Install the built wheel in an isolated environment and import `lm_resiliency`.
- [x] Validate the minimum supported Python and PyTorch versions in addition to the primary development versions.
- [ ] Make the workflow required for pull requests to the default branch.

The workflow is defined in `.github/workflows/ci.yml`.
It uses read-only permissions, SHA-pinned actions, minimum and primary dependency matrices, and separately built release artifacts.
After creating the public repository, require `Quality`, `Unit (minimum)`, `Unit (primary)`, and `Package` before merging to the default branch.

### 2. Define the Supported Version Matrix

The primary development environment uses PyTorch 2.13.0, TorchTitan 0.2.2, Megatron Core 0.18.2, and DeepSpeed 0.19.4.
The minimum CPU environment uses Python 3.10.20, NumPy 1.26.4, and PyTorch 2.10.0.

- [x] Test the declared minimum versions or narrow the dependency ranges to versions covered by integration tests.
- [x] Document tested and supported versions separately.
- [x] Define how compatibility is handled when framework internals change.
- [x] Add a manager compatibility test that installs released wheels of both repositories and exercises their public contract.

The supported ranges and qualification policy are documented in [Compatibility](compatibility.md).
The manager-owned wheel contract passed with independently built `0.1.0` wheels.

### 3. Establish a Clean Formatting Baseline

- [x] Run the configured Ruff formatter over the repository.
- [x] Review and commit the formatting-only change separately.
- [x] Confirm that `pre-commit run --all-files` passes without modifying files.

The formatting baseline is recorded in commit `d72d253`.

### 4. Complete License and Attribution Review

- [x] Confirm that the named copyright holder has authority to release all project code under BSD-3-Clause.
- [x] Add the required MIT copyright and permission notice for the Triton-derived grouped-GEMM validation code in `tests/support/triton_grouped_expert.py`.
- [x] Record the upstream Triton tutorial URL and source revision used by the derived implementation.
- [x] Review the tree for any other copied, adapted, or generated code that requires attribution.

The full notice is stored in `THIRD_PARTY_NOTICES.md` and included in source and wheel distributions.
Zhuang Wang confirmed release authority for all non-third-party code on 2026-08-09.

### 5. Define the Stable Public API

The package exposes separate user, manager, and experimental namespaces.

- [x] Limit package-root exports to interfaces intended for users and external managers, or document every supported export.
- [x] Mark experimental interfaces explicitly.
- [x] Define the compatibility policy for the `0.x` release series.
- [x] Add contract tests for the user-facing and manager-facing import paths.

The stable contracts are `lm_resiliency`, `lm_resiliency.manager_api`, and the explicit framework integration entry points.
Low-level interfaces under `lm_resiliency.experimental` may change within the `0.x` release series.
The core suite passed 632 tests, the manager suite passed 49 tests, and independently built `0.1.0` wheels passed the manager compatibility check.

### 6. Add Security and Contribution Policies

- [x] Add `SECURITY.md` with a private vulnerability-reporting channel and supported-version policy.
- [x] Add `CONTRIBUTING.md` with environment setup, CPU checks, optional GPU checks, and pull-request expectations.
- [x] Add a code of conduct appropriate for public contributions.
- [x] Run a dedicated secret scanner against the exact snapshot before creating the public repository.
- [x] Run a dependency vulnerability audit and triage findings before the first release.

`detect-secrets` 1.5.0 scanned all 198 intended tracked files without a baseline, file exclusions, or suppressed findings.
No candidate secrets were found.
`pip-audit` 2.9.0 audited every resolved third-party distribution in the isolated Python 3.12, NumPy 2.5.1, and PyTorch 2.13.0 CPU wheel environment.
The PyTorch `+cpu` local-version suffix was normalized to its upstream `2.13.0` version for the advisory query.
One final clean-install repeat initially resolved `setuptools` 78.1.0 from the PyTorch CPU index and reported `PYSEC-2025-49` and `PYSEC-2026-3447`.
The isolated runtime setup now upgrades to `setuptools>=83`; the repeated audit found no known vulnerabilities or collection failures.
No vulnerability exceptions are required.
CI repeats both audits without a secret baseline or ignored vulnerability identifiers.

### 7. Complete Package and Release Metadata

- [x] Add authors or maintainers, project URLs, keywords, and package classifiers to `pyproject.toml`.
- [x] Confirm that the `lm-resiliency` package name is available in the target package index.
- [x] Add a changelog or release-notes process.
- [x] Add `CITATION.cff` using the final GEMINI and SCOUT citations.
- [x] Include and smoke-test the installed MoE regime-discovery CLI in the release wheel.
- [ ] Publish a release candidate to TestPyPI or an equivalent staging index and test installation from that index.
- [ ] Tag the reviewed commit as `v0.1.0` and publish artifacts built by CI.

The PyPI and TestPyPI JSON APIs both returned `404` for the normalized `lm-resiliency` project name on 2026-08-09.
This check establishes current availability but does not reserve the name.
`CITATION.cff` passes the Citation File Format 1.2.0 schema, and the source distribution and wheel pass `twine check`.
The `Release` workflow builds immutable artifacts, supports trusted TestPyPI staging with an isolated installation test, and publishes version-matching tags to protected PyPI and GitHub environments.
TestPyPI publication requires the public repository's trusted-publisher configuration.
The final tag remains blocked until TestPyPI staging succeeds and the public repository's required checks are configured.

The public repository, protected GitHub environments, Actions policy, and TestPyPI and PyPI
trusted publishers are configured.
The following steps must wait until the refactor and release snapshot are complete:

- Copy the reviewed source snapshot to the public repository without copying the private
  repository's history.
- Run CI in the public repository, then require **Quality**, **Unit (minimum)**,
  **Unit (primary)**, and **Package** for pull requests to `main`.
- Publish and install the TestPyPI release candidate.
- Change the version to `0.1.0`, rerun the release gates, and create the production tag.

#### TestPyPI Staging Procedure

Prepare and publish the candidate as follows:

1. Set the package version in `pyproject.toml` to a PEP 440 prerelease such as `0.1.0rc1`.
2. Run the release-gate checks and commit and push the complete candidate. A GitHub Actions run
   cannot include local uncommitted changes.
3. Open **Actions**, select **Release**, select **Run workflow**, and choose the branch containing
   the candidate commit.
4. Approve the `testpypi` environment deployment if an approval rule is configured.
5. Confirm that **Build release artifacts**, **Publish to TestPyPI**, and
   **Test TestPyPI installation** all pass.

Do not create or push an RC tag with the current workflow.
Every `v*` tag enables the production PyPI publication job, so TestPyPI staging must use the
manual `workflow_dispatch` trigger.

The installation job waits for the TestPyPI index to expose the uploaded version, installs the
supported PyTorch and NumPy versions separately, and then installs only `lm-resiliency` from
TestPyPI:

```bash
python -m pip install \
  --index-url https://download.pytorch.org/whl/cpu \
  --only-binary=:all: \
  "torch==2.13.0"
python -m pip install "numpy==2.5.1" "setuptools>=83"
python -m pip install \
  --index-url https://test.pypi.org/simple/ \
  --no-deps \
  "lm-resiliency==0.1.0rc1"
python -m pip check
cd "$(mktemp -d)"
python -c "import lm_resiliency; print(lm_resiliency.__file__)"
lm-resiliency-discover-moe-regimes --help
```

Run those commands in a new Python 3.12 virtual environment when reproducing the workflow
manually.
Installing dependencies separately avoids depending on incomplete dependency mirrors in
TestPyPI while ensuring that the project wheel itself comes from the staging index.
The Python Packaging User Guide also documents
[installation from TestPyPI](https://packaging.python.org/guides/using-testpypi/).

Record the candidate version, commit SHA, workflow-run URL, and completion date in this checklist.
If the candidate needs changes after upload, increment the prerelease version, for example from
`0.1.0rc1` to `0.1.0rc2`, before staging it again.
After staging succeeds, change the version to `0.1.0`, rerun the release gates, review the final
commit, and create the production `v0.1.0` tag.

### 8. Validate Documentation From a Clean Checkout

- [x] Consolidate overlapping public guides and retain validation evidence in `validation.md` and `moe_execution_regimes.md`.
- [x] Check all relative links and heading anchors automatically.
- [x] Run the Quick Start in a clean environment without relying on the existing development environment.
- [x] State how to install `uv`, or provide an equivalent standard `pip` installation path.
- [x] Ensure every documented framework example uses only its supported public entry point.
- [x] Keep validation claims tied to the documented software and hardware versions.

Lychee 0.24.2 checked 77 Markdown link and heading-fragment references with no errors after documentation consolidation; CI runs the same offline check on every Markdown file.
The README now uses the standard-library `venv` module and `pip` rather than requiring `uv`.
A detached clean worktree completed the documented Python 3.12 setup and installed `.[torchtitan]` without the development environment.
The resulting Python 3.12.3 environment contained `lm-resiliency` 0.1.0, PyTorch 2.13.0, and TorchTitan 0.2.2, and `pip check` reported no broken requirements.
Package-root, manager, health-monitor, and `torchtitan.train.Trainer` imports passed from outside the checkout.
An AST audit parsed every Python example in the README and API guide and found nine `lm_resiliency` imports, all through documented stable namespaces.
Validation summaries now identify the hardware and software stacks for the two-host framework, production MoE, and Triton regime-compression campaigns.

## Recommended After Publication

- [ ] Add scheduled or manually dispatched GPU smoke tests for the supported framework matrix.
- [ ] Add issue and pull-request templates.
- [ ] Add automated dependency updates with framework compatibility review.
- [ ] Publish signed artifacts and provenance attestations through trusted CI publishing.
- [ ] Add coverage reporting for the CPU suite.
- [ ] Add a `py.typed` marker if downstream static type checking is part of the support contract.

## Release Gate

The repository is ready for public release when all required items above are complete and the following commands pass from a clean checkout:

```bash
CUDA_VISIBLE_DEVICES="" python -m pytest -q
ruff check .
ruff format --check .
pre-commit run --all-files
python -m build
```

The final review must also confirm an isolated wheel import, a clean secret scan, valid documentation links, and compatibility with the separately released manager package.
