# Changelog

All notable changes to this project are documented in this file.
The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Pre-training system-failure classifications for host and GPU exhaustion,
  storage, PCIe, fabric, input-pipeline, environment, fail-slow, control-plane,
  and SDC coverage-gap campaigns, with cause/effect validation and ground-truth
  report propagation.

## [0.2.0] - 2026-08-19

### Added

- A native `torchrun` rendezvous backend that provides closed-loop recovery through active/standby admission, same-node restart, faulty-node replacement, and exact manager-selected GEMINI recovery.
- Automatic zero-import worker adapters for native PyTorch, DeepSpeed, Megatron Core, and TorchTitan while preserving each framework's training loop and recovery state.
- Runnable production-loop, adapter-bootstrap, and resiliency-cycle examples, including a single-host campaign covering all 21 canonical fault types.
- A runnable CPU quick-start example with a complete native PyTorch training loop and GEMINI checkpoint resume.
- A scheduled and maintainer-dispatched two-GPU qualification workflow with machine-readable revision, environment, topology, command, log, and checksum evidence.
- A framework-neutral fault injection evaluation kit with incident schedules, automatic training-iteration hooks, verified ground truth, capability-checked executors, neutral localization scoring, and a systematic eight-GPU DDP/SCOUT evaluation matrix.

### Changed

- Quick Start installs the core package from PyPI before introducing optional framework integrations.
- Release publishing now requires a protected default-branch tag, revalidates source and exact artifact digests, uses a pinned build/audit toolchain, emits build provenance, and publishes draft-populated immutable GitHub releases.
- GEMINI node-local checkpoints now use a versioned, schema-validated, weights-only format that rejects arbitrary pickle globals. The unrestricted-pickle format written by `0.1.0` is intentionally not loadable after this change.
- The release Quick Start now runs an entry point shipped in the installed wheel instead of an example from the moving `main` branch.

## [0.1.0] - 2026-08-12

### Added

- GEMINI asynchronous in-memory checkpointing, peer replication, node-local persistence, and recovery.
- SCOUT replay-based localization for silent data corruption, stragglers, collective desynchronization, and process stalls.
- SCOUT-certified durable checkpoint selection.
- Unified integrations for native PyTorch, TorchTitan, Megatron Core, and DeepSpeed.
- Platform-neutral orchestration and manager APIs.
- Stable user and manager namespaces with an explicit experimental namespace.
- CPU, distributed GPU, MoE execution-regime, framework-parallelism, and recovery validation suites.

## Release Process

Accumulate user-visible changes under `Unreleased`.
Before tagging a release, move those entries under the new version, replace `Pending` with the release date, and update the versions in `pyproject.toml` and `CITATION.cff`.
Release artifacts are built and published by the [Release workflow](.github/workflows/release.yml) under the documented [release process](docs/release.md).
