# Changelog

All notable changes to this project are documented in this file.
The project follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and uses [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Quick Start installs the released package from PyPI, while editable installation remains in the development workflow.

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
Release artifacts are built and published by the [Release workflow](.github/workflows/release.yml).
