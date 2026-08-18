# LM Resiliency Codex Guidance

## Repository purpose

LM Resiliency protects long-running distributed LLM training with two complementary mechanisms:

- **GEMINI** provides frequent asynchronous in-memory checkpoints, peer replication, and fast recovery.
- **SCOUT** detects and localizes silent data corruption (SDC), stragglers, collective desynchronization, process stalls, and selected hardware failures, and participates in checkpoint certification.

Correctness under partial failure is more important than convenience. Prefer conservative behavior when evidence is incomplete or contradictory.

## Working guidance

- Read `CONTRIBUTING.md` before changing code.
- For checkpoint or recovery changes, read `docs/gemini.md` and the relevant tests.
- For detection, replay, consensus, hang, telemetry, or checkpoint-certification changes, read `docs/scout.md` and the relevant tests.
- For public APIs, framework adapters, package imports, or supported versions, read `docs/compatibility.md` and `docs/api.md`.
- Treat this file as review guidance, not as a replacement for tests or the documented runtime contracts.
- Do not duplicate deterministic CI feedback. Ruff, formatting, pre-commit, packaging, and the CPU unit suite are handled by CI.
- Do not weaken a documented safety property merely to make a test pass. Update implementation, tests, and documentation together when the contract intentionally changes.
- Keep optional framework dependencies lazy. Importing `lm_resiliency` must not require DeepSpeed, Megatron Core, TorchTitan, Triton, or CUDA-only packages.

## Pull request review workflow

Use Codex review in batches rather than after every revision commit.

Do not begin editing when the first review comment arrives.

For the current PR head SHA:

1. Wait until the Codex GitHub review has been submitted.
2. Read every unresolved review thread.
3. Run a complete independent review of the branch diff against the base branch.
4. Combine and deduplicate all findings.
5. Fix every accepted finding in one revision, run the relevant tests, and push once.
6. Do not request another broad review until this revision is complete.

- The native Codex GitHub integration handles the initial review when a pull request is opened for review or moved from draft to ready.
- When addressing review feedback, first inspect **all unresolved Codex review threads** and treat the complete set of actionable findings as one fix batch.
- Implement the whole batch, add or update focused regression tests, run the relevant deterministic checks, and inspect the resulting diff before asking for another review.
- Do **not** request `@codex review` after each commit, formatting fix, test-only adjustment, or other intermediate revision.
- Resolve a review thread only after its underlying finding is actually addressed or intentionally rejected with a documented rationale.
- After the current batch is addressed and the relevant CI checks pass, request **one** fresh `@codex review` to look for newly introduced or previously missed issues.
- If that re-review produces new actionable findings, repeat the same batch process and request one additional review only after the next batch is complete.
- A clean later review does not automatically resolve earlier GitHub review threads; verify the old findings are addressed and explicitly resolve those conversations before merge.

## Code Review Rules

Review for concrete correctness, safety, compatibility, and performance regressions. Prefer a small number of high-confidence findings over speculative comments. Do not report style-only issues that deterministic tooling can catch.

Use these severities:

- **P0**: can corrupt training state, select unsafe recovery state, cause widespread data loss, or create a severe security issue.
- **P1**: can deadlock/hang workers, misattribute a failure, break recovery, violate a documented compatibility contract, or introduce a major regression in a supported path.
- **P2**: real but narrower correctness, robustness, test-coverage, or performance issue that should be fixed before relying on the affected path.

For every finding, explain the concrete failure mode and point to the smallest relevant file/line range. Do not raise a finding when the concern is only hypothetical and cannot be tied to reachable behavior.

### Distributed correctness and liveness

- Verify all participating ranks execute compatible collectives in the same order and with compatible tensor metadata.
- Flag one-sided waits, mismatched barriers, lock ordering hazards, background-thread races, unsafe process-group reuse/destruction, and shutdown paths that can strand peers.
- Treat timeout, cancellation, signal, exception, worker-loss, and partial-initialization paths as first-class behavior.
- Check that rank-local decisions do not accidentally become job-wide decisions without consensus, and that job-wide decisions are actually agreed across the required ranks.
- Preserve topology semantics across DP, DDP, FSDP/HSDP, TP, SP, CP, PP, EP, expert TP, ZeRO, and framework-specific process groups when the changed code applies to them.

### GEMINI checkpoint and recovery invariants

- A checkpoint must represent one coherent optimizer step. Flag any path that can mix buffers, metadata, RNG, scheduler, sampler/input position, optimizer state, or model state from different steps.
- Only completed checkpoint slots may be serialized, replicated as complete, or selected for recovery.
- Buffer reuse must not race an unfinished GPU-to-host copy, peer transfer, serialization, integrity check, or recovery read.
- Recovery-step selection must remain rank-consistent. A newer step available on only part of the job must not be selected as a global recovery point.
- Peer replication and local persistence must preserve ownership, step, metadata, and trust state across failures and restart.
- Detected SDC, inaccessible required ranks/machines, incomplete emergency evidence, or failed integrity checks must not promote an unsafe checkpoint.
- Dynamic-catalog `CANDIDATE` and `RECOVERY_VERIFIED` transitions must preserve the documented two-cycle certification semantics; dense accepted checks may certify immediately as documented.
- Do not treat CRC/checksum success as proof that source GPU state was numerically correct.

### SCOUT detection and certification invariants

- Attribution based on peer comparison requires equivalent peers and the documented healthy-majority/consensus conditions. A mismatch alone is not enough to name a faulty rank.
- Preserve `Agree`, `Attributed`, and `Inconclusive` semantics. Inconclusive exact evidence must remain conservative and must not certify unsafe checkpoint state.
- Job-wide checkpoint trust decisions must incorporate every required comparison group; one group detecting SDC must not be ignored by another group.
- Emergency replay must fail safe when evidence is missing, incomplete, stale, unavailable, or internally contradictory.
- Rank, host, GPU, and communication-endpoint attribution must not overclaim beyond the evidence source and available topology/resource mapping.
- Keep common-mode and single-owner limitations explicit; do not silently turn unsupported detection cases into "healthy" results.
- Out-of-band progress, replay, health telemetry, and training-process lifecycle changes must not create new deadlocks or false liveness assumptions.

### Framework integration and compatibility

- Preserve stable public APIs, checkpoint formats, manager contracts, and documented import paths unless the change explicitly updates the compatibility contract.
- Adapter discovery must not silently bind the wrong framework object or topology.
- Framework-specific code must remain isolated behind optional dependencies and capability checks.
- Validate changed behavior against the supported version ranges in `pyproject.toml` and `docs/compatibility.md`; do not assume APIs from an unpinned upstream development branch.
- Manager-facing reports and `RecoveryDecision`/fault records must remain JSON-serializable and semantically consistent when applicable.

### Concurrency, resources, and failure cleanup

- Check ownership and cleanup of CUDA events/streams, CPU pinned memory, files, sockets, process groups, subprocesses, threads, signal handlers, and temporary state.
- Flag use-after-close, double-finalization, leaked background work, daemon-thread assumptions, or cleanup that can block indefinitely after a failure.
- Ensure error paths preserve the original actionable failure while still performing bounded cleanup.

### Performance regressions

- The healthy training path is latency-sensitive. Flag new global synchronizations, device synchronizations, blocking filesystem I/O, blocking network operations, or unbounded Python work on frequent optimizer boundaries unless justified.
- Check that diagnostic/replay work remains sampled or bounded as documented and that large tensors are not accidentally cloned, serialized, or transferred on every step.
- Treat silent changes in communication volume, checkpoint cadence, replication chunking, or memory footprint as compatibility/performance concerns when they materially affect production behavior.

### Tests and validation

- Require focused regression tests for changed failure semantics, trust-state transitions, recovery selection, consensus, or public contracts.
- CPU tests are necessary but not sufficient for code whose correctness depends on CUDA, distributed collectives, multiple ranks, multiple nodes, or a specific training framework.
- For distributed/GPU-only behavior, require the smallest meaningful targeted validation and ensure the PR documents any hardware-dependent validation that remains outstanding.
- Do not require unrelated broad GPU testing when a focused test can establish the changed invariant.
