# LM Resiliency Documentation

LM Resiliency provides GEMINI checkpoint recovery and SCOUT failure localization for distributed LLM training.

## Start here

- [API Guide](api.md) — public APIs, framework integration, recovery, and manager hooks.
- [Compatibility](compatibility.md) — supported and tested Python/framework versions.
- [GEMINI](gemini.md) — checkpoint tiers, replication, persistence, and recovery semantics.
- [SCOUT](scout.md) — failure coverage, replay, consensus, and localization contracts.
- [Validation](validation.md) — qualification evidence, environments, and known boundaries.
- [Release Process](release.md) — artifact validation and publication workflow.

## Documentation versions

The hosted site is versioned with `mike`:

- `/dev/` tracks the current `main` branch.
- `/<version>/` contains documentation published from a `v<version>` release tag.
- `/latest/` aliases the most recently published release documentation.

The repository Markdown remains the source of truth for documentation changes. CI builds the site with warnings treated as errors before changes can satisfy the required CI gate.
