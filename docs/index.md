# LM Resiliency Documentation

LM Resiliency provides GEMINI checkpoint recovery and SCOUT failure localization for distributed LLM training.

## Start here

- [API Guide](api.md) — public APIs, framework integration, recovery, and manager hooks.
- [Compatibility](compatibility.md) — supported and tested Python/framework versions.
- [GEMINI](gemini.md) — checkpoint tiers, replication, persistence, and recovery semantics.
- [SCOUT](scout.md) — failure coverage, replay, consensus, and localization contracts.
- [Torchrun Resiliency](torchrun_resiliency.md) — fixed-size replacement architecture,
  coordinator protocol, fencing, and restart context.
- [Validation](validation.md) — qualification evidence, environments, and known boundaries.
- [Release Process](release.md) — artifact validation and publication workflow.

## Documentation versions

The hosted site publishes immutable static trees by source revision:

- `/dev/` tracks the current `main` branch.
- `/<version>/` contains documentation built from the successfully validated `v<version>` release revision.
- `/latest/` mirrors the most recently published stable release documentation.

Before the first stable documentation tree is published, the site root redirects to `/dev/`. Once a stable release is published through the repository release workflow, the root redirects to `/latest/`.

Publication workflows stage the complete versioned static tree on `gh-pages`. A dedicated trusted workflow then uploads the current staged tree as a GitHub Pages artifact and deploys it through GitHub's Pages deployment API. Repository Pages settings therefore need to use **GitHub Actions** as the publishing source rather than building directly from the `gh-pages` branch.

The repository Markdown remains the source of truth for documentation changes. CI builds the site with warnings treated as errors before changes can satisfy the required CI gate.
