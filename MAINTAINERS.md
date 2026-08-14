# Maintainers

LM Resiliency currently uses joint maintainership rather than separate subsystem teams.

## Current maintainers

- `@zhuangwang93`
- `@lmresiliencydev`

Both maintainers are listed as code owners for the repository's current critical surfaces. This keeps review routing explicit without implying a specialization split that does not yet exist.

## Ownership map

| Surface | Current owners | Review focus |
|---|---|---|
| GEMINI checkpointing and recovery | `@zhuangwang93`, `@lmresiliencydev` | checkpoint coherence, persistence, recovery selection, resource cleanup |
| SCOUT detection and localization | `@zhuangwang93`, `@lmresiliencydev` | replay safety, consensus, localization semantics, fail-safe behavior |
| Framework integrations | `@zhuangwang93`, `@lmresiliencydev` | framework lifecycle, topology mapping, optional imports, compatibility |
| Public API and orchestration | `@zhuangwang93`, `@lmresiliencydev` | stable interfaces, manager contracts, backward compatibility |
| Tests and validation | `@zhuangwang93`, `@lmresiliencydev` | regression coverage, reproducibility, qualification boundaries |
| Documentation and compatibility | `@zhuangwang93`, `@lmresiliencydev` | release alignment, supported-version claims, user-facing contracts |
| CI, release, governance, and security metadata | `@zhuangwang93`, `@lmresiliencydev` | branch/release safety, workflow permissions, reporting and policy surfaces |

## Review and fallback policy

`CODEOWNERS` is used to route review requests. Code-owner review is not a substitute for the repository's branch rules, CI checks, or required conversation resolution.

When a change spans multiple surfaces, request review from the owners of every affected contract. If an owner is temporarily unavailable, keep the pull request open and document which review is outstanding rather than bypassing validation. If ownership becomes a recurring bottleneck, update this file and `CODEOWNERS` in the same pull request to assign another trusted maintainer.

Security vulnerabilities must follow [SECURITY.md](SECURITY.md) and must not be disclosed in public review threads.

## Changing ownership

Ownership changes require a focused pull request that updates both `MAINTAINERS.md` and `.github/CODEOWNERS`. The pull request should explain the new responsibility boundary and verify that representative paths request the intended reviewer(s).
