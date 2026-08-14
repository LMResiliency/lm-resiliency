# Release Process

Production releases use one identified wheel and source distribution from build through validation, PyPI publication, and the immutable GitHub release.

## Repository Controls

- The active `release tag protection` ruleset matches `refs/tags/v*`. Only organization administrators can create, update, or delete matching tags; non-fast-forward tag changes are blocked.
- Repository release immutability is enabled for releases created after `v0.1.0`. Published release tags and assets cannot be changed. The `v0.1.0` release predates this setting and remains outside the immutable-release guarantee.
- PyPI publication uses the protected `pypi` environment and OIDC trusted publishing. TestPyPI uses its separate protected environment.

## Production Release

1. Merge the release changes through protected `main` and wait for required CI and review policy.
2. Update `pyproject.toml`, `CITATION.cff`, and this changelog to the same version.
3. An organization administrator creates and pushes `v<version>` at the intended `main` commit. Do not reuse or move a release tag.
4. The release workflow verifies that the tag matches `pyproject.toml`, resolves to the checked-out commit, and is an ancestor of current `origin/main`.
5. The workflow reruns the pinned primary CPU suite, builds with a pinned toolchain, and writes `release-manifest.json` and `SHA256SUMS` for the wheel and source distribution.
6. The exact downloaded artifacts pass manifest verification, clean installation, Quick Start, `pip check`, dependency audit, and GitHub build-provenance verification.
7. After protected-environment approval, the same artifact bundle is published to PyPI.
8. The workflow creates a draft GitHub release, attaches every artifact plus its manifest and checksums, publishes it, then verifies the immutable release and each local asset.

The workflow fails before production publication if any tag, revision, version, digest, artifact membership, validation, audit, provenance, environment approval, or immutable-release check fails.

## TestPyPI

A trusted maintainer can dispatch the same workflow on a selected revision. It builds and validates an identified artifact bundle, publishes it to TestPyPI after environment approval, and verifies a clean installation. Manual dispatch never enters the production PyPI or GitHub release jobs because those jobs require a `refs/tags/v*` event.

## Verification

Consumers can verify a future immutable release and a downloaded asset with GitHub CLI:

```bash
gh release verify vX.Y.Z --repo LMResiliency/lm-resiliency
gh release verify-asset vX.Y.Z ./lm_resiliency-X.Y.Z-py3-none-any.whl \
  --repo LMResiliency/lm-resiliency
gh attestation verify ./lm_resiliency-X.Y.Z-py3-none-any.whl \
  --repo LMResiliency/lm-resiliency
```
