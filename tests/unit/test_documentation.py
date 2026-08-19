from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

import docs_hooks

ROOT = Path(__file__).parents[2]


def test_source_links_use_checked_out_revision(monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "event-sha-that-was-not-checked-out")
    reloaded = importlib.reload(docs_hooks)
    checked_out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    rewritten = reloaded.on_page_markdown("[source](../README.md)")

    assert f"/blob/{checked_out}/README.md" in rewritten
    assert "event-sha-that-was-not-checked-out" not in rewritten


def test_deployment_contracts_cover_every_destination():
    mkdocs = (ROOT / "mkdocs.yml").read_text()
    development = (ROOT / ".github/workflows/docs.yml").read_text()
    online_links = (ROOT / ".github/workflows/docs-online-links.yml").read_text()
    release = (ROOT / ".github/workflows/release.yml").read_text()
    pages = (ROOT / ".github/workflows/pages.yml").read_text()

    assert "site_url:" not in mkdocs
    assert "Torchrun Resiliency: torchrun_resiliency.md" in mkdocs
    assert "[Torchrun Resiliency](torchrun_resiliency.md)" in (ROOT / "docs/index.md").read_text()
    assert "git rm -rf --ignore-unmatch ." in development
    assert "git rm -rf --ignore-unmatch ." in release
    assert "group: release-docs-${{ github.ref_name }}" in release
    assert "for attempt in $(seq 1 5)" in development
    assert "for attempt in $(seq 1 5)" in release
    assert "git reset --hard origin/gh-pages" in development
    assert "git rebase origin/gh-pages" not in development
    assert "git reset --hard origin/gh-pages" in release
    assert "git rebase origin/gh-pages" not in release
    assert "retention-days: 30" in release
    assert "release_flags+=(--prerelease)" in release
    assert "workflow_dispatch" not in development
    assert "scripts/release_versions.py is-prerelease" in release
    assert "ref: ${{ github.sha }}" in release
    assert 'cp scripts/release_versions.py "$RUNNER_TEMP/release_versions.py"' in release
    assert 'python "$RELEASE_VERSION_HELPER" latest-stable-tag' in release
    assert "sort -V" not in release
    assert '"torch==2.13.0"' in release

    assert "schedule:" in online_links
    assert "workflow_dispatch:" in online_links
    assert "pull_request:" not in online_links
    assert "--exclude-all-private" in online_links
    assert "--exclude '^https://doi\\.org/10\\.1145/3600006\\.3613145$'" in online_links
    assert '--github-token "$GH_TOKEN"' in online_links
    assert "--max-retries 3" in online_links
    assert "--offline" not in online_links

    assert "workflow_run:" in pages
    assert "workflow_dispatch:" in pages
    assert "- Documentation" in pages
    assert "- Release" in pages
    assert "github.event.workflow_run.conclusion == 'success'" in pages
    assert "github.event.workflow_run.event == 'push'" in pages
    assert "github.event_name == 'workflow_dispatch'" in pages
    assert "ref: gh-pages" in pages
    assert "group: pages-deploy" in pages
    assert "cancel-in-progress: true" in pages
    assert "pages: write" in pages
    assert "id-token: write" in pages
    assert "name: github-pages" in pages
    assert "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9" in pages
    assert "actions/deploy-pages@cd2ce8fcbc39b97be8ca5fce6e763baed58fa128" in pages
