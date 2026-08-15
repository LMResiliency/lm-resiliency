"""MkDocs hooks for links that intentionally leave the documentation tree."""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path
from urllib.parse import quote

_REPOSITORY = os.getenv("GITHUB_REPOSITORY", "LMResiliency/lm-resiliency")
_REPO_LINK = re.compile(r"(?P<prefix>\]\()(?P<target>\.\./[^)\s]+)(?P<suffix>\))")


def _checked_out_revision() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "main"
    return completed.stdout.strip() or "main"


_SOURCE_REF = _checked_out_revision()


def _repository_url(target: str) -> str:
    path_and_fragment = target[3:]
    path, separator, fragment = path_and_fragment.partition("#")
    url = f"https://github.com/{_REPOSITORY}/blob/{_SOURCE_REF}/{quote(path, safe='/')}"
    if separator:
        url = f"{url}#{fragment}"
    return url


def on_page_markdown(markdown: str, **_: object) -> str:
    """Rewrite repository-relative source links to revision-pinned GitHub URLs."""

    return _REPO_LINK.sub(
        lambda match: (
            f"{match.group('prefix')}"
            f"{_repository_url(match.group('target'))}"
            f"{match.group('suffix')}"
        ),
        markdown,
    )
