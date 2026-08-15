"""Classify and order release tags according to PEP 440."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterable

from packaging.version import InvalidVersion, Version


def is_prerelease(value: str) -> bool:
    """Return whether a validated package version is a pre- or development release."""
    return Version(value).is_prerelease


def latest_stable_tag(tags: Iterable[str]) -> str:
    """Return the newest stable ``v`` tag, ignoring unrelated release tags."""
    candidates: list[tuple[Version, str]] = []
    for raw_tag in tags:
        tag = raw_tag.strip()
        if not tag.startswith("v"):
            continue
        try:
            version = Version(tag[1:])
        except InvalidVersion:
            continue
        if not version.is_prerelease:
            candidates.append((version, tag))
    if not candidates:
        raise ValueError("no stable PEP 440 release tags were provided")
    return max(candidates)[1]


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    prerelease = subparsers.add_parser("is-prerelease")
    prerelease.add_argument("version")
    subparsers.add_parser("latest-stable-tag")
    args = parser.parse_args()

    try:
        if args.command == "is-prerelease":
            print(str(is_prerelease(args.version)).lower())
        else:
            print(latest_stable_tag(sys.stdin))
    except (InvalidVersion, ValueError) as error:
        print(f"release version validation failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
