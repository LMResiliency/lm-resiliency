"""Install an LM Resiliency worker adapter before the user module starts."""

from __future__ import annotations

import os
import sys
import traceback
from importlib.machinery import PathFinder
from importlib.util import module_from_spec
from pathlib import Path


def _run_existing_sitecustomize() -> None:
    current = Path(__file__).resolve()
    bootstrap_dir = current.parent
    search_path = [item for item in sys.path if item and Path(item).resolve() != bootstrap_dir]
    spec = PathFinder.find_spec("sitecustomize", search_path)
    if spec is None or spec.loader is None or spec.origin is None:
        return
    try:
        if Path(spec.origin).resolve() == current:
            return
    except OSError:
        return
    delegated = module_from_spec(spec)
    current_module = sys.modules.get("sitecustomize")
    sys.modules["sitecustomize"] = delegated
    try:
        spec.loader.exec_module(delegated)
    finally:
        if current_module is None:
            sys.modules.pop("sitecustomize", None)
        else:
            sys.modules["sitecustomize"] = current_module


def _install() -> None:
    try:
        _run_existing_sitecustomize()
        from lm_resiliency.integrations.torchrun.worker_adapter import (
            bootstrap_worker_from_environment,
        )

        bootstrap_worker_from_environment()
    except BaseException:  # Python otherwise suppresses sitecustomize failures.
        traceback.print_exc()
        os._exit(78)


_install()
