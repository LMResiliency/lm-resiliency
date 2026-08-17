"""Install an LM Resiliency worker adapter before the user module starts."""

from __future__ import annotations

import os
import traceback


def _install() -> None:
    try:
        from lm_resiliency.integrations.torchrun.worker_adapter import (
            bootstrap_worker_from_environment,
        )

        bootstrap_worker_from_environment()
    except BaseException:  # Python otherwise suppresses sitecustomize failures.
        traceback.print_exc()
        os._exit(78)


_install()
