"""Validation wrapper around one unchanged framework production loop."""

from __future__ import annotations

import argparse
import importlib
import sys

from examples.torchrun.adapter_bootstrap._validation import (
    assert_torchrun_adapter_attached,
)

FRAMEWORKS = ("pytorch", "deepspeed", "megatron", "torchtitan")


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    arguments, remaining = parser.parse_known_args()
    importlib.import_module(arguments.framework)
    module_name = f"examples.production_loops.{arguments.framework}"
    module = importlib.import_module(module_name)
    sys.argv = [module_name, *remaining]
    module.main()
    assert_torchrun_adapter_attached()


if __name__ == "__main__":
    main()
