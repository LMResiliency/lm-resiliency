"""Validation wrapper around one unchanged framework production loop."""

from __future__ import annotations

import argparse
import importlib
import sys

from examples.torchrun.adapter_bootstrap._validation import ObserveTorchrunAdapterAttachment

FRAMEWORKS = ("pytorch", "deepspeed", "megatron", "torchtitan")
FRAMEWORK_IMPORT_ROOTS = {
    "pytorch": "torch",
    "deepspeed": "deepspeed",
    "megatron": "megatron",
    "torchtitan": "torchtitan",
}


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--framework", choices=FRAMEWORKS, required=True)
    arguments, remaining = parser.parse_known_args()
    importlib.import_module(FRAMEWORK_IMPORT_ROOTS[arguments.framework])
    module_name = f"examples.production_loops.{arguments.framework}"
    module = importlib.import_module(module_name)
    sys.argv = [module_name, *remaining]
    with ObserveTorchrunAdapterAttachment():
        module.main()


if __name__ == "__main__":
    main()
