"""Config/version drift detection across the nodes of a run.

A silent mismatch in the software stack across nodes — different NCCL, CUDA, cuDNN, torch,
GPU driver, or GPU model — is a classic source of *subtle* failures: hangs in a collective,
numerical divergence between ranks, or a crash deep in a kernel, all far from the real
cause (one node was reimaged / rebuilt / scheduled on different hardware). Cheaper to catch
up front than to debug at step 10k.

This is a **preflight, report-only** check: each worker publishes a small version
fingerprint to the control plane at startup; the supervisor gathers them once the cohort is
up and logs any key whose value is not identical across all ranks. It does not block or
recover — it surfaces the mismatch so an operator can decide (the failure it prevents is
subtle, and a false abort on a benign difference would be worse).
"""

from __future__ import annotations

import platform

import torch


def local_fingerprint() -> dict[str, str]:
    """This process's software-stack fingerprint (the versions that must agree across ranks)."""
    fp: dict[str, str] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_build": str(torch.version.cuda),  # CUDA torch was built against
        "cudnn": str(torch.backends.cudnn.version()),
    }
    try:
        fp["nccl"] = ".".join(str(v) for v in torch.cuda.nccl.version())
    except Exception:  # noqa: BLE001 — no CUDA / NCCL unavailable
        pass
    try:
        import pynvml

        pynvml.nvmlInit()
        drv = pynvml.nvmlSystemGetDriverVersion()
        fp["driver"] = drv.decode() if isinstance(drv, bytes) else str(drv)
    except Exception:  # noqa: BLE001 — NVML absent (CPU box) / not permitted
        pass
    if torch.cuda.is_available():
        try:
            fp["gpu"] = torch.cuda.get_device_name(0)
        except Exception:  # noqa: BLE001
            pass
    return fp


def find_drift(fingerprints: dict[str, dict[str, str]]) -> dict[str, dict[str, str]]:
    """Return the fingerprint keys that are **not identical** across all reporting nodes.

    Args:
        fingerprints: node/rank id → its ``local_fingerprint()``.
    Returns:
        differing key → {node id → value} (``"<missing>"`` where a node didn't report the
        key). Empty when the whole cohort agrees.
    """
    drift: dict[str, dict[str, str]] = {}
    if len(fingerprints) < 2:
        return drift
    keys: set[str] = set().union(*(fp.keys() for fp in fingerprints.values()))
    for key in sorted(keys):
        by_node = {node: fp.get(key, "<missing>") for node, fp in fingerprints.items()}
        if len(set(by_node.values())) > 1:
            drift[key] = by_node
    return drift


def format_drift(drift: dict[str, dict[str, str]]) -> str:
    """Human-readable one-block summary of the drift (grouping nodes by shared value)."""
    lines = []
    for key, by_node in drift.items():
        groups: dict[str, list[str]] = {}
        for node, val in by_node.items():
            groups.setdefault(val, []).append(node)
        parts = "; ".join(f"{val!r} on {sorted(nodes)}" for val, nodes in groups.items())
        lines.append(f"  {key}: {parts}")
    return "\n".join(lines)
