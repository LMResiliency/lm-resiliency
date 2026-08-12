"""Unit tests for config/version drift detection (detection/config_drift.py)."""

from __future__ import annotations

from lm_resiliency.detection.config_drift import find_drift, format_drift, local_fingerprint


def test_local_fingerprint_has_core_keys():
    fp = local_fingerprint()
    assert fp["python"] and fp["torch"]  # always present
    assert set(fp) >= {"python", "torch", "cuda_build", "cudnn"}


def test_no_drift_when_all_agree():
    fp = {"nccl": "2.28.9", "cuda_build": "13.0", "torch": "2.11.0"}
    assert find_drift({"0": dict(fp), "1": dict(fp), "2": dict(fp)}) == {}


def test_single_node_never_drifts():
    assert find_drift({"0": {"nccl": "2.28.9"}}) == {}


def test_detects_a_mismatched_key():
    drift = find_drift(
        {
            "0": {"nccl": "2.28.9", "cuda_build": "13.0"},
            "1": {"nccl": "2.20.5", "cuda_build": "13.0"},  # NCCL differs
        }
    )
    assert set(drift) == {"nccl"}
    assert drift["nccl"] == {"0": "2.28.9", "1": "2.20.5"}


def test_missing_key_on_one_node_is_drift():
    drift = find_drift({"0": {"driver": "595.71.05"}, "1": {}})
    assert drift["driver"] == {"0": "595.71.05", "1": "<missing>"}


def test_format_groups_nodes_by_value():
    drift = find_drift(
        {
            "0": {"nccl": "2.28.9"},
            "1": {"nccl": "2.28.9"},
            "2": {"nccl": "2.20.5"},
        }
    )
    out = format_drift(drift)
    assert "nccl:" in out
    assert "'2.28.9'" in out and "'2.20.5'" in out
    assert "['0', '1']" in out  # the agreeing majority grouped together
