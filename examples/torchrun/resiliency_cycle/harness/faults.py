"""Isolated executors for the resiliency-cycle failure-type matrix."""

from __future__ import annotations

import hashlib
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from lm_resiliency import (
    CallbackFaultExecutor,
    FailureType,
    FaultExecutionRequest,
    FaultExecutionResult,
    SafetyClass,
)


def isolated_fault_executor(root: Path) -> CallbackFaultExecutor:
    """Return an executor that confines destructive effects to disposable resources."""

    root.mkdir(parents=True, exist_ok=True)

    def validate(request: FaultExecutionRequest) -> None:
        metadata = request.fault.target.metadata
        if metadata.get("executor") != "isolated_validation":
            raise ValueError("resiliency-cycle faults require executor='isolated_validation'")
        resource = request.fault.target.resource
        if not isinstance(resource, str) or not resource.startswith("resiliency-cycle:"):
            raise ValueError("resiliency-cycle faults require an isolated resource target")

    return CallbackFaultExecutor(
        name="resiliency-cycle-isolated",
        supported_types=set(FailureType),
        activate=lambda request: _activate_isolated(root, request),
        validate=validate,
        one_shot=True,
        max_safety=SafetyClass.CLUSTER_DESTRUCTIVE,
    )


def _activate_isolated(
    root: Path,
    request: FaultExecutionRequest,
) -> FaultExecutionResult:
    failure_type = request.fault.type
    occurrence_root = root / _safe_occurrence_name(request.occurrence_id)
    occurrence_root.mkdir(parents=True, exist_ok=True)
    handlers = {
        FailureType.TENSOR_CORRUPTION: _tensor_corruption,
        FailureType.STALE_STATE: _stale_state,
        FailureType.DROP: _drop,
        FailureType.DUPLICATE: _duplicate,
        FailureType.REORDER: _reorder,
        FailureType.DELAY: _delay,
        FailureType.HANG: _hang,
        FailureType.TIMEOUT: _timeout,
        FailureType.EXCEPTION: _exception,
        FailureType.RESOURCE_EXHAUSTION: _resource_exhaustion,
        FailureType.PROCESS_TERMINATION: _process_termination,
        FailureType.RESOURCE_UNAVAILABLE: _resource_unavailable,
        FailureType.CHECKPOINT_CORRUPTION: _checkpoint_corruption,
        FailureType.CHECKPOINT_TRUNCATION: _checkpoint_truncation,
        FailureType.CHECKPOINT_MISSING: _checkpoint_missing,
        FailureType.IO_ERROR: _io_error,
        FailureType.PAYLOAD_CORRUPTION: _payload_corruption,
        FailureType.COLLECTIVE_DESYNC: _collective_desync,
        FailureType.MESSAGE_DROP: _message_drop,
        FailureType.NETWORK_PARTITION: _network_partition,
        FailureType.CONFIG_DRIFT: _config_drift,
    }
    evidence = handlers[failure_type](occurrence_root, request)
    if not _effect_observed(failure_type, evidence):
        raise RuntimeError(f"isolated {failure_type.value} effect was not observed")
    return FaultExecutionResult(
        verified=True,
        active=False,
        evidence={
            "failure_type": failure_type.value,
            "isolation": "rank-local disposable sandbox",
            **evidence,
        },
    )


def _effect_observed(failure_type: FailureType, evidence: dict[str, Any]) -> bool:
    predicates = {
        FailureType.TENSOR_CORRUPTION: lambda: evidence["after"] != evidence["before"],
        FailureType.STALE_STATE: lambda: evidence["selected_step"] < evidence["latest_step"],
        FailureType.DROP: lambda: evidence["output_count"] < evidence["input_count"],
        FailureType.DUPLICATE: lambda: evidence["output_count"] > evidence["input_count"],
        FailureType.REORDER: lambda: evidence["observed"] != evidence["expected"],
        FailureType.DELAY: lambda: evidence["elapsed_ms"] >= evidence["requested_ms"],
        FailureType.HANG: lambda: evidence["bounded_watchdog_detected_hang"],
        FailureType.TIMEOUT: lambda: evidence["operation_timed_out"],
        FailureType.EXCEPTION: lambda: bool(evidence["caught_exception"]),
        FailureType.RESOURCE_EXHAUSTION: lambda: evidence["exhausted"],
        FailureType.PROCESS_TERMINATION: lambda: evidence["terminated"],
        FailureType.RESOURCE_UNAVAILABLE: lambda: evidence["unavailable"],
        FailureType.CHECKPOINT_CORRUPTION: lambda: evidence["corrupted"],
        FailureType.CHECKPOINT_TRUNCATION: lambda: (
            evidence["truncated_bytes"] < evidence["original_bytes"]
        ),
        FailureType.CHECKPOINT_MISSING: lambda: evidence["missing"],
        FailureType.IO_ERROR: lambda: bool(evidence["io_error"]),
        FailureType.PAYLOAD_CORRUPTION: lambda: evidence["corrupted"],
        FailureType.COLLECTIVE_DESYNC: lambda: evidence["desynchronized"],
        FailureType.MESSAGE_DROP: lambda: evidence["dropped"] > 0,
        FailureType.NETWORK_PARTITION: lambda: evidence["connection_blocked"],
        FailureType.CONFIG_DRIFT: lambda: evidence["drifted"],
    }
    return bool(predicates[failure_type]())


def _safe_occurrence_name(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:24]


def _tensor_corruption(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    before = [1.0, -2.0]
    after = [-value for value in before]
    return {"before": before, "after": after, "operation": "sign_flip"}


def _stale_state(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    history = [{"step": 1, "value": 11}, {"step": 2, "value": 22}]
    selected = history[0]
    return {"latest_step": history[-1]["step"], "selected_step": selected["step"]}


def _drop(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    values = ["a", "b", "c"]
    observed = values[:1] + values[2:]
    return {"input_count": len(values), "output_count": len(observed)}


def _duplicate(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    values = ["a", "b"]
    observed = [values[0], values[0], values[1]]
    return {"input_count": len(values), "output_count": len(observed)}


def _reorder(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    expected = [0, 1, 2]
    observed = [1, 0, 2]
    return {"expected": expected, "observed": observed}


def _delay(_root: Path, request: FaultExecutionRequest) -> dict[str, Any]:
    requested_ms = float(request.fault.parameters["delay_ms"])
    started = time.monotonic()
    time.sleep(requested_ms / 1_000.0)
    elapsed_ms = (time.monotonic() - started) * 1_000.0
    return {"elapsed_ms": elapsed_ms, "requested_ms": requested_ms}


def _hang(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    timed_out = False
    try:
        process.wait(timeout=0.02)
    except subprocess.TimeoutExpired:
        timed_out = True
    finally:
        process.terminate()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=1)
    return {"bounded_watchdog_detected_hang": timed_out}


def _timeout(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    timed_out = False
    try:
        subprocess.run(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=0.02,
        )
    except subprocess.TimeoutExpired:
        timed_out = True
    return {"operation_timed_out": timed_out}


def _exception(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    message = ""
    try:
        raise RuntimeError("isolated validation exception")
    except RuntimeError as error:
        message = str(error)
    return {"caught_exception": message}


def _resource_exhaustion(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    quota = 1_024
    requested = 2_048
    exhausted = requested > quota
    return {"exhausted": exhausted, "quota_bytes": quota, "requested_bytes": requested}


def _process_termination(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    process = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(60)"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    process.terminate()
    return_code = process.wait(timeout=1)
    return {"child_return_code": return_code, "terminated": return_code != 0}


def _resource_unavailable(
    root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    missing = root / "unavailable"
    unavailable = False
    try:
        missing.read_bytes()
    except FileNotFoundError:
        unavailable = True
    return {"resource": str(missing), "unavailable": unavailable}


def _checkpoint_corruption(
    root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    path = root / "checkpoint.bin"
    original = b"checkpoint-payload"
    path.write_bytes(original)
    corrupted = bytearray(path.read_bytes())
    corrupted[0] ^= 0xFF
    path.write_bytes(corrupted)
    return {
        "corrupted": path.read_bytes() != original,
        "path": str(path),
    }


def _checkpoint_truncation(
    root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    path = root / "checkpoint.bin"
    original = b"checkpoint-payload"
    path.write_bytes(original)
    path.write_bytes(original[:4])
    return {
        "original_bytes": len(original),
        "path": str(path),
        "truncated_bytes": path.stat().st_size,
    }


def _checkpoint_missing(
    root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    path = root / "checkpoint.bin"
    path.write_bytes(b"checkpoint")
    path.unlink()
    return {"missing": not path.exists(), "path": str(path)}


def _io_error(root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    path = root / "directory"
    path.mkdir()
    error_name = ""
    try:
        path.read_bytes()
    except OSError as error:
        error_name = type(error).__name__
    return {"io_error": error_name, "path": str(path)}


def _payload_corruption(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    original = b"collective-payload"
    corrupted = bytearray(original)
    corrupted[-1] ^= 0x01
    return {
        "corrupted": bytes(corrupted) != original,
        "original_digest": hashlib.sha256(original).hexdigest(),
        "observed_digest": hashlib.sha256(corrupted).hexdigest(),
    }


def _collective_desync(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    expected_sequence = [10, 11, 12]
    observed_sequence = [10, 12, 11]
    return {
        "desynchronized": observed_sequence != expected_sequence,
        "expected_sequence": expected_sequence,
        "observed_sequence": observed_sequence,
    }


def _message_drop(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    sent = ["message"]
    received: list[str] = []
    return {"dropped": len(sent) - len(received), "sent": len(sent)}


def _network_partition(
    _root: Path,
    _request: FaultExecutionRequest,
) -> dict[str, Any]:
    sender, receiver = socket.socketpair()
    blocked = False
    try:
        receiver.close()
        try:
            sender.sendall(b"probe")
        except (BrokenPipeError, ConnectionResetError):
            blocked = True
    finally:
        sender.close()
        receiver.close()
    return {"connection_blocked": blocked, "transport": "socketpair"}


def _config_drift(_root: Path, _request: FaultExecutionRequest) -> dict[str, Any]:
    expected = b'{"learning_rate": 0.001}'
    observed = b'{"learning_rate": 0.002}'
    return {
        "drifted": expected != observed,
        "expected_digest": hashlib.sha256(expected).hexdigest(),
        "observed_digest": hashlib.sha256(observed).hexdigest(),
    }
