"""Contract tests for torchrun runtime configuration and context handoff."""

from __future__ import annotations

import fcntl
import json
import os
import queue
import stat
import threading
import time
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, cast

import pytest
from torch.distributed import HashStore
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
    RendezvousConnectionError,
    RendezvousInfo,
    RendezvousParameters,
    RendezvousStateError,
    RendezvousStoreInfo,
    RendezvousTimeoutError,
)

from lm_resiliency.integrations.torchrun._agent_registration import (
    AgentRegistrationManager,
    AgentRegistrationReader,
)
from lm_resiliency.integrations.torchrun._control_store import InMemoryControlStore
from lm_resiliency.integrations.torchrun._coordinator_lease import (
    CoordinatorLeaseManager,
)
from lm_resiliency.integrations.torchrun._generation_state import GenerationStateManager
from lm_resiliency.integrations.torchrun._protocol import (
    FaultEvent,
    HardwareFaultReport,
    RankAssignment,
    RestartContext,
    RestartPlan,
    SlotAssignment,
    WorkerIdentity,
    validate_event_reporter,
)
from lm_resiliency.integrations.torchrun._runtime import (
    RestartContextFile,
    RestartContextFileError,
    SlotAwareRendezvousHandler,
    TorchrunRendezvousPolicy,
    TorchrunRuntimeConfig,
    TorchrunRuntimeConfigError,
)

RUN_ID = "training-run"
POLICY = """\
schema_version = 1
control_endpoint = "control.example:443"
replacement_only = true
max_replacement_generations = 1
registration_lease_duration_ms = 30000
poll_interval_ms = 1000
join_timeout_ms = 300000
"""


class ManualClock:
    def __init__(self, now_unix_ms: int = 1_000) -> None:
        self._now_unix_ms = now_unix_ms
        self._lock = threading.Lock()

    def __call__(self) -> int:
        with self._lock:
            return self._now_unix_ms

    def advance(self, duration_ms: int) -> None:
        with self._lock:
            self._now_unix_ms += duration_ms

    def set(self, now_unix_ms: int) -> None:
        with self._lock:
            self._now_unix_ms = now_unix_ms


class ManualMonotonicClock:
    def __init__(self) -> None:
        self._now = 0.0
        self._lock = threading.Lock()

    def __call__(self) -> float:
        with self._lock:
            return self._now

    def advance(self, duration_seconds: float) -> None:
        with self._lock:
            self._now += duration_seconds


def _write_policy(path: Path, content: str = POLICY) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _parameters(
    path: Path,
    *,
    min_nodes: int = 2,
    max_nodes: int = 3,
    **config: object,
) -> RendezvousParameters:
    runtime_config = {
        "local_world_size": "2",
        "environment_digest": "environment-v1",
        "resource_ids": "gpu-node-a-0;gpu-node-a-1",
        **config,
    }
    return RendezvousParameters(
        backend="lm_resiliency",
        endpoint="rdzv.example:29400",
        run_id=RUN_ID,
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        local_addr="node-a.example",
        config=str(path),
        **runtime_config,
    )


def _environment(context_path: Path) -> dict[str, str]:
    return {
        "LM_RESILIENCY_NODE_ID": "node-a",
        "LM_RESILIENCY_RESTART_CONTEXT": str(context_path),
    }


def _context(*, plan_id: str = "plan-1") -> RestartContext:
    return RestartContext(
        plan_id=plan_id,
        run_id=RUN_ID,
        generation=1,
        node_id="node-a",
        logical_node_slot=0,
        first_global_rank=0,
        local_world_size=2,
        expected_world_size=4,
        topology_digest="topology-v1",
        recovery_mode="latest",
        checkpoint_source="gemini",
        checkpoint_step=40,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-40",
        reason_code="attributed_sdc",
    )


class StaticRecoveryStateReader:
    def __init__(self, state: object) -> None:
        self._state = state
        self.calls = 0

    def read_recovery_state(self) -> object:
        self.calls += 1
        return self._state


def _static_recovery_state(
    plan: RestartPlan,
    *,
    committed_at_unix_ms: int = 1_000,
) -> SimpleNamespace:
    return SimpleNamespace(
        plan=plan,
        publication=SimpleNamespace(committed_at_unix_ms=committed_at_unix_ms),
    )


def _replacement_plan(
    *,
    node_id: str = "node-b",
    restart_deadline_unix_ms: int = 5_000,
) -> RestartPlan:
    return RestartPlan(
        plan_id="plan-1",
        intent_id="intent-1",
        run_id=RUN_ID,
        from_generation=0,
        to_generation=1,
        incident_ids=("incident-1",),
        reason_code="attributed_sdc",
        recovery_mode="latest",
        checkpoint_source="gemini",
        checkpoint_step=40,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-40",
        slot_assignments=(SlotAssignment(0, node_id, 0, 2),),
        quarantined_node_ids=("node-a",),
        expected_world_size=2,
        topology_digest="topology-v1",
        restart_deadline_unix_ms=restart_deadline_unix_ms,
    )


def test_runtime_config_loads_shared_policy_and_node_inputs(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    context_path = tmp_path / "private" / "restart-context.json"

    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(source),
        environment=_environment(context_path),
    )

    assert config.policy.control_endpoint == "control.example:443"
    assert config.policy.max_replacement_generations == 1
    assert (
        config.policy.digest
        == TorchrunRendezvousPolicy(
            control_endpoint="control.example:443",
        ).digest
    )
    assert config.run_id == RUN_ID
    assert config.endpoint == "rdzv.example:29400"
    assert config.min_nodes == 2
    assert config.max_nodes == 3
    assert config.node_id == "node-a"
    assert config.resource_ids == ("gpu-node-a-0", "gpu-node-a-1")
    assert config.restart_context_path == context_path
    assert config.local_addr == "node-a.example"
    assert config.source_path == source


def test_runtime_config_digest_ignores_toml_formatting_and_node_inputs(tmp_path: Path):
    first = _write_policy(tmp_path / "first.toml")
    second = _write_policy(
        tmp_path / "second.toml",
        """
join_timeout_ms=300000
poll_interval_ms=1000
registration_lease_duration_ms=30000
max_replacement_generations=1
replacement_only=true
control_endpoint="control.example:443"
schema_version=1
""",
    )

    first_config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            first,
            node_id="node-a",
            resource_ids="gpu-node-a-0;gpu-node-a-1",
            restart_context_path="/run/a.json",
        ),
        environment={},
    )
    second_config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            second,
            node_id="node-b",
            resource_ids="gpu-node-b-0;gpu-node-b-1",
            restart_context_path="/run/b.json",
        ),
        environment={},
    )

    assert first_config.policy == second_config.policy
    assert first_config.policy.digest == second_config.policy.digest
    assert first_config.registration_digest == second_config.registration_digest
    assert first_config.node_id != second_config.node_id


def test_runtime_config_registration_digest_includes_node_range(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    common: Any = {
        "node_id": "node-a",
        "restart_context_path": "/run/context.json",
    }
    first = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, max_nodes=3, **common),
        environment={},
    )
    second = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, max_nodes=4, **common),
        environment={},
    )

    assert first.policy.digest == second.policy.digest
    assert first.registration_digest != second.registration_digest


def test_runtime_config_registration_digest_includes_local_world_size(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    common: Any = {
        "node_id": "node-a",
        "restart_context_path": "/run/context.json",
    }
    first = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, local_world_size="2", **common),
        environment={},
    )
    second = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, local_world_size="4", **common),
        environment={},
    )

    assert first.registration_digest != second.registration_digest


def test_runtime_config_builds_agent_identity(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            source,
            node_id="node-a",
            restart_context_path="/run/context.json",
        ),
        environment={},
    )

    identity = config.build_agent_identity(
        agent_id="agent-a",
        hostname="host-a",
    )

    assert identity.run_id == RUN_ID
    assert identity.node_id == "node-a"
    assert identity.agent_id == "agent-a"
    assert identity.hostname == "host-a"
    assert identity.local_world_size == 2
    assert identity.resource_ids == ("gpu-node-a-0", "gpu-node-a-1")
    assert identity.environment_digest == config.agent_environment_digest


def test_runtime_config_agent_identity_authorizes_registered_hardware_report(
    tmp_path: Path,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            source,
            node_id="node-a",
            restart_context_path="/run/context.json",
        ),
        environment={},
    )
    identity = config.build_agent_identity(
        agent_id="agent-a",
        hostname="host-a",
    )
    assignment = RankAssignment(
        run_id=RUN_ID,
        generation=0,
        active_nodes=1,
        local_world_size=2,
        slot_to_node_id={0: "node-a"},
        slot_to_rank_range={0: (0, 2)},
        topology_digest="topology-v1",
    )
    reporter = WorkerIdentity(
        run_id=RUN_ID,
        generation=0,
        node_id="node-a",
        agent_id="agent-a",
        logical_node_slot=0,
        global_rank=0,
        local_rank=0,
        local_world_size=2,
        hostname="host-a",
        gpu_uuid="gpu-node-a-0",
        topology_digest="topology-v1",
    )
    event = FaultEvent(
        event_id="fault-gpu-a0",
        incident_id="incident-a",
        run_id=RUN_ID,
        generation=0,
        reporter=reporter,
        optimizer_step=10,
        report=HardwareFaultReport(
            kind="hardware",
            resource_kind="gpu",
            resource_id="gpu-node-a-0",
            metric="uncorrectable_ecc",
            value=1.0,
            severity="fatal",
            message="fatal ECC",
        ),
    )

    validate_event_reporter(
        event,
        assignment,
        agent_identity=identity,
        resource_to_node_id={
            "gpu-node-a-0": "node-a",
            "gpu-node-a-1": "node-a",
        },
        resource_to_kind={
            "gpu-node-a-0": "gpu",
            "gpu-node-a-1": "gpu",
        },
        resource_to_global_rank={
            "gpu-node-a-0": 0,
            "gpu-node-a-1": 1,
        },
    )


def test_runtime_config_resolves_agent_identity_inputs_from_environment(
    tmp_path: Path,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    environment = {
        "LM_RESILIENCY_NODE_ID": "node-a",
        "LM_RESILIENCY_LOCAL_WORLD_SIZE": "4",
        "LM_RESILIENCY_ENVIRONMENT_DIGEST": "environment-v2",
        "LM_RESILIENCY_RESOURCE_IDS": "gpu-node-a-1;gpu-node-a-0",
        "LM_RESILIENCY_RESTART_CONTEXT": "/run/context.json",
    }

    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            source,
            local_world_size=None,
            environment_digest=None,
            resource_ids=None,
        ),
        environment=environment,
    )

    assert config.node_id == "node-a"
    assert config.local_world_size == 4
    assert config.environment_digest == "environment-v2"
    assert config.resource_ids == ("gpu-node-a-0", "gpu-node-a-1")
    assert config.restart_context_path == Path("/run/context.json")


def test_runtime_config_agent_environment_digest_binds_workload_and_runtime(
    tmp_path: Path,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    common: Any = {
        "node_id": "node-a",
        "restart_context_path": "/run/context.json",
    }
    first = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, environment_digest="environment-v1", **common),
        environment={},
    )
    different_workload = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, environment_digest="environment-v2", **common),
        environment={},
    )
    different_runtime = TorchrunRuntimeConfig.from_parameters(
        _parameters(source, max_nodes=4, **common),
        environment={},
    )

    assert first.agent_environment_digest != different_workload.agent_environment_digest
    assert first.agent_environment_digest != different_runtime.agent_environment_digest


def test_runtime_config_rendezvous_overrides_change_resolved_policy(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    context_path = tmp_path / "restart-context.json"

    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            source,
            control_endpoint="override.example:8443",
            max_replacement_generations="1",
            registration_lease_duration_ms="45000",
            poll_interval_ms="500",
            join_timeout_ms="120000",
            replacement_only="true",
        ),
        environment=_environment(context_path),
    )

    assert config.policy.control_endpoint == "override.example:8443"
    assert config.policy.max_replacement_generations == 1
    assert config.policy.registration_lease_duration_ms == 45_000
    assert config.policy.poll_interval_ms == 500
    assert config.policy.join_timeout_ms == 120_000


@pytest.mark.parametrize(
    ("content", "match"),
    [
        ('schema_version = 1.0\ncontrol_endpoint = "control"\n', "schema_version"),
        ('schema_version = 2\ncontrol_endpoint = "control"\n', "schema_version"),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\nunknown = 1\n',
            "unknown rendezvous config",
        ),
        ("schema_version = 1\n", "control_endpoint"),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\nreplacement_only = false\n',
            "replacement_only=false",
        ),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\nreplacement_only = 1\n',
            "replacement_only",
        ),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\n'
            "poll_interval_ms = 1000\nregistration_lease_duration_ms = 1000\n",
            "registration_lease_duration_ms",
        ),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\nmax_replacement_generations = 1.5\n',
            "max_replacement_generations",
        ),
        (
            'schema_version = 1\ncontrol_endpoint = "control"\nmax_replacement_generations = 2\n',
            "exactly one replacement generation",
        ),
    ],
)
def test_runtime_config_rejects_unsafe_policy(
    tmp_path: Path,
    content: str,
    match: str,
):
    source = _write_policy(tmp_path / "rendezvous.toml", content)

    with pytest.raises(TorchrunRuntimeConfigError, match=match):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source),
            environment=_environment(tmp_path / "context.json"),
        )


def test_runtime_config_rejects_unknown_rendezvous_option(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")

    with pytest.raises(TorchrunRuntimeConfigError, match="unknown rendezvous"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source, unexpected="value"),
            environment=_environment(tmp_path / "context.json"),
        )


def test_runtime_config_rejects_fifo_policy_without_blocking(tmp_path: Path):
    source = tmp_path / "rendezvous.toml"
    os.mkfifo(source, mode=0o600)

    with pytest.raises(TorchrunRuntimeConfigError, match="regular file"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source),
            environment=_environment(tmp_path / "context.json"),
        )


@pytest.mark.parametrize(
    ("parameter_name", "environment_name", "parameter_value", "environment_value"),
    [
        ("node_id", "LM_RESILIENCY_NODE_ID", "node-a", "node-b"),
        ("local_world_size", "LM_RESILIENCY_LOCAL_WORLD_SIZE", "2", "4"),
        (
            "environment_digest",
            "LM_RESILIENCY_ENVIRONMENT_DIGEST",
            "environment-v1",
            "environment-v2",
        ),
        (
            "resource_ids",
            "LM_RESILIENCY_RESOURCE_IDS",
            "gpu-node-a-0;gpu-node-a-1",
            "gpu-node-a-0;hca-node-a",
        ),
        (
            "restart_context_path",
            "LM_RESILIENCY_RESTART_CONTEXT",
            "/run/a.json",
            "/run/b.json",
        ),
    ],
)
def test_runtime_config_rejects_conflicting_node_inputs(
    tmp_path: Path,
    parameter_name: str,
    environment_name: str,
    parameter_value: str,
    environment_value: str,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    environment = {
        "LM_RESILIENCY_NODE_ID": "node-a",
        "LM_RESILIENCY_RESTART_CONTEXT": "/run/context.json",
    }
    environment[environment_name] = environment_value

    with pytest.raises(TorchrunRuntimeConfigError, match="conflicts"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(
                source,
                **cast(Any, {parameter_name: parameter_value}),
            ),
            environment=environment,
        )


@pytest.mark.parametrize(
    "environment",
    [
        {"LM_RESILIENCY_RESTART_CONTEXT": "/run/context.json"},
        {"LM_RESILIENCY_NODE_ID": "node-a"},
    ],
)
def test_runtime_config_requires_node_inputs(
    tmp_path: Path,
    environment: dict[str, str],
):
    source = _write_policy(tmp_path / "rendezvous.toml")

    with pytest.raises(TorchrunRuntimeConfigError, match="must be provided"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source),
            environment=environment,
        )


@pytest.mark.parametrize(
    ("config", "environment", "match"),
    [
        (
            {"local_world_size": None},
            {},
            "LM_RESILIENCY_LOCAL_WORLD_SIZE",
        ),
        (
            {"environment_digest": None},
            {},
            "LM_RESILIENCY_ENVIRONMENT_DIGEST",
        ),
        (
            {"resource_ids": None},
            {},
            "LM_RESILIENCY_RESOURCE_IDS",
        ),
        (
            {"local_world_size": "0"},
            {},
            "local_world_size must be positive",
        ),
    ],
)
def test_runtime_config_requires_agent_identity_inputs(
    tmp_path: Path,
    config: dict[str, object],
    environment: dict[str, str],
    match: str,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    runtime_config: Any = {
        "node_id": "node-a",
        "restart_context_path": "/run/context.json",
        **config,
    }

    with pytest.raises(TorchrunRuntimeConfigError, match=match):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source, **runtime_config),
            environment=environment,
        )


@pytest.mark.parametrize(
    ("resource_ids", "match"),
    [
        ("gpu-node-a-0;;gpu-node-a-1", r"resource_ids\[1\]"),
        ("gpu-node-a-0;gpu-node-a-0", "must be unique"),
        ('["gpu-node-a-0","gpu-node-a-1"]', "semicolon-delimited"),
        ({"gpu-node-a-0": True}, "semicolon-delimited"),
    ],
)
def test_runtime_config_rejects_invalid_resource_ids(
    tmp_path: Path,
    resource_ids: object,
    match: str,
):
    source = _write_policy(tmp_path / "rendezvous.toml")

    with pytest.raises(TorchrunRuntimeConfigError, match=match):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(
                source,
                node_id="node-a",
                resource_ids=resource_ids,
                restart_context_path="/run/context.json",
            ),
            environment={},
        )


def test_runtime_config_accepts_explicit_empty_resource_inventory(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")

    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(
            source,
            node_id="node-a",
            resource_ids="[]",
            restart_context_path="/run/context.json",
        ),
        environment={},
    )

    assert config.resource_ids == ()


def test_runtime_config_respects_explicitly_empty_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    source = _write_policy(tmp_path / "rendezvous.toml")
    monkeypatch.setenv("LM_RESILIENCY_NODE_ID", "process-node")
    monkeypatch.setenv(
        "LM_RESILIENCY_RESTART_CONTEXT",
        str(tmp_path / "process-context.json"),
    )

    with pytest.raises(TorchrunRuntimeConfigError, match="must be provided"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source),
            environment={},
        )


def test_runtime_config_requires_absolute_paths_and_standby_capacity(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")

    with pytest.raises(TorchrunRuntimeConfigError, match="config path must be absolute"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(Path("relative.toml")),
            environment=_environment(tmp_path / "context.json"),
        )
    with pytest.raises(TorchrunRuntimeConfigError, match="restart_context_path"):
        TorchrunRuntimeConfig.from_parameters(
            _parameters(source, restart_context_path="relative.json"),
            environment={"LM_RESILIENCY_NODE_ID": "node-a"},
        )
    with pytest.raises(TorchrunRuntimeConfigError, match="standby capacity"):
        TorchrunRuntimeConfig.from_parameters(
            RendezvousParameters(
                backend="lm_resiliency",
                endpoint="rdzv.example:29400",
                run_id=RUN_ID,
                min_nodes=2,
                max_nodes=2,
                config=str(source),
                local_world_size="2",
                environment_digest="environment-v1",
                resource_ids="[]",
            ),
            environment=_environment(tmp_path / "context.json"),
        )


def test_runtime_config_rejects_wrong_backend_and_type(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")

    with pytest.raises(TorchrunRuntimeConfigError, match="backend"):
        TorchrunRuntimeConfig.from_parameters(
            RendezvousParameters(
                backend="c10d",
                endpoint="rdzv.example:29400",
                run_id=RUN_ID,
                min_nodes=2,
                max_nodes=3,
                config=str(source),
            ),
            environment=_environment(tmp_path / "context.json"),
        )
    with pytest.raises(TypeError, match="RendezvousParameters"):
        TorchrunRuntimeConfig.from_parameters(cast(Any, object()))


def test_restart_context_file_writes_reads_replaces_and_clears(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)

    cleanup_token = context_file.write(_context())

    assert context_file.read() == _context()
    assert context_file.read_with_token() == (cleanup_token, _context())
    assert json.loads(path.read_text(encoding="utf-8")) == _context().to_dict()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    context_file.write(_context(plan_id="plan-2"))
    assert context_file.read() == _context(plan_id="plan-2")
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []

    context_file.clear()
    assert not path.exists()
    context_file.clear()


def test_restart_context_file_clears_only_the_matching_cleanup_token(
    tmp_path: Path,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)

    first_token = context_file.write(_context())
    second_token = context_file.write(_context())

    assert first_token != second_token
    assert context_file.clear_if_token(first_token) is False
    assert context_file.read() == _context()
    assert context_file.clear_if_token(second_token) is True
    assert not path.exists()


def test_restart_context_file_invalidates_context_while_directory_lock_is_held(
    tmp_path: Path,
):
    path = tmp_path / "private" / "restart-context.json"
    monotonic_clock = ManualMonotonicClock()
    context_file = RestartContextFile(path, monotonic_clock=monotonic_clock)
    cleanup_token = context_file.write(_context())
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    fcntl.flock(directory, fcntl.LOCK_EX)
    try:
        context_file.invalidate(cleanup_token)
        with pytest.raises(RestartContextFileError, match="timed out locking"):
            context_file.clear_if_token(
                cleanup_token,
                deadline=monotonic_clock(),
            )
    finally:
        fcntl.flock(directory, fcntl.LOCK_UN)
        os.close(directory)

    with pytest.raises(RestartContextFileError, match="invalidated"):
        context_file.read()


def test_restart_context_file_preserves_every_invalidated_token(
    tmp_path: Path,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    old_token = context_file.write(_context(plan_id="plan-old"))
    current_token = context_file.write(_context(plan_id="plan-current"))

    context_file.invalidate(current_token)
    context_file.invalidate(old_token)

    with pytest.raises(RestartContextFileError, match="invalidated"):
        context_file.read()

    context_file.write(_context(plan_id="plan-new"))
    assert context_file.read() == _context(plan_id="plan-new")


def test_restart_context_file_invalidates_long_valid_basename(
    tmp_path: Path,
):
    path = tmp_path / "private" / ("c" * 180)
    context_file = RestartContextFile(path)
    cleanup_token = context_file.write(_context())

    context_file.invalidate(cleanup_token)

    with pytest.raises(RestartContextFileError, match="invalidated"):
        context_file.read()


def test_restart_context_file_preserves_previous_value_when_replace_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())

    def fail_replace(
        source: object,
        destination: object,
        **kwargs: object,
    ) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RestartContextFileError, match="failed to publish"):
        context_file.write(_context(plan_id="plan-2"))

    assert context_file.read() == _context()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_restart_context_file_rejects_oversized_value_before_replace(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    oversized = replace(_context(plan_id="oversized"), reason_code="x" * (70 * 1024))

    with pytest.raises(RestartContextFileError, match="too large"):
        context_file.write(oversized)

    assert context_file.read() == _context()
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_restart_context_file_closes_descriptor_when_permission_setup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    real_close = os.close
    closed: list[int] = []
    failed_descriptor: int | None = None

    def fail_fchmod(descriptor: int, mode: int) -> None:
        nonlocal failed_descriptor
        failed_descriptor = descriptor
        raise OSError("injected permission failure")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(RestartContextFileError, match="failed to publish"):
        context_file.write(_context())

    assert failed_descriptor in closed
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []


def test_restart_context_file_rejects_duplicate_json_fields(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    encoded = path.read_text(encoding="utf-8").replace(
        '"plan_id":"plan-1"',
        '"plan_id":"plan-1","plan_id":"substituted"',
        1,
    )
    path.write_text(encoded, encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(RestartContextFileError, match="duplicate field"):
        context_file.read()


def test_restart_context_file_rejects_context_changed_without_metadata(
    tmp_path: Path,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    path.write_text(_context(plan_id="substituted").to_json() + "\n", encoding="utf-8")
    path.chmod(0o600)

    with pytest.raises(RestartContextFileError, match="metadata does not match"):
        context_file.read()


def test_restart_context_file_rejects_insecure_directory(tmp_path: Path):
    parent = tmp_path / "shared"
    parent.mkdir(mode=0o755)
    parent.chmod(0o755)
    context_file = RestartContextFile(parent / "restart-context.json")

    with pytest.raises(RestartContextFileError, match="group or other"):
        context_file.write(_context())


def test_restart_context_file_rejects_symlink_path(tmp_path: Path):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    target = parent / "target.json"
    target.write_text("{}", encoding="utf-8")
    target.chmod(0o600)
    path = parent / "restart-context.json"
    path.symlink_to(target)
    context_file = RestartContextFile(path)

    with pytest.raises(RestartContextFileError, match="symlink"):
        context_file.write(_context())
    with pytest.raises(RestartContextFileError, match="failed to open"):
        context_file.read()
    with pytest.raises(RestartContextFileError, match="symlink"):
        context_file.clear()


def test_restart_context_file_rejects_fifo_without_blocking(tmp_path: Path):
    parent = tmp_path / "private"
    parent.mkdir(mode=0o700)
    path = parent / "restart-context.json"
    os.mkfifo(path, mode=0o600)
    context_file = RestartContextFile(path)

    with pytest.raises(RestartContextFileError, match="not a regular file"):
        context_file.read()


def test_restart_context_file_rejects_symlink_parent(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    context_file = RestartContextFile(linked / "restart-context.json")

    with pytest.raises(RestartContextFileError, match="real directories"):
        context_file.write(_context())


def test_restart_context_file_rejects_symlinked_ancestor(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    context_file = RestartContextFile(linked / "private" / "restart-context.json")

    with pytest.raises(RestartContextFileError, match="real directories"):
        context_file.write(_context())


def test_restart_context_file_rejects_insecure_file_permissions(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    path.chmod(0o644)

    with pytest.raises(RestartContextFileError, match="group or other"):
        context_file.read()


def test_restart_context_file_retries_directory_sync_after_failed_clear(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    original_sync = RestartContextFile._fsync_directory
    calls = 0

    def flaky_sync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected directory sync failure")
        original_sync(descriptor)

    monkeypatch.setattr(
        RestartContextFile,
        "_fsync_directory",
        staticmethod(flaky_sync),
    )

    with pytest.raises(OSError, match="injected directory sync failure"):
        context_file.clear()

    assert not path.exists()
    context_file.clear()
    assert calls == 2


def test_restart_context_file_clear_accepts_missing_parent(tmp_path: Path):
    path = tmp_path / "missing" / "restart-context.json"

    RestartContextFile(path).clear()

    assert not path.parent.exists()


def test_restart_context_file_lock_acquisition_is_bounded(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    path.parent.mkdir(mode=0o700)
    directory_descriptor = os.open(path.parent, os.O_RDONLY)
    try:
        fcntl.flock(directory_descriptor, fcntl.LOCK_EX)

        with pytest.raises(RestartContextFileError, match="timed out locking"):
            RestartContextFile(path).write(
                _context(),
                deadline=time.monotonic() + 0.03,
            )
    finally:
        fcntl.flock(directory_descriptor, fcntl.LOCK_UN)
        os.close(directory_descriptor)

    assert not path.exists()


@pytest.mark.parametrize("path", [Path("relative.json"), cast(Any, "not-a-path")])
def test_restart_context_file_validates_path(path: object):
    expected = TypeError if isinstance(path, str) else RestartContextFileError

    with pytest.raises(expected):
        RestartContextFile(cast(Any, path))


def _handler_config(
    tmp_path: Path,
    *,
    node_id: str,
    min_nodes: int,
    max_nodes: int,
    registration_lease_duration_ms: int = 30_000,
    poll_interval_ms: int = 10,
    join_timeout_ms: int = 500,
) -> TorchrunRuntimeConfig:
    return TorchrunRuntimeConfig(
        policy=TorchrunRendezvousPolicy(
            control_endpoint="control.example:443",
            registration_lease_duration_ms=registration_lease_duration_ms,
            poll_interval_ms=poll_interval_ms,
            join_timeout_ms=join_timeout_ms,
        ),
        run_id=RUN_ID,
        endpoint="rdzv.example:29400",
        min_nodes=min_nodes,
        max_nodes=max_nodes,
        local_world_size=2,
        node_id=node_id,
        environment_digest="environment-v1",
        resource_ids=(f"gpu-{node_id}-0", f"gpu-{node_id}-1"),
        restart_context_path=tmp_path / node_id / "restart-context.json",
        local_addr=f"{node_id}.example",
        source_path=tmp_path / "rendezvous.toml",
    )


def _initialize_generation(
    store: InMemoryControlStore,
    clock: ManualClock,
    node_ids: tuple[str, ...],
):
    lease = CoordinatorLeaseManager(
        store,
        run_id=RUN_ID,
        coordinator_id="coordinator-a",
        lease_duration_ms=10_000,
        clock=clock,
    ).acquire()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=0,
        assignments=tuple(
            SlotAssignment(
                logical_node_slot=slot,
                node_id=node_id,
                first_global_rank=slot * 2,
                local_world_size=2,
            )
            for slot, node_id in enumerate(node_ids)
        ),
        topology_digest="topology-v1",
    )
    manager = GenerationStateManager(store, run_id=RUN_ID)
    return manager, lease, manager.initialize(lease, assignment)


def _handler(
    config: TorchrunRuntimeConfig,
    *,
    store: InMemoryControlStore,
    clock: ManualClock,
    agent_id: str,
    rendezvous_store: HashStore | None = None,
    bootstrap_store_info: RendezvousStoreInfo | None = RendezvousStoreInfo(
        "master.example",
        29500,
    ),
    monotonic_clock: Callable[[], float] = time.monotonic,
) -> SlotAwareRendezvousHandler:
    return SlotAwareRendezvousHandler(
        config,
        control_store=store,
        rendezvous_store=rendezvous_store or HashStore(),
        clock=clock,
        agent_id=agent_id,
        hostname=f"{config.node_id}.example",
        bootstrap_store_info=bootstrap_store_info,
        monotonic_clock=monotonic_clock,
    )


def _wait_until(predicate: Callable[[], bool], *, timeout: float = 2.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true before the timeout")


def _start_rendezvous(
    handler: SlotAwareRendezvousHandler,
) -> tuple[threading.Thread, queue.Queue[BaseException | object]]:
    outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous() -> None:
        try:
            outcome.put(handler.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    thread = threading.Thread(target=rendezvous)
    thread.start()
    return thread, outcome


def _seed_assigned_arrival(
    handler: SlotAwareRendezvousHandler,
    store: InMemoryControlStore,
    *,
    generation: int,
    slot: int,
) -> None:
    deadline = time.monotonic() + 1
    handler._ensure_registered(deadline)
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    current = handler._read_generation()
    assert current is not None
    attempt = handler._claim_shared_arrival_attempt(
        assignment=current.snapshot.record.assignment,
        slot=slot,
        registration=registration,
        deadline=deadline,
    )
    store.compare_set(
        handler._arrival_key(generation, attempt, slot),
        expected_revision=None,
        value=handler._arrival_value(
            generation=generation,
            attempt=attempt,
            slot=slot,
            registration=registration,
        ),
    )

    def complete_attempt() -> None:
        while time.monotonic() < deadline:
            handler._try_complete_arrival_attempt(
                current.snapshot.record.assignment,
                attempt,
                registration,
            )
            if store.get(handler._arrival_completion_key(generation, attempt)) is not None:
                return
            time.sleep(0.005)

    threading.Thread(target=complete_attempt, daemon=True).start()


def _stub_replacement_return_evidence(
    handler: SlotAwareRendezvousHandler,
    monkeypatch: pytest.MonkeyPatch,
    *,
    slot: int = 0,
) -> tuple[object, object]:
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    completion = object()
    admission = object()
    consumption = SimpleNamespace(revision=101, transaction_sequence=202)
    monkeypatch.setattr(
        handler,
        "_validate_replacement_admission_entry",
        lambda *_args, **_kwargs: {
            str(slot): {
                "agent_id": registration.record.agent_identity.agent_id,
                "consumption_revision": consumption.revision,
                "consumption_transaction_sequence": consumption.transaction_sequence,
                "node_id": registration.record.agent_identity.node_id,
                "registration_id": registration.record.registration_id,
                "registration_revision": registration.fencing_token,
            }
        },
    )
    monkeypatch.setattr(
        handler,
        "_read_arrival_consumption",
        lambda *_args, **_kwargs: consumption,
    )
    return completion, admission


def test_slot_aware_handler_refreshes_leader_registration_before_completion(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _, _, current = _initialize_generation(store, clock, ("node-a", "node-b"))
    leader = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    deadline = time.monotonic() + 1
    leader._ensure_registered(deadline)
    peer._ensure_registered(deadline)
    with leader._registration_lock:
        original_leader_registration = leader._registration
    with peer._registration_lock:
        peer_registration = peer._registration
    assert original_leader_registration is not None
    assert peer_registration is not None
    attempt = leader._claim_shared_arrival_attempt(
        assignment=current.snapshot.record.assignment,
        slot=0,
        registration=original_leader_registration,
        deadline=deadline,
    )
    for slot, handler, registration in (
        (0, leader, original_leader_registration),
        (1, peer, peer_registration),
    ):
        store.compare_set(
            leader._arrival_key(0, attempt, slot),
            expected_revision=None,
            value=handler._arrival_value(
                generation=0,
                attempt=attempt,
                slot=slot,
                registration=registration,
            ),
        )
    renewed = leader._registration_manager.renew(original_leader_registration)
    with leader._registration_lock:
        leader._registration = renewed

    leader._try_complete_arrival_attempt(
        current.snapshot.record.assignment,
        attempt,
        original_leader_registration,
    )

    completion = store.get(leader._arrival_completion_key(0, attempt))
    assert completion is not None
    assert completion.guard_revision == renewed.fencing_token
    assert leader.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_classifies_duplicate_completion_fields_as_corruption(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _, _, current = _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    deadline = time.monotonic() + 1
    handler._ensure_registered(deadline)
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    attempt = handler._claim_shared_arrival_attempt(
        assignment=current.snapshot.record.assignment,
        slot=0,
        registration=registration,
        deadline=deadline,
    )
    store.compare_set(
        handler._arrival_key(0, attempt, 0),
        expected_revision=None,
        value=handler._arrival_value(
            generation=0,
            attempt=attempt,
            slot=0,
            registration=registration,
        ),
    )
    handler._try_complete_arrival_attempt(
        current.snapshot.record.assignment,
        attempt,
        registration,
    )
    completion = store.get(handler._arrival_completion_key(0, attempt))
    assert completion is not None
    corrupted = replace(
        completion,
        value=completion.value.replace(
            b'"attempt":1,',
            b'"attempt":1,"attempt":1,',
            1,
        ),
    )

    with pytest.raises(RendezvousStateError, match="malformed"):
        handler._validate_arrival_completion_entry(
            corrupted,
            assignment=current.snapshot.record.assignment,
            attempt=attempt,
            registration=registration,
        )

    assert handler.shutdown() is True


def test_slot_aware_handler_admits_initial_assignment_with_stable_rank(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=2,
        max_nodes=3,
    )
    context_file = RestartContextFile(config.restart_context_path)
    context_file.write(replace(_context(), run_id="previous-run"))
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    peer_thread, peer_outcome = _start_rendezvous(peer)

    info = handler.next_rendezvous()
    peer_thread.join(timeout=2)
    peer_info = peer_outcome.get_nowait()

    assert handler.get_backend() == "lm_resiliency"
    assert handler.get_run_id() == RUN_ID
    assert handler.use_agent_store is True
    assert info.rank == 1
    assert info.world_size == 2
    assert info.bootstrap_store_info == RendezvousStoreInfo("master.example", 29500)
    assert isinstance(peer_info, RendezvousInfo)
    assert peer_info.rank == 0
    assert not config.restart_context_path.exists()
    registration = AgentRegistrationReader(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).get("node-b")
    assert registration is not None
    assert registration.record.agent_identity == handler.agent_identity

    assert handler.shutdown() is True
    assert peer.shutdown() is True
    assert handler.shutdown() is True
    assert (
        AgentRegistrationReader(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).get("node-b")
        is None
    )


def test_slot_aware_handler_times_out_when_an_assigned_slot_never_arrives(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )

    before = time.monotonic()
    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()

    assert time.monotonic() - before < 0.5
    assert handler.shutdown() is True


def test_slot_aware_handler_reuses_generation_scoped_bootstrap_store(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    rendezvous_store = HashStore()
    handler_a = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    handler_b = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )

    def rendezvous_b(outcome: queue.Queue[BaseException | object]) -> None:
        try:
            outcome.put(handler_b.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    first_outcome: queue.Queue[BaseException | object] = queue.Queue()
    first_thread = threading.Thread(target=rendezvous_b, args=(first_outcome,))
    first_thread.start()
    time.sleep(0.02)
    assert first_thread.is_alive()
    first_a = handler_a.next_rendezvous()
    first_thread.join(timeout=2)
    first_b = first_outcome.get_nowait()
    assert isinstance(first_b, type(first_a))
    assert first_b.bootstrap_store_info == first_a.bootstrap_store_info
    assert first_a.store.timeout == timedelta(milliseconds=handler_a._config.policy.join_timeout_ms)
    assert first_b.store.timeout == first_a.store.timeout

    assert handler_b.shutdown() is True
    replacement_b = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b-replacement",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )

    def replacement_rendezvous_b(outcome: queue.Queue[BaseException | object]) -> None:
        try:
            outcome.put(replacement_b.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    second_outcome: queue.Queue[BaseException | object] = queue.Queue()
    second_thread = threading.Thread(
        target=replacement_rendezvous_b,
        args=(second_outcome,),
    )
    second_thread.start()
    second_thread.join(timeout=2)
    replacement_first = second_outcome.get_nowait()
    assert isinstance(replacement_first, RendezvousInfo)
    assert replacement_first.rank == 1

    second_outcome = queue.Queue()
    second_thread = threading.Thread(
        target=replacement_rendezvous_b,
        args=(second_outcome,),
    )
    second_thread.start()
    time.sleep(0.02)
    assert second_thread.is_alive()
    second_a = handler_a.next_rendezvous()
    second_thread.join(timeout=2)
    assert not second_thread.is_alive()
    second_b = second_outcome.get_nowait()
    assert isinstance(second_b, type(second_a))
    assert second_b.bootstrap_store_info == second_a.bootstrap_store_info
    assert handler_a.use_agent_store is False
    assert replacement_b.use_agent_store is False

    assert handler_a.shutdown() is True
    assert replacement_b.shutdown() is True


def test_slot_aware_handler_reuses_incomplete_attempt_after_leader_replacement(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    original_leader = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a-original",
    )
    deadline = time.monotonic() + 1
    original_leader._ensure_registered(deadline)
    with original_leader._registration_lock:
        original_registration = original_leader._registration
    assert original_registration is not None
    current = original_leader._read_generation()
    assert current is not None
    assert (
        original_leader._claim_shared_arrival_attempt(
            assignment=current.snapshot.record.assignment,
            slot=0,
            registration=original_registration,
            deadline=deadline,
        )
        == 1
    )
    assert original_leader.shutdown() is True

    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    replacement = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a-replacement",
    )
    peer_thread, peer_outcome = _start_rendezvous(peer)

    replacement_info = replacement.next_rendezvous()
    peer_thread.join(timeout=2)
    peer_info = peer_outcome.get_nowait()

    assert replacement_info.rank == 0
    assert isinstance(peer_info, RendezvousInfo)
    assert peer_info.rank == 1
    attempt_entry = store.get(replacement._arrival_attempt_key(0))
    assert attempt_entry is not None
    attempt, _, _ = replacement._validate_arrival_attempt_entry(
        attempt_entry,
        generation=0,
    )
    assert attempt == 1
    assert replacement.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_replacement_consumes_completed_attempt_before_advance(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _, _, current = _initialize_generation(store, clock, ("node-a", "node-b"))
    original_leader = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a-original",
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    deadline = time.monotonic() + 1
    original_leader._ensure_registered(deadline)
    peer._ensure_registered(deadline)
    with original_leader._registration_lock:
        original_registration = original_leader._registration
    with peer._registration_lock:
        peer_registration = peer._registration
    assert original_registration is not None
    assert peer_registration is not None
    assignment = current.snapshot.record.assignment
    attempt = original_leader._claim_shared_arrival_attempt(
        assignment=assignment,
        slot=0,
        registration=original_registration,
        deadline=deadline,
    )
    for slot, handler, registration in (
        (0, original_leader, original_registration),
        (1, peer, peer_registration),
    ):
        store.compare_set(
            original_leader._arrival_key(0, attempt, slot),
            expected_revision=None,
            value=handler._arrival_value(
                generation=0,
                attempt=attempt,
                slot=slot,
                registration=registration,
            ),
        )
    original_leader._try_complete_arrival_attempt(
        assignment,
        attempt,
        original_registration,
    )
    completion = store.get(original_leader._arrival_completion_key(0, attempt))
    assert completion is not None
    original_leader._publish_arrival_consumption(
        current,
        assignment,
        attempt,
        0,
        completion,
        deadline,
    )
    original_consumption = store.get(
        original_leader._arrival_consumption_key(
            0,
            attempt,
            0,
            original_registration.record.registration_id,
        )
    )
    assert original_consumption is not None
    assert original_leader.shutdown() is True

    peer_info = peer.next_rendezvous()
    peer_consumption = store.get(
        peer._arrival_consumption_key(
            0,
            attempt,
            1,
            peer_registration.record.registration_id,
        )
    )
    assert peer_info.rank == 1
    assert peer_consumption is not None

    replacement = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a-replacement",
    )
    replacement_info = replacement.next_rendezvous()
    with replacement._registration_lock:
        replacement_registration = replacement._registration
    assert replacement_registration is not None
    replacement_consumption = store.get(
        replacement._arrival_consumption_key(
            0,
            attempt,
            0,
            replacement_registration.record.registration_id,
        )
    )
    attempt_entry = store.get(replacement._arrival_attempt_key(0))

    assert replacement_info.rank == 0
    assert replacement_consumption is not None
    assert replacement_consumption.value != original_consumption.value
    assert attempt_entry is not None
    current_attempt, _, _ = replacement._validate_arrival_attempt_entry(
        attempt_entry,
        generation=0,
    )
    assert current_attempt == attempt
    assert peer.shutdown() is True
    assert replacement.shutdown() is True


def test_slot_aware_handler_retries_consumption_after_heartbeat_refresh(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_compare = store.compare_set_many_guarded
    refreshed = False

    def refresh_before_consumption(*args: Any, **kwargs: Any) -> Any:
        nonlocal refreshed
        writes = args[0]
        if not refreshed and any("/consumed/" in key for key in writes):
            refreshed = True
            with handler._registration_lock:
                registration = handler._registration
            assert registration is not None
            renewed = handler._registration_manager.renew(registration)
            with handler._registration_lock:
                handler._registration = renewed
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(store, "compare_set_many_guarded", refresh_before_consumption)

    info = handler.next_rendezvous()

    assert info.rank == 0
    assert refreshed
    assert handler.shutdown() is True


def test_slot_aware_handler_fences_closure_during_consumption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_compare = store.compare_set_many_guarded
    closed = False

    def close_before_consumption(*args: Any, **kwargs: Any) -> Any:
        nonlocal closed
        writes = args[0]
        if not closed and any("/consumed/" in key for key in writes):
            closed = True
            store.compare_set(
                handler.closure_key,
                expected_revision=None,
                value=handler._closure_value,
            )
        return original_compare(*args, **kwargs)

    monkeypatch.setattr(store, "compare_set_many_guarded", close_before_consumption)

    with pytest.raises(RendezvousClosedError):
        handler.next_rendezvous()

    assert closed
    assert handler.shutdown() is True


def test_slot_aware_handler_shutdown_interrupts_initial_closure_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
            join_timeout_ms=1_000,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    started = threading.Event()
    unblock = threading.Event()
    original_read = handler._read_closure_entry

    def blocked_read() -> Any:
        started.set()
        unblock.wait(timeout=2)
        return original_read()

    monkeypatch.setattr(handler, "_read_closure_entry", blocked_read)
    thread, outcome = _start_rendezvous(handler)
    assert started.wait(timeout=1)

    before = time.monotonic()
    assert handler.shutdown() is True
    thread.join(timeout=1)
    elapsed = time.monotonic() - before

    assert not thread.is_alive()
    assert elapsed < 0.5
    assert isinstance(outcome.get_nowait(), RendezvousClosedError)
    unblock.set()


def test_slot_aware_handler_bounds_initial_closure_read_by_formation_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    started = threading.Event()
    unblock = threading.Event()

    def blocked_read() -> Any:
        started.set()
        unblock.wait(timeout=2)
        return None

    monkeypatch.setattr(handler, "_read_closure_entry", blocked_read)

    before = time.monotonic()
    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()
    elapsed = time.monotonic() - before

    assert started.is_set()
    assert elapsed < 0.5
    unblock.set()
    assert handler.shutdown() is True


def test_slot_aware_handler_bounds_bootstrap_read_and_ignores_stale_unprefixed_keys(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    rendezvous_store = HashStore()
    rendezvous_store.set("MASTER_ADDR", b"stale.example")
    rendezvous_store.set("MASTER_PORT", b"12345")
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    _seed_assigned_arrival(peer, store, generation=0, slot=0)

    before = time.monotonic()
    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()
    elapsed = time.monotonic() - before

    assert elapsed < 0.5
    assert handler.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_rechecks_closure_after_bootstrap_wait(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    rendezvous_store = HashStore()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    _seed_assigned_arrival(peer, store, generation=0, slot=0)
    outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous() -> None:
        try:
            outcome.put(handler.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    thread = threading.Thread(target=rendezvous)
    thread.start()
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-b")
            is not None
        )
    )

    handler.set_closed()
    bootstrap_store = handler._bootstrap_store(0)
    bootstrap_store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, "master.example")
    bootstrap_store.set(RendezvousStoreInfo.MASTER_PORT_KEY, "29500")
    thread.join(timeout=2)

    assert isinstance(outcome.get_nowait(), RendezvousClosedError)
    assert handler.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_rejects_generation_change_after_bootstrap_wait(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(
        store,
        clock,
        ("node-a", "node-b"),
    )
    rendezvous_store = HashStore()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    _seed_assigned_arrival(peer, store, generation=0, slot=0)
    thread, outcome = _start_rendezvous(handler)
    _wait_until(lambda: store.get(handler._arrival_completion_key(0, 1)) is not None)
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=(
            SlotAssignment(0, "node-a", 0, 2),
            SlotAssignment(1, "node-b", 2, 2),
        ),
        topology_digest="topology-v1",
    )
    manager.commit_successor(lease, current, successor)
    bootstrap_store = handler._bootstrap_store(0)
    bootstrap_store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, "master.example")
    bootstrap_store.set(RendezvousStoreInfo.MASTER_PORT_KEY, "29500")
    thread.join(timeout=2)

    assert isinstance(outcome.get_nowait(), RendezvousConnectionError)
    assert handler.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_rechecks_heartbeat_after_bootstrap_wait(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    rendezvous_store = HashStore()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
        rendezvous_store=rendezvous_store,
        bootstrap_store_info=None,
    )
    _seed_assigned_arrival(peer, store, generation=0, slot=0)
    outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous() -> None:
        try:
            outcome.put(handler.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    thread = threading.Thread(target=rendezvous)
    thread.start()
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-b")
            is not None
        )
    )

    handler._heartbeat_error = RuntimeError("registration lost")
    bootstrap_store = handler._bootstrap_store(0)
    bootstrap_store.set(RendezvousStoreInfo.MASTER_ADDR_KEY, "master.example")
    bootstrap_store.set(RendezvousStoreInfo.MASTER_PORT_KEY, "29500")
    thread.join(timeout=2)

    assert isinstance(outcome.get_nowait(), RendezvousConnectionError)
    assert handler.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_does_not_publish_arrival_before_context_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    peer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=2,
            max_nodes=3,
            join_timeout_ms=80,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    leader = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
            join_timeout_ms=80,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_clear = RestartContextFile.clear_stale_for_run

    def fail_peer_clear(
        context_file: RestartContextFile,
        run_id: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        if context_file.path == peer._config.restart_context_path:
            raise RestartContextFileError("injected context cleanup failure")
        return original_clear(context_file, run_id, deadline=deadline)

    monkeypatch.setattr(RestartContextFile, "clear_stale_for_run", fail_peer_clear)
    peer_thread, peer_outcome = _start_rendezvous(peer)

    with pytest.raises(RendezvousTimeoutError):
        leader.next_rendezvous()
    peer_thread.join(timeout=2)

    assert isinstance(peer_outcome.get_nowait(), RendezvousStateError)
    attempt_entry = store.get(leader._arrival_attempt_key(0))
    assert attempt_entry is not None
    attempt, _, _ = leader._validate_arrival_attempt_entry(
        attempt_entry,
        generation=0,
    )
    assert store.get(leader._arrival_key(0, attempt, 1)) is None
    assert leader.shutdown() is True
    assert peer.shutdown() is True


def test_slot_aware_handler_refuses_to_clear_current_run_context_for_initial_generation(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _, _, current = _initialize_generation(store, clock, ("node-a",))
    config = _handler_config(
        tmp_path,
        node_id="node-a",
        min_nodes=1,
        max_nodes=2,
    )
    context_file = RestartContextFile(config.restart_context_path)
    context_file.write(_context())
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-a",
    )

    with pytest.raises(RendezvousStateError, match="stale initial restart context"):
        handler._prepare_restart_context(
            current,
            None,
            time.monotonic() + 1,
        )

    assert context_file.read() == _context()
    assert handler.shutdown() is True


def test_slot_aware_handler_reads_registration_histories_linearly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b", "node-c"))
    handlers = tuple(
        _handler(
            _handler_config(
                tmp_path,
                node_id=node_id,
                min_nodes=3,
                max_nodes=4,
            ),
            store=store,
            clock=clock,
            agent_id=f"agent-{node_id}",
        )
        for node_id in ("node-a", "node-b", "node-c")
    )
    history_reads = 0
    history_lock = threading.Lock()
    original_get_history = store.get_history

    def count_registration_histories(key: str):
        nonlocal history_reads
        if "/agent-registrations/" in key:
            with history_lock:
                history_reads += 1
        return original_get_history(key)

    monkeypatch.setattr(store, "get_history", count_registration_histories)
    deadline = time.monotonic() + 1
    handlers[0]._ensure_registered(deadline)
    with handlers[0]._registration_lock:
        leader_registration = handlers[0]._registration
    assert leader_registration is not None
    current = handlers[0]._read_generation()
    assert current is not None
    attempt = handlers[0]._claim_shared_arrival_attempt(
        assignment=current.snapshot.record.assignment,
        slot=0,
        registration=leader_registration,
        deadline=deadline,
    )
    assert attempt == 1
    peer_threads = tuple(_start_rendezvous(handler) for handler in handlers[1:])
    _wait_until(
        lambda: all(
            store.get(handlers[0]._arrival_key(0, attempt, slot)) is not None for slot in (1, 2)
        )
    )

    leader_info = handlers[0].next_rendezvous()
    peer_infos: list[RendezvousInfo] = []
    for thread, outcome in peer_threads:
        thread.join(timeout=2)
        info = outcome.get_nowait()
        assert isinstance(info, RendezvousInfo)
        peer_infos.append(info)

    assert leader_info.rank == 0
    assert {info.rank for info in peer_infos} == {1, 2}
    assert history_reads <= 4 * len(handlers)
    assert all(handler.shutdown() for handler in handlers)


def test_slot_aware_handler_parks_standby_without_reporting_waiting_node(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    standby = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    closer = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous() -> None:
        try:
            outcome.put(standby.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    thread = threading.Thread(target=rendezvous)
    thread.start()
    reader = AgentRegistrationReader(store, run_id=RUN_ID, clock=clock)
    _wait_until(lambda: reader.get("node-b") is not None)
    time.sleep(0.08)

    assert thread.is_alive()
    assert standby.num_nodes_waiting() == 0

    closer.set_closed()
    thread.join(timeout=2)

    assert not thread.is_alive()
    assert isinstance(outcome.get_nowait(), RendezvousClosedError)
    assert reader.get("node-b") is None
    assert standby.shutdown() is True
    assert closer.shutdown() is True


def test_slot_aware_handler_local_shutdown_does_not_close_shared_run(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    first = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    second = _handler(
        _handler_config(
            tmp_path,
            node_id="node-c",
            min_nodes=1,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-c",
    )
    first_outcome: queue.Queue[BaseException | object] = queue.Queue()
    second_outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous(
        handler: SlotAwareRendezvousHandler,
        outcome: queue.Queue[BaseException | object],
    ) -> None:
        try:
            outcome.put(handler.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    first_thread = threading.Thread(target=rendezvous, args=(first, first_outcome))
    second_thread = threading.Thread(target=rendezvous, args=(second, second_outcome))
    first_thread.start()
    second_thread.start()
    reader = AgentRegistrationReader(store, run_id=RUN_ID, clock=clock)
    _wait_until(lambda: reader.get("node-b") is not None)
    _wait_until(lambda: reader.get("node-c") is not None)

    assert first.shutdown() is True
    first_thread.join(timeout=2)
    assert isinstance(first_outcome.get_nowait(), RendezvousClosedError)
    assert not store.has_history(first.closure_key)
    assert not second.is_closed()
    assert second_thread.is_alive()

    second.set_closed()
    second_thread.join(timeout=2)

    assert isinstance(second_outcome.get_nowait(), RendezvousClosedError)
    assert second.shutdown() is True


def test_slot_aware_handler_renews_active_registration(tmp_path: Path):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    config = _handler_config(
        tmp_path,
        node_id="node-a",
        min_nodes=1,
        max_nodes=2,
        registration_lease_duration_ms=90,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    handler.next_rendezvous()
    reader = AgentRegistrationReader(store, run_id=RUN_ID, clock=clock)
    initial = reader.get("node-a")
    assert initial is not None

    clock.advance(20)
    _wait_until(
        lambda: (
            (renewed := reader.get("node-a")) is not None
            and renewed.fencing_token != initial.fencing_token
        )
    )

    renewed = reader.get("node-a")
    assert renewed is not None
    assert renewed.expires_at_unix_ms > initial.expires_at_unix_ms
    assert handler.shutdown() is True


def test_slot_aware_handler_schedules_renewal_from_remaining_lease_lifetime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
            registration_lease_duration_ms=90,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    handler.next_rendezvous()
    original_renew = handler._registration_manager.renew
    renew_count = 0
    second_renewal_succeeded = threading.Event()
    timer: threading.Timer | None = None

    def delayed_first_response(registration: Any) -> Any:
        nonlocal renew_count, timer
        renew_count += 1
        renewed = original_renew(registration)
        if renew_count == 1:
            clock.advance(85)
            timer = threading.Timer(0.01, lambda: clock.advance(10))
            timer.start()
        else:
            second_renewal_succeeded.set()
        return renewed

    monkeypatch.setattr(
        handler._registration_manager,
        "renew",
        delayed_first_response,
    )

    assert second_renewal_succeeded.wait(timeout=1)
    if timer is not None:
        timer.join(timeout=1)
    assert handler.shutdown() is True


def test_slot_aware_handler_shutdown_is_bounded_by_stuck_heartbeat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
            registration_lease_duration_ms=90,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    handler.next_rendezvous()
    started = threading.Event()
    release = threading.Event()

    def block_renewal(registration: Any) -> Any:
        started.set()
        release.wait(timeout=2)
        return registration

    monkeypatch.setattr(handler._registration_manager, "renew", block_renewal)
    assert started.wait(timeout=1)

    before = time.monotonic()
    assert handler.shutdown() is False
    elapsed = time.monotonic() - before

    assert elapsed < 0.5
    release.set()
    _wait_until(
        lambda: handler._heartbeat_thread is not None and not handler._heartbeat_thread.is_alive()
    )
    assert handler.shutdown() is True


def test_slot_aware_handler_shutdown_is_bounded_by_stuck_registration_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    handler.next_rendezvous()
    original_release = handler._registration_manager.release
    started = threading.Event()
    unblock = threading.Event()

    def delayed_release(registration: Any) -> int:
        started.set()
        unblock.wait(timeout=2)
        return original_release(registration)

    monkeypatch.setattr(
        handler._registration_manager,
        "release",
        delayed_release,
    )

    before = time.monotonic()
    assert handler.shutdown() is False
    elapsed = time.monotonic() - before

    assert elapsed < 0.5
    assert started.is_set()
    unblock.set()
    _wait_until(lambda: handler._registration is None)
    assert handler.shutdown() is True


def test_slot_aware_handler_times_out_and_releases_standby_registration(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()

    assert (
        AgentRegistrationReader(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).get("node-b")
        is None
    )
    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_incompatible_node_id_reuse(tmp_path: Path):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    original_config = _handler_config(
        tmp_path,
        node_id="node-a",
        min_nodes=1,
        max_nodes=2,
    )
    original_identity = original_config.build_agent_identity(
        agent_id="agent-a-original",
        hostname="node-a.example",
    )
    registration_manager = AgentRegistrationManager(
        store,
        agent_identity=original_identity,
        lease_duration_ms=original_config.policy.registration_lease_duration_ms,
        clock=clock,
    )
    original_registration = registration_manager.register()
    registration_manager.release(original_registration)
    incompatible_config = replace(
        original_config,
        environment_digest="environment-v2",
    )
    handler = _handler(
        incompatible_config,
        store=store,
        clock=clock,
        agent_id="agent-a-replacement",
    )

    with pytest.raises(RendezvousStateError, match="incompatible"):
        handler.next_rendezvous()

    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_job_wide_environment_drift(tmp_path: Path):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a", "node-b"))
    first = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    second = _handler(
        replace(
            _handler_config(
                tmp_path,
                node_id="node-b",
                min_nodes=2,
                max_nodes=3,
            ),
            environment_digest="environment-v2",
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    first._ensure_registered(time.monotonic() + 1)
    current = first._read_generation()
    assert current is not None
    slot, recovery_state, _ = first._generation_admission(
        current,
        time.monotonic() + 1,
    )
    assert slot == 0
    assert recovery_state is None
    with pytest.raises(RendezvousStateError, match="committed workload environment"):
        second.next_rendezvous()

    assert first.shutdown() is True
    assert second.shutdown() is True


def test_slot_aware_handler_standby_cannot_commit_job_compatibility(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    standby = _handler(
        replace(
            _handler_config(
                tmp_path,
                node_id="node-b",
                min_nodes=1,
                max_nodes=2,
            ),
            environment_digest="environment-v2",
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    active = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    standby_outcome: queue.Queue[BaseException | object] = queue.Queue()

    def rendezvous_standby() -> None:
        try:
            standby_outcome.put(standby.next_rendezvous())
        except BaseException as error:
            standby_outcome.put(error)

    standby_thread = threading.Thread(target=rendezvous_standby)
    standby_thread.start()
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-b")
            is not None
        )
    )

    assert store.get(active._compatibility_key) is None
    assert active.next_rendezvous().rank == 0
    assert store.get(active._compatibility_key) is not None

    assert standby.shutdown() is True
    standby_thread.join(timeout=2)
    assert isinstance(standby_outcome.get_nowait(), RendezvousClosedError)
    assert active.shutdown() is True


def test_slot_aware_handler_shutdown_is_bounded_during_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_register = handler._registration_manager.register
    started = threading.Event()
    release = threading.Event()
    outcome: queue.Queue[BaseException | object] = queue.Queue()

    def delayed_registration() -> Any:
        registration = original_register()
        started.set()
        release.wait(timeout=2)
        return registration

    def rendezvous() -> None:
        try:
            outcome.put(handler.next_rendezvous())
        except BaseException as error:
            outcome.put(error)

    monkeypatch.setattr(
        handler._registration_manager,
        "register",
        delayed_registration,
    )
    thread = threading.Thread(target=rendezvous)
    thread.start()
    assert started.wait(timeout=1)

    before = time.monotonic()
    assert handler.shutdown() is True
    assert time.monotonic() - before < 0.5

    release.set()
    thread.join(timeout=2)
    assert isinstance(outcome.get_nowait(), RendezvousClosedError)
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-a")
            is None
        )
    )


def test_slot_aware_handler_bounds_initial_registration_by_formation_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
            join_timeout_ms=60,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_register = handler._registration_manager.register
    started = threading.Event()
    release = threading.Event()

    def delayed_registration() -> Any:
        registration = original_register()
        started.set()
        release.wait(timeout=2)
        return registration

    monkeypatch.setattr(
        handler._registration_manager,
        "register",
        delayed_registration,
    )

    before = time.monotonic()
    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()

    assert started.is_set()
    assert time.monotonic() - before < 0.5
    release.set()
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-a")
            is None
        )
    )
    assert handler.shutdown() is True


def test_slot_aware_handler_cleans_up_registration_on_cancellation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    def cancel(_deadline: float | None) -> bool:
        raise KeyboardInterrupt

    monkeypatch.setattr(handler, "_wait_for_change", cancel)

    with pytest.raises(KeyboardInterrupt):
        handler.next_rendezvous()

    assert (
        AgentRegistrationReader(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).get("node-b")
        is None
    )
    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_successor_without_restart_plan_publication(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=(SlotAssignment(0, "node-b", 0, 2),),
        topology_digest="topology-v1",
    )
    manager.commit_successor(lease, current, successor)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    with pytest.raises(RendezvousStateError, match="restart-plan publication"):
        handler.next_rendezvous()

    assert handler.shutdown() is True


def test_slot_aware_handler_admits_replacement_and_writes_restart_context(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    state = _static_recovery_state(plan)
    handler._publication_reader = cast(Any, StaticRecoveryStateReader(state))

    rendezvous = handler.next_rendezvous()

    assert rendezvous.rank == 0
    assert rendezvous.world_size == 1
    assert RestartContextFile(config.restart_context_path).read() == RestartContext.from_plan(
        plan,
        "node-b",
    )
    admission = store.get(handler._arrival_admission_key(1, 1))
    assert admission is not None
    assert handler.num_nodes_waiting() == 0
    assert handler.shutdown() is True


def test_slot_aware_handler_waits_for_return_ack_before_advancing_attempt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )
    reached_return_ack = threading.Event()
    release_return_ack = threading.Event()
    original_publish = handler._publish_replacement_return_acknowledgement

    def delay_return_ack(*args: Any, **kwargs: Any) -> Any:
        reached_return_ack.set()
        assert release_return_ack.wait(timeout=2)
        return original_publish(*args, **kwargs)

    monkeypatch.setattr(
        handler,
        "_publish_replacement_return_acknowledgement",
        delay_return_ack,
    )
    thread, outcome = _start_rendezvous(handler)
    assert reached_return_ack.wait(timeout=2)
    current_generation = handler._read_generation()
    assert current_generation is not None
    attempt_entry = store.get(handler._arrival_attempt_key(1))
    completion = store.get(handler._arrival_completion_key(1, 1))
    assert attempt_entry is not None
    assert completion is not None
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None

    assert (
        handler._advance_arrival_attempt(
            successor,
            attempt_entry,
            1,
            2,
            handler._arrival_attempt_value(
                generation=1,
                attempt=2,
                registration=registration,
            ),
            registration,
            completion,
        )
        is None
    )
    release_return_ack.set()
    thread.join(timeout=2)
    assert not thread.is_alive()
    assert isinstance(outcome.get_nowait(), RendezvousInfo)

    advanced = handler._advance_arrival_attempt(
        successor,
        attempt_entry,
        1,
        2,
        handler._arrival_attempt_value(
            generation=1,
            attempt=2,
            registration=registration,
        ),
        registration,
        completion,
    )

    assert advanced is not None
    attempt, _, _ = handler._validate_arrival_attempt_entry(
        advanced,
        generation=1,
        expected_attempt=2,
        leader_node_id="node-b",
    )
    assert attempt == 2
    assert handler.shutdown() is True


def test_slot_aware_handler_never_partially_admits_replacement_after_deadline(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(
        store,
        clock,
        ("node-a", "node-d"),
    )
    plan = replace(
        _replacement_plan(restart_deadline_unix_ms=clock() + 50),
        slot_assignments=(
            SlotAssignment(0, "node-b", 0, 2),
            SlotAssignment(1, "node-c", 2, 2),
        ),
        expected_world_size=4,
    )
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    handlers = tuple(
        _handler(
            _handler_config(
                tmp_path,
                node_id=node_id,
                min_nodes=2,
                max_nodes=4,
            ),
            store=store,
            clock=clock,
            agent_id=f"agent-{node_id}",
        )
        for node_id in ("node-b", "node-c")
    )
    recovery_state = cast(
        Any,
        _static_recovery_state(plan),
    )
    deadline = time.monotonic() + 1
    for handler in handlers:
        handler._publication_reader = cast(
            Any,
            StaticRecoveryStateReader(recovery_state),
        )
        handler._ensure_registered(deadline)
        handler._prepare_restart_context(
            cast(Any, handler._read_generation()),
            recovery_state,
            deadline,
        )
    generation = handlers[0]._read_generation()
    assert generation is not None
    with handlers[0]._registration_lock:
        leader_registration = handlers[0]._registration
    assert leader_registration is not None
    attempt = handlers[0]._claim_shared_arrival_attempt(
        assignment=successor,
        slot=0,
        registration=leader_registration,
        deadline=deadline,
    )
    for slot, handler in enumerate(handlers):
        with handler._registration_lock:
            registration = handler._registration
        assert registration is not None
        store.compare_set(
            handler._arrival_key(1, attempt, slot),
            expected_revision=None,
            value=handler._arrival_value(
                generation=1,
                attempt=attempt,
                slot=slot,
                registration=registration,
            ),
        )
    handlers[0]._try_complete_arrival_attempt(
        successor,
        attempt,
        leader_registration,
    )
    completion = store.get(handlers[0]._arrival_completion_key(1, attempt))
    assert completion is not None
    context_tokens = []
    for handler in handlers:
        context_tokens.append(handler._restart_context.read_with_token()[0])
    for handler, context_token in zip(handlers, context_tokens, strict=True):
        handler._validate_final_admission_state(
            generation,
            recovery_state,
            deadline,
            expected_context_token=context_token,
        )
    handlers[0]._publish_arrival_consumption(
        generation,
        successor,
        attempt,
        0,
        completion,
        deadline,
        recovery_state,
    )

    clock.advance(50)
    with pytest.raises(RendezvousTimeoutError):
        handlers[1]._publish_arrival_consumption(
            generation,
            successor,
            attempt,
            1,
            completion,
            deadline,
            recovery_state,
        )
    handlers[0]._try_commit_replacement_admission(
        generation,
        successor,
        attempt,
        completion,
        recovery_state,
        deadline,
    )

    assert store.get(handlers[0]._arrival_admission_key(1, attempt)) is None
    for handler in handlers:
        assert handler.shutdown() is True


def test_slot_aware_handler_gives_selected_standby_a_fresh_formation_deadline(
    tmp_path: Path,
):
    clock = ManualClock()
    monotonic_clock = ManualMonotonicClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
        join_timeout_ms=50,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
        monotonic_clock=monotonic_clock,
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )
    thread, outcome = _start_rendezvous(handler)
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-b")
            is not None
        )
    )
    monotonic_clock.advance(0.08)
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)

    result = outcome.get(timeout=2)
    thread.join(timeout=2)

    assert isinstance(result, RendezvousInfo)
    assert result.rank == 0
    assert RestartContextFile(config.restart_context_path).read() == RestartContext.from_plan(
        plan,
        "node-b",
    )
    assert handler.shutdown() is True


def test_slot_aware_handler_keeps_unselected_standby_outside_plan_deadline(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan(restart_deadline_unix_ms=clock())
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-c",
            min_nodes=1,
            max_nodes=3,
            join_timeout_ms=50,
        ),
        store=store,
        clock=clock,
        agent_id="agent-c",
    )
    recovery_reader = StaticRecoveryStateReader(_static_recovery_state(plan))
    handler._publication_reader = cast(Any, recovery_reader)

    thread, outcome = _start_rendezvous(handler)
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-c")
            is not None
        )
    )
    time.sleep(0.08)

    assert thread.is_alive()
    assert recovery_reader.calls == 0
    assert handler.shutdown() is True
    thread.join(timeout=2)
    assert isinstance(outcome.get_nowait(), RendezvousClosedError)


def test_slot_aware_handler_rejects_expired_replacement_without_leaving_context(
    tmp_path: Path,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan(restart_deadline_unix_ms=clock())
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )

    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()

    assert not config.restart_context_path.exists()
    assert handler.shutdown() is True


def test_slot_aware_handler_clears_context_when_publication_fails_after_replace(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )
    original_write = RestartContextFile.write

    def publish_then_fail(
        context_file: RestartContextFile,
        context: RestartContext,
        *,
        deadline: float | None = None,
    ) -> None:
        token = original_write(context_file, context, deadline=deadline)
        raise RestartContextFileError(
            "injected post-publication failure",
            published_token=token,
        )

    monkeypatch.setattr(RestartContextFile, "write", publish_then_fail)

    with pytest.raises(RendezvousStateError, match="publish the replacement restart context"):
        handler.next_rendezvous()

    assert not config.restart_context_path.exists()
    assert handler.shutdown() is True


def test_slot_aware_handler_clears_replacement_context_after_admission_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )

    def fail_after_context(*_args: object) -> object:
        assert RestartContextFile(config.restart_context_path).read() == RestartContext.from_plan(
            plan,
            "node-b",
        )
        raise RendezvousConnectionError("injected arrival failure")

    monkeypatch.setattr(handler, "_wait_for_assigned_arrivals", fail_after_context)

    with pytest.raises(RendezvousConnectionError, match="injected arrival failure"):
        handler.next_rendezvous()

    assert not config.restart_context_path.exists()
    assert handler.shutdown() is True


def test_slot_aware_handler_always_cleans_registration_when_context_cleanup_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
    )
    cleanup_token = "a" * 64
    cleaned = False
    current = SimpleNamespace()
    monkeypatch.setattr(handler, "_is_closed_bounded", lambda _deadline: False)
    monkeypatch.setattr(handler, "_ensure_registered", lambda _deadline: None)
    monkeypatch.setattr(handler, "_read_generation", lambda: current)
    monkeypatch.setattr(
        handler,
        "_generation_admission",
        lambda *_args: (0, object(), time.monotonic() + 1),
    )
    monkeypatch.setattr(
        handler,
        "_prepare_restart_context",
        lambda *_args: cleanup_token,
    )
    monkeypatch.setattr(
        handler,
        "_wait_for_assigned_arrivals",
        lambda *_args: (_ for _ in ()).throw(
            RendezvousConnectionError("original admission failure")
        ),
    )
    monkeypatch.setattr(
        handler,
        "_invalidate_and_clear_restart_context",
        lambda _token: (_ for _ in ()).throw(RendezvousStateError("context cleanup failure")),
    )

    def cleanup() -> bool:
        nonlocal cleaned
        cleaned = True
        return True

    monkeypatch.setattr(handler, "_cleanup_local_resources", cleanup)

    with pytest.raises(RendezvousConnectionError, match="original admission failure"):
        handler.next_rendezvous()

    assert cleaned is True


def test_slot_aware_handler_invalidates_context_when_cleanup_cannot_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    manager, lease, current = _initialize_generation(store, clock, ("node-a",))
    plan = _replacement_plan()
    successor = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    manager.commit_successor(lease, current, successor)
    config = _handler_config(
        tmp_path,
        node_id="node-b",
        min_nodes=1,
        max_nodes=2,
    )
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._publication_reader = cast(
        Any,
        StaticRecoveryStateReader(_static_recovery_state(plan)),
    )

    def fail_after_context(*_args: object) -> object:
        assert RestartContextFile(config.restart_context_path).read() == RestartContext.from_plan(
            plan,
            "node-b",
        )
        raise RendezvousConnectionError("injected arrival failure")

    def fail_cleanup(
        _context_file: RestartContextFile,
        _cleanup_token: str,
        *,
        deadline: float | None = None,
    ) -> bool:
        del deadline
        raise RestartContextFileError("injected cleanup lock timeout")

    monkeypatch.setattr(handler, "_wait_for_assigned_arrivals", fail_after_context)
    monkeypatch.setattr(RestartContextFile, "clear_if_token", fail_cleanup)

    with pytest.raises(RendezvousConnectionError, match="injected arrival failure"):
        handler.next_rendezvous()

    assert config.restart_context_path.exists()
    with pytest.raises(RestartContextFileError, match="invalidated"):
        RestartContextFile(config.restart_context_path).read()
    handler.shutdown()


def test_slot_aware_handler_rejects_replacement_clock_behind_publication(
    tmp_path: Path,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
    )
    recovery_state = _static_recovery_state(
        _replacement_plan(),
        committed_at_unix_ms=clock() + 1,
    )

    with pytest.raises(RendezvousConnectionError, match="clock regressed"):
        handler._replacement_deadline(cast(Any, recovery_state), None)


def test_slot_aware_handler_uses_authoritative_store_time_for_replacement_deadline(
    tmp_path: Path,
):
    client_clock = ManualClock(1_000)
    store_clock = ManualClock(1_100)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=store_clock),
        clock=client_clock,
        agent_id="agent-b",
    )
    recovery_state = _static_recovery_state(
        _replacement_plan(restart_deadline_unix_ms=1_050),
        committed_at_unix_ms=1_000,
    )

    with pytest.raises(RendezvousTimeoutError):
        handler._replacement_deadline(cast(Any, recovery_state), None)


def test_slot_aware_handler_rejects_clock_regression_and_retains_first_deadline(
    tmp_path: Path,
):
    clock = ManualClock()
    monotonic_clock = ManualMonotonicClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
        monotonic_clock=monotonic_clock,
    )
    recovery_state = _static_recovery_state(_replacement_plan())
    deadline = handler._replacement_deadline(cast(Any, recovery_state), None)

    clock.set(clock() - 1)
    with pytest.raises(
        RendezvousConnectionError,
        match="authoritative replacement rendezvous time",
    ):
        handler._validate_replacement_deadline(cast(Any, recovery_state), deadline)

    clock.set(1_000)
    monotonic_clock.advance(deadline)
    with pytest.raises(RendezvousTimeoutError):
        handler._validate_replacement_deadline(cast(Any, recovery_state), deadline)


def test_slot_aware_handler_rechecks_deadline_before_returning_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    monotonic_clock = ManualMonotonicClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
        monotonic_clock=monotonic_clock,
    )
    plan = _replacement_plan(restart_deadline_unix_ms=clock() + 50)
    recovery_state = _static_recovery_state(plan)
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    deadline = monotonic_clock() + 0.05
    handler._ensure_registered(time.monotonic() + 1)

    def validate_after_blocking_work(*_args: object, **_kwargs: object) -> None:
        clock.advance(50)
        monotonic_clock.advance(0.05)

    monkeypatch.setattr(
        handler,
        "_validate_final_admission_state",
        validate_after_blocking_work,
    )
    monkeypatch.setattr(handler, "_validate_current_arrival_attempt", lambda *_args: None)
    completion, admission = _stub_replacement_return_evidence(
        handler,
        monkeypatch,
    )

    with pytest.raises(RendezvousTimeoutError):
        handler._validate_replacement_return(
            cast(Any, object()),
            assignment,
            1,
            0,
            cast(Any, completion),
            cast(Any, admission),
            cast(Any, recovery_state),
            deadline,
            "a" * 64,
        )
    assert handler.shutdown() is True


def test_slot_aware_handler_binds_admission_to_returning_registration(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
    )
    handler._ensure_registered(time.monotonic() + 1)
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    plan = _replacement_plan()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    monkeypatch.setattr(handler, "_validate_final_admission_state", lambda *_a, **_k: None)
    monkeypatch.setattr(
        handler,
        "_validate_replacement_admission_entry",
        lambda *_a, **_k: {
            "0": {
                "agent_id": "old-agent-b",
                "consumption_revision": 10,
                "consumption_transaction_sequence": 11,
                "node_id": "node-b",
                "registration_id": "old-registration",
                "registration_revision": registration.fencing_token,
            }
        },
    )

    with pytest.raises(
        RendezvousConnectionError,
        match="does not include this agent registration",
    ):
        handler._validate_replacement_return(
            cast(Any, object()),
            assignment,
            1,
            0,
            cast(Any, object()),
            cast(Any, object()),
            cast(Any, _static_recovery_state(plan)),
            time.monotonic() + 1,
            "a" * 64,
        )

    assert handler.shutdown() is True


def test_slot_aware_handler_accepts_heartbeat_renewal_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
    )
    handler._ensure_registered(time.monotonic() + 1)
    with handler._registration_lock:
        original = handler._registration
    assert original is not None
    completion, admission = _stub_replacement_return_evidence(
        handler,
        monkeypatch,
    )
    renewed = handler._registration_manager.renew(original)
    with handler._registration_lock:
        handler._registration = renewed
    plan = _replacement_plan()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    monkeypatch.setattr(
        handler,
        "_validate_final_admission_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(handler, "_validate_current_arrival_attempt", lambda *_args: None)
    monkeypatch.setattr(
        handler,
        "_publish_replacement_return_acknowledgement",
        lambda *_args, **_kwargs: None,
    )

    handler._validate_replacement_return(
        cast(Any, object()),
        assignment,
        1,
        0,
        cast(Any, completion),
        cast(Any, admission),
        cast(Any, _static_recovery_state(plan)),
        time.monotonic() + 1,
        "a" * 64,
    )

    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_replaced_restart_context_token(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-b",
    )
    plan = _replacement_plan()
    recovery_state = _static_recovery_state(plan)
    current = object()
    expected_context = handler._restart_context_for_plan(plan)
    original_token = handler._restart_context.write(expected_context)
    replacement_token = handler._restart_context.write(expected_context)
    assert replacement_token != original_token
    monkeypatch.setattr(handler, "_read_generation", lambda: current)
    monkeypatch.setattr(handler, "_read_recovery_state", lambda: recovery_state)

    with pytest.raises(RendezvousStateError, match="context changed"):
        handler._validate_final_admission_state(
            cast(Any, current),
            cast(Any, recovery_state),
            time.monotonic() + 1,
            validate_deadline=False,
            expected_context_token=original_token,
        )

    handler._restart_context.clear()


def test_slot_aware_handler_treats_missing_peer_registration_as_not_ready(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=InMemoryControlStore(clock=clock),
        clock=clock,
        agent_id="agent-a",
    )
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=(SlotAssignment(0, "node-b", 0, 2),),
        topology_digest="topology-v1",
    )
    monkeypatch.setattr(
        handler,
        "_try_current_assigned_registration",
        lambda _node_id: None,
    )

    assert (
        handler._arrival_consumption_state(
            assignment,
            1,
            cast(Any, object()),
        )
        is None
    )


def test_slot_aware_handler_requires_live_registration_before_returning_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    deadline = time.monotonic() + 1
    handler._ensure_registered(deadline)
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    assert handler._stop_heartbeat() is True
    handler._registration_manager.release(registration)
    replacement_manager = AgentRegistrationManager(
        store,
        agent_identity=replace(
            handler.agent_identity,
            agent_id="replacement-agent-b",
        ),
        lease_duration_ms=handler._config.policy.registration_lease_duration_ms,
        clock=clock,
    )
    replacement_manager.register()
    monkeypatch.setattr(
        handler,
        "_validate_final_admission_state",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(handler, "_validate_current_arrival_attempt", lambda *_args: None)
    plan = _replacement_plan()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    completion, admission = _stub_replacement_return_evidence(
        handler,
        monkeypatch,
    )

    with pytest.raises(
        RendezvousConnectionError,
        match="registration changed before admission",
    ):
        handler._validate_replacement_return(
            cast(Any, object()),
            assignment,
            1,
            0,
            cast(Any, completion),
            cast(Any, admission),
            cast(Any, _static_recovery_state(plan)),
            deadline,
            "a" * 64,
        )


def test_slot_aware_handler_uses_authoritative_time_for_final_registration_expiry(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    client_clock = ManualClock()
    store_clock = ManualClock()
    store = InMemoryControlStore(clock=store_clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=client_clock,
        agent_id="agent-b",
    )
    handler._ensure_registered(time.monotonic() + 1)
    assert handler._stop_heartbeat() is True
    plan = _replacement_plan(restart_deadline_unix_ms=100_000)
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    monkeypatch.setattr(handler, "_validate_final_admission_state", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(handler, "_validate_current_arrival_attempt", lambda *_args: None)
    completion, admission = _stub_replacement_return_evidence(
        handler,
        monkeypatch,
    )
    store_clock.advance(handler._config.policy.registration_lease_duration_ms)

    with pytest.raises(
        RendezvousConnectionError,
        match="registration has expired",
    ):
        handler._validate_replacement_return(
            cast(Any, object()),
            assignment,
            1,
            0,
            cast(Any, completion),
            cast(Any, admission),
            cast(Any, _static_recovery_state(plan)),
            time.monotonic() + 1,
            "a" * 64,
        )
    handler.shutdown()


def test_slot_aware_handler_rejects_advanced_attempt_before_returning_admission(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )
    handler._ensure_registered(time.monotonic() + 1)
    with handler._registration_lock:
        registration = handler._registration
    assert registration is not None
    plan = _replacement_plan()
    assignment = RankAssignment.from_assignments(
        run_id=RUN_ID,
        generation=1,
        assignments=plan.slot_assignments,
        topology_digest=plan.topology_digest,
    )
    store.compare_set(
        handler._arrival_attempt_key(assignment.generation),
        expected_revision=None,
        value=handler._arrival_attempt_value(
            generation=assignment.generation,
            attempt=2,
            registration=registration,
        ),
    )
    monkeypatch.setattr(handler, "_validate_final_admission_state", lambda *_args, **_kwargs: None)
    completion, admission = _stub_replacement_return_evidence(
        handler,
        monkeypatch,
    )

    with pytest.raises(
        RendezvousStateError,
        match="does not match its generation",
    ):
        handler._validate_replacement_return(
            cast(Any, object()),
            assignment,
            1,
            0,
            cast(Any, completion),
            cast(Any, admission),
            cast(Any, _static_recovery_state(plan)),
            time.monotonic() + 1,
            "a" * 64,
        )

    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_assignment_size_mismatch(tmp_path: Path):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=2,
            max_nodes=3,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )

    with pytest.raises(RendezvousStateError, match="active node count"):
        handler.next_rendezvous()

    assert handler.shutdown() is True


def test_slot_aware_handler_rejects_deleted_closure_record(tmp_path: Path):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    first = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    first.set_closed()
    closure = store.get(first.closure_key)
    assert closure is not None
    store.compare_delete(first.closure_key, expected_revision=closure.revision)
    second = _handler(
        _handler_config(
            tmp_path,
            node_id="node-b",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    with pytest.raises(RendezvousStateError, match="deleted"):
        second.is_closed()

    assert second.shutdown() is True


def test_slot_aware_handler_bounds_closure_publication_before_cleanup(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    _initialize_generation(store, clock, ("node-a",))
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    handler._ensure_registered(time.monotonic() + 1)
    original_read = handler._read_closure_entry
    started = threading.Event()
    unblock = threading.Event()

    def delayed_read():
        started.set()
        unblock.wait(timeout=2)
        return original_read()

    monkeypatch.setattr(handler, "_read_closure_entry", delayed_read)

    before = time.monotonic()
    with pytest.raises(RendezvousConnectionError, match="timed out"):
        handler.set_closed()
    elapsed = time.monotonic() - before

    assert started.is_set()
    assert elapsed < 0.5
    _wait_until(
        lambda: (
            AgentRegistrationReader(
                store,
                run_id=RUN_ID,
                clock=clock,
            ).get("node-a")
            is None
        )
    )
    unblock.set()
    _wait_until(lambda: store.get(handler.closure_key) is not None)
    assert handler.shutdown() is True


def test_slot_aware_handler_retries_closure_created_during_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    clock = ManualClock()
    store = InMemoryControlStore(clock=clock)
    handler = _handler(
        _handler_config(
            tmp_path,
            node_id="node-a",
            min_nodes=1,
            max_nodes=2,
        ),
        store=store,
        clock=clock,
        agent_id="agent-a",
    )
    original_has_history = store.has_history
    published = False

    def publish_before_history(key: str) -> bool:
        nonlocal published
        if key == handler.closure_key and not published:
            published = True
            store.compare_set(
                handler.closure_key,
                expected_revision=None,
                value=handler._closure_value,
            )
        return original_has_history(key)

    monkeypatch.setattr(store, "has_history", publish_before_history)

    assert handler.is_closed()
    assert handler.shutdown() is True
