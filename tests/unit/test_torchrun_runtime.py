"""Contract tests for torchrun runtime configuration and context handoff."""

from __future__ import annotations

import os
import queue
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, cast

import pytest
from torch.distributed import HashStore
from torch.distributed.elastic.rendezvous import (
    RendezvousClosedError,
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
max_replacement_generations = 2
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


def test_runtime_config_loads_shared_policy_and_node_inputs(tmp_path: Path):
    source = _write_policy(tmp_path / "rendezvous.toml")
    context_path = tmp_path / "private" / "restart-context.json"

    config = TorchrunRuntimeConfig.from_parameters(
        _parameters(source),
        environment=_environment(context_path),
    )

    assert config.policy.control_endpoint == "control.example:443"
    assert config.policy.max_replacement_generations == 2
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
max_replacement_generations=2
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
            max_replacement_generations="4",
            registration_lease_duration_ms="45000",
            poll_interval_ms="500",
            join_timeout_ms="120000",
            replacement_only="true",
        ),
        environment=_environment(context_path),
    )

    assert config.policy.control_endpoint == "override.example:8443"
    assert config.policy.max_replacement_generations == 4
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

    context_file.write(_context())

    assert context_file.read() == _context()
    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(path.stat().st_mode) == 0o600

    context_file.write(_context(plan_id="plan-2"))
    assert context_file.read() == _context(plan_id="plan-2")
    assert list(path.parent.glob(f".{path.name}.*.tmp")) == []

    context_file.clear()
    assert not path.exists()
    context_file.clear()


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
    registration_lease_duration_ms: int = 120,
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
    context_file.write(_context())
    handler = _handler(
        config,
        store=store,
        clock=clock,
        agent_id="agent-b",
    )

    info = handler.next_rendezvous()

    assert handler.get_backend() == "lm_resiliency"
    assert handler.get_run_id() == RUN_ID
    assert handler.use_agent_store is True
    assert info.rank == 1
    assert info.world_size == 2
    assert info.bootstrap_store_info == RendezvousStoreInfo("master.example", 29500)
    assert not config.restart_context_path.exists()
    registration = AgentRegistrationReader(
        store,
        run_id=RUN_ID,
        clock=clock,
    ).get("node-b")
    assert registration is not None
    assert registration.record.agent_identity == handler.agent_identity

    assert handler.shutdown() is True
    assert handler.shutdown() is True
    assert (
        AgentRegistrationReader(
            store,
            run_id=RUN_ID,
            clock=clock,
        ).get("node-b")
        is None
    )


def test_slot_aware_handler_uses_attempt_scoped_bootstrap_store(
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

    second_outcome: queue.Queue[BaseException | object] = queue.Queue()
    second_thread = threading.Thread(target=rendezvous_b, args=(second_outcome,))
    second_thread.start()
    time.sleep(0.02)
    assert second_thread.is_alive()
    second_a = handler_a.next_rendezvous()
    second_thread.join(timeout=2)
    second_b = second_outcome.get_nowait()
    assert isinstance(second_b, type(second_a))
    assert second_b.bootstrap_store_info == second_a.bootstrap_store_info

    assert handler_a.shutdown() is True
    assert handler_b.shutdown() is True


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

    before = time.monotonic()
    with pytest.raises(RendezvousTimeoutError):
        handler.next_rendezvous()
    elapsed = time.monotonic() - before

    assert elapsed < 0.5
    assert handler.shutdown() is True


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


def test_slot_aware_handler_rejects_generation_after_initial_assignment(
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

    with pytest.raises(RendezvousStateError, match="replacement generations"):
        handler.next_rendezvous()

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

    assert second.shutdown() is False
