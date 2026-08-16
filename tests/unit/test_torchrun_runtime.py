"""Contract tests for torchrun runtime configuration and context handoff."""

from __future__ import annotations

import os
import stat
from pathlib import Path
from typing import Any, cast

import pytest
from torch.distributed.elastic.rendezvous import RendezvousParameters

from lm_resiliency.integrations.torchrun._protocol import RestartContext
from lm_resiliency.integrations.torchrun._runtime import (
    RestartContextFile,
    RestartContextFileError,
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


def _write_policy(path: Path, content: str = POLICY) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _parameters(path: Path, **config: object) -> RendezvousParameters:
    return RendezvousParameters(
        backend="lm_resiliency",
        endpoint="rdzv.example:29400",
        run_id=RUN_ID,
        min_nodes=2,
        max_nodes=3,
        local_addr="node-a.example",
        config=str(path),
        **config,
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
        _parameters(first, node_id="node-a", restart_context_path="/run/a.json"),
        environment={},
    )
    second_config = TorchrunRuntimeConfig.from_parameters(
        _parameters(second, node_id="node-b", restart_context_path="/run/b.json"),
        environment={},
    )

    assert first_config.policy == second_config.policy
    assert first_config.policy.digest == second_config.policy.digest
    assert first_config.node_id != second_config.node_id


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


@pytest.mark.parametrize(
    ("parameter_name", "environment_name", "parameter_value", "environment_value"),
    [
        ("node_id", "LM_RESILIENCY_NODE_ID", "node-a", "node-b"),
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
            _parameters(source, **{parameter_name: parameter_value}),
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

    def fail_replace(source: object, destination: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr(os, "replace", fail_replace)

    with pytest.raises(RestartContextFileError, match="failed to publish"):
        context_file.write(_context(plan_id="plan-2"))

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

    def fail_fchmod(descriptor: int, mode: int) -> None:
        raise OSError("injected permission failure")

    def record_close(descriptor: int) -> None:
        closed.append(descriptor)
        real_close(descriptor)

    monkeypatch.setattr(os, "fchmod", fail_fchmod)
    monkeypatch.setattr(os, "close", record_close)

    with pytest.raises(RestartContextFileError, match="failed to publish"):
        context_file.write(_context())

    assert len(closed) == 1
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


def test_restart_context_file_rejects_symlink_parent(tmp_path: Path):
    actual = tmp_path / "actual"
    actual.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(actual, target_is_directory=True)
    context_file = RestartContextFile(linked / "restart-context.json")

    with pytest.raises(RestartContextFileError, match="parent directory.*symlink"):
        context_file.write(_context())


def test_restart_context_file_rejects_insecure_file_permissions(tmp_path: Path):
    path = tmp_path / "private" / "restart-context.json"
    context_file = RestartContextFile(path)
    context_file.write(_context())
    path.chmod(0o644)

    with pytest.raises(RestartContextFileError, match="group or other"):
        context_file.read()


@pytest.mark.parametrize("path", [Path("relative.json"), cast(Any, "not-a-path")])
def test_restart_context_file_validates_path(path: object):
    expected = TypeError if isinstance(path, str) else RestartContextFileError

    with pytest.raises(expected):
        RestartContextFile(cast(Any, path))
