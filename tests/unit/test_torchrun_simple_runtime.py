"""Tests for the minimal manager-owned torchrun runtime."""

from __future__ import annotations

import hashlib
import json
import os
import queue
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from torch.distributed import HashStore
from torch.distributed.elastic.rendezvous import (
    RendezvousInfo,
    RendezvousParameters,
    RendezvousTimeoutError,
)

from lm_resiliency.integrations.torchrun._protocol import (
    ProtocolValidationError,
    RestartContext,
    RestartPlan,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._simple_runtime import (
    RecoveryPlanConflict,
    RecoveryPlanCorrupt,
    SimpleRecoveryPlanStore,
    SimpleRendezvousHandler,
    SimpleRestartContextFile,
    SimpleRuntimeConfig,
    _create_rendezvous_handler,
    _machine_node_id,
    _node_id_from_machine_id,
)

RUN_ID = "simple-run"


def _machine_id(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()[:32]


def _machine_id_path(tmp_path: Path, label: str) -> Path:
    path = tmp_path / f"{label}.machine-id"
    path.write_text(_machine_id(label) + "\n", encoding="ascii")
    return path.resolve()


def _machine_environment(tmp_path: Path, label: str) -> dict[str, str]:
    return {"LM_RESILIENCY_MACHINE_ID_PATH": str(_machine_id_path(tmp_path, label))}


def _config(tmp_path: Path, node_id: str) -> SimpleRuntimeConfig:
    return SimpleRuntimeConfig(
        run_id=RUN_ID,
        node_id=node_id,
        min_nodes=2,
        max_nodes=3,
        restart_context_path=(tmp_path / node_id / "restart-context.json").resolve(),
        join_timeout_ms=5_000,
        poll_interval_ms=10,
        heartbeat_timeout_ms=1_000,
    )


def _handler(
    tmp_path: Path,
    store: HashStore,
    node_id: str,
) -> SimpleRendezvousHandler:
    return SimpleRendezvousHandler(
        _config(tmp_path, node_id),
        store=store,
        local_addr="127.0.0.1",
    )


def _plan() -> RestartPlan:
    return RestartPlan(
        plan_id="plan-1",
        intent_id="intent-1",
        run_id=RUN_ID,
        from_generation=0,
        to_generation=1,
        incident_ids=("incident-1",),
        reason_code="attributed_sdc",
        recovery_mode="recovery_verified",
        checkpoint_source="durable",
        checkpoint_step=40,
        checkpoint_id="checkpoint-40",
        checkpoint_manifest_id="manifest-40",
        slot_assignments=(
            SlotAssignment(0, "node-a", 0, 1),
            SlotAssignment(1, "node-c", 1, 1),
        ),
        quarantined_node_ids=("node-b",),
        expected_world_size=2,
        topology_digest="topology-v2",
        restart_deadline_unix_ms=time.time_ns() // 1_000_000 + 60_000,
    )


def test_restart_context_write_creates_private_parent(tmp_path: Path) -> None:
    path = tmp_path / "missing-context-directory" / "restart-context.json"
    context = RestartContext.from_plan(_plan(), "node-a")
    context_file = SimpleRestartContextFile(path)

    context_file.write(context)

    assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700
    assert context_file.read() == context


def _start(
    handler: SimpleRendezvousHandler,
) -> tuple[threading.Thread, queue.Queue[BaseException | RendezvousInfo]]:
    result: queue.Queue[BaseException | RendezvousInfo] = queue.Queue()

    def run() -> None:
        try:
            result.put(handler.next_rendezvous())
        except BaseException as error:
            result.put(error)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, result


def _result(
    thread: threading.Thread,
    result: queue.Queue[BaseException | RendezvousInfo],
) -> RendezvousInfo:
    thread.join(timeout=5)
    assert not thread.is_alive()
    value = result.get_nowait()
    if isinstance(value, BaseException):
        raise value
    return value


def test_config_resolves_torchrun_fields(tmp_path: Path) -> None:
    worker_config = (tmp_path / "worker.toml").resolve()
    environment = _machine_environment(tmp_path, "node-a")
    params = RendezvousParameters(
        backend="lm_resiliency",
        endpoint=str(tmp_path / "rdzv"),
        run_id=RUN_ID,
        min_nodes=2,
        max_nodes=3,
        lm_resiliency_restart_context_path=str((tmp_path / "context.json").resolve()),
        lm_resiliency_worker_config=str(worker_config),
        store_type="file",
        timeout="60",
    )

    config = SimpleRuntimeConfig.from_parameters(params, environment=environment)

    assert config.node_id == _node_id_from_machine_id(_machine_id("node-a"))
    assert config.min_nodes == 2
    assert config.max_nodes == 3
    assert config.worker_config == worker_config


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("lm_resiliency_restart_context_path", "relative.json", "must be absolute"),
        ("lm_resiliency_worker_config", "relative.toml", "must be absolute"),
    ],
)
def test_config_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    config = {
        "lm_resiliency_restart_context_path": str((tmp_path / "context.json").resolve()),
    }
    config[field] = value
    params = RendezvousParameters(
        backend="lm_resiliency",
        endpoint=str(tmp_path / "rdzv"),
        run_id=RUN_ID,
        min_nodes=2,
        max_nodes=3,
        **config,
    )

    with pytest.raises(ValueError, match=message):
        SimpleRuntimeConfig.from_parameters(
            params,
            environment=_machine_environment(tmp_path, "node-a"),
        )


def test_config_rejects_unknown_rendezvous_fields(tmp_path: Path) -> None:
    params = RendezvousParameters(
        backend="lm_resiliency",
        endpoint=str(tmp_path / "rdzv"),
        run_id=RUN_ID,
        min_nodes=2,
        max_nodes=3,
        lm_resiliency_restart_context_path=str((tmp_path / "context.json").resolve()),
        unexpected="unused",
    )

    with pytest.raises(ValueError, match="unknown rendezvous configuration"):
        SimpleRuntimeConfig.from_parameters(
            params,
            environment=_machine_environment(tmp_path, "node-a"),
        )


@pytest.mark.parametrize(
    "value",
    [
        "",
        "0" * 32,
        "not-a-machine-id",
        "g" * 32,
    ],
)
def test_machine_identity_rejects_invalid_values(
    tmp_path: Path,
    value: str,
) -> None:
    path = tmp_path / "machine-id"
    path.write_text(value + "\n", encoding="ascii")

    with pytest.raises(ValueError, match="machine identity"):
        _machine_node_id(path)


def test_machine_identity_is_hashed_and_stable(tmp_path: Path) -> None:
    path = _machine_id_path(tmp_path, "node-a")

    first = _machine_node_id(path)
    second = _machine_node_id(path)

    assert first == second == _node_id_from_machine_id(_machine_id("node-a"))
    assert _machine_id("node-a") not in first


def test_machine_identity_requires_a_regular_file(tmp_path: Path) -> None:
    path = tmp_path / "machine-id"
    path.mkdir()

    with pytest.raises(ValueError, match="regular file"):
        _machine_node_id(path)


def test_worker_config_is_the_automatic_adapter_opt_in(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    activation = "LM_RESILIENCY_TORCHRUN_BOOTSTRAP"
    removed_width = "LM_RESILIENCY_TORCHRUN_LOCAL_WORLD_SIZE"
    monkeypatch.delenv(activation, raising=False)
    monkeypatch.delenv(removed_width, raising=False)
    monkeypatch.setenv(
        "LM_RESILIENCY_MACHINE_ID_PATH",
        str(_machine_id_path(tmp_path, "node-a")),
    )

    def parameters(name: str, *, worker_config: Path | None) -> RendezvousParameters:
        config: dict[str, str] = {
            "lm_resiliency_restart_context_path": str((tmp_path / name / "context.json").resolve()),
            "store_type": "file",
        }
        if worker_config is not None:
            config["lm_resiliency_worker_config"] = str(worker_config)
        return RendezvousParameters(
            backend="lm_resiliency",
            endpoint=str(tmp_path / f"{name}.rdzv"),
            run_id=name,
            min_nodes=1,
            max_nodes=1,
            **config,
        )

    explicit = _create_rendezvous_handler(parameters("explicit", worker_config=None))
    try:
        assert activation not in os.environ
    finally:
        assert explicit.shutdown()

    policy = (tmp_path / "worker.toml").resolve()
    automatic = _create_rendezvous_handler(parameters("automatic", worker_config=policy))
    try:
        assert os.environ[activation] == "1"
        assert removed_width not in os.environ
    finally:
        assert automatic.shutdown()


def test_plan_store_is_create_once_and_advances_generation() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id=RUN_ID)
    plan = _plan()

    plans.publish(plan)
    plans.publish(plan)

    assert plans.current_generation() == 1
    assert plans.read(1) == plan
    with pytest.raises(RecoveryPlanConflict):
        plans.publish(replace(plan, plan_id="other-plan"))


def test_initial_nodes_are_committed_once_in_registration_order() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id=RUN_ID)

    plans.register_node("node-b", "agent-b", max_nodes=3)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) is None
    plans.register_node("node-a", "agent-a", max_nodes=3)

    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) == (
        "node-b",
        "node-a",
    )
    plans.register_node("node-c", "agent-c", max_nodes=3)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) == (
        "node-b",
        "node-a",
    )
    assert plans.read_initial_nodes() == ("node-b", "node-a")
    assert plans.registered_nodes(max_nodes=3) == (
        "node-b",
        "node-a",
        "node-c",
    )


def test_duplicate_live_agents_with_one_machine_identity_fail_closed() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id=RUN_ID)

    plans.register_node("node-a", "agent-a", max_nodes=2)
    with pytest.raises(RecoveryPlanCorrupt, match="same machine identity"):
        plans.register_node("node-a", "agent-b", max_nodes=2)


def test_registration_schema_rejects_boolean_version() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id=RUN_ID)
    store.append(
        f"{plans.prefix}/initial/registrations",
        b'{"agent_id":"agent-a","node_id":"node-a","schema_version":true}\n',
    )

    with pytest.raises(RecoveryPlanCorrupt, match="invalid fields"):
        plans.registered_nodes(max_nodes=2)


def test_registration_schema_rejects_float_version() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id=RUN_ID)
    store.append(
        f"{plans.prefix}/initial/registrations",
        b'{"agent_id":"agent-a","node_id":"node-a","schema_version":1.0}\n',
    )

    with pytest.raises(RecoveryPlanCorrupt, match="invalid fields"):
        plans.registered_nodes(max_nodes=2)


def test_plan_and_context_have_canonical_round_trips() -> None:
    plan = _plan()
    context = RestartContext.from_plan(plan, "node-c")

    assert RestartPlan.from_json(plan.to_json()) == plan
    assert RestartContext.from_json(context.to_json()) == context
    assert context.logical_node_slot == 1
    assert context.first_global_rank == 1
    assert context.checkpoint_step == 40


def test_plan_decoder_rejects_duplicate_fields() -> None:
    encoded = _plan().to_json()
    duplicated = encoded.replace(
        '"schema_version":1',
        '"schema_version":1,"schema_version":1',
        1,
    )

    with pytest.raises(ProtocolValidationError, match="duplicate JSON field"):
        RestartPlan.from_json(duplicated)


@pytest.mark.parametrize(
    "mutation",
    [
        {"schema_version": 1.0},
        {"unexpected": "value"},
    ],
)
def test_plan_decoder_rejects_noncanonical_wire_fields(
    mutation: dict[str, object],
) -> None:
    payload = _plan().to_dict()
    payload.update(mutation)

    with pytest.raises(ProtocolValidationError):
        RestartPlan.from_json(json.dumps(payload))


def test_automatic_initial_nodes_and_selected_standby_replace_failed_node(
    tmp_path: Path,
) -> None:
    store = HashStore()
    active_a = _handler(tmp_path, store, "node-a")
    active_b = _handler(tmp_path, store, "node-b")
    thread_a, result_a = _start(active_a)
    thread_b, result_b = _start(active_b)

    first_a = _result(thread_a, result_a)
    first_b = _result(thread_b, result_b)
    assert (first_a.rank, first_a.world_size) == (0, 2)
    assert (first_b.rank, first_b.world_size) == (1, 2)

    standby = _handler(tmp_path, store, "node-c")
    standby_thread, standby_result = _start(standby)
    time.sleep(0.05)
    assert standby_thread.is_alive()

    plan = _plan()
    SimpleRecoveryPlanStore(store, run_id=RUN_ID).publish(plan)
    assert active_a.num_nodes_waiting() == 1

    replacement_thread, replacement_result = _start(active_a)
    second_a = _result(replacement_thread, replacement_result)
    second_c = _result(standby_thread, standby_result)

    assert (second_a.rank, second_a.world_size) == (0, 2)
    assert (second_c.rank, second_c.world_size) == (1, 2)
    assert active_a.num_nodes_waiting() == 0
    assert SimpleRestartContextFile(
        _config(tmp_path, "node-a").restart_context_path
    ).read() == RestartContext.from_plan(plan, "node-a")
    assert SimpleRestartContextFile(
        _config(tmp_path, "node-c").restart_context_path
    ).read() == RestartContext.from_plan(plan, "node-c")

    assert active_a.shutdown()
    assert active_b.shutdown()
    assert standby.shutdown()


def test_same_node_successor_plan_signals_job_restart(tmp_path: Path) -> None:
    store = HashStore()
    active_a = _handler(tmp_path, store, "node-a")
    active_b = _handler(tmp_path, store, "node-b")
    first_a_thread, first_a_result = _start(active_a)
    first_b_thread, first_b_result = _start(active_b)
    _result(first_a_thread, first_a_result)
    _result(first_b_thread, first_b_result)

    plan = replace(
        _plan(),
        plan_id="same-node-restart",
        intent_id="transient-process-stall",
        reason_code="process_stall",
        recovery_mode="latest",
        checkpoint_source="gemini",
        checkpoint_step=41,
        checkpoint_id=None,
        checkpoint_manifest_id="gemini-latest-41",
        slot_assignments=(
            SlotAssignment(0, "node-a", 0, 1),
            SlotAssignment(1, "node-b", 1, 1),
        ),
        quarantined_node_ids=(),
    )
    SimpleRecoveryPlanStore(store, run_id=RUN_ID).publish(plan)

    assert active_a.num_nodes_waiting() == 1
    second_a_thread, second_a_result = _start(active_a)
    second_b_thread, second_b_result = _start(active_b)
    second_a = _result(second_a_thread, second_a_result)
    second_b = _result(second_b_thread, second_b_result)

    assert (second_a.rank, second_b.rank) == (0, 1)
    assert active_a.num_nodes_waiting() == 0
    assert active_a.shutdown()
    assert active_b.shutdown()


def test_initial_admission_timeout_is_bounded(tmp_path: Path) -> None:
    store = HashStore()
    handler = SimpleRendezvousHandler(
        replace(_config(tmp_path, "node-a"), join_timeout_ms=10),
        store=store,
        local_addr="127.0.0.1",
        monotonic_clock=iter((0.0, 1.0)).__next__,
    )

    with pytest.raises(RendezvousTimeoutError, match="initial node registrations"):
        handler.next_rendezvous()

    assert handler.shutdown()


def test_global_close_gracefully_exits_parked_standby(tmp_path: Path) -> None:
    store = HashStore()
    standby = _handler(tmp_path, store, "node-c")
    thread, result = _start(standby)
    time.sleep(0.05)

    SimpleRecoveryPlanStore(store, run_id=RUN_ID).close_run()
    thread.join(timeout=5)

    assert not thread.is_alive()
    value = result.get_nowait()
    assert isinstance(value, SystemExit)
    assert value.code == 0
