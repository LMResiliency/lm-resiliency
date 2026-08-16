"""Tests for the minimal manager-owned torchrun runtime."""

from __future__ import annotations

import json
import queue
import threading
import time
from dataclasses import replace
from pathlib import Path

import pytest
from torch.distributed import HashStore
from torch.distributed.elastic.rendezvous import (
    RendezvousInfo,
    RendezvousParameters,
)

from lm_resiliency.integrations.torchrun._protocol import (
    ProtocolValidationError,
    RestartContext,
    RestartPlan,
    SlotAssignment,
)
from lm_resiliency.integrations.torchrun._simple_runtime import (
    RecoveryPlanConflict,
    SimpleRecoveryPlanStore,
    SimpleRendezvousHandler,
    SimpleRestartContextFile,
    SimpleRuntimeConfig,
)

RUN_ID = "simple-run"


def _config(tmp_path: Path, node_id: str) -> SimpleRuntimeConfig:
    return SimpleRuntimeConfig(
        run_id=RUN_ID,
        node_id=node_id,
        active_nodes=("node-a", "node-b"),
        local_world_size=1,
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
    params = RendezvousParameters(
        backend="lm_resiliency",
        endpoint=str(tmp_path / "rdzv"),
        run_id=RUN_ID,
        min_nodes=2,
        max_nodes=3,
        node_id="node-a",
        active_nodes="node-a;node-b",
        local_world_size="2",
        restart_context_path=str((tmp_path / "context.json").resolve()),
        store_type="file",
        timeout="60",
    )

    config = SimpleRuntimeConfig.from_parameters(params, environment={})

    assert config.node_id == "node-a"
    assert config.active_nodes == ("node-a", "node-b")
    assert config.local_world_size == 2


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("active_nodes", "node-a", "exactly min_nodes"),
        ("local_world_size", "1.0", "positive integer"),
        ("restart_context_path", "relative.json", "must be absolute"),
    ],
)
def test_config_rejects_invalid_fields(
    tmp_path: Path,
    field: str,
    value: str,
    message: str,
) -> None:
    config = {
        "node_id": "node-a",
        "active_nodes": "node-a;node-b",
        "local_world_size": "1",
        "restart_context_path": str((tmp_path / "context.json").resolve()),
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
        SimpleRuntimeConfig.from_parameters(params, environment={})


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


def test_active_nodes_rendezvous_and_selected_standby_replaces_failed_node(
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
