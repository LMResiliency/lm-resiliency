"""Tests for public torchrun manager and launch helpers."""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import pytest
from torch.distributed import HashStore

from lm_resiliency.integrations.torchrun import (
    TorchrunLaunchConfig,
    TorchrunRecoveryCoordinator,
    TorchrunRecoveryRequest,
    derive_torchrun_node_id,
)
from lm_resiliency.integrations.torchrun._simple_runtime import SimpleRecoveryPlanStore


def _request(*, generation: int, mode: str = "latest") -> TorchrunRecoveryRequest:
    return TorchrunRecoveryRequest(
        plan_id=f"plan-{generation + 1}",
        intent_id=f"incident-{generation + 1}",
        reason_code="process_stall" if mode == "latest" else "sdc_detected",
        recovery_mode=mode,
        checkpoint_source="gemini",
        checkpoint_step=generation + 1,
        checkpoint_manifest_id=f"gemini-{generation + 1}",
        topology_digest="ddp-world-2",
        restart_deadline_unix_ms=9_999_999_999_999,
    )


def test_coordinator_returns_committed_active_and_standby_nodes() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=3)
    plans.register_node("node-b", "agent-b", max_nodes=3)
    plans.register_node("node-c", "agent-c", max_nodes=3)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) == ("node-a", "node-b")

    placement = TorchrunRecoveryCoordinator(store, run_id="run").initial_placement(
        active_node_count=2,
        allocated_node_count=3,
    )

    assert placement is not None
    assert placement.active_node_ids == ("node-a", "node-b")
    assert placement.standby_node_ids == ("node-c",)


def test_coordinator_publishes_same_node_and_replacement_generations() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=3)
    plans.register_node("node-b", "agent-b", max_nodes=3)
    plans.register_node("node-c", "agent-c", max_nodes=3)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) == ("node-a", "node-b")
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")

    try:
        first = coordinator.publish_successor(
            generation=0,
            active_node_ids=("node-a", "node-b"),
            quarantined_node_ids=(),
            request=_request(generation=0),
            local_world_size=2,
        )
        second = coordinator.publish_successor(
            generation=1,
            active_node_ids=first.active_node_ids,
            quarantined_node_ids=first.quarantined_node_ids,
            request=_request(generation=1, mode="recovery_verified"),
            local_world_size=2,
            replacement=(1, "node-c"),
        )

        assert first.active_node_ids == ("node-a", "node-b")
        assert first.quarantined_node_ids == ()
        assert second.generation == 2
        assert second.active_node_ids == ("node-a", "node-c")
        assert second.quarantined_node_ids == ("node-b",)
        plan = SimpleRecoveryPlanStore(store, run_id="run").read(2)
        assert plan is not None
        assert plan.expected_world_size == 4
        assert plan.slot_assignments[1].first_global_rank == 2
    finally:
        coordinator.close()


def test_coordinator_rejects_stale_or_uncommitted_manager_state() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=3)
    plans.register_node("node-b", "agent-b", max_nodes=3)
    plans.register_node("node-c", "agent-c", max_nodes=3)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=3) == ("node-a", "node-b")
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")

    try:
        with pytest.raises(RuntimeError, match="active_node_ids"):
            coordinator.publish_successor(
                generation=0,
                active_node_ids=("node-b", "node-a"),
                quarantined_node_ids=(),
                request=_request(generation=0),
                local_world_size=1,
            )
        first = coordinator.publish_successor(
            generation=0,
            active_node_ids=("node-a", "node-b"),
            quarantined_node_ids=(),
            request=_request(generation=0),
            local_world_size=1,
            replacement=(1, "node-c"),
        )

        with pytest.raises(ValueError, match="not registered"):
            coordinator.publish_successor(
                generation=1,
                active_node_ids=first.active_node_ids,
                quarantined_node_ids=first.quarantined_node_ids,
                request=_request(generation=1),
                local_world_size=1,
                replacement=(0, "node-unregistered"),
            )
        with pytest.raises(RuntimeError, match="stale"):
            coordinator.publish_successor(
                generation=0,
                active_node_ids=("node-a", "node-b"),
                quarantined_node_ids=(),
                request=_request(generation=0),
                local_world_size=1,
            )
        with pytest.raises(RuntimeError, match="quarantine history"):
            coordinator.publish_successor(
                generation=1,
                active_node_ids=first.active_node_ids,
                quarantined_node_ids=(),
                request=_request(generation=1),
                local_world_size=1,
            )
        with pytest.raises(RuntimeError, match="local_world_size"):
            coordinator.publish_successor(
                generation=1,
                active_node_ids=first.active_node_ids,
                quarantined_node_ids=first.quarantined_node_ids,
                request=_request(generation=1),
                local_world_size=2,
            )
        with pytest.raises(RuntimeError, match="topology_digest"):
            coordinator.publish_successor(
                generation=1,
                active_node_ids=first.active_node_ids,
                quarantined_node_ids=first.quarantined_node_ids,
                request=replace(_request(generation=1), topology_digest="different-topology"),
                local_world_size=1,
            )
    finally:
        coordinator.close()


def test_coordinator_renews_plan_lease_until_close() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=2)
    plans.register_node("node-b", "agent-b", max_nodes=2)
    assert plans.ensure_initial_nodes(min_nodes=2, max_nodes=2) == ("node-a", "node-b")
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")

    coordinator.publish_successor(
        generation=0,
        active_node_ids=("node-a", "node-b"),
        quarantined_node_ids=(),
        request=replace(
            _request(generation=0),
            restart_deadline_unix_ms=time.time_ns() // 1_000_000 + 5_000,
        ),
        local_world_size=1,
    )
    plan = plans.read(1)
    assert plan is not None
    initial_lease = plans.read_plan_lease(plan)
    assert initial_lease is not None
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        lease = plans.read_plan_lease(plan)
        if lease is not None and lease[0] > initial_lease[0]:
            break
        time.sleep(0.02)
    else:
        pytest.fail("coordinator did not renew the plan lease")

    coordinator.close()
    stopped_lease = plans.read_plan_lease(plan)
    time.sleep(0.15)
    assert plans.read_plan_lease(plan) == stopped_lease


def test_coordinator_retries_transient_plan_lease_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=1)
    assert plans.ensure_initial_nodes(min_nodes=1, max_nodes=1) == ("node-a",)
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")
    original = coordinator._plans.renew_plan_lease
    calls = 0

    def flaky_renew(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("transient store failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator._plans, "renew_plan_lease", flaky_renew)
    coordinator.publish_successor(
        generation=0,
        active_node_ids=("node-a",),
        quarantined_node_ids=(),
        request=_request(generation=0),
        local_world_size=1,
    )
    deadline = time.monotonic() + 2
    while calls < 3 and time.monotonic() < deadline:
        time.sleep(0.02)

    assert calls >= 3
    coordinator.check_health()
    coordinator.close()


def test_coordinator_retries_initial_plan_lease_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=1)
    assert plans.ensure_initial_nodes(min_nodes=1, max_nodes=1) == ("node-a",)
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")
    original = coordinator._plans.renew_plan_lease
    calls = 0

    def flaky_renew(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RuntimeError("transient initial store failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator._plans, "renew_plan_lease", flaky_renew)
    placement = coordinator.publish_successor(
        generation=0,
        active_node_ids=("node-a",),
        quarantined_node_ids=(),
        request=_request(generation=0),
        local_world_size=1,
    )
    plan = plans.read(placement.generation)
    assert plan is not None
    deadline = time.monotonic() + 2
    while plans.read_plan_lease(plan) is None and time.monotonic() < deadline:
        time.sleep(0.02)

    assert calls >= 2
    assert plans.read_plan_lease(plan) is not None
    coordinator.check_health()
    coordinator.close()


def test_coordinator_surfaces_persistent_plan_lease_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=1)
    assert plans.ensure_initial_nodes(min_nodes=1, max_nodes=1) == ("node-a",)
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")
    original = coordinator._plans.renew_plan_lease
    calls = 0

    def failing_renew(*args: object, **kwargs: object) -> int:
        nonlocal calls
        calls += 1
        if calls > 1:
            raise RuntimeError("persistent store failure")
        return original(*args, **kwargs)

    monkeypatch.setattr(coordinator._plans, "renew_plan_lease", failing_renew)
    coordinator.publish_successor(
        generation=0,
        active_node_ids=("node-a",),
        quarantined_node_ids=(),
        request=_request(generation=0),
        local_world_size=1,
    )
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        try:
            coordinator.check_health()
        except RuntimeError:
            break
        time.sleep(0.02)
    else:
        pytest.fail("coordinator did not surface persistent lease failure")

    with pytest.raises(RuntimeError, match="lease renewal failed"):
        coordinator.current_generation
    with pytest.raises(RuntimeError, match="lease renewal failed"):
        coordinator.close()


def test_coordinator_rejects_expired_request_before_publish() -> None:
    store = HashStore()
    plans = SimpleRecoveryPlanStore(store, run_id="run")
    plans.register_node("node-a", "agent-a", max_nodes=1)
    assert plans.ensure_initial_nodes(min_nodes=1, max_nodes=1) == ("node-a",)
    coordinator = TorchrunRecoveryCoordinator(store, run_id="run")

    with pytest.raises(ValueError, match="deadline must be in the future"):
        coordinator.publish_successor(
            generation=0,
            active_node_ids=("node-a",),
            quarantined_node_ids=(),
            request=replace(
                _request(generation=0),
                restart_deadline_unix_ms=time.time_ns() // 1_000_000 - 1,
            ),
            local_world_size=1,
        )

    assert plans.read(1) is None
    coordinator.close()


def test_launch_config_builds_namespaced_framework_neutral_command(tmp_path: Path) -> None:
    config = TorchrunLaunchConfig(
        run_id="run",
        rendezvous_endpoint="host:1234",
        restart_context_path=(tmp_path / "context.json").resolve(),
        min_nodes=2,
        max_nodes=3,
        nproc_per_node=8,
        max_restarts=4,
        is_host=False,
    )

    command = config.command(module="package.train", module_args=("--steps=10",))

    assert "--rdzv-backend=lm_resiliency" in command
    assert "--nnodes=2:3" in command
    assert "--nproc-per-node=8" in command
    assert "--max-restarts=4" in command
    assert "package.train" in command
    assert command[-1] == "--steps=10"
    rendezvous = next(item for item in command if item.startswith("--rdzv-conf="))
    assert "lm_resiliency_restart_context_path=" in rendezvous
    assert "is_host=false" in rendezvous


def test_synthetic_machine_identity_uses_public_node_id_derivation() -> None:
    machine_id = "0123456789abcdef0123456789abcdef"

    assert derive_torchrun_node_id(machine_id) == derive_torchrun_node_id(machine_id)
