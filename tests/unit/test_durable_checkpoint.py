"""Tests for SCOUT-gated durable checkpoint certification."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from lm_resiliency.checkpointing.durable import (
    DurableCheckpointConfig,
    DurableCheckpointCoordinator,
    DurableCheckpointEvent,
    DurableCheckpointRecord,
)
from lm_resiliency.detection.layer_replay import ReplayResult


@dataclass
class RecordingAdapter:
    saved: list[DurableCheckpointRecord] = field(default_factory=list)
    loaded: list[DurableCheckpointRecord] = field(default_factory=list)
    committed: list[tuple[DurableCheckpointRecord, DurableCheckpointRecord | None]] = field(
        default_factory=list
    )
    quarantined: list[tuple[DurableCheckpointRecord, str]] = field(default_factory=list)

    def save_candidate(self, candidate):
        self.saved.append(candidate)
        return {"framework_manifest_sha256": f"digest-{candidate.step}"}

    def load_checkpoint(self, checkpoint):
        self.loaded.append(checkpoint)
        return checkpoint.step

    def commit_candidate(self, checkpoint, previous):
        self.committed.append((checkpoint, previous))

    def quarantine_candidate(self, checkpoint, reason):
        self.quarantined.append((checkpoint, reason))


def _coordinator(
    tmp_path: Path,
    adapter: RecordingAdapter,
    *,
    shape_ids: tuple[str, ...] = ("small", "large"),
    checkpoint_io=None,
) -> DurableCheckpointCoordinator:
    return DurableCheckpointCoordinator(
        DurableCheckpointConfig(
            manifest_dir=str(tmp_path),
            environment_id="a100-cuda12-production-plan",
            adapter=adapter,
        ),
        shape_plan_id="qualified-shape-plan",
        shape_ids=shape_ids,
        checkpoint_io=checkpoint_io,
    )


def _result(shape_id: str, *, sdc: bool = False) -> ReplayResult:
    return ReplayResult(
        sdc_bitmap=[int(sdc), 0, 0],
        straggler_bitmap=[0, 0, 0],
        replay_time_ms=1.0,
        layer_id=0,
        checked_shape_ids=[shape_id],
        checked_shapes=[None],
        completed_shape_cycle=False,
        shape_cycle_size=2,
    )


def test_candidate_commits_only_after_every_shape_is_clean(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter)

    candidate = coordinator.begin_candidate(step=100, first_shape_id="small")
    first = coordinator.observe(_result("small"), step=110)
    second = coordinator.observe(_result("large"), step=120)

    assert first is DurableCheckpointEvent.PROGRESS
    assert second is DurableCheckpointEvent.COMMITTED
    assert coordinator.pending is None
    assert coordinator.latest_validated is not None
    assert coordinator.latest_validated.checkpoint_id == candidate.checkpoint_id
    assert coordinator.latest_validated.checked_shape_ids == ("small", "large")
    assert coordinator.latest_validated.completed_step == 120
    assert len(adapter.saved) == 1
    assert len(adapter.committed) == 1

    manifest = json.loads((tmp_path / "LATEST_SCOUT_VALIDATED").read_text())
    assert manifest["status"] == "recovery_verified"
    assert manifest["verdict"] == "clean"
    assert manifest["artifacts"]["framework_manifest_sha256"] == "digest-100"
    assert not (tmp_path / "PENDING_SCOUT_CANDIDATE").exists()


def test_framework_checkpoint_save_and_load_are_instrumented(tmp_path):
    adapter = RecordingAdapter()
    observed = []

    class Boundary:
        def __init__(self, operation, name):
            self.value = (operation, name)

        def __enter__(self):
            observed.append(("enter", *self.value))

        def __exit__(self, *_args):
            observed.append(("exit", *self.value))

    coordinator = _coordinator(
        tmp_path,
        adapter,
        shape_ids=("captured",),
        checkpoint_io=lambda operation, name: Boundary(operation, name),
    )
    candidate = coordinator.begin_candidate(step=10, first_shape_id="captured")
    coordinator.observe(_result("captured"), step=10)
    coordinator.load_latest_validated()

    assert observed == [
        ("enter", "write", candidate.checkpoint_id),
        ("exit", "write", candidate.checkpoint_id),
        ("enter", "read", candidate.checkpoint_id),
        ("exit", "read", candidate.checkpoint_id),
    ]


def test_dense_one_shape_candidate_commits_in_one_replay(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter, shape_ids=("captured",))

    coordinator.begin_candidate(step=10, first_shape_id="captured")
    event = coordinator.observe(_result("captured"), step=10)

    assert event is DurableCheckpointEvent.COMMITTED
    assert coordinator.latest_validated is not None
    assert coordinator.latest_validated.step == 10


def test_shape_specific_sdc_rejects_candidate_and_preserves_previous(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter)
    coordinator.begin_candidate(step=100, first_shape_id="small")
    coordinator.observe(_result("small"), step=110)
    coordinator.observe(_result("large"), step=120)
    previous = coordinator.latest_validated

    coordinator.begin_candidate(step=200, first_shape_id="small")
    coordinator.observe(_result("small"), step=210)
    event = coordinator.observe(_result("large", sdc=True), step=220)

    assert event is DurableCheckpointEvent.REJECTED
    assert coordinator.pending is None
    assert coordinator.latest_validated == previous
    assert adapter.quarantined[-1][0].step == 200
    assert "numerical divergence" in adapter.quarantined[-1][1]
    latest = json.loads((tmp_path / "LATEST_SCOUT_VALIDATED").read_text())
    assert latest["step"] == 100


def test_sdc_from_another_peer_group_rejects_job_wide_candidate(
    tmp_path,
    monkeypatch,
):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter)
    coordinator.begin_candidate(step=100, first_shape_id="small")
    monkeypatch.setattr(
        coordinator,
        "_all_gather_object",
        lambda value: [
            value,
            ("SCOUT detected numerical divergence", ()),
        ],
    )

    event = coordinator.observe(_result("small"), step=110)

    assert event is DurableCheckpointEvent.REJECTED
    assert coordinator.latest_validated is None
    assert "numerical divergence" in adapter.quarantined[-1][1]


def test_restart_quarantines_incomplete_candidate(tmp_path):
    adapter = RecordingAdapter()
    first = _coordinator(tmp_path, adapter)
    first.begin_candidate(step=100, first_shape_id="small")
    first.observe(_result("small"), step=110)

    restarted = _coordinator(tmp_path, adapter)

    assert restarted.pending is None
    assert not (tmp_path / "PENDING_SCOUT_CANDIDATE").exists()
    assert adapter.quarantined[-1][0].step == 100
    assert "restarted" in adapter.quarantined[-1][1]


def test_restart_after_pointer_publish_does_not_quarantine_committed_candidate(
    tmp_path,
):
    adapter = RecordingAdapter()
    first = _coordinator(tmp_path, adapter, shape_ids=("captured",))
    first.begin_candidate(step=100, first_shape_id="captured")
    first.observe(_result("captured"), step=100)
    latest = (tmp_path / "LATEST_SCOUT_VALIDATED").read_text()
    (tmp_path / "PENDING_SCOUT_CANDIDATE").write_text(latest)

    restarted = _coordinator(tmp_path, adapter, shape_ids=("captured",))

    assert restarted.latest_validated is not None
    assert restarted.latest_validated.step == 100
    assert adapter.quarantined == []
    assert not (tmp_path / "PENDING_SCOUT_CANDIDATE").exists()


def test_recovery_loads_validated_manifest_never_newer_pending_candidate(tmp_path):
    adapter = RecordingAdapter()
    first = _coordinator(tmp_path, adapter)
    first.begin_candidate(step=100, first_shape_id="small")
    first.observe(_result("small"), step=110)
    first.observe(_result("large"), step=120)
    first.begin_candidate(step=200, first_shape_id="small")

    restarted = _coordinator(tmp_path, adapter)
    recovered = restarted.load_latest_validated()

    assert recovered == 100
    assert [record.step for record in adapter.loaded] == [100]
    assert adapter.quarantined[-1][0].step == 200


def test_recovery_rejects_loader_that_does_not_confirm_the_loaded_step(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter, shape_ids=("captured",))
    coordinator.begin_candidate(step=100, first_shape_id="captured")
    coordinator.observe(_result("captured"), step=100)
    adapter.load_checkpoint = lambda checkpoint: None

    with pytest.raises(RuntimeError, match="did not return the loaded step"):
        coordinator.load_latest_validated()


def test_out_of_order_shape_evidence_rejects_candidate(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter)
    coordinator.begin_candidate(step=100, first_shape_id="small")

    event = coordinator.observe(_result("large"), step=110)

    assert event is DurableCheckpointEvent.REJECTED
    assert coordinator.latest_validated is None
    assert "shape order" in adapter.quarantined[-1][1]


def test_unavailable_replay_rejects_candidate(tmp_path):
    adapter = RecordingAdapter()
    coordinator = _coordinator(tmp_path, adapter)
    coordinator.begin_candidate(step=100, first_shape_id="small")

    event = coordinator.observe(None, step=110)

    assert event is DurableCheckpointEvent.REJECTED
    assert coordinator.latest_validated is None
    assert "did not produce evidence" in adapter.quarantined[-1][1]
