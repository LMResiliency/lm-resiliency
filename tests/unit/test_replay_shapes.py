"""Tests for the unified dense/MoE replay-shape API."""

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency.detection.layer_replay import ReplayInvocation, ReplayResult
from lm_resiliency.detection.replay_harness import (
    ModelReplayHarness,
    ReplayHarnessConfig,
    enable_replay_detection,
)
from lm_resiliency.detection.replay_shapes import (
    GroupedExpertMaterializer,
    LeadingDimensionMaterializer,
    ReplayShape,
    ReplayShapePlan,
    ReplayShapePlanMismatch,
    ReplayShapeScheduler,
    ReplayWorkload,
)
from lm_resiliency.detection.topology import ReplayPeerGroup, ReplayPeerRole


class RoutedExpertModel(nn.Module):
    def __init__(self, hidden_dim: int = 4):
        super().__init__()
        self.expert = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        routed = tokens + 1
        output = self.expert(routed)
        routed.add_(100)
        return output


def _result(*, sdc: bool = False) -> ReplayResult:
    bitmap = [0, int(sdc), 0]
    return ReplayResult(
        sdc_bitmap=bitmap,
        straggler_bitmap=[0, 0, 0],
        replay_time_ms=1.0,
        layer_id=0,
        peer_ranks=[0, 1, 2],
        replay_times_ms=[1.0, 1.0, 1.0],
        sdc_source_bitmaps={"output": bitmap},
        spatial_straggler_bitmap=[0, 0, 0],
    )


def test_dense_plan_contains_one_captured_identity_shape():
    plan = ReplayShapePlan.dense()

    assert plan.shapes == (ReplayShape.captured(),)
    assert plan.shapes[0].dimensions is None


def test_replay_workload_normalizes_and_validates_peer_role():
    workload = ReplayWorkload(peer_role="expert")

    assert workload.peer_role is ReplayPeerRole.EXPERT
    with pytest.raises(ValueError, match="unsupported replay peer role"):
        ReplayWorkload(peer_role="pipeline")


def test_harness_rejects_peer_group_for_a_different_model_state_role():
    model = RoutedExpertModel()
    workload = ReplayWorkload.dense([model.expert])

    with pytest.raises(ValueError, match="dense replay requires a matching peer group"):
        ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(check_interval=0, workload=workload),
            peer_group=ReplayPeerGroup(ReplayPeerRole.EXPERT, None, None),
        )


def test_leading_dimension_materializer_resizes_inputs_and_backward_signal():
    tokens = torch.arange(12, dtype=torch.float32).view(3, 4)
    metadata = torch.arange(3)
    fixed = torch.ones(2, 4)
    invocation = ReplayInvocation(
        args=(tokens, fixed),
        kwargs={"metadata": metadata},
        input_requires_grad=[True, False, False],
        grad_output=(tokens + 50,),
    )

    resized = LeadingDimensionMaterializer()(
        invocation,
        ReplayShape("n_exec=5", dimensions=(5,)),
    )

    assert resized.args[0].shape == (5, 4)
    assert torch.equal(resized.args[0][3:], tokens[:2])
    assert resized.args[1] is fixed
    assert torch.equal(resized.kwargs["metadata"], torch.tensor([0, 1, 2, 0, 1]))
    assert resized.grad_output[0].shape == (5, 4)
    assert resized.input_requires_grad == invocation.input_requires_grad


def test_grouped_expert_materializer_rebuilds_counts_and_resizes_token_inputs():
    tokens = torch.arange(20, dtype=torch.float32).view(5, 4)
    counts = torch.tensor([3, 0, 2], dtype=torch.int64)
    probabilities = torch.linspace(0.1, 0.5, 5)
    invocation = ReplayInvocation(
        args=(tokens, counts, probabilities),
        kwargs={},
        input_requires_grad=[True, False, False],
        grad_output=(tokens + 50, None),
    )

    resized = GroupedExpertMaterializer()(
        invocation,
        ReplayShape("n_exec=10", dimensions=(10,)),
    )

    assert resized.args[0].shape == (30, 4)
    assert resized.args[2].shape == (30,)
    assert torch.equal(resized.args[1], torch.tensor([10, 10, 10]))
    assert resized.args[1].sum().item() == 30
    assert resized.grad_output[0].shape == (30, 4)
    assert resized.grad_output[1] is None
    assert resized.input_requires_grad == invocation.input_requires_grad


def test_grouped_expert_materializer_sets_uniform_counts_at_same_packed_extent():
    tokens = torch.randn(6, 4)
    invocation = ReplayInvocation(
        args=(tokens,),
        kwargs={"tokens_per_expert": torch.tensor([6, 0])},
    )

    resized = GroupedExpertMaterializer(counts_input="tokens_per_expert")(
        invocation,
        ReplayShape("n_exec=3", dimensions=(3,)),
    )

    assert resized.args[0] is tokens
    assert torch.equal(resized.kwargs["tokens_per_expert"], torch.tensor([3, 3]))


def test_grouped_expert_materializer_rejects_vector_recipe():
    invocation = ReplayInvocation(
        args=(torch.randn(6, 4), torch.tensor([3, 2, 1])),
        kwargs={},
    )

    with pytest.raises(ValueError, match="one concrete per-expert replay dimension"):
        GroupedExpertMaterializer()(
            invocation,
            ReplayShape("counts=0x4x4", dimensions=(0, 4, 4)),
        )


def test_grouped_expert_materializer_enforces_backend_alignment():
    invocation = ReplayInvocation(
        args=(torch.randn(8, 4), torch.tensor([4, 4])),
        kwargs={},
    )
    materializer = GroupedExpertMaterializer(alignment=4)

    resized = materializer(invocation, ReplayShape("n_exec=12", dimensions=(12,)))
    assert torch.equal(resized.args[1], torch.tensor([12, 12]))

    with pytest.raises(ValueError, match="not aligned"):
        materializer(invocation, ReplayShape("n_exec=10", dimensions=(10,)))


@pytest.mark.parametrize(
    ("counts", "message"),
    [
        (torch.tensor([3, 3]), "must sum"),
        (torch.tensor([5, -1]), "non-negative"),
        (torch.tensor([True, True, True, True]), "integer dtype"),
    ],
)
def test_grouped_expert_materializer_rejects_invalid_captured_counts(counts, message):
    invocation = ReplayInvocation(
        args=(torch.randn(4, 4), counts),
        kwargs={},
    )

    with pytest.raises(ValueError, match=message):
        GroupedExpertMaterializer()(
            invocation,
            ReplayShape("n_exec=8", dimensions=(8,)),
        )


def test_shape_scheduler_restores_position_only_for_the_same_plan():
    plan = ReplayShapePlan.from_dimensions([(2,), (5,)], source_id="qualified")
    scheduler = ReplayShapeScheduler(plan)
    scheduler.advance()
    state = scheduler.state_dict()

    restored = ReplayShapeScheduler(plan)
    restored.load_state_dict(state)
    assert restored.current_shape.dimensions == (5,)

    changed = ReplayShapeScheduler(
        ReplayShapePlan.from_dimensions([(2,), (6,)], source_id="qualified")
    )
    with pytest.raises(ReplayShapePlanMismatch, match="different shape plan"):
        changed.load_state_dict(state)


def test_concrete_shapes_require_owned_capture_storage():
    model = RoutedExpertModel()
    workload = ReplayWorkload(
        shape_plan=ReplayShapePlan.from_dimensions([(2,), (5,)]),
        replay_modules=(model.expert,),
        materializer=LeadingDimensionMaterializer(),
    )

    with pytest.raises(ValueError, match="capture_inputs_by_value=True"):
        ModelReplayHarness(
            model,
            config=ReplayHarnessConfig(check_interval=0, workload=workload),
        )


def test_convenience_api_accepts_the_common_shape_workload():
    model = RoutedExpertModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    workload = ReplayWorkload.from_shapes(
        [(2,), (5,)],
        replay_modules=[model.expert],
        materializer=LeadingDimensionMaterializer(),
    )

    with patch("lm_resiliency.detection.replay_harness.ModelReplayHarness") as harness_constructor:
        enable_replay_detection(model, optimizer, check_interval=7, workload=workload)

    config = harness_constructor.call_args.kwargs["config"]
    assert config.workload is workload
    assert config.capture_inputs_by_value is True
    assert config.check_interval == 7


def test_harness_rotates_shapes_and_reports_shape_specific_injected_failure():
    model = RoutedExpertModel()
    workload = ReplayWorkload(
        shape_plan=ReplayShapePlan.from_dimensions([(2,), (5,)]),
        replay_modules=(model.expert,),
        materializer=LeadingDimensionMaterializer(),
    )
    harness = ModelReplayHarness(
        model,
        config=ReplayHarnessConfig(
            check_interval=0,
            capture_inputs_by_value=True,
            workload=workload,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
        ),
    )
    model(torch.randn(3, 4))

    observed_shapes = []

    def replay_with_shape(*, invocation, **kwargs):
        del kwargs
        n_exec = invocation.args[0].shape[0]
        observed_shapes.append(n_exec)
        return _result(sdc=n_exec == 5)

    detector = MagicMock()
    detector.replay_shape_consensus.return_value = (True, [0, 0, 0])
    detector.replay_invocation.side_effect = replay_with_shape
    harness._detector = detector

    first = harness.check()
    second = harness.check()

    assert observed_shapes == [2, 5]
    assert first.sdc_bitmap == [0, 0, 0]
    assert first.replay_shape == (2,)
    assert second.sdc_bitmap == [0, 1, 0]
    assert second.replay_shape == (5,)
    assert "shape-1-5" in second.replay_shape_id
    assert harness.current_replay_shape.dimensions == (2,)


def test_emergency_shape_cycle_preserves_normal_scheduler_position():
    model = RoutedExpertModel()
    workload = ReplayWorkload.from_shapes(
        [(2,), (5,)],
        replay_modules=[model.expert],
        materializer=LeadingDimensionMaterializer(),
    )
    harness = ModelReplayHarness(
        model,
        config=ReplayHarnessConfig(
            check_interval=0,
            capture_inputs_by_value=True,
            workload=workload,
            rotate_layers=False,
            enable_temporal=False,
            scale_factors=[],
        ),
    )
    model(torch.randn(3, 4))

    observed_shapes = []

    def replay_with_shape(*, invocation, **kwargs):
        del kwargs
        observed_shapes.append(invocation.args[0].shape[0])
        return _result()

    detector = MagicMock()
    detector.replay_shape_consensus.return_value = (True, [0, 0, 0])
    detector.replay_invocation.side_effect = replay_with_shape
    harness._detector = detector

    result = harness.check_shape_cycle(preserve_scheduler=True)

    assert result.completed_shape_cycle
    assert observed_shapes == [2, 5]
    assert harness.current_replay_shape.dimensions == (2,)

    harness.check()
    assert observed_shapes == [2, 5, 2]
    assert harness.current_replay_shape.dimensions == (5,)


def test_harness_rejects_cross_peer_shape_schedule_drift_without_advancing():
    model = RoutedExpertModel()
    workload = ReplayWorkload.from_shapes(
        [(2,), (5,)],
        replay_modules=[model.expert],
        materializer=LeadingDimensionMaterializer(),
    )
    harness = ModelReplayHarness(
        model,
        config=ReplayHarnessConfig(
            check_interval=0,
            capture_inputs_by_value=True,
            workload=workload,
            rotate_layers=False,
        ),
    )
    model(torch.randn(3, 4))
    detector = MagicMock()
    detector.replay_shape_consensus.return_value = (False, [0, 1, 0])
    harness._detector = detector

    with pytest.raises(RuntimeError, match="schedule differs across peers"):
        harness.check()

    detector.replay_invocation.assert_not_called()
    assert harness.current_replay_shape.dimensions == (2,)


def test_harness_checkpoint_state_preserves_shape_rotation():
    model = RoutedExpertModel()
    workload = ReplayWorkload(
        shape_plan=ReplayShapePlan.from_dimensions([(2,), (5,)]),
        replay_modules=(model.expert,),
        materializer=LeadingDimensionMaterializer(),
    )
    config = ReplayHarnessConfig(
        check_interval=4,
        capture_inputs_by_value=True,
        workload=workload,
        rotate_layers=False,
        enable_temporal=False,
        scale_factors=[],
    )
    harness = ModelReplayHarness(model, config=config)
    harness._shape_scheduler.advance()
    state = harness.temporal_state_dict()

    restored = ModelReplayHarness(model, config=config)
    restored.load_temporal_state_dict(state)

    assert restored.current_replay_shape.dimensions == (5,)
    assert restored.replay_shape_cycle_steps == 8
