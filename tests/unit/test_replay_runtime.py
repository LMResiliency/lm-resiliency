"""Focused tests for replay RNG and FSDP gradient-communication runtime helpers."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import torch
import torch.nn as nn

from lm_resiliency.detection._utils import synchronized_replay_rng
from lm_resiliency.detection.c3 import C3Result, C3Status
from lm_resiliency.detection.layer_replay import (
    FSDP_PARAMETER_ALL_GATHER,
    PARAMETER_STATE,
    LayerReplayDetector,
    OpTiming,
    ReplayInvocation,
    ReplayResult,
    _differentiable_outputs_and_grads,
    _replay_autocast,
)
from lm_resiliency.integrations.pytorch.gradient_replay import (
    replay_fsdp_gradient_communication,
)


def test_synchronized_replay_rng_installs_source_state_and_restores_local_state():
    local_state = {"torch_cpu": torch.tensor([1], dtype=torch.uint8)}
    source_state = {"torch_cpu": torch.tensor([2], dtype=torch.uint8)}

    def broadcast(payload, **_kwargs):
        assert payload == [None]
        payload[0] = source_state

    with (
        patch(
            "lm_resiliency.detection._utils.capture_rng_state",
            return_value=local_state,
        ),
        patch("lm_resiliency.detection._utils.restore_rng_state") as restore,
        patch("lm_resiliency.detection._utils.dist.is_available", return_value=True),
        patch("lm_resiliency.detection._utils.dist.is_initialized", return_value=True),
        patch("lm_resiliency.detection._utils.dist.get_rank", return_value=3),
        patch(
            "lm_resiliency.detection._utils.dist.broadcast_object_list",
            side_effect=broadcast,
        ) as broadcast_object_list,
    ):
        with synchronized_replay_rng("gloo", source_global_rank=1) as installed:
            assert installed is source_state
            restore.assert_called_once_with(source_state)

    assert restore.call_args_list == [call(source_state), call(local_state)]
    broadcast_object_list.assert_called_once_with([source_state], src=1, group="gloo")


def test_structured_backward_replay_preserves_gradient_for_tensor_output():
    output = torch.randn(3, 4, requires_grad=True)
    captured_gradient = torch.full_like(output, 7)

    outputs, gradients = _differentiable_outputs_and_grads(
        (output, None),
        (captured_gradient, None),
    )

    assert outputs == [output]
    assert gradients == [captured_gradient]


def test_replay_restores_captured_autocast_context():
    layer = nn.Linear(4, 4).to(dtype=torch.bfloat16)
    invocation = ReplayInvocation(
        args=(torch.ones(2, 4, dtype=torch.float32),),
        kwargs={},
        autocast_enabled=True,
        autocast_device_type="cpu",
        autocast_dtype=torch.bfloat16,
    )

    with _replay_autocast(invocation):
        output = layer(*invocation.args)

    assert output.dtype == torch.bfloat16


def test_invocation_broadcast_materializes_noncontiguous_source_tensors():
    detector = object.__new__(LayerReplayDetector)
    detector._broadcast_src = 0
    detector._peer_ranks = [0]
    detector._group = "gloo"
    detector._nccl_group = None
    detector._device = torch.device("cpu")
    detector._rank = 0
    source = torch.arange(12).view(3, 4).transpose(0, 1)
    assert not source.is_contiguous()

    with (
        patch("lm_resiliency.detection.layer_replay.dist.broadcast_object_list"),
        patch("lm_resiliency.detection.layer_replay.dist.broadcast") as broadcast,
    ):
        shared = detector._broadcast_invocation(ReplayInvocation(args=(source,), kwargs={}))

    transmitted = broadcast.call_args.args[0]
    assert transmitted.is_contiguous()
    assert shared.args[0].is_contiguous()
    assert torch.equal(shared.args[0], source)


def test_replay_retains_input_and_rng_precondition_evidence():
    detector = object.__new__(LayerReplayDetector)
    detector._compare_parameter_state = False
    detector._broadcast_invocation = lambda invocation: invocation
    detector._peer_ranks = [0, 1, 2]
    detector._broadcast_src = 0
    detector._group = None
    detector._synchronize_rng = True
    detector._straggler_min_slowdown_ratio = 1.1
    detector._straggler_min_slowdown_ms = 2.0
    detector._check_forward_sdc = MagicMock(return_value=({"output": [torch.ones(2)]}, 1.0))
    prepared = ReplayInvocation(args=(torch.full((2,), 2.0),), kwargs={})
    detector._invocation_preparer = MagicMock(return_value=prepared)
    detector._c3 = MagicMock()
    detector._c3.run_structure.side_effect = [
        C3Result(C3Status.AGREE, [0, 0, 0], [10, 10, 10]),
        C3Result(C3Status.ATTRIBUTED, [0, 1, 0], [20, 21, 20]),
    ]
    detector._c3.run_tensor_groups.return_value = {
        "output": C3Result(C3Status.AGREE, [0, 0, 0], [30, 30, 30])
    }
    detector._c3.run_scalar.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [1.0, 1.0, 1.0],
    )
    invocation = ReplayInvocation(args=(torch.ones(2),), kwargs={"scale": 1.0})
    rng_state = {"torch_cpu": torch.tensor([1], dtype=torch.uint8)}
    layer = nn.Identity()

    with patch(
        "lm_resiliency.detection.layer_replay.synchronized_replay_rng",
        return_value=nullcontext(rng_state),
    ):
        result = detector.replay_invocation(invocation=invocation, layer=layer)

    assert set(result.c3_results) == {
        "replay_input",
        "replay_rng_state",
        "output",
    }
    assert result.sdc_bitmap == [0, 1, 0]
    assert result.sdc_sources == ["replay_rng_state"]
    detector._invocation_preparer.assert_called_once()
    detector._check_forward_sdc.assert_called_once_with(
        layer,
        prepared.args,
        prepared.kwargs,
    )


def test_replay_prepares_framework_evidence_before_c3():
    detector = object.__new__(LayerReplayDetector)
    detector._compare_parameter_state = False
    detector._broadcast_invocation = lambda invocation: invocation
    detector._peer_ranks = [0, 1, 2]
    detector._broadcast_src = 0
    detector._group = None
    detector._synchronize_rng = False
    detector._invocation_preparer = None
    detector._straggler_min_slowdown_ratio = 1.1
    detector._straggler_min_slowdown_ms = 2.0
    prepared = torch.full((2,), 7.0)
    detector._evidence_preparer = MagicMock(
        return_value={"output": [prepared]},
    )
    detector._check_forward_sdc = MagicMock(return_value=({"output": [torch.zeros(2)]}, 1.0))
    detector._c3 = MagicMock()
    detector._c3.run_structure.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [10, 10, 10],
    )
    detector._c3.run_tensor_groups.return_value = {
        "output": C3Result(C3Status.AGREE, [0, 0, 0], [20, 20, 20])
    }
    detector._c3.run_scalar.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [1.0, 1.0, 1.0],
    )

    detector.replay_invocation(
        invocation=ReplayInvocation(args=(torch.ones(2),), kwargs={}),
        layer=nn.Identity(),
    )

    detector._evidence_preparer.assert_called_once()
    assert detector._c3.run_tensor_groups.call_args.args[0] == {
        "output": [prepared],
    }


def test_parameter_state_is_compared_before_replay_executes():
    class TrackingLinear(nn.Linear):
        replayed = False

        def forward(self, value):
            self.replayed = True
            return super().forward(value)

    detector = object.__new__(LayerReplayDetector)
    detector._compare_parameter_state = True
    detector._broadcast_invocation = lambda invocation: invocation
    detector._peer_ranks = [0, 1, 2]
    detector._broadcast_src = 0
    detector._group = None
    detector._synchronize_rng = False
    detector._straggler_min_slowdown_ratio = 1.1
    detector._straggler_min_slowdown_ms = 2.0
    detector._check_forward_sdc = MagicMock(
        side_effect=lambda layer, *_args: (
            {"output": [layer(torch.ones(1, 2))]},
            1.0,
        )
    )
    detector._c3 = MagicMock()
    layer = TrackingLinear(2, 2)

    def compare_parameters(_tensors):
        assert not layer.replayed
        return C3Result(C3Status.AGREE, [0, 0, 0], [10, 10, 10])

    detector._c3.run_tensor_sequence.side_effect = compare_parameters
    detector._c3.run_structure.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [20, 20, 20],
    )
    detector._c3.run_tensor_groups.return_value = {
        "output": C3Result(C3Status.AGREE, [0, 0, 0], [30, 30, 30])
    }
    detector._c3.run_scalar.return_value = C3Result(
        C3Status.AGREE,
        [0, 0, 0],
        [1.0, 1.0, 1.0],
    )

    result = detector.replay_invocation(
        invocation=ReplayInvocation(args=(torch.ones(1, 2),), kwargs={}),
        layer=layer,
    )

    assert layer.replayed
    assert result.c3_results[PARAMETER_STATE].status is C3Status.AGREE
    detector._c3.run_tensor_sequence.assert_called_once()
    assert PARAMETER_STATE not in detector._c3.run_tensor_groups.call_args.args[0]


def test_external_fsdp_timing_is_communication_evidence():
    detector = object.__new__(LayerReplayDetector)
    detector._peer_ranks = [4, 8, 12]
    detector._rank = 1
    detector._straggler_min_slowdown_ratio = 1.1
    detector._straggler_min_slowdown_ms = 2.0
    detector._c3 = MagicMock()
    detector._c3.run_scalar.return_value = C3Result(
        C3Status.ATTRIBUTED,
        [0, 1, 0],
        [10.0, 30.0, 10.0],
    )
    result = ReplayResult(
        sdc_bitmap=[0, 0, 0],
        straggler_bitmap=[0, 0, 0],
        replay_time_ms=5.0,
        layer_id=0,
        spatial_straggler_bitmap=[0, 0, 0],
    )

    detector.add_communication_timing(
        result,
        name=FSDP_PARAMETER_ALL_GATHER,
        elapsed_ms=30.0,
        group_ranks=(8, 9),
        topology_role="fsdp",
    )

    assert result.straggler_bitmap == [0, 1, 0]
    assert result.straggler_detail is not None
    assert result.straggler_detail.straggler_rank == 8
    assert result.straggler_detail.straggler_type == "communication"
    assert result.communication_peer_times_ms[FSDP_PARAMETER_ALL_GATHER] == [
        10.0,
        30.0,
        10.0,
    ]
    assert result.c3_results[f"{FSDP_PARAMETER_ALL_GATHER}.timing"].bitmap == [
        0,
        1,
        0,
    ]
    assert len(result.collective_timings) == 1
    assert result.collective_timings[0].group_ranks == (8, 9)
    assert result.collective_timings[0].slow


def test_collective_timings_are_classified_across_equivalent_peer_groups():
    detector = object.__new__(LayerReplayDetector)
    detector._c3 = SimpleNamespace(_world_size=4)
    detector._group = object()
    detector._rank = 1
    detector._straggler_min_slowdown_ratio = 1.1
    detector._straggler_min_slowdown_ms = 2.0
    local = OpTiming(
        name="all_reduce",
        type="communication",
        time_ms=5.0,
        group_ranks=(1, 4),
        message_bytes=4096,
        sequence=0,
    )
    peers = [
        [
            OpTiming(
                "all_reduce",
                "communication",
                1.0,
                group_ranks=(0, 3),
                message_bytes=4096,
            )
        ],
        [local],
        [
            OpTiming(
                "all_reduce",
                "communication",
                1.0,
                group_ranks=(2, 5),
                message_bytes=4096,
            )
        ],
        [
            OpTiming(
                "all_reduce",
                "communication",
                1.0,
                group_ranks=(3, 6),
                message_bytes=4096,
            )
        ],
    ]

    def gather(output, value, group):
        assert value == [local]
        assert group is detector._group
        output[:] = peers

    with patch(
        "lm_resiliency.detection.layer_replay.dist.all_gather_object",
        side_effect=gather,
    ):
        samples = detector._classify_collective_timings([local])

    assert len(samples) == 1
    assert samples[0].group_ranks == (1, 4)
    assert samples[0].slow


def test_fsdp_replay_times_forward_before_synchronous_backward_communication():
    layer = nn.Linear(4, 4)
    invocation = ReplayInvocation(
        args=(torch.ones(2, 4),),
        kwargs={},
        input_requires_grad=[True],
        grad_output=(torch.ones(2, 4),),
    )
    detector = object.__new__(LayerReplayDetector)
    detector._device = torch.device("cpu")
    detector._deterministic = False
    events = []

    def clock():
        events.append("clock")
        return (10.0, 10.125)[events.count("clock") - 1]

    def gradient_communication(_layer, gradients):
        assert gradients
        events.append("communication")

    detector._gradient_communication = gradient_communication
    with (
        patch(
            "lm_resiliency.detection.layer_replay.torch.cuda.synchronize",
            side_effect=lambda _device: events.append("synchronize"),
        ),
        patch(
            "lm_resiliency.detection.layer_replay.time.perf_counter",
            side_effect=clock,
        ),
    ):
        _, elapsed_ms = detector._check_forward_backward_sdc(layer, invocation)

    assert elapsed_ms == 125.0
    assert events.index("communication") > max(
        index for index, event in enumerate(events) if event == "clock"
    )
    assert events[-1] == "synchronize"


def test_fsdp_gradient_replay_uses_padded_layer_volume_and_reduction_dtype():
    layer = nn.Linear(5, 3)
    layer._fsdp_state = SimpleNamespace(
        _fsdp_param_group=SimpleNamespace(
            _reduce_scatter_process_group="shard",
            _all_reduce_process_group="replicate",
            _reduce_dtype=torch.float32,
        )
    )
    gradients = [
        torch.ones(3, 5, dtype=torch.bfloat16),
        torch.ones(3, dtype=torch.bfloat16),
    ]

    def world_size(group):
        return {"shard": 2, "replicate": 4}[group]

    with (
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.is_available",
            return_value=True,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.get_world_size",
            side_effect=world_size,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.reduce_scatter_tensor"
        ) as reduce_scatter,
        patch("lm_resiliency.integrations.pytorch.gradient_replay.dist.all_reduce") as all_reduce,
    ):
        replay_fsdp_gradient_communication(layer, gradients)

    output, input_tensor = reduce_scatter.call_args.args
    # Weight: (3, 5) pads dim 0 to (4, 5) = 20. Bias: (3,) pads to (4,) = 4.
    assert input_tensor.numel() == 24
    assert output.numel() == 12
    assert input_tensor.dtype == torch.float32
    assert reduce_scatter.call_args.kwargs["group"] == "shard"
    all_reduce.assert_called_once_with(output, group="replicate")


def test_pure_fsdp_gradient_replay_skips_replication_all_reduce():
    layer = nn.Linear(4, 4, bias=False)
    layer._fsdp_state = SimpleNamespace(
        _fsdp_param_group=SimpleNamespace(
            _reduce_scatter_process_group="shard",
            _all_reduce_process_group=None,
            _reduce_dtype=None,
        )
    )

    with (
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.is_available",
            return_value=True,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.is_initialized",
            return_value=True,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.get_world_size",
            return_value=2,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.gradient_replay.dist.reduce_scatter_tensor"
        ) as reduce_scatter,
        patch("lm_resiliency.integrations.pytorch.gradient_replay.dist.all_reduce") as all_reduce,
    ):
        replay_fsdp_gradient_communication(layer, [torch.ones(4, 4)])

    reduce_scatter.assert_called_once()
    all_reduce.assert_not_called()
