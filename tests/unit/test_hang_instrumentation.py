"""Training-side OOB progress instrumentation tests."""

from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.nn as nn

from lm_resiliency.detection.hang_instrumentation import (
    HangInstrumentation,
    collective_metadata_fingerprint,
)


def test_layer_and_collective_boundaries_advance_tracker():
    layers = nn.ModuleList([nn.Linear(4, 4), nn.ReLU()])
    model = nn.Sequential(*layers)

    with patch.object(dist, "all_reduce", return_value=None) as collective:
        instrumentation = HangInstrumentation(model, layers, rank=197)
        try:
            model(torch.ones(2, 4))
            after_layers = instrumentation.tracker.op_id
            assert after_layers == 4

            pending_fingerprints = []
            collective.side_effect = lambda *_args, **_kwargs: pending_fingerprints.append(
                instrumentation.tracker.metadata_fingerprint
            )
            dist.all_reduce(torch.ones(1))
            assert instrumentation.tracker.op_id == after_layers + 2
            assert pending_fingerprints[0] != 0
            assert instrumentation.tracker.metadata_fingerprint == 0
            collective.assert_called_once()
        finally:
            instrumentation.close()


def _fingerprint(
    name: str,
    *args,
    group_ranks: tuple[int, ...] = (0, 1, 2, 3),
    **kwargs,
) -> int:
    return collective_metadata_fingerprint(name, args, kwargs, group_ranks=group_ranks)


def test_fingerprint_changes_for_collective_shape_dtype_and_group():
    baseline = _fingerprint("all_reduce", torch.ones(4, dtype=torch.float32))

    assert _fingerprint("all_reduce", torch.ones(8, dtype=torch.float32)) != baseline
    assert _fingerprint("all_reduce", torch.ones(4, dtype=torch.float16)) != baseline
    assert (
        _fingerprint(
            "all_reduce",
            torch.ones(4, dtype=torch.float32),
            group_ranks=(0, 1, 4, 5),
        )
        != baseline
    )


def test_fingerprint_changes_for_collective_name_and_reduction():
    tensor = torch.ones(4)

    assert _fingerprint("all_reduce", tensor) != _fingerprint("broadcast", tensor, src=0)
    assert _fingerprint("broadcast", tensor, src=0) != _fingerprint("broadcast", tensor, src=1)
    assert _fingerprint("all_reduce", tensor) != _fingerprint(
        "all_reduce", tensor, op=dist.ReduceOp.MAX
    )


def test_dynamic_all_to_all_shapes_do_not_create_false_mismatch():
    first = _fingerprint(
        "all_to_all_single",
        torch.empty(7),
        torch.empty(7),
        output_split_sizes=[1, 2, 3, 1],
        input_split_sizes=[2, 1, 1, 3],
    )
    second = _fingerprint(
        "all_to_all_single",
        torch.empty(11),
        torch.empty(11),
        output_split_sizes=[4, 1, 2, 4],
        input_split_sizes=[3, 3, 4, 1],
    )

    assert first == second


def test_dynamic_all_to_all_accepts_tensor_split_sizes():
    tensor_splits = _fingerprint(
        "all_to_all_single",
        torch.empty(7),
        torch.empty(7),
        output_split_sizes=torch.tensor([1, 2, 3, 1]),
        input_split_sizes=torch.tensor([2, 1, 1, 3]),
    )
    list_splits = _fingerprint(
        "all_to_all_single",
        torch.empty(11),
        torch.empty(11),
        output_split_sizes=[4, 1, 2, 4],
        input_split_sizes=[3, 3, 4, 1],
    )

    assert tensor_splits == list_splits


def test_exact_all_to_all_recipe_is_retained_until_step_boundary():
    layers = nn.ModuleList([nn.Identity(), nn.Identity()])
    model = nn.Sequential(*layers)
    group = object()
    inputs = torch.empty(7, 4, dtype=torch.float16)
    outputs = torch.empty(7, 4, dtype=torch.float16)

    with (
        patch.object(dist, "all_to_all_single", return_value=None) as collective,
        patch.object(dist, "is_initialized", return_value=True),
        patch.object(dist, "get_process_group_ranks", return_value=[0, 1, 2, 3]),
    ):
        instrumentation = HangInstrumentation(model, layers, rank=198)
        try:
            dist.all_to_all_single(
                outputs,
                inputs,
                output_split_sizes=[1, 2, 3, 1],
                input_split_sizes=[2, 1, 1, 3],
                group=group,
            )

            (recipe,) = instrumentation.all_to_all_recipes
            assert recipe.sequence == 0
            assert recipe.collective == "all_to_all_single"
            assert recipe.group_ranks == (0, 1, 2, 3)
            assert recipe.input_split_sizes == (2, 1, 1, 3)
            assert recipe.output_split_sizes == (1, 2, 3, 1)
            assert recipe.async_op is False
            assert recipe.group is group
            assert recipe.inputs[0].shape == (7, 4)
            assert recipe.inputs[0].dtype == torch.float16
            assert recipe.input_bytes == 56
            assert recipe.output_bytes == 56
            collective.assert_called_once()

            equal_inputs = torch.empty(8, 4)
            equal_outputs = torch.empty(8, 4)
            dist.all_to_all_single(equal_outputs, equal_inputs, group=group)
            equal_recipe = instrumentation.all_to_all_recipes[1]
            assert equal_recipe.sequence == 1
            assert equal_recipe.input_split_sizes == (2, 2, 2, 2)
            assert equal_recipe.output_split_sizes == (2, 2, 2, 2)

            with instrumentation.suspend_all_to_all_capture():
                dist.all_to_all_single(equal_outputs, equal_inputs, group=group)
            assert len(instrumentation.all_to_all_recipes) == 2

            instrumentation.step_boundary()
            assert instrumentation.all_to_all_recipes == ()
        finally:
            instrumentation.close()
