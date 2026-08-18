"""Unit tests for native PyTorch DDP, FSDP2, and HSDP integration."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from lm_resiliency.cadence import ResiliencyCadence
from lm_resiliency.checkpointing.config import InMemoryCkptConfig
from lm_resiliency.checkpointing.rng import RNG_KEY
from lm_resiliency.detection.layer_replay import ReplayResult
from lm_resiliency.detection.replay_harness import ReplayHarnessConfig
from lm_resiliency.detection.temporal import SCOUT_TEMPORAL_KEY
from lm_resiliency.integrations._common import recover_with_fallback
from lm_resiliency.integrations.pytorch import enable_resiliency
from lm_resiliency.integrations.pytorch.fsdp import (
    PyTorchFSDPResiliency,
    _effective_replay_config,
    _fsdp_communication_ranks,
    _is_hsdp_model,
    _materialize_pure_fsdp_evidence,
    _prepare_root_managed_fsdp_invocation,
    infer_parallelism_info,
)
from lm_resiliency.integrations.pytorch.gradient_replay import (
    replay_fsdp_gradient_communication,
)


class _Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList([nn.Linear(4, 4), nn.Linear(4, 4)])

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            value = layer(value)
        return value


class _Parallelism:
    def __init__(self, dp_replicate: int, dp_shard: int, cp: int = 1) -> None:
        self.dp_replicate = dp_replicate
        self.dp_shard = dp_shard
        self.cp = cp


def _replay_result(sdc_bitmap: list[int]) -> ReplayResult:
    return ReplayResult(
        sdc_bitmap=sdc_bitmap,
        straggler_bitmap=[0] * len(sdc_bitmap),
        replay_time_ms=1.0,
        layer_id=0,
        peer_ranks=list(range(len(sdc_bitmap))),
        sdc_source_bitmaps={"output": [0] * len(sdc_bitmap)},
    )


def test_pure_fsdp_disables_optimizer_recipe_without_replica_oracle():
    config = ReplayHarnessConfig(
        check_interval=3,
        compare_parameter_state=True,
        optimizer_check_interval=None,
    )

    effective = _effective_replay_config(
        config,
        has_fsdp=True,
        is_hsdp=False,
        compare_updated_weights=False,
    )

    assert effective.compare_parameter_state is False
    assert effective.optimizer_check_interval == 0
    assert effective.hidden_check_interval is None


def test_replicated_and_ddp_models_use_common_feature_wiring():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sentinel = MagicMock()
    gloo_group = object()
    nccl_group = object()

    with (
        patch(
            "lm_resiliency.integrations.pytorch.has_dtensor_params",
            return_value=False,
        ),
        patch(
            "lm_resiliency.integrations.pytorch._wire_features",
            return_value=sentinel,
        ) as wire_features,
    ):
        result = enable_resiliency(
            model,
            optimizer,
            interval=7,
            group=gloo_group,
            nccl_group=nccl_group,
        )

    assert result is sentinel
    assert wire_features.call_args.args == (model, optimizer)
    assert wire_features.call_args.kwargs["checkpoint"].interval == 7
    assert wire_features.call_args.kwargs["replay"].check_interval == 7
    assert wire_features.call_args.kwargs["group"] is gloo_group
    assert wire_features.call_args.kwargs["nccl_group"] is nccl_group


def test_dtensor_model_uses_sharded_runtime():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sentinel = MagicMock()
    replay = ReplayHarnessConfig(check_interval=3)

    with (
        patch(
            "lm_resiliency.integrations.pytorch.has_dtensor_params",
            return_value=True,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.enable_fsdp2_resiliency",
            return_value=sentinel,
        ) as enable_fsdp2,
    ):
        result = enable_resiliency(
            model,
            optimizer,
            interval=7,
            enable_checkpoint=False,
            replay=replay,
            device=torch.device("cpu"),
        )

    assert result is sentinel
    assert enable_fsdp2.call_args.kwargs["ckpt_config"] is None
    assert enable_fsdp2.call_args.kwargs["detection_config"].check_interval == 7
    assert enable_fsdp2.call_args.kwargs["device"] == torch.device("cpu")


def test_native_pytorch_registers_automatic_cleanup():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    handle = MagicMock()

    with (
        patch(
            "lm_resiliency.integrations.pytorch.has_dtensor_params",
            return_value=False,
        ),
        patch(
            "lm_resiliency.integrations.pytorch._wire_features",
            return_value=handle,
        ),
        patch(
            "lm_resiliency.integrations.pytorch.register_automatic_cleanup",
        ) as register_cleanup,
    ):
        result = enable_resiliency(model, optimizer)

    assert result is handle
    register_cleanup.assert_called_once_with(handle)


@pytest.mark.parametrize("missing", ["group", "nccl_group"])
def test_native_pytorch_rejects_one_detection_group(missing: str):
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    kwargs = {"group": object(), "nccl_group": object()}
    kwargs[missing] = None

    with pytest.raises(ValueError, match="must be supplied together"):
        enable_resiliency(
            model,
            optimizer,
            enable_checkpoint=False,
            **kwargs,
        )


def test_hsdp_is_inferred_from_named_parameter_device_mesh():
    class Mesh:
        mesh_dim_names = ("dp_replicate", "dp_shard")
        mesh = torch.empty(4, 2)

        def size(self, index):
            return (4, 2)[index]

    model = _Model()
    next(model.parameters()).device_mesh = Mesh()

    info = infer_parallelism_info(model)
    assert (info.dp_replicate, info.dp_shard) == (4, 2)
    assert _is_hsdp_model(model, parallelism_info=None)


def test_pure_fsdp_is_inferred_from_unnamed_one_dimensional_mesh():
    class Mesh:
        mesh_dim_names = None
        mesh = torch.empty(8)

        def size(self, index):
            assert index == 0
            return 8

    model = _Model()
    next(model.parameters()).device_mesh = Mesh()

    info = infer_parallelism_info(model)
    assert (info.dp_replicate, info.dp_shard) == (1, 8)
    assert not _is_hsdp_model(model, parallelism_info=None)


def test_explicit_parallelism_metadata_takes_precedence():
    model = _Model()
    explicit = _Parallelism(dp_replicate=3, dp_shard=2)

    assert infer_parallelism_info(model, explicit) is explicit
    assert _is_hsdp_model(model, explicit)


def test_torchtitan_context_parallelism_contributes_to_fsdp_degree():
    model = _Model()
    parallelism = _Parallelism(dp_replicate=2, dp_shard=1, cp=2)

    assert _is_hsdp_model(model, parallelism)


def test_fsdp2_replay_disables_replica_only_comparisons():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = MagicMock()
    harness = MagicMock()
    harness.target_layer = model.layers[0]
    harness.step.return_value = _replay_result([0, 1, 0, 0])

    with patch(
        "lm_resiliency.integrations.pytorch.fsdp.ModelReplayHarness",
        return_value=harness,
    ) as harness_cls:
        resiliency = PyTorchFSDPResiliency(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=1),
            device=torch.device("cpu"),
            fault_callback=callback,
            group="gloo",
            nccl_group="nccl",
            parallelism_info=_Parallelism(dp_replicate=1, dp_shard=4),
        )
        optimizer.step()

    config = harness_cls.call_args.kwargs["config"]
    assert config.compare_parameter_state is False
    assert (
        harness_cls.call_args.kwargs["gradient_communication"] is replay_fsdp_gradient_communication
    )
    assert harness_cls.call_args.kwargs["evidence_preparer"] is _materialize_pure_fsdp_evidence
    harness.check_local_parameter_shards.assert_not_called()
    harness.step.assert_called_once_with()
    callback.assert_called_once_with(harness.step.return_value)
    resiliency.close()


def test_hsdp_replay_compares_materialized_parameters():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    harness = MagicMock()
    harness.target_layer = model.layers[0]

    with patch(
        "lm_resiliency.integrations.pytorch.fsdp.ModelReplayHarness",
        return_value=harness,
    ) as harness_cls:
        resiliency = PyTorchFSDPResiliency(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=1),
            device=torch.device("cpu"),
            group="gloo",
            nccl_group="nccl",
            parallelism_info=_Parallelism(dp_replicate=2, dp_shard=2),
        )

    assert harness_cls.call_args.kwargs["config"].compare_parameter_state is True
    assert (
        harness_cls.call_args.kwargs["invocation_preparer"] is _prepare_root_managed_fsdp_invocation
    )
    assert harness_cls.call_args.kwargs["evidence_preparer"] is None
    resiliency.close()


def test_pure_fsdp_evidence_materializes_global_dtensor_values():
    class DTensor:
        def __init__(self, full: torch.Tensor) -> None:
            self.full = full
            self.calls = 0

        def full_tensor(self) -> torch.Tensor:
            self.calls += 1
            return self.full

    distributed = DTensor(torch.arange(4))
    local = torch.ones(2)

    prepared = _materialize_pure_fsdp_evidence(
        {"output": [distributed, local]},
    )

    assert prepared == {"output": [distributed.full, local]}
    assert distributed.calls == 1


def test_fsdp2_honors_recipe_specific_replay_steps():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = MagicMock()
    result = _replay_result([0, 1])
    harness = MagicMock()
    harness.target_layer = model.layers[0]
    harness.dense_boundary_modules = ()
    harness.replay_due.side_effect = [False, True]
    harness.step.side_effect = [None, result]

    with patch(
        "lm_resiliency.integrations.pytorch.fsdp.ModelReplayHarness",
        return_value=harness,
    ):
        resiliency = PyTorchFSDPResiliency(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(
                check_interval=10,
                hidden_check_interval=2,
            ),
            device=torch.device("cpu"),
            fault_callback=callback,
            group="gloo",
            nccl_group="nccl",
            parallelism_info=_Parallelism(dp_replicate=1, dp_shard=2),
        )
        optimizer.step()
        optimizer.step()

    assert harness.replay_due.call_args_list[0].args == (1,)
    assert harness.replay_due.call_args_list[1].args == (2,)
    assert harness.step.call_count == 2
    callback.assert_called_once_with(result)
    resiliency.close()


def test_fsdp2_materializes_dense_boundary_modules_for_replay():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
    )
    target = model.layers[0]
    boundary = model.layers[1]
    target.unshard = MagicMock()
    target.reshard = MagicMock()
    boundary.unshard = MagicMock()
    boundary.reshard = MagicMock()
    harness = MagicMock()
    harness.target_layer = target
    harness.dense_boundary_modules = (boundary,)
    resiliency.replay_harness = harness
    result = _replay_result([0])

    observed = resiliency._run_with_unsharded_model(lambda: result)

    assert observed is result
    target.unshard.assert_called_once()
    boundary.unshard.assert_called_once()
    target.reshard.assert_called_once()
    boundary.reshard.assert_called_once()
    harness.add_communication_timing.assert_called_once()
    timing_call = harness.add_communication_timing.call_args
    assert timing_call.args == (result,)
    assert timing_call.kwargs["name"] == "fsdp_parameter_all_gather"
    assert timing_call.kwargs["elapsed_ms"] >= 0.0
    resiliency.close()


def test_fsdp_communication_timing_uses_the_state_shard_group():
    mesh = MagicMock()
    mesh.mesh_dim_names = ("dp_replicate", "dp_shard", "tp")
    shard_group = object()
    mesh.get_group.return_value = shard_group
    parameter = MagicMock()
    parameter.device_mesh = mesh
    module = MagicMock()
    module.parameters.return_value = [parameter]
    module.modules.return_value = []

    with patch(
        "lm_resiliency.integrations.pytorch.fsdp.dist.get_process_group_ranks",
        return_value=[4, 5],
    ):
        ranks = _fsdp_communication_ranks(module, None)

    assert ranks == (4, 5)
    mesh.get_group.assert_called_once_with("dp_shard")


def test_hsdp_checks_local_shards_and_updated_weights():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = MagicMock()
    harness = MagicMock()
    harness.target_layer = model.layers[0]
    harness.check_local_parameter_shards.return_value = [0, 1]
    harness.step.return_value = _replay_result([0, 0])

    with patch(
        "lm_resiliency.integrations.pytorch.fsdp.ModelReplayHarness",
        return_value=harness,
    ):
        resiliency = PyTorchFSDPResiliency(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=1),
            device=torch.device("cpu"),
            fault_callback=callback,
            group="gloo",
            nccl_group="nccl",
            parallelism_info=_Parallelism(dp_replicate=2, dp_shard=2),
        )
        optimizer.step()

    harness.check_local_parameter_shards.assert_called_once()
    harness.step.assert_called_once_with(
        optimizer=optimizer,
        allow_local_dtensor_shards=True,
    )
    result = callback.call_args.args[0]
    assert result.sdc_bitmap == [0, 1]
    assert result.sdc_source_bitmaps["local_parameter_shard"] == [0, 1]
    resiliency.close()


def test_hsdp_reuses_replay_group_for_local_shard_consensus():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    group = object()
    harness = MagicMock()
    harness.target_layer = model.layers[0]
    local_shard_result = MagicMock(bitmap=[0, 1, 0, 0])

    with (
        patch(
            "lm_resiliency.integrations.pytorch.fsdp.ModelReplayHarness",
            return_value=harness,
        ),
        patch("lm_resiliency.integrations.pytorch.fsdp.C3") as c3_cls,
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=4),
    ):
        resiliency = PyTorchFSDPResiliency(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=1),
            device=torch.device("cpu"),
            group=group,
            nccl_group=object(),
            parallelism_info=_Parallelism(dp_replicate=4, dp_shard=2),
        )

    c3_cls.return_value.run_tensor_sequence.return_value = local_shard_result
    c3_cls.assert_called_once_with(group=group)
    assert resiliency._check_local_parameter_shards(model.layers[0]) == [0, 1, 0, 0]
    c3_cls.return_value.run_tensor_sequence.assert_called_once()
    resiliency.close()


def test_hsdp_gemini_skips_explicit_peer_replication(tmp_path):
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(
            interval=1,
            disk_folder=str(tmp_path),
            pin_memory=False,
        ),
        device=torch.device("cpu"),
        parallelism_info=_Parallelism(dp_replicate=2, dp_shard=2),
    )

    assert resiliency.ckpt_manager is not None
    assert resiliency.ckpt_manager._skip_replication is True
    assert resiliency.ckpt_manager._replicator.enabled is False
    resiliency.close()


def test_empty_recovery_does_not_materialize_optimizer_state():
    model = _Model()
    optimizer = torch.optim.AdamW(model.parameters())
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
    )
    manager = MagicMock()
    manager.load_tensors.return_value = None
    resiliency.ckpt_manager = manager

    assert resiliency.try_recover() == -1
    assert not optimizer.state
    resiliency.close()


def test_recovery_restores_tensors_extra_state_and_step():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    restored_extra = {}
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
        load_extra_state_fn=restored_extra.update,
    )
    saved = [torch.full_like(parameter, 3.0) for parameter in model.parameters()]
    manager = MagicMock()
    manager.load_tensors.return_value = (
        saved,
        9,
        {
            "scheduler": {"last_epoch": 8},
            RNG_KEY: None,
            SCOUT_TEMPORAL_KEY: None,
        },
    )
    resiliency.ckpt_manager = manager

    assert resiliency.try_recover() == 9
    assert resiliency.recovered_step == 9
    assert resiliency.step_count == 9
    assert restored_extra == {"scheduler": {"last_epoch": 8}}
    assert all(
        torch.equal(parameter, expected) for parameter, expected in zip(model.parameters(), saved)
    )
    resiliency.close()


@pytest.mark.parametrize(
    "saved",
    [
        [torch.zeros(1)],
        [torch.zeros(3, 3)] * 4,
        [torch.zeros(4, 4, dtype=torch.float64)] * 4,
    ],
)
def test_recovery_rejects_tensor_layout_mismatch(saved: list[torch.Tensor]):
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
    )
    manager = MagicMock()
    manager.load_tensors.return_value = (saved, 4, None)
    resiliency.ckpt_manager = manager

    with pytest.raises(RuntimeError, match="tensor layout does not match"):
        resiliency.try_recover()
    resiliency.close()


def test_public_durable_recovery_updates_common_handle_step():
    handle = MagicMock()
    handle.try_recover.return_value = -1
    handle.durable_checkpoint.load_latest_validated.return_value = 11

    recover_with_fallback(handle, load_fallback=MagicMock())

    assert handle.step_count == 11


def test_sharded_handle_lifecycle_is_idempotent():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    callback = MagicMock()
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
    )
    resiliency.add_close_callback(callback)

    optimizer.step()
    assert resiliency.step_count == 1
    resiliency.close()
    resiliency.close()

    assert resiliency.closed
    callback.assert_called_once()


def test_checkpoint_sdc_skips_capture():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    resiliency = PyTorchFSDPResiliency(
        model,
        optimizer,
        ckpt_config=InMemoryCkptConfig(enable=False),
        device=torch.device("cpu"),
    )
    manager = MagicMock()
    resiliency.ckpt_manager = manager
    resiliency._cadence = ResiliencyCadence(
        interval=1,
        checkpoint_enabled=True,
        detection_enabled=True,
    )
    harness = MagicMock()
    harness.target_layer = model.layers[0]
    harness.step.return_value = _replay_result([1])
    resiliency.replay_harness = harness

    optimizer.step()

    manager.save_tensors.assert_not_called()
    resiliency.close()


def test_torchtitan_entry_point_remains_a_compatibility_wrapper():
    model = _Model()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    sentinel = object()

    with patch(
        "lm_resiliency.integrations.torchtitan._enable_pytorch_resiliency",
        return_value=sentinel,
    ) as enable_pytorch:
        from lm_resiliency.integrations.torchtitan import (
            enable_resiliency as enable_torchtitan,
        )

        result = enable_torchtitan(
            model,
            optimizer,
            ckpt_config=InMemoryCkptConfig(enable=False),
            detection_config=ReplayHarnessConfig(check_interval=5),
        )

    assert result is sentinel
    assert enable_pytorch.call_args.args == (model, optimizer)
    assert enable_pytorch.call_args.kwargs["checkpoint"].enable is False
    assert enable_pytorch.call_args.kwargs["replay"].check_interval == 5
