"""Tests for zero-import torchrun worker adapters."""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast

import pytest

from lm_resiliency.integrations.torchrun._protocol import RestartContext
from lm_resiliency.integrations.torchrun._simple_runtime import SimpleRestartContextFile
from lm_resiliency.integrations.torchrun.worker_adapter import (
    DeepSpeedWorkerAdapter,
    MegatronWorkerAdapter,
    NativePyTorchAdapter,
    NativePyTorchDDPAdapter,
    TorchrunWorkerAdapterError,
    TorchrunWorkerContext,
    TorchTitanWorkerAdapter,
    _context_from_environment,
    _feature_options,
    _load_adapter,
    _load_config,
    configure_worker_bootstrap_environment,
)


def _context(tmp_path: Path) -> TorchrunWorkerContext:
    return TorchrunWorkerContext(
        run_id="adapter-run",
        node_id="node-a",
        local_world_size=1,
        restart_context_path=(tmp_path / "restart-context.json").resolve(),
    )


def _recovery_context(
    tmp_path: Path,
    *,
    checkpoint_source: str = "gemini",
    checkpoint_step: int = 4,
) -> TorchrunWorkerContext:
    return TorchrunWorkerContext(
        run_id="adapter-run",
        node_id="node-a",
        local_world_size=1,
        restart_context_path=(tmp_path / "restart-context.json").resolve(),
        generation=1,
        logical_node_slot=0,
        first_global_rank=0,
        checkpoint_step=checkpoint_step,
        checkpoint_id=("checkpoint-4" if checkpoint_source == "durable" else None),
        checkpoint_source=checkpoint_source,
        recovery_mode="recovery_verified",
    )


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("LM_RESILIENCY_TORCHRUN_"):
            environment.pop(name)
    return environment


def test_custom_adapter_bootstraps_user_script_without_lm_imports(
    tmp_path: Path,
) -> None:
    marker = tmp_path / "adapter.json"
    success = tmp_path / "success.txt"
    adapter_module = tmp_path / "custom_adapter.py"
    adapter_module.write_text(
        textwrap.dedent(
            """
            import json
            import os
            from pathlib import Path


            class Adapter:
                def install(self, context):
                    Path(os.environ["ADAPTER_MARKER"]).write_text(
                        json.dumps(
                            {
                                "run_id": context.run_id,
                                "node_id": context.node_id,
                                "generation": context.generation,
                            },
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )


            def create(context):
                return Adapter()
            """
        ),
        encoding="utf-8",
    )
    user_script = tmp_path / "user_train.py"
    user_source = textwrap.dedent(
        """
        import os
        from pathlib import Path

        marker = Path(os.environ["ADAPTER_MARKER"])
        if not marker.exists():
            raise RuntimeError("adapter did not run before user code")
        Path(os.environ["USER_SUCCESS"]).write_text("ok", encoding="utf-8")
        """
    )
    assert "lm_resiliency" not in user_source
    user_script.write_text(user_source, encoding="utf-8")
    environment = _clean_environment()
    environment["ADAPTER_MARKER"] = str(marker)
    environment["USER_SUCCESS"] = str(success)
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[2])])
    configure_worker_bootstrap_environment(
        adapter_spec="custom_adapter:create",
        context=_context(tmp_path),
        environment=environment,
    )

    completed = subprocess.run(
        [sys.executable, str(user_script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert success.read_text(encoding="utf-8") == "ok"
    assert json.loads(marker.read_text(encoding="utf-8")) == {
        "generation": 0,
        "node_id": "node-a",
        "run_id": "adapter-run",
    }


def test_worker_bootstrap_fails_closed_before_user_code(tmp_path: Path) -> None:
    adapter_module = tmp_path / "broken_adapter.py"
    adapter_module.write_text(
        "def create(context):\n    raise RuntimeError('broken adapter')\n",
        encoding="utf-8",
    )
    user_script = tmp_path / "user_train.py"
    user_script.write_text(
        "raise RuntimeError('user code must not execute')\n",
        encoding="utf-8",
    )
    environment = _clean_environment()
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[2])])
    configure_worker_bootstrap_environment(
        adapter_spec="broken_adapter:create",
        context=_context(tmp_path),
        environment=environment,
    )

    completed = subprocess.run(
        [sys.executable, str(user_script)],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 78
    assert "broken adapter" in completed.stderr
    assert "user code must not execute" not in completed.stderr


def test_worker_bootstrap_environment_conflict_is_transactional(tmp_path: Path) -> None:
    environment = _clean_environment()
    environment["LM_RESILIENCY_TORCHRUN_NODE_ID"] = "node-b"
    before = dict(environment)

    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="LM_RESILIENCY_TORCHRUN_NODE_ID",
    ):
        configure_worker_bootstrap_environment(
            adapter_spec="pytorch",
            context=_context(tmp_path),
            environment=environment,
        )

    assert environment == before


@pytest.fixture
def process_group(tmp_path: Path) -> Generator[None, None, None]:
    import torch.distributed as dist

    rendezvous = tmp_path / "ddp-rendezvous"
    dist.init_process_group(
        "gloo",
        init_method=f"file://{rendezvous}",
        rank=0,
        world_size=1,
    )
    try:
        yield
    finally:
        dist.destroy_process_group()


def test_native_ddp_adapter_attaches_before_first_forward(
    tmp_path: Path,
    process_group: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    calls: list[tuple[Any, Any, dict[str, Any]]] = []
    sentinel = object()

    def fake_enable(model: Any, optimizer: Any, **kwargs: Any) -> object:
        calls.append((model, optimizer, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        fake_enable,
    )
    adapter = NativePyTorchDDPAdapter(
        {
            "interval": 3,
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        model = DistributedDataParallel(torch.nn.Linear(4, 2))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model(torch.ones(1, 4))
        model(torch.ones(1, 4))

        assert adapter.attached
        assert adapter.handle is sentinel
        assert os.environ["LM_RESILIENCY_TORCHRUN_ADAPTER_ATTACHED"] == "1"
        assert calls == [
            (
                model,
                optimizer,
                {
                    "checkpoint": None,
                    "enable_checkpoint": False,
                    "enable_detection": False,
                    "interval": 3,
                    "recovery_mode": None,
                    "replay": None,
                },
            )
        ]
    finally:
        adapter.close()


def test_native_ddp_adapter_rejects_ambiguous_optimizers(
    tmp_path: Path,
    process_group: None,
) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    adapter = NativePyTorchDDPAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = DistributedDataParallel(torch.nn.Linear(4, 2))
        optimizer_a = torch.optim.SGD(model.parameters(), lr=0.1)
        optimizer_b = torch.optim.AdamW(model.parameters(), lr=0.1)

        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="multiple optimizers",
        ):
            model(torch.ones(1, 4))
        assert optimizer_a is not optimizer_b
    finally:
        adapter.close()


def test_native_ddp_adapter_rejects_optimizer_with_foreign_parameters(
    tmp_path: Path,
    process_group: None,
) -> None:
    import torch
    from torch.nn.parallel import DistributedDataParallel

    adapter = NativePyTorchDDPAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = DistributedDataParallel(torch.nn.Linear(4, 2))
        foreign = torch.nn.Parameter(torch.ones(1))
        optimizer = torch.optim.SGD([*model.parameters(), foreign], lr=0.1)

        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="no optimizer owns exactly one model",
        ):
            model(torch.ones(1, 4))
        assert optimizer.param_groups
    finally:
        adapter.close()


def test_native_pytorch_adapter_delegates_plain_root_module_without_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    calls: list[tuple[Any, Any, dict[str, Any]]] = []
    sentinel = object()

    def fake_enable(model: Any, optimizer: Any, **kwargs: Any) -> object:
        calls.append((model, optimizer, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        fake_enable,
    )
    adapter = NativePyTorchAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        model(torch.ones(1, 4))

        assert calls[0][0] is model
        assert calls[0][1] is optimizer
        assert "parallelism_info" not in calls[0][2]
        assert adapter.handle is sentinel
    finally:
        adapter.close()


def test_native_pytorch_disabled_policy_is_a_distributed_noop(
    tmp_path: Path,
    process_group: None,
) -> None:
    import torch

    adapter = NativePyTorchAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    handle: Any | None = None
    try:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        loss = model(torch.ones(1, 4)).sum()
        loss.backward()
        optimizer.step()

        handle = adapter.handle
        assert adapter.attached
        assert handle is not None
        assert handle.ckpt_manager is None
        assert handle.replay_harness is None
        assert handle.step_count == 1
    finally:
        if handle is not None:
            handle.close()
        adapter.close()


def test_native_pytorch_adapter_requires_exact_manager_selected_step(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    closed: list[bool] = []
    handle = SimpleNamespace(
        recovered_step=3,
        close=lambda: closed.append(True),
    )
    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        lambda *_args, **_kwargs: handle,
    )
    adapter = NativePyTorchAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_recovery_context(tmp_path, checkpoint_step=4))
    try:
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="recovered step 3, expected manager-selected step 4",
        ):
            model(torch.ones(1, 4))

        assert optimizer.param_groups
        assert closed == [True]
        assert not adapter.attached
    finally:
        adapter.close()


def test_builtin_adapter_rejects_durable_restart_context(tmp_path: Path) -> None:
    adapter = NativePyTorchAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )

    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="durable recovery requires a custom worker adapter",
    ):
        adapter.install(_recovery_context(tmp_path, checkpoint_source="durable"))


def test_torchtitan_adapter_delegates_exact_trainer_without_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("torchtitan")
    package.__path__ = []
    train_module = types.ModuleType("torchtitan.train")

    class Trainer:
        def train(self) -> str:
            return "trained"

    train_module.Trainer = Trainer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "torchtitan", package)
    monkeypatch.setitem(sys.modules, "torchtitan.train", train_module)
    calls: list[tuple[Any, dict[str, Any]]] = []
    sentinel = object()

    def fake_enable(trainer: Any, **kwargs: Any) -> object:
        calls.append((trainer, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.torchtitan.enable_resiliency",
        fake_enable,
    )
    adapter = TorchTitanWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    trainer = Trainer()
    try:
        assert trainer.train() == "trained"
        assert calls[0][0] is trainer
        assert "parallelism_info" not in calls[0][1]
        assert adapter.handle is sentinel
    finally:
        adapter.close()


def test_megatron_adapter_delegates_exact_setup_result_without_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("megatron")
    package.__path__ = []
    training_package = types.ModuleType("megatron.training")
    training_package.__path__ = []
    training_module = types.ModuleType("megatron.training.training")
    model = [object(), object()]
    optimizer = object()
    scheduler = object()
    arguments = SimpleNamespace(iteration=0)
    training_module.setup_model_and_optimizer = lambda: (  # type: ignore[attr-defined]
        model,
        optimizer,
        scheduler,
    )
    training_module.train = lambda **_kwargs: "trained"  # type: ignore[attr-defined]
    training_module.get_args = lambda: arguments  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "megatron", package)
    monkeypatch.setitem(sys.modules, "megatron.training", training_package)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_module)
    calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    sentinel = SimpleNamespace(step_count=17)

    def fake_enable(
        passed_model: Any,
        passed_optimizer: Any,
        passed_scheduler: Any,
        **kwargs: Any,
    ) -> Any:
        calls.append((passed_model, passed_optimizer, passed_scheduler, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.megatron.training.enable_resiliency",
        fake_enable,
    )
    adapter = MegatronWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        result = training_module.setup_model_and_optimizer()
        assert result == (model, optimizer, scheduler)
        assert calls[0][0] is model
        assert calls[0][1] is optimizer
        assert calls[0][2] is scheduler
        assert "parallelism_info" not in calls[0][3]
        assert arguments.iteration == 17
        assert adapter.handle is sentinel
    finally:
        adapter.close()


def test_megatron_adapter_delegates_manual_train_objects_without_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("megatron")
    package.__path__ = []
    training_package = types.ModuleType("megatron.training")
    training_package.__path__ = []
    training_module = types.ModuleType("megatron.training.training")
    model = [object(), object()]
    optimizer = object()
    scheduler = object()
    arguments = SimpleNamespace(iteration=0)
    training_module.setup_model_and_optimizer = lambda: (  # type: ignore[attr-defined]
        model,
        optimizer,
        scheduler,
    )

    def train(
        forward_step_func: Any,
        model: Any,
        optimizer: Any,
        opt_param_scheduler: Any,
    ) -> Any:
        return (forward_step_func, model, optimizer, opt_param_scheduler)

    training_module.train = train  # type: ignore[attr-defined]
    training_module.get_args = lambda: arguments  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "megatron", package)
    monkeypatch.setitem(sys.modules, "megatron.training", training_package)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_module)
    calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    sentinel = SimpleNamespace(step_count=19)

    def fake_enable(
        passed_model: Any,
        passed_optimizer: Any,
        passed_scheduler: Any,
        **kwargs: Any,
    ) -> Any:
        calls.append((passed_model, passed_optimizer, passed_scheduler, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.megatron.training.enable_resiliency",
        fake_enable,
    )
    adapter = MegatronWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    forward_step = object()
    try:
        result = training_module.train(
            forward_step_func=forward_step,
            model=model,
            optimizer=optimizer,
            opt_param_scheduler=scheduler,
        )
        assert result == (forward_step, model, optimizer, scheduler)
        assert calls[0][0] is model
        assert calls[0][1] is optimizer
        assert calls[0][2] is scheduler
        assert "parallelism_info" not in calls[0][3]
        assert arguments.iteration == 19
        assert adapter.handle is sentinel
    finally:
        adapter.close()


def test_deepspeed_adapter_delegates_exact_engine_without_parallelism(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    engine = object()
    optimizer = object()
    dataloader = object()
    scheduler = object()
    deepspeed.initialize = lambda: (  # type: ignore[attr-defined]
        engine,
        optimizer,
        dataloader,
        scheduler,
    )
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    calls: list[tuple[Any, dict[str, Any]]] = []
    sentinel = object()

    def fake_enable(passed_engine: Any, **kwargs: Any) -> object:
        calls.append((passed_engine, kwargs))
        return sentinel

    monkeypatch.setattr(
        "lm_resiliency.integrations.deepspeed.training.enable_resiliency",
        fake_enable,
    )
    adapter = DeepSpeedWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        result = deepspeed.initialize()
        assert result == (engine, optimizer, dataloader, scheduler)
        assert calls[0][0] is engine
        assert "parallelism_info" not in calls[0][1]
        assert adapter.handle is sentinel
    finally:
        adapter.close()


@pytest.mark.parametrize(
    ("name", "expected_type"),
    [
        ("pytorch", NativePyTorchAdapter),
        ("pytorch_ddp", NativePyTorchAdapter),
        ("torchtitan", TorchTitanWorkerAdapter),
        ("megatron", MegatronWorkerAdapter),
        ("deepspeed", DeepSpeedWorkerAdapter),
    ],
)
def test_builtin_adapter_names(
    tmp_path: Path,
    name: str,
    expected_type: type[Any],
) -> None:
    assert isinstance(_load_adapter(name, _context(tmp_path)), expected_type)


def test_builtin_adapter_construction_does_not_import_optional_frameworks(
    tmp_path: Path,
) -> None:
    optional_roots = {"torchtitan", "megatron", "deepspeed"}
    before = set(sys.modules)

    for name in ("torchtitan", "megatron", "deepspeed"):
        _load_adapter(name, _context(tmp_path))

    imported = {module_name.split(".", 1)[0] for module_name in set(sys.modules) - before}
    assert imported.isdisjoint(optional_roots)


def test_megatron_adapter_rejects_incomplete_setup_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = types.ModuleType("megatron")
    package.__path__ = []
    training_package = types.ModuleType("megatron.training")
    training_package.__path__ = []
    training_module = types.ModuleType("megatron.training.training")
    setup_model_and_optimizer = cast(Any, lambda: ([], object()))
    setattr(training_module, "setup_model_and_optimizer", setup_model_and_optimizer)
    setattr(training_module, "train", lambda: None)
    monkeypatch.setitem(sys.modules, "megatron", package)
    monkeypatch.setitem(sys.modules, "megatron.training", training_package)
    monkeypatch.setitem(sys.modules, "megatron.training.training", training_module)
    adapter = MegatronWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="must return \\(model, optimizer, scheduler\\)",
        ):
            cast(Any, getattr(training_module, "setup_model_and_optimizer"))()
    finally:
        adapter.close()


def test_deepspeed_adapter_rejects_invalid_initialize_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    initialize = cast(Any, lambda: object())
    setattr(deepspeed, "initialize", initialize)
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    adapter = DeepSpeedWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="must return a tuple beginning with the engine",
        ):
            cast(Any, getattr(deepspeed, "initialize"))()
    finally:
        adapter.close()


def test_worker_config_is_strict(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    config.write_text(
        "schema_version = 1.0\nunknown = true\n",
        encoding="utf-8",
    )

    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="unknown worker config fields",
    ):
        _load_config(config)

    config.write_text("schema_version = 1.0\n", encoding="utf-8")
    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="schema_version must be integer 1",
    ):
        _load_config(config)

    config.write_text("enable_checkpoint = false\n", encoding="utf-8")
    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="requires schema_version",
    ):
        _load_config(config)


@pytest.mark.parametrize("section", ["checkpoint", "replay"])
def test_disabled_worker_features_still_validate_their_sections(
    tmp_path: Path,
    section: str,
) -> None:
    payload = {
        "schema_version": 1,
        "enable_checkpoint": False,
        "enable_detection": False,
        section: {"unknown": True},
    }

    with pytest.raises(
        TorchrunWorkerAdapterError,
        match=f"unknown {section} fields",
    ):
        _feature_options(payload, _context(tmp_path))


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("run_id", "another-run", "another run"),
        ("node_id", "node-b", "another node"),
        ("local_world_size", 2, "changes local_world_size"),
        ("generation", 0, "generation must be positive"),
    ],
)
def test_worker_context_rejects_stale_restart_handoff(
    tmp_path: Path,
    field: str,
    value: Any,
    message: str,
) -> None:
    context_path = (tmp_path / "restart-context.json").resolve()
    restart = RestartContext(
        plan_id="plan-1",
        run_id="adapter-run",
        generation=1,
        node_id="node-a",
        logical_node_slot=0,
        first_global_rank=0,
        local_world_size=1,
        expected_world_size=1,
        topology_digest="topology",
        recovery_mode="recovery_verified",
        checkpoint_source="gemini",
        checkpoint_step=4,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-4",
        reason_code="attributed_sdc",
    )
    environment = {
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LM_RESILIENCY_TORCHRUN_LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
    }
    if field == "generation":
        restart = replace(restart, generation=value)
    else:
        environment[f"LM_RESILIENCY_TORCHRUN_{field.upper()}"] = str(value)
    SimpleRestartContextFile(context_path).write(restart)

    with pytest.raises(TorchrunWorkerAdapterError, match=message):
        _context_from_environment(environment)


def test_torchrun_package_exports_stable_worker_adapter_api() -> None:
    from lm_resiliency.integrations import torchrun

    expected = {
        "DeepSpeedWorkerAdapter",
        "MegatronWorkerAdapter",
        "NativePyTorchAdapter",
        "NativePyTorchDDPAdapter",
        "TorchTitanWorkerAdapter",
        "TorchrunWorkerAdapter",
        "TorchrunWorkerAdapterError",
        "TorchrunWorkerContext",
        "get_rendezvous_handler_creator",
    }

    assert set(torchrun.__all__) == expected
    for symbol in expected:
        assert getattr(torchrun, symbol) is not None


@pytest.mark.parametrize(
    "module_name",
    [
        "deepspeed.py",
        "megatron.py",
        "pytorch.py",
        "torchtitan.py",
        "torchrun_user_train.py",
    ],
)
def test_user_training_examples_have_no_lm_resiliency_imports(
    module_name: str,
) -> None:
    source_path = Path(__file__).parents[2] / "examples" / "production_loops" / module_name
    tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
    imported_modules = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_modules.update(
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )

    assert not any(
        module == "lm_resiliency" or module.startswith("lm_resiliency.")
        for module in imported_modules
    )


def test_checked_in_worker_policies_are_valid(tmp_path: Path) -> None:
    root = Path(__file__).parents[2] / "examples" / "production_loops"

    production = _load_config(root / "worker_resiliency.toml")
    checkpoint, replay, options = _feature_options(production, _context(tmp_path))
    assert options == {
        "enable_checkpoint": True,
        "enable_detection": True,
        "interval": 1,
    }
    assert checkpoint is not None
    assert checkpoint.replication_jump == 4
    assert checkpoint.disk_flush_interval == 0
    assert replay is not None
    assert replay.rotate_layers is False

    smoke = _load_config(root / "torchrun_worker_smoke.toml")
    checkpoint, replay, options = _feature_options(smoke, _context(tmp_path))
    assert options["enable_checkpoint"] is False
    assert options["enable_detection"] is False
    assert checkpoint is None
    assert replay is None
