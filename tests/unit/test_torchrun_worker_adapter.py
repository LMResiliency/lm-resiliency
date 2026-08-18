"""Tests for zero-import torchrun worker adapters."""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import subprocess
import sys
import textwrap
import threading
import types
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generator, cast
from unittest.mock import MagicMock

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
    _AutoFrameworkAdapter,
    _context_from_environment,
    _feature_options,
    _load_config,
    configure_worker_bootstrap_environment,
    configure_worker_generation_environment,
    get_torchrun_worker_context,
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
        topology_digest="topology",
        restart_deadline_unix_ms=9_999_999_999_999,
    )


def _framework_disagreement_worker(
    rank: int,
    rendezvous_path: str,
    output_dir: str,
) -> None:
    import torch.distributed as dist

    adapter = _AutoFrameworkAdapter({"enable_checkpoint": False, "enable_detection": False})
    context = TorchrunWorkerContext(
        run_id="framework-agreement",
        node_id=f"node-{rank}",
        local_world_size=1,
        restart_context_path=(Path(output_dir) / f"context-{rank}.json").resolve(),
    )
    if rank == 1:
        deepspeed = types.ModuleType("deepspeed")
        deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
        sys.modules["deepspeed"] = deepspeed
    adapter.install(context)
    try:
        __import__("torch" if rank == 0 else "deepspeed")
        try:
            dist.init_process_group(
                "gloo",
                init_method=f"file://{rendezvous_path}",
                rank=rank,
                world_size=2,
            )
        except TorchrunWorkerAdapterError as error:
            outcome = str(error)
        else:
            outcome = "framework agreement unexpectedly succeeded"
    finally:
        adapter.close()
        if dist.is_initialized():
            dist.destroy_process_group()
    (Path(output_dir) / f"framework-agreement-{rank}.txt").write_text(
        outcome,
        encoding="utf-8",
    )


def _clean_environment() -> dict[str, str]:
    environment = dict(os.environ)
    for name in tuple(environment):
        if name.startswith("LM_RESILIENCY_TORCHRUN_"):
            environment.pop(name)
    environment.pop("LM_RESILIENCY_GENERATION", None)
    environment.pop("LOCAL_WORLD_SIZE", None)
    return environment


def _configure_bootstrap(
    context: TorchrunWorkerContext,
    environment: dict[str, str],
) -> None:
    before = dict(environment)
    environment.update(
        {
            "GROUP_RANK": str(context.logical_node_slot or 0),
            "LOCAL_RANK": "0",
            "LOCAL_WORLD_SIZE": str(context.local_world_size),
            "RANK": str(context.first_global_rank or 0),
            "WORLD_SIZE": str(context.local_world_size),
        }
    )
    try:
        configure_worker_bootstrap_environment(
            run_id=context.run_id,
            node_id=context.node_id,
            restart_context_path=context.restart_context_path,
            config_path=context.config_path,
            environment=environment,
        )
        configure_worker_generation_environment(context.generation, environment)
    except BaseException:
        environment.clear()
        environment.update(before)
        raise


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
                                "runtime_generation": os.environ[
                                    "LM_RESILIENCY_GENERATION"
                                ],
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
    config = tmp_path / "worker.toml"
    config.write_text(
        'schema_version = 1\nadapter = "custom_adapter:create"\n',
        encoding="utf-8",
    )
    environment = _clean_environment()
    environment["LOCAL_WORLD_SIZE"] = "1"
    environment["ADAPTER_MARKER"] = str(marker)
    environment["USER_SUCCESS"] = str(success)
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[2])])
    _configure_bootstrap(
        replace(_context(tmp_path), config_path=config.resolve()),
        environment,
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
        "runtime_generation": "0",
    }


def test_worker_bootstrap_preserves_existing_sitecustomize(tmp_path: Path) -> None:
    marker = tmp_path / "sitecustomize.txt"
    existing = tmp_path / "existing"
    existing.mkdir()
    (existing / "sitecustomize.py").write_text(
        "from pathlib import Path\n"
        "import os\n"
        "Path(os.environ['SITE_MARKER']).write_text('loaded', encoding='utf-8')\n",
        encoding="utf-8",
    )
    config = tmp_path / "worker.toml"
    config.write_text(
        'schema_version = 1\nadapter = "custom_adapter:create"\n',
        encoding="utf-8",
    )
    (tmp_path / "custom_adapter.py").write_text(
        "class Adapter:\n"
        "    def install(self, context):\n"
        "        pass\n"
        "def create(context):\n"
        "    return Adapter()\n",
        encoding="utf-8",
    )
    environment = _clean_environment()
    environment["LOCAL_WORLD_SIZE"] = "1"
    environment["SITE_MARKER"] = str(marker)
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(existing), str(tmp_path), str(Path(__file__).parents[2])]
    )
    _configure_bootstrap(
        replace(_context(tmp_path), config_path=config.resolve()),
        environment,
    )

    completed = subprocess.run(
        [sys.executable, "-c", "pass"],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert marker.read_text(encoding="utf-8") == "loaded"


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
    config = tmp_path / "worker.toml"
    config.write_text(
        'schema_version = 1\nadapter = "broken_adapter:create"\n',
        encoding="utf-8",
    )
    environment = _clean_environment()
    environment["LOCAL_WORLD_SIZE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join([str(tmp_path), str(Path(__file__).parents[2])])
    _configure_bootstrap(
        replace(_context(tmp_path), config_path=config.resolve()),
        environment,
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
        _configure_bootstrap(_context(tmp_path), environment)

    assert environment == before


def test_worker_bootstrap_rejects_policy_changed_after_rendezvous(tmp_path: Path) -> None:
    config = tmp_path / "worker.toml"
    original = b"schema_version = 1\ninterval = 1\n"
    config.write_bytes(original)
    digest = hashlib.sha256(b"lm-resiliency/worker-policy/v1\0" + original).hexdigest()
    config.write_text("schema_version = 1\ninterval = 2\n", encoding="utf-8")

    with pytest.raises(TorchrunWorkerAdapterError, match="changed after rendezvous"):
        _load_config(config, expected_digest=digest)


def test_worker_context_uses_torchrun_local_world_size(tmp_path: Path) -> None:
    context_path = (tmp_path / "restart-context.json").resolve()
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "8",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "0",
        "RANK": "0",
        "WORLD_SIZE": "8",
    }

    context = _context_from_environment(environment)

    assert context.local_world_size == 8


def test_public_worker_context_sets_manager_generation(tmp_path: Path) -> None:
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(
            (tmp_path / "restart-context.json").resolve()
        ),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    context = get_torchrun_worker_context(environment)

    assert context.generation == 0
    assert environment["LM_RESILIENCY_GENERATION"] == "0"
    assert environment["LM_RESILIENCY_TORCHRUN_CHECKPOINT_STEP"] == "0"


def test_framework_inference_replaces_tentative_pytorch_with_deepspeed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        __import__("torch")
        assert adapter.selected_framework is None
        assert isinstance(adapter.delegate, NativePyTorchAdapter)

        __import__("deepspeed")
        assert adapter.selected_framework == "deepspeed"
        assert isinstance(adapter.delegate, DeepSpeedWorkerAdapter)
    finally:
        adapter.close()


def test_framework_inference_observes_transitive_deepspeed_import(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        adapter._run_import("user_helper", lambda: __import__("deepspeed"))

        assert adapter.selected_framework == "deepspeed"
        assert isinstance(adapter.delegate, DeepSpeedWorkerAdapter)
    finally:
        adapter.close()


def test_framework_inference_wraps_transitively_imported_alias(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    engine = object()
    deepspeed.initialize = lambda: (engine,)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    monkeypatch.setattr(
        "lm_resiliency.integrations.deepspeed.training.enable_resiliency",
        lambda *_args, **_kwargs: object(),
    )
    adapter = _AutoFrameworkAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))

    def import_helper() -> Any:
        from deepspeed import initialize

        return initialize

    try:
        initialize = adapter._run_import("user_helper", import_helper)
        initialize()
        assert isinstance(adapter.delegate, DeepSpeedWorkerAdapter)
        assert adapter.delegate.attached
    finally:
        adapter.close()


def test_failed_outer_import_does_not_select_transitive_framework(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))

    def failed_import() -> None:
        __import__("deepspeed")
        raise ImportError("helper failed")

    try:
        with pytest.raises(ImportError, match="helper failed"):
            adapter._run_import("user_helper", failed_import)
        assert adapter.selected_framework is None
        assert adapter.delegate is None
    finally:
        adapter.close()


def test_failed_import_cannot_rollback_another_thread_framework_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    first_started = threading.Event()
    release_first = threading.Event()
    second_calling = threading.Event()
    second_operation_started = threading.Event()
    second_done = threading.Event()
    errors: list[BaseException] = []

    def failed_operation() -> None:
        first_started.set()
        release_first.wait(timeout=5)
        raise ImportError("unrelated import failed")

    def first_import() -> None:
        try:
            adapter._run_import("user_helper", failed_operation)
        except ImportError:
            return
        except BaseException as error:
            errors.append(error)

    def successful_operation() -> Any:
        second_operation_started.set()
        return deepspeed

    def second_import() -> None:
        second_calling.set()
        try:
            adapter._run_import("deepspeed", successful_operation)
        except BaseException as error:
            errors.append(error)
        finally:
            second_done.set()

    first = threading.Thread(target=first_import)
    second = threading.Thread(target=second_import)
    try:
        first.start()
        assert first_started.wait(timeout=5)
        second.start()
        assert second_calling.wait(timeout=5)
        assert not second_operation_started.wait(timeout=1)
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)

        assert not first.is_alive()
        assert not second.is_alive()
        assert second_done.is_set()
        assert not errors
        assert adapter.selected_framework == "deepspeed"
        assert isinstance(adapter.delegate, DeepSpeedWorkerAdapter)
    finally:
        release_first.set()
        first.join(timeout=5)
        second.join(timeout=5)
        adapter.close()


def test_framework_inference_rejects_multiple_specialized_frameworks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deepspeed = types.ModuleType("deepspeed")
    deepspeed.initialize = lambda: (object(),)  # type: ignore[attr-defined]
    megatron = types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    try:
        __import__("deepspeed")
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="multiple supported training frameworks",
        ):
            __import__("megatron")
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="multiple supported training frameworks",
        ):
            deepspeed.initialize()
    finally:
        adapter.close()


def test_framework_inference_rejects_cross_rank_disagreement_before_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch.distributed as dist

    deepspeed = types.ModuleType("deepspeed")
    engine = object()
    deepspeed.initialize = lambda: (engine,)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    monkeypatch.setattr(dist, "init_process_group", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)

    def gather_frameworks(gathered: list[str | None], local: str | None) -> None:
        gathered[:] = [local, "pytorch"]

    monkeypatch.setattr(dist, "all_gather_object", gather_frameworks)
    enable = MagicMock()
    monkeypatch.setattr(
        "lm_resiliency.integrations.deepspeed.training.enable_resiliency",
        enable,
    )
    adapter = _AutoFrameworkAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        __import__("deepspeed")

        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="inferred different training frameworks",
        ):
            dist.init_process_group("gloo")
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="inferred different training frameworks",
        ):
            deepspeed.initialize()

        enable.assert_not_called()
    finally:
        adapter.close()


def test_framework_inference_disagreement_fails_all_gloo_ranks(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    rendezvous = tmp_path / "framework-agreement-store"
    processes = [
        context.Process(
            target=_framework_disagreement_worker,
            args=(rank, str(rendezvous), str(tmp_path)),
        )
        for rank in range(2)
    ]

    original_sys_path = list(sys.path)
    repository_root = str(Path(__file__).resolve().parents[2])
    original_python_path = os.environ.get("PYTHONPATH")
    bootstrap_environment = {
        name: os.environ.pop(name)
        for name in tuple(os.environ)
        if name.startswith("LM_RESILIENCY_TORCHRUN_")
    }
    sys.path[:] = [repository_root, *(path for path in sys.path if path != repository_root)]
    os.environ["PYTHONPATH"] = os.pathsep.join(
        path for path in (repository_root, original_python_path) if path
    )
    try:
        for process in processes:
            process.start()
    finally:
        sys.path[:] = original_sys_path
        if original_python_path is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = original_python_path
        os.environ.update(bootstrap_environment)
    for process in processes:
        process.join(timeout=30)
        if process.is_alive():
            process.terminate()
            process.join(timeout=5)

    assert [process.exitcode for process in processes] == [0, 0]
    outcomes = [
        (tmp_path / f"framework-agreement-{rank}.txt").read_text(encoding="utf-8")
        for rank in range(2)
    ]
    assert all("inferred different training frameworks" in outcome for outcome in outcomes)


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


def test_native_pytorch_adapter_rejects_multiple_single_process_training_pairs(
    tmp_path: Path,
) -> None:
    import torch

    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model_a = torch.nn.Linear(4, 2)
        model_b = torch.nn.Linear(4, 2)
        optimizer_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
        optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=0.1)

        with pytest.raises(TorchrunWorkerAdapterError, match="multiple optimizers"):
            model_a(torch.ones(1, 4))

        assert optimizer_a is not optimizer_b
        assert not adapter.attached
    finally:
        adapter.close()


def test_native_pytorch_distributed_warmup_fails_before_collective_attach(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    enable = MagicMock()
    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        enable,
    )
    monkeypatch.setattr(NativePyTorchAdapter, "_distributed_world_size", lambda _self: 2)
    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = torch.nn.Linear(4, 2)
        torch.optim.SGD(model.parameters(), lr=0.1)

        with pytest.raises(
            TorchrunWorkerAdapterError,
            match="DDP or FSDP construction boundary",
        ):
            model(torch.ones(1, 4))

        enable.assert_not_called()
    finally:
        adapter.close()


def test_native_pytorch_waits_for_outermost_optimizer_constructor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class ParentOptimizer(torch.optim.SGD):
        def __init__(self, parameters: Any) -> None:
            super().__init__(parameters, lr=0.1)
            self.parent_ready = True

    class ChildOptimizer(ParentOptimizer):
        def __init__(self, parameters: Any) -> None:
            super().__init__(parameters)
            self.child_ready = True

    attached: list[ChildOptimizer] = []

    def fake_enable(_model: Any, optimizer: ChildOptimizer, **_kwargs: Any) -> object:
        assert optimizer.parent_ready
        assert optimizer.child_ready
        attached.append(optimizer)
        return object()

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        fake_enable,
    )
    monkeypatch.setattr(NativePyTorchAdapter, "_distributed_world_size", lambda _self: 2)
    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = torch.nn.Linear(4, 2)
        adapter._register_distributed_model(model)

        optimizer = ChildOptimizer(model.parameters())

        assert attached == [optimizer]
        assert adapter.attached
    finally:
        adapter.close()


def test_native_pytorch_instruments_optimizer_defined_after_bootstrap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    attached: list[Any] = []

    def fake_enable(_model: Any, optimizer: Any, **_kwargs: Any) -> object:
        assert optimizer.ready
        attached.append(optimizer)
        return object()

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        fake_enable,
    )
    monkeypatch.setattr(NativePyTorchAdapter, "_distributed_world_size", lambda _self: 2)
    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = torch.nn.Linear(4, 2)
        adapter._register_distributed_model(model)

        class UserOptimizer(torch.optim.Optimizer):
            def __init__(self, parameters: Any) -> None:
                super().__init__(parameters, {"lr": 0.1})
                self.ready = True

            def step(self, closure: Any | None = None) -> None:
                del closure

        optimizer = UserOptimizer(model.parameters())

        assert attached == [optimizer]
        assert adapter.attached
    finally:
        adapter.close()


def test_native_pytorch_selects_outermost_bottom_up_fsdp_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = torch.nn.Linear(4, 4)
            self.second = torch.nn.Linear(4, 2)

    calls: list[tuple[Any, Any]] = []

    def fake_enable(model: Any, optimizer: Any, **_kwargs: Any) -> object:
        calls.append((model, optimizer))
        return object()

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        fake_enable,
    )
    monkeypatch.setattr(NativePyTorchAdapter, "_distributed_world_size", lambda _self: 2)
    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    try:
        model = Model()
        adapter._register_distributed_model(model.first)
        adapter._register_distributed_model(model.second)
        adapter._register_distributed_model(model)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        assert calls == [(model, optimizer)]
        assert adapter.attached
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


def test_native_adapter_closes_handle_before_process_group_teardown(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import torch.distributed as dist

    events: list[str] = []

    class Handle:
        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        lambda *_args, **_kwargs: Handle(),
    )
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda _group=None: events.append("destroy"),
    )
    adapter = NativePyTorchAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 4))

    dist.destroy_process_group()

    assert optimizer.param_groups
    assert events == ["close", "destroy"]
    assert not adapter.attached


def test_native_adapter_runs_process_group_teardown_when_close_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import torch.distributed as dist

    events: list[str] = []

    class Handle:
        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close failed")

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        lambda *_args, **_kwargs: Handle(),
    )
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda _group=None: events.append("destroy"),
    )
    adapter = NativePyTorchAdapter({"enable_checkpoint": False, "enable_detection": False})
    adapter.install(_context(tmp_path))
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    model(torch.ones(1, 4))

    with pytest.raises(RuntimeError, match="close failed"):
        dist.destroy_process_group()

    assert events == ["close", "destroy"]
    assert optimizer.param_groups


def test_native_adapter_intercepts_teardown_alias_captured_before_attachment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    import torch.distributed as dist

    events: list[str] = []

    class Handle:
        def close(self) -> None:
            events.append("close")

    monkeypatch.setattr(
        "lm_resiliency.integrations.pytorch.enable_resiliency",
        lambda *_args, **_kwargs: Handle(),
    )
    monkeypatch.setattr(
        dist,
        "destroy_process_group",
        lambda _group=None: events.append("destroy"),
    )
    adapter = _AutoFrameworkAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_context(tmp_path))
    namespace: dict[str, Any] = {}
    try:
        exec(
            "from torch.distributed import destroy_process_group as teardown",
            namespace,
        )
        delegate = adapter.delegate
        assert isinstance(delegate, NativePyTorchAdapter)
        model = torch.nn.Linear(4, 2)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        model(torch.ones(1, 4))

        namespace["teardown"]()

        assert optimizer.param_groups
        assert events == ["close", "destroy"]
        assert not delegate.attached
    finally:
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
    core_package = types.ModuleType("megatron.core")
    core_package.__path__ = []
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    teardown_events: list[str] = []
    parallel_state.destroy_model_parallel = lambda: teardown_events.append(  # type: ignore[attr-defined]
        "destroy"
    )
    model = [object(), object()]
    optimizer = object()
    scheduler = SimpleNamespace(
        state_dict=lambda: {"num_steps": 17},
        load_state_dict=lambda state: setattr(scheduler, "restored", state),
    )
    arguments = SimpleNamespace(
        iteration=0,
        consumed_train_samples=34,
        skipped_train_samples=2,
    )
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
    monkeypatch.setitem(sys.modules, "megatron.core", core_package)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
    calls: list[tuple[Any, Any, Any, dict[str, Any]]] = []
    sentinel = SimpleNamespace(
        step_count=17,
        close=lambda: teardown_events.append("close"),
    )

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
        capture = calls[0][3]["extra_state_fn"]
        restore = calls[0][3]["load_extra_state_fn"]
        arguments.curr_iteration = 18
        assert capture() == {
            "torchrun_megatron_loop": {
                "iteration": 19,
                "consumed_train_samples": 34,
                "skipped_train_samples": 2,
                "scheduler": {"num_steps": 17},
            }
        }
        restore(
            {
                "torchrun_megatron_loop": {
                    "iteration": 11,
                    "consumed_train_samples": 22,
                    "skipped_train_samples": 1,
                    "scheduler": {"num_steps": 11},
                }
            }
        )
        assert (
            arguments.iteration,
            arguments.consumed_train_samples,
            arguments.skipped_train_samples,
        ) == (11, 22, 1)
        assert scheduler.restored == {"num_steps": 11}
        parallel_state.destroy_model_parallel()  # type: ignore[attr-defined]
        assert teardown_events == ["close", "destroy"]
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
    core_package = types.ModuleType("megatron.core")
    core_package.__path__ = []
    parallel_state = types.ModuleType("megatron.core.parallel_state")
    parallel_state.destroy_model_parallel = lambda: None  # type: ignore[attr-defined]
    model = [object(), object()]
    optimizer = object()
    scheduler = object()
    arguments = SimpleNamespace(
        iteration=0,
        consumed_train_samples=0,
        skipped_train_samples=0,
    )
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
    monkeypatch.setitem(sys.modules, "megatron.core", core_package)
    monkeypatch.setitem(sys.modules, "megatron.core.parallel_state", parallel_state)
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


def test_deepspeed_adapter_rejects_framework_load_after_manager_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Engine:
        closed = False

        def load_checkpoint(self) -> str:
            return "framework checkpoint loaded"

        def destroy(self) -> None:
            assert closed

    deepspeed = types.ModuleType("deepspeed")
    engine = Engine()
    deepspeed.initialize = lambda: (engine, None, None, None)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "deepspeed", deepspeed)
    closed = False

    def close() -> None:
        nonlocal closed
        closed = True

    handle = SimpleNamespace(recovered_step=4, close=close)
    monkeypatch.setattr(
        "lm_resiliency.integrations.deepspeed.training.enable_resiliency",
        lambda *_args, **_kwargs: handle,
    )
    adapter = DeepSpeedWorkerAdapter(
        {
            "enable_checkpoint": False,
            "enable_detection": False,
        }
    )
    adapter.install(_recovery_context(tmp_path))
    deepspeed.initialize()
    try:
        with pytest.raises(
            TorchrunWorkerAdapterError,
            match=r"load_checkpoint\(\) cannot run after manager-selected GEMINI recovery",
        ):
            engine.load_checkpoint()
    finally:
        engine.destroy()

    assert closed
    assert engine.load_checkpoint() == "framework checkpoint loaded"


def test_builtin_adapter_construction_does_not_import_optional_frameworks() -> None:
    optional_roots = {"torchtitan", "megatron", "deepspeed"}
    before = set(sys.modules)

    for adapter_type in (
        TorchTitanWorkerAdapter,
        MegatronWorkerAdapter,
        DeepSpeedWorkerAdapter,
    ):
        adapter_type()

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

    config.write_text(
        'schema_version = 1\nadapter = "pytorch"\n',
        encoding="utf-8",
    )
    with pytest.raises(
        TorchrunWorkerAdapterError,
        match="module:factory",
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
        ("generation", 2, "does not match rendezvous generation 1"),
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
        restart_deadline_unix_ms=9_999_999_999_999,
    )
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    if field == "generation":
        restart = replace(restart, generation=value)
    elif field == "local_world_size":
        environment["LOCAL_WORLD_SIZE"] = str(value)
    else:
        environment[f"LM_RESILIENCY_TORCHRUN_{field.upper()}"] = str(value)
    SimpleRestartContextFile(context_path).write(restart)

    with pytest.raises(TorchrunWorkerAdapterError, match=message):
        _context_from_environment(environment)


def test_successor_worker_requires_exact_restart_context_generation(tmp_path: Path) -> None:
    context_path = (tmp_path / "restart-context.json").resolve()
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    with pytest.raises(TorchrunWorkerAdapterError, match="requires a restart context"):
        _context_from_environment(environment)


def test_successor_worker_preserves_manager_topology_digest(tmp_path: Path) -> None:
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
        topology_digest="checkpoint-topology",
        recovery_mode="recovery_verified",
        checkpoint_source="gemini",
        checkpoint_step=4,
        checkpoint_id=None,
        checkpoint_manifest_id="manifest-4",
        reason_code="attributed_sdc",
        restart_deadline_unix_ms=9_999_999_999_999,
    )
    SimpleRestartContextFile(context_path).write(restart)
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    context = _context_from_environment(environment)

    assert context.topology_digest == "checkpoint-topology"


@pytest.mark.parametrize(
    ("environment_changes", "context_changes", "message"),
    [
        ({"WORLD_SIZE": "2"}, {}, "changes WORLD_SIZE"),
        (
            {"GROUP_RANK": "1", "RANK": "1", "WORLD_SIZE": "2"},
            {"expected_world_size": 2},
            "changes GROUP_RANK",
        ),
    ],
)
def test_successor_worker_rejects_rank_topology_mismatch(
    tmp_path: Path,
    environment_changes: dict[str, str],
    context_changes: dict[str, Any],
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
        restart_deadline_unix_ms=9_999_999_999_999,
    )
    if context_changes:
        restart = replace(restart, **context_changes)
    SimpleRestartContextFile(context_path).write(restart)
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }
    environment.update(environment_changes)

    with pytest.raises(TorchrunWorkerAdapterError, match=message):
        _context_from_environment(environment)


def test_successor_worker_rejects_expired_restart_context(tmp_path: Path) -> None:
    context_path = (tmp_path / "restart-context.json").resolve()
    SimpleRestartContextFile(context_path).write(
        RestartContext(
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
            restart_deadline_unix_ms=1,
        )
    )
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "1",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    with pytest.raises(TorchrunWorkerAdapterError, match="deadline elapsed"):
        _context_from_environment(environment)


def test_initial_worker_rejects_stale_restart_context(tmp_path: Path) -> None:
    context_path = (tmp_path / "restart-context.json").resolve()
    SimpleRestartContextFile(context_path).write(
        RestartContext(
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
            restart_deadline_unix_ms=9_999_999_999_999,
        )
    )
    environment = {
        "GROUP_RANK": "0",
        "LOCAL_RANK": "0",
        "LM_RESILIENCY_TORCHRUN_RUN_ID": "adapter-run",
        "LM_RESILIENCY_TORCHRUN_NODE_ID": "node-a",
        "LOCAL_WORLD_SIZE": "1",
        "LM_RESILIENCY_TORCHRUN_RESTART_CONTEXT": str(context_path),
        "LM_RESILIENCY_TORCHRUN_EXPECTED_GENERATION": "0",
        "RANK": "0",
        "WORLD_SIZE": "1",
    }

    with pytest.raises(TorchrunWorkerAdapterError, match="must not have a restart context"):
        _context_from_environment(environment)


def test_torchrun_package_exports_stable_worker_adapter_api() -> None:
    from lm_resiliency.integrations import torchrun

    expected = {
        "DeepSpeedWorkerAdapter",
        "MegatronWorkerAdapter",
        "NativePyTorchAdapter",
        "NativePyTorchDDPAdapter",
        "TorchTitanWorkerAdapter",
        "TorchrunInitialPlacement",
        "TorchrunLaunchConfig",
        "TorchrunRecoveryCoordinator",
        "TorchrunRecoveryRequest",
        "TorchrunSuccessorPlacement",
        "TorchrunWorkerAdapter",
        "TorchrunWorkerAdapterError",
        "TorchrunWorkerContext",
        "create_rendezvous_handler",
        "derive_torchrun_node_id",
        "get_torchrun_worker_context",
        "get_rendezvous_handler_creator",
    }

    assert set(torchrun.__all__) == expected
    for symbol in expected:
        assert getattr(torchrun, symbol) is not None
