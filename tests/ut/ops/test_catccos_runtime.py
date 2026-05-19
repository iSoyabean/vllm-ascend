import importlib
import logging
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _install_lightweight_imports(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    ops_module = ModuleType("vllm_ascend.ops")
    ops_module.__path__ = [str(repo_root / "vllm_ascend" / "ops")]
    catccos_module = ModuleType("vllm_ascend.ops.catccos")
    catccos_module.__path__ = [str(repo_root / "vllm_ascend" / "ops" / "catccos")]
    vllm_module = ModuleType("vllm")
    logger_module = ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger

    monkeypatch.setitem(sys.modules, "vllm_ascend.ops", ops_module)
    monkeypatch.setitem(sys.modules, "vllm_ascend.ops.catccos", catccos_module)
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)
    sys.modules.pop("vllm_ascend.ops.catccos.register", None)


def _reload_runtime(monkeypatch, *, enabled=False, so_path="", smoke=False):
    _install_lightweight_imports(monkeypatch)
    monkeypatch.setenv("VLLM_ASCEND_ENABLE_CATCCOS", "1" if enabled else "0")
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_OPS_SO", so_path)
    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST", "1" if smoke else "0")
    module = importlib.import_module("vllm_ascend.ops.catccos.register")
    return importlib.reload(module)


def test_disabled_runtime_does_not_load_library(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=False, so_path="/fake/libcatccos_torch.so")
    fake_ops = SimpleNamespace(load_library=MagicMock())
    monkeypatch.setattr(runtime.torch, "ops", fake_ops)

    runtime.load_catccos_library()

    fake_ops.load_library.assert_not_called()
    assert not runtime.is_catccos_loaded()
    assert not runtime.is_catccos_initialized()


def test_enabled_runtime_requires_library_path(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="")

    with pytest.raises(RuntimeError, match="VLLM_ASCEND_CATCCOS_OPS_SO"):
        runtime.load_catccos_library()


def test_enabled_runtime_requires_existing_library(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="/missing/libcatccos_torch.so")
    monkeypatch.setattr(runtime.os.path, "exists", lambda _: False)

    with pytest.raises(RuntimeError, match="not found"):
        runtime.load_catccos_library()


def test_enabled_runtime_loads_library_once(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="/fake/libcatccos_torch.so")
    monkeypatch.setattr(runtime.os.path, "exists", lambda _: True)
    fake_ops = SimpleNamespace(load_library=MagicMock())
    monkeypatch.setattr(runtime.torch, "ops", fake_ops)

    runtime.load_catccos_library()
    runtime.load_catccos_library()

    fake_ops.load_library.assert_called_once_with("/fake/libcatccos_torch.so")
    assert runtime.is_catccos_loaded()


def test_init_uses_master_addr_and_default_port(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="/fake/libcatccos_torch.so")
    monkeypatch.setenv("MASTER_ADDR", "10.1.2.3")
    monkeypatch.setattr(runtime, "_loaded", True)
    mock_atexit_register = MagicMock()
    monkeypatch.setattr(runtime.atexit, "register", mock_atexit_register)
    mock_barrier = MagicMock()
    monkeypatch.setattr(runtime, "_barrier_if_distributed", mock_barrier)
    fake_catccos = SimpleNamespace(init=MagicMock(return_value=0), finalize=MagicMock())
    monkeypatch.setattr(runtime.torch, "ops", SimpleNamespace(catccos=fake_catccos))

    runtime.init_catccos_shmem(rank=1, world_size=4)
    runtime.init_catccos_shmem(rank=1, world_size=4)

    fake_catccos.init.assert_called_once_with(1, 4, 1024**3, "tcp://10.1.2.3:28735")
    mock_atexit_register.assert_called_once_with(runtime.finalize_catccos_shmem)
    mock_barrier.assert_called_once_with()
    assert runtime.is_catccos_initialized()


def test_finalize_is_idempotent(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="/fake/libcatccos_torch.so")
    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_shmem_initialized", True)
    fake_catccos = SimpleNamespace(finalize=MagicMock(return_value=0))
    monkeypatch.setattr(runtime.torch, "ops", SimpleNamespace(catccos=fake_catccos))

    runtime.finalize_catccos_shmem()
    runtime.finalize_catccos_shmem()

    fake_catccos.finalize.assert_called_once_with()
    assert not runtime.is_catccos_initialized()


def test_smoke_test_is_opt_in_and_does_not_finalize(monkeypatch):
    runtime = _reload_runtime(monkeypatch, enabled=True, so_path="/fake/libcatccos_torch.so", smoke=False)
    monkeypatch.setattr(runtime, "_loaded", True)
    monkeypatch.setattr(runtime, "_shmem_initialized", True)
    fake_out = SimpleNamespace(shape=(256, 128), dtype="float16")
    fake_catccos = SimpleNamespace(
        allgather_matmul=MagicMock(return_value=fake_out),
        finalize=MagicMock(),
    )
    fake_torch = SimpleNamespace(
        ops=SimpleNamespace(catccos=fake_catccos),
        float16="float16",
        npu=SimpleNamespace(synchronize=MagicMock()),
    )
    fake_a = SimpleNamespace(
        shape=(128, 256),
        dtype="float16",
        device="npu",
        layout="strided",
        is_meta=False,
        is_contiguous=MagicMock(return_value=True),
        data_ptr=MagicMock(return_value=123),
    )
    fake_b = SimpleNamespace(
        shape=(256, 128),
        dtype="float16",
        device="npu",
        layout="strided",
        is_meta=False,
        is_contiguous=MagicMock(return_value=True),
        data_ptr=MagicMock(return_value=456),
    )
    fake_materialize = MagicMock(side_effect=[fake_a, fake_b])
    mock_barrier = MagicMock()
    monkeypatch.setattr(runtime, "torch", fake_torch)
    monkeypatch.setattr(runtime, "_materialize_smoke_tensor", fake_materialize)
    monkeypatch.setattr(runtime, "_barrier_if_distributed", mock_barrier)

    runtime.run_catccos_smoke_test(world_size=2)
    fake_catccos.allgather_matmul.assert_not_called()

    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST", "1")
    runtime.run_catccos_smoke_test(world_size=2)

    fake_materialize.assert_any_call((128, 256))
    fake_materialize.assert_any_call((256, 128))
    fake_a.data_ptr.assert_called_once_with()
    fake_b.data_ptr.assert_called_once_with()
    fake_catccos.allgather_matmul.assert_called_once_with(fake_a, fake_b, 2)
    assert mock_barrier.call_count == 3
    fake_catccos.finalize.assert_not_called()
