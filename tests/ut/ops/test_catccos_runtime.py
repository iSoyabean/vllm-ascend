import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def _reload_runtime(monkeypatch, *, enabled=False, so_path="", smoke=False):
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
    fake_catccos = SimpleNamespace(init=MagicMock(return_value=0), finalize=MagicMock())
    monkeypatch.setattr(runtime.torch, "ops", SimpleNamespace(catccos=fake_catccos))

    runtime.init_catccos_shmem(rank=1, world_size=4)
    runtime.init_catccos_shmem(rank=1, world_size=4)

    fake_catccos.init.assert_called_once_with(1, 4, 1024**3, "tcp://10.1.2.3:28735")
    mock_atexit_register.assert_called_once_with(runtime.finalize_catccos_shmem)
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
        ones=MagicMock(return_value=object()),
        float16="float16",
        npu=SimpleNamespace(synchronize=MagicMock()),
    )
    monkeypatch.setattr(runtime, "torch", fake_torch)

    runtime.run_catccos_smoke_test(world_size=2)
    fake_catccos.allgather_matmul.assert_not_called()

    monkeypatch.setenv("VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST", "1")
    runtime.run_catccos_smoke_test(world_size=2)

    fake_catccos.allgather_matmul.assert_called_once()
    fake_catccos.finalize.assert_not_called()
