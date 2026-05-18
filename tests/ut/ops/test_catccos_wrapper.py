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
    sys.modules.pop("vllm_ascend.ops.catccos.wrapper", None)


def test_allgather_matmul_requires_initialized_runtime(monkeypatch):
    _install_lightweight_imports(monkeypatch)
    wrapper = importlib.import_module("vllm_ascend.ops.catccos.wrapper")
    wrapper = importlib.reload(wrapper)
    monkeypatch.setattr(wrapper.runtime, "is_catccos_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="catccos is not initialized"):
        wrapper.allgather_matmul(object(), object(), 2)


def test_allgather_matmul_delegates_to_torch_op(monkeypatch):
    _install_lightweight_imports(monkeypatch)
    wrapper = importlib.import_module("vllm_ascend.ops.catccos.wrapper")
    wrapper = importlib.reload(wrapper)
    monkeypatch.setattr(wrapper.runtime, "is_catccos_initialized", lambda: True)
    fake_result = object()
    fake_catccos = SimpleNamespace(allgather_matmul=MagicMock(return_value=fake_result))
    monkeypatch.setattr(wrapper.torch, "ops", SimpleNamespace(catccos=fake_catccos))
    a = object()
    b = object()

    result = wrapper.allgather_matmul(a, b, 4)

    assert result is fake_result
    fake_catccos.allgather_matmul.assert_called_once_with(a, b, 4)
