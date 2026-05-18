import importlib
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest


def test_allgather_matmul_requires_initialized_runtime(monkeypatch):
    wrapper = importlib.import_module("vllm_ascend.ops.catccos.wrapper")
    wrapper = importlib.reload(wrapper)
    monkeypatch.setattr(wrapper.runtime, "is_catccos_initialized", lambda: False)

    with pytest.raises(RuntimeError, match="catccos is not initialized"):
        wrapper.allgather_matmul(object(), object(), 2)


def test_allgather_matmul_delegates_to_torch_op(monkeypatch):
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
