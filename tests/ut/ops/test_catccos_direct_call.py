# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import sys
from types import ModuleType, SimpleNamespace

import torch

from vllm_ascend.ops.catccos import runtime
from vllm_ascend.ops import linear_op


def _reset_runtime_state():
    runtime._CATCCOS_SHMEM_INITIALIZED = False
    runtime._CATCCOS_ATEXIT_REGISTERED = False


def test_catccos_runtime_disabled_does_not_import_or_init(monkeypatch):
    _reset_runtime_state()
    monkeypatch.setattr(runtime.envs_ascend, "VLLM_ASCEND_ENABLE_CATCCOS", False, raising=False)
    sys.modules.pop("vllm_ascend.vllm_ascend_catccos_C", None)

    runtime.init_catccos_shmem(rank=0, world_size=2)

    assert not runtime.is_catccos_shmem_initialized()
    assert "vllm_ascend.vllm_ascend_catccos_C" not in sys.modules


def test_catccos_runtime_calls_registered_ops(monkeypatch):
    _reset_runtime_state()
    calls = []

    def fake_init(rank, world_size, local_mem_size, ip_port):
        calls.append(("init", rank, world_size, local_mem_size, ip_port))
        return 0

    def fake_finalize():
        calls.append(("finalize",))
        return 0

    fake_ops = SimpleNamespace(catccos_init=fake_init, catccos_finalize=fake_finalize)
    monkeypatch.setattr(runtime.torch, "ops", SimpleNamespace(_C_ascend=fake_ops))
    monkeypatch.setattr(runtime.envs_ascend, "VLLM_ASCEND_ENABLE_CATCCOS", True, raising=False)
    monkeypatch.setenv("MASTER_ADDR", "10.0.0.8")
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.vllm_ascend_catccos_C",
        ModuleType("vllm_ascend.vllm_ascend_catccos_C"),
    )

    runtime.init_catccos_shmem(rank=1, world_size=2)
    runtime.init_catccos_shmem(rank=1, world_size=2)

    assert runtime.is_catccos_shmem_initialized()
    assert calls == [("init", 1, 2, 1024**3, "tcp://10.0.0.8:28735")]

    runtime.finalize_catccos_shmem()

    assert not runtime.is_catccos_shmem_initialized()
    assert calls[-1] == ("finalize",)


def test_matmul_allreduce_row_parallel_uses_catccos_and_adds_bias(monkeypatch):
    class AscendUnquantizedLinearMethod:
        pass

    op = linear_op.MatmulAllreduceRowParallelOp.__new__(linear_op.MatmulAllreduceRowParallelOp)
    op.input_is_parallel = True
    op.reduce_results = True
    op.skip_bias_add = False
    op.bias = torch.tensor([1.0, 2.0], dtype=torch.float16)
    op.layer = SimpleNamespace(
        weight=torch.tensor([[1.0, 2.0, 3.0, 4.0], [2.0, 1.0, 0.0, 1.0]], dtype=torch.float16)
    )
    op.quant_method = AscendUnquantizedLinearMethod()

    calls = []

    def fake_catccos(input_parallel, weight, tp_size):
        calls.append((input_parallel, weight, tp_size))
        return input_parallel @ weight

    fake_ops = SimpleNamespace(_C_ascend=SimpleNamespace(catccos_matmul_allreduce=fake_catccos))
    monkeypatch.setattr(linear_op.MatmulAllreduceRowParallelOp, "tp_rank", property(lambda self: 0))
    monkeypatch.setattr(linear_op.MatmulAllreduceRowParallelOp, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(linear_op, "catccos_matmul_allreduce_enable", lambda: True)
    monkeypatch.setattr(linear_op.torch, "ops", fake_ops)

    input_tensor = torch.tensor([[1.0, 0.0, 1.0, 0.0]], dtype=torch.float16)
    output, output_bias = op.apply_impl(input_tensor)

    expected = input_tensor @ op.layer.weight.t() + op.bias
    assert torch.equal(output, expected)
    assert output_bias is None
    assert calls[0][2] == 2


def test_matmul_allreduce_row_parallel_keeps_torch_npu_fallback_when_gate_off(monkeypatch):
    class AscendUnquantizedLinearMethod:
        pass

    op = linear_op.MatmulAllreduceRowParallelOp.__new__(linear_op.MatmulAllreduceRowParallelOp)
    op.input_is_parallel = True
    op.reduce_results = True
    op.skip_bias_add = True
    op.bias = torch.tensor([1.0, 2.0], dtype=torch.float16)
    op.layer = SimpleNamespace(weight=torch.ones((2, 4), dtype=torch.float16))
    op.quant_method = AscendUnquantizedLinearMethod()
    op.hcomm_info = "hcomm"

    fallback_output = torch.ones((1, 2), dtype=torch.float16)
    catccos_calls = []
    fallback_calls = []

    def fake_catccos(*args):
        catccos_calls.append(args)
        raise AssertionError("catccos should not be called when gate is off")

    fake_ops = SimpleNamespace(_C_ascend=SimpleNamespace(catccos_matmul_allreduce=fake_catccos))

    def fake_mm_all_reduce_base(input_parallel, weight, hcomm_info, bias=None):
        fallback_calls.append((input_parallel, weight, hcomm_info, bias))
        return fallback_output

    monkeypatch.setattr(linear_op.MatmulAllreduceRowParallelOp, "tp_rank", property(lambda self: 0))
    monkeypatch.setattr(linear_op.MatmulAllreduceRowParallelOp, "tp_size", property(lambda self: 2))
    monkeypatch.setattr(linear_op, "catccos_matmul_allreduce_enable", lambda: False)
    monkeypatch.setattr(linear_op.torch, "ops", fake_ops)
    monkeypatch.setattr(linear_op.torch_npu, "npu_mm_all_reduce_base", fake_mm_all_reduce_base)

    output, output_bias = op.apply_impl(torch.ones((1, 4), dtype=torch.float16))

    assert output is fallback_output
    assert output_bias is op.bias
    assert catccos_calls == []
    assert fallback_calls[0][2] == "hcomm"
    assert fallback_calls[0][3] is None


def test_row_parallel_selection_accepts_catccos_gate(monkeypatch):
    def fake_init(self, layer):
        self.layer = layer

    monkeypatch.setattr(linear_op, "enable_dsa_cp_with_layer_shard", lambda: False)
    monkeypatch.setattr(linear_op, "mlp_tp_enable", lambda: False)
    monkeypatch.setattr(linear_op, "oproj_tp_enable", lambda: False)
    monkeypatch.setattr(linear_op, "matmul_allreduce_enable", lambda: False)
    monkeypatch.setattr(linear_op, "catccos_matmul_allreduce_enable", lambda: True)
    monkeypatch.setattr(linear_op, "flashcomm2_enable", lambda: False)
    monkeypatch.setattr(linear_op, "enable_sp", lambda: False)
    monkeypatch.setattr(linear_op.MatmulAllreduceRowParallelOp, "__init__", fake_init)

    op = linear_op._get_row_parallel_op("self_attn.o_proj", object())

    assert isinstance(op, linear_op.MatmulAllreduceRowParallelOp)
