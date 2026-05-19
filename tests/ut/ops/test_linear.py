import os
import unittest
from unittest import mock
from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend import ascend_config
from vllm_ascend.distributed import parallel_state
from vllm_ascend.ops.linear import (
    AscendMergedColumnParallelLinear,
    AscendReplicatedLinear,
    AscendRowParallelLinear,
    AscendUnquantizedLinearMethod,
)
from vllm_ascend.ops.linear_op import CatccosMLPColumnParallelOp, MLPColumnParallelOp


class BaseLinearTest(unittest.TestCase):
    def setUp(self):
        self.mock_group = mock.MagicMock()
        self.mock_group.world_size = 2
        self.mock_group.rank_in_group = 0

        parallel_state._MLP_TP = self.mock_group
        parallel_state._OTP = self.mock_group

        self.mock_ascend_config = MagicMock()
        self.mock_ascend_config.finegrained_tp_config.oproj_tensor_parallel_size = 2
        self.mock_ascend_config.finegrained_tp_config.mlp_tensor_parallel_size = 2

        self.patches = [
            patch("vllm_ascend.ascend_config.get_ascend_config", return_value=self.mock_ascend_config),
            patch("vllm_ascend.distributed.parallel_state.get_otp_group", return_value=self.mock_group),
            patch("vllm_ascend.distributed.parallel_state.get_mlp_tp_group", return_value=self.mock_group),
            patch("vllm_ascend.ops.linear_op.get_tp_group", return_value=self.mock_group),
            patch(
                "vllm.distributed.parallel_state.get_tp_group",
                return_value=self.mock_group,
            ),
            patch("vllm_ascend.utils.mlp_tp_enable", return_value=True),
            patch("vllm_ascend.utils.oproj_tp_enable", return_value=True),
        ]

        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in self.patches:
            p.stop()


class TestAscendUnquantizedLinearMethod(TestBase):
    def setUp(self):
        self.method = AscendUnquantizedLinearMethod()
        self.layer = mock.MagicMock()
        mock_dtype = mock.PropertyMock(return_value=torch.float16)
        type(self.layer.weight.data).dtype = mock_dtype

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "0"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz0(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "1"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz1(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_not_called()

    @patch.dict(os.environ, {"VLLM_ASCEND_ENABLE_NZ": "2"})
    @mock.patch("torch_npu.npu_format_cast")
    def test_process_weights_after_loading_with_nz2(self, mock_format_cast):
        self.method.process_weights_after_loading(self.layer)
        mock_format_cast.assert_called_once()


class TestAscendRowParallelLinear(BaseLinearTest):
    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm.config.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_mlp_optimize(self, mock_get_current_vllm_config, mock_get_weight_prefetch_method):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="down_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)

    @patch("vllm_ascend.ops.linear_op.get_weight_prefetch_method", return_value=MagicMock())
    @patch("vllm.config.get_current_vllm_config", return_value=MagicMock())
    @patch(
        "vllm_ascend.ops.linear.AscendUnquantizedLinearMethod.apply",
        new=lambda self, layer, x, bias=None: torch.nn.functional.linear(x, layer.weight, bias),
    )
    def test_oproj_tp(self, mock_get_current_vllm_config, mock_get_weight_prefetch_method):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.oproj_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendRowParallelLinear(
            input_size=16,
            output_size=8,
            prefix="o_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._OTP)

        input_tensor = torch.randn(16, 8)
        linear(input_tensor)


class TestAscendMergedColumnParallelLinear(BaseLinearTest):
    def test_merged_mlp_tp_init(self):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="gate_up_proj",
        )
        self.assertEqual(linear.custom_op.comm_group, parallel_state._MLP_TP)


class TestCatccosMLPColumnParallelOp(BaseLinearTest):
    def _prepare_config(self):
        ascend_config._ASCEND_CONFIG = MagicMock()
        ascend_config._ASCEND_CONFIG.recompute_scheduler_enable = False
        ascend_config._ASCEND_CONFIG.finegrained_tp_config.mlp_tensor_parallel_size = 2
        ascend_config._ASCEND_CONFIG.ascend_scheduler_config.enabled = False

    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_prefix_enabled", return_value=True)
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_enable", return_value=True)
    def test_catccos_mlp_column_parallel_selected(self, mock_enable, mock_prefix_enabled):
        self._prepare_config()

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="model.layers.0.mlp.gate_up_proj",
        )

        self.assertIsInstance(linear.custom_op, CatccosMLPColumnParallelOp)

    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_prefix_enabled", return_value=False)
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_enable", return_value=True)
    def test_catccos_mlp_column_parallel_requires_prefix_match(self, mock_enable, mock_prefix_enabled):
        self._prepare_config()

        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="model.layers.0.mlp.gate_up_proj",
        )

        self.assertIsInstance(linear.custom_op, MLPColumnParallelOp)
        self.assertNotIsInstance(linear.custom_op, CatccosMLPColumnParallelOp)

    @patch("vllm_ascend.ops.catccos.allgather_matmul")
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_prefix_enabled", return_value=True)
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_enable", return_value=True)
    def test_catccos_mlp_column_parallel_apply(self, mock_enable, mock_prefix_enabled, mock_allgather_matmul):
        self._prepare_config()
        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            prefix="model.layers.0.mlp.gate_up_proj",
        )
        fake_output = torch.randn(8, 16)
        mock_allgather_matmul.return_value = fake_output

        input_tensor = torch.randn(4, 16)
        output = linear(input_tensor)

        self.assertIs(output, fake_output)
        mock_allgather_matmul.assert_called_once()
        call_args = mock_allgather_matmul.call_args.args
        self.assertEqual(call_args[0].shape, input_tensor.shape)
        self.assertEqual(call_args[1].shape, (16, 16))
        self.assertEqual(call_args[2], 2)


    @patch("vllm_ascend.ops.catccos.allgather_matmul")
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_prefix_enabled", return_value=True)
    @patch("vllm_ascend.ops.linear_op.catccos_allgather_matmul_enable", return_value=True)
    def test_catccos_mlp_column_parallel_falls_back_for_unsupported_quant_method(
        self, mock_enable, mock_prefix_enabled, mock_allgather_matmul
    ):
        self._prepare_config()
        linear = AscendMergedColumnParallelLinear(
            input_size=16,
            output_sizes=[8, 8],
            bias=False,
            prefix="model.layers.0.mlp.gate_up_proj",
        )
        fake_output = torch.randn(4, 16)
        input_tensor = torch.randn(4, 16)
        gathered_input = torch.randn(8, 16)

        class UnsupportedQuantMethod:
            def __init__(self):
                self.apply = MagicMock(return_value=fake_output)

        fake_quant_method = UnsupportedQuantMethod()
        linear.quant_method = fake_quant_method
        linear.custom_op.update_attrs()
        self.mock_group.all_gather.return_value = gathered_input

        output = linear(input_tensor)

        self.assertIs(output, fake_output)
        mock_allgather_matmul.assert_not_called()
        self.mock_group.all_gather.assert_called_once_with(input_tensor, 0)
        fake_quant_method.apply.assert_called_once_with(linear, gathered_input, None)


class TestAscendReplicatedLinear(BaseLinearTest):
    def test_init_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))

    def test_init_without_disable_tp(self):
        linear = AscendReplicatedLinear(
            input_size=16,
            output_size=8,
        )
        self.assertTrue(isinstance(linear.quant_method, AscendUnquantizedLinearMethod))


if __name__ == "__main__":
    unittest.main()
