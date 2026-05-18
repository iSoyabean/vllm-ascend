"""Public Python wrappers for catccos torch operators."""

import torch

from vllm_ascend.ops.catccos import register as runtime


def allgather_matmul(a: torch.Tensor, b: torch.Tensor, world_size: int) -> torch.Tensor:
    if not runtime.is_catccos_initialized():
        raise RuntimeError(
            "catccos is not initialized. Set VLLM_ASCEND_ENABLE_CATCCOS=1 "
            "and VLLM_ASCEND_CATCCOS_OPS_SO before starting vLLM-Ascend."
        )

    return torch.ops.catccos.allgather_matmul(a, b, world_size)
