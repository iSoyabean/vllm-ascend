"""Public Python wrappers for catccos torch operators."""

import torch

from vllm_ascend.ops.catccos import register as runtime


def _device_type(tensor: torch.Tensor) -> str | None:
    return getattr(getattr(tensor, "device", None), "type", None)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _validate_allgather_matmul_inputs(a: torch.Tensor, b: torch.Tensor, world_size: int) -> None:
    if not runtime.is_catccos_initialized():
        raise RuntimeError(
            "catccos is not initialized. Set VLLM_ASCEND_ENABLE_CATCCOS=1 "
            "and VLLM_ASCEND_CATCCOS_OPS_SO before starting vLLM-Ascend."
        )

    if world_size <= 0:
        raise RuntimeError(f"catccos allgather_matmul requires world_size > 0, got {world_size}.")

    if _device_type(a) != "npu" or _device_type(b) != "npu":
        raise RuntimeError(
            "catccos allgather_matmul requires NPU tensors, "
            f"got a.device={getattr(a, 'device', None)} b.device={getattr(b, 'device', None)}."
        )
    if getattr(a, "device", None) != getattr(b, "device", None):
        raise RuntimeError(
            "catccos allgather_matmul requires a and b on the same device, "
            f"got a.device={a.device} b.device={b.device}."
        )

    if a.dtype != torch.float16 or b.dtype != torch.float16:
        raise RuntimeError(
            "catccos allgather_matmul currently supports float16 only, "
            f"got a.dtype={a.dtype} b.dtype={b.dtype}."
        )

    a_shape = _shape(a)
    b_shape = _shape(b)
    if len(a_shape) != 2 or len(b_shape) != 2:
        raise RuntimeError(
            "catccos allgather_matmul requires 2D tensors, "
            f"got a.shape={a_shape} b.shape={b_shape}."
        )
    if a_shape[1] != b_shape[0]:
        raise RuntimeError(
            "catccos allgather_matmul shape mismatch, "
            f"a.shape={a_shape} b.shape={b_shape}."
        )


def allgather_matmul(a: torch.Tensor, b: torch.Tensor, world_size: int) -> torch.Tensor:
    _validate_allgather_matmul_inputs(a, b, world_size)
    return torch.ops.catccos.allgather_matmul(a, b, world_size)
