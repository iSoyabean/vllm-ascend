"""Public Python wrappers for catccos torch operators."""

import torch
from vllm.logger import logger

from vllm_ascend.ops.catccos import register as runtime

_CATCCOS_WRAPPER_LOG_LIMIT = 16
_catccos_wrapper_log_count = 0


def _log_catccos_wrapper_once(message: str, *args) -> None:
    global _catccos_wrapper_log_count
    if _catccos_wrapper_log_count >= _CATCCOS_WRAPPER_LOG_LIMIT:
        return
    _catccos_wrapper_log_count += 1
    logger.info(message, *args)


def _device_type(tensor: torch.Tensor) -> str | None:
    return getattr(getattr(tensor, "device", None), "type", None)


def _shape(tensor: torch.Tensor) -> tuple[int, ...]:
    return tuple(tensor.shape)


def _shape_or_none(tensor: object) -> tuple[int, ...] | None:
    shape = getattr(tensor, "shape", None)
    return tuple(shape) if shape is not None else None


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
    _log_catccos_wrapper_once(
        "catccos torch op call: a_shape=%s b_shape=%s a_dtype=%s b_dtype=%s "
        "a_device=%s b_device=%s world_size=%s",
        _shape(a),
        _shape(b),
        a.dtype,
        b.dtype,
        a.device,
        b.device,
        world_size,
    )
    output = torch.ops.catccos.allgather_matmul(a, b, world_size)
    _log_catccos_wrapper_once(
        "catccos torch op returned: output_shape=%s output_dtype=%s output_device=%s",
        _shape_or_none(output),
        getattr(output, "dtype", None),
        getattr(output, "device", None),
    )
    return output
