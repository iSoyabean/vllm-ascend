"""Catccos .so loader, SHMEM init, and safety probe."""

import os

import torch

import vllm_ascend.envs as envs_ascend
from vllm.logger import init_logger

logger = init_logger(__name__)

_loaded = False
_shmem_initialized = False


def catccos_enabled() -> bool:
    so = envs_ascend.VLLM_ASCEND_CATCCOS_OPS_SO
    return bool(so and os.path.exists(so))


def is_catccos_loaded() -> bool:
    return _loaded


def load_catccos_library() -> None:
    """Load libcatccos_torch.so. Called once in NPUWorker.__init__."""
    global _loaded

    if _loaded:
        return

    so_path = envs_ascend.VLLM_ASCEND_CATCCOS_OPS_SO
    if not so_path:
        return

    if not os.path.exists(so_path):
        logger.warning(
            "VLLM_ASCEND_CATCCOS_OPS_SO is set but file not found: %s", so_path
        )
        return

    logger.info("catccos loading library: %s", so_path)
    torch.ops.load_library(so_path)
    _loaded = True
    logger.info("catccos library loaded")


def init_catccos_shmem(rank: int, world_size: int) -> None:
    """Initialize SHMEM via torch.ops.catccos.init().

    Call AFTER HCCL/distributed init, BEFORE any model forward.
    """
    global _shmem_initialized

    if not _loaded:
        return

    if _shmem_initialized:
        return

    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    ip_port = f"tcp://{master_addr}:28735"

    status = torch.ops.catccos.init(rank, world_size, 1024 ** 3, ip_port)
    if status != 0:
        raise RuntimeError(
            f"catccos SHMEM init failed: rank={rank} world_size={world_size} status={status}"
        )

    _shmem_initialized = True
    logger.info(
        "catccos SHMEM init ok: rank=%s world_size=%s ip_port=%s",
        rank,
        world_size,
        ip_port,
    )


def probe_catccos(world_size: int) -> None:
    """Run one allgather_matmul to verify the end-to-end catccos pipeline.

    Call AFTER model weights are loaded (in load_model).
    """
    global _shmem_initialized

    if not _shmem_initialized:
        return

    m, k, n = 128, 256, 128
    a = torch.ones((m, k), dtype=torch.float16, device="npu")
    b = torch.ones((k, n), dtype=torch.float16, device="npu")
    out = torch.ops.catccos.allgather_matmul(a, b, world_size)
    torch.npu.synchronize()

    expected_shape = (m * world_size, n)
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(
            f"catccos probe shape mismatch: got={tuple(out.shape)} expected={expected_shape}"
        )

    torch.ops.catccos.finalize()
    _shmem_initialized = False

    logger.info(
        "catccos probe passed: output_shape=%s dtype=%s",
        tuple(out.shape),
        out.dtype,
    )
