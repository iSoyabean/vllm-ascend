"""Catccos .so loader, SHMEM lifecycle, and optional smoke test."""

import atexit
import os

import torch
import torch.distributed as dist
from vllm.logger import init_logger

import vllm_ascend.envs as envs_ascend

logger = init_logger(__name__)

_DEFAULT_SHMEM_PORT = 28735
_DEFAULT_SHMEM_LOCAL_MEM_SIZE = 1024**3

_loaded = False
_shmem_initialized = False
_atexit_registered = False


def catccos_enabled() -> bool:
    return bool(envs_ascend.VLLM_ASCEND_ENABLE_CATCCOS)


def is_catccos_loaded() -> bool:
    return _loaded


def is_catccos_initialized() -> bool:
    return _shmem_initialized



def _barrier_if_distributed() -> None:
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def _materialize_smoke_tensor(shape: tuple[int, int]) -> torch.Tensor:
    tensor = torch.empty(shape, dtype=torch.float16, device="npu")
    tensor.fill_(1)
    return tensor.contiguous()


def _log_smoke_tensor(name: str, tensor: torch.Tensor) -> None:
    try:
        data_ptr = tensor.data_ptr()
    except RuntimeError:
        logger.exception(
            "catccos smoke %s has no storage: shape=%s dtype=%s device=%s layout=%s is_meta=%s",
            name,
            tuple(tensor.shape),
            tensor.dtype,
            tensor.device,
            tensor.layout,
            getattr(tensor, "is_meta", False),
        )
        raise

    logger.info(
        "catccos smoke %s ready: shape=%s dtype=%s device=%s contiguous=%s data_ptr=%s",
        name,
        tuple(tensor.shape),
        tensor.dtype,
        tensor.device,
        tensor.is_contiguous(),
        data_ptr,
    )

def _get_shmem_ip_port() -> str:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    return f"tcp://{master_addr}:{_DEFAULT_SHMEM_PORT}"


def load_catccos_library() -> None:
    """Load libcatccos_torch.so into the current worker process."""
    global _loaded

    if not catccos_enabled() or _loaded:
        return

    so_path = envs_ascend.VLLM_ASCEND_CATCCOS_OPS_SO
    if not so_path:
        raise RuntimeError(
            "VLLM_ASCEND_ENABLE_CATCCOS=1 requires "
            "VLLM_ASCEND_CATCCOS_OPS_SO to point to libcatccos_torch.so."
        )
    if not os.path.exists(so_path):
        raise RuntimeError(f"catccos ops library not found: {so_path}")

    logger.info("catccos loading library: %s", so_path)
    torch.ops.load_library(so_path)
    _loaded = True
    logger.info("catccos library loaded")


def init_catccos_shmem(rank: int, world_size: int) -> None:
    """Initialize catccos SHMEM after vLLM distributed setup is ready."""
    global _atexit_registered, _shmem_initialized

    if not catccos_enabled() or _shmem_initialized:
        return

    load_catccos_library()

    ip_port = _get_shmem_ip_port()
    status = torch.ops.catccos.init(rank, world_size, _DEFAULT_SHMEM_LOCAL_MEM_SIZE, ip_port)
    if status != 0:
        raise RuntimeError(
            f"catccos SHMEM init failed: rank={rank} world_size={world_size} "
            f"ip_port={ip_port} status={status}"
        )

    _shmem_initialized = True
    if not _atexit_registered:
        atexit.register(finalize_catccos_shmem)
        _atexit_registered = True

    _barrier_if_distributed()

    logger.info(
        "catccos SHMEM init ok: rank=%s world_size=%s ip_port=%s",
        rank,
        world_size,
        ip_port,
    )


def finalize_catccos_shmem() -> None:
    """Finalize catccos SHMEM once per worker process."""
    global _shmem_initialized

    if not _shmem_initialized:
        return

    status = torch.ops.catccos.finalize()
    _shmem_initialized = False
    if status != 0:
        logger.warning("catccos SHMEM finalize returned non-zero status: %s", status)
        return

    logger.info("catccos SHMEM finalized")


def run_catccos_smoke_test(world_size: int) -> None:
    """Run a tiny allgather_matmul if the debug smoke-test switch is enabled."""
    if not envs_ascend.VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST or not _shmem_initialized:
        return

    _barrier_if_distributed()

    m, k, n = 128, 256, 128
    a = _materialize_smoke_tensor((m, k))
    b = _materialize_smoke_tensor((k, n))
    if dist.is_available() and dist.is_initialized():
        dist.broadcast(b, src=0)
    torch.npu.synchronize()

    _log_smoke_tensor("a", a)
    _log_smoke_tensor("b", b)
    _barrier_if_distributed()

    out = torch.ops.catccos.allgather_matmul(a, b, world_size)
    torch.npu.synchronize()
    _barrier_if_distributed()

    expected_shape = (m * world_size, n)
    if tuple(out.shape) != expected_shape:
        raise RuntimeError(
            f"catccos smoke test shape mismatch: got={tuple(out.shape)} "
            f"expected={expected_shape}"
        )

    logger.info(
        "catccos smoke test passed: output_shape=%s dtype=%s",
        tuple(out.shape),
        out.dtype,
    )
