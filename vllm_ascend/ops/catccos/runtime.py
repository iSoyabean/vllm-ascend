# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.

import atexit
import os

import torch
from vllm.logger import logger

import vllm_ascend.envs as envs_ascend

_CATCCOS_SHMEM_INITIALIZED = False
_CATCCOS_ATEXIT_REGISTERED = False
_DEFAULT_CATCCOS_SHMEM_LOCAL_MEM_SIZE = 1024**3
_DEFAULT_CATCCOS_SHMEM_PORT = 28735


def _catccos_enabled() -> bool:
    return bool(envs_ascend.VLLM_ASCEND_ENABLE_CATCCOS)


def _get_catccos_shmem_ip_port() -> str:
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    return f"tcp://{master_addr}:{_DEFAULT_CATCCOS_SHMEM_PORT}"


def is_catccos_shmem_initialized() -> bool:
    return _CATCCOS_SHMEM_INITIALIZED


def finalize_catccos_shmem() -> None:
    global _CATCCOS_SHMEM_INITIALIZED

    if not _CATCCOS_SHMEM_INITIALIZED:
        return

    status = torch.ops._C_ascend.catccos_finalize()
    _CATCCOS_SHMEM_INITIALIZED = False
    if status != 0:
        logger.warning("catccos SHMEM finalize returned non-zero status: %s", status)


def init_catccos_shmem(rank: int, world_size: int) -> None:
    global _CATCCOS_ATEXIT_REGISTERED, _CATCCOS_SHMEM_INITIALIZED

    if not _catccos_enabled() or _CATCCOS_SHMEM_INITIALIZED:
        return

    import vllm_ascend.vllm_ascend_C  # noqa: F401

    ip_port = _get_catccos_shmem_ip_port()
    status = torch.ops._C_ascend.catccos_init(
        rank,
        world_size,
        _DEFAULT_CATCCOS_SHMEM_LOCAL_MEM_SIZE,
        ip_port,
    )
    if status != 0:
        raise RuntimeError(
            "catccos SHMEM init failed: "
            f"rank={rank} world_size={world_size} ip_port={ip_port} status={status}"
        )

    _CATCCOS_SHMEM_INITIALIZED = True
    if not _CATCCOS_ATEXIT_REGISTERED:
        atexit.register(finalize_catccos_shmem)
        _CATCCOS_ATEXIT_REGISTERED = True
