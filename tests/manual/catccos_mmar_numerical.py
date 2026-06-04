#!/usr/bin/env python3
"""Numerical check for in-tree catccos MMAR.

Run after building/installing vllm-ascend in the NPU docker::

    torchrun --nproc_per_node=2 tests/manual/catccos_mmar_numerical.py

The script initializes ``_C_ascend.catccos_*`` directly, runs
``catccos_matmul_allreduce(a, b, world_size)``, and compares it with
``dist.all_reduce(a @ b)``. It is intended as a focused operator check before
running a full vLLM model.
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="catccos MMAR numerical check")
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--shmem-size", type=int, default=1024**3)
    parser.add_argument("--catccos-port", type=int, default=28735)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def _set_npu_device(local_rank: int) -> torch.device:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available")
    torch.npu.set_device(local_rank)
    return torch.device(f"npu:{local_rank}")


def _sync_device() -> None:
    torch.npu.synchronize()


def main() -> None:
    args = parse_args()
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    ip_port = f"tcp://{master_addr}:{args.catccos_port}"

    device = _set_npu_device(local_rank)
    dist.init_process_group(backend="hccl", rank=rank, world_size=world_size)

    import vllm_ascend.vllm_ascend_C  # noqa: F401

    init_status = torch.ops._C_ascend.catccos_init(
        rank,
        world_size,
        args.shmem_size,
        ip_port,
    )
    if init_status != 0:
        raise RuntimeError(f"catccos init failed: status={init_status}")

    try:
        torch.manual_seed(args.seed + rank)
        a = torch.randn(args.m, args.k, device=device, dtype=torch.float16)
        b = torch.randn(args.k, args.n, device=device, dtype=torch.float16)
        dist.barrier()

        out = torch.ops._C_ascend.catccos_matmul_allreduce(a, b, world_size)
        _sync_device()

        ref = a.matmul(b)
        dist.all_reduce(ref, op=dist.ReduceOp.SUM)
        _sync_device()

        max_diff = (out - ref).abs().max().item()
        passed = torch.allclose(out, ref, rtol=args.rtol, atol=args.atol)
        print(
            f"rank={rank} shape={tuple(out.shape)} max_diff={max_diff:.6f} "
            f"passed={passed}",
            flush=True,
        )
        if not passed:
            raise AssertionError("catccos MMAR differs from all_reduce(a @ b)")
    finally:
        dist.barrier()
        finalize_status = torch.ops._C_ascend.catccos_finalize()
        if finalize_status != 0:
            print(
                f"rank={rank} catccos finalize returned status={finalize_status}",
                flush=True,
            )
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
