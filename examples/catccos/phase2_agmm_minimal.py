# Copyright (c) 2026.
# This file is a part of the vllm-ascend catccos integration experiments.
"""Minimal catccos allgather_matmul check.

Run on an NPU machine with two ranks, for example:

    torchrun --nproc_per_node=2 examples/catccos/phase2_agmm_minimal.py \
        --catccos-ops-so /workspace/catccos/build/lib/libcatccos_torch.so
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.distributed as dist


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Minimal catccos AGMM check")
    parser.add_argument(
        "--catccos-ops-so",
        default=os.getenv("VLLM_ASCEND_CATCCOS_OPS_SO", ""),
        help="Path to libcatccos_torch.so.",
    )
    parser.add_argument("--m", type=int, default=128)
    parser.add_argument("--k", type=int, default=256)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--shmem-size", type=int, default=1024**3)
    parser.add_argument("--catccos-port", type=int, default=28735)
    parser.add_argument("--rtol", type=float, default=1e-2)
    parser.add_argument("--atol", type=float, default=1e-2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.catccos_ops_so:
        raise ValueError("--catccos-ops-so or VLLM_ASCEND_CATCCOS_OPS_SO is required")

    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    world_size = int(os.environ["WORLD_SIZE"])
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    ip_port = f"tcp://{master_addr}:{args.catccos_port}"

    if hasattr(torch, "npu"):
        torch.npu.set_device(local_rank)
        device = torch.device(f"npu:{local_rank}")
        backend = "hccl"
    else:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(local_rank)
        backend = "nccl"

    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)
    torch.ops.load_library(args.catccos_ops_so)
    init_status = torch.ops.catccos.init(rank, world_size, args.shmem_size, ip_port)
    if init_status != 0:
        raise RuntimeError(f"catccos init failed: status={init_status}")

    try:
        torch.manual_seed(2026 + rank)
        a_local = torch.randn(args.m, args.k, device=device, dtype=torch.float16)
        if rank == 0:
            b = torch.randn(args.k, args.n, device=device, dtype=torch.float16)
        else:
            b = torch.empty(args.k, args.n, device=device, dtype=torch.float16)
        dist.broadcast(b, src=0)
        dist.barrier()

        out = torch.ops.catccos.allgather_matmul(a_local, b, world_size)
        if hasattr(torch, "npu"):
            torch.npu.synchronize()
        else:
            torch.cuda.synchronize()

        gathered = [torch.empty_like(a_local) for _ in range(world_size)]
        dist.all_gather(gathered, a_local)
        ref = torch.cat(gathered, dim=0).matmul(b)

        max_diff = (out - ref).abs().max().item()
        passed = torch.allclose(out, ref, rtol=args.rtol, atol=args.atol)
        print(
            f"rank={rank} shape={tuple(out.shape)} max_diff={max_diff:.6f} "
            f"passed={passed}",
            flush=True,
        )
        if not passed:
            raise AssertionError("catccos allgather_matmul differs from reference")
    finally:
        torch.ops.catccos.finalize()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
