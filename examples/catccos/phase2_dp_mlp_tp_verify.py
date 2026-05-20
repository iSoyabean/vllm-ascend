# Copyright (c) 2026.
# This file is a part of the vllm-ascend catccos integration experiments.
"""Verify catccos Phase 2 with DP-based fine-grained MLP TP.

This script launches one vLLM instance per DP rank, enables
finegrained_tp_config.mlp_tensor_parallel_size=dp_size, and runs a short eager
inference. It is intended for NPU machines while validating the catccos manual
allgather_matmul replacement path.

Example:
    export VLLM_ASCEND_ENABLE_CATCCOS=1
    export VLLM_ASCEND_CATCCOS_OPS_SO=/workspace/catccos/build/lib/libcatccos_torch.so
    export VLLM_ASCEND_ENABLE_CATCCOS_ALLGATHER_MATMUL=1
    export VLLM_ASCEND_CATCCOS_ALLGATHER_MATMUL_PREFIXES=gate_up_proj
    export VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST=0
    export VLLM_LOGGING_LEVEL=INFO

    python examples/catccos/phase2_dp_mlp_tp_verify.py \
        --model /root/.cache/Qwen3-0.6B \
        --dp-size 2 \
        --tp-size 1 \
        --catccos-ops-so /workspace/catccos/build/lib/libcatccos_torch.so
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import os
from multiprocessing import Process
from time import sleep

import torch
from vllm import LLM, SamplingParams
from vllm.distributed.parallel_state import (
    destroy_distributed_environment,
    destroy_model_parallel,
)
from vllm.utils.network_utils import get_open_port

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="catccos Phase 2 DP + MLP TP verification"
    )
    parser.add_argument("--model", type=str, default="/root/.cache/Qwen3-0.6B")
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--node-size", type=int, default=1)
    parser.add_argument("--node-rank", type=int, default=0)
    parser.add_argument("--master-addr", type=str, default="")
    parser.add_argument("--master-port", type=int, default=0)
    parser.add_argument(
        "--catccos-ops-so",
        type=str,
        default=os.getenv("VLLM_ASCEND_CATCCOS_OPS_SO", ""),
    )
    parser.add_argument(
        "--prefixes",
        type=str,
        default=os.getenv(
            "VLLM_ASCEND_CATCCOS_ALLGATHER_MATMUL_PREFIXES", "gate_up_proj"
        ),
    )
    parser.add_argument(
        "--prompt", type=str, default="Write one short sentence about Ascend NPU."
    )
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.6)
    parser.add_argument("--dtype", type=str, default="float16")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--disable-catccos",
        action="store_true",
        help="Run the same DP/MLP-TP setup without Phase2 replacement.",
    )
    return parser.parse_args()


def cleanup_env_and_memory() -> None:
    with contextlib.suppress(Exception):
        destroy_model_parallel()
    with contextlib.suppress(Exception):
        destroy_distributed_environment()
    with contextlib.suppress(AssertionError, RuntimeError):
        torch.distributed.destroy_process_group()
    gc.collect()
    if hasattr(torch, "npu"):
        with contextlib.suppress(Exception):
            torch.npu.empty_cache()
        with contextlib.suppress(Exception):
            torch.npu.reset_peak_memory_stats()


def configure_catccos_env(args: argparse.Namespace) -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
    os.environ["VLLM_ASCEND_ENABLE_CATCCOS"] = "0" if args.disable_catccos else "1"
    os.environ["VLLM_ASCEND_ENABLE_CATCCOS_ALLGATHER_MATMUL"] = (
        "0" if args.disable_catccos else "1"
    )
    os.environ["VLLM_ASCEND_CATCCOS_ALLGATHER_MATMUL_PREFIXES"] = args.prefixes
    os.environ.setdefault("VLLM_ASCEND_CATCCOS_RUN_SMOKE_TEST", "0")
    if args.catccos_ops_so:
        os.environ["VLLM_ASCEND_CATCCOS_OPS_SO"] = args.catccos_ops_so
    elif not args.disable_catccos:
        raise ValueError(
            "--catccos-ops-so or VLLM_ASCEND_CATCCOS_OPS_SO is required "
            "when catccos is enabled"
        )


def run_dp_rank(
    *,
    args: argparse.Namespace,
    local_dp_rank: int,
    global_dp_rank: int,
    dp_master_ip: str,
    dp_master_port: int,
) -> None:
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(args.dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ.setdefault("MASTER_ADDR", dp_master_ip)
    os.environ.setdefault("MASTER_PORT", str(dp_master_port))

    configure_catccos_env(args)

    prompts = [f"[dp_rank={global_dp_rank}] {args.prompt}"]
    sampling_params = SamplingParams(temperature=0, max_tokens=args.max_tokens)

    print(
        "catccos phase2 worker config: "
        f"dp_rank={global_dp_rank} local_dp_rank={local_dp_rank} "
        f"dp_size={args.dp_size} tp_size={args.tp_size} "
        f"catccos_enabled={os.environ['VLLM_ASCEND_ENABLE_CATCCOS']} "
        f"prefixes={os.environ['VLLM_ASCEND_CATCCOS_ALLGATHER_MATMUL_PREFIXES']}",
        flush=True,
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp_size,
        dtype=args.dtype,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=True,
        trust_remote_code=args.trust_remote_code,
        additional_config={
            "finegrained_tp_config": {
                "mlp_tensor_parallel_size": args.dp_size,
            }
        },
    )
    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        print(
            f"DP rank {global_dp_rank}, prompt={output.prompt!r}, "
            f"generated={output.outputs[0].text!r}",
            flush=True,
        )

    sleep(3)
    del llm
    cleanup_env_and_memory()


def main() -> None:
    args = parse_args()
    if args.dp_size < 2:
        raise ValueError("Phase2 DP MLP TP verification requires --dp-size >= 2")
    if args.dp_size % args.node_size != 0:
        raise ValueError("--dp-size must be divisible by --node-size")
    configure_catccos_env(args)

    dp_master_ip = "127.0.0.1" if args.node_size == 1 else args.master_addr
    dp_master_port = get_open_port() if args.node_size == 1 else args.master_port
    if not dp_master_ip or not dp_master_port:
        raise ValueError(
            "--master-addr and --master-port are required for multi-node runs"
        )

    dp_per_node = args.dp_size // args.node_size
    procs: list[Process] = []
    for local_dp_rank, global_dp_rank in enumerate(
        range(
            args.node_rank * dp_per_node,
            (args.node_rank + 1) * dp_per_node,
        )
    ):
        proc = Process(
            target=run_dp_rank,
            kwargs={
                "args": args,
                "local_dp_rank": local_dp_rank,
                "global_dp_rank": global_dp_rank,
                "dp_master_ip": dp_master_ip,
                "dp_master_port": dp_master_port,
            },
        )
        proc.start()
        procs.append(proc)

    exit_code = 0
    for proc in procs:
        proc.join(timeout=900)
        if proc.exitcode is None:
            print(
                f"Killing process {proc.pid} that did not stop within 15 minutes.",
                flush=True,
            )
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
