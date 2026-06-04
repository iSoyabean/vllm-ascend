#!/usr/bin/env python3
"""Smoke checks for catccos deep integration.

Usage after building/installing vllm-ascend in the NPU docker::

    PYTHONPATH=. python tests/manual/catccos_deep_integration_smoke.py

The default check imports ``vllm_ascend_catccos_C``, verifies that the
``_C_ascend::catccos_*`` dispatcher schemas are registered, and checks that
the MMAR Meta implementation returns the expected output shape. It does not
launch the real MMAR kernel.

To additionally verify that an NPU tensor call fails clearly before
``catccos_init``::

    PYTHONPATH=. python tests/manual/catccos_deep_integration_smoke.py \
        --run-uninitialized-call
"""

import argparse

import torch


def _import_extension() -> None:
    import vllm_ascend.vllm_ascend_catccos_C  # noqa: F401


def _require_op(name: str) -> None:
    qualified_name = f"_C_ascend::{name}"
    try:
        torch._C._dispatch_find_schema_or_throw(qualified_name, "")
    except RuntimeError as exc:
        raise AssertionError(f"{qualified_name} is not registered") from exc


def check_registration() -> None:
    _import_extension()
    for name in (
        "catccos_init",
        "catccos_matmul_allreduce",
        "catccos_finalize",
    ):
        _require_op(name)


def check_meta() -> None:
    a = torch.empty((3, 5), device="meta", dtype=torch.float16)
    b = torch.empty((5, 7), device="meta", dtype=torch.float16)
    out = torch.ops._C_ascend.catccos_matmul_allreduce(a, b, 2)
    assert out.device.type == "meta", out.device
    assert out.shape == (3, 7), out.shape
    assert out.dtype == torch.float16, out.dtype


def check_uninitialized_call() -> None:
    if not hasattr(torch, "npu") or not torch.npu.is_available():
        raise RuntimeError("NPU is not available; skip --run-uninitialized-call")
    a = torch.empty((3, 5), device="npu", dtype=torch.float16)
    b = torch.empty((5, 7), device="npu", dtype=torch.float16)
    try:
        torch.ops._C_ascend.catccos_matmul_allreduce(a, b, 2)
    except RuntimeError as exc:
        if "catccos is not initialized" in str(exc):
            return
        raise
    raise AssertionError("MMAR call succeeded before catccos_init")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-uninitialized-call",
        action="store_true",
        help="also call MMAR on NPU tensors and expect a pre-init error",
    )
    args = parser.parse_args()

    check_registration()
    check_meta()
    if args.run_uninitialized_call:
        check_uninitialized_call()
    print("catccos deep integration smoke checks passed")


if __name__ == "__main__":
    main()
