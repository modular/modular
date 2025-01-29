# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: NVIDIA-GPU
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext, DeviceStream
from gpu.host._nvidia_cuda import CUDA, CUcontext


fn _run_cuda_context(ctx: DeviceContext) raises:
    print("-")
    print("_run_cuda_context()")

    var cuda_ctx: CUcontext = CUDA(ctx)
    print("CUcontext: " + String(cuda_ctx))


fn _run_cuda_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_cuda_stream()")

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()
    var cuda_stream = CUDA(stream)
    print("CUstream: " + String(cuda_stream))


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_cuda_context(ctx)
    _run_cuda_stream(ctx)

    print("Done.")
