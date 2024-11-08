# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V1 %s
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cuda %s

from asyncrt_test_utils import (
    create_test_device_context,
    expect_eq,
    is_v2_context,
)

from gpu.host import DeviceBuffer, DeviceContext, DeviceAttribute, DeviceStream
from gpu.host.nvidia_cuda import CUDA, CUcontext


fn _run_cuda_context(ctx: DeviceContext) raises:
    print("-")
    print("_run_cuda_context()")

    if not is_v2_context():
        print("Skipping test.")
        return

    var cuda_ctx: CUcontext = CUDA(ctx)
    print("CUcontext: " + str(cuda_ctx))


fn _run_cuda_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_cuda_stream()")

    if not is_v2_context():
        print("Skipping test.")
        return

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()
    var cuda_stream = CUDA(stream)
    print("CUstream: " + str(cuda_stream))


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_cuda_context(ctx)
    _run_cuda_stream(ctx)

    print("Done.")
