# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: AMD-GPU
# COM: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceAttribute, DeviceBuffer, DeviceContext, DeviceStream
from gpu.host._amdgpu_hip import HIP, hipDevice_t


fn _run_hip_context(ctx: DeviceContext) raises:
    print("-")
    print("_run_hip_context()")

    var hip_ctx: hipDevice_t = HIP(ctx)
    print("hipDevice_t: " + String(hip_ctx))


fn _run_hip_stream(ctx: DeviceContext) raises:
    print("-")
    print("_run_hip_stream()")

    print("Getting the stream.")
    var stream = ctx.stream()
    print("Synchronizing on `stream`.")
    stream.synchronize()
    var hip_stream = HIP(stream)
    print("hipStream_t: " + String(hip_stream))


fn main() raises:
    var ctx = create_test_device_context()
    print("-------")
    print("Running test_smoke(" + ctx.name() + "):")

    _run_hip_context(ctx)
    _run_hip_stream(ctx)

    print("Done.")
