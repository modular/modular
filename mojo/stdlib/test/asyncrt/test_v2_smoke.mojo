# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s | FileCheck %s

from gpu import *
from gpu.host import DeviceContextVariant
from sys.param_env import is_defined, env_get_string


fn test_smoke(ctx: DeviceContextVariant) raises:
    alias length = 64

    var in_host = ctx.malloc_host[Float32](length)
    var out_host = ctx.malloc_host[Float32](length)
    var in_dev = ctx.create_buffer[DType.float32](length)
    var out_dev = ctx.create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.enqueue_copy_to_device(in_dev, in_host)
    ctx.enqueue_copy_device_to_device(out_dev, in_dev)
    ctx.enqueue_copy_from_device(out_host, out_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    # CHECK: at index 0 the value is 0.0
    # CHECK: at index 1 the value is 1.0
    # CHECK: at index 2 the value is 2.0
    # CHECK: at index 3 the value is 3.0
    # CHECK: at index 4 the value is 4.0
    # CHECK: at index 5 the value is 5.0
    # CHECK: at index 6 the value is 6.0
    # CHECK: at index 7 the value is 7.0
    # CHECK: at index 8 the value is 8.0
    # CHECK: at index 9 the value is 9.0
    for i in range(10):
        print("at index", i, "the value is", out_host[i])

    ctx.free_host(out_host)
    ctx.free_host(in_host)


def main():
    # Create an instance of the DeviceContextVariant
    var ctx: DeviceContextVariant

    print(
        "Using DeviceContextVariant for "
        + env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2", "V1"]()
    )

    @parameter
    if is_defined["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]():
        ctx = DeviceContextVariant(
            env_get_string["MODULAR_ASYNCRT_DEVICE_CONTEXT_V2"]()
        )
    else:
        ctx = DeviceContextVariant()

    # Execute our test with the context
    test_smoke(ctx)
