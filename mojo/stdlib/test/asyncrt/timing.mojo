# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceContext


@parameter
fn _timed_iter_func(context: DeviceContext, iter: Int) raises:
    alias length = 64

    var in_host = context.malloc_host[Float32](length)
    var out_host = context.malloc_host[Float32](length)
    var in_dev = context.enqueue_create_buffer[DType.float32](length)
    var out_dev = context.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i + iter
        out_host[i] = length + i

    # Copy to and from device buffers.
    context.enqueue_copy_to_device(in_dev, in_host)
    context.enqueue_copy_device_to_device(out_dev, in_dev)
    context.enqueue_copy_from_device(out_host, out_dev)

    # Wait for the copies to be completed.
    context.synchronize()

    for i in range(length):
        expect_eq(
            out_host[i],
            i + iter,
            "at index " + str(i) + " the value is " + str(out_host[i]),
        )

    context.free_host(out_host)
    context.free_host(in_host)


@parameter
fn _timed_func(context: DeviceContext) raises:
    _timed_iter_func(context, 2)


fn main() raises:
    var ctx = create_test_device_context()
    print("Running test_timing(" + ctx.name() + "):")

    # Measure the time to run the function 100 times.
    var elapsed_time = ctx.execution_time[_timed_func](100)
    print("Elapsed time for _timed_func: " + str(elapsed_time / 1e9) + "s")

    elapsed_time = ctx.execution_time_iter[_timed_iter_func](100)
    print("Elapsed time for _timed_iter_func: " + str(elapsed_time / 1e9) + "s")

    print("Done.")
