# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=cpu %s
# RUN: %mojo-no-debug -D MODULAR_ASYNCRT_DEVICE_CONTEXT_V2=gpu %s

from asyncrt_test_utils import create_test_device_context, expect_eq
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer


fn _run_memcpy(ctx: DeviceContext, length: Int) raises:
    print("-")
    print("_run_memcpy(", length, ")")

    var in_host = ctx.malloc_host[Float32](length)
    var out_host = UnsafePointer[Float32].alloc(length)
    var in_dev = ctx.create_buffer_sync[DType.float32](length)
    var out_dev = ctx.create_buffer_sync[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.copy_sync(in_dev, in_host)
    ctx.copy_sync(out_dev, in_dev)
    ctx.copy_sync(out_host, out_dev)

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(out_host[i], i, "at index ", i, " the value is ", out_host[i])

    out_host.free()
    ctx.free_host(in_host)


fn _run_memcpy_async(ctx: DeviceContext, length: Int, use_context: Bool) raises:
    print("-")
    print("_run_memcpy_async(", length, ")")

    var in_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.float32](length)
    var in_dev = ctx.enqueue_create_buffer[DType.float32](length)
    var out_dev = ctx.enqueue_create_buffer[DType.float32](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    ctx.enqueue_copy(in_dev, in_host.unsafe_ptr())
    if use_context:
        ctx.enqueue_copy(out_dev, in_dev)
    else:
        in_dev.enqueue_copy_to(out_dev)
    out_dev.enqueue_copy_to(out_host)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(out_host[i], i, "at index ", i, " the value is ", out_host[i])


fn _run_sub_memcpy_async(ctx: DeviceContext, length: Int) raises:
    print("-")
    print("_run_sub_memcpy_async(", length, ")")

    var half_length = length // 2

    var in_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var in_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var first_out_dev = out_dev.create_sub_buffer[DType.int64](0, half_length)
    var second_out_dev = out_dev.create_sub_buffer[DType.int64](
        half_length, half_length
    )

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(in_dev)
    ctx.enqueue_copy(out_dev, in_dev)

    # Swap halves on copy back.
    # TODO(iposva): Investigate failure with this code:
    # second_out_dev.enqueue_copy_to(out_host)
    ctx.enqueue_copy(out_host.unsafe_ptr(), second_out_dev)
    ctx.enqueue_copy(out_host.unsafe_ptr().offset(half_length), first_out_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        var expected: Int
        if i < half_length:
            expected = i + half_length
        else:
            expected = i - half_length
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], expected, "at index ", i, " the value is ", out_host[i]
        )


fn _run_fake_memcpy_async(
    ctx: DeviceContext, length: Int, use_take_ptr: Bool
) raises:
    print("-")
    print("_run_fake_memcpy_async(", length, ", take_ptr = ", use_take_ptr, ")")

    var half_length = length // 2

    var in_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var out_host = ctx.enqueue_create_host_buffer[DType.int64](length)
    var in_dev = ctx.enqueue_create_buffer[DType.int64](length)
    var out_dev = ctx.enqueue_create_buffer[DType.int64](length)

    # Initialize the input and outputs with known values.
    for i in range(length):
        in_host[i] = i
        out_host[i] = length + i

    # Copy to and from device buffers.
    in_host.enqueue_copy_to(in_dev)
    ctx.enqueue_copy(out_dev, in_dev)

    var out_ptr: UnsafePointer[Int64]
    if use_take_ptr:
        out_ptr = out_dev.take_ptr()
    else:
        out_ptr = out_dev.unsafe_ptr()

    var first_out_dev = DeviceBuffer[DType.int64](
        ctx, out_ptr, half_length, owning=use_take_ptr
    )
    var interior_out_ptr: UnsafePointer[Int64] = out_ptr.offset(half_length)
    var second_out_dev = DeviceBuffer[DType.int64](
        ctx, interior_out_ptr, half_length, owning=False
    )

    # Swap halves on copy back.
    # TODO(iposva): Investigate failure with this code:
    # second_out_dev.enqueue_copy_to(out_host)
    ctx.enqueue_copy(out_host.unsafe_ptr(), second_out_dev)
    ctx.enqueue_copy(out_host.unsafe_ptr().offset(half_length), first_out_dev)

    # Wait for the copies to be completed.
    ctx.synchronize()

    for i in range(length):
        var expected: Int
        if i < half_length:
            expected = i + half_length
        else:
            expected = i - half_length
        if i < 10:
            print("at index", i, "the value is", out_host[i])
        expect_eq(
            out_host[i], expected, "at index ", i, " the value is ", out_host[i]
        )


fn main() raises:
    var ctx = create_test_device_context(buffer_cache_size=6 * 1024 * 1024)

    print("-------")
    print("Running test_copies(" + ctx.name() + "):")

    alias one_mb = 1024 * 1024

    _run_memcpy(ctx, 64)
    _run_memcpy(ctx, one_mb)

    _run_memcpy_async(ctx, 64, True)
    _run_memcpy_async(ctx, one_mb, True)
    _run_memcpy_async(ctx, 64, False)
    _run_memcpy_async(ctx, one_mb, False)

    _run_sub_memcpy_async(ctx, 64)

    _run_fake_memcpy_async(ctx, 64, False)
    # _run_fake_memcpy_async(ctx, 64, True)

    print("Done.")
