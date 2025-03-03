# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-no-debug-no-assert %s

import time
from collections import InlineArray
from math import floor
from sys import sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.all_reduce import MAX_GPUS, MAX_NUM_BLOCKS_DEFAULT, Signal, all_reduce
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal


fn _pretty_print_float(val: Float64) -> String:
    """This converts the float value to a string, but omits the fractional part
    if not needed (e.g. prints 2 instead of 2.0).
    """
    if Float64(floor(val)) == val:
        return String(Int(val))
    return String(val)


fn _human_memory(size: Int) -> String:
    alias KB = 1024
    alias MB = KB * KB
    alias GB = MB * KB

    if size >= GB:
        return _pretty_print_float(Float64(size) / GB) + "GB"

    if size >= MB:
        return _pretty_print_float(Float64(size) / MB) + "MB"

    if size >= KB:
        return _pretty_print_float(Float64(size) / KB) + "KB"

    return String(size) + "B"


fn all_reduce_test[
    type: DType, rank: Int, ngpus: Int
](list_of_ctx: List[DeviceContext], length: Int) raises:
    alias num_warmups = 5
    alias num_iters = 100
    alias max_num_blocks = MAX_NUM_BLOCKS_DEFAULT

    constrained[ngpus in (1, 2, 4, 8), "ngpus must be 1, 2, 4, or 8"]()
    constrained[rank == 1, "this test code currently assumes rank 1"]()

    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[type]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[type]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[type]]](capacity=ngpus)

    # Create signal buffers for synchronization
    var signal_buffers = List[DeviceBuffer[DType.uint8]](capacity=ngpus)
    var rank_sigs = InlineArray[
        UnsafePointer[Signal[max_num_blocks]], MAX_GPUS
    ](UnsafePointer[Signal[max_num_blocks]]())

    # Set up temp buffers for GPUs to reduce-scatter into / all-gather from.
    var temp_buffer_num_bytes = ngpus * sizeof[type]() * length

    # Initialize buffers for each GPU
    @parameter
    for i in range(ngpus):
        # Create and store device buffers
        in_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[type](length))
        out_bufs_list.append(list_of_ctx[i].enqueue_create_buffer[type](length))

        # Create and initialize host buffers
        var host_buffer = UnsafePointer[Scalar[type]].alloc(length)
        host_buffers.append(host_buffer)

        # Initialize host buffer with values (i + 1).0
        var host_nd_buf = NDBuffer[type, rank](host_buffer, DimList(length))
        host_nd_buf.fill(Scalar[type](i + 1))

        # Create and initialize signal buffers
        signal_buffers.append(
            list_of_ctx[i].create_buffer_sync[DType.uint8](
                sizeof[Signal[max_num_blocks]]() + temp_buffer_num_bytes
            )
        )
        list_of_ctx[i].enqueue_memset[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = (
            signal_buffers[i].unsafe_ptr().bitcast[Signal[max_num_blocks]]()
        )

        # Copy data to device
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create and initialize input and output buffers.
    var in_bufs = InlineArray[NDBuffer[type, rank], ngpus](
        NDBuffer[type, rank]()
    )
    var out_bufs = InlineArray[NDBuffer[type, rank], ngpus](
        NDBuffer[type, rank]()
    )

    for i in range(ngpus):
        in_bufs[i] = NDBuffer[type, rank](
            in_bufs_list[i].unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[type, rank](
            out_bufs_list[i].unsafe_ptr(), DimList(length)
        )

    # Synchronize all devices to ensure setup has propagated.
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Warm up.
    for _ in range(num_warmups):
        all_reduce(list_of_ctx, in_bufs, out_bufs, rank_sigs)

    # Synchronize all devices.
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    # Perform a benchmarked allreduce.
    start_t = time.perf_counter_ns()

    @parameter
    for _ in range(num_iters):
        all_reduce(list_of_ctx, in_bufs, out_bufs, rank_sigs)

    # Synchronize all devices.
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    end_t = time.perf_counter_ns()

    # Quick and dirty benchmark since benchmark module doesn't support
    # multi-device contexts.
    print("Time taken (ms):", (end_t - start_t) / (1_000_000 * num_iters))

    # Copy results back and verify
    var expected_sum = Scalar[type](0)

    @parameter
    for i in range(ngpus):
        expected_sum += i + 1
        list_of_ctx[i].enqueue_copy(host_buffers[i], out_bufs_list[i])

    # Verify results
    @parameter
    for i in range(ngpus):
        for j in range(length):
            try:
                assert_almost_equal(host_buffers[i][j], expected_sum)
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_buffers[i][j])
                print("Expected:", expected_sum)
                raise e

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
    _ = signal_buffers^


fn _get_test_str[type: DType](ngpus: Int, length: Int) -> String:
    return String(
        "====allreduce-",
        type,
        "-",
        ngpus,
        "-",
        _human_memory(sizeof[type]() * length),
    )


def main():
    # Test configurations covering edge cases
    # fmt: off
    alias test_lengths = (
        8 * 1024,           # Small latency bound.
        128 * 1024,         # Larger latency bound.
        256 * 1024,         # Smallest bandwidth bound.
        16 * 1024 * 1024,   # Bandwidth bound.
        64 * 1024 * 1024,   # Bandwidth bound: 8192 chunk size at dim = 8192.
    )
    # fmt: on

    # Test hyperparameters.
    alias test_dtypes = (DType.bfloat16, DType.float32)
    alias test_gpu_counts = (2, 4, 8)

    # Run tests for each configuration.
    @parameter
    for gpu_idx in range(len(test_gpu_counts)):
        alias num_gpus = test_gpu_counts[gpu_idx]
        if DeviceContext.number_of_devices() < num_gpus:
            break

        # Create GPU context.
        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        # Test all cases for this test configuration.
        @parameter
        for dtype_idx in range(len(test_dtypes)):
            alias dtype = test_dtypes[dtype_idx]

            @parameter
            for length_idx in range(len(test_lengths)):
                alias length = test_lengths[length_idx]

                # Generate descriptive test name.
                print(_get_test_str[dtype](num_gpus, length))

                # Execute test.
                all_reduce_test[type=dtype, rank=1, ngpus=num_gpus](ctx, length)
