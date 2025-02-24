# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# FIXME: KERN-1377
# UNSUPPORTED: AMD-GPU
# RUN: %mojo-build-no-debug-no-assert %s

import time
from collections import InlineArray
from math import floor
from sys import sizeof

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.all_reduce import MAX_GPUS, Signal, all_reduce
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal

from utils.index import IndexList, StaticTuple

from sys import env_get_dtype, env_get_int
from internal_utils import arg_parse
from testing import assert_true

from benchmark import (
    Bench,
    BenchConfig,
    Bencher,
    BenchId,
    BenchMetric,
    ThroughputMeasure,
    keep,
)


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


# TODO: convert 'ngpus' to runtime variable
fn bench_reduce[
    type: DType, rank: Int, ngpus: Int, max_num_blocks: Int
](mut m: Bench, list_of_ctx: List[DeviceContext], num_bytes: Int) raises:
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
    var temp_buffer_num_bytes = ngpus * num_bytes
    var length = num_bytes // sizeof[type]()

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
        list_of_ctx[i].memset_sync[DType.uint8](signal_buffers[i], 0)
        rank_sigs[i] = (
            signal_buffers[i].unsafe_ptr().bitcast[Signal[max_num_blocks]]()
        )

        # Copy data to device
        list_of_ctx[i].enqueue_copy_to_device(in_bufs_list[i], host_buffers[i])

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

    @parameter
    @always_inline
    fn bench_iter(mut b: Bencher) raises:
        @parameter
        @always_inline
        fn call_fn() raises:
            # TODO: add multi-device (context) iter_custum function to bencher
            all_reduce(list_of_ctx, in_bufs, out_bufs, rank_sigs)

            # Synchronize all devices.
            @parameter
            for i in range(ngpus):
                list_of_ctx[i].synchronize()

        b.iter[call_fn]()

    var name = String(_get_test_str[type](ngpus, length))
    m.bench_function[bench_iter](
        BenchId(name),
        # add data movement to measures
        ThroughputMeasure(BenchMetric.bytes, num_bytes),
    )

    # Copy results back and verify
    var expected_sum = Scalar[type](0)

    @parameter
    for i in range(ngpus):
        expected_sum += i + 1
        list_of_ctx[i].enqueue_copy_from_device(
            host_buffers[i], out_bufs_list[i]
        )

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


fn _get_test_str[type: DType](ngpus: Int, num_bytes: Int) -> String:
    return String("allreduce-", type, "-", ngpus, "-", _human_memory(num_bytes))


def main():
    var num_bytes = arg_parse("num_bytes", 16 * 1024)

    alias dtype = env_get_dtype["dtype", DType.bfloat16]()
    alias num_gpus = env_get_int["num_gpus", 2]()
    alias rank = env_get_int["rank", 1]()
    # Force passing `max_num_blocks` explicitly.
    alias max_num_blocks = env_get_int["TUNE_MAX_NUM_BLOCKS", -1]()

    assert_true(DeviceContext.number_of_devices() >= num_gpus)
    assert_true(num_bytes % sizeof[dtype]() == 0)

    # Create GPU context.
    var ctx = List[DeviceContext]()
    for i in range(num_gpus):
        ctx.append(DeviceContext(device_id=i))

    # Generate descriptive test name.
    print(_get_test_str[dtype](num_gpus, num_bytes))

    var m = Bench()

    bench_reduce[
        type=dtype, rank=rank, ngpus=num_gpus, max_num_blocks=max_num_blocks
    ](m, ctx, num_bytes)

    m.dump_report()
