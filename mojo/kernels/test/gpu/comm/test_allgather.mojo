# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s


import time
from collections import InlineArray
from math import floor
from sys import sizeof
from utils import IndexList, StaticTuple

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.comm.allgather import allgather
from gpu.host import DeviceBuffer, DeviceContext
from memory import UnsafePointer
from testing import assert_almost_equal, assert_equal


fn all_gather_test[
    type: DType, rank: Int, ngpus: Int
](list_of_ctx: List[DeviceContext], lengths: List[Int]) raises:
    # Create device buffers for all GPUs
    var in_bufs_list = List[DeviceBuffer[type]](capacity=ngpus)
    var out_bufs_list = List[DeviceBuffer[type]](capacity=ngpus)
    var host_buffers = List[UnsafePointer[Scalar[type]]](capacity=ngpus)
    var host_output_buffers = List[UnsafePointer[Scalar[type]]](capacity=ngpus)

    var total_length = 0

    @parameter
    for i in range(ngpus):
        total_length += lengths[i]

    var expected_output = UnsafePointer[Scalar[type]].alloc(total_length)
    var start_index = 0

    # Initialize buffers for each GPU
    @parameter
    for i in range(ngpus):
        var length = lengths[i]

        # Create and store device buffers
        in_bufs_list.append(list_of_ctx[i].create_buffer_sync[type](length))
        out_bufs_list.append(
            list_of_ctx[i].create_buffer_sync[type](total_length)
        )

        # Create and initialize host buffers
        var host_buffer = UnsafePointer[Scalar[type]].alloc(length)
        host_buffers.append(host_buffer)
        var host_output_buffer = UnsafePointer[Scalar[type]].alloc(total_length)
        host_output_buffers.append(host_output_buffer)

        # Initialize host buffer with values so that the expected allgather
        # output is range(total_length).
        var host_nd_buf = NDBuffer[type, rank](host_buffer, DimList(length))
        for j in range(length):
            var element = Scalar[type](start_index + j)
            host_nd_buf[j] = element
            expected_output[start_index + j] = element
        start_index += length

        # Copy data to device
        list_of_ctx[i].enqueue_copy(in_bufs_list[i], host_buffers[i])

    # Create and initialize input and output buffers.
    var in_bufs = InlineArray[NDBuffer[type, rank, MutableAnyOrigin], ngpus](
        NDBuffer[type, rank, MutableAnyOrigin]()
    )
    var out_bufs = InlineArray[NDBuffer[type, rank, MutableAnyOrigin], ngpus](
        NDBuffer[type, rank, MutableAnyOrigin]()
    )

    @parameter
    for i in range(ngpus):
        var length = lengths[i]
        in_bufs[i] = NDBuffer[type, rank](
            in_bufs_list[i]._unsafe_ptr(), DimList(length)
        )
        out_bufs[i] = NDBuffer[type, rank](
            out_bufs_list[i]._unsafe_ptr(), DimList(total_length)
        )

    # Perform allgather
    allgather(in_bufs, out_bufs, list_of_ctx)

    # Synchronize all devices
    @parameter
    for i in range(ngpus):
        list_of_ctx[i].synchronize()

    @parameter
    for i in range(ngpus):
        list_of_ctx[i].enqueue_copy(host_output_buffers[i], out_bufs_list[i])

    # Verify results
    @parameter
    for i in range(ngpus):
        for j in range(total_length):
            try:
                assert_equal(host_output_buffers[i][j], expected_output[j])
            except e:
                print("Verification failed at GPU", i, "index", j)
                print("Value:", host_output_buffers[i][j])
                print("Expected:", expected_output[j])
                raise e

    # Cleanup
    for i in range(ngpus):
        host_buffers[i].free()
        host_output_buffers[i].free()


def main():
    # Test configurations
    alias test_lengths = (
        List[Int](8 * 1024, 8 * 1024),
        List[Int](128 * 1024, 8 * 1024),
        List[Int](8 * 1024, 256 * 1024),
        List[Int](8 * 1024, 8 * 1024, 8 * 1024, 8 * 1024),
        List[Int](128 * 1024, 256 * 1024, 8 * 1024, 64 * 1024),
    )

    @parameter
    for test_idx in range(len(test_lengths)):
        alias lengths = test_lengths[test_idx]
        alias num_gpus = len(lengths)

        if DeviceContext.number_of_devices() < num_gpus:
            continue

        var ctx = List[DeviceContext]()
        for i in range(num_gpus):
            ctx.append(DeviceContext(device_id=i))

        all_gather_test[DType.bfloat16, rank=1, ngpus=num_gpus](ctx, lengths)
