# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: has_cuda_device
# RUN: %mojo %s | FileCheck %s

from sys import argv
from time import time_function as time_function_sync

from algorithm.functional import _get_start_indices_of_nth_subvolume
from buffer import NDBuffer
from buffer.list import DimList
from gpu import BlockIdx, ThreadIdx
from gpu.host import Context, Function, Stream
from gpu.host.event import time_function as time_function_cuda
from gpu.host.memory import (
    _copy_device_to_host,
    _copy_host_to_device,
    _free,
    _malloc,
    _memset,
)
from gpu.host.sync import synchronize
from nn.concat import _concat_gpu, _concat_inner_most_single_dim
from utils import StaticTuple


fn _create_buffer_host[
    rank: Int, dtype: DType
](dims: DimList) -> NDBuffer[dtype, rank]:
    var total_size: Int = dims.product[rank]().value.value()
    var mem_ptr = DTypePointer[dtype].alloc(total_size)
    var buffer = NDBuffer[dtype, rank](mem_ptr, dims)
    return buffer


fn _create_buffer_device[
    rank: Int, dtype: DType
](dims: DimList) raises -> NDBuffer[dtype, rank]:
    var total_size: Int = dims.product[rank]().value.value()
    var mem_ptr = _malloc[dtype](total_size)
    var buffer = NDBuffer[dtype, rank](mem_ptr, dims)
    return buffer


fn _fill_buffer[rank: Int, dtype: DType](buffer: NDBuffer[dtype, rank]):
    for i in range(buffer.num_elements()):
        buffer.flatten()[i] = i


fn _fill_buffer[
    rank: Int, dtype: DType
](buffer: NDBuffer[dtype, rank], val: Scalar[dtype],):
    for i in range(buffer.num_elements()):
        buffer.flatten()[i] = val


fn test_concat_4_inputs_rank5() raises:
    print("== test_concat_4_inputs_rank5")

    alias rank = 5
    alias dtype = DType.float32

    alias d0 = 1
    alias d1 = 128
    alias d2 = 32
    alias d3 = 64
    alias d4 = 1

    var input_shape = DimList(d0, d1, d2, d3, d4)
    var output_shape = DimList(d0, d1, d2, d3, 4)

    var input_0_host = _create_buffer_host[rank, dtype](input_shape)
    var input_1_host = _create_buffer_host[rank, dtype](input_shape)
    var input_2_host = _create_buffer_host[rank, dtype](input_shape)
    var input_3_host = _create_buffer_host[rank, dtype](input_shape)

    _fill_buffer(input_0_host)
    _fill_buffer(input_1_host)
    _fill_buffer(input_2_host)
    _fill_buffer(input_3_host)

    var input_0_device = _create_buffer_device[rank, dtype](input_shape)
    var input_1_device = _create_buffer_device[rank, dtype](input_shape)
    var input_2_device = _create_buffer_device[rank, dtype](input_shape)
    var input_3_device = _create_buffer_device[rank, dtype](input_shape)

    _copy_host_to_device(
        input_0_device.data, input_0_host.data, input_0_host.size()
    )
    _copy_host_to_device(
        input_1_device.data, input_1_host.data, input_1_host.size()
    )
    _copy_host_to_device(
        input_2_device.data, input_2_host.data, input_2_host.size()
    )
    _copy_host_to_device(
        input_3_device.data, input_3_host.data, input_3_host.size()
    )

    var output_device = _create_buffer_device[rank, dtype](output_shape)

    alias B_SIZE = 32

    var func = Function[
        _concat_inner_most_single_dim[
            rank=rank, type=dtype, num_inputs=4, block_size=B_SIZE
        ]
    ]()

    var stream = Stream()

    @always_inline
    @parameter
    fn run_concat_inner_most_single_dim(stream: Stream) raises:
        func(
            output_device,
            StaticTuple[NDBuffer[dtype, rank], 4](
                input_0_device,
                input_1_device,
                input_2_device,
                input_3_device,
            ),
            stream=stream,
            grid_dim=(d0 * d1 * d2 * d3 * d4 // B_SIZE),
            block_dim=(B_SIZE),
        )

    var nstime_kernel = time_function_cuda[run_concat_inner_most_single_dim](
        stream
    )
    print("concat_inner_most_single_dim time = ", nstime_kernel * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device.bytecount() * 2 * 1e9 / (1024**3) / nstime_kernel,
        "GB/s",
    )

    var output_host = _create_buffer_host[rank, dtype](output_shape)
    _copy_device_to_host(
        output_host.data, output_device.data, output_device.size()
    )
    _memset(output_device.data, 0, output_device.num_elements())

    # CHECK: Test passed
    fn validate_results():
        var validTest = True
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        var not_match_0 = output_host[
                            i, j, k, l, 0
                        ] != input_0_host[i, j, k, l, 0]
                        var not_match_1 = output_host[
                            i, j, k, l, 1
                        ] != input_1_host[i, j, k, l, 0]
                        var not_match_2 = output_host[
                            i, j, k, l, 2
                        ] != input_2_host[i, j, k, l, 0]
                        var not_match_3 = output_host[
                            i, j, k, l, 3
                        ] != input_3_host[i, j, k, l, 0]
                        if (
                            not_match_0
                            or not_match_1
                            or not_match_2
                            or not_match_3
                        ):
                            validTest = False
        if not validTest:
            print("❌ Test failed!")
            return
        else:
            print("✅ Test passed!")

    validate_results()

    @always_inline
    @parameter
    fn run_concat_gpu():
        try:
            # uses default stream
            _concat_gpu(
                output_device,
                4,
                StaticTuple[NDBuffer[dtype, rank], 4](
                    input_0_device,
                    input_1_device,
                    input_2_device,
                    input_3_device,
                ),
            )
        except e:
            abort(e)

    var nstime = time_function_sync[run_concat_gpu]()
    print("concat_gpu time = ", nstime * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device.bytecount() * 2 * 1e9 / (1024**3) / nstime,
        "GB/s",
    )

    _copy_device_to_host(
        output_host.data, output_device.data, output_device.size()
    )

    # CHECK: Test passed
    validate_results()

    _free(input_0_device.data)
    _free(input_1_device.data)
    _free(input_2_device.data)
    _free(input_3_device.data)
    _free(output_device.data)


fn main() raises:
    with Context() as ctx:
        test_concat_4_inputs_rank5()
