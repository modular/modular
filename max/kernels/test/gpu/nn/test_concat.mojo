# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

from collections import OptionalReg

from buffer import NDBuffer
from buffer.dimlist import DimList
from gpu.host import DeviceContext
from nn.concat import (
    _concat_gpu,
    _concat_inner_most_single_dim,
    elementwise_epilogue_type,
)
from testing import assert_true

from utils import IndexList, StaticTuple


fn _create_buffer_host[
    rank: Int, dtype: DType
](dims: DimList) -> NDBuffer[dtype, rank, MutableAnyOrigin]:
    var total_size: Int = dims.product[rank]().get()
    var mem_ptr = UnsafePointer[Scalar[dtype]].alloc(total_size)
    var buffer = NDBuffer[dtype, rank](mem_ptr, dims)
    return buffer


fn _fill_buffer[
    rank: Int, dtype: DType
](buffer: NDBuffer[mut=True, dtype, rank]):
    for i in range(buffer.num_elements()):
        buffer.flatten()[i] = i


fn _fill_buffer[
    rank: Int, dtype: DType
](buffer: NDBuffer[mut=True, dtype, rank], val: Scalar[dtype]):
    for i in range(buffer.num_elements()):
        buffer.flatten()[i] = val


fn test_concat_4_inputs_rank5[test_epilogue: Bool](ctx: DeviceContext) raises:
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

    var total_size_inp: Int = input_shape.product[rank]().get()
    var input_0_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_1_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_2_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_3_device = ctx.enqueue_create_buffer[dtype](total_size_inp)

    var input_0_device_ref = NDBuffer[dtype, rank](
        input_0_device._unsafe_ptr(), input_shape
    )
    var input_1_device_ref = NDBuffer[dtype, rank](
        input_1_device._unsafe_ptr(), input_shape
    )
    var input_2_device_ref = NDBuffer[dtype, rank](
        input_2_device._unsafe_ptr(), input_shape
    )
    var input_3_device_ref = NDBuffer[dtype, rank](
        input_3_device._unsafe_ptr(), input_shape
    )

    ctx.enqueue_copy(input_0_device, input_0_host.data)
    ctx.enqueue_copy(input_1_device, input_1_host.data)
    ctx.enqueue_copy(input_2_device, input_2_host.data)
    ctx.enqueue_copy(input_3_device, input_3_host.data)

    var total_size_outp: Int = output_shape.product[rank]().get()
    var output_device = ctx.enqueue_create_buffer[dtype](total_size_outp)
    var output_device_ref = NDBuffer[dtype, rank](
        output_device._unsafe_ptr(), output_shape
    )

    alias B_SIZE = 32

    @parameter
    @always_inline
    @__copy_capture(output_device_ref)
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        output_device_ref.store[width=width](
            rebind[IndexList[rank]](indices),
            rebind[SIMD[dtype, width]](val + 1),
        )

    alias kernel = _concat_inner_most_single_dim[
        rank=rank,
        dtype=dtype,
        num_inputs=4,
        block_size=B_SIZE,
        epilogue_fn = OptionalReg[elementwise_epilogue_type](
            epilogue_plus_one
        ) if test_epilogue else None,
    ]

    @always_inline
    @__copy_capture(
        output_device_ref,
        input_0_device_ref,
        input_1_device_ref,
        input_2_device_ref,
        input_3_device_ref,
    )
    @parameter
    fn run_concat_inner_most_single_dim(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel](
            output_device_ref,
            StaticTuple[NDBuffer[dtype, rank, MutableAnyOrigin], 4](
                input_0_device_ref,
                input_1_device_ref,
                input_2_device_ref,
                input_3_device_ref,
            ),
            grid_dim=(d0 * d1 * d2 * d3 * d4 // B_SIZE),
            block_dim=(B_SIZE),
        )

    var nstime_kernel = ctx.execution_time[run_concat_inner_most_single_dim](1)
    print("concat_inner_most_single_dim time = ", nstime_kernel * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device_ref.bytecount() * 2 * 1e9 / (1024**3) / nstime_kernel,
        "GB/s",
    )

    var output_host = _create_buffer_host[rank, dtype](output_shape)
    ctx.enqueue_copy(output_host.data, output_device)

    fn validate_results() raises:
        var validTest = True
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        alias tail_val = 1 if test_epilogue else 0
                        var not_match_0 = (
                            output_host[i, j, k, l, 0]
                            != input_0_host[i, j, k, l, 0] + tail_val
                        )
                        var not_match_1 = (
                            output_host[i, j, k, l, 1]
                            != input_1_host[i, j, k, l, 0] + tail_val
                        )
                        var not_match_2 = (
                            output_host[i, j, k, l, 2]
                            != input_2_host[i, j, k, l, 0] + tail_val
                        )
                        var not_match_3 = (
                            output_host[i, j, k, l, 3]
                            != input_3_host[i, j, k, l, 0] + tail_val
                        )
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

        assert_true(validTest)

    validate_results()

    @always_inline
    @__copy_capture(
        output_device_ref,
        input_0_device_ref,
        input_1_device_ref,
        input_2_device_ref,
        input_3_device_ref,
    )
    @parameter
    fn run_concat_gpu(ctx: DeviceContext) raises:
        # uses default stream
        _concat_gpu[
            epilogue_fn = OptionalReg[elementwise_epilogue_type](
                epilogue_plus_one
            ) if test_epilogue else None
        ](
            output_device_ref,
            4,
            StaticTuple[NDBuffer[dtype, rank, MutableAnyOrigin], 4](
                input_0_device_ref,
                input_1_device_ref,
                input_2_device_ref,
                input_3_device_ref,
            ),
            ctx,
        )

    var nstime = ctx.execution_time[run_concat_gpu](1)
    print("concat_gpu time = ", nstime * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device_ref.bytecount() * 2 * 1e9 / (1024**3) / nstime,
        "GB/s",
    )

    ctx.enqueue_copy(output_host.data, output_device)

    validate_results()

    _ = input_0_device
    _ = input_1_device
    _ = input_2_device
    _ = input_3_device
    _ = output_device


def main():
    with DeviceContext() as ctx:
        test_concat_4_inputs_rank5[True](ctx)
        test_concat_4_inputs_rank5[False](ctx)
