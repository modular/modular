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
from sys import size_of

from gpu.host import DeviceContext
from layout import UNKNOWN_VALUE, Layout, LayoutTensor, RuntimeLayout
from layout._fillers import arange
from nn.concat import (
    _concat_gpu,
    _concat_inner_most_single_dim,
    elementwise_epilogue_type,
)
from testing import assert_true

from utils import IndexList, StaticTuple
from utils.index import product


fn _create_buffer_host[
    rank: Int, dtype: DType
](dims: IndexList[rank]) -> LayoutTensor[
    dtype, Layout.row_major[rank](), MutableAnyOrigin
]:
    var total_size: Int = product(dims)
    var mem_ptr = UnsafePointer[Scalar[dtype]].alloc(total_size)
    alias layout = Layout.row_major[rank]()
    var buffer = LayoutTensor[dtype, layout, MutableAnyOrigin](
        mem_ptr, RuntimeLayout[layout].row_major(dims)
    )
    return buffer


fn test_concat_4_inputs_rank5[test_epilogue: Bool](ctx: DeviceContext) raises:
    print("== test_concat_4_inputs_rank5")

    alias rank = 5
    alias dtype = DType.float32

    alias d0 = 1
    alias d1 = 128
    alias d2 = 32
    alias d3 = 64
    alias d4 = 1

    var input_layout = Layout.row_major(d0, d1, d2, d3, d4)
    var output_layout = Layout.row_major(d0, d1, d2, d3, 4)
    var input_shape = IndexList[5](d0, d1, d2, d3, d4)
    var output_shape = IndexList[5](d0, d1, d2, d3, 4)

    var input_0_host = _create_buffer_host[rank, dtype](input_shape)
    var input_1_host = _create_buffer_host[rank, dtype](input_shape)
    var input_2_host = _create_buffer_host[rank, dtype](input_shape)
    var input_3_host = _create_buffer_host[rank, dtype](input_shape)

    arange(input_0_host)
    arange(input_1_host)
    arange(input_2_host)
    arange(input_3_host)

    var total_size_inp: Int = product(input_shape)
    var input_0_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_1_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_2_device = ctx.enqueue_create_buffer[dtype](total_size_inp)
    var input_3_device = ctx.enqueue_create_buffer[dtype](total_size_inp)

    alias layout = Layout.row_major[rank]()
    var input_0_device_ref = LayoutTensor[dtype, layout](
        input_0_device,
        RuntimeLayout[layout].row_major(input_shape),
    )
    var input_1_device_ref = LayoutTensor[dtype, layout](
        input_1_device,
        RuntimeLayout[layout].row_major(input_shape),
    )
    var input_2_device_ref = LayoutTensor[dtype, layout](
        input_2_device,
        RuntimeLayout[layout].row_major(input_shape),
    )
    var input_3_device_ref = LayoutTensor[dtype, layout](
        input_3_device,
        RuntimeLayout[layout].row_major(input_shape),
    )

    ctx.enqueue_copy(input_0_device, input_0_host.ptr)
    ctx.enqueue_copy(input_1_device, input_1_host.ptr)
    ctx.enqueue_copy(input_2_device, input_2_host.ptr)
    ctx.enqueue_copy(input_3_device, input_3_host.ptr)

    var total_size_outp: Int = product(output_shape)
    var output_device = ctx.enqueue_create_buffer[dtype](total_size_outp)
    var output_device_ref = LayoutTensor[dtype, layout](
        output_device,
        RuntimeLayout[layout].row_major(output_shape),
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
        output_layout=layout,
        inputs_layout=layout,
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
        ctx.enqueue_function_checked[kernel, kernel](
            output_device_ref.as_any_origin(),
            StaticTuple[LayoutTensor[dtype, layout, MutableAnyOrigin], 4](
                input_0_device_ref.as_any_origin(),
                input_1_device_ref.as_any_origin(),
                input_2_device_ref.as_any_origin(),
                input_3_device_ref.as_any_origin(),
            ),
            grid_dim=(d0 * d1 * d2 * d3 * d4 // B_SIZE),
            block_dim=(B_SIZE),
        )

    var nstime_kernel = ctx.execution_time[run_concat_inner_most_single_dim](1)
    print("concat_inner_most_single_dim time = ", nstime_kernel * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device_ref.size()
        * size_of[UInt8]()
        * 2
        * 1e9
        / (1024**3)
        / nstime_kernel,
        "GB/s",
    )

    var output_host = _create_buffer_host[rank, dtype](output_shape)
    ctx.enqueue_copy(output_host.ptr, output_device)

    fn validate_results() raises:
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
                            assert_true(False, msg="❌ Test failed!")
                            return

        print("✅ Test passed!")

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
            output_device_ref.as_any_origin(),
            4,
            StaticTuple[LayoutTensor[dtype, layout, MutableAnyOrigin], 4](
                input_0_device_ref.as_any_origin(),
                input_1_device_ref.as_any_origin(),
                input_2_device_ref.as_any_origin(),
                input_3_device_ref.as_any_origin(),
            ),
            ctx,
        )

    var nstime = ctx.execution_time[run_concat_gpu](1)
    print("concat_gpu time = ", nstime * 1e-6, " ms")
    print(
        "transfer rate = ",
        output_device_ref.size()
        * size_of[UInt8]()
        * 2
        * 1e9
        / (1024**3)
        / nstime,
        "GB/s",
    )

    ctx.enqueue_copy(output_host.ptr, output_device)

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
