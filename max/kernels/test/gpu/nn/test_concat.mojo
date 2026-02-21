# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
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

from collections import Optional
from sys import size_of

from gpu.host import DeviceContext, HostBuffer
from layout._coord import Coord, CoordLike, Idx
from layout._layout import row_major
from layout._tile_tensor import TileTensor
from nn.concat import (
    _concat_gpu,
    _concat_inner_most_single_dim,
    elementwise_epilogue_type,
)
from testing import assert_true

from utils import IndexList, StaticTuple
from utils.index import product


fn test_concat_4_inputs_rank5[test_epilogue: Bool](ctx: DeviceContext) raises:
    print("== test_concat_4_inputs_rank5")

    comptime rank = 5
    comptime dtype = DType.float32

    comptime d0 = 1
    comptime d1 = 128
    comptime d2 = 32
    comptime d3 = 64
    comptime d4 = 1

    comptime input_layout = row_major[d0, d1, d2, d3, d4]()
    comptime output_layout = row_major[d0, d1, d2, d3, 4]()

    # Create host buffers
    var input_0_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_1_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_2_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var input_3_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        input_layout.product()
    )
    var output_host_buffer = ctx.enqueue_create_host_buffer[dtype](
        output_layout.product()
    )
    ctx.synchronize()

    # Create TileTensors from host buffers and fill with arange pattern
    var input_0_host = TileTensor(input_0_host_buffer, input_layout)
    var input_1_host = TileTensor(input_1_host_buffer, input_layout)
    var input_2_host = TileTensor(input_2_host_buffer, input_layout)
    var input_3_host = TileTensor(input_3_host_buffer, input_layout)

    # Fill with arange pattern
    for i in range(input_layout.product()):
        input_0_host_buffer[i] = Float32(i)
        input_1_host_buffer[i] = Float32(i)
        input_2_host_buffer[i] = Float32(i)
        input_3_host_buffer[i] = Float32(i)

    # Create device buffers
    var input_0_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_1_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_2_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var input_3_device_buffer = ctx.enqueue_create_buffer[dtype](
        input_layout.product()
    )
    var output_device_buffer = ctx.enqueue_create_buffer[dtype](
        output_layout.product()
    )

    # Copy host to device
    ctx.enqueue_copy(input_0_device_buffer, input_0_host_buffer)
    ctx.enqueue_copy(input_1_device_buffer, input_1_host_buffer)
    ctx.enqueue_copy(input_2_device_buffer, input_2_host_buffer)
    ctx.enqueue_copy(input_3_device_buffer, input_3_host_buffer)
    ctx.synchronize()

    # Create TileTensors from device buffers using dynamic layouts
    # Use Coord(IndexList) pattern that benchmark uses to create runtime layouts
    var input_shape = IndexList[rank](d0, d1, d2, d3, d4)
    var output_shape = IndexList[rank](d0, d1, d2, d3, 4)

    var input_0_dyn = TileTensor(
        input_0_device_buffer.unsafe_ptr(), row_major(Coord(input_shape))
    )
    var input_1_dyn = TileTensor(
        input_1_device_buffer.unsafe_ptr(), row_major(Coord(input_shape))
    )
    var input_2_dyn = TileTensor(
        input_2_device_buffer.unsafe_ptr(), row_major(Coord(input_shape))
    )
    var input_3_dyn = TileTensor(
        input_3_device_buffer.unsafe_ptr(), row_major(Coord(input_shape))
    )
    var output_dyn = TileTensor(
        output_device_buffer.unsafe_ptr(), row_major(Coord(output_shape))
    )

    comptime B_SIZE = 32

    @parameter
    @always_inline
    @__copy_capture(output_dyn)
    fn epilogue_plus_one[
        c_type: DType, _rank: Int, width: Int, *, alignment: Int
    ](indices: IndexList[_rank], val: SIMD[c_type, width]):
        var coord = Coord(indices)
        comptime assert coord.flat_rank == output_dyn.flat_rank
        output_dyn.store[width=width](
            coord,
            rebind[SIMD[dtype, width]](val + 1),
        )

    comptime kernel = _concat_inner_most_single_dim[
        OutputLayoutType = output_dyn.LayoutType,
        output_origin=MutAnyOrigin,
        InputLayoutType = input_0_dyn.LayoutType,
        input_origin=ImmutAnyOrigin,
        dtype=dtype,
        num_inputs=4,
        block_size=B_SIZE,
        epilogue_fn = Optional[elementwise_epilogue_type](
            epilogue_plus_one
        ) if test_epilogue else None,
    ]

    @always_inline
    @__copy_capture(
        output_dyn,
        input_0_dyn,
        input_1_dyn,
        input_2_dyn,
        input_3_dyn,
    )
    @parameter
    fn run_concat_inner_most_single_dim(ctx: DeviceContext) raises:
        ctx.enqueue_function[kernel, kernel](
            output_dyn.as_any_origin(),
            StaticTuple[
                TileTensor[dtype, input_0_dyn.LayoutType, ImmutAnyOrigin],
                4,
            ](
                input_0_dyn.as_any_origin().as_immut(),
                input_1_dyn.as_any_origin().as_immut(),
                input_2_dyn.as_any_origin().as_immut(),
                input_3_dyn.as_any_origin().as_immut(),
            ),
            grid_dim=(d0 * d1 * d2 * d3 * d4 // B_SIZE),
            block_dim=(B_SIZE),
        )

    var nstime_kernel = ctx.execution_time[run_concat_inner_most_single_dim](1)
    print(
        "concat_inner_most_single_dim time = ",
        Float64(nstime_kernel) * 1e-6,
        " ms",
    )
    print(
        "transfer rate = ",
        Float64(output_dyn.numel() * size_of[UInt8]() * 2)
        * 1e9
        / Float64((1024**3))
        / Float64(nstime_kernel),
        "GB/s",
    )

    # Copy output back to host
    var output_host = TileTensor(output_host_buffer, output_layout)
    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    fn validate_results() raises:
        for i in range(d0):
            for j in range(d1):
                for k in range(d2):
                    for l in range(d3):
                        comptime tail_val = 1 if test_epilogue else 0
                        var not_match_0 = output_host[
                            i, j, k, l, 0
                        ] != input_0_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_1 = output_host[
                            i, j, k, l, 1
                        ] != input_1_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_2 = output_host[
                            i, j, k, l, 2
                        ] != input_2_host[i, j, k, l, 0] + Float32(tail_val)
                        var not_match_3 = output_host[
                            i, j, k, l, 3
                        ] != input_3_host[i, j, k, l, 0] + Float32(tail_val)
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
        output_dyn,
        input_0_dyn,
        input_1_dyn,
        input_2_dyn,
        input_3_dyn,
    )
    @parameter
    fn run_concat_gpu(ctx: DeviceContext) raises:
        # uses default stream
        _concat_gpu[
            epilogue_fn = Optional[elementwise_epilogue_type](
                epilogue_plus_one
            ) if test_epilogue else None
        ](
            output_dyn.as_any_origin(),
            4,
            StaticTuple[
                TileTensor[dtype, input_0_dyn.LayoutType, ImmutAnyOrigin],
                4,
            ](
                input_0_dyn.as_any_origin().as_immut(),
                input_1_dyn.as_any_origin().as_immut(),
                input_2_dyn.as_any_origin().as_immut(),
                input_3_dyn.as_any_origin().as_immut(),
            ),
            ctx,
        )

    var nstime = ctx.execution_time[run_concat_gpu](1)
    print("concat_gpu time = ", Float64(nstime) * 1e-6, " ms")
    print(
        "transfer rate = ",
        Float64(output_dyn.numel() * size_of[UInt8]() * 2)
        * 1e9
        / Float64((1024**3))
        / Float64(nstime),
        "GB/s",
    )

    ctx.enqueue_copy(output_host_buffer, output_device_buffer)
    ctx.synchronize()

    validate_results()

    _ = input_0_device_buffer
    _ = input_1_device_buffer
    _ = input_2_device_buffer
    _ = input_3_device_buffer
    _ = output_device_buffer


def main():
    with DeviceContext() as ctx:
        test_concat_4_inputs_rank5[True](ctx)
        test_concat_4_inputs_rank5[False](ctx)
