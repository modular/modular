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

from gpu import block_dim, block_idx, grid_dim, thread_idx
from gpu.host import DeviceContext, DeviceBuffer, DeviceAttribute
from layout._coord import Coord, Idx
from layout._layout import TensorLayout, Layout
from layout._tile_tensor import TileTensor
from math import ceildiv
from sys.info import align_of
from utils.index import IndexList


fn _fill_strides_indexlist[
    rank: Int,
](input_shape: IndexList[rank], mut strides: IndexList[rank],):
    """
    Fill `strides`, which will be an array of strides indexed by axis, assuming
    `buf` contains contiguous buf.

    Note that `buf` is only used for querying its dimensions.
    """
    comptime assert rank > 0
    strides[rank - 1] = 1

    @parameter
    for idx in range(rank - 1):
        comptime axis = rank - idx - 2
        var next_axis_stride = strides[axis + 1]
        var next_axis_dim = input_shape[axis + 1]
        var curr_axis_stride = next_axis_stride * next_axis_dim
        strides[axis] = curr_axis_stride


@always_inline
fn get_row_offset[
    dtype: DType,
](
    input_tensor: TileTensor[dtype, ...],
    output_tensor: TileTensor[mut=True, dtype, ...],
    row_length: Int,
    row: Int,
) -> Int:
    var coord = input_tensor.layout.idx2crd(row * row_length)
    var offset = output_tensor.layout(coord)
    return Int(offset)


@always_inline
fn scalar_copy_row[
    dtype: DType,
](
    input_ptr: UnsafePointer[Scalar[dtype]],
    output_ptr: UnsafePointer[mut=True, Scalar[dtype]],
    row_length: Int,
    threads_per_row: Int,
):
    var start_col = Int(thread_idx.x) % threads_per_row
    for col in range(start_col, row_length, threads_per_row):
        output_ptr[col] = input_ptr[col]


@always_inline
fn vector_copy_row[
    dtype: DType,
    simd_width: Int,
](
    input_ptr: UnsafePointer[Scalar[dtype]],
    output_ptr: UnsafePointer[mut=True, Scalar[dtype]],
    scaled_row_length: Int,
    row_length: Int,
    threads_per_row: Int,
):
    comptime alignment = align_of[SIMD[dtype, simd_width]]()

    var input_aligned = Int(input_ptr) % alignment == 0
    var output_aligned = Int(output_ptr) % alignment == 0

    # NOTE: possible perf improvement check for simd_width > 1
    # first then do alignment check

    if input_aligned and output_aligned and simd_width > 1:  # vectorized loads
        var iter_width = threads_per_row * simd_width
        var start_col = (Int(thread_idx.x) % threads_per_row) * simd_width
        for col in range(start_col, scaled_row_length, iter_width):
            output_ptr.store[width=simd_width](
                col, input_ptr.load[width=simd_width](col)
            )

        if scaled_row_length < row_length:
            var out_ptr = output_ptr + scaled_row_length
            var in_ptr = input_ptr + scaled_row_length
            scalar_copy_row(
                in_ptr, out_ptr, row_length - scaled_row_length, threads_per_row
            )

    else:  # default to scalar loads
        scalar_copy_row(input_ptr, output_ptr, row_length, threads_per_row)


fn padded_copy_kernel[
    InputLayoutType: TensorLayout,
    input_origin: ImmutOrigin,
    OutputLayoutType: TensorLayout,
    output_origin: MutOrigin,
    dtype: DType,
    simd_width: Int,
](
    input_tensor: TileTensor[dtype, InputLayoutType, input_origin],
    output_tensor: TileTensor[dtype, OutputLayoutType, output_origin],
    rows_per_sm: Int,
    total_rows: Int,
    row_length: Int,
    scaled_row_length: Int,
):
    var start_row = Int(block_idx.x) * rows_per_sm
    var threads_per_row = Int(block_dim.x)

    var rows_per_iter = Int(block_dim.y)
    var end_row = min(start_row + rows_per_sm, total_rows)

    start_row += Int(thread_idx.y)

    for row in range(start_row, end_row, rows_per_iter):
        var output_offset = get_row_offset(
            input_tensor, output_tensor, row_length, row
        )
        var input_offset = row * row_length

        var output_ptr = output_tensor.ptr + (output_offset)
        var input_ptr = input_tensor.ptr + (input_offset)

        vector_copy_row[dtype, simd_width](
            input_ptr,
            output_ptr,
            scaled_row_length,
            row_length,
            threads_per_row,
        )


fn _pad_constant_impl[
    dtype: DType,
    simd_width: Int = 1,
    max_threads: Int = 256,
    threads_per_row: Int = 16,
](
    input_tensor: TileTensor[dtype, address_space = AddressSpace.GENERIC, ...],
    output_tensor: TileTensor[
        mut=True, dtype, address_space = AddressSpace.GENERIC, ...
    ],
    ctx: DeviceContext,
) raises:
    var row_length = Int(input_tensor.dim(input_tensor.rank - 1))
    var total_rows = input_tensor.numel() // row_length

    comptime assert threads_per_row > 0 and max_threads % threads_per_row == 0

    var sm_count = ctx.get_attribute(DeviceAttribute.MULTIPROCESSOR_COUNT)

    var linear_block_count: Int

    var rows_per_block: Int

    if sm_count > total_rows:
        linear_block_count = total_rows
        rows_per_block = 1
    else:
        linear_block_count = sm_count
        rows_per_block = ceildiv(total_rows, sm_count)

    var scaled_row_length = (row_length // simd_width) * simd_width
    comptime block_rows = max_threads // threads_per_row
    comptime kernel = padded_copy_kernel[
        input_origin = ImmutOrigin(input_tensor.origin),
        InputLayoutType = input_tensor.LayoutType,
        output_origin = output_tensor.origin,
        OutputLayoutType = output_tensor.LayoutType,
        dtype=dtype,
        simd_width=simd_width,
    ]

    ctx.enqueue_function_experimental[kernel](
        input_tensor.as_immut(),
        output_tensor,
        rows_per_block,
        total_rows,
        row_length,
        scaled_row_length,
        grid_dim=(linear_block_count),
        block_dim=(threads_per_row, block_rows),
    )


fn pad_constant[
    rank: Int, dtype: DType, padding_type: DType
](
    output: UnsafePointer[mut=True, Scalar[dtype]],
    output_shape: IndexList[rank],
    input: UnsafePointer[Scalar[dtype]],
    input_shape: IndexList[rank],
    paddings: UnsafePointer[Scalar[padding_type]],
    constant: Scalar[dtype],
    ctx: DeviceContext,
) raises:
    """
    Fill `output` with values from `input`, and edges padded with `constant`
    based on `paddings`.

    Args:
        output: The output buffer.
        output_shape: The output shape.
        input: The input buffer.
        input_shape: The input shape.
        paddings: Ordered (before, after) padding sizes for each axis.
        constant: The constant to pad output with.
        ctx: Device context for participating GPU.

    Example:
        ```mojo
        var input_shape = (X, Y, Z)
        var paddings = [x0, x1, y0, y1, z0, z1]

        out[x, y, z] =
          input[x - x0, y - y0, z - z0] if x ∈ [x0, x0 + X] &&
                                           y ∈ [y0, y0 + Y] &&
                                           z ∈ [z0, z0 + Z]
          else constant
        ```
    """

    var input_strides = IndexList[rank]()
    var output_strides = IndexList[rank]()

    var output_size: Int = 1

    @parameter
    for i in range(rank):
        output_size *= output_shape[i]

    var output_buffer = DeviceBuffer[dtype](
        ctx, output, output_size, owning=False
    )
    output_buffer.enqueue_fill(constant)

    _fill_strides_indexlist[rank](input_shape, input_strides)
    _fill_strides_indexlist[rank](output_shape, output_strides)

    var input_tensor = TileTensor(
        input,
        Layout(Coord(input_shape), Coord(input_strides)),
    )

    var pre_pad_offset = 0

    for i in range(rank):
        pre_pad_offset += Int(paddings[2 * i]) * output_strides[i]

    var adjusted_output_ptr = output + pre_pad_offset

    var output_tensor = TileTensor(
        adjusted_output_ptr,
        Layout(Coord(input_shape), Coord(output_strides)),
    )

    _pad_constant_impl[dtype](
        input_tensor,
        output_tensor,
        ctx,
    )


fn get_padding_output_shape[
    rank: Int
](
    input_shape: IndexList[rank],
    paddings: TileTensor[DType.int, ...],
) -> IndexList[rank]:
    comptime assert (
        paddings.flat_rank == 1 and paddings.static_shape[0] == 2 * rank
    )
    var output_shape = IndexList[rank]()
    for i in range(rank):
        var before = paddings[2 * i]
        var after = paddings[2 * i + 1]
        output_shape[i] = Int(before) + input_shape[i] + Int(after)
    return output_shape
