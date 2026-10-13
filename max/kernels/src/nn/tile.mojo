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
"""Implements the ONNX Tile operator, which replicates a tensor a specified number of times along each axis."""

from layout import Coord, TileTensor, coord_to_index_list
from max.algorithm.functional import elementwise
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from std.memory import unsafe_memcpy
from std.sys.info import simd_width_of
from std.utils import IndexList, StaticTuple

# TODO: This implementation supports up to 4 dimensions.

# Note: ONNX spec specifies that `tile` behaves like Numpy's tile, but without
#       broadcast. This means that `repeats` is a 1D int64 tensor of the SAME
#       length as input's rank (unlike Numpy that allows `repeats` to have more
#       or less elements than the input's rank).


@inline(.always)
def _tile_cpu[
    dtype: DType, type_repeats: DType
](
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    repeats: TileTensor[type_repeats, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
) raises:
    """
    Implements the `Tile` operator from the ONNX spec. This behaves like Numpy
    tile, but without broadcast.

    Parameters:
        dtype: Type of the input and output tensors.
        type_repeats: Type of the repeats tensor.

    Args:
        input: The input tensor. Currently <= 4 dimensions are supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        output: The output tensor. Has the same dimensions and type as input.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output must have the same rank"

    comptime assert (
        input.rank <= 4
    ), "Currently only inputs of up to three dimensions are supported."

    comptime assert (
        repeats.flat_rank == 1 and type_repeats == .int64
    ), "Rank of repeats tensor needs to be one-dimensional and of int64 type."

    if input.rank != Int(repeats.dim(0)):
        raise Error(
            "Length of repeats tensor should be equal to the rank of the input"
            " tensor."
        )

    var num_dp_input = 1
    var num_depth_input = 1
    var num_rows_input = 1

    comptime if input.rank == 4:
        num_dp_input = Int(input.dim(input.rank - 4))

    comptime if input.rank >= 3:
        num_depth_input = Int(input.dim(input.rank - 3))

    comptime if input.rank >= 2:
        num_rows_input = Int(input.dim(input.rank - 2))
    var num_cols_input = Int(input.dim(input.rank - 1))

    var repeats_len = Int(repeats.dim(0))

    # Pre-compute repeat values, using 1 for dimensions that don't exist
    # to avoid negative index access when rank < 4
    var repeat_cols = Int(repeats[repeats_len - 1])
    var repeat_rows = 1
    var repeat_depth = 1
    var repeat_dp = 1

    comptime if input.rank >= 2:
        repeat_rows = Int(repeats[repeats_len - 2])

    comptime if input.rank >= 3:
        repeat_depth = Int(repeats[repeats_len - 3])

    comptime if input.rank >= 4:
        repeat_dp = Int(repeats[repeats_len - 4])

    # Initializes output by first copying in the original input to the
    # appropriate output elements, and then handles tiling across the column
    # (last) dimension.
    # e.g., for:
    #   input:  [[1, 2, 3],
    #            [4, 5, 6]]
    #   and repeats = [2,2], the below will handle:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [X, X, X, X, X, X],
    #            [X, X, X, X, X, X]]
    #   where 'X' denotes parts of the output that are not yet calculated.
    #   These (i.e., for dimensions beyond the innermost) are handled later.
    for dp in range(num_dp_input):
        for d in range(num_depth_input):
            for r in range(num_rows_input):
                # print(dp, d, r)
                var input_src_index = (
                    dp * num_depth_input * num_rows_input * num_cols_input
                    + d * num_rows_input * num_cols_input
                    + r * num_cols_input
                )
                var output_src_index = (
                    dp
                    * num_depth_input
                    * repeat_depth
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + d
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + r * num_cols_input * repeat_cols
                )
                var output_src_stride = num_cols_input
                var count = output_src_stride
                for rep in range(repeat_cols):
                    var src_ptr = input.ptr + input_src_index
                    var dst_ptr = output.ptr + (
                        output_src_index + rep * output_src_stride
                    )
                    unsafe_memcpy(dest=dst_ptr, src=src_ptr, count=count)

    # Handles tiling across the second lowest dimension (if tensor rank >= 2).
    # Continuing with the example above, this will handle the 'X's, which
    # correspond to the first '2' in the repeats = [2, 2] tensor.
    # The result is:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6]]
    # Moving from the inner to the outermost dimension, we can unsafe_memcpy to
    # replicate contiguous memory areas (representing a dimension to be tiled).
    comptime if input.rank >= 2:
        var src_index_stride = num_rows_input * num_cols_input * repeat_cols
        var count = src_index_stride
        for dp in range(num_dp_input):
            for d in range(num_depth_input):
                var src_index = (
                    dp
                    * num_depth_input
                    * repeat_depth
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                    + d
                    * num_rows_input
                    * repeat_rows
                    * num_cols_input
                    * repeat_cols
                )
                for rep in range(repeat_rows - 1):
                    var src_ptr = output.ptr + src_index
                    var dst_ptr = output.ptr + (
                        src_index + (rep + 1) * src_index_stride
                    )
                    # dst_ptr and src_ptr are non-overlapping slices of the
                    # same `output` buffer (shared origin). Opt out of
                    # exclusivity with an unsafe any-origin:
                    # unsafe_memcpy's non-overlap
                    # is a caller contract the checker can't prove.
                    unsafe_memcpy(
                        dest=dst_ptr,
                        src=src_ptr.as_unsafe_any_origin(),
                        count=count,
                    )

    # Handles tiling across the third dimension from the end (if tensor rank >= 3)
    comptime if input.rank >= 3:
        var src_index_stride = (
            num_depth_input
            * repeat_rows
            * num_rows_input
            * num_cols_input
            * repeat_cols
        )
        var count = src_index_stride
        for dp in range(num_dp_input):
            var src_index = (
                dp
                * num_depth_input
                * repeat_depth
                * num_rows_input
                * repeat_rows
                * num_cols_input
                * repeat_cols
            )
            for rep in range(repeat_depth - 1):
                var src_ptr = output.ptr + src_index
                var dst_ptr = output.ptr + (
                    src_index + (rep + 1) * src_index_stride
                )
                # dst_ptr and src_ptr are non-overlapping slices of the
                # same `output` buffer (shared origin). Opt out of
                # exclusivity with an unsafe any-origin:
                # unsafe_memcpy's non-overlap
                # is a caller contract the checker can't prove.
                unsafe_memcpy(
                    dest=dst_ptr,
                    src=src_ptr.as_unsafe_any_origin(),
                    count=count,
                )

    # Handles tiling across the fourth dimension from the end (if tensor rank == 4)
    comptime if input.rank == 4:
        var src_index_stride = (
            num_dp_input
            * repeat_depth
            * num_depth_input
            * repeat_rows
            * num_rows_input
            * num_cols_input
            * repeat_cols
        )
        var count = src_index_stride
        var src_index = 0
        for rep in range(repeat_dp - 1):
            var src_ptr = output.ptr + src_index
            var dst_ptr = output.ptr + (
                src_index + (rep + 1) * src_index_stride
            )
            # dst_ptr and src_ptr are non-overlapping slices of the
            # same `output` buffer (shared origin). Opt out of
            # exclusivity with an unsafe any-origin:
            # unsafe_memcpy's non-overlap
            # is a caller contract the checker can't prove.
            unsafe_memcpy(
                dest=dst_ptr,
                src=src_ptr.as_unsafe_any_origin(),
                count=count,
            )


@inline(.always)
def _tile_gpu[
    dtype: DType,
    type_repeats: DType,
    target: StaticString,
](
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    repeats: TileTensor[type_repeats, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Tiles `input` into `output` on a GPU.

    The original kernel ran one thread per scalar element with a divmod
    chain per axis, so the SMs spent their issue slots on integer
    division (75% SM throughput at 3% L2 throughput on a B200) instead
    of moving data. This version splits by repeat pattern:

    - `repeat_inner == 1` (every output row IS one input row -- the
      common outer-repeats-only case, e.g. expand-embeddings and MoE row
      blocks): a vector-width elementwise pass over the original rank-N
      shape. The framework packs consecutive elements per closure, which
      stay in one row because the inner extent must divide by the vector
      width (else the launch falls back to scalar); each closure wraps
      every axis by the input extent once per vector. Measured 6.7-7.5x
      faster than the per-element kernel on a B200.
    - otherwise (the last axis repeats too): the per-element kernel,
      which a chunk-per-thread variant regressed against on B200.

    The input extents are read on the host and captured by value; the
    coordinate decode is arithmetic against those, never through
    device-side layout round-trips.
    """
    comptime assert input.rank == output.rank
    comptime assert input.rank <= 4
    comptime assert (
        repeats.flat_rank == 1 and type_repeats == .int64
    ), "repeats must be rank-1 int64"
    if input.rank != Int(repeats.dim(0)):
        raise Error(
            "Length of repeats tensor should be equal to the rank of the"
            " input tensor."
        )

    var num_elements = output.num_elements()
    if num_elements == 0:
        return

    # All extents are read on the host and captured by value: the source
    # coordinate of each output element wraps per axis by the input extent.
    var input_dims = coord_to_index_list(input.layout.shape_coord())
    var output_dims = coord_to_index_list(output.layout.shape_coord())
    comptime rank = input.rank
    var in_dims_t = StaticTuple[Int64, rank](fill=0)
    comptime for axis in range(rank):
        in_dims_t[axis] = Int64(input_dims[axis])

    var in_inner = Int(input_dims[rank - 1])
    var out_inner = Int(output_dims[rank - 1])
    if in_inner == 0:
        return
    var repeat_inner = out_inner // in_inner

    comptime full_width = simd_width_of[dtype]()
    # The vector pass stays inside one row only when the inner extent
    # divides by the vector width (the framework packs consecutive
    # elements; a ragged extent would let a group wrap rows).
    var width = full_width if in_inner % full_width == 0 else 1

    if repeat_inner == 1:
        # Every output row IS one input row: wrap every axis by the input
        # extent and vectorize along the innermost.
        @inline(.always)
        def copy_row[
            simd_width: Int, alignment: Int = 1
        ](output_index: Coord) {var}:
            # The framework may probe or fall back to other widths (the
            # scalar tail path); every width <= full_width stays within a
            # row because the inner extent divides by the launch width.
            var out_coord = coord_to_index_list(output_index)
            var in_coord = out_coord
            comptime for axis in range(rank):
                in_coord[axis] = out_coord[axis] % Int(in_dims_t[axis])
            output.store[width=simd_width](
                Coord(out_coord),
                input.load[width=simd_width](Coord(in_coord)),
            )

        comptime if full_width > 1:
            if width == full_width:
                elementwise[
                    simd_width=full_width,
                    target=target,
                    _trace_description="tile",
                ](copy_row, output.layout.shape_coord(), ctx)
            else:
                elementwise[
                    simd_width=1,
                    target=target,
                    _trace_description="tile",
                ](copy_row, output.layout.shape_coord(), ctx)
        else:
            elementwise[
                simd_width=1,
                target=target,
                _trace_description="tile",
            ](copy_row, output.layout.shape_coord(), ctx)
    else:
        # Inner axis repeats too: keep the per-element kernel. A
        # chunk-per-thread variant regressed on B200 (the extra grid
        # dimension costs more than the outer-axis wraps it saves), so
        # only the vectorizable `repeat_inner == 1` case takes the fast
        # path above.
        @inline(.always)
        def copy_element[
            simd_width: Int, alignment: Int = 1
        ](output_index: Coord) {var}:
            comptime assert simd_width == 1
            # `output_index` is the output's per-axis coordinate. The
            # source coordinate wraps each axis by the input's extent.
            var output_coord = coord_to_index_list(output_index)
            var input_index = output_coord
            comptime for axis in range(rank):
                input_index[axis] = output_coord[axis] % Int(in_dims_t[axis])
            output.store(
                Coord(output_coord),
                input.load(Coord(input_index)),
            )

        elementwise[
            simd_width=1,
            target=target,
            _trace_description="tile",
        ](copy_element, output.layout.shape_coord(), ctx)


@inline(.always)
def tile[
    dtype: DType, type_repeats: DType
](
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    repeats: TileTensor[type_repeats, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
) raises:
    """Tiles `input` into `output` on the CPU."""
    _tile_cpu(input, repeats, output)


@inline(.always)
def tile[
    dtype: DType,
    type_repeats: DType,
    target: StaticString,
](
    input: TileTensor[dtype, address_space=.GENERIC, ...],
    repeats: TileTensor[type_repeats, address_space=.GENERIC, ...],
    output: TileTensor[mut=True, dtype, address_space=.GENERIC, ...],
    ctx: DeviceContext,
) raises:
    """Tiles `input` into `output` on CPU or GPU."""
    comptime if is_cpu[target]():
        _tile_cpu(input, repeats, output)
    else:
        _tile_gpu[dtype, type_repeats, target](input, repeats, output, ctx)


@inline(.always)
def tile_shape[
    input_type: DType,
    repeats_type: DType,
](
    input_buf: TileTensor[input_type, ...],
    repeats_buf: TileTensor[repeats_type, ...],
) raises -> IndexList[input_buf.rank]:
    """
    Compute the output shape of a `tile` operation, and assert the inputs are
    compatible.

    Parameters:
        input_type: Type of the input tensor.
        repeats_type: Type of the repeats tensor.

    Args:
        input_buf: The input tensor.
        repeats_buf: The repeats tensor.

    Returns:
        The output shape.
    """
    comptime assert repeats_buf.flat_rank == 1, "repeats_buf must be of rank 1"

    # TODO add runtime test once we support dynamic rank execution, currently
    # MLIR verifier of `MO::TileOp` prevents testing this with static rank.
    if Int(repeats_buf.dim(0)) != input_buf.rank:
        raise Error("[tile] requires (len(repeats) == input_rank)")

    # Compute and return the output shape. Negative repeats would produce a
    # negative dimension, so reject them here (zero stays valid, like NumPy).
    var output_shape = IndexList[input_buf.rank]()

    comptime for i in range(input_buf.rank):
        var repeat = Int(repeats_buf[i])
        if repeat < 0:
            raise Error("[tile] requires non-negative repeats")
        output_shape[i] = Int(input_buf.dim(i)) * repeat

    return output_shape
