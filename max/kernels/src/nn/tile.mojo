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

# Note: This implementation supports tensors of any rank.

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

    Tiling runs from the innermost axis outward, so every axis beyond the
    current one replicates an already-contiguous region of `output`:

    1. Copy each input row (inner axis) into the first output row slot,
       then unsafe_memcpy it `repeat[0] - 1` more times along the inner
       axis. This fills one full output "row plane" per input row.
    2. For each outer axis (inner to outer), the filled block of
       `prod(inner dims * repeats)` elements is memcpy'd to the remaining
       `repeat[axis] - 1` block slots at their block strides. Each copy
       never overlaps an earlier one: destination blocks are separated by
       the full block extent, and block 0 stays the source for every copy.

    Parameters:
        dtype: Type of the input and output tensors.
        type_repeats: Type of the repeats tensor.

    Args:
        input: The input tensor. Any rank is supported.
        repeats: One-dimensional tensor that specifies the number of repeated
                 copies along each of the input's dimensions. Length equals
                 input tensor rank.
        output: The output tensor. Has the same dimensions and type as input.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output must have the same rank"

    comptime assert (
        repeats.flat_rank == 1 and type_repeats == .int64
    ), "Rank of repeats tensor needs to be one-dimensional and of int64 type."

    if input.rank != Int(repeats.dim(0)):
        raise Error(
            "Length of repeats tensor should be equal to the rank of the input"
            " tensor."
        )

    comptime rank = input.rank

    # Zero repeats or zero input extents empty the output; nothing to copy.
    # (This also keeps the offset math below from writing past a
    # zero-sized output buffer.)
    if output.num_elements() == 0:
        return

    # Extents and repeat counts, innermost axis first (the ONNX `repeats`
    # order is outermost-first).
    var dims = IndexList[rank]()
    var reps = IndexList[rank]()
    var odims = IndexList[rank]()
    comptime for axis in range(rank):
        dims[axis] = Int(input.dim(rank - 1 - axis))
        reps[axis] = Int(repeats[rank - 1 - axis])
        odims[axis] = dims[axis] * reps[axis]

    # Row-major offset step of each axis in the input and the output: the
    # product of every inner extent.
    var in_stride = IndexList[rank]()
    var out_stride = IndexList[rank]()
    in_stride[0] = 1
    out_stride[0] = 1
    comptime for axis in range(1, rank):
        in_stride[axis] = in_stride[axis - 1] * dims[axis - 1]
        out_stride[axis] = out_stride[axis - 1] * odims[axis - 1]

    # Stage 0 (innermost axis): copy every input row (the inner axis is
    # contiguous) into the first slot of its output row plane, then repeat
    # it along the inner axis. This initializes output by first copying in
    # the original input to the appropriate output elements, and then
    # handles tiling across the column (last) dimension.
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
    var num_rows = 1
    comptime for axis in range(1, rank):
        num_rows *= dims[axis]

    for row in range(num_rows):
        # Decode `row` into the input coordinates of every axis above the
        # inner one (innermost of those fastest), and map them to the
        # input and output offsets of the row.
        var rem = row
        var src_index = 0
        var dst_index = 0
        for axis in range(1, rank):
            var coord = rem % dims[axis]
            rem = (rem - coord) // dims[axis]
            src_index += coord * in_stride[axis]
            dst_index += coord * out_stride[axis]

        unsafe_memcpy(
            dest=output.ptr.unsafe_offset(dst_index),
            src=input.ptr.unsafe_offset(src_index),
            count=dims[0],
        )
        for rep in range(1, reps[0]):
            # The row block just written to `output` is repeated in place
            # along the inner axis; dest and src are non-overlapping
            # slices of the same `output` buffer (shared origin). Opt out
            # of exclusivity with an unsafe any-origin: unsafe_memcpy's
            # non-overlap is a caller contract the checker can't prove.
            unsafe_memcpy(
                dest=output.ptr.unsafe_offset(dst_index).unsafe_offset(
                    rep * dims[0]
                ),
                src=output.ptr.unsafe_offset(dst_index).as_unsafe_any_origin(),
                count=dims[0],
            )

    # Stages 1..rank-1 (outer axes, inner to outer): replicate the region
    # already filled by the previous stage across the remaining slots of
    # that axis. Continuing with the example above, this will handle the
    # 'X's, which correspond to the first '2' in the repeats = [2, 2]
    # tensor. The result is:
    #   output: [[1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6],
    #            [1, 2, 3, 1, 2, 3],
    #            [4, 5, 6, 4, 5, 6]]
    # Moving from the inner to the outermost dimension, we can unsafe_memcpy
    # to replicate contiguous memory areas (representing a dimension to be
    # tiled).
    comptime for axis in range(1, rank):
        # The filled region of one value of this axis: every inner
        # (already tiled) extent times this axis' input extent, one
        # contiguous chunk.
        var chunk = dims[axis] * out_stride[axis]
        var num_outer = 1
        comptime for outer in range(axis + 1, rank):
            num_outer *= dims[outer]

        for outer_index in range(num_outer):
            # Decode `outer_index` into the input coordinates of every
            # axis above this one; those select which filled chunk to
            # repeat.
            var rem = outer_index
            var base_index = 0
            for outer in range(axis + 1, rank):
                var coord = rem % dims[outer]
                rem = (rem - coord) // dims[outer]
                base_index += coord * out_stride[outer]

            for rep in range(1, reps[axis]):
                var dst_ptr = output.ptr.unsafe_offset(
                    base_index
                ).unsafe_offset(rep * chunk)
                # dst_ptr and src_ptr are non-overlapping slices of the
                # same `output` buffer (shared origin). Opt out of
                # exclusivity with an unsafe any-origin:
                # unsafe_memcpy's non-overlap is a caller contract the
                # checker can't prove.
                unsafe_memcpy(
                    dest=dst_ptr,
                    src=output.ptr.unsafe_offset(
                        base_index
                    ).as_unsafe_any_origin(),
                    count=chunk,
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
    comptime assert (
        input.rank == output.rank
    ), "input and output must have the same rank"
    # Any rank: the elementwise decode wraps each axis by the input extent.
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
            output.store(
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
