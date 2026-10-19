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
"""Implements `arg_nonzero`, which returns the indices of all nonzero elements in a tensor."""


from max.algorithm.functional import _get_start_indices_of_nth_subvolume
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.primitives import block
from max.runtime.tracing import Trace, TraceLevel
from std.math import ceildiv
from std.utils import StaticTuple
from std.utils.index import IndexList

# Default GPU block size; override per launch via `_arg_nonzero_gpu[BLOCK_SIZE=...]`.
comptime _ARG_NONZERO_BLOCK_SIZE = 256

# ===-----------------------------------------------------------------------===#
# arg_nonzero
# ===-----------------------------------------------------------------------===#


@inline(.always)
def arg_nonzero[
    dtype: DType,
    output_type: DType,
](
    input_buffer: TileTensor[dtype, ...],
    output_buffer: TileTensor[mut=True, output_type, ...],
) raises:
    """Gather the indices of all non-zero elements in input buffer storing
    the indices in the output_buffer.

    Parameters:
        dtype: The element dtype.
        output_type: The integer dtype to store the indices in.

    Args:
        input_buffer: The tensor to count the non-zeros in.
        output_buffer: The indices of all non-zero elements.
    """
    comptime assert (
        output_buffer.flat_rank == 2
    ), "output_buffer must be of rank 2"
    # Provide evidence that flat_rank >= 2 for the Coord(..., ...) stores below.
    comptime assert output_buffer.flat_rank >= 2

    with Trace[TraceLevel.OP, target=StaticString("cpu")]("arg_nonzero"):
        var numel = input_buffer.num_elements()
        if numel == 0:
            return

        var j: Int = 0
        for i in range(numel):
            var indices = _get_start_indices_of_nth_subvolume[0](
                i, coord_to_index_list(input_buffer.layout.shape_coord())
            )
            var offset = input_buffer.layout(Coord(indices))
            if input_buffer[Coord(indices)] != 0:
                var out_indices = IndexList[2]()
                out_indices[0] = j
                j += 1

                # Get the coordinates for the current index
                var coords = input_buffer.layout.idx2crd(Int(offset))

                # Write each coordinate to the output buffer
                comptime for k in range(input_buffer.rank):
                    out_indices[1] = k
                    output_buffer.store(
                        (out_indices[0], out_indices[1]),
                        output_buffer.ElementType(coords[k].value()),
                    )


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"arg_nonzero_gpu_count_{dtype}_{BLOCK_SIZE}")
def _arg_nonzero_count_kernel[
    dtype: DType,
    BLOCK_SIZE: Int,
    InputLayout: TensorLayout,
    CountsLayout: TensorLayout,
    InputEngine: TensorEngine,
    CountsEngine: TensorEngine,
](
    input_buffer: TileTensor[
        dtype, InputLayout, ImmutAnyOrigin, Engine=InputEngine
    ],
    counts: TileTensor[
        mut=True,
        DType.int64,
        CountsLayout,
        MutAnyOrigin,
        Engine=CountsEngine,
    ],
    tile_len_arg: Int64,
    num_elements_arg: Int64,
):
    """Counts the nonzero elements of one tile of the input.

    The tile index is the block index. Each block walks its tile in
    `BLOCK_SIZE`-element chunks, one element per lane per chunk, so tiles
    longer than `BLOCK_SIZE` are fully covered. The final tile may be
    shorter than `tile_len_arg`, so reads are bounded by `num_elements_arg`
    as well.
    """
    var tile_len = Int(tile_len_arg)
    var num_elements = Int(num_elements_arg)
    var tile_start = tile_len * block_idx.x
    var local: Int32 = 0
    var input_flat = input_buffer.reshape(row_major(num_elements))
    for chunk in range(0, tile_len, BLOCK_SIZE):
        var flat_index = tile_start + chunk + thread_idx.x
        if chunk + thread_idx.x < tile_len and flat_index < num_elements:
            # 32-bit lanes: warp shuffles move 32- and 64-bit lanes only,
            # and the 32-bit path is the one every other scan in this tree
            # exercises.
            if input_flat[flat_index] != 0:
                local += 1
    var count = block.sum[block_size=BLOCK_SIZE](local)
    if thread_idx.x == 0:
        counts[block_idx.x] = Int64(count)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"arg_nonzero_gpu_offsets_{BLOCK_SIZE}")
def _arg_nonzero_offsets_kernel[
    BLOCK_SIZE: Int,
    CountsLayout: TensorLayout,
    CountsEngine: TensorEngine,
](
    counts: TileTensor[
        mut=True, DType.int64, CountsLayout, MutAnyOrigin, Engine=CountsEngine
    ],
    num_tiles_arg: Int64,
):
    """Replaces each per-tile count with its exclusive prefix, in place.

    One block scans every tile count, so `_arg_nonzero_gpu` caps the
    count/emit grid at `BLOCK_SIZE` tiles; larger inputs take
    proportionally longer tiles instead. The scan uses 64-bit lanes so
    the offsets can reach the full nonzero count.
    """
    var num_tiles = Int(num_tiles_arg)
    var tid = thread_idx.x
    var selected: Int64 = 0
    if tid < num_tiles:
        selected = Int64(counts[tid])
    var offset = block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](
        selected
    )
    if tid < num_tiles:
        counts[tid] = Int64(offset)


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](Int32(BLOCK_SIZE))
)
@__name(t"arg_nonzero_gpu_emit_{dtype}_{output_type}_{rank}_{BLOCK_SIZE}")
def _arg_nonzero_emit_kernel[
    dtype: DType,
    output_type: DType,
    rank: Int,
    BLOCK_SIZE: Int,
    InputLayout: TensorLayout,
    OutputLayout: TensorLayout,
    OffsetsLayout: TensorLayout,
    InputEngine: TensorEngine,
    OutputEngine: TensorEngine,
    OffsetsEngine: TensorEngine,
](
    input_buffer: TileTensor[
        dtype, InputLayout, ImmutAnyOrigin, Engine=InputEngine
    ],
    output_buffer: TileTensor[
        mut=True,
        output_type,
        OutputLayout,
        MutAnyOrigin,
        Engine=OutputEngine,
    ],
    offsets: TileTensor[
        DType.int64, OffsetsLayout, ImmutAnyOrigin, Engine=OffsetsEngine
    ],
    tile_len_arg: Int64,
    num_elements_arg: Int64,
    dims: StaticTuple[Int64, rank],
    strides: StaticTuple[Int64, rank],
):
    """Writes the row-major coordinates of one tile.

    The tile's base row is the scanned per-tile offset. Each block walks
    its tile in `BLOCK_SIZE`-element chunks: within a chunk the
    block-wide exclusive prefix sum ranks the selected elements, and the
    running base advances by each chunk's total. Since chunks are walked
    in ascending order and rows are placed at `base + prefix`, rows stay
    in ascending row-major order across chunks, tiles, and the grid. The
    final tile may be shorter than `tile_len_arg`, so reads are bounded
    by `num_elements_arg` as well.
    """
    var tile_len = Int(tile_len_arg)
    var num_elements = Int(num_elements_arg)
    var tid = thread_idx.x
    var tile_start = tile_len * block_idx.x
    var base = Int(offsets[block_idx.x])
    var input_flat = input_buffer.reshape(row_major(num_elements))
    for chunk in range(0, tile_len, BLOCK_SIZE):
        var selected: Int32 = 0
        if chunk + tid < tile_len and tile_start + chunk + tid < num_elements:
            if input_flat[tile_start + chunk + tid] != 0:
                selected = 1

        var local_row = block.prefix_sum[block_size=BLOCK_SIZE, exclusive=True](
            selected
        )

        if selected != 0:
            var output_row = base + Int(local_row)
            var remainder = tile_start + chunk + tid
            comptime for axis in range(rank):
                var coordinate = (remainder // Int(strides[axis])) % Int(
                    dims[axis]
                )
                remainder -= coordinate * Int(strides[axis])
                output_buffer.store(
                    (output_row, axis),
                    output_buffer.ElementType(coordinate),
                )

        base += Int(block.sum[block_size=BLOCK_SIZE](selected))


def _arg_nonzero_gpu[
    dtype: DType,
    output_type: DType,
    //,
    BLOCK_SIZE: Int = _ARG_NONZERO_BLOCK_SIZE,
](
    input_buffer: TileTensor[dtype, ...],
    output_buffer: TileTensor[mut=True, output_type, ...],
    ctx: DeviceContext,
) raises:
    """Gathers nonzero coordinates with a multi-grid count/offsets/emit pass.

    The count kernel runs one block per tile across the full grid; the
    offsets kernel turns the per-tile counts into exclusive prefixes in
    one block; the emit kernel re-walks the tiles in parallel, ranking
    each selected element by its tile offset plus its in-chunk prefix.
    All three launches share the stream, so each sees the prior's writes.
    """
    var num_elements = input_buffer.num_elements()
    if num_elements == 0:
        return

    # Decode metadata is computed on the host, where the layout round-trips
    # are known-good, and passed by value. Rank is comptime from the layout.
    comptime rank = input_buffer.rank
    var dims = coord_to_index_list(input_buffer.layout.shape_coord())
    var dims_tuple = StaticTuple[Int64, rank](fill=0)
    var strides_tuple = StaticTuple[Int64, rank](fill=0)
    comptime for axis in range(rank):
        dims_tuple[axis] = Int64(dims[axis])
        var stride = 1
        comptime for later in range(axis + 1, rank):
            stride *= Int(dims[later])
        strides_tuple[axis] = Int64(stride)

    # The kernels below walk memory linearly (a flat `row_major` view) and
    # decode coordinates from row-major strides, so both hold only for a
    # row-major input. The graph compiler guarantees this: strided views
    # (slice/broadcast/reshape) never reach a non-fused kernel argument;
    # they are materialized into dense row-major buffers first. Guard so a
    # future non-row-major input fails loudly here, not as silently
    # misdecoded coordinates.
    var actual_strides = coord_to_index_list(input_buffer.layout.stride_coord())
    var row_major_strides = dims.get_row_major_strides()
    for axis in range(rank):
        if actual_strides[axis] != row_major_strides[axis]:
            raise Error(
                "arg_nonzero GPU path requires a row-major input; axis "
                + String(axis)
                + " has stride "
                + String(actual_strides[axis])
                + ", expected "
                + String(row_major_strides[axis])
            )

    # One tile per block. The offsets kernel scans the tile counts in a
    # single block, so the grid is capped at BLOCK_SIZE tiles; larger
    # inputs take proportionally longer tiles, which the count/emit
    # kernels walk in BLOCK_SIZE-element chunks.
    var num_tiles = min(ceildiv(num_elements, BLOCK_SIZE), BLOCK_SIZE)
    var tile_len = ceildiv(num_elements, num_tiles)

    var counts_buffer = ctx.enqueue_create_buffer[.int64](num_tiles)
    var counts = TileTensor(counts_buffer, row_major(num_tiles))

    var input_any = input_buffer.as_imm().as_unsafe_any_origin()
    var output_any = output_buffer.as_unsafe_any_origin()
    var counts_imm = counts.as_imm().as_unsafe_any_origin()
    comptime count_kernel = _arg_nonzero_count_kernel[
        dtype,
        BLOCK_SIZE,
        input_any.LayoutType,
        counts.LayoutType,
        input_any.Engine,
        counts.Engine,
    ]
    ctx.enqueue_function[count_kernel](
        input_any,
        counts,
        Int64(tile_len),
        Int64(num_elements),
        grid_dim=num_tiles,
        block_dim=BLOCK_SIZE,
    )

    comptime offsets_kernel = _arg_nonzero_offsets_kernel[
        BLOCK_SIZE,
        counts_imm.LayoutType,
        counts_imm.Engine,
    ]
    ctx.enqueue_function[offsets_kernel](
        counts,
        Int64(num_tiles),
        grid_dim=1,
        block_dim=BLOCK_SIZE,
    )

    comptime emit_kernel = _arg_nonzero_emit_kernel[
        dtype,
        output_type,
        rank,
        BLOCK_SIZE,
        input_any.LayoutType,
        output_any.LayoutType,
        counts_imm.LayoutType,
        input_any.Engine,
        output_any.Engine,
        counts_imm.Engine,
    ]
    ctx.enqueue_function[emit_kernel](
        input_any,
        output_any,
        counts_imm,
        Int64(tile_len),
        Int64(num_elements),
        dims_tuple,
        strides_tuple,
        grid_dim=num_tiles,
        block_dim=BLOCK_SIZE,
    )


def arg_nonzero[
    dtype: DType,
    output_type: DType,
    target: StaticString,
](
    input_buffer: TileTensor[dtype, ...],
    output_buffer: TileTensor[mut=True, output_type, ...],
    ctx: DeviceContext,
) raises:
    """Gathers nonzero coordinates on `target`.

    Parameters:
        dtype: The element dtype.
        output_type: The integer dtype used for coordinates.
        target: The target device.

    Args:
        input_buffer: The tensor whose nonzero coordinates are returned.
        output_buffer: The `[nnz, input.rank]` coordinate tensor.
        ctx: Device context used to enqueue GPU work.
    """
    with Trace[TraceLevel.OP, target=target]("arg_nonzero"):
        comptime if is_cpu[target]():
            arg_nonzero(input_buffer, output_buffer)
        else:
            _arg_nonzero_gpu(input_buffer, output_buffer, ctx)


# Where has the shape 2D shape [NumNonZeros, InputRank]
@inline(.always)
def arg_nonzero_shape[
    dtype: DType
](input_buffer: TileTensor[dtype, ...]) -> IndexList[2]:
    """Return [NumNonZeros, InputRank] where NumNonZeros are the number of
    non-zero elements in the input.

    Parameters:
        dtype: The element dtype.

    Args:
        input_buffer: The tensor to count the non-zeros in.

    Returns:
        Shape of the arg_nonzero kernel for this input [NumNonZeros, InputRank].
    """

    var shape = IndexList[2]()
    shape[1] = input_buffer.rank

    var numel = input_buffer.num_elements()

    var j: Int = 0
    for i in range(numel):
        if input_buffer[i] != 0:
            j += 1

    shape[0] = j
    return shape
