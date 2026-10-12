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
"""Implements the ONNX CumSum operator, computing prefix sums along a specified tensor axis."""

from std.math import align_up, ceildiv
from std.memory import stack_allocation
from std.sys import bit_width_of
from std.utils.static_tuple import StaticTuple

from layout import (
    Coord,
    Idx,
    ImmTileTensor,
    MutTileTensor,
    TensorLayout,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    block_idx,
    global_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import is_cpu
from max.gpu.memory import AddressSpace
from max.gpu.primitives import block
from max.gpu.sync import barrier
from max.runtime.tracing import Trace, TraceLevel, get_safe_task_id

from std.utils.numerics import get_accum_type

# Threads per block, and contiguous elements per thread, for the row-scan
# kernel. Each block covers `_CUMSUM_BLOCK_SIZE * _CUMSUM_ITEMS` elements of a
# row per loop trip.
comptime _CUMSUM_BLOCK_SIZE = 256
comptime _CUMSUM_ITEMS = 4
# Independent loads each thread keeps in flight per trip in the chunk-sum
# kernel.
comptime _CUMSUM_TOTAL_ITEMS = 16

# Contiguous rows at least this long always take the row-scan kernel. Shorter
# rows take it only when they are too few to occupy the GPU one thread each
# (see `_use_row_scan`): with many short rows, a 256-thread block per row is
# mostly idle threads, and one thread per row is faster.
comptime _CUMSUM_MIN_ROW_SCAN_LEN = 128

# Threads per block of the one-thread-per-line kernels.
comptime _CUMSUM_SERIAL_BLOCK_SIZE = 256

# A line is split into chunks scanned by separate blocks (or threads) only when
# there are too few lines to occupy the GPU. Row-scan blocks per SM that count
# as "full", the blocks per SM to aim for when splitting, and the smallest
# chunk worth a block.
comptime _CUMSUM_FILL_BLOCKS_PER_SM = 2
comptime _CUMSUM_TARGET_BLOCKS_PER_SM = 8
comptime _CUMSUM_MIN_CHUNK_TILES = 4
# The same for the one-thread-per-line kernels, in threads per SM and elements.
comptime _CUMSUM_TARGET_THREADS_PER_SM = 2048
comptime _CUMSUM_MIN_SERIAL_CHUNK = 32


@inline(.always)
def cumsum[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
](output: TileTensor[mut=True, dtype, ...], input: TileTensor[dtype, ...],):
    """
    Implements the CumSum operator from the ONNX spec:
    https://github.com/onnx/onnx/blob/main/docs/Operators.md#CumSum
    Computes cumulative sum of the input elements along the given axis.
    Cumulative sum can be inclusive or exclusive of the top element, and
    normal or reverse (direction along a given axis).

    Parameters:
        dtype: Type of the input and output tensors.
        exclusive: If set to True, return exclusive sum (top element not included).
        reverse: If set to True, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.

    Args:
        output: The output tensor.
        input: The input tensor.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output should have the same rank."

    comptime accum_type = DType.float64 if dtype == DType.float32 else get_accum_type[
        dtype
    ]()
    comptime assert (
        -input.rank <= axis < input.rank
    ), "Axis value must be in range [-rank, rank)"
    comptime axis_pos = axis if axis >= 0 else axis + input.rank

    var shape = coord_to_index_list(input.layout.shape_coord())

    var inner = 1
    var outer = 1
    var depth = 1
    for i in range(input.rank):
        if i < axis_pos:
            inner *= shape[i]
        elif i > axis_pos:
            outer *= shape[i]
        else:
            depth = shape[i]

    var output_data = TileTensor(
        output.ptr,
        row_major(output.num_elements()),
    )
    var input_data = TileTensor(
        input.ptr,
        row_major(input.num_elements()),
    )

    for outer_index in range(outer):
        var outer_index_adj: Int

        comptime if reverse:
            outer_index_adj = (outer - 1) - outer_index
        else:
            outer_index_adj = outer_index

        for inner_index in range(inner):
            var accumulator: Scalar[accum_type] = 0
            var inner_index_adj: Int

            comptime if reverse:
                inner_index_adj = (inner - 1) - inner_index
            else:
                inner_index_adj = inner_index

            for depth_index in range(depth):
                var depth_index_adj: Int

                comptime if reverse:
                    depth_index_adj = (depth - 1) - depth_index
                else:
                    depth_index_adj = depth_index

                var index = (
                    outer_index_adj
                    + inner_index_adj * depth * outer
                    + depth_index_adj * outer
                )

                comptime if exclusive:
                    output_data[index] = accumulator.cast[dtype]()
                    accumulator = (
                        accumulator + input_data[index].cast[accum_type]()
                    )
                else:
                    accumulator = (
                        accumulator + input_data[index].cast[accum_type]()
                    )
                    output_data[index] = accumulator.cast[dtype]()


def _gpu_accum_type[dtype: DType]() -> DType:
    # Warp shuffles move 32- and 64-bit lanes only, so narrow integers scan in
    # 32 bits. Truncating back to `dtype` gives the same wrapped result.
    comptime if dtype.is_integral() and bit_width_of[dtype]() < 32:
        return DType.int32 if dtype.is_signed() else DType.uint32
    return get_accum_type[dtype]()


@inline(.always)
def _cumsum_shape[axis: Int](input: ImmTileTensor[...]) -> Tuple[Int, Int, Int]:
    """Returns `(num_rows, axis_len, stride)` for a row-major `input`.

    `num_rows` is the product of the dims before `axis` and `stride` the
    product of the dims after it, so `input` reshapes to
    `[num_rows, axis_len, stride]`.
    """
    comptime axis_pos = axis if axis >= 0 else axis + input.rank
    var shape = coord_to_index_list(input.layout.shape_coord())
    var num_rows = 1
    var stride = 1
    comptime for i in range(input.rank):
        comptime if i < axis_pos:
            num_rows *= shape[i]
        elif i > axis_pos:
            stride *= shape[i]
    return (num_rows, shape[axis_pos], stride)


@inline(.always)
def _scan_pos[reverse: Bool](index: Int, axis_len: Int) -> Int:
    """Maps the `index`-th element in scan order to its position on the axis."""
    comptime if reverse:
        return axis_len - 1 - index
    return index


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_BLOCK_SIZE)
    )
)
def _cumsum_row_scan_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans one `chunk_len`-long chunk of a row of a `[num_rows, axis_len]`
    input; block `block_idx.x` takes chunk `block_idx.x % num_chunks` of row
    `block_idx.x // num_chunks`.

    Each thread folds `_CUMSUM_ITEMS` consecutive elements serially, a block
    scan over the per-thread totals places each thread, and the tile total
    carries into the next trip along the chunk. With `has_carry`, the sum of
    all earlier chunks of the row, `carry[row, chunk]`, seeds the first trip.
    """
    comptime assert input.flat_rank == 2 and output.flat_rank == 2
    comptime tile = _CUMSUM_BLOCK_SIZE * _CUMSUM_ITEMS
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    # Without a carry, the chunk is the whole row; skip the 64-bit division.
    var row = block_idx.x
    var chunk = 0
    comptime if has_carry:
        row, chunk = divmod(block_idx.x, ceildiv(axis_len, chunk_len))
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)
    var tid = thread_idx.x
    var tile_total = stack_allocation[
        1, Scalar[accum_type], address_space=AddressSpace.SHARED
    ]()

    var carry_in: Scalar[accum_type] = 0
    comptime if has_carry:
        carry_in = carry[row, chunk]
    for tile_start in range(begin, end, tile):
        var first = tile_start + tid * _CUMSUM_ITEMS
        var partial = SIMD[accum_type, _CUMSUM_ITEMS](0)
        var thread_sum: Scalar[accum_type] = 0
        comptime for k in range(_CUMSUM_ITEMS):
            var val: Scalar[accum_type] = 0
            if first + k < end:
                val = input[row, _scan_pos[reverse](first + k, axis_len)].cast[
                    accum_type
                ]()
            comptime if exclusive:
                partial[k] = thread_sum
                thread_sum += val
            else:
                thread_sum += val
                partial[k] = thread_sum

        var offset = carry_in + block.prefix_sum[
            block_size=_CUMSUM_BLOCK_SIZE, exclusive=True
        ](thread_sum)

        comptime for k in range(_CUMSUM_ITEMS):
            if first + k < end:
                output[row, _scan_pos[reverse](first + k, axis_len)] = (
                    offset + partial[k]
                ).cast[dtype]()

        if tid == _CUMSUM_BLOCK_SIZE - 1:
            tile_total[0] = offset + thread_sum
        barrier()
        carry_in = tile_total[0]
        # The last thread rewrites `tile_total` on the next trip.
        barrier()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_BLOCK_SIZE)
    )
)
def _cumsum_row_total_kernel[
    dtype: DType,
    accum_type: DType,
    reverse: Bool,
    InLayoutType: TensorLayout,
    TotalLayoutType: TensorLayout,
](
    totals: TileTensor[accum_type, TotalLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Writes the sum of each `chunk_len`-long chunk of a `[num_rows,
    axis_len]` input to `totals[row, chunk]`, one block per chunk."""
    comptime assert input.flat_rank == 2 and totals.flat_rank == 2
    comptime tile = _CUMSUM_BLOCK_SIZE * _CUMSUM_TOTAL_ITEMS
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var row, chunk = divmod(block_idx.x, ceildiv(axis_len, chunk_len))
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)
    var tid = thread_idx.x

    var total: Scalar[accum_type] = 0
    for tile_start in range(begin, end, tile):
        # Consecutive threads read consecutive elements for coalescing.
        comptime for k in range(_CUMSUM_TOTAL_ITEMS):
            var index = tile_start + k * _CUMSUM_BLOCK_SIZE + tid
            if index < end:
                total += input[row, _scan_pos[reverse](index, axis_len)].cast[
                    accum_type
                ]()
    total = block.sum[block_size=_CUMSUM_BLOCK_SIZE, broadcast=False](total)
    if tid == 0:
        totals[row, chunk] = total


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_SERIAL_BLOCK_SIZE)
    )
)
def _cumsum_serial_kernel[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
    OutLayoutType: TensorLayout,
    InLayoutType: TensorLayout,
    CarryLayoutType: TensorLayout,
](
    output: TileTensor[dtype, OutLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    carry: TileTensor[accum_type, CarryLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Scans one `chunk_len`-long chunk of one `[row, :, col]` line of a
    `[num_rows, axis_len, stride]` input per thread.

    Adjacent threads own adjacent columns, so when `stride > 1` each step
    along the axis is a coalesced load across the warp. With `has_carry`,
    `carry[row * stride + col, chunk]` is the sum of all earlier chunks of the
    line.
    """
    comptime assert input.flat_rank == 3 and output.flat_rank == 3
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var stride = Int(input.dim[2]())
    # Without a carry, the chunk is the whole line; skip the 64-bit division.
    var num_chunks = 1
    comptime if has_carry:
        num_chunks = ceildiv(axis_len, chunk_len)
    if global_idx.x >= Int(input.dim[0]()) * stride * num_chunks:
        return
    var row_chunk, col = divmod(global_idx.x, stride)
    var row = row_chunk
    var chunk = 0
    comptime if has_carry:
        row, chunk = divmod(row_chunk, num_chunks)
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)

    var acc: Scalar[accum_type] = 0
    comptime if has_carry:
        acc = carry[row * stride + col, chunk]
    for i in range(begin, end):
        var pos = _scan_pos[reverse](i, axis_len)
        var val = input[row, pos, col].cast[accum_type]()
        comptime if exclusive:
            output[row, pos, col] = acc.cast[dtype]()
            acc += val
        else:
            acc += val
            output[row, pos, col] = acc.cast[dtype]()


@__llvm_metadata(
    MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](
        Int32(_CUMSUM_SERIAL_BLOCK_SIZE)
    )
)
def _cumsum_serial_total_kernel[
    dtype: DType,
    accum_type: DType,
    reverse: Bool,
    InLayoutType: TensorLayout,
    TotalLayoutType: TensorLayout,
](
    totals: TileTensor[accum_type, TotalLayoutType, MutAnyOrigin],
    input: TileTensor[dtype, InLayoutType, ImmutAnyOrigin],
    chunk_len_arg: Int32,
):
    """Writes the sum of each `chunk_len`-long chunk of each line of a
    `[num_rows, axis_len, stride]` input to `totals[row * stride + col, chunk]`,
    one thread per chunk."""
    comptime assert input.flat_rank == 3 and totals.flat_rank == 2
    var chunk_len = Int(chunk_len_arg)
    var axis_len = Int(input.dim[1]())
    var stride = Int(input.dim[2]())
    var num_chunks = ceildiv(axis_len, chunk_len)
    if global_idx.x >= Int(input.dim[0]()) * stride * num_chunks:
        return
    var row_chunk, col = divmod(global_idx.x, stride)
    var row, chunk = divmod(row_chunk, num_chunks)
    var begin = chunk * chunk_len
    var end = min(begin + chunk_len, axis_len)

    var total: Scalar[accum_type] = 0
    for i in range(begin, end):
        total += input[row, _scan_pos[reverse](i, axis_len), col].cast[
            accum_type
        ]()
    totals[row * stride + col, chunk] = total


def _enqueue_scan[
    dtype: DType,
    accum_type: DType,
    exclusive: Bool,
    reverse: Bool,
    has_carry: Bool,
](
    output: MutTileTensor[dtype, ...],
    input: ImmTileTensor[dtype, ...],
    carry: ImmTileTensor[accum_type, ...],
    num_rows: Int,
    axis_len: Int,
    stride: Int,
    chunk_len: Int,
    row_scan: Bool,
    ctx: DeviceContext,
) raises:
    """Scans every `chunk_len`-long chunk of `[num_rows, axis_len, stride]`.

    Without `has_carry` each line is one chunk starting at zero (`chunk_len`
    must be `axis_len`) and `carry` is never read. `row_scan` selects
    the block-per-chunk kernel, which needs `stride == 1`.
    """
    var num_chunks = ceildiv(axis_len, chunk_len)
    if row_scan:
        var rows_layout = row_major((num_rows, axis_len))
        var out_rows = TileTensor(output.unsafe_ptr(), rows_layout)
        var in_rows = TileTensor(input.unsafe_ptr(), rows_layout)
        comptime kernel = _cumsum_row_scan_kernel[
            dtype,
            accum_type,
            exclusive,
            reverse,
            has_carry,
            out_rows.LayoutType,
            in_rows.LayoutType,
            carry.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            out_rows,
            in_rows.as_imm(),
            carry,
            Int32(chunk_len),
            grid_dim=num_rows * num_chunks,
            block_dim=_CUMSUM_BLOCK_SIZE,
        )
        return

    var lines_layout = row_major((num_rows, axis_len, stride))
    var out_lines = TileTensor(output.unsafe_ptr(), lines_layout)
    var in_lines = TileTensor(input.unsafe_ptr(), lines_layout)
    comptime kernel = _cumsum_serial_kernel[
        dtype,
        accum_type,
        exclusive,
        reverse,
        has_carry,
        out_lines.LayoutType,
        in_lines.LayoutType,
        carry.LayoutType,
    ]
    ctx.enqueue_function[kernel](
        out_lines,
        in_lines.as_imm(),
        carry,
        Int32(chunk_len),
        grid_dim=ceildiv(
            num_rows * stride * num_chunks, _CUMSUM_SERIAL_BLOCK_SIZE
        ),
        block_dim=_CUMSUM_SERIAL_BLOCK_SIZE,
    )


def _enqueue_chunk_totals[
    dtype: DType, accum_type: DType, reverse: Bool
](
    totals: MutTileTensor[accum_type, ...],
    input: ImmTileTensor[dtype, ...],
    num_rows: Int,
    axis_len: Int,
    stride: Int,
    chunk_len: Int,
    row_scan: Bool,
    ctx: DeviceContext,
) raises:
    """Writes the per-chunk sums of `[num_rows, axis_len, stride]` to the
    `[num_rows * stride, num_chunks]` tensor `totals`, in scan order."""
    var num_chunks = ceildiv(axis_len, chunk_len)
    if row_scan:
        var in_rows = TileTensor(
            input.unsafe_ptr(), row_major((num_rows, axis_len))
        )
        comptime kernel = _cumsum_row_total_kernel[
            dtype,
            accum_type,
            reverse,
            in_rows.LayoutType,
            totals.LayoutType,
        ]
        ctx.enqueue_function[kernel](
            totals,
            in_rows.as_imm(),
            Int32(chunk_len),
            grid_dim=num_rows * num_chunks,
            block_dim=_CUMSUM_BLOCK_SIZE,
        )
        return

    var in_lines = TileTensor(
        input.unsafe_ptr(), row_major((num_rows, axis_len, stride))
    )
    comptime kernel = _cumsum_serial_total_kernel[
        dtype, accum_type, reverse, in_lines.LayoutType, totals.LayoutType
    ]
    ctx.enqueue_function[kernel](
        totals,
        in_lines.as_imm(),
        Int32(chunk_len),
        grid_dim=ceildiv(
            num_rows * stride * num_chunks, _CUMSUM_SERIAL_BLOCK_SIZE
        ),
        block_dim=_CUMSUM_SERIAL_BLOCK_SIZE,
    )


def _chunk_len(
    row_scan: Bool, num_lines: Int, axis_len: Int, sm_count: Int
) -> Int:
    """Returns the axis span one block (or thread) scans; `axis_len` means the
    axis is not split.

    A line is split only when there are too few lines to occupy the GPU. The
    decision depends on shapes alone so that it replays under graph capture.
    """
    var chunk_len: Int
    if row_scan:
        if num_lines >= _CUMSUM_FILL_BLOCKS_PER_SM * sm_count:
            return axis_len
        comptime tile = _CUMSUM_BLOCK_SIZE * _CUMSUM_ITEMS
        var num_chunks = ceildiv(
            _CUMSUM_TARGET_BLOCKS_PER_SM * sm_count, num_lines
        )
        chunk_len = align_up(ceildiv(axis_len, num_chunks), tile)
        chunk_len = max(chunk_len, _CUMSUM_MIN_CHUNK_TILES * tile)
    else:
        if num_lines >= _CUMSUM_TARGET_THREADS_PER_SM * sm_count:
            return axis_len
        var num_chunks = ceildiv(
            _CUMSUM_TARGET_THREADS_PER_SM * sm_count, num_lines
        )
        chunk_len = max(ceildiv(axis_len, num_chunks), _CUMSUM_MIN_SERIAL_CHUNK)
    # Two chunks do not repay the extra launches.
    return axis_len if 2 * chunk_len >= axis_len else chunk_len


def _use_row_scan(
    num_lines: Int, axis_len: Int, stride: Int, sm_count: Int
) -> Bool:
    """Returns whether `num_lines` lines of `axis_len` take the row-scan
    kernel rather than the one-thread-per-line one.

    Row scan needs contiguous lines (`stride == 1`). Below
    `_CUMSUM_MIN_ROW_SCAN_LEN`, it is chosen only where the one-thread-per-line
    kernel would split the lines into chunks, which costs three launches and
    two scratch buffers where row scan takes one launch.
    """
    if stride != 1:
        return False
    return (
        axis_len >= _CUMSUM_MIN_ROW_SCAN_LEN
        or _chunk_len(False, num_lines, axis_len, sm_count) < axis_len
    )


def _cumsum_gpu[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
](
    output: MutTileTensor[dtype, ...],
    input: ImmTileTensor[dtype, ...],
    ctx: DeviceContext,
) raises:
    # Unlike the CPU path, float32 accumulates in float32: float64 is slow on
    # most GPUs and unsupported on Apple silicon.
    comptime accum_type = _gpu_accum_type[dtype]()
    # Graph tensors are contiguous but do not carry static row-major strides,
    # so the views below are built from the base pointer, as on CPU.
    var num_rows, axis_len, stride = _cumsum_shape[axis](input)
    if num_rows * axis_len * stride == 0:
        return

    var num_lines = num_rows * stride
    var sm_count = ctx.default_device_info.sm_count
    var row_scan = _use_row_scan(num_lines, axis_len, stride, sm_count)
    var chunk_len = _chunk_len(row_scan, num_lines, axis_len, sm_count)
    # Stands in for `carry` wherever `has_carry` is False; never read.
    var no_carry = TileTensor(
        input.unsafe_ptr().unsafe_bitcast[Scalar[accum_type]](),
        row_major((1, 1)),
    )
    if chunk_len == axis_len:
        _enqueue_scan[dtype, accum_type, exclusive, reverse, False](
            output,
            input,
            no_carry,
            num_rows,
            axis_len,
            stride,
            axis_len,
            row_scan,
            ctx,
        )
        return

    # Split each line into chunks: sum every chunk, scan the sums into
    # per-chunk carries, then scan each chunk starting from its carry.
    var num_chunks = ceildiv(axis_len, chunk_len)
    var totals_buf = ctx.enqueue_create_buffer[accum_type](
        num_lines * num_chunks
    )
    var carries_buf = ctx.enqueue_create_buffer[accum_type](
        num_lines * num_chunks
    )
    var chunks_layout = row_major((num_lines, num_chunks))
    var totals = TileTensor(totals_buf, chunks_layout)
    var carries = TileTensor(carries_buf, chunks_layout)
    _enqueue_chunk_totals[dtype, accum_type, reverse](
        totals, input, num_rows, axis_len, stride, chunk_len, row_scan, ctx
    )
    _enqueue_scan[accum_type, accum_type, True, False, False](
        carries,
        totals.as_imm(),
        no_carry,
        num_lines,
        num_chunks,
        1,
        num_chunks,
        _use_row_scan(num_lines, num_chunks, 1, sm_count),
        ctx,
    )
    _enqueue_scan[dtype, accum_type, exclusive, reverse, True](
        output,
        input,
        carries.as_imm(),
        num_rows,
        axis_len,
        stride,
        chunk_len,
        row_scan,
        ctx,
    )


def cumsum[
    dtype: DType,
    exclusive: Bool,
    reverse: Bool,
    *,
    axis: Int,
    target: StaticString,
](
    output: TileTensor[mut=True, dtype, ...],
    input: TileTensor[dtype, ...],
    ctx: DeviceContext,
) raises:
    """Computes the cumulative sum of `input` along `axis` on `target`.

    See the CPU overload for the semantics. On GPU, float32 inputs accumulate
    in float32 rather than float64.

    Parameters:
        dtype: Type of the input and output tensors.
        exclusive: If set to True, return exclusive sum (top element not included).
        reverse: If set to True, perform cumsum operation in reverse direction.
        axis: The axis on which to perform the cumsum operation.
        target: The target device, `"cpu"` or a GPU target.

    Args:
        output: The output tensor.
        input: The input tensor.
        ctx: The device context used to enqueue GPU kernels.

    Raises:
        Error: If the GPU kernel launch fails.
    """
    comptime assert (
        input.rank == output.rank
    ), "input and output should have the same rank."
    comptime assert (
        -input.rank <= axis < input.rank
    ), "Axis value must be in range [-rank, rank)"
    with Trace[TraceLevel.OP, target=target](
        "cumsum", task_id=get_safe_task_id(ctx)
    ):
        comptime if is_cpu[target]():
            cumsum[dtype, exclusive, reverse, axis=axis](output, input)
        else:
            _cumsum_gpu[dtype, exclusive, reverse, axis=axis](
                output, input.as_imm(), ctx
            )
