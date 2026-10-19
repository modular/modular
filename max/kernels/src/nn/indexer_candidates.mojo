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
"""Prepare sparse indexer candidates and restore request-local token positions."""

from std.bit import log2_floor
from std.collections import Array
from std.math import ceildiv, clamp
from std.memory import unsafe_stack_allocation
from std.utils.index import StaticTuple
from layout import TileTensor
from max.gpu import (
    MAX_THREADS_PER_BLOCK_METADATA,
    WARP_SIZE,
    block_idx,
    thread_idx,
)
from max.gpu.host import DeviceContext
from max.gpu.sync import barrier
import max.gpu.primitives.block as block
import max.gpu.primitives.warp as warp


@inline(.always)
def _raw[
    dtype: DType
](t: TileTensor[dtype, ...]) -> MutPointer[Scalar[dtype], MutUntrackedOrigin]:
    # The kernels below index the pointer as dense row-major, so a strided
    # layout would read and write the wrong addresses.
    comptime assert (
        t.is_row_major
    ), "indexer_candidates tensors must be row-major"
    return (
        t.ptr.unsafe_mut_cast[True]()
        .unsafe_origin_cast[MutUntrackedOrigin]()
        .unsafe_address_space_cast[.GENERIC]()
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"indexer_expand_k{candidates}_r{ratio}_b{block_kv}")
def _expand[
    candidates: Int, ratio: Int, block_kv: Int
](
    input: ImmPointer[Int32, ImmutAnyOrigin],
    starts: ImmPointer[Int32, ImmutAnyOrigin],
    stops: ImmPointer[Int32, ImmutAnyOrigin],
    output: MutPointer[Int32, MutAnyOrigin],
    ends: MutPointer[Int32, MutAnyOrigin],
    input_stride: Int32,
    start_stride: Int32,
    stop_stride: Int32,
    output_stride: Int32,
    end_stride: Int32,
):
    comptime size = candidates * ratio
    comptime items = ceildiv(size, 128)
    comptime capacity = items * 128
    var row = block_idx.x
    var lane = thread_idx.x
    var start = starts[unsafe_offset=row * Int(start_stride)]
    var stop = stops[unsafe_offset=row * Int(stop_stride)]
    var limit = ceildiv(max(stop - start, Int32(0)), Int32(block_kv))
    var values = Array[Int32, items](fill=Int32.MAX)
    var count = Int32(0)
    var last = Int32(-1)
    comptime for item in range(items):
        var index = item * 128 + lane
        if index < size:
            var candidate = input[
                unsafe_offset=row * Int(input_stride) + index // ratio
            ] * Int32(ratio) + Int32(index % ratio)
            if candidate >= 0 and candidate < limit:
                values[item] = candidate
                count += 1
                last = max(last, candidate)
    count = block.sum[block_size=128](count)
    last = block.max[block_size=128](last)
    var shared = unsafe_stack_allocation[
        capacity, Int32, address_space=.SHARED
    ]()
    # Register pairs and warp shuffles avoid a block barrier at every sort step.
    comptime for level in range(1, log2_floor(capacity) + 1):
        comptime stage = 1 << level
        comptime for step in range(level):
            comptime distance = 1 << (level - step - 1)
            var partners = Array[Int32, items](fill=0)
            comptime if distance >= 128:
                comptime for item in range(items):
                    partners[item] = values[item ^ (distance // 128)]
            elif distance < WARP_SIZE:
                comptime for item in range(items):
                    partners[item] = warp.shuffle_xor(
                        values[item], UInt32(distance)
                    )
            else:
                comptime for item in range(items):
                    shared[unsafe_offset=item * 128 + lane] = values[item]
                barrier()
                comptime for item in range(items):
                    partners[item] = shared[
                        unsafe_offset=(item * 128 + lane) ^ distance
                    ]
                barrier()
            comptime for item in range(items):
                var index = item * 128 + lane
                var take_min = ((index & stage) == 0) == (
                    (index & distance) == 0
                )
                values[item] = min(
                    values[item], partners[item]
                ) if take_min else max(values[item], partners[item])
    var base = start // Int32(block_kv)
    var pad = base + (last if count > 0 else Int32(0))
    comptime for item in range(items):
        var index = item * 128 + lane
        if index < size:
            output[unsafe_offset=row * Int(output_stride) + index] = (
                values[item] + base if index < Int(count) else pad
            )
    if lane == 0:
        var fill = clamp(
            stop - (last * Int32(block_kv) + start), Int32(1), Int32(block_kv)
        )
        ends[unsafe_offset=row * Int(end_stride)] = (count - 1) * Int32(
            block_kv
        ) + fill if count > 0 else Int32(0)


def indexer_expand_candidates[
    candidates: Int, ratio: Int, block_kv: Int
](
    input: TileTensor[mut=False, .int32, ...],
    starts: TileTensor[mut=False, .int32, ...],
    stops: TileTensor[mut=False, .int32, ...],
    output: TileTensor[mut=True, .int32, ...],
    ends: TileTensor[mut=True, .int32, ...],
    ctx: DeviceContext,
) raises:
    """Expand and sort candidate blocks; pad with the final valid block.

    Each tensor is a row-major `[rows, stride]` tile whose row stride is
    its second dimension. Candidates are nonnegative request-local block
    ids or negative sentinels. Bounds are token positions in a shared
    workspace; their base need not be block-aligned. Duplicates are
    retained. `ends` counts valid token columns, including a partial final
    block. All buffers are caller-owned and disjoint.

    Parameters:
        candidates: Candidate capacity per row, a power of two.
        ratio: Compression group size, either 1 or 2.
        block_kv: Tokens per block, either 8 or 16.

    Args:
        input: Candidate ids. Shape: `[rows, input_stride]` with at least
            `candidates` columns used.
        starts: First token position per row, in column 0.
            Shape: `[rows, start_stride]`.
        stops: Last-plus-one token position per row, in column 0.
            Shape: `[rows, stop_stride]`.
        output: Sorted expanded block ids. Shape: `[rows, output_stride]`
            with at least `candidates * ratio` columns used.
        ends: Valid token count per row, in column 0.
            Shape: `[rows, end_stride]`.
        ctx: The device context.
    """
    comptime assert candidates > 0 and candidates.is_power_of_two()
    comptime assert ratio == 1 or ratio == 2
    comptime assert block_kv == 8 or block_kv == 16
    comptime assert input.rank == 2, "input must be [rows, input_stride]"
    comptime assert starts.rank == 2, "starts must be [rows, start_stride]"
    comptime assert stops.rank == 2, "stops must be [rows, stop_stride]"
    comptime assert output.rank == 2, "output must be [rows, output_stride]"
    comptime assert ends.rank == 2, "ends must be [rows, end_stride]"
    var rows = Int(input.dim[0]())
    var input_stride = Int(input.dim[1]())
    var start_stride = Int(starts.dim[1]())
    var stop_stride = Int(stops.dim[1]())
    var output_stride = Int(output.dim[1]())
    var end_stride = Int(ends.dim[1]())
    debug_assert(
        Int(starts.dim[0]()) == rows
        and Int(stops.dim[0]()) == rows
        and Int(output.dim[0]()) == rows
        and Int(ends.dim[0]()) == rows
    )
    debug_assert(input_stride >= candidates)
    debug_assert(output_stride >= candidates * ratio)
    debug_assert(start_stride > 0 and stop_stride > 0 and end_stride > 0)
    if rows == 0:
        return
    ctx.enqueue_function[_expand[candidates, ratio, block_kv]](
        _raw(input).as_imm().as_unsafe_any_origin(),
        _raw(starts).as_imm().as_unsafe_any_origin(),
        _raw(stops).as_imm().as_unsafe_any_origin(),
        _raw(output).as_unsafe_any_origin(),
        _raw(ends).as_unsafe_any_origin(),
        Int32(input_stride),
        Int32(start_stride),
        Int32(stop_stride),
        Int32(output_stride),
        Int32(end_stride),
        grid_dim=rows,
        block_dim=128,
    )


@__llvm_metadata(MAX_THREADS_PER_BLOCK_METADATA=StaticTuple[Int32, 1](128))
@__name(t"indexer_remap_b{block_kv}")
def _remap[
    block_kv: Int
](
    columns: ImmPointer[Int32, ImmutAnyOrigin],
    blocks: ImmPointer[Int32, ImmutAnyOrigin],
    starts: ImmPointer[Int32, ImmutAnyOrigin],
    output: MutPointer[Int32, MutAnyOrigin],
    count: Int32,
    width: Int32,
    column_stride: Int32,
    block_stride: Int32,
    start_stride: Int32,
    output_stride: Int32,
):
    var row = block_idx.y
    var index = block_idx.x * 128 + thread_idx.x
    if index < Int(count):
        var column = columns[unsafe_offset=row * Int(column_stride) + index]
        var value = Int32(-1)
        if column >= 0 and column < width:
            var base = starts[unsafe_offset=row * Int(start_stride)] // Int32(
                block_kv
            )
            var sparse = blocks[
                unsafe_offset=row * Int(block_stride)
                + Int(column // Int32(block_kv))
            ]
            value = (sparse - base) * Int32(block_kv) + column % Int32(block_kv)
        output[unsafe_offset=row * Int(output_stride) + index] = value


def indexer_remap_sparse_columns[
    block_kv: Int
](
    columns: TileTensor[mut=False, .int32, ...],
    blocks: TileTensor[mut=False, .int32, ...],
    starts: TileTensor[mut=False, .int32, ...],
    output: TileTensor[mut=True, .int32, ...],
    count: Int,
    width: Int,
    ctx: DeviceContext,
) raises:
    """Map selected sparse columns to local tokens; invalid columns yield -1.

    Each tensor is a row-major `[rows, stride]` tile whose row stride is
    its second dimension. `width` is the scored sparse-column width, not a
    row's valid-prefix length. The selector owns masking by the latter
    before this operation.

    Parameters:
        block_kv: Tokens per block, either 8 or 16.

    Args:
        columns: Selected sparse columns. Shape: `[rows, column_stride]`
            with at least `count` columns used.
        blocks: Expanded block ids, as `indexer_expand_candidates` writes.
            Shape: `[rows, block_stride]` with at least
            `ceildiv(width, block_kv)` columns used.
        starts: First token position per row, in column 0.
            Shape: `[rows, start_stride]`.
        output: Local token positions. Shape: `[rows, output_stride]`
            with at least `count` columns used.
        count: Selected columns per row.
        width: Scored sparse-column width.
        ctx: The device context.
    """
    comptime assert block_kv == 8 or block_kv == 16
    comptime assert columns.rank == 2, "columns must be [rows, column_stride]"
    comptime assert blocks.rank == 2, "blocks must be [rows, block_stride]"
    comptime assert starts.rank == 2, "starts must be [rows, start_stride]"
    comptime assert output.rank == 2, "output must be [rows, output_stride]"
    var rows = Int(columns.dim[0]())
    var column_stride = Int(columns.dim[1]())
    var block_stride = Int(blocks.dim[1]())
    var start_stride = Int(starts.dim[1]())
    var output_stride = Int(output.dim[1]())
    debug_assert(rows >= 0 and count >= 0 and width >= 0)
    debug_assert(
        Int(blocks.dim[0]()) == rows
        and Int(starts.dim[0]()) == rows
        and Int(output.dim[0]()) == rows
    )
    debug_assert(column_stride >= count and output_stride >= count)
    debug_assert(block_stride >= ceildiv(width, block_kv) and start_stride > 0)
    if rows == 0 or count == 0:
        return
    ctx.enqueue_function[_remap[block_kv]](
        _raw(columns).as_imm().as_unsafe_any_origin(),
        _raw(blocks).as_imm().as_unsafe_any_origin(),
        _raw(starts).as_imm().as_unsafe_any_origin(),
        _raw(output).as_unsafe_any_origin(),
        Int32(count),
        Int32(width),
        Int32(column_stride),
        Int32(block_stride),
        Int32(start_stride),
        Int32(output_stride),
        grid_dim=(ceildiv(count, 128), rows),
        block_dim=128,
    )
