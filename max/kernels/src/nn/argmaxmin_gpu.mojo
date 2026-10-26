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
"""Provides a GPU argmax of logits plus a bias over the inner-most dimension."""


from std.bit import next_power_of_two
from max.gpu import block_dim, block_idx, thread_idx
from max.gpu.primitives.grid_controls import (
    PDL,
    PDLLevel,
    pdl_launch_attributes,
)
from std.math import align_down, ceildiv, iota
from std.sys import align_of, size_of

from max.gpu.host import DeviceContext
from max.runtime.tracing import Trace, TraceLevel, trace_arg
from layout import (
    Coord,
    TensorLayout,
    TensorEngine,
    TileTensor,
    coord_to_index_list,
    row_major,
)
from nn.topk import TopK_2, _block_reduce_topk, _topk_dead_val


# Elements each thread pulls per loop trip, as a multiple of the vector width.
# The (value, index) update is loop-carried, so without a widened trip a warp
# keeps only one load in flight and the scan runs at HBM-latency rate rather
# than at bandwidth. Four 128-bit loads per trip is enough to saturate a
# streaming CTA while staying well inside the 64-register budget at 1024
# threads.
comptime _ARGMAX_UNROLL = 4

# Row bytes per scan block. One block saturates at the per-SM load rate, so
# long rows have to be split across blocks to reach device bandwidth; the
# split costs one extra (tiny) combine launch, which is why short rows are
# left on a single block.
comptime _ARGMAX_BYTES_PER_BLOCK = 32 * 1024

# Cap on slices per row so the combine stays one block of at most 1024
# threads and the scratch stays negligible.
comptime _ARGMAX_MAX_SPLITS = 64


@inline(.always)
def _argmax_vec_update[
    dtype: DType, simd_width: Int
](
    mut best_vals: SIMD[dtype, simd_width],
    mut best_idxs: SIMD[.int32, simd_width],
    vals: SIMD[dtype, simd_width],
    base: Int,
):
    """Folds one loaded vector into the per-lane running (max, index).

    The comparison is strict so that, scanning indices in increasing order,
    the lowest index wins a tie. NaN never compares greater and is therefore
    ignored, matching `TopK_2.insert`.
    """
    comptime lane_offsets = iota[.int32, simd_width]()
    var better = vals.gt(best_vals)
    best_vals = better.select(vals, best_vals)
    best_idxs = better.select(lane_offsets + Int32(base), best_idxs)


@inline(.always)
def _argmax_scan[
    dtype: DType,
    simd_width: Int,
    unroll: Int,
    alignment: Int,
](
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    num_elements: Int,
    tid: Int,
    block_size: Int,
    mut best_vals: SIMD[dtype, simd_width],
    mut best_idxs: SIMD[.int32, simd_width],
):
    """Streams one row once, accumulating a per-lane (max, index)."""
    var lane_stride = block_size * simd_width
    var trip = lane_stride * unroll
    var num_trips = num_elements // trip
    var vec_end = align_down(num_elements, simd_width)

    for t in range(num_trips):
        var offset = t * trip + tid * simd_width
        var staged = Array[SIMD[dtype, simd_width], unroll](
            fill=SIMD[dtype, simd_width](_topk_dead_val[dtype, True]())
        )
        comptime for u in range(unroll):
            staged[u] = row.unsafe_load[width=simd_width, alignment=alignment](
                offset + u * lane_stride
            )
        comptime for u in range(unroll):
            _argmax_vec_update[dtype](
                best_vals, best_idxs, staged[u], offset + u * lane_stride
            )

    var i = num_trips * trip + tid * simd_width
    while i + simd_width <= vec_end:
        _argmax_vec_update[dtype](
            best_vals,
            best_idxs,
            row.unsafe_load[width=simd_width, alignment=alignment](i),
            i,
        )
        i += lane_stride


@inline(.always)
def _argmax_collapse_lanes[
    dtype: DType, simd_width: Int
](
    best_vals: SIMD[dtype, simd_width],
    best_idxs: SIMD[.int32, simd_width],
) -> TopK_2[dtype, True]:
    """Collapses per-lane running (max, index) pairs into one.

    Lanes hold indices from different trips, so the tie-break has to compare
    indices explicitly rather than rely on lane order.
    """
    var partial = TopK_2[dtype, True]()
    comptime for lane in range(simd_width):
        var val = best_vals[lane]
        var idx = Int(best_idxs[lane])
        if val > partial.u or (val == partial.u and idx < partial.p):
            partial.u = val
            partial.p = idx
    return partial


@inline(.always)
def _argmax_block_partial[
    dtype: DType, simd_width: Int, unroll: Int
](
    row: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    begin: Int,
    count: Int,
    aligned: Bool,
    tid: Int,
    block_size: Int,
) -> TopK_2[dtype, True]:
    """Reduces `row[begin : begin + count]` to one (max, global index)."""
    var best_vals = SIMD[dtype, simd_width](_topk_dead_val[dtype, True]())
    var best_idxs = SIMD[.int32, simd_width](0)
    var chunk = row.unsafe_offset(begin)

    if aligned:
        _argmax_scan[
            dtype,
            simd_width,
            unroll,
            align_of[SIMD[dtype, simd_width]](),
        ](chunk, count, tid, block_size, best_vals, best_idxs)
    else:
        _argmax_scan[dtype, simd_width, unroll, align_of[Scalar[dtype]]()](
            chunk, count, tid, block_size, best_vals, best_idxs
        )

    var partial = _argmax_collapse_lanes[dtype](best_vals, best_idxs)

    # Ragged tail: fewer than `simd_width` elements, all at higher indices
    # than anything scanned above, so a strict insert keeps first-index.
    var vec_end = align_down(count, simd_width)
    if tid < count - vec_end:
        partial.insert(chunk.unsafe_offset(vec_end + tid)[], vec_end + tid)

    partial.p += begin
    return partial


@__name(t"argmax_combine_{dtype}_{out_idx_type}")
def _argmax_combine_kernel[
    dtype: DType,
    out_idx_type: DType,
    OutIdxLayoutType: TensorLayout,
    OutIdxEngine: TensorEngine,
](
    out_idxs: TileTensor[
        mut=True,
        out_idx_type,
        OutIdxLayoutType,
        MutAnyOrigin,
        Engine=OutIdxEngine,
    ],
    part_vals: ImmPointer[Scalar[dtype], ImmutAnyOrigin],
    part_idxs: ImmPointer[Int32, ImmutAnyOrigin],
    num_splits_arg: Int32,
):
    """Reduces one row's per-slice winners to the final index.

    Parameters:
        dtype: Element type of the values.
        out_idx_type: Output index dtype.
        OutIdxLayoutType: Layout of the output indices.
        OutIdxEngine: Engine of the output indices.

    Args:
        out_idxs: One index per row.
        part_vals: Per-slice winning values from the scan kernel.
        part_idxs: Per-slice winning indices from the scan kernel.
        num_splits_arg: Slices per row.
    """
    var num_splits = Int(num_splits_arg)
    var row_id = block_idx.x
    var tid = thread_idx.x
    var base = row_id * num_splits

    with PDL():
        # Slices are scanned in increasing index order and carry global
        # indices, so a strict insert plus the block reduce's lowest-index
        # tie-break reproduces a single-pass first-index scan.
        var partial = TopK_2[dtype, True]()
        for i in range(tid, num_splits, block_dim.x):
            partial.insert(
                part_vals.unsafe_offset(base + i)[],
                Int(part_idxs.unsafe_offset(base + i)[]),
            )

        var total = _block_reduce_topk[ascending=True](partial)

        if tid == 0:
            out_idxs.reshape(row_major(out_idxs.num_elements())).store(
                Coord(row_id), Int(total.p).cast[out_idx_type]()
            )


# ===-----------------------------------------------------------------------===#
# argmax of logits + bias
# ===-----------------------------------------------------------------------===#


@inline(.always)
def _argmax_biased_scan[
    dtype: DType,
    bias_dtype: DType,
    simd_width: Int,
    unroll: Int,
    alignment: Int,
    bias_alignment: Int,
    logits_layout: TensorLayout,
    logits_engine: TensorEngine,
    bias_layout: TensorLayout,
    bias_engine: TensorEngine,
](
    logits: TileTensor[
        dtype, logits_layout, ImmutAnyOrigin, Engine=logits_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_layout, ImmutAnyOrigin, Engine=bias_engine
    ],
    row_id: Int,
    position: Int,
    col_begin: Int,
    num_elements: Int,
    tid: Int,
    block_size: Int,
    mut best_vals: SIMD[.float32, simd_width],
    mut best_idxs: SIMD[.int32, simd_width],
):
    """Like `_argmax_scan`, over `logits + bias` added in float32.

    Reads `logits[row_id, position, col_begin:]` and `bias[row_id, col_begin:]`.
    """
    var lane_stride = block_size * simd_width
    var trip = lane_stride * unroll
    var num_trips = num_elements // trip
    var vec_end = align_down(num_elements, simd_width)

    for t in range(num_trips):
        var offset = t * trip + tid * simd_width
        var staged = Array[SIMD[dtype, simd_width], unroll](
            fill=SIMD[dtype, simd_width](0)
        )
        var staged_bias = Array[SIMD[bias_dtype, simd_width], unroll](
            fill=SIMD[bias_dtype, simd_width](0)
        )
        comptime for u in range(unroll):
            staged[u] = logits.load[
                width=simd_width, alignment=alignment, invariant=False
            ]((row_id, position, col_begin + offset + u * lane_stride))
            staged_bias[u] = bias.load[
                width=simd_width, alignment=bias_alignment, invariant=False
            ]((row_id, col_begin + offset + u * lane_stride))
        comptime for u in range(unroll):
            _argmax_vec_update[.float32](
                best_vals,
                best_idxs,
                staged[u].cast[.float32]() + staged_bias[u].cast[.float32](),
                offset + u * lane_stride,
            )

    var i = num_trips * trip + tid * simd_width
    while i + simd_width <= vec_end:
        _argmax_vec_update[.float32](
            best_vals,
            best_idxs,
            logits.load[width=simd_width, alignment=alignment, invariant=False](
                (row_id, position, col_begin + i)
            ).cast[.float32]()
            + bias.load[
                width=simd_width, alignment=bias_alignment, invariant=False
            ]((row_id, col_begin + i)).cast[.float32](),
            i,
        )
        i += lane_stride


@__name(t"argmax_biased_scan_{dtype}_{bias_dtype}")
def _argmax_biased_scan_kernel[
    dtype: DType,
    bias_dtype: DType,
    simd_width: Int,
    unroll: Int,
    logits_layout: TensorLayout,
    logits_engine: TensorEngine,
    bias_layout: TensorLayout,
    bias_engine: TensorEngine,
](
    logits: TileTensor[
        dtype, logits_layout, ImmutAnyOrigin, Engine=logits_engine
    ],
    bias: TileTensor[
        bias_dtype, bias_layout, ImmutAnyOrigin, Engine=bias_engine
    ],
    part_vals: MutPointer[Float32, MutAnyOrigin],
    part_idxs: MutPointer[Int32, MutAnyOrigin],
    num_elements_arg: Int32,
    position_arg: Int32,
    split_len_arg: Int32,
    num_splits_arg: Int32,
    aligned_arg: Int32,
):
    """Streams one slice of one row of `logits + bias` and emits its winner.

    Row `r` reads `logits[r, position_arg, :]`, so the caller can pick one
    position out of a `[rows, positions, n]` tensor without a copy.

    Parameters:
        dtype: Element type of the logits.
        bias_dtype: Element type of the bias.
        simd_width: Elements per vector load.
        unroll: Vector loads issued back to back per loop trip.
        logits_layout: Layout of the logits.
        logits_engine: Engine of the logits.
        bias_layout: Layout of the bias.
        bias_engine: Engine of the bias.

    Args:
        logits: `[rows, positions, n]` logits.
        bias: `[rows, n]` bias.
        part_vals: Per-slice winning float32 values, `rows * num_splits`.
        part_idxs: Per-slice winning indices, row-relative.
        num_elements_arg: Length of each row.
        position_arg: Position of `logits` to read.
        split_len_arg: Elements per slice, a multiple of `simd_width`.
        num_splits_arg: Slices per row.
        aligned_arg: Non-zero when vector loads may assume vector alignment
            on both inputs.
    """
    var num_elements = Int(num_elements_arg)
    var row_id = block_idx.x
    var split = block_idx.y
    var tid = thread_idx.x

    var begin = min(split * Int(split_len_arg), num_elements)
    var count = min(Int(split_len_arg), num_elements - begin)

    with PDL():
        var best_vals = SIMD[.float32, simd_width](
            _topk_dead_val[.float32, True]()
        )
        var best_idxs = SIMD[.int32, simd_width](0)
        if aligned_arg != 0:
            _argmax_biased_scan[
                dtype,
                bias_dtype,
                simd_width,
                unroll,
                align_of[SIMD[dtype, simd_width]](),
                align_of[SIMD[bias_dtype, simd_width]](),
                logits_layout,
                logits_engine,
                bias_layout,
                bias_engine,
            ](
                logits,
                bias,
                row_id,
                Int(position_arg),
                begin,
                count,
                tid,
                block_dim.x,
                best_vals,
                best_idxs,
            )
        else:
            _argmax_biased_scan[
                dtype,
                bias_dtype,
                simd_width,
                unroll,
                align_of[Scalar[dtype]](),
                align_of[Scalar[bias_dtype]](),
                logits_layout,
                logits_engine,
                bias_layout,
                bias_engine,
            ](
                logits,
                bias,
                row_id,
                Int(position_arg),
                begin,
                count,
                tid,
                block_dim.x,
                best_vals,
                best_idxs,
            )

        var partial = _argmax_collapse_lanes[.float32](best_vals, best_idxs)
        # Ragged tail, as in `_argmax_block_partial`.
        var vec_end = align_down(count, simd_width)
        if tid < count - vec_end:
            partial.insert(
                rebind[Scalar[dtype]](
                    logits.load[invariant=False](
                        (row_id, Int(position_arg), begin + vec_end + tid)
                    )
                ).cast[.float32]()
                + rebind[Scalar[bias_dtype]](
                    bias.load[invariant=False]((row_id, begin + vec_end + tid))
                ).cast[.float32](),
                vec_end + tid,
            )
        partial.p += begin

        var total = _block_reduce_topk[ascending=True](partial)
        if tid == 0:
            var slot = row_id * Int(num_splits_arg) + split
            part_vals.unsafe_offset(slot)[] = total.u
            part_idxs.unsafe_offset(slot)[] = Int32(total.p)


def argmax_with_bias_gpu[
    dtype: DType, bias_dtype: DType, output_type: DType
](
    ctx: DeviceContext,
    logits: TileTensor[dtype, ...],
    bias: TileTensor[bias_dtype, ...],
    position: Int,
    output: TileTensor[mut=True, output_type, ...],
) raises:
    """Takes the argmax of one position's logits plus a per-row bias.

    For each row `r`, writes the index of the largest
    `float32(logits[r, position, i]) + float32(bias[r, i])`. Both inputs
    are upcast to float32 in registers and added there, which is exact for
    16-bit inputs, so the result is the same as materializing the float32 sum
    and taking its argmax: the lowest index wins a tie, NaN never wins,
    and a row with nothing above -inf reports 0. No input is copied or
    relaid out: the kernel reads `logits[:, position, :]` in place.

    Parameters:
        dtype: Element type of the logits.
        bias_dtype: Element type of the bias.
        output_type: Element type of the output indices.

    Args:
        ctx: Device context for launching the kernels.
        logits: `[rows, positions, n]` logits, contiguous.
        bias: `[rows, n]` bias, contiguous.
        position: Which position of `logits` to read, in `[0, positions)`.
        output: `[rows]` output indices.
    """
    comptime assert logits.rank == 3, "logits must be rank 3"
    comptime assert bias.rank == 2, "bias must be rank 2"
    comptime assert output.rank == 1, "output must be rank 1"

    var shape = coord_to_index_list(logits.layout.shape_coord())
    var bias_shape = coord_to_index_list(bias.layout.shape_coord())
    var num_rows = shape[0]
    var positions = shape[1]
    var num_elements = shape[2]

    def trace_information() {imm} -> String:
        return String(";").join(
            Span(
                [
                    trace_arg("logits", shape, dtype),
                    trace_arg("bias", bias_shape, bias_dtype),
                    "position=" + String(position),
                ]
            )
        )

    with Trace[TraceLevel.OP, target=StaticString("gpu")](
        "argmax_with_bias_gpu",
        Trace[TraceLevel.OP]._get_detail_str(trace_information),
        task_id=Int(ctx.id()),
    ):
        if position < 0 or position >= positions:
            raise Error(
                "argmax_with_bias_gpu: position ",
                position,
                " is outside [0, ",
                positions,
                ")",
            )
        if bias_shape[0] != num_rows or bias_shape[1] != num_elements:
            raise Error("argmax_with_bias_gpu: bias must be [rows, n]")
        if Int(coord_to_index_list(output.layout.shape_coord())[0]) != num_rows:
            raise Error("argmax_with_bias_gpu: output must be [rows]")
        if num_rows == 0 or num_elements == 0:
            return
        if num_elements > Int(Int32.MAX):
            raise Error(
                "argmax_with_bias_gpu: inner dimension exceeds the int32 index"
                " range"
            )

        # Eight 16-bit elements per load; the float32 sum stays in registers.
        comptime simd_width = 8

        # Split rows by bytes read, counting both inputs.
        var row_bytes = num_elements * (
            size_of[Scalar[dtype]]() + size_of[Scalar[bias_dtype]]()
        )
        var num_splits = min(
            ceildiv(row_bytes, _ARGMAX_BYTES_PER_BLOCK),
            max(Int(ctx.default_device_info.sm_count) // num_rows, 1),
        )
        num_splits = min(max(num_splits, 1), _ARGMAX_MAX_SPLITS)
        var split_len = (
            ceildiv(ceildiv(num_elements, num_splits), simd_width) * simd_width
        )

        var logits_ptr = logits.ptr.unsafe_offset(position * num_elements)
        var aligned = (
            num_elements % simd_width == 0
            and Int(logits_ptr) % align_of[SIMD[dtype, simd_width]]() == 0
            and Int(bias.ptr) % align_of[SIMD[bias_dtype, simd_width]]() == 0
        )

        comptime warp_size = ctx.default_device_info.warp_size
        var block_size = min(
            next_power_of_two(ceildiv(split_len, simd_width)), 1024
        )
        block_size = max(block_size, warp_size)
        # Same-type scans with both vocabularies static use 70--77 registers per
        # thread on B200, exceeding the 64-register budget at 1024 threads.
        # Use 512 threads for these specializations; keep other specializations
        # on their existing launch sizes.
        comptime if (
            ctx.target.is_nvidia_gpu()
            and dtype == bias_dtype
            and (
                logits.LayoutType._shape_types[2].is_static_value
                and bias.LayoutType._shape_types[1].is_static_value
            )
        ):
            block_size = min(block_size, 512)
        var combine_block_size = max(next_power_of_two(num_splits), warp_size)
        comptime if ctx.target.is_apple_gpu():
            block_size = warp_size
            combine_block_size = warp_size

        var part_vals = ctx.enqueue_create_buffer[.float32](
            num_rows * num_splits
        )
        var part_idxs = ctx.enqueue_create_buffer[.int32](num_rows * num_splits)

        comptime scan_kernel = _argmax_biased_scan_kernel[
            dtype,
            bias_dtype,
            simd_width,
            _ARGMAX_UNROLL,
            type_of(logits).LayoutType,
            type_of(logits).Engine,
            type_of(bias).LayoutType,
            type_of(bias).Engine,
        ]
        ctx.enqueue_function[scan_kernel](
            logits.as_unsafe_any_origin(),
            bias.as_unsafe_any_origin(),
            part_vals,
            part_idxs,
            Int32(num_elements),
            Int32(position),
            Int32(split_len),
            Int32(num_splits),
            Int32(1) if aligned else Int32(0),
            grid_dim=(num_rows, num_splits),
            block_dim=block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )
        comptime combine_kernel = _argmax_combine_kernel[
            .float32,
            output_type,
            output.LayoutType,
            type_of(output).Engine,
        ]
        ctx.enqueue_function[combine_kernel](
            output,
            part_vals,
            part_idxs,
            Int32(num_splits),
            grid_dim=num_rows,
            block_dim=combine_block_size,
            attributes=pdl_launch_attributes(PDLLevel.ON),
        )

        _ = part_vals^
        _ = part_idxs^
