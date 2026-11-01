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
"""Benchmark the current token-parallel `gated_delta_conv1d_fwd_gpu`.

Compare source revisions with the same harness for optimization A/Bs. The
optional `--legacy=True` also times the retired sequential kernel below;
that comparison is not evidence for a change to the current kernel.

The adjacent YAML manifest owns the grids. Named CLI arguments select one
sequence-length vector and channel count.
Each sample starts from identical state restored outside the event interval.
Timing uses 5 warmup launches, 30 single-launch samples, and no L2 flush.
"""

from max.gpu import block_idx, thread_idx
from std.benchmark import Batch, Bench, BenchmarkInfo, Report
from std.builtin.sort import sort
from std.math import ceildiv

from internal_utils import arg_parse
from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from layout.tile_layout import Layout
from state_space.gated_delta_conv1d import (
    CONV1D_TOKENS_PER_BLOCK,
    gated_delta_conv1d_fwd_gpu,
)

comptime KERNEL_SIZE = 4
comptime CONV1D_BLOCK_DIM: Int = 128
comptime WARMUP: Int = 5
comptime ITERS: Int = 30


def _median(times: List[Float64]) -> Float64:
    var ordered = times.copy()
    sort(Span(ordered))
    var middle = len(ordered) // 2
    if len(ordered) % 2 == 0:
        return (ordered[middle - 1] + ordered[middle]) / 2
    return ordered[middle]


def gated_delta_conv1d_sequential_reference[
    work_dtype: DType,  # for qkv_input_ragged / conv_weight / conv_output_ragged
    state_dtype: DType,  # for the conv_state pool (typically bf16)
    KERNEL_SIZE: Int,
    CONV1D_BLOCK_DIM: Int,
    qkv_input_ragged_LT: TensorLayout,
    conv_weight_LT: TensorLayout,
    conv_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
    conv_output_ragged_LT: TensorLayout,
    Engine: TensorEngine,
](
    batch_size: Int32,
    total_seq_len: Int32,
    conv_dim: Int32,
    qkv_input_ragged: TileTensor[
        work_dtype, qkv_input_ragged_LT, MutUntrackedOrigin, Engine=Engine
    ],
    conv_weight: TileTensor[
        work_dtype, conv_weight_LT, MutUntrackedOrigin, Engine=Engine
    ],
    conv_state: TileTensor[
        state_dtype, conv_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    slot_idx: TileTensor[
        .uint32, slot_idx_LT, MutUntrackedOrigin, Engine=Engine
    ],
    input_row_offsets: TileTensor[
        .uint32, input_row_offsets_LT, MutUntrackedOrigin, Engine=Engine
    ],
    conv_output_ragged: TileTensor[
        work_dtype, conv_output_ragged_LT, MutUntrackedOrigin, Engine=Engine
    ],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_input_seqlen_stride: UInt32,  # stride along total_seq_len axis
    qkv_input_channel_stride: UInt32,  # stride along conv_dim axis (usually 1)
    conv_weight_channel_stride: UInt32,  # stride along conv_dim axis
    conv_weight_offset_stride: UInt32,  # stride along kernel_size axis
    # Output strides (match input strides for conv_output_ragged)
    conv_output_seqlen_stride: UInt32,
    conv_output_channel_stride: UInt32,
):
    """Retired sequential kernel, kept as the bit-exact reference."""
    var batch_item_idx = block_idx.x
    var conv_channel_idx = block_idx.y * CONV1D_BLOCK_DIM + thread_idx.x

    if batch_item_idx >= Int(batch_size) or conv_channel_idx >= Int(conv_dim):
        return

    var slot = Int(rebind[UInt32](slot_idx.load((batch_item_idx,))))
    debug_assert(
        0 <= slot < Int(conv_state.dim[0]()), "conv state slot out of range"
    )
    var sequence_start_flat_idx = Int(
        rebind[UInt32](input_row_offsets.load((batch_item_idx,)))
    )
    var sequence_end_flat_idx = Int(
        rebind[UInt32](input_row_offsets.load((batch_item_idx + 1,)))
    )
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    var qkv_input_view = qkv_input_ragged.reshape(
        Layout(
            Coord(Int(total_seq_len), Int(conv_dim)),
            Coord(Int(qkv_input_seqlen_stride), Int(qkv_input_channel_stride)),
        )
    )
    var conv_weight_view = conv_weight.reshape(
        Layout(
            Coord(Int(conv_dim), KERNEL_SIZE),
            Coord(
                Int(conv_weight_channel_stride), Int(conv_weight_offset_stride)
            ),
        )
    )
    var conv_output_view = conv_output_ragged.reshape(
        Layout(
            Coord(Int(total_seq_len), Int(conv_dim)),
            Coord(
                Int(conv_output_seqlen_stride), Int(conv_output_channel_stride)
            ),
        )
    )

    var weight_register = SIMD[work_dtype, KERNEL_SIZE](0)
    comptime for kernel_offset_k in range(KERNEL_SIZE):
        weight_register[kernel_offset_k] = rebind[Scalar[work_dtype]](
            conv_weight_view.load((conv_channel_idx, kernel_offset_k))
        )

    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    for token_position_in_sequence in range(sequence_length):
        var flat_token_idx = (
            sequence_start_flat_idx + token_position_in_sequence
        )
        var conv_sum = Float32(0.0)

        comptime for kernel_offset_k in range(KERNEL_SIZE):
            var lookback_position = token_position_in_sequence - (
                KERNEL_SIZE_MINUS_ONE - kernel_offset_k
            )

            var input_value: Float32 = 0

            if lookback_position >= 0:
                input_value = Float32(
                    rebind[Scalar[work_dtype]](
                        qkv_input_view.load(
                            (
                                sequence_start_flat_idx + lookback_position,
                                conv_channel_idx,
                            )
                        )
                    )
                )
            else:
                var window_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                if window_idx >= 0:
                    input_value = Float32(
                        conv_state.load((slot, conv_channel_idx, window_idx))[0]
                    )

            conv_sum += input_value * Float32(weight_register[kernel_offset_k])

        conv_output_view.store(
            (flat_token_idx, conv_channel_idx),
            Scalar[work_dtype](conv_sum),
        )

    comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
        var source_position_in_sequence = (
            sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
        )

        var state_value: Scalar[state_dtype] = 0
        if source_position_in_sequence >= 0:
            state_value = Scalar[state_dtype](
                rebind[Scalar[work_dtype]](
                    qkv_input_view.load(
                        (
                            sequence_start_flat_idx
                            + source_position_in_sequence,
                            conv_channel_idx,
                        )
                    )
                )
            )
        else:
            var old_window_idx = (
                KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            )
            if old_window_idx >= 0:
                state_value = conv_state.load(
                    (slot, conv_channel_idx, old_window_idx)
                )[0]

        conv_state.store((slot, conv_channel_idx, state_slot_j), state_value)


def _bench(
    ctx: DeviceContext,
    seq_lengths: List[Int],
    conv_dim: Int,
    compare_legacy: Bool = False,
) raises:
    comptime work_dtype = DType.float32
    comptime state_dtype = DType.bfloat16
    comptime state_len = KERNEL_SIZE - 1
    var num_seqs = len(seq_lengths)
    var total_T = 0
    var length_strings = List[String]()
    for seq_len in seq_lengths:
        total_T += seq_len
        length_strings.append(String(seq_len))
    var lengths = ":".join(length_strings)
    # Same tiling rule as the op registration.
    var tokens_per_block = min(
        CONV1D_TOKENS_PER_BLOCK, max(1, ceildiv(total_T, num_seqs))
    )
    var max_slots = num_seqs
    var pool_size = max_slots * conv_dim * state_len

    var qkv_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)
    var weight_d = ctx.enqueue_create_buffer[work_dtype](conv_dim * KERNEL_SIZE)
    var pool_new_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var pool_old_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var pool_initial_d = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_seqs)
    var offsets_d = ctx.enqueue_create_buffer[.uint32](num_seqs + 1)
    var out_new_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)
    var out_old_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)

    var qkv_h = ctx.enqueue_create_host_buffer[work_dtype](total_T * conv_dim)
    for i in range(total_T * conv_dim):
        qkv_h[i] = Scalar[work_dtype](Float32((i % 1000) - 500) * 0.003)
    ctx.enqueue_copy(qkv_d, qkv_h)
    var weight_h = ctx.enqueue_create_host_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )
    for i in range(conv_dim * KERNEL_SIZE):
        weight_h[i] = Scalar[work_dtype](0.25 - Float32(i % 7) * 0.01)
    ctx.enqueue_copy(weight_d, weight_h)
    var pool_h = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    for i in range(pool_size):
        pool_h[i] = Scalar[state_dtype](0.1 - Float32(i % 11) * 0.005)
    ctx.enqueue_copy(pool_initial_d, pool_h)
    var slot_h = ctx.enqueue_create_host_buffer[.uint32](num_seqs)
    for b in range(num_seqs):
        slot_h[b] = UInt32(b)
    ctx.enqueue_copy(slot_d, slot_h)
    var offsets_h = ctx.enqueue_create_host_buffer[.uint32](num_seqs + 1)
    var offset = 0
    offsets_h[0] = 0
    for b in range(num_seqs):
        offset += seq_lengths[b]
        offsets_h[b + 1] = UInt32(offset)
    ctx.enqueue_copy(offsets_d, offsets_h)
    ctx.synchronize()

    var qkv_tt = TileTensor(qkv_d, row_major(total_T, conv_dim))
    var weight_tt = TileTensor(weight_d, row_major(conv_dim, KERNEL_SIZE))
    var pool_new_tt = TileTensor(
        pool_new_d, row_major(max_slots, conv_dim, state_len)
    )
    var pool_old_tt = TileTensor(
        pool_old_d, row_major(max_slots, conv_dim, state_len)
    )
    var slot_tt = TileTensor(slot_d, row_major(num_seqs))
    var offsets_tt = TileTensor(offsets_d, row_major(num_seqs + 1))
    var out_new_tt = TileTensor(out_new_d, row_major(total_T, conv_dim))
    var out_old_tt = TileTensor(out_old_d, row_major(total_T, conv_dim))

    var compiled_new = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            True,
            qkv_tt.LayoutType,
            weight_tt.LayoutType,
            pool_new_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            out_new_tt.LayoutType,
            qkv_tt.Engine,
        ]
    ]()
    var compiled_old = ctx.compile_function[
        gated_delta_conv1d_sequential_reference[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            qkv_tt.LayoutType,
            weight_tt.LayoutType,
            pool_old_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            out_old_tt.LayoutType,
            qkv_tt.Engine,
        ]
    ]()

    def launch_new(
        lctx: DeviceContext,
    ) raises {
        imm compiled_new,
        imm tokens_per_block,
        imm qkv_tt,
        imm weight_tt,
        imm pool_new_tt,
        imm slot_tt,
        imm offsets_tt,
        imm out_new_tt,
        imm out_old_tt,
        imm total_T,
        imm conv_dim,
        imm num_seqs,
    }:
        lctx.enqueue_function(
            compiled_new,
            Int32(num_seqs),
            Int32(total_T),
            Int32(conv_dim),
            Int32(tokens_per_block),
            qkv_tt,
            weight_tt,
            pool_new_tt,
            slot_tt,
            offsets_tt,
            out_new_tt,
            # Never written without CAPTURE_INPUT.
            out_old_tt,
            grid_dim=(
                ceildiv(total_T, tokens_per_block),
                ceildiv(conv_dim, CONV1D_BLOCK_DIM),
            ),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

    def launch_old(
        lctx: DeviceContext,
    ) raises {
        imm compiled_old,
        imm qkv_tt,
        imm weight_tt,
        imm pool_old_tt,
        imm slot_tt,
        imm offsets_tt,
        imm out_old_tt,
        imm total_T,
        imm conv_dim,
        imm num_seqs,
    }:
        lctx.enqueue_function(
            compiled_old,
            Int32(num_seqs),
            Int32(total_T),
            Int32(conv_dim),
            qkv_tt,
            weight_tt,
            pool_old_tt,
            slot_tt,
            offsets_tt,
            out_old_tt,
            UInt32(conv_dim),
            UInt32(1),
            UInt32(KERNEL_SIZE),
            UInt32(1),
            UInt32(conv_dim),
            UInt32(1),
            grid_dim=(num_seqs, ceildiv(conv_dim, CONV1D_BLOCK_DIM)),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

    var shape = String(
        t"work_dtype={work_dtype} state_dtype={state_dtype} "
        t"conv_dim={conv_dim} kernel_size={KERNEL_SIZE} "
        t"batch={num_seqs} sequence_lengths={lengths} total_tokens={total_T} "
        t"tokens_per_block={tokens_per_block} write_state=true"
    )
    for _ in range(WARMUP):
        ctx.enqueue_copy(pool_new_d, pool_initial_d)
        launch_new(ctx)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(ITERS):
        ctx.enqueue_copy(pool_new_d, pool_initial_d)
        ctx.synchronize()
        times.append(Float64(ctx.execution_time(launch_new, 1)))
    ctx.synchronize()
    for sample in range(len(times)):
        print(
            t"sample kernel=gated_delta_conv1d {shape} "
            t"index={sample} ns={times[sample]}"
        )
    var benchmark = Bench()
    var report = Report()
    for elapsed in times:
        report.runs.append(Batch(Int(elapsed), 1, True))
    benchmark.info_vec.append(BenchmarkInfo(shape, report^))
    var median_ns = _median(times)
    print(
        t"summary kernel=gated_delta_conv1d {shape} "
        t"samples={ITERS} median_ns={median_ns}"
    )

    if compare_legacy:
        for _ in range(WARMUP):
            ctx.enqueue_copy(pool_old_d, pool_initial_d)
            launch_old(ctx)
        ctx.synchronize()
        times = List[Float64]()
        for _ in range(ITERS):
            ctx.enqueue_copy(pool_old_d, pool_initial_d)
            ctx.synchronize()
            times.append(Float64(ctx.execution_time(launch_old, 1)))
        ctx.synchronize()
        for sample in range(len(times)):
            print(
                t"sample kernel=gated_delta_conv1d_legacy {shape} "
                t"index={sample} ns={times[sample]}"
            )
        var legacy_median_ns = _median(times)
        print(
            t"summary kernel=gated_delta_conv1d_legacy {shape} "
            t"samples={ITERS} median_ns={legacy_median_ns}"
        )

        var legacy_report = Report()
        for elapsed in times:
            legacy_report.runs.append(Batch(Int(elapsed), 1, True))
        benchmark.info_vec.append(
            BenchmarkInfo(String(t"legacy {shape}"), legacy_report^)
        )
    benchmark.dump_report()
    _ = qkv_d^
    _ = weight_d^
    _ = pool_new_d^
    _ = pool_old_d^
    _ = pool_initial_d^
    _ = slot_d^
    _ = offsets_d^
    _ = out_new_d^
    _ = out_old_d^


def main() raises:
    var lengths = List[Int]()
    for value in (
        arg_parse("sequence_lengths", String("")).strip("[]").split(",")
    ):
        lengths.append(Int(value))
    var channels = arg_parse("channels", 0)
    var compare_legacy = arg_parse("legacy", False)
    if len(lengths) == 0 or channels <= 0:
        raise Error(
            "Sequence lengths and a positive channel count are required"
        )
    for length in lengths:
        if length < 0:
            raise Error("Sequence lengths must be nonnegative")
    with DeviceContext() as ctx:
        print(
            t"config warmup={WARMUP} samples={ITERS} "
            t"launches_per_sample=1 state_reset=outside_timing l2_flush=false"
            t" csv_statistic=mean raw_statistic=median"
        )
        _bench(ctx, lengths, channels, compare_legacy)
