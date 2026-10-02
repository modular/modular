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
"""Benchmark the token-parallel `gated_delta_conv1d_fwd_gpu` against the
retired sequential kernel (inlined below as the reference).

The sequential kernel runs one thread per (batch_item, conv_channel); the
production kernel runs one thread per (token tile, conv_channel), with the
head thread producing positions 0..K-2. Timing follows
`bench_gated_delta_recurrence.mojo`: 5 warmup, 30 timed, median.

Shapes: the Qwen3.8-27B TP2 conv_dim shard (5120) at prefill chunk lengths,
and decode/spec-decode batches.
"""

from max.gpu import block_idx, thread_idx
from std.math import ceildiv

from max.gpu.host import DeviceContext
from layout import (
    Coord,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from state_space.gated_delta_conv1d import (
    CONV1D_TOKENS_PER_BLOCK,
    gated_delta_conv1d_fwd_gpu,
)

comptime KERNEL_SIZE: Int = 4
comptime CONV1D_BLOCK_DIM: Int = 128
comptime WARMUP: Int = 5
comptime ITERS: Int = 30


def _median(times: List[Float64]) -> Float64:
    var s = List[Float64]()
    for t in times:
        var inserted = False
        for j in range(len(s)):
            if t < s[j]:
                s.insert(j, t)
                inserted = True
                break
        if not inserted:
            s.append(t)
    return s[len(s) // 2]


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

    var slot = Int(slot_idx.raw_load(batch_item_idx))
    debug_assert(
        0 <= slot < Int(conv_state.dim[0]()), "conv state slot out of range"
    )
    var sequence_start_flat_idx = Int(
        input_row_offsets.raw_load(batch_item_idx)
    )
    var sequence_end_flat_idx = Int(
        input_row_offsets.raw_load(batch_item_idx + 1)
    )
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    var weight_register = SIMD[work_dtype, KERNEL_SIZE](0)
    comptime for kernel_offset_k in range(KERNEL_SIZE):
        var weight_flat_offset = (
            UInt32(conv_channel_idx) * conv_weight_channel_stride
            + UInt32(kernel_offset_k) * conv_weight_offset_stride
        )
        weight_register[kernel_offset_k] = conv_weight.raw_load(
            weight_flat_offset
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
                var ragged_flat_offset = (
                    UInt32(sequence_start_flat_idx + lookback_position)
                    * qkv_input_seqlen_stride
                    + UInt32(conv_channel_idx) * qkv_input_channel_stride
                )
                input_value = Float32(
                    qkv_input_ragged.raw_load(ragged_flat_offset)
                )
            else:
                var window_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                if window_idx >= 0:
                    input_value = Float32(
                        conv_state.load(
                            Coord(slot, conv_channel_idx, window_idx)
                        )[0]
                    )

            conv_sum += input_value * Float32(weight_register[kernel_offset_k])

        var output_flat_offset = (
            UInt32(flat_token_idx) * conv_output_seqlen_stride
            + UInt32(conv_channel_idx) * conv_output_channel_stride
        )
        conv_output_ragged.raw_store(
            output_flat_offset, Scalar[work_dtype](conv_sum)
        )

    comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
        var source_position_in_sequence = (
            sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
        )

        var state_value: Scalar[state_dtype] = 0
        if source_position_in_sequence >= 0:
            var source_flat_offset = (
                UInt32(sequence_start_flat_idx + source_position_in_sequence)
                * qkv_input_seqlen_stride
                + UInt32(conv_channel_idx) * qkv_input_channel_stride
            )
            state_value = Scalar[state_dtype](
                qkv_input_ragged.raw_load(source_flat_offset)
            )
        else:
            var old_window_idx = (
                KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            )
            if old_window_idx >= 0:
                state_value = conv_state.load(
                    Coord(slot, conv_channel_idx, old_window_idx)
                )[0]

        conv_state.store(
            Coord(slot, conv_channel_idx, state_slot_j), state_value
        )


def _bench(
    ctx: DeviceContext,
    num_seqs: Int,
    seq_len: Int,
    conv_dim: Int,
) raises:
    comptime work_dtype = DType.float32
    comptime state_dtype = DType.bfloat16
    comptime state_len = KERNEL_SIZE - 1
    var total_T = seq_len * num_seqs
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
    var slot_d = ctx.enqueue_create_buffer[.uint32](num_seqs)
    var offsets_d = ctx.enqueue_create_buffer[.uint32](num_seqs + 1)
    var out_new_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)
    var out_old_d = ctx.enqueue_create_buffer[work_dtype](total_T * conv_dim)

    var qkv_h = alloc[Scalar[work_dtype]](total_T * conv_dim)
    for i in range(total_T * conv_dim):
        qkv_h[i] = Scalar[work_dtype](Float32((i % 1000) - 500) * 0.003)
    ctx.enqueue_copy(qkv_d, qkv_h)
    var weight_h = alloc[Scalar[work_dtype]](conv_dim * KERNEL_SIZE)
    for i in range(conv_dim * KERNEL_SIZE):
        weight_h[i] = Scalar[work_dtype](0.25 - Float32(i % 7) * 0.01)
    ctx.enqueue_copy(weight_d, weight_h)
    var pool_h = alloc[Scalar[state_dtype]](pool_size)
    for i in range(pool_size):
        pool_h[i] = Scalar[state_dtype](0.1 - Float32(i % 11) * 0.005)
    ctx.enqueue_copy(pool_new_d, pool_h)
    ctx.enqueue_copy(pool_old_d, pool_h)
    var slot_h = alloc[Scalar[.uint32]](num_seqs)
    for b in range(num_seqs):
        slot_h[b] = Scalar[.uint32](b)
    ctx.enqueue_copy(slot_d, slot_h)
    var offsets_h = alloc[Scalar[.uint32]](num_seqs + 1)
    for b in range(num_seqs + 1):
        offsets_h[b] = Scalar[.uint32](b * seq_len)
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
            UInt32(conv_dim),
            UInt32(1),
            UInt32(KERNEL_SIZE),
            UInt32(1),
            UInt32(conv_dim),
            UInt32(1),
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

    for _ in range(WARMUP):
        launch_new(ctx)
    ctx.synchronize()
    var times = List[Float64]()
    for _ in range(ITERS):
        times.append(Float64(ctx.execution_time(launch_new, 1)))
    ctx.synchronize()
    var new_us = _median(times) / 1000.0

    for _ in range(WARMUP):
        launch_old(ctx)
    ctx.synchronize()
    times = List[Float64]()
    for _ in range(ITERS):
        times.append(Float64(ctx.execution_time(launch_old, 1)))
    ctx.synchronize()
    var old_us = _median(times) / 1000.0

    print(
        "conv_dim=",
        conv_dim,
        "  ",
        num_seqs,
        " rows x ",
        seq_len,
        " tokens  span=",
        tokens_per_block,
        "   old ",
        old_us,
        " us   new ",
        new_us,
        " us   speedup ",
        old_us / new_us,
        sep="",
    )

    _ = qkv_d^
    _ = weight_d^
    _ = pool_new_d^
    _ = pool_old_d^
    _ = slot_d^
    _ = offsets_d^
    _ = out_new_d^
    _ = out_old_d^


def main() raises:
    with DeviceContext() as ctx:
        print("gated_delta_conv1d, median us per call (fp32 work, bf16 pool)")
        # Production Qwen3.8-27B TP2 prefill shard.
        _bench(ctx, 1, 2048, 5120)
        _bench(ctx, 1, 8192, 5120)
        # Decode / spec-decode.
        _bench(ctx, 20, 1, 5120)
        _bench(ctx, 20, 4, 5120)
