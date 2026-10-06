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

from std.math import ceildiv

from max.gpu import block_idx, thread_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import (
    Coord,
    Idx,
    TensorEngine,
    TensorLayout,
    TileTensor,
    row_major,
)
from std.random import rand
from state_space.gated_delta_conv1d import (
    CONV1D_TOKENS_PER_BLOCK,
    gated_delta_conv1d_fwd_gpu,
)
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.utils.index import Index, IndexList


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
    """Retired sequential kernel, kept as the FMA-tolerance reference."""
    var batch_item_idx = block_idx.x
    var conv_channel_idx = block_idx.y * CONV1D_BLOCK_DIM + thread_idx.x

    if batch_item_idx >= Int(batch_size) or conv_channel_idx >= Int(conv_dim):
        return

    var slot = Int(slot_idx.raw_load(batch_item_idx))
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


def run_slot_indexed_gpu[
    work_dtype: DType,
    state_dtype: DType,
    KERNEL_SIZE: Int,
    WRITE_STATE: Bool = True,
](
    batch_size: Int,
    total_seq_len: Int,
    conv_dim: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    slot_assignments: IndexList,  # [batch_size] slot indices into the pool
    ctx: DeviceContext,
    tokens_per_block: Int = CONV1D_TOKENS_PER_BLOCK,
    rtol: Float64 = 0.01,
    # Host reference is too slow at production sizes; the sequential-kernel
    # comparison (matches within FMA contraction tolerance) still runs.
    cpu_reference: Bool = True,
) raises:
    """Run the slot-indexed conv1d kernel and check it against a CPU reference.

    Differences from the old gather/scatter path that this exercises:
      - The conv-state pool has shape [max_slots, conv_dim, K-1] and the
        kernel reads/writes slot ``slot_assignments[b]`` for batch item b,
        so slots not referenced by ``slot_assignments`` must remain
        untouched.
      - In-place mutation: there is no conv_state_out tensor.

    With ``WRITE_STATE = False`` the reference pool is left unchanged, so
    the pool comparison checks the kernel wrote nothing.
    """
    comptime CONV1D_BLOCK_DIM = 128

    var state_len = KERNEL_SIZE - 1

    # ── Host tensors ────────────────────────────────────────────────────────
    # qkv_input_ragged: [total_seq_len, conv_dim]
    var qkv_input_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    # conv_weight: [conv_dim, KERNEL_SIZE]
    var conv_weight_heap = ctx.enqueue_create_host_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )

    # Pool: [max_slots, conv_dim, K-1]. Filled with a recognisable pattern so
    # that any unintended write to the wrong slot is caught by the equality
    # check at the end.
    var pool_size = max_slots * conv_dim * state_len
    var conv_state_initial_h_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    rand[state_dtype](conv_state_initial_h_heap.unsafe_ptr(), pool_size)

    # Slot assignments device buffer.
    var slot_idx_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        slot_idx_heap[b] = UInt32(slot_assignments[b])

    # input_row_offsets: [batch_size + 1]
    var input_row_offsets_heap = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cumsum = 0
    input_row_offsets_heap[0] = UInt32(0)
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        input_row_offsets_heap[b + 1] = UInt32(cumsum)

    var conv_output_gpu_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    var pool_after_gpu_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )

    var conv_output_ref_gpu_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var pool_after_ref_gpu_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )

    rand[work_dtype](qkv_input_heap.unsafe_ptr(), len(qkv_input_heap))
    rand[work_dtype](conv_weight_heap.unsafe_ptr(), len(conv_weight_heap))

    # ── Device buffers ──────────────────────────────────────────────────────
    var qkv_input_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var conv_weight_device = ctx.enqueue_create_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )
    var conv_state_device = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    var conv_output_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var conv_state_ref_device = ctx.enqueue_create_buffer[state_dtype](
        pool_size
    )
    var conv_output_ref_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_input_device, qkv_input_heap)
        ctx.enqueue_copy(conv_weight_device, conv_weight_heap)
        ctx.enqueue_copy(conv_state_device, conv_state_initial_h_heap)
        ctx.enqueue_copy(conv_state_ref_device, conv_state_initial_h_heap)
        ctx.enqueue_copy(slot_idx_device, slot_idx_heap)
        ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_heap)

    var qkv_input_tt = TileTensor(
        qkv_input_device, row_major(total_seq_len, conv_dim)
    )
    var conv_weight_tt = TileTensor(
        conv_weight_device, row_major(conv_dim, KERNEL_SIZE)
    )
    var conv_state_tt = TileTensor(
        conv_state_device,
        row_major(max_slots, conv_dim, state_len),
    )
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var input_row_offsets_tt = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )
    var conv_output_tt = TileTensor(
        conv_output_device, row_major(total_seq_len, conv_dim)
    )
    var conv_state_ref_tt = TileTensor(
        conv_state_ref_device,
        row_major(max_slots, conv_dim, state_len),
    )
    var conv_output_ref_tt = TileTensor(
        conv_output_ref_device, row_major(total_seq_len, conv_dim)
    )

    var qkv_input_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_input_channel_stride: UInt32 = 1
    var conv_weight_channel_stride: UInt32 = UInt32(KERNEL_SIZE)
    var conv_weight_offset_stride: UInt32 = 1
    # Not passed to the kernel (it indexes `conv_state_tt` via `Coord`), but
    # the CPU reference below still addresses `pool_ref_heap` by hand.
    var conv_state_pool_stride: UInt32 = UInt32(conv_dim * state_len)
    var conv_state_channel_stride: UInt32 = UInt32(state_len)
    var conv_state_window_stride: UInt32 = 1
    var conv_output_seqlen_stride: UInt32 = UInt32(conv_dim)
    var conv_output_channel_stride: UInt32 = 1

    var compiled_func = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            WRITE_STATE,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_tt.LayoutType,
            slot_idx_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
            qkv_input_tt.Engine,
        ]
    ]()
    var compiled_ref = ctx.compile_function[
        gated_delta_conv1d_sequential_reference[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_tt.LayoutType,
            slot_idx_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
            qkv_input_tt.Engine,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch_size),
            Int32(total_seq_len),
            Int32(conv_dim),
            Int32(tokens_per_block),
            qkv_input_tt,
            conv_weight_tt,
            conv_state_tt,
            slot_idx_tt,
            input_row_offsets_tt,
            conv_output_tt,
            grid_dim=(
                ceildiv(total_seq_len, tokens_per_block),
                ceildiv(conv_dim, CONV1D_BLOCK_DIM),
            ),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

        ctx.enqueue_function(
            compiled_ref,
            Int32(batch_size),
            Int32(total_seq_len),
            Int32(conv_dim),
            qkv_input_tt,
            conv_weight_tt,
            conv_state_ref_tt,
            slot_idx_tt,
            input_row_offsets_tt,
            conv_output_ref_tt,
            qkv_input_seqlen_stride,
            qkv_input_channel_stride,
            conv_weight_channel_stride,
            conv_weight_offset_stride,
            conv_output_seqlen_stride,
            conv_output_channel_stride,
            grid_dim=(batch_size, ceildiv(conv_dim, CONV1D_BLOCK_DIM)),
            block_dim=(CONV1D_BLOCK_DIM,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(conv_output_gpu_heap, conv_output_device)
        ctx.enqueue_copy(pool_after_gpu_heap, conv_state_device)
        ctx.enqueue_copy(conv_output_ref_gpu_heap, conv_output_ref_device)
        ctx.enqueue_copy(pool_after_ref_gpu_heap, conv_state_ref_device)
    ctx.synchronize()

    for i in range(total_seq_len * conv_dim):
        assert_almost_equal(
            conv_output_gpu_heap[i],
            conv_output_ref_gpu_heap[i],
            rtol=1e-5,
        )

    for i in range(pool_size):
        comptime if WRITE_STATE:
            assert_equal(pool_after_gpu_heap[i], pool_after_ref_gpu_heap[i])
        else:
            # The reference always writes the window; here the pool must be
            # unchanged.
            assert_equal(pool_after_gpu_heap[i], conv_state_initial_h_heap[i])

    if not cpu_reference:
        return

    # ── CPU reference: scalar gather/scatter to the same pool. ───────────────
    # Only the slots referenced by slot_assignments should change.
    var pool_ref_heap = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    for i in range(pool_size):
        pool_ref_heap[i] = conv_state_initial_h_heap[i]

    var conv_output_ref_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    for b in range(batch_size):
        var slot = slot_assignments[b]
        var seq_start = Int(input_row_offsets_heap[b])
        var seq_end = Int(input_row_offsets_heap[b + 1])
        var seq_len = seq_end - seq_start

        for c in range(conv_dim):
            for t in range(seq_len):
                var conv_sum = Float32(0.0)
                comptime for k in range(KERNEL_SIZE):
                    var lookback = t - (KERNEL_SIZE_MINUS_ONE - k)
                    var input_value = Float32(0.0)
                    if lookback >= 0:
                        input_value = Float32(
                            qkv_input_heap.unsafe_ptr()[
                                UInt32(seq_start + lookback)
                                * qkv_input_seqlen_stride
                                + UInt32(c) * qkv_input_channel_stride
                            ]
                        )
                    else:
                        var slot_pos = KERNEL_SIZE_MINUS_ONE + lookback
                        if slot_pos >= 0:
                            input_value = Float32(
                                pool_ref_heap.unsafe_ptr()[
                                    UInt32(slot) * conv_state_pool_stride
                                    + UInt32(c) * conv_state_channel_stride
                                    + UInt32(slot_pos)
                                    * conv_state_window_stride
                                ]
                            )
                    var w = Float32(
                        conv_weight_heap.unsafe_ptr()[
                            UInt32(c) * conv_weight_channel_stride
                            + UInt32(k) * conv_weight_offset_stride
                        ]
                    )
                    conv_sum = conv_sum + input_value * w

                conv_output_ref_heap.unsafe_ptr().store(
                    UInt32(seq_start + t) * conv_output_seqlen_stride
                    + UInt32(c) * conv_output_channel_stride,
                    Scalar[work_dtype](conv_sum),
                )

            # Update the slot's window with the last K-1 raw inputs (or
            # carry-forward if seq_len < K-1). Reads of the old window
            # complete before any write because the write loop runs after the
            # token loop.
            comptime if WRITE_STATE:
                var old_window = Array[
                    Scalar[state_dtype], KERNEL_SIZE_MINUS_ONE
                ](fill=0)
                comptime for j in range(KERNEL_SIZE_MINUS_ONE):
                    old_window[j] = pool_ref_heap.unsafe_ptr()[
                        UInt32(slot) * conv_state_pool_stride
                        + UInt32(c) * conv_state_channel_stride
                        + UInt32(j) * conv_state_window_stride
                    ]

                comptime for j in range(KERNEL_SIZE_MINUS_ONE):
                    var src = seq_len - KERNEL_SIZE_MINUS_ONE + j
                    var v: Scalar[state_dtype] = 0
                    if src >= 0:
                        v = Scalar[state_dtype](
                            qkv_input_heap.unsafe_ptr()[
                                UInt32(seq_start + src)
                                * qkv_input_seqlen_stride
                                + UInt32(c) * qkv_input_channel_stride
                            ]
                        )
                    else:
                        var old_slot = KERNEL_SIZE_MINUS_ONE + src
                        if old_slot >= 0:
                            v = old_window[old_slot]
                    pool_ref_heap.unsafe_ptr().store(
                        UInt32(slot) * conv_state_pool_stride
                        + UInt32(c) * conv_state_channel_stride
                        + UInt32(j) * conv_state_window_stride,
                        v,
                    )

    # ── Compare ──────────────────────────────────────────────────────────────
    for i in range(total_seq_len * conv_dim):
        assert_almost_equal(
            conv_output_gpu_heap[i],
            conv_output_ref_heap[i],
            rtol=rtol,
        )

    for i in range(pool_size):
        assert_almost_equal(
            pool_after_gpu_heap[i],
            pool_ref_heap[i],
            rtol=rtol,
        )


def test_slot_indexed_single_sequence_targets_chosen_slot() raises:
    """One-sequence smoke test: writes only to the slot named in slot_idx."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4](
        batch_size=1,
        total_seq_len=5,
        conv_dim=8,
        max_slots=3,
        seq_lengths=Index(5),
        slot_assignments=Index(2),
        ctx=ctx,
    )


def test_slot_indexed_two_sequences_disjoint_slots() raises:
    """Two sequences hitting non-adjacent slots; bf16 pool, fp32 work."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=2,
        total_seq_len=7,
        conv_dim=8,
        max_slots=4,
        seq_lengths=Index(4, 3),
        slot_assignments=Index(3, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_slot_indexed_short_sequence_carries_state_forward() raises:
    """When seq_len < K-1: window must carry forward from existing pool entry.
    """
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4](
        batch_size=1,
        total_seq_len=2,
        conv_dim=8,
        max_slots=2,
        seq_lengths=Index(2),
        slot_assignments=Index(1),
        ctx=ctx,
    )


def test_suppressing_the_window_write_leaves_the_pool_alone() raises:
    """Checks ``WRITE_STATE = False`` leaves the pool unchanged."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4, WRITE_STATE=False](
        batch_size=2,
        total_seq_len=7,
        conv_dim=8,
        max_slots=4,
        seq_lengths=Index(4, 3),
        slot_assignments=Index(3, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_suppressing_the_window_write_spares_a_carry_forward_row() raises:
    """Checks the same for a row shorter than the window."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4, WRITE_STATE=False](
        batch_size=1,
        total_seq_len=2,
        conv_dim=8,
        max_slots=2,
        seq_lengths=Index(2),
        slot_assignments=Index(1),
        ctx=ctx,
    )


def _launch_conv[
    WRITE_STATE: Bool
](
    ctx: DeviceContext,
    qkv: DeviceBuffer[DType.float32],
    weight: DeviceBuffer[DType.float32],
    pool: DeviceBuffer[DType.float32],
    slot: DeviceBuffer[DType.uint32],
    offsets: DeviceBuffer[DType.uint32],
    conv_out: DeviceBuffer[DType.float32],
    seq_len: Int,
) raises:
    """Runs one kernel-size-4 conv launch over one row of `seq_len` tokens."""
    comptime KERNEL_SIZE = 4
    comptime CONV_DIM = 8
    comptime MAX_SLOTS = 2
    comptime CONV1D_BLOCK_DIM = 128
    var qkv_tt = TileTensor(qkv, row_major(seq_len, CONV_DIM))
    var weight_tt = TileTensor(weight, row_major(CONV_DIM, KERNEL_SIZE))
    var pool_tt = TileTensor(
        pool, row_major(MAX_SLOTS, CONV_DIM, KERNEL_SIZE - 1)
    )
    var slot_tt = TileTensor(slot, row_major(1))
    var offsets_tt = TileTensor(offsets, row_major(2))
    var out_tt = TileTensor(conv_out, row_major(seq_len, CONV_DIM))
    var kernel = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            DType.float32,
            DType.float32,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            WRITE_STATE,
            qkv_tt.LayoutType,
            weight_tt.LayoutType,
            pool_tt.LayoutType,
            slot_tt.LayoutType,
            offsets_tt.LayoutType,
            out_tt.LayoutType,
            qkv_tt.Engine,
        ]
    ]()
    comptime tokens_per_block = 1
    ctx.enqueue_function(
        kernel,
        Int32(1),
        Int32(seq_len),
        Int32(CONV_DIM),
        Int32(tokens_per_block),
        qkv_tt,
        weight_tt,
        pool_tt,
        slot_tt,
        offsets_tt,
        out_tt,
        grid_dim=(
            ceildiv(seq_len, tokens_per_block),
            ceildiv(CONV_DIM, CONV1D_BLOCK_DIM),
        ),
        block_dim=(CONV1D_BLOCK_DIM,),
    )
    ctx.synchronize()


def run_deferred_window_write(accepted: Int) raises:
    """Checks a deferred window, written at `accepted` tokens, is the forward's.

    One launch covers a four-token window with the write suppressed, and a
    second writes the window over the first `accepted` tokens. The pool must
    match a single launch over those tokens from the same starting window,
    which below `KERNEL_SIZE - 1` carries slots of that window forward.
    """
    comptime KERNEL_SIZE = 4
    comptime CONV_DIM = 8
    comptime WINDOW = 4
    comptime POOL_ELEMS = 2 * CONV_DIM * (KERNEL_SIZE - 1)
    var ctx = DeviceContext()

    var qkv_h = ctx.enqueue_create_host_buffer[DType.float32](WINDOW * CONV_DIM)
    rand[DType.float32](qkv_h.unsafe_ptr(), WINDOW * CONV_DIM)
    var weight_h = ctx.enqueue_create_host_buffer[DType.float32](
        CONV_DIM * KERNEL_SIZE
    )
    rand[DType.float32](weight_h.unsafe_ptr(), CONV_DIM * KERNEL_SIZE)
    var pool_h = ctx.enqueue_create_host_buffer[DType.float32](POOL_ELEMS)
    rand[DType.float32](pool_h.unsafe_ptr(), POOL_ELEMS)
    var slot_h = ctx.enqueue_create_host_buffer[DType.uint32](1)
    slot_h[0] = 1
    var window_offsets_h = ctx.enqueue_create_host_buffer[DType.uint32](2)
    window_offsets_h[0] = 0
    window_offsets_h[1] = UInt32(WINDOW)
    var accepted_offsets_h = ctx.enqueue_create_host_buffer[DType.uint32](2)
    accepted_offsets_h[0] = 0
    accepted_offsets_h[1] = UInt32(accepted)

    var qkv = ctx.enqueue_create_buffer[DType.float32](WINDOW * CONV_DIM)
    var weight = ctx.enqueue_create_buffer[DType.float32](
        CONV_DIM * KERNEL_SIZE
    )
    var pool = ctx.enqueue_create_buffer[DType.float32](POOL_ELEMS)
    var reference_pool = ctx.enqueue_create_buffer[DType.float32](POOL_ELEMS)
    var slot = ctx.enqueue_create_buffer[DType.uint32](1)
    var window_offsets = ctx.enqueue_create_buffer[DType.uint32](2)
    var accepted_offsets = ctx.enqueue_create_buffer[DType.uint32](2)
    var conv_out = ctx.enqueue_create_buffer[DType.float32](WINDOW * CONV_DIM)
    ctx.enqueue_copy(qkv, qkv_h)
    ctx.enqueue_copy(weight, weight_h)
    ctx.enqueue_copy(pool, pool_h)
    ctx.enqueue_copy(reference_pool, pool_h)
    ctx.enqueue_copy(slot, slot_h)
    ctx.enqueue_copy(window_offsets, window_offsets_h)
    ctx.enqueue_copy(accepted_offsets, accepted_offsets_h)
    ctx.synchronize()

    _launch_conv[False](
        ctx, qkv, weight, pool, slot, window_offsets, conv_out, WINDOW
    )
    _launch_conv[True](
        ctx, qkv, weight, pool, slot, accepted_offsets, conv_out, WINDOW
    )
    _launch_conv[True](
        ctx,
        qkv,
        weight,
        reference_pool,
        slot,
        accepted_offsets,
        conv_out,
        WINDOW,
    )

    var pool_after = ctx.enqueue_create_host_buffer[DType.float32](POOL_ELEMS)
    var reference_after = ctx.enqueue_create_host_buffer[DType.float32](
        POOL_ELEMS
    )
    ctx.enqueue_copy(pool_after, pool)
    ctx.enqueue_copy(reference_after, reference_pool)
    ctx.synchronize()
    for i in range(POOL_ELEMS):
        assert_equal(
            pool_after[i],
            reference_after[i],
            "the deferred window diverged at element " + String(i),
        )


def test_a_deferred_window_carries_the_old_window_forward() raises:
    run_deferred_window_write(1)
    run_deferred_window_write(2)


def test_a_deferred_window_rebuilt_from_inputs() raises:
    run_deferred_window_write(3)


# =============================================================================
# Regression test: a `conv_state` row past a 32-bit offset
# =============================================================================


def test_gated_delta_conv1d_gpu_deep_slot_no_alias() raises:
    """A `conv_state` slot past 2**32 elements must not alias onto the front
    of the pool.

    The row is kept tiny (`conv_dim=4`, `KERNEL_SIZE=2`, `UInt8`) so that
    `slot_deep * row_elements` lands on 2**32 with a ~4 GiB pool. Only
    the two sentinel bytes are touched from the host.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return

    comptime state_dtype = DType.uint8
    comptime work_dtype = DType.float32
    comptime KERNEL_SIZE = 2
    comptime state_len = KERNEL_SIZE - 1  # 1
    comptime conv_dim = 4
    comptime row_elements = conv_dim * state_len  # 4, a power of two
    comptime batch_size = 1
    comptime total_seq_len = 1
    comptime CONV1D_BLOCK_DIM = 128

    # slot_deep * row_elements == 2**32 exactly: one past UInt32.MAX.
    var slot_deep = 1 << 30
    var num_slots = slot_deep + 1
    var deep_offset = slot_deep * row_elements

    var front_sentinel = Scalar[state_dtype](111)
    var deep_sentinel = Scalar[state_dtype](222)
    var x_val = Float32(5)

    var qkv_input_heap = List(length=conv_dim * total_seq_len, fill=x_val)
    var qkv_input_h = TileTensor(
        qkv_input_heap, row_major(total_seq_len, conv_dim)
    )
    var conv_weight_heap = List(length=conv_dim * KERNEL_SIZE, fill=Float32(0))
    var conv_weight_h = TileTensor(
        conv_weight_heap, row_major(conv_dim, KERNEL_SIZE)
    )
    for c in range(conv_dim):
        conv_weight_h.raw_store(c * KERNEL_SIZE, Float32(1))
    var slot_idx_heap = List(length=batch_size, fill=UInt32(slot_deep))
    var slot_idx_h = TileTensor(slot_idx_heap, row_major(batch_size))
    var input_row_offsets_heap = List(length=batch_size + 1, fill=UInt32(0))
    var input_row_offsets_h = TileTensor(
        input_row_offsets_heap, row_major(batch_size + 1)
    )
    input_row_offsets_h.raw_store(1, UInt32(total_seq_len))

    var qkv_input_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    ctx.enqueue_copy(qkv_input_device, qkv_input_h._storage)
    var conv_weight_device = ctx.enqueue_create_buffer[work_dtype](
        conv_dim * KERNEL_SIZE
    )
    ctx.enqueue_copy(conv_weight_device, conv_weight_h._storage)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(slot_idx_device, slot_idx_h._storage)
    var input_row_offsets_device = ctx.enqueue_create_buffer[.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(input_row_offsets_device, input_row_offsets_h._storage)
    var conv_output_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )

    var qkv_input_tt = TileTensor(
        qkv_input_device, row_major(total_seq_len, conv_dim)
    )
    var conv_weight_tt = TileTensor(
        conv_weight_device, row_major(conv_dim, KERNEL_SIZE)
    )
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var input_row_offsets_tt = TileTensor(
        input_row_offsets_device, row_major(batch_size + 1)
    )
    var conv_output_tt = TileTensor(
        conv_output_device, row_major(total_seq_len, conv_dim)
    )

    var conv_state_device = ctx.enqueue_create_buffer[state_dtype](
        num_slots * row_elements
    )
    var conv_state_tt = TileTensor(
        conv_state_device, row_major(num_slots, conv_dim, state_len)
    )

    var front_sub = conv_state_device.create_sub_buffer[state_dtype](0, 1)
    var deep_sub = conv_state_device.create_sub_buffer[state_dtype](
        deep_offset, 1
    )
    var front_seed_heap = List(length=1, fill=front_sentinel)
    var front_seed_h = TileTensor(front_seed_heap, row_major(1))
    var deep_seed_heap = List(length=1, fill=deep_sentinel)
    var deep_seed_h = TileTensor(deep_seed_heap, row_major(1))
    ctx.enqueue_copy(front_sub, front_seed_h._storage)
    ctx.enqueue_copy(deep_sub, deep_seed_h._storage)
    ctx.synchronize()

    var compiled_func = ctx.compile_function[
        gated_delta_conv1d_fwd_gpu[
            work_dtype,
            state_dtype,
            KERNEL_SIZE,
            CONV1D_BLOCK_DIM,
            True,
            qkv_input_tt.LayoutType,
            conv_weight_tt.LayoutType,
            conv_state_tt.LayoutType,
            slot_idx_tt.LayoutType,
            input_row_offsets_tt.LayoutType,
            conv_output_tt.LayoutType,
            qkv_input_tt.Engine,
        ]
    ]()
    ctx.enqueue_function(
        compiled_func,
        Int32(batch_size),
        Int32(total_seq_len),
        Int32(conv_dim),
        Int32(1),
        qkv_input_tt,
        conv_weight_tt,
        conv_state_tt,
        slot_idx_tt,
        input_row_offsets_tt,
        conv_output_tt,
        grid_dim=(
            ceildiv(total_seq_len, CONV1D_TOKENS_PER_BLOCK),
            ceildiv(conv_dim, CONV1D_BLOCK_DIM),
        ),
        block_dim=(CONV1D_BLOCK_DIM,),
    )

    var output_readback_heap = List(
        length=total_seq_len * conv_dim, fill=Float32(0)
    )
    var output_h = TileTensor(
        output_readback_heap, row_major(total_seq_len, conv_dim)
    )
    var front_readback_heap = List(length=1, fill=Scalar[state_dtype](0))
    var front_readback_h = TileTensor(front_readback_heap, row_major(1))
    var deep_readback_heap = List(length=1, fill=Scalar[state_dtype](0))
    var deep_readback_h = TileTensor(deep_readback_heap, row_major(1))
    ctx.enqueue_copy(output_h._storage, conv_output_device)
    ctx.enqueue_copy(front_readback_h._storage, front_sub)
    ctx.enqueue_copy(deep_readback_h._storage, deep_sub)
    ctx.synchronize()

    # The deep sentinel, not the front row it would alias onto.
    assert_equal(output_h.raw_load(0), Scalar[work_dtype](deep_sentinel))

    # The front of the pool must be untouched.
    assert_equal(front_readback_h.raw_load(0), front_sentinel)

    # The deep slot itself must carry the write.
    assert_equal(deep_readback_h.raw_load(0), Scalar[state_dtype](x_val))


def test_slot_indexed_ragged_short_sequences() raises:
    """Ragged batch of 5 sequences with lengths 1..5 (the K-1 boundary cases);
    bf16 work and state; conv_dim not a multiple of the block size."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.bfloat16, DType.bfloat16, 4](
        batch_size=5,
        total_seq_len=15,
        conv_dim=264,
        max_slots=5,
        seq_lengths=Index(1, 2, 3, 4, 5),
        slot_assignments=Index(4, 2, 0, 3, 1),
        ctx=ctx,
    )


def test_slot_indexed_seq4_seq5_boundary() raises:
    """Lengths equal to K and K+1: every head-thread output plus the first
    fast-path output at position K-1."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.bfloat16, DType.float32, 4](
        batch_size=3,
        total_seq_len=13,
        conv_dim=130,
        max_slots=3,
        seq_lengths=Index(4, 5, 4),
        slot_assignments=Index(1, 2, 0),
        ctx=ctx,
        tokens_per_block=1,
    )


def test_slot_indexed_ragged_mixed_lengths() raises:
    """Ragged batch of 8 sequences mixing decode-length rows with a 2048-token
    prefill row, out-of-order slots."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=8,
        total_seq_len=2188,
        conv_dim=512,
        max_slots=8,
        seq_lengths=Index(64, 1, 2048, 3, 2, 64, 1, 5),
        slot_assignments=Index(7, 6, 5, 4, 3, 2, 1, 0),
        ctx=ctx,
    )


def test_slot_indexed_ragged_40_rows_binary_search() raises:
    """40 rows (> warp size, so the binary-search lookup) mixing zero-length,
    short and one long row."""
    comptime BATCH = 40
    var lengths = IndexList[BATCH]()
    var slots = IndexList[BATCH]()
    var total = 0
    for b in range(BATCH):
        var length = 1 + (b * 5) % 6
        if b % 9 == 0:
            length = 0
        if b == 17:
            length = 700
        lengths[b] = length
        slots[b] = (b * 7) % BATCH
        total += length
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=BATCH,
        total_seq_len=total,
        conv_dim=264,
        max_slots=BATCH,
        seq_lengths=lengths,
        slot_assignments=slots,
        ctx=ctx,
    )


def test_slot_indexed_production_prefill_2048() raises:
    """Single 2048-token prefill chunk at the production TP2 shard width
    (conv_dim = (key_dim*2 + value_dim)/2 = 5120 for Qwen3.8-27B)."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=1,
        total_seq_len=2048,
        conv_dim=5120,
        max_slots=1,
        seq_lengths=Index(2048),
        slot_assignments=Index(0),
        ctx=ctx,
        cpu_reference=False,
    )


def test_slot_indexed_production_prefill_16384() raises:
    """16384-token sequence with conv_dim 10200 (not a multiple of the block
    size); exercises the 32-bit flat-offset arithmetic at scale."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 4](
        batch_size=1,
        total_seq_len=16384,
        conv_dim=10200,
        max_slots=1,
        seq_lengths=Index(16384),
        slot_assignments=Index(0),
        ctx=ctx,
        cpu_reference=False,
    )


def test_slot_indexed_decode_20_rows() raises:
    """Decode shape: 20 rows x 1 token, every thread is a head thread."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.bfloat16, DType.float32, 4](
        batch_size=20,
        total_seq_len=20,
        conv_dim=5120,
        max_slots=20,
        seq_lengths=Index(
            1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
        ),
        slot_assignments=Index(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ),
        ctx=ctx,
        tokens_per_block=1,
    )


def test_slot_indexed_spec_decode_20_rows() raises:
    """Spec-decode shape: 20 rows x 4 tokens (== K) at the unsharded
    conv_dim 10240, a multiple of the block size."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 4](
        batch_size=20,
        total_seq_len=80,
        conv_dim=10240,
        max_slots=20,
        seq_lengths=Index(
            4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4
        ),
        slot_assignments=Index(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
        ),
        ctx=ctx,
        tokens_per_block=1,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
