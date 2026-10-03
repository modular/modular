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
"""Causal depthwise conv1d for the Gated DeltaNet two-pass prefill.

This is Pass 1 of the two-pass gated delta rule prefill path.  It computes
the causal 1-D convolution over a ragged (variable-length) batch of sequences
and updates the per-sequence sliding-window conv state.

Unlike the existing causal_conv1d_varlen_fwd (which uses [dim, total_seqlen]
layout for Mamba compatibility), this kernel uses [total_seqlen, conv_dim]
layout to match the gated_deltanet.py convention where all per-token tensors
are seqlen-first.

Tensor shapes
-------------
Inputs:
  qkv_input_ragged   : [total_seq_len, conv_dim]              float32
      Flat projected QKV input, all sequences concatenated.
  conv_weight        : [conv_dim, kernel_size]                float32
      Depthwise conv weights (one weight per channel per time offset).
  conv_state         : [max_slots, conv_dim, kernel_size-1]
      Mutable sliding-window conv state pool.  The kernel reads/writes
      slot `slot_idx[batch_item]` in place; all other slots are
      untouched.  Slots within a single pool entry are ordered
      oldest-to-newest: window slot 0 is the token at position -(K-1)
      relative to the current sequence start.  Pool dtype is independent
      of the working dtype.
  slot_idx           : [batch_size]                           uint32
      Pool slot index for each batch item.
  input_row_offsets  : [batch_size + 1]                       uint32
      Exclusive prefix sums of sequence lengths.  Sequence b spans
      token indices [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  conv_output_ragged : [total_seq_len, conv_dim]              float32
      Causal conv output in the same ragged layout as the input.
  (conv_state is mutated in place; there is no separate state-out
   tensor.  Window slot j ends up holding the raw input at position
   seq_len - (kernel_size-1) + j within the sequence, carrying forward
   from the old window when seq_len is shorter.  Under
   WRITE_STATE = False the window is left alone and the only result is
   the conv output.)

Thread mapping (GPU)
--------------------
  Grid  : (ceildiv(total_seq_len, tokens_per_block),
          ceildiv(conv_dim, CONV1D_BLOCK_DIM))
  Block : (CONV1D_BLOCK_DIM,)
  One thread per (token tile, conv_channel).  The conv is parallel over
  tokens, so long prefill rows spread across the whole GPU.
"""

from std.bit import count_leading_zeros
from std.sys.info import is_nvidia_gpu

import max.gpu.primitives.warp as warp
from max.gpu import (
    WARP_SIZE,
    block_dim,
    block_idx,
    lane_id,
    thread_idx,
)
from layout import TensorEngine, TensorLayout, TileTensor
from std.utils.index import IndexList


# Upper bound on the runtime `tokens_per_block`; the host shrinks it to the
# average row length so short rows do not serialize on one thread.
comptime CONV1D_TOKENS_PER_BLOCK: Int = 8


# ===----------------------------------------------------------------------=== #
# GPU Kernel
# ===----------------------------------------------------------------------=== #


def gated_delta_conv1d_fwd_gpu[
    work_dtype: DType,  # for qkv_input_ragged / conv_weight / conv_output_ragged
    state_dtype: DType,  # for the conv_state pool (typically bf16)
    KERNEL_SIZE: Int,
    CONV1D_BLOCK_DIM: Int,
    WRITE_STATE: Bool,
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
    tokens_per_block: Int32,
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
):
    """Slot-indexed causal depthwise conv1d over a ragged batch.

    The thread owning a sequence's first token computes the outputs that
    see the old conv window and then rewrites the window; later tokens read
    only the ragged input. With `WRITE_STATE` False the window is left
    unchanged.
    """
    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    var _total_seq_len = Int(total_seq_len)
    var _batch_size = Int(batch_size)
    var _tokens_per_block = Int(tokens_per_block)
    var conv_channel_idx = block_idx.y * CONV1D_BLOCK_DIM + thread_idx.x

    # Out-of-range threads must stay converged for the warp-wide batch lookup,
    # so their reads are clamped and only their stores are suppressed.
    var active = conv_channel_idx < Int(conv_dim)
    var channel_read_idx = min(conv_channel_idx, Int(conv_dim) - 1)

    var weight_register = SIMD[work_dtype, KERNEL_SIZE](0)
    comptime for kernel_offset_k in range(KERNEL_SIZE):
        weight_register[kernel_offset_k] = conv_weight.load[width=1](
            (channel_read_idx, kernel_offset_k)
        )

    # Cached across the consecutive tokens this thread owns.
    var batch_item_idx = 0
    var sequence_start_flat_idx = 0
    var sequence_end_flat_idx = 0

    var first_flat_token_idx = block_idx.x * _tokens_per_block
    for flat_token_idx in range(
        first_flat_token_idx,
        min(first_flat_token_idx + _tokens_per_block, _total_seq_len),
    ):
        if flat_token_idx >= sequence_end_flat_idx:
            # Rightmost b with input_row_offsets[b] <= flat_token_idx, which
            # skips zero-length sequences.
            if _batch_size == 1:
                batch_item_idx = 0
            else:
                var looked_up = False
                # The 32-bit ballot does not lower on 64-lane AMD wavefronts.
                comptime if is_nvidia_gpu():
                    comptime assert WARP_SIZE == 32, "ballot is 32 bits wide"
                    if _batch_size <= WARP_SIZE:
                        # One parallel load instead of a dependent-load chain.
                        var lane = lane_id()
                        var found = False
                        if lane < _batch_size:
                            found = (
                                Int(input_row_offsets.load[width=1]((lane,)))
                                <= flat_token_idx
                            )
                        var ballot = warp.vote[.uint32](found)
                        var warp_lookup = (
                            WARP_SIZE - 1 - Int(count_leading_zeros(ballot))
                        )
                        if warp_lookup < 0:
                            warp_lookup = 0
                        batch_item_idx = warp_lookup
                        looked_up = True
                if not looked_up:
                    var lo = 0
                    var hi = _batch_size - 1
                    while lo < hi:
                        var mid = (lo + hi + 1) >> 1
                        if (
                            Int(input_row_offsets.load[width=1]((mid,)))
                            <= flat_token_idx
                        ):
                            lo = mid
                        else:
                            hi = mid - 1
                    batch_item_idx = lo
            sequence_start_flat_idx = Int(
                input_row_offsets.load[width=1]((batch_item_idx,))
            )
            sequence_end_flat_idx = Int(
                input_row_offsets.load[width=1]((batch_item_idx + 1,))
            )

        var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx
        var token_position_in_sequence = (
            flat_token_idx - sequence_start_flat_idx
        )

        if token_position_in_sequence >= KERNEL_SIZE_MINUS_ONE:
            # All K taps come from the ragged input; conv_state is untouched.
            var conv_sum = Float32(0.0)
            comptime for kernel_offset_k in range(KERNEL_SIZE):
                var lookback_position = token_position_in_sequence - (
                    KERNEL_SIZE_MINUS_ONE - kernel_offset_k
                )
                conv_sum += Float32(
                    qkv_input_ragged.load[width=1](
                        (
                            sequence_start_flat_idx + lookback_position,
                            channel_read_idx,
                        )
                    )
                ) * Float32(weight_register[kernel_offset_k])

            if active:
                conv_output_ragged.store(
                    (flat_token_idx, conv_channel_idx),
                    Scalar[work_dtype](conv_sum),
                )
            continue

        if token_position_in_sequence >= 1:
            # The head thread already computed this output.
            continue

        var slot = Int(slot_idx.load[width=1]((batch_item_idx,)))
        debug_assert(
            0 <= slot < Int(conv_state.dim[0]()), "conv state slot out of range"
        )

        for head_position in range(min(KERNEL_SIZE_MINUS_ONE, sequence_length)):
            var conv_sum = Float32(0.0)

            comptime for kernel_offset_k in range(KERNEL_SIZE):
                var lookback_position = head_position - (
                    KERNEL_SIZE_MINUS_ONE - kernel_offset_k
                )

                # Cast on read so the work-dtype qkv input and the state-dtype
                # pool produce the same Float32 conv_sum accumulator.
                var input_value: Float32 = 0

                if lookback_position >= 0:
                    input_value = Float32(
                        qkv_input_ragged.load[width=1](
                            (
                                sequence_start_flat_idx + lookback_position,
                                channel_read_idx,
                            )
                        )
                    )
                else:
                    # Window index 0 is the oldest entry.
                    var window_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                    if window_idx >= 0:
                        input_value = Float32(
                            conv_state.load[width=1](
                                (slot, channel_read_idx, window_idx)
                            )
                        )

                conv_sum += input_value * Float32(
                    weight_register[kernel_offset_k]
                )

            if active:
                conv_output_ragged.store(
                    (sequence_start_flat_idx + head_position, conv_channel_idx),
                    Scalar[work_dtype](conv_sum),
                )

        # Safe in place: every old-window read precedes the write to its slot.
        # Never return here; this thread may own more tokens in its tile.
        comptime if WRITE_STATE:
            comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
                var source_position_in_sequence = (
                    sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
                )

                var state_value: Scalar[state_dtype] = 0
                if source_position_in_sequence >= 0:
                    state_value = Scalar[state_dtype](
                        qkv_input_ragged.load[width=1](
                            (
                                sequence_start_flat_idx
                                + source_position_in_sequence,
                                channel_read_idx,
                            )
                        )
                    )
                else:
                    var old_window_idx = (
                        KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
                    )
                    if old_window_idx >= 0:
                        state_value = conv_state.load[width=1](
                            (slot, channel_read_idx, old_window_idx)
                        )

                if active:
                    conv_state.store(
                        (slot, conv_channel_idx, state_slot_j),
                        state_value,
                    )
