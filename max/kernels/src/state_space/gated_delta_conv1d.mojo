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
  qkv_input_ragged   : [total_seq_len, conv_dim]          float32
      Flat projected QKV input, all sequences concatenated.
  conv_weight        : [conv_dim, kernel_size]             float32
      Depthwise conv weights (one weight per channel per time offset).
  conv_state_in      : [batch_size, conv_dim, kernel_size-1] float32
      Sliding-window conv state from the previous segment.  Slots are
      ordered oldest-to-newest: slot 0 is the token at position -(K-1)
      relative to the current sequence start.
  input_row_offsets  : [batch_size + 1]                   uint32
      Exclusive prefix sums of sequence lengths.  Sequence b spans
      token indices [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  conv_output_ragged : [total_seq_len, conv_dim]          float32
      Causal conv output in the same ragged layout as the input.
  conv_state_out     : [batch_size, conv_dim, kernel_size-1] float32
      Updated sliding-window state.  Slot j holds the raw (pre-activation)
      token at position seq_len - (kernel_size-1) + j within each sequence.

Thread mapping (GPU)
--------------------
  Grid  : (batch_size, ceildiv(conv_dim, CONV1D_BLOCK_DIM))
  Block : (CONV1D_BLOCK_DIM,)
  One thread per (batch_item, conv_channel).  Each thread processes its
  channel's full sequence sequentially, reading from conv_state_in for
  the look-back that extends before the current sequence.
"""

from std.gpu import (
    block_dim_uint as block_dim,
    block_idx_uint as block_idx,
    thread_idx_uint as thread_idx,
)
from layout import TensorLayout, TileTensor
from std.utils.index import IndexList


# ===----------------------------------------------------------------------=== #
# GPU Kernel
# ===----------------------------------------------------------------------=== #


def gated_delta_conv1d_fwd_gpu[
    dtype: DType,
    KERNEL_SIZE: Int,  # conv kernel width, compile-time (e.g. 4 for Qwen3.5)
    CONV1D_BLOCK_DIM: Int,  # threads per block along conv_dim axis
    qkv_input_ragged_LT: TensorLayout,
    conv_weight_LT: TensorLayout,
    conv_state_in_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
    conv_output_ragged_LT: TensorLayout,
    conv_state_out_LT: TensorLayout,
](
    batch_size: Int,
    total_seq_len: Int,
    conv_dim: Int,
    qkv_input_ragged: TileTensor[dtype, qkv_input_ragged_LT, MutExternalOrigin],
    conv_weight: TileTensor[dtype, conv_weight_LT, MutExternalOrigin],
    conv_state_in: TileTensor[dtype, conv_state_in_LT, MutExternalOrigin],
    input_row_offsets: TileTensor[
        DType.uint32, input_row_offsets_LT, MutExternalOrigin
    ],
    conv_output_ragged: TileTensor[
        dtype, conv_output_ragged_LT, MutExternalOrigin
    ],
    conv_state_out: TileTensor[dtype, conv_state_out_LT, MutExternalOrigin],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_input_seqlen_stride: UInt32,  # stride along total_seq_len axis
    qkv_input_channel_stride: UInt32,  # stride along conv_dim axis (usually 1)
    conv_weight_channel_stride: UInt32,  # stride along conv_dim axis
    conv_weight_offset_stride: UInt32,  # stride along kernel_size axis
    # Strides for [batch_size, conv_dim, kernel_size-1] tensors
    conv_state_batch_stride: UInt32,
    conv_state_channel_stride: UInt32,
    conv_state_slot_stride: UInt32,
    # Output strides (match input strides for conv_output_ragged)
    conv_output_seqlen_stride: UInt32,
    conv_output_channel_stride: UInt32,
):
    """GPU kernel: causal depthwise conv1d over ragged batch, seqlen-first layout.

    One thread handles one (batch_item, conv_channel) pair for the entire
    sequence length.  The weights for that channel are kept in registers
    to avoid repeated global-memory reads across token loop iterations.
    """
    var batch_item_idx = Int(block_idx.x)
    var channel_block_idx = Int(block_idx.y)
    var thread_within_block = Int(thread_idx.x)

    var conv_channel_idx = (
        channel_block_idx * CONV1D_BLOCK_DIM + thread_within_block
    )

    if batch_item_idx >= batch_size or conv_channel_idx >= conv_dim:
        return

    # ── Sequence boundaries from ragged offsets ─────────────────────────────
    var sequence_start_flat_idx = Int(input_row_offsets.ptr[batch_item_idx])
    var sequence_end_flat_idx = Int(input_row_offsets.ptr[batch_item_idx + 1])
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    # ── Load conv weights for this channel into registers ───────────────────
    # Avoids re-reading the same KERNEL_SIZE weights on every token step.
    var weight_register = SIMD[dtype, KERNEL_SIZE](0)
    comptime for kernel_offset_k in range(KERNEL_SIZE):
        var weight_flat_offset = (
            UInt32(conv_channel_idx) * conv_weight_channel_stride
            + UInt32(kernel_offset_k) * conv_weight_offset_stride
        )
        weight_register[kernel_offset_k] = conv_weight.ptr[weight_flat_offset]

    comptime KERNEL_SIZE_MINUS_ONE = KERNEL_SIZE - 1

    # ── Process each token in this sequence ─────────────────────────────────
    for token_position_in_sequence in range(sequence_length):
        var flat_token_idx = (
            sequence_start_flat_idx + token_position_in_sequence
        )
        var conv_sum = Float32(0.0)

        # Convolve: sum over K offsets, looking back K-1 tokens.
        # For kernel offset k, the input token is at relative position:
        #   lookback_position = token_position_in_sequence - (KERNEL_SIZE - 1 - k)
        # A negative lookback_position falls before the current sequence and
        # is read from conv_state_in.  Non-negative positions come from the
        # ragged input buffer.
        comptime for kernel_offset_k in range(KERNEL_SIZE):
            var lookback_position = token_position_in_sequence - (
                KERNEL_SIZE_MINUS_ONE - kernel_offset_k
            )

            var input_value: Scalar[dtype] = 0

            if lookback_position >= 0:
                # Within the current sequence: read from ragged input buffer
                var ragged_flat_offset = (
                    UInt32(sequence_start_flat_idx + lookback_position)
                    * qkv_input_seqlen_stride
                    + UInt32(conv_channel_idx) * qkv_input_channel_stride
                )
                input_value = qkv_input_ragged.ptr[ragged_flat_offset]
            else:
                # Before the current sequence: read from conv_state_in.
                # Slot index maps the negative lookback to the state array:
                #   state_slot = KERNEL_SIZE_MINUS_ONE + lookback_position
                # When lookback_position = -(KERNEL_SIZE-1) this is slot 0 (oldest).
                # When lookback_position = -1 this is slot KERNEL_SIZE-2 (newest).
                var state_slot_idx = KERNEL_SIZE_MINUS_ONE + lookback_position
                if state_slot_idx >= 0:
                    var state_flat_offset = (
                        UInt32(batch_item_idx) * conv_state_batch_stride
                        + UInt32(conv_channel_idx) * conv_state_channel_stride
                        + UInt32(state_slot_idx) * conv_state_slot_stride
                    )
                    input_value = conv_state_in.ptr[state_flat_offset]

            conv_sum += Float32(input_value) * Float32(
                weight_register[kernel_offset_k]
            )

        var output_flat_offset = (
            UInt32(flat_token_idx) * conv_output_seqlen_stride
            + UInt32(conv_channel_idx) * conv_output_channel_stride
        )
        conv_output_ragged.ptr[output_flat_offset] = Scalar[dtype](conv_sum)

    # ── Update conv_state_out: the last KERNEL_SIZE-1 raw input tokens ──────
    # slot j should hold the raw input at position seq_len - (KERNEL_SIZE-1) + j.
    # If that position is negative (sequence shorter than KERNEL_SIZE-1), the
    # slot carries forward from conv_state_in at position KERNEL_SIZE_MINUS_ONE
    # + (seq_len - KERNEL_SIZE_MINUS_ONE + j) = seq_len + j.
    comptime for state_slot_j in range(KERNEL_SIZE_MINUS_ONE):
        var source_position_in_sequence = (
            sequence_length - KERNEL_SIZE_MINUS_ONE + state_slot_j
        )

        var state_value: Scalar[dtype] = 0
        if source_position_in_sequence >= 0:
            # Source token is within the current sequence
            var source_flat_offset = (
                UInt32(sequence_start_flat_idx + source_position_in_sequence)
                * qkv_input_seqlen_stride
                + UInt32(conv_channel_idx) * qkv_input_channel_stride
            )
            state_value = qkv_input_ragged.ptr[source_flat_offset]
        else:
            # Source token is before the current sequence; carry from old state.
            # The old state slot is KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            # (which is still negative-offset from the old window end).
            var old_state_slot_idx = (
                KERNEL_SIZE_MINUS_ONE + source_position_in_sequence
            )
            if old_state_slot_idx >= 0:
                var old_state_flat_offset = (
                    UInt32(batch_item_idx) * conv_state_batch_stride
                    + UInt32(conv_channel_idx) * conv_state_channel_stride
                    + UInt32(old_state_slot_idx) * conv_state_slot_stride
                )
                state_value = conv_state_in.ptr[old_state_flat_offset]

        var new_state_flat_offset = (
            UInt32(batch_item_idx) * conv_state_batch_stride
            + UInt32(conv_channel_idx) * conv_state_channel_stride
            + UInt32(state_slot_j) * conv_state_slot_stride
        )
        conv_state_out.ptr[new_state_flat_offset] = state_value
