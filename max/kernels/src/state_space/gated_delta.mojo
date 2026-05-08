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
"""Gated DeltaNet recurrence kernel for Qwen3.5 — Pass 2 of two-pass prefill.

Implements the gated delta rule recurrence over a ragged (variable-length)
batch of sequences.  This is Pass 2 of the prefill path; it consumes the
conv1d output produced by Pass 1 (gated_delta_conv1d_fwd).

The five steps of the gated delta rule at each token t for value-dim element
vd_i and value head h are:

  1. Apply per-head scalar decay to the entire state column:
       state_col[k]  ←  decay[t,h] * state_col[k]    for k in [0, KD)

  2. Compute kv_memory by taking the dot product of the decayed state column
     with the L2-normalised key vector (summing over the key_dim axis):
       kv_memory_vd_i  =  Σ_k  state_col[k] * key_normalised[t,h,k]

  3. Compute the delta correction using beta and the value residual:
       delta_correction_vd_i  =  beta[t,h] * (value[t,h,vd_i] - kv_memory_vd_i)

  4. Outer-product update of the state column with the key and delta:
       state_col[k]  ←  state_col[k]  +  key_normalised[t,h,k] * delta_correction_vd_i

  5. Read out the output by dotting the updated state with the scaled,
     L2-normalised query vector:
       output[t, h*VD + vd_i]  =  Σ_k  state_col[k] * query_scaled[t,h,k]

All of steps 1–5 run over the key_dim loop (k = 0..KD-1) which is a compile-
time constant.  This allows the inner loop to be fully unrolled and the KD-
element state column to live in GPU registers, eliminating shared-memory
traffic.

L2 normalisation and Q scaling are fused into the kernel body.  The raw Q/K/V
vectors are read directly from the conv1d output (qkv_conv_output), with the
channel layout: Q at [0..key_dim), K at [key_dim..2*key_dim), V at [2*key_dim..).

GQA (grouped query attention) is handled by computing the key head index as:
  key_head_idx = value_head_idx // heads_expansion_ratio

where heads_expansion_ratio = num_value_heads / num_key_heads is a runtime
integer, so no compile-time specialisation per model is required.

Tensor shapes
-------------
Inputs:
  qkv_conv_output    : [total_seq_len, conv_dim]              float32
      Conv1d output from Pass 1.  Channel layout:
        Q: channels [0, key_dim)
        K: channels [key_dim, 2*key_dim)
        V: channels [2*key_dim, 2*key_dim + value_dim)
      where key_dim  = num_key_heads  * key_head_dim
            value_dim = num_value_heads * value_head_dim
            conv_dim  = key_dim * 2 + value_dim
  decay_per_token    : [total_seq_len, num_value_heads]        float32
      Per-token, per-head scalar decay factor (exp(-softplus) pre-applied).
  beta_per_token     : [total_seq_len, num_value_heads]        float32
      Per-token, per-head beta gate (sigmoid pre-applied).
  recurrent_state    : [max_slots, num_value_heads, key_head_dim, value_head_dim]
      Mutable recurrent-state pool.  The kernel reads/writes slot
      `slot_idx[batch_item]` in place; all other slots are untouched.
      Pool dtype is independent of the working dtype, so the caller can
      keep per-token tensors at float32 while storing the pool at the
      model's native dtype (typically bfloat16).
  slot_idx           : [batch_size]                            uint32
      Pool slot index for each batch item.
  input_row_offsets  : [batch_size + 1]                        uint32
      Ragged offsets: sequence b spans flat indices
      [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  recurrence_output  : [total_seq_len, value_dim]              float32
      Flat output for all tokens.  Indexed as
      output[flat_t, value_head_idx * value_head_dim + vd_element_idx].
  (recurrent_state is mutated in place; there is no separate state-out
   tensor.)

Thread mapping (GPU)
--------------------
  total_threads = batch_size * num_value_heads * value_head_dim
  Grid  : ceildiv(total_threads, RECURRENCE_BLOCK_SIZE) blocks of size 1-D
  Block : RECURRENCE_BLOCK_SIZE threads

  Thread decomposition:
    flat_thread_idx       = block_idx * RECURRENCE_BLOCK_SIZE + thread_idx
    batch_item_idx        = flat_thread_idx // (num_value_heads * value_head_dim)
    value_head_idx        = (flat_thread_idx % (num_value_heads * value_head_dim))
                            // value_head_dim
    vd_element_idx        = flat_thread_idx % value_head_dim
    key_head_idx          = value_head_idx // heads_expansion_ratio

  Each thread owns the KD-element column
    state_col[0..KD-1] =
      recurrent_state[slot_idx[batch_item], value_head, 0..KD-1, vd_element]
  in registers and iterates over its sequence sequentially.
"""

import std.math
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


def gated_delta_recurrence_fwd_gpu[
    work_dtype: DType,  # for qkv/decay/beta/recurrence_output (typically fp32)
    state_dtype: DType,  # for the recurrent_state pool (typically bf16)
    KEY_HEAD_DIM: Int,  # key_head_dim, compile-time (e.g. 128 for Qwen3.5)
    VALUE_HEAD_DIM: Int,  # value_head_dim, compile-time (e.g. 128 for Qwen3.5)
    RECURRENCE_BLOCK_SIZE: Int,
    recurrence_output_LT: TensorLayout,
    qkv_conv_output_LT: TensorLayout,
    decay_per_token_LT: TensorLayout,
    beta_per_token_LT: TensorLayout,
    recurrent_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
](
    total_threads: Int,  # batch_size * num_value_heads * value_head_dim
    batch_size: Int,
    total_seq_len: Int,
    num_value_heads: Int,  # nv
    num_key_heads: Int,  # nk; heads_expansion_ratio = nv / nk
    key_dim: Int,  # num_key_heads * key_head_dim
    value_dim: Int,  # num_value_heads * value_head_dim
    conv_dim: Int,  # key_dim * 2 + value_dim
    recurrence_output: TileTensor[
        work_dtype, recurrence_output_LT, MutExternalOrigin
    ],
    recurrent_state: TileTensor[
        state_dtype, recurrent_state_LT, MutExternalOrigin
    ],
    slot_idx: TileTensor[DType.uint32, slot_idx_LT, MutExternalOrigin],
    qkv_conv_output: TileTensor[
        work_dtype, qkv_conv_output_LT, MutExternalOrigin
    ],
    decay_per_token: TileTensor[
        work_dtype, decay_per_token_LT, MutExternalOrigin
    ],
    beta_per_token: TileTensor[
        work_dtype, beta_per_token_LT, MutExternalOrigin
    ],
    input_row_offsets: TileTensor[
        DType.uint32, input_row_offsets_LT, MutExternalOrigin
    ],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_conv_output_seqlen_stride: UInt32,
    qkv_conv_output_channel_stride: UInt32,
    # Strides for [total_seq_len, num_value_heads] tensors (decay, beta)
    per_token_seqlen_stride: UInt32,
    per_token_head_stride: UInt32,
    # Strides for [max_slots, nv, KD, VD] recurrent state pool.
    # `recurrent_state_slot_stride` is the stride between adjacent slots —
    # same numeric meaning as the previous `recurrent_state_batch_stride`
    # for the old [B, nv, KD, VD] tensor.
    recurrent_state_slot_stride: UInt32,
    recurrent_state_value_head_stride: UInt32,
    recurrent_state_key_dim_stride: UInt32,
    recurrent_state_value_dim_stride: UInt32,
    # Strides for [total_seq_len, value_dim] recurrence output
    recurrence_output_seqlen_stride: UInt32,
    recurrence_output_valuedim_stride: UInt32,
):
    """GPU kernel: slot-indexed gated delta rule recurrence.

    The recurrent state lives in a single mutable pool of shape
    ``[max_slots, nv, KD, VD]``; this kernel reads/writes pool slot
    ``slot_idx[batch_item_idx]`` for batch item ``batch_item_idx`` and avoids
    the gather/scatter copies the host-side state cache used to do.
    One thread per (batch_item, value_head, vd_element) triple; the
    KD-element state column lives entirely in registers.
    """
    # Cast to Int before multiplication to avoid UInt32 overflow
    var flat_thread_idx = Int(block_dim.x) * Int(block_idx.x) + Int(
        thread_idx.x
    )
    if flat_thread_idx >= total_threads:
        return

    # ── Decompose thread index into (batch, value_head, vd_element) ─────────
    var threads_per_batch_item = num_value_heads * VALUE_HEAD_DIM
    var batch_item_idx = flat_thread_idx // threads_per_batch_item
    var within_batch_flat_idx = flat_thread_idx % threads_per_batch_item
    var value_head_idx = within_batch_flat_idx // VALUE_HEAD_DIM
    var vd_element_idx = within_batch_flat_idx % VALUE_HEAD_DIM

    if batch_item_idx >= batch_size:
        return

    # Read the pool slot for this batch item exactly once. The caller
    # (`GatedDeltaNetStateCache.claim`) guarantees `slot < max_slots`.
    var slot = Int(slot_idx.ptr[batch_item_idx])

    # GQA: map value head to key head
    var heads_expansion_ratio = num_value_heads // num_key_heads
    var key_head_idx = value_head_idx // heads_expansion_ratio

    # ── Load initial state column from pool[slot, ...] ──────────────────────
    # state_col[k] = recurrent_state[slot, value_head, k, vd_element]
    var state_col = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
    comptime for kd_element_k in range(KEY_HEAD_DIM):
        var state_flat_offset = (
            UInt32(slot) * recurrent_state_slot_stride
            + UInt32(value_head_idx) * recurrent_state_value_head_stride
            + UInt32(kd_element_k) * recurrent_state_key_dim_stride
            + UInt32(vd_element_idx) * recurrent_state_value_dim_stride
        )
        state_col[kd_element_k] = Scalar[DType.float32](
            recurrent_state.ptr[state_flat_offset]
        )

    # ── Sequence boundaries from ragged offsets ──────────────────────────────
    var sequence_start_flat_idx = Int(input_row_offsets.ptr[batch_item_idx])
    var sequence_end_flat_idx = Int(input_row_offsets.ptr[batch_item_idx + 1])
    var sequence_length = sequence_end_flat_idx - sequence_start_flat_idx

    # Precompute constant channel offsets for Q, K, V in the conv_dim layout.
    # Q occupies channels [0, key_dim).
    # K occupies channels [key_dim, 2 * key_dim).
    # V occupies channels [2 * key_dim, 2 * key_dim + value_dim).
    var query_channel_base = UInt32(key_head_idx * KEY_HEAD_DIM)
    var key_channel_base = UInt32(key_dim + key_head_idx * KEY_HEAD_DIM)
    var value_channel_offset = UInt32(
        2 * key_dim + value_head_idx * VALUE_HEAD_DIM + vd_element_idx
    )

    var query_scale = Float32(1.0) / Float32(
        std.math.sqrt(Float32(KEY_HEAD_DIM))
    )

    # ── Iterate over sequence tokens ─────────────────────────────────────────
    for token_position_in_sequence in range(sequence_length):
        var flat_token_idx = (
            sequence_start_flat_idx + token_position_in_sequence
        )

        # ── Load Q and K vectors for this key head (KD elements each) ─────
        # Simultaneously accumulate L2 norms for normalisation.
        var query_raw = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
        var key_raw = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
        var query_squared_sum = Float32(0.0)
        var key_squared_sum = Float32(0.0)

        var token_qkv_row_offset = (
            UInt32(flat_token_idx) * qkv_conv_output_seqlen_stride
        )
        comptime for kd_element_k in range(KEY_HEAD_DIM):
            var query_flat_offset = (
                token_qkv_row_offset
                + (query_channel_base + UInt32(kd_element_k))
                * qkv_conv_output_channel_stride
            )
            var key_flat_offset = (
                token_qkv_row_offset
                + (key_channel_base + UInt32(kd_element_k))
                * qkv_conv_output_channel_stride
            )
            var q_raw_k = Scalar[DType.float32](
                qkv_conv_output.ptr[query_flat_offset]
            )
            var k_raw_k = Scalar[DType.float32](
                qkv_conv_output.ptr[key_flat_offset]
            )
            query_raw[kd_element_k] = q_raw_k
            key_raw[kd_element_k] = k_raw_k
            query_squared_sum = query_squared_sum + q_raw_k * q_raw_k
            key_squared_sum = key_squared_sum + k_raw_k * k_raw_k

        # ── L2 normalise Q and K, scale Q by 1/sqrt(KD) ───────────────────
        var query_inv_norm = Float32(1.0) / std.math.sqrt(
            query_squared_sum + Float32(1e-6)
        )
        var key_inv_norm = Float32(1.0) / std.math.sqrt(
            key_squared_sum + Float32(1e-6)
        )

        var query_normalised_scaled = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
        var key_normalised = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
        comptime for kd_element_k in range(KEY_HEAD_DIM):
            query_normalised_scaled[kd_element_k] = (
                query_raw[kd_element_k] * query_inv_norm * query_scale
            )
            key_normalised[kd_element_k] = key_raw[kd_element_k] * key_inv_norm

        # ── Load V element (only the vd_element_idx column) ───────────────
        var value_flat_offset = (
            token_qkv_row_offset
            + value_channel_offset * qkv_conv_output_channel_stride
        )
        var value_element = Scalar[DType.float32](
            qkv_conv_output.ptr[value_flat_offset]
        )

        # ── Load per-token decay and beta for this value head ──────────────
        var head_token_offset = (
            UInt32(flat_token_idx) * per_token_seqlen_stride
            + UInt32(value_head_idx) * per_token_head_stride
        )
        var decay_value = Scalar[DType.float32](
            decay_per_token.ptr[head_token_offset]
        )
        var beta_value = Scalar[DType.float32](
            beta_per_token.ptr[head_token_offset]
        )

        # ── Step 1: Apply decay; Step 2: Compute kv_memory ────────────────
        # Both require iterating over kd_element_k, so they are fused.
        # Note: Step 2's dot product uses state_col[kd_element_k] after in-place
        # decay (Step 1) — this is intentional per the recurrence definition.
        var kv_memory_value = Float32(0.0)
        comptime for kd_element_k in range(KEY_HEAD_DIM):
            state_col[kd_element_k] = state_col[kd_element_k] * decay_value
            kv_memory_value = (
                kv_memory_value
                + state_col[kd_element_k] * key_normalised[kd_element_k]
            )

        # ── Step 3: Compute delta correction ──────────────────────────────
        var delta_correction_vd_i = beta_value * (
            value_element - kv_memory_value
        )

        # ── Step 4: Outer-product state update; Step 5: Query readout ─────
        # Both iterate over kd_element_k, so they are fused.
        var output_value = Float32(0.0)
        comptime for kd_element_k in range(KEY_HEAD_DIM):
            state_col[kd_element_k] = (
                state_col[kd_element_k]
                + key_normalised[kd_element_k] * delta_correction_vd_i
            )
            output_value = (
                output_value
                + state_col[kd_element_k]
                * query_normalised_scaled[kd_element_k]
            )

        # ── Write output for this token ────────────────────────────────────
        # Output layout: [total_seq_len, value_dim] where the value_dim index
        # is value_head_idx * VALUE_HEAD_DIM + vd_element_idx.
        var recurrence_output_flat_offset = (
            UInt32(flat_token_idx) * recurrence_output_seqlen_stride
            + UInt32(value_head_idx * VALUE_HEAD_DIM + vd_element_idx)
            * recurrence_output_valuedim_stride
        )
        recurrence_output.ptr[recurrence_output_flat_offset] = Scalar[
            work_dtype
        ](output_value)

    # ── Write final state column back into pool[slot, ...] ──────────────────
    comptime for kd_element_k in range(KEY_HEAD_DIM):
        var state_flat_offset = (
            UInt32(slot) * recurrent_state_slot_stride
            + UInt32(value_head_idx) * recurrent_state_value_head_stride
            + UInt32(kd_element_k) * recurrent_state_key_dim_stride
            + UInt32(vd_element_idx) * recurrent_state_value_dim_stride
        )
        recurrent_state.ptr[state_flat_offset] = Scalar[state_dtype](
            state_col[kd_element_k]
        )
