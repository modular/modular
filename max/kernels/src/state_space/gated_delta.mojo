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
"""Gated DeltaNet recurrence kernel for Qwen3.5: Pass 2 of two-pass prefill.

Implements the gated delta rule recurrence over a ragged (variable-length)
batch of sequences. This is Pass 2 of the prefill path; it consumes the
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

Thread mapping (GPU)
--------------------
One CTA owns one (batch_item, value_head); the block has VALUE_HEAD_DIM
threads. Thread `tid == vd_element` owns the KD-element state column

    state_col[k] = recurrent_state[slot_idx[batch_item], value_head, k, tid]

in registers as a `SIMD[.float32, KEY_HEAD_DIM]` and iterates over its
sequence sequentially. KEY_HEAD_DIM is a compile-time constant, so the state
column stays in registers across the whole sequence.

  Grid  : (batch_size * num_value_heads,) 1-D
  Block : (VALUE_HEAD_DIM,) 1-D

The per-token raw Q and K vectors for this value head's key head are loaded
once per block into shared memory (one element per thread, coalesced). Each
L2 norm is a warp shuffle plus a cross-warp combine through shared memory.
L2 normalisation and the 1/sqrt(KD) query scale are folded in as scalars
factored out of the KD reductions, so no normalised Q/K array is
materialised.

GQA (grouped query attention) is handled by computing the key head index as:
  key_head_idx = value_head_idx // heads_expansion_ratio

where heads_expansion_ratio = num_value_heads / num_key_heads is a runtime
integer, so no compile-time specialisation per model is required.

Tensor shapes
-------------
Inputs:
  qkv_conv_output    : [total_seq_len, conv_dim]              float32
      Conv1d output from Pass 1. Channel layout:
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
      Mutable recurrent-state pool. The kernel reads/writes slot
      `slot_idx[batch_item]` in place; all other slots are untouched.
      Pool dtype is independent of the working dtype, so the caller can
      keep per-token tensors at float32 while storing the pool at the
      model's native dtype (bfloat16).
  slot_idx           : [batch_size]                            uint32
      Pool slot index for each batch item.
  input_row_offsets  : [batch_size + 1]                        uint32
      Ragged offsets: sequence b spans flat indices
      [input_row_offsets[b], input_row_offsets[b+1]).

Outputs:
  recurrence_output  : [total_seq_len, value_dim]              float32
      Flat output for all tokens. Indexed as
      output[flat_t, value_head_idx * value_head_dim + vd_element_idx].
  (recurrent_state is mutated in place; there is no separate state-out
   tensor.)
"""

import std.math
from max.gpu import (
    WARP_SIZE,
    block_idx,
    lane_id,
    thread_idx,
)
from max.gpu.primitives import warp
from max.gpu.sync import barrier
from std.math import fma, rsqrt
from std.memory import unsafe_stack_allocation
from layout import Coord, TensorEngine, TensorLayout, TileTensor


comptime _SharedF32 = Pointer[
    Float32, MutUntrackedOrigin, address_space=.SHARED
]


@always_inline
def _gated_delta_token_step[
    KEY_HEAD_DIM: Int
](
    mut state_col: SIMD[.float32, KEY_HEAD_DIM],
    q_raw_s: _SharedF32,
    k_raw_s: _SharedF32,
    q_warp_sumsq_s: _SharedF32,
    k_warp_sumsq_s: _SharedF32,
    q_value: Float32,
    k_value: Float32,
    value_element: Float32,
    decay_value: Float32,
    beta_value: Float32,
    mut key_update_factor: Float32,
) -> Float32:
    """Advances one thread's state column over one token and returns its readout.

    Every thread of the block calls this once per token with its own element of
    the raw Q and K rows. Both recurrence kernels advance their state through
    this one function, which is what keeps them bit-identical.

    Parameters:
        KEY_HEAD_DIM: Compile-time key head dimension.

    Args:
        state_col: This thread's `KEY_HEAD_DIM`-element state column.
        q_raw_s: `KEY_HEAD_DIM` shared floats for the raw query row.
        k_raw_s: `KEY_HEAD_DIM` shared floats for the raw key row.
        q_warp_sumsq_s: One shared partial sum of squares per warp.
        k_warp_sumsq_s: One shared partial sum of squares per warp.
        q_value: This thread's raw query element.
        k_value: This thread's raw key element.
        value_element: This thread's value element.
        decay_value: The token's decay for this value head.
        beta_value: The token's beta gate for this value head.
        key_update_factor: Set to the factor the key row is scaled by in the
            state update, `beta * (v - kv_memory) / ||k||`.

    Returns:
        The readout for this thread's value element.
    """
    var tid = Int(thread_idx.x)
    q_raw_s[tid] = q_value
    k_raw_s[tid] = k_value

    comptime NUM_WARPS = (KEY_HEAD_DIM + WARP_SIZE - 1) // WARP_SIZE
    # A head narrower than a warp leaves lanes unlaunched, which must not
    # join the shuffle.
    comptime REDUCE_LANES = min(KEY_HEAD_DIM, WARP_SIZE)
    var warp_q_sum = warp.lane_group_sum[num_lanes=REDUCE_LANES](
        q_value * q_value
    )
    var warp_k_sum = warp.lane_group_sum[num_lanes=REDUCE_LANES](
        k_value * k_value
    )
    if lane_id() == 0:
        q_warp_sumsq_s[tid // WARP_SIZE] = warp_q_sum
        k_warp_sumsq_s[tid // WARP_SIZE] = warp_k_sum
    barrier()
    var q_squared_sum = Float32(0.0)
    var key_squared_sum = Float32(0.0)
    comptime for w in range(NUM_WARPS):
        q_squared_sum = q_squared_sum + q_warp_sumsq_s[w]
        key_squared_sum = key_squared_sum + k_warp_sumsq_s[w]
    # Fold the 1/sqrt(KD) query scale into the query inverse-norm scalar.
    var query_scale = Float32(1.0) / std.math.sqrt(Float32(KEY_HEAD_DIM))
    var query_factor = rsqrt(q_squared_sum + Float32(1e-6)) * query_scale
    var key_inv_norm = rsqrt(key_squared_sum + Float32(1e-6))

    # Steps 1+2: kv_memory = key_inv_norm * Σ_k (decay·state_col[k]) · k_raw[k].
    # Scalar on purpose, since a full-width `k_raw` vector spills at the
    # 255-register cap.
    var kv_raw = Float32(0.0)
    comptime for kd in range(KEY_HEAD_DIM):
        state_col[kd] = state_col[kd] * decay_value
        kv_raw = kv_raw + state_col[kd] * Float32(k_raw_s[kd])

    # Step 3: k_normalised[k]·delta = k_raw[k] · (key_inv_norm·delta).
    key_update_factor = (
        beta_value * (value_element - kv_raw * key_inv_norm) * key_inv_norm
    )

    # Steps 4+5: outer-product update, query readout. The `fma` pins the
    # rounding `gated_delta_state_fold_gpu` reproduces.
    var out_raw = Float32(0.0)
    comptime for kd in range(KEY_HEAD_DIM):
        state_col[kd] = fma(
            Float32(k_raw_s[kd]), key_update_factor, state_col[kd]
        )
        out_raw = out_raw + state_col[kd] * Float32(q_raw_s[kd])

    # WAR: all reads of q_raw_s/k_raw_s must finish before the next token's
    # cooperative load overwrites them.
    barrier()
    return out_raw * query_factor


# ===----------------------------------------------------------------------=== #
# GPU Kernel
# ===----------------------------------------------------------------------=== #


def gated_delta_recurrence_fwd_gpu[
    work_dtype: DType,  # for qkv/decay/beta/recurrence_output (fp32)
    state_dtype: DType,  # for the recurrent_state pool (bf16)
    KEY_HEAD_DIM: Int,  # key_head_dim, compile-time (e.g. 128 for Qwen3.5)
    VALUE_HEAD_DIM: Int,  # value_head_dim, compile-time (e.g. 128 for Qwen3.5)
    recurrence_output_LT: TensorLayout,
    qkv_conv_output_LT: TensorLayout,
    decay_per_token_LT: TensorLayout,
    beta_per_token_LT: TensorLayout,
    recurrent_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
    Engine: TensorEngine,
](
    batch_size: Int32,
    num_value_heads: Int32,  # nv
    num_key_heads: Int32,  # nk; heads_expansion_ratio = nv / nk
    key_dim: Int32,  # num_key_heads * key_head_dim
    recurrence_output: TileTensor[
        work_dtype, recurrence_output_LT, MutUntrackedOrigin, Engine=Engine
    ],
    recurrent_state: TileTensor[
        state_dtype, recurrent_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    slot_idx: TileTensor[
        .uint32, slot_idx_LT, MutUntrackedOrigin, Engine=Engine
    ],
    qkv_conv_output: TileTensor[
        work_dtype, qkv_conv_output_LT, MutUntrackedOrigin, Engine=Engine
    ],
    decay_per_token: TileTensor[
        work_dtype, decay_per_token_LT, MutUntrackedOrigin, Engine=Engine
    ],
    beta_per_token: TileTensor[
        work_dtype, beta_per_token_LT, MutUntrackedOrigin, Engine=Engine
    ],
    input_row_offsets: TileTensor[
        .uint32, input_row_offsets_LT, MutUntrackedOrigin, Engine=Engine
    ],
    # Strides for [total_seq_len, conv_dim] tensors
    qkv_conv_output_seqlen_stride: UInt32,
    qkv_conv_output_channel_stride: UInt32,
    # Strides for [total_seq_len, num_value_heads] tensors (decay, beta)
    per_token_seqlen_stride: UInt32,
    per_token_head_stride: UInt32,
    # Strides for [total_seq_len, value_dim] recurrence output
    recurrence_output_seqlen_stride: UInt32,
    recurrence_output_valuedim_stride: UInt32,
):
    """GPU kernel: slot-indexed gated delta rule recurrence, one CTA per head.

    One CTA owns one (batch_item, value_head); thread `tid == vd_element` owns
    the KD-element state column ``recurrent_state[slot, value_head, :, tid]`` in
    registers for the whole sequence. The per-token raw Q/K for this value
    head's key head are staged once per block in shared memory (one element per
    thread, coalesced) so the KD reductions read them from shared memory rather
    than every vd-thread re-reading the same KD elements from global memory;
    L2 normalisation and the 1/sqrt(KD) query scale are folded in as scalars
    factored out of the reductions.

    Parameters:
        work_dtype: `DType` for the per-token input and output tensors
            (`qkv_conv_output`, `decay_per_token`, `beta_per_token`,
            `recurrence_output`), `float32`.
        state_dtype: `DType` for the `recurrent_state` pool (`bfloat16`).
        KEY_HEAD_DIM: Compile-time key head dimension (e.g. 128 for
            Qwen3.5).
        VALUE_HEAD_DIM: Compile-time value head dimension; must equal
            `KEY_HEAD_DIM`.
        recurrence_output_LT: `TensorLayout` for `recurrence_output`.
        qkv_conv_output_LT: `TensorLayout` for `qkv_conv_output`.
        decay_per_token_LT: `TensorLayout` for `decay_per_token`.
        beta_per_token_LT: `TensorLayout` for `beta_per_token`.
        recurrent_state_LT: `TensorLayout` for `recurrent_state`.
        slot_idx_LT: `TensorLayout` for `slot_idx`.
        input_row_offsets_LT: `TensorLayout` for
            `input_row_offsets`.
        Engine: Engine shared by all tile operands.

    Args:
        batch_size: Number of sequences in the ragged batch.
        num_value_heads: Number of value heads (`nv`).
        num_key_heads: Number of key heads (`nk`); the GQA expansion
            ratio is `num_value_heads / num_key_heads`.
        key_dim: Total key dimension, equal to
            `num_key_heads * KEY_HEAD_DIM`.
        recurrence_output: Flat output of shape
            `[total_seq_len, value_dim]` holding the recurrence result
            for every token.
        recurrent_state: Mutable state pool of shape
            `[max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM]`;
            the kernel reads and writes slot `slot_idx[batch_item]` in
            place.
        slot_idx: Pool slot index for each batch item, shape
            `[batch_size]`, `uint32`.
        qkv_conv_output: Conv1d output from Pass 1, shape
            `[total_seq_len, conv_dim]`, with Q in channels
            `[0, key_dim)`, K in `[key_dim, 2*key_dim)`, V in
            `[2*key_dim, 2*key_dim + value_dim)`.
        decay_per_token: Per-token per-head scalar decay factor, shape
            `[total_seq_len, num_value_heads]`.
        beta_per_token: Per-token per-head beta gate, shape
            `[total_seq_len, num_value_heads]`.
        input_row_offsets: Ragged offsets of shape `[batch_size + 1]`;
            sequence `b` spans flat indices
            `[input_row_offsets[b], input_row_offsets[b+1])`.
        qkv_conv_output_seqlen_stride: Stride between consecutive
            sequence positions in `qkv_conv_output`.
        qkv_conv_output_channel_stride: Stride between consecutive
            channels in `qkv_conv_output`.
        per_token_seqlen_stride: Stride between consecutive sequence
            positions in `decay_per_token` and `beta_per_token`.
        per_token_head_stride: Stride between consecutive heads in
            `decay_per_token` and `beta_per_token`.
        recurrence_output_seqlen_stride: Stride between consecutive
            sequence positions in `recurrence_output`.
        recurrence_output_valuedim_stride: Stride between consecutive
            value-dim elements in `recurrence_output`.
    """
    var _num_value_heads = Int(num_value_heads)
    var _key_dim = Int(key_dim)
    comptime assert (
        KEY_HEAD_DIM == VALUE_HEAD_DIM
    ), "gated_delta_recurrence_fwd_gpu requires KEY_HEAD_DIM == VALUE_HEAD_DIM"

    var tid = Int(thread_idx.x)
    var block = Int(block_idx.x)

    # ── block -> (batch_item, value_head) ───────────────────────────────────
    var batch_item_idx, value_head_idx = divmod(block, _num_value_heads)
    if batch_item_idx >= Int(batch_size):
        return

    # GQA: map value head to key head.
    var key_head_idx = value_head_idx // (
        _num_value_heads // Int(num_key_heads)
    )

    # Read the pool slot for this batch item exactly once. The caller
    # (`GatedDeltaNetStateCache.claim`) guarantees `slot < max_slots`.
    var slot = Int(slot_idx.raw_load(batch_item_idx))

    var q_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM, Float32, address_space=.SHARED
    ]()
    var k_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM, Float32, address_space=.SHARED
    ]()
    comptime NUM_WARPS = (KEY_HEAD_DIM + WARP_SIZE - 1) // WARP_SIZE
    var q_warp_sumsq_s = unsafe_stack_allocation[
        NUM_WARPS, Float32, address_space=.SHARED
    ]()
    var k_warp_sumsq_s = unsafe_stack_allocation[
        NUM_WARPS, Float32, address_space=.SHARED
    ]()

    # The pool is dense row-major, so its strides follow from the shape
    # instead of per-element `Coord` resolution.
    comptime pool_stride_head = KEY_HEAD_DIM * VALUE_HEAD_DIM
    comptime pool_stride_kd = VALUE_HEAD_DIM
    # 64-bit, since a deep slot's offset exceeds 2**32. Recomputed at the
    # final store instead of held live across the per-token loop.
    var pool_base_offset = (
        slot * (_num_value_heads * pool_stride_head)
        + value_head_idx * pool_stride_head
        + tid
    )

    # ── Load this thread's KD-element state column from pool[slot, ...] ──────
    var state_col = SIMD[.float32, KEY_HEAD_DIM](0.0)
    comptime for kd in range(KEY_HEAD_DIM):
        state_col[kd] = Float32(
            recurrent_state.raw_load(pool_base_offset + kd * pool_stride_kd)
        )

    var sequence_start_flat_idx = Int(
        input_row_offsets.raw_load(batch_item_idx)
    )
    var sequence_end_flat_idx = Int(
        input_row_offsets.raw_load(batch_item_idx + 1)
    )

    # Precompute constant channel offsets for Q, K, V in the conv_dim layout.
    var query_channel = UInt32(key_head_idx * KEY_HEAD_DIM + tid)
    var key_channel = UInt32(_key_dim + key_head_idx * KEY_HEAD_DIM + tid)
    var value_channel = UInt32(
        2 * _key_dim + value_head_idx * VALUE_HEAD_DIM + tid
    )

    for flat_token_idx in range(sequence_start_flat_idx, sequence_end_flat_idx):
        var token_qkv_row_offset = (
            UInt32(flat_token_idx) * qkv_conv_output_seqlen_stride
        )
        var head_token_offset = (
            UInt32(flat_token_idx) * per_token_seqlen_stride
            + UInt32(value_head_idx) * per_token_head_stride
        )
        var key_update_factor = Float32(0.0)
        var output_value = _gated_delta_token_step(
            state_col,
            q_raw_s,
            k_raw_s,
            q_warp_sumsq_s,
            k_warp_sumsq_s,
            Float32(
                qkv_conv_output.raw_load(
                    token_qkv_row_offset
                    + query_channel * qkv_conv_output_channel_stride
                )
            ),
            Float32(
                qkv_conv_output.raw_load(
                    token_qkv_row_offset
                    + key_channel * qkv_conv_output_channel_stride
                )
            ),
            Float32(
                qkv_conv_output.raw_load(
                    token_qkv_row_offset
                    + value_channel * qkv_conv_output_channel_stride
                )
            ),
            Float32(decay_per_token.raw_load(head_token_offset)),
            Float32(beta_per_token.raw_load(head_token_offset)),
            key_update_factor,
        )
        recurrence_output.raw_store(
            UInt32(flat_token_idx) * recurrence_output_seqlen_stride
            + UInt32(value_head_idx * VALUE_HEAD_DIM + tid)
            * recurrence_output_valuedim_stride,
            Scalar[work_dtype](output_value),
        )

    # ── Write final state column back into pool[slot, ...] ──────────────────
    var pool_final_base_offset = (
        slot * (_num_value_heads * pool_stride_head)
        + value_head_idx * pool_stride_head
        + tid
    )
    comptime for kd in range(KEY_HEAD_DIM):
        recurrent_state.raw_store(
            pool_final_base_offset + kd * pool_stride_kd,
            Scalar[state_dtype](state_col[kd]),
        )


# ===----------------------------------------------------------------------=== #
# Speculative verify: ring capture + commit-time fold
# ===----------------------------------------------------------------------=== #
#
# A verify advances the recurrence over every drafted position, but only the
# accepted prefix may be kept. `gated_delta_recurrence_verify_ring_gpu` runs
# the verify from the live pool without writing it back and records, per
# token, the decay, the raw key and the delta factor its update consumed.
# `gated_delta_state_fold_gpu` then applies the accepted records to the live
# pool, for every layer in one launch.
#
# The ring is addressed like the state pool, a `[rows, ...]` buffer plus a
# per-request row index. A ring row is `[num_key_heads, RING_LEN,
# record_stride]`, one record per key head and window position, laid out by
# `gated_delta_ring_record_elements`. A record is written at the token's position in the
# window, so a ring holds one window and the caller must fold after every
# verify.


@always_inline
def gated_delta_ring_record_elements[
    KEY_HEAD_DIM: Int, VALUE_HEAD_DIM: Int
](group_size: Int) -> Int:
    """Returns the elements one ring record holds, before any padding.

    The record for a key head at one window position holds, in order, the
    token's raw key, one `VALUE_HEAD_DIM` row of delta factors per value head
    of the key head's GQA group, and one decay per value head of the group.
    A ring's `record_stride` is at least this.

    Parameters:
        KEY_HEAD_DIM: Compile-time key head dimension.
        VALUE_HEAD_DIM: Compile-time value head dimension.

    Args:
        group_size: Value heads per key head.

    Returns:
        The record's element count.
    """
    return KEY_HEAD_DIM + group_size * (VALUE_HEAD_DIM + 1)


@always_inline
def _ring_delta_offset[
    KEY_HEAD_DIM: Int, VALUE_HEAD_DIM: Int
](group_head: Int) -> Int:
    return KEY_HEAD_DIM + group_head * VALUE_HEAD_DIM


@always_inline
def _ring_decay_offset[
    KEY_HEAD_DIM: Int, VALUE_HEAD_DIM: Int
](group_size: Int, group_head: Int) -> Int:
    return KEY_HEAD_DIM + group_size * VALUE_HEAD_DIM + group_head


def gated_delta_recurrence_verify_ring_gpu[
    work_dtype: DType,
    state_dtype: DType,
    ring_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    RING_LEN: Int,
    recurrence_output_LT: TensorLayout,
    qkv_conv_output_LT: TensorLayout,
    decay_per_token_LT: TensorLayout,
    beta_per_token_LT: TensorLayout,
    recurrent_state_LT: TensorLayout,
    slot_idx_LT: TensorLayout,
    input_row_offsets_LT: TensorLayout,
    ring_LT: TensorLayout,
    ring_slot_idx_LT: TensorLayout,
    Engine: TensorEngine,
](
    batch_size: Int32,
    num_value_heads: Int32,
    num_key_heads: Int32,
    key_dim: Int32,
    recurrence_output: TileTensor[
        work_dtype, recurrence_output_LT, MutUntrackedOrigin, Engine=Engine
    ],
    recurrent_state: TileTensor[
        state_dtype, recurrent_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    slot_idx: TileTensor[
        .uint32, slot_idx_LT, MutUntrackedOrigin, Engine=Engine
    ],
    qkv_conv_output: TileTensor[
        work_dtype, qkv_conv_output_LT, MutUntrackedOrigin, Engine=Engine
    ],
    decay_per_token: TileTensor[
        work_dtype, decay_per_token_LT, MutUntrackedOrigin, Engine=Engine
    ],
    beta_per_token: TileTensor[
        work_dtype, beta_per_token_LT, MutUntrackedOrigin, Engine=Engine
    ],
    input_row_offsets: TileTensor[
        .uint32, input_row_offsets_LT, MutUntrackedOrigin, Engine=Engine
    ],
    ring: TileTensor[ring_dtype, ring_LT, MutUntrackedOrigin, Engine=Engine],
    ring_slot_idx: TileTensor[
        .uint32, ring_slot_idx_LT, MutUntrackedOrigin, Engine=Engine
    ],
    ring_record_stride: Int32,
    qkv_conv_output_seqlen_stride: UInt32,
    qkv_conv_output_channel_stride: UInt32,
    per_token_seqlen_stride: UInt32,
    per_token_head_stride: UInt32,
    recurrence_output_seqlen_stride: UInt32,
    recurrence_output_valuedim_stride: UInt32,
):
    """GPU kernel: runs the gated delta recurrence over a verify window and
    records a ring instead of writing the state back.

    Produces the same `recurrence_output` as `gated_delta_recurrence_fwd_gpu`
    and leaves `recurrent_state` unchanged. Each token writes the record
    `gated_delta_state_fold_gpu` needs to advance the state over it.

    A row longer than `RING_LEN` records nothing and writes its state back
    as `gated_delta_recurrence_fwd_gpu` does, committing the whole row. The
    fold skips a row whose count exceeds `RING_LEN`, so a caller folds such a
    row with its full length.

    Parameters:
        work_dtype: `DType` of the per-token input and output tensors.
        state_dtype: `DType` of the `recurrent_state` pool.
        ring_dtype: `DType` of the ring pool. The fold is bit-exact against a
            forward over the accepted prefix only for `float32`.
        KEY_HEAD_DIM: Compile-time key head dimension.
        VALUE_HEAD_DIM: Compile-time value head dimension, equal to
            `KEY_HEAD_DIM`.
        RING_LEN: Compile-time record capacity of one ring row.
        recurrence_output_LT: `TensorLayout` for `recurrence_output`.
        qkv_conv_output_LT: `TensorLayout` for `qkv_conv_output`.
        decay_per_token_LT: `TensorLayout` for `decay_per_token`.
        beta_per_token_LT: `TensorLayout` for `beta_per_token`.
        recurrent_state_LT: `TensorLayout` for `recurrent_state`.
        slot_idx_LT: `TensorLayout` for `slot_idx`.
        input_row_offsets_LT: `TensorLayout` for `input_row_offsets`.
        ring_LT: `TensorLayout` for `ring`.
        ring_slot_idx_LT: `TensorLayout` for `ring_slot_idx`.
        Engine: Engine shared by all tile operands.

    Args:
        batch_size: Number of sequences in the ragged batch.
        num_value_heads: Number of value heads (`nv`).
        num_key_heads: Number of key heads (`nk`).
        key_dim: `num_key_heads * key_head_dim`.
        recurrence_output: `[total_seq_len, value_dim]` readout, written for
            every token of the window.
        recurrent_state: `[max_slots, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM]` dense
            live pool, written only for a row longer than `RING_LEN`.
        slot_idx: `[batch_size]` live pool row per batch item.
        qkv_conv_output: `[total_seq_len, conv_dim]` conv output.
        decay_per_token: `[total_seq_len, nv]` per-token decay.
        beta_per_token: `[total_seq_len, nv]` per-token beta gate.
        input_row_offsets: `[batch_size + 1]` ragged offsets.
        ring: `[ring_rows, nk, RING_LEN, ring_record_stride]` dense ring pool.
        ring_slot_idx: `[batch_size]` ring row per batch item.
        ring_record_stride: Elements between consecutive records, at least
            `gated_delta_ring_record_elements(nv // nk)`.
        qkv_conv_output_seqlen_stride: Stride between sequence positions in
            `qkv_conv_output`.
        qkv_conv_output_channel_stride: Stride between channels in
            `qkv_conv_output`.
        per_token_seqlen_stride: Stride between sequence positions in
            `decay_per_token` and `beta_per_token`.
        per_token_head_stride: Stride between heads in `decay_per_token` and
            `beta_per_token`.
        recurrence_output_seqlen_stride: Stride between sequence positions in
            `recurrence_output`.
        recurrence_output_valuedim_stride: Stride between value-dim elements
            in `recurrence_output`.
    """
    var _num_value_heads = Int(num_value_heads)
    var _key_dim = Int(key_dim)
    comptime assert KEY_HEAD_DIM == VALUE_HEAD_DIM, (
        "gated_delta_recurrence_verify_ring_gpu requires KEY_HEAD_DIM =="
        " VALUE_HEAD_DIM"
    )

    var tid = Int(thread_idx.x)
    var block = Int(block_idx.x)

    var batch_item_idx, value_head_idx = divmod(block, _num_value_heads)
    if batch_item_idx >= Int(batch_size):
        return

    var group_size = _num_value_heads // Int(num_key_heads)
    var key_head_idx = value_head_idx // group_size
    var group_head = value_head_idx % group_size
    # One value head per GQA group writes the shared raw key.
    var writes_ring_key = group_head == 0

    var slot = Int(slot_idx.raw_load(batch_item_idx))
    var ring_row = Int(ring_slot_idx.raw_load(batch_item_idx))

    var q_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM, Float32, address_space=.SHARED
    ]()
    var k_raw_s = unsafe_stack_allocation[
        KEY_HEAD_DIM, Float32, address_space=.SHARED
    ]()
    comptime NUM_WARPS = (KEY_HEAD_DIM + WARP_SIZE - 1) // WARP_SIZE
    var q_warp_sumsq_s = unsafe_stack_allocation[
        NUM_WARPS, Float32, address_space=.SHARED
    ]()
    var k_warp_sumsq_s = unsafe_stack_allocation[
        NUM_WARPS, Float32, address_space=.SHARED
    ]()

    comptime pool_stride_head = KEY_HEAD_DIM * VALUE_HEAD_DIM
    comptime pool_stride_kd = VALUE_HEAD_DIM
    var pool_base_offset = (
        slot * (_num_value_heads * pool_stride_head)
        + value_head_idx * pool_stride_head
        + tid
    )
    var state_col = SIMD[.float32, KEY_HEAD_DIM](0.0)
    comptime for kd in range(KEY_HEAD_DIM):
        state_col[kd] = Float32(
            recurrent_state.raw_load(pool_base_offset + kd * pool_stride_kd)
        )

    # 64-bit, like the pool offset.
    var ring_record_base = (
        (ring_row * Int(num_key_heads) + key_head_idx)
        * RING_LEN
        * Int(ring_record_stride)
    )
    var ring_delta_offset = (
        _ring_delta_offset[KEY_HEAD_DIM, VALUE_HEAD_DIM](group_head) + tid
    )
    var ring_decay_offset = _ring_decay_offset[KEY_HEAD_DIM, VALUE_HEAD_DIM](
        group_size, group_head
    )

    var sequence_start_flat_idx = Int(
        input_row_offsets.raw_load(batch_item_idx)
    )
    var sequence_length = (
        Int(input_row_offsets.raw_load(batch_item_idx + 1))
        - sequence_start_flat_idx
    )
    # Block-uniform, since a CTA covers one batch item.
    var records = sequence_length <= RING_LEN

    var query_channel = UInt32(key_head_idx * KEY_HEAD_DIM + tid)
    var key_channel = UInt32(_key_dim + key_head_idx * KEY_HEAD_DIM + tid)
    var value_channel = UInt32(
        2 * _key_dim + value_head_idx * VALUE_HEAD_DIM + tid
    )

    for position in range(sequence_length):
        var flat_token_idx = sequence_start_flat_idx + position
        var token_qkv_row_offset = (
            UInt32(flat_token_idx) * qkv_conv_output_seqlen_stride
        )
        var head_token_offset = (
            UInt32(flat_token_idx) * per_token_seqlen_stride
            + UInt32(value_head_idx) * per_token_head_stride
        )
        var k_value = Float32(
            qkv_conv_output.raw_load(
                token_qkv_row_offset
                + key_channel * qkv_conv_output_channel_stride
            )
        )
        var decay_value = Float32(decay_per_token.raw_load(head_token_offset))
        var key_update_factor = Float32(0.0)
        var output_value = _gated_delta_token_step(
            state_col,
            q_raw_s,
            k_raw_s,
            q_warp_sumsq_s,
            k_warp_sumsq_s,
            Float32(
                qkv_conv_output.raw_load(
                    token_qkv_row_offset
                    + query_channel * qkv_conv_output_channel_stride
                )
            ),
            k_value,
            Float32(
                qkv_conv_output.raw_load(
                    token_qkv_row_offset
                    + value_channel * qkv_conv_output_channel_stride
                )
            ),
            decay_value,
            Float32(beta_per_token.raw_load(head_token_offset)),
            key_update_factor,
        )
        recurrence_output.raw_store(
            UInt32(flat_token_idx) * recurrence_output_seqlen_stride
            + UInt32(value_head_idx * VALUE_HEAD_DIM + tid)
            * recurrence_output_valuedim_stride,
            Scalar[work_dtype](output_value),
        )

        if records:
            var record = ring_record_base + position * Int(ring_record_stride)
            ring.raw_store(
                record + ring_delta_offset,
                Scalar[ring_dtype](key_update_factor),
            )
            if tid == 0:
                ring.raw_store(
                    record + ring_decay_offset, Scalar[ring_dtype](decay_value)
                )
            if writes_ring_key:
                ring.raw_store(record + tid, Scalar[ring_dtype](k_value))

    # TODO(MXSERV-555): this writeback keeps state_col live to the exit, which
    # costs the recording path about 5% in spills even though no row the
    # graph builds reaches it. An early-return split spills more. If the
    # verify shows up in a profile, launch rows longer than the ring on a
    # separate kernel, which needs the host row offsets as an operand.
    if not records:
        comptime for kd in range(KEY_HEAD_DIM):
            recurrent_state.raw_store(
                pool_base_offset + kd * pool_stride_kd,
                Scalar[state_dtype](state_col[kd]),
            )


def gated_delta_state_fold_gpu[
    state_dtype: DType,
    ring_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
    RING_LEN: Int,
    recurrent_state_LT: TensorLayout,
    row_ids_LT: TensorLayout,
    ring_LT: TensorLayout,
    ring_row_ids_LT: TensorLayout,
    num_accepted_LT: TensorLayout,
    Engine: TensorEngine,
    KEY_DIM_TILE: Int = 8,
](
    batch_size: Int32,
    num_layers: Int32,
    num_value_heads: Int32,
    num_key_heads: Int32,
    recurrent_state: TileTensor[
        state_dtype, recurrent_state_LT, MutUntrackedOrigin, Engine=Engine
    ],
    row_ids: TileTensor[.uint32, row_ids_LT, MutUntrackedOrigin, Engine=Engine],
    ring: TileTensor[ring_dtype, ring_LT, MutUntrackedOrigin, Engine=Engine],
    ring_row_ids: TileTensor[
        .uint32, ring_row_ids_LT, MutUntrackedOrigin, Engine=Engine
    ],
    ring_record_stride: Int32,
    num_accepted: TileTensor[
        .uint32, num_accepted_LT, MutUntrackedOrigin, Engine=Engine
    ],
):
    """GPU kernel: applies a verify's accepted records to the state pool.

    Advances `recurrent_state` over the first `num_accepted[b]` records
    written by `gated_delta_recurrence_verify_ring_gpu`. The result is
    bit-exact against running `gated_delta_recurrence_fwd_gpu` over the
    accepted tokens. One launch covers every layer, with the layer as a grid
    coordinate.

    A row is not read or written when `num_accepted[b]` is zero, or when it
    exceeds `RING_LEN`, which names a row the verify was too long to record
    and committed itself. Rows that alias, as padding requests do on the null
    block, race, and nothing reads such a row.

    Parameters:
        state_dtype: `DType` of the `recurrent_state` pool.
        ring_dtype: `DType` of the ring pool.
        KEY_HEAD_DIM: Compile-time key head dimension.
        VALUE_HEAD_DIM: Compile-time value head dimension, equal to
            `KEY_HEAD_DIM`.
        RING_LEN: Compile-time record capacity of one ring row.
        recurrent_state_LT: `TensorLayout` for `recurrent_state`.
        row_ids_LT: `TensorLayout` for `row_ids`.
        ring_LT: `TensorLayout` for `ring`.
        ring_row_ids_LT: `TensorLayout` for `ring_row_ids`.
        num_accepted_LT: `TensorLayout` for `num_accepted`.
        Engine: Engine shared by all tile operands.
        KEY_DIM_TILE: Key-dim elements a thread holds in registers at once.
            Must divide `KEY_HEAD_DIM`.

    Args:
        batch_size: Number of sequences the verify covered.
        num_layers: Number of layers, each with its own pool row.
        num_value_heads: Number of value heads (`nv`).
        num_key_heads: Number of key heads (`nk`).
        recurrent_state: `[rows, nv, KEY_HEAD_DIM, VALUE_HEAD_DIM]` dense live
            pool, folded in place.
        row_ids: `[num_layers, batch_size]` live pool row per (layer, request).
        ring: `[ring_rows, nk, RING_LEN, ring_record_stride]` dense ring pool.
        ring_row_ids: `[num_layers, batch_size]` ring row per (layer,
            request).
        ring_record_stride: Elements between consecutive records.
        num_accepted: `[batch_size]` records to fold.
    """
    var _batch_size = Int(batch_size)
    var _num_layers = Int(num_layers)
    var _num_value_heads = Int(num_value_heads)
    comptime assert (
        KEY_HEAD_DIM == VALUE_HEAD_DIM
    ), "gated_delta_state_fold_gpu requires KEY_HEAD_DIM == VALUE_HEAD_DIM"
    comptime assert (
        KEY_HEAD_DIM % KEY_DIM_TILE == 0
    ), "gated_delta_state_fold_gpu requires KEY_DIM_TILE to divide KEY_HEAD_DIM"

    var tid = Int(thread_idx.x)
    var block = Int(block_idx.x)

    # block -> (batch_item, layer, value_head), value head fastest.
    var layer_and_batch, value_head_idx = divmod(block, _num_value_heads)
    var batch_item_idx, layer_idx = divmod(layer_and_batch, _num_layers)
    if batch_item_idx >= _batch_size:
        return

    var accepted = Int(num_accepted.raw_load(batch_item_idx))
    if accepted <= 0 or accepted > RING_LEN:
        return

    var group_size = _num_value_heads // Int(num_key_heads)
    var key_head_idx = value_head_idx // group_size
    var group_head = value_head_idx % group_size

    var table_offset = layer_idx * _batch_size + batch_item_idx
    var slot = Int(row_ids.raw_load(table_offset))
    var ring_row = Int(ring_row_ids.raw_load(table_offset))
    var ring_record_base = (
        (ring_row * Int(num_key_heads) + key_head_idx)
        * RING_LEN
        * Int(ring_record_stride)
    )
    var ring_delta_offset = (
        _ring_delta_offset[KEY_HEAD_DIM, VALUE_HEAD_DIM](group_head) + tid
    )
    var ring_decay_offset = _ring_decay_offset[KEY_HEAD_DIM, VALUE_HEAD_DIM](
        group_size, group_head
    )

    # Every thread of the block reads every key, so stage them once.
    var k_raw_s = unsafe_stack_allocation[
        RING_LEN * KEY_HEAD_DIM,
        Float32,
        address_space=.SHARED,
    ]()
    comptime for record_idx in range(RING_LEN):
        if record_idx < accepted:
            k_raw_s[record_idx * KEY_HEAD_DIM + tid] = Float32(
                ring.raw_load(
                    ring_record_base
                    + record_idx * Int(ring_record_stride)
                    + tid
                )
            )
    barrier()

    # Compile-time indices keep these in registers.
    var decays = SIMD[.float32, RING_LEN](1.0)
    var update_factors = SIMD[.float32, RING_LEN](0.0)
    comptime for record_idx in range(RING_LEN):
        if record_idx < accepted:
            var record = ring_record_base + record_idx * Int(ring_record_stride)
            decays[record_idx] = Float32(
                ring.raw_load(record + ring_decay_offset)
            )
            update_factors[record_idx] = Float32(
                ring.raw_load(record + ring_delta_offset)
            )

    comptime pool_stride_head = KEY_HEAD_DIM * VALUE_HEAD_DIM
    comptime pool_stride_kd = VALUE_HEAD_DIM
    var pool_base_offset = (
        slot * (_num_value_heads * pool_stride_head)
        + value_head_idx * pool_stride_head
        + tid
    )

    # The update is elementwise in the key dim, so the column is processed
    # in register-sized tiles. Each tile applies every record, so the state
    # is still read and written once.
    comptime NUM_TILES = KEY_HEAD_DIM // KEY_DIM_TILE
    comptime for tile_idx in range(NUM_TILES):
        comptime tile_base = tile_idx * KEY_DIM_TILE
        var state_tile = SIMD[.float32, KEY_DIM_TILE](0.0)
        comptime for i in range(KEY_DIM_TILE):
            state_tile[i] = Float32(
                recurrent_state.raw_load(
                    pool_base_offset + (tile_base + i) * pool_stride_kd
                )
            )

        comptime for record_idx in range(RING_LEN):
            if record_idx < accepted:
                # Steps 1 and 4 of `_gated_delta_token_step`, with the same
                # rounding.
                comptime for i in range(KEY_DIM_TILE):
                    state_tile[i] = fma(
                        Float32(
                            k_raw_s[record_idx * KEY_HEAD_DIM + tile_base + i]
                        ),
                        update_factors[record_idx],
                        state_tile[i] * decays[record_idx],
                    )

        comptime for i in range(KEY_DIM_TILE):
            recurrent_state.raw_store(
                pool_base_offset + (tile_base + i) * pool_stride_kd,
                Scalar[state_dtype](state_tile[i]),
            )
