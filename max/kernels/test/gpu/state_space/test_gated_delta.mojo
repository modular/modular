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

import std.math
from std.math import sqrt

from max.gpu.host import DeviceContext
from layout import (
    Idx,
    TileTensor,
    row_major,
)
from max.gpu.host import DeviceBuffer
from std.random import rand, random_float64, seed
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from std.testing import TestSuite, assert_almost_equal, assert_equal
from std.utils.index import Index, IndexList


def run_slot_indexed_gpu[
    work_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
](
    batch_size: Int,
    total_seq_len: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    max_slots: Int,
    seq_lengths: IndexList,
    slot_assignments: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Run the slot-indexed recurrence kernel and check it against a CPU reference.

    Verifies (a) recurrence_output matches the scalar five-step recurrence and
    (b) only the pool slots named in ``slot_assignments`` are mutated; the
    remaining slots must equal their initial random fill.
    """

    var key_dim = num_key_heads * KEY_HEAD_DIM
    var value_dim = num_value_heads * VALUE_HEAD_DIM
    var conv_dim = key_dim * 2 + value_dim

    # ── Host tensors ────────────────────────────────────────────────────────
    # qkv_conv_output: [total_seq_len, conv_dim]
    var qkv_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    rand[work_dtype](qkv_heap.unsafe_ptr(), len(qkv_heap))

    # decay_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var decay_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    rand[work_dtype](decay_heap.unsafe_ptr(), len(decay_heap))
    # Decay in (0, 1): use |x| / (|x| + 1) to keep values in (0, 1)
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(decay_heap[i]))
        decay_heap[i] = Scalar[work_dtype](v / (v + Float32(1.0)))

    # beta_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var beta_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    rand[work_dtype](beta_heap.unsafe_ptr(), len(beta_heap))
    # Beta in (0, 1): same trick
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(beta_heap[i]))
        beta_heap[i] = Scalar[work_dtype](v / (v + Float32(1.0)))

    # input_row_offsets: [batch_size + 1]
    var offsets_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size + 1)
    var cumsum = 0
    offsets_heap[0] = UInt32(0)
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        offsets_heap[b + 1] = UInt32(cumsum)

    # Pool [max_slots, nv, KD, VD] zeroed so initial state for any slot is 0.
    var pool_size = max_slots * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    var pool_initial_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )
    for i in range(pool_size):
        pool_initial_heap[i] = Scalar[state_dtype](0)

    var slot_idx_heap = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        slot_idx_heap[b] = UInt32(slot_assignments[b])

    var recur_out_gpu_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * value_dim
    )
    var pool_after_gpu_heap = ctx.enqueue_create_host_buffer[state_dtype](
        pool_size
    )

    # ── Device buffers ──────────────────────────────────────────────────────
    var qkv_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var decay_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var beta_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var pool_device = ctx.enqueue_create_buffer[state_dtype](pool_size)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    var recur_out_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * value_dim
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_device, qkv_heap)
        ctx.enqueue_copy(decay_device, decay_heap)
        ctx.enqueue_copy(beta_device, beta_heap)
        ctx.enqueue_copy(offsets_device, offsets_heap)
        ctx.enqueue_copy(pool_device, pool_initial_heap)
        ctx.enqueue_copy(slot_idx_device, slot_idx_heap)

    var qkv_tt = TileTensor(qkv_device, row_major(total_seq_len, conv_dim))
    var decay_tt = TileTensor(
        decay_device, row_major(total_seq_len, num_value_heads)
    )
    var beta_tt = TileTensor(
        beta_device, row_major(total_seq_len, num_value_heads)
    )
    var offsets_tt = TileTensor(offsets_device, row_major(batch_size + 1))
    var pool_tt = TileTensor(
        pool_device,
        row_major(
            max_slots,
            num_value_heads,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
        ),
    )
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var recur_out_tt = TileTensor(
        recur_out_device, row_major(total_seq_len, value_dim)
    )

    var qkv_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_channel_stride: UInt32 = 1
    var per_token_seqlen_stride: UInt32 = UInt32(num_value_heads)
    var per_token_head_stride: UInt32 = 1
    # Not passed to the kernel (it indexes `pool_tt` via `Coord`), but the
    # CPU reference below still addresses `pool_ref_heap` by hand.
    var pool_slot_stride: UInt32 = UInt32(
        num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var pool_value_head_stride: UInt32 = UInt32(KEY_HEAD_DIM * VALUE_HEAD_DIM)
    var pool_key_dim_stride: UInt32 = UInt32(VALUE_HEAD_DIM)
    var pool_value_dim_stride: UInt32 = 1
    var output_seqlen_stride: UInt32 = UInt32(value_dim)
    var output_valuedim_stride: UInt32 = 1

    # One CTA per (batch, value_head); the block has VALUE_HEAD_DIM threads.
    var num_blocks = batch_size * num_value_heads

    var compiled_func = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            work_dtype,
            state_dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            recur_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_tt.LayoutType,
            slot_idx_tt.LayoutType,
            offsets_tt.LayoutType,
            recur_out_tt.Engine,
        ]
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(batch_size),
            Int32(num_value_heads),
            Int32(num_key_heads),
            Int32(key_dim),
            recur_out_tt,
            pool_tt,
            slot_idx_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            qkv_seqlen_stride,
            qkv_channel_stride,
            per_token_seqlen_stride,
            per_token_head_stride,
            output_seqlen_stride,
            output_valuedim_stride,
            grid_dim=(num_blocks,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    with ctx.push_context():
        ctx.enqueue_copy(recur_out_gpu_heap, recur_out_device)
        ctx.enqueue_copy(pool_after_gpu_heap, pool_device)
    ctx.synchronize()

    # ── CPU reference: scalar implementation of the five-step gated delta rule ─
    # Mirrors the GPU kernel logic exactly, iterating over every
    # (batch, value_head, vd_element) thread and every token.
    var pool_ref_heap = ctx.enqueue_create_host_buffer[state_dtype](pool_size)
    for i in range(pool_size):
        pool_ref_heap[i] = pool_initial_heap[i]

    var recur_out_ref_heap = ctx.enqueue_create_host_buffer[work_dtype](
        total_seq_len * value_dim
    )

    var heads_expansion_ratio = num_value_heads // num_key_heads
    var query_scale = Float32(1.0) / sqrt(Float32(KEY_HEAD_DIM))

    for b in range(batch_size):
        var slot = slot_assignments[b]
        var seq_start = Int(offsets_heap[b])
        var seq_end = Int(offsets_heap[b + 1])
        var seq_len = seq_end - seq_start

        for vh in range(num_value_heads):
            var kh = vh // heads_expansion_ratio
            for vd in range(VALUE_HEAD_DIM):
                # Load initial state column for this (batch, value_head, vd_element)
                var state_col = SIMD[.float32, KEY_HEAD_DIM](0.0)
                comptime for kd in range(KEY_HEAD_DIM):
                    state_col[kd] = Float32(
                        pool_ref_heap.unsafe_ptr()[
                            UInt32(slot) * pool_slot_stride
                            + UInt32(vh) * pool_value_head_stride
                            + UInt32(kd) * pool_key_dim_stride
                            + UInt32(vd) * pool_value_dim_stride
                        ]
                    )

                var q_base = UInt32(kh * KEY_HEAD_DIM)
                var k_base = UInt32(key_dim + kh * KEY_HEAD_DIM)
                var v_off_const = UInt32(2 * key_dim + vh * VALUE_HEAD_DIM + vd)

                for t in range(seq_len):
                    var token = seq_start + t
                    var token_row = UInt32(token) * qkv_seqlen_stride

                    # Load Q, K raw vectors and accumulate squared norms
                    var q_raw = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var k_raw = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var q_sq = Float32(0.0)
                    var k_sq = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        var q_val = Float32(
                            qkv_heap.unsafe_ptr()[
                                token_row
                                + (q_base + UInt32(kd)) * qkv_channel_stride
                            ]
                        )
                        var k_val = Float32(
                            qkv_heap.unsafe_ptr()[
                                token_row
                                + (k_base + UInt32(kd)) * qkv_channel_stride
                            ]
                        )
                        q_raw[kd] = q_val
                        k_raw[kd] = k_val
                        q_sq = q_sq + q_val * q_val
                        k_sq = k_sq + k_val * k_val

                    # L2 normalise Q (also scaled) and K
                    var q_inv = Float32(1.0) / sqrt(q_sq + Float32(1e-6))
                    var k_inv = Float32(1.0) / sqrt(k_sq + Float32(1e-6))
                    var q_ns = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    var k_n = SIMD[.float32, KEY_HEAD_DIM](0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        q_ns[kd] = q_raw[kd] * q_inv * query_scale
                        k_n[kd] = k_raw[kd] * k_inv

                    # Load V element
                    var v_elem = Float32(
                        qkv_heap.unsafe_ptr()[
                            token_row + v_off_const * qkv_channel_stride
                        ]
                    )
                    # Load decay and beta
                    var head_off = (
                        UInt32(token) * per_token_seqlen_stride
                        + UInt32(vh) * per_token_head_stride
                    )
                    var dec = Float32(decay_heap[Int(head_off)])
                    var bet = Float32(beta_heap[Int(head_off)])

                    # Step 1+2: decay state, accumulate kv_memory
                    var kv_mem = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        state_col[kd] = state_col[kd] * dec
                        kv_mem = kv_mem + state_col[kd] * k_n[kd]

                    # Step 3: delta correction
                    var delta = bet * (v_elem - kv_mem)

                    # Step 4+5: update state, read out output
                    var out_val = Float32(0.0)
                    comptime for kd in range(KEY_HEAD_DIM):
                        state_col[kd] = state_col[kd] + k_n[kd] * delta
                        out_val = out_val + state_col[kd] * q_ns[kd]

                    recur_out_ref_heap.unsafe_ptr().store(
                        UInt32(token) * output_seqlen_stride
                        + UInt32(vh * VALUE_HEAD_DIM + vd)
                        * output_valuedim_stride,
                        Scalar[work_dtype](out_val),
                    )

                # Write final state column
                comptime for kd in range(KEY_HEAD_DIM):
                    pool_ref_heap.unsafe_ptr().store(
                        UInt32(slot) * pool_slot_stride
                        + UInt32(vh) * pool_value_head_stride
                        + UInt32(kd) * pool_key_dim_stride
                        + UInt32(vd) * pool_value_dim_stride,
                        Scalar[state_dtype](state_col[kd]),
                    )

    # ── Compare GPU vs CPU ─────────────────────────────────────────────────────
    for i in range(total_seq_len * value_dim):
        assert_almost_equal(
            recur_out_gpu_heap[i], recur_out_ref_heap[i], rtol=rtol
        )
    for i in range(pool_size):
        assert_almost_equal(pool_after_gpu_heap[i], pool_ref_heap[i], rtol=rtol)


def test_slot_indexed_single_sequence_targets_chosen_slot() raises:
    """Single sequence, KD=VD=128: writes only to slot 1 of a 3-slot pool."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.float32, 128, 128](
        batch_size=1,
        total_seq_len=4,
        num_value_heads=1,
        num_key_heads=1,
        max_slots=3,
        seq_lengths=Index(4),
        slot_assignments=Index(1),
        ctx=ctx,
    )


def test_slot_indexed_gqa_two_sequences() raises:
    """GQA (nv=2 nk=1), two sequences hitting non-adjacent slots, bf16 pool."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=2,
        total_seq_len=5,
        num_value_heads=2,
        num_key_heads=1,
        max_slots=4,
        seq_lengths=Index(3, 2),
        slot_assignments=Index(3, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_decode_gqa_batch() raises:
    """Decode shape (seq_len 1 per request), GQA ratio 2, multi-request batch
    hitting distinct slots — mirrors Qwen3.5 decode."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=4,
        total_seq_len=4,
        num_value_heads=8,
        num_key_heads=4,
        max_slots=8,
        seq_lengths=Index(1, 1, 1, 1),
        slot_assignments=Index(5, 1, 7, 2),
        ctx=ctx,
        rtol=0.05,
    )


def test_prefill_gqa_multi_seq() raises:
    """Prefill shape (longer seqs), GQA ratio 2, two requests in distinct slots.
    """
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=2,
        total_seq_len=24,
        num_value_heads=8,
        num_key_heads=4,
        max_slots=4,
        seq_lengths=Index(16, 8),
        slot_assignments=Index(2, 0),
        ctx=ctx,
        rtol=0.05,
    )


def test_slot_indexed_batch1_single_token_production_shape() raises:
    """Checks batch 1 with a single token at 128-dim heads, 48 value heads."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=1,
        total_seq_len=1,
        num_value_heads=48,
        num_key_heads=16,
        max_slots=2,
        seq_lengths=Index(1),
        slot_assignments=Index(1),
        ctx=ctx,
        rtol=0.05,
    )


def test_slot_indexed_mixed_ragged_batch_production_shape() raises:
    """Checks one launch mixing multi-token and single-token rows."""
    var ctx = DeviceContext()
    run_slot_indexed_gpu[.float32, DType.bfloat16, 128, 128](
        batch_size=4,
        total_seq_len=10,  # 1 (decode) + 5 (prefill) + 1 (decode) + 3 (prefill)
        num_value_heads=48,
        num_key_heads=16,
        max_slots=5,
        seq_lengths=Index(1, 5, 1, 3),
        slot_assignments=Index(4, 0, 2, 1),
        ctx=ctx,
        rtol=0.05,
    )


def _fill_random_signed[dtype: DType](mut data: List[Scalar[dtype]]) raises:
    """Fills `data` with values in [-1, 1)."""
    for i in range(len(data)):
        data[i] = Scalar[dtype](Float32(random_float64(-1.0, 1.0)))


def _fill_random_unit_interval[
    dtype: DType
](mut data: List[Scalar[dtype]]) raises:
    """Fills `data` with `|x| / (|x| + 1)` values in [0, 1)."""
    for i in range(len(data)):
        var v = abs(Float32(random_float64(-2.0, 2.0)))
        data[i] = Scalar[dtype](v / (v + Float32(1.0)))


def _launch_gdn_ragged_batch[
    work_dtype: DType,
    state_dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
](
    num_value_heads: Int,
    num_key_heads: Int,
    conv_dim: Int,
    value_dim: Int,
    qkv_full: List[Scalar[work_dtype]],
    decay_full: List[Scalar[work_dtype]],
    beta_full: List[Scalar[work_dtype]],
    row_starts: IndexList,
    row_lengths: IndexList,
    row_slots: IndexList,
    max_slots: Int,
    pool_device: DeviceBuffer[state_dtype],
    ctx: DeviceContext,
) raises -> List[Scalar[work_dtype]]:
    """Launches the recurrence once over `len(row_lengths)` ragged rows.

    Row `r` takes `row_lengths[r]` tokens of the `*_full` inputs starting at
    `row_starts[r]`, so rows may overlap. `pool_device` persists across
    calls. Returns the `[total_seq_len, value_dim]` output.
    """
    var num_rows = len(row_lengths)
    var total_seq_len = 0
    for r in range(num_rows):
        total_seq_len += row_lengths[r]
    var key_dim = num_key_heads * KEY_HEAD_DIM

    var qkv_h = List[Scalar[work_dtype]](
        length=total_seq_len * conv_dim, fill=Scalar[work_dtype](0)
    )
    var decay_h = List[Scalar[work_dtype]](
        length=total_seq_len * num_value_heads, fill=Scalar[work_dtype](0)
    )
    var beta_h = List[Scalar[work_dtype]](
        length=total_seq_len * num_value_heads, fill=Scalar[work_dtype](0)
    )
    var offsets_h = List[Scalar[.uint32]](
        length=num_rows + 1, fill=Scalar[.uint32](0)
    )
    var slot_idx_h = List[Scalar[.uint32]](
        length=num_rows, fill=Scalar[.uint32](0)
    )

    var dest_tok = 0
    var cumsum = UInt32(0)
    offsets_h[0] = cumsum
    for r in range(num_rows):
        var src_start = row_starts[r]
        var length = row_lengths[r]
        for t in range(length):
            var src_tok = src_start + t
            for c in range(conv_dim):
                qkv_h[dest_tok * conv_dim + c] = qkv_full[
                    src_tok * conv_dim + c
                ]
            for h in range(num_value_heads):
                decay_h[dest_tok * num_value_heads + h] = decay_full[
                    src_tok * num_value_heads + h
                ]
                beta_h[dest_tok * num_value_heads + h] = beta_full[
                    src_tok * num_value_heads + h
                ]
            dest_tok += 1
        cumsum += UInt32(length)
        offsets_h[r + 1] = cumsum
        slot_idx_h[r] = UInt32(row_slots[r])

    var qkv_tt_h = TileTensor(qkv_h, row_major(total_seq_len, conv_dim))
    var decay_tt_h = TileTensor(
        decay_h, row_major(total_seq_len, num_value_heads)
    )
    var beta_tt_h = TileTensor(
        beta_h, row_major(total_seq_len, num_value_heads)
    )
    var offsets_tt_h = TileTensor(offsets_h, row_major(num_rows + 1))
    var slot_idx_tt_h = TileTensor(slot_idx_h, row_major(num_rows))

    var qkv_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    var decay_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var beta_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    var offsets_device = ctx.enqueue_create_buffer[.uint32](num_rows + 1)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](num_rows)
    var recur_out_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * value_dim
    )

    ctx.enqueue_copy(qkv_device, qkv_tt_h._storage)
    ctx.enqueue_copy(decay_device, decay_tt_h._storage)
    ctx.enqueue_copy(beta_device, beta_tt_h._storage)
    ctx.enqueue_copy(offsets_device, offsets_tt_h._storage)
    ctx.enqueue_copy(slot_idx_device, slot_idx_tt_h._storage)

    var qkv_tt = TileTensor(qkv_device, row_major(total_seq_len, conv_dim))
    var decay_tt = TileTensor(
        decay_device, row_major(total_seq_len, num_value_heads)
    )
    var beta_tt = TileTensor(
        beta_device, row_major(total_seq_len, num_value_heads)
    )
    var offsets_tt = TileTensor(offsets_device, row_major(num_rows + 1))
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(num_rows))
    var pool_tt = TileTensor(
        pool_device,
        row_major(max_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM),
    )
    var recur_out_tt = TileTensor(
        recur_out_device, row_major(total_seq_len, value_dim)
    )

    var compiled_func = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            work_dtype,
            state_dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            recur_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_tt.LayoutType,
            slot_idx_tt.LayoutType,
            offsets_tt.LayoutType,
            recur_out_tt.Engine,
        ]
    ]()
    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            Int32(num_rows),
            Int32(num_value_heads),
            Int32(num_key_heads),
            Int32(key_dim),
            recur_out_tt,
            pool_tt,
            slot_idx_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            offsets_tt,
            UInt32(conv_dim),
            UInt32(1),
            UInt32(num_value_heads),
            UInt32(1),
            UInt32(value_dim),
            UInt32(1),
            grid_dim=(num_rows * num_value_heads,),
            block_dim=(VALUE_HEAD_DIM,),
        )

    var out_h = List[Scalar[work_dtype]](
        length=total_seq_len * value_dim, fill=Scalar[work_dtype](0)
    )
    var out_tt_h = TileTensor(out_h, row_major(total_seq_len, value_dim))
    with ctx.push_context():
        ctx.enqueue_copy(out_tt_h._storage, recur_out_device)
    ctx.synchronize()
    return out_h^


def test_gated_delta_chunk_invariance_and_self_consistency() raises:
    """Checks one sequence run whole, in chunks, and duplicated agrees.

      (a) One launch of 6 tokens into slot 0.
      (b) Three launches of 2, 3 and 1 tokens into slot 3.
      (c) Two copies of the sequence in one batch, slots 1 and 2.

    (a) and (c) must agree exactly. (b) rounds its state to bfloat16 at two
    extra launch boundaries, so it is compared against (a) within the bound
    those roundings allow.
    """
    seed(20260918)

    comptime work_dtype = DType.float32
    comptime state_dtype = DType.bfloat16
    comptime KEY_HEAD_DIM = 128
    comptime VALUE_HEAD_DIM = 128
    comptime num_value_heads = 48
    comptime num_key_heads = 16
    comptime key_dim = num_key_heads * KEY_HEAD_DIM
    comptime value_dim = num_value_heads * VALUE_HEAD_DIM
    comptime conv_dim = key_dim * 2 + value_dim
    comptime seq_len = 6
    comptime max_slots = 4

    var ctx = DeviceContext()

    var qkv_full = List[Scalar[work_dtype]](
        length=seq_len * conv_dim, fill=Scalar[work_dtype](0)
    )
    _fill_random_signed[work_dtype](qkv_full)
    var decay_full = List[Scalar[work_dtype]](
        length=seq_len * num_value_heads, fill=Scalar[work_dtype](0)
    )
    _fill_random_unit_interval[work_dtype](decay_full)
    var beta_full = List[Scalar[work_dtype]](
        length=seq_len * num_value_heads, fill=Scalar[work_dtype](0)
    )
    _fill_random_unit_interval[work_dtype](beta_full)

    var pool_size = max_slots * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    var pool_device = ctx.enqueue_create_buffer[state_dtype](pool_size)
    ctx.enqueue_memset(pool_device, 0)

    # (a) monolithic -> slot 0.
    var out_mono = _launch_gdn_ragged_batch[
        work_dtype, state_dtype, KEY_HEAD_DIM, VALUE_HEAD_DIM
    ](
        num_value_heads,
        num_key_heads,
        conv_dim,
        value_dim,
        qkv_full,
        decay_full,
        beta_full,
        row_starts=Index(0),
        row_lengths=Index(seq_len),
        row_slots=Index(0),
        max_slots=max_slots,
        pool_device=pool_device,
        ctx=ctx,
    )

    # (c) self-consistency: the identical sequence duplicated -> slots 1, 2.
    var out_dup = _launch_gdn_ragged_batch[
        work_dtype, state_dtype, KEY_HEAD_DIM, VALUE_HEAD_DIM
    ](
        num_value_heads,
        num_key_heads,
        conv_dim,
        value_dim,
        qkv_full,
        decay_full,
        beta_full,
        row_starts=Index(0, 0),
        row_lengths=Index(seq_len, seq_len),
        row_slots=Index(1, 2),
        max_slots=max_slots,
        pool_device=pool_device,
        ctx=ctx,
    )

    # (a) and both rows of (c) run identical arithmetic, so they agree bit for
    # bit whatever else shares the launch.
    for i in range(seq_len * value_dim):
        assert_equal(out_dup[i], out_dup[seq_len * value_dim + i])
        assert_equal(out_mono[i], out_dup[i])

    # (b) chunked: three separate launches (2 + 3 + 1 tokens) -> slot 3.
    var chunk_starts: List[Int] = [0, 2, 5]
    var chunk_lengths: List[Int] = [2, 3, 1]
    var out_chunks = List[Scalar[work_dtype]](
        length=seq_len * value_dim, fill=Scalar[work_dtype](0)
    )
    var dest = 0
    for c in range(3):
        var piece = _launch_gdn_ragged_batch[
            work_dtype, state_dtype, KEY_HEAD_DIM, VALUE_HEAD_DIM
        ](
            num_value_heads,
            num_key_heads,
            conv_dim,
            value_dim,
            qkv_full,
            decay_full,
            beta_full,
            row_starts=Index(chunk_starts[c]),
            row_lengths=Index(chunk_lengths[c]),
            row_slots=Index(3),
            max_slots=max_slots,
            pool_device=pool_device,
            ctx=ctx,
        )
        for i in range(len(piece)):
            out_chunks[dest * value_dim + i] = piece[i]
        dest += chunk_lengths[c]

    # The state is rounded to `state_dtype` at each launch boundary, so the
    # chunked slot carries two more roundings than the whole one.
    var pool_readback = List[Scalar[state_dtype]](
        length=pool_size, fill=Scalar[state_dtype](0)
    )
    var pool_readback_tt = TileTensor(pool_readback, row_major(pool_size))
    with ctx.push_context():
        ctx.enqueue_copy(pool_readback_tt._storage, pool_device)
    ctx.synchronize()
    var row_elements = num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    var max_abs_state = Float32(0.0)
    var max_column_norm = Float32(0.0)
    for h in range(num_value_heads):
        for vd in range(VALUE_HEAD_DIM):
            var squared_norm = Float32(0.0)
            for kd in range(KEY_HEAD_DIM):
                var v = Float32(
                    pool_readback[(h * KEY_HEAD_DIM + kd) * VALUE_HEAD_DIM + vd]
                )
                squared_norm += v * v
                max_abs_state = max(max_abs_state, abs(v))
            max_column_norm = max(max_column_norm, std.math.sqrt(squared_norm))
    # Machine epsilon, twice the unit roundoff, so the bounds are loose.
    comptime bfloat16_epsilon = 0.0078125  # 2**-7
    comptime extra_bf16_roundtrips = 2.0  # chunked: 3 launches, mono: 1
    var state_atol = (
        Float64(max_abs_state) * bfloat16_epsilon * extra_bf16_roundtrips + 1e-6
    )
    for i in range(row_elements):
        var slot0_val = Float32(pool_readback[0 * row_elements + i])
        var slot3_val = Float32(pool_readback[3 * row_elements + i])
        assert_almost_equal(slot0_val, slot3_val, atol=state_atol, rtol=0.0)

    # A readout dots the state column with a unit query scaled by
    # 1/sqrt(KD), and the delta-rule update does not grow an error in the
    # column, so each extra rounding moves an output by at most epsilon
    # times a column norm over sqrt(KD).
    var out_atol = (
        Float64(max_column_norm)
        * bfloat16_epsilon
        * extra_bf16_roundtrips
        / std.math.sqrt(Float64(KEY_HEAD_DIM))
        + 1e-6
    )
    for i in range(seq_len * value_dim):
        assert_almost_equal(out_mono[i], out_chunks[i], atol=out_atol, rtol=0.0)


# =============================================================================
# Regression test: a `recurrent_state` row past a 32-bit offset
# =============================================================================


def test_gated_delta_recurrence_gpu_deep_slot_no_alias() raises:
    """A `recurrent_state` slot past 2**32 elements must not alias onto the
    front of the pool.

    The row is kept tiny (`nv=4`, `KD=VD=1`, `UInt8`) so that
    `slot_deep * row_elements` lands on 2**32 with a ~4 GiB pool. `K=0`
    leaves decay as the only term driving the update, and decay is 0.5
    rather than 1.0 so an aliased write cannot pass as a no-op.
    """
    var ctx = DeviceContext()
    if not ctx.is_compatible():
        return

    comptime state_dtype = DType.uint8
    comptime work_dtype = DType.float32
    comptime KEY_HEAD_DIM = 1
    comptime VALUE_HEAD_DIM = 1
    comptime num_value_heads = 4
    comptime num_key_heads = 1
    comptime row_elements = num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM  # 4
    comptime key_dim = num_key_heads * KEY_HEAD_DIM  # 1
    comptime value_dim = num_value_heads * VALUE_HEAD_DIM  # 4
    comptime conv_dim = key_dim * 2 + value_dim  # 6: [Q, K, V0..V3]
    comptime total_seq_len = 1
    comptime batch_size = 1

    # slot_deep * row_elements == 2**32 exactly: one past UInt32.MAX.
    var slot_deep = 1 << 30
    var num_slots = slot_deep + 1
    var deep_offset = slot_deep * row_elements

    var front_sentinel = Scalar[state_dtype](200)
    var deep_sentinel = Scalar[state_dtype](100)
    var decay_value = Float32(0.5)

    var qkv_heap = List(length=conv_dim, fill=Float32(0))
    var qkv_h = TileTensor(qkv_heap, row_major(total_seq_len, conv_dim))
    qkv_h.raw_store(0, Float32(1))
    var decay_heap = List(length=num_value_heads, fill=Float32(0))
    var decay_h = TileTensor(
        decay_heap, row_major(total_seq_len, num_value_heads)
    )
    decay_h.raw_store(0, decay_value)
    var beta_heap = List(length=num_value_heads, fill=Float32(0))
    var beta_h = TileTensor(
        beta_heap, row_major(total_seq_len, num_value_heads)
    )
    var offsets_heap = List(length=batch_size + 1, fill=UInt32(0))
    var offsets_h = TileTensor(offsets_heap, row_major(batch_size + 1))
    offsets_h.raw_store(1, UInt32(total_seq_len))
    var slot_idx_heap = List(length=batch_size, fill=UInt32(slot_deep))
    var slot_idx_h = TileTensor(slot_idx_heap, row_major(batch_size))

    var qkv_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * conv_dim
    )
    ctx.enqueue_copy(qkv_device, qkv_h._storage)
    var decay_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    ctx.enqueue_copy(decay_device, decay_h._storage)
    var beta_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * num_value_heads
    )
    ctx.enqueue_copy(beta_device, beta_h._storage)
    var offsets_device = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    ctx.enqueue_copy(offsets_device, offsets_h._storage)
    var slot_idx_device = ctx.enqueue_create_buffer[.uint32](batch_size)
    ctx.enqueue_copy(slot_idx_device, slot_idx_h._storage)
    var recur_out_device = ctx.enqueue_create_buffer[work_dtype](
        total_seq_len * value_dim
    )

    var qkv_tt = TileTensor(qkv_device, row_major(total_seq_len, conv_dim))
    var decay_tt = TileTensor(
        decay_device, row_major(total_seq_len, num_value_heads)
    )
    var beta_tt = TileTensor(
        beta_device, row_major(total_seq_len, num_value_heads)
    )
    var offsets_tt = TileTensor(offsets_device, row_major(batch_size + 1))
    var slot_idx_tt = TileTensor(slot_idx_device, row_major(batch_size))
    var recur_out_tt = TileTensor(
        recur_out_device, row_major(total_seq_len, value_dim)
    )

    var pool_device = ctx.enqueue_create_buffer[state_dtype](
        num_slots * row_elements
    )
    var pool_device_tt = TileTensor(
        pool_device,
        row_major(num_slots, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM),
    )

    var front_sub = pool_device.create_sub_buffer[state_dtype](0, 1)
    var deep_sub = pool_device.create_sub_buffer[state_dtype](deep_offset, 1)
    var front_seed_heap = List(length=1, fill=front_sentinel)
    var front_seed_h = TileTensor(front_seed_heap, row_major(1))
    var deep_seed_heap = List(length=1, fill=deep_sentinel)
    var deep_seed_h = TileTensor(deep_seed_heap, row_major(1))
    ctx.enqueue_copy(front_sub, front_seed_h._storage)
    ctx.enqueue_copy(deep_sub, deep_seed_h._storage)
    ctx.synchronize()

    var compiled_func = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            work_dtype,
            state_dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            recur_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            pool_device_tt.LayoutType,
            slot_idx_tt.LayoutType,
            offsets_tt.LayoutType,
            recur_out_tt.Engine,
        ]
    ]()
    ctx.enqueue_function(
        compiled_func,
        Int32(batch_size),
        Int32(num_value_heads),
        Int32(num_key_heads),
        Int32(key_dim),
        recur_out_tt,
        pool_device_tt,
        slot_idx_tt,
        qkv_tt,
        decay_tt,
        beta_tt,
        offsets_tt,
        UInt32(conv_dim),  # qkv_conv_output_seqlen_stride
        UInt32(1),  # qkv_conv_output_channel_stride
        UInt32(num_value_heads),  # per_token_seqlen_stride
        UInt32(1),  # per_token_head_stride
        UInt32(value_dim),  # recurrence_output_seqlen_stride
        UInt32(1),  # recurrence_output_valuedim_stride
        grid_dim=(batch_size * num_value_heads,),
        block_dim=(VALUE_HEAD_DIM,),
    )

    var output_readback_heap = List(
        length=total_seq_len * value_dim, fill=Float32(0)
    )
    var output_h = TileTensor(
        output_readback_heap, row_major(total_seq_len, value_dim)
    )
    var front_readback_heap = List(length=1, fill=Scalar[state_dtype](0))
    var front_readback_h = TileTensor(front_readback_heap, row_major(1))
    var deep_readback_heap = List(length=1, fill=Scalar[state_dtype](0))
    var deep_readback_h = TileTensor(deep_readback_heap, row_major(1))
    ctx.enqueue_copy(output_h._storage, recur_out_device)
    ctx.enqueue_copy(front_readback_h._storage, front_sub)
    ctx.enqueue_copy(deep_readback_h._storage, deep_sub)
    ctx.synchronize()

    # The deep sentinel, not the front row it would alias onto.
    assert_almost_equal(output_h.raw_load(0), Float32(50), rtol=0.01)

    # The front of the pool must be untouched.
    assert_equal(front_readback_h.raw_load(0), front_sentinel)

    # The deep slot itself must carry the write.
    assert_equal(deep_readback_h.raw_load(0), Scalar[state_dtype](50))


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
