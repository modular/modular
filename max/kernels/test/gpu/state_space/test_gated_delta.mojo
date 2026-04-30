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

from std.math import ceildiv, sqrt

from std.gpu.host import DeviceContext
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from std.random import rand
from state_space.gated_delta import gated_delta_recurrence_fwd_gpu
from std.testing import TestSuite, assert_almost_equal
from std.utils.index import Index, IndexList


def run_gated_delta_recurrence_gpu[
    dtype: DType,
    KEY_HEAD_DIM: Int,
    VALUE_HEAD_DIM: Int,
](
    batch_size: Int,
    total_seq_len: Int,
    num_value_heads: Int,
    num_key_heads: Int,
    seq_lengths: IndexList,
    ctx: DeviceContext,
    rtol: Float64 = 0.01,
) raises:
    """Test gated_delta_recurrence_fwd_gpu against a scalar CPU reference.

    Runs the five-step gated delta rule on CPU element-by-element and
    compares recurrence_output and recurrent_state_out against GPU results.

    Args:
        batch_size: Number of sequences in the batch.
        total_seq_len: Total flattened sequence length (sum of seq_lengths).
        num_value_heads: Number of value attention heads (nv).
        num_key_heads: Number of key attention heads (nk); nv must be a
            multiple of nk for GQA.
        seq_lengths: Per-sequence lengths summing to total_seq_len.
        ctx: GPU device context.
        rtol: Relative tolerance for GPU vs CPU comparison.
    """
    comptime RECURRENCE_BLOCK_SIZE = 128
    comptime layout_2d = Layout.row_major[2]()
    comptime layout_4d = Layout.row_major[4]()
    comptime layout_1d = Layout(UNKNOWN_VALUE)

    var key_dim = num_key_heads * KEY_HEAD_DIM
    var value_dim = num_value_heads * VALUE_HEAD_DIM
    var conv_dim = key_dim * 2 + value_dim

    # ── Allocate host tensors ──────────────────────────────────────────────────
    # qkv_conv_output: [total_seq_len, conv_dim]
    var qkv_heap = List(length=total_seq_len * conv_dim, fill=Scalar[dtype](0))
    var qkv_h = LayoutTensor[dtype, layout_2d, _](
        qkv_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, conv_dim)),
    )

    # decay_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var decay_heap = List(
        length=total_seq_len * num_value_heads, fill=Scalar[dtype](0)
    )
    var decay_h = LayoutTensor[dtype, layout_2d, _](
        decay_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(total_seq_len, num_value_heads)
        ),
    )

    # beta_per_token: [total_seq_len, num_value_heads], values in (0, 1)
    var beta_heap = List(
        length=total_seq_len * num_value_heads, fill=Scalar[dtype](0)
    )
    var beta_h = LayoutTensor[dtype, layout_2d, _](
        beta_heap,
        RuntimeLayout[layout_2d].row_major(
            Index(total_seq_len, num_value_heads)
        ),
    )

    # input_row_offsets: [batch_size + 1]
    var offsets_heap = List(length=batch_size + 1, fill=Scalar[DType.uint32](0))
    var offsets_h = LayoutTensor[DType.uint32, layout_1d, _](
        offsets_heap,
        RuntimeLayout[layout_1d].row_major(Index(batch_size + 1)),
    )
    var cumsum = 0
    offsets_h.ptr.store(0, Scalar[DType.uint32](0))
    for b in range(batch_size):
        cumsum += seq_lengths[b]
        offsets_h.ptr.store(b + 1, Scalar[DType.uint32](cumsum))

    # recurrent_state_in: [batch_size, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM]
    var state_in_heap = List(
        length=batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM,
        fill=Scalar[dtype](0),
    )
    var state_in_h = LayoutTensor[dtype, layout_4d, _](
        state_in_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )

    # GPU outputs
    var recurrence_output_gpu_heap = List(
        length=total_seq_len * value_dim, fill=Scalar[dtype](0)
    )
    var recurrence_output_gpu_h = LayoutTensor[dtype, layout_2d, _](
        recurrence_output_gpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, value_dim)),
    )

    var state_out_gpu_heap = List(
        length=batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM,
        fill=Scalar[dtype](0),
    )
    var state_out_gpu_h = LayoutTensor[dtype, layout_4d, _](
        state_out_gpu_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )

    # CPU outputs
    var recurrence_output_cpu_heap = List(
        length=total_seq_len * value_dim, fill=Scalar[dtype](0)
    )
    var recurrence_output_cpu_h = LayoutTensor[dtype, layout_2d, _](
        recurrence_output_cpu_heap,
        RuntimeLayout[layout_2d].row_major(Index(total_seq_len, value_dim)),
    )

    var state_out_cpu_heap = List(
        length=batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM,
        fill=Scalar[dtype](0),
    )
    var state_out_cpu_h = LayoutTensor[dtype, layout_4d, _](
        state_out_cpu_heap,
        RuntimeLayout[layout_4d].row_major(
            Index(batch_size, num_value_heads, KEY_HEAD_DIM, VALUE_HEAD_DIM)
        ),
    )

    # ── Fill inputs with random values ─────────────────────────────────────────
    rand[dtype](qkv_h.ptr, qkv_h.size())

    # Decay in (0, 1): use |x| / (|x| + 1) to keep values in (0, 1)
    rand[dtype](decay_h.ptr, decay_h.size())
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(decay_h.ptr[i]))
        decay_h.ptr.store(i, Scalar[dtype](v / (v + Float32(1.0))))

    # Beta in (0, 1): same trick
    rand[dtype](beta_h.ptr, beta_h.size())
    for i in range(total_seq_len * num_value_heads):
        var v = abs(Float32(beta_h.ptr[i]))
        beta_h.ptr.store(i, Scalar[dtype](v / (v + Float32(1.0))))

    # ── Allocate device buffers and copy inputs ────────────────────────────────
    var qkv_device = ctx.enqueue_create_buffer[dtype](total_seq_len * conv_dim)
    var decay_device = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_value_heads
    )
    var beta_device = ctx.enqueue_create_buffer[dtype](
        total_seq_len * num_value_heads
    )
    var offsets_device = ctx.enqueue_create_buffer[DType.uint32](batch_size + 1)
    var state_in_device = ctx.enqueue_create_buffer[dtype](
        batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var recurrence_output_device = ctx.enqueue_create_buffer[dtype](
        total_seq_len * value_dim
    )
    var state_out_device = ctx.enqueue_create_buffer[dtype](
        batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )

    with ctx.push_context():
        ctx.enqueue_copy(qkv_device, qkv_h.ptr)
        ctx.enqueue_copy(decay_device, decay_h.ptr)
        ctx.enqueue_copy(beta_device, beta_h.ptr)
        ctx.enqueue_copy(offsets_device, offsets_h.ptr)
        ctx.enqueue_copy(state_in_device, state_in_h.ptr)

    # ── Build device TileTensors ───────────────────────────────────────────────
    var qkv_tt = TileTensor(
        qkv_device, row_major(Idx(total_seq_len), Idx(conv_dim))
    )
    var decay_tt = TileTensor(
        decay_device, row_major(Idx(total_seq_len), Idx(num_value_heads))
    )
    var beta_tt = TileTensor(
        beta_device, row_major(Idx(total_seq_len), Idx(num_value_heads))
    )
    var offsets_tt = TileTensor(offsets_device, row_major(Idx(batch_size + 1)))
    var state_in_tt = TileTensor(
        state_in_device,
        row_major(
            Idx(batch_size),
            Idx(num_value_heads),
            Idx(KEY_HEAD_DIM),
            Idx(VALUE_HEAD_DIM),
        ),
    )
    var recurrence_output_tt = TileTensor(
        recurrence_output_device,
        row_major(Idx(total_seq_len), Idx(value_dim)),
    )
    var state_out_tt = TileTensor(
        state_out_device,
        row_major(
            Idx(batch_size),
            Idx(num_value_heads),
            Idx(KEY_HEAD_DIM),
            Idx(VALUE_HEAD_DIM),
        ),
    )

    # ── Strides ────────────────────────────────────────────────────────────────
    var qkv_seqlen_stride: UInt32 = UInt32(conv_dim)
    var qkv_channel_stride: UInt32 = 1
    var per_token_seqlen_stride: UInt32 = UInt32(num_value_heads)
    var per_token_head_stride: UInt32 = 1
    var state_batch_stride: UInt32 = UInt32(
        num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    var state_value_head_stride: UInt32 = UInt32(KEY_HEAD_DIM * VALUE_HEAD_DIM)
    var state_key_dim_stride: UInt32 = UInt32(VALUE_HEAD_DIM)
    var state_value_dim_stride: UInt32 = 1
    var output_seqlen_stride: UInt32 = UInt32(value_dim)
    var output_valuedim_stride: UInt32 = 1

    # ── Launch GPU kernel ──────────────────────────────────────────────────────
    var total_threads = batch_size * num_value_heads * VALUE_HEAD_DIM
    var compiled_func = ctx.compile_function[
        gated_delta_recurrence_fwd_gpu[
            dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            RECURRENCE_BLOCK_SIZE,
            recurrence_output_tt.LayoutType,
            state_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            state_in_tt.LayoutType,
            offsets_tt.LayoutType,
        ],
        gated_delta_recurrence_fwd_gpu[
            dtype,
            KEY_HEAD_DIM,
            VALUE_HEAD_DIM,
            RECURRENCE_BLOCK_SIZE,
            recurrence_output_tt.LayoutType,
            state_out_tt.LayoutType,
            qkv_tt.LayoutType,
            decay_tt.LayoutType,
            beta_tt.LayoutType,
            state_in_tt.LayoutType,
            offsets_tt.LayoutType,
        ],
    ]()

    with ctx.push_context():
        ctx.enqueue_function(
            compiled_func,
            total_threads,
            batch_size,
            total_seq_len,
            num_value_heads,
            num_key_heads,
            key_dim,
            value_dim,
            conv_dim,
            recurrence_output_tt,
            state_out_tt,
            qkv_tt,
            decay_tt,
            beta_tt,
            state_in_tt,
            offsets_tt,
            qkv_seqlen_stride,
            qkv_channel_stride,
            per_token_seqlen_stride,
            per_token_head_stride,
            state_batch_stride,
            state_value_head_stride,
            state_key_dim_stride,
            state_value_dim_stride,
            output_seqlen_stride,
            output_valuedim_stride,
            grid_dim=(ceildiv(total_threads, RECURRENCE_BLOCK_SIZE),),
            block_dim=(RECURRENCE_BLOCK_SIZE,),
        )

    # ── Copy GPU results back ──────────────────────────────────────────────────
    with ctx.push_context():
        ctx.enqueue_copy(recurrence_output_gpu_h.ptr, recurrence_output_device)
        ctx.enqueue_copy(state_out_gpu_h.ptr, state_out_device)
    ctx.synchronize()

    # ── CPU reference: scalar implementation of the five-step gated delta rule ─
    # Mirrors the GPU kernel logic exactly, iterating over every
    # (batch, value_head, vd_element) thread and every token.
    var heads_expansion_ratio = num_value_heads // num_key_heads
    var query_scale = Float32(1.0) / sqrt(Float32(KEY_HEAD_DIM))

    for batch_item_idx in range(batch_size):
        var seq_start = Int(offsets_h.ptr.load(batch_item_idx))
        var seq_end = Int(offsets_h.ptr.load(batch_item_idx + 1))
        var seq_len = seq_end - seq_start

        for value_head_idx in range(num_value_heads):
            var key_head_idx = value_head_idx // heads_expansion_ratio

            for vd_element_idx in range(VALUE_HEAD_DIM):
                # Load initial state column for this (batch, value_head, vd_element)
                var state_col = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
                comptime for kd_k in range(KEY_HEAD_DIM):
                    var state_off = (
                        UInt32(batch_item_idx) * state_batch_stride
                        + UInt32(value_head_idx) * state_value_head_stride
                        + UInt32(kd_k) * state_key_dim_stride
                        + UInt32(vd_element_idx) * state_value_dim_stride
                    )
                    state_col[kd_k] = Scalar[DType.float32](
                        state_in_h.ptr.load(state_off)
                    )

                var query_channel_base = UInt32(key_head_idx * KEY_HEAD_DIM)
                var key_channel_base = UInt32(
                    key_dim + key_head_idx * KEY_HEAD_DIM
                )
                var value_channel_offset = UInt32(
                    2 * key_dim
                    + value_head_idx * VALUE_HEAD_DIM
                    + vd_element_idx
                )

                for token_pos in range(seq_len):
                    var flat_token_idx = seq_start + token_pos
                    var token_row = UInt32(flat_token_idx) * qkv_seqlen_stride

                    # Load Q, K raw vectors and accumulate squared norms
                    var query_raw = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
                    var key_raw = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
                    var query_sq_sum = Float32(0.0)
                    var key_sq_sum = Float32(0.0)

                    comptime for kd_k in range(KEY_HEAD_DIM):
                        var q_off = (
                            token_row
                            + (query_channel_base + UInt32(kd_k))
                            * qkv_channel_stride
                        )
                        var k_off = (
                            token_row
                            + (key_channel_base + UInt32(kd_k))
                            * qkv_channel_stride
                        )
                        var q_val = Scalar[DType.float32](qkv_h.ptr.load(q_off))
                        var k_val = Scalar[DType.float32](qkv_h.ptr.load(k_off))
                        query_raw[kd_k] = q_val
                        key_raw[kd_k] = k_val
                        query_sq_sum = query_sq_sum + q_val * q_val
                        key_sq_sum = key_sq_sum + k_val * k_val

                    # L2 normalise Q (also scaled) and K
                    var q_inv_norm = Float32(1.0) / sqrt(
                        query_sq_sum + Float32(1e-6)
                    )
                    var k_inv_norm = Float32(1.0) / sqrt(
                        key_sq_sum + Float32(1e-6)
                    )

                    var query_ns = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
                    var key_n = SIMD[DType.float32, KEY_HEAD_DIM](0.0)
                    comptime for kd_k in range(KEY_HEAD_DIM):
                        query_ns[kd_k] = (
                            query_raw[kd_k] * q_inv_norm * query_scale
                        )
                        key_n[kd_k] = key_raw[kd_k] * k_inv_norm

                    # Load V element
                    var v_off = (
                        token_row + value_channel_offset * qkv_channel_stride
                    )
                    var value_element = Scalar[DType.float32](
                        qkv_h.ptr.load(v_off)
                    )

                    # Load decay and beta
                    var head_tok_off = (
                        UInt32(flat_token_idx) * per_token_seqlen_stride
                        + UInt32(value_head_idx) * per_token_head_stride
                    )
                    var decay_val = Scalar[DType.float32](
                        decay_h.ptr.load(head_tok_off)
                    )
                    var beta_val = Scalar[DType.float32](
                        beta_h.ptr.load(head_tok_off)
                    )

                    # Step 1+2: decay state, accumulate kv_memory
                    var kv_memory = Float32(0.0)
                    comptime for kd_k in range(KEY_HEAD_DIM):
                        state_col[kd_k] = state_col[kd_k] * decay_val
                        kv_memory = kv_memory + state_col[kd_k] * key_n[kd_k]

                    # Step 3: delta correction
                    var delta = beta_val * (value_element - kv_memory)

                    # Step 4+5: update state, read out output
                    var output_val = Float32(0.0)
                    comptime for kd_k in range(KEY_HEAD_DIM):
                        state_col[kd_k] = state_col[kd_k] + key_n[kd_k] * delta
                        output_val = (
                            output_val + state_col[kd_k] * query_ns[kd_k]
                        )

                    var out_off = (
                        UInt32(flat_token_idx) * output_seqlen_stride
                        + UInt32(
                            value_head_idx * VALUE_HEAD_DIM + vd_element_idx
                        )
                        * output_valuedim_stride
                    )
                    recurrence_output_cpu_h.ptr.store(
                        out_off, Scalar[dtype](output_val)
                    )

                # Write final state column
                comptime for kd_k in range(KEY_HEAD_DIM):
                    var state_off = (
                        UInt32(batch_item_idx) * state_batch_stride
                        + UInt32(value_head_idx) * state_value_head_stride
                        + UInt32(kd_k) * state_key_dim_stride
                        + UInt32(vd_element_idx) * state_value_dim_stride
                    )
                    state_out_cpu_h.ptr.store(
                        state_off, Scalar[dtype](state_col[kd_k])
                    )

    # ── Compare GPU vs CPU ─────────────────────────────────────────────────────
    var output_size = total_seq_len * value_dim
    for i in range(output_size):
        assert_almost_equal(
            recurrence_output_gpu_h.ptr[i],
            recurrence_output_cpu_h.ptr[i],
            rtol=rtol,
        )

    var state_size = (
        batch_size * num_value_heads * KEY_HEAD_DIM * VALUE_HEAD_DIM
    )
    for i in range(state_size):
        assert_almost_equal(
            state_out_gpu_h.ptr[i],
            state_out_cpu_h.ptr[i],
            rtol=rtol,
        )
    _ = state_out_cpu_heap^
    _ = recurrence_output_cpu_heap^
    _ = state_out_gpu_heap^
    _ = recurrence_output_gpu_heap^
    _ = state_in_heap^
    _ = offsets_heap^
    _ = beta_heap^
    _ = decay_heap^
    _ = qkv_heap^


# =============================================================================
# Test functions
# =============================================================================


def test_gated_delta_recurrence_single_sequence() raises:
    """Test recurrence with a single sequence."""
    var ctx = DeviceContext()
    run_gated_delta_recurrence_gpu[DType.float32, 128, 128](
        batch_size=1,
        total_seq_len=10,
        num_value_heads=8,
        num_key_heads=8,
        seq_lengths=Index(10),
        ctx=ctx,
    )


def test_gated_delta_recurrence_multiple_sequences() raises:
    """Test recurrence with multiple variable-length sequences."""
    var ctx = DeviceContext()
    run_gated_delta_recurrence_gpu[DType.float32, 128, 128](
        batch_size=3,
        total_seq_len=15,
        num_value_heads=8,
        num_key_heads=8,
        seq_lengths=Index(5, 7, 3),
        ctx=ctx,
    )


def test_gated_delta_recurrence_grouped_query_attention() raises:
    """Test recurrence with GQA (more value heads than key heads)."""
    var ctx = DeviceContext()
    run_gated_delta_recurrence_gpu[DType.float32, 128, 128](
        batch_size=2,
        total_seq_len=20,
        num_value_heads=16,
        num_key_heads=8,
        seq_lengths=Index(8, 12),
        ctx=ctx,
    )


def test_gated_delta_recurrence_short_sequences() raises:
    """Test recurrence with sequences of length 1."""
    var ctx = DeviceContext()
    run_gated_delta_recurrence_gpu[DType.float32, 128, 128](
        batch_size=2,
        total_seq_len=3,
        num_value_heads=4,
        num_key_heads=4,
        seq_lengths=Index(2, 1),
        ctx=ctx,
    )


def test_gated_delta_recurrence_larger_sequence() raises:
    """Test recurrence with a longer sequence to exercise sequential token loop.
    """
    var ctx = DeviceContext()
    run_gated_delta_recurrence_gpu[DType.float32, 128, 128](
        batch_size=1,
        total_seq_len=64,
        num_value_heads=8,
        num_key_heads=8,
        seq_lengths=Index(64),
        ctx=ctx,
    )


def main() raises:
    TestSuite.discover_tests[__functions_in_module()]().run()
