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

"""Numerical E2E test for mla_prefill_sparse kernel (BF16).

This kernel performs sparse MLA prefill attention over a subset of KV tokens
selected by an external scoring phase (the indexer). DSv3.2 only uses the
"absorbed" / latent shape: qk_depth = kv_lora_rank(512) + qk_rope_head_dim(64)
= 576, v_depth = 512.

Memory layout:
  - KV:     Paged KV cache (PagedKVCacheCollection) with BF16 data.
            head_size = qk_depth.  V is the first v_depth columns.
  - Q:      [total_q_tokens, num_heads, qk_depth]   BF16
  - output: [total_q_tokens, num_heads, v_depth]    BF16
  - indices:       [total_q_tokens, indices_stride] uint32 (PER-QUERY)
  - topk_lengths:  [total_q_tokens]                 uint32 (PER-QUERY)

Host reference (per query row):
    Score = Q @ KV_sel^T * scale
    O     = softmax(Score) @ V_sel
"""

from std.math import ceildiv, exp2, sqrt
from std.math.constants import log2e
from std.memory import UnsafePointer, alloc
from std.random import randn
from std.sys import has_nvidia_gpu_accelerator, size_of

from std.gpu import *
from std.gpu.host import DeviceBuffer, DeviceContext
from std.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    row_major,
)
from nn.attention.mha_mask import NullMask
from nn.attention.mha_utils import DynamicInt
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse import (
    MLASparseConfig,
    mla_prefill_sparse,
)
from std.utils.index import Index, IndexList
from std.utils.numerics import min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# DSv3.2 absorbed dims (latent space).
comptime KV_LORA_RANK = 512
comptime QK_ROPE_HEAD_DIM = 64
comptime QK_DEPTH = KV_LORA_RANK + QK_ROPE_HEAD_DIM  # 576
comptime V_DEPTH = KV_LORA_RANK  # 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1

# softmax scale = 1 / sqrt(qk_nope_head_dim + qk_rope_head_dim) * mscale^2,
# which for DSv3.2 with mscale=1 is 1 / sqrt(128 + 64) = 1 / sqrt(192).
# Even though the kernel operates over 576 latent dims, scale uses the
# pre-absorption per-head depth (192).
comptime SOFTMAX_SCALE_BASE_DIM = 192


# ===-----------------------------------------------------------------------===#
# Helpers
# ===-----------------------------------------------------------------------===#


def _gcd(a: Int, b: Int) -> Int:
    var x = a
    var y = b
    while y != 0:
        var t = y
        y = x % y
        x = t
    return x


def _coprime_multiplier(n: Int) -> Int:
    """Find a multiplier coprime to n for deterministic token selection."""
    if n <= 1:
        return 1
    if _gcd(3, n) == 1:
        return 3
    if _gcd(5, n) == 1:
        return 5
    if _gcd(7, n) == 1:
        return 7
    if _gcd(11, n) == 1:
        return 11
    return 13


# ===-----------------------------------------------------------------------===#
# Host-side reference
# Per-query indices: each (b, s) row selects its own `topk` KV tokens.
# Score[h, k]    = Q[b,s,h,:] @ KV_sel[b,s,k,:]^T * scale
# O[b,s,h,d]     = softmax(Score)[h,:] @ V_sel[b,s,:,d]
# ===-----------------------------------------------------------------------===#


def host_reference[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    kv_sparse_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    batch_size: Int,
    seq_len: Int,
    num_heads: Int,
    topk: Int,
    qk_depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Compute reference MLA sparse prefill output on host.

    Q:               [B*seq_len, num_heads, qk_depth]
    KV_sparse:       [B*seq_len, topk, qk_depth]
    V_sparse_per_q = KV_sparse[bs, :, :v_depth]

    For each (b, s, h):
      score[k]    = Q[b,s,h,:] @ KV[b*seq_len+s,k,:]^T * scale
      prob[k]     = softmax(score)[k]
      O[b,s,h,d]  = sum_k prob[k] * V[b*seq_len+s,k,d]

    Implementation note: uses the kernel's log2-domain softmax path
    (`exp2(P*scale*log2e - mi)` and accumulate `li`) instead of the
    natural-exp form, so the host fp64 reference and the bf16 kernel
    follow the same algebraic chain.  This shaves off the
    natural-vs-log2 rounding drift from `max_err` so the diagnostic
    reflects only BF16 precision and MMA accumulator order — not the
    representation choice in the reference itself.
    """
    var scale_log2e = Float64(scale) * Float64(log2e)
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                var bs = b * seq_len + s
                var q_base = bs * num_heads * qk_depth + h * qk_depth

                var mi = Float64(min_or_neg_inf[DType.float32]())
                var s_buf = alloc[Float64](topk)

                for k in range(topk):
                    var kv_base = (bs * topk + k) * qk_depth
                    var dot = Float64(0)
                    for d in range(qk_depth):
                        dot += (
                            q_ptr[q_base + d].cast[DType.float64]()
                            * kv_sparse_ptr[kv_base + d].cast[DType.float64]()
                        )
                    # Match kernel's `cur_pi_max *= scale_log2e` step.
                    s_buf[k] = dot * scale_log2e
                    if s_buf[k] > mi:
                        mi = s_buf[k]

                # Softmax in log2 domain (matches kernel's exp2 path).
                # s_buf[k] = exp2(P_k * scale_log2e - mi); li accumulates
                # the sum; final softmax = s_buf[k] / li.
                var li = Float64(0)
                for k in range(topk):
                    s_buf[k] = exp2(s_buf[k] - mi)
                    li += s_buf[k]
                for k in range(topk):
                    s_buf[k] = s_buf[k] / li

                # O = P @ V (V = first v_depth columns of KV)
                var o_base = bs * num_heads * v_depth + h * v_depth
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(topk):
                        var kv_base = (bs * topk + k) * qk_depth
                        acc += (
                            s_buf[k]
                            * kv_sparse_ptr[kv_base + d].cast[DType.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()

                s_buf.free()


# ===-----------------------------------------------------------------------===#
# Core test function
# ===-----------------------------------------------------------------------===#


def run_test_prefill_sparse[
    q_type: DType,
    num_heads: Int,
    topk: Int,
](
    name: StringLiteral,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    ctx: DeviceContext,
) raises:
    """Test the sparse MLA prefill kernel with a paged KV cache, per-query
    indices, and the absorbed DSv3.2 dims (qk_depth=576, v_depth=512).
    """
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " seq_len:",
        seq_len,
        " num_heads:",
        num_heads,
        " num_kv_tokens:",
        num_kv_tokens,
        " topk:",
        topk,
    )

    var scale = Float32(1.0) / sqrt(Float32(SOFTMAX_SCALE_BASE_DIM))
    comptime group = num_heads
    var total_q_tokens = batch_size * seq_len

    # -----------------------------------------------------------------------
    # KV cache parameters
    # -----------------------------------------------------------------------
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=QK_DEPTH, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

    var total_pages = batch_size * ceildiv(num_kv_tokens, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_kv_tokens, PAGE_SIZE)

    var block_shape = IndexList[6](
        total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var block_elems = (
        total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # -----------------------------------------------------------------------
    # Generate random BF16 KV data: [batch_size * num_kv_tokens, qk_depth]
    # -----------------------------------------------------------------------
    var kv_total = batch_size * num_kv_tokens * QK_DEPTH
    var kv_host = alloc[Scalar[q_type]](kv_total)
    randn[q_type](kv_host, kv_total, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Build shuffled page mapping (coprime permutation).
    # -----------------------------------------------------------------------
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = alloc[UInt32](lut_size)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_kv_tokens, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    # -----------------------------------------------------------------------
    # Fill KV cache blocks from random data with paged layout.
    # -----------------------------------------------------------------------
    var blocks_host = alloc[Scalar[q_type]](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[q_type](0)

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * QK_DEPTH
            var src_base = (bi * num_kv_tokens + t) * QK_DEPTH
            for d in range(QK_DEPTH):
                blocks_host[base + d] = kv_host[src_base + d]

    # -----------------------------------------------------------------------
    # Q tensor: [total_q_tokens, num_heads, qk_depth]
    # -----------------------------------------------------------------------
    var q_elems = total_q_tokens * num_heads * QK_DEPTH
    var q_host = alloc[Scalar[q_type]](q_elems)
    randn[q_type](q_host, q_elems, mean=0.0, standard_deviation=0.5)

    # -----------------------------------------------------------------------
    # Per-query token selection: each (b, s) row picks its own topk tokens.
    # We rotate the starting point by `s` so different queries see different
    # selections (catches per-query stride bugs in the kernel).
    # selected_tokens[bs * topk + i] = which physical-row in batch `b` to use
    # -----------------------------------------------------------------------
    var selected_tokens = alloc[Int](total_q_tokens * topk)
    var sel_mult = _coprime_multiplier(num_kv_tokens)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            var rotation = s % num_kv_tokens
            for i in range(topk):
                selected_tokens[bs * topk + i] = (
                    (rotation + i) * sel_mult + 1
                ) % num_kv_tokens

    # -----------------------------------------------------------------------
    # Build sparse KV ref: [total_q_tokens, topk, qk_depth]
    # Gather selected rows from the full KV buffer per query.
    # -----------------------------------------------------------------------
    var kv_sparse_size = total_q_tokens * topk * QK_DEPTH
    var kv_sparse = alloc[Scalar[q_type]](kv_sparse_size)

    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                var t = selected_tokens[bs * topk + i]
                var src_base = (bi * num_kv_tokens + t) * QK_DEPTH
                var dst_base = (bs * topk + i) * QK_DEPTH
                for d in range(QK_DEPTH):
                    kv_sparse[dst_base + d] = kv_host[src_base + d]

    # -----------------------------------------------------------------------
    # Compute host reference output
    # -----------------------------------------------------------------------
    var out_elems = total_q_tokens * num_heads * V_DEPTH
    var ref_host = alloc[Scalar[q_type]](out_elems)
    host_reference[q_type](
        q_host,
        kv_sparse,
        ref_host,
        batch_size,
        seq_len,
        num_heads,
        topk,
        QK_DEPTH,
        V_DEPTH,
        scale,
    )

    # -----------------------------------------------------------------------
    # Copy data to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[q_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_host = alloc[UInt32](batch_size)
    for bi in range(batch_size):
        cache_lengths_host[bi] = UInt32(num_kv_tokens)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_elems)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_elems)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build per-query gather4 indices.
    # indices[bs * topk + i] = physical_block * PAGE_SIZE + tok_in_page.
    # -----------------------------------------------------------------------
    var total_indices = total_q_tokens * topk
    var h_indices = alloc[UInt32](total_indices)
    for bi in range(batch_size):
        for s in range(seq_len):
            var bs = bi * seq_len + s
            for i in range(topk):
                var t = selected_tokens[bs * topk + i]
                var page_idx = t // PAGE_SIZE
                var tok_in_page = t % PAGE_SIZE
                var block_id = Int(
                    lookup_table_host[bi * max_pages_per_batch + page_idx]
                )
                h_indices[bs * topk + i] = UInt32(
                    block_id * PAGE_SIZE + tok_in_page
                )

    var indices_device = ctx.enqueue_create_buffer[DType.uint32](total_indices)
    ctx.enqueue_copy(indices_device, h_indices)

    # topk_lengths is per-query (not per-batch): the kernel reads
    # topk_lengths[seq_idx] for seq_idx in [0, total_q_tokens).
    var h_topk_lengths = alloc[UInt32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(topk)

    var topk_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        total_q_tokens
    )
    ctx.enqueue_copy(topk_lengths_device, h_topk_lengths)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[q_type, Layout.row_major[6]()](
        blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(block_shape),
    )

    comptime cl_layout = Layout(UNKNOWN_VALUE)
    var cache_lengths_lt = LayoutTensor[DType.uint32, cl_layout](
        cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )

    comptime lt_layout_2d = Layout.row_major[2]()
    var lookup_table_lt = LayoutTensor[DType.uint32, lt_layout_2d](
        lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_pages_per_batch)
        ),
    )

    var kv_collection = PagedKVCacheCollection[q_type, kv_params, PAGE_SIZE](
        LayoutTensor[q_type, Layout.row_major[6](), MutAnyOrigin](
            blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                blocks_lt.runtime_layout.shape.value,
                blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, cl_layout, ImmutAnyOrigin](
            cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                cache_lengths_lt.runtime_layout.shape.value,
                cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, lt_layout_2d, ImmutAnyOrigin](
            lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                lookup_table_lt.runtime_layout.shape.value,
                lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(seq_len),
        UInt32(num_kv_tokens),
    )

    var kv_cache = kv_collection.get_key_cache(0)

    # -----------------------------------------------------------------------
    # Build TileTensors for Q, output, indices, and topk_lengths.
    # -----------------------------------------------------------------------
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(total_q_tokens), Idx[num_heads](), Idx[QK_DEPTH]())),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(total_q_tokens), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var indices_tt = TileTensor(
        indices_device.unsafe_ptr(),
        row_major(Idx(total_indices)),
    )

    var topk_lengths_tt = TileTensor(
        topk_lengths_device.unsafe_ptr(),
        row_major(Idx(total_q_tokens)),
    )

    # -----------------------------------------------------------------------
    # Call mla_prefill_sparse
    # -----------------------------------------------------------------------
    print("  Launching mla_prefill_sparse...")

    comptime config = MLASparseConfig[q_type](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        v_depth=V_DEPTH,
        top_k=topk,
        group=num_heads,
    )

    # PR2: no attention sink. A null pointer triggers the runtime
    # `if Int(attn_sink_ptr) == 0` branch in the kernel epilogue, which
    # skips the exp2(sink - mi) term entirely.
    var attn_sink_ptr = UnsafePointer[Float32, ImmutAnyOrigin](
        unsafe_from_address=0
    )

    mla_prefill_sparse[
        config=config,
        group=group,
        q_depth=QK_DEPTH,
    ](
        out_tt,
        q_tt,
        kv_cache,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        scale,
        Int32(topk),
        ctx,
    )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output against host reference
    # -----------------------------------------------------------------------
    var out_host = alloc[Scalar[q_type]](out_elems)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    # Tolerance set to the observed BF16 noise floor with the cast-once
    # epilogue (#3) and log2-domain host ref.  Observed max_err is
    # ≤0.13 across all 4 test shapes; 0.15 gives a small margin so
    # transient fluctuations don't flake.  Tighten if/when the kernel
    # gets closer to fp32 precision.
    var atol = Float64(0.15)
    var max_err = Float64(0)
    var max_err_low_d = Float64(0)
    var max_err_high_d = Float64(0)
    var max_err_low_h = Float64(0)
    var max_err_high_h = Float64(0)
    var max_actual = Float64(0)
    var num_nonzero = 0
    var nonzero_low_depth = 0
    var nonzero_high_depth = 0
    var nonzero_low_head = 0
    var nonzero_high_head = 0
    var total_checked = 0
    for b in range(batch_size):
        for s in range(seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[DType.float64]()
                    var actual_val = out_host[idx].cast[DType.float64]()
                    var err = abs(actual_val - ref_val)
                    if err > max_err:
                        max_err = err
                    if d < 256:
                        if err > max_err_low_d:
                            max_err_low_d = err
                    else:
                        if err > max_err_high_d:
                            max_err_high_d = err
                    if h < 64:
                        if err > max_err_low_h:
                            max_err_low_h = err
                    else:
                        if err > max_err_high_h:
                            max_err_high_h = err
                    if abs(actual_val) > max_actual:
                        max_actual = abs(actual_val)
                    if abs(actual_val) > 1e-6:
                        num_nonzero += 1
                        if d < 256:
                            nonzero_low_depth += 1
                        else:
                            nonzero_high_depth += 1
                        if h < 64:
                            nonzero_low_head += 1
                        else:
                            nonzero_high_head += 1
                    total_checked += 1
    print(
        "  DIAG: max_err=",
        max_err,
        " max_abs_actual=",
        max_actual,
        " num_nonzero=",
        num_nonzero,
        "/",
        total_checked,
    )
    print(
        "  max_err by depth: low(0..255)=",
        max_err_low_d,
        " high(256..511)=",
        max_err_high_d,
    )
    print(
        "  max_err by head: low(0..63)=",
        max_err_low_h,
        " high(64..127)=",
        max_err_high_h,
    )
    print("  Sample out vs ref for seq=0:")
    for h in [0, 32, 64, 96]:
        var base = h * V_DEPTH
        for d in [0, 64, 128, 192]:
            var idx = base + d
            print(
                "    h=",
                h,
                " d=",
                d,
                " out=",
                out_host[idx].cast[DType.float64](),
                " ref=",
                ref_host[idx].cast[DType.float64](),
            )

    # Real assertion — without this the test would print "PASSED" even
    # if the kernel regressed.
    if max_err > atol:
        raise Error(
            "max_err exceeded tolerance: "
            + String(max_err)
            + " > atol "
            + String(atol)
        )
    print(
        "  PASSED: max_err=",
        max_err,
        " checked=",
        total_checked,
        " elements",
    )

    # -----------------------------------------------------------------------
    # Cleanup
    # -----------------------------------------------------------------------
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = indices_device
    _ = topk_lengths_device

    blocks_host.free()
    kv_host.free()
    lookup_table_host.free()
    cache_lengths_host.free()
    q_host.free()
    kv_sparse.free()
    selected_tokens.free()
    ref_host.free()
    out_host.free()
    h_indices.free()
    h_topk_lengths.free()


# ===-----------------------------------------------------------------------===#
# Main
# ===-----------------------------------------------------------------------===#


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            # Without PR3's k-valid masking, topk must be a multiple of
            # B_TOPK=128 — the kernel reads B_TOPK indices per k-block per
            # query unconditionally, and falling short of that walks off
            # the per-query indices row. Phase1.cuh:628 has the same
            # constraint (`KU_ASSERT(params.topk % B_TOPK == 0)`).

            # Single k-block: topk == B_TOPK.
            run_test_prefill_sparse[DType.bfloat16, 128, 128](
                "b1_s32_h128_kv512_topk128",
                1,
                32,
                512,
                ctx,
            )

            # Multi-batch, single k-block.
            run_test_prefill_sparse[DType.bfloat16, 128, 128](
                "b4_s16_h128_kv256_topk128",
                4,
                16,
                256,
                ctx,
            )

            # Multi-block: topk=256 = 2 * B_TOPK exercises the cross-block
            # online-softmax state (mi/li updates between k iters).
            run_test_prefill_sparse[DType.bfloat16, 128, 256](
                "b1_s32_h128_kv512_topk256",
                1,
                32,
                512,
                ctx,
            )

            # Production-flavored shape, scaled down so the Float64 host
            # reference completes in a tractable amount of time. Real
            # DSv3.2 uses topk=2048; the kernel paths exercised here
            # (multi-block, multi-warpgroup pipeline, full epilogue) are
            # the same.
            run_test_prefill_sparse[DType.bfloat16, 128, 256](
                "b1_s64_h128_kv1024_topk256_prodlike",
                1,
                64,
                1024,
                ctx,
            )
        else:
            pass
