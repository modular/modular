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

"""Numerical E2E test for MLA_SM100_Decode_Sparse_KV_FP8.

This kernel stores the full 576-byte KV row (nope+rope) as FP8 in HBM,
loaded via a single INT64/SWIZZLE_NONE gather4 (tile_width=72 INT64 =
576 B). The parent kernel (`MLA_SM100_Decode_Sparse`) instead keeps rope
in BF16 (128 bytes) so its rows are 640 bytes.

Layout tested here per KV row:
    [nope: 512 FP8 bytes] [rope: 64 FP8 bytes]  = 576 bytes total

This test covers the full feature matrix supported by the kernel:
  * base (NullMask, no extra features)
  * CausalMask
  * multi-token Q (q_max_seq_len > 1)
  * multi-batch (batch_size > 1)
  * variable per-batch topk (has_variable_topk)
  * attention sinks (has_attn_sink)
  * extra KV (has_extra_kv)
  * topk clamping (small cache with topk > actual_tokens)

For FP8 correctness, every variant:
  1. Casts random BF16 K data to FP8 before storing in the paged cache.
  2. Reads back the FP8 cache and casts to BF16 for the reference.
This ensures the reference sees exactly the same quantized values as the
kernel, so the tolerance measures kernel correctness, not quantization
error.
"""

from std.math import ceildiv, exp
from std.memory import UnsafePointer, alloc, bitcast
from std.random import randn, seed
from std.sys import has_nvidia_gpu_accelerator, size_of

from std.gpu import *
from std.gpu.host import DeviceBuffer, DeviceContext, FuncAttribute
from std.gpu.host.info import _is_sm10x_gpu
from std.gpu.host.nvidia.tma import TensorMapSwizzle
from std.gpu.memory import AddressSpace
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Idx,
    Layout,
    LayoutTensor,
    RuntimeLayout,
    TileTensor,
    UNKNOWN_VALUE,
    lt_to_tt,
    row_major,
)
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_operand import KVCacheMHAOperand
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
    compute_mla_dispatch_scalars,
)
from std.testing import assert_almost_equal
from std.utils.index import IndexList
from std.utils.numerics import isnan, min_or_neg_inf


# ===-----------------------------------------------------------------------===#
# Test constants
# ===-----------------------------------------------------------------------===#

# MLA dimensions (matching DeepSeek V3 production config).
comptime Q_DEPTH = 576  # Full Q depth: 512 nope + 64 rope
comptime V_DEPTH = 512  # Output depth (nope only)
comptime ROPE_DEPTH = 64  # Rope dimension
comptime PAGE_SIZE = 128  # Standard page size
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1  # MLA has 1 KV head

# ALL-FP8 KV layout: 512 (FP8 nope) + 64 (FP8 rope) = 576 FP8 bytes.
# Unlike test_mla_sparse.mojo (640), rope is ALSO FP8 here.
comptime KV_HEAD_SIZE = V_DEPTH + ROPE_DEPTH  # 576


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
# Host-side reference: BF16 Q (576) x combined K^T (576) -> P -> O
# ===-----------------------------------------------------------------------===#


def host_reference[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    k_bf16_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,  # Q_DEPTH = 576
    v_depth: Int,  # V_DEPTH = 512
    scale: Float32,
    q_max_seq_len: Int = 1,
    use_causal: Bool = False,
    cache_len: Int = 0,
):
    """Reference MLA output on host with same FP8-rounded data the kernel sees.

    Q: [batch_size * q_max_seq_len, num_heads, depth(576)] (ragged)
    K: [batch_size, num_keys, depth(576)] in BF16 (already FP8-roundtripped)
    V = K[:, :, :v_depth]

    If `use_causal` is True, query token `s` only attends to key tokens
    with index <= (cache_len + s) inside the same batch.
    """
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                var q_base = (
                    b * q_max_seq_len * num_heads * depth
                    + s * num_heads * depth
                    + h * depth
                )

                var max_s = Float64(min_or_neg_inf[DType.float32]())
                var s_buf = List(length=num_keys, fill=Float64(0))
                var valid = List(length=num_keys, fill=False)

                # Determine causal limit for this query row within the batch.
                var causal_limit = num_keys
                if use_causal:
                    causal_limit = cache_len + s + 1
                    if causal_limit > num_keys:
                        causal_limit = num_keys

                for k in range(num_keys):
                    if use_causal and k >= causal_limit:
                        valid[k] = False
                        s_buf[k] = Float64(min_or_neg_inf[DType.float32]())
                        continue
                    valid[k] = True
                    var k_base = b * num_keys * depth + k * depth
                    var dot = Float64(0)
                    for d in range(depth):
                        dot += (
                            q_ptr[q_base + d].cast[DType.float64]()
                            * k_bf16_ptr[k_base + d].cast[DType.float64]()
                        )
                    s_buf[k] = dot * Float64(scale)
                    if s_buf[k] > max_s:
                        max_s = s_buf[k]

                var sum_exp = Float64(0)
                for k in range(num_keys):
                    if not valid[k]:
                        s_buf[k] = Float64(0)
                        continue
                    s_buf[k] = exp(s_buf[k] - max_s)
                    sum_exp += s_buf[k]
                for k in range(num_keys):
                    if valid[k]:
                        s_buf[k] = s_buf[k] / sum_exp

                var o_base = (
                    b * q_max_seq_len * num_heads * v_depth
                    + s * num_heads * v_depth
                    + h * v_depth
                )
                for d in range(v_depth):
                    var acc = Float64(0)
                    for k in range(num_keys):
                        if not valid[k]:
                            continue
                        var k_base = b * num_keys * depth + k * depth
                        acc += (
                            s_buf[k]
                            * k_bf16_ptr[k_base + d].cast[DType.float64]()
                        )
                    output_ptr[o_base + d] = acc.cast[q_type]()


def host_reference_with_attn_sink[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    k_bf16_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    attn_sink_host: UnsafePointer[Float32, _],
    batch_size: Int,
    num_heads: Int,
    num_keys: Int,
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA with attn_sink correction (natural log domain).

    attn_sink is shape [num_heads_q]. The softmax denominator is adjusted:
      sum_exp += exp(attn_sink[h] - max_s)
    which captures the aggregate contribution of non-selected tokens.
    """
    for b in range(batch_size):
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[DType.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = b * num_keys * depth + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[DType.float64]()
                        * k_bf16_ptr[k_base + d].cast[DType.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var attn_sink_val = Float64(attn_sink_host[h])
            if attn_sink_val > max_s:
                max_s = attn_sink_val

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            sum_exp += exp(attn_sink_val - max_s)

            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = b * num_keys * depth + k * depth
                    acc += (
                        s_buf[k] * k_bf16_ptr[k_base + d].cast[DType.float64]()
                    )
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^


def host_reference_varkeys[
    q_type: DType,
](
    q_ptr: UnsafePointer[Scalar[q_type], _],
    k_bf16_ptr: UnsafePointer[Scalar[q_type], _],
    output_ptr: UnsafePointer[mut=True, Scalar[q_type], _],
    batch_size: Int,
    num_heads: Int,
    num_keys_per_batch: List[Int],
    depth: Int,
    v_depth: Int,
    scale: Float32,
):
    """Reference MLA for variable-length per-batch sparse attention.

    K buffer is packed contiguously per batch: k_bf16_ptr contains
    sum_b num_keys_per_batch[b] * depth elements.
    """
    var k_offsets = List(length=batch_size, fill=Int(0))
    var running = 0
    for b in range(batch_size):
        k_offsets[b] = running
        running += num_keys_per_batch[b] * depth

    for b in range(batch_size):
        var num_keys = num_keys_per_batch[b]
        var k_base_offset = k_offsets[b]
        for h in range(num_heads):
            var q_base = b * num_heads * depth + h * depth

            var max_s = Float64(min_or_neg_inf[DType.float32]())
            var s_buf = List(length=num_keys, fill=Float64(0))
            for k in range(num_keys):
                var k_base = k_base_offset + k * depth
                var dot = Float64(0)
                for d in range(depth):
                    dot += (
                        q_ptr[q_base + d].cast[DType.float64]()
                        * k_bf16_ptr[k_base + d].cast[DType.float64]()
                    )
                s_buf[k] = dot * Float64(scale)
                if s_buf[k] > max_s:
                    max_s = s_buf[k]

            var sum_exp = Float64(0)
            for k in range(num_keys):
                s_buf[k] = exp(s_buf[k] - max_s)
                sum_exp += s_buf[k]
            for k in range(num_keys):
                s_buf[k] = s_buf[k] / sum_exp

            var o_base = b * num_heads * v_depth + h * v_depth
            for d in range(v_depth):
                var acc = Float64(0)
                for k in range(num_keys):
                    var k_base = k_base_offset + k * depth
                    acc += (
                        s_buf[k] * k_bf16_ptr[k_base + d].cast[DType.float64]()
                    )
                output_ptr[o_base + d] = acc.cast[q_type]()
            _ = s_buf^
    _ = k_offsets^


# ===-----------------------------------------------------------------------===#
# Core test function (all-FP8 KV, sparse, tensorwise)
# Parametrized over mask type (NullMask / CausalMask) via a `use_causal` flag
# at runtime — the call to flare_mla_decoding picks the right constructor.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8[
    q_type: DType,
    kv_type: DType,  # float8_e4m3fn
    num_heads: Int,
    use_causal: Bool = False,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
    q_max_seq_len: Int = 1,
) raises:
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
        " q_max_seq_len:",
        q_max_seq_len,
        " causal:",
        use_causal,
    )

    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)

    # All-FP8 layout: head_size = 576 (V_DEPTH + ROPE_DEPTH).
    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1  # MLA: is_mla=True => dim[1]=1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

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

    # Allocate KV cache on host, zero-initialized.
    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    # Generate random BF16 data for nope (512) + rope (64) per token.
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # Token stride in the KV cache = head_size = 576 FP8 slots.
    var tok_stride = kv_params.head_size  # 576 FP8 slots

    # Build shuffled page table (so gather4 actually exercises scatter).
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        lut_size
    )
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)

    # Fill KV cache: nope (512) AND rope (64) both cast BF16 -> FP8.
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH

            # Write ALL 576 slots as FP8 (nope + rope both cast).
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    # Reference K: read back FP8 bytes as BF16 so the reference sees
    # exactly the same quantized values as the kernel.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH

            for d in range(Q_DEPTH):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

    # Q tensor: [batch_size * q_max_seq_len, num_heads, Q_DEPTH] (ragged).
    var q_size = batch_size * q_max_seq_len * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Select topk unique tokens per batch via deterministic permutation.
    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    # Build sparse reference K buffer [batch_size, topk, Q_DEPTH].
    var k_sparse_ref_size = batch_size * topk * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](k_sparse_ref_size)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            var dst_base = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]

    # For the causal reference we need per-batch virtual token positions
    # for each selected entry. Since each batch uses the same permutation
    # we can reconstruct this from selected_tokens.
    # However the kernel applies causal to logical positions
    # (cache_len + s + 1 relative to the full cache, NOT the sparse set).
    # To match, we use the logical token index `selected_tokens[i]`
    # as the key's position, and clamp each q-token's reach to `cache_len + s + 1`.
    var out_size = batch_size * q_max_seq_len * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)

    if use_causal:
        # Build the causal sparse reference using logical token positions.
        for b in range(batch_size):
            for s in range(q_max_seq_len):
                var causal_limit = cache_len + s + 1
                for h in range(num_heads):
                    var q_base = (
                        b * q_max_seq_len * num_heads * Q_DEPTH
                        + s * num_heads * Q_DEPTH
                        + h * Q_DEPTH
                    )
                    var max_s = Float64(min_or_neg_inf[DType.float32]())
                    var s_buf = List(length=topk, fill=Float64(0))
                    var valid = List(length=topk, fill=False)
                    for i in range(topk):
                        var tok = selected_tokens[b * topk + i]
                        if tok >= causal_limit:
                            valid[i] = False
                            s_buf[i] = Float64(min_or_neg_inf[DType.float32]())
                            continue
                        valid[i] = True
                        var k_base = b * topk * Q_DEPTH + i * Q_DEPTH
                        var dot = Float64(0)
                        for d in range(Q_DEPTH):
                            dot += (
                                q_host[q_base + d].cast[DType.float64]()
                                * k_sparse_ref[k_base + d].cast[DType.float64]()
                            )
                        s_buf[i] = dot * Float64(scale)
                        if s_buf[i] > max_s:
                            max_s = s_buf[i]

                    var sum_exp = Float64(0)
                    for i in range(topk):
                        if not valid[i]:
                            s_buf[i] = Float64(0)
                            continue
                        s_buf[i] = exp(s_buf[i] - max_s)
                        sum_exp += s_buf[i]
                    if sum_exp > Float64(0):
                        for i in range(topk):
                            if valid[i]:
                                s_buf[i] = s_buf[i] / sum_exp

                    var o_base = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                    )
                    for d in range(V_DEPTH):
                        var acc = Float64(0)
                        for i in range(topk):
                            if not valid[i]:
                                continue
                            var k_base = b * topk * Q_DEPTH + i * Q_DEPTH
                            acc += (
                                s_buf[i]
                                * k_sparse_ref[k_base + d].cast[DType.float64]()
                            )
                        ref_host[o_base + d] = acc.cast[q_type]()
                    _ = valid^
                    _ = s_buf^
    else:
        host_reference[q_type](
            q_host.unsafe_ptr(),
            k_sparse_ref.unsafe_ptr(),
            ref_host.unsafe_ptr(),
            batch_size,
            num_heads,
            topk,
            Q_DEPTH,
            V_DEPTH,
            scale,
            q_max_seq_len,
        )

    # -----------------------------------------------------------------------
    # Copy to device
    # -----------------------------------------------------------------------
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build PagedKVCacheCollection on device
    # -----------------------------------------------------------------------
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
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
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )

    var kv_cache = kv_collection.get_key_cache(0)
    var kv_lut = KVCacheMHAOperand(kv_cache)

    # -----------------------------------------------------------------------
    # Build gather4 indices for the selected topk tokens.
    #   d_indices[batch * topk + i] = physical_block * PAGE_SIZE + offset.
    # -----------------------------------------------------------------------
    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[DType.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    var d_indices_device = ctx.enqueue_create_buffer[DType.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Build TileTensors and call flare_mla_decoding through dispatch.
    # -----------------------------------------------------------------------
    var total_q_tokens = batch_size * q_max_seq_len
    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(total_q_tokens), Idx[num_heads](), Idx[Q_DEPTH]())),
    )

    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(total_q_tokens), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)
    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = topk

    print(
        "  Launching MLA sparse KV_FP8 decode kernel...",
        " topk=",
        topk,
        " num_keys=",
        num_keys,
    )

    comptime if use_causal:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            lt_to_tt(scalar_args_buf_lt),
            d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[q_type](num_heads, Q_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            kv_cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            lt_to_tt(scalar_args_buf_lt),
            d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
                d_indices_device.unsafe_ptr()
            ),
            indices_stride=indices_stride,
        )

    ctx.synchronize()

    # -----------------------------------------------------------------------
    # Verify output: max abs error must be < 5e-2 and zero NaN allowed.
    # -----------------------------------------------------------------------
    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
    var nan_count = 0
    var total_checked = 0
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[DType.float64]()
                    var actual_val = out_host[idx].cast[DType.float64]()
                    if isnan(actual_val):
                        nan_count += 1
                        if nan_count <= 5:
                            print(
                                "  NaN at b=",
                                b,
                                " s=",
                                s,
                                " h=",
                                h,
                                " d=",
                                d,
                            )
                        continue
                    var err = abs(actual_val - ref_val)
                    if err > max_err:
                        max_err = err
                    total_checked += 1
                    if err > 1e-1:
                        print(
                            "  large err b=",
                            b,
                            " s=",
                            s,
                            " h=",
                            h,
                            " d=",
                            d,
                            " got=",
                            actual_val,
                            " ref=",
                            ref_val,
                            " err=",
                            err,
                        )

    if nan_count > 0:
        print(
            "  FAILED: ",
            nan_count,
            "NaN values in output (max_err over non-NaN:",
            max_err,
            ")",
        )
        raise Error("NaN in kernel output")

    # Run asserts only after NaN scan (so we see all NaNs before failing).
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            for h in range(num_heads):
                for d in range(V_DEPTH):
                    var idx = (
                        b * q_max_seq_len * num_heads * V_DEPTH
                        + s * num_heads * V_DEPTH
                        + h * V_DEPTH
                        + d
                    )
                    var ref_val = ref_host[idx].cast[DType.float64]()
                    var actual_val = out_host[idx].cast[DType.float64]()
                    assert_almost_equal(
                        actual_val, ref_val, atol=atol, rtol=rtol
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
    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Variable-topk sparse test (has_variable_topk=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_variable_topk[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Variable-topk per-batch with all-FP8 KV (576-byte row).

    Per-batch cache lengths and topk counts. indices_stride = max(topk).
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
        )
    print("  indices_stride (max topk)=", max_topk)

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

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
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        lut_size
    )
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH

    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    # FP8-roundtripped reference K.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH

            # Write all 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()

        k_offset += nk * Q_DEPTH

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # d_indices: [batch_size * max_topk] padded with zeros beyond each
    # batch's actual topk.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[DType.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += topk_per_batch[bi] * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(topk_per_batch[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    var d_indices_device = ctx.enqueue_create_buffer[DType.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[DType.int32](
        batch_size
    )
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[DType.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
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
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[Q_DEPTH]())),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = max_topk
    print("  Launching MLA sparse KV_FP8 (variable topk)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        lt_to_tt(scalar_args_buf_lt),
        d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=indices_stride,
        topk_lengths=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
    var total_checked = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Attention sink sparse test (has_attn_sink=True)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_attn_sink[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    batch_size: Int,
    cache_len: Int,
    ctx: DeviceContext,
    topk: Int,
) raises:
    """All-FP8 sparse MLA decode with attn_sink correction.

    attn_sink shape is [num_heads_q] (NOT batch_size) per the MOGG fix.
    """
    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " cache_len:",
        cache_len,
        " num_heads:",
        num_heads,
        " topk:",
        topk,
    )

    comptime q_max_seq_len = 1
    var num_keys = cache_len + q_max_seq_len
    comptime scale = Float32(0.125)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

    var total_pages = batch_size * ceildiv(num_keys, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(num_keys, PAGE_SIZE)

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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var k_bf16_total = batch_size * num_keys * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var tok_stride = kv_params.head_size

    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        lut_size
    )
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np = ceildiv(num_keys, PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            var shuffled_p = (p * mult + 1) % np
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            for d in range(Q_DEPTH):
                blocks_host[base + d] = k_bf16_host[k_base + d].cast[kv_type]()

    # FP8-roundtripped reference K.
    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](k_bf16_total)
    for bi in range(batch_size):
        for t in range(num_keys):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = block_id * page_stride_elems + tok_in_page * tok_stride
            var k_base = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            for d in range(Q_DEPTH):
                k_ref_host[k_base + d] = blocks_host[base + d].cast[q_type]()

    # Select topk tokens via deterministic permutation.
    var selected_tokens = List(length=batch_size * topk, fill=Int(0))
    for bi in range(batch_size):
        var mult = _coprime_multiplier(num_keys)
        for i in range(topk):
            selected_tokens[bi * topk + i] = (i * mult + 1) % num_keys

    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        batch_size * topk * Q_DEPTH
    )
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var src = bi * num_keys * Q_DEPTH + t * Q_DEPTH
            var dst = bi * topk * Q_DEPTH + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst + d] = k_ref_host[src + d]

    # attn_sink per head (natural log domain).
    var attn_sink_host = ctx.enqueue_create_host_buffer[DType.float32](
        num_heads
    )
    for h in range(num_heads):
        attn_sink_host[h] = Float32(
            -1.0 + 3.0 * Float32(h) / Float32(num_heads)
        )

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.3)

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_with_attn_sink[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        attn_sink_host.unsafe_ptr(),
        batch_size,
        num_heads,
        topk,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Uploads.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)

    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_len)
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)

    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)

    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    var attn_sink_device = ctx.enqueue_create_buffer[DType.float32](num_heads)
    ctx.enqueue_copy(attn_sink_device, attn_sink_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
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
        UInt32(q_max_seq_len),
        UInt32(cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var total_indices = batch_size * topk
    var h_indices = ctx.enqueue_create_host_buffer[DType.int32](total_indices)
    for bi in range(batch_size):
        for i in range(topk):
            var t = selected_tokens[bi * topk + i]
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var block_id = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * topk + i] = Int32(block_id * PAGE_SIZE + tok_in_page)

    var d_indices_device = ctx.enqueue_create_buffer[DType.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    ctx.synchronize()

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[Q_DEPTH]())),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = topk
    print("  Launching MLA sparse KV_FP8 (attn_sink)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        lt_to_tt(scalar_args_buf_lt),
        d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=indices_stride,
        attn_sink_ptr=rebind[
            UnsafePointer[Scalar[DType.float32], origin=MutAnyOrigin]
        ](attn_sink_device.unsafe_ptr()),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var max_err = Float64(0)
    var total_checked = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=5e-2, rtol=1e-2)

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = row_offsets_device
    _ = attn_sink_device
    _ = selected_tokens^


# ===-----------------------------------------------------------------------===#
# Extra KV sparse test (has_extra_kv=True)
# Both the original and extra KV caches use the same 576-byte all-FP8 layout.
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_extra_kv[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    extra_cache_lengths: List[Int],
    extra_topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Two paged KV caches (main + always-attend). Both all-FP8, 576 B/row."""
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var max_extra_topk = 0
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        if extra_topk_per_batch[bi] > max_extra_topk:
            max_extra_topk = extra_topk_per_batch[bi]

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " extra_cache_len=",
            extra_cache_lengths[i],
            " extra_topk=",
            extra_topk_per_batch[i],
        )

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1
    var tok_stride = kv_params.head_size
    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    # --- ORIGINAL cache ----------------------------------------------------
    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)
    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

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

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        lut_size
    )
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH
            # All 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()
        k_offset += nk * Q_DEPTH

    # --- EXTRA cache -------------------------------------------------------
    var max_extra_cache_len = 0
    var extra_total_pages = 0
    var extra_num_keys_list = List[Int]()
    for i in range(batch_size):
        var ecl = extra_cache_lengths[i]
        if ecl > max_extra_cache_len:
            max_extra_cache_len = ecl
        var enk = ecl + q_max_seq_len
        extra_num_keys_list.append(enk)
        extra_total_pages += ceildiv(enk, PAGE_SIZE)
    var max_extra_num_keys = max_extra_cache_len + q_max_seq_len
    var max_extra_pages_per_batch = ceildiv(max_extra_num_keys, PAGE_SIZE)

    var extra_block_shape = IndexList[6](
        extra_total_pages,
        kv_dim2,
        NUM_LAYERS,
        PAGE_SIZE,
        kv_params.num_heads,
        kv_params.head_size,
    )
    var extra_block_elems = (
        extra_total_pages
        * kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var extra_blocks_host = ctx.enqueue_create_host_buffer[kv_type](
        extra_block_elems
    )
    for i in range(extra_block_elems):
        extra_blocks_host[i] = Scalar[kv_type](0)

    var extra_lut_size = batch_size * max_extra_pages_per_batch
    var extra_lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        extra_lut_size
    )
    for i in range(extra_lut_size):
        extra_lookup_table_host[i] = UInt32(0)
    var extra_page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(extra_num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            extra_lookup_table_host[
                bi * max_extra_pages_per_batch + p
            ] = UInt32(extra_page_offset + shuffled_p)
        extra_page_offset += np_bi

    var extra_cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        extra_cache_lengths_host[i] = UInt32(extra_cache_lengths[i])

    var extra_total_k_elems = 0
    for bi in range(batch_size):
        extra_total_k_elems += extra_num_keys_list[bi] * Q_DEPTH
    var extra_k_bf16_host = ctx.enqueue_create_host_buffer[q_type](
        extra_total_k_elems
    )
    randn(
        extra_k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var extra_k_ref_host = ctx.enqueue_create_host_buffer[q_type](
        extra_total_k_elems
    )

    var ek_offset = 0
    for bi in range(batch_size):
        var enk = extra_num_keys_list[bi]
        for t in range(enk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = ek_offset + t * Q_DEPTH
            # All 576 slots as FP8 — same layout as the main cache.
            for d in range(Q_DEPTH):
                var fp8_val = extra_k_bf16_host[k_base + d].cast[kv_type]()
                extra_blocks_host[base + d] = fp8_val
                extra_k_ref_host[k_base + d] = fp8_val.cast[q_type]()
        ek_offset += enk * Q_DEPTH

    # Q
    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # Select indices for both caches; build combined reference.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[DType.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(0)
    var extra_total_indices = batch_size * max_extra_topk
    var extra_h_indices = ctx.enqueue_create_host_buffer[DType.int32](
        extra_total_indices
    )
    for i in range(extra_total_indices):
        extra_h_indices[i] = Int32(0)
    var total_combined_ref_elems = 0
    for bi in range(batch_size):
        total_combined_ref_elems += (
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        ) * Q_DEPTH
    var k_combined_ref = ctx.enqueue_create_host_buffer[q_type](
        total_combined_ref_elems
    )

    var combined_ref_offset = 0
    var k_offset_src = 0
    var ek_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var topk_bi = topk_per_batch[bi]
        var mult = _coprime_multiplier(nk)
        for i in range(topk_bi):
            var t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = k_ref_host[src_base + d]
        combined_ref_offset += topk_bi * Q_DEPTH

        var enk = extra_num_keys_list[bi]
        var extra_topk_bi = extra_topk_per_batch[bi]
        var extra_mult = _coprime_multiplier(enk)
        for i in range(extra_topk_bi):
            var t = (i * extra_mult + 1) % enk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                extra_lookup_table_host[
                    bi * max_extra_pages_per_batch + page_idx
                ]
            )
            extra_h_indices[bi * max_extra_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = ek_offset_src + t * Q_DEPTH
            var dst_base = combined_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_combined_ref[dst_base + d] = extra_k_ref_host[src_base + d]
        combined_ref_offset += extra_topk_bi * Q_DEPTH
        k_offset_src += nk * Q_DEPTH
        ek_offset_src += enk * Q_DEPTH

    var combined_num_keys_list = List[Int]()
    for bi in range(batch_size):
        combined_num_keys_list.append(
            topk_per_batch[bi] + extra_topk_per_batch[bi]
        )

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_combined_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        combined_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Device uploads.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)

    var extra_blocks_device = ctx.enqueue_create_buffer[kv_type](
        extra_block_elems
    )
    ctx.enqueue_copy(extra_blocks_device, extra_blocks_host)
    var extra_cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(extra_cache_lengths_device, extra_cache_lengths_host)
    var extra_lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](
        extra_lut_size
    )
    ctx.enqueue_copy(extra_lookup_table_device, extra_lookup_table_host)

    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)

    var d_indices_device = ctx.enqueue_create_buffer[DType.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)
    var extra_d_indices_device = ctx.enqueue_create_buffer[DType.int32](
        extra_total_indices
    )
    ctx.enqueue_copy(extra_d_indices_device, extra_h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[DType.int32](
        batch_size
    )
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])
    var topk_lengths_device = ctx.enqueue_create_buffer[DType.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    var extra_topk_lengths_host = ctx.enqueue_create_host_buffer[DType.int32](
        batch_size
    )
    for bi in range(batch_size):
        extra_topk_lengths_host[bi] = Int32(extra_topk_per_batch[bi])
    var extra_topk_lengths_device = ctx.enqueue_create_buffer[DType.int32](
        batch_size
    )
    ctx.enqueue_copy(extra_topk_lengths_device, extra_topk_lengths_host)

    ctx.synchronize()

    # Build KV collections (original + extra).
    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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
    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
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
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var extra_blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
        extra_blocks_device.unsafe_ptr(),
        RuntimeLayout[Layout.row_major[6]()].row_major(extra_block_shape),
    )
    var extra_cache_lengths_lt = LayoutTensor[DType.uint32, cl_layout](
        extra_cache_lengths_device.unsafe_ptr(),
        RuntimeLayout[cl_layout].row_major(IndexList[1](batch_size)),
    )
    var extra_lookup_table_lt = LayoutTensor[DType.uint32, lt_layout_2d](
        extra_lookup_table_device.unsafe_ptr(),
        RuntimeLayout[lt_layout_2d].row_major(
            IndexList[2](batch_size, max_extra_pages_per_batch)
        ),
    )
    var extra_kv_collection = PagedKVCacheCollection[
        kv_type, kv_params, PAGE_SIZE
    ](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
            extra_blocks_lt.ptr,
            RuntimeLayout[Layout.row_major[6]()](
                extra_blocks_lt.runtime_layout.shape.value,
                extra_blocks_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, cl_layout, ImmutAnyOrigin](
            extra_cache_lengths_lt.ptr,
            RuntimeLayout[cl_layout](
                extra_cache_lengths_lt.runtime_layout.shape.value,
                extra_cache_lengths_lt.runtime_layout.stride.value,
            ),
        ),
        LayoutTensor[DType.uint32, lt_layout_2d, ImmutAnyOrigin](
            extra_lookup_table_lt.ptr,
            RuntimeLayout[lt_layout_2d](
                extra_lookup_table_lt.runtime_layout.shape.value,
                extra_lookup_table_lt.runtime_layout.stride.value,
            ),
        ),
        UInt32(q_max_seq_len),
        UInt32(max_extra_cache_len),
    )
    var extra_kv_cache = extra_kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[Q_DEPTH]())),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    print("  Launching MLA sparse KV_FP8 (extra KV)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        lt_to_tt(scalar_args_buf_lt),
        d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=max_topk,
        topk_lengths=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
        extra_k=extra_kv_cache,
        extra_d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            extra_d_indices_device.unsafe_ptr()
        ),
        extra_indices_stride=max_extra_topk,
        extra_topk_lengths=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            extra_topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
    var total_checked = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = extra_blocks_device
    _ = extra_cache_lengths_device
    _ = extra_lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = extra_d_indices_device
    _ = topk_lengths_device
    _ = extra_topk_lengths_device
    _ = row_offsets_device


# ===-----------------------------------------------------------------------===#
# Topk-clamping sparse test (small caches with topk > actual_tokens)
# ===-----------------------------------------------------------------------===#


def run_test_sparse_kv_fp8_topk_clamping[
    q_type: DType,
    kv_type: DType,
    num_heads: Int,
](
    name: StringLiteral,
    cache_lengths: List[Int],
    topk_per_batch: List[Int],
    ctx: DeviceContext,
) raises:
    """Topk clamping with all-FP8 KV layout.

    When topk_per_batch[b] > actual_tokens[b] (= cache_length + seq_len),
    the kernel should clamp to actual_tokens. Indices beyond effective_topk
    are filled with -1 to catch OOB reads.
    """
    var batch_size = len(cache_lengths)
    comptime q_max_seq_len = 1
    comptime scale = Float32(0.125)

    var max_topk = 0
    var effective_topk_list = List[Int]()
    for bi in range(batch_size):
        if topk_per_batch[bi] > max_topk:
            max_topk = topk_per_batch[bi]
        var actual_tokens = cache_lengths[bi] + q_max_seq_len
        var eff_topk = topk_per_batch[bi]
        if eff_topk > actual_tokens:
            eff_topk = actual_tokens
        effective_topk_list.append(eff_topk)

    print(
        "test:",
        name,
        " batch_size:",
        batch_size,
        " num_heads:",
        num_heads,
    )
    for i in range(batch_size):
        print(
            "  batch",
            i,
            ": cache_len=",
            cache_lengths[i],
            " topk=",
            topk_per_batch[i],
            " actual_tokens=",
            cache_lengths[i] + q_max_seq_len,
            " effective_topk=",
            effective_topk_list[i],
        )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_list = List[Int]()
    for i in range(batch_size):
        var cl = cache_lengths[i]
        if cl > max_cache_len:
            max_cache_len = cl
        var nk = cl + q_max_seq_len
        num_keys_list.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)

    var max_num_keys = max_cache_len + q_max_seq_len
    var max_pages_per_batch = ceildiv(max_num_keys, PAGE_SIZE)

    comptime kv_params = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=KV_HEAD_SIZE, is_mla=True
    )
    comptime kv_dim2 = 1

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
    var tok_stride = kv_params.head_size

    var blocks_host = ctx.enqueue_create_host_buffer[kv_type](block_elems)
    for i in range(block_elems):
        blocks_host[i] = Scalar[kv_type](0)
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[DType.uint32](
        lut_size
    )
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for bi in range(batch_size):
        var np_bi = ceildiv(num_keys_list[bi], PAGE_SIZE)
        var mult_bi = _coprime_multiplier(np_bi)
        for p in range(np_bi):
            var shuffled_p = (p * mult_bi + 1) % np_bi
            lookup_table_host[bi * max_pages_per_batch + p] = UInt32(
                page_offset + shuffled_p
            )
        page_offset += np_bi

    var cache_lengths_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size
    )
    for i in range(batch_size):
        cache_lengths_host[i] = UInt32(cache_lengths[i])

    var page_stride_elems = (
        kv_dim2
        * NUM_LAYERS
        * PAGE_SIZE
        * kv_params.num_heads
        * kv_params.head_size
    )

    var total_k_elems = 0
    for bi in range(batch_size):
        total_k_elems += num_keys_list[bi] * Q_DEPTH
    var k_bf16_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)
    randn(
        k_bf16_host.as_span(),
        mean=0.0,
        standard_deviation=0.5,
    )

    var k_ref_host = ctx.enqueue_create_host_buffer[q_type](total_k_elems)

    var k_offset = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        for t in range(nk):
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            var base = (
                physical_page * page_stride_elems + tok_in_page * tok_stride
            )
            var k_base = k_offset + t * Q_DEPTH
            # All 576 slots as FP8.
            for d in range(Q_DEPTH):
                var fp8_val = k_bf16_host[k_base + d].cast[kv_type]()
                blocks_host[base + d] = fp8_val
                k_ref_host[k_base + d] = fp8_val.cast[q_type]()
        k_offset += nk * Q_DEPTH

    var q_size = batch_size * num_heads * Q_DEPTH
    var q_host = ctx.enqueue_create_host_buffer[q_type](q_size)
    randn(q_host.as_span(), mean=0.0, standard_deviation=0.5)

    # d_indices: first effective_topk entries valid; rest are -1 sentinel.
    var total_indices = batch_size * max_topk
    var h_indices = ctx.enqueue_create_host_buffer[DType.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(-1)

    var total_sparse_ref_elems = 0
    for bi in range(batch_size):
        total_sparse_ref_elems += effective_topk_list[bi] * Q_DEPTH
    var k_sparse_ref = ctx.enqueue_create_host_buffer[q_type](
        total_sparse_ref_elems
    )

    var sparse_ref_offset = 0
    var k_offset_src = 0
    for bi in range(batch_size):
        var nk = num_keys_list[bi]
        var eff_topk = effective_topk_list[bi]
        var mult = _coprime_multiplier(nk) if nk > 1 else 1
        for i in range(eff_topk):
            var t: Int
            if nk == 1:
                t = 0
            else:
                t = (i * mult + 1) % nk
            var page_idx = t // PAGE_SIZE
            var tok_in_page = t % PAGE_SIZE
            var physical_page = Int(
                lookup_table_host[bi * max_pages_per_batch + page_idx]
            )
            h_indices[bi * max_topk + i] = Int32(
                physical_page * PAGE_SIZE + tok_in_page
            )
            var src_base = k_offset_src + t * Q_DEPTH
            var dst_base = sparse_ref_offset + i * Q_DEPTH
            for d in range(Q_DEPTH):
                k_sparse_ref[dst_base + d] = k_ref_host[src_base + d]
        sparse_ref_offset += eff_topk * Q_DEPTH
        k_offset_src += nk * Q_DEPTH

    var sparse_num_keys_list = List[Int]()
    for bi in range(batch_size):
        sparse_num_keys_list.append(effective_topk_list[bi])

    var out_size = batch_size * num_heads * V_DEPTH
    var ref_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    host_reference_varkeys[q_type](
        q_host.unsafe_ptr(),
        k_sparse_ref.unsafe_ptr(),
        ref_host.unsafe_ptr(),
        batch_size,
        num_heads,
        sparse_num_keys_list,
        Q_DEPTH,
        V_DEPTH,
        scale,
    )

    # Uploads.
    var blocks_device = ctx.enqueue_create_buffer[kv_type](block_elems)
    ctx.enqueue_copy(blocks_device, blocks_host)
    var cache_lengths_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size
    )
    ctx.enqueue_copy(cache_lengths_device, cache_lengths_host)
    var lookup_table_device = ctx.enqueue_create_buffer[DType.uint32](lut_size)
    ctx.enqueue_copy(lookup_table_device, lookup_table_host)
    var q_device = ctx.enqueue_create_buffer[q_type](q_size)
    ctx.enqueue_copy(q_device, q_host)
    var out_device = ctx.enqueue_create_buffer[q_type](out_size)
    var d_indices_device = ctx.enqueue_create_buffer[DType.int32](total_indices)
    ctx.enqueue_copy(d_indices_device, h_indices)

    var topk_lengths_host = ctx.enqueue_create_host_buffer[DType.int32](
        batch_size
    )
    for bi in range(batch_size):
        topk_lengths_host[bi] = Int32(topk_per_batch[bi])  # UNCLAMPED
    var topk_lengths_device = ctx.enqueue_create_buffer[DType.int32](batch_size)
    ctx.enqueue_copy(topk_lengths_device, topk_lengths_host)

    ctx.synchronize()

    var blocks_lt = LayoutTensor[kv_type, Layout.row_major[6]()](
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

    var kv_collection = PagedKVCacheCollection[kv_type, kv_params, PAGE_SIZE](
        LayoutTensor[kv_type, Layout.row_major[6](), MutAnyOrigin](
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
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var kv_cache = kv_collection.get_key_cache(0)

    var q_tt = TileTensor(
        q_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[Q_DEPTH]())),
    )
    var out_tt = TileTensor(
        out_device.unsafe_ptr(),
        row_major((Idx(batch_size), Idx[num_heads](), Idx[V_DEPTH]())),
    )

    var row_offsets_host = ctx.enqueue_create_host_buffer[DType.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i)
    var row_offsets_device = ctx.enqueue_create_buffer[DType.uint32](
        batch_size + 1
    )
    ctx.enqueue_copy(row_offsets_device, row_offsets_host)
    ctx.synchronize()

    var row_offsets_tt = TileTensor(
        row_offsets_device.unsafe_ptr(),
        row_major(Idx(batch_size + 1)),
    )

    var mla_args = MLADispatchScalarArgs[
        num_heads=num_heads,
        is_fp8_kv=True,
    ](batch_size, max_cache_len, q_max_seq_len, ctx)
    var scalar_args_buf_lt = mla_args.gpu_layout_tensor()

    comptime sm_count = ctx.default_device_info.sm_count
    var dispatch_scalars = compute_mla_dispatch_scalars[
        num_heads=num_heads, is_fp8_kv=True, half_sms=sm_count // 2
    ](batch_size, max_cache_len, q_max_seq_len, sm_count)
    var num_partitions = dispatch_scalars[2]
    print(
        "  num_partitions=",
        num_partitions,
        " (split-K",
        "ACTIVE" if num_partitions > 1 else "OFF",
        ")",
    )

    var indices_stride = max_topk
    print("  Launching MLA sparse KV_FP8 (topk clamping)...")

    flare_mla_decoding[
        rank=3,
        config=MHAConfig[q_type](num_heads, Q_DEPTH),
        ragged=True,
        sparse=True,
    ](
        out_tt,
        q_tt,
        kv_cache,
        NullMask(),
        row_offsets_tt,
        scale,
        ctx,
        lt_to_tt(scalar_args_buf_lt),
        d_indices=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            d_indices_device.unsafe_ptr()
        ),
        indices_stride=indices_stride,
        topk_lengths=rebind[UnsafePointer[Int32, MutAnyOrigin]](
            topk_lengths_device.unsafe_ptr()
        ),
    )
    ctx.synchronize()

    var out_host = ctx.enqueue_create_host_buffer[q_type](out_size)
    ctx.enqueue_copy(out_host, out_device)
    ctx.synchronize()

    var rtol = Float64(1e-2)
    var atol = Float64(5e-2)
    var max_err = Float64(0)
    var total_checked = 0
    for b in range(batch_size):
        for h in range(num_heads):
            for d in range(V_DEPTH):
                var ref_val = ref_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var actual_val = out_host[
                    b * num_heads * V_DEPTH + h * V_DEPTH + d
                ].cast[DType.float64]()
                var err = abs(actual_val - ref_val)
                if err > max_err:
                    max_err = err
                total_checked += 1
                if err > 1e-1:
                    print(b, h, d, actual_val, ref_val, err)
                assert_almost_equal(actual_val, ref_val, atol=atol, rtol=rtol)

    print("  PASSED: max_err=", max_err, " checked=", total_checked)

    _ = mla_args
    _ = blocks_device
    _ = cache_lengths_device
    _ = lookup_table_device
    _ = q_device
    _ = out_device
    _ = d_indices_device
    _ = topk_lengths_device
    _ = row_offsets_device


def main() raises:
    with DeviceContext() as ctx:
        comptime if has_nvidia_gpu_accelerator() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            # =====================================================
            # Base variants: NullMask, no feature flags.
            # =====================================================

            # Minimal config: bs=1, h=16, cl=128, topk=8.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_min_b1_h16_cl128_topk8",
                1,
                128,
                ctx,
                topk=8,
            )

            # bs=1, h=16, cl=512, topk=64.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Production-ish: bs=1, h=128, cl=512, topk=64.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 128](
                "sparse_kv_fp8_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # Larger cache: bs=1, h=128, cl=2048, topk=64 (split-K likely).
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 128](
                "sparse_kv_fp8_b1_h128_cl2048_topk64",
                1,
                2048,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-batch (batch_size > 1). Exercises per-batch indexing.
            # =====================================================

            # bs=2, h=16, cl=128, topk=64.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b2_h16_cl128_topk64",
                2,
                128,
                ctx,
                topk=64,
            )

            # bs=4, h=16, cl=256, topk=64.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Multi-token query (q_max_seq_len > 1) — speculative decode.
            # =====================================================

            # q_max_seq_len=2.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl256_topk64_seq2",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=2,
            )

            # q_max_seq_len=4.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b1_h16_cl256_topk64_seq4",
                1,
                256,
                ctx,
                topk=64,
                q_max_seq_len=4,
            )

            # q_max_seq_len=8, multi-batch.
            run_test_sparse_kv_fp8[DType.bfloat16, DType.float8_e4m3fn, 16](
                "sparse_kv_fp8_b2_h16_cl256_topk64_seq8",
                2,
                256,
                ctx,
                topk=64,
                q_max_seq_len=8,
            )

            # =====================================================
            # CausalMask variant.
            # =====================================================

            # Causal, single-token.
            run_test_sparse_kv_fp8[
                DType.bfloat16, DType.float8_e4m3fn, 16, use_causal=True
            ](
                "sparse_kv_fp8_causal_b1_h16_cl256_topk64",
                1,
                256,
                ctx,
                topk=64,
            )

            # Causal + multi-token: we intentionally skip this combination.
            # For sparse MLA decode the kernel's causal mask interacts with
            # logical token positions inside the full cache, but the sparse
            # indices expose only a subset of those positions. The exact
            # mapping from (q_seq, sparse_slot) -> causal validity isn't
            # straightforward to reference on the host without duplicating
            # kernel internals; the parent test_mla_sparse.mojo also does
            # not cover causal+multi-seq directly. The seq_len=1 causal
            # case above exercises the mask wiring end-to-end.

            # =====================================================
            # Variable per-batch topk (has_variable_topk=True).
            # =====================================================

            # bs=2, variable cache + variable topk.
            var vt_cls_2: List[Int] = [256, 128]
            var vt_topk_2: List[Int] = [64, 32]
            run_test_sparse_kv_fp8_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_variable_topk_b2_h16",
                vt_cls_2,
                vt_topk_2,
                ctx,
            )

            # bs=4, variable cache + variable topk.
            var vt_cls_4: List[Int] = [256, 384, 128, 512]
            var vt_topk_4: List[Int] = [64, 128, 32, 64]
            run_test_sparse_kv_fp8_variable_topk[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_variable_topk_b4_h16",
                vt_cls_4,
                vt_topk_4,
                ctx,
            )

            # =====================================================
            # Attention sink (has_attn_sink=True).
            # =====================================================

            # attn_sink, small cache, 16 heads.
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_attn_sink_b1_h16_cl128_topk64",
                1,
                128,
                ctx,
                topk=64,
            )

            # attn_sink, production-ish (128 heads).
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 128
            ](
                "sparse_kv_fp8_attn_sink_b1_h128_cl512_topk64",
                1,
                512,
                ctx,
                topk=64,
            )

            # attn_sink, multi-batch.
            run_test_sparse_kv_fp8_attn_sink[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_attn_sink_b4_h16_cl256_topk64",
                4,
                256,
                ctx,
                topk=64,
            )

            # =====================================================
            # Extra KV (has_extra_kv=True).
            # =====================================================

            # bs=1, 16 heads, 64 extra tokens.
            var ek_cls_1: List[Int] = [256]
            var ek_topk_1: List[Int] = [64]
            var ek_ecls_1: List[Int] = [64]
            var ek_etopk_1: List[Int] = [64]
            run_test_sparse_kv_fp8_extra_kv[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_extra_kv_b1_h16_topk64_extra64",
                ek_cls_1,
                ek_topk_1,
                ek_ecls_1,
                ek_etopk_1,
                ctx,
            )

            # bs=2, 16 heads, variable extra.
            var ek_cls_2: List[Int] = [256, 384]
            var ek_topk_2: List[Int] = [64, 64]
            var ek_ecls_2: List[Int] = [64, 128]
            var ek_etopk_2: List[Int] = [64, 64]
            run_test_sparse_kv_fp8_extra_kv[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_extra_kv_b2_h16_variable",
                ek_cls_2,
                ek_topk_2,
                ek_ecls_2,
                ek_etopk_2,
                ctx,
            )

            # =====================================================
            # Topk clamping (small caches).
            # =====================================================

            # First decode: cache_length=0, actual=1, topk=64 -> clamp to 1.
            var tc_cls_1: List[Int] = [0]
            var tc_topk_1: List[Int] = [64]
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_first_exec_b1_h16",
                tc_cls_1,
                tc_topk_1,
                ctx,
            )

            # Small cache: cache_length=5, actual=6, topk=64 -> clamp to 6.
            var tc_cls_2: List[Int] = [5]
            var tc_topk_2: List[Int] = [64]
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_small_cache_b1_h16",
                tc_cls_2,
                tc_topk_2,
                ctx,
            )

            # Mixed batch: cl=0 and cl=256.
            var tc_cls_3: List[Int] = [0, 256]
            var tc_topk_3: List[Int] = [64, 64]
            run_test_sparse_kv_fp8_topk_clamping[
                DType.bfloat16, DType.float8_e4m3fn, 16
            ](
                "sparse_kv_fp8_topk_clamp_mixed_b2_h16",
                tc_cls_3,
                tc_topk_3,
                ctx,
            )
