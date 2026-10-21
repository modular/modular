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

"""Exact-oracle test for `mla_prefill_sparse_qkv_fp8` on a packed NVFP4 cache.

Target family: Blackwell sm_100.

The cache is `uint8` and holds each MLA row as a 512-wide e2m1 latent (two per
byte, element 0 in the low nibble) with one `float8_e4m3fn` scale per 16
elements, followed by the 64 rope dims as raw FP8 bytes. The kernel dequantizes
the latent to FP8 in shared memory and copies the rope bytes, so the oracle is
the same kernel on an FP8 cache holding the dequantized latent `e4m3(e2m1 *
sf)`, rounded once, and the same rope bytes. Both runs feed the MMA identical
bytes, so their outputs must match bit for bit.

The scales are paged apart from the values, as the Jenga KV manager does: the
NVFP4 cache gets its own page table, a permutation of the values', and the
sparse index rows carry each selection's scale-pool row after its value row
(`[value rows | scale rows]`, as `PagedKVScaleRemap` writes them). A kernel
that took a scale row from its value row would read other tokens' scales.

Each case also runs a canary: the scales of one selected row of the packed cache
are raised after the reference is built, and the outputs must then differ, so a
passing case cannot be one that never reads the scales.
"""

from std.math import ceildiv
from std.memory import bitcast
from std.random import randn, seed
from std.testing import assert_equal, assert_true

from max.gpu import *
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Idx, TileTensor, row_major
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_qkv_fp8 import (
    mla_prefill_sparse_qkv_fp8,
)
from nn.attention.gpu.nvidia.sm100.mla_prefill_sparse_utils import (
    MLASparseConfig,
)

comptime QK_DEPTH = 576  # 512 latent + 64 rope
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1
# `quantization_granularity` counts packed bytes (16 elements = 8 bytes).
comptime QUANT_GRANULARITY = 8

comptime E2M1_MAGNITUDES = SIMD[DType.float32, 8](
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
)


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
    for m in [3, 5, 7, 11, 13, 17, 19, 23]:
        if _gcd(m, n) == 1:
            return m
    return 1


struct _Lcg:
    """A deterministic generator so a failing case reproduces."""

    var state: UInt64

    def __init__(out self, seed: UInt64):
        self.state = seed

    def next(mut self) -> UInt32:
        self.state = self.state * 6364136223846793005 + 1442695040888963407
        return UInt32(self.state >> 33)


def _scale_byte(mut rng: _Lcg) -> UInt8:
    """A random E4M3 scale byte: 2^-5 .. 2^1 times 1 .. 1.875, or zero.

    The range keeps `6 * sf` far below the E4M3 maximum so no product saturates.
    """
    var r = rng.next()
    if r % 16 == 0:
        return UInt8(0)
    var exponent = 2 + Int((r >> 4) % 7)
    var mantissa = Int((r >> 8) % 8)
    return UInt8(exponent * 8 + mantissa)


def _e4m3_value(byte: UInt8) -> Float32:
    return bitcast[DType.float8_e4m3fn, 1](SIMD[DType.uint8, 1](byte)).cast[
        DType.float32
    ]()[0]


def _dequant_e4m3_byte(code: Int, sf_byte: UInt8) -> UInt8:
    """`e4m3(e2m1(code) * sf)` as the byte the FP8 cache stores."""
    var mag = E2M1_MAGNITUDES[code & 7]
    var rounded = bitcast[DType.uint8, 1](
        SIMD[DType.float32, 1](mag * _e4m3_value(sf_byte)).cast[
            DType.float8_e4m3fn
        ]()
    )[0]
    return rounded | (UInt8(0x80) if (code & 8) != 0 else UInt8(0))


def _prefill[
    num_heads: Int,
    row_depth: Int,
    indices_stride: Int,
    cache_t: KVCacheT,
](
    cache: cache_t,
    out_tt: TileTensor[mut=True, address_space=.GENERIC, ...],
    q_tt: TileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    indices_tt: TileTensor[.uint32, address_space=.GENERIC, ...],
    topk_lengths_tt: TileTensor[.uint32, address_space=.GENERIC, ...],
    attn_sink_ptr: Optional[ImmPointer[Float32, ImmutAnyOrigin]],
    ctx: DeviceContext,
) raises:
    # head<=64 runs the single-CTA tile, head=128 the 2-CTA tile; the same
    # tiling the production dispatch picks.
    comptime cta_group = 2 if num_heads == 128 else 1
    comptime b_topk = 128 if cta_group == 2 else 64
    comptime num_mbars = 2 if cta_group == 2 else 4
    comptime config = MLASparseConfig[
        DType.bfloat16,
        b_topk_=b_topk,
        num_mbars_=num_mbars,
        cta_group_=cta_group,
    ](
        num_q_heads=num_heads,
        num_kv_heads=1,
        qk_depth=QK_DEPTH,
        input_qk_depth=row_depth,
        v_depth=V_DEPTH,
        indices_stride=indices_stride,
        group=num_heads,
    )
    mla_prefill_sparse_qkv_fp8[
        config=config,
        group=num_heads,
        q_depth=row_depth,
        scale_block_size=0,
    ](
        out_tt,
        q_tt,
        cache,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        Float32(1.0) / 13.856406,
        Int32(indices_stride),
        ctx,
    )


def run_nvfp4_vs_fp8[
    num_heads: Int,
    row_depth: Int = QK_DEPTH,
    topk: Int = 2048,
    perf_iters: Int = 0,
](
    name: StringLiteral,
    batch_size: Int,
    seq_len: Int,
    num_kv_tokens: Int,
    ctx: DeviceContext,
    *,
    valid_topk: Int = -1,
    with_sink: Bool = False,
    unit_scales: Bool = False,
) raises:
    var n_valid = topk if valid_topk < 0 else valid_topk
    print(
        "test:",
        name,
        " batch:",
        batch_size,
        " seq:",
        seq_len,
        " heads:",
        num_heads,
        " row_depth:",
        row_depth,
        " kv_tokens:",
        num_kv_tokens,
        " topk:",
        topk,
        " valid:",
        n_valid,
    )

    comptime rope_depth = row_depth - V_DEPTH
    comptime packed_row = V_DEPTH // 2 + rope_depth
    comptime sf_per_row = V_DEPTH // 16
    var total_q_tokens = batch_size * seq_len
    var pages_per_batch = ceildiv(num_kv_tokens, PAGE_SIZE)
    var total_pages = batch_size * pages_per_batch

    comptime kv_params_fp8 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=row_depth, is_mla=True
    )
    comptime kv_params_fp4 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS,
        head_size=packed_row,
        is_mla=True,
        fp8_tail_size=rope_depth,
    )

    # Shuffled page table (exercises scatter), shared by both caches.
    var lut_size = batch_size * pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    var page_offset = 0
    for b in range(batch_size):
        var mult = _coprime_multiplier(pages_per_batch)
        for p in range(pages_per_batch):
            lookup_table_host[b * pages_per_batch + p] = UInt32(
                page_offset + (p * mult + 1) % pages_per_batch
            )
        page_offset += pages_per_batch

    # The scales' own page table: a permutation of the value block ids, as a
    # Jenga scales leaf has block ids of its own.
    var scale_mult = _coprime_multiplier(total_pages)
    var scales_lookup_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        scales_lookup_host[i] = UInt32(
            (Int(lookup_table_host[i]) * scale_mult + 1) % total_pages
        )

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(num_kv_tokens)

    # Random packed cache and its exact FP8 dequantization.
    var fp8_elems = total_pages * PAGE_SIZE * row_depth
    var fp4_elems = total_pages * PAGE_SIZE * packed_row
    var sf_elems = total_pages * PAGE_SIZE * sf_per_row
    var fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](fp8_elems)
    var fp4_host = ctx.enqueue_create_host_buffer[.uint8](fp4_elems)
    var sf_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](sf_elems)
    for i in range(fp8_elems):
        fp8_host[i] = bitcast[DType.float8_e4m3fn, 1](SIMD[DType.uint8, 1](0))[
            0
        ]
    for i in range(fp4_elems):
        fp4_host[i] = UInt8(0)
    for i in range(sf_elems):
        sf_host[i] = bitcast[DType.float8_e4m3fn, 1](SIMD[DType.uint8, 1](0))[0]

    var rng = _Lcg(
        UInt64(0x9E3779B97F4A7C15) ^ UInt64(total_pages * 131 + topk)
    )
    for b in range(batch_size):
        for t in range(num_kv_tokens):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var block_id = Int(
                lookup_table_host[b * pages_per_batch + page_idx]
            )
            var row = block_id * PAGE_SIZE + tok_in_page
            var scale_row = (
                Int(scales_lookup_host[b * pages_per_batch + page_idx])
                * PAGE_SIZE
                + tok_in_page
            )
            for g in range(sf_per_row):
                var sf_byte = _scale_byte(rng)
                if unit_scales:
                    sf_byte = UInt8(0x38)
                sf_host[scale_row * sf_per_row + g] = bitcast[
                    DType.float8_e4m3fn, 1
                ](SIMD[DType.uint8, 1](sf_byte))[0]
                for j in range(16):
                    var code = Int(rng.next() & 15)
                    var e = g * 16 + j
                    var byte_idx = row * packed_row + e // 2
                    var cur = fp4_host[byte_idx]
                    if e % 2 == 0:
                        fp4_host[byte_idx] = cur | UInt8(code)
                    else:
                        fp4_host[byte_idx] = cur | (UInt8(code) << 4)
                    fp8_host[row * row_depth + e] = bitcast[
                        DType.float8_e4m3fn, 1
                    ](SIMD[DType.uint8, 1](_dequant_e4m3_byte(code, sf_byte)))[
                        0
                    ]
            # The rope dims are FP8 bytes in both caches.
            for e in range(rope_depth):
                var rope_byte = _scale_byte(rng) | UInt8(
                    0x80 if rng.next() & 1 == 1 else 0
                )
                fp4_host[row * packed_row + V_DEPTH // 2 + e] = rope_byte
                fp8_host[row * row_depth + V_DEPTH + e] = bitcast[
                    DType.float8_e4m3fn, 1
                ](SIMD[DType.uint8, 1](rope_byte))[0]

    # Q (FP8) shared by both runs.
    var q_size = total_q_tokens * num_heads * row_depth
    var q_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_scratch[i].cast[.float8_e4m3fn]()

    # Per-query selection: rotated coprime walk over the batch's tokens. Slots
    # past `n_valid` are the -1 sentinel so a read through them would show.
    # The FP8 reference reads `[tokens, topk]` rows; the NVFP4 cache reads
    # `[tokens, 2 * topk]`: the same rows, then each slot's scale-pool row.
    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.uint32](total_indices)
    var total_wide = 2 * total_indices
    var h_wide = ctx.enqueue_create_host_buffer[.uint32](total_wide)
    for i in range(total_wide):
        h_wide[i] = UInt32(0xFFFFFFFF)
    var sel_mult = _coprime_multiplier(num_kv_tokens)
    var first_selected_scale_row = -1
    for b in range(batch_size):
        for s in range(seq_len):
            var bs = b * seq_len + s
            var rotation = s % num_kv_tokens
            for i in range(topk):
                if i < n_valid:
                    var t = ((rotation + i) * sel_mult + 1) % num_kv_tokens
                    var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                    var block_id = Int(
                        lookup_table_host[b * pages_per_batch + page_idx]
                    )
                    var row = block_id * PAGE_SIZE + tok_in_page
                    var scale_row = (
                        Int(scales_lookup_host[b * pages_per_batch + page_idx])
                        * PAGE_SIZE
                        + tok_in_page
                    )
                    h_indices[bs * topk + i] = UInt32(row)
                    h_wide[bs * 2 * topk + i] = UInt32(row)
                    h_wide[bs * 2 * topk + topk + i] = UInt32(scale_row)
                    if bs == 0 and i == 0:
                        first_selected_scale_row = scale_row
                else:
                    h_indices[bs * topk + i] = UInt32(0xFFFFFFFF)

    var h_topk_lengths = ctx.enqueue_create_host_buffer[.uint32](total_q_tokens)
    for i in range(total_q_tokens):
        h_topk_lengths[i] = UInt32(n_valid)

    var sink_host = ctx.enqueue_create_host_buffer[.float32](num_heads)
    for h in range(num_heads):
        sink_host[h] = Float32(h % 5) - 2.0

    # Device buffers.
    var fp8_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](fp8_elems)
    var fp4_dev = ctx.enqueue_create_buffer[.uint8](fp4_elems)
    var sf_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](sf_elems)
    var sf_canary_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](sf_elems)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var q_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    var scales_lookup_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var indices_dev = ctx.enqueue_create_buffer[.uint32](total_indices)
    var wide_dev = ctx.enqueue_create_buffer[.uint32](total_wide)
    var topk_lengths_dev = ctx.enqueue_create_buffer[.uint32](total_q_tokens)
    var sink_dev = ctx.enqueue_create_buffer[.float32](num_heads)
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_fp8_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_fp4_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_canary_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)

    ctx.enqueue_copy(fp8_dev, fp8_host)
    ctx.enqueue_copy(fp4_dev, fp4_host)
    ctx.enqueue_copy(sf_dev, sf_host)
    ctx.enqueue_copy(sf_canary_dev, sf_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_dev, lookup_table_host)
    ctx.enqueue_copy(scales_lookup_dev, scales_lookup_host)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(indices_dev, h_indices)
    ctx.enqueue_copy(wide_dev, h_wide)
    ctx.enqueue_copy(topk_lengths_dev, h_topk_lengths)
    ctx.enqueue_copy(sink_dev, sink_host)
    ctx.synchronize()
    # The canary cache differs from the real one in every scale of one selected
    # row, raised enough that the row dominates the softmax.
    with sf_canary_dev.map_to_host() as mapped:
        for g in range(sf_per_row):
            mapped[first_selected_scale_row * sf_per_row + g] = bitcast[
                DType.float8_e4m3fn, 1
            ](SIMD[DType.uint8, 1](UInt8(0x58)))[0]
    ctx.synchronize()

    # Collections.
    comptime kv_dim2 = 1
    var fp8_tt = TileTensor(
        fp8_dev,
        row_major(
            Int64(total_pages),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[row_depth],
        ),
    )
    var fp4_tt = TileTensor(
        fp4_dev,
        row_major(
            Int64(total_pages),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[packed_row],
        ),
    )
    var sf_tt = TileTensor(
        sf_dev,
        row_major(
            Int64(total_pages),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[sf_per_row],
        ),
    )
    var sf_canary_tt = TileTensor(
        sf_canary_dev,
        row_major(
            Int64(total_pages),
            Idx[kv_dim2],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[sf_per_row],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_dev, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_dev,
        row_major(Int64(batch_size), Int64(pages_per_batch)),
    )
    var scales_lookup_tt = TileTensor(
        scales_lookup_dev,
        row_major(Int64(batch_size), Int64(pages_per_batch)),
    )

    comptime CollectionFP8 = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        kv_params_fp8,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
    ]
    comptime CollectionFP4 = PagedKVCacheCollection[
        DType.uint8,
        kv_params_fp4,
        PAGE_SIZE,
        MutAnyOrigin,
        ImmutAnyOrigin,
        ImmutAnyOrigin,
        MutAnyOrigin,
        scale_dtype_=DType.float8_e4m3fn,
        quantization_granularity_=QUANT_GRANULARITY,
    ]
    var coll_fp8 = CollectionFP8(
        rebind[CollectionFP8.blocks_tt_type](fp8_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(num_kv_tokens),
    )
    var coll_fp4 = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](fp4_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(num_kv_tokens),
        rebind[CollectionFP4.scales_tt_type](sf_tt.as_unsafe_any_origin()),
        scales_lookup_tt.as_imm().as_unsafe_any_origin(),
    )
    var coll_canary = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](fp4_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(seq_len),
        UInt32(num_kv_tokens),
        rebind[CollectionFP4.scales_tt_type](
            sf_canary_tt.as_unsafe_any_origin()
        ),
        scales_lookup_tt.as_imm().as_unsafe_any_origin(),
    )
    var kv_fp8 = coll_fp8.get_key_cache(0)
    var kv_fp4 = coll_fp4.get_key_cache(0)
    var kv_canary = coll_canary.get_key_cache(0)

    var q_tt = TileTensor(
        q_dev, row_major((total_q_tokens, Idx[num_heads], Idx[row_depth]))
    )
    var out_fp8_tt = TileTensor(
        out_fp8_dev, row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH]))
    )
    var out_fp4_tt = TileTensor(
        out_fp4_dev, row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH]))
    )
    var out_canary_tt = TileTensor(
        out_canary_dev,
        row_major((total_q_tokens, Idx[num_heads], Idx[V_DEPTH])),
    )
    var indices_tt = TileTensor(indices_dev, row_major(total_indices))
    var wide_tt = TileTensor(wide_dev, row_major(total_wide))
    var topk_lengths_tt = TileTensor(
        topk_lengths_dev, row_major(total_q_tokens)
    )

    var attn_sink_ptr = Optional[ImmPointer[Float32, ImmutAnyOrigin]](None)
    if with_sink:
        attn_sink_ptr = Optional[ImmPointer[Float32, ImmutAnyOrigin]](
            sink_dev.unsafe_ptr()
            .unsafe_bitcast[Float32]()
            .as_unsafe_any_origin()
        )

    _prefill[num_heads, row_depth, topk](
        kv_fp8,
        out_fp8_tt,
        q_tt,
        indices_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        ctx,
    )
    _prefill[num_heads, row_depth, 2 * topk](
        kv_fp4,
        out_fp4_tt,
        q_tt,
        wide_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        ctx,
    )
    _prefill[num_heads, row_depth, 2 * topk](
        kv_canary,
        out_canary_tt,
        q_tt,
        wide_tt,
        topk_lengths_tt,
        attn_sink_ptr,
        ctx,
    )
    ctx.synchronize()

    var out_fp8 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var out_fp4 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var out_canary = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_fp8, out_fp8_dev)
    ctx.enqueue_copy(out_fp4, out_fp4_dev)
    ctx.enqueue_copy(out_canary, out_canary_dev)
    ctx.synchronize()

    var mismatches = 0
    var canary_mismatches = 0
    var nonzero = 0
    var nonfinite = 0
    for i in range(out_size):
        var a = bitcast[DType.uint16, 1](SIMD[DType.bfloat16, 1](out_fp8[i]))[0]
        var b = bitcast[DType.uint16, 1](SIMD[DType.bfloat16, 1](out_fp4[i]))[0]
        var c = bitcast[DType.uint16, 1](
            SIMD[DType.bfloat16, 1](out_canary[i])
        )[0]
        if a != b:
            if mismatches < 4:
                print(
                    "  mismatch at",
                    i,
                    ": fp8 ref",
                    out_fp8[i],
                    " nvfp4",
                    out_fp4[i],
                )
            mismatches += 1
        if a != c:
            canary_mismatches += 1
        if abs(out_fp8[i].cast[.float32]()) > 1e-6:
            nonzero += 1
        var v = out_fp4[i].cast[.float32]()
        if v != v or abs(v) > 1e30:
            nonfinite += 1

    print(
        "  mismatches:",
        mismatches,
        "/",
        out_size,
        " canary_mismatches:",
        canary_mismatches,
        " nonzero:",
        nonzero,
    )
    assert_equal(nonfinite, 0)
    assert_true(nonzero > out_size // 2, "reference output is mostly zero")
    assert_equal(mismatches, 0)
    assert_true(
        canary_mismatches > 0, "a changed scale did not change the output"
    )

    comptime if perf_iters > 0:

        def _launch_fp8(ctx: DeviceContext) raises {imm}:
            _prefill[num_heads, row_depth, topk](
                kv_fp8,
                out_fp8_tt,
                q_tt,
                indices_tt,
                topk_lengths_tt,
                attn_sink_ptr,
                ctx,
            )

        def _launch_fp4(ctx: DeviceContext) raises {imm}:
            _prefill[num_heads, row_depth, 2 * topk](
                kv_fp4,
                out_fp4_tt,
                q_tt,
                wide_tt,
                topk_lengths_tt,
                attn_sink_ptr,
                ctx,
            )

        for _ in range(5):
            _launch_fp8(ctx)
            _launch_fp4(ctx)
        ctx.synchronize()
        var fp8_us = (
            Float64(ctx.execution_time(_launch_fp8, perf_iters))
            / Float64(perf_iters)
            / 1000.0
        )
        var fp4_us = (
            Float64(ctx.execution_time(_launch_fp4, perf_iters))
            / Float64(perf_iters)
            / 1000.0
        )
        print("PERF", name, " fp8_us:", fp8_us, " nvfp4_us:", fp4_us)

    _ = fp8_dev
    _ = fp4_dev
    _ = sf_dev
    _ = sf_canary_dev
    _ = cache_lengths_dev
    _ = lookup_table_dev
    _ = scales_lookup_dev
    _ = q_dev
    _ = indices_dev
    _ = wide_dev
    _ = topk_lengths_dev
    _ = sink_dev
    _ = out_fp8_dev
    _ = out_fp4_dev
    _ = out_canary_dev


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            # GLM 5.3: 64 heads / TP8 = 8 per GPU, topk 2048, page 128.
            run_nvfp4_vs_fp8[8](
                "h8_unit_scales", 1, 8, 4096, ctx, unit_scales=True
            )
            run_nvfp4_vs_fp8[8]("h8_b2", 2, 16, 4096, ctx)
            run_nvfp4_vs_fp8[8]("h8_b3_partial_page", 3, 9, 3000, ctx)
            run_nvfp4_vs_fp8[8](
                "h8_masked_partial_tile", 2, 8, 4096, ctx, valid_topk=1000
            )
            run_nvfp4_vs_fp8[8, topk=128](
                "h8_tiny", 2, 4, 100, ctx, valid_topk=5
            )
            run_nvfp4_vs_fp8[8]("h8_sink", 2, 8, 4096, ctx, with_sink=True)
            run_nvfp4_vs_fp8[16]("h16_b2", 2, 8, 3000, ctx)
            run_nvfp4_vs_fp8[32]("h32_b2", 2, 8, 4096, ctx, valid_topk=1500)
            run_nvfp4_vs_fp8[64]("h64_b2", 2, 8, 4096, ctx)
            run_nvfp4_vs_fp8[64](
                "h64_sink_masked",
                2,
                8,
                3000,
                ctx,
                valid_topk=1900,
                with_sink=True,
            )

            # head=128 runs the 2-CTA tile: K is CTA-split, V is a per-CTA
            # depth slice gathered separately.
            run_nvfp4_vs_fp8[128]("h128_b2", 2, 8, 4096, ctx)
            run_nvfp4_vs_fp8[128](
                "h128_masked_partial_tile", 2, 8, 3000, ctx, valid_topk=1900
            )
            run_nvfp4_vs_fp8[128, topk=128](
                "h128_tiny", 2, 4, 100, ctx, valid_topk=5
            )
            run_nvfp4_vs_fp8[128]("h128_sink", 2, 8, 4096, ctx, with_sink=True)

            # Timing only (prints PERF lines): FP8 cache vs packed NVFP4.
            run_nvfp4_vs_fp8[8, perf_iters=20]("perf_h8", 1, 2048, 8192, ctx)
            run_nvfp4_vs_fp8[16, perf_iters=20]("perf_h16", 1, 1024, 8192, ctx)
            run_nvfp4_vs_fp8[64, perf_iters=20]("perf_h64", 1, 512, 8192, ctx)
            run_nvfp4_vs_fp8[128, perf_iters=20]("perf_h128", 1, 512, 8192, ctx)
            print("nvfp4 sparse prefill: all cases match the FP8 oracle")
