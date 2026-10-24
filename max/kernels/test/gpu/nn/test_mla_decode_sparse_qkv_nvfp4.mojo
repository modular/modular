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

"""Exact-oracle test of MLA_SM100_Decode_Sparse_QKV_FP8 on a packed NVFP4 cache.

Target family: Blackwell sm_100.

The KV operand is a `uint8` cache holding the 576-wide MLA row as a 512-wide
e2m1 latent (two per byte, element 0 in the low nibble, 256 bytes) with one
`float8_e4m3fn` scale per 16 elements (32 per row), followed by the 64 rope
dims as raw FP8 bytes (320 bytes per row). The kernel dequantizes the latent to
FP8 in shared memory and copies the rope bytes, so the oracle is the same
kernel on an FP8 cache that holds the dequantized latent, `e4m3(e2m1 * sf)`
rounded once, and the same rope bytes. Both runs feed the MMA identical bytes,
so their outputs must match bit for bit.

The scales are paged apart from the values, as the Jenga KV manager does: the
NVFP4 cache gets its own page table, a permutation of the values', and the
sparse index rows carry each selection's scale-pool row after its value row
(`[value rows | scale rows]`, as `PagedKVScaleRemap` writes them). A kernel
that took a scale row from its value row would read other tokens' scales.

Each case also runs a canary: the scales of one selected row in the packed
cache are raised after the reference is built, and the outputs must then
differ, so a passing case cannot be one that never reads the scales.
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
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_utils import MHAConfig
from nn.attention.gpu.mla import flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    MLADispatchScalarArgs,
)
from std.utils.index import IndexList

comptime ROW_DEPTH = 576  # 512 latent + 64 rope
comptime V_DEPTH = 512
comptime PAGE_SIZE = 128
comptime NUM_LAYERS = 1
comptime KV_NUM_HEADS = 1
comptime ROPE_DEPTH = ROW_DEPTH - V_DEPTH
# The packed latent, then the FP8 rope bytes.
comptime PACKED_ROW = V_DEPTH // 2 + ROPE_DEPTH  # 320 bytes per row
comptime SF_PER_ROW = V_DEPTH // 16  # 32 scales per row
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


def _decode[
    num_heads: Int, use_causal: Bool, cache_t: KVCacheT
](
    cache: cache_t,
    out_tt: TileTensor[mut=True, address_space=.GENERIC, ...],
    q_tt: TileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    row_offsets_tt: TileTensor[.uint32, address_space=.GENERIC, ...],
    scalar_args_buf_tt: TileTensor[.int64, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    logical: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    ctx: DeviceContext,
) raises:
    comptime scale = Float32(0.125)
    comptime if use_causal:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, ROW_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=indices,
            indices_stride=indices_stride,
            logical_indices=logical,
        )
    else:
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, ROW_DEPTH),
            ragged=True,
            sparse=True,
        ](
            out_tt,
            q_tt,
            cache,
            NullMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_args_buf_tt,
            d_indices=indices,
            indices_stride=indices_stride,
        )


def run_nvfp4_vs_fp8[
    num_heads: Int,
    use_causal: Bool = False,
    unit_scales: Bool = False,
    perf_iters: Int = 0,
](
    name: StringLiteral,
    cache_lens: List[Int],
    topk: Int,
    ctx: DeviceContext,
    q_max_seq_len: Int = 1,
) raises:
    var batch_size = len(cache_lens)
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)
    print(
        "test:",
        name,
        " batch:",
        batch_size,
        " heads:",
        num_heads,
        " topk:",
        topk,
        " q_len:",
        q_max_seq_len,
        " causal:",
        use_causal,
    )

    var max_cache_len = 0
    var total_pages = 0
    var num_keys_of = List[Int]()
    for b in range(batch_size):
        max_cache_len = max(max_cache_len, cache_lens[b])
        var nk = cache_lens[b] + q_max_seq_len
        num_keys_of.append(nk)
        total_pages += ceildiv(nk, PAGE_SIZE)
    var max_pages_per_batch = ceildiv(max_cache_len + q_max_seq_len, PAGE_SIZE)

    comptime kv_params_fp8 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS, head_size=ROW_DEPTH, is_mla=True
    )
    comptime kv_params_fp4 = KVCacheStaticParams(
        num_heads=KV_NUM_HEADS,
        head_size=PACKED_ROW,
        is_mla=True,
        fp8_tail_size=ROPE_DEPTH,
    )

    # Shuffled page table (exercises scatter), shared by both caches.
    var lut_size = batch_size * max_pages_per_batch
    var lookup_table_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        lookup_table_host[i] = UInt32(0)
    var page_offset = 0
    for b in range(batch_size):
        var np = ceildiv(num_keys_of[b], PAGE_SIZE)
        var mult = _coprime_multiplier(np)
        for p in range(np):
            lookup_table_host[b * max_pages_per_batch + p] = UInt32(
                page_offset + (p * mult + 1) % np
            )
        page_offset += np

    # The scales' own page table: a permutation of the value block ids, as a
    # Jenga scales leaf has block ids of its own.
    var scale_mult = _coprime_multiplier(total_pages)
    var scales_lookup_host = ctx.enqueue_create_host_buffer[.uint32](lut_size)
    for i in range(lut_size):
        scales_lookup_host[i] = UInt32(0)
    for b in range(batch_size):
        for p in range(ceildiv(num_keys_of[b], PAGE_SIZE)):
            scales_lookup_host[b * max_pages_per_batch + p] = UInt32(
                (
                    Int(lookup_table_host[b * max_pages_per_batch + p])
                    * scale_mult
                    + 1
                )
                % total_pages
            )

    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    for b in range(batch_size):
        cache_lengths_host[b] = UInt32(cache_lens[b])

    # Random packed cache and its exact FP8 dequantization.
    var fp8_elems = total_pages * PAGE_SIZE * ROW_DEPTH
    var fp4_elems = total_pages * PAGE_SIZE * PACKED_ROW
    var sf_elems = total_pages * PAGE_SIZE * SF_PER_ROW
    var fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](fp8_elems)
    var fp4_host = ctx.enqueue_create_host_buffer[.uint8](fp4_elems)
    var sf_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](sf_elems)
    for i in range(fp8_elems):
        fp8_host[i] = bitcast[DType.float8_e4m3fn, 1](SIMD[DType.uint8, 1](0))[
            0
        ]
    for i in range(fp4_elems):
        fp4_host[i] = UInt8(0)

    var rng = _Lcg(
        UInt64(0x9E3779B97F4A7C15) ^ UInt64(total_pages * 131 + topk)
    )
    for b in range(batch_size):
        for t in range(num_keys_of[b]):
            var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
            var block_id = Int(
                lookup_table_host[b * max_pages_per_batch + page_idx]
            )
            var row = block_id * PAGE_SIZE + tok_in_page
            var scale_row = (
                Int(scales_lookup_host[b * max_pages_per_batch + page_idx])
                * PAGE_SIZE
                + tok_in_page
            )
            for g in range(SF_PER_ROW):
                var sf_byte = _scale_byte(rng)
                comptime if unit_scales:
                    sf_byte = UInt8(0x38)
                sf_host[scale_row * SF_PER_ROW + g] = bitcast[
                    DType.float8_e4m3fn, 1
                ](SIMD[DType.uint8, 1](sf_byte))[0]
                for j in range(16):
                    var code = Int(rng.next() & 15)
                    var e = g * 16 + j
                    var byte_idx = row * PACKED_ROW + e // 2
                    var cur = fp4_host[byte_idx]
                    if e % 2 == 0:
                        fp4_host[byte_idx] = cur | UInt8(code)
                    else:
                        fp4_host[byte_idx] = cur | (UInt8(code) << 4)
                    fp8_host[row * ROW_DEPTH + e] = bitcast[
                        DType.float8_e4m3fn, 1
                    ](SIMD[DType.uint8, 1](_dequant_e4m3_byte(code, sf_byte)))[
                        0
                    ]
            # The rope dims are FP8 bytes in both caches.
            for e in range(ROPE_DEPTH):
                var rope_byte = _scale_byte(rng) | UInt8(
                    0x80 if rng.next() & 1 == 1 else 0
                )
                fp4_host[row * PACKED_ROW + V_DEPTH // 2 + e] = rope_byte
                fp8_host[row * ROW_DEPTH + V_DEPTH + e] = bitcast[
                    DType.float8_e4m3fn, 1
                ](SIMD[DType.uint8, 1](rope_byte))[0]

    # Q (FP8) and the sparse selection, shared by both runs.
    var q_size = total_q_tokens * num_heads * ROW_DEPTH
    var q_scratch = ctx.enqueue_create_host_buffer[.bfloat16](q_size)
    randn(q_scratch.as_span(), mean=0.0, standard_deviation=0.5)
    var q_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](q_size)
    for i in range(q_size):
        q_host[i] = q_scratch[i].cast[.float8_e4m3fn]()

    # Slots past the live key count stay -1 so a read through them would show.
    # The FP8 reference reads `[tokens, topk]` rows; the NVFP4 cache reads
    # `[tokens, 2 * topk]`: the same rows, then each slot's scale-pool row.
    var total_indices = total_q_tokens * topk
    var h_indices = ctx.enqueue_create_host_buffer[.int32](total_indices)
    var h_logical = ctx.enqueue_create_host_buffer[.int32](total_indices)
    for i in range(total_indices):
        h_indices[i] = Int32(-1)
        h_logical[i] = Int32(-1)
    var total_wide = 2 * total_indices
    var h_wide = ctx.enqueue_create_host_buffer[.int32](total_wide)
    var h_wide_logical = ctx.enqueue_create_host_buffer[.int32](total_wide)
    for i in range(total_wide):
        h_wide[i] = Int32(-1)
        h_wide_logical[i] = Int32(-1)
    var first_selected_scale_row = -1
    for b in range(batch_size):
        var nk = num_keys_of[b]
        var n_valid = min(topk, nk)
        var mult = _coprime_multiplier(nk)
        for s in range(q_max_seq_len):
            var g = b * q_max_seq_len + s
            for i in range(n_valid):
                var t = (i * mult + 1 + s) % nk
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[b * max_pages_per_batch + page_idx]
                )
                var scale_row = (
                    Int(scales_lookup_host[b * max_pages_per_batch + page_idx])
                    * PAGE_SIZE
                    + tok_in_page
                )
                h_indices[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
                h_logical[g * topk + i] = Int32(t)
                h_wide[g * 2 * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
                h_wide[g * 2 * topk + topk + i] = Int32(scale_row)
                h_wide_logical[g * 2 * topk + i] = Int32(t)
                if b == 0 and s == 0 and i == 0:
                    first_selected_scale_row = scale_row

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)

    # Device buffers.
    var fp8_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](fp8_elems)
    var fp4_dev = ctx.enqueue_create_buffer[.uint8](fp4_elems)
    var sf_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](sf_elems)
    var sf_canary_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](sf_elems)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lookup_table_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var q_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](q_size)
    var scales_lookup_dev = ctx.enqueue_create_buffer[.uint32](lut_size)
    var indices_dev = ctx.enqueue_create_buffer[.int32](total_indices)
    var logical_dev = ctx.enqueue_create_buffer[.int32](total_indices)
    var wide_dev = ctx.enqueue_create_buffer[.int32](total_wide)
    var wide_logical_dev = ctx.enqueue_create_buffer[.int32](total_wide)
    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_fp8_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_fp4_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_canary_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)

    # The canary cache differs from the real one in every scale of one selected
    # row, raised enough that the row dominates the softmax.
    var canary_row = first_selected_scale_row
    ctx.enqueue_copy(sf_canary_dev, sf_host)
    ctx.enqueue_copy(fp8_dev, fp8_host)
    ctx.enqueue_copy(fp4_dev, fp4_host)
    ctx.enqueue_copy(sf_dev, sf_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.enqueue_copy(lookup_table_dev, lookup_table_host)
    ctx.enqueue_copy(scales_lookup_dev, scales_lookup_host)
    ctx.enqueue_copy(q_dev, q_host)
    ctx.enqueue_copy(indices_dev, h_indices)
    ctx.enqueue_copy(logical_dev, h_logical)
    ctx.enqueue_copy(wide_dev, h_wide)
    ctx.enqueue_copy(wide_logical_dev, h_wide_logical)
    ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
    ctx.synchronize()
    with sf_canary_dev.map_to_host() as mapped:
        for g in range(SF_PER_ROW):
            mapped[canary_row * SF_PER_ROW + g] = bitcast[
                DType.float8_e4m3fn, 1
            ](SIMD[DType.uint8, 1](UInt8(0x58)))[0]
    ctx.synchronize()

    # Collections.
    var fp8_tt = TileTensor(
        fp8_dev,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[ROW_DEPTH],
        ),
    )
    var fp4_tt = TileTensor(
        fp4_dev,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[PACKED_ROW],
        ),
    )
    var sf_tt = TileTensor(
        sf_dev,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[SF_PER_ROW],
        ),
    )
    var sf_canary_tt = TileTensor(
        sf_canary_dev,
        row_major(
            Int64(total_pages),
            Idx[1],
            Int64(NUM_LAYERS),
            Idx[PAGE_SIZE],
            Idx[KV_NUM_HEADS],
            Idx[SF_PER_ROW],
        ),
    )
    var cache_lengths_tt = TileTensor(
        cache_lengths_dev, row_major(Int64(batch_size))
    )
    var lookup_table_tt = TileTensor(
        lookup_table_dev,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
    )
    var scales_lookup_tt = TileTensor(
        scales_lookup_dev,
        row_major(Int64(batch_size), Int64(max_pages_per_batch)),
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
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
    )
    var coll_fp4 = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](fp4_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
        rebind[CollectionFP4.scales_tt_type](sf_tt.as_unsafe_any_origin()),
        scales_lookup_tt.as_imm().as_unsafe_any_origin(),
    )
    var coll_canary = CollectionFP4(
        rebind[CollectionFP4.blocks_tt_type](fp4_tt.as_unsafe_any_origin()),
        cache_lengths_tt.as_imm().as_unsafe_any_origin(),
        lookup_table_tt.as_imm().as_unsafe_any_origin(),
        UInt32(q_max_seq_len),
        UInt32(max_cache_len),
        rebind[CollectionFP4.scales_tt_type](
            sf_canary_tt.as_unsafe_any_origin()
        ),
        scales_lookup_tt.as_imm().as_unsafe_any_origin(),
    )
    var kv_fp8 = coll_fp8.get_key_cache(0)
    var kv_fp4 = coll_fp4.get_key_cache(0)
    var kv_canary = coll_canary.get_key_cache(0)

    var q_tt = TileTensor(
        q_dev, row_major(total_q_tokens, Idx[num_heads], Idx[ROW_DEPTH])
    )
    var out_fp8_tt = TileTensor(
        out_fp8_dev, row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH])
    )
    var out_fp4_tt = TileTensor(
        out_fp4_dev, row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH])
    )
    var out_canary_tt = TileTensor(
        out_canary_dev, row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH])
    )
    var row_offsets_tt = TileTensor(row_offsets_dev, row_major(batch_size + 1))

    var mla_args = MLADispatchScalarArgs[num_heads=num_heads, is_fp8_kv=True](
        batch_size, max_cache_len, q_max_seq_len, ctx
    )
    var scalar_args_buf_tt = mla_args.gpu_tile_tensor()

    var indices_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
        indices_dev.unsafe_ptr()
    )
    var logical_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
        logical_dev.unsafe_ptr()
    )
    var wide_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
        wide_dev.unsafe_ptr()
    )
    var wide_logical_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
        wide_logical_dev.unsafe_ptr()
    )
    _decode[num_heads, use_causal](
        kv_fp8,
        out_fp8_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        indices_ptr,
        logical_ptr,
        topk,
        ctx,
    )
    _decode[num_heads, use_causal](
        kv_fp4,
        out_fp4_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        wide_ptr,
        wide_logical_ptr,
        2 * topk,
        ctx,
    )
    _decode[num_heads, use_causal](
        kv_canary,
        out_canary_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        wide_ptr,
        wide_logical_ptr,
        2 * topk,
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
            _decode[num_heads, use_causal](
                kv_fp8,
                out_fp8_tt,
                q_tt,
                row_offsets_tt,
                scalar_args_buf_tt,
                indices_ptr,
                logical_ptr,
                topk,
                ctx,
            )

        def _launch_fp4(ctx: DeviceContext) raises {imm}:
            _decode[num_heads, use_causal](
                kv_fp4,
                out_fp4_tt,
                q_tt,
                row_offsets_tt,
                scalar_args_buf_tt,
                wide_ptr,
                wide_logical_ptr,
                2 * topk,
                ctx,
            )

        for _ in range(20):
            _launch_fp8(ctx)
            _launch_fp4(ctx)
        ctx.synchronize()
        var fp8_ns = ctx.execution_time(_launch_fp8, perf_iters)
        var fp4_ns = ctx.execution_time(_launch_fp4, perf_iters)
        var fp8_us = Float64(fp8_ns) / Float64(perf_iters) / 1000.0
        var fp4_us = Float64(fp4_ns) / Float64(perf_iters) / 1000.0
        print("PERF", name, " fp8_us:", fp8_us, " nvfp4_us:", fp4_us)

    _ = mla_args
    _ = fp8_dev
    _ = fp4_dev
    _ = sf_dev
    _ = sf_canary_dev
    _ = cache_lengths_dev
    _ = lookup_table_dev
    _ = scales_lookup_dev
    _ = q_dev
    _ = indices_dev
    _ = logical_dev
    _ = wide_dev
    _ = wide_logical_dev
    _ = row_offsets_dev
    _ = out_fp8_dev
    _ = out_fp4_dev
    _ = out_canary_dev


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)

            run_nvfp4_vs_fp8[8, unit_scales=True](
                "unit_scales", [4096], 2048, ctx
            )
            # GLM 5.3: 64 heads / TP8 = 8 per GPU, topk 2048, page 128.
            run_nvfp4_vs_fp8[8]("glm_b1_cl4096", [4096], 2048, ctx)
            run_nvfp4_vs_fp8[8](
                "glm_b4_ragged_partial_pages",
                [3000, 2571, 2200, 2049],
                2048,
                ctx,
            )
            run_nvfp4_vs_fp8[8](
                "glm_b8_mixed",
                [8191, 4096, 2300, 2048, 129, 5000, 3333, 7777],
                2048,
                ctx,
            )

            # Fewer live keys than topk: trailing -1 slots, partial last page,
            # and a partial last tile.
            run_nvfp4_vs_fp8[8]("masked_b3_short", [1000, 200, 5], 2048, ctx)
            run_nvfp4_vs_fp8[8]("masked_tiny", [1, 2], 2048, ctx)
            run_nvfp4_vs_fp8[8]("topk_not_tile", [900], 100, ctx)

            # 16 query heads, and the MTP verify shape (seq_len > 1).
            run_nvfp4_vs_fp8[16]("h16_b2", [3000, 2100], 2048, ctx)
            run_nvfp4_vs_fp8[8]("mtp2_b2", [3000, 2100], 2048, ctx, 2)
            run_nvfp4_vs_fp8[8]("mtp4_b2_masked", [900, 2100], 2048, ctx, 4)

            run_nvfp4_vs_fp8[8, use_causal=True](
                "causal_mtp2", [3000, 2100], 2048, ctx, 2
            )

            # Timing only (prints PERF lines): FP8 cache vs packed NVFP4.
            run_nvfp4_vs_fp8[8, perf_iters=200](
                "perf_b1_cl8192", [8192], 2048, ctx
            )
            run_nvfp4_vs_fp8[8, perf_iters=200](
                "perf_b8_cl8192", List[Int](length=8, fill=8192), 2048, ctx
            )
            run_nvfp4_vs_fp8[8, perf_iters=200](
                "perf_b32_cl4096", List[Int](length=32, fill=4096), 2048, ctx
            )
            run_nvfp4_vs_fp8[8, perf_iters=200](
                "perf_b8_mtp3_cl8192",
                List[Int](length=8, fill=8192),
                2048,
                ctx,
                3,
            )
            run_nvfp4_vs_fp8[16, perf_iters=200](
                "perf_h16_b8_cl8192", List[Int](length=8, fill=8192), 2048, ctx
            )
            print("nvfp4 sparse decode: all cases match the FP8 oracle")
