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

The scale rows come from `_remap_scale_one`, the indexer's own encoder. With
`scale_block_base` the scale block ids start there, so their rows can pass
2**31 as a production scale pool's do (247380 blocks of 9984 rows); a kernel
that read such a row as signed would take it for an unused slot.
"""

from std.collections import Optional
from std.math import ceildiv
from std.memory import bitcast
from std.os import getenv
from std.random import randn, seed
from std.sys import get_defined_int
from std.testing import assert_equal, assert_true

from max.gpu import *
from max.gpu.host import DeviceContext
from max.gpu.host.info import _is_sm10x_gpu
from kv_cache.paged_sparse_kv_index_remap import _remap_scale_one
from kv_cache.types import (
    KVCacheStaticParams,
    KVCacheT,
    PagedKVCacheCollection,
)
from layout import Idx, TileTensor, row_major
from layout.tile_tensor import NullableTileTensor
from nn.attention.mha_mask import CausalMask, NullMask
from nn.attention.mha_utils import MHAConfig, maybe_null_pointer
from nn.attention.gpu.mla import SM100MLA, flare_mla_decoding
from nn.attention.gpu.nvidia.sm100.mla_decode_dispatch import (
    compute_mla_dispatch_scalars,
)
from nn.attention.gpu.nvidia.sm100.mla_decode_utils import (
    _MLA_NVFP4_PACKED_LINEAR,
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
    num_heads: Int,
    use_causal: Bool,
    cache_t: KVCacheT,
    skip: Bool = False,
    has_lengths: Bool = False,
](
    cache: cache_t,
    out_tt: TileTensor[mut=True, address_space=.GENERIC, ...],
    q_tt: TileTensor[.float8_e4m3fn, address_space=.GENERIC, ...],
    row_offsets_tt: TileTensor[.uint32, address_space=.GENERIC, ...],
    scalar_args_buf_tt: TileTensor[.int64, address_space=.GENERIC, ...],
    indices: MutPointer[Int32, MutAnyOrigin],
    logical: MutPointer[Int32, MutAnyOrigin],
    indices_stride: Int,
    ctx: DeviceContext[],
    lengths: MutPointer[Int32, MutAnyOrigin] = MutPointer[
        Int32, MutAnyOrigin
    ].unsafe_dangling(),
    np: Int = 1,
) raises:
    comptime scale = Float32(0.125)
    var np_arg: Optional[Int] = None
    if np > 0:
        np_arg = Optional[Int](np)
    comptime api_mode = get_defined_int["MLA_NVFP4_TEST_PARTITION_API", 0]()
    var partition_input = np_arg
    var legacy_partitions = np_arg
    comptime if api_mode == 1 or api_mode == 2:
        partition_input = None
    elif api_mode == 3 and cache_t.dtype == DType.uint8:
        legacy_partitions = Optional[Int](np + 1)
    var scalar_input: NullableTileTensor[
        DType.int64,
        type_of(scalar_args_buf_tt).LayoutType,
        ImmOrigin(type_of(scalar_args_buf_tt).origin),
        Engine=type_of(scalar_args_buf_tt).Engine,
        address_space=type_of(scalar_args_buf_tt).address_space,
        linear_idx_type=type_of(scalar_args_buf_tt).linear_idx_type,
    ] = scalar_args_buf_tt
    comptime if api_mode == 2:
        scalar_input = {None, scalar_args_buf_tt.layout}
    comptime if skip:
        # The prefill-on-decode route: rows packed causally (token s draws
        # its keys from [0, cache_len + s]), so the kernel's causal-count
        # bound skips each token's empty tail tiles.
        flare_mla_decoding[
            rank=3,
            config=MHAConfig[.float8_e4m3fn](num_heads, ROW_DEPTH),
            ragged=True,
            sparse=True,
            skip_masked_tiles=True,
            kernels=SM100MLA,
        ](
            out_tt,
            q_tt,
            cache,
            CausalMask(),
            row_offsets_tt,
            scale,
            ctx,
            scalar_input,
            d_indices=indices,
            indices_stride=indices_stride,
            topk_lengths=maybe_null_pointer[has_lengths](lengths),
            num_partitions=legacy_partitions,
            num_partitions_in=partition_input,
        )
    else:
        comptime if use_causal:
            flare_mla_decoding[
                rank=3,
                config=MHAConfig[.float8_e4m3fn](num_heads, ROW_DEPTH),
                ragged=True,
                sparse=True,
                kernels=SM100MLA,
            ](
                out_tt,
                q_tt,
                cache,
                CausalMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_input,
                d_indices=indices,
                indices_stride=indices_stride,
                logical_indices=logical,
                num_partitions=legacy_partitions,
                num_partitions_in=partition_input,
            )
        else:
            flare_mla_decoding[
                rank=3,
                config=MHAConfig[.float8_e4m3fn](num_heads, ROW_DEPTH),
                ragged=True,
                sparse=True,
                kernels=SM100MLA,
            ](
                out_tt,
                q_tt,
                cache,
                NullMask(),
                row_offsets_tt,
                scale,
                ctx,
                scalar_input,
                d_indices=indices,
                indices_stride=indices_stride,
                num_partitions=legacy_partitions,
                num_partitions_in=partition_input,
            )


def run_nvfp4_vs_fp8[
    num_heads: Int,
    use_causal: Bool = False,
    unit_scales: Bool = False,
    perf_iters: Int = 0,
    skip: Bool = False,
    has_lengths: Bool = False,
](
    name: StringLiteral,
    cache_lens: List[Int],
    topk: Int,
    ctx: DeviceContext[],
    q_max_seq_len: Int = 1,
    scale_block_base: Int = 0,
    np: Int = 1,
    canary_slot: Int = 0,
    value_canary_slot: Int = -1,
    readout_token: Int = -1,
    readout_profile: Bool = False,
    readout_zero_q_rope: Bool = False,
    readout_zero_q_latent: Bool = False,
    readout_rope: Bool = False,
    readout_all_rows: Bool = False,
    readout_block_markers: Bool = False,
    readout_stage_markers: Bool = False,
    readout_pair: Bool = False,
    readout_qonehot: Int = -1,
) raises:
    var batch_size = len(cache_lens)
    var total_q_tokens = batch_size * q_max_seq_len
    comptime scale = Float32(0.125)
    print(
        t"test: {name}  batch: {batch_size}  heads: {num_heads}  topk: {topk} "
        t" q_len: {q_max_seq_len}  causal: {use_causal}  skip: {skip}"
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
    # Jenga scales leaf has block ids of its own, starting at
    # `scale_block_base`.
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
                + scale_block_base
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
            # Row in the scale buffer, which holds blocks from
            # `scale_block_base` on.
            var scale_row = (
                Int(scales_lookup_host[b * max_pages_per_batch + page_idx])
                - scale_block_base
            ) * PAGE_SIZE + tok_in_page
            # Readout crafting (`readout_token`): every token's latent and
            # rope is zero with unit scales except one token's, whose
            # element e holds e2m1 code (e & 15) -- so both caches hold
            # known per-dim values and the output reduces to
            # `w_readout(h) * value[v]` over the selected keys, which
            # separates a wrong K staging (weight error, uniform per head)
            # from a wrong V staging (value error, per dim).
            var _readout = readout_token >= 0
            var _is_readout = _readout and (
                readout_all_rows or t == readout_token
            )
            for g in range(SF_PER_ROW):
                var sf_byte = _scale_byte(rng)
                if _readout:
                    sf_byte = UInt8(0x38)
                comptime if unit_scales:
                    sf_byte = UInt8(0x38)
                sf_host[scale_row * SF_PER_ROW + g] = bitcast[
                    DType.float8_e4m3fn, 1
                ](SIMD[DType.uint8, 1](sf_byte))[0]
                for j in range(16):
                    var code = Int(rng.next() & 15)
                    if _readout:
                        code = ((g * 16 + j) & 15) if _is_readout else 0
                        if readout_block_markers and _is_readout:
                            # One distinct magnitude per 64-column block:
                            # a K-view block permutation then shows up as a
                            # per-block weight error the probe can attribute
                            # offline against the known Q.
                            code = ((g // 4) & 7) + 1
                        if readout_stage_markers and not _is_readout:
                            # Tokens 0..63 stage a 0.5-constant row, 64..127
                            # a 1.5-constant one, so a K operand read from
                            # the wrong ring stage is identifiable per head:
                            # S' would be 0.5 or 1.5 times the head's latent
                            # Q sum instead of the readout row's dot.
                            code = 1 if t < 64 else (3 if t < 128 else 0)
                        if readout_pair:
                            # Two readout keys with disjoint V dims: token 1
                            # (slot 0, correct territory) owns dims 0..255
                            # and the readout token (slot 128) dims 256..511.
                            # Each key's weight is then measurable per head
                            # per arm with no model, which splits an
                            # own-score error from a denominator error.
                            var _pe = g * 16 + j
                            if t == 1:
                                code = (_pe & 15) if _pe < 256 else 0
                            elif _is_readout:
                                code = ((_pe - 256) & 15) if _pe >= 256 else 0
                            else:
                                code = 0
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
                if _readout:
                    rope_byte = UInt8(0x38 + e % 8) if (
                        _is_readout and readout_rope
                    ) else UInt8(0)
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
        var _dim = i % ROW_DEPTH
        if readout_zero_q_rope and _dim >= V_DEPTH:
            q_host[i] = bitcast[DType.float8_e4m3fn, 1](
                SIMD[DType.uint8, 1](UInt8(0))
            )[0]
        if readout_zero_q_latent and _dim < V_DEPTH:
            q_host[i] = bitcast[DType.float8_e4m3fn, 1](
                SIMD[DType.uint8, 1](UInt8(0))
            )[0]
        if readout_qonehot >= 0:
            var _head = i // ROW_DEPTH
            q_host[i] = bitcast[DType.float8_e4m3fn, 1](
                SIMD[DType.uint8, 1](UInt8(0))
            )[0]
            if _dim == readout_qonehot * 64 + _head:
                q_host[i] = bitcast[DType.float8_e4m3fn, 1](
                    SIMD[DType.uint8, 1](UInt8(0x50))
                )[0]

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
    var readout_value_row = -1
    for b in range(batch_size):
        var nk = num_keys_of[b]
        var n_valid = min(topk, nk)
        var mult = _coprime_multiplier(nk)
        for s in range(q_max_seq_len):
            var g = b * q_max_seq_len + s
            # In skip mode the row is packed the way the indexer packs it for
            # the prefill-on-decode route: token s draws min(topk, cache + s
            # + 1) keys from [0, cache + s], and the rest is -1 padding the
            # causal-count bound skips. With per-token lengths the row keeps
            # the full n_valid live entries (drawn from the whole sequence,
            # as the not-skip cases are), so the slots past the length hold
            # live keys only the length excludes; the FP8-vs-NVFP4 oracle
            # reads the same rows either way.
            var ring = cache_lens[b] + s + 1
            var ring_mult = _coprime_multiplier(ring)
            comptime if skip and not has_lengths:
                n_valid = min(topk, ring)
            for i in range(n_valid):
                var t = (i * mult + 1 + s) % nk
                comptime if skip and not has_lengths:
                    t = (i * ring_mult + 1 + s) % ring
                var page_idx, tok_in_page = divmod(t, PAGE_SIZE)
                var block_id = Int(
                    lookup_table_host[b * max_pages_per_batch + page_idx]
                )
                h_indices[g * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
                h_logical[g * topk + i] = Int32(t)
                h_wide[g * 2 * topk + i] = Int32(
                    block_id * PAGE_SIZE + tok_in_page
                )
                h_wide[g * 2 * topk + topk + i] = _remap_scale_one(
                    Int32(t),
                    UInt32(b),
                    scales_lookup_host.unsafe_ptr().as_imm(),
                    max_pages_per_batch,
                    batch_size,
                    PAGE_SIZE,
                    UInt32(scale_block_base + total_pages),
                    NUM_LAYERS * PAGE_SIZE,
                )
                h_wide_logical[g * 2 * topk + i] = Int32(t)
                if b == 0 and s == 0 and t == readout_token:
                    readout_value_row = Int(h_wide[g * 2 * topk + i])
                if b == 0 and s == 0 and i == canary_slot:
                    first_selected_scale_row = (
                        Int(
                            scales_lookup_host[
                                b * max_pages_per_batch + page_idx
                            ]
                        )
                        - scale_block_base
                    ) * PAGE_SIZE + tok_in_page

    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    for i in range(batch_size + 1):
        row_offsets_host[i] = UInt32(i * q_max_seq_len)

    # Per-token causal counts, the top-k lengths the prefill-on-decode graph
    # op supplies. With `has_lengths` the packed row holds more live entries
    # than early tokens' lengths, so the length is the only thing excluding
    # them.
    var lengths_host = ctx.enqueue_create_host_buffer[.int32](total_q_tokens)
    for b in range(batch_size):
        for s in range(q_max_seq_len):
            lengths_host[b * q_max_seq_len + s] = Int32(
                min(topk, cache_lens[b] + s + 1)
            )

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
    var lengths_dev = ctx.enqueue_create_buffer[.int32](total_q_tokens)
    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var out_size = total_q_tokens * num_heads * V_DEPTH
    var out_fp8_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_fp4_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
    var out_canary_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)

    # The canary cache differs from the real one in every scale of one selected
    # row, raised enough that the row dominates the softmax.
    var canary_row = first_selected_scale_row
    if readout_token >= 0:
        for b in range(batch_size):
            if readout_token < num_keys_of[b]:
                var page_idx, tok_in_page = divmod(readout_token, PAGE_SIZE)
                canary_row = (
                    Int(scales_lookup_host[b * max_pages_per_batch + page_idx])
                    - scale_block_base
                ) * PAGE_SIZE + tok_in_page
                break
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
    ctx.enqueue_copy(lengths_dev, lengths_host)
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
    # The scale pool spans blocks 0 .. scale_block_base + total_pages, but only
    # the last total_pages are ever read, so its view starts scale_block_base
    # blocks before the buffer rather than allocating the blocks below.
    var sf_view_offset = -scale_block_base * NUM_LAYERS * PAGE_SIZE * SF_PER_ROW
    var sf_layout = row_major(
        Int64(scale_block_base + total_pages),
        Idx[1],
        Int64(NUM_LAYERS),
        Idx[PAGE_SIZE],
        Idx[KV_NUM_HEADS],
        Idx[SF_PER_ROW],
    )
    var sf_tt = TileTensor(
        sf_dev.unsafe_ptr().unsafe_offset(sf_view_offset), sf_layout
    )
    var sf_canary_tt = TileTensor(
        sf_canary_dev.unsafe_ptr().unsafe_offset(sf_view_offset), sf_layout
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

    # The kernel takes its split count from this pack while the launcher takes
    # the grid's from `num_partitions`/`num_partitions_in`. Stage the split the
    # cell launches, or the kernel's own split differs from the grid: every CTA
    # then walks a slice no CTA owns, and both arms drop the same keys, so the
    # comparison passes without ever computing the selection.
    comptime sm_count = ctx.default_device_info.sm_count
    var plain_scalars = compute_mla_dispatch_scalars[num_heads, is_fp8_kv=True](
        batch_size, max_cache_len, q_max_seq_len, sm_count
    )
    var launched_partitions = np if np > 0 else plain_scalars[2]
    var dispatch_args_host = ctx.enqueue_create_host_buffer[.int64](3)
    dispatch_args_host[0] = Int64(batch_size)
    dispatch_args_host[1] = Int64(q_max_seq_len)
    dispatch_args_host[2] = Int64(launched_partitions)
    var dispatch_args_dev = ctx.enqueue_create_buffer[.int64](3)
    ctx.enqueue_copy(dispatch_args_dev, dispatch_args_host)
    var scalar_args_buf_tt = TileTensor(
        dispatch_args_dev.unsafe_ptr(), row_major((Idx[3],))
    )

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
    var lengths_ptr = rebind[MutPointer[Int32, MutAnyOrigin]](
        lengths_dev.unsafe_ptr()
    )
    if getenv("MLA_NVFP4_DIAG_SYNC"):
        print("diagnostic launch: FP8")
        ctx.synchronize()
    _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
        kv_fp8,
        out_fp8_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        indices_ptr,
        logical_ptr,
        topk,
        ctx,
        lengths_ptr,
        np,
    )
    if getenv("MLA_NVFP4_DIAG_SYNC"):
        ctx.synchronize()
        print("diagnostic completed: FP8; launch: NVFP4")
    _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
        kv_fp4,
        out_fp4_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        wide_ptr,
        wide_logical_ptr,
        2 * topk,
        ctx,
        lengths_ptr,
        np,
    )
    if getenv("MLA_NVFP4_DIAG_SYNC"):
        ctx.synchronize()
        print("diagnostic completed: NVFP4; launch: scale canary")
    _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
        kv_canary,
        out_canary_tt,
        q_tt,
        row_offsets_tt,
        scalar_args_buf_tt,
        wide_ptr,
        wide_logical_ptr,
        2 * topk,
        ctx,
        lengths_ptr,
        np,
    )
    # The value canary: one selected row's packed bytes flipped in place, so
    # a dead output says the gather never fetched that row.
    var out_vcanary_dev = out_canary_dev
    var value_canary_on = False
    var value_canary_hit = -1
    if value_canary_slot >= 0 and Int(h_wide[value_canary_slot]) >= 0:
        value_canary_on = True
        value_canary_hit = 0
        var fp4_vcanary_dev = ctx.enqueue_create_buffer[.uint8](fp4_elems)
        ctx.enqueue_copy(fp4_vcanary_dev, fp4_dev)
        ctx.synchronize()
        with fp4_vcanary_dev.map_to_host() as mapped:
            var vrow = Int(h_wide[value_canary_slot])
            for i in range(PACKED_ROW):
                mapped[vrow * PACKED_ROW + i] ^= UInt8(0x5A)
        ctx.synchronize()
        var fp4_vcanary_tt = TileTensor(
            fp4_vcanary_dev,
            row_major(
                Int64(total_pages),
                Idx[1],
                Int64(NUM_LAYERS),
                Idx[PAGE_SIZE],
                Idx[KV_NUM_HEADS],
                Idx[PACKED_ROW],
            ),
        )
        var coll_vcanary = CollectionFP4(
            rebind[CollectionFP4.blocks_tt_type](
                fp4_vcanary_tt.as_unsafe_any_origin()
            ),
            cache_lengths_tt.as_imm().as_unsafe_any_origin(),
            lookup_table_tt.as_imm().as_unsafe_any_origin(),
            UInt32(q_max_seq_len),
            UInt32(max_cache_len),
            rebind[CollectionFP4.scales_tt_type](sf_tt.as_unsafe_any_origin()),
            scales_lookup_tt.as_imm().as_unsafe_any_origin(),
        )
        out_vcanary_dev = ctx.enqueue_create_buffer[.bfloat16](out_size)
        _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
            coll_vcanary.get_key_cache(0),
            TileTensor(
                out_vcanary_dev,
                row_major(total_q_tokens, Idx[num_heads], Idx[V_DEPTH]),
            ),
            q_tt,
            row_offsets_tt,
            scalar_args_buf_tt,
            wide_ptr,
            wide_logical_ptr,
            2 * topk,
            ctx,
            lengths_ptr,
            np,
        )
    ctx.synchronize()

    var out_fp8 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var out_fp4 = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    var out_canary = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
    ctx.enqueue_copy(out_fp8, out_fp8_dev)
    ctx.enqueue_copy(out_fp4, out_fp4_dev)
    ctx.enqueue_copy(out_canary, out_canary_dev)
    ctx.synchronize()
    var out_vcanary = out_canary
    if value_canary_on:
        out_vcanary = ctx.enqueue_create_host_buffer[.bfloat16](out_size)
        ctx.enqueue_copy(out_vcanary, out_vcanary_dev)
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
            if mismatches < 32:
                print(
                    t"  mismatch at {i} : fp8 ref {out_fp8[i]}  nvfp4"
                    t" {out_fp4[i]}"
                )
            mismatches += 1
        if b != c:
            canary_mismatches += 1
        if value_canary_on:
            if (
                b
                != bitcast[DType.uint16, 1](
                    SIMD[DType.bfloat16, 1](out_vcanary[i])
                )[0]
            ):
                value_canary_hit += 1
        if abs(out_fp8[i].cast[.float32]()) > 1e-6:
            nonzero += 1
        var v = out_fp4[i].cast[.float32]()
        if v != v or abs(v) > 1e30:
            nonfinite += 1

    print(
        t"  mismatches: {mismatches} / {out_size}  canary_mismatches:"
        t" {canary_mismatches}  nonzero: {nonzero}"
    )
    if value_canary_on:
        print("  value_canary_diffs:", value_canary_hit)
    if readout_profile and num_heads <= 64 and q_max_seq_len == 1:
        # The full Q matrix: with the per-head weights above it forms the
        # sparse-recovery system q_h . delta_k = d(h), whose solution names
        # the leaked K bytes' positions and values directly.
        # The one-hot cells skip the dump: their Q is trivial.
        if readout_qonehot < 0:
            for h in range(num_heads):
                var qrow = List[Float64]()
                for e in range(ROW_DEPTH):
                    qrow.append(
                        Float64(q_host[h * ROW_DEPTH + e].cast[.float32]())
                    )
                print("  qrow h:", h, " q:", qrow)
        # Head 0's per-dim readout: the crafted output is
        # `w_readout * value[v]`, so a per-dim constant ratio means the V
        # bytes are right and only the weight (the K side) is off; a
        # broken ratio at some v localizes corrupted latent dims.
        var _prof = 0
        for v in range(V_DEPTH):
            var a0 = bitcast[DType.uint16, 1](
                SIMD[DType.bfloat16, 1](out_fp8[v])
            )[0]
            var b0 = bitcast[DType.uint16, 1](
                SIMD[DType.bfloat16, 1](out_fp4[v])
            )[0]
            if a0 != b0:
                var ra = out_fp8[v].cast[.float32]()
                var rb = out_fp4[v].cast[.float32]()
                var ratio = (
                    Float64(rb) / Float64(ra) if abs(ra) > 1e-12 else -1.0
                )
                print(
                    "  v:",
                    v,
                    " ref:",
                    out_fp8[v],
                    " nvfp4:",
                    out_fp4[v],
                    " ratio:",
                    ratio,
                )
                _prof += 1
        print("  head0 mismatched dims:", _prof)
        if readout_pair:
            # Two crafted keys with disjoint dims: out[h][v] = w(h)*val[v]
            # per range, so each key's weight is a plain average per arm.
            for h in range(num_heads):
                var sums = List[Float64]()
                for owner in range(2):
                    for arm in range(2):
                        var acc = 0.0
                        var n = 0
                        for v in range(256):
                            var vv = v + 256 * owner
                            var val = Float64(E2M1_MAGNITUDES[vv & 7]) * (
                                -1.0 if (vv & 8) != 0 else 1.0
                            )
                            if val == 0:
                                continue
                            var idx = h * V_DEPTH + vv
                            var o = (
                                out_fp8[idx].cast[.float32]() if arm
                                == 0 else out_fp4[idx].cast[.float32]()
                            )
                            acc += Float64(o) / Float64(val)
                            n += 1
                        sums.append(acc / Float64(n))
                print(
                    "  pair h:",
                    h,
                    " wA_ref:",
                    sums[0],
                    " wA_nv:",
                    sums[1],
                    " wB_ref:",
                    sums[2],
                    " wB_nv:",
                    sums[3],
                )

        # Per-head weight factor: the crafted output is w(h)*value[v], so
        # every mismatching dim holds the same per-head ratio; average it
        # over all of them to wash out bf16 rounding.
        for h in range(num_heads):
            var _hm = 0
            var _hratio_sum = 0.0
            var _w_ref_sum = 0.0
            var _w_nv_sum = 0.0
            var _w_n = 0
            for v in range(V_DEPTH):
                var base = h * V_DEPTH + v
                var ra = out_fp8[base].cast[.float32]()
                var rb = out_fp4[base].cast[.float32]()
                if abs(ra) > 1e-12 and ra != rb:
                    _hm += 1
                    _hratio_sum += Float64(rb) / Float64(ra)
                # The crafted value of dim v: code (v & 15) with unit scales.
                var val = Float64(E2M1_MAGNITUDES[v & 7]) * (
                    -1.0 if (v & 8) != 0 else 1.0
                )
                if val != 0:
                    _w_ref_sum += Float64(ra) / val
                    _w_nv_sum += Float64(rb) / val
                    _w_n += 1
            if _w_n > 0 and not readout_pair:
                print(
                    "  head:",
                    h,
                    " mismatched_dims:",
                    _hm,
                    " w_ref:",
                    _w_ref_sum / Float64(_w_n),
                    " w_nv:",
                    _w_nv_sum / Float64(_w_n),
                )
        if readout_value_row >= 0:
            # Per-head Q dots against the readout token's dequantized row:
            # with every other token zero, 2^S = e^(q.k * 0.125), so the
            # measured c(h) must equal (2^S + 128) / (129 * 2^S) if the bad
            # key's K operand is read as zeros (a wrong ring stage).
            var _vrow = readout_value_row
            for h in range(num_heads):
                var qk_latent = Float64(0)
                var qk_rope = Float64(0)
                var qsum_latent = Float64(0)
                for e in range(ROW_DEPTH):
                    var qv = q_host[h * ROW_DEPTH + e].cast[.float32]()
                    var kv = fp8_host[_vrow * ROW_DEPTH + e].cast[.float32]()
                    if e < V_DEPTH:
                        qk_latent += Float64(qv) * Float64(kv)
                        qsum_latent += Float64(qv)
                    else:
                        qk_rope += Float64(qv) * Float64(kv)
                # Per-chunk Q sums: 16 latent chunks of 32 dims and 4 rope
                # chunks of 16, so a chunk- or block-granular K-view
                # permutation is fittable offline (d(h) = sum_c dc*Qc(h)).
                var chunk_sums = List[Float64]()
                for c in range(20):
                    var csum = Float64(0)
                    for e in range(32 if c < 16 else 16):
                        var idx = c * 32 + e if c < 16 else (
                            V_DEPTH + (c - 16) * 16 + e
                        )
                        csum += Float64(
                            q_host[h * ROW_DEPTH + idx].cast[.float32]()
                        )
                    chunk_sums.append(csum)
                print(
                    "  qdot h:",
                    h,
                    " qk_latent:",
                    qk_latent,
                    " qk_rope:",
                    qk_rope,
                    " qsum_latent:",
                    qsum_latent,
                    " chunks:",
                    chunk_sums,
                )
    assert_equal(nonfinite, 0)
    assert_true(nonzero > out_size // 2, "reference output is mostly zero")
    comptime if get_defined_int["MLA_NVFP4_BREAK", 0]() != 0:
        assert_true(mismatches > 0, "NVFP4 negative control did not fire")
    else:
        assert_equal(mismatches, 0)
        assert_true(
            canary_mismatches > 0, "a changed scale did not change the output"
        )

    comptime if perf_iters > 0:

        def _launch_fp8(ctx: DeviceContext[]) raises {imm}:
            _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
                kv_fp8,
                out_fp8_tt,
                q_tt,
                row_offsets_tt,
                scalar_args_buf_tt,
                indices_ptr,
                logical_ptr,
                topk,
                ctx,
                lengths_ptr,
            )

        def _launch_fp4(ctx: DeviceContext[]) raises {imm}:
            _decode[num_heads, use_causal, skip=skip, has_lengths=has_lengths](
                kv_fp4,
                out_fp4_tt,
                q_tt,
                row_offsets_tt,
                scalar_args_buf_tt,
                wide_ptr,
                wide_logical_ptr,
                2 * topk,
                ctx,
                lengths_ptr,
            )

        for _ in range(20):
            _launch_fp8(ctx)
            _launch_fp4(ctx)
        ctx.synchronize()
        var fp8_ns = ctx.execution_time(_launch_fp8, perf_iters)
        var fp4_ns = ctx.execution_time(_launch_fp4, perf_iters)
        var fp8_us = Float64(fp8_ns) / Float64(perf_iters) / 1000.0
        var fp4_us = Float64(fp4_ns) / Float64(perf_iters) / 1000.0
        print(t"PERF {name}  fp8_us: {fp8_us}  nvfp4_us: {fp4_us}")

    _ = dispatch_args_dev
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
    _ = lengths_dev
    _ = row_offsets_dev
    _ = out_fp8_dev
    _ = out_fp4_dev
    _ = out_canary_dev
    _ = out_vcanary_dev


def _probe_layout_e_cell[
    num_heads: Int
](
    name: StringLiteral,
    cache_lens: List[Int],
    topk: Int,
    ctx: DeviceContext[],
    np: Int = 1,
    canary_slot: Int = 0,
    value_canary_slot: Int = -1,
    readout_token: Int = -1,
    readout_profile: Bool = False,
    readout_zero_q_rope: Bool = False,
    readout_zero_q_latent: Bool = False,
    readout_rope: Bool = False,
    readout_all_rows: Bool = False,
    readout_block_markers: Bool = False,
    readout_stage_markers: Bool = False,
    readout_pair: Bool = False,
    readout_qonehot: Int = -1,
) raises:
    """Reports a historical bisect cell with matched execution geometry."""
    try:
        run_nvfp4_vs_fp8[num_heads](
            name,
            cache_lens,
            topk,
            ctx,
            np=np,
            canary_slot=canary_slot,
            value_canary_slot=value_canary_slot,
            readout_token=readout_token,
            readout_profile=readout_profile,
            readout_zero_q_rope=readout_zero_q_rope,
            readout_zero_q_latent=readout_zero_q_latent,
            readout_rope=readout_rope,
            readout_all_rows=readout_all_rows,
            readout_block_markers=readout_block_markers,
            readout_stage_markers=readout_stage_markers,
            readout_pair=readout_pair,
            readout_qonehot=readout_qonehot,
        )
        print("  probe", name, ": PASS")
    except e:
        print("  probe", name, ": FAIL (", String(e), ")")


def _test_layout_e_boundaries[num_heads: Int](ctx: DeviceContext[]) raises:
    for topk in [63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 384]:
        run_nvfp4_vs_fp8[num_heads]("layout_e_boundary", [4096], topk, ctx)
    for np in [1, 2, 3, 4, 16]:
        run_nvfp4_vs_fp8[num_heads](
            "layout_e_wide_split", [4096], 2048, ctx, np=np
        )
    run_nvfp4_vs_fp8[num_heads](
        "layout_e_empty_splits", [192, 64, 5], 192, ctx, np=4
    )
    run_nvfp4_vs_fp8[num_heads](
        "layout_e_unsigned_scales", [4096], 2048, ctx, scale_block_base=20000000
    )
    run_nvfp4_vs_fp8[num_heads, use_causal=True](
        "layout_e_causal", [900, 4096], 257, ctx, 2
    )
    run_nvfp4_vs_fp8[num_heads, skip=True, has_lengths=True](
        "layout_e_skip_lengths", [129, 4096], 2048, ctx, 2, np=2
    )
    # A packed row whose live count is not a multiple of the tile: the causal
    # bound admits the last tile's window, and only the gathered row can show
    # that its slots past 1501 are padding. That makes the sentinel
    # load-bearing, so a broken all-live test changes the output.
    run_nvfp4_vs_fp8[num_heads, skip=True](
        "layout_e_skip_sentinel", [1500, 4096], 2048, ctx
    )
    run_nvfp4_vs_fp8[num_heads](
        "layout_e_onehot",
        [129],
        129,
        ctx,
        readout_token=125,
        readout_qonehot=0,
        canary_slot=128,
    )


def main() raises:
    with DeviceContext() as ctx:
        comptime if ctx.target.is_nvidia_gpu() and _is_sm10x_gpu(
            ctx.default_device_info
        ):
            seed(42)
            comptime if get_defined_int["MLA_NVFP4_BREAK", 0]() != 0:
                run_nvfp4_vs_fp8[128](
                    "nvfp4_negative_control", [4096], 384, ctx
                )
                return
            comptime if get_defined_int[
                "MLA_NVFP4_TEST_PARTITION_API", 0
            ]() != 0:
                _test_layout_e_boundaries[64](ctx)
                return
            comptime if get_defined_int["MLA_NVFP4_TEST_PAIR", 0]() != 0:
                _test_layout_e_boundaries[128](ctx)
                return

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

            # The prefill-on-decode route (`skip_masked_tiles`): rows packed
            # causally per token, so each token's empty tail tiles are
            # skipped; both caches must still feed the MMA identical bytes.
            run_nvfp4_vs_fp8[8, skip=True](
                "skip_b3_s4", [3000, 2100, 900], 2048, ctx, 4
            )
            run_nvfp4_vs_fp8[8, skip=True](
                "skip_short_cache", [1000, 200, 5], 2048, ctx
            )
            run_nvfp4_vs_fp8[8, skip=True]("skip_tiny", [1, 2], 2048, ctx)
            run_nvfp4_vs_fp8[16, skip=True](
                "skip_h16_s2", [3000, 2100], 2048, ctx, 2
            )
            run_nvfp4_vs_fp8[32, skip=True]("skip_h32_b1", [4096], 2048, ctx, 2)
            # The exact oracle fixes the split and key tile in both arms.
            comptime if _MLA_NVFP4_PACKED_LINEAR != 0:
                _test_layout_e_boundaries[64](ctx)
                _test_layout_e_boundaries[128](ctx)
                run_nvfp4_vs_fp8[64]("h64_b2_page", [128, 128], 128, ctx)
                run_nvfp4_vs_fp8[64](
                    "h64_b8_page", List[Int](length=8, fill=128), 128, ctx
                )
                run_nvfp4_vs_fp8[64, skip=True](
                    "skip_h64_s2_page", [128, 128], 128, ctx, 2
                )
                run_nvfp4_vs_fp8[64, skip=True, has_lengths=True](
                    "skip_h64_lengths_page", [128, 100], 128, ctx, 2
                )
                run_nvfp4_vs_fp8[64]("masked_h64_short", [100, 60, 5], 128, ctx)
                run_nvfp4_vs_fp8[128]("h128_b2_page", [128, 128], 128, ctx)
                run_nvfp4_vs_fp8[128, skip=True](
                    "skip_h128_s2_page", [128, 128], 128, ctx, 2
                )
                run_nvfp4_vs_fp8[128, skip=True, has_lengths=True](
                    "skip_h128_lengths_page", [128, 90], 128, ctx, 2
                )

                # Retain the historical probes after correcting their split override.
                if getenv("MLA_NVFP4_LAYOUT_E_BISECT"):
                    _probe_layout_e_cell[64](
                        "bisect_cl192_topk128", [192], 128, ctx
                    )
                    _probe_layout_e_cell[64](
                        "bisect_cl4096_topk128", [4096], 128, ctx
                    )
                    _probe_layout_e_cell[64](
                        "bisect_cl129_topk129", [129], 129, ctx
                    )
                    _probe_layout_e_cell[64](
                        "bisect_cl4096_topk192", [4096], 192, ctx
                    )
                    _probe_layout_e_cell[64](
                        "bisect_cl256_topk192", [256], 192, ctx
                    )
                    _probe_layout_e_cell[128](
                        "bisect_h128_cl192_topk128", [192], 128, ctx
                    )
                    # Split-K cells: page-128 splits hand split 1 the global
                    # slots 128 and up as its own one-tile walk, so these
                    # separate "the third tile of a CTA's walk" from "the
                    # global slot position" (np 1 vs np 2 at topk 129).
                    _probe_layout_e_cell[64](
                        "bisect_np2_cl129_topk129", [129], 129, ctx, np=2
                    )
                    _probe_layout_e_cell[64](
                        "bisect_np2_cl4096_topk192", [4096], 192, ctx, np=2
                    )
                    _probe_layout_e_cell[64](
                        "bisect_np3_cl4096_topk384", [4096], 384, ctx, np=3
                    )
                    # Canary localization at the failing slot 128 of
                    # cl129_topk129: whether the scale fetch and the value
                    # fetch read that slot's true rows.
                    _probe_layout_e_cell[64](
                        "bisect_canary_s0_cl129",
                        [129],
                        129,
                        ctx,
                        canary_slot=0,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_canary_s128_cl129",
                        [129],
                        129,
                        ctx,
                        canary_slot=128,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_canary_v128_cl129",
                        [129],
                        129,
                        ctx,
                        value_canary_slot=128,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_canary_v0_cl129",
                        [129],
                        129,
                        ctx,
                        value_canary_slot=0,
                    )
                    # Readout crafting: every token's latent and rope zero
                    # except one, whose dim e holds code (e & 15), so the
                    # output is `w_readout(h) * value[v]` and the mismatch
                    # pattern across v (V staging) and h (K staging)
                    # localizes the wrong bytes. Token 125 is slot 128's
                    # token: t = (128 * 3 + 1) % 130, nk = cache + q_len.
                    _probe_layout_e_cell[64](
                        "bisect_readout_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_profile=True,
                    )
                    # Test B: Q's rope dims zeroed. delta(h) = q_h . deltak,
                    # so if the K-view corruption lives in the tile's rope
                    # columns (512-575, K-only territory) this passes; if it
                    # is a misread of latent columns, it still fails.
                    _probe_layout_e_cell[64](
                        "bisect_readout_q0rope_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_zero_q_rope=True,
                    )
                    # Test C: Q's latent dims zeroed and the readout
                    # token's rope bytes crafted, so S reads the rope view
                    # alone.
                    _probe_layout_e_cell[64](
                        "bisect_readout_q0latent_rope_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_zero_q_latent=True,
                        readout_rope=True,
                    )
                    # Test D: every token's row crafted IDENTICALLY, so a
                    # K-view that permutes bytes within the tile reads the
                    # same bytes and must agree; a failure here means the K
                    # view reads bytes from outside the tile (or the fault
                    # is past the K read).
                    _probe_layout_e_cell[64](
                        "bisect_readout_allrows_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_all_rows=True,
                    )
                    # Test E: block markers -- the readout row holds one
                    # distinct magnitude per 64-column block, so a K-view
                    # block permutation is attributable offline.
                    _probe_layout_e_cell[64](
                        "bisect_readout_blockmarkers_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_block_markers=True,
                        readout_profile=True,
                    )
                    # Test F: stage markers -- tokens 0..63 hold a 0.5-
                    # constant row and 64..127 a 1.5-constant one, so if
                    # the bad key's K operand is gathered from the wrong
                    # ring stage its score becomes 0.5 or 1.5 times the
                    # head's latent Q sum, which the qdot profile makes
                    # attributable offline.
                    _probe_layout_e_cell[64](
                        "bisect_readout_stagemarks_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_stage_markers=True,
                        readout_profile=True,
                    )
                    # Test G: paired readout -- a correct-territory key
                    # (token 1 at slot 0) and the bad key (slot 128) hold
                    # disjoint V dims, so both keys' weights are measurable
                    # per head per arm. wA moving means the denominator moved
                    # (other keys' scores corrupted); wA exact means the bad
                    # key's own score is the only fault.
                    _probe_layout_e_cell[64](
                        "bisect_readout_pair_cl129",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_pair=True,
                        readout_profile=True,
                    )
                    # One-hot readout: head h's query is 8.0 at exactly one
                    # K dim (64*k + h), so each head's weight reads the
                    # corrupted K row at one dimension directly; nine
                    # variants tile all 576 dims.
                    # Layout G control for the same byte readout: the
                    # routed head counts run the G form, which passes the
                    # oracle, so its P quantization is the reference
                    # behavior to compare against the E form's.
                    comptime if _MLA_NVFP4_PACKED_LINEAR != 0:
                        pass
                    _probe_layout_e_cell[32](
                        "bisect_readout_onehot_g0",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=0,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot0",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=0,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot1",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=1,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot2",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=2,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot3",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=3,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot4",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=4,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot5",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=5,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot6",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=6,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot7",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=7,
                        readout_profile=True,
                    )
                    _probe_layout_e_cell[64](
                        "bisect_readout_onehot8",
                        [129],
                        129,
                        ctx,
                        readout_token=125,
                        readout_qonehot=8,
                        readout_profile=True,
                    )
            # Per-token top-k lengths, as the graph op supplies them: rows
            # keep live entries past early tokens' lengths, which only the
            # length excludes.
            run_nvfp4_vs_fp8[8, skip=True, has_lengths=True](
                "skip_lengths_b2_s4", [3000, 900], 2048, ctx, 4
            )
            run_nvfp4_vs_fp8[32, skip=True, has_lengths=True](
                "skip_h32_lengths_b2", [3000, 900], 2048, ctx, 2
            )

            # Scale rows on both sides of 2**31: blocks from 2**24 - 58 on, at
            # 128 rows each (MODELS-1696).
            run_nvfp4_vs_fp8[8](
                "scale_rows_past_int32",
                [8191, 4096, 2300, 129],
                2048,
                ctx,
                scale_block_base=(1 << 24) - 58,
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
