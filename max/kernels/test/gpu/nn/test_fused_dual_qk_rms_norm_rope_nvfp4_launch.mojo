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
"""Bitwise test of the warp launch of
`fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged` against the row-per-block
launch it replaces (NVIDIA SM100).

Both launches run on one staged `[T, 9856]` projection at MiniMax-M3's shapes
(main Q 64 heads, K/V 4, IndexQ 4, IndexK 1, head_dim 128, rope_dim 64, NVFP4
main and index caches, FP8 Q outputs) and must write identical Q, IndexQ and
cache bytes, poison included. Cases cover the decode sizes, the sizes around
the dispatch threshold and the 16,384-token prefill plan with one or two
ragged rows and cached prefixes. The staging holds zero rows, tiny rows,
outliers, and a few non-finite values.
"""

from std.math import ceildiv
from std.sys.info import size_of

from internal_utils import arg_parse
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import Idx, RowMajorLayout, TileTensor, row_major
from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
from max.gpu import global_idx
from max.gpu.host import DeviceBuffer, DeviceContext
from nn.kv_cache import (
    NVFP4_QK_LAUNCH_ROW,
    NVFP4_QK_LAUNCH_WARP,
    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged,
)
from std.math import cos, sin
from std.testing import assert_equal
from std.utils.numerics import inf, nan

comptime dtype = DType.bfloat16
comptime freq_dtype = DType.float32
comptime q_out_dtype = DType.float8_e4m3fn
comptime head_dim = 128
comptime packed_dim = head_dim // 2
comptime sf_cols = head_dim // NVFP4_SF_VECTOR_SIZE
comptime page_size = 128
comptime q_heads = 64
comptime kv_heads = 4
comptime index_q_heads = 4
comptime index_k_heads = 1
comptime rope_dim = 64
# Columns of the staging row: Q | K | V | IndexQ | IndexK.
comptime col_k = q_heads * head_dim
comptime col_v = col_k + kv_heads * head_dim
comptime col_iq = col_v + kv_heads * head_dim
comptime col_ik = col_iq + index_q_heads * head_dim
comptime staging_cols = col_ik + index_k_heads * head_dim
comptime FILL_BLOCK = 256


def _fill_kernel(
    buf: MutPointer[Scalar[dtype], MutAnyOrigin], n: Int64, salt: UInt64
):
    var i = global_idx.x
    if Int64(i) < n:
        var h = (UInt64(i) + salt) * 0x9E3779B97F4A7C15
        h ^= h >> 29
        h *= 0xBF58476D1CE4E5B9
        h ^= h >> 32
        # Uniform in [-3, 3) with a few larger outliers so scales vary.
        var u = Float32(h & 0xFFFFF) / Float32(0x100000)
        var v = (u - 0.5) * 6.0
        if (h >> 40) & 0x3FF == 0:
            v *= 40.0
        # Row-wide edge cases: zeros, tiny values, and a few non-finite
        # elements; a row is `head_dim` elements.
        var row = i // head_dim
        if row % 61 == 1:
            v = 0.0
        elif row % 53 == 2:
            v *= 1e-4
        elif row % 4001 == 3 and i % head_dim == 5:
            v = inf[DType.float32]()
        elif row % 4003 == 4 and i % head_dim == 70:
            v = nan[DType.float32]()
        buf[i] = v.cast[dtype]()


def _fill(
    ctx: DeviceContext[], buf: DeviceBuffer[dtype], n: Int, salt: UInt64
) raises:
    ctx.enqueue_function[_fill_kernel](
        buf.unsafe_ptr(),
        Int64(n),
        salt,
        grid_dim=ceildiv(n, FILL_BLOCK),
        block_dim=FILL_BLOCK,
    )


def _differing_bytes[
    dt: DType
](ctx: DeviceContext[], a: DeviceBuffer[dt], b: DeviceBuffer[dt]) raises -> Int:
    """Number of bytes that differ between two device buffers."""
    var n = len(a)
    var ha = ctx.enqueue_create_host_buffer[dt](n)
    var hb = ctx.enqueue_create_host_buffer[dt](n)
    ctx.enqueue_copy(ha, a)
    ctx.enqueue_copy(hb, b)
    ctx.synchronize()
    var pa = ha.unsafe_ptr().unsafe_bitcast[UInt8]()
    var pb = hb.unsafe_ptr().unsafe_bitcast[UInt8]()
    var diffs = 0
    for i in range(n * size_of[dt]()):
        if pa[i] != pb[i]:
            if diffs < 12:
                print(
                    "  diff at byte",
                    i,
                    "row/col",
                    i // 128,
                    i % 128,
                    "row",
                    pa[i],
                    "warp",
                    pb[i],
                )
            diffs += 1
    return diffs


def blocks[
    dt: DType
](
    mut dev: DeviceBuffer[dt],
    num_blocks: Int,
    kv_dim: Int,
    heads: Int,
    width: Int,
) -> TileTensor[dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], MutAnyOrigin]:
    return TileTensor(
        dev, row_major(num_blocks, kv_dim, 1, page_size, heads, width)
    ).as_unsafe_any_origin()


def run_case(
    ctx: DeviceContext[],
    tokens: Int,
    rows: Int,
    cache_len: Int,
) raises:
    var max_seq = (cache_len * rows + tokens) + page_size
    var base = tokens // rows
    var lens = List[Int]()
    for r in range(rows):
        lens.append(base if r < rows - 1 else tokens - base * (rows - 1))

    var lut_cols = 0
    var cache_lens = List[Int]()
    for r in range(rows):
        var c = cache_len * (r + 1)
        cache_lens.append(c)
        lut_cols = max(lut_cols, ceildiv(c + lens[r], page_size))
    var num_blocks = 0
    for r in range(rows):
        num_blocks += (
            ceildiv(cache_lens[r] + lens[r], page_size)
            - cache_lens[r] // page_size
            + 1
        )
    num_blocks += 2

    var row_offsets_h = ctx.enqueue_create_host_buffer[DType.uint32](rows + 1)
    var cache_lengths_h = ctx.enqueue_create_host_buffer[DType.uint32](rows)
    var lut_h = ctx.enqueue_create_host_buffer[DType.uint32](rows * lut_cols)
    ctx.synchronize()
    var off = 0
    var next_block = 0
    for r in range(rows):
        row_offsets_h[r] = UInt32(off)
        cache_lengths_h[r] = UInt32(cache_lens[r])
        off += lens[r]
        for c in range(lut_cols):
            lut_h[r * lut_cols + c] = UInt32(num_blocks)
        # Only the pages that receive new tokens get a block.
        var first = cache_lens[r] // page_size
        var last = ceildiv(cache_lens[r] + lens[r], page_size)
        for c in range(first, last):
            lut_h[r * lut_cols + c] = UInt32(next_block)
            next_block += 1
    row_offsets_h[rows] = UInt32(off)
    var row_offsets_d = ctx.enqueue_create_buffer[DType.uint32](rows + 1)
    var cache_lengths_d = ctx.enqueue_create_buffer[DType.uint32](rows)
    var lut_d = ctx.enqueue_create_buffer[DType.uint32](rows * lut_cols)
    ctx.enqueue_copy(row_offsets_d, row_offsets_h)
    ctx.enqueue_copy(cache_lengths_d, cache_lengths_h)
    ctx.enqueue_copy(lut_d, lut_h)

    var freq_n = max_seq * rope_dim
    var freqs_d = ctx.enqueue_create_buffer[freq_dtype](freq_n)
    var freqs_h = ctx.enqueue_create_host_buffer[freq_dtype](freq_n)
    ctx.synchronize()
    for p in range(max_seq):
        for j in range(rope_dim // 2):
            # cos/sin of an angle that varies with position and pair.
            var ang = (
                Float32(p % 4096)
                * Float32(0.01)
                * Float32(1.0 / Float32(1 + j))
            )
            freqs_h[p * rope_dim + 2 * j] = cos(ang)
            freqs_h[p * rope_dim + 2 * j + 1] = sin(ang)
    ctx.enqueue_copy(freqs_d, freqs_h)

    var gammas = List[DeviceBuffer[dtype]]()
    for g in range(4):
        var gh = ctx.enqueue_create_host_buffer[dtype](head_dim)
        ctx.synchronize()
        for d in range(head_dim):
            gh[d] = Scalar[dtype](Float32(((d * 7 + g * 13) % 17) - 8) / 16.0)
        var gd = ctx.enqueue_create_buffer[dtype](head_dim)
        ctx.enqueue_copy(gd, gh)
        ctx.synchronize()
        gammas.append(gd)

    var staging_n = tokens * staging_cols
    var staging_d = ctx.enqueue_create_buffer[dtype](staging_n)
    _fill(ctx, staging_d, staging_n, 12345)

    var q_main_n = tokens * q_heads * head_dim
    var q_index_n = tokens * index_q_heads * head_dim
    var q_main_a = ctx.enqueue_create_buffer[q_out_dtype](q_main_n)
    var q_main_b = ctx.enqueue_create_buffer[q_out_dtype](q_main_n)
    var q_index_a = ctx.enqueue_create_buffer[q_out_dtype](q_index_n)
    var q_index_b = ctx.enqueue_create_buffer[q_out_dtype](q_index_n)

    var main_n = num_blocks * 2 * page_size * kv_heads * packed_dim
    var main_sf_n = num_blocks * 2 * page_size * kv_heads * sf_cols
    var idx_n = num_blocks * page_size * index_k_heads * packed_dim
    var idx_sf_n = num_blocks * page_size * index_k_heads * sf_cols
    var main_a = ctx.enqueue_create_buffer[DType.uint8](main_n)
    var main_b = ctx.enqueue_create_buffer[DType.uint8](main_n)
    var main_sf_a = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](main_sf_n)
    var main_sf_b = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](main_sf_n)
    var idx_a = ctx.enqueue_create_buffer[DType.uint8](idx_n)
    var idx_b = ctx.enqueue_create_buffer[DType.uint8](idx_n)
    var idx_sf_a = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](idx_sf_n)
    var idx_sf_b = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](idx_sf_n)
    ctx.enqueue_memset(q_main_a, Scalar[q_out_dtype](0))
    ctx.enqueue_memset(q_main_b, Scalar[q_out_dtype](0))
    ctx.enqueue_memset(q_index_a, Scalar[q_out_dtype](0))
    ctx.enqueue_memset(q_index_b, Scalar[q_out_dtype](0))
    ctx.enqueue_memset(main_a, UInt8(0xAB))
    ctx.enqueue_memset(main_b, UInt8(0xAB))
    ctx.enqueue_memset(idx_a, UInt8(0xAB))
    ctx.enqueue_memset(idx_b, UInt8(0xAB))
    ctx.enqueue_memset(main_sf_a, Scalar[NVFP4_SF_DTYPE](0))
    ctx.enqueue_memset(main_sf_b, Scalar[NVFP4_SF_DTYPE](0))
    ctx.enqueue_memset(idx_sf_a, Scalar[NVFP4_SF_DTYPE](0))
    ctx.enqueue_memset(idx_sf_b, Scalar[NVFP4_SF_DTYPE](0))
    ctx.synchronize()

    var row_offsets_tt = TileTensor(row_offsets_d, row_major(rows + 1))
    var freqs_tt = TileTensor(freqs_d, row_major((max_seq, Idx[rope_dim])))
    var g_q_main = TileTensor(gammas[0], row_major[head_dim]())
    var g_k_main = TileTensor(gammas[1], row_major[head_dim]())
    var g_q_index = TileTensor(gammas[2], row_major[head_dim]())
    var g_k_index = TileTensor(gammas[3], row_major[head_dim]())
    var staging_tt = TileTensor[mut=False](
        staging_d, row_major((tokens, Idx[staging_cols]))
    ).as_unsafe_any_origin()
    var cache_lengths_tt = TileTensor[mut=False](
        cache_lengths_d, row_major(rows)
    ).as_unsafe_any_origin()
    var lut_tt = TileTensor[mut=False](
        lut_d, row_major((rows, lut_cols))
    ).as_unsafe_any_origin()

    var q_main_a_tt = TileTensor(
        q_main_a, row_major((tokens, Idx[q_heads], Idx[head_dim]))
    )
    var q_main_b_tt = TileTensor(
        q_main_b, row_major((tokens, Idx[q_heads], Idx[head_dim]))
    )
    var q_index_a_tt = TileTensor(
        q_index_a, row_major((tokens, Idx[index_q_heads], Idx[head_dim]))
    )
    var q_index_b_tt = TileTensor(
        q_index_b, row_major((tokens, Idx[index_q_heads], Idx[head_dim]))
    )

    var max_prompt = UInt32(max(base, tokens - base * (rows - 1)))
    var max_cache = UInt32(cache_len * rows)

    var main_coll_a = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(num_heads=kv_heads, head_size=packed_dim),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks(main_a, num_blocks, 2, kv_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        max_prompt,
        max_cache,
        scales=blocks(main_sf_a, num_blocks, 2, kv_heads, sf_cols),
        scales_lookup_table=lut_tt,
    )
    var main_coll_b = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(num_heads=kv_heads, head_size=packed_dim),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks(main_b, num_blocks, 2, kv_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        max_prompt,
        max_cache,
        scales=blocks(main_sf_b, num_blocks, 2, kv_heads, sf_cols),
        scales_lookup_table=lut_tt,
    )
    var index_coll_a = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(
            num_heads=index_k_heads, head_size=packed_dim, is_mla=True
        ),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks(idx_a, num_blocks, 1, index_k_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        max_prompt,
        max_cache,
        scales=blocks(idx_sf_a, num_blocks, 1, index_k_heads, sf_cols),
        scales_lookup_table=lut_tt,
    )
    var index_coll_b = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(
            num_heads=index_k_heads, head_size=packed_dim, is_mla=True
        ),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks(idx_b, num_blocks, 1, index_k_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        max_prompt,
        max_cache,
        scales=blocks(idx_sf_b, num_blocks, 1, index_k_heads, sf_cols),
        scales_lookup_table=lut_tt,
    )

    @inline(.always)
    def q_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var staging_tt} -> SIMD[dtype, width]:
        return staging_tt.load[width=width]((token, head * head_dim + col))

    @inline(.always)
    def k_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var staging_tt} -> SIMD[dtype, width]:
        return staging_tt.load[width=width](
            (token, col_k + head * head_dim + col)
        )

    @inline(.always)
    def v_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var staging_tt} -> SIMD[dtype, width]:
        return staging_tt.load[width=width](
            (token, col_v + head * head_dim + col)
        )

    @inline(.always)
    def q_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var staging_tt} -> SIMD[dtype, width]:
        return staging_tt.load[width=width](
            (token, col_iq + head * head_dim + col)
        )

    @inline(.always)
    def k_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var staging_tt} -> SIMD[dtype, width]:
        return staging_tt.load[width=width](
            (token, col_ik + head * head_dim + col)
        )

    @inline(.always)
    def launch_row(c: DeviceContext[]) raises {imm}:
        fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
            multiply_before_cast=True,
            interleaved=False,
            launch=NVFP4_QK_LAUNCH_ROW,
        ](
            main_coll_a,
            index_coll_a,
            g_q_main,
            g_k_main,
            g_q_index,
            g_k_index,
            freqs_tt,
            Float32(1e-6),
            Float32(1e-5),
            Scalar[dtype](1.0),
            UInt32(0),
            row_offsets_tt,
            q_main_fn,
            k_main_fn,
            v_main_fn,
            q_index_fn,
            k_index_fn,
            q_main_a_tt,
            q_index_a_tt,
            c,
        )

    @inline(.always)
    def launch_warp(c: DeviceContext[]) raises {imm}:
        fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
            multiply_before_cast=True,
            interleaved=False,
            launch=NVFP4_QK_LAUNCH_WARP,
        ](
            main_coll_b,
            index_coll_b,
            g_q_main,
            g_k_main,
            g_q_index,
            g_k_index,
            freqs_tt,
            Float32(1e-6),
            Float32(1e-5),
            Scalar[dtype](1.0),
            UInt32(0),
            row_offsets_tt,
            q_main_fn,
            k_main_fn,
            v_main_fn,
            q_index_fn,
            k_index_fn,
            q_main_b_tt,
            q_index_b_tt,
            c,
        )

    launch_row(ctx)
    launch_warp(ctx)
    ctx.synchronize()

    var diffs = 0
    diffs += _differing_bytes(ctx, q_main_a, q_main_b)
    diffs += _differing_bytes(ctx, q_index_a, q_index_b)
    diffs += _differing_bytes(ctx, main_a, main_b)
    diffs += _differing_bytes(ctx, main_sf_a, main_sf_b)
    diffs += _differing_bytes(ctx, idx_a, idx_b)
    diffs += _differing_bytes(ctx, idx_sf_a, idx_sf_b)

    print(t"T={tokens} rows={rows} cache_len={cache_len}: byte diffs {diffs}")
    assert_equal(diffs, 0)
    _ = staging_d^


def main() raises:
    with DeviceContext() as ctx:
        # Decode sizes: 8 speculative tokens per row.
        run_case(ctx, 1, 1, 1000)
        run_case(ctx, 16, 2, 1000)
        run_case(ctx, 64, 8, 1000)
        run_case(ctx, 128, 16, 1000)
        run_case(ctx, 512, 64, 1000)
        # Around the threshold, with a ragged last warp block.
        run_case(ctx, 2047, 3, 129)
        run_case(ctx, 2049, 1, 0)
        # Prefill plans: u60 chunks and a long cached prefix.
        run_case(ctx, 16384, 1, 0)
        run_case(ctx, 16384, 2, 16384)
        run_case(ctx, 16384, 1, 320000)
