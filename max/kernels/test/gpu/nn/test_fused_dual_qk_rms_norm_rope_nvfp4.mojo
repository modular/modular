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
"""Bitwise test for `fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged`.

The reference is the in-place dual op on a bf16 main cache seeded with the raw
K. The staging op must reproduce its Q, IndexQ and IndexK exactly, and its
NVFP4 K and V must equal MiniMax's quantizer applied on the host to the
reference's post-RoPE K and to the raw V. Every cache byte the op does not own
must keep its poison.
"""

from std.collections import Set
from std.math import ceildiv
from std.memory import bitcast
from std.random import random_float64, random_ui64, seed

from max.gpu.host import DeviceBuffer, DeviceContext, HostBuffer
from kv_cache.types import KVCacheStaticParams, PagedKVCacheCollection
from layout import (
    Coord,
    Idx,
    RowMajorLayout,
    TileTensor,
    row_major,
)
from linalg.fp4_utils import NVFP4_SF_DTYPE, NVFP4_SF_VECTOR_SIZE
from nn.kv_cache import (
    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged,
    fused_dual_qk_rms_norm_rope_ragged_paged,
)
from std.testing import assert_equal, assert_true
from std.utils import Index
from std.utils.numerics import inf, nan

comptime dtype = DType.bfloat16
comptime freq_dtype = DType.float32
comptime q_main_out_dtype = DType.float8_e4m3fn
comptime head_dim = 128
comptime packed_dim = head_dim // 2
comptime sf_cols = head_dim // NVFP4_SF_VECTOR_SIZE
comptime page_size = 128
comptime num_layers = 2
comptime layer_idx = 1
comptime max_seq_len = 1024
comptime index_q_heads = 4
comptime index_k_heads = 1
comptime value_poison = UInt8(0xAB)
# An unwritten scale reads as NaN.
comptime scale_poison = UInt8(0x7F)


def e2m1_magnitude_rne(a: Float32) -> UInt8:
    """MiniMax's round-half-to-even ladder on `a = min(|x / scale|, 6)`."""
    if a > 5.0:
        return 7
    if a >= 3.5:
        return 6
    if a > 2.5:
        return 5
    if a >= 1.75:
        return 4
    if a > 1.25:
        return 3
    if a >= 0.75:
        return 2
    if a > 0.25:
        return 1
    return 0


def nvfp4_group_scale(amax: Float32) -> Float32:
    """`E4M3(max(amax, 1e-12) / 6)` clamped to [2^-9, 448]; the cast
    saturates."""
    var scale = (
        (max(amax, Float32(1e-12)) / 6.0)
        .cast[NVFP4_SF_DTYPE]()
        .cast[DType.float32]()
    )
    return min(Float32(448.0), max(scale, Float32(1.0 / 512.0)))


def quantize_nvfp4_row(
    x: List[Float32], mut packed: List[UInt8], mut scales: List[UInt8]
):
    """Quantizes one `head_dim` row with MiniMax's formula (tensor scale 1):
    element 2i in the low nibble, zero magnitude stored as +0."""
    packed.clear()
    scales.clear()
    for g in range(sf_cols):
        var base = g * NVFP4_SF_VECTOR_SIZE
        var amax = Float32(0)
        for e in range(NVFP4_SF_VECTOR_SIZE):
            amax = max(amax, abs(x[base + e]))
        var scale = nvfp4_group_scale(amax)
        scales.append(bitcast_e4m3(scale.cast[NVFP4_SF_DTYPE]()))
        for e in range(0, NVFP4_SF_VECTOR_SIZE, 2):
            var byte = UInt8(0)
            for h in range(2):
                var xs = x[base + e + h] / scale
                var mag = e2m1_magnitude_rne(min(abs(xs), Float32(6.0)))
                var code = mag | (UInt8(8) if xs < 0 and mag != 0 else 0)
                byte |= code << UInt8(4 * h)
            packed.append(byte)


def bitcast_e4m3(v: Scalar[NVFP4_SF_DTYPE]) -> UInt8:
    return bitcast[DType.uint8, 1](v)


def edge_row_value(kind: Int, d: Int, v: Float32) -> Float32:
    """Plants the quantizer's edge cases by `kind = row % 8`.

    1: zeros (scale floor, all +0). 2: x 1e-2 (subnormal E4M3 scales).
    3: x 1e-3 (scale clamped to 2^-9). 4: every ladder midpoint at scale 2^-2
    (ties to even). 5: amax 6.3125, whose scale rounds down so |x / scale|
    saturates at 6. 6: an outlier per group (scale clamped to 448).
    """
    var ties = SIMD[DType.float32, 8](
        6.0, 5.0, 3.5, 2.5, 1.75, 1.25, 0.75, 0.25
    )
    var e = d % NVFP4_SF_VECTOR_SIZE
    if kind == 1:
        return 0.0
    if kind == 2:
        return v * 1e-2
    if kind == 3:
        return v * 1e-3
    if kind == 4:
        return ties[e % 8] * Float32(0.25 if d % 3 else -0.25)
    if kind == 5 and e == 0:
        return 6.3125
    if kind == 6 and e == 0:
        return 3000.0
    return v


def fill_staging(
    mut buf: HostBuffer[dtype], rows: Int, heads: Int, edge_rows: Bool
):
    for r in range(rows):
        for h in range(heads):
            for d in range(head_dim):
                var v = Float32(random_float64(-2.0, 2.0))
                if edge_rows:
                    v = edge_row_value((r * heads + h) % 8, d, v)
                buf[(r * heads + h) * head_dim + d] = v.cast[dtype]()


def cache_offset(
    block: Int,
    kv: Int,
    kv_dim: Int,
    pos: Int,
    head: Int,
    heads: Int,
    width: Int,
    page_elems: Int = -1,
) -> Int:
    """Element offset of a cache row; `page_elems` is a padded page stride
    (-1 for packed pages)."""
    var inner = (
        ((kv * num_layers + layer_idx) * page_size + pos) * heads + head
    ) * width
    var page = (
        kv_dim * num_layers * page_size * heads * width if page_elems
        < 0 else page_elems
    )
    return block * page + inner


def nonfinite_group_codes(
    row: List[Float32], group: Int, mut packed: List[UInt8]
):
    """The bytes a group holding a +inf gets: scale 448, inf to code 7, every
    finite element through the ladder at scale 448."""
    for e in range(0, NVFP4_SF_VECTOR_SIZE, 2):
        var byte = UInt8(0)
        for h in range(2):
            var x = row[group * NVFP4_SF_VECTOR_SIZE + e + h]
            var mag = e2m1_magnitude_rne(min(abs(x) / 448.0, Float32(6.0)))
            var code = mag | (UInt8(8) if x < 0 and mag != 0 else 0)
            byte |= code << UInt8(4 * h)
        packed[group * (NVFP4_SF_VECTOR_SIZE // 2) + e // 2] = byte


def blocks_tt[
    dt: DType
](
    mut dev: DeviceBuffer[dt],
    num_blocks: Int,
    kv_dim: Int,
    heads: Int,
    width: Int,
) -> TileTensor[dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], MutAnyOrigin]:
    return TileTensor(
        dev, row_major(num_blocks, kv_dim, num_layers, page_size, heads, width)
    ).as_unsafe_any_origin()


def run_nvfp4_dual[
    main_q_heads: Int,
    main_k_heads: Int,
    rope_dim: Int = 64,
](
    ctx: DeviceContext,
    value_pad_rows: Int = 0,
    scale_pad_rows: Int = 0,
    separate_scales_lut: Bool = False,
    nonfinite: Bool = False,
) raises:
    """Checks one launch. `value_pad_rows` / `scale_pad_rows` pad each NVFP4
    value / scale page (a padded `page_stride`), `separate_scales_lut` pages
    the scales through their own reversed table, and `nonfinite` plants a +inf
    group and a NaN group in one V row."""
    print(
        "== run_nvfp4_dual main=",
        main_q_heads,
        "/",
        main_k_heads,
        " rope_dim=",
        rope_dim,
        " pad=",
        value_pad_rows,
        "/",
        scale_pad_rows,
        " separate_scales_lut=",
        separate_scales_lut,
        " nonfinite=",
        nonfinite,
        sep="",
    )
    seed(42)
    var prompt_lens: List[Int] = [17, 1, 31, 130, 3]
    var cache_lens: List[Int] = [0, 127, 13, 250, 6]
    var batch_size = len(prompt_lens)

    var total_length = 0
    var max_prompt_length = 0
    var max_cache_length = 0
    var max_context = 0
    for i in range(batch_size):
        total_length += prompt_lens[i]
        max_prompt_length = max(max_prompt_length, prompt_lens[i])
        max_cache_length = max(max_cache_length, cache_lens[i])
        max_context = max(max_context, cache_lens[i] + prompt_lens[i])

    var lut_cols = ceildiv(max_context, page_size)
    var num_blocks = 0
    for i in range(batch_size):
        num_blocks += ceildiv(cache_lens[i] + prompt_lens[i], page_size)
    num_blocks += 4

    # Shared metadata. The LUT hands out distinct random blocks.
    var row_offsets_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size + 1
    )
    var cache_lengths_host = ctx.enqueue_create_host_buffer[.uint32](batch_size)
    var lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.synchronize()
    var offset = 0
    var used = Set[Int]()
    for i in range(batch_size):
        row_offsets_host[i] = UInt32(offset)
        cache_lengths_host[i] = UInt32(cache_lens[i])
        offset += prompt_lens[i]
        for c in range(lut_cols):
            lut_host[i * lut_cols + c] = UInt32(num_blocks)
        for c in range(ceildiv(cache_lens[i] + prompt_lens[i], page_size)):
            var b = Int(random_ui64(0, UInt64(num_blocks - 1)))
            while b in used:
                b = Int(random_ui64(0, UInt64(num_blocks - 1)))
            used.add(b)
            lut_host[i * lut_cols + c] = UInt32(b)
    row_offsets_host[batch_size] = UInt32(offset)

    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](batch_size + 1)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](batch_size)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](batch_size * lut_cols)
    ctx.enqueue_copy(row_offsets_dev, row_offsets_host)
    ctx.enqueue_copy(cache_lengths_dev, cache_lengths_host)
    ctx.enqueue_copy(lut_dev, lut_host)

    # RoPE table and gammas.
    var freqs_host = ctx.enqueue_create_host_buffer[freq_dtype](
        max_seq_len * rope_dim
    )
    ctx.synchronize()
    for i in range(max_seq_len * rope_dim):
        freqs_host[i] = Float32(random_float64(-1.0, 1.0))
    var freqs_dev = ctx.enqueue_create_buffer[freq_dtype](
        max_seq_len * rope_dim
    )
    ctx.enqueue_copy(freqs_dev, freqs_host)
    var gamma_devs = List[DeviceBuffer[dtype]]()
    for _ in range(4):
        var gamma_host = ctx.enqueue_create_host_buffer[dtype](head_dim)
        ctx.synchronize()
        for d in range(head_dim):
            gamma_host[d] = Float32(random_float64(-1.0, 1.0)).cast[dtype]()
        var dev = ctx.enqueue_create_buffer[dtype](head_dim)
        ctx.enqueue_copy(dev, gamma_host)
        ctx.synchronize()
        gamma_devs.append(dev)

    # Staging: what the projection writes.
    var q_main_n = total_length * main_q_heads * head_dim
    var kv_main_n = total_length * main_k_heads * head_dim
    var q_index_n = total_length * index_q_heads * head_dim
    var k_index_n = total_length * index_k_heads * head_dim
    var q_main_host = ctx.enqueue_create_host_buffer[dtype](q_main_n)
    var k_main_host = ctx.enqueue_create_host_buffer[dtype](kv_main_n)
    var v_main_host = ctx.enqueue_create_host_buffer[dtype](kv_main_n)
    var q_index_host = ctx.enqueue_create_host_buffer[dtype](q_index_n)
    var k_index_host = ctx.enqueue_create_host_buffer[dtype](k_index_n)
    ctx.synchronize()
    fill_staging(q_main_host, total_length, main_q_heads, False)
    fill_staging(k_main_host, total_length, main_k_heads, True)
    fill_staging(v_main_host, total_length, main_k_heads, True)
    # Token 1 (the single-token request ending on a page boundary), head 0.
    comptime nf_tok = 1
    if nonfinite:
        var base = nf_tok * main_k_heads * head_dim
        v_main_host[base + 16 + 3] = inf[dtype]()
        v_main_host[base + 32 + 5] = nan[dtype]()
    fill_staging(q_index_host, total_length, index_q_heads, False)
    fill_staging(k_index_host, total_length, index_k_heads, False)
    var q_main_dev = ctx.enqueue_create_buffer[dtype](q_main_n)
    var k_main_dev = ctx.enqueue_create_buffer[dtype](kv_main_n)
    var v_main_dev = ctx.enqueue_create_buffer[dtype](kv_main_n)
    var q_index_dev = ctx.enqueue_create_buffer[dtype](q_index_n)
    var k_index_dev = ctx.enqueue_create_buffer[dtype](k_index_n)
    ctx.enqueue_copy(q_main_dev, q_main_host)
    ctx.enqueue_copy(k_main_dev, k_main_host)
    ctx.enqueue_copy(v_main_dev, v_main_host)
    ctx.enqueue_copy(q_index_dev, q_index_host)
    ctx.enqueue_copy(k_index_dev, k_index_host)

    # Reference caches hold the raw K / IndexK at each new token's slot.
    var main_ref_n = (
        num_blocks * 2 * num_layers * page_size * main_k_heads * (head_dim)
    )
    var index_n = num_blocks * num_layers * page_size * index_k_heads * head_dim
    var main_ref_host = ctx.enqueue_create_host_buffer[dtype](main_ref_n)
    var index_ref_host = ctx.enqueue_create_host_buffer[dtype](index_n)
    ctx.synchronize()
    for i in range(main_ref_n):
        main_ref_host[i] = BFloat16(0)
    for i in range(index_n):
        index_ref_host[i] = BFloat16(-7.0)
    var tok = 0
    for bs in range(batch_size):
        for t in range(prompt_lens[bs]):
            var pos = cache_lens[bs] + t
            var block = Int(lut_host[bs * lut_cols + pos // page_size])
            for h in range(main_k_heads):
                var dst = cache_offset(
                    block, 0, 2, pos % page_size, h, main_k_heads, head_dim
                )
                for d in range(head_dim):
                    main_ref_host[dst + d] = k_main_host[
                        (tok * main_k_heads + h) * head_dim + d
                    ]
            var dst = cache_offset(
                block, 0, 1, pos % page_size, 0, index_k_heads, head_dim
            )
            for d in range(head_dim):
                index_ref_host[dst + d] = k_index_host[tok * head_dim + d]
            tok += 1
    var main_ref_dev = ctx.enqueue_create_buffer[dtype](main_ref_n)
    var index_ref_dev = ctx.enqueue_create_buffer[dtype](index_n)
    ctx.enqueue_copy(main_ref_dev, main_ref_host)
    ctx.enqueue_copy(index_ref_dev, index_ref_host)

    # Caches under test, poisoned (padding included).
    var value_page = (2 * num_layers * page_size + value_pad_rows) * (
        main_k_heads * packed_dim
    )
    var scale_page = (2 * num_layers * page_size + scale_pad_rows) * (
        main_k_heads * sf_cols
    )
    var main4_n = num_blocks * value_page
    var sf_n = num_blocks * scale_page
    # Scale page ids: reversed when the scales have their own table.
    var sf_lut_host = ctx.enqueue_create_host_buffer[.uint32](
        batch_size * lut_cols
    )
    ctx.synchronize()
    for i in range(batch_size * lut_cols):
        var b = Int(lut_host[i])
        sf_lut_host[i] = UInt32(
            num_blocks - 1 - b if separate_scales_lut and b < num_blocks else b
        )
    var sf_lut_dev = ctx.enqueue_create_buffer[.uint32](batch_size * lut_cols)
    ctx.enqueue_copy(sf_lut_dev, sf_lut_host)
    var main4_host = ctx.enqueue_create_host_buffer[.uint8](main4_n)
    var sf_host = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sf_n)
    ctx.synchronize()
    for i in range(main4_n):
        main4_host[i] = value_poison
    var sf_host_bytes = sf_host.unsafe_ptr().bitcast[UInt8]()
    for i in range(sf_n):
        sf_host_bytes[i] = scale_poison
    var main4_dev = ctx.enqueue_create_buffer[.uint8](main4_n)
    var sf_dev = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](sf_n)
    var index_dev = ctx.enqueue_create_buffer[dtype](index_n)
    ctx.enqueue_copy(main4_dev, main4_host)
    ctx.enqueue_copy(sf_dev, sf_host)
    ctx.enqueue_copy(index_dev, index_ref_host)

    var q_main_ref_dev = ctx.enqueue_create_buffer[q_main_out_dtype](q_main_n)
    var q_main_out_dev = ctx.enqueue_create_buffer[q_main_out_dtype](q_main_n)
    var q_index_ref_dev = ctx.enqueue_create_buffer[dtype](q_index_n)
    var q_index_out_dev = ctx.enqueue_create_buffer[dtype](q_index_n)

    var row_offsets_tt = TileTensor(row_offsets_dev, row_major(batch_size + 1))
    var freqs_tt = TileTensor(freqs_dev, row_major[max_seq_len, rope_dim]())
    var g_q_main = TileTensor(gamma_devs[0], row_major[head_dim]())
    var g_k_main = TileTensor(gamma_devs[1], row_major[head_dim]())
    var g_q_index = TileTensor(gamma_devs[2], row_major[head_dim]())
    var g_k_index = TileTensor(gamma_devs[3], row_major[head_dim]())

    var q_main_tt = TileTensor(
        q_main_dev, row_major((total_length, Idx[main_q_heads], Idx[head_dim]))
    )
    var k_main_tt = TileTensor(
        k_main_dev, row_major((total_length, Idx[main_k_heads], Idx[head_dim]))
    )
    var v_main_tt = TileTensor(
        v_main_dev, row_major((total_length, Idx[main_k_heads], Idx[head_dim]))
    )
    var q_index_tt = TileTensor(
        q_index_dev,
        row_major((total_length, Idx[index_q_heads], Idx[head_dim])),
    )
    var k_index_tt = TileTensor(
        k_index_dev,
        row_major((total_length, Idx[index_k_heads], Idx[head_dim])),
    )
    var q_main_ref_tt = TileTensor(
        q_main_ref_dev,
        row_major((total_length, Idx[main_q_heads], Idx[head_dim])),
    )
    var q_main_out_tt = TileTensor(
        q_main_out_dev,
        row_major((total_length, Idx[main_q_heads], Idx[head_dim])),
    )
    var q_index_ref_tt = TileTensor(
        q_index_ref_dev,
        row_major((total_length, Idx[index_q_heads], Idx[head_dim])),
    )
    var q_index_out_tt = TileTensor(
        q_index_out_dev,
        row_major((total_length, Idx[index_q_heads], Idx[head_dim])),
    )

    var cache_lengths_tt = TileTensor[mut=False](
        cache_lengths_dev, row_major(batch_size)
    ).as_unsafe_any_origin()
    var lut_tt = TileTensor[mut=False](
        lut_dev, row_major((batch_size, lut_cols))
    ).as_unsafe_any_origin()
    var sf_lut_tt = TileTensor[mut=False](
        sf_lut_dev, row_major((batch_size, lut_cols))
    ).as_unsafe_any_origin()

    var main_ref_coll = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=main_k_heads, head_size=head_dim),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(main_ref_dev, num_blocks, 2, main_k_heads, head_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var index_ref_coll = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=index_k_heads, head_size=head_dim, is_mla=True
        ),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(index_ref_dev, num_blocks, 1, index_k_heads, head_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var main4_coll = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(num_heads=main_k_heads, head_size=packed_dim),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks_tt(main4_dev, num_blocks, 2, main_k_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
        scales=blocks_tt(sf_dev, num_blocks, 2, main_k_heads, sf_cols),
        scales_lookup_table=sf_lut_tt,
        page_stride=value_page if value_pad_rows else -1,
        scales_page_stride=scale_page if scale_pad_rows else -1,
    )
    var index_coll = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(
            num_heads=index_k_heads, head_size=head_dim, is_mla=True
        ),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(index_dev, num_blocks, 1, index_k_heads, head_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )

    @inline(.always)
    def q_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_main_tt} -> SIMD[dtype, width]:
        return q_main_tt.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def k_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var k_main_tt} -> SIMD[dtype, width]:
        return k_main_tt.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def v_main_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var v_main_tt} -> SIMD[dtype, width]:
        return v_main_tt.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def q_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var q_index_tt} -> SIMD[dtype, width]:
        return q_index_tt.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def k_index_fn[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var k_index_tt} -> SIMD[dtype, width]:
        return k_index_tt.load[width=width](Coord(Index(token, head, col)))

    var main_epsilon = Float32(1e-6)
    var index_epsilon = Float32(1e-5)

    fused_dual_qk_rms_norm_rope_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        main_ref_coll,
        index_ref_coll,
        g_q_main,
        g_k_main,
        g_q_index,
        g_k_index,
        freqs_tt,
        main_epsilon,
        index_epsilon,
        Scalar[dtype](1.0),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_fn,
        q_index_fn,
        q_main_ref_tt,
        q_index_ref_tt,
        ctx,
    )
    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        main4_coll,
        index_coll,
        g_q_main,
        g_k_main,
        g_q_index,
        g_k_index,
        freqs_tt,
        main_epsilon,
        index_epsilon,
        Scalar[dtype](1.0),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_fn,
        k_main_fn,
        v_main_fn,
        q_index_fn,
        k_index_fn,
        q_main_out_tt,
        q_index_out_tt,
        ctx,
    )

    # The same launch into an NVFP4 index-K cache (K-only, so one kv slot per
    # page), padded and paged like the main cache, with an fp8 IndexQ out.
    # Main K/V get the same bytes again, so the main-cache checks below are
    # unaffected.
    var idx_value_page = (num_layers * page_size + value_pad_rows) * (
        index_k_heads * packed_dim
    )
    var idx_scale_page = (num_layers * page_size + scale_pad_rows) * (
        index_k_heads * sf_cols
    )
    var idx4_n = num_blocks * idx_value_page
    var idx_sf_n = num_blocks * idx_scale_page
    var idx4_host = ctx.enqueue_create_host_buffer[.uint8](idx4_n)
    var idx_sf_host = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](idx_sf_n)
    ctx.synchronize()
    for i in range(idx4_n):
        idx4_host[i] = value_poison
    var idx_sf_host_bytes = idx_sf_host.unsafe_ptr().bitcast[UInt8]()
    for i in range(idx_sf_n):
        idx_sf_host_bytes[i] = scale_poison
    var idx4_dev = ctx.enqueue_create_buffer[.uint8](idx4_n)
    var idx_sf_dev = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](idx_sf_n)
    ctx.enqueue_copy(idx4_dev, idx4_host)
    ctx.enqueue_copy(idx_sf_dev, idx_sf_host)
    var idx4_coll = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(
            num_heads=index_k_heads, head_size=packed_dim, is_mla=True
        ),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks_tt(idx4_dev, num_blocks, 1, index_k_heads, packed_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
        scales=blocks_tt(idx_sf_dev, num_blocks, 1, index_k_heads, sf_cols),
        scales_lookup_table=sf_lut_tt,
        page_stride=idx_value_page if value_pad_rows else -1,
        scales_page_stride=idx_scale_page if scale_pad_rows else -1,
    )
    var q_index_out2_dev = ctx.enqueue_create_buffer[q_main_out_dtype](
        q_index_n
    )
    var q_index_out2_tt = TileTensor(
        q_index_out2_dev,
        row_major((total_length, Idx[index_q_heads], Idx[head_dim])),
    )
    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        main4_coll,
        idx4_coll,
        g_q_main,
        g_k_main,
        g_q_index,
        g_k_index,
        freqs_tt,
        main_epsilon,
        index_epsilon,
        Scalar[dtype](1.0),
        UInt32(layer_idx),
        row_offsets_tt,
        q_main_fn,
        k_main_fn,
        v_main_fn,
        q_index_fn,
        k_index_fn,
        q_main_out_tt,
        q_index_out2_tt,
        ctx,
    )
    var idx4_out = ctx.enqueue_create_host_buffer[.uint8](idx4_n)
    var idx_sf_out = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](idx_sf_n)
    var q_index_out2 = ctx.enqueue_create_host_buffer[q_main_out_dtype](
        q_index_n
    )
    ctx.enqueue_copy(idx4_out, idx4_dev)
    ctx.enqueue_copy(idx_sf_out, idx_sf_dev)
    ctx.enqueue_copy(q_index_out2, q_index_out2_dev)

    var q_main_ref_out = ctx.enqueue_create_host_buffer[q_main_out_dtype](
        q_main_n
    )
    var q_main_out = ctx.enqueue_create_host_buffer[q_main_out_dtype](q_main_n)
    var q_index_ref_out = ctx.enqueue_create_host_buffer[dtype](q_index_n)
    var q_index_out = ctx.enqueue_create_host_buffer[dtype](q_index_n)
    var main_ref_out = ctx.enqueue_create_host_buffer[dtype](main_ref_n)
    var index_ref_out = ctx.enqueue_create_host_buffer[dtype](index_n)
    var index_out = ctx.enqueue_create_host_buffer[dtype](index_n)
    var main4_out = ctx.enqueue_create_host_buffer[.uint8](main4_n)
    var sf_out = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sf_n)
    ctx.enqueue_copy(q_main_ref_out, q_main_ref_dev)
    ctx.enqueue_copy(q_main_out, q_main_out_dev)
    ctx.enqueue_copy(q_index_ref_out, q_index_ref_dev)
    ctx.enqueue_copy(q_index_out, q_index_out_dev)
    ctx.enqueue_copy(main_ref_out, main_ref_dev)
    ctx.enqueue_copy(index_ref_out, index_ref_dev)
    ctx.enqueue_copy(index_out, index_dev)
    ctx.enqueue_copy(main4_out, main4_dev)
    ctx.enqueue_copy(sf_out, sf_dev)
    ctx.synchronize()

    print("comparing Q (main) and IndexQ")
    var q8_ref = q_main_ref_out.unsafe_ptr().bitcast[UInt8]()
    var q8_out = q_main_out.unsafe_ptr().bitcast[UInt8]()
    for i in range(q_main_n):
        assert_equal(q8_out[i], q8_ref[i], "main Q byte")
    var qi_ref = q_index_ref_out.unsafe_ptr().bitcast[UInt16]()
    var qi_out = q_index_out.unsafe_ptr().bitcast[UInt16]()
    for i in range(q_index_n):
        assert_equal(qi_out[i], qi_ref[i], "IndexQ bits")

    print("comparing IndexK cache")
    var ik_ref = index_ref_out.unsafe_ptr().bitcast[UInt16]()
    var ik_out = index_out.unsafe_ptr().bitcast[UInt16]()
    for i in range(index_n):
        assert_equal(ik_out[i], ik_ref[i], "IndexK cache bits")

    print("comparing NVFP4 K/V against the host quantizer")
    var sf_bytes = sf_out.unsafe_ptr().bitcast[UInt8]()
    var owned_value = List[Bool](length=main4_n, fill=False)
    var owned_scale = List[Bool](length=sf_n, fill=False)
    var row = List[Float32](length=head_dim, fill=0.0)
    var packed = List[UInt8]()
    var scales = List[UInt8]()
    var nonfloor_scales = 0
    tok = 0
    for bs in range(batch_size):
        for t in range(prompt_lens[bs]):
            var pos = cache_lens[bs] + t
            var block = Int(lut_host[bs * lut_cols + pos // page_size])
            for h in range(main_k_heads):
                for kv in range(2):
                    if kv == 0:
                        var src = cache_offset(
                            block,
                            0,
                            2,
                            pos % page_size,
                            h,
                            main_k_heads,
                            head_dim,
                        )
                        for d in range(head_dim):
                            row[d] = main_ref_out[src + d].cast[.float32]()
                    else:
                        for d in range(head_dim):
                            row[d] = v_main_host[
                                (tok * main_k_heads + h) * head_dim + d
                            ].cast[.float32]()
                    quantize_nvfp4_row(row, packed, scales)
                    # The planted row: the +inf group has a fixed answer and
                    # the NaN group is unspecified; neither may touch the
                    # other groups.
                    var skip_group = -1
                    if nonfinite and kv == 1 and tok == nf_tok and h == 0:
                        nonfinite_group_codes(row, 1, packed)
                        scales[1] = 0x7E
                        skip_group = 2
                    var vdst = cache_offset(
                        block,
                        kv,
                        2,
                        pos % page_size,
                        h,
                        main_k_heads,
                        packed_dim,
                        value_page,
                    )
                    for b in range(packed_dim):
                        owned_value[vdst + b] = True
                        if b // (NVFP4_SF_VECTOR_SIZE // 2) == skip_group:
                            continue
                        if main4_out[vdst + b] != packed[b]:
                            raise Error(
                                String(
                                    "NVFP4 ",
                                    "K" if kv == 0 else "V",
                                    " byte mismatch: req ",
                                    bs,
                                    " tok ",
                                    t,
                                    " head ",
                                    h,
                                    " byte ",
                                    b,
                                    ": got ",
                                    main4_out[vdst + b],
                                    " want ",
                                    packed[b],
                                )
                            )
                    var sdst = cache_offset(
                        Int(sf_lut_host[bs * lut_cols + pos // page_size]),
                        kv,
                        2,
                        pos % page_size,
                        h,
                        main_k_heads,
                        sf_cols,
                        scale_page,
                    )
                    for g in range(sf_cols):
                        owned_scale[sdst + g] = True
                        if g == skip_group:
                            continue
                        if scales[g] != 0x08:
                            nonfloor_scales += 1
                        if sf_bytes[sdst + g] != scales[g]:
                            raise Error(
                                String(
                                    "NVFP4 ",
                                    "K" if kv == 0 else "V",
                                    " scale mismatch: req ",
                                    bs,
                                    " tok ",
                                    t,
                                    " head ",
                                    h,
                                    " group ",
                                    g,
                                    ": got ",
                                    sf_bytes[sdst + g],
                                    " want ",
                                    scales[g],
                                )
                            )
            tok += 1
    assert_true(nonfloor_scales > 0, "every scale sat at the floor")

    print("checking the poison outside the written rows")
    for i in range(main4_n):
        if not owned_value[i]:
            assert_equal(main4_out[i], value_poison, "stray value write")
    for i in range(sf_n):
        if not owned_scale[i]:
            assert_equal(sf_bytes[i], scale_poison, "stray scale write")

    print("comparing NVFP4 IndexK against the host quantizer")
    # The fp8 IndexQ is the bf16 IndexQ through a plain cast.
    var qi2 = q_index_out2.unsafe_ptr().bitcast[UInt8]()
    for i in range(q_index_n):
        assert_equal(
            qi2[i],
            bitcast[DType.uint8, 1](
                q_index_ref_out[i].cast[q_main_out_dtype]()
            ),
            "fp8 IndexQ byte",
        )
    var idx_sf_bytes = idx_sf_out.unsafe_ptr().bitcast[UInt8]()
    var idx_owned_value = List[Bool](length=idx4_n, fill=False)
    var idx_owned_scale = List[Bool](length=idx_sf_n, fill=False)
    for bs in range(batch_size):
        for t in range(prompt_lens[bs]):
            var pos = cache_lens[bs] + t
            var block = Int(lut_host[bs * lut_cols + pos // page_size])
            var src = cache_offset(
                block, 0, 1, pos % page_size, 0, index_k_heads, head_dim
            )
            for d in range(head_dim):
                row[d] = index_ref_out[src + d].cast[.float32]()
            quantize_nvfp4_row(row, packed, scales)
            var vdst = cache_offset(
                block,
                0,
                1,
                pos % page_size,
                0,
                index_k_heads,
                packed_dim,
                idx_value_page,
            )
            for b in range(packed_dim):
                idx_owned_value[vdst + b] = True
                assert_equal(idx4_out[vdst + b], packed[b], "IndexK byte")
            var sdst = cache_offset(
                Int(sf_lut_host[bs * lut_cols + pos // page_size]),
                0,
                1,
                pos % page_size,
                0,
                index_k_heads,
                sf_cols,
                idx_scale_page,
            )
            for g in range(sf_cols):
                idx_owned_scale[sdst + g] = True
                assert_equal(idx_sf_bytes[sdst + g], scales[g], "IndexK scale")
    for i in range(idx4_n):
        if not idx_owned_value[i]:
            assert_equal(idx4_out[i], value_poison, "stray IndexK write")
    for i in range(idx_sf_n):
        if not idx_owned_scale[i]:
            assert_equal(idx_sf_bytes[i], scale_poison, "stray IndexK scale")

    _ = row_offsets_dev^
    _ = cache_lengths_dev^
    _ = lut_dev^
    _ = freqs_dev^
    _ = gamma_devs^
    _ = q_main_dev^
    _ = k_main_dev^
    _ = v_main_dev^
    _ = q_index_dev^
    _ = k_index_dev^
    _ = main_ref_dev^
    _ = index_ref_dev^
    _ = main4_dev^
    _ = sf_dev^
    _ = sf_lut_dev^
    _ = index_dev^
    _ = idx4_dev^
    _ = idx_sf_dev^
    _ = q_index_out2_dev^
    _ = q_main_ref_dev^
    _ = q_main_out_dev^
    _ = q_index_ref_dev^
    _ = q_index_out_dev^


def run_nvfp4_dual_empty(ctx: DeviceContext) raises:
    """A data-parallel replica with no requests: zero rows, nothing written."""
    print("== run_nvfp4_dual_empty")
    comptime n_blocks = 2
    comptime value_n = n_blocks * 2 * num_layers * page_size * packed_dim
    comptime scale_n = n_blocks * 2 * num_layers * page_size * sf_cols
    comptime index_n = n_blocks * num_layers * page_size * head_dim
    var main4_dev = ctx.enqueue_create_buffer[.uint8](value_n)
    var sf_dev = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](scale_n)
    var index_dev = ctx.enqueue_create_buffer[dtype](index_n)
    main4_dev.enqueue_fill(value_poison)
    var sf_bytes_dev = DeviceBuffer[.uint8](
        ctx, sf_dev.unsafe_ptr().bitcast[UInt8](), scale_n, owning=False
    )
    sf_bytes_dev.enqueue_fill(scale_poison)
    var row_offsets_dev = ctx.enqueue_create_buffer[.uint32](1)
    row_offsets_dev.enqueue_fill(0)
    var cache_lengths_dev = ctx.enqueue_create_buffer[.uint32](1)
    var lut_dev = ctx.enqueue_create_buffer[.uint32](1)
    var freqs_dev = ctx.enqueue_create_buffer[freq_dtype](max_seq_len * 64)
    var gamma_dev = ctx.enqueue_create_buffer[dtype](head_dim)
    var st0 = ctx.enqueue_create_buffer[dtype](head_dim)
    var st1 = ctx.enqueue_create_buffer[dtype](head_dim)
    var st2 = ctx.enqueue_create_buffer[dtype](head_dim)
    var st3 = ctx.enqueue_create_buffer[dtype](head_dim)
    var st4 = ctx.enqueue_create_buffer[dtype](head_dim)
    var q_dev = ctx.enqueue_create_buffer[q_main_out_dtype](head_dim)
    var qi_dev = ctx.enqueue_create_buffer[dtype](head_dim)

    var cache_lengths_tt = TileTensor[mut=False](
        cache_lengths_dev, row_major(0)
    ).as_unsafe_any_origin()
    var lut_tt = TileTensor[mut=False](
        lut_dev, row_major((0, 1))
    ).as_unsafe_any_origin()
    var main4_coll = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(num_heads=1, head_size=packed_dim),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks_tt(main4_dev, n_blocks, 2, 1, packed_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(0),
        UInt32(0),
        scales=blocks_tt(sf_dev, n_blocks, 2, 1, sf_cols),
        scales_lookup_table=lut_tt,
    )
    var index_coll = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=1, head_size=head_dim, is_mla=True),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(index_dev, n_blocks, 1, 1, head_dim),
        cache_lengths_tt,
        lut_tt,
        UInt32(0),
        UInt32(0),
    )
    var gamma = TileTensor(gamma_dev, row_major[head_dim]())
    var s0 = TileTensor(st0, row_major((0, Idx[1], Idx[head_dim])))
    var s1 = TileTensor(st1, row_major((0, Idx[1], Idx[head_dim])))
    var s2 = TileTensor(st2, row_major((0, Idx[1], Idx[head_dim])))
    var s3 = TileTensor(st3, row_major((0, Idx[1], Idx[head_dim])))
    var s4 = TileTensor(st4, row_major((0, Idx[1], Idx[head_dim])))

    @inline(.always)
    def f0[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var s0} -> SIMD[dtype, width]:
        return s0.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def f1[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var s1} -> SIMD[dtype, width]:
        return s1.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def f2[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var s2} -> SIMD[dtype, width]:
        return s2.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def f3[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var s3} -> SIMD[dtype, width]:
        return s3.load[width=width](Coord(Index(token, head, col)))

    @inline(.always)
    def f4[
        width: Int, alignment: Int
    ](token: Int, head: Int, col: Int) {var s4} -> SIMD[dtype, width]:
        return s4.load[width=width](Coord(Index(token, head, col)))

    fused_dual_qk_rms_norm_rope_nvfp4_ragged_paged[
        target="gpu", multiply_before_cast=True, interleaved=False
    ](
        main4_coll,
        index_coll,
        gamma,
        gamma,
        gamma,
        gamma,
        TileTensor(freqs_dev, row_major[max_seq_len, 64]()),
        Float32(1e-6),
        Float32(1e-6),
        Scalar[dtype](1.0),
        UInt32(layer_idx),
        TileTensor(row_offsets_dev, row_major(1)),
        f0,
        f1,
        f2,
        f3,
        f4,
        TileTensor(q_dev, row_major((0, Idx[16], Idx[head_dim]))),
        TileTensor(qi_dev, row_major((0, Idx[4], Idx[head_dim]))),
        ctx,
    )
    var main4_out = ctx.enqueue_create_host_buffer[.uint8](value_n)
    var sf_out = ctx.enqueue_create_host_buffer[.uint8](scale_n)
    ctx.enqueue_copy(main4_out, main4_dev)
    ctx.enqueue_copy(sf_out, sf_bytes_dev)
    ctx.synchronize()
    for i in range(value_n):
        assert_equal(main4_out[i], value_poison, "empty launch wrote a value")
    for i in range(scale_n):
        assert_equal(sf_out[i], scale_poison, "empty launch wrote a scale")
    _ = main4_dev^
    _ = sf_dev^
    _ = index_dev^
    _ = row_offsets_dev^
    _ = cache_lengths_dev^
    _ = lut_dev^
    _ = freqs_dev^
    _ = gamma_dev^
    _ = st0^
    _ = st1^
    _ = st2^
    _ = st3^
    _ = st4^
    _ = q_dev^
    _ = qi_dev^


def main() raises:
    with DeviceContext() as ctx:
        # M3 at TP4: 16 Q heads over one KV head per device.
        run_nvfp4_dual[main_q_heads=16, main_k_heads=1](ctx)
        run_nvfp4_dual[main_q_heads=8, main_k_heads=2](ctx)
        # TP1/TP2 layouts keep more KV heads per device.
        run_nvfp4_dual[main_q_heads=16, main_k_heads=4](ctx)
        # Non-interleaved RoPE over the whole head.
        run_nvfp4_dual[main_q_heads=16, main_k_heads=1, rope_dim=128](ctx)
        # Pool layouts: padded pages (independent value and scale strides),
        # scales paged through their own table, and non-finite V values.
        run_nvfp4_dual[main_q_heads=16, main_k_heads=1](
            ctx,
            value_pad_rows=32,
            scale_pad_rows=16,
            separate_scales_lut=True,
            nonfinite=True,
        )
        run_nvfp4_dual[main_q_heads=8, main_k_heads=2](
            ctx, value_pad_rows=16, scale_pad_rows=48, separate_scales_lut=True
        )
        run_nvfp4_dual_empty(ctx)
