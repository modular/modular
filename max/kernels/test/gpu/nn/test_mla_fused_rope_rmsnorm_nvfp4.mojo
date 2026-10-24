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
"""Bitwise test for the NVFP4 latent-cache write of
`mla_fused_rope_rmsnorm_quantization` (Blackwell sm_100).

The reference is the same op on a bf16 cache, which holds the normed and roped
576-wide latent row rounded to bf16 (the value the NVFP4 path quantizes). A
host quantizer, written independently of the kernel, must reproduce every
packed latent byte and E4M3 scale exactly, the roped query must match bit for
bit, and every cache byte the op does not own must keep its poison.

The 64 rope dims are not quantized: the NVFP4 row stores them as FP8 after the
256 packed latent bytes. Those bytes must equal what the FP8 cache path writes
for the same input, so the test runs the op on an FP8 cache too.
"""

from std.collections import Set
from std.math import ceildiv, sqrt
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
from nn.attention.gpu.mla_graph import mla_fused_rope_rmsnorm_quantization
from std.testing import assert_equal, assert_true
from std.utils import IndexList

comptime dtype = DType.bfloat16
comptime freq_dtype = DType.float32
comptime kv_norm_dim = 512
comptime rope_dim = 64
comptime head_size = kv_norm_dim + rope_dim
comptime latent_packed_size = kv_norm_dim // 2
# The packed latent, then the FP8 rope bytes.
comptime packed_size = latent_packed_size + rope_dim
comptime sf_cols = kv_norm_dim // NVFP4_SF_VECTOR_SIZE
comptime page_size = 128
comptime num_layers = 2
comptime layer_idx = 1
comptime max_seq_len = 1024
comptime epsilon = Float32(1e-6)
comptime value_poison = UInt8(0xAB)
# An unwritten scale reads as NaN.
comptime scale_poison = UInt8(0x7F)


def e2m1_magnitude_rne(a: Float32) -> UInt8:
    """Round-half-to-even ladder on `a = min(|x / scale|, 6)`."""
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


def e2m1_value(code: UInt8) -> Float32:
    var mag = code & 7
    var v = SIMD[DType.float32, 8](0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)[
        Int(mag)
    ]
    return -v if (code & 8) != 0 else v


def nvfp4_group_scale(amax: Float32) -> Float32:
    """`E4M3(max(amax, 1e-12) / 6)` clamped to [2^-9, 448]; the cast
    saturates."""
    var scale = (
        (max(amax, Float32(1e-12)) / 6.0)
        .cast[NVFP4_SF_DTYPE]()
        .cast[DType.float32]()
    )
    return min(Float32(448.0), max(scale, Float32(1.0 / 512.0)))


def bitcast_e4m3(v: Scalar[NVFP4_SF_DTYPE]) -> UInt8:
    return bitcast[DType.uint8, 1](v)


def quantize_row(
    x: List[Float32], mut packed: List[UInt8], mut scales: List[UInt8]
):
    """Quantizes one latent row (tensor scale 1): element 2i in the low nibble,
    zero magnitude stored as +0."""
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


def cache_offset(block: Int, pos: Int, width: Int, kv_dim: Int = 1) -> Int:
    """Element offset of the MLA cache row `(block, layer_idx, pos)`."""
    return (
        block * kv_dim * num_layers * page_size * width
        + (layer_idx * page_size + pos) * width
    )


def blocks_tt[
    dt: DType
](mut dev: DeviceBuffer[dt], num_blocks: Int, width: Int) -> TileTensor[
    dt, RowMajorLayout[Int, Int, Int, Int, Int, Int], MutAnyOrigin
]:
    return TileTensor(
        dev, row_major(num_blocks, 1, num_layers, page_size, 1, width)
    ).as_unsafe_any_origin()


def run_case[num_q_heads: Int](ctx: DeviceContext) raises:
    print("== test_mla_fused_rope_rmsnorm_nvfp4 num_q_heads=", num_q_heads)
    seed(7)
    # Request 1 ends on a page boundary, request 3 crosses one, and request 0
    # starts a fresh page.
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

    var freqs_host = ctx.enqueue_create_host_buffer[freq_dtype](
        max_seq_len * rope_dim
    )
    var gamma_host = ctx.enqueue_create_host_buffer[dtype](kv_norm_dim)
    ctx.synchronize()
    for i in range(max_seq_len * rope_dim):
        freqs_host[i] = Float32(random_float64(-1.0, 1.0))
    for d in range(kv_norm_dim):
        gamma_host[d] = Float32(random_float64(0.5, 1.5)).cast[dtype]()
    var freqs_dev = ctx.enqueue_create_buffer[freq_dtype](
        max_seq_len * rope_dim
    )
    var gamma_dev = ctx.enqueue_create_buffer[dtype](kv_norm_dim)
    ctx.enqueue_copy(freqs_dev, freqs_host)
    ctx.enqueue_copy(gamma_dev, gamma_host)

    # Latent staging [total, 576] plus the roped query. Rows cycle through
    # edge cases: 1 zeros, 2 tiny, 3 an outlier per 16-element group (scale
    # clamped to 448), the rest random.
    var kv_n = total_length * head_size
    var q_n = total_length * num_q_heads * rope_dim
    var kv_host = ctx.enqueue_create_host_buffer[dtype](kv_n)
    var q_host = ctx.enqueue_create_host_buffer[dtype](q_n)
    ctx.synchronize()
    for r in range(total_length):
        for d in range(head_size):
            var v = Float32(random_float64(-3.0, 3.0))
            var kind = r % 6
            if kind == 1:
                v = 0.0
            elif kind == 2:
                v *= 1e-3
            elif kind == 3 and d % NVFP4_SF_VECTOR_SIZE == 0:
                v = 3000.0
            kv_host[r * head_size + d] = v.cast[dtype]()
    for i in range(q_n):
        q_host[i] = Float32(random_float64(-2.0, 2.0)).cast[dtype]()
    var kv_dev = ctx.enqueue_create_buffer[dtype](kv_n)
    var q_dev = ctx.enqueue_create_buffer[dtype](q_n)
    ctx.enqueue_copy(kv_dev, kv_host)
    ctx.enqueue_copy(q_dev, q_host)

    # Caches. The bf16 cache is the reference row; the NVFP4 one is under test,
    # and the FP8 one is the reference for its rope bytes.
    var ref_n = num_blocks * num_layers * page_size * head_size
    var packed_n = num_blocks * num_layers * page_size * packed_size
    var sf_n = num_blocks * num_layers * page_size * sf_cols
    var ref_dev = ctx.enqueue_create_buffer[dtype](ref_n)
    var packed_host = ctx.enqueue_create_host_buffer[.uint8](packed_n)
    var sf_host = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sf_n)
    ctx.synchronize()
    ctx.enqueue_memset(ref_dev, BFloat16(0))
    var fp8_dev = ctx.enqueue_create_buffer[.float8_e4m3fn](ref_n)
    ctx.enqueue_memset(fp8_dev, Float8_e4m3fn(0))
    for i in range(packed_n):
        packed_host[i] = value_poison
    var sf_host_bytes = sf_host.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(sf_n):
        sf_host_bytes[i] = scale_poison
    var packed_dev = ctx.enqueue_create_buffer[.uint8](packed_n)
    var sf_dev = ctx.enqueue_create_buffer[NVFP4_SF_DTYPE](sf_n)
    ctx.enqueue_copy(packed_dev, packed_host)
    ctx.enqueue_copy(sf_dev, sf_host)

    var q_ref_dev = ctx.enqueue_create_buffer[dtype](q_n)
    var q_out_dev = ctx.enqueue_create_buffer[dtype](q_n)

    var row_offsets_tt = TileTensor[mut=False](
        row_offsets_dev, row_major(batch_size + 1)
    )
    var freqs_tt = TileTensor[mut=False](
        freqs_dev, row_major[max_seq_len, rope_dim]()
    )
    var gamma_tt = TileTensor[mut=False](gamma_dev, row_major[kv_norm_dim]())
    var kv_tt = TileTensor(
        kv_dev, row_major((total_length, Idx[head_size]))
    ).as_unsafe_any_origin()
    var q_tt = TileTensor[mut=False](
        q_dev, row_major((total_length, Idx[num_q_heads], Idx[rope_dim]))
    )
    var q_ref_tt = TileTensor(
        q_ref_dev,
        row_major((total_length, Idx[num_q_heads], Idx[rope_dim])),
    )
    var q_out_tt = TileTensor(
        q_out_dev,
        row_major((total_length, Idx[num_q_heads], Idx[rope_dim])),
    )
    var cache_lengths_tt = TileTensor[mut=False](
        cache_lengths_dev, row_major(batch_size)
    ).as_unsafe_any_origin()
    var lut_tt = TileTensor[mut=False](
        lut_dev, row_major((batch_size, lut_cols))
    ).as_unsafe_any_origin()

    var ref_coll = PagedKVCacheCollection[
        dtype,
        KVCacheStaticParams(num_heads=1, head_size=head_size, is_mla=True),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(ref_dev, num_blocks, head_size),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var fp8_coll = PagedKVCacheCollection[
        DType.float8_e4m3fn,
        KVCacheStaticParams(num_heads=1, head_size=head_size, is_mla=True),
        page_size,
        scales_origin=MutAnyOrigin,
    ](
        blocks_tt(fp8_dev, num_blocks, head_size),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
    )
    var fp4_coll = PagedKVCacheCollection[
        DType.uint8,
        KVCacheStaticParams(
            num_heads=1,
            head_size=packed_size,
            is_mla=True,
            fp8_tail_size=rope_dim,
        ),
        page_size,
        scale_dtype_=NVFP4_SF_DTYPE,
        quantization_granularity_=NVFP4_SF_VECTOR_SIZE // 2,
    ](
        blocks_tt(packed_dev, num_blocks, packed_size),
        cache_lengths_tt,
        lut_tt,
        UInt32(max_prompt_length),
        UInt32(max_cache_length),
        scales=blocks_tt(sf_dev, num_blocks, sf_cols),
        scales_lookup_table=lut_tt,
    )

    @inline(.always)
    def kv_input_fn[
        width: Int
    ](coords: IndexList[2]) {var kv_tt} -> SIMD[DType.bfloat16, width]:
        return kv_tt.load[width=width]((coords[0], coords[1]))

    mla_fused_rope_rmsnorm_quantization(
        q_ref_tt,
        q_tt,
        row_offsets_tt,
        freqs_tt,
        gamma_tt,
        ref_coll,
        UInt32(layer_idx),
        epsilon,
        kv_input_fn,
        ctx,
    )
    mla_fused_rope_rmsnorm_quantization(
        q_out_tt,
        q_tt,
        row_offsets_tt,
        freqs_tt,
        gamma_tt,
        fp4_coll,
        UInt32(layer_idx),
        epsilon,
        kv_input_fn,
        ctx,
    )
    var q_fp8_dev = ctx.enqueue_create_buffer[dtype](q_n)
    var q_fp8_tt = TileTensor(
        q_fp8_dev,
        row_major((total_length, Idx[num_q_heads], Idx[rope_dim])),
    )
    mla_fused_rope_rmsnorm_quantization(
        q_fp8_tt,
        q_tt,
        row_offsets_tt,
        freqs_tt,
        gamma_tt,
        fp8_coll,
        UInt32(layer_idx),
        epsilon,
        kv_input_fn,
        ctx,
    )
    ctx.synchronize()

    var ref_host = ctx.enqueue_create_host_buffer[dtype](ref_n)
    var fp8_host = ctx.enqueue_create_host_buffer[.float8_e4m3fn](ref_n)
    var q_ref_host = ctx.enqueue_create_host_buffer[dtype](q_n)
    var q_out_host = ctx.enqueue_create_host_buffer[dtype](q_n)
    var packed_out = ctx.enqueue_create_host_buffer[.uint8](packed_n)
    var sf_out = ctx.enqueue_create_host_buffer[NVFP4_SF_DTYPE](sf_n)
    ctx.enqueue_copy(ref_host, ref_dev)
    ctx.enqueue_copy(fp8_host, fp8_dev)
    ctx.enqueue_copy(q_ref_host, q_ref_dev)
    ctx.enqueue_copy(q_out_host, q_out_dev)
    ctx.enqueue_copy(packed_out, packed_dev)
    ctx.enqueue_copy(sf_out, sf_dev)
    ctx.synchronize()

    # The roped query does not depend on the cache dtype.
    for i in range(q_n):
        assert_equal(q_out_host[i], q_ref_host[i])

    # Expected cache images: poison everywhere, the reference quantization at
    # every token the op owns.
    var expected = List[UInt8](length=packed_n, fill=value_poison)
    var expected_sf = List[UInt8](length=sf_n, fill=scale_poison)
    var row = List[Float32](length=head_size, fill=0)
    var packed = List[UInt8]()
    var scales = List[UInt8]()
    var sq_err = Float64(0)
    var sq_ref = Float64(0)
    var fp4_rope_rows = List[Int]()
    var fp8_rope_rows = List[Int]()
    var tok = 0
    for bs in range(batch_size):
        for t in range(prompt_lens[bs]):
            var pos = cache_lens[bs] + t
            var block = Int(lut_host[bs * lut_cols + pos // page_size])
            var src = cache_offset(block, pos % page_size, head_size)
            for d in range(head_size):
                row[d] = ref_host[src + d].cast[DType.float32]()
            quantize_row(row, packed, scales)
            var dst = cache_offset(block, pos % page_size, packed_size)
            var sf_dst = cache_offset(block, pos % page_size, sf_cols)
            for i in range(latent_packed_size):
                expected[dst + i] = packed[i]
            # The rope bytes are what the FP8 cache holds for those dims.
            for i in range(rope_dim):
                expected[dst + latent_packed_size + i] = bitcast[
                    DType.uint8, 1
                ](fp8_host[src + kv_norm_dim + i])
            fp4_rope_rows.append(dst + latent_packed_size)
            fp8_rope_rows.append(src + kv_norm_dim)
            for g in range(sf_cols):
                expected_sf[sf_dst + g] = scales[g]

            # Dequantization error vs the bf16 row. The outlier rows (kind 3)
            # are checked by the exact comparison only: one 3000 per group
            # flushes the rest of the group to zero by design.
            if tok % 6 != 3:
                for g in range(sf_cols):
                    var sc = bitcast[NVFP4_SF_DTYPE, 1](
                        SIMD[DType.uint8, 1](scales[g])
                    ).cast[DType.float32]()
                    for e in range(NVFP4_SF_VECTOR_SIZE):
                        var d = g * NVFP4_SF_VECTOR_SIZE + e
                        var byte = packed[d // 2]
                        var code = (byte >> UInt8(4 * (d % 2))) & 0xF
                        var dq = e2m1_value(code) * sc[0]
                        var diff = Float64(dq - row[d])
                        sq_err += diff * diff
                        sq_ref += Float64(row[d]) * Float64(row[d])
            tok += 1

    var packed_bad = 0
    for i in range(packed_n):
        if packed_out[i] != expected[i]:
            packed_bad += 1
    var sf_bad = 0
    var sf_out_bytes = sf_out.unsafe_ptr().unsafe_bitcast[UInt8]()
    for i in range(sf_n):
        if sf_out_bytes[i] != expected_sf[i]:
            sf_bad += 1
    print("packed byte mismatches:", packed_bad, " scale mismatches:", sf_bad)
    assert_equal(packed_bad, 0)
    assert_equal(sf_bad, 0)

    # The rope bytes of the NVFP4 row are the FP8 cache's bytes, bit for bit.
    var rope_bad = 0
    for r in range(len(fp4_rope_rows)):
        for i in range(rope_dim):
            if packed_out[fp4_rope_rows[r] + i] != bitcast[DType.uint8, 1](
                fp8_host[fp8_rope_rows[r] + i]
            ):
                rope_bad += 1
    print(
        "rope bytes checked:", len(fp4_rope_rows) * rope_dim, " bad:", rope_bad
    )
    assert_equal(rope_bad, 0)

    var rel = sqrt(sq_err / sq_ref)
    print("dequantized relative RMS error:", rel)
    assert_true(rel < 0.2, "NVFP4 dequantization error beyond expectation")

    _ = row_offsets_dev
    _ = cache_lengths_dev
    _ = lut_dev
    _ = freqs_dev
    _ = gamma_dev
    _ = kv_dev
    _ = q_dev
    _ = ref_dev
    _ = fp8_dev
    _ = q_fp8_dev
    _ = packed_dev
    _ = sf_dev
    _ = q_ref_dev
    _ = q_out_dev


def main() raises:
    with DeviceContext() as ctx:
        # Few heads put the query and key lanes in one warp; many heads put the
        # key lanes in a second rope block.
        run_case[2](ctx)
        run_case[16](ctx)
