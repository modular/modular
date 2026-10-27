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

from std.math import ceildiv, cos, sin
from std.testing import assert_equal, assert_raises, assert_true
from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from layout._fillers import random
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    rope_quantize_dynamic_block_scaled,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    NVFP4_SF_DTYPE,
    NVFP4_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from nn.rope import rope_ragged


def _quantize[
    out_dtype: DType, scales_dtype: DType, SF_VECTOR_SIZE: Int, N: Int
](
    ctx: DeviceContext,
    mut dst: DeviceBuffer[DType.uint8],
    mut sf: DeviceBuffer[DType.uint8],
    src: DeviceBuffer[DType.bfloat16],
    m: Int,
) raises:
    comptime out_N = N // 2 if out_dtype == DType.uint8 else N
    quantize_dynamic_scaled_fp4fp8[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        num_max_threads=512 if out_dtype == DType.uint8 else 1024,
    ](
        ctx,
        TileTensor(
            dst.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            row_major(m, Idx[out_N]),
        ).as_unsafe_any_origin(),
        TileTensor(
            sf.unsafe_ptr().unsafe_bitcast[Scalar[scales_dtype]](),
            row_major(
                ceildiv(m, SF_MN_GROUP_SIZE),
                ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
                Idx[SF_ATOM_M[0]],
                Idx[SF_ATOM_M[1]],
                Idx[SF_ATOM_K],
            ),
        ).as_unsafe_any_origin(),
        TileTensor(src, row_major(m, Idx[N])).as_unsafe_any_origin(),
        num_cols=N,
        num_cols_padded=N,
    )


def test_rope_quantize_matches_unfused[
    out_dtype: DType,
    scales_dtype: DType,
    SF_VECTOR_SIZE: Int,
    heads: Int,
    head_dim: Int,
    rope_dim: Int,
    inverse: Bool,
](ctx: DeviceContext, m: Int) raises:
    """Checks the fused RoPE quantize writes the bytes of `rope_ragged`
    followed by the plain quantize.

    The reference rotates by a conjugated table when `inverse`. Rows span
    2^-8..2^8, some values sit far past the clamp, and every fifth row has
    an all-zero block. Both arms write into buffers pre-filled with a
    sentinel, so a byte either skips, padded scale rows included, differs.
    """
    comptime N = heads * head_dim
    comptime out_N = N // 2 if out_dtype == DType.uint8 else N

    var x_layout = row_major(m, Idx[N])
    var heads_layout = row_major(m, Idx[heads], Idx[head_dim])
    var freqs_layout = row_major(m, Idx[rope_dim])
    var out_layout = row_major(m, Idx[out_N])
    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        ceildiv(N, SF_VECTOR_SIZE * SF_ATOM_K),
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var x = ctx.enqueue_create_buffer[DType.bfloat16](x_layout.size())
    var rotated = ctx.enqueue_create_buffer[DType.bfloat16](x_layout.size())
    var freqs = ctx.enqueue_create_buffer[DType.float32](freqs_layout.size())
    var ref_freqs = ctx.enqueue_create_buffer[DType.float32](
        freqs_layout.size()
    )
    var row_offsets = ctx.enqueue_create_buffer[DType.uint32](2)
    var start_pos = ctx.enqueue_create_buffer[DType.uint32](1)
    var out_ref = ctx.enqueue_create_buffer[DType.uint8](out_layout.size())
    var out_fused = ctx.enqueue_create_buffer[DType.uint8](out_layout.size())
    var out_plain = ctx.enqueue_create_buffer[DType.uint8](out_layout.size())
    var sf_ref = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    var sf_fused = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    var sf_plain = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    out_ref.enqueue_fill(0xAB)
    out_fused.enqueue_fill(0xAB)
    out_plain.enqueue_fill(0xAB)
    sf_ref.enqueue_fill(0xAB)
    sf_fused.enqueue_fill(0xAB)
    sf_plain.enqueue_fill(0xAB)

    with x.map_to_host() as host:
        var t = TileTensor(host, x_layout)
        random(t, min=-1.0, max=1.0)
        for row in range(m):
            var row_scale = Float32(2.0) ** Float32(row % 17 - 8)
            for col in range(N):
                var v = t[row, col].cast[DType.float32]() * row_scale
                if row % 5 == 0 and col < SF_VECTOR_SIZE:
                    v = 0
                elif col % 37 == row % 37:
                    v *= 1000
                t[row, col] = v.cast[DType.bfloat16]()
    with freqs.map_to_host() as f, ref_freqs.map_to_host() as rf:
        for row in range(m):
            for j in range(rope_dim // 2):
                var theta = Float32(row + 1) * (
                    Float32(10000.0) ** (Float32(-2 * j) / Float32(rope_dim))
                )
                var c = cos(theta)
                var s = sin(theta)
                var i = row * rope_dim + 2 * j
                f[i] = c
                f[i + 1] = s
                rf[i] = c
                rf[i + 1] = -s if inverse else s
    with row_offsets.map_to_host() as h:
        h[0] = 0
        h[1] = UInt32(m)
    with start_pos.map_to_host() as h:
        h[0] = 0

    var rotated_heads = TileTensor(rotated, heads_layout)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](idx: Coord, val: SIMD[DType.bfloat16, width]) {var rotated_heads} -> None:
        rotated_heads.store(idx, val)

    rope_ragged[
        DType.bfloat16,
        DType.float32,
        interleaved=True,
        target=StaticString("gpu"),
    ](
        x=TileTensor(x, heads_layout).as_unsafe_any_origin(),
        input_row_offsets=TileTensor(
            row_offsets, row_major(Idx[2])
        ).as_unsafe_any_origin(),
        start_pos=TileTensor(
            start_pos, row_major(Idx[1])
        ).as_unsafe_any_origin(),
        freqs_cis=TileTensor(ref_freqs, freqs_layout).as_unsafe_any_origin(),
        context=ctx,
        output_fn=output_fn,
    )

    _quantize[out_dtype, scales_dtype, SF_VECTOR_SIZE, N](
        ctx, out_ref, sf_ref, rotated, m
    )
    _quantize[out_dtype, scales_dtype, SF_VECTOR_SIZE, N](
        ctx, out_plain, sf_plain, x, m
    )
    rope_quantize_dynamic_block_scaled[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE, head_dim=head_dim, inverse=inverse
    ](
        TileTensor(
            out_fused.unsafe_ptr().unsafe_bitcast[Scalar[out_dtype]](),
            out_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_fused.unsafe_ptr().unsafe_bitcast[Scalar[scales_dtype]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        TileTensor(x, x_layout).as_unsafe_any_origin(),
        TileTensor(freqs, freqs_layout).as_unsafe_any_origin(),
        Float32(1.0),
        ctx,
    )
    ctx.synchronize()

    with out_ref.map_to_host() as a, out_fused.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                assert_equal(a[i], b[i], String("M = ", m, ", output byte ", i))
    with sf_ref.map_to_host() as a, sf_fused.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                assert_equal(a[i], b[i], String("M = ", m, ", scale byte ", i))
    # Guards against a rotation that is a no-op in both arms.
    var rotated_bytes = 0
    with out_plain.map_to_host() as a, out_fused.map_to_host() as b:
        for i in range(len(a)):
            if a[i] != b[i]:
                rotated_bytes += 1
    assert_true(rotated_bytes > 0, "the RoPE columns came out unrotated")
    print(
        "fused matches unfused:",
        out_dtype,
        "M =",
        m,
        "heads =",
        heads,
        "head_dim =",
        head_dim,
        "inverse =",
        inverse,
    )


def test_rope_quantize_rejects_unpacked_fp4_output(
    ctx: DeviceContext,
) raises:
    """Checks the wrapper refuses an FP4 output one byte per value wide,
    which the kernel would otherwise write past."""
    comptime head_dim = 192
    comptime N = 5 * head_dim
    comptime m = 3
    var x = ctx.enqueue_create_buffer[DType.bfloat16](m * N)
    var out = ctx.enqueue_create_buffer[DType.uint8](m * N)
    var freqs = ctx.enqueue_create_buffer[DType.float32](m * 64)
    var scales_layout = row_major(
        Idx[1],
        Idx[ceildiv(N, NVFP4_SF_VECTOR_SIZE * SF_ATOM_K)],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )
    var sf = ctx.enqueue_create_buffer[DType.uint8](scales_layout.size())
    with assert_raises(contains="output must be"):
        rope_quantize_dynamic_block_scaled[
            SF_VECTOR_SIZE=NVFP4_SF_VECTOR_SIZE,
            head_dim=head_dim,
            inverse=False,
        ](
            TileTensor(out, row_major(Idx[m], Idx[N])).as_unsafe_any_origin(),
            TileTensor(
                sf.unsafe_ptr().unsafe_bitcast[Scalar[NVFP4_SF_DTYPE]](),
                scales_layout,
            ).as_unsafe_any_origin(),
            TileTensor(x, row_major(Idx[m], Idx[N])).as_unsafe_any_origin(),
            TileTensor(
                freqs, row_major(Idx[m], Idx[64])
            ).as_unsafe_any_origin(),
            Float32(1.0),
            ctx,
        )


def main() raises:
    with DeviceContext() as ctx:
        for m in [1, 96, 129, 1000]:
            test_rope_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                64,
                512,
                64,
                True,
            ](ctx, m)
            test_rope_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                3,
                128,
                64,
                False,
            ](ctx, m)
            # The whole head rotates.
            test_rope_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                2,
                64,
                64,
                True,
            ](ctx, m)
            test_rope_quantize_matches_unfused[
                DType.uint8,
                NVFP4_SF_DTYPE,
                NVFP4_SF_VECTOR_SIZE,
                5,
                192,
                64,
                True,
            ](ctx, m)
        # The model's shape at decode and prefill row counts, prime counts,
        # and both sides of each block-size switch.
        for m in [48, 144, 256, 257, 577, 1536, 1537, 4093, 4096, 4097, 8168]:
            test_rope_quantize_matches_unfused[
                DType.float8_e4m3fn,
                MXFP8_SF_DTYPE,
                MXFP8_SF_VECTOR_SIZE,
                64,
                512,
                64,
                True,
            ](ctx, m)
        test_rope_quantize_rejects_unpacked_fp4_output(ctx)
