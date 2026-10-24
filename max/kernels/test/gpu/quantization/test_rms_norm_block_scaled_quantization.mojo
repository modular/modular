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
"""Fused q/kv RMSNorm + MXFP8 quantize vs the unfused kernels.

`rms_norm_quantize_mxfp8_q_kv` must write what `rms_norm` on the q and kv
parts followed by `quantize_dynamic_scaled_fp4fp8` on the normalized q write.
The two only differ in the order of each row's sum of squares. The exact test
feeds integer values whose squares sum exactly in fp32 in any order, so every
byte must match. The random test allows the rare rounding flip a different
order causes. Every output starts as poison, so a byte either path skips,
padded scale rows included, shows up.
"""

from std.math import ceildiv
from std.random import rand, seed
from std.testing import assert_equal, assert_raises, assert_true
from std.utils.coord import ComptimeInt

from max.gpu.host import DeviceBuffer, DeviceContext
from layout import Coord, Idx, TileTensor, row_major
from linalg.block_scaled_quantization import (
    quantize_dynamic_scaled_fp4fp8,
    rms_norm_quantize_mxfp8_q_kv,
)
from linalg.fp4_utils import (
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
)
from builtin_kernels.rowwise_normalization import rms_norm

comptime IN_DTYPE = DType.bfloat16
comptime OUT_DTYPE = DType.float8_e4m3fn
comptime EPS = Float32(1e-20)
comptime POISON = 0xAB


def _reference_rms_norm[
    cols: Int
](
    ctx: DeviceContext,
    x_tt: TileTensor[IN_DTYPE, ...],
    normed_tt: TileTensor[mut=True, IN_DTYPE, ...],
    gamma_tt: TileTensor[IN_DTYPE, ...],
    shape: Coord,
) raises:
    @inline(.always)
    def input_fn[
        width: Int, alignment: Int
    ](coords: Coord) {var x_tt} -> SIMD[IN_DTYPE, width]:
        return x_tt.load[width=width, alignment=alignment](coords)

    @inline(.always)
    def output_fn[
        width: SIMDLength, alignment: Int
    ](coords: Coord, val: SIMD[IN_DTYPE, width]) {var normed_tt}:
        normed_tt.store[alignment=alignment](
            coords, rebind[SIMD[IN_DTYPE, width]](val)
        )

    rms_norm[
        IN_DTYPE,
        2,
        target="gpu",
        epsilon=EPS,
        weight_offset=Float32(0.0),
        multiply_before_cast=True,
    ](
        input_fn,
        output_fn,
        shape,
        ComptimeInt[cols](),
        gamma_tt,
        ctx,
    )


def test_matches_unfused[
    q_cols: Int, kv_cols: Int
](ctx: DeviceContext, m: Int, exact: Bool, fault: Int = 0) raises:
    comptime width = q_cols + kv_cols
    comptime sf_cols = ceildiv(q_cols, MXFP8_SF_VECTOR_SIZE * SF_ATOM_K)
    var n_scales = ceildiv(m, SF_MN_GROUP_SIZE) * sf_cols * 128 * SF_ATOM_K

    var x_host = ctx.enqueue_create_host_buffer[IN_DTYPE](m * width)
    var gamma_q_host = ctx.enqueue_create_host_buffer[IN_DTYPE](q_cols)
    var gamma_kv_host = ctx.enqueue_create_host_buffer[IN_DTYPE](kv_cols)
    var q_in_host = ctx.enqueue_create_host_buffer[IN_DTYPE](m * q_cols)
    var kv_in_host = ctx.enqueue_create_host_buffer[IN_DTYPE](m * kv_cols)
    var vals = ctx.enqueue_create_host_buffer[.float32](m * width)
    seed(m * 13 + q_cols)
    rand[.float32](vals.unsafe_ptr(), m * width, min=-16.0, max=16.0)
    for r in range(m):
        var row_scale = Float32(1 << (r % 3))
        for c in range(width):
            var v = vals[r * width + c]
            if exact:
                # Integers: every square and partial sum is exact in fp32.
                v = Float32(Int(v)) * row_scale
                if c == (r * 7) % width:
                    v = 100.0 if r % 2 == 0 else -100.0
            else:
                var gain = Float32(20.0) if c % 41 == 0 else Float32(1.0)
                v = v * row_scale * gain
            # An all-zero scale block in q, and an all-zero row.
            if r % 5 == 0 and c < MXFP8_SF_VECTOR_SIZE:
                v = 0.0
            if r % 11 == 3:
                v = 0.0
            x_host[r * width + c] = v.cast[IN_DTYPE]()
    for r in range(m):
        for c in range(q_cols):
            q_in_host[r * q_cols + c] = x_host[r * width + c]
        for c in range(kv_cols):
            kv_in_host[r * kv_cols + c] = x_host[r * width + q_cols + c]
    for c in range(q_cols):
        var g = 0.5 + Float32(c % 13) / 8.0
        gamma_q_host[c] = (-g if c % 5 == 0 else g).cast[IN_DTYPE]()
    for c in range(kv_cols):
        var g = 0.75 + Float32(c % 7) / 8.0
        gamma_kv_host[c] = (-g if c % 9 == 0 else g).cast[IN_DTYPE]()

    var x = ctx.enqueue_create_buffer[IN_DTYPE](m * width)
    var q_in = ctx.enqueue_create_buffer[IN_DTYPE](m * q_cols)
    var kv_in = ctx.enqueue_create_buffer[IN_DTYPE](m * kv_cols)
    var gamma_q = ctx.enqueue_create_buffer[IN_DTYPE](q_cols)
    var gamma_kv = ctx.enqueue_create_buffer[IN_DTYPE](kv_cols)
    ctx.enqueue_copy(x, x_host)
    ctx.enqueue_copy(q_in, q_in_host)
    ctx.enqueue_copy(kv_in, kv_in_host)
    ctx.enqueue_copy(gamma_q, gamma_q_host)
    ctx.enqueue_copy(gamma_kv, gamma_kv_host)

    var q_normed = ctx.enqueue_create_buffer[IN_DTYPE](m * q_cols)
    var kv_ref = ctx.enqueue_create_buffer[IN_DTYPE](m * kv_cols)
    var q_ref = ctx.enqueue_create_buffer[DType.uint8](m * q_cols)
    var sf_ref = ctx.enqueue_create_buffer[DType.uint8](n_scales)
    var q_fused = ctx.enqueue_create_buffer[DType.uint8](m * q_cols)
    var sf_fused = ctx.enqueue_create_buffer[DType.uint8](n_scales)
    var kv_fused_bytes = ctx.enqueue_create_buffer[DType.uint8](m * kv_cols * 2)
    q_ref.enqueue_fill(POISON)
    sf_ref.enqueue_fill(POISON)
    q_fused.enqueue_fill(POISON)
    sf_fused.enqueue_fill(POISON)
    kv_fused_bytes.enqueue_fill(POISON)

    var scales_layout = row_major(
        ceildiv(m, SF_MN_GROUP_SIZE),
        Idx[sf_cols],
        Idx[SF_ATOM_M[0]],
        Idx[SF_ATOM_M[1]],
        Idx[SF_ATOM_K],
    )

    # Unfused: rms_norm of each part, then the plain quantize of q.
    _reference_rms_norm[q_cols](
        ctx,
        TileTensor(q_in, row_major(m, Idx[q_cols])),
        TileTensor(q_normed, row_major(m, Idx[q_cols])),
        TileTensor(gamma_q, row_major(Idx[q_cols])),
        Coord(m, q_cols),
    )
    _reference_rms_norm[kv_cols](
        ctx,
        TileTensor(kv_in, row_major(m, Idx[kv_cols])),
        TileTensor(kv_ref, row_major(m, Idx[kv_cols])),
        TileTensor(gamma_kv, row_major(Idx[kv_cols])),
        Coord(m, kv_cols),
    )
    quantize_dynamic_scaled_fp4fp8[
        SF_VECTOR_SIZE=MXFP8_SF_VECTOR_SIZE, num_max_threads=1024
    ](
        ctx,
        TileTensor(
            q_ref.unsafe_ptr().unsafe_bitcast[Scalar[OUT_DTYPE]](),
            row_major(m, Idx[q_cols]),
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_ref.unsafe_ptr().unsafe_bitcast[Scalar[MXFP8_SF_DTYPE]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        TileTensor(q_normed, row_major(m, Idx[q_cols])).as_unsafe_any_origin(),
        num_cols=q_cols,
        num_cols_padded=q_cols,
    )

    rms_norm_quantize_mxfp8_q_kv(
        TileTensor(
            q_fused.unsafe_ptr().unsafe_bitcast[Scalar[OUT_DTYPE]](),
            row_major(m, Idx[q_cols]),
        ).as_unsafe_any_origin(),
        TileTensor(
            sf_fused.unsafe_ptr().unsafe_bitcast[Scalar[MXFP8_SF_DTYPE]](),
            scales_layout,
        ).as_unsafe_any_origin(),
        TileTensor(
            kv_fused_bytes.unsafe_ptr().unsafe_bitcast[Scalar[IN_DTYPE]](),
            row_major(m, Idx[kv_cols]),
        ).as_unsafe_any_origin(),
        TileTensor(x, row_major(m, Idx[width])).as_unsafe_any_origin(),
        TileTensor(gamma_q, row_major(Idx[q_cols])).as_unsafe_any_origin(),
        TileTensor(gamma_kv, row_major(Idx[kv_cols])).as_unsafe_any_origin(),
        EPS,
        ctx,
    )
    ctx.synchronize()
    # Planted faults the checks must catch: one flipped q byte, or the last
    # padding row's scale left as poison.
    if fault == 1:
        with q_fused.map_to_host() as b:
            b[m * q_cols // 2] = b[m * q_cols // 2] ^ 1
    elif fault == 2:
        with sf_fused.map_to_host() as b:
            b[n_scales - 1] = POISON

    var q_bad = 0
    var sf_bad = 0
    var kv_bad = 0
    with q_ref.map_to_host() as a, q_fused.map_to_host() as b:
        for i in range(len(a)):
            q_bad += Int(a[i] != b[i])
    with sf_ref.map_to_host() as a, sf_fused.map_to_host() as b:
        for i in range(len(a)):
            sf_bad += Int(a[i] != b[i])
    with kv_ref.map_to_host() as a, kv_fused_bytes.map_to_host() as raw:
        var b = raw.unsafe_ptr().unsafe_bitcast[Scalar[IN_DTYPE]]()
        for i in range(len(a)):
            var x32 = a[i].cast[DType.float32]()
            var y32 = b[i].cast[DType.float32]()
            # A different sum order moves a value by at most one bf16 ulp.
            if x32 != y32:
                kv_bad += 1
                if not exact:
                    assert_true(
                        abs(x32 - y32) <= 0.008 * max(abs(x32), abs(y32)),
                        String("M = ", m, ", kv element ", i),
                    )
    if exact:
        assert_equal(q_bad, 0, String("M = ", m, ", q bytes"))
        assert_equal(sf_bad, 0, String("M = ", m, ", scale bytes"))
        assert_equal(kv_bad, 0, String("M = ", m, ", kv elements"))
    else:
        var q_total = m * q_cols
        assert_true(q_bad * 500 <= q_total, String("M = ", m, ", q bytes"))
        assert_true(sf_bad * 500 <= n_scales, String("M = ", m, ", scales"))
        assert_true(
            kv_bad * 500 <= m * kv_cols, String("M = ", m, ", kv elements")
        )
    print(
        "fused matches unfused:",
        "M =",
        m,
        "q_cols =",
        q_cols,
        "kv_cols =",
        kv_cols,
        "exact =",
        exact,
        "mismatches q/scales/kv =",
        q_bad,
        sf_bad,
        kv_bad,
    )


def main() raises:
    with DeviceContext() as ctx:
        # Decode rows (6 a request), prefill chunks, and prime and ragged
        # counts around the 128-row scale tiles.
        for m in [1, 5, 6, 7, 48, 96, 127, 128, 129, 144, 577, 1000, 1031]:
            test_matches_unfused[1280, 512](ctx, m, True)
            test_matches_unfused[1280, 512](ctx, m, False)
            test_matches_unfused[512, 256](ctx, m, True)
        for m in [4093, 8168]:
            test_matches_unfused[1280, 512](ctx, m, True)
            test_matches_unfused[1280, 512](ctx, m, False)
        for fault in [1, 2]:
            with assert_raises():
                test_matches_unfused[1280, 512](ctx, 129, True, fault)
