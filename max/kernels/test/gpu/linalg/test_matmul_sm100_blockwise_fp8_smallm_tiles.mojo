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

"""SM100 blockwise FP8 matmul: tiles `blockwise_scaled_fp8_with_epilogue` picks
for M <= 128 against the naive blockwise reference and the previous dispatch.

For a static N the dispatcher picks, per runtime M band:

  M <= 64    a 64-row, one-CTA-per-tile config whose MMA_N is the narrowest of
             64/128/192/256 that fits the N grid in one wave of SMs;
  M <= 128   the 128-wide 2-CTA pair while its CTAs fit in one wave
             (N <= 9472), else a 128-wide one-CTA-per-tile config.

The N values select each MMA_N the M <= 64 band can pick (64, 128, 192, 256)
and sit on both sides of the one-wave boundary (9472 / 9552). 576 and 9552 end
in a partial 128-wide tile, and with 64-wide scales the tile stays within two
scale blocks. M = 1 uses a 4-wide A-scales row stride, as the TMA needs
16 byte rows.

Per case this checks, in addition to the naive reference:

- with 128-wide scales, bitwise equality with the previous dispatch's config
  (2-CTA pair, 256-wide for ceildiv(N, 128) >= 37) launched directly. With
  64-wide scales that old 256-wide tile mis-scales, so only the reference
  applies there;
- the `has_epilogue_fn` path (used by the KV-cache projections) stores the
  same bits as the plain path;
- a dynamic N, which keeps the 128-wide pair, produces the same bits too.

Every candidate tile gives the same bits, so those checks cannot see which
tile was chosen. `_check_tile_selection` pins the two selection helpers the
dispatcher calls (the `M <= 64` MMA_N and the `64 < M <= 128` pair bound) at
their switch points; the M = 64 / 128 band edges themselves are only exercised
by the M sweep. PDL ordering for these bands is covered by
`test_matmul_sm100_blockwise_fp8_pdl`.
"""

from std.math import align_up, ceildiv
from std.memory import bitcast
from std.random import rand, seed

from internal_utils import assert_almost_equal
from layout import Idx, TileTensor, row_major
from linalg.fp8_quantization import (
    _one_wave_mma_n,
    _pair_fits_wave,
    blockwise_scaled_fp8_with_epilogue,
    matmul_dynamic_scaled_fp8,
    naive_blockwise_scaled_fp8_matmul,
)
from linalg.matmul.gpu.sm100_structured.blockwise_fp8.blockwise_fp8_matmul import (
    blockwise_fp8_matmul,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    GEMMKind,
    MatmulConfig,
)
from max.gpu.host import DeviceContext
from max.gpu.host.info import B200
from std.testing import assert_equal, assert_false, assert_true
from std.utils.coord import Coord


def _assert_bitwise_equal(
    what: String,
    got: ImmPointer[Scalar[DType.bfloat16], ...],
    want: ImmPointer[Scalar[DType.bfloat16], ...],
    count: Int,
) raises:
    # Compare the raw bits: `!=` on floats treats +0.0 and -0.0 as equal and
    # every NaN as different from itself.
    var mismatches = 0
    for i in range(count):
        if bitcast[DType.uint16](got[i]) != bitcast[DType.uint16](want[i]):
            mismatches += 1
    if mismatches != 0:
        raise Error(
            String(
                what,
                ": ",
                mismatches,
                " of ",
                count,
                " outputs differ bitwise",
            )
        )


def _run_one[
    N: Int, K: Int, n_g: Int, extra_paths: Bool
](ctx: DeviceContext, m: Int) raises:
    comptime a_type = DType.float8_e4m3fn
    comptime c_type = DType.bfloat16
    comptime scale_type = DType.float32
    comptime k_blocks = K // 128
    comptime n_blocks = ceildiv(N, n_g)

    var sw = align_up(m, 4)
    print(t"M= {m}  N= {N}  K= {K}  n_scale_granularity= {n_g}")

    var a_host = ctx.enqueue_create_host_buffer[a_type](m * K)
    var b_host = ctx.enqueue_create_host_buffer[a_type](N * K)
    var as_host = ctx.enqueue_create_host_buffer[scale_type](k_blocks * sw)
    var bs_host = ctx.enqueue_create_host_buffer[scale_type](
        n_blocks * k_blocks
    )
    var c_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    var c_ref_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    var c_other_host = ctx.enqueue_create_host_buffer[c_type](m * N)
    rand(a_host.unsafe_ptr(), m * K)
    rand(b_host.unsafe_ptr(), N * K)
    rand(as_host.unsafe_ptr(), k_blocks * sw)
    rand(bs_host.unsafe_ptr(), n_blocks * k_blocks)

    var a_dev = ctx.enqueue_create_buffer[a_type](m * K)
    var b_dev = ctx.enqueue_create_buffer[a_type](N * K)
    var as_dev = ctx.enqueue_create_buffer[scale_type](k_blocks * sw)
    var bs_dev = ctx.enqueue_create_buffer[scale_type](n_blocks * k_blocks)
    var c_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_ref_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_other_dev = ctx.enqueue_create_buffer[c_type](m * N)
    var c_scratch_dev = ctx.enqueue_create_buffer[c_type](m * N)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(as_dev, as_host)
    ctx.enqueue_copy(bs_dev, bs_host)

    var a = TileTensor(a_dev, row_major(m, Idx[K]))
    var b = TileTensor(b_dev, row_major(Idx[N], Idx[K]))
    var a_scales = TileTensor(as_dev, row_major(Idx[k_blocks], sw))
    var b_scales = TileTensor(bs_dev, row_major(Idx[n_blocks], Idx[k_blocks]))
    var c = TileTensor(c_dev, row_major(m, Idx[N]))
    var c_ref = TileTensor(c_ref_dev, row_major(m, Idx[N]))

    naive_blockwise_scaled_fp8_matmul[
        BLOCK_DIM=16,
        transpose_b=True,
        scales_granularity_mnk=Coord[Int, Int, Int](1, n_g, 128),
    ](c_ref, a, b, a_scales, b_scales, ctx)

    matmul_dynamic_scaled_fp8[
        input_scale_granularity="block",
        weight_scale_granularity="block",
        m_scale_granularity=1,
        n_scale_granularity=n_g,
        k_scale_granularity=128,
        transpose_b=True,
        target="gpu",
    ](c, a, b, a_scales, b_scales, ctx)

    ctx.enqueue_copy(c_host, c_dev)
    ctx.enqueue_copy(c_ref_host, c_ref_dev)
    ctx.synchronize()

    assert_almost_equal(
        c_host.unsafe_ptr(),
        c_ref_host.unsafe_ptr(),
        m * N,
        atol=1e-2,
        rtol=1e-2,
    )

    # The previous dispatch for M <= 128: a 2-CTA pair, 256-wide when the N
    # grid was wide. Its 256-wide tile mis-scales 64-wide scales, so it is
    # only a valid baseline at 128.
    comptime if n_g == 128:
        comptime old_wide = ceildiv(N, 128) >= B200.sm_count // 4
        comptime old_config = MatmulConfig[a_type, a_type, c_type, True](
            cluster_shape=(2, 1, 1),
            mma_shape=(128, 256 if old_wide else 128, 32),
            cta_group=2,
            gemm_kind=GEMMKind.BLOCK_SCALED_1D2D_FP8,
        )
        var c_old = TileTensor(c_other_dev, row_major(m, Idx[N]))
        blockwise_fp8_matmul[
            transpose_b=True,
            a_scales_type=scale_type,
            b_scales_type=scale_type,
            config=old_config,
            n_scale_granularity=n_g,
        ](c_old, a, b, a_scales, b_scales, ctx)
        ctx.enqueue_copy(c_other_host, c_other_dev)
        ctx.synchronize()
        _assert_bitwise_equal(
            "previous dispatch config",
            c_host.unsafe_ptr(),
            c_other_host.unsafe_ptr(),
            m * N,
        )

    comptime if extra_paths:
        # `has_epilogue_fn`: the matmul lands in a scratch buffer and the
        # epilogue closure stores the final values.
        var c_out = TileTensor(c_other_dev, row_major(m, Idx[N]))
        var c_scratch = TileTensor(c_scratch_dev, row_major(m, Idx[N]))
        ctx.enqueue_memset[c_type](c_other_dev, 0)

        @always_inline
        def epilogue_fn[
            dtype: DType, width: SIMDLength, *, alignment: Int
        ](coords: Coord, val: SIMD[dtype, width]) {var c_out}:
            comptime assert coords.rank == 2
            comptime assert c_out.flat_rank >= 2
            c_out.store[alignment=alignment](
                coords, rebind[SIMD[c_type, width]](val)
            )

        blockwise_scaled_fp8_with_epilogue[
            transpose_b=True,
            scales_granularity_mnk=Coord[Int, Int, Int](1, n_g, 128),
            has_epilogue_fn=True,
        ](c_scratch, a, b, a_scales, b_scales, ctx, epilogue_fn)
        ctx.enqueue_copy(c_other_host, c_other_dev)
        ctx.synchronize()
        _assert_bitwise_equal(
            "epilogue path",
            c_host.unsafe_ptr(),
            c_other_host.unsafe_ptr(),
            m * N,
        )

        # Dynamic N: the grid size is unknown, so the 128-wide pair is kept.
        var n_rt = N
        var nb_rt = n_blocks
        var b_dyn = TileTensor(b_dev, row_major(n_rt, Idx[K]))
        var bs_dyn = TileTensor(bs_dev, row_major(nb_rt, Idx[k_blocks]))
        var c_dyn = TileTensor(c_other_dev, row_major(m, n_rt))
        ctx.enqueue_memset[c_type](c_other_dev, 0)
        matmul_dynamic_scaled_fp8[
            input_scale_granularity="block",
            weight_scale_granularity="block",
            m_scale_granularity=1,
            n_scale_granularity=n_g,
            k_scale_granularity=128,
            transpose_b=True,
            target="gpu",
        ](c_dyn, a, b_dyn, a_scales, bs_dyn, ctx)
        ctx.enqueue_copy(c_other_host, c_other_dev)
        ctx.synchronize()
        _assert_bitwise_equal(
            "dynamic N",
            c_host.unsafe_ptr(),
            c_other_host.unsafe_ptr(),
            m * N,
        )


def _run_m_sweep[
    N: Int, K: Int, n_g: Int, extra_paths: Bool = False
](ctx: DeviceContext) raises:
    # M = 1 (4-wide scales), both sides of the 64 and 128 band edges, and a
    # row count that is not a multiple of the MMA M tile.
    for m in [1, 4, 8, 60, 64, 68, 124, 128]:
        _run_one[N, K, n_g, extra_paths](ctx, m)


def _check_tile_selection() raises:
    # M <= 64 MMA_N with 128-wide scales (cap 256): the narrowest tile whose
    # N grid fits the 148 SMs.
    assert_equal(_one_wave_mma_n(576, 256), 64)
    assert_equal(_one_wave_mma_n(9472, 256), 64)
    assert_equal(_one_wave_mma_n(9473, 256), 128)
    assert_equal(_one_wave_mma_n(18944, 256), 128)
    assert_equal(_one_wave_mma_n(18945, 256), 192)
    assert_equal(_one_wave_mma_n(24576, 256), 192)
    assert_equal(_one_wave_mma_n(28416, 256), 192)
    assert_equal(_one_wave_mma_n(28417, 256), 256)
    assert_equal(_one_wave_mma_n(36864, 256), 256)
    # 64-wide scales cap the tile at two scale blocks.
    assert_equal(_one_wave_mma_n(9472, 128), 64)
    assert_equal(_one_wave_mma_n(24576, 128), 128)
    assert_equal(_one_wave_mma_n(36864, 128), 128)
    # 64 < M <= 128: the 2-CTA pair while 2 * ceildiv(N, 128) <= 148.
    assert_true(_pair_fits_wave(576))
    assert_true(_pair_fits_wave(9472))
    assert_false(_pair_fits_wave(9473))
    assert_false(_pair_fits_wave(36864))
    # Dynamic N keeps the pair.
    assert_true(_pair_fits_wave(-1))


def main() raises:
    seed(2758)
    _check_tile_selection()
    with DeviceContext() as ctx:
        # 128-wide scales (production DeepSeek/Qwen3/GLM weights). The
        # epilogue and dynamic-N paths run on a narrow N (pair in the 65..128
        # band) and a wide N (one CTA per tile there).
        _run_m_sweep[576, 512, 128, True](ctx)
        # Last N that fits the 2-CTA pair in one wave, and the first that
        # does not.
        _run_m_sweep[9472, 512, 128](ctx)
        _run_m_sweep[9552, 512, 128, True](ctx)
        # 192- and 256-wide M <= 64 tiles.
        _run_m_sweep[24576, 256, 128](ctx)
        _run_m_sweep[28672, 256, 128](ctx)
        # 64-wide scales: the M <= 64 tile is capped at two scale blocks.
        _run_m_sweep[5120, 256, 64](ctx)
        _run_m_sweep[9600, 512, 64](ctx)
        _run_m_sweep[24576, 256, 64](ctx)
