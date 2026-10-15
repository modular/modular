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
"""Shared harness for the SM100 W4A8 block-scaled matmul tests.

A is E4M3 (`K` bytes per row), B is E2M1 nibble-packed into `uint8` (`K/2`
bytes per row, element 2i in the low nibble), and both carry E8M0 scales over
32-element K blocks, consumed by one `kind::mxf8f6f4` MMA. `K` is always
counted in elements.

No vendor BLAS takes this operand pair and the naive block-scaled kernel
rejects mixed operand types, so the reference is computed on the host in
FP32. Two data modes:

- Exact (default). A comes from `A_VALUES`, B from all 16 E2M1 codes, scales
  from {1, 2, 4}, and `alpha` must be a power of two. Every product is a
  multiple of 0.25 and at most `2 * 6 * 4 * 4 = 192` in magnitude, so while
  `192 * K <= 2**22` every partial sum is exact in FP32 regardless of the
  accumulation order. The BF16 store is then the only rounding: one BF16 ulp.
- Realistic. A is drawn from a continuous range and rounded to E4M3, so FP32
  accumulation order matters. The bound adds `K * 2**-24 * sum(|terms|)`,
  the standard worst-case summation error, to the BF16 store rounding.

C is filled with NaN before launch: an output tile the kernel never writes
fails the full-output NaN scan, whatever the sampled comparison covers.
"""

from std.math import ceildiv
from std.random import random_float64, random_ui64, seed
from std.utils.numerics import isnan, nan

from max.gpu.host import DeviceContext, HostBuffer
from layout import Coord, Idx, TileTensor, row_major
from linalg.fp4_utils import (
    E2M1_TO_FLOAT32,
    MXFP8_SF_DTYPE,
    MXFP8_SF_VECTOR_SIZE,
    SF_ATOM_K,
    SF_ATOM_M,
    SF_MN_GROUP_SIZE,
    W4A8_A_DTYPE,
    W4A8_B_DTYPE,
)
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    blackwell_block_scaled_matmul_tma_umma_warp_specialized,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig, GEMMKind
from linalg.utils import _as_compute_fn
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from std.simd import _convert_f32_to_float8_ue8m0
from std.testing import assert_almost_equal, assert_false, assert_true
from std.utils.index import IndexList

comptime SF_VECTOR_SIZE = MXFP8_SF_VECTOR_SIZE
comptime SF_DTYPE = MXFP8_SF_DTYPE
comptime C_DTYPE = DType.bfloat16
comptime MMA_K = 32
"""`kind::mxf8f6f4` consumes 32 K elements (32 shared-memory bytes) per MMA."""

comptime A_VALUES = SIMD[.float32, 8](0.0, 0.5, 1.0, 2.0, -0.5, -1.0, -2.0, 1.5)
"""Exact in E4M3; with E2M1 and scales in {1, 2, 4}, products stay small."""

comptime MAX_EXACT_TERM = 192
"""Largest |a * b * sfa * sfb| in exact mode: 2 * 6 * 4 * 4."""

comptime BF16_RTOL = 1.0 / 256.0
"""One BF16 ulp, the rounding of the FP32 accumulator on the output store."""

comptime FP32_UNIT_ROUNDOFF: Float32 = 1.0 / 16777216.0
"""2**-24."""


def _e8m0_pow2(exponent: Int) -> Scalar[SF_DTYPE]:
    return _convert_f32_to_float8_ue8m0[SF_DTYPE](Float32(1 << exponent))


def _sf_offset(row: Int, col: Int, k_groups: Int) -> Int:
    """Returns the offset of `(row, col)`'s scale in one batch's scale block.

    The block is row-major `(rows/128, K/128, 32, 4, 4)`, the coordinates
    `set_scale_factor` and `get_scale_factor` use.
    """
    return (
        (
            (
                (row // SF_MN_GROUP_SIZE) * k_groups
                + col // (SF_VECTOR_SIZE * SF_ATOM_K)
            )
            * SF_ATOM_M[0]
            + row % SF_ATOM_M[0]
        )
        * SF_ATOM_M[1]
        + (row % SF_MN_GROUP_SIZE) // SF_ATOM_M[0]
    ) * SF_ATOM_K + (col // SF_VECTOR_SIZE) % SF_ATOM_K


def run_w4a8_block_scaled_matmul[
    N: Int,
    K: Int,
    *,
    mma_n: Int,
    cta_group: Int,
    cluster_shape: IndexList[3],
    swapAB: Bool = False,
    k_group_size: Int = 1,
    block_swizzle_size: Int = 0,
    num_accum_pipeline_stages: Int = 0,
    num_clc_pipeline_stages: Int = 2,
    prefetch_tiles_n: Int = 0,
    fused_epilogue: Bool = False,
    batch: Int = 1,
    batched: Bool = False,
    realistic_data: Bool = False,
](ctx: DeviceContext, M: Int, alpha: Float32 = 1.0) raises:
    """Runs one W4A8 matmul through the SM100 block-scaled kernel and checks it.

    Parameters:
        N: Output columns (B rows).
        K: Reduction extent in elements; a multiple of 128, which the packed
            FP4 TMA copy requires.
        mma_n: MMA N extent; the MMA shape is `(cta_group * 128, mma_n, 32)`.
        cta_group: 1 or 2 CTAs per MMA.
        cluster_shape: Thread-block cluster shape.
        swapAB: Computes `C^T = B A^T`, putting the FP4 operand in the A slot.
        k_group_size: K tiles per pipeline stage.
        block_swizzle_size: Tile raster swizzle.
        num_accum_pipeline_stages: TMEM accumulator stages; 0 picks the
            default for `mma_n`.
        num_clc_pipeline_stages: Cluster-launch-control stages.
        prefetch_tiles_n: Weight tiles loaded before the PDL wait.
        fused_epilogue: Routes the output through a compute closure that
            doubles it, so a skipped closure fails the comparison.
        batch: Batch count; must be 1 unless `batched`.
        batched: Passes rank-3 operands and rank-6 scales.
        realistic_data: Uses continuous A values and a derived error bound
            instead of the exact data set.

    Args:
        ctx: Device context.
        M: Output rows (A rows), runtime.
        alpha: Output scale; a power of two in exact mode.
    """
    comptime assert K % 128 == 0, "packed FP4 TMA needs K % 128 == 0"
    comptime assert (
        realistic_data or MAX_EXACT_TERM * K <= 2**22
    ), "exact mode loses FP32 exactness past K = 21845"
    comptime assert batched or batch == 1, "batch > 1 requires batched"
    comptime assert not (
        fused_epilogue and batched
    ), "the fused-epilogue closure is written for rank-2 output"
    comptime a_type = W4A8_A_DTYPE
    comptime b_type = W4A8_B_DTYPE
    comptime K_PACKED = K // 2
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    comptime output_scale = Float32(2.0) if fused_epilogue else Float32(1.0)

    print(
        t"[w4a8] batch={batch} M={M} N={N} K={K} mma_n={mma_n}"
        t" cta_group={cta_group} cluster={cluster_shape} swapAB={swapAB}"
        t" k_group_size={k_group_size} prefetch_tiles_n={prefetch_tiles_n}"
        t" fused_epilogue={fused_epilogue} realistic={realistic_data}"
        t" alpha={alpha}"
    )
    seed(1234)

    var a_elems = batch * M * K
    var b_elems = batch * N * K_PACKED
    var c_elems = batch * M * N
    var sfa_batch_stride = ceildiv(M, SF_MN_GROUP_SIZE) * k_groups * sf_atom
    var sfb_batch_stride = ceildiv(N, SF_MN_GROUP_SIZE) * k_groups * sf_atom

    var a_host = ctx.enqueue_create_host_buffer[a_type](a_elems)
    var b_host = ctx.enqueue_create_host_buffer[b_type](b_elems)
    var c_host = ctx.enqueue_create_host_buffer[C_DTYPE](c_elems)
    var sfa_host = ctx.enqueue_create_host_buffer[SF_DTYPE](
        batch * sfa_batch_stride
    )
    var sfb_host = ctx.enqueue_create_host_buffer[SF_DTYPE](
        batch * sfb_batch_stride
    )
    ctx.synchronize()
    fill_w4a8_operands[N, K, realistic_data](
        a_host, b_host, sfa_host, sfb_host, batch, M
    )

    var a_dev = ctx.enqueue_create_buffer[a_type](a_elems)
    var b_dev = ctx.enqueue_create_buffer[b_type](b_elems)
    var c_dev = ctx.enqueue_create_buffer[C_DTYPE](c_elems)
    var sfa_dev = ctx.enqueue_create_buffer[SF_DTYPE](batch * sfa_batch_stride)
    var sfb_dev = ctx.enqueue_create_buffer[SF_DTYPE](batch * sfb_batch_stride)
    ctx.enqueue_copy(a_dev, a_host)
    ctx.enqueue_copy(b_dev, b_host)
    ctx.enqueue_copy(sfa_dev, sfa_host)
    ctx.enqueue_copy(sfb_dev, sfb_host)
    ctx.enqueue_memset(c_dev, nan[C_DTYPE]())

    comptime default_accum_stages = 1 if mma_n in (192, 256) else 2
    comptime accum_stages = (
        num_accum_pipeline_stages if num_accum_pipeline_stages
        > 0 else default_accum_stages
    )
    comptime config = BlockScaledMatmulConfig[
        a_type, b_type, C_DTYPE, SF_DTYPE, SF_DTYPE, True
    ](
        scaling_kind=UMMAKind.KIND_MXF8F6F4,
        cluster_shape=cluster_shape,
        mma_shape=(cta_group * 128, mma_n, MMA_K),
        block_swizzle_size=block_swizzle_size,
        cta_group=cta_group,
        AB_swapped=swapAB,
        k_group_size=k_group_size,
        num_accum_pipeline_stages=accum_stages,
        num_clc_pipeline_stages=num_clc_pipeline_stages,
        gemm_kind=GEMMKind.BMM if batched else GEMMKind.GEMM,
        prefetch_tiles_n=prefetch_tiles_n,
    )

    comptime if batched:
        blackwell_block_scaled_matmul_tma_umma_warp_specialized[
            transpose_b=True, K=K, config=config
        ](
            TileTensor(c_dev, row_major(Coord(Idx[batch], M, Idx[N]))),
            TileTensor(a_dev, row_major(Coord(Idx[batch], M, Idx[K]))),
            TileTensor(
                b_dev, row_major(Coord(Idx[batch], Idx[N], Idx[K_PACKED]))
            ),
            TileTensor(
                sfa_dev,
                row_major(
                    Coord(
                        Idx[batch],
                        ceildiv(M, SF_MN_GROUP_SIZE),
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            ),
            TileTensor(
                sfb_dev,
                row_major(
                    Coord(
                        Idx[batch],
                        Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
                        Idx[k_groups],
                        Idx[SF_ATOM_M[0]],
                        Idx[SF_ATOM_M[1]],
                        Idx[SF_ATOM_K],
                    )
                ),
            ),
            ctx,
            alpha,
        )
    else:
        var c_tensor = TileTensor(c_dev, row_major(Coord(M, Idx[N])))
        var a_tensor = TileTensor(a_dev, row_major(Coord(M, Idx[K])))
        var b_tensor = TileTensor(
            b_dev, row_major(Coord(Idx[N], Idx[K_PACKED]))
        )
        var sfa_tensor = TileTensor(
            sfa_dev,
            row_major(
                Coord(
                    ceildiv(M, SF_MN_GROUP_SIZE),
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )
        var sfb_tensor = TileTensor(
            sfb_dev,
            row_major(
                Coord(
                    Idx[ceildiv(N, SF_MN_GROUP_SIZE)],
                    Idx[k_groups],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                )
            ),
        )

        @inline(.always)
        def double[
            _dtype: DType, width: SIMDLength, *, alignment: Int
        ](idx: Coord, val: SIMD[_dtype, width]) -> SIMD[_dtype, width]:
            comptime assert (
                idx.rank == 2
            ), "the matmul epilogue contract is rank 2"
            return val * 2

        comptime if fused_epilogue:
            blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                transpose_b=True, K=K, config=config
            ](
                c_tensor,
                a_tensor,
                b_tensor,
                sfa_tensor,
                sfb_tensor,
                _as_compute_fn(double),
                ctx,
                alpha,
            )
        else:
            blackwell_block_scaled_matmul_tma_umma_warp_specialized[
                transpose_b=True, K=K, config=config
            ](c_tensor, a_tensor, b_tensor, sfa_tensor, sfb_tensor, ctx, alpha)

    ctx.enqueue_copy(c_host, c_dev)
    ctx.synchronize()

    assert_w4a8_output_matches_reference[N, K, realistic_data](
        c_host,
        a_host,
        b_host,
        sfa_host,
        sfb_host,
        batch,
        M,
        alpha * output_scale,
    )
    print("  passed")

    _ = a_dev^
    _ = b_dev^
    _ = c_dev^
    _ = sfa_dev^
    _ = sfb_dev^


def fill_w4a8_operands[
    N: Int, K: Int, realistic_data: Bool
](
    a_host: HostBuffer[W4A8_A_DTYPE],
    b_host: HostBuffer[W4A8_B_DTYPE],
    sfa_host: HostBuffer[SF_DTYPE],
    sfb_host: HostBuffer[SF_DTYPE],
    batch: Int,
    M: Int,
):
    """Fills operands and scales for the chosen data mode.

    Scale rows past M or N are padding the kernel still reads; they are
    zeroed, since nonzero padding corrupts the tiles sharing their 128-row
    group.
    """
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    var sfa_batch_stride = ceildiv(M, SF_MN_GROUP_SIZE) * k_groups * sf_atom
    var sfb_batch_stride = ceildiv(N, SF_MN_GROUP_SIZE) * k_groups * sf_atom

    for i in range(batch * M * K):
        comptime if realistic_data:
            a_host[i] = Float32(random_float64(-8.0, 8.0)).cast[W4A8_A_DTYPE]()
        else:
            a_host[i] = A_VALUES[Int(random_ui64(0, 7))].cast[W4A8_A_DTYPE]()
    # `K // 2` is a multiple of 64, so whole 8-byte words tile the buffer.
    for word in range(batch * N * (K // 2) // 8):
        var bits = random_ui64(0, UInt64.MAX)
        for byte in range(8):
            b_host[word * 8 + byte] = UInt8((bits >> UInt64(8 * byte)) & 0xFF)

    for i in range(batch * sfa_batch_stride):
        sfa_host[i] = Scalar[SF_DTYPE](0.0)
    for i in range(batch * sfb_batch_stride):
        sfb_host[i] = Scalar[SF_DTYPE](0.0)
    for b in range(batch):
        for row in range(M):
            for col in range(0, K, SF_VECTOR_SIZE):
                sfa_host[
                    b * sfa_batch_stride + _sf_offset(row, col, k_groups)
                ] = _e8m0_pow2(Int(random_ui64(0, 2)))
        for row in range(N):
            for col in range(0, K, SF_VECTOR_SIZE):
                sfb_host[
                    b * sfb_batch_stride + _sf_offset(row, col, k_groups)
                ] = _e8m0_pow2(Int(random_ui64(0, 2)))


def assert_w4a8_output_matches_reference[
    N: Int, K: Int, realistic_data: Bool
](
    c_host: HostBuffer[C_DTYPE],
    a_host: HostBuffer[W4A8_A_DTYPE],
    b_host: HostBuffer[W4A8_B_DTYPE],
    sfa_host: HostBuffer[SF_DTYPE],
    sfb_host: HostBuffer[SF_DTYPE],
    batch: Int,
    M: Int,
    output_scale: Float32,
) raises:
    """Checks every output for the NaN sentinel, then a sample against FP32.

    Exhaustive checking is O(batch * M * N * K) on the host in a debug build,
    so each row compares ~32 columns (rotated per row so every column residue
    is hit across rows) plus its last column. Each compared element is held
    to the full bound.
    """
    comptime k_groups = ceildiv(K, SF_VECTOR_SIZE * SF_ATOM_K)
    comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
    comptime K_PACKED = K // 2
    var sfa_batch_stride = ceildiv(M, SF_MN_GROUP_SIZE) * k_groups * sf_atom
    var sfb_batch_stride = ceildiv(N, SF_MN_GROUP_SIZE) * k_groups * sf_atom

    for i in range(batch * M * N):
        assert_false(
            isnan(c_host[i]), msg=String("output ", i, " was never written")
        )

    var n_stride = max(N // 32, 1)
    for b in range(batch):
        for m in range(M):
            for j in range(0, N + n_stride, n_stride):
                var n = N - 1 if j >= N else (j + m * 7) % N
                var acc = Float32(0.0)
                var abs_acc = Float32(0.0)
                for kb in range(0, K, SF_VECTOR_SIZE):
                    var scale = (
                        sfa_host[
                            b * sfa_batch_stride + _sf_offset(m, kb, k_groups)
                        ].cast[.float32]()
                        * sfb_host[
                            b * sfb_batch_stride + _sf_offset(n, kb, k_groups)
                        ].cast[.float32]()
                    )
                    var block = Float32(0.0)
                    var abs_block = Float32(0.0)
                    for k in range(kb, kb + SF_VECTOR_SIZE):
                        var av = a_host[(b * M + m) * K + k].cast[.float32]()
                        var packed = Int(
                            b_host[(b * N + n) * K_PACKED + k // 2]
                        )
                        var nibble = (packed >> 4) if k % 2 else (packed & 0xF)
                        var product = av * E2M1_TO_FLOAT32[nibble]
                        block += product
                        abs_block += abs(product)
                    acc += block * scale
                    abs_acc += abs_block * scale
                var expected = acc * output_scale
                var got = c_host[(b * M + m) * N + n].cast[.float32]()
                var msg = String("batch=", b, " m=", m, " n=", n)
                comptime if realistic_data:
                    var accum_bound = (
                        Float32(K)
                        * FP32_UNIT_ROUNDOFF
                        * abs_acc
                        * abs(output_scale)
                    )
                    assert_true(
                        abs(got - expected)
                        <= Float32(BF16_RTOL) * abs(expected) + accum_bound,
                        msg=msg + String(" got=", got, " expected=", expected),
                    )
                else:
                    assert_almost_equal(
                        got, expected, atol=0.0, rtol=BF16_RTOL, msg=msg
                    )
