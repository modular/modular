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


"""Provides dispatch logic for SM100 block-scaled (NVFP4, MXFP4, MXFP8) matmul kernels with optional elementwise epilogue."""


from std.math import ceildiv
from max.gpu.host import DeviceContext
from max.gpu.primitives.grid_controls import PDLLevel
from layout import Coord, Idx, TileTensor
from layout.tile_layout import Layout as TileLayout
from std.logger import Logger
from linalg.fp4_utils import (
    SF_ATOM_M,
    SF_ATOM_K,
    SF_MN_GROUP_SIZE,
    NVFP4_SF_VECTOR_SIZE,
    MXFP4_SF_VECTOR_SIZE,
    MXFP8_SF_VECTOR_SIZE,
    get_scaling_kind,
)
from max.gpu.host.info import B200, _is_sm10x_gpu
from std.collections import Optional
from linalg.utils import (
    ElementwiseComputeFn,
    ElementwiseEpilogueFn,
    no_compute_fn,
    no_epilogue_fn,
)
from max.gpu.host.nvidia.tma import TensorMapSwizzle
from std.sys import size_of
from std.sys.info import simd_width_of
from max.algorithm import elementwise
from max.gpu.compute.arch.mma_nvidia_sm100 import UMMAKind
from linalg.matmul.gpu.sm100.block_scaled_matmul import (
    _blackwell_block_scaled_matmul_impl,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.matmul.gpu.sm100_structured.default.tuning_configs import (
    TuningConfigSM100,
    _get_tuning_list_sm100_nvfp4,
    _get_tuning_list_sm100_mxfp4,
    _get_tuning_list_sm100_mxfp8,
)
from internal_utils import Table
from linalg.matmul.gpu.sm100_structured.structured_kernels.config import (
    BLOCK_SCALED_CONFIGS_MAX_M,
    build_block_scaled_configs,
    choose_block_scaled_config,
)

comptime logger = Logger()

# Largest M whose compute closure runs in a separate elementwise pass instead
# of inside the GEMM epilogue; B200 measurements cover M <= 512.
comptime compute_fn_unfused_m_cap = 512

comptime DISPATCH_MISS = 0
comptime DISPATCH_HIT = 1


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an SM100 block-scaled matmul by selecting a tuning config from
    per-format outlier tables for specific M ranges, falling back to a
    small-BN config for GEMVs (`m == 1`) and a heuristic config table for the
    remaining cases. Returns `DISPATCH_HIT` when a matching config is found and
    launched, or `DISPATCH_MISS` when no config matches.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the LHS input tensor `a` (inferred).
        b_type: Element type of the RHS input tensor `b` (inferred).
        scales_dtype: Element type of the per-block scale tensors
            `a_scales` and `b_scales` (inferred).
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
            Must match the format: 16 for NVFP4, 32 for MXFP4, or 32
            for MXFP8.
        transpose_b: Whether `b` is stored transposed. Must be `True`
            (defaults to `True`).
        pdl_level: Programmatic Dependent Launch scheduling level for
            overlapping this kernel with prior GPU work (defaults to
            `PDLLevel()`).

    Args:
        c: Output TileTensor accumulating the matmul result.
        a: LHS input TileTensor.
        b: RHS input TileTensor (must be transposed).
        a_scales: Per-block scales for `a`.
        b_scales: Per-block scales for `b`.
        tensor_sf: Global tensor scaling factor applied as `alpha`.
        ctx: Device context used to launch the kernel.

    Returns:
        `DISPATCH_HIT` if a config was selected and the kernel launched, otherwise `DISPATCH_MISS`.
    """
    return _heuristic_and_outliers_dispatch_impl[
        SF_VECTOR_SIZE,
        transpose_b,
        pdl_level=pdl_level,
        has_epilogue_fn=False,
        has_compute_fn=False,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        tensor_sf,
        no_epilogue_fn,
        no_compute_fn,
        ctx,
    )


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    //,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    epilogue_fn: EpilogueFnType,
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an SM100 block-scaled matmul (see the other overload),
    delivering each output value to an epilogue closure instead of writing
    `c`.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the LHS input tensor `a` (inferred).
        b_type: Element type of the RHS input tensor `b` (inferred).
        scales_dtype: Element type of the per-block scale tensors (inferred).
        EpilogueFnType: Type of `epilogue_fn` (inferred).
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
        transpose_b: Whether `b` is stored transposed. Must be `True`.
        pdl_level: Programmatic Dependent Launch scheduling level.

    Args:
        c: Output TileTensor; only its shape is read.
        a: LHS input TileTensor.
        b: RHS input TileTensor (must be transposed).
        a_scales: Per-block scales for `a`.
        b_scales: Per-block scales for `b`.
        tensor_sf: Global tensor scaling factor applied as `alpha`.
        epilogue_fn: Stores each output value at its global `(m, n)`.
        ctx: Device context used to launch the kernel.

    Returns:
        `DISPATCH_HIT` if a config was selected and the kernel launched, otherwise `DISPATCH_MISS`.
    """
    return _heuristic_and_outliers_dispatch_impl[
        SF_VECTOR_SIZE,
        transpose_b,
        pdl_level=pdl_level,
        has_epilogue_fn=True,
        has_compute_fn=False,
    ](c, a, b, a_scales, b_scales, tensor_sf, epilogue_fn, no_compute_fn, ctx)


def heuristic_and_outliers_dispatch[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    ComputeFnType: ElementwiseComputeFn,
    //,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    pdl_level: PDLLevel = PDLLevel(),
    compute_fn_needs_vectors: Bool = False,
    compute_fn_col_period: Int = 0,
    compute_fn_col_start: Int = 0,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises -> Int:
    """Dispatches an SM100 block-scaled matmul (see the other overload),
    storing `compute_fn(idx, value)` into `c`.

    Parameters:
        c_type: Element type of the output tensor `c` (inferred).
        a_type: Element type of the LHS input tensor `a` (inferred).
        b_type: Element type of the RHS input tensor `b` (inferred).
        scales_dtype: Element type of the per-block scale tensors (inferred).
        ComputeFnType: Type of `compute_fn` (inferred).
        SF_VECTOR_SIZE: Number of elements each scale factor covers.
        transpose_b: Whether `b` is stored transposed. Must be `True`.
        pdl_level: Programmatic Dependent Launch scheduling level.
        compute_fn_needs_vectors: Whether `compute_fn` reads each value
            with its neighbors in the row, so it must get consecutive
            columns rather than one element at a time.
        compute_fn_col_period: When nonzero, `compute_fn` returns its
            input unchanged at every column whose index modulo this
            period is below `compute_fn_col_start`, so an unfused pass
            visits only the remaining columns.
        compute_fn_col_start: The first column of each period that
            `compute_fn` may change.

    Args:
        c: Output TileTensor accumulating the matmul result.
        a: LHS input TileTensor.
        b: RHS input TileTensor (must be transposed).
        a_scales: Per-block scales for `a`.
        b_scales: Per-block scales for `b`.
        tensor_sf: Global tensor scaling factor applied as `alpha`.
        compute_fn: Transforms each output value before it is stored.
        ctx: Device context used to launch the kernel.

    Returns:
        `DISPATCH_HIT` if a config was selected and the kernel launched, otherwise `DISPATCH_MISS`.
    """
    return _heuristic_and_outliers_dispatch_impl[
        SF_VECTOR_SIZE,
        transpose_b,
        pdl_level=pdl_level,
        has_epilogue_fn=False,
        has_compute_fn=True,
        compute_fn_needs_vectors=compute_fn_needs_vectors,
        compute_fn_col_period=compute_fn_col_period,
        compute_fn_col_start=compute_fn_col_start,
    ](c, a, b, a_scales, b_scales, tensor_sf, no_epilogue_fn, compute_fn, ctx)


def _heuristic_and_outliers_dispatch_impl[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool = True,
    *,
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_fn: Bool,
    has_compute_fn: Bool,
    compute_fn_needs_vectors: Bool = False,
    compute_fn_col_period: Int = 0,
    compute_fn_col_start: Int = 0,
    groups: Int = 1,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises -> Int:
    var m = Int(c.dim[0]())

    comptime scaling_kind = get_scaling_kind[
        a_type, scales_dtype, SF_VECTOR_SIZE, b_type
    ]()
    comptime is_fp4 = (
        scaling_kind == UMMAKind.KIND_MXF4NVF4
        or scaling_kind == UMMAKind.KIND_MXF4
    )

    comptime MMA_K = 32
    comptime BK = (TensorMapSwizzle.SWIZZLE_128B.bytes() // size_of[a_type]())
    # Config choice treats `groups` block-diagonal blocks of [M, N/groups,
    # K/groups] as one [M, N, K/groups] problem: the same tiles and K loop.
    comptime group_K_phys = a.static_shape[1] // groups
    comptime num_k_iters = ceildiv(group_K_phys, BK)
    comptime static_N = c.static_shape[1]
    comptime static_K = group_K_phys * 2 if is_fp4 else group_K_phys

    comptime assert _is_sm10x_gpu(
        ctx.default_device_info
    ), "This kernel is only supported on SM100"

    comptime assert transpose_b, "Only support transposed B"

    comptime assert (
        (
            scaling_kind == UMMAKind.KIND_MXF4NVF4
            and SF_VECTOR_SIZE == NVFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF4
            and SF_VECTOR_SIZE == MXFP4_SF_VECTOR_SIZE
        )
        or (
            scaling_kind == UMMAKind.KIND_MXF8F6F4
            and SF_VECTOR_SIZE == MXFP8_SF_VECTOR_SIZE
        )
    ), "Only support NVFP4, MXFP4, or MXFP8 scale/dtype combinations."

    comptime assert (
        a_scales.static_shape[1] == groups * b_scales.static_shape[1]
    ), "Both A and B scales must have the same shape in K dimension"
    comptime assert (
        a_scales.static_shape[2] == b_scales.static_shape[2] == SF_ATOM_M[0]
    ), ""
    comptime assert (
        a_scales.static_shape[3] == b_scales.static_shape[3] == SF_ATOM_M[1]
    ), ""
    comptime assert (
        a_scales.static_shape[4] == b_scales.static_shape[4] == SF_ATOM_K
    ), ""

    comptime outliers = Table(
        _get_tuning_list_sm100_nvfp4(), "nvfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4NVF4 else Table(
        _get_tuning_list_sm100_mxfp4(), "mxfp4_heuristic_outliers"
    ) if scaling_kind == UMMAKind.KIND_MXF4 else Table(
        _get_tuning_list_sm100_mxfp8(), "mxfp8_heuristic_outliers"
    )

    comptime outlier_configs = outliers.find(
        rule=lambda (x: TuningConfigSM100) -> Bool: x.K == static_K
        and x.N == static_N
    )

    comptime for tuning_config in outlier_configs:
        if m >= tuning_config.M and m < tuning_config.M_end:
            comptime table_config = BlockScaledMatmulConfig[
                a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
            ](
                scaling_kind=scaling_kind,
                mma_shape=tuning_config.mma_shape,
                cta_group=tuning_config.cta_group,
                cluster_shape=tuning_config.cluster_shape,
                block_swizzle_size=tuning_config.block_swizzle_size,
                raster_order=tuning_config.rasterize_order,
                AB_swapped=tuning_config.swapAB,
                num_accum_pipeline_stages=tuning_config.num_accum_pipeline_stages,
                num_clc_pipeline_stages=tuning_config.num_clc_pipeline_stages,
                k_group_size=tuning_config.k_group_size,
                num_split_k=tuning_config.num_split_k,
                num_pipeline_stages=Optional(
                    tuning_config.num_pipeline_stages
                ) if tuning_config.num_pipeline_stages
                > 0 else None,
                is_small_bn=tuning_config.is_small_bn,
            )
            # At decode M the GEMM is resident while its producer drains, so
            # it can start the weight stream before the PDL wait. A long K loop
            # also gains from prefetching its weights into L2 ahead of their
            # loads, which keeps more DRAM reads in flight than the ring does.
            comptime matmul_config = table_config.with_weight_prefetch(
                l2_depth=4 if tuning_config.is_small_bn
                and static_K >= 8192 else 0
            ) if (
                pdl_level > PDLLevel.OFF
                and scaling_kind == UMMAKind.KIND_MXF8F6F4
                and tuning_config.M_end <= 256
            ) else table_config

            logger.info("Using tuning config: ", matmul_config)

            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=matmul_config,
                pdl_level=pdl_level,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                compute_fn_needs_vectors=compute_fn_needs_vectors,
                compute_fn_col_period=compute_fn_col_period,
                compute_fn_col_start=compute_fn_col_start,
                groups=groups,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                tensor_sf,
                epilogue_fn,
                compute_fn,
                ctx,
            )

            return DISPATCH_HIT

    # Dispatch to the small-BN kernel for the small-M decode regime (m <= 16:
    # m == 1 GEMV plus small-batch / speculative-decode m ~ 8, 16). It is
    # optimized for skinny GEMMs; MMA_N=8 tiles the M dim (m=16 -> 2 tiles).
    # Larger M keeps the cta_group=2 prefill heuristic below.
    #
    # For MXFP8, m in (8, 16] needs two M tiles, so every weight tile is
    # launched twice. When that exceeds one wave of SMs (N >= ~9.5K, e.g. the
    # fused M3.1 QKV projection), the heuristic config wins: 14us vs 22us at
    # M=16, N=9856, K=6144 on B200. FP4 formats keep the small-BN kernel.
    # The wave estimate uses B200's SM count; revisit for other SM10x parts.
    # Skipping small-BN exposes heuristic misses to the vendor fallback.
    comptime two_m_tiles_fit_one_wave = (
        scaling_kind != UMMAKind.KIND_MXF8F6F4
        or ceildiv(static_N, 128) * 2 <= B200.sm_count
    )
    if m <= 8 or (m <= 16 and two_m_tiles_fit_one_wave):
        # Larger k-groups shorten the mainloop for this latency-bound skinny
        # decode GEMM (K=6144 -> 48 k-iters; kg=4 -> 12 groups vs 24), which an
        # isolated B200 sweep at the served QKV shapes (N in {2304,2560}, K=6144,
        # M<=16) showed is ~8% faster than kg=2 and ~40% faster than vendor
        # cuBLASLt, bit-exact. Needs num_pipeline_stages % kg == 0 (12 % 4 == 0).
        comptime k_group_size = (
            4 if num_k_iters % 4 == 0 else (2 if num_k_iters % 2 == 0 else 1)
        )
        comptime config = BlockScaledMatmulConfig[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](
            scaling_kind=scaling_kind,
            cta_group=1,
            mma_shape=(128, 8, 32),
            cluster_shape=(1, 1, 1),
            block_swizzle_size=8,
            num_accum_pipeline_stages=1,
            k_group_size=k_group_size,
            num_clc_pipeline_stages=0,
            AB_swapped=True,
            is_small_bn=True,
        )
        _block_scaled_matmul_with_epilogue[
            SF_VECTOR_SIZE=SF_VECTOR_SIZE,
            transpose_b=transpose_b,
            config=config,
            pdl_level=pdl_level,
            has_epilogue_fn=has_epilogue_fn,
            has_compute_fn=has_compute_fn,
            compute_fn_needs_vectors=compute_fn_needs_vectors,
            compute_fn_col_period=compute_fn_col_period,
            compute_fn_col_start=compute_fn_col_start,
            groups=groups,
        ](
            c,
            a,
            b,
            a_scales,
            b_scales,
            tensor_sf,
            epilogue_fn,
            compute_fn,
            ctx,
        )

        logger.info("Using small-BN config: ", config)
        return DISPATCH_HIT

    comptime configs = build_block_scaled_configs[
        a_type,
        b_type,
        c_type,
        scales_dtype,
        scales_dtype,
        static_N,
        static_K,
        transpose_b,
    ]()
    var config_runtime = choose_block_scaled_config[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ](m, static_N, static_K)

    comptime for config in configs:
        if config_runtime == config:
            logger.info("Using heuristic config: ", config)
            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=config,
                pdl_level=pdl_level,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                compute_fn_needs_vectors=compute_fn_needs_vectors,
                compute_fn_col_period=compute_fn_col_period,
                compute_fn_col_start=compute_fn_col_start,
                groups=groups,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                tensor_sf,
                epilogue_fn,
                compute_fn,
                ctx,
            )
            return DISPATCH_HIT

    # A block-diagonal matmul has no vendor fallback. Every prebuilt config
    # runs any M, so a miss runs the one chosen at the end of the M sweep.
    comptime if groups > 1:
        comptime fallback = choose_block_scaled_config[
            a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
        ](BLOCK_SCALED_CONFIGS_MAX_M, static_N, static_K)
        comptime if fallback in configs:
            logger.info("Using fallback config: ", fallback)
            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=fallback,
                pdl_level=pdl_level,
                has_epilogue_fn=has_epilogue_fn,
                has_compute_fn=has_compute_fn,
                compute_fn_needs_vectors=compute_fn_needs_vectors,
                compute_fn_col_period=compute_fn_col_period,
                compute_fn_col_start=compute_fn_col_start,
                groups=groups,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                tensor_sf,
                epilogue_fn,
                compute_fn,
                ctx,
            )
            return DISPATCH_HIT

    return DISPATCH_MISS


def block_scaled_matmul_block_diagonal[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    //,
    *,
    groups: Int,
    SF_VECTOR_SIZE: Int,
    pdl_level: PDLLevel = PDLLevel(),
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    ctx: DeviceContext,
) raises:
    """Computes a block-diagonal block-scaled matmul on SM100.

    Block `g` of `groups` multiplies A's columns `[g * K, (g + 1) * K)` by
    B's rows `[g * N, (g + 1) * N)` and writes C's columns
    `[g * N, (g + 1) * N)`, where `K = b.dim[1]` and
    `N = c.dim[1] / groups`. The blocks run as one batched matmul over
    strided views, so A and C stay token-major and the scales stay in the
    layouts the dense quantize and the weights already use.

    Parameters:
        c_type: Element type of `c` (inferred).
        a_type: Element type of `a` (inferred).
        b_type: Element type of `b` (inferred).
        scales_dtype: Element type of both scale tensors (inferred).
        groups: The number of diagonal blocks.
        SF_VECTOR_SIZE: Elements per scale factor.
        pdl_level: Programmatic dependent launch level.

    Args:
        c: Output `[M, groups * N]`.
        a: `[M, groups * K]` operand.
        b: `[groups * N, K]` operand, K-major.
        a_scales: `a`'s scales in the 5D SF-atom layout of the whole
            `[M, groups * K]` operand.
        b_scales: `b`'s scales in the 5D SF-atom layout.
        tensor_sf: Scalar multiplier applied to the product.
        ctx: Device context used to launch the kernel.

    Raises:
        If no SM100 kernel config covers the shape.
    """
    comptime assert c.rank == 2 and a.rank == 2 and b.rank == 2
    comptime assert a_scales.rank == 5 and b_scales.rank == 5
    comptime N_g = c.static_shape[1] // groups
    comptime assert (
        c.static_shape[1] == groups * N_g
        and b.static_shape[0] == groups * N_g
        and a.static_shape[1] == groups * b.static_shape[1]
    ), "c, a and b must hold `groups` equal blocks"
    comptime assert N_g % SF_MN_GROUP_SIZE == 0, (
        "each block's N must be a multiple of the scale tile rows, so its"
        " B scales start on a tile"
    )
    comptime assert (
        a_scales.static_shape[1] == groups * b_scales.static_shape[1]
    ), "each block's K must be a whole number of scale tiles"

    if c.dim[0]() == 0:
        return

    var status = _heuristic_and_outliers_dispatch_impl[
        SF_VECTOR_SIZE=SF_VECTOR_SIZE,
        transpose_b=True,
        pdl_level=pdl_level,
        has_epilogue_fn=False,
        has_compute_fn=False,
        groups=groups,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        tensor_sf,
        no_epilogue_fn,
        no_compute_fn,
        ctx,
    )
    if status != DISPATCH_HIT:
        raise Error("no SM100 block-scaled config for the block-diagonal shape")


########################################################
# SM100 Block Scaled matmul with normal epilogue kernel dispatch
########################################################


def _block_scaled_matmul_with_epilogue[
    c_type: DType,
    a_type: DType,
    b_type: DType,
    scales_dtype: DType,
    EpilogueFnType: ElementwiseEpilogueFn,
    ComputeFnType: ElementwiseComputeFn,
    //,
    *,
    SF_VECTOR_SIZE: Int,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, scales_dtype, scales_dtype, transpose_b
    ],
    pdl_level: PDLLevel = PDLLevel(),
    has_epilogue_fn: Bool = False,
    has_compute_fn: Bool = False,
    compute_fn_needs_vectors: Bool = False,
    compute_fn_col_period: Int = 0,
    compute_fn_col_start: Int = 0,
    groups: Int = 1,
](
    c: TileTensor[mut=True, c_type, ...],
    a: TileTensor[a_type, ...],
    b: TileTensor[b_type, ...],
    a_scales: TileTensor[scales_dtype, ...],
    b_scales: TileTensor[scales_dtype, ...],
    tensor_sf: Float32,
    epilogue_fn: EpilogueFnType,
    compute_fn: ComputeFnType,
    ctx: DeviceContext,
) raises:
    """Launch the SM100 block-scaled matmul, fusing the elementwise epilogue
    in-kernel.

    With `has_epilogue_fn` the matmul stores are redirected through
    `epilogue_fn` inside the kernel (TMEM -> registers -> closure, no scratch
    round-trip) on both the main and small-BN paths; with `has_compute_fn` the
    kernel stores `compute_fn(idx, value)`. Otherwise the kernel performs its
    default store. Callers must still allocate `c`; the default-store path
    writes it, and it defines the (row, col) coordinate space the closures
    address.
    """

    var m = Int(c.dim[0]())
    var n = Int(c.dim[1]())
    if m == 0 or n == 0:
        return

    comptime K_phys = a.static_shape[1] // groups

    comptime if groups > 1:
        comptime assert not (
            has_epilogue_fn or has_compute_fn
        ), "a block-diagonal matmul stores c directly"
        # Block g is a batch of strided views: A columns and C columns
        # [g * K, (g + 1) * K) and [g * N, (g + 1) * N), B rows
        # [g * N, (g + 1) * N), and the matching scale tiles.
        comptime N_g = c.static_shape[1] // groups
        comptime sf_k = b_scales.static_shape[1]
        comptime sf_atom = SF_ATOM_M[0] * SF_ATOM_M[1] * SF_ATOM_K
        var a_3d = a.reshape(
            TileLayout(
                Coord(Idx[groups], Int64(m), Idx[K_phys]),
                Coord(Idx[K_phys], Idx[groups * K_phys], Idx[1]),
            )
        )
        var b_3d = b.reshape(
            TileLayout(
                Coord(Idx[groups], Idx[N_g], Idx[K_phys]),
                Coord(Idx[N_g * K_phys], Idx[K_phys], Idx[1]),
            )
        )
        var c_3d = c.reshape(
            TileLayout(
                Coord(Idx[groups], Int64(m), Idx[N_g]),
                Coord(Idx[N_g], Idx[groups * N_g], Idx[1]),
            )
        )
        var a_scales_6d = a_scales.reshape(
            TileLayout(
                Coord(
                    Idx[groups],
                    Int64(a_scales.dim[0]()),
                    Idx[sf_k],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                ),
                Coord(
                    Idx[sf_k * sf_atom],
                    Idx[groups * sf_k * sf_atom],
                    Idx[sf_atom],
                    Idx[SF_ATOM_M[1] * SF_ATOM_K],
                    Idx[SF_ATOM_K],
                    Idx[1],
                ),
            )
        )
        var b_scales_6d = b_scales.reshape(
            TileLayout(
                Coord(
                    Idx[groups],
                    Idx[N_g // SF_MN_GROUP_SIZE],
                    Idx[sf_k],
                    Idx[SF_ATOM_M[0]],
                    Idx[SF_ATOM_M[1]],
                    Idx[SF_ATOM_K],
                ),
                Coord(
                    Idx[N_g // SF_MN_GROUP_SIZE * sf_k * sf_atom],
                    Idx[sf_k * sf_atom],
                    Idx[sf_atom],
                    Idx[SF_ATOM_M[1] * SF_ATOM_K],
                    Idx[SF_ATOM_K],
                    Idx[1],
                ),
            )
        )
        _blackwell_block_scaled_matmul_impl[
            transpose_b=transpose_b,
            K=K_phys,
            config=config,
            pdl_level=pdl_level,
            has_epilogue_fn=False,
            has_compute_fn=False,
        ](
            c_3d,
            a_3d,
            b_3d,
            a_scales_6d,
            b_scales_6d,
            epilogue_fn,
            compute_fn,
            ctx,
            tensor_sf,
        )
        return

    # At decode-sized M the compute closure runs inside the single-tile
    # mainloop tail, once per scalar element behind dependent global loads, and
    # costs several microseconds on top of a GEMM that is itself ~10us. Writing
    # the plain GEMM result to `c` and applying `compute_fn` from an
    # elementwise kernel hides those latencies across threads. The small-BN
    # kernel keeps the fused closure: its epilogue is cheap there, and the
    # extra launch would lose.
    #
    # A kernel that swaps A and B hands the closure one element at a time, so
    # a closure that needs consecutive columns runs from the elementwise
    # kernel at every M.
    comptime per_element = compute_fn_needs_vectors and config.AB_swapped
    comptime if has_compute_fn and (
        per_element
        or (
            not config.is_small_bn
            and config.scaling_kind == UMMAKind.KIND_MXF8F6F4
        )
    ):
        if per_element or m <= compute_fn_unfused_m_cap:
            _block_scaled_matmul_with_epilogue[
                SF_VECTOR_SIZE=SF_VECTOR_SIZE,
                transpose_b=transpose_b,
                config=config,
                pdl_level=pdl_level,
            ](
                c,
                a,
                b,
                a_scales,
                b_scales,
                tensor_sf,
                no_epilogue_fn,
                no_compute_fn,
                ctx,
            )

            comptime simd_size = simd_width_of[c_type, target=ctx.target]()
            comptime if compute_fn_col_period > 0:
                comptime period = compute_fn_col_period
                comptime start = compute_fn_col_start
                comptime assert (
                    0 <= start < period and (period - start) % simd_size == 0
                ), "the changed columns must be whole SIMD vectors"
                if n % period != 0:
                    raise Error("the output width must be whole column periods")

                # Visits only the columns `compute_fn` changes: column
                # `p * period + start + j` is `(row, p, j)`.
                def apply_compute_fn_cols[
                    width: Int, alignment: Int = 1
                ](idx: Coord) {var c, var compute_fn}:
                    var at = Coord(
                        Int64(Int(idx[0])),
                        Int64(Int(idx[1]) * period + start + Int(idx[2])),
                    )
                    c.store[alignment=alignment * size_of[c_type]()](
                        at,
                        compute_fn[c_type, width, alignment=alignment](
                            at, c.load[width=width](at)
                        ),
                    )

                elementwise[
                    simd_size,
                    target="gpu",
                    _trace_description="elementwise_block_scaled_matmul_compute",
                ](apply_compute_fn_cols, (m, n // period, period - start), ctx)
            else:

                def apply_compute_fn[
                    width: Int, alignment: Int = 1
                ](idx: Coord) {var c, var compute_fn}:
                    c.store[alignment=alignment * size_of[c_type]()](
                        idx,
                        compute_fn[c_type, width, alignment=alignment](
                            idx, c.load[width=width](idx)
                        ),
                    )

                elementwise[
                    simd_size,
                    target="gpu",
                    _trace_description="elementwise_block_scaled_matmul_compute",
                ](apply_compute_fn, (m, n), ctx)
            return

    # Store-redirect epilogues and the remaining compute closures stay fused
    # through TileWriter on both the main and small-BN paths.
    _blackwell_block_scaled_matmul_impl[
        transpose_b=transpose_b,
        K=K_phys,
        config=config,
        pdl_level=pdl_level,
        has_epilogue_fn=has_epilogue_fn,
        has_compute_fn=has_compute_fn,
    ](
        c,
        a,
        b,
        a_scales,
        b_scales,
        epilogue_fn,
        compute_fn,
        ctx,
        tensor_sf,
    )
