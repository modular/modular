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

"""CPU entry points for blockwise FP8 SM100 matmul.

Creates TMA descriptors for A, B, C and A-scales, then launches the
warp-specialized blockwise FP8 kernel with register-based accumulation.
"""

from math import align_up, ceildiv
from sys import env_get_bool, size_of

from gpu.host import DeviceContext, FuncAttribute
from gpu.host.nvidia.tma import TensorMapSwizzle
from gpu.host.info import B200
from layout import Layout as LegacyLayout, LayoutTensor
from layout._tile_tensor import TileTensor
from ..structured_kernels.tile_types import create_tma_tile, lt_to_tt

from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from ..structured_kernels.config import MatmulConfig
from .blockwise_fp8_smem import BlockwiseFP8Smem
from .blockwise_fp8_matmul_kernel import BlackwellBlockwiseFP8MatmulKernel

# Legacy kernel for fallback via -D USE_LEGACY_BLOCKWISE_FP8=True
from ...sm100.warp_specialized_blockwise_fp8 import (
    sm100_warp_specialized_blockwise_fp8,
)


# =============================================================================
# Public API: blockwise_fp8_matmul
# =============================================================================


fn blockwise_fp8_matmul[
    transpose_b: Bool,
    a_scales_type: DType,
    b_scales_type: DType,
    *,
    config: MatmulConfig[_, _, _, transpose_b],
](
    c: TileTensor,
    a: TileTensor[mut=False, ...],
    b: TileTensor[mut=False, ...],
    a_scales: TileTensor[mut=False, ...],
    b_scales: TileTensor[mut=False, ...],
    ctx: DeviceContext,
) raises:
    comptime a_type = config.a_type
    comptime b_type = config.b_type
    comptime c_type = config.c_type
    """Launch blockwise FP8 matmul kernel.

    Args:
        c: Output matrix (M x N).
        a: Input matrix A (M x K), FP8.
        b: Input matrix B (K x N or N x K if transposed), FP8.
        a_scales: Scaling factors for A (M x ceil(K/128)), FP32.
        b_scales: Scaling factors for B (ceil(N/128) x ceil(K/128)), FP32.
        ctx: Device context for kernel launch.

    Environment:
        USE_LEGACY_BLOCKWISE_FP8: If True, use legacy kernel instead of structured.
    """

    # Legacy kernel path disabled -- incompatible with TileTensor API.
    # To re-enable, update sm100_warp_specialized_blockwise_fp8 to accept TileTensor.
    comptime assert not env_get_bool[
        "USE_LEGACY_BLOCKWISE_FP8", False
    ](), "Legacy blockwise FP8 kernel not supported with TileTensor API"

    comptime assert transpose_b, "Only support transposed B"
    comptime assert (
        a_type == b_type and a_type == DType.float8_e4m3fn
    ), "Only support float8_e4m3fn"
    comptime assert (
        a_scales_type == b_scales_type
    ), "Only support float32 for scales"

    if (Int(a_scales.dim[1]()) * size_of[a_scales_type]()) % 16 != 0:
        raise Error(
            "a_scales should be a multiple of 16 bytes on the M dimension"
        )

    comptime MMA_M = config.mma_shape[0]
    comptime MMA_N = config.mma_shape[1]
    comptime MMA_K = config.mma_shape[2]

    comptime BM = MMA_M // config.cta_group
    comptime BN = MMA_N // config.cta_group
    comptime BK = config.block_tile_shape[2]

    comptime assert config.cta_group in (
        1,
        2,
    ), "Only support cta_group == 1 or 2"
    comptime assert not config.AB_swapped, "Swapped AB is not supported"

    # ==== Compute correct max_pipeline_stages for blockwise FP8 ====
    # MatmulConfig._maximize_pipeline_stages_by_default() doesn't account for
    # a_scales_smem, so we must recalculate and potentially reduce num_pipeline_stages.
    comptime b200_smem = B200.shared_memory_per_multiprocessor - 1024
    comptime MBAR_BYTES_EARLY = size_of[Int64]()
    comptime CLC_RESPONSE_BYTES_EARLY = size_of[Int128]()
    comptime TMEM_ADDR_BYTES_EARLY = size_of[Int32]()

    comptime a_smem_bytes_per_stage_early = BM * BK * size_of[a_type]()
    comptime b_smem_bytes_per_stage_early = BN * BK * size_of[b_type]()
    comptime a_scales_smem_bytes_per_stage_early = BM * size_of[a_scales_type]()
    comptime AB_smem_per_stage_early = a_smem_bytes_per_stage_early + b_smem_bytes_per_stage_early

    comptime c_smem_bytes_early = config.output_tile_shape[
        0
    ] * config.output_tile_shape[1] * config.num_output_stages * size_of[
        c_type
    ]()

    comptime accum_full_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_accum_pipeline_stages
    comptime accum_empty_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_accum_pipeline_stages
    comptime clc_response_bytes_early = CLC_RESPONSE_BYTES_EARLY * config.num_clc_pipeline_stages
    comptime clc_full_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_clc_pipeline_stages
    comptime clc_empty_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_clc_pipeline_stages
    comptime clc_throttle_full_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_clc_pipeline_stages
    comptime clc_throttle_empty_mbar_bytes_early = MBAR_BYTES_EARLY * config.num_clc_pipeline_stages
    comptime tmem_writeout_smem_early = c_smem_bytes_early + TMEM_ADDR_BYTES_EARLY + MBAR_BYTES_EARLY
    comptime accum_smem_early = accum_full_mbar_bytes_early + accum_empty_mbar_bytes_early
    comptime clc_smem_early = (
        clc_response_bytes_early
        + clc_full_mbar_bytes_early
        + clc_empty_mbar_bytes_early
        + clc_throttle_full_mbar_bytes_early
        + clc_throttle_empty_mbar_bytes_early
    )
    comptime smem_leftover_early = b200_smem - (
        clc_smem_early + accum_smem_early + tmem_writeout_smem_early
    )

    comptime tma_mbar_bytes_per_stage_early = MBAR_BYTES_EARLY
    comptime mma_mbar_bytes_per_stage_early = MBAR_BYTES_EARLY
    comptime producer_consumer_smem_per_stage_early = (
        AB_smem_per_stage_early
        + a_scales_smem_bytes_per_stage_early
        + tma_mbar_bytes_per_stage_early
        + mma_mbar_bytes_per_stage_early
    )

    comptime max_pipeline_stages_early = smem_leftover_early // producer_consumer_smem_per_stage_early

    # Use the minimum of config value and computed max to avoid SMEM overflow
    comptime corrected_pipeline_stages = min(
        config.num_pipeline_stages, max_pipeline_stages_early
    )

    # Create corrected config with proper pipeline stages for blockwise FP8
    comptime corrected_config = MatmulConfig[
        a_type, b_type, c_type, transpose_b
    ](
        cta_group=config.cta_group,
        mma_shape=config.mma_shape,
        cluster_shape=config.cluster_shape,
        AB_swapped=config.AB_swapped,
        num_pipeline_stages=corrected_pipeline_stages,
        num_accum_pipeline_stages=config.num_accum_pipeline_stages,
        num_clc_pipeline_stages=config.num_clc_pipeline_stages,
        block_swizzle_size=config.block_swizzle_size,
        raster_order=config.raster_order,
        k_group_size=config.k_group_size,
    )

    var M = Int(c.dim[0]())
    var N = Int(c.dim[1]())
    var K = Int(a.dim[1]())

    # Compute SMEM size - use corrected_config which accounts for a_scales
    comptime SmemType = BlockwiseFP8Smem[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        transpose_b,
        config=corrected_config,
    ]

    # Use the values computed earlier (with _early suffix) - they're the same
    comptime producer_consumer_smem = producer_consumer_smem_per_stage_early * corrected_pipeline_stages

    comptime smem_size = (
        clc_smem_early
        + accum_smem_early
        + producer_consumer_smem
        + tmem_writeout_smem_early
    )

    # Instantiate kernel type - use corrected_config which has proper num_pipeline_stages
    comptime Kernel = BlackwellBlockwiseFP8MatmulKernel[
        a_type,
        b_type,
        c_type,
        a_scales_type,
        b_scales_type,
        type_of(b_scales).LayoutType,
        transpose_b=transpose_b,
        config=corrected_config,
        cluster_shape = StaticTuple[Int32, 3](
            Int32(corrected_config.cluster_shape[0]),
            Int32(corrected_config.cluster_shape[1]),
            Int32(corrected_config.cluster_shape[2]),
        ),
    ]

    # Create TMA descriptors using kernel's derived legacy layouts
    a_tma_op = create_tma_tile[
        Kernel.ATmaTile.tile_layout,
        Kernel.ATmaTile.desc_layout,
        Index(BM // config.cluster_shape[1], BK),
        swizzle_mode = config.a_swizzle,
    ](ctx, a)

    b_tma_op = create_tma_tile[
        Kernel.BTmaTile.tile_layout,
        Kernel.BTmaTile.desc_layout,
        Index(
            BN // (config.cluster_shape[0] // config.cta_group), BK
        ) if transpose_b else Index(
            BK, BN // (config.cluster_shape[0] // config.cta_group)
        ),
        swizzle_mode = config.b_swizzle,
    ](ctx, b)

    a_scales_tma_op = create_tma_tile[
        Kernel.AScalesTmaTile.tile_layout,
        Kernel.AScalesTmaTile.desc_layout,
        Index(1, BM),
    ](ctx, a_scales)

    comptime c_tma_tile_shape_mma128 = Index(64, config.output_tile_shape[1])
    comptime c_tma_tile_shape = config.output_tile_shape if (
        MMA_M == 256 or config.cta_group == 1
    ) else c_tma_tile_shape_mma128

    var c_tma_op = create_tma_tile[
        Kernel.CTmaTile.tile_layout,
        Kernel.CTmaTile.desc_layout,
        c_tma_tile_shape,
        swizzle_mode = config.c_swizzle,
    ](ctx, c)

    var grid_dim = (
        align_up(ceildiv(M, BM), config.cluster_shape[0]),
        align_up(ceildiv(N, MMA_N), config.cluster_shape[1]),
        1,
    )

    var cluster_dim = StaticTuple[Int32, 3](
        Int32(ceildiv(grid_dim[0], config.cluster_shape[0])),
        Int32(ceildiv(grid_dim[1], config.cluster_shape[1])),
        1,
    )

    var problem_shape = StaticTuple[Int32, 3](Int32(M), Int32(N), Int32(K))

    ctx.enqueue_function[Kernel.run, Kernel.run, dump_asm=False](
        a_tma_op,
        b_tma_op,
        c_tma_op,
        a_scales_tma_op,
        cluster_dim,
        UInt(ceildiv(K, BK)),
        b_scales,
        problem_shape,
        grid_dim=grid_dim,
        # 1 TMA, 1 MMA, 1 Scheduler, 4 EPILOGUE warps = 7 warps
        block_dim=(32 * 7),
        shared_mem_bytes=smem_size,
        func_attribute=FuncAttribute.MAX_DYNAMIC_SHARED_SIZE_BYTES(
            UInt32(smem_size)
        ),
    )
