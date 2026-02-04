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

"""Shared memory layout for block-scaled SM100 matmul.

Extends standard SMEM with scaling factor tile storage (SFA, SFB) following
MXFP8 layout conventions. Also includes all pipeline barriers and TMEM state.
"""

from gpu.memory import AddressSpace
from layout import Layout
from layout.tma_async import SharedMemBarrier
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
    tile_sf_layout_k_major,
)

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from ..structured_kernels.pipeline_storage import (
    InputPipelineStorage,
    OutputPipelineStorage,
    ClcPipelineStorage,
    TmemDeallocStorage,
    BlockScaledTileStorage,
)
from ..structured_kernels.tile_pipeline import BlockScaledTilePayload
from ..structured_kernels.tile_types import (
    SMemTile,
    internal_k_major_128B,
)
from linalg.structuring import SMemArray


struct BlockScaledSmem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    *,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]:
    """SMEM struct containing A/B tiles, scaling factors, C output, and barriers.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]
    comptime MMA_M = Self.config.mma_shape[0]
    comptime MMA_N = Self.config.mma_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages = Self.config.num_accum_pipeline_stages
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages

    # ========== Layout Definitions ==========
    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.config.a_swizzle
    ]()

    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]()

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size
    # This determines how many K elements each scaling factor covers
    comptime SF_K_GROUP_SIZE = SF_ATOM_K * Self.config.vec_sf_size

    # SF layouts use config.vec_sf_size (MXFP8=32, NVFP4=16) and num_sf_k_tiles
    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        Self.MMA_N,
        Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles,
        Self.config.vec_sf_size,
    ]()

    # SF tile dimensions (computed from tile_sf_layout_k_major formula)
    # The layout uses a nested structure, so we compute dimensions directly:
    # Row: (BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0] = (BM // 128) * 32
    # Col: (BK // (SF_ATOM_K * vec_sf_size)) * (SF_ATOM_M[1] * SF_ATOM_K)
    #    = (BK // (4 * vec_sf_size)) * 16
    comptime SF_BK = Self.SF_K_GROUP_SIZE * Self.config.num_sf_k_tiles
    comptime SFA_DIM0 = (Self.BM // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
    comptime SFA_DIM1 = (
        Self.SF_BK // (SF_ATOM_K * Self.config.vec_sf_size)
    ) * (SF_ATOM_M[1] * SF_ATOM_K)
    comptime SFB_DIM0 = (Self.MMA_N // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]
    comptime SFB_DIM1 = (
        Self.SF_BK // (SF_ATOM_K * Self.config.vec_sf_size)
    ) * (SF_ATOM_M[1] * SF_ATOM_K)

    # ========== Tile Storage (Single Source of Truth) ==========
    # Combined storage preserves SMEM layout: a, b, c, sfa, sfb
    # Note: Layouts are still defined above for LayoutTensor boundary conversion
    comptime Tiles = BlockScaledTileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        # A tile dimensions (BM x BK)
        Self.BM,
        Self.BK,
        # B tile dimensions (BN x BK)
        Self.BN,
        Self.BK,
        # C tile dimensions (OutputM x OutputN)
        Self.OutputM,
        Self.OutputN,
        # SFA tile dimensions
        Self.SFA_DIM0,
        Self.SFA_DIM1,
        # SFB tile dimensions
        Self.SFB_DIM0,
        Self.SFB_DIM1,
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Re-export tile array types (all TileTensor-based now)
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray  # TileTensor-based
    comptime SFATileArray = Self.Tiles.SFATileArray
    comptime SFBTileArray = Self.Tiles.SFBTileArray

    # ========== Tile Storage Field ==========
    var tiles: Self.Tiles

    # ========== Tile Accessors (TileTensor - Delegated) ==========
    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor (TileTensor-based)."""
        return self.tiles.a_tiles()

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor (TileTensor-based)."""
        return self.tiles.b_tiles()

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor (TileTensor-based)."""
        return self.tiles.c_tiles_tt()

    @always_inline
    fn sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        """Get SFA tile array accessor (TileTensor-based)."""
        return self.tiles.sfa_tiles()

    @always_inline
    fn sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        """Get SFB tile array accessor (TileTensor-based)."""
        return self.tiles.sfb_tiles()

    # ========== Pipeline Storage (Embedded) ==========
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages,
        BlockScaledTilePayload[
            Self.a_type,
            Self.b_type,
            Self.sfa_dtype,
            Self.sfb_dtype,
            # A tile dimensions (BM x BK)
            Self.BM,
            Self.BK,
            # B tile dimensions (BN x BK)
            Self.BN,
            Self.BK,
            # SFA tile dimensions
            Self.SFA_DIM0,
            Self.SFA_DIM1,
            # SFB tile dimensions
            Self.SFB_DIM0,
            Self.SFB_DIM1,
            Self.num_pipeline_stages,
        ],
    ]
    comptime OutputPipeline = OutputPipelineStorage[
        Self.num_accum_pipeline_stages
    ]
    comptime ClcPipeline = ClcPipelineStorage[Self.num_clc_pipeline_stages]
    comptime TmemDeallocPipeline = TmemDeallocStorage

    # Storage fields - embedded in SMEM
    var input_pipeline: Self.InputPipeline
    var output_pipeline: Self.OutputPipeline
    var clc_pipeline: Self.ClcPipeline
    var tmem_dealloc_pipeline: Self.TmemDeallocPipeline

    # Type aliases for accessor return types
    comptime InputBarriers = Self.InputPipeline.BarrierArray
    comptime AccumBarriers = Self.OutputPipeline.BarrierArray
    comptime ClcBarriers = Self.ClcPipeline.BarrierArray
    comptime ClcThrottleBarriers = Self.ClcPipeline.ThrottleArray
    comptime ClcResponse = Self.ClcPipeline.ResponseArray
    comptime TmemDealloc = Self.TmemDeallocPipeline.BarrierArray
    comptime TmemAddr = Self.TmemDeallocPipeline.AddrArray

    # ========== Barrier Accessors (Delegated to Pipelines) ==========
    @always_inline
    fn input_barriers(ref[AddressSpace.SHARED] self) -> Self.InputBarriers:
        """Returns input tile pipeline barriers."""
        return self.input_pipeline.barriers.barriers()

    @always_inline
    fn accum_barriers(ref[AddressSpace.SHARED] self) -> Self.AccumBarriers:
        """Returns accumulator pipeline barriers."""
        return self.output_pipeline.barriers.barriers()

    @always_inline
    fn clc_mbars_full(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        return self.clc_pipeline.full()

    @always_inline
    fn clc_mbars_empty(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        return self.clc_pipeline.empty()

    @always_inline
    fn clc_throttle_mbars(
        ref[AddressSpace.SHARED] self,
    ) -> Self.ClcThrottleBarriers:
        return self.clc_pipeline.throttle()

    @always_inline
    fn clc_response(ref[AddressSpace.SHARED] self) -> Self.ClcResponse:
        return self.clc_pipeline.response()

    @always_inline
    fn tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        """Returns TMEM deallocation barrier."""
        return self.tmem_dealloc_pipeline.barrier()

    @always_inline
    fn tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        return self.tmem_dealloc_pipeline.addr()

    # ========== Size Utilities ==========
    @staticmethod
    @always_inline
    fn ab_pipeline_size() -> Int:
        """Total size of A+B tiles for all pipeline stages (in elements)."""
        return Self.ATileArray.num_elements + Self.BTileArray.num_elements

    @staticmethod
    @always_inline
    fn sf_pipeline_size() -> Int:
        """Total size of SFA+SFB tiles for all pipeline stages (in elements)."""
        return Self.SFATileArray.num_elements + Self.SFBTileArray.num_elements

    @staticmethod
    @always_inline
    fn c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    fn total_tile_size() -> Int:
        """Total tile storage size (A+B+SFA+SFB+C) in elements."""
        return (
            Self.ab_pipeline_size()
            + Self.sf_pipeline_size()
            + Self.c_output_size()
        )


# =============================================================================
# Helper Functions for Scaling Factor SMEM Layout
# =============================================================================


@always_inline
fn get_sfa_smem_layout[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]() -> Layout:
    """Get the SMEM layout for A scaling factors."""
    comptime SF_K_GROUP_SIZE = SF_ATOM_K * config.vec_sf_size
    return tile_sf_layout_k_major[
        config.block_tile_shape[0],
        SF_K_GROUP_SIZE * config.num_sf_k_tiles,
        config.vec_sf_size,
    ]()


@always_inline
fn get_sfb_smem_layout[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_dtype: DType,
    sfb_dtype: DType,
    transpose_b: Bool,
    config: BlockScaledMatmulConfig[
        a_type, b_type, c_type, sfa_dtype, sfb_dtype, transpose_b
    ],
]() -> Layout:
    """Get the SMEM layout for B scaling factors."""
    comptime SF_K_GROUP_SIZE = SF_ATOM_K * config.vec_sf_size
    return tile_sf_layout_k_major[
        config.mma_shape[1],
        SF_K_GROUP_SIZE * config.num_sf_k_tiles,
        config.vec_sf_size,
    ]()


# =============================================================================
# TMEM Column Calculations for Scaling Factors
# =============================================================================


@always_inline
fn get_sfa_num_cols[
    config: BlockScaledMatmulConfig,
]() -> Int:
    """Get the number of TMEM columns needed for A scaling factors."""
    return config.block_tile_shape[0] // 32


@always_inline
fn get_sfb_num_cols[
    config: BlockScaledMatmulConfig,
]() -> Int:
    """Get the number of TMEM columns needed for B scaling factors."""
    return config.mma_shape[1] // 32
