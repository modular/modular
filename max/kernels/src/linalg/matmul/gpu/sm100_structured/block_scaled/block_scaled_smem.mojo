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
from layout.tensor_core_async import tile_sf_layout_k_major

from linalg.fp4_utils import (
    SF_MN_GROUP_SIZE,
    SF_ATOM_M,
    SF_ATOM_K,
)
from ..structured_kernels.config import BlockScaledMatmulConfig
from ..structured_kernels.pipeline_storage import (
    BlockScaledTileStorage,
    SmemPipelineBundle,
    SmemLayouts,
)
from ..structured_kernels.tile_pipeline import BlockScaledTilePayload


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
    comptime Layouts = SmemLayouts[
        Self.a_type,
        Self.b_type,
        Self.BM,
        Self.BN,
        Self.BK,
        Self.OutputM,
        Self.OutputN,
        Self.config.a_swizzle,
        Self.config.b_swizzle,
        Self.transpose_b,
    ]
    comptime a_smem_layout = Self.Layouts.a_smem_layout
    comptime b_smem_layout = Self.Layouts.b_smem_layout
    comptime c_smem_layout = Self.Layouts.c_smem_layout

    # SF_K_GROUP_SIZE = SF_ATOM_K * vec_sf_size
    # This determines how many K elements each scaling factor covers
    comptime SF_K_GROUP_SIZE = sf_k_group_size[Self.config]()

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

    # SF tile dimensions (computed via shared helper functions)
    comptime SF_BK = sf_bk[Self.config]()
    comptime SFA_DIM0 = sfa_dim0[Self.config]()
    comptime SFA_DIM1 = sfa_dim1[Self.config]()
    comptime SFB_DIM0 = sfb_dim0[Self.config]()
    comptime SFB_DIM1 = sfb_dim1[Self.config]()

    # ========== Tile Storage (Single Source of Truth) ==========
    # Combined storage preserves SMEM layout: a, b, c, sfa, sfb
    # Layouts are used by tile storage types for allocation and sizing
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

    # Re-export tile array types
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime SFATileArray = Self.Tiles.SFATileArray
    comptime SFBTileArray = Self.Tiles.SFBTileArray

    # ========== Tile Storage Field ==========
    var tiles: Self.Tiles

    # ========== Tile Accessors (TileTensor - Delegated) ==========
    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return self.tiles.a_tiles()

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return self.tiles.b_tiles()

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        """Get C tile array accessor."""
        return self.tiles.c_tiles()

    @always_inline
    fn sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        """Get SFA tile array accessor."""
        return self.tiles.sfa_tiles()

    @always_inline
    fn sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        """Get SFB tile array accessor."""
        return self.tiles.sfb_tiles()

    # ========== Pipeline Storage (Composed Bundle) ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.num_group_pipeline_stages,
        Self.num_accum_pipeline_stages,
        Self.num_clc_pipeline_stages,
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
    var pipelines: Self.Pipelines

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
# Scaling Factor Dimension Helpers
# =============================================================================


@always_inline
fn sf_k_group_size[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SF_K_GROUP_SIZE from config."""
    return SF_ATOM_K * config.vec_sf_size


@always_inline
fn sf_bk[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SF_BK from config."""
    return sf_k_group_size[config]() * config.num_sf_k_tiles


@always_inline
fn sfa_dim0[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFA first dimension from config."""
    return (config.block_tile_shape[0] // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]


@always_inline
fn sfa_dim1[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFA second dimension from config."""
    return (sf_bk[config]() // (SF_ATOM_K * config.vec_sf_size)) * (
        SF_ATOM_M[1] * SF_ATOM_K
    )


@always_inline
fn sfb_dim0[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFB first dimension from config."""
    return (config.mma_shape[1] // SF_MN_GROUP_SIZE) * SF_ATOM_M[0]


@always_inline
fn sfb_dim1[config: BlockScaledMatmulConfig]() -> Int:
    """Compute SFB second dimension from config."""
    return (sf_bk[config]() // (SF_ATOM_K * config.vec_sf_size)) * (
        SF_ATOM_M[1] * SF_ATOM_K
    )
