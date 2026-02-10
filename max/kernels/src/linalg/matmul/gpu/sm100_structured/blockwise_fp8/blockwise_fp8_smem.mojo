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

"""Shared memory layout for blockwise FP8 SM100 matmul.

This module provides the SMEM struct for blockwise FP8 matmul kernels where:
- A-scales are loaded via TMA and stored in SMEM (1D: 1 x BM per stage)
- B-scales are read directly from global memory (not stored in SMEM)
- Scaling is applied post-MMA in CUDA cores, not within the MMA unit

Unlike block-scaled matmul, blockwise FP8 uses register-based accumulation
across K iterations, with scales applied per-iteration.
"""

from gpu.memory import AddressSpace
from layout import Layout

from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.pipeline_storage import (
    BlockwiseFP8TileStorage,
    SmemPipelineBundle,
    SmemLayouts,
)
from ..structured_kernels.tile_pipeline import BlockwiseFP8TilePayload


struct BlockwiseFP8Smem[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    transpose_b: Bool,
    *,
    config: MatmulConfig[a_type, b_type, c_type, transpose_b],
]:
    """SMEM struct for blockwise FP8 matmul: A/B tiles, A-scales, C output, barriers.

    Key differences from BlockScaledSmem:
    - A-scales stored in SMEM (1D: 1 x BM per pipeline stage)
    - No B-scales in SMEM (read from global memory during epilogue)
    - Used with register-based accumulation pattern
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
    comptime num_output_stages = Self.config.num_output_stages
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

    # A-scales layout: 1D row vector with BM elements (one scale per row)
    comptime a_scales_smem_layout = Layout.row_major(1, Self.BM)

    # ========== Tile Storage (Single Source of Truth) ==========
    # Combined storage preserves SMEM layout: a, b, c, a_scales
    # Layouts are used by tile storage types for allocation and sizing
    comptime Tiles = BlockwiseFP8TileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        # A tile dimensions (BM x BK)
        Self.BM,
        Self.BK,
        # B tile dimensions (BN x BK)
        Self.BN,
        Self.BK,
        # C tile dimensions (OutputM x OutputN)
        Self.OutputM,
        Self.OutputN,
        # A-scales dimensions (1 x BM)
        1,
        Self.BM,
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Re-export tile array types
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime AScalesTileArray = Self.Tiles.AScalesTileArray

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
    fn a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        """Get A-scales tile array accessor."""
        return self.tiles.a_scales_tiles()

    # ========== Pipeline Storage (Composed Bundle) ==========
    comptime Pipelines = SmemPipelineBundle[
        Self.num_group_pipeline_stages,
        Self.num_accum_pipeline_stages,
        Self.num_clc_pipeline_stages,
        BlockwiseFP8TilePayload[
            Self.a_type,
            Self.b_type,
            Self.a_scales_type,
            # A tile dimensions (BM x BK)
            Self.BM,
            Self.BK,
            # B tile dimensions (BN x BK)
            Self.BN,
            Self.BK,
            # A-scales dimensions (1 x BM)
            1,
            Self.BM,
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
    fn a_scales_pipeline_size() -> Int:
        """Total size of A-scales tiles for all pipeline stages (in elements).
        """
        return Self.AScalesTileArray.num_elements

    @staticmethod
    @always_inline
    fn c_output_size() -> Int:
        """Size of C tiles for all output stages (in elements)."""
        return Self.CTileArray.num_elements

    @staticmethod
    @always_inline
    fn total_tile_size() -> Int:
        """Total tile storage size (A+B+A-scales+C) in elements."""
        return (
            Self.ab_pipeline_size()
            + Self.a_scales_pipeline_size()
            + Self.c_output_size()
        )
