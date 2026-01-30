# ===----------------------------------------------------------------------=== #
# Copyright (c) 2025, Modular Inc. All rights reserved.
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
from layout.tma_async import SharedMemBarrier
from layout.tensor_core_async import (
    tile_layout_k_major,
    tile_layout_mn_major,
)

from ..structured_kernels.config import MatmulConfig
from ..structured_kernels.pipeline_storage import (
    InputPipelineStorage,
    OutputPipelineStorage,
    ClcPipelineStorage,
    TmemDeallocStorage,
    BlockwiseFP8TileStorage,
)
from ..structured_kernels.tile_pipeline import BlockwiseFP8TilePayload
from linalg.structuring import SMemTileArray, SMemArray


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
    comptime a_smem_layout = tile_layout_k_major[
        Self.a_type, Self.BM, Self.BK, swizzle_mode = Self.config.a_swizzle
    ]()

    comptime b_smem_layout = tile_layout_k_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]() if Self.transpose_b else tile_layout_mn_major[
        Self.b_type, Self.BN, Self.BK, swizzle_mode = Self.config.b_swizzle
    ]()

    comptime c_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # A-scales layout: 1D row vector with BM elements (one scale per row)
    comptime a_scales_smem_layout = Layout.row_major(1, Self.BM)

    # ========== Tile Storage (Single Source of Truth) ==========
    # Combined storage preserves SMEM layout: a, b, c, a_scales
    comptime Tiles = BlockwiseFP8TileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.a_scales_type,
        Self.a_smem_layout,
        Self.b_smem_layout,
        Self.c_smem_layout,
        Self.a_scales_smem_layout,
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Re-export tile array types for external use
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime AScalesTileArray = Self.Tiles.AScalesTileArray

    # ========== Tile Storage Field ==========
    var tiles: Self.Tiles

    # ========== Tile Accessors (Delegated) ==========
    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        return self.tiles.a_tiles()

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        return self.tiles.b_tiles()

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return self.tiles.c_tiles()

    @always_inline
    fn a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        return self.tiles.a_scales_tiles()

    # ========== Pipeline Storage (Embedded) ==========
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages,
        BlockwiseFP8TilePayload[
            Self.a_type,
            Self.b_type,
            Self.a_scales_type,
            Self.a_smem_layout,
            Self.b_smem_layout,
            Self.a_scales_smem_layout,
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
