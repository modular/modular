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
"""Shared memory layout for grouped 1D-1D block-scaled SM100 matmul.

This is a simplified SMEM structure for the 1D-1D kernel variant that uses
offset-based addressing instead of pointer-per-group. Key differences from
the standard GroupedBlockScaledSmem:

1. No tensormap descriptors - TMAs are grid-constant (not updated per-group)
2. No CLC pipeline storage - uses 3-warp specialization (no scheduler warp)
3. Simpler barrier structure optimized for the 1D-1D workload

The 1D-1D layout uses:
- A tensor: Contiguous (total_tokens, K) with a_offsets for per-group access
- B tensor: Batched (num_experts, N, K) weights
- C tensor: Contiguous (total_tokens, N) output
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
    TmemDeallocStorage,
    BlockScaledTileStorage,
)
from ..structured_kernels.tile_pipeline import BlockScaledTilePayload
from linalg.structuring import SMemTileArray, SMemArray


struct Grouped1D1DSmem[
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
    """SMEM struct for grouped 1D-1D block-scaled GEMM.

    Simplified version of GroupedBlockScaledSmem for offset-based addressing.
    Uses 3-warp specialization (Load, MMA, Epilogue) without a scheduler warp,
    so CLC pipeline storage is not needed.

    Layout in SMEM:
    1. A tiles (input pipeline stages)
    2. B tiles (input pipeline stages)
    3. C tiles (output stages)
    4. SFA tiles (scaling factors for A)
    5. SFB tiles (scaling factors for B)
    6. Input pipeline barriers
    7. Output pipeline barriers (accum barriers)
    8. TMEM deallocation state
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

    # ========== Tile Storage (Single Source of Truth) ==========
    # Combined storage preserves SMEM layout: a, b, c, sfa, sfb
    comptime Tiles = BlockScaledTileStorage[
        Self.a_type,
        Self.b_type,
        Self.c_type,
        Self.sfa_dtype,
        Self.sfb_dtype,
        Self.a_smem_layout,
        Self.b_smem_layout,
        Self.c_smem_layout,
        Self.sfa_smem_layout,
        Self.sfb_smem_layout,
        Self.num_pipeline_stages,
        Self.num_output_stages,
    ]

    # Re-export tile array types for external use
    comptime ATileArray = Self.Tiles.ATileArray
    comptime BTileArray = Self.Tiles.BTileArray
    comptime CTileArray = Self.Tiles.CTileArray
    comptime SFATileArray = Self.Tiles.SFATileArray
    comptime SFBTileArray = Self.Tiles.SFBTileArray

    # ========== Tile Storage Field ==========
    var tiles: Self.Tiles

    # ========== Pipeline Storage (Embedded) ==========
    # Note: No ClcPipeline - 1D-1D uses 3-warp specialization without scheduler
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages,
        BlockScaledTilePayload[
            Self.a_type,
            Self.b_type,
            Self.sfa_dtype,
            Self.sfb_dtype,
            Self.a_smem_layout,
            Self.b_smem_layout,
            Self.sfa_smem_layout,
            Self.sfb_smem_layout,
            Self.num_pipeline_stages,
        ],
    ]
    comptime OutputPipeline = OutputPipelineStorage[
        Self.num_accum_pipeline_stages
    ]
    comptime TmemDeallocPipeline = TmemDeallocStorage

    # Storage fields - embedded in SMEM
    var input_pipeline: Self.InputPipeline
    var output_pipeline: Self.OutputPipeline
    var tmem_dealloc_pipeline: Self.TmemDeallocPipeline

    # Type aliases for accessor return types
    comptime InputBarriers = Self.InputPipeline.BarrierArray
    comptime AccumBarriers = Self.OutputPipeline.BarrierArray
    comptime TmemDealloc = Self.TmemDeallocPipeline.BarrierArray
    comptime TmemAddr = Self.TmemDeallocPipeline.AddrArray

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
    fn sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        return self.tiles.sfa_tiles()

    @always_inline
    fn sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        return self.tiles.sfb_tiles()

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
    fn tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        """Returns TMEM deallocation barrier."""
        return self.tmem_dealloc_pipeline.barrier()

    @always_inline
    fn tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        """Returns TMEM address storage for deallocation."""
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
