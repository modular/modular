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
    MXFP8_SF_VECTOR_SIZE,
)
from linalg.matmul.gpu.sm100.config import BlockScaledMatmulConfig
from linalg.structuring import SMemTileArrayType, SMemArrayType


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
    comptime num_pipeline_stages = Int(Self.config.num_pipeline_stages)
    comptime num_group_pipeline_stages = (
        Self.num_pipeline_stages // Int(Self.config.k_group_size)
    )
    comptime num_output_stages = Int(Self.config.num_output_stages)
    comptime num_accum_pipeline_stages = Int(
        Self.config.num_accum_pipeline_stages
    )
    comptime num_clc_pipeline_stages = Int(Self.config.num_clc_pipeline_stages)

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

    comptime sfa_smem_layout = tile_sf_layout_k_major[
        Self.BM, Self.BK, MXFP8_SF_VECTOR_SIZE
    ]()

    comptime sfb_smem_layout = tile_sf_layout_k_major[
        Self.MMA_N, Self.BK, MXFP8_SF_VECTOR_SIZE
    ]()

    # ========== Tile Array Type Aliases ==========
    comptime ATileArray = SMemTileArrayType[
        Self.a_type,
        Self.a_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime BTileArray = SMemTileArrayType[
        Self.b_type,
        Self.b_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime CTileArray = SMemTileArrayType[
        Self.c_type,
        Self.c_smem_layout,
        Self.num_output_stages,
        alignment=128,
    ]
    comptime SFATileArray = SMemTileArrayType[
        Self.sfa_dtype,
        Self.sfa_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime SFBTileArray = SMemTileArrayType[
        Self.sfb_dtype,
        Self.sfb_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]

    # ========== Storage Fields ==========
    var a_tiles_storage: Self.ATileArray.StorageType
    var b_tiles_storage: Self.BTileArray.StorageType
    var c_tiles_storage: Self.CTileArray.StorageType
    var sfa_tiles_storage: Self.SFATileArray.StorageType
    var sfb_tiles_storage: Self.SFBTileArray.StorageType

    @always_inline
    fn a_tiles(ref [AddressSpace.SHARED]self) -> Self.ATileArray:
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref [AddressSpace.SHARED]self) -> Self.BTileArray:
        return Self.BTileArray(self.b_tiles_storage)

    @always_inline
    fn c_tiles(ref [AddressSpace.SHARED]self) -> Self.CTileArray:
        return Self.CTileArray(self.c_tiles_storage)

    @always_inline
    fn sfa_tiles(ref [AddressSpace.SHARED]self) -> Self.SFATileArray:
        return Self.SFATileArray(self.sfa_tiles_storage)

    @always_inline
    fn sfb_tiles(ref [AddressSpace.SHARED]self) -> Self.SFBTileArray:
        return Self.SFBTileArray(self.sfb_tiles_storage)

    # ========== Barrier Type Aliases ==========
    comptime InputBarriers = SMemArrayType[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]
    comptime AccumBarriers = SMemArrayType[
        SharedMemBarrier, Self.num_accum_pipeline_stages * 2
    ]
    comptime ClcBarriers = SMemArrayType[
        SharedMemBarrier, Self.num_clc_pipeline_stages
    ]
    comptime ClcThrottleBarriers = SMemArrayType[
        SharedMemBarrier, Self.num_clc_pipeline_stages * 2
    ]
    comptime ClcResponse = SMemArrayType[UInt128, Self.num_clc_pipeline_stages]
    comptime TmemDealloc = SMemArrayType[SharedMemBarrier, 1]
    comptime TmemAddr = SMemArrayType[UInt32, 1]

    # ========== Barrier Storage Fields ==========
    var tma_mma_mbars_storage: Self.InputBarriers.StorageType
    var accum_mbars_storage: Self.AccumBarriers.StorageType
    var clc_mbars_full_storage: Self.ClcBarriers.StorageType
    var clc_mbars_empty_storage: Self.ClcBarriers.StorageType
    var clc_throttle_mbars_storage: Self.ClcThrottleBarriers.StorageType
    var clc_response_storage: Self.ClcResponse.StorageType
    var tmem_dealloc_mbar_storage: Self.TmemDealloc.StorageType
    var tmem_addr_storage: Self.TmemAddr.StorageType

    @always_inline
    fn tma_mma_mbars(ref [AddressSpace.SHARED]self) -> Self.InputBarriers:
        return Self.InputBarriers(self.tma_mma_mbars_storage)

    @always_inline
    fn accum_mbars(ref [AddressSpace.SHARED]self) -> Self.AccumBarriers:
        return Self.AccumBarriers(self.accum_mbars_storage)

    @always_inline
    fn clc_mbars_full(ref [AddressSpace.SHARED]self) -> Self.ClcBarriers:
        return Self.ClcBarriers(self.clc_mbars_full_storage)

    @always_inline
    fn clc_mbars_empty(ref [AddressSpace.SHARED]self) -> Self.ClcBarriers:
        return Self.ClcBarriers(self.clc_mbars_empty_storage)

    @always_inline
    fn clc_throttle_mbars(
        ref [AddressSpace.SHARED]self,
    ) -> Self.ClcThrottleBarriers:
        return Self.ClcThrottleBarriers(self.clc_throttle_mbars_storage)

    @always_inline
    fn clc_response(ref [AddressSpace.SHARED]self) -> Self.ClcResponse:
        return Self.ClcResponse(self.clc_response_storage)

    @always_inline
    fn tmem_dealloc_mbar(ref [AddressSpace.SHARED]self) -> Self.TmemDealloc:
        return Self.TmemDealloc(self.tmem_dealloc_mbar_storage)

    @always_inline
    fn tmem_addr(ref [AddressSpace.SHARED]self) -> Self.TmemAddr:
        return Self.TmemAddr(self.tmem_addr_storage)

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
    return tile_sf_layout_k_major[
        config.block_tile_shape[0],
        config.block_tile_shape[2],
        MXFP8_SF_VECTOR_SIZE,
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
    return tile_sf_layout_k_major[
        config.mma_shape[1],
        config.block_tile_shape[2],
        MXFP8_SF_VECTOR_SIZE,
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
