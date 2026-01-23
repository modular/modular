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

from .config import MatmulConfig
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
    comptime num_output_stages = Int(Self.config.num_output_stages)
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

    # ========== Tile Array Type Aliases ==========
    comptime ATileArray = SMemTileArray[
        Self.a_type,
        Self.a_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime BTileArray = SMemTileArray[
        Self.b_type,
        Self.b_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime CTileArray = SMemTileArray[
        Self.c_type,
        Self.c_smem_layout,
        Self.num_output_stages,
        alignment=128,
    ]
    comptime AScalesTileArray = SMemTileArray[
        Self.a_scales_type,
        Self.a_scales_smem_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]

    # ========== Storage Fields ==========
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var a_scales_tiles_storage: Self.AScalesTileArray.Storage

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
    fn a_scales_tiles(ref [AddressSpace.SHARED]self) -> Self.AScalesTileArray:
        return Self.AScalesTileArray(self.a_scales_tiles_storage)

    # ========== Barrier Type Aliases ==========
    comptime InputBarriers = SMemArray[
        SharedMemBarrier, Self.num_group_pipeline_stages * 2
    ]
    comptime AccumBarriers = SMemArray[
        SharedMemBarrier, Self.num_accum_pipeline_stages * 2
    ]
    comptime ClcBarriers = SMemArray[
        SharedMemBarrier, Self.num_clc_pipeline_stages
    ]
    comptime ClcThrottleBarriers = SMemArray[
        SharedMemBarrier, Self.num_clc_pipeline_stages * 2
    ]
    comptime ClcResponse = SMemArray[UInt128, Self.num_clc_pipeline_stages]
    comptime TmemDealloc = SMemArray[SharedMemBarrier, 1]
    comptime TmemAddr = SMemArray[UInt32, 1]

    # ========== Barrier Storage Fields ==========
    var tma_mma_mbars_storage: Self.InputBarriers.Storage
    var accum_mbars_storage: Self.AccumBarriers.Storage
    var clc_mbars_full_storage: Self.ClcBarriers.Storage
    var clc_mbars_empty_storage: Self.ClcBarriers.Storage
    var clc_throttle_mbars_storage: Self.ClcThrottleBarriers.Storage
    var clc_response_storage: Self.ClcResponse.Storage
    var tmem_dealloc_mbar_storage: Self.TmemDealloc.Storage
    var tmem_addr_storage: Self.TmemAddr.Storage

    # ========== Barrier Accessors ==========
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

    # ========== Standard API Aliases ==========

    @always_inline
    fn input_barriers(ref [AddressSpace.SHARED]self) -> Self.InputBarriers:
        return self.tma_mma_mbars()

    @always_inline
    fn accum_barriers(ref [AddressSpace.SHARED]self) -> Self.AccumBarriers:
        return self.accum_mbars()

    @always_inline
    fn tmem_dealloc(ref [AddressSpace.SHARED]self) -> Self.TmemDealloc:
        return self.tmem_dealloc_mbar()

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
