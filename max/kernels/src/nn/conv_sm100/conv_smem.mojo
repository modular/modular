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

"""Shared memory layout for SM100 Conv2D kernel.

This module provides the Conv2dSmem struct which defines the shared memory
organization for the Conv2D fprop kernel. The layout is similar to
B200MatmulSmem but uses conv-specific naming.

SMEM Organization:
- Activation tiles (from im2col): Multi-stage pipelined
- Filter tiles: Multi-stage pipelined
- Output tiles: Double-buffered for epilogue
- Pipeline barriers: For producer-consumer synchronization
- CLC barriers: For work scheduling
- TMEM storage: For accumulator address sharing
"""

from sys import align_of, size_of

from gpu.memory import AddressSpace
from layout import Layout
from layout.tensor_core_async import tile_layout_k_major, tile_layout_mn_major

from linalg.structuring import SMemTileArray, SMemArray

# Import pipeline storage from matmul structured kernels
from linalg.matmul.gpu.sm100_structured.structured_kernels.pipeline_storage import (
    InputPipelineStorage,
    OutputPipelineStorage,
    ClcPipelineStorage,
    TmemDeallocStorage,
    StandardTileStorage,
    OutputTileStorage,
)
from linalg.matmul.gpu.sm100_structured.structured_kernels.tile_pipeline import (
    StandardTilePayload,
)

from .conv_config import Conv2dConfig


# =============================================================================
# Conv2dSmem - Shared memory layout for Conv2D kernel
# =============================================================================


struct Conv2dSmem[
    act_type: DType,
    filter_type: DType,
    out_type: DType,
    *,
    config: Conv2dConfig[act_type, filter_type, out_type],
]:
    """Shared memory layout for SM100 Conv2D fprop kernel.

    This struct manages shared memory allocation for:
    - Activation tiles (after im2col transformation)
    - Filter tiles
    - Output tiles for accumulation
    - Synchronization barriers

    The layout mirrors B200MatmulSmem but with conv-specific semantics:
    - A tiles = im2col'd activation (M x K where M = N*H*W, K = C*R*S)
    - B tiles = filter (transposed, K x N where K = C*R*S, N = K_out)
    - C tiles = output (M x N)

    Parameters:
        act_type: Activation data type.
        filter_type: Filter data type.
        out_type: Output data type.
        config: Kernel configuration.
    """

    # ========== Derived Constants ==========
    comptime BM = Self.config.block_tile_shape[0]
    comptime BN = Self.config.block_tile_shape[1]
    comptime BK = Self.config.block_tile_shape[2]
    comptime OutputM = Self.config.output_tile_shape[0]
    comptime OutputN = Self.config.output_tile_shape[1]

    # Pipeline stage counts
    comptime num_pipeline_stages: Int = Self.config.num_pipeline_stages
    comptime num_group_pipeline_stages: Int = (
        Self.num_pipeline_stages // Self.config.k_group_size
    )
    comptime num_output_stages: Int = Self.config.num_output_stages
    comptime num_accum_pipeline_stages: Int = Self.config.num_accum_pipeline_stages
    comptime num_clc_pipeline_stages: Int = Self.config.num_clc_pipeline_stages

    # ========== Layout Definitions ==========
    # Activation tiles use K-major layout (im2col'd)
    comptime act_smem_layout = tile_layout_k_major[
        Self.act_type,
        Self.BM,
        Self.BK,
        swizzle_mode = Self.config.a_swizzle,
    ]()

    # Filter tiles use K-major layout (transposed GEMM B)
    comptime filter_smem_layout = tile_layout_k_major[
        Self.filter_type,
        Self.BN,
        Self.BK,
        swizzle_mode = Self.config.b_swizzle,
    ]()

    # Output tiles use row-major layout
    comptime out_smem_layout = Layout.row_major(Self.OutputM, Self.OutputN)

    # ========== Tile Storage ==========
    # Reuse StandardTileStorage from matmul (same structure)
    comptime InputTiles = StandardTileStorage[
        Self.act_type,
        Self.filter_type,
        # Activation tile dimensions (BM x BK)
        Self.BM,
        Self.BK,
        # Filter tile dimensions (BN x BK)
        Self.BN,
        Self.BK,
        Self.num_pipeline_stages,
    ]

    comptime OutputTiles = OutputTileStorage[
        Self.out_type,
        Self.OutputM,
        Self.OutputN,
        Self.num_output_stages,
    ]

    # Re-export tile array types
    comptime ActTileArray = Self.InputTiles.ATileArray
    comptime FilterTileArray = Self.InputTiles.BTileArray
    comptime OutTileArray = Self.OutputTiles.CTileArray

    # ========== Storage Fields ==========
    var input_tiles: Self.InputTiles
    var output_tiles: Self.OutputTiles

    # ========== Tile Accessors ==========
    @always_inline
    fn act_tiles(ref[AddressSpace.SHARED] self) -> Self.ActTileArray:
        """Get activation tiles (im2col'd)."""
        return self.input_tiles.a_tiles()

    @always_inline
    fn filter_tiles(ref[AddressSpace.SHARED] self) -> Self.FilterTileArray:
        """Get filter tiles."""
        return self.input_tiles.b_tiles()

    @always_inline
    fn out_tiles(ref[AddressSpace.SHARED] self) -> Self.OutTileArray:
        """Get output tiles."""
        return self.output_tiles.c_tiles_tt()

    # ========== Pipeline Storage ==========
    comptime InputPipeline = InputPipelineStorage[
        Self.num_group_pipeline_stages,
        StandardTilePayload[
            Self.act_type,
            Self.filter_type,
            Self.BM,
            Self.BK,
            Self.BN,
            Self.BK,
            Self.num_pipeline_stages,
        ],
    ]
    comptime OutputPipeline = OutputPipelineStorage[
        Self.num_accum_pipeline_stages
    ]
    comptime ClcPipeline = ClcPipelineStorage[Self.num_clc_pipeline_stages]
    comptime TmemDeallocPipeline = TmemDeallocStorage

    var input_pipeline: Self.InputPipeline
    var output_pipeline: Self.OutputPipeline
    var clc_pipeline: Self.ClcPipeline
    var tmem_dealloc_pipeline: Self.TmemDeallocPipeline

    # ========== Barrier Type Aliases ==========
    comptime InputBarriers = Self.InputPipeline.BarrierArray
    comptime AccumBarriers = Self.OutputPipeline.BarrierArray
    comptime ClcBarriers = Self.ClcPipeline.BarrierArray
    comptime ClcThrottleBarriers = Self.ClcPipeline.ThrottleArray
    comptime ClcResponse = Self.ClcPipeline.ResponseArray
    comptime TmemDealloc = Self.TmemDeallocPipeline.BarrierArray
    comptime TmemAddr = Self.TmemDeallocPipeline.AddrArray

    # ========== Barrier Accessors ==========
    @always_inline
    fn input_barriers(ref[AddressSpace.SHARED] self) -> Self.InputBarriers:
        return self.input_pipeline.barriers.barriers()

    @always_inline
    fn accum_barriers(ref[AddressSpace.SHARED] self) -> Self.AccumBarriers:
        return self.output_pipeline.barriers.barriers()

    @always_inline
    fn clc_full(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        return self.clc_pipeline.full()

    @always_inline
    fn clc_empty(ref[AddressSpace.SHARED] self) -> Self.ClcBarriers:
        return self.clc_pipeline.empty()

    @always_inline
    fn clc_throttle(ref[AddressSpace.SHARED] self) -> Self.ClcThrottleBarriers:
        return self.clc_pipeline.throttle()

    @always_inline
    fn clc_response(ref[AddressSpace.SHARED] self) -> Self.ClcResponse:
        return self.clc_pipeline.response()

    @always_inline
    fn tmem_dealloc(ref[AddressSpace.SHARED] self) -> Self.TmemDealloc:
        return self.tmem_dealloc_pipeline.barrier()

    @always_inline
    fn tmem_addr(ref[AddressSpace.SHARED] self) -> Self.TmemAddr:
        return self.tmem_dealloc_pipeline.addr()
