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
"""Unified Pipeline Storage Framework for SM100 Structured Kernels.

This module provides a single-source-of-truth framework for pipeline storage,
where stage count determines barrier count, and tile storage type determines
the SMEM layout for input tiles.

Design Principles
-----------------
1. **Single Source of Truth**: Stage count parameterizes barrier count
2. **Single Source of Truth**: Tile storage types define array types once
3. **Composable**: SMEM structs compose storage objects
4. **Extensible**: Easy to add new storage types
5. **Escape Hatch**: Raw storage access when framework doesn't fit

Architecture
------------
```
┌─────────────────────────────────────────────────────────────────────┐
│  Tile Storage (defines tile arrays and storage)                     │
│                                                                     │
│  StandardTileStorage[a_type, b_type, layouts, stages]              │
│      ├── ATileArray = SMemTileArray[...]                           │
│      ├── BTileArray = SMemTileArray[...]                           │
│      ├── var a_tiles_storage                                        │
│      ├── var b_tiles_storage                                        │
│      └── fn a_tiles(), b_tiles()                                    │
│                                                                     │
│  BlockScaledTileStorage[..., sfa_type, sfb_type, ...]              │
│  BlockwiseFP8TileStorage[..., a_scales_type, ...]                  │
│  OutputTileStorage[c_type, c_layout, num_stages]                   │
├─────────────────────────────────────────────────────────────────────┤
│  Pipeline Storage (defines barriers)                                │
│                                                                     │
│  InputPipelineStorage[num_stages, Payload]                         │
│      └── var barriers: BarrierPair[num_stages]                     │
│                                                                     │
│  OutputPipelineStorage[num_stages]                                 │
│  ClcPipelineStorage[num_stages]                                    │
│  TmemDeallocStorage                                                │
├─────────────────────────────────────────────────────────────────────┤
│  SMEM composes both:                                                │
│                                                                     │
│  struct MySmem:                                                     │
│      var tiles: StandardTileStorage[...]      # Tile storage       │
│      var output_tiles: OutputTileStorage[...] # Output tiles       │
│      var input_pipeline: InputPipelineStorage[...]  # Barriers     │
│      var output_pipeline: OutputPipelineStorage[...]                │
│      var clc_pipeline: ClcPipelineStorage[...]                     │
└─────────────────────────────────────────────────────────────────────┘
```

Example Usage
-------------
```
struct MyKernelSmem[config: MyConfig]:
    # Tile storage (single source of truth for tile types)
    comptime Tiles = StandardTileStorage[
        config.a_type, config.b_type,
        config.a_layout, config.b_layout,
        config.num_pipeline_stages,
    ]
    var tiles: Self.Tiles

    # Output tile storage (separate stage count)
    comptime OutputTiles = OutputTileStorage[
        config.c_type, config.c_layout, config.num_output_stages
    ]
    var output_tiles: Self.OutputTiles

    # Pipeline storage (barriers)
    var input_pipeline: InputPipelineStorage[...]
    var output_pipeline: OutputPipelineStorage[...]

    # Accessors delegate to composed storage
    fn a_tiles(ref[SHARED] self) -> Self.Tiles.ATileArray:
        return self.tiles.a_tiles()

    fn c_tiles(ref[SHARED] self) -> Self.OutputTiles.CTileArray:
        return self.output_tiles.c_tiles()
```

Extensibility
-------------
To add a new tile storage type:
1. Create a new struct with comptime type aliases and storage fields
2. Add accessors that construct tile arrays from storage
3. Use in SMEM via composition

Escape Hatch
------------
When the framework doesn't fit:
1. Use raw SMemArray for custom tile layouts
2. Use RawBarrierStorage for non-standard barrier patterns
3. Add custom storage fields to SMEM struct
"""

from gpu.memory import AddressSpace
from layout import Layout
from layout.tma_async import SharedMemBarrier

from linalg.structuring import SMemArray, SMemTileArray, SMemPtr

comptime MbarPtr = SMemPtr[SharedMemBarrier]

from .pipeline import ProducerConsumerPipeline
from .tile_pipeline import TilePayload


# =============================================================================
# Tile Storage - Single source of truth for input tile layouts
# =============================================================================


struct StandardTileStorage[
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
]:
    """Storage for standard matmul tiles (A and B).

    This is the single source of truth for tile array types and storage.
    SMEM structs embed this rather than defining tile arrays separately.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_tile_layout: SMEM layout for A tiles.
        b_tile_layout: SMEM layout for B tiles.
        num_pipeline_stages: Number of pipeline stages (determines array depth).
    """

    comptime ATileArray = SMemTileArray[
        Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime BTileArray = SMemTileArray[
        Self.b_type, Self.b_tile_layout, Self.num_pipeline_stages, alignment=128
    ]

    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage

    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        """Get A tile array accessor."""
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        """Get B tile array accessor."""
        return Self.BTileArray(self.b_tiles_storage)


struct BlockScaledTileStorage[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_output_stages: Int,
]:
    """Storage for block-scaled matmul tiles (A, B, C, SFA, SFB).

    Single source of truth for block-scaled tile arrays and storage.

    IMPORTANT: Field order preserves SMEM layout compatibility: a, b, c, sfa, sfb.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        c_type: Data type for C matrix tiles.
        sfa_type: Data type for A scale factor tiles.
        sfb_type: Data type for B scale factor tiles.
        a_tile_layout: SMEM layout for A tiles.
        b_tile_layout: SMEM layout for B tiles.
        c_tile_layout: SMEM layout for C tiles.
        sfa_tile_layout: SMEM layout for A scale tiles.
        sfb_tile_layout: SMEM layout for B scale tiles.
        num_pipeline_stages: Number of input pipeline stages.
        num_output_stages: Number of output pipeline stages.
    """

    comptime ATileArray = SMemTileArray[
        Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime BTileArray = SMemTileArray[
        Self.b_type, Self.b_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_tile_layout, Self.num_output_stages, alignment=128
    ]
    comptime SFATileArray = SMemTileArray[
        Self.sfa_type,
        Self.sfa_tile_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime SFBTileArray = SMemTileArray[
        Self.sfb_type,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]

    # Field order preserves SMEM layout: a, b, c, sfa, sfb
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var sfa_tiles_storage: Self.SFATileArray.Storage
    var sfb_tiles_storage: Self.SFBTileArray.Storage

    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        return Self.BTileArray(self.b_tiles_storage)

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return Self.CTileArray(self.c_tiles_storage)

    @always_inline
    fn sfa_tiles(ref[AddressSpace.SHARED] self) -> Self.SFATileArray:
        return Self.SFATileArray(self.sfa_tiles_storage)

    @always_inline
    fn sfb_tiles(ref[AddressSpace.SHARED] self) -> Self.SFBTileArray:
        return Self.SFBTileArray(self.sfb_tiles_storage)


struct BlockwiseFP8TileStorage[
    a_type: DType,
    b_type: DType,
    c_type: DType,
    a_scales_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    c_tile_layout: Layout,
    a_scales_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_output_stages: Int,
]:
    """Storage for blockwise FP8 matmul tiles (A, B, C, A-scales).

    Single source of truth for blockwise FP8 tile arrays and storage.
    B-scales are read directly from global memory during epilogue.

    IMPORTANT: Field order preserves SMEM layout compatibility: a, b, c, a_scales.

    Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        c_type: Data type for C matrix tiles.
        a_scales_type: Data type for A scale tiles.
        a_tile_layout: SMEM layout for A tiles.
        b_tile_layout: SMEM layout for B tiles.
        c_tile_layout: SMEM layout for C tiles.
        a_scales_tile_layout: SMEM layout for A scale tiles.
        num_pipeline_stages: Number of input pipeline stages.
        num_output_stages: Number of output pipeline stages.
    """

    comptime ATileArray = SMemTileArray[
        Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime BTileArray = SMemTileArray[
        Self.b_type, Self.b_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_tile_layout, Self.num_output_stages, alignment=128
    ]
    comptime AScalesTileArray = SMemTileArray[
        Self.a_scales_type,
        Self.a_scales_tile_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]

    # Field order preserves SMEM layout: a, b, c, a_scales
    var a_tiles_storage: Self.ATileArray.Storage
    var b_tiles_storage: Self.BTileArray.Storage
    var c_tiles_storage: Self.CTileArray.Storage
    var a_scales_tiles_storage: Self.AScalesTileArray.Storage

    @always_inline
    fn a_tiles(ref[AddressSpace.SHARED] self) -> Self.ATileArray:
        return Self.ATileArray(self.a_tiles_storage)

    @always_inline
    fn b_tiles(ref[AddressSpace.SHARED] self) -> Self.BTileArray:
        return Self.BTileArray(self.b_tiles_storage)

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return Self.CTileArray(self.c_tiles_storage)

    @always_inline
    fn a_scales_tiles(ref[AddressSpace.SHARED] self) -> Self.AScalesTileArray:
        return Self.AScalesTileArray(self.a_scales_tiles_storage)


struct OutputTileStorage[
    c_type: DType,
    c_tile_layout: Layout,
    num_output_stages: Int,
]:
    """Storage for output tiles (C matrix).

    Single source of truth for output tile array and storage.
    Separate from input tiles since output has different stage count.

    Parameters:
        c_type: Data type for C matrix tiles.
        c_tile_layout: SMEM layout for C tiles.
        num_output_stages: Number of output pipeline stages.
    """

    comptime CTileArray = SMemTileArray[
        Self.c_type, Self.c_tile_layout, Self.num_output_stages, alignment=128
    ]

    var c_tiles_storage: Self.CTileArray.Storage

    @always_inline
    fn c_tiles(ref[AddressSpace.SHARED] self) -> Self.CTileArray:
        return Self.CTileArray(self.c_tiles_storage)


# =============================================================================
# Barrier Storage - Foundational building block
# =============================================================================


struct BarrierPair[num_stages: Int]:
    """Storage for a producer-consumer barrier pair (full + empty).

    Each stage has two barriers:
    - full[i]: Producer signals when stage i is filled
    - empty[i]: Consumer signals when stage i is consumed

    Parameters:
        num_stages: Number of pipeline stages (ring buffer depth).
    """

    comptime Array = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    var storage: Self.Array.Storage

    @always_inline
    fn barriers(ref[AddressSpace.SHARED] self) -> Self.Array:
        """Get barrier array accessor."""
        return Self.Array(self.storage)

    @always_inline
    fn ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get raw barrier pointer for initialization or custom usage."""
        return self.barriers().ptr

    @always_inline
    fn create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create a runtime pipeline from this barrier storage."""
        return ProducerConsumerPipeline[Self.num_stages](self.barriers().ptr)

    # Note: Barrier initialization is left to the kernel code as patterns vary.
    # Use ptr() to get raw access for initialization.


# =============================================================================
# Input Pipeline Storage - For TMA → MMA tile transfer
# =============================================================================


struct InputPipelineStorage[
    num_stages: Int,
    Payload: TilePayload,
]:
    """Unified storage for input tile pipeline (barriers + payload).

    Bundles barrier storage with tile payload storage, ensuring they're
    always consistent. The pipeline can only be created from matching storage.

    Parameters:
        num_stages: Number of pipeline stages.
        Payload: Tile payload type (defines what's in each stage).

    Example:
        ```
        struct MySmem[...]:
            var input: InputPipelineStorage[
                4,  # 4 stages
                StandardTilePayload[float16, float16, a_layout, b_layout],
            ]

            fn get_pipeline(ref[SHARED] self):
                return self.input.create_pipeline()
        ```
    """

    # Type alias for barrier array (exposed for type-level access)
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    # Barrier storage (derived from stage count)
    var barriers: BarrierPair[Self.num_stages]

    # Payload storage would go here, but Payload types currently
    # don't define their own storage. For now, tile storage remains
    # separate in SMEM. This is the extensibility point for future.
    #
    # TODO: When Payload types define Storage, add:
    # var payload: Payload.Storage[num_stages]

    @always_inline
    fn create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create runtime pipeline from this storage."""
        return self.barriers.create_pipeline()

    @always_inline
    fn barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Escape hatch: Get raw barrier pointer for custom initialization."""
        return self.barriers.barriers().ptr


# =============================================================================
# Output Pipeline Storage - For MMA → Epilogue TMEM transfer
# =============================================================================


struct OutputPipelineStorage[num_stages: Int]:
    """Unified storage for output/accumulator pipeline.

    For MMA → Epilogue synchronization. TMEM stages are allocated
    dynamically, so this only stores barriers.

    Parameters:
        num_stages: Number of accumulator pipeline stages.
    """

    # Type alias for barrier array (exposed for type-level access)
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]

    var barriers: BarrierPair[Self.num_stages]

    @always_inline
    fn create_pipeline(
        ref[AddressSpace.SHARED] self,
    ) -> ProducerConsumerPipeline[Self.num_stages]:
        """Create runtime pipeline from this storage."""
        return self.barriers.create_pipeline()

    @always_inline
    fn barrier_ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Escape hatch: Get raw barrier pointer."""
        return self.barriers.barriers().ptr


# =============================================================================
# CLC Pipeline Storage - For scheduler coordination
# =============================================================================


struct ClcPipelineStorage[num_stages: Int]:
    """Storage for CLC (Cluster Launch Control) scheduler pipeline.

    CLC has a different barrier pattern:
    - full/empty: Standard producer-consumer for work items
    - throttle: Rate limiting barriers (2 per stage)
    - response: CLC response storage (UInt128 per stage)

    Parameters:
        num_stages: Number of CLC pipeline stages.
    """

    # Standard full/empty barriers
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages]
    var full_storage: Self.BarrierArray.Storage
    var empty_storage: Self.BarrierArray.Storage

    # Throttle barriers (2 per stage for rate limiting)
    comptime ThrottleArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]
    var throttle_storage: Self.ThrottleArray.Storage

    # CLC response storage
    comptime ResponseArray = SMemArray[UInt128, Self.num_stages]
    var response_storage: Self.ResponseArray.Storage

    @always_inline
    fn full(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.full_storage)

    @always_inline
    fn empty(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.empty_storage)

    @always_inline
    fn throttle(ref[AddressSpace.SHARED] self) -> Self.ThrottleArray:
        return Self.ThrottleArray(self.throttle_storage)

    @always_inline
    fn response(ref[AddressSpace.SHARED] self) -> Self.ResponseArray:
        return Self.ResponseArray(self.response_storage)


# =============================================================================
# TMEM Deallocation Storage - Single barrier for TMEM lifecycle
# =============================================================================


struct TmemDeallocStorage:
    """Storage for TMEM deallocation synchronization.

    Single barrier + address storage for TMEM lifecycle management.
    """

    comptime BarrierArray = SMemArray[SharedMemBarrier, 1]
    comptime AddrArray = SMemArray[UInt32, 1]

    var barrier_storage: Self.BarrierArray.Storage
    var addr_storage: Self.AddrArray.Storage

    @always_inline
    fn barrier(ref[AddressSpace.SHARED] self) -> Self.BarrierArray:
        return Self.BarrierArray(self.barrier_storage)

    @always_inline
    fn addr(ref[AddressSpace.SHARED] self) -> Self.AddrArray:
        return Self.AddrArray(self.addr_storage)


# =============================================================================
# Escape Hatch - Raw barrier storage for custom patterns
# =============================================================================


struct RawBarrierStorage[count: Int]:
    """Escape hatch: Raw barrier storage for custom patterns.

    Use this when the standard pipeline storage doesn't fit your needs.
    You're responsible for initialization and synchronization semantics.

    Parameters:
        count: Total number of barriers to allocate.

    Example:
        ```
        # Custom barrier layout for specialized synchronization
        struct MyCustomSmem:
            var custom_barriers: RawBarrierStorage[8]

            fn init_custom(ref[SHARED] self):
                ptr = self.custom_barriers.ptr()
                # Custom initialization...
        ```
    """

    comptime Array = SMemArray[SharedMemBarrier, Self.count]

    var storage: Self.Array.Storage

    @always_inline
    fn barriers(ref[AddressSpace.SHARED] self) -> Self.Array:
        return Self.Array(self.storage)

    @always_inline
    fn ptr(ref[AddressSpace.SHARED] self) -> MbarPtr:
        """Get raw pointer for custom usage."""
        return self.barriers().ptr
