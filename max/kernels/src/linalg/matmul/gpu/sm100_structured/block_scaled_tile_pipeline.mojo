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

"""Block-scaled tile pipeline for SM100 matmul.

Extends TilePipeline to manage 4 tile arrays (A, B, SFA, SFB) with
synchronized producer-consumer access for TMA loading and MMA consumption.
"""

from layout.tma_async import SharedMemBarrier
from .pipeline import ProducerConsumerPipeline
from .tmem import TmemAllocation, TmemStage
from linalg.structuring import SMemPtr, SMemTileArrayType, SMemArrayType


comptime MbarPtr = SMemPtr[SharedMemBarrier]


# =============================================================================
# BlockScaledTilePipeline - Extended pipeline with scaling factors
# =============================================================================


@register_passable("trivial")
struct BlockScaledTilePipeline[
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Staged tile storage for A, B, SFA, SFB with producer-consumer sync."""

    comptime Pipeline = ProducerConsumerPipeline[Self.num_group_stages]

    comptime ATileArray = SMemTileArrayType[
        Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime BTileArray = SMemTileArrayType[
        Self.b_type, Self.b_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime SFATileArray = SMemTileArrayType[
        Self.sfa_type,
        Self.sfa_tile_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]
    comptime SFBTileArray = SMemTileArrayType[
        Self.sfb_type,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        alignment=128,
    ]

    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime SFATile = Self.SFATileArray.Tile
    comptime SFBTile = Self.SFBTileArray.Tile

    var pipeline: Self.Pipeline
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var sfa_tiles: Self.SFATileArray
    var sfb_tiles: Self.SFBTileArray

    @staticmethod
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread."""
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    comptime BarrierArray = SMemArrayType[
        SharedMemBarrier, Self.num_group_stages * 2
    ]

    @always_inline
    fn __init__(
        out self,
        barriers: Self.BarrierArray,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        sfa_tiles: Self.SFATileArray,
        sfb_tiles: Self.SFBTileArray,
    ):
        """Initialize from typed barrier array and all tile arrays."""
        self.pipeline = Self.Pipeline(barriers.ptr)
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.sfa_tiles = sfa_tiles
        self.sfb_tiles = sfb_tiles

    @always_inline
    fn producer[
        origin: MutOrigin
    ](ref [origin]self) -> BlockScaledInputProducer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get producer view for TMA Load warp."""
        return BlockScaledInputProducer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin
    ](ref [origin]self) -> BlockScaledInputConsumer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get consumer view for MMA warp."""
        return BlockScaledInputConsumer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn _acquire_producer_stage(
        mut self,
    ) -> Tuple[
        UInt32,
        MbarPtr,
        Self.ATileArray,
        Self.BTileArray,
        Self.SFATileArray,
        Self.SFBTileArray,
    ]:
        """Wait for slot availability and return stage info."""
        self.pipeline.wait_consumer()
        var stage = self.pipeline.producer_stage()
        return (
            stage,
            self.pipeline.producer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
            self.sfa_tiles,
            self.sfb_tiles,
        )

    @always_inline
    fn _release_producer_stage(mut self):
        """Signal completion and advance stage."""
        self.pipeline.producer_step()

    @always_inline
    fn _acquire_consumer_stage(
        mut self,
    ) -> Tuple[
        UInt32,
        MbarPtr,
        Self.ATileArray,
        Self.BTileArray,
        Self.SFATileArray,
        Self.SFBTileArray,
    ]:
        """Wait for data availability and return stage info."""
        self.pipeline.wait_producer()
        var stage = self.pipeline.consumer_stage()
        return (
            stage,
            self.pipeline.consumer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
            self.sfa_tiles,
            self.sfb_tiles,
        )

    @always_inline
    fn _release_consumer_stage(mut self):
        """Signal completion and advance stage."""
        self.pipeline.consumer_step()

    @always_inline
    fn producer_stage(self) -> UInt32:
        return self.pipeline.producer_stage()

    @always_inline
    fn consumer_stage(self) -> UInt32:
        return self.pipeline.consumer_stage()


# =============================================================================
# BlockScaledProducerStage - Producer stage with scaling factor access
# =============================================================================


@register_passable("trivial")
struct BlockScaledProducerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for producer tile access with scaling factors."""

    comptime TilePipelineType = BlockScaledTilePipeline[
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.TilePipelineType.ATileArray
    comptime BTileArray = Self.TilePipelineType.BTileArray
    comptime SFATileArray = Self.TilePipelineType.SFATileArray
    comptime SFBTileArray = Self.TilePipelineType.SFBTileArray
    comptime ATile = Self.TilePipelineType.ATile
    comptime BTile = Self.TilePipelineType.BTile
    comptime SFATile = Self.TilePipelineType.SFATile
    comptime SFBTile = Self.TilePipelineType.SFBTile

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: UInt32
    var _barrier: MbarPtr
    var _a_tiles: Self.ATileArray
    var _b_tiles: Self.BTileArray
    var _sfa_tiles: Self.SFATileArray
    var _sfb_tiles: Self.SFBTileArray

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: UInt32,
        barrier: MbarPtr,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        sfa_tiles: Self.SFATileArray,
        sfb_tiles: Self.SFBTileArray,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._barrier = barrier
        self._a_tiles = a_tiles
        self._b_tiles = b_tiles
        self._sfa_tiles = sfa_tiles
        self._sfb_tiles = sfb_tiles

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[]._release_producer_stage()

    @always_inline
    fn get_tile(self, k_idx: Int) -> Tuple[Self.ATile, Self.BTile]:
        """Get A and B tiles at the specified k-group index."""
        var idx = self._stage * Self.k_group_size + k_idx
        return (self._a_tiles[idx], self._b_tiles[idx])

    @always_inline
    fn get_sf_tile(self, k_idx: Int) -> Tuple[Self.SFATile, Self.SFBTile]:
        """Get A and B scaling factor tiles at the specified k-group index."""
        var idx = self._stage * Self.k_group_size + k_idx
        return (self._sfa_tiles[idx], self._sfb_tiles[idx])

    @always_inline
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_sfa_tile(self, k_idx: Int) -> Self.SFATile:
        """Get A scaling factor tile at the specified k-group index."""
        return self._sfa_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_sfb_tile(self, k_idx: Int) -> Self.SFBTile:
        """Get B scaling factor tile at the specified k-group index."""
        return self._sfb_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn expect_bytes(self, num_bytes: Int):
        """Set expected bytes on the barrier for TMA loads."""
        self._barrier[0].expect_bytes(num_bytes)

    @always_inline
    fn barrier(self) -> MbarPtr:
        """Get the barrier pointer for TMA multicast loads."""
        return self._barrier

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage


# =============================================================================
# BlockScaledConsumerStage - Consumer stage with scaling factor access
# =============================================================================


@register_passable("trivial")
struct BlockScaledConsumerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for consumer tile access with scaling factors."""

    comptime TilePipelineType = BlockScaledTilePipeline[
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.TilePipelineType.ATileArray
    comptime BTileArray = Self.TilePipelineType.BTileArray
    comptime SFATileArray = Self.TilePipelineType.SFATileArray
    comptime SFBTileArray = Self.TilePipelineType.SFBTileArray
    comptime ATile = Self.TilePipelineType.ATile
    comptime BTile = Self.TilePipelineType.BTile
    comptime SFATile = Self.TilePipelineType.SFATile
    comptime SFBTile = Self.TilePipelineType.SFBTile

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: UInt32
    var _mbar: MbarPtr
    var _a_tiles: Self.ATileArray
    var _b_tiles: Self.BTileArray
    var _sfa_tiles: Self.SFATileArray
    var _sfb_tiles: Self.SFBTileArray

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: UInt32,
        mbar: MbarPtr,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        sfa_tiles: Self.SFATileArray,
        sfb_tiles: Self.SFBTileArray,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._mbar = mbar
        self._a_tiles = a_tiles
        self._b_tiles = b_tiles
        self._sfa_tiles = sfa_tiles
        self._sfb_tiles = sfb_tiles

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[]._release_consumer_stage()

    @always_inline
    fn get_tile(self, k_idx: Int) -> Tuple[Self.ATile, Self.BTile]:
        """Get A and B tiles at the specified k-group index."""
        var idx = self._stage * Self.k_group_size + k_idx
        return (self._a_tiles[idx], self._b_tiles[idx])

    @always_inline
    fn get_sf_tile(self, k_idx: Int) -> Tuple[Self.SFATile, Self.SFBTile]:
        """Get A and B scaling factor tiles at the specified k-group index."""
        var idx = self._stage * Self.k_group_size + k_idx
        return (self._sfa_tiles[idx], self._sfb_tiles[idx])

    @always_inline
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_sfa_tile(self, k_idx: Int) -> Self.SFATile:
        """Get A scaling factor tile at the specified k-group index."""
        return self._sfa_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_sfb_tile(self, k_idx: Int) -> Self.SFBTile:
        """Get B scaling factor tile at the specified k-group index."""
        return self._sfb_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the barrier pointer for MMA commit."""
        return self._mbar

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage


# =============================================================================
# BlockScaledInputProducer - Producer view with scaling factor support
# =============================================================================


@fieldwise_init
@register_passable("trivial")
struct BlockScaledInputProducer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Producer view for TMA Load warp (block-scaled input pipeline)."""

    comptime TilePipelineType = BlockScaledTilePipeline[
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn drain(mut self):
        """Drain pipeline to prevent CTA exit while peer is still working."""

        @parameter
        for _ in range(Self.num_group_stages):
            self.pipeline_ptr[].pipeline.wait_consumer()
            self.pipeline_ptr[].pipeline.producer_step()

    @always_inline
    fn acquire(
        mut self,
    ) -> BlockScaledProducerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for slot availability."""
        var stage, barrier, a_tiles, b_tiles, sfa_tiles, sfb_tiles = (
            self.pipeline_ptr[]._acquire_producer_stage()
        )
        return BlockScaledProducerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            barrier=barrier,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
            sfa_tiles=sfa_tiles,
            sfb_tiles=sfb_tiles,
        )


# =============================================================================
# BlockScaledInputConsumer - Consumer view with scaling factor support
# =============================================================================


@fieldwise_init
@register_passable("trivial")
struct BlockScaledInputConsumer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    sfa_tile_layout: Layout,
    sfb_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Consumer view for MMA warp (block-scaled input pipeline)."""

    comptime TilePipelineType = BlockScaledTilePipeline[
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn acquire(
        mut self,
    ) -> BlockScaledConsumerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.sfa_type,
        Self.sfb_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.sfa_tile_layout,
        Self.sfb_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for tiles to be ready."""
        var stage, mbar, a_tiles, b_tiles, sfa_tiles, sfb_tiles = (
            self.pipeline_ptr[]._acquire_consumer_stage()
        )
        return BlockScaledConsumerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            mbar=mbar,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
            sfa_tiles=sfa_tiles,
            sfb_tiles=sfb_tiles,
        )
