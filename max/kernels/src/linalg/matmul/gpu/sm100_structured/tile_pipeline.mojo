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

"""Tile pipeline for SM100 producer-consumer synchronization.

Provides staged tile storage with producer-consumer barrier synchronization
for TMA-MMA pipeline coordination.

Key abstractions:
- TilePipeline: Manages A/B tiles across pipeline stages with sync barriers
- OutputTilePipeline: Manages TMEM accumulator stages for MMA→Epilogue pipeline

Usage:
    var tile_pipeline = TilePipeline[...](storage_ptr, a_tiles, b_tiles)

    # Producer (TMA Load warp):
    with tile_pipeline.producer() as producer:
        with producer.acquire() as tiles:
            tiles.expect_bytes(expected_bytes)
            for j in range(k_group_size):
                var a_tile, b_tile = tiles.get_tile(j)
                # TMA load into a_tile, b_tile
        # Automatically signals completion on exit

    # Consumer (MMA warp):
    with tile_pipeline.consumer() as consumer:
        with consumer.acquire() as tiles:
            for j in range(k_group_size):
                var a_tile, b_tile = tiles.get_tile(j)
                # MMA using a_tile, b_tile
        # Automatically signals completion on exit
"""

from layout.tma_async import SharedMemBarrier
from .pipeline import ProducerConsumerPipeline
from .tmem import TmemAllocation, TmemStage
from linalg.structuring import SMemPtr, SMemTileArrayType, SMemArrayType


comptime MbarPtr = SMemPtr[SharedMemBarrier]


# ============================================================================
# TilePipeline - Staged tile storage with producer-consumer synchronization
# ============================================================================


@register_passable("trivial")
struct TilePipeline[
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Staged tile storage with producer-consumer synchronization for SM100.

    Manages a fixed set of pipeline stages (not a FIFO queue) where:
    - Producer (TMA Load) fills tiles into the current stage
    - Consumer (MMA) reads tiles from the current stage
    - Barriers coordinate access between producer and consumer

    Template Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_tile_layout: Memory layout for A tiles.
        b_tile_layout: Memory layout for B tiles.
        num_pipeline_stages: Total number of tile stages (stages * k_group_size).
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_group_stages]
    comptime ATileArray = SMemTileArrayType[
        Self.a_type, Self.a_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime BTileArray = SMemTileArrayType[
        Self.b_type, Self.b_tile_layout, Self.num_pipeline_stages, alignment=128
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile

    var pipeline: Self.Pipeline
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

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
    ):
        """Initialize from typed barrier array and tile arrays."""
        self.pipeline = Self.Pipeline(barriers.ptr)
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles

    @always_inline
    fn producer[
        origin: MutOrigin
    ](ref [origin]self) -> InputProducer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get producer view for TMA Load warp."""
        return InputProducer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin
    ](ref [origin]self) -> InputConsumer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get consumer view for MMA warp."""
        return InputConsumer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn _acquire_producer_stage(
        mut self,
    ) -> Tuple[UInt32, MbarPtr, Self.ATileArray, Self.BTileArray]:
        """Wait for slot availability and return stage info."""
        self.pipeline.wait_consumer()
        var stage = self.pipeline.producer_stage()
        return (
            stage,
            self.pipeline.producer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
        )

    @always_inline
    fn _release_producer_stage(mut self):
        """Signal completion and advance stage."""
        self.pipeline.producer_step()

    @always_inline
    fn _acquire_consumer_stage(
        mut self,
    ) -> Tuple[UInt32, MbarPtr, Self.ATileArray, Self.BTileArray]:
        """Wait for data availability and return stage info."""
        self.pipeline.wait_producer()
        var stage = self.pipeline.consumer_stage()
        return (
            stage,
            self.pipeline.consumer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
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

    @always_inline
    fn producer_mbar(self, stage: UInt32) -> MbarPtr:
        return self.pipeline.producer_mbar(stage)

    @always_inline
    fn consumer_mbar(self, stage: UInt32) -> MbarPtr:
        return self.pipeline.consumer_mbar(stage)


@register_passable("trivial")
struct ProducerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for producer tile access with encapsulated stage indexing.
    """

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.TilePipelineType.ATileArray
    comptime BTileArray = Self.TilePipelineType.BTileArray
    comptime ATile = Self.TilePipelineType.ATile
    comptime BTile = Self.TilePipelineType.BTile

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: UInt32
    var _barrier: MbarPtr
    var _a_tiles: Self.ATileArray
    var _b_tiles: Self.BTileArray

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: UInt32,
        barrier: MbarPtr,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._barrier = barrier
        self._a_tiles = a_tiles
        self._b_tiles = b_tiles

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
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[self._stage * Self.k_group_size + k_idx]

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


@register_passable("trivial")
struct ConsumerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for consumer tile access with encapsulated stage indexing.
    """

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.TilePipelineType.ATileArray
    comptime BTileArray = Self.TilePipelineType.BTileArray
    comptime ATile = Self.TilePipelineType.ATile
    comptime BTile = Self.TilePipelineType.BTile

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: UInt32
    var _mbar: MbarPtr
    var _a_tiles: Self.ATileArray
    var _b_tiles: Self.BTileArray

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: UInt32,
        mbar: MbarPtr,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._mbar = mbar
        self._a_tiles = a_tiles
        self._b_tiles = b_tiles

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
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[self._stage * Self.k_group_size + k_idx]

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the barrier pointer for MMA commit."""
        return self._mbar

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage


@fieldwise_init
@register_passable("trivial")
struct InputProducer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Producer view for TMA Load warp (input pipeline)."""

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
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
    ) -> ProducerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for slot availability."""
        var stage, barrier, a_tiles, b_tiles = (
            self.pipeline_ptr[]._acquire_producer_stage()
        )
        return ProducerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            barrier=barrier,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


@fieldwise_init
@register_passable("trivial")
struct InputConsumer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Consumer view for MMA warp (input pipeline)."""

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
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
    ) -> ConsumerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for tiles to be ready."""
        var stage, mbar, a_tiles, b_tiles = (
            self.pipeline_ptr[]._acquire_consumer_stage()
        )
        return ConsumerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            mbar=mbar,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


@register_passable("trivial")
struct OutputStage[
    num_stages: Int,
    stage_stride: Int,
    cta_group: Int,
]:
    """Acquired output stage with TMEM handle and pipeline reference."""

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]
    comptime Tmem = TmemStage[
        Self.num_stages, Self.stage_stride, Self.cta_group
    ]

    var index: UInt32
    var tmem: Self.Tmem
    var pipeline: Self.Pipeline

    @always_inline
    fn __init__(
        out self,
        index: UInt32,
        tmem: Self.Tmem,
        pipeline: Self.Pipeline,
    ):
        self.index = index
        self.tmem = tmem
        self.pipeline = pipeline


@register_passable("trivial")
struct OutputTilePipeline[
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Pipeline for MMA→Epilogue TMEM stage synchronization."""

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]
    comptime BarrierArray = SMemArrayType[SharedMemBarrier, Self.num_stages * 2]
    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline: Self.Pipeline
    var tmem: Self.Tmem
    var mma_complete_mask: UInt16

    @staticmethod
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread."""
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    @always_inline
    fn __init__(
        out self,
        barriers: Self.BarrierArray,
        tmem: Self.Tmem,
        mma_complete_mask: UInt16,
    ):
        """Initialize from barrier array, TMEM allocation, and multicast mask.
        """
        self.pipeline = Self.Pipeline(barriers.ptr)
        self.tmem = tmem
        self.mma_complete_mask = mma_complete_mask

    @always_inline
    fn acquire_for_mma(self) -> Self.Stage:
        """Acquire stage for MMA, waiting for epilogue to finish."""
        var idx = self.pipeline.producer_stage()
        self.pipeline.wait_consumer()
        var tmem = Self.Stage.Tmem(self.tmem, Int(idx))
        return Self.Stage(idx, tmem, self.pipeline)

    @always_inline
    fn release_from_mma(mut self, stage: Self.Stage):
        """Signal MMA completion using mma_arrive (1-SM) or multicast (2-SM)."""
        from gpu.cluster import elect_one_sync
        from gpu.mma_sm100 import mma_arrive, mma_arrive_multicast

        if elect_one_sync():

            @parameter
            if Self.cta_group == 1:
                mma_arrive[Self.cta_group](
                    self.pipeline.producer_mbar(stage.index)
                )
            else:
                mma_arrive_multicast[Self.cta_group](
                    self.pipeline.producer_mbar(stage.index),
                    self.mma_complete_mask,
                )
        self.pipeline.producer_step()

    @always_inline
    fn acquire_for_epilogue(self) -> Self.Stage:
        """Acquire stage for epilogue, waiting for MMA to complete."""
        var idx = self.pipeline.consumer_stage()
        self.pipeline.wait_producer()
        var tmem = Self.Stage.Tmem(self.tmem, Int(idx))
        return Self.Stage(idx, tmem, self.pipeline)

    @always_inline
    fn release_from_epilogue(mut self):
        """Signal epilogue completion, freeing stage for MMA reuse."""
        self.pipeline.consumer_step()

    @always_inline
    fn producer[
        origin: MutOrigin, //
    ](ref [origin]self) -> OutputProducer[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get producer view for MMA warp."""
        return OutputProducer(Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin, //
    ](ref [origin]self) -> OutputConsumer[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get consumer view for epilogue warp."""
        return OutputConsumer(Pointer(to=self))

    @always_inline
    fn get_pipeline(self) -> Self.Pipeline:
        """Get underlying pipeline (used during barrier initialization)."""
        return self.pipeline


@register_passable("trivial")
struct OutputProducer[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Producer view for MMA warp (output pipeline)."""

    comptime TilePipelineType = OutputTilePipeline[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var stage: Self.Stage

    @always_inline
    fn __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.Stage.Tmem(0, 0)
        self.stage = Self.Stage(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_stages](MbarPtr()),
        )

    @always_inline
    fn __enter__(mut self) -> Self.Stage:
        self.stage = self.pipeline_ptr[].acquire_for_mma()
        return self.stage

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[].release_from_mma(self.stage)


@register_passable("trivial")
struct OutputConsumer[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Consumer view for epilogue warp (output pipeline)."""

    comptime TilePipelineType = OutputTilePipeline[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    fn __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr

    @always_inline
    fn __enter__(mut self) -> Self.Stage:
        return self.pipeline_ptr[].acquire_for_epilogue()

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[].release_from_epilogue()
