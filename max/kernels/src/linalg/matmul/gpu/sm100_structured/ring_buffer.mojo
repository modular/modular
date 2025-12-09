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

"""Ring buffer for SM100 producer-consumer synchronization.

Provides SM90-style get_tiles() API for TMA-MMA pipeline synchronization.

Usage:
    var ring_buffer = RingBuffer[...](pipeline, a_tiles, b_tiles)

    # Producer: tiles contains stage, barrier, a_tiles, b_tiles
    with ring_buffer.producer() as producer:
        with producer.get_tiles() as tiles:
            load_tiles(tiles)  # Access tiles.a_tiles[stage * k_group + j]

    # Consumer: tiles contains stage, mbar, a_tiles, b_tiles
    with ring_buffer.consumer() as consumer:
        with consumer.get_tiles() as tiles:
            mma_tiles(tiles)  # Access tiles.a_tiles[stage * k_group + j]
"""

from memory import LegacyUnsafePointer as UnsafePointer

from layout import LayoutTensor
from layout.tma_async import PipelineState, SharedMemBarrier
from .pipeline import ProducerConsumerPipeline
from ....structuring import SMemPtr, SMemTileArrayType


comptime MbarPtr = SMemPtr[SharedMemBarrier]


# ============================================================================
# RingBuffer - Full ring buffer with tile storage (SM90-style API)
# ============================================================================


@register_passable("trivial")
struct RingBuffer[
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Ring buffer with tile storage for SM100 producer-consumer sync.

    This is the SM90-style API where tiles are stored in the ring buffer
    and returned directly from get_tiles().

    Template Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_tile_layout: Memory layout for A tiles.
        b_tile_layout: Memory layout for B tiles.
        num_pipeline_stages: Total number of tile stages.
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

    # ========== Barrier Initialization (called once) ==========

    @staticmethod
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            producer_arv_count: Expected arrival count for producer barriers.
            consumer_arv_count: Expected arrival count for consumer barriers.
        """
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    # ========== Constructor ==========

    @always_inline
    fn __init__(
        out self,
        storage_ptr: MbarPtr,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
    ):
        """Initialize ring buffer from storage pointer.

        Creates pipeline internally from storage pointer. Barriers must be
        initialized via init_barriers() before first use.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            a_tiles: A matrix tile array in shared memory.
            b_tiles: B matrix tile array in shared memory.
        """
        self.pipeline = Self.Pipeline(storage_ptr)
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles

    # ========== Producer/Consumer view factories ==========

    @always_inline
    fn producer[
        origin: Origin[True]
    ](ref [origin]self) -> Producer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get producer view with get_tiles() API."""
        return Producer(ring_buffer_ptr=Pointer(to=self))

    @always_inline
    fn consumer[
        origin: Origin[True]
    ](ref [origin]self) -> Consumer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get consumer view with get_tiles() API."""
        return Consumer(ring_buffer_ptr=Pointer(to=self))

    # ========== Producer API ==========

    @always_inline
    fn get_producer_tiles(
        mut self,
    ) -> Tuple[UInt32, MbarPtr, Self.ATileArray, Self.BTileArray]:
        """Wait for slot and return stage, barrier, and tile arrays.

        Synchronization is handled internally - waits for consumer to release slot.
        Use stage to index: tiles.a_tiles[stage * k_group_size + j]

        Returns:
            Tuple of (stage, barrier, a_tiles, b_tiles).
        """
        self.pipeline.wait_consumer()
        var stage = self.pipeline.producer_stage()
        return (
            stage,
            self.pipeline.producer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
        )

    @always_inline
    fn enqueue_tile(mut self):
        """Signal producer finished loading and advance stage."""
        self.pipeline.producer_step()

    @always_inline
    fn get_tile[
        tile_idx_in_group: Int
    ](self, stage: UInt32) -> Tuple[Self.ATile, Self.BTile]:
        """Get tiles at specific index within the current k_group."""
        var tile_idx = stage * Self.k_group_size + tile_idx_in_group
        return (self.a_tiles[tile_idx], self.b_tiles[tile_idx])

    # ========== Consumer API ==========

    @always_inline
    fn get_consumer_tiles(
        mut self,
    ) -> Tuple[UInt32, MbarPtr, Self.ATileArray, Self.BTileArray]:
        """Wait for slot and return stage, barrier, and tile arrays.

        Synchronization is handled internally - waits for producer to fill slot.
        Use stage to index: tiles.a_tiles[stage * k_group_size + j]

        Returns:
            Tuple of (stage, mbar, a_tiles, b_tiles).
        """
        self.pipeline.wait_producer()
        var stage = self.pipeline.consumer_stage()
        return (
            stage,
            self.pipeline.consumer_mbar(stage),
            self.a_tiles,
            self.b_tiles,
        )

    @always_inline
    fn release_slot(mut self):
        """Signal consumer finished and advance stage."""
        self.pipeline.consumer_step()

    # ========== Direct access ==========

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


# ============================================================================
# ProducerTiles/ConsumerTiles - Context managers with direct tile access
# ============================================================================


@fieldwise_init
@register_passable("trivial")
struct ProducerTiles[
    origin: Origin[True],
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for producer access with stage, barrier, and tile arrays.

    Provides everything needed to load k_group tiles in a single context:
        - stage: Current pipeline stage index
        - barrier: Barrier for synchronization (call expect_bytes once for all k_group)
        - a_tiles: Full A tile array (index with stage * k_group_size + j)
        - b_tiles: Full B tile array (index with stage * k_group_size + j)
    """

    comptime RingBufferType = RingBuffer[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.RingBufferType.ATileArray
    comptime BTileArray = Self.RingBufferType.BTileArray

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]
    var stage: UInt32
    var barrier: MbarPtr
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.ring_buffer_ptr[].enqueue_tile()


@fieldwise_init
@register_passable("trivial")
struct ConsumerTiles[
    origin: Origin[True],
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Context manager for consumer access with stage, barrier, and tile arrays.

    Provides everything needed to process k_group tiles in a single context:
        - stage: Current pipeline stage index
        - mbar: Barrier for synchronization (tiles ready when context entered)
        - a_tiles: Full A tile array (index with stage * k_group_size + j)
        - b_tiles: Full B tile array (index with stage * k_group_size + j)
    """

    comptime RingBufferType = RingBuffer[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]
    comptime ATileArray = Self.RingBufferType.ATileArray
    comptime BTileArray = Self.RingBufferType.BTileArray

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]
    var stage: UInt32
    var mbar: MbarPtr
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.ring_buffer_ptr[].release_slot()


# ============================================================================
# Producer/Consumer views with get_tiles()
# ============================================================================


@fieldwise_init
@register_passable("trivial")
struct Producer[
    origin: Origin[True],
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Producer view with get_tiles() API."""

    comptime RingBufferType = RingBuffer[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn drain(mut self):
        """Drain all pending pipeline stages.

        Prevents the CTA from exiting while a peer CTA is still working on MMA.
        Waits for each consumer slot and releases it, cycling through all
        num_group_stages stages.
        """

        @parameter
        for _ in range(Self.num_group_stages):
            self.ring_buffer_ptr[].pipeline.wait_consumer()
            self.ring_buffer_ptr[].pipeline.producer_step()

    @always_inline
    fn get_tiles(
        mut self,
    ) -> ProducerTiles[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get the next available slot with stage, barrier, and tile arrays.

        Synchronization is handled internally - waits for slot availability.
        """
        var stage, barrier, a_tiles, b_tiles = (
            self.ring_buffer_ptr[].get_producer_tiles()
        )
        return ProducerTiles(
            ring_buffer_ptr=self.ring_buffer_ptr,
            stage=stage,
            barrier=barrier,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


@fieldwise_init
@register_passable("trivial")
struct Consumer[
    origin: Origin[True],
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
]:
    """Consumer view with get_tiles() API."""

    comptime RingBufferType = RingBuffer[
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn get_tiles(
        mut self,
    ) -> ConsumerTiles[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_tile_layout,
        Self.b_tile_layout,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get the next slot with stage, barrier, and tile arrays.

        Synchronization is handled internally - waits for tiles to be ready.
        """
        var stage, mbar, a_tiles, b_tiles = (
            self.ring_buffer_ptr[].get_consumer_tiles()
        )
        return ConsumerTiles(
            ring_buffer_ptr=self.ring_buffer_ptr,
            stage=stage,
            mbar=mbar,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


# ============================================================================
# OutputRingBuffer - Ring buffer for MMA→Epilogue pipeline (TMEM stages)
# ============================================================================


@register_passable("trivial")
struct OutputStage[num_stages: Int]:
    """Stage info for output pipeline.

    Contains the stage index, computed TMEM offset, and a copy of the pipeline.
    This makes the stage self-contained, eliminating the need to pass the
    pipeline separately to functions like multi_stage_store_C.

    Template Parameters:
        num_stages: Number of pipeline stages (must match the OutputRingBuffer).
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]

    var stage: UInt32
    var tmem_offset: UInt32
    var pipeline: Self.Pipeline

    @always_inline
    fn __init__(
        out self, stage: UInt32, tmem_offset: UInt32, pipeline: Self.Pipeline
    ):
        self.stage = stage
        self.tmem_offset = tmem_offset
        self.pipeline = pipeline


@register_passable("trivial")
struct OutputRingBuffer[
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Ring buffer for MMA→Epilogue output pipeline.

    Manages TMEM accumulator stage synchronization between MMA warps (producer)
    and Epilogue warps (consumer). Unlike RingBuffer which manages SMEM tiles,
    this manages stage indices and computes TMEM offsets.

    The TMEM itself is allocated separately via tcgen05_alloc; this struct
    only coordinates access to different stages within that allocation.

    Template Parameters:
        num_stages: Number of accumulator pipeline stages.
        stage_stride_cols: TMEM column stride between stages.
        cta_group: CTA group size (1 or 2) for multicast signaling.

    Usage:
        # Initialize barriers once (elect_one_warp/elect_one_thread):
        OutputRingBuffer[...].init_barriers(storage_ptr, prod_cnt, cons_cnt)

        # Create ring buffer (each warp creates its own):
        var output_rb = OutputRingBuffer[...](storage_ptr, tmem_addr, mask)

        # MMA warp (producer):
        with output_rb.producer() as stage:
            # ... perform MMA into stage.tmem_offset ...

        # Epilogue warp (consumer):
        with output_rb.consumer() as stage:
            # ... read from stage.tmem_offset, write to GMEM ...
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]

    var pipeline: Self.Pipeline
    var tmem_base_addr: UInt32
    var mma_complete_mask: UInt16

    # ========== Barrier Initialization (called once) ==========

    @staticmethod
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            producer_arv_count: Expected arrival count for producer barriers.
            consumer_arv_count: Expected arrival count for consumer barriers.
        """
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    # ========== Constructor ==========

    @always_inline
    fn __init__(
        out self,
        storage_ptr: MbarPtr,
        tmem_base_addr: UInt32,
        mma_complete_mask: UInt16,
    ):
        """Initialize output ring buffer.

        Creates pipeline internally from storage pointer. Barriers must be
        initialized via init_barriers() before first use.

        Args:
            storage_ptr: Pointer to shared memory barrier storage.
            tmem_base_addr: Base TMEM address for accumulators.
            mma_complete_mask: Multicast mask for 2-SM MMA completion signaling.
        """
        self.pipeline = Self.Pipeline(storage_ptr)
        self.tmem_base_addr = tmem_base_addr
        self.mma_complete_mask = mma_complete_mask

    comptime Stage = OutputStage[Self.num_stages]

    # ========== MMA (Producer) Interface ==========

    @always_inline
    fn acquire_for_mma(self) -> Self.Stage:
        """Acquire a stage for MMA computation.

        Waits for the epilogue to finish with this stage, then returns
        the stage info with computed TMEM offset and pipeline reference.

        Returns:
            OutputStage with stage index, TMEM offset, and pipeline for signaling.
        """
        var stage = self.pipeline.producer_stage()
        self.pipeline.wait_consumer()
        var tmem_offset = self.tmem_base_addr + (
            stage * UInt32(Self.stage_stride_cols)
        )
        return Self.Stage(stage, tmem_offset, self.pipeline)

    @always_inline
    fn release_from_mma(mut self, stage: Self.Stage):
        """Signal MMA completion and advance to next stage.

        Signals the epilogue that accumulator data is ready, using either
        mma_arrive (1-SM) or mma_arrive_multicast (2-SM).

        Args:
            stage: The stage being released (from acquire_for_mma).
        """
        from gpu.cluster import elect_one_sync
        from gpu.mma_sm100 import mma_arrive, mma_arrive_multicast

        if elect_one_sync():

            @parameter
            if Self.cta_group == 1:
                mma_arrive[Self.cta_group](
                    self.pipeline.producer_mbar(stage.stage)
                )
            else:
                mma_arrive_multicast[Self.cta_group](
                    self.pipeline.producer_mbar(stage.stage),
                    self.mma_complete_mask,
                )
        self.pipeline.producer_step()

    # ========== Epilogue (Consumer) Interface ==========

    @always_inline
    fn acquire_for_epilogue(self) -> Self.Stage:
        """Acquire a stage for epilogue processing.

        Waits for MMA to complete this stage, then returns the stage info.

        Returns:
            OutputStage with stage index, TMEM offset, and pipeline for signaling.
        """
        var stage = self.pipeline.consumer_stage()
        self.pipeline.wait_producer()
        var tmem_offset = self.tmem_base_addr + (
            stage * UInt32(Self.stage_stride_cols)
        )
        return Self.Stage(stage, tmem_offset, self.pipeline)

    @always_inline
    fn release_from_epilogue(mut self):
        """Signal epilogue completion and advance to next stage.

        Signals MMA that this accumulator stage is free for reuse.
        """
        self.pipeline.consumer_step()

    # ========== Context Manager Interface ==========

    @always_inline
    fn producer[
        origin: MutOrigin, //
    ](ref [origin]self) -> OutputProducerContext[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get a producer context for MMA warp.

        Usage:
            with output_rb.producer() as stage:
                # MMA into stage.tmem_offset
            # release_from_mma called automatically
        """
        return OutputProducerContext(Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin, //
    ](ref [origin]self) -> OutputConsumerContext[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get a consumer context for epilogue warp.

        Usage:
            with output_rb.consumer() as stage:
                # Read from stage.tmem_offset, write to GMEM
            # release_from_epilogue called automatically
        """
        return OutputConsumerContext(Pointer(to=self))

    # ========== Pipeline Access (for barrier initialization) ==========

    @always_inline
    fn get_pipeline(self) -> Self.Pipeline:
        """Get the underlying pipeline for barrier initialization.

        Note: With OutputStage now carrying the pipeline, most code no longer
        needs this. It's retained for init_barriers() which needs the raw
        pipeline before any OutputStage instances exist.
        """
        return self.pipeline


@register_passable("trivial")
struct OutputProducerContext[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Context manager for MMA producer access to OutputRingBuffer.

    Automatically calls acquire_for_mma on enter and release_from_mma on exit.

    Usage:
        with output_rb.producer() as stage:
            # ... MMA into stage.tmem_offset ...
        # release_from_mma called automatically
    """

    comptime RingBufferType = OutputRingBuffer[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[Self.num_stages]

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]
    var stage: Self.Stage

    @always_inline
    fn __init__(
        out self, ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]
    ):
        self.ring_buffer_ptr = ring_buffer_ptr
        # Dummy initialization - will be set in __enter__
        self.stage = Self.Stage(
            0, 0, ProducerConsumerPipeline[Self.num_stages](MbarPtr())
        )

    @always_inline
    fn __enter__(mut self) -> Self.Stage:
        self.stage = self.ring_buffer_ptr[].acquire_for_mma()
        return self.stage

    @always_inline
    fn __exit__(mut self):
        self.ring_buffer_ptr[].release_from_mma(self.stage)


@register_passable("trivial")
struct OutputConsumerContext[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Context manager for epilogue consumer access to OutputRingBuffer.

    Automatically calls acquire_for_epilogue on enter and release_from_epilogue
    on exit.

    Usage:
        with output_rb.consumer() as stage:
            # ... read from stage.tmem_offset, write to GMEM ...
        # release_from_epilogue called automatically
    """

    comptime RingBufferType = OutputRingBuffer[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[Self.num_stages]

    var ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]

    @always_inline
    fn __init__(
        out self, ring_buffer_ptr: Pointer[Self.RingBufferType, Self.origin]
    ):
        self.ring_buffer_ptr = ring_buffer_ptr

    @always_inline
    fn __enter__(mut self) -> Self.Stage:
        return self.ring_buffer_ptr[].acquire_for_epilogue()

    @always_inline
    fn __exit__(mut self):
        self.ring_buffer_ptr[].release_from_epilogue()
