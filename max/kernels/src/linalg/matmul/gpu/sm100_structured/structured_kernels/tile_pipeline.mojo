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

"""Tile pipeline for SM100 producer-consumer synchronization.

Provides staged tile storage with producer-consumer barrier synchronization
for TMA-MMA pipeline coordination. All barrier operations are encapsulated
in context managers for safety and clarity.

All tiles use TileTensor natively. Convert to LayoutTensor at TMA/MMA
boundaries using {ptr} syntax or explicit LayoutTensor construction.

Key Abstractions
----------------
- InputTilePipeline[Payload]: Generic pipeline with payload abstraction
- TilePipeline: Standard pipeline with explicit A/B tile types
- OutputTilePipeline: TMEM accumulator stages for MMA→Epilogue pipeline

Naming Conventions
------------------
- *Pipeline: Multi-stage buffer (InputTilePipeline, OutputTilePipeline)
- *Producer/*Consumer: Role handles (InputProducer, OutputConsumer)
- acquire(): Context manager to get one pipeline stage

Context Manager Semantics
-------------------------
Each `with` block handles barrier synchronization automatically:

    with producer.acquire() as tiles:   # BLOCKS until consumer releases stage
        load_tiles(tiles)                # safe to write
                                         # EXIT: signals producer barrier, advances

    with consumer.acquire() as tiles:   # BLOCKS until producer fills stage
        use_tiles(tiles)                 # safe to read
                                         # EXIT: signals consumer barrier, advances

Example: TMA Load Warp (Producer)
---------------------------------
    with input_pipeline.producer() as producer:  # producer role for this warp
        while work_iter.has_work():
            with work_iter.next() as current:
                for i in range(num_iters):
                    with producer.acquire() as tiles:  # waits for consumer
                        tma_load(tiles.a_tile(), tiles.b_tile())
        producer.drain()  # wait for all stages consumed before CTA exits

Example: MMA Warp (Consumer + Output Producer)
----------------------------------------------
    with mma_ctx:  # TMEM lifecycle
        while work_iter.has_work():
            with work_iter.wait_and_advance():  # blocks on CLC response
                with output_pipeline.producer() as output_stage:  # waits for epilogue
                    with input_pipeline.consumer() as consumer:
                        for i in range(num_iters):
                            with consumer.acquire() as input_tiles:  # waits for TMA
                                mma(output_stage.tmem, input_tiles)

Example: Epilogue Warp (Output Consumer)
----------------------------------------
    with epi_ctx:  # signals TMEM dealloc on exit
        while work_iter.has_work():
            with work_iter.next() as current:
                with output_pipeline.consumer() as output_stage:  # waits for MMA
                    write_output(output_stage)
"""

from layout import Layout
from layout.tma_async import SharedMemBarrier
from .pipeline import ProducerConsumerPipeline
from .tmem import TmemAllocation, TmemStage

# SMemArray for barriers (non-tile arrays), SMemPtr for pointers
from linalg.structuring import SMemPtr, SMemArray

# LayoutTensor-based SMemTileArray for tiles needing element-level access (A-scales)
from linalg.structuring import SMemTileArray as LTSMemTileArray

# TileTensor-based tile arrays for most tile storage (A, B)
from .tile_types import (
    SMemTileArray2D,
    SMemTileArrayWithLayout,
    internal_sf_k_major,
)

comptime MbarPtr = SMemPtr[SharedMemBarrier]


# ============================================================================
# Tile Payloads - Data containers for pipeline tile arrays
# ============================================================================


trait TilePayload(TrivialRegisterPassable):
    """Trait for tile payload types. Must be extend TrivialRegisterPassable."""

    pass


struct StandardTilePayload[
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload):
    """Tile payload for standard matmul (A and B tiles).

    Uses explicit dimensions for tile arrays. The tiles are stored as TileTensor
    with row_major layout. TileTensors are passed directly to TMA/MMA.
    at TMA/MMA boundaries.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.num_pipeline_stages,
        128,
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type,
        Self.b_dim0,
        Self.b_dim1,
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

    @always_inline
    fn __init__(out self, a_tiles: Self.ATileArray, b_tiles: Self.BTileArray):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles

    @always_inline
    fn get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[Self.ATile, Self.BTile]:
        """Get A and B tiles at the specified stage and k-group index."""
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (self.a_tiles[idx], self.b_tiles[idx])

    @always_inline
    fn get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index."""
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index."""
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


struct BlockScaledTilePayload[
    a_type: DType,
    b_type: DType,
    sfa_type: DType,
    sfb_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # SFA tile dimensions
    sfa_dim0: Int,
    sfa_dim1: Int,
    # SFB tile dimensions
    sfb_dim0: Int,
    sfb_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload):
    """Tile payload for block-scaled matmul (A, B, SFA, SFB tiles)."""

    comptime ATileArray = SMemTileArray2D[
        Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type, Self.b_dim0, Self.b_dim1, Self.num_pipeline_stages, 128
    ]
    # SF tiles use internal_sf_k_major layout (matches tile_sf_layout_k_major).
    # MMA extracts layout directly from TileTensor type parameters.
    comptime sfa_layout = internal_sf_k_major[Self.sfa_dim0, Self.sfa_dim1]
    comptime sfb_layout = internal_sf_k_major[Self.sfb_dim0, Self.sfb_dim1]
    comptime SFATileArray = SMemTileArrayWithLayout[
        Self.sfa_type,
        Self.sfa_layout,
        Self.num_pipeline_stages,
        128,
    ]
    comptime SFBTileArray = SMemTileArrayWithLayout[
        Self.sfb_type,
        Self.sfb_layout,
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime SFATile = Self.SFATileArray.Tile
    comptime SFBTile = Self.SFBTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var sfa_tiles: Self.SFATileArray
    var sfb_tiles: Self.SFBTileArray

    @always_inline
    fn __init__(
        out self,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        sfa_tiles: Self.SFATileArray,
        sfb_tiles: Self.SFBTileArray,
    ):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.sfa_tiles = sfa_tiles
        self.sfb_tiles = sfb_tiles

    @always_inline
    fn get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[
        Self.ATile, Self.BTile, Self.SFATile, Self.SFBTile
    ]:
        """Get A, B, SFA, SFB tiles at the specified stage and k-group index."""
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles[idx],
            self.b_tiles[idx],
            self.sfa_tiles[idx],
            self.sfb_tiles[idx],
        )

    @always_inline
    fn get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index."""
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index."""
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_sfa_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.SFATile:
        """Get SFA tile at the specified stage and k-group index."""
        return self.sfa_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_sfb_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.SFBTile:
        """Get SFB tile at the specified stage and k-group index."""
        return self.sfb_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


struct BlockwiseFP8TilePayload[
    a_type: DType,
    b_type: DType,
    a_scales_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # A-scales tile dimensions
    a_scales_dim0: Int,
    a_scales_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
](TilePayload, TrivialRegisterPassable):
    """Tile payload for blockwise FP8 matmul (A, B, A-scales tiles).

    Unlike BlockScaledTilePayload, this only stores A-scales in SMEM.
    B-scales are read directly from global memory during the epilogue phase.
    """

    comptime ATileArray = SMemTileArray2D[
        Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type, Self.b_dim0, Self.b_dim1, Self.num_pipeline_stages, 128
    ]
    # TileTensor-based for A-scales (explicit dimensions)
    comptime AScalesTileArray = SMemTileArray2D[
        Self.a_scales_type,
        Self.a_scales_dim0,
        Self.a_scales_dim1,
        Self.num_pipeline_stages,
        128,
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile
    comptime AScalesTile = Self.AScalesTileArray.Tile

    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray
    var a_scales_tiles: Self.AScalesTileArray

    @always_inline
    fn __init__(
        out self,
        a_tiles: Self.ATileArray,
        b_tiles: Self.BTileArray,
        a_scales_tiles: Self.AScalesTileArray,
    ):
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles
        self.a_scales_tiles = a_scales_tiles

    @always_inline
    fn get_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Tuple[
        Self.ATile, Self.BTile, Self.AScalesTile
    ]:
        """Get A, B, A-scales tiles at the specified stage and k-group index."""
        var idx = stage * UInt32(k_group_size) + UInt32(k_idx)
        return (
            self.a_tiles[idx],
            self.b_tiles[idx],
            self.a_scales_tiles[idx],
        )

    @always_inline
    fn get_a_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified stage and k-group index."""
        return self.a_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_b_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified stage and k-group index."""
        return self.b_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]

    @always_inline
    fn get_a_scales_tile[
        k_group_size: Int
    ](self, stage: UInt32, k_idx: Int) -> Self.AScalesTile:
        """Get A-scales tile at the specified stage and k-group index."""
        return self.a_scales_tiles[stage * UInt32(k_group_size) + UInt32(k_idx)]


# ============================================================================
# InputTilePipeline - Generic pipeline parameterized by payload type
# ============================================================================


struct InputTilePipeline[
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Tile pipeline with configurable payload type.

    Separates synchronization from tile storage. The Payload parameter
    (e.g., StandardTilePayload or BlockScaledTilePayload) holds tile arrays.
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_group_stages]
    comptime BarrierArray = SMemArray[
        SharedMemBarrier, Self.num_group_stages * 2
    ]

    var pipeline: Self.Pipeline
    var payload: Self.Payload

    @staticmethod
    @always_inline
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread."""
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    @always_inline
    fn __init__(out self, barriers: Self.BarrierArray, payload: Self.Payload):
        """Initialize from typed barrier array and payload."""
        self.pipeline = Self.Pipeline(barriers.ptr)
        self.payload = payload

    @always_inline
    fn acquire_producer(mut self) -> Tuple[UInt32, MbarPtr]:
        """Wait for slot availability and return (stage, barrier)."""
        self.pipeline.wait_consumer()
        var stage = self.pipeline.producer_stage()
        return (stage, self.pipeline.producer_mbar(stage))

    @always_inline
    fn release_producer(mut self):
        """Signal completion and advance producer stage."""
        self.pipeline.producer_step()

    @always_inline
    fn acquire_consumer(mut self) -> Tuple[UInt32, MbarPtr]:
        """Wait for data availability and return (stage, barrier)."""
        self.pipeline.wait_producer()
        var stage = self.pipeline.consumer_stage()
        return (stage, self.pipeline.consumer_mbar(stage))

    @always_inline
    fn release_consumer(mut self):
        """Signal completion and advance consumer stage."""
        self.pipeline.consumer_step()

    # ========== Try-Acquire Pattern Methods ==========
    # These enable overlapping barrier checks with useful work.

    @always_inline
    fn try_acquire_producer(self) -> Bool:
        """Non-blocking check if next producer stage is available.

        Returns:
            True if consumer has freed the stage, False otherwise.

        Example (TMA Load warp):
            ```
            var ready = pipeline.try_acquire_producer()
            # ... do other work while potentially waiting ...
            pipeline.wait_producer_if_needed(ready)
            var stage = pipeline.producer_stage()
            # ... load tiles ...
            ```
        """
        return self.pipeline.try_wait_consumer()

    @always_inline
    fn try_acquire_consumer(self) -> Bool:
        """Non-blocking check if next consumer stage has data.

        Returns:
            True if producer has filled the stage, False otherwise.

        Example (MMA warp):
            ```
            var ready = pipeline.try_acquire_consumer()
            # ... do other work while potentially waiting ...
            pipeline.wait_consumer_if_needed(ready)
            var stage = pipeline.consumer_stage()
            # ... process tiles ...
            ```
        """
        return self.pipeline.try_wait_producer()

    @always_inline
    fn wait_producer_if_needed(self, already_ready: Bool):
        """Conditionally wait for producer stage if not already ready.

        Args:
            already_ready: Result from try_acquire_consumer().
        """
        self.pipeline.wait_producer_if_needed(already_ready)

    @always_inline
    fn wait_consumer_if_needed(self, already_ready: Bool):
        """Conditionally wait for consumer to free stage if not already ready.

        Args:
            already_ready: Result from try_acquire_producer().
        """
        self.pipeline.wait_consumer_if_needed(already_ready)

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

    @always_inline
    fn producer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputProducer[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Get producer view for TMA Load warp."""
        return InputProducer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn consumer[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputConsumer[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Get consumer view for MMA warp."""
        return InputConsumer(pipeline_ptr=Pointer(to=self))

    # =========================================================================
    # Linear Type API - Compiler-enforced resource management
    # =========================================================================

    @always_inline
    fn acquire_producer_linear[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputProducerStage[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire a producer stage handle using linear types.

        Waits for the consumer to free the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var tiles = pipeline.acquire_producer_linear()
            load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
            tiles^.release()  # Advances to next stage

        Returns:
            An InputProducerStage handle that must be released.
        """
        var stage, barrier = self.acquire_producer()
        return InputProducerStage(
            pipeline_ptr=Pointer(to=self), stage=stage, barrier=barrier
        )

    @always_inline
    fn acquire_consumer_linear[
        mut_origin: MutOrigin
    ](ref[mut_origin] self) -> InputConsumerStage[
        mut_origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire a consumer stage handle using linear types.

        Waits for the producer to fill the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var tiles = pipeline.acquire_consumer_linear()
            process_tiles(tiles.payload(), tiles.stage())
            tiles^.release()  # Signals complete and advances

        Returns:
            An InputConsumerStage handle that must be released.
        """
        var stage, mbar = self.acquire_consumer()
        return InputConsumerStage(
            pipeline_ptr=Pointer(to=self), stage=stage, mbar=mbar
        )

    @always_inline
    fn drain_producer(mut self):
        """Drain pipeline to prevent CTA exit while peer is still working.

        Call this after all producer iterations are complete.
        This is the linear type equivalent of InputProducer.drain().
        """

        comptime for _ in range(Self.num_group_stages):
            self.pipeline.wait_consumer()
            self.pipeline.producer_step()


# ============================================================================
# InputProducerStage/InputConsumerStage - Unified linear types for tile access
# ============================================================================
#
# These types can be used in two ways:
#
# 1. Linear Type API (flat, explicit):
#    var tiles = input_pipeline.acquire_producer()
#    load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
#    tiles^.release()  # Compiler enforces this call
#
# 2. Context Manager API (scoped, automatic):
#    with producer.acquire() as tiles:
#        load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
#    # release() called automatically by context manager
#
# ============================================================================


struct InputProducerStage[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Handle for producer tile access - works as context manager or linear-style.

    Two usage patterns:

    1. Context manager (scoped):
        with producer.acquire() as tiles:
            load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
        # release() called automatically by __exit__

    2. Linear-style (flat):
        var tiles = producer.acquire()
        load_tiles(tiles.payload(), tiles.stage(), tiles.barrier())
        tiles.release()  # Manual release

    Lifecycle:
    1. Created via `producer.acquire()` - waits for consumer
    2. Use `payload()`, `stage()`, `barrier()` for TMA operations
    3. Call `release()` or let `__exit__` advance producer stage

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _barrier: MbarPtr

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        barrier: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._barrier = barrier

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[].release_producer()

    @always_inline
    fn payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    fn expect_bytes(self, num_bytes: Int):
        """Set expected bytes on the barrier for TMA loads."""
        self._barrier[0].expect_bytes(Int32(num_bytes))

    @always_inline
    fn barrier(self) -> MbarPtr:
        """Get the barrier pointer for TMA multicast loads."""
        return self._barrier

    @always_inline
    fn release(mut self):
        """Advance producer to next stage (linear-style API).

        Use this for flat code structure instead of context manager.
        Equivalent to what __exit__ does.
        """
        self.pipeline_ptr[].release_producer()


struct InputConsumerStage[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Handle for consumer tile access - works as context manager or linear-style.

    Two usage patterns:

    1. Context manager (scoped):
        with consumer.acquire() as tiles:
            process_tiles(tiles.payload(), tiles.stage())
        # release() called automatically by __exit__

    2. Linear-style (flat):
        var tiles = consumer.acquire()
        process_tiles(tiles.payload(), tiles.stage())
        tiles.release()  # Manual release

    Lifecycle:
    1. Created via `consumer.acquire()` - waits for producer
    2. Use `payload()`, `stage()` for tile access
    3. Call `release()` or let `__exit__` signal and advance

    Parameters:
        origin: Origin of the pipeline reference.
        Payload: The tile payload type.
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]
    var _stage: UInt32
    var _mbar: MbarPtr

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.PipelineType, Self.origin],
        stage: UInt32,
        mbar: MbarPtr,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage
        self._mbar = mbar

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        self.pipeline_ptr[].release_consumer()

    @always_inline
    fn payload(self) -> Self.Payload:
        """Get the tile payload for direct access."""
        return self.pipeline_ptr[].payload

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the barrier pointer."""
        return self._mbar

    @always_inline
    fn release(mut self):
        """Signal consumption and advance to next stage (linear-style API).

        Use this for flat code structure instead of context manager.
        Equivalent to what __exit__ does.
        """
        self.pipeline_ptr[].release_consumer()


# ============================================================================
# InputProducer/InputConsumer - Role handles with acquire() for InputTilePipeline
# ============================================================================


@fieldwise_init
struct InputProducer[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Producer view for TMA Load warp. Use acquire() to get stages."""

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn drain(mut self):
        """Drain pipeline to prevent CTA exit while peer is still working."""

        comptime for _ in range(Self.num_group_stages):
            self.pipeline_ptr[].pipeline.wait_consumer()
            self.pipeline_ptr[].pipeline.producer_step()

    @always_inline
    fn acquire(
        mut self,
    ) -> InputProducerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage, waiting for slot availability.

        Returns a context manager for loading tiles.
        """
        var stage_idx, barrier = self.pipeline_ptr[].acquire_producer()
        return InputProducerStage(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, barrier=barrier
        )

    @always_inline
    fn try_acquire(mut self) -> Bool:
        """Non-blocking check if next producer stage is available.

        Returns:
            True if the stage is ready, False if waiting is needed.

        Use with acquire_if_needed() for the try-acquire pattern:
        ```
        var ready = producer.try_acquire()
        # ... do other work ...
        with producer.acquire_if_needed(ready) as tiles:
            load_tiles()
        ```
        """
        return self.pipeline_ptr[].try_acquire_producer()

    @always_inline
    fn acquire_if_needed(
        mut self, already_ready: Bool
    ) -> InputProducerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire stage, only waiting if not already ready.

        Args:
            already_ready: Result from try_acquire(). Skips wait if True.

        Returns:
            The producer stage for loading tiles.
        """
        self.pipeline_ptr[].wait_consumer_if_needed(already_ready)
        var stage_idx = self.pipeline_ptr[].producer_stage()
        var barrier = self.pipeline_ptr[].producer_mbar(stage_idx)
        return InputProducerStage(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, barrier=barrier
        )


@fieldwise_init
struct InputConsumer[
    origin: MutOrigin,
    Payload: TilePayload,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Consumer view for MMA warp. Use acquire() to get stages."""

    comptime PipelineType = InputTilePipeline[
        Self.Payload, Self.num_group_stages, Self.k_group_size
    ]

    var pipeline_ptr: Pointer[Self.PipelineType, Self.origin]

    @always_inline
    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn __exit__(mut self):
        pass

    @always_inline
    fn acquire(
        mut self,
    ) -> InputConsumerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire next stage, waiting for tiles to be ready.

        Returns a context manager for processing tiles.
        """
        var stage_idx, mbar = self.pipeline_ptr[].acquire_consumer()
        return InputConsumerStage(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, mbar=mbar
        )

    @always_inline
    fn try_acquire(mut self) -> Bool:
        """Non-blocking check if next consumer stage has data.

        Returns:
            True if the stage has data, False if waiting is needed.

        Use with acquire_if_needed() for the try-acquire pattern:
        ```
        var ready = consumer.try_acquire()
        # ... do other work ...
        with consumer.acquire_if_needed(ready) as tiles:
            process_tiles()
        ```
        """
        return self.pipeline_ptr[].try_acquire_consumer()

    @always_inline
    fn acquire_if_needed(
        mut self, already_ready: Bool
    ) -> InputConsumerStage[
        Self.origin, Self.Payload, Self.num_group_stages, Self.k_group_size
    ]:
        """Acquire stage, only waiting if not already ready.

        Args:
            already_ready: Result from try_acquire(). Skips wait if True.

        Returns:
            The consumer stage for processing tiles.
        """
        self.pipeline_ptr[].wait_producer_if_needed(already_ready)
        var stage_idx = self.pipeline_ptr[].consumer_stage()
        var mbar = self.pipeline_ptr[].consumer_mbar(stage_idx)
        return InputConsumerStage(
            pipeline_ptr=self.pipeline_ptr, stage=stage_idx, mbar=mbar
        )


# ============================================================================
# TilePipeline - Staged tile storage with producer-consumer synchronization
# ============================================================================


struct TilePipeline[
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Staged tile storage with producer-consumer synchronization for SM100.

    Manages a fixed set of pipeline stages (not a FIFO queue) where:
    - Producer (TMA Load) fills tiles into the current stage
    - Consumer (MMA) reads tiles from the current stage
    - Barriers coordinate access between producer and consumer

    Template Parameters:
        a_type: Data type for A matrix tiles.
        b_type: Data type for B matrix tiles.
        a_dim0: First dimension for A tiles.
        a_dim1: Second dimension for A tiles.
        b_dim0: First dimension for B tiles.
        b_dim1: Second dimension for B tiles.
        num_pipeline_stages: Total number of tile stages (stages * k_group_size).
        num_group_stages: Number of synchronization stages.
        k_group_size: Number of tiles per synchronization stage.
    """

    comptime Pipeline = ProducerConsumerPipeline[Self.num_group_stages]
    comptime ATileArray = SMemTileArray2D[
        Self.a_type, Self.a_dim0, Self.a_dim1, Self.num_pipeline_stages, 128
    ]
    comptime BTileArray = SMemTileArray2D[
        Self.b_type, Self.b_dim0, Self.b_dim1, Self.num_pipeline_stages, 128
    ]
    comptime ATile = Self.ATileArray.Tile
    comptime BTile = Self.BTileArray.Tile

    var pipeline: Self.Pipeline
    var a_tiles: Self.ATileArray
    var b_tiles: Self.BTileArray

    @staticmethod
    @always_inline
    fn init_barriers(
        storage_ptr: MbarPtr,
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize pipeline barriers. Called once by elect_one thread."""
        var pipeline = Self.Pipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    comptime BarrierArray = SMemArray[
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
    ](ref[origin] self) -> StandardTileProducer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get producer view for TMA Load warp."""
        return StandardTileProducer(pipeline_ptr=Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin
    ](ref[origin] self) -> StandardTileConsumer[
        origin,
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Get consumer view for MMA warp."""
        return StandardTileConsumer(pipeline_ptr=Pointer(to=self))

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


struct StandardProducerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Context manager for producer tile access with encapsulated stage indexing.
    """

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
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
        var idx = self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        return (self._a_tiles[idx], self._b_tiles[idx])

    @always_inline
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[
            self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        ]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[
            self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        ]

    @always_inline
    fn expect_bytes(self, num_bytes: Int):
        """Set expected bytes on the barrier for TMA loads."""
        self._barrier[0].expect_bytes(Int32(num_bytes))

    @always_inline
    fn barrier(self) -> MbarPtr:
        """Get the barrier pointer for TMA multicast loads."""
        return self._barrier

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage


struct StandardConsumerStage[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Context manager for consumer tile access with encapsulated stage indexing.
    """

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
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
        var idx = self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        return (self._a_tiles[idx], self._b_tiles[idx])

    @always_inline
    fn get_a_tile(self, k_idx: Int) -> Self.ATile:
        """Get A tile at the specified k-group index."""
        return self._a_tiles[
            self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        ]

    @always_inline
    fn get_b_tile(self, k_idx: Int) -> Self.BTile:
        """Get B tile at the specified k-group index."""
        return self._b_tiles[
            self._stage * UInt32(Self.k_group_size) + UInt32(k_idx)
        ]

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the barrier pointer for MMA commit."""
        return self._mbar

    @always_inline
    fn stage(self) -> UInt32:
        """Get the current stage index."""
        return self._stage


@fieldwise_init
struct StandardTileProducer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Producer view for TMA Load warp (standard tile pipeline)."""

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
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

        comptime for _ in range(Self.num_group_stages):
            self.pipeline_ptr[].pipeline.wait_consumer()
            self.pipeline_ptr[].pipeline.producer_step()

    @always_inline
    fn acquire(
        mut self,
    ) -> StandardProducerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for slot availability."""
        var stage, barrier, a_tiles, b_tiles = (
            self.pipeline_ptr[]._acquire_producer_stage()
        )
        return StandardProducerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            barrier=barrier,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


@fieldwise_init
struct StandardTileConsumer[
    origin: MutOrigin,
    a_type: DType,
    b_type: DType,
    # A tile dimensions
    a_dim0: Int,
    a_dim1: Int,
    # B tile dimensions
    b_dim0: Int,
    b_dim1: Int,
    # Pipeline stages
    num_pipeline_stages: Int,
    num_group_stages: Int,
    k_group_size: Int,
](TrivialRegisterPassable):
    """Consumer view for MMA warp (standard tile pipeline)."""

    comptime TilePipelineType = TilePipeline[
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
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
    ) -> StandardConsumerStage[
        Self.origin,
        Self.a_type,
        Self.b_type,
        Self.a_dim0,
        Self.a_dim1,
        Self.b_dim0,
        Self.b_dim1,
        Self.num_pipeline_stages,
        Self.num_group_stages,
        Self.k_group_size,
    ]:
        """Acquire next stage, waiting for tiles to be ready."""
        var stage, mbar, a_tiles, b_tiles = (
            self.pipeline_ptr[]._acquire_consumer_stage()
        )
        return StandardConsumerStage(
            pipeline_ptr=self.pipeline_ptr,
            stage=stage,
            mbar=mbar,
            a_tiles=a_tiles,
            b_tiles=b_tiles,
        )


struct OutputStage[
    num_stages: Int,
    stage_stride: Int,
    cta_group: Int,
](TrivialRegisterPassable):
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

    @staticmethod
    @always_inline
    fn from_raw(
        pipeline: Self.Pipeline,
        stage_index: UInt32,
        tmem_offset: UInt32,
    ) -> Self:
        """Create OutputStage from raw pipeline, stage index, and TMEM offset.

        Useful when not using OutputTilePipeline's consumer() context manager.

        Args:
            pipeline: The ProducerConsumerPipeline for barrier signaling.
            stage_index: Current pipeline stage index.
            tmem_offset: Pre-computed TMEM offset for this stage.

        Returns:
            OutputStage with the given parameters.
        """
        var tmem = Self.Tmem.from_offset(Int(tmem_offset), Int(stage_index))
        return Self(stage_index, tmem, pipeline)


struct OutputTilePipeline[
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
    """Pipeline for MMA→Epilogue TMEM stage synchronization."""

    comptime Pipeline = ProducerConsumerPipeline[Self.num_stages]
    comptime BarrierArray = SMemArray[SharedMemBarrier, Self.num_stages * 2]
    comptime Tmem = TmemAllocation[Self.cta_group]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline: Self.Pipeline
    var tmem: Self.Tmem
    var mma_complete_mask: UInt16

    @staticmethod
    @always_inline
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
        from gpu.primitives.cluster import elect_one_sync
        from gpu.compute.arch.mma_nvidia_sm100 import (
            mma_arrive,
            mma_arrive_multicast,
        )

        if elect_one_sync():
            comptime if Self.cta_group == 1:
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
    ](ref[origin] self) -> OutputProducer[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get producer view for MMA warp."""
        return OutputProducer(Pointer(to=self))

    @always_inline
    fn consumer[
        origin: MutOrigin, //
    ](ref[origin] self) -> OutputConsumer[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get consumer view for epilogue warp."""
        return OutputConsumer(Pointer(to=self))

    # =========================================================================
    # Linear Type API - Compiler-enforced resource management
    # =========================================================================

    @always_inline
    fn acquire_mma_linear[
        origin: MutOrigin, //
    ](ref[origin] self) -> MmaStage[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Acquire a stage for MMA using linear types.

        Waits for the epilogue to free the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = output_pipeline.acquire_mma_linear()
            mma_op.mma(a_tile, b_tile, stage.tmem_offset())
            mma_op.commit(stage.mbar())
            stage^.release()  # Signals mma_arrive and advances

        Returns:
            An MmaStage handle that must be released.
        """
        var stage = self.acquire_for_mma()
        return MmaStage(Pointer(to=self), stage)

    @always_inline
    fn acquire_epilogue_linear[
        origin: MutOrigin, //
    ](ref[origin] self) -> EpilogueStage[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Acquire a stage for epilogue using linear types.

        Waits for MMA to complete the current stage, then returns a
        linear type handle that MUST be released (compiler-enforced).

        Usage:
            var stage = output_pipeline.acquire_epilogue_linear()
            process_tmem(stage.tmem())
            stage^.release()  # Advances to next stage

        Returns:
            An EpilogueStage handle that must be released.
        """
        var stage = self.acquire_for_epilogue()
        return EpilogueStage(Pointer(to=self), stage)

    @always_inline
    fn get_pipeline(self) -> Self.Pipeline:
        """Get underlying pipeline (used during barrier initialization)."""
        return self.pipeline

    @always_inline
    fn per_k[
        origin: MutOrigin, //
    ](ref[origin] self) -> OutputKPipeline[
        origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get per-K-iteration view for kernels with per-K signaling.

        Unlike producer()/consumer() which signal once per tile (after all K
        iterations), this view signals after each K iteration. Use for kernels
        with per-K accumulation patterns (e.g., blockwise FP8).

        Returns:
            OutputKPipeline view that provides produce()/consume() context
            managers for per-K-iteration barrier signaling.
        """
        return OutputKPipeline(Pointer(to=self))

    @always_inline
    fn per_k_epilogue[
        output_origin: MutOrigin,
        input_origin: MutOrigin,
        num_input_stages: Int,
    ](
        ref[output_origin] self,
        ref[input_origin] input_pipeline: ProducerConsumerPipeline[
            num_input_stages
        ],
    ) -> EpilogueKContext[
        output_origin,
        input_origin,
        Self.num_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        num_input_stages,
    ]:
        """Get combined per-K epilogue context for blockwise FP8.

        Bundles output pipeline (MMA->Epilogue sync) and input pipeline
        (A-scales consumption) into a single context manager.

        Example:
            for k_iter in range(num_iters):
                with output_pipeline.per_k_epilogue(input_pipeline) as stage:
                    accum.promote(stage, ...)
                # Both pipelines signaled automatically

        Args:
            input_pipeline: The input pipeline for A-scales consumption.

        Returns:
            EpilogueKContext context manager that handles both pipelines.
        """
        return EpilogueKContext(Pointer(to=self), Pointer(to=input_pipeline))


struct OutputProducer[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
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


struct OutputConsumer[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
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


# =============================================================================
# Unified Linear Types for OutputTilePipeline
# =============================================================================
#
# These types work both as linear types (direct use) and within context managers.
#
# Linear Type API (flat):
#     var stage = output_pipeline.acquire_mma_linear()
#     mma_op.mma(a_tile, b_tile, stage.tmem_offset())
#     stage^.release()
#
# Context Manager API (scoped):
#     with output_pipeline.producer() as stage:
#         mma_op.mma(a_tile, b_tile, stage.tmem.offset())
#


@explicit_destroy("Must call release() to signal MMA completion and advance")
struct MmaStage[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Unified linear type handle for MMA stage in output pipeline.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `output_pipeline.acquire_mma_linear()` - waits for epilogue
    2. Use `tmem()`, `tmem_offset()`, `mbar()` for MMA operations
    3. Must call `release()` to signal mma_arrive and advance (compiler-enforced)

    Parameters:
        origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        stage_stride_cols: TMEM column stride between stages.
        cta_group: CTA group size (1 or 2).
    """

    comptime TilePipelineType = OutputTilePipeline[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: Self.Stage

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: Self.Stage,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage

    @always_inline
    fn tmem(self) -> Self.Stage.Tmem:
        """Get the TMEM stage handle."""
        return self._stage.tmem

    @always_inline
    fn tmem_offset(self) -> Int:
        """Get the TMEM offset for MMA accumulator."""
        return self._stage.tmem.offset()

    @always_inline
    fn index(self) -> UInt32:
        """Get the current stage index."""
        return self._stage.index

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the producer barrier for MMA commit."""
        return self.pipeline_ptr[].pipeline.producer_mbar(self._stage.index)

    @always_inline
    fn release(deinit self):
        """Signal MMA completion and advance to next stage.

        This is the only way to destroy this linear type.
        Internally calls mma_arrive (1-SM) or mma_arrive_multicast (2-SM).
        """
        self.pipeline_ptr[].release_from_mma(self._stage)


@explicit_destroy("Must call release() to free stage for MMA reuse")
struct EpilogueStage[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
]:
    """Unified linear type handle for epilogue stage in output pipeline.

    Works as both a linear type (direct use) and within context managers.

    Lifecycle:
    1. Created via `output_pipeline.acquire_epilogue_linear()` - waits for MMA
    2. Use `tmem()`, `tmem_offset()` for reading MMA results
    3. Must call `release()` to advance (compiler-enforced)

    Parameters:
        origin: Origin of the pipeline reference.
        num_stages: Number of pipeline stages.
        stage_stride_cols: TMEM column stride between stages.
        cta_group: CTA group size (1 or 2).
    """

    comptime TilePipelineType = OutputTilePipeline[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime Stage = OutputStage[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    var _stage: Self.Stage

    @always_inline
    fn __init__(
        out self,
        pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin],
        stage: Self.Stage,
    ):
        self.pipeline_ptr = pipeline_ptr
        self._stage = stage

    @always_inline
    fn tmem(self) -> Self.Stage.Tmem:
        """Get the TMEM stage handle."""
        return self._stage.tmem

    @always_inline
    fn tmem_offset(self) -> Int:
        """Get the TMEM offset for reading MMA results."""
        return self._stage.tmem.offset()

    @always_inline
    fn index(self) -> UInt32:
        """Get the current stage index."""
        return self._stage.index

    @always_inline
    fn release(deinit self):
        """Free stage for MMA reuse and advance to next stage.

        This is the only way to destroy this linear type.
        """
        self.pipeline_ptr[].release_from_epilogue()


# =============================================================================
# Per-K Output Pipeline Views
# =============================================================================
# These types provide per-K-iteration signaling for kernels that need to
# signal after each K iteration rather than once per tile (e.g., blockwise FP8).


struct OutputKPipeline[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
    """Per-K-iteration view of OutputTilePipeline.

    Unlike standard producer()/consumer() which signal once per tile (after
    all K iterations), this view signals after each K iteration. Use for
    kernels with per-K accumulation patterns (e.g., blockwise FP8).

    Example (MMA warp):
        for i in range(num_iters):
            with mma_ctx.output_pipeline.per_k().produce() as stage:
                mma(stage.tmem, ...)
            # __exit__ signals mma_arrive for this K iteration

    Example (Epilogue warp):
        for k_iter in range(num_iters):
            with epi_ctx.output_pipeline.per_k().consume() as stage:
                promote(stage.tmem, ...)
            # __exit__ signals consumer_step for this K iteration
    """

    comptime TilePipelineType = OutputTilePipeline[
        Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]

    var pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]

    @always_inline
    fn __init__(
        out self, pipeline_ptr: Pointer[Self.TilePipelineType, Self.origin]
    ):
        self.pipeline_ptr = pipeline_ptr

    @always_inline
    fn produce(
        self,
    ) -> MmaKStage[
        Self.origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get MMA stage context manager for one K iteration.

        Returns:
            Context manager that acquires stage on enter and signals
            mma_arrive on exit.
        """
        return MmaKStage(self.pipeline_ptr)

    @always_inline
    fn consume(
        self,
    ) -> PerKConsumerStage[
        Self.origin, Self.num_stages, Self.stage_stride_cols, Self.cta_group
    ]:
        """Get consumer context manager for one K iteration.

        Returns:
            Context manager that waits for MMA on enter and signals
            consumer_step on exit.
        """
        return PerKConsumerStage(self.pipeline_ptr)


struct MmaKStage[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
    """Per-K stage context for MMA warp in blockwise FP8.

    __enter__: Acquires stage, waits for epilogue to release previous stage
    __exit__: Signals mma_arrive to notify epilogue, advances producer stage
    """

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


struct PerKConsumerStage[
    origin: MutOrigin,
    num_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
](TrivialRegisterPassable):
    """Context manager for per-K epilogue consumption.

    __enter__: Acquires stage, waits for MMA to complete this K iteration
    __exit__: Signals consumer barrier to release stage for MMA reuse

    IMPORTANT: Unlike standard per-tile consumption, per-K consumption must
    signal the consumer barrier explicitly. The MMA warp waits on this barrier
    before each K iteration, so we must signal after each K iteration.
    """

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
        self.stage = self.pipeline_ptr[].acquire_for_epilogue()
        return self.stage

    @always_inline
    fn __exit__(mut self):
        # Signal the consumer barrier to tell MMA we're done with this stage.
        # This is critical for per-K synchronization - MMA waits on this
        # barrier before each K iteration.
        from gpu.sync import mbarrier_arrive, umma_arrive_leader_cta

        comptime if Self.cta_group == 1:
            _ = mbarrier_arrive(
                self.pipeline_ptr[].pipeline.consumer_mbar(self.stage.index)
            )
        else:
            umma_arrive_leader_cta(
                self.pipeline_ptr[].pipeline.consumer_mbar(self.stage.index)
            )

        self.pipeline_ptr[].release_from_epilogue()


# =============================================================================
# Epilogue Per-K Stage (for blockwise FP8)
# =============================================================================


struct EpilogueKStage[
    num_output_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    num_input_stages: Int,
](TrivialRegisterPassable):
    """Per-K stage for epilogue warp in blockwise FP8.

    Returned from `EpilogueKContext.__enter__()`. Bundles:
    - output_stage: TMEM access (offset for reading MMA results)
    - input_stage_index: Current A-scales stage
    - input_pipeline: For signaling A-scales consumption
    """

    comptime OutputStageType = OutputStage[
        Self.num_output_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime InputPipelineType = ProducerConsumerPipeline[Self.num_input_stages]

    var output_stage: Self.OutputStageType
    var input_stage_index: UInt32
    var input_pipeline: Self.InputPipelineType

    @always_inline
    fn __init__(
        out self,
        output_stage: Self.OutputStageType,
        input_stage_index: UInt32,
        input_pipeline: Self.InputPipelineType,
    ):
        self.output_stage = output_stage
        self.input_stage_index = input_stage_index
        self.input_pipeline = input_pipeline

    @always_inline
    fn arrive_input(self):
        """Arrive on the input pipeline's consumer barrier.

        Use with lane-guarded patterns:
            if lane_id() < cluster_size:
                epi_stage.arrive_input()
        """
        _ = self.input_pipeline.consumer_mbar(self.input_stage_index)[
            0
        ].arrive()


# =============================================================================
# Epilogue Per-K Context Manager (for blockwise FP8)
# =============================================================================


struct EpilogueKContext[
    origin: MutOrigin,
    input_origin: MutOrigin,
    num_output_stages: Int,
    stage_stride_cols: Int,
    cta_group: Int,
    num_input_stages: Int,
](TrivialRegisterPassable):
    """Per-K context manager for epilogue warp in blockwise FP8.

    Bundles output pipeline (MMA→Epilogue sync) and input pipeline (A-scales)
    into a single context manager for clean per-K iteration handling.

    Example usage:
        for k_iter in range(num_iters):
            with epi_ctx.per_k_stage(input_pipeline) as epi_stage:
                accum.promote(epi_stage, ...)
            # __exit__ signals BOTH pipelines

    __enter__: Waits for MMA to complete this K iteration, returns EpilogueKStage
    __exit__: Signals both output consumer barrier AND input consumer_step
    """

    comptime OutputPipelineType = OutputTilePipeline[
        Self.num_output_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime OutputStageType = OutputStage[
        Self.num_output_stages, Self.stage_stride_cols, Self.cta_group
    ]
    comptime InputPipelineType = ProducerConsumerPipeline[Self.num_input_stages]

    # Combined stage type returned from __enter__
    comptime CombinedStageType = EpilogueKStage[
        Self.num_output_stages,
        Self.stage_stride_cols,
        Self.cta_group,
        Self.num_input_stages,
    ]

    var output_pipeline_ptr: Pointer[Self.OutputPipelineType, Self.origin]
    var input_pipeline_ptr: Pointer[Self.InputPipelineType, Self.input_origin]
    var output_stage: Self.OutputStageType
    var input_stage_index: UInt32

    @always_inline
    fn __init__(
        out self,
        output_pipeline_ptr: Pointer[Self.OutputPipelineType, Self.origin],
        input_pipeline_ptr: Pointer[Self.InputPipelineType, Self.input_origin],
    ):
        self.output_pipeline_ptr = output_pipeline_ptr
        self.input_pipeline_ptr = input_pipeline_ptr
        self.input_stage_index = 0
        # Placeholder stage - set properly in __enter__
        var placeholder_tmem = Self.OutputStageType.Tmem(0, 0)
        self.output_stage = Self.OutputStageType(
            0,
            placeholder_tmem,
            ProducerConsumerPipeline[Self.num_output_stages](MbarPtr()),
        )

    @always_inline
    fn __enter__(mut self) -> Self.CombinedStageType:
        self.output_stage = self.output_pipeline_ptr[].acquire_for_epilogue()
        self.input_stage_index = self.input_pipeline_ptr[].consumer_stage()
        return Self.CombinedStageType(
            self.output_stage, self.input_stage_index, self.input_pipeline_ptr[]
        )

    @always_inline
    fn __exit__(mut self):
        # Signal input pipeline consumer_step (for A-scales consumption)
        self.input_pipeline_ptr[].consumer_step()

        # Signal output pipeline consumer barrier (for MMA synchronization)
        from gpu.sync import mbarrier_arrive, umma_arrive_leader_cta

        comptime if Self.cta_group == 1:
            _ = mbarrier_arrive(
                self.output_pipeline_ptr[].pipeline.consumer_mbar(
                    self.output_stage.index
                )
            )
        else:
            umma_arrive_leader_cta(
                self.output_pipeline_ptr[].pipeline.consumer_mbar(
                    self.output_stage.index
                )
            )

        self.output_pipeline_ptr[].release_from_epilogue()
