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

from gpu import warp_id as get_warp_id
from layout import Layout
from linalg.structuring import SMemArrayType
from utils import StaticTuple

from .structured import AMDSharedMemoryBarrier, SMemBuffer


@register_passable("trivial")
struct ProducerTile[
    origin: Origin[True],
    ring_buffer_type: type_of(RingBuffer),
    warps_processed_per_producer: Int,
    producer_warps: Int,
]:
    """Context manager for producer access to a single ring buffer tile."""

    alias ProducerViewType = RingBufferProducer[
        origin, ring_buffer_type, warps_processed_per_producer, producer_warps
    ]
    alias ProducerViewPtrType = Pointer[Self.ProducerViewType, origin]

    var producer_view_ptr: Self.ProducerViewPtrType
    var tile: ring_buffer_type.SmemBufferType.WarpTileType
    var stage: Int
    var warp_tile_idx: Int

    @always_inline
    fn __init__(
        out self,
        producer_view_ptr: Self.ProducerViewPtrType,
        stage: Int,
        warp_tile_idx: Int,
    ):
        self.producer_view_ptr = producer_view_ptr
        self.stage = stage
        self.warp_tile_idx = warp_tile_idx
        # Acquire the tile
        self.tile = self.producer_view_ptr[].get_tile(stage, warp_tile_idx)

    @always_inline
    fn __enter__(mut self) -> ring_buffer_type.SmemBufferType.WarpTileType:
        """Return the acquired tile for use."""
        return self.tile

    @always_inline
    fn __exit__(mut self):
        """Release the tile back to consumers."""
        self.producer_view_ptr[].release_tile(self.stage, self.warp_tile_idx)


@register_passable("trivial")
struct ConsumerTile[
    origin: Origin[True],
    ring_buffer_type: type_of(RingBuffer),
    warps_computed_per_consumer: Int,
    num_consumer_warps: Int,
]:
    """Context manager for consumer access to a single ring buffer tile."""

    alias ConsumerViewType = RingBufferConsumer[
        origin,
        ring_buffer_type,
        warps_computed_per_consumer,
        num_consumer_warps,
    ]
    alias ConsumerViewPtrType = Pointer[Self.ConsumerViewType, origin]

    var consumer_view_ptr: Self.ConsumerViewPtrType
    var tile: ring_buffer_type.SmemBufferType.WarpTileType
    var stage: Int
    var warp_tile_idx: Int

    @always_inline
    fn __init__(
        out self,
        consumer_view_ptr: Self.ConsumerViewPtrType,
        stage: Int,
        local_tile_count: Int,
        warp_tile_idx: Int,
    ):
        self.consumer_view_ptr = consumer_view_ptr
        self.stage = stage
        self.warp_tile_idx = warp_tile_idx
        # Acquire the tile
        self.tile = self.consumer_view_ptr[].get_tile(
            stage, local_tile_count, warp_tile_idx
        )

    @always_inline
    fn __enter__(mut self) -> ring_buffer_type.SmemBufferType.WarpTileType:
        """Return the acquired tile for use."""
        return self.tile

    @always_inline
    fn __exit__(mut self):
        """Release the tile back to producers."""
        self.consumer_view_ptr[].release_tile(self.stage, self.warp_tile_idx)


@register_passable("trivial")
struct RingBufferProducer[
    origin: Origin[True],
    ring_buffer_type: type_of(RingBuffer),
    warps_processed_per_producer: Int,
    producer_warps: Int,
]:
    """Producer view of the ring buffer with phase management."""

    alias RingBufferPtrType = Pointer[ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType
    var phases: StaticTuple[
        Int32, ring_buffer_type.pipeline_stages * warps_processed_per_producer
    ]

    @always_inline
    fn __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr
        self.phases = StaticTuple[
            Int32,
            ring_buffer_type.pipeline_stages * warps_processed_per_producer,
        ](fill=0)

    @always_inline
    fn __enter__(mut self) -> Self:
        """Context manager entry - no special initialization needed for producer.
        """
        return self

    @always_inline
    fn __exit__(mut self):
        """Context manager exit - no cleanup needed."""
        pass

    @always_inline
    fn get_tile(
        mut self,
        stage: Int,
        warp_tile_idx: Int,
    ) -> ring_buffer_type.SmemBufferType.WarpTileType:
        # Compute phase_idx directly from stage and warp_tile_idx
        var phase_idx = stage * warps_processed_per_producer + (
            warp_tile_idx // producer_warps
        )
        return self.ring_buffer_ptr[].get_tile(
            self.phases[phase_idx], stage, warp_tile_idx
        )

    @always_inline
    fn release_tile(mut self, stage: Int, warp_tile_idx: Int):
        self.ring_buffer_ptr[].release_tile(stage, warp_tile_idx)

    alias ProducerTileType = ProducerTile[
        origin, ring_buffer_type, warps_processed_per_producer, producer_warps
    ]

    @always_inline
    fn acquire_tile(
        mut self,
        stage: Int,
        warp_tile_idx: Int,
    ) -> Self.ProducerTileType:
        """Get a context manager for accessing a tile."""
        return Self.ProducerTileType(
            rebind[Pointer[Self, origin]](Pointer(to=self)),
            stage,
            warp_tile_idx,
        )


@register_passable("trivial")
struct RingBufferConsumer[
    origin: Origin[True],
    ring_buffer_type: type_of(RingBuffer),
    warps_computed_per_consumer: Int,
    num_consumer_warps: Int,
]:
    """Consumer view of the ring buffer with phase management."""

    alias RingBufferPtrType = Pointer[ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType
    var phases: StaticTuple[
        Int32, ring_buffer_type.pipeline_stages * warps_computed_per_consumer
    ]
    var consumer_warp_id: Int

    @always_inline
    fn __init__(
        out self, ring_buffer_ptr: Self.RingBufferPtrType, consumer_warp_id: Int
    ):
        self.ring_buffer_ptr = ring_buffer_ptr
        self.phases = StaticTuple[
            Int32,
            ring_buffer_type.pipeline_stages * warps_computed_per_consumer,
        ](fill=1)
        self.consumer_warp_id = consumer_warp_id

    @always_inline
    fn __enter__(mut self) -> Self:
        """Context manager entry for consumer."""
        # Note: AMD version doesn't have arrive_empty_barriers like SM90
        # The synchronization is handled differently via the barrier array
        return self

    @always_inline
    fn __exit__(mut self):
        """Context manager exit - no cleanup needed."""
        pass

    @always_inline
    fn get_tile(
        mut self,
        stage: Int,
        local_tile_count: Int,
        warp_tile_idx: Int,
    ) -> ring_buffer_type.SmemBufferType.WarpTileType:
        # Phase idx is computed from stage and local_tile_count
        var phase_idx = stage * warps_computed_per_consumer + local_tile_count
        return self.ring_buffer_ptr[].get_tile(
            self.phases[phase_idx], stage, warp_tile_idx
        )

    @always_inline
    fn release_tile(mut self, stage: Int, warp_tile_idx: Int):
        self.ring_buffer_ptr[].release_tile(stage, warp_tile_idx)

    alias ConsumerTileType = ConsumerTile[
        origin,
        ring_buffer_type,
        warps_computed_per_consumer,
        num_consumer_warps,
    ]

    @always_inline
    fn acquire_tile(
        mut self,
        stage: Int,
        local_tile_count: Int,
        warp_tile_idx: Int,
    ) -> Self.ConsumerTileType:
        """Get a context manager for accessing a tile."""
        return Self.ConsumerTileType(
            rebind[Pointer[Self, origin]](Pointer(to=self)),
            stage,
            local_tile_count,
            warp_tile_idx,
        )


struct RingBuffer[
    dtype: DType,
    layout: Layout,
    pipeline_stages: Int,
    B_rows: Int,  # BM for A, BN for B
    B_cols: Int,  # BK for both
    W_rows: Int,  # WM for A, WN for B
    W_cols: Int, //,  # WK for both
    consumer_warps: Int,
    reads_per_warp_block: Int,  # how many times the warp block will be read from shared memory
]:

    """Manages access to shared memory tiles using barriers based in shared memory.
    """

    # NOTE: smem can be 3D if pipelined, in that case we need a way to extract
    # the 2D tiles that's what this does

    # The barrier consists of integers. Producers and
    # consumers should wait if the barrier integer value does not fit into their expected range.
    # The rows of the barrier represent the warp tile desired. the columns consist of pipeline stages
    # each with consumer_warps slots. If pipeline_stages is > 1 then shared memory buffering is being used.
    # There are also consumer_warps slots for each pipeline stage, since each warp can write to the barrier
    # at the same time causing race conditions.

    alias writes_per_warp_block = 1  # only one producer writes to one warp tile

    alias block_warps = B_rows // W_rows

    alias BarrierArray = SMemArrayType[
        AMDSharedMemoryBarrier[consumer_warps],
        Self.block_warps * pipeline_stages,
    ]

    var barrier: Self.BarrierArray
    alias SmemBufferType = SMemBuffer[
        dtype, layout, pipeline_stages, B_rows, B_cols, W_rows, W_cols
    ]

    var smem_buffer: Self.SmemBufferType

    @always_inline
    fn __init__(
        out self,
        smem_buffer: Self.SmemBufferType,
    ):
        self.smem_buffer = smem_buffer
        self.barrier = Self.BarrierArray.stack_allocation[alignment=32]()

        @parameter
        for i in range(type_of(self.barrier).size):
            self.barrier[i][].initialize()

    @always_inline
    fn _wait(
        self,
        barrier: Self.BarrierArray,
        phase: Int32,
        tile_idx: Int,
        stage: Int,
    ):
        barrier[
            tile_idx * pipeline_stages + stage
        ][].wait_until_greater_or_equal_to(phase)

    @always_inline
    fn get_tile(
        mut self,
        mut phase: Int32,
        stage: Int,
        tile_idx: Int,
    ) -> Self.SmemBufferType.WarpTileType:
        self._wait(self.barrier, phase, tile_idx, stage)

        phase += Self.writes_per_warp_block + Self.reads_per_warp_block
        var staged_smem_tile = self.smem_buffer.get_tile(stage)
        return staged_smem_tile.tile[W_rows, W_cols](tile_idx, 0)

    @always_inline
    fn release_tile(mut self, stage: Int, tile_idx: Int):
        self.barrier[tile_idx * pipeline_stages + stage][].increment(
            Int(get_warp_id() % UInt(consumer_warps))
        )

    @always_inline
    fn producer[
        warps_processed_per_producer: Int, producer_warps: Int
    ](
        mut self,
    ) -> RingBufferProducer[
        origin_of(self),
        type_of(self),
        warps_processed_per_producer,
        producer_warps,
    ]:
        """Create a producer view of this ring buffer."""
        return RingBufferProducer[
            origin_of(self),
            type_of(self),
            warps_processed_per_producer,
            producer_warps,
        ](Pointer(to=self))

    @always_inline
    fn consumer[
        warps_computed_per_consumer: Int, num_consumer_warps: Int
    ](mut self, consumer_warp_id: Int) -> RingBufferConsumer[
        origin_of(self),
        type_of(self),
        warps_computed_per_consumer,
        num_consumer_warps,
    ]:
        """Create a consumer view of this ring buffer."""
        return RingBufferConsumer[
            origin_of(self),
            type_of(self),
            warps_computed_per_consumer,
            num_consumer_warps,
        ](Pointer(to=self), consumer_warp_id)
