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

"""Ring buffer implementation for producer-consumer synchronization in GPU kernels.

This module provides a ring buffer abstraction that enables efficient overlap of
memory transfers and computation in matrix multiplication kernels. The pattern
divides work between:

- Producer: One warp group that loads tiles from global to shared memory
- Consumers: Multiple warp groups that process tiles using tensor cores

The ring buffer uses barrier synchronization to coordinate access to a circular
queue of tile buffers, allowing the producer to work ahead while consumers process
previously loaded data.

Usage Example:
    # Create ring buffer during kernel initialization
    var ring_buffer = RingBuffer[...](full_mbar, empty_mbar, ...)

    # Producer pattern
    with ring_buffer.producer() as producer:
        while has_work():
            with producer.get_tiles() as tiles:
                # Load data into tiles.a_tile and tiles.b_tile
                load_tile(tiles.a_tile, tiles.barrier)

    # Consumer pattern
    with ring_buffer.consumer() as consumer:
        while has_work():
            with consumer.get_tiles() as tiles:
                # Process tiles.a_tile and tiles.b_tile
                gemm(tiles.a_tile, tiles.b_tile, output)
"""
from layout.tma_async import PipelineState
from ....structuring import SMemBarrier
from gpu.sync import async_copy_arrive
from gpu.globals import WARPGROUP_SIZE
from gpu.memory import AddressSpace
from ....structuring import NVIDIASharedMemoryManager as SharedMemoryManager
from ....structuring import SMemBarrier


# Ring buffer abstraction for producer-consumer synchronization
# This module implements a multi-stage pipeline where one producer warp group
# loads tiles into shared memory while multiple consumer warp groups process them


@register_passable("trivial")
struct ProducerTiles[
    origin: Origin[True],
    ring_buffer_type: __type_of(RingBuffer),
]:
    """Context manager for producer access to ring buffer tiles.

    This struct provides safe access to a single tile slot in the ring buffer
    for the producer to fill. It automatically handles barrier synchronization
    when entering and exiting the context.
    """

    alias ATileType = ring_buffer_type.ATileType
    alias BTileType = ring_buffer_type.BTileType
    alias RingBufferPtrType = Pointer[Self.ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType

    var barrier: SMemBarrier
    var a_tile: Self.ATileType
    var b_tile: Self.BTileType

    fn __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr
        # Get the next available slot and its associated tiles
        self.barrier, self.a_tile, self.b_tile = (
            self.ring_buffer_ptr[].get_producer_tiles()
        )

    fn __enter__(mut self) -> Self:
        return self

    fn __exit__(mut self):
        # Signal that the tile has been filled and is ready for consumption
        self.ring_buffer_ptr[].enqueue_tile()


@register_passable("trivial")
struct ConsumerTiles[
    origin: Origin[True],
    ring_buffer_type: __type_of(RingBuffer),
]:
    """Context manager for consumer access to ring buffer tiles.

    This struct provides safe access to a single tile slot in the ring buffer
    for consumers to read. It tracks the read index and automatically releases
    the slot when exiting the context.
    """

    alias ATileType = ring_buffer_type.ATileType
    alias BTileType = ring_buffer_type.BTileType
    alias RingBufferPtrType = Pointer[Self.ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType

    var read_idx: UInt32
    var a_tile: Self.ATileType
    var b_tile: Self.BTileType

    fn __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr
        # Wait for a full slot and get its tiles
        self.read_idx, self.a_tile, self.b_tile = (
            self.ring_buffer_ptr[].get_consumer_tiles()
        )

    fn __enter__(mut self) -> Self:
        return self

    fn __exit__(mut self):
        # Signal that we're done with this slot, making it available for reuse
        self.ring_buffer_ptr[].release_slot(self.read_idx)


@register_passable("trivial")
struct RingBufferConsumer[
    origin: Origin[True],
    ring_buffer_type: __type_of(RingBuffer),
]:
    """Consumer view of the ring buffer.

    This struct provides the consumer interface to the ring buffer, allowing
    consumers to wait for and access tiles loaded by the producer. It handles
    the initial barrier arrival when entering the consumer context.
    """

    alias ATileType = ring_buffer_type.ATileType
    alias BTileType = ring_buffer_type.BTileType
    alias RingBufferPtrType = Pointer[Self.ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType

    fn __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr

    fn __enter__(mut self) -> Self:
        # Arrive at all empty barriers to signal consumer readiness
        self.ring_buffer_ptr[].arrive_empty_barriers()
        return self

    @always_inline
    fn get_tiles(
        mut self,
    ) -> ConsumerTiles[origin, Self.ring_buffer_type]:
        """Get a context manager for accessing the next available tile."""
        return {self.ring_buffer_ptr}


@register_passable("trivial")
struct RingBufferProducer[
    origin: Origin[True],
    ring_buffer_type: __type_of(RingBuffer),
]:
    """Producer view of the ring buffer.

    This struct provides the producer interface to the ring buffer, allowing
    the producer to wait for empty slots and fill them with new tiles.
    """

    alias ATileType = ring_buffer_type.ATileType
    alias BTileType = ring_buffer_type.BTileType
    alias RingBufferPtrType = Pointer[Self.ring_buffer_type, origin]

    var ring_buffer_ptr: Self.RingBufferPtrType

    fn __init__(out self, ring_buffer_ptr: Self.RingBufferPtrType):
        self.ring_buffer_ptr = ring_buffer_ptr

    fn __enter__(mut self) -> Self:
        return self

    @always_inline
    fn get_tiles(
        mut self,
    ) -> ProducerTiles[origin, Self.ring_buffer_type]:
        """Get a context manager for accessing the next empty tile slot."""
        return {self.ring_buffer_ptr}


struct RingBuffer[
    a_type: DType,
    b_type: DType,
    a_tile_layout: Layout,
    b_tile_layout: Layout,
    num_pipeline_stages: Int,
    num_consumers: Int,
    cluster_size: Int,
    use_async_copy: Bool = False,
](ImplicitlyCopyable):
    """Ring buffer for managing pipeline synchronization between producers and consumers.

    This struct encapsulates the synchronization logic for a multi-stage pipeline
    with one producer and multiple consumers, supporting both single-block and
    multi-cluster configurations.

    The ring buffer uses two sets of barriers:
    - full_mbar: Signals when tiles are ready for consumption
    - empty_mbar: Signals when slots are available for production

    Template Parameters:
        a_type: Data type for A matrix tiles
        b_type: Data type for B matrix tiles
        a_tile_layout: Memory layout for A tiles
        b_tile_layout: Memory layout for B tiles
        num_pipeline_stages: Number of stages in the circular buffer
        num_consumers: Number of consumer warp groups
        cluster_size: Number of blocks in the cluster (1 for single-block)
        use_async_copy: Whether to use cp.async instructions (default: False)
    """

    alias SMM = SharedMemoryManager[]

    # Tile iterator types for managing shared memory tiles
    alias ATileIterType = Self.SMM.TileIter[
        a_type, a_tile_layout, num_pipeline_stages
    ]
    alias BTileIterType = Self.SMM.TileIter[
        b_type, b_tile_layout, num_pipeline_stages
    ]
    # Actual tile tensor types that hold the data
    alias ATileType = Self.ATileIterType.T.LayoutTensorType
    alias BTileType = Self.BTileIterType.T.LayoutTensorType

    # Barriers for synchronization:
    # - full_mbar[i]: Signaled by producer when slot i contains data
    # - empty_mbar[i]: Signaled by consumers when slot i is free
    var full_mbar: SMemBarrier
    var empty_mbar: SMemBarrier

    # Pipeline states track current position and phase in the ring buffer
    var read_state: PipelineState[num_pipeline_stages]  # Consumer's position
    var write_state: PipelineState[num_pipeline_stages]  # Producer's position

    # Thread index within the warp group (0-127 for 4 warps)
    var warp_group_thread_idx: UInt

    # Tile storage arrays in shared memory
    var a_tiles: Self.ATileIterType.T
    var b_tiles: Self.BTileIterType.T

    fn __init__(
        out self,
        full_mbar: SMemBarrier,
        empty_mbar: SMemBarrier,
        warp_group_thread_idx: UInt,
        a_tiles: Self.ATileIterType.T,
        b_tiles: Self.BTileIterType.T,
    ):
        """Initialize ring buffer with barrier pointers.

        Args:
            full_mbar: Barrier array signaling when tiles are ready.
            empty_mbar: Barrier array signaling when slots are empty.
            warp_group_thread_idx: Thread index within the warp group.
            a_tiles: Iterator over A matrix tile storage.
            b_tiles: Iterator over B matrix tile storage.
        """
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.read_state = PipelineState[num_pipeline_stages]()
        self.write_state = PipelineState[num_pipeline_stages]()
        self.warp_group_thread_idx = warp_group_thread_idx
        self.a_tiles = a_tiles
        self.b_tiles = b_tiles

        # Only thread 0 initializes the barriers
        if thread_idx.x == 0:

            @parameter
            for i in range(num_pipeline_stages):
                # Full barrier: expects arrivals from producer threads
                # For async_copy, all threads in warp group participate
                self.full_mbar[i].init(WARPGROUP_SIZE if use_async_copy else 1)
                # Empty barrier: expects arrivals from all consumers across cluster
                self.empty_mbar[i].init(num_consumers * cluster_size)

    fn __enter__(mut self) -> Self:
        """Context manager entry."""
        return self^

    @always_inline
    @staticmethod
    fn get_expected_bytes() -> Int:
        """Calculate expected bytes per pipeline stage for TMA transfers."""
        return (
            Self.ATileIterType.storage_size + Self.BTileIterType.storage_size
        ) // num_pipeline_stages

    @always_inline
    fn get_slot(mut self) -> UInt32:
        """Producer waits for empty buffer slot and prepares for loading.

        This method blocks until the current write slot is empty (all consumers
        have finished with it), then prepares the barrier for new data.

        Returns:
            Index of the available slot in the ring buffer.
        """
        var write_idx = self.write_state.index()
        # Wait for all consumers to signal this slot is empty
        self.empty_mbar[Int(write_idx)].wait(self.write_state.phase())
        # For TMA transfers, set expected bytes for the barrier
        if not use_async_copy:
            alias expected_bytes = Self.get_expected_bytes()
            self.full_mbar[Int(write_idx)].expect_bytes(expected_bytes)
        return write_idx

    @always_inline
    fn get_producer_tiles(
        mut self,
    ) -> Tuple[SMemBarrier, Self.ATileType, Self.BTileType]:
        """Get the next available slot for the producer to fill.

        Returns:
            Tuple of (barrier, a_tile, b_tile) for the producer to use.
        """
        var idx = self.get_slot()
        return (
            self.full_mbar.offset(idx),
            self.a_tiles.next(idx)[],
            self.b_tiles.next(idx)[],
        )

    @always_inline
    fn enqueue_tile(mut self):
        """Producer signals that tile loading is complete.

        This handles the specific signaling pattern needed:
        - For cp.async: Signal async copy arrival and barrier arrival
        - For TMA: Barrier arrival is handled by hardware

        After signaling, advances to the next pipeline stage.
        """
        if use_async_copy:
            var write_idx = self.write_state.index()
            # Signal that async copy operations have been issued
            async_copy_arrive(self.full_mbar[write_idx].unsafe_ptr())
            # Arrive at barrier to signal tile is ready
            _ = self.full_mbar[write_idx].arrive()
        # Move to next slot in the ring buffer
        self.write_state.step()

    @always_inline
    fn get_tile(mut self) -> UInt32:
        """Consumer waits for full buffer slot.

        This method blocks until the producer has filled the current read slot.

        Returns:
            Index of the available tile to consume.
        """
        var read_idx = self.read_state.index()
        # Wait for producer to signal this slot is full
        self.full_mbar[Int(read_idx)].wait(self.read_state.phase())
        return read_idx

    @always_inline
    fn get_consumer_tiles(
        mut self,
    ) -> Tuple[UInt32, Self.ATileType, Self.BTileType]:
        """Consumer waits for full buffer slot and returns the tiles.

        Returns:
            Tuple of (read_idx, a_tile, b_tile) for the consumer to process.
        """
        var read_idx = self.get_tile()
        return (
            read_idx,
            self.a_tiles.next(read_idx)[],
            self.b_tiles.next(read_idx)[],
        )

    @always_inline
    fn release_slot(
        mut self,
        read_idx: UInt32,
    ):
        """Consumer signals that buffer slot is empty.

        This allows the producer to reuse this slot in the ring buffer.
        Different arrival patterns are used for single-block vs multi-cluster.

        Args:
            read_idx: Index of the slot to release.
        """

        @parameter
        if cluster_size > 1:
            # In multi-cluster mode, one thread per block arrives
            if self.warp_group_thread_idx < UInt(cluster_size):
                _ = self.empty_mbar[Int(read_idx)].arrive_cluster(
                    self.warp_group_thread_idx
                )
        else:
            # In single-block mode, only thread 0 arrives
            if self.warp_group_thread_idx == 0:
                _ = self.empty_mbar[Int(read_idx)].arrive()

        # Advance to next pipeline stage
        self.read_state.step()

    @always_inline
    fn consumer(mut self) -> RingBufferConsumer[__origin_of(self), Self]:
        """Create a consumer view of this ring buffer."""
        return {Pointer(to=self)}

    @always_inline
    fn producer(mut self) -> RingBufferProducer[__origin_of(self), Self]:
        """Create a producer view of this ring buffer."""
        return {Pointer(to=self)}

    @always_inline
    fn arrive_empty_barriers(
        self,
    ):
        """Helper to arrive at empty barriers during consumer initialization.

        This is called when consumers enter their context to signal they are
        ready to consume tiles. It ensures all pipeline stages start with
        empty slots available for the producer.
        """

        @parameter
        for i in range(num_pipeline_stages):
            # Same arrival pattern as release_slot for consistency
            @parameter
            if cluster_size > 1:
                if self.warp_group_thread_idx < UInt(cluster_size):
                    _ = self.empty_mbar[i].arrive_cluster(
                        self.warp_group_thread_idx
                    )
            else:
                if self.warp_group_thread_idx == 0:
                    _ = self.empty_mbar[i].arrive()
