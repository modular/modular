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
"""Producer-consumer pipeline utilities for SM100 structured kernels.

This module provides pipeline synchronization primitives for warp-specialized
GPU kernels, enabling efficient producer-consumer patterns between warps.

Key abstraction:
- ProducerConsumerPipeline: Low-level barrier management for N-stage pipelines

Context manager API (recommended):
    # Producer side (e.g., MMA warp producing to epilogue):
    with pipeline.produce() as stage:
        # stage.index() - current stage index
        # stage.mbar() - barrier for signaling (use with mma_arrive)
        # ... do work ...
        # Must signal via stage.mbar() before exit
    # __exit__ calls producer_step()

    # Consumer side (e.g., epilogue consuming from MMA):
    with pipeline.consume() as stage:
        # stage.index() - current stage index
        # ... do work ...
    # __exit__ signals consumption complete and calls consumer_step()

Direct API (for special cases):
    pipeline.wait_producer() / wait_consumer()
    pipeline.producer_step() / consumer_step()
    pipeline.producer_mbar(stage) / consumer_mbar(stage)
"""

from sys import size_of

from layout.tma_async import SharedMemBarrier
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]


comptime MbarPtr = UnsafePointer[
    SharedMemBarrier, address_space = AddressSpace.SHARED
]


@register_passable("trivial")
struct ProducerConsumerPipeline[num_stages: Int]:
    """A producer-consumer pipeline using shared memory barriers to
    enforce synchronization (between producer and consumer warps).

    Parameters:
        num_stages: The number of pipeline stages.

    This struct is commonly used with warp specialization to pipeline operations
    between two warps/warpgroups with data dependencies.
    """

    # Full implies data has been produced. Producer signals this barrier
    # and consumer waits on this barrier.
    var full: MbarPtr

    # Empty implies data has been consumed. Consumer signals this barrier
    # and producer waits on this barrier.
    var empty: MbarPtr

    # The stage in pipeline, from 0 to num_stages-1
    var _consumer_stage: UInt32
    var _producer_stage: UInt32

    # The phase for shared memory barrier, between 0 and 1
    var _consumer_phase: UInt32
    var _producer_phase: UInt32

    @always_inline
    fn __init__(out self, ptr: MbarPtr):
        """Initialize the producer-consumer pipeline with default phases.

        Args:
            ptr: Pointer to shared memory barriers.
        """
        self.full = ptr
        self.empty = ptr + Self.num_stages
        self._producer_stage = 0
        self._consumer_stage = 0
        # This ensures producer's wait_consumer() passes trivially at
        # the beginning when it tries to initialize data buffer.
        self._producer_phase = 1
        self._consumer_phase = 0

    @always_inline
    fn wait_producer(self):
        """Consumer waits for producer."""
        self.full[self._consumer_stage].wait(self._consumer_phase)

    @always_inline
    fn wait_consumer(self):
        """Producer waits for consumer."""
        self.empty[self._producer_stage].wait(self._producer_phase)

    @always_inline
    fn producer_mbar(self, stage: UInt32) -> MbarPtr:
        """Get the producer barrier for a specific stage.

        Args:
            stage: The pipeline stage.

        Returns:
            The shared memory barrier that the producer signals.
        """
        return self.full + stage

    @always_inline
    fn consumer_mbar(self, stage: UInt32) -> MbarPtr:
        """Get the consumer barrier for a specific stage.

        Args:
            stage: The pipeline stage.

        Returns:
            The shared memory barrier that the consumer signals.
        """
        return self.empty + stage

    @always_inline
    fn producer_stage(self) -> UInt32:
        """Get the current producer stage index.

        Returns:
            The current stage index for the producer (0 to num_stages-1).
        """
        return self._producer_stage

    @always_inline
    fn consumer_stage(self) -> UInt32:
        """Get the current consumer stage index.

        Returns:
            The current stage index for the consumer (0 to num_stages-1).
        """
        return self._consumer_stage

    @always_inline
    fn consumer_step(mut self):
        """Advance the consumer to the next pipeline stage.

        Increments the consumer stage and wraps to 0 when reaching num_stages,
        toggling the phase bit on wrap-around.
        Only switch phase at end of pipeline because we assume all barriers
        are at the same consumer/producer phase before checked. Once checked,
        the execution moves to next barrier.
        """
        self._consumer_stage += 1

        if self._consumer_stage == Self.num_stages:
            self._consumer_stage = 0
            self._consumer_phase ^= 1

    @always_inline
    fn producer_step(mut self):
        """Advance the producer to the next pipeline stage.

        Increments the producer stage and wraps to 0 when reaching num_stages,
        toggling the phase bit on wrap-around.
        """
        self._producer_stage += 1

        if self._producer_stage == Self.num_stages:
            self._producer_stage = 0
            self._producer_phase ^= 1

    @staticmethod
    @always_inline
    fn smem_bytes() -> UInt32:
        """Calculate the shared memory bytes required for pipeline barriers.

        Returns:
            The total number of bytes needed for all pipeline barriers
            (2 * num_stages barriers).
        """
        return 2 * Self.num_stages * size_of[SharedMemBarrier]()

    @always_inline
    fn init_mbars(
        self, producer_arrive_count: Int32, consumer_arrive_count: Int32
    ):
        """
        Initialize the smem barriers for the producer and consumer.

        Args:
            producer_arrive_count: The number of threads that will arrive at the barrier marking data as produced.
            consumer_arrive_count: The number of threads that will arrive at the barrier marking data as consumed.

        This function must be called by a single thread and must be called before any the pipeline object is used.
        """

        @parameter
        for i in range(Self.num_stages):
            self.full[i].init(producer_arrive_count)
            self.empty[i].init(consumer_arrive_count)

    @always_inline
    fn producer_signal_and_step(mut self):
        """Wait for consumer, signal production, and advance stage.

        Combined operation for CLC throttling (Load warp):
        1. Wait for consumer to finish with current stage
        2. Signal that producer has new data
        3. Advance to next stage
        """
        self.wait_consumer()
        _ = self.full[self._producer_stage].arrive()
        self.producer_step()

    @always_inline
    fn consumer_signal_and_step(mut self):
        """Wait for producer, signal consumption, and advance stage.

        Combined operation for CLC throttling (Scheduler warp):
        1. Wait for producer to have data ready
        2. Signal that consumer has consumed data
        3. Advance to next stage
        """
        self.wait_producer()
        _ = self.empty[self._consumer_stage].arrive()
        self.consumer_step()

    # =========================================================================
    # Context Manager API - Encapsulated barrier operations
    # =========================================================================

    @always_inline
    fn produce[
        origin: MutOrigin, //
    ](ref [origin]self) -> ProduceContext[origin, Self.num_stages]:
        """Produce one pipeline stage with encapsulated barriers.

        Usage:
            with pipeline.produce() as stage:
                # stage.index() gives current stage
                # stage.mbar() gives barrier for signaling
                # __exit__ calls producer_step()

        Returns:
            Context that waits for consumer on enter, advances on exit.
        """
        return ProduceContext(Pointer(to=self))

    @always_inline
    fn consume[
        origin: MutOrigin, //
    ](ref [origin]self) -> ConsumeContext[origin, Self.num_stages]:
        """Consume one pipeline stage with encapsulated barriers.

        Usage:
            with pipeline.consume() as stage:
                # stage.index() gives current stage
                # __exit__ signals consumer done and advances

        Returns:
            Context that waits for producer on enter, signals+advances on exit.
        """
        return ConsumeContext(Pointer(to=self))

    @always_inline
    fn consume_explicit[
        origin: MutOrigin, //
    ](ref [origin]self) -> ExplicitConsumeContext[origin, Self.num_stages]:
        """Consume one pipeline stage with EXPLICIT barrier arrive.

        Use this for kernels requiring lane-guarded or specialized signaling.

        Usage:
            with pipeline.consume_explicit() as stage:
                # ... do work ...
                if lane_id() < CLUSTER_SIZE:
                    stage.arrive()  # Lane-guarded arrive
            # __exit__ only advances, does NOT arrive

        For specialized signaling (e.g., umma_arrive_leader_cta):
            with pipeline.consume_explicit() as stage:
                if cta_group == 1:
                    stage.arrive()
                else:
                    umma_arrive_leader_cta(stage.mbar())

        Returns:
            Context that waits for producer on enter, advances only on exit.
        """
        return ExplicitConsumeContext(Pointer(to=self))


# =============================================================================
# Context Managers for ProducerConsumerPipeline
# =============================================================================


@register_passable("trivial")
struct ProducerStage:
    """Stage info returned by ProduceContext.__enter__."""

    var _index: UInt32
    var _mbar: MbarPtr

    @always_inline
    fn __init__(out self, index: UInt32, mbar: MbarPtr):
        self._index = index
        self._mbar = mbar

    @always_inline
    fn index(self) -> UInt32:
        """Get the current stage index."""
        return self._index

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the barrier to signal when production is complete.

        Caller is responsible for signaling via mma_arrive or similar.
        """
        return self._mbar


@register_passable("trivial")
struct ProduceContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
]:
    """Context for producing one pipeline stage.

    - __enter__: Waits for consumer to be ready, returns stage info
    - __exit__: Advances producer to next stage

    Note: The actual production signal (mma_arrive) is kernel-specific
    and must be called by the user before exiting the context.
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
    ]

    @always_inline
    fn __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
        ],
    ):
        self.pipeline = pipeline

    @always_inline
    fn __enter__(self) -> ProducerStage:
        """Wait for consumer and return stage info."""
        self.pipeline[].wait_consumer()
        return ProducerStage(
            self.pipeline[].producer_stage(),
            self.pipeline[].producer_mbar(self.pipeline[].producer_stage()),
        )

    @always_inline
    fn __exit__(self):
        """Advance producer to next stage."""
        self.pipeline[].producer_step()


@register_passable("trivial")
struct ConsumerStage:
    """Stage info returned by ConsumeContext.__enter__."""

    var _index: UInt32
    var _mbar: MbarPtr
    var _pipeline_ptr: UnsafePointer[NoneType]  # For accessing empty barriers

    @always_inline
    fn __init__(out self, index: UInt32):
        self._index = index
        self._mbar = MbarPtr()
        self._pipeline_ptr = UnsafePointer[NoneType]()

    @always_inline
    fn __init__(out self, index: UInt32, mbar: MbarPtr):
        self._index = index
        self._mbar = mbar
        self._pipeline_ptr = UnsafePointer[NoneType]()

    @always_inline
    fn index(self) -> UInt32:
        """Get the current stage index."""
        return self._index

    @always_inline
    fn mbar(self) -> MbarPtr:
        """Get the empty barrier for manual signaling.

        Prefer using `signal()` for cleaner code. Use this for
        specialized signaling like `umma_arrive_leader_cta`.
        """
        return self._mbar

    @always_inline
    fn arrive(self):
        """Arrive on this stage's consumer barrier.

        Use with lane-guarded patterns:
            if lane_id() < CLUSTER_SIZE:
                stage.arrive()

        For specialized signaling (e.g., umma_arrive_leader_cta),
        use stage.mbar() directly.
        """
        _ = self._mbar[0].arrive()


@register_passable("trivial")
struct ConsumeContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
]:
    """Context for consuming one pipeline stage.

    - __enter__: Waits for producer to be ready, returns stage info
    - __exit__: Signals consumption complete and advances to next stage
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
    ]

    @always_inline
    fn __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
        ],
    ):
        self.pipeline = pipeline

    @always_inline
    fn __enter__(self) -> ConsumerStage:
        """Wait for producer and return stage info."""
        self.pipeline[].wait_producer()
        return ConsumerStage(self.pipeline[].consumer_stage())

    @always_inline
    fn __exit__(self):
        """Signal consumption complete and advance."""
        _ = self.pipeline[].empty[self.pipeline[].consumer_stage()].arrive()
        self.pipeline[].consumer_step()


@register_passable("trivial")
struct ExplicitConsumeContext[
    pipeline_origin: MutOrigin,
    num_stages: Int,
]:
    """Context for consuming one pipeline stage with EXPLICIT arrive.

    Use this when you need lane-guarded or specialized barrier signaling.

    - __enter__: Waits for producer to be ready, returns stage info with mbar
    - __exit__: Only advances stage counter, does NOT arrive on barrier

    The caller is responsible for calling arrive via stage.arrive() or stage.mbar():
        with pipeline.consume_explicit() as stage:
            # ... do work ...
            if lane_id() < CLUSTER_SIZE:
                stage.arrive()
        # __exit__ only calls consumer_step(), not arrive()
    """

    var pipeline: Pointer[
        ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
    ]

    @always_inline
    fn __init__(
        out self,
        pipeline: Pointer[
            ProducerConsumerPipeline[Self.num_stages], Self.pipeline_origin
        ],
    ):
        self.pipeline = pipeline

    @always_inline
    fn __enter__(self) -> ConsumerStage:
        """Wait for producer and return stage info with barrier access."""
        self.pipeline[].wait_producer()
        var stage_idx = self.pipeline[].consumer_stage()
        return ConsumerStage(
            stage_idx,
            self.pipeline[].consumer_mbar(stage_idx),
        )

    @always_inline
    fn __exit__(self):
        """Advance to next stage WITHOUT signaling barrier."""
        # Caller is responsible for signaling via stage.mbar()
        self.pipeline[].consumer_step()
