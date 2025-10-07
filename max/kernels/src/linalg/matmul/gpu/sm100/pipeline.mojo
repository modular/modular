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


from sys import size_of

from gpu.memory import AddressSpace
from layout.tma_async import SharedMemBarrier


alias MbarPtr = UnsafePointer[
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
        self.empty = ptr + num_stages
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

        if self._consumer_stage == num_stages:
            self._consumer_stage = 0
            self._consumer_phase ^= 1

    @always_inline
    fn producer_step(mut self):
        """Advance the producer to the next pipeline stage.

        Increments the producer stage and wraps to 0 when reaching num_stages,
        toggling the phase bit on wrap-around.
        """
        self._producer_stage += 1

        if self._producer_stage == num_stages:
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
        return 2 * num_stages * size_of[SharedMemBarrier]()

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
        for i in range(num_stages):
            self.full[i].init(producer_arrive_count)
            self.empty[i].init(consumer_arrive_count)
