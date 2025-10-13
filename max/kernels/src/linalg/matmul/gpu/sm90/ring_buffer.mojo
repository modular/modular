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
from layout.tma_async import PipelineState
from ....structuring import SMemBarrier
from gpu.sync import async_copy_arrive
from gpu.globals import WARPGROUP_SIZE


# Ring buffer abstraction for producer-consumer synchronization
@register_passable("trivial")
struct RingBuffer[
    num_pipeline_stages: Int,
    num_consumers: Int,
    cluster_size: Int,
    use_async_copy: Bool = False,
]:
    """Ring buffer for managing pipeline synchronization between producers and consumers.

    This struct encapsulates the synchronization logic for a multi-stage pipeline
    with one producer and multiple consumers, supporting both single-block and
    multi-cluster configurations.
    """

    # Barriers for synchronization
    var full_mbar: SMemBarrier
    var empty_mbar: SMemBarrier

    # Pipeline states
    var read_state: PipelineState[num_pipeline_stages]
    var write_state: PipelineState[num_pipeline_stages]

    var warp_group_thread_idx: UInt

    fn __init__(
        out self,
        full_mbar: SMemBarrier,
        empty_mbar: SMemBarrier,
        warp_group_thread_idx: UInt,
    ):
        """Initialize ring buffer with barrier pointers."""
        self.full_mbar = full_mbar
        self.empty_mbar = empty_mbar
        self.read_state = PipelineState[num_pipeline_stages]()
        self.write_state = PipelineState[num_pipeline_stages]()
        self.warp_group_thread_idx = warp_group_thread_idx

        if thread_idx.x == 0:

            @parameter
            for i in range(num_pipeline_stages):
                self.full_mbar[i].init(WARPGROUP_SIZE if use_async_copy else 1)
                self.empty_mbar[i].init(num_consumers * cluster_size)

    fn __enter__(mut self) -> Self:
        """Context manager entry."""
        return self

    @always_inline
    fn get_slot[
        expected_bytes: Int = 0,
    ](mut self) -> UInt32:
        """Producer waits for empty buffer slot and prepares for loading."""
        var write_idx = self.write_state.index()
        self.empty_mbar[Int(write_idx)].wait(self.write_state.phase())
        if expected_bytes > 0:
            self.full_mbar[Int(write_idx)].expect_bytes(expected_bytes)
        return write_idx

    @always_inline
    fn enqueue_tile(mut self):
        """Producer signals for cp.async operations.

        This handles the specific signaling pattern needed for cp.async:
        1. Signal async copy arrival
        2. Arrive at the barrier
        3. Advance to next stage
        """
        if use_async_copy:
            var write_idx = self.write_state.index()
            async_copy_arrive(self.full_mbar[write_idx].unsafe_ptr())
            _ = self.full_mbar[write_idx].arrive()
        self.write_state.step()

    @always_inline
    fn get_tile(mut self) -> UInt32:
        """Consumer waits for full buffer slot."""
        var read_idx = self.read_state.index()
        self.full_mbar[Int(read_idx)].wait(self.read_state.phase())
        return read_idx

    @always_inline
    fn release_slot(
        mut self,
        read_idx: UInt32,
    ):
        """Consumer signals that buffer slot is empty."""

        @parameter
        if cluster_size > 1:
            if self.warp_group_thread_idx < UInt(cluster_size):
                _ = self.empty_mbar[Int(read_idx)].arrive_cluster(
                    self.warp_group_thread_idx
                )
        else:
            if self.warp_group_thread_idx == 0:
                _ = self.empty_mbar[Int(read_idx)].arrive()

        # Advance to next pipeline stage
        self.read_state.step()

    @always_inline
    fn arrive_empty_barriers(
        self,
    ):
        """Helper to arrive at empty barriers during consumer initialization."""

        @parameter
        for i in range(num_pipeline_stages):

            @parameter
            if cluster_size > 1:
                if self.warp_group_thread_idx < UInt(cluster_size):
                    _ = self.empty_mbar[i].arrive_cluster(
                        self.warp_group_thread_idx
                    )
            else:
                if self.warp_group_thread_idx == 0:
                    _ = self.empty_mbar[i].arrive()
