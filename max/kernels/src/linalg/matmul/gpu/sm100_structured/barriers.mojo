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
"""Barrier abstractions for SM100 structured matmul kernels.

This module provides type-safe wrappers around low-level barrier primitives,
improving code readability and reducing error potential.
"""

from gpu.cluster import block_rank_in_cluster
from gpu.sync import named_barrier, named_barrier_arrive
from layout.tma_async import SharedMemBarrier

from linalg.structuring import SMemArrayType

from .tmem import TmemAllocation


@register_passable("trivial")
struct WarpGroupBarrier[num_threads: Int, barrier_id: Int = 0]:
    """Named barrier for warp group synchronization.

    Wraps `named_barrier` and `named_barrier_arrive` with compile-time
    thread count and barrier ID for type-safe synchronization.
    """

    @staticmethod
    @always_inline
    fn arrive():
        """Signal arrival without blocking (non-blocking arrive)."""
        named_barrier_arrive[Self.num_threads](Self.barrier_id)

    @staticmethod
    @always_inline
    fn wait():
        """Block until all threads have arrived."""
        named_barrier[Self.num_threads](Self.barrier_id)

    @staticmethod
    @always_inline
    fn sync():
        """Full barrier: arrive and wait for all threads."""
        named_barrier[Self.num_threads](Self.barrier_id)


@register_passable("trivial")
struct TmemDeallocBarrier[cta_group: Int]:
    """TMEM deallocation synchronization barrier.

    Handles cluster-aware synchronization patterns for TMEM deallocation,
    supporting both single-CTA and multi-CTA (cta_group=2) configurations.
    """

    var barrier: SMemArrayType[SharedMemBarrier, 1]

    fn __init__(out self, barrier: SMemArrayType[SharedMemBarrier, 1]):
        """Initialize with shared memory barrier array."""
        self.barrier = barrier

    @always_inline
    fn signal_peer(self):
        """Signal peer CTA in cluster (cta_group=2 only)."""

        @parameter
        if Self.cta_group == 2:
            _ = self.barrier.ptr[].arrive_cluster(block_rank_in_cluster() ^ 1)

    @always_inline
    fn signal_self(self):
        """Signal own arrival at barrier."""
        _ = self.barrier.ptr[].arrive()

    @always_inline
    fn wait(self):
        """Wait for barrier completion."""
        self.barrier.ptr[].wait()

    @always_inline
    fn complete_dealloc[
        max_cols: Int = 512
    ](self, tmem: TmemAllocation[Self.cta_group, max_cols]):
        """Complete TMEM deallocation sequence (MMA warp side).

        Releases the allocation lock, waits for epilogue completion,
        then deallocates the TMEM.
        """
        tmem.release_lock()
        self.wait()
        tmem.deallocate()

    @always_inline
    fn signal_complete(self):
        """Signal TMEM consumption complete (Epilogue warp side).

        For cta_group=2, signals peer CTA first, then signals self.
        """
        self.signal_peer()
        self.signal_self()
