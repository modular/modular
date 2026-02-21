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
"""Shared kernel components for SM100 warp-specialized matmul kernels.

This module contains common components used by all SM100 matmul kernel variants:
- WarpRole: Warp specialization roles (MMA, Load, Scheduler, Epilogue)
- KernelContext: Common kernel state (election vars, CTA coords, masks)
- consumer_main_loop: Legacy MMA consumer loop (deprecated but kept for compatibility)
"""

from gpu import WARP_SIZE
from gpu import warp_id as get_warp_id
from gpu import block_id_in_cluster
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host.nvidia.tma import TensorMapSwizzle

from utils.index import IndexList

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.structuring import SMemPtr, SMemArray, SMemTileIter
from .pipeline import ProducerConsumerPipeline


# =============================================================================
# WarpRole - Warp specialization roles
# =============================================================================


@fieldwise_init
struct WarpRole(TrivialRegisterPassable):
    """Warp role identifiers for SM100 warp-specialized kernel.

    Warp assignment (8 warps total = 256 threads):
    - Epilogue: warp IDs 0-3 (4 warps, 128 threads)
    - Scheduler: warp ID 4 (1 warp, 32 threads)
    - MainLoad: warp ID 5 (1 warp, 32 threads)
    - Mma: warp ID 6 (1 warp, 32 threads)
    - EpilogueLoad: warp ID 7 (1 warp, 32 threads) - loads source C for residual

    Note: When epilogue load is not needed (no residual), warp 7 exits early.
    """

    var _role: Int32

    comptime EpilogueLoad = Self(7)
    comptime Mma = Self(6)
    comptime MainLoad = Self(5)
    comptime Scheduler = Self(4)
    comptime Epilogue = Self(3)

    @always_inline
    fn __eq__(self, other: UInt) -> Bool:
        return self._role == Int32(other)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._role == other._role

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._role != other._role

    @always_inline
    fn __ge__(self, other: UInt) -> Bool:
        return self._role >= Int32(other)

    @staticmethod
    @always_inline
    fn is_main_load() -> Bool:
        return Self.MainLoad == get_warp_id()

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        return Self.Mma == get_warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        return Self.Epilogue >= get_warp_id()

    @staticmethod
    @always_inline
    fn is_scheduler() -> Bool:
        return Self.Scheduler == get_warp_id()

    @staticmethod
    @always_inline
    fn is_epilogue_load() -> Bool:
        """Check if current warp is the epilogue load warp (loads source C)."""
        return Self.EpilogueLoad == get_warp_id()


# =============================================================================
# KernelContext - Common state for kernel entry points
# =============================================================================


struct KernelContext[
    num_clc_pipeline_stages: Int,
    cta_group: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
](Copyable, Movable):
    """Shared kernel state: election vars, CTA coords, multicast masks, pipeline states.
    """

    # ===== Election Variables =====
    var elect_one_warp: Bool
    var elect_one_thread: Bool
    var elect_one_cta: Bool
    var is_first_cta_in_cluster: Bool
    var warp_id: UInt32

    # ===== CTA Coordinates =====
    var rank_m: UInt
    var rank_n: UInt
    var peer_cta_coord: Tuple[UInt, UInt, UInt]

    # ===== Multicast Masks =====
    var a_multicast_mask: UInt16
    var b_multicast_mask: UInt16
    var mma_complete_mask: Int

    # Note: Pipeline states (producer and consumer) are now managed by
    # SchedulerWorkIterator and WorkIterator respectively.

    # ===== TMEM Pointer =====
    comptime TmemAddrArray = SMemArray[UInt32, 1]
    var ptr_tmem_addr: SMemPtr[UInt32]

    @always_inline
    fn __init__(out self, ptr_tmem_addr: SMemPtr[UInt32]):
        """Initialize context from TMEM pointer; computes all derived state."""
        # Election variables
        self.warp_id = UInt32(get_warp_id())
        self.elect_one_warp = self.warp_id == 0
        self.elect_one_thread = elect_one_sync_with_mask()
        self.elect_one_cta = (
            block_rank_in_cluster() % 2 == 0 if Self.cta_group == 2 else True
        )
        self.is_first_cta_in_cluster = block_rank_in_cluster() == 0

        # CTA coordinates
        self.rank_m = block_id_in_cluster.x
        self.rank_n = block_id_in_cluster.y

        # Peer CTA coordinate: (peer_id, mma_coord_m, mma_coord_n)
        self.peer_cta_coord = (
            self.rank_m % UInt(Self.cta_group),
            self.rank_m // UInt(Self.cta_group),
            self.rank_n,
        )

        # Compute multicast masks
        self.a_multicast_mask = 0x0
        self.b_multicast_mask = 0x0

        comptime for i in range(Self.CLUSTER_N):
            self.a_multicast_mask |= UInt16(1 << (i * Self.CLUSTER_M))

        comptime for i in range(Self.CLUSTER_M // Self.cta_group):
            self.b_multicast_mask |= UInt16(1 << (i * Self.cta_group))

        self.a_multicast_mask <<= UInt16(self.rank_m)
        self.b_multicast_mask <<= UInt16(self.peer_cta_coord[0])
        self.b_multicast_mask <<= UInt16(self.rank_n * UInt(Self.CLUSTER_M))

        # MMA completion mask for barrier synchronization
        # For 2SM: peer is the other CTA in the cluster (XOR with 1)
        var self_mask = 1 << Int(block_rank_in_cluster())
        var peer_rank = (
            block_rank_in_cluster() ^ 1 if Self.cta_group
            == 2 else block_rank_in_cluster()
        )
        var peer_mask = 1 << Int(peer_rank)
        self.mma_complete_mask = self_mask | peer_mask

        # TMEM pointer
        self.ptr_tmem_addr = ptr_tmem_addr

    @always_inline
    fn __init__(out self, tmem_addr: Self.TmemAddrArray):
        """Initialize context from typed TMEM address array."""
        self = Self(tmem_addr.ptr)


# =============================================================================
# consumer_main_loop - MMA consumer loop (external API)
# =============================================================================


# DEPRECATED: Use TilePipeline with StandardConsumerStage and BlackwellMatmulSM100Kernel.mma()
# instead. This legacy function uses raw SMemTileIter rather than encapsulated
# StandardConsumerStage access. Kept for backward compatibility with external callers.
@always_inline
fn consumer_main_loop[
    accum_type: DType,
    c_type: DType,
    a_type: DType,
    b_type: DType,
    a_smem_layout: Layout,
    b_smem_layout: Layout,
    a_swizzle: TensorMapSwizzle,
    b_swizzle: TensorMapSwizzle,
    transpose_b: Bool,
    pipeline_stages: Int,
    /,
    *,
    block_tile_shape: IndexList[3],
    mma_shape: IndexList[3],
    cta_group: Int = 1,
    cluster_shape: IndexList[3] = IndexList[3](1, 1, 1),
    k_group_size: Int = 1,
](
    tmem_addr: Int,
    a_smem_iter: SMemTileIter[a_type, a_smem_layout],
    b_smem_iter: SMemTileIter[b_type, b_smem_layout],
    load_mma_pipeline: ProducerConsumerPipeline[pipeline_stages],
    mma_op: MmaOpSM100_SS[
        c_type,
        a_type,
        b_type,
        block_tile_shape,
        mma_shape,
        accum_type=accum_type,
        cta_group=cta_group,
        cluster_shape=cluster_shape,
        a_swizzle=a_swizzle,
        b_swizzle=b_swizzle,
        transpose_b=transpose_b,
    ],
    elect_one_warp: Bool,
    iter_idx: UInt32,
    k_start: UInt32,
):
    """DEPRECATED: Legacy MMA consumer loop for external callers.

    Use TilePipeline with StandardConsumerStage and BlackwellMatmulSM100Kernel.mma()
    for new code. This function is kept for backward compatibility.
    """
    var stage = load_mma_pipeline.consumer_stage()

    load_mma_pipeline.wait_producer()

    if elect_one_sync():
        comptime for j in range(k_group_size):
            var a_smem_tile = a_smem_iter.next(
                stage * UInt32(k_group_size) + UInt32(j)
            )[]
            var b_smem_tile = b_smem_iter.next(
                stage * UInt32(k_group_size) + UInt32(j)
            )[]
            mma_op.mma(
                a_smem_tile,
                b_smem_tile,
                UInt32(tmem_addr),
                init_c=(iter_idx + UInt32(j) == k_start),
            )
        mma_op.commit(load_mma_pipeline.consumer_mbar(stage))
