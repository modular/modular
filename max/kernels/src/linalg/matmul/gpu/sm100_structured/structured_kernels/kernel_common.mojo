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
- WarpRole: Warp specialization roles for 4-warp kernels (MMA, Load, Scheduler, Epilogue)
- WarpRole1D1D: Warp specialization roles for 3-warp kernels (MMA, Load, Epilogue)
- KernelContext: Common kernel state (election vars, CTA coords, masks)
- Barrier init helpers: compute_input_consumer_count, init_core_barriers, init_clc_barriers
- _Batched3DLayout / _to_batched_3d: Reshape 2D TileTensor to 3D (batch=1)
- consumer_main_loop: Legacy MMA consumer loop (deprecated but kept for compatibility)
"""

from gpu import WARP_SIZE, thread_idx
from gpu import warp_id as get_warp_id
from gpu import block_id_in_cluster
from gpu.primitives.cluster import (
    block_rank_in_cluster,
    elect_one_sync,
    elect_one_sync_with_mask,
)
from gpu.host.nvidia.tma import TensorMapSwizzle
from layout.tma_async import SharedMemBarrier
from layout._layout import RowMajorLayout, TensorLayout, row_major
from layout.coord import ComptimeInt, Coord, Idx
from layout.tile_tensor import TileTensor

from utils.index import IndexList
from utils.static_tuple import StaticTuple

from linalg.arch.sm100 import MmaOpSM100_SS
from linalg.structuring import SMemPtr, SMemArray, SMemTileIter
from .pipeline import ProducerConsumerPipeline

comptime MbarPtr = SMemPtr[SharedMemBarrier]


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
# WarpRole1D1D - 3-warp specialization (no scheduler)
# =============================================================================


struct WarpRole1D1D(TrivialRegisterPassable):
    """Warp role for 1D-1D kernels with 3-warp specialization.

    Thread layout (192 threads total):
    - Warps 0-3 (threads 0-127): Epilogue (4 warps)
    - Warp 4 (threads 128-159): TMA Load
    - Warp 5 (threads 160-191): MMA

    The epilogue warps being at 0-3 is important because TMAStoreCoords
    uses `warp_id == 0` for election.

    No scheduler warp — work distribution uses linear grid traversal.
    """

    comptime EPILOGUE_WARP_START = 0
    comptime LOAD_WARP_START = 128
    comptime MMA_WARP_START = 160

    comptime NUM_EPILOGUE_THREADS = 128  # 4 warps
    comptime NUM_LOAD_THREADS = 32
    comptime NUM_MMA_THREADS = 32

    comptime TOTAL_THREADS = 192

    @staticmethod
    @always_inline
    fn is_epilogue() -> Bool:
        """Returns True if current thread is in an epilogue warp (warps 0-3)."""
        return thread_idx.x < Self.LOAD_WARP_START

    @staticmethod
    @always_inline
    fn is_load() -> Bool:
        """Returns True if current thread is in the TMA load warp (warp 4)."""
        return (
            thread_idx.x >= Self.LOAD_WARP_START
            and thread_idx.x < Self.MMA_WARP_START
        )

    @staticmethod
    @always_inline
    fn is_mma() -> Bool:
        """Returns True if current thread is in the MMA warp (warp 5)."""
        return thread_idx.x >= Self.MMA_WARP_START


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
# TMA tile dimension and barrier count helpers
# =============================================================================


@always_inline
fn compute_tma_tile_dims[
    BM: Int,
    BN: Int,
    MMA_M: Int,
    OutputM: Int,
    CLUSTER_M: Int,
    CLUSTER_N: Int,
    cta_group: Int,
    AB_swapped: Bool = False,
]() -> StaticTuple[Int, 3]:
    """Compute TMA tile dimensions (a_tile_dim0, b_tile_dim0, c_tile_dim0).

    Returns:
        StaticTuple of (a_tile_dim0, b_tile_dim0, c_tile_dim0).
    """
    comptime a_tile_dim0 = BM // CLUSTER_N
    comptime b_tile_dim0 = BN // (CLUSTER_M // cta_group)
    comptime c_tile_dim0 = OutputM if (
        MMA_M == 256 or cta_group == 1 or AB_swapped
    ) else 64
    return StaticTuple[Int, 3](a_tile_dim0, b_tile_dim0, c_tile_dim0)


@always_inline
fn compute_clc_barrier_counts[
    SCHEDULER_THREADS: Int,
    TMA_LOAD_THREADS: Int,
    MMA_THREADS: Int,
    EPILOGUE_THREADS: Int,
    CLUSTER_SIZE: Int,
    cta_group: Int,
]() -> StaticTuple[Int, 4]:
    """Compute CLC barrier arrival counts.

    Returns:
        StaticTuple of (producer, consumer, throttle_producer, throttle_consumer).
    """
    return StaticTuple[Int, 4](
        1,  # clc_producer_arv_count
        SCHEDULER_THREADS
        + CLUSTER_SIZE
        * (
            TMA_LOAD_THREADS + MMA_THREADS + EPILOGUE_THREADS
        ),  # clc_consumer_arv_count
        TMA_LOAD_THREADS,  # clc_throttle_producer_arv_count
        SCHEDULER_THREADS,  # clc_throttle_consumer_arv_count
    )


@always_inline
fn compute_accum_barrier_counts[
    EPILOGUE_THREADS: Int,
    cta_group: Int,
]() -> StaticTuple[Int, 2]:
    """Compute accumulator pipeline barrier arrival counts.

    Returns:
        StaticTuple of (producer_arv_count, consumer_arv_count).
    """
    return StaticTuple[Int, 2](
        1,  # accum_pipeline_producer_arv_count (MMA warp via mma_arrive)
        cta_group * EPILOGUE_THREADS,  # accum_pipeline_consumer_arv_count
    )


# =============================================================================
# Barrier initialization helpers
# =============================================================================


@always_inline
fn compute_input_consumer_count[
    CLUSTER_M: Int,
    CLUSTER_N: Int,
    cta_group: Int,
    CLUSTER_SIZE: Int = 0,
    epilogue_threads: Int = 0,
]() -> Int:
    """Compute input pipeline barrier consumer count.

    For standard kernels, consumers are the MMA warps across the cluster.
    For blockwise FP8 kernels, epilogue warps also consume input tiles
    (A-scales), so pass CLUSTER_SIZE and epilogue_threads to include them.
    """
    comptime base = CLUSTER_M // cta_group + CLUSTER_N - 1

    comptime if epilogue_threads > 0:
        return base + CLUSTER_SIZE * (epilogue_threads // 32)
    else:
        return base


@always_inline
fn init_core_barriers[
    num_input_stages: Int,
    num_accum_stages: Int,
](
    input_barriers_ptr: MbarPtr,
    input_consumer_count: Int32,
    accum_barriers_ptr: MbarPtr,
    accum_producer_arv_count: Int32,
    accum_consumer_arv_count: Int32,
    tmem_dealloc_ptr: MbarPtr,
    tmem_dealloc_thread_count: Int32,
):
    """Initialize input, output, and TMEM deallocation barriers.

    Called inside the elect_one_warp && elect_one_thread guard.
    Handles the three barrier init steps shared by all SM100 kernels.
    """
    ProducerConsumerPipeline[num_input_stages](input_barriers_ptr).init_mbars(
        Int32(1), input_consumer_count
    )
    ProducerConsumerPipeline[num_accum_stages](accum_barriers_ptr).init_mbars(
        accum_producer_arv_count, accum_consumer_arv_count
    )
    tmem_dealloc_ptr[].init(tmem_dealloc_thread_count)


@always_inline
fn init_clc_barriers[
    num_clc_stages: Int
](
    clc_full_ptr: MbarPtr,
    clc_empty_ptr: MbarPtr,
    clc_producer_arv_count: Int32,
    clc_consumer_arv_count: Int32,
):
    """Initialize CLC full/empty barrier pairs.

    Called inside the elect_one_warp && elect_one_thread guard for
    CLC-enabled kernels (default, block_scaled, blockwise_fp8, grouped_2sm).
    """
    comptime for i in range(num_clc_stages):
        clc_full_ptr[i].init(clc_producer_arv_count)
        clc_empty_ptr[i].init(clc_consumer_arv_count)


# =============================================================================
# consumer_main_loop - MMA consumer loop (external API)
# =============================================================================


# DEPRECATED: Use InputTilePipeline with InputConsumerStage instead.
# This legacy function uses raw SMemTileIter rather than encapsulated
# stage access. Kept for backward compatibility with external callers.
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

    Use InputTilePipeline with InputConsumerStage for new code.
    This function is kept for backward compatibility.
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


# =============================================================================
# _Batched3DLayout / _to_batched_3d - 2D → 3D TileTensor reshape
# =============================================================================


comptime _Batched3DLayout[L: TensorLayout] = RowMajorLayout[
    ComptimeInt[1], L._shape_types[0], L._shape_types[1]
]
"""3D batched layout from a 2D layout: prepend batch=1, preserve shape types."""


fn _to_batched_3d(
    tensor: TileTensor[...],
) -> tensor.ViewType[_Batched3DLayout[type_of(tensor).LayoutType]]:
    """Reshape 2D TileTensor to 3D by prepending batch=1: (M, K) -> (1, M, K).

    The input must be rank 2. Shape types (static/dynamic) are preserved.
    """
    comptime L = type_of(tensor).LayoutType
    comptime assert L.rank == 2, "expected rank-2 TileTensor"
    return tensor.reshape(
        row_major(
            Coord(
                Idx[1](),
                tensor.layout.shape[0](),
                tensor.layout.shape[1](),
            )
        )
    )
