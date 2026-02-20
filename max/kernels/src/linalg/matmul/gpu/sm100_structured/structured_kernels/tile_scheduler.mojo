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

from math import ceildiv
from sys import _RegisterPackType, size_of
from sys._assembly import inlined_assembly

from gpu.primitives.cluster import (
    clusterlaunchcontrol_query_cancel_get_first_ctaid_v4,
    clusterlaunchcontrol_query_cancel_is_canceled,
    clusterlaunchcontrol_try_cancel,
    elect_one_sync,
)
from gpu import block_id_in_cluster, block_idx, lane_id, warp_id
from gpu.memory import fence_async_view_proxy
from layout.tma_async import PipelineState, SharedMemBarrier

from utils.fast_div import FastDiv

from linalg.structuring import SMemPtr, SMemArray
from .pipeline import ProducerConsumerPipeline
from utils.index import Index, IndexList
from utils.static_tuple import StaticTuple

from linalg.matmul.gpu.tile_scheduler import RasterOrder


@fieldwise_init
struct WorkInfo(Stringable, TrivialRegisterPassable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @always_inline
    fn coord(self) -> Tuple[UInt, UInt]:
        """Get (m, n) tile coordinates as a tuple."""
        return (UInt(self.m), UInt(self.n))

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to(self, mut writer: Some[Writer]):
        writer.write(
            "(",
            self.m,
            ", ",
            self.n,
            ", ",
            self.k_start,
            ", ",
            self.is_valid_tile,
            ")",
        )


# =============================================================================
# Work Iteration Context Managers
# =============================================================================
#
# Two patterns exist for iterating over work items in warp-specialized kernels:
#
# 1. ADVANCE AFTER WORK (Load/Scheduler/Epilogue warps):
#    - Do work with current work_info
#    - Then fetch next, assign, step
#    Usage:
#      with scheduler.advance_after_work(work_info, state) as current:
#          do_work(current)
#      # After: work_info updated, state stepped
#
# 2. PREFETCH BEFORE WORK (MMA warp - software pipelining):
#    - Fetch next and step BEFORE doing work
#    - Do work with current work_info
#    - Then assign prefetched value
#    Usage:
#      with scheduler.prefetch_before_work(work_info, state) as current:
#          do_mma(current)
#      # After: work_info updated to prefetched value
# =============================================================================


struct AdvanceAfterWorkContext[
    work_origin: MutOrigin,
    state_origin: MutOrigin,
    num_stages: Int,
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
](TrivialRegisterPassable):
    """Context for warps that do work THEN advance (Load/Scheduler/Epilogue).

    - __enter__: Returns current work_info for use in the block
    - __exit__: Fetches next work, assigns to work_info, steps state
    """

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]

    var scheduler: Self.SchedulerType
    var work_info_ptr: Pointer[WorkInfo, Self.work_origin]
    var consumer_state_ptr: Pointer[
        PipelineState[Self.num_stages], Self.state_origin
    ]

    @always_inline
    fn __init__(
        out self,
        scheduler: Self.SchedulerType,
        work_info_ptr: Pointer[WorkInfo, Self.work_origin],
        consumer_state_ptr: Pointer[
            PipelineState[Self.num_stages], Self.state_origin
        ],
    ):
        self.scheduler = scheduler
        self.work_info_ptr = work_info_ptr
        self.consumer_state_ptr = consumer_state_ptr

    @always_inline
    fn __enter__(self) -> WorkInfo:
        return self.work_info_ptr[]

    @always_inline
    fn __exit__(mut self):
        var next = self.scheduler.fetch_next_work(
            self.work_info_ptr[],
            self.consumer_state_ptr[],
        )
        self.work_info_ptr[] = next
        self.consumer_state_ptr[].step()


struct WaitAndAdvanceContext[
    work_origin: MutOrigin,
](TrivialRegisterPassable):
    """Context for waiting on CLC barrier and advancing work iterator.

    Encapsulates the CLC response barrier synchronization:
    - Construction: Waits for CLC response, fetches next work
    - __enter__: Returns current work_info for processing
    - __exit__: Assigns fetched work as current

    Usage:
        with work_iter.wait_and_advance() as current:
            # current is the work item to process NOW
            process(current)
        # After exit, work_iter.work_info is the NEXT work item
    """

    var work_info_ptr: Pointer[WorkInfo, Self.work_origin]
    var next_work: WorkInfo

    @always_inline
    fn __init__(
        out self,
        work_info_ptr: Pointer[WorkInfo, Self.work_origin],
        next_work: WorkInfo,
    ):
        self.work_info_ptr = work_info_ptr
        self.next_work = next_work

    @always_inline
    fn __enter__(self) -> WorkInfo:
        return self.work_info_ptr[]

    @always_inline
    fn __exit__(mut self):
        self.work_info_ptr[] = self.next_work


# =============================================================================
# WorkIterator - Per-warp iterator encapsulating scheduler + pipeline state
# =============================================================================


struct WorkIterator[
    num_stages: Int,
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
](TrivialRegisterPassable):
    """Per-warp work iterator that owns work_info and pipeline state.

    Each warp creates its own WorkIterator which internally manages both
    the current work item and the CLC pipeline consumer state. Throttle
    pipeline is obtained from the scheduler.

    Usage:
        var work_iter = scheduler.work_iterator()
        while work_iter.has_work():
            with work_iter.next() as current:
                work_iter.throttle_signal(ctx.is_first_cta_in_cluster)
                do_work(current)
    """

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]
    comptime ThrottlePipeline = Self.SchedulerType.ThrottlePipeline

    var scheduler: Self.SchedulerType
    var work_info: WorkInfo
    var consumer_state: PipelineState[Self.num_stages]
    var throttle_pipeline: Self.ThrottlePipeline

    @always_inline
    fn __init__(out self, scheduler: Self.SchedulerType, work_info: WorkInfo):
        """Create work iterator with initial work_info. Throttle from scheduler.
        """
        self.scheduler = scheduler
        self.work_info = work_info
        self.consumer_state = PipelineState[Self.num_stages]()
        self.throttle_pipeline = scheduler.throttle_pipeline

    @always_inline
    fn has_work(self) -> Bool:
        """Check if there is more work to process."""
        return self.work_info.is_valid()

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> AdvanceAfterWorkContext[
        origin_of(self.work_info),
        origin_of(self.consumer_state),
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]:
        """Get next work item (advance AFTER work pattern)."""
        return AdvanceAfterWorkContext(
            self.scheduler,
            Pointer(to=self.work_info),
            Pointer(to=self.consumer_state),
        )

    @always_inline
    fn wait_and_advance[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> WaitAndAdvanceContext[
        origin_of(self.work_info)
    ]:
        """Wait for next work from CLC and advance iterator.

        Encapsulates the CLC barrier wait:
        - __enter__: Waits for CLC response, returns current work
        - __exit__: Assigns fetched work as current

        Usage:
            with work_iter.wait_and_advance() as current:
                # Process current work item
            # After exit, work_iter points to next work
        """
        var next = self.scheduler.fetch_next_work(
            self.work_info, self.consumer_state
        )
        self.consumer_state.step()
        return WaitAndAdvanceContext(Pointer(to=self.work_info), next)

    # ========== CLC Throttle (Producer Side) ==========

    @always_inline
    fn throttle_signal(mut self, is_first_cta_in_cluster: Bool):
        """Signal CLC throttle if this is the first CTA in cluster.

        The Load warp acts as producer for CLC throttle, signaling that it has
        started processing a new work item. This prevents the scheduler from
        getting too far ahead.

        Args:
            is_first_cta_in_cluster: Only first CTA signals to avoid duplicates.
        """
        if is_first_cta_in_cluster:
            self.throttle_pipeline.producer_signal_and_step()


# =============================================================================
# SchedulerWorkIterator - For Scheduler warp (owns both producer and consumer state)
# =============================================================================


struct SchedulerWorkIterator[
    num_stages: Int,
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
](TrivialRegisterPassable):
    """Work iterator for Scheduler warp - owns work_info and both pipeline states.

    The Scheduler warp uniquely needs to:
    1. Consume work responses (like other warps) via next()
    2. Signal throttle and produce new work requests via signal_and_advance()
    3. Drain pending requests at exit via drain()

    Usage:
        var sched_iter = scheduler.scheduler_iterator()
        while sched_iter.has_work():
            with sched_iter.next():
                sched_iter.signal_and_advance()
        sched_iter.drain()
    """

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]
    comptime ThrottlePipeline = Self.SchedulerType.ThrottlePipeline

    var scheduler: Self.SchedulerType
    var work_info: WorkInfo
    var consumer_state: PipelineState[Self.num_stages]
    var producer_state: PipelineState[Self.num_stages]
    var throttle_pipeline: Self.ThrottlePipeline

    @always_inline
    fn __init__(out self, scheduler: Self.SchedulerType, work_info: WorkInfo):
        """Create scheduler iterator. Throttle pipeline from scheduler."""
        self.scheduler = scheduler
        self.work_info = work_info
        self.consumer_state = PipelineState[Self.num_stages]()
        self.producer_state = PipelineState[Self.num_stages](0, 1, 0)
        self.throttle_pipeline = scheduler.throttle_pipeline

    @always_inline
    fn has_work(self) -> Bool:
        """Check if there is more work to process."""
        return self.work_info.is_valid()

    # ========== Work Iteration (Consumer Side) ==========

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> AdvanceAfterWorkContext[
        origin_of(self.work_info),
        origin_of(self.consumer_state),
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]:
        """Get next work item."""
        return AdvanceAfterWorkContext(
            self.scheduler,
            Pointer(to=self.work_info),
            Pointer(to=self.consumer_state),
        )

    # ========== CLC Throttle + Work Request ==========

    @always_inline
    fn signal_and_advance(mut self):
        """Signal CLC throttle consumer and advance to next work request.

        Combines two operations that always happen together in Scheduler warp:
        1. Signal throttle consumer (tells Load warp we've consumed a response)
        2. Issue next CLC work request (producer side)
        """
        self.throttle_pipeline.consumer_signal_and_step()
        self.producer_state = self.scheduler.advance_to_next_work(
            self.producer_state
        )

    # ========== Cleanup ==========

    @always_inline
    fn drain(mut self):
        """Drain all pending CLC requests before kernel exit.

        Must be called after the work loop completes to ensure all
        CLC pipeline stages are properly synchronized before exit.
        """

        comptime for i in range(Self.num_stages):
            self.scheduler.empty_mbar[self.producer_state.index()].wait(
                self.producer_state.phase()
            )
            self.producer_state.step()


struct TileScheduler[
    num_stages: Int,
    cluster_shape: IndexList[3, element_type = DType.uint32] = Index[
        dtype = DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
](TrivialRegisterPassable):
    comptime cluster_size = Self.cluster_shape[0] * Self.cluster_shape[
        1
    ] * Self.cluster_shape[2]
    comptime log_cluster_m = FastDiv[DType.uint32](Self.cluster_shape[0])
    comptime log_cluster_n = FastDiv[DType.uint32](Self.cluster_shape[1])
    comptime log_cluster_k = FastDiv[DType.uint32](Self.cluster_shape[2])

    comptime ThrottlePipeline = ProducerConsumerPipeline[Self.num_stages]

    # Typed barrier array aliases for clean API
    comptime ClcResponseArray = SMemArray[UInt128, Self.num_stages]
    comptime ClcBarrierArray = SMemArray[SharedMemBarrier, Self.num_stages]
    comptime ThrottleBarrierArray = SMemArray[
        SharedMemBarrier, Self.num_stages * 2
    ]

    var cluster_dim: StaticTuple[Int32, 3]
    var log_cluster_dim_m: FastDiv[DType.uint32]
    var log_cluster_dim_n: FastDiv[DType.uint32]
    var log_cluster_dim_k: FastDiv[DType.uint32]

    var clc_response: SMemPtr[UInt128]
    var full_mbar: SMemPtr[SharedMemBarrier]
    var empty_mbar: SMemPtr[SharedMemBarrier]
    var throttle_pipeline: Self.ThrottlePipeline

    @staticmethod
    fn init_throttle_barriers(
        storage_ptr: SMemPtr[SharedMemBarrier],
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize throttle pipeline barriers. Called once by elect_one thread.
        """
        var pipeline = Self.ThrottlePipeline(storage_ptr)
        pipeline.init_mbars(producer_arv_count, consumer_arv_count)

    @always_inline
    fn __init__(
        out self,
        cluster_dim: StaticTuple[Int32, 3],
        clc_response: Self.ClcResponseArray,
        clc_full: Self.ClcBarrierArray,
        clc_empty: Self.ClcBarrierArray,
        clc_throttle: Self.ThrottleBarrierArray,
    ):
        """Initialize from typed barrier arrays."""
        constrained[
            Self.block_swizzle_size in [0, 1, 2, 4, 8],
            "block_swizzle_size must be 0, 1, 2, 4, or 8",
        ]()

        self.cluster_dim = cluster_dim
        self.log_cluster_dim_m = FastDiv[DType.uint32](Int(cluster_dim[0]))
        self.log_cluster_dim_n = FastDiv[DType.uint32](Int(cluster_dim[1]))
        self.log_cluster_dim_k = FastDiv[DType.uint32](Int(cluster_dim[2]))
        self.clc_response = clc_response.ptr
        self.full_mbar = clc_full.ptr
        self.empty_mbar = clc_empty.ptr
        self.throttle_pipeline = Self.ThrottlePipeline(clc_throttle.ptr)

    @always_inline
    @staticmethod
    fn work_info_from_clc_response(result: SMemPtr[UInt128]) -> WorkInfo:
        comptime asm = """{
            .reg .pred p1;
            .reg .b128 clc_result;
            ld.shared.b128 clc_result, [$4];
            clusterlaunchcontrol.query_cancel.is_canceled.pred.b128 p1, clc_result;
            selp.u32 $3, 1, 0, p1;
            @p1 clusterlaunchcontrol.query_cancel.get_first_ctaid.v4.b32.b128 {$0, $1, $2, _}, clc_result;
        }"""
        var ret_val = inlined_assembly[
            asm,
            _RegisterPackType[UInt32, UInt32, UInt32, UInt32],
            has_side_effect=True,
            constraints="=r,=r,=r,=r,r",
        ](UInt32(Int(result)))

        fence_async_view_proxy()

        return WorkInfo(
            m=ret_val[0],
            n=ret_val[1],
            k_start=ret_val[2],
            is_valid_tile=(ret_val[3] == 1),
        )

    @always_inline
    @staticmethod
    fn work_info_from_cluster(
        work_info: WorkInfo,
        cluster_dim: StaticTuple[Int32, 3],
        log_cluster_dim_m: FastDiv[DType.uint32],
        log_cluster_dim_n: FastDiv[DType.uint32],
    ) -> WorkInfo:
        comptime FastUInt = Scalar[FastDiv[DType.uint32].uint_type]

        var normalized_m = FastUInt(work_info.m) / Self.log_cluster_m
        var normalized_n = FastUInt(work_info.n) / Self.log_cluster_n
        comptime log_block_swizzle_size = FastDiv[DType.uint32](
            Self.block_swizzle_size
        )

        var linear_cluster_id = (
            normalized_m * FastUInt(cluster_dim[1]) + normalized_n
        )

        # CLC rasterize along M by default.
        comptime if Self.rasterize_order == RasterOrder.AlongM:
            new_normalized_m = normalized_m
            new_normalized_n = normalized_n
        else:
            new_normalized_m = linear_cluster_id % log_cluster_dim_m
            new_normalized_n = linear_cluster_id / log_cluster_dim_m

        comptime if Self.block_swizzle_size != 0:
            var swizzle_m_size = (
                FastUInt(cluster_dim[0]) / log_block_swizzle_size
            )
            var swizzle_n_size = (
                FastUInt(cluster_dim[1]) / log_block_swizzle_size
            )

            var m_local = (new_normalized_m / log_block_swizzle_size) + (
                (swizzle_m_size) * (new_normalized_n % log_block_swizzle_size)
            )
            var n_local = new_normalized_m % log_block_swizzle_size

            var is_even_subtile = Int(
                (new_normalized_n / log_block_swizzle_size) % 2 == 0
            )

            var m_bound = swizzle_m_size * FastUInt(Self.block_swizzle_size)
            var n_bound = swizzle_n_size * FastUInt(Self.block_swizzle_size)
            if new_normalized_m < m_bound and new_normalized_n < n_bound:
                new_m_global = FastUInt(is_even_subtile) * m_local + FastUInt(
                    1 - is_even_subtile
                ) * (m_bound - m_local - 1)
                new_n_global = n_local + FastUInt(
                    Int(new_normalized_n / log_block_swizzle_size)
                    * Self.block_swizzle_size
                )
            else:
                new_m_global = new_normalized_m
                new_n_global = new_normalized_n
        else:
            new_m_global = new_normalized_m
            new_n_global = new_normalized_n

        return WorkInfo(
            m=UInt32(
                Int(new_m_global) * Self.cluster_shape[0]
                + Int(block_id_in_cluster.x)
            ),
            n=UInt32(
                Int(new_n_global) * Self.cluster_shape[1]
                + Int(block_id_in_cluster.y)
            ),
            k_start=work_info.k_start,
            is_valid_tile=work_info.is_valid_tile,
        )

    @always_inline
    fn initial_work_info(self) -> WorkInfo:
        return self.work_info_from_cluster(
            WorkInfo(
                UInt32(block_idx.x),
                UInt32(block_idx.y),
                UInt32(block_idx.z),
                is_valid_tile=True,
            ),
            self.cluster_dim,
            self.log_cluster_dim_m,
            self.log_cluster_dim_n,
        )

    @always_inline
    fn fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[Self.num_stages],
    ) -> WorkInfo:
        # num_stages == 0 implies there is only one wave. Only initial
        # work info is valid, next work info is invalid.
        comptime if Self.num_stages == 0:
            return WorkInfo(0, 0, 0, False)

        self.full_mbar[consumer_state.index()].wait(consumer_state.phase())
        var work_tile = self.work_info_from_clc_response(
            self.clc_response + consumer_state.index()
        )
        # Only cta 0 in a cluster is used for scheduling.
        self.empty_mbar[consumer_state.index()].arrive_cluster(0)
        return self.work_info_from_cluster(
            work_tile,
            self.cluster_dim,
            self.log_cluster_dim_m,
            self.log_cluster_dim_n,
        )

    # =========================================================================
    # Work Iteration Context Managers
    # =========================================================================

    @always_inline
    fn advance_after_work[
        work_origin: MutOrigin, state_origin: MutOrigin, //
    ](
        self,
        ref[work_origin] work_info: WorkInfo,
        ref[state_origin] consumer_state: PipelineState[Self.num_stages],
    ) -> AdvanceAfterWorkContext[
        work_origin,
        state_origin,
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]:
        """Context for warps that do work THEN advance (Load/Scheduler/Epilogue).

        Usage:
            with scheduler.advance_after_work(work_info, state) as current:
                do_work(current)
                syncwarp()
            # After: work_info updated, state stepped
        """
        return AdvanceAfterWorkContext(
            self,
            Pointer(to=work_info),
            Pointer(to=consumer_state),
        )

    @always_inline
    fn wait_and_advance_work[
        work_origin: MutOrigin, //
    ](
        self,
        ref[work_origin] work_info: WorkInfo,
        mut consumer_state: PipelineState[Self.num_stages],
    ) -> WaitAndAdvanceContext[work_origin]:
        """Wait for next work from CLC and advance.

        Encapsulates the CLC barrier wait (called on scheduler directly).

        Usage:
            with scheduler.wait_and_advance_work(work_info, state) as current:
                do_mma(current)
            # After: work_info updated to next value
        """
        var next = self.fetch_next_work(work_info, consumer_state)
        consumer_state.step()
        return WaitAndAdvanceContext(Pointer(to=work_info), next)

    @always_inline
    fn work_iterator(
        self,
    ) -> WorkIterator[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]:
        """Create a per-warp work iterator with internally managed state.

        Each warp should create its own work iterator. The iterator owns
        work_info, pipeline state, and throttle internally.

        Usage:
            var work_iter = scheduler.work_iterator()
            while work_iter.has_work():
                with work_iter.next() as current:
                    work_iter.throttle_signal(ctx.is_first_cta_in_cluster)
                    do_work(current)
        """
        return WorkIterator(self, self.initial_work_info())

    @always_inline
    fn scheduler_iterator(
        self,
    ) -> SchedulerWorkIterator[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]:
        """Create iterator for Scheduler warp (owns work_info and both pipeline states).

        The Scheduler warp uniquely needs to both consume work responses and
        produce new work requests. This iterator owns everything internally.

        Usage:
            var sched_iter = scheduler.scheduler_iterator()
            while sched_iter.has_work():
                with sched_iter.next():
                    sched_iter.signal_and_advance()
            sched_iter.drain()
        """
        return SchedulerWorkIterator(self, self.initial_work_info())

    @always_inline
    fn advance_to_next_work(
        self,
        mut clc_state: PipelineState[Self.num_stages],
    ) -> PipelineState[Self.num_stages]:
        comptime multicast = True if Self.cluster_size > 1 else False
        var lane_id = lane_id()
        var pred: UInt32 = UInt32(1) if lane_id < UInt(
            Self.cluster_size
        ) else UInt32(0)
        self.empty_mbar[clc_state.index()].wait(clc_state.phase())
        self.full_mbar[clc_state.index()].arrive_and_expect_bytes(
            Int32(size_of[UInt128]()),
            UInt32(lane_id),
            pred,
        )

        if elect_one_sync():
            clusterlaunchcontrol_try_cancel[multicast=multicast](
                self.clc_response + clc_state.index(),
                (self.full_mbar + clc_state.index()).bitcast[Int64](),
            )

        return clc_state.next()
