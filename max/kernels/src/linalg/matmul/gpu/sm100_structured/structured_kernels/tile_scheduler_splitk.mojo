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
from gpu.memory import AddressSpace
from .tile_scheduler import TileScheduler as B200TileScheduler
from .tile_scheduler import WorkInfo as B200WorkInfo
from linalg.matmul.gpu.tile_scheduler import RasterOrder
from layout._layout import TensorLayout, row_major
from layout._coord import Coord, Idx
from layout._tile_tensor import TileTensor
from layout.tma_async import SharedMemBarrier, PipelineState
from utils.static_tuple import StaticTuple
from .tile_types import static_row_major, _StridedLayout, _strided_layout
from gpu import (
    grid_dim,
    thread_idx,
    lane_id,
    NamedBarrierSemaphore,
    WARP_SIZE,
)
from gpu.primitives.cluster import elect_one_sync
from gpu.globals import WARPGROUP_SIZE
from gpu.compute.arch.tcgen05 import *
from gpu.sync import named_barrier
from memory import LegacyUnsafePointer

comptime UnsafePointer = LegacyUnsafePointer[mut=True, ...]
from std.bit import prev_power_of_two

from linalg.structuring import SMemPtr
from .tmem import TmemAddress, TmemTensor


@fieldwise_init
struct WorkInfo(Stringable, TrivialRegisterPassable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    var num_k_tiles: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    comptime INVALID_WORK_INFO = Self(0, 0, 0, 0, False)

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @always_inline
    fn is_final_split(self, k_tiles_per_output_tile: UInt32) -> Bool:
        return (self.k_start + self.num_k_tiles) == k_tiles_per_output_tile

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
# Work Iteration Context Managers (Split-K variant)
# =============================================================================
#
# See tile_scheduler.mojo for pattern documentation. These are split-K versions
# that work with the split-K WorkInfo type.
# =============================================================================


struct AdvanceAfterWorkContextSplitK[
    work_origin: MutOrigin,
    state_origin: MutOrigin,
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
    num_split_k: Int,
](TrivialRegisterPassable):
    """Context for warps that do work THEN advance (Load/Scheduler/Epilogue)."""

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
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


struct WaitAndAdvanceContextSplitK[
    work_origin: MutOrigin,
](TrivialRegisterPassable):
    """Context for waiting on CLC barrier and advancing work iterator (Split-K).

    Encapsulates the CLC response barrier synchronization:
    - Construction: Waits for CLC response, fetches next work
    - __enter__: Returns current work_info for processing
    - __exit__: Assigns fetched work as current
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
# WorkIteratorSplitK - Per-warp iterator encapsulating scheduler + pipeline state
# =============================================================================


struct WorkIteratorSplitK[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
    num_split_k: Int,
](TrivialRegisterPassable):
    """Per-warp work iterator for split-K that owns work_info and pipeline state.
    Throttle pipeline is obtained from the scheduler.
    """

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]
    comptime ThrottlePipeline = Self.SchedulerType.ThrottlePipeline

    var scheduler: Self.SchedulerType
    var work_info: WorkInfo
    var consumer_state: PipelineState[Self.num_stages]
    var throttle_pipeline: Self.ThrottlePipeline

    @always_inline
    fn __init__(out self, scheduler: Self.SchedulerType, work_info: WorkInfo):
        """Create work iterator. Throttle pipeline from scheduler."""
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
    ) -> AdvanceAfterWorkContextSplitK[
        origin_of(self.work_info),
        origin_of(self.consumer_state),
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Get next work item (advance AFTER work pattern)."""
        return AdvanceAfterWorkContextSplitK(
            self.scheduler,
            Pointer(to=self.work_info),
            Pointer(to=self.consumer_state),
        )

    @always_inline
    fn wait_and_advance[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> WaitAndAdvanceContextSplitK[
        origin_of(self.work_info)
    ]:
        """Wait for next work from CLC and advance iterator (Split-K).

        Encapsulates the CLC barrier wait:
        - __enter__: Waits for CLC response, returns current work
        - __exit__: Assigns fetched work as current
        """
        var next = self.scheduler.fetch_next_work(
            self.work_info, self.consumer_state
        )
        self.consumer_state.step()
        return WaitAndAdvanceContextSplitK(Pointer(to=self.work_info), next)

    # ========== CLC Throttle (Producer Side) ==========

    @always_inline
    fn throttle_signal(mut self, is_first_cta_in_cluster: Bool):
        """Signal CLC throttle if this is the first CTA in cluster.

        Args:
            is_first_cta_in_cluster: Only first CTA signals to avoid duplicates.
        """
        if is_first_cta_in_cluster:
            self.throttle_pipeline.producer_signal_and_step()


# =============================================================================
# SchedulerWorkIteratorSplitK - For Scheduler warp (split-K variant)
# =============================================================================


struct SchedulerWorkIteratorSplitK[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32],
    rasterize_order: RasterOrder,
    block_swizzle_size: Int,
    num_split_k: Int,
](TrivialRegisterPassable):
    """Work iterator for Scheduler warp (split-K) - owns work_info and both states.
    Throttle pipeline is obtained from the scheduler.
    """

    comptime SchedulerType = TileScheduler[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
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

    @always_inline
    fn next[
        state_origin: MutOrigin, //
    ](
        ref[state_origin] self,
    ) -> AdvanceAfterWorkContextSplitK[
        origin_of(self.work_info),
        origin_of(self.consumer_state),
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Get next work item."""
        return AdvanceAfterWorkContextSplitK(
            self.scheduler,
            Pointer(to=self.work_info),
            Pointer(to=self.consumer_state),
        )

    @always_inline
    fn signal_and_advance(mut self):
        """Signal CLC throttle consumer and advance to next work request."""
        self.throttle_pipeline.consumer_signal_and_step()
        self.producer_state = self.scheduler.advance_to_next_work(
            self.producer_state
        )

    @always_inline
    fn drain(mut self):
        """Drain all pending CLC requests before kernel exit."""

        comptime for i in range(Self.num_stages):
            # Split-K wraps underlying scheduler, so access via scheduler.scheduler
            self.scheduler.scheduler.empty_mbar[
                self.producer_state.index()
            ].wait(self.producer_state.phase())
            self.producer_state.step()


struct TileScheduler[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32] = Index[
        dtype = DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
    num_split_k: Int = 1,
](TrivialRegisterPassable):
    comptime UnderlyingScheduler = B200TileScheduler[
        Self.num_stages,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
    ]
    comptime BM = Self.reduction_tile_shape[0]
    comptime MMA_N = Self.reduction_tile_shape[1]
    comptime BK = Self.reduction_tile_shape[2]
    comptime ROW_SIZE = Self.MMA_N if Self.BM == 128 else Self.MMA_N // 2
    comptime ThrottlePipeline = Self.UnderlyingScheduler.ThrottlePipeline

    # Typed barrier array aliases (delegate to underlying scheduler)
    comptime ClcResponseArray = Self.UnderlyingScheduler.ClcResponseArray
    comptime ClcBarrierArray = Self.UnderlyingScheduler.ClcBarrierArray
    comptime ThrottleBarrierArray = Self.UnderlyingScheduler.ThrottleBarrierArray

    var locks_ptr: UnsafePointer[Int32]
    var scheduler: Self.UnderlyingScheduler
    var total_k_tiles: UInt32
    var k_tiles_per_split: UInt32
    var throttle_pipeline: Self.ThrottlePipeline

    @staticmethod
    fn init_throttle_barriers(
        storage_ptr: SMemPtr[SharedMemBarrier],
        producer_arv_count: Int32,
        consumer_arv_count: Int32,
    ):
        """Initialize throttle pipeline barriers. Called once by elect_one thread.
        """
        Self.UnderlyingScheduler.init_throttle_barriers(
            storage_ptr, producer_arv_count, consumer_arv_count
        )

    @always_inline
    fn __init__(
        out self,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        clc_response: Self.ClcResponseArray,
        clc_full: Self.ClcBarrierArray,
        clc_empty: Self.ClcBarrierArray,
        clc_throttle: Self.ThrottleBarrierArray,
        locks_ptr: UnsafePointer[UInt8],
    ):
        """Initialize from typed barrier arrays."""
        self.scheduler = Self.UnderlyingScheduler(
            cluster_dim,
            clc_response,
            clc_full,
            clc_empty,
            clc_throttle,
        )
        self.total_k_tiles = ceildiv(
            mnk[2], UInt32(Self.reduction_tile_shape[2])
        )
        self.k_tiles_per_split = ceildiv(
            self.total_k_tiles, UInt32(Self.num_split_k)
        )
        self.locks_ptr = locks_ptr.bitcast[Int32]()
        self.throttle_pipeline = Self.ThrottlePipeline(clc_throttle.ptr)

    @always_inline
    fn convert_to_splitk_work_info(self, work_info: B200WorkInfo) -> WorkInfo:
        var current_k_start = work_info.k_start * self.k_tiles_per_split
        var remaining_k_tiles = self.total_k_tiles - current_k_start
        return WorkInfo(
            work_info.m,
            work_info.n,
            current_k_start,
            min(self.k_tiles_per_split, remaining_k_tiles),
            work_info.is_valid_tile,
        )

    @always_inline
    fn initial_work_info(self) -> WorkInfo:
        return self.convert_to_splitk_work_info(
            self.scheduler.initial_work_info()
        )

    @always_inline
    fn advance_to_next_work(
        self,
        mut clc_state: PipelineState[Self.num_stages],
    ) -> PipelineState[Self.num_stages]:
        return self.scheduler.advance_to_next_work(clc_state)

    @always_inline
    fn fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[Self.num_stages],
    ) -> WorkInfo:
        var underlying_workinfo = B200WorkInfo(
            work_info.m, work_info.n, work_info.k_start, work_info.is_valid_tile
        )
        return self.convert_to_splitk_work_info(
            self.scheduler.fetch_next_work(underlying_workinfo, consumer_state)
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
    ) -> AdvanceAfterWorkContextSplitK[
        work_origin,
        state_origin,
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Context for warps that do work THEN advance (Load/Scheduler/Epilogue).

        Usage:
            with scheduler.advance_after_work(work_info, state) as current:
                do_work(current)
                syncwarp()
            # After: work_info updated, state stepped
        """
        return AdvanceAfterWorkContextSplitK(
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
    ) -> WaitAndAdvanceContextSplitK[work_origin]:
        """Wait for next work from CLC and advance (Split-K).

        Encapsulates the CLC barrier wait (called on scheduler directly).

        Usage:
            with scheduler.wait_and_advance_work(work_info, state) as current:
                do_mma(current)
            # After: work_info updated to next value
        """
        var next = self.fetch_next_work(work_info, consumer_state)
        consumer_state.step()
        return WaitAndAdvanceContextSplitK(Pointer(to=work_info), next)

    @always_inline
    fn work_iterator(
        self,
    ) -> WorkIteratorSplitK[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Create a per-warp work iterator that owns work_info internally.
        Throttle pipeline is obtained from the scheduler.
        """
        return WorkIteratorSplitK(self, self.initial_work_info())

    @always_inline
    fn scheduler_iterator(
        self,
    ) -> SchedulerWorkIteratorSplitK[
        Self.num_stages,
        Self.reduction_tile_shape,
        Self.cluster_shape,
        Self.rasterize_order,
        Self.block_swizzle_size,
        Self.num_split_k,
    ]:
        """Create iterator for Scheduler warp (owns work_info and both states).
        Throttle pipeline is obtained from the scheduler.
        """
        return SchedulerWorkIteratorSplitK(self, self.initial_work_info())

    @always_inline
    fn is_last_split(self, work_tile_info: WorkInfo) -> Bool:
        return work_tile_info.is_valid() and work_tile_info.is_final_split(
            self.total_k_tiles
        )

    @always_inline
    fn output_tile_index(self, work_info: WorkInfo) -> UInt32:
        return work_info.m * UInt32(grid_dim.y) + work_info.n

    comptime WorkspaceTileLayout = static_row_major[Self.BM, Self.MMA_N]

    @always_inline
    fn _get_workspace_tile[
        accum_type: DType, workspace_layout: TensorLayout
    ](
        self,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        reduction_tile_idx: UInt32,
    ) -> TileTensor[accum_type, Self.WorkspaceTileLayout, MutAnyOrigin]:
        var offset = reduction_tile_idx * UInt32(Self.BM) * UInt32(Self.MMA_N)
        return TileTensor[accum_type, Self.WorkspaceTileLayout, MutAnyOrigin](
            UnsafePointer(to=reduction_workspace.ptr[Int(offset)]),
            row_major[Self.BM, Self.MMA_N](),
        )

    @always_inline
    @staticmethod
    fn _get_max_width_per_stage[max_width: Int]() -> Int:
        return min(max_width, Self.ROW_SIZE & -Self.ROW_SIZE)

    @always_inline
    @staticmethod
    fn _get_widths_per_stage[
        max_width: Int
    ]() -> Tuple[InlineArray[Int, 4], Int]:
        """helper functions to decompose MMA_N into widths that are powers of two
        """
        var arr = InlineArray[Int, 4](uninitialized=True)
        var current_width = Self.ROW_SIZE
        var first_width: Int
        var second_width: Int

        var i = 0
        while current_width > 0:
            first_width = min(max_width, prev_power_of_two(current_width))
            second_width = current_width - first_width
            arr[i] = first_width
            i += 1
            current_width = second_width

        return (arr^, i)

    @always_inline
    @staticmethod
    fn _to_next_subtile[
        accum_type: DType,
        tile_layout: TensorLayout,
        /,
        *,
        widths: InlineArray[Int, 4],
        curr_stage: Int,
    ](
        tensor: TileTensor[accum_type, tile_layout, MutAnyOrigin],
    ) -> TileTensor[
        accum_type,
        # Shape narrows to [height, stage_width], but stride is preserved
        # from the parent [parent_stride, 1] -- NOT row_major of the
        # narrowed shape. The sub-tile is a strided view into wider rows.
        _StridedLayout[
            tile_layout.static_shape[0],
            widths[curr_stage],
            tile_layout.static_stride[0],
        ],
        MutAnyOrigin,
    ]:
        @parameter
        fn _get_current_width(
            widths: InlineArray[Int, 4], curr_stage: Int
        ) -> Int:
            var width = 0
            for i in range(curr_stage):
                width += widths[i]
            return width

        comptime current_width = _get_current_width(widths, curr_stage)

        return TileTensor[
            accum_type,
            _StridedLayout[
                tile_layout.static_shape[0],
                widths[curr_stage],
                tile_layout.static_stride[0],
            ],
            MutAnyOrigin,
        ](
            UnsafePointer(to=tensor.ptr[current_width]),
            _strided_layout[
                tile_layout.static_shape[0],
                widths[curr_stage],
                tile_layout.static_stride[0],
            ](),
        )

    @always_inline
    fn store_to_workspace[
        accum_type: DType,
        workspace_layout: TensorLayout,
        /,
        *,
        do_reduction: Bool = False,
        write_back: Bool = False,
    ](
        self,
        tmem: TmemAddress,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        epilogue_thread_idx: UInt,
        reduction_tile_idx: UInt32,
    ):
        # 128 is a magic number that is provided by the NVCC backend.
        # register size that is greater than that will not compile.
        comptime widths_per_stage = Self._get_widths_per_stage[128]()
        comptime widths = widths_per_stage[0]
        comptime num_widths = widths_per_stage[1]

        # TmemTensor for split-K reduction.
        # Use cta_group=2 to force is_lower_required=True - split-K always
        # needs both upper and lower fragments for the full reduction.
        comptime accum_layout = Layout.row_major(Self.BM, Self.ROW_SIZE)
        comptime AccumTmem = TmemTensor[accum_type, accum_layout, cta_group=2]

        var local_warp_id = epilogue_thread_idx // UInt(WARP_SIZE)

        # workspace has layout (X, BM, MMA_N)
        var workspace_tile = self._get_workspace_tile(
            reduction_workspace, reduction_tile_idx
        )

        comptime REDUCTION_BM = Self.BM // 4 if Self.BM == 128 else Self.BM // 2
        comptime REDUCTION_BN = Self.MMA_N if Self.BM == 128 else Self.MMA_N // 2
        var warp_id_x = local_warp_id if Self.BM == 128 else local_warp_id % 2
        var warp_id_y = 0 if Self.BM == 128 else local_warp_id // 2

        var reduction_frag = workspace_tile.tile[REDUCTION_BM, REDUCTION_BN](
            Coord(Idx(Int(warp_id_x)), Idx(Int(warp_id_y)))
        )
        var reduction_upper = reduction_frag.tile[16, REDUCTION_BN](
            Coord(Idx(0), Idx(0))
        )
        var reduction_lower = reduction_frag.tile[16, REDUCTION_BN](
            Coord(Idx(1), Idx(0))
        )
        var stage_addr = tmem  # Track address for iteration

        comptime for stage in range(num_widths):
            comptime stage_width = widths[stage]
            comptime stage_rep = stage_width // 8

            var stage_tmem = AccumTmem(stage_addr)
            var frags = stage_tmem.load_fragments[stage_rep]()
            AccumTmem.wait_load()

            # Get workspace subtiles for this stage
            var ws_upper = (
                Self._to_next_subtile[widths=widths, curr_stage=stage](
                    reduction_upper
                )
                .vectorize[1, 2]()
                .distribute[row_major[8, 4]()](Int(lane_id()))
            )
            var ws_lower = (
                Self._to_next_subtile[widths=widths, curr_stage=stage](
                    reduction_lower
                )
                .vectorize[1, 2]()
                .distribute[row_major[8, 4]()](Int(lane_id()))
            )

            comptime num_m = type_of(ws_upper).static_shape[0]
            comptime num_n = type_of(ws_upper).static_shape[1]

            comptime for m in range(num_m):
                comptime for n in range(num_n):
                    comptime i = m * num_n + n

                    var v2_upper = rebind[type_of(ws_upper).ElementType](
                        SIMD[accum_type, 2](
                            frags.upper[2 * i], frags.upper[2 * i + 1]
                        )
                    )
                    var v2_lower = rebind[type_of(ws_lower).ElementType](
                        SIMD[accum_type, 2](
                            frags.lower[2 * i], frags.lower[2 * i + 1]
                        )
                    )

                    comptime if do_reduction:
                        v2_upper += ws_upper[m, n]
                        v2_lower += ws_lower[m, n]

                    comptime if write_back:
                        ws_upper[m, n] = v2_upper
                        ws_lower[m, n] = v2_lower
                    else:
                        frags.upper[2 * i] = v2_upper[0]
                        frags.upper[2 * i + 1] = v2_upper[1]
                        frags.lower[2 * i] = v2_lower[0]
                        frags.lower[2 * i + 1] = v2_lower[1]

            # Store modified fragments back to TMEM
            comptime if not write_back:
                stage_tmem.store_fragments[stage_rep](frags)
                AccumTmem.wait_store()

            stage_addr = stage_addr + stage_width

    @always_inline
    fn reduction[
        accum_type: DType,
        workspace_layout: TensorLayout,
    ](
        self,
        reduction_workspace: TileTensor[
            accum_type, workspace_layout, MutAnyOrigin
        ],
        tmem: TmemAddress,
        epilogue_thread_idx: UInt,
        work_info: WorkInfo,
    ) -> Bool:
        var reduction_tile_idx = self.output_tile_index(work_info)

        var lock_idx = reduction_tile_idx

        if not self.is_last_split(work_info):
            if work_info.k_start == 0:
                # first split don't wait and just write to workspace.
                self.store_to_workspace[do_reduction=False, write_back=True](
                    tmem,
                    reduction_workspace,
                    epilogue_thread_idx,
                    reduction_tile_idx,
                )
            else:
                Self.wait_eq(
                    self.locks_ptr,
                    0,
                    Int(epilogue_thread_idx),
                    lock_idx,
                    work_info.k_start,
                )

                self.store_to_workspace[do_reduction=True, write_back=True](
                    tmem,
                    reduction_workspace,
                    epilogue_thread_idx,
                    reduction_tile_idx,
                )

            var increment = work_info.num_k_tiles + work_info.k_start

            Self.arrive_set(
                self.locks_ptr,
                0,
                Int(epilogue_thread_idx),
                lock_idx,
                increment,
            )

            return False
        else:
            Self.wait_eq(
                self.locks_ptr,
                0,
                Int(epilogue_thread_idx),
                lock_idx,
                work_info.k_start,
            )
            self.store_to_workspace[do_reduction=True, write_back=False](
                tmem,
                reduction_workspace,
                epilogue_thread_idx,
                reduction_tile_idx,
            )

            return True

    @always_inline
    @staticmethod
    fn wait_eq(
        lock_ptr: UnsafePointer[Int32],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        val: UInt32,
    ):
        var sema = NamedBarrierSemaphore[Int32(WARPGROUP_SIZE), 4, 1](
            lock_ptr + lock_idx, barrier_group_thread_idx
        )
        sema.wait_eq(barrier_id, Int32(val))

    @staticmethod
    @always_inline
    fn wait_lt(
        lock_ptr: UnsafePointer[Int32],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        count: UInt32,
    ):
        pass

    @staticmethod
    @always_inline
    fn arrive_set(
        lock_ptr: UnsafePointer[Int32],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        val: UInt32,
    ):
        var sema = NamedBarrierSemaphore[Int32(WARPGROUP_SIZE), 4, 1](
            lock_ptr + lock_idx, barrier_group_thread_idx
        )
        sema.arrive_set(barrier_id, Int32(val))


@always_inline
fn get_num_tiles(
    problem_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[2],
) -> IndexList[2]:
    var num_block_m = ceildiv(problem_shape[0], block_tile_shape[0])
    var num_block_n = ceildiv(problem_shape[1], block_tile_shape[1])

    var problem_blocks_m = align_up(num_block_m, cluster_shape[0])
    var problem_blocks_n = align_up(num_block_n, cluster_shape[1])

    return Index(problem_blocks_m, problem_blocks_n)


@always_inline
fn get_required_locks_buffer_size_bytes[
    accum_type: DType
](
    problem_shape: IndexList[3],
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[2],
) -> Int:
    var problem_blocks = get_num_tiles(
        problem_shape, block_tile_shape, cluster_shape
    )
    var num_output_tiles = problem_blocks[0] * problem_blocks[1]

    var locks_workspace_bytes = num_output_tiles * size_of[Int32]()

    return locks_workspace_bytes
