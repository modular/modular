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
from gpu.memory import AddressSpace
from layout.layout_tensor import LayoutTensorIter
from .tile_scheduler import TileScheduler as B200TileScheduler
from .tile_scheduler import WorkInfo as B200WorkInfo
from ..tile_scheduler import RasterOrder
from layout.tma_async import SharedMemBarrier, PipelineState
from utils.static_tuple import StaticTuple
from gpu.id import grid_dim, thread_idx, lane_id
from gpu.cluster import elect_one_sync
from gpu import NamedBarrierSemaphore, WARP_SIZE
from gpu.globals import WARPGROUP_SIZE
from gpu.tcgen05 import *
from gpu.sync import named_barrier
from memory import LegacyUnsafePointer as UnsafePointer
from stdlib.bit import prev_power_of_two


@fieldwise_init
@register_passable("trivial")
struct WorkInfo(ImplicitlyCopyable, Stringable, Writable):
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


@register_passable("trivial")
struct TileScheduler[
    num_stages: Int,
    reduction_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32] = Index[
        dtype = DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
    num_split_k: Int = 1,
]:
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

    var locks_ptr: UnsafePointer[Int32]
    var scheduler: Self.UnderlyingScheduler
    var total_k_tiles: UInt32
    var k_tiles_per_split: UInt32

    @always_inline
    fn __init__(
        out self,
        cluster_dim: StaticTuple[Int32, 3],
        mnk: StaticTuple[UInt32, 3],
        clc_response_ptr: UnsafePointer[
            UInt128, address_space = AddressSpace.SHARED
        ],
        full_mbar_ptr: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED
        ],
        empty_mbar_ptr: UnsafePointer[
            SharedMemBarrier, address_space = AddressSpace.SHARED
        ],
        locks_ptr: UnsafePointer[UInt8],
    ):
        self.scheduler = Self.UnderlyingScheduler(
            cluster_dim, clc_response_ptr, full_mbar_ptr, empty_mbar_ptr
        )
        self.total_k_tiles = ceildiv(mnk[2], Self.reduction_tile_shape[2])
        self.k_tiles_per_split = ceildiv(self.total_k_tiles, Self.num_split_k)
        self.locks_ptr = locks_ptr.bitcast[Int32]()

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

    @always_inline
    fn is_last_split(self, work_tile_info: WorkInfo) -> Bool:
        return work_tile_info.is_valid() and work_tile_info.is_final_split(
            self.total_k_tiles
        )

    @always_inline
    fn output_tile_index(self, work_info: WorkInfo) -> UInt32:
        return work_info.m * grid_dim.y + work_info.n

    @always_inline
    fn _get_workspace_tile[
        accum_type: DType, workspace_layout: Layout
    ](
        self,
        reduction_workspace: LayoutTensor[accum_type, workspace_layout],
        reduction_tile_idx: UInt32,
        out result: LayoutTensor[
            accum_type, Layout.row_major(Self.BM, Self.MMA_N), MutAnyOrigin
        ],
    ):
        return type_of(result)(
            reduction_workspace.ptr
            + (reduction_tile_idx * Self.BM * Self.MMA_N)
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

        return (arr, i)

    @staticmethod
    fn _get_new_layout[
        layout: Layout,
        width: Int,
    ]() -> Layout:
        comptime new_shape = IntTuple(layout.shape[0].value(), width)
        return Layout(new_shape, layout.stride)

    @always_inline
    @staticmethod
    fn _to_next_subtile[
        accum_type: DType,
        layout: Layout,
        /,
        *,
        widths: InlineArray[Int, 4],
        curr_stage: Int,
    ](
        tensor: LayoutTensor[accum_type, layout, MutAnyOrigin, **_],
        out result: LayoutTensor[
            accum_type,
            Self._get_new_layout[layout, widths[curr_stage]](),
            MutAnyOrigin,
            address_space = type_of(tensor).address_space,
        ],
    ):
        @parameter
        fn _get_current_width(
            widths: InlineArray[Int, 4], curr_stage: Int
        ) -> Int:
            var width = 0
            for i in range(curr_stage):
                width += widths[i]
            return width

        comptime current_width = _get_current_width(widths, curr_stage)

        return type_of(result)(tensor.ptr + current_width)

    @always_inline
    fn store_to_workspace[
        accum_type: DType,
        workspace_layout: Layout,
        /,
        *,
        do_reduction: Bool = False,
        write_back: Bool = False,
    ](
        self,
        tmem_addr: UInt32,
        reduction_workspace: LayoutTensor[accum_type, workspace_layout],
        epilogue_thread_idx: UInt,
        reduction_tile_idx: UInt32,
    ):
        comptime data_paths = 16  # same as lanes
        comptime bits = 256
        # only load from TMEM when not using split-k
        comptime total_rep = Self.ROW_SIZE // 8

        # 128 is a magic number that is provided by the NVCC backend.
        # register size that is greater than that will not compile.
        comptime widths_per_stage = Self._get_widths_per_stage[128]()
        comptime widths = widths_per_stage[0]
        comptime num_widths = widths_per_stage[1]

        # alias stage_rep = width_per_stage // 8
        comptime fragment_size = (data_paths * (bits // 32)) // WARP_SIZE
        # alias stage_frag_size = stage_rep * fragment_size

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
            Int(warp_id_x), Int(warp_id_y)
        )
        var reduction_upper = reduction_frag.tile[16, REDUCTION_BN](0, 0)
        var reduction_lower = reduction_frag.tile[16, REDUCTION_BN](1, 0)
        var stage_tmem_addr = tmem_addr

        @parameter
        for stage in range(num_widths):
            comptime stage_width = widths[stage]
            comptime stage_rep = stage_width // 8
            comptime stage_frag_size = stage_rep * fragment_size

            var stage_frag_upper = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=stage_rep,
                dtype=accum_type,
                pack=False,
                width=stage_frag_size,
            ](stage_tmem_addr)

            var stage_frag_lower = tcgen05_ld[
                datapaths=data_paths,
                bits=bits,
                repeat=stage_rep,
                dtype=accum_type,
                pack=False,
                width=stage_frag_size,
            ](stage_tmem_addr + (16 << 16))
            tcgen05_load_wait()

            var reduction_upper_subtile = Self._to_next_subtile[
                widths=widths, curr_stage=stage
            ](reduction_upper)
            var reduction_lower_subtile = Self._to_next_subtile[
                widths=widths, curr_stage=stage
            ](reduction_lower)

            var reduction_frag_upper = reduction_upper_subtile.vectorize[
                1, 2
            ]().distribute[Layout.row_major(8, 4)](lane_id())

            var reduction_frag_lower = reduction_lower_subtile.vectorize[
                1, 2
            ]().distribute[Layout.row_major(8, 4)](lane_id())

            comptime num_m = reduction_frag_upper.layout.shape[0].value()
            comptime num_n = reduction_frag_upper.layout.shape[1].value()

            @parameter
            for m in range(num_m):

                @parameter
                for n in range(num_n):
                    comptime i = m * num_n + n

                    var v2_upper = rebind[reduction_frag_upper.element_type](
                        SIMD[accum_type, 2](
                            stage_frag_upper[2 * i], stage_frag_upper[2 * i + 1]
                        )
                    )
                    var v2_lower = rebind[reduction_frag_lower.element_type](
                        SIMD[accum_type, 2](
                            stage_frag_lower[2 * i], stage_frag_lower[2 * i + 1]
                        )
                    )

                    @parameter
                    if do_reduction:
                        v2_upper += reduction_frag_upper[m, n]
                        v2_lower += reduction_frag_lower[m, n]

                    @parameter
                    if write_back:
                        reduction_frag_upper[m, n] = v2_upper
                        reduction_frag_lower[m, n] = v2_lower
                    else:
                        stage_frag_upper[2 * i] = v2_upper[0]
                        stage_frag_upper[2 * i + 1] = v2_upper[1]
                        stage_frag_lower[2 * i] = v2_lower[0]
                        stage_frag_lower[2 * i + 1] = v2_lower[1]

            # we can't hold all accumulators in registers, so we need to store to TMEM
            @parameter
            if not write_back:
                tcgen05_st[
                    datapaths=data_paths,
                    bits=bits,
                    repeat=stage_rep,
                    pack=False,
                ](stage_tmem_addr, stage_frag_upper)
                tcgen05_st[
                    datapaths=data_paths,
                    bits=bits,
                    repeat=stage_rep,
                    pack=False,
                ](stage_tmem_addr + (16 << 16), stage_frag_lower)
                tcgen05_store_wait()

            stage_tmem_addr += stage_width

    @always_inline
    fn reduction[
        accum_type: DType,
        workspace_layout: Layout,
    ](
        self,
        reduction_workspace: LayoutTensor[accum_type, workspace_layout],
        tmem_addr: UInt32,
        epilogue_thread_idx: UInt,
        work_info: WorkInfo,
    ) -> Bool:
        var reduction_tile_idx = self.output_tile_index(work_info)

        var lock_idx = reduction_tile_idx

        if not self.is_last_split(work_info):
            if work_info.k_start == 0:
                # first split don't wait and just write to workspace.
                self.store_to_workspace[do_reduction=False, write_back=True](
                    tmem_addr,
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
                    tmem_addr,
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
                tmem_addr,
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
            lock_ptr.offset(lock_idx), barrier_group_thread_idx
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
            lock_ptr.offset(lock_idx), barrier_group_thread_idx
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

    return Int(locks_workspace_bytes)
