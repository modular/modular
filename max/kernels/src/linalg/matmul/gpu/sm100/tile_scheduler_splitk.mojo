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


@fieldwise_init
@register_passable("trivial")
struct WorkInfo(ImplicitlyCopyable, Movable, Stringable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    var num_k_tiles: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    alias INVALID_WORK_INFO = Self(0, 0, 0, 0, False)

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
    block_tile_shape: IndexList[3],
    cluster_shape: IndexList[3, element_type = DType.uint32] = Index[
        dtype = DType.uint32
    ](1, 1, 1),
    rasterize_order: RasterOrder = RasterOrder.AlongM,
    block_swizzle_size: Int = 8,
    num_split_k: Int = 1,
]:
    alias UnderlyingScheduler = B200TileScheduler[
        num_stages, cluster_shape, rasterize_order, block_swizzle_size
    ]
    alias BM = block_tile_shape[0]
    alias BN = block_tile_shape[1]
    alias BK = block_tile_shape[2]

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
        locks_ptr: UnsafePointer[Int8],
    ):
        self.scheduler = Self.UnderlyingScheduler(
            cluster_dim, clc_response_ptr, full_mbar_ptr, empty_mbar_ptr
        )
        self.total_k_tiles = ceildiv(mnk[2], block_tile_shape[2])
        self.k_tiles_per_split = ceildiv(self.total_k_tiles, num_split_k)
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
        mut clc_state: PipelineState[num_stages],
    ) -> PipelineState[num_stages]:
        return self.scheduler.advance_to_next_work(clc_state)

    @always_inline
    fn fetch_next_work(
        self,
        work_info: WorkInfo,
        consumer_state: PipelineState[num_stages],
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
            accum_type, Layout.row_major(Self.BM, Self.BN), MutableAnyOrigin
        ],
    ):
        return type_of(result)(
            reduction_workspace.ptr + (reduction_tile_idx * Self.BM * Self.BN)
        )

    @always_inline
    fn store_to_workspace[
        accum_type: DType,
        size: Int,
        workspace_layout: Layout,
        /,
        *,
        repeat: Int,
        do_reduction: Bool = False,
        write_back: Bool = False,
    ](
        self,
        mut reg_tile_upper: SIMD[accum_type, size],
        mut reg_tile_lower: SIMD[accum_type, size],
        tmem_addr: UInt32,
        reduction_workspace: LayoutTensor[accum_type, workspace_layout],
        epilogue_thread_idx: UInt,
        reduction_tile_idx: UInt32,
    ):
        var local_warp_id = epilogue_thread_idx // UInt(WARP_SIZE)
        reg_tile_upper = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat=repeat,
            dtype=accum_type,
            pack=False,
            width=size,
        ](tmem_addr)

        reg_tile_lower = tcgen05_ld[
            datapaths=16,
            bits=256,
            repeat=repeat,
            dtype=accum_type,
            pack=False,
            width=size,
        ](tmem_addr + (16 << 16))
        tcgen05_load_wait()

        # workspace has layout (X, BM, BN)
        var workspace_tile = self._get_workspace_tile(
            reduction_workspace, reduction_tile_idx
        )

        var reduction_frag = workspace_tile.tile[Self.BM // 4, Self.BN](
            Int(local_warp_id), 0
        )

        var reduction_frag_upper = (
            reduction_frag.tile[16, Self.BN](0, 0)
            .vectorize[1, 2]()
            .distribute[Layout.row_major(8, 4)](lane_id())
        )
        var reduction_frag_lower = (
            reduction_frag.tile[16, Self.BN](1, 0)
            .vectorize[1, 2]()
            .distribute[Layout.row_major(8, 4)](lane_id())
        )

        alias num_m = reduction_frag_upper.layout.shape[0].value()
        alias num_n = reduction_frag_upper.layout.shape[1].value()

        @parameter
        for m in range(num_m):

            @parameter
            for n in range(num_n):
                alias i = m * num_n + n

                var v2_upper = rebind[reduction_frag_upper.element_type](
                    SIMD[accum_type, 2](
                        reg_tile_upper[2 * i], reg_tile_upper[2 * i + 1]
                    )
                )
                var v2_lower = rebind[reduction_frag_lower.element_type](
                    SIMD[accum_type, 2](
                        reg_tile_lower[2 * i], reg_tile_lower[2 * i + 1]
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
                    reg_tile_upper[2 * i] = v2_upper[0]
                    reg_tile_upper[2 * i + 1] = v2_upper[1]
                    reg_tile_lower[2 * i] = v2_lower[0]
                    reg_tile_lower[2 * i + 1] = v2_lower[1]

    @always_inline
    fn reduction[
        accum_type: DType,
        size: Int,
        workspace_layout: Layout,
        /,
        *,
        repeat: Int,
    ](
        self,
        mut reg_tile_upper: SIMD[accum_type, size],
        mut reg_tile_lower: SIMD[accum_type, size],
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
                self.store_to_workspace[
                    repeat=repeat, do_reduction=False, write_back=True
                ](
                    reg_tile_upper,
                    reg_tile_lower,
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

                self.store_to_workspace[
                    repeat=repeat, do_reduction=True, write_back=True
                ](
                    reg_tile_upper,
                    reg_tile_lower,
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
            self.store_to_workspace[
                repeat=repeat, do_reduction=True, write_back=False
            ](
                reg_tile_upper,
                reg_tile_lower,
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
