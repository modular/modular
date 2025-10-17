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

from math import align_up, ceildiv
from os.atomic import Atomic
from sys import size_of

from gpu import NamedBarrierSemaphore
from gpu.globals import WARPGROUP_SIZE
from gpu.host.info import H100
from gpu.id import block_idx, grid_dim, thread_idx
from gpu.memory import AddressSpace
from layout import Layout, LayoutTensor
from layout.runtime_layout import RuntimeLayout
from stdlib.bit import log2_floor

from utils.index import Index, IndexList

from .tile_scheduler import RasterOrder, WorkInfo

from ...structuring import RegTileType


@always_inline("nodebug")
fn _check_scheduler_constraints[
    prob_shape_nk: IndexList[2],
    tile_shape: IndexList[3],
    splits: UInt32,
    num_consumer: UInt32,
    num_pipeline_stages: UInt32,
    cluster_shape: IndexList[2],
    raster_order: RasterOrder,
    reduction_mode: ReductionMode,
]():
    alias num_k_iters = ceildiv(prob_shape_nk[1], tile_shape[2])

    constrained[
        reduction_mode == ReductionMode.Deterministic,
        "Currently SplitK only supports Deterministic reduction",
    ]()

    constrained[
        splits <= H100.sm_count,
        "splits must be less than or equal to the number of SMs",
    ]()

    constrained[
        splits <= num_k_iters,
        "splits must be less than or equal to the number of output tiles",
    ]()
    constrained[(num_k_iters % splits) == 0, "BK must be divisible by splits"]()


@fieldwise_init
@register_passable("trivial")
struct ReductionMode(ImplicitlyCopyable, Movable):
    var _value: Int32

    # CTAs perform reduction in a serialized fashion so we will have deterministic numeric behavior
    alias Deterministic = Self(0)

    # CTAs perform reduction atomically but we will have nondeterministic numeric behavior
    alias Nondeterministic = Self(1)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct SplitKTileScheduler[
    problem_shape_nk: IndexList[2],
    tile_shape: IndexList[3],
    splits: UInt32,
    num_consumer: UInt32,
    num_pipeline_stages: UInt32,
    cluster_shape: IndexList[2],
    raster_order: RasterOrder,
    reduction_mode: ReductionMode = ReductionMode.Deterministic,
]:
    var prob_shape: IndexList[3]  # M x N x K
    var block_id_in_cluster: IndexList[2]
    var blocks_per_problem: UInt32
    var current_work_linear_idx: UInt32

    # Like tile blocks that are in a 2D grid and have `m` and `n` indices and a they have a notion of major and minor for each dimension,
    # block clusters can also be thought of as a smaller 2D grid of sub-blocks.
    # Therefore, we have a notion of major and minor dimensions for each dimension of a block cluster.
    # For example, when we rasterize along N, the major dimension is blocks on the N dimension and the minor dimension is blocks on the M dimension.
    # and therefore, cluster_shape_major and cluster_shape_minor for sub blocks in the each cluster will be CLUSTER_N and CLUSTER_M, respectively.
    var log_cluster_shape_major: UInt32
    var log_cluster_shape_minor: UInt32

    var cluster_blk_major: UInt32

    var locks_ptr: UnsafePointer[Int32]

    alias k_tiles_per_output_tile = ceildiv(problem_shape_nk[1], tile_shape[2])
    # we don't support uneven splits so for num_iters per split can be compile time constant
    alias k_tiles_per_split = ceildiv(
        problem_shape_nk[1], tile_shape[2]
    ) // splits
    # cluster size is power of 2 (1, 2 ,4)
    alias log_cluster_size = log2_floor(cluster_shape[0] * cluster_shape[1])

    alias WorkTileType[dtype: DType, layout: Layout] = LayoutTensor[
        dtype, layout, MutableAnyOrigin
    ]

    @always_inline
    fn __init__(
        out self,
        prob_shape: IndexList[3],
        block_id_in_cluster: IndexList[2],
        locks_ptr: UnsafePointer[NoneType],
    ):
        _check_scheduler_constraints[
            problem_shape_nk,
            tile_shape,
            splits,
            num_consumer,
            num_pipeline_stages,
            cluster_shape,
            raster_order,
            reduction_mode,
        ]()

        self.prob_shape = prob_shape
        self.block_id_in_cluster = block_id_in_cluster

        self.locks_ptr = locks_ptr.bitcast[Int32]()

        var problem_blocks = Self.get_problem_blocks_shape(
            prob_shape, tile_shape, cluster_shape
        )
        var problem_blocks_m = align_up(
            UInt(problem_blocks[0]),
            UInt(self.cluster_shape[0]),
        )
        var problem_blocks_n = align_up(
            UInt(problem_blocks[1]),
            UInt(self.cluster_shape[1]),
        )

        @parameter
        if raster_order == RasterOrder.AlongN:
            self.current_work_linear_idx = (
                block_idx.x + grid_dim.x * block_idx.y
            )
            self.log_cluster_shape_major = log2_floor(self.cluster_shape[1])
            self.log_cluster_shape_minor = log2_floor(self.cluster_shape[0])
            self.cluster_blk_major = (
                problem_blocks_n >> self.log_cluster_shape_major
            )

        else:  # rasterize along M
            self.current_work_linear_idx = (
                block_idx.x * grid_dim.y + block_idx.y
            )
            self.log_cluster_shape_major = log2_floor(self.cluster_shape[0])
            self.log_cluster_shape_minor = log2_floor(self.cluster_shape[1])
            self.cluster_blk_major = (
                problem_blocks_m >> self.log_cluster_shape_major
            )

        self.blocks_per_problem = problem_blocks_m * problem_blocks_n

    @always_inline
    fn get_sm_num(self) -> UInt32:
        @parameter
        if raster_order == RasterOrder.AlongN:
            return block_idx.x + grid_dim.x * block_idx.y
        else:
            return block_idx.x * grid_dim.y + block_idx.y

    @staticmethod
    @always_inline
    fn get_problem_blocks_shape(
        problem_shape: IndexList[3],
        dyn_tile_shape: IndexList[3],
        dyn_cluster_shape: IndexList[2],
    ) -> IndexList[2]:
        var num_blocks_m = (
            problem_shape[0] + dyn_tile_shape[0] - 1
        ) // dyn_tile_shape[0]
        var num_blocks_n = (
            problem_shape[1] + dyn_tile_shape[1] - 1
        ) // dyn_tile_shape[1]

        var problem_blocks_m = (
            (num_blocks_m + dyn_cluster_shape[0] - 1) // dyn_cluster_shape[0]
        ) * dyn_cluster_shape[0]
        var problem_blocks_n = (
            (num_blocks_n + dyn_cluster_shape[1] - 1) // dyn_cluster_shape[1]
        ) * dyn_cluster_shape[1]

        return IndexList[2](
            problem_blocks_m,
            problem_blocks_n,
        )

    @always_inline
    fn initial_work_tile_info(mut self) -> WorkInfo:
        return self.get_current_work_info()

    @always_inline
    fn get_current_work_info(mut self) -> WorkInfo:
        if self.current_work_linear_idx >= self.blocks_per_problem * splits:
            return WorkInfo.INVALID_WORK_INFO

        var work_tile_info = WorkInfo()
        self.assign_work(work_tile_info, self.current_work_linear_idx)

        work_tile_info.is_valid_tile = True

        return work_tile_info

    @always_inline
    fn get_worktile_m_n_idx(
        mut self,
        mut work_tile_info: WorkInfo,
        linear_tile_id: UInt32,
    ):
        var rank_m_in_cluster = self.block_id_in_cluster[0]
        var rank_n_in_cluster = self.block_id_in_cluster[1]

        var tile_id = linear_tile_id % self.blocks_per_problem

        var cta_per_grid_dim = tile_id >> self.log_cluster_shape_minor

        var cluster_id = cta_per_grid_dim >> self.log_cluster_shape_major
        var cluster_major_offset = cta_per_grid_dim & (
            (1 << self.log_cluster_shape_major) - 1
        )

        var cluster_minor_offset: UInt32

        @parameter
        if self.raster_order == RasterOrder.AlongN:
            cluster_minor_offset = rank_m_in_cluster
        else:
            cluster_minor_offset = rank_n_in_cluster

        var cluster_idx_minor = cluster_id / self.cluster_blk_major
        var cluster_idx_major = cluster_id % self.cluster_blk_major

        var minor_work_idx = (
            cluster_idx_minor << self.log_cluster_shape_minor
        ) + cluster_minor_offset

        var major_work_idx = (
            cluster_idx_major << self.log_cluster_shape_major
        ) + cluster_major_offset

        var work_idx_m: UInt32
        var work_idx_n: UInt32

        @parameter
        if self.raster_order == RasterOrder.AlongN:
            work_idx_m = minor_work_idx
            work_idx_n = major_work_idx
        else:
            work_idx_m = major_work_idx
            work_idx_n = minor_work_idx

        work_tile_info.m = work_idx_m
        work_tile_info.n = work_idx_n

    @always_inline
    fn assign_work(mut self, mut work_tile_info: WorkInfo, linear_idx: UInt32):
        var linear_tile_id = self.get_k_start_and_linear_tile_id(
            work_tile_info, linear_idx
        )

        self.get_worktile_m_n_idx(work_tile_info, linear_tile_id)

    @always_inline
    fn get_k_start_and_linear_tile_id(
        mut self, mut work_tile_info: WorkInfo, linear_idx: UInt32
    ) -> UInt32:
        var linear_cluster_id = linear_idx >> self.log_cluster_size
        var num_tile_clusters = self.blocks_per_problem >> self.log_cluster_size

        var split = linear_cluster_id / num_tile_clusters
        var cluster_linear_idx = linear_cluster_id % num_tile_clusters

        # Bring the linearized tile ID back into the space of tiles, rather than clusters
        var linear_tile_id = cluster_linear_idx << self.log_cluster_size

        var rank_m_in_cluster = self.block_id_in_cluster[0]
        var rank_n_in_cluster = self.block_id_in_cluster[1]

        # The final linearized tile ID is in units of the cluster dimension over which we rasterize.
        @parameter
        if self.raster_order == RasterOrder.AlongN:
            linear_tile_id += rank_n_in_cluster << self.log_cluster_shape_minor
        else:
            linear_tile_id += rank_m_in_cluster << self.log_cluster_shape_minor

        work_tile_info.k_start = self.k_tiles_per_split * split
        work_tile_info.num_k_tiles = (
            ceildiv(problem_shape_nk[1], tile_shape[2]) // splits
        )

        return linear_tile_id  # basically linear index of the output tile

    @always_inline
    fn fetch_next_work(mut self, mut work_tile_info: WorkInfo) -> WorkInfo:
        self.advance_to_next_work()
        return self.get_current_work_info()

    @always_inline
    fn requires_reduction(self, work_tile_info: WorkInfo) -> Bool:
        var m = work_tile_info.m * self.tile_shape[0]
        var n = work_tile_info.n * self.tile_shape[1]
        var is_valid = m < self.prob_shape[0] and n < self.prob_shape[1]

        return (
            is_valid
            and work_tile_info.is_valid()
            and work_tile_info.num_k_tiles != self.k_tiles_per_output_tile
        )

    @always_inline
    fn advance_to_next_work(mut self):
        self.current_work_linear_idx += grid_dim.x * grid_dim.y * grid_dim.z

    @always_inline
    fn is_last_split(
        self,
        work_tile_info: WorkInfo,
    ) -> Bool:
        var m = work_tile_info.m * self.tile_shape[0]
        var n = work_tile_info.n * self.tile_shape[1]
        var is_valid = m < self.prob_shape[0] and n < self.prob_shape[1]
        return (
            is_valid
            and work_tile_info.is_valid()
            and work_tile_info.is_final_split(self.k_tiles_per_output_tile)
        )

    @staticmethod
    @always_inline
    fn get_grid_shape(
        dyn_cluster_shape: IndexList[3],
        dyn_raster_order: RasterOrder = RasterOrder.AlongN,
    ) raises -> IndexList[3]:
        var launch_grid_shape = IndexList[3](1, 1, 1)

        if dyn_raster_order == RasterOrder.AlongN:
            launch_grid_shape[0] = dyn_cluster_shape[0]
        else:
            launch_grid_shape[1] = dyn_cluster_shape[1]

        var cluster_size = dyn_cluster_shape[0] * dyn_cluster_shape[1]

        if cluster_size == 1:
            if dyn_raster_order == RasterOrder.AlongN:
                launch_grid_shape[1] = Int(H100.sm_count)
            else:
                launch_grid_shape[0] = Int(H100.sm_count)
        else:
            if dyn_raster_order == RasterOrder.AlongN:
                launch_grid_shape[1] = Int(H100.sm_count) // Int(
                    dyn_cluster_shape[0]
                )
            else:
                launch_grid_shape[0] = Int(H100.sm_count) // Int(
                    dyn_cluster_shape[1]
                )

        return launch_grid_shape

    @staticmethod
    @always_inline
    fn get_num_tiles(
        problem_shape: IndexList[3],
        dyn_tile_shape: IndexList[3],
        dyn_cluster_shape: IndexList[2],
    ) -> Int:
        var problem_blocks = Self.get_problem_blocks_shape(
            problem_shape, dyn_tile_shape, dyn_cluster_shape
        )

        var problem_blocks_m = align_up(
            UInt(problem_blocks[0]),
            UInt(dyn_cluster_shape[0]),
        )
        var problem_blocks_n = align_up(
            UInt(problem_blocks[1]),
            UInt(dyn_cluster_shape[1]),
        )
        return Int(problem_blocks_m * problem_blocks_n)

    @staticmethod
    @always_inline
    fn get_required_locks_buffer_size_bytes[
        accum_type: DType, dyn_num_consumer: UInt32
    ](
        problem_shape: IndexList[3],
        dyn_tile_shape: IndexList[3],
        dyn_cluster_shape: IndexList[2],
    ) -> Int:
        var problem_blocks = Self.get_problem_blocks_shape(
            problem_shape, dyn_tile_shape, dyn_cluster_shape
        )

        var problem_blocks_m = align_up(
            UInt(problem_blocks[0]),
            UInt(dyn_cluster_shape[0]),
        )
        var problem_blocks_n = align_up(
            UInt(problem_blocks[1]),
            UInt(dyn_cluster_shape[1]),
        )

        constrained[
            accum_type == DType.float32,
            "Only support float32 accumulator type",
        ]()

        var num_output_tiles = problem_blocks_m * problem_blocks_n

        var locks_workspace_bytes = (
            num_output_tiles * UInt(size_of[Int32]()) * dyn_num_consumer
        )

        return Int(locks_workspace_bytes)

    @always_inline
    fn get_linear_idx_from_m_and_n(
        self, tile_m: UInt32, tile_n: UInt32
    ) -> UInt32:
        var minor_work_idx: UInt32
        var major_work_idx: UInt32
        var cluster_minor_offset: UInt32

        @parameter
        if self.raster_order == RasterOrder.AlongN:
            minor_work_idx = tile_m
            major_work_idx = tile_n
            var cluster_m = (
                tile_m >> self.log_cluster_shape_minor
            ) << self.log_cluster_shape_minor
            cluster_minor_offset = tile_m - cluster_m
        else:
            major_work_idx = tile_m
            minor_work_idx = tile_n
            var cluster_n = (
                tile_n >> self.log_cluster_shape_minor
            ) << self.log_cluster_shape_minor
            cluster_minor_offset = tile_n - cluster_n

        var cluster_idx_minor = (
            minor_work_idx - cluster_minor_offset
        ) >> self.log_cluster_shape_minor

        var cluster_idx_major = major_work_idx >> self.log_cluster_shape_major
        var cluster_major_offset = major_work_idx & (
            (1 << self.log_cluster_shape_major) - 1
        )

        var cluster_id = (
            cluster_idx_minor * self.cluster_blk_major + cluster_idx_major
        )

        var linear_idx = (
            (
                (cluster_id << self.log_cluster_shape_major)
                + cluster_major_offset
            )
            << self.log_cluster_shape_minor
        ) + cluster_minor_offset

        return linear_idx

    @always_inline
    fn output_tile_index(self, work_tile_info: WorkInfo) -> UInt32:
        return self.get_linear_idx_from_m_and_n(
            work_tile_info.m, work_tile_info.n
        )

    @always_inline
    fn reduction[
        accum_type: DType,
        c_reg_layout: Layout,
        workspace_layout: Layout,
    ](
        self,
        reduction_workspace: Self.WorkTileType[accum_type, workspace_layout],
        c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
        work_tile_info: WorkInfo,
        num_barriers: UInt32,
        warp_group_local_idx: UInt32,
    ):
        if not self.requires_reduction(work_tile_info):
            return

        var reduction_tile_idx = self.output_tile_index(work_tile_info)

        # Index of the lock on which to wait
        var lock_idx = (
            reduction_tile_idx * num_barriers
        ) + warp_group_local_idx

        var num_peers = 0
        var reduction_peer_offset = 0

        var warp_group_thread_idx = thread_idx.x % UInt(WARPGROUP_SIZE)

        if not self.is_last_split(work_tile_info):
            if work_tile_info.k_start == 0:
                # The first split of the tile initializes the workspace partials,
                self.store_accumulator(
                    reduction_workspace,
                    c_reg_tile,
                    reduction_tile_idx,
                    warp_group_local_idx,
                    warp_group_thread_idx,
                )

            else:

                @parameter
                if reduction_mode == ReductionMode.Deterministic:
                    # Wait until the preceding split added its accumulators
                    Self.wait_eq(
                        self.locks_ptr,
                        Int32(warp_group_local_idx),
                        Int(warp_group_thread_idx),
                        lock_idx,
                        work_tile_info.k_start,
                    )

                else:
                    Self.wait_lt(
                        self.locks_ptr,
                        Int32(warp_group_local_idx),
                        Int(warp_group_thread_idx),
                        lock_idx,
                        1,
                    )

                self.reduce_add[write_back=True](
                    reduction_workspace,
                    c_reg_tile,
                    reduction_tile_idx,
                    warp_group_local_idx,
                    warp_group_thread_idx,
                )

            var increment = work_tile_info.num_k_tiles + work_tile_info.k_start

            Self.arrive_set(
                self.locks_ptr,
                Int32(warp_group_local_idx),
                Int(warp_group_thread_idx),
                lock_idx,
                increment,
            )

        else:
            # last split of the tile. Wait until all the other splits have written their accumulators
            Self.wait_eq(
                self.locks_ptr,
                Int32(warp_group_local_idx),
                Int(warp_group_thread_idx),
                lock_idx,
                work_tile_info.k_start,
            )

            self.reduce_add[write_back=False](
                reduction_workspace,
                c_reg_tile,
                reduction_tile_idx,
                warp_group_local_idx,
                warp_group_thread_idx,
            )

    @staticmethod
    @always_inline
    fn wait_eq(
        lock_ptr: UnsafePointer[Int32],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        val: UInt32,
    ):
        var sema = NamedBarrierSemaphore[
            Int32(WARPGROUP_SIZE), 4, Int32(num_consumer)
        ](lock_ptr.offset(lock_idx), barrier_group_thread_idx)
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
        var sema = NamedBarrierSemaphore[
            Int32(WARPGROUP_SIZE), 4, Int32(num_consumer)
        ](lock_ptr.offset(lock_idx), barrier_group_thread_idx)
        sema.wait_lt(barrier_id, Int32(count))

    @staticmethod
    @always_inline
    fn arrive_set(
        lock_ptr: UnsafePointer[Int32],
        barrier_id: Int32,
        barrier_group_thread_idx: Int,
        lock_idx: UInt32,
        increment: UInt32,
    ):
        var sema = NamedBarrierSemaphore[
            Int32(WARPGROUP_SIZE), 4, Int32(num_consumer)
        ](lock_ptr.offset(lock_idx), barrier_group_thread_idx)
        sema.arrive_set(barrier_id, Int32(increment))

    @always_inline
    fn store_accumulator[
        accum_type: DType,
        c_reg_layout: Layout,
        workspace_layout: Layout,
    ](
        self,
        reduction_workspace: Self.WorkTileType[accum_type, workspace_layout],
        c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
        reduction_tile_idx: UInt32,
        warp_group_local_idx: UInt32,
        warp_group_thread_idx: UInt32,
    ):
        alias BM = workspace_layout.shape[1].value()
        alias BN = workspace_layout.shape[2].value()

        constrained[
            accum_type == DType.float32,
            "Only support float32 accumulator type",
        ]()

        alias num_mma = c_reg_tile.layout.shape[0].value()
        alias c_frag_size = c_reg_tile.layout.shape[1].value()

        var workspace_tile = self._get_workspace_tile_reshaped(
            reduction_workspace, reduction_tile_idx
        )

        var tile_crd_idx = workspace_tile.tile_with_offset[
            Int(BM // num_consumer), BN
        ](Int(warp_group_local_idx), 0)
        var work_space_tile_split = tile_crd_idx[0]
        var work_space_tile_reshaped = work_space_tile_split.reshape[
            Layout.row_major(
                (Int(BM // num_consumer) * BN) // WARPGROUP_SIZE,
                WARPGROUP_SIZE,
            )
        ]()

        @parameter
        for mma_id in range(num_mma):

            @parameter
            for i in range(c_frag_size):
                work_space_tile_reshaped[
                    Int(mma_id * c_frag_size + i), Int(warp_group_thread_idx)
                ] = c_reg_tile[mma_id, i]

    @always_inline
    fn reduce_add[
        accum_type: DType,
        c_reg_layout: Layout,
        workspace_layout: Layout, //,
        *,
        write_back: Bool,
    ](
        self,
        reduction_workspace: Self.WorkTileType[accum_type, workspace_layout],
        c_reg_tile: RegTileType[accum_type, c_reg_layout, _],
        reduction_tile_idx: UInt32,
        warp_group_local_idx: UInt32,
        warp_group_thread_idx: UInt32,
    ):
        alias BM = workspace_layout.shape[1].value()
        alias BN = workspace_layout.shape[2].value()

        constrained[
            accum_type == DType.float32,
            "Only support float32 accumulator type",
        ]()

        alias num_mma = c_reg_tile.layout.shape[0].value()
        alias c_frag_size = c_reg_tile.layout.shape[1].value()

        var workspace_tile = self._get_workspace_tile_reshaped(
            reduction_workspace, reduction_tile_idx
        )

        var tile_crd_idx = workspace_tile.tile_with_offset[
            Int(BM // num_consumer), BN
        ](Int(warp_group_local_idx), 0)
        var work_space_tile_split = tile_crd_idx[0]
        var work_space_tile_reshaped = work_space_tile_split.reshape[
            Layout.row_major(
                (Int(BM // num_consumer) * BN) // WARPGROUP_SIZE,
                WARPGROUP_SIZE,
            )
        ]()

        @parameter
        for mma_id in range(num_mma):

            @parameter
            for i in range(c_frag_size):
                var sum_val = (
                    work_space_tile_reshaped[
                        Int(mma_id * c_frag_size + i),
                        Int(warp_group_thread_idx),
                    ]
                    + c_reg_tile[mma_id, i]
                )

                @parameter
                if write_back:

                    @parameter
                    if reduction_mode == ReductionMode.Nondeterministic:
                        var offset = (
                            mma_id * c_frag_size + i
                        ) * WARPGROUP_SIZE + warp_group_thread_idx

                        _ = Atomic.fetch_add(
                            work_space_tile_reshaped.ptr + offset,
                            rebind[Scalar[accum_type]](c_reg_tile[mma_id, i]),
                        )
                    else:
                        work_space_tile_reshaped[
                            Int(mma_id * c_frag_size + i),
                            Int(warp_group_thread_idx),
                        ] = sum_val
                else:
                    c_reg_tile[mma_id, i] = sum_val

    @always_inline
    fn _get_workspace_tile_reshaped[
        accum_type: DType,
        workspace_layout: Layout,
    ](
        self,
        reduction_workspace: Self.WorkTileType[accum_type, workspace_layout],
        reduction_tile_idx: UInt32,
        out reshaped_workspace: Self.WorkTileType[
            accum_type,
            Layout.row_major(
                reduction_workspace.shape[1](), reduction_workspace.shape[2]()
            ),
        ],
    ):
        alias BM = workspace_layout.shape[1].value()
        alias BN = workspace_layout.shape[2].value()

        return {
            reduction_workspace.ptr + reduction_tile_idx * BM * BN,
            RuntimeLayout[reshaped_workspace.layout].row_major(Index(BM, BN)),
        }
