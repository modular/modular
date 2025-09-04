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

from math import ceildiv

from gpu.id import block_idx, grid_dim
from utils.fast_div import FastDiv

from utils.index import Index, IndexList

from linalg.utils_gpu import block_swizzle


@fieldwise_init
@register_passable("trivial")
struct RasterOrder(ImplicitlyCopyable, Movable):
    var _value: Int32

    alias AlongN = Self(0)
    alias AlongM = Self(1)

    @always_inline
    fn __eq__(self, other: Self) -> Bool:
        return self._value == other._value

    @always_inline
    fn __ne__(self, other: Self) -> Bool:
        return self._value != other._value


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
    fn __init__(
        out self,
    ):
        self.m = 0
        self.n = 0
        self.k_start = 0
        self.num_k_tiles = 0
        self.is_valid_tile = False

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @always_inline
    fn is_final_split(self, k_tiles_per_output_tile: UInt32) -> Bool:
        return (self.k_start + self.num_k_tiles) == k_tiles_per_output_tile

    @always_inline
    fn get_k_start(self) -> UInt32:
        return self.k_start

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
            self.num_k_tiles,
            ", ",
            self.is_valid_tile,
            ")",
        )


@fieldwise_init
@register_passable("trivial")
struct MatmulSchedule(ImplicitlyCopyable, Movable):
    var _value: Int32

    alias NONE = Self(-1)
    alias TILE1D = Self(0)
    alias TILE2D = Self(1)
    alias DS_SCHEDULER = Self(2)

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
struct TileScheduler[
    problem_shape: IndexList[3],
    tile_shape: IndexList[3],
    grid_shape: IndexList[2],
    cluster: IndexList[3] = Index(1, 1, 1),
    raster_dim: UInt32 = 1,
    schedule: MatmulSchedule = MatmulSchedule.TILE2D,
]:
    # grid_shape[0], [1] map to x, y, to N and M in output matrix.
    # tile_shape[0], [1] map to M and N
    # wave_shape[0], [1] map to M and N
    alias wave_shape = Index[dtype = DType.uint32](
        tile_shape[0] * grid_shape[1], tile_shape[1] * grid_shape[0]
    )
    # This has to match the grid dimension for the kernel launch.
    alias num_grids: UInt32 = grid_shape[0] * grid_shape[1]
    var idx: UInt32
    var prob_shape: IndexList[3]  # M x N x K
    var num_waves_m: UInt32
    var num_waves_n: UInt32
    var log_num_waves_n: FastDiv[DType.uint32]

    # Member variables for DeepSeek Scheduler
    var current_iter: Int  # Tracks the scheduler's progress across kernel launches
    var num_aligned_m_blocks: UInt32  # Number of blocks needed for the M dimension
    var num_blocks: UInt32  # Total number of blocks for non-masked types

    alias kNum1DBlocksPerGroup: UInt32 = 16
    alias kNumNBlocks: UInt32 = ceildiv(problem_shape[1], tile_shape[1])

    @always_inline
    fn __init__(out self, prob_shape: IndexList[3]):
        @parameter
        if schedule == MatmulSchedule.TILE2D:
            constrained[
                _check_cluster(cluster, raster_dim),
                "Only support block cluster in along raster dimension.",
            ]()

        if schedule == MatmulSchedule.DS_SCHEDULER:
            constrained[
                cluster[0] == cluster[1] == cluster[2] == 1,
                (
                    "Currently multicasting is not supported for DeepSeek"
                    " Scheduler"
                ),
            ]()

        self.prob_shape = prob_shape
        self.num_waves_m = ceildiv(self.prob_shape[0], Self.wave_shape[0])
        self.num_waves_n = ceildiv(self.prob_shape[1], Self.wave_shape[1])
        self.log_num_waves_n = FastDiv[DType.uint32](Int(self.num_waves_n))

        self.current_iter = -1
        self.num_aligned_m_blocks = ceildiv(prob_shape[0], tile_shape[0])
        self.num_blocks = self.num_aligned_m_blocks * Self.kNumNBlocks

        @parameter
        if raster_dim == 0:  # rasterize along M
            self.idx = block_idx.x * grid_dim.y + block_idx.y
        else:
            self.idx = block_idx.x + grid_dim.x * block_idx.y

    @always_inline
    fn get_current_work_info(mut self) -> WorkInfo:
        @parameter
        if schedule == MatmulSchedule.DS_SCHEDULER:
            var m_block_idx: UInt32 = 0
            var n_block_idx: UInt32 = 0
            var is_valid = self._get_next_block(m_block_idx, n_block_idx)
            var m = UInt(m_block_idx * tile_shape[0])
            var n = UInt(n_block_idx * tile_shape[1])

            return WorkInfo(
                m, n, 0, ceildiv(problem_shape[2], tile_shape[2]), is_valid
            )
        else:
            m, n = self._index_to_mn()
            is_valid = m < UInt(self.prob_shape[0]) and n < UInt(
                self.prob_shape[1]
            )
            return WorkInfo(
                m, n, 0, ceildiv(self.prob_shape[2], tile_shape[2]), is_valid
            )

    @always_inline
    fn advance(mut self):
        self.idx += Self.num_grids

    @always_inline
    fn fetch_next_work(mut self) -> WorkInfo:
        @parameter
        if schedule == MatmulSchedule.DS_SCHEDULER:
            return self.fetch_next_work_ds()
        else:
            self.advance()
            return self.get_current_work_info()

    @always_inline
    fn _index_to_mn(self) -> Tuple[UInt, UInt]:
        """Map the thread block's index to coordinates of work tile."""

        @parameter
        if schedule == MatmulSchedule.TILE2D:
            return self._index_to_mn_tile2d()

        return self._index_to_mn_tile1d()

    @always_inline
    fn _index_to_mn_tile1d(self) -> Tuple[UInt, UInt]:
        # Grid dim as if there is no persist kernel
        logical_grid_dim = Index[dtype = DType.uint32](
            ceildiv(self.prob_shape[1], tile_shape[1]),
            ceildiv(self.prob_shape[0], tile_shape[0]),
        )

        by, bx = divmod(UInt(self.idx), UInt(logical_grid_dim[0]))
        block_xy_swizzle = block_swizzle(
            Index[dtype = DType.uint32](bx, by), logical_grid_dim
        )

        m = UInt(block_xy_swizzle[1] * tile_shape[0])
        n = UInt(block_xy_swizzle[0] * tile_shape[1])

        return (m, n)

    @always_inline
    fn _index_to_mn_tile2d(self) -> Tuple[UInt, UInt]:
        # We consider a sweep on busy SMs a wave, not all SMs
        alias log_num_grids = FastDiv[DType.uint32](Int(Self.num_grids))
        alias log_grid_shape = FastDiv[DType.uint32](Int(grid_shape[0]))

        num_waves_executed = Int(self.idx) / log_num_grids
        idx_in_wave = Int(self.idx) % log_num_grids

        num_waves_executed_m = Int(num_waves_executed) / self.log_num_waves_n
        num_waves_executed_n = Int(num_waves_executed) % self.log_num_waves_n

        # The wave maps to a BM x grid_shape[1] by BN x grid_shape[0]
        # submatrix in C.
        wave_m = num_waves_executed_m * Self.wave_shape[0]
        wave_n = num_waves_executed_n * Self.wave_shape[1]

        m_in_wave = Int(idx_in_wave) / log_grid_shape
        n_in_wave = Int(idx_in_wave) % log_grid_shape

        return (
            UInt(wave_m + m_in_wave * tile_shape[0]),
            UInt(wave_n + n_in_wave * tile_shape[1]),
        )

    @always_inline
    fn num_output_tiles(self) -> UInt:
        return UInt(
            ceildiv(self.prob_shape[0], Self.wave_shape[0])
            * ceildiv(self.prob_shape[1], Self.wave_shape[1])
        )

    @always_inline
    fn fetch_next_work_ds(mut self) -> WorkInfo:
        var m_block_idx: UInt32 = 0
        var n_block_idx: UInt32 = 0
        var is_valid = self._get_next_block(m_block_idx, n_block_idx)

        var m = UInt(m_block_idx * tile_shape[0])
        var n = UInt(n_block_idx * tile_shape[1])
        # Only support K starting from 0 for now.
        return WorkInfo(
            m, n, 0, ceildiv(problem_shape[2], tile_shape[2]), is_valid
        )

    # Calculates swizzled M and N block indices for better cache utilization
    @always_inline
    fn _get_swizzled_block_idx(
        self, num_m_blocks: UInt32, block_idx: Int
    ) -> Tuple[UInt32, UInt32]:
        """
        Calculates swizzled (m_block_idx, n_block_idx) based on the overall block_idx.
        The swizzling pattern depends on kIsTMAMulticastOnA.
        Returns a tuple (m_block_idx, n_block_idx).
        """

        var m_block_idx: UInt32
        var n_block_idx: UInt32

        # Swizzle for better L2 usages
        var primary_num_blocks = num_m_blocks
        alias secondary_num_blocks = Self.kNumNBlocks
        alias num_blocks_per_group = secondary_num_blocks * Self.kNum1DBlocksPerGroup
        var group_idx = block_idx / num_blocks_per_group
        var first_block_idx = group_idx * Self.kNum1DBlocksPerGroup
        var in_group_idx = block_idx % num_blocks_per_group
        var num_blocks_in_group = min(
            Self.kNum1DBlocksPerGroup, primary_num_blocks - first_block_idx
        )

        m_block_idx = first_block_idx + in_group_idx % num_blocks_in_group
        n_block_idx = in_group_idx / num_blocks_in_group

        return (m_block_idx, n_block_idx)

    # Gets the next (m_block_idx, n_block_idx) pair for the current thread block to process
    @always_inline
    fn _get_next_block(
        mut self, mut m_block_idx: UInt32, mut n_block_idx: UInt32
    ) -> Bool:
        """
        Calculates and returns the next (m_block_idx, n_block_idx) pair for the
        calling thread block. Returns None if no more blocks are available.
        This implements the core logic of the persistent block scheduler.
        """

        self.current_iter += 1
        var next_block_idx = self.current_iter * grid_dim.x + block_idx.x

        # Check if the calculated index exceeds the total number of blocks
        if next_block_idx >= Int(self.num_blocks):
            return False  # No more work

        # Get swizzled indices based on the total number of aligned M blocks
        m_block_idx, n_block_idx = self._get_swizzled_block_idx(
            self.num_aligned_m_blocks, Int(next_block_idx)
        )
        return True


fn _check_cluster(cluster_dims: IndexList[3], raster_dim: UInt32) -> Bool:
    """Check if block cluster is along the raster dimension."""

    @parameter
    for i in range(3):
        if cluster_dims[i] > 1 and i != Int(raster_dim):
            return False

    return True
