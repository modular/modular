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
from linalg.fast_div import FastDiv

from utils.index import Index, IndexList

from .utils_gpu import block_swizzle


@value
@register_passable("trivial")
struct WorkInfo(Stringable, Writable):
    # Coordinates in output matrix
    var m: UInt32
    var n: UInt32
    # Starting k index in A and B for the output tile's mma.
    var k_start: UInt32
    var num_k_tiles: UInt32
    # Whether work tile is completely OOB.
    var is_valid_tile: Bool

    @always_inline
    fn is_valid(self) -> Bool:
        return self.is_valid_tile

    @no_inline
    fn __str__(self) -> String:
        return String.write(self)

    @no_inline
    fn write_to[W: Writer](self, mut writer: W):
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


@value
@register_passable("trivial")
struct MatmulSchedule:
    var _value: Int32

    alias NONE = Self(-1)
    alias TILE1D = Self(0)
    alias TILE2D = Self(1)

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

    @always_inline
    fn __init__(out self, prob_shape: IndexList[3]):
        @parameter
        if schedule == MatmulSchedule.TILE2D:
            constrained[
                _check_cluster(cluster, raster_dim),
                "Only support block cluster in along raster dimention.",
            ]()

        self.prob_shape = prob_shape
        self.num_waves_m = ceildiv(self.prob_shape[0], Self.wave_shape[0])
        self.num_waves_n = ceildiv(self.prob_shape[1], Self.wave_shape[1])
        self.log_num_waves_n = FastDiv[DType.uint32](Int(self.num_waves_n))

        @parameter
        if raster_dim == 0:  # rasterize along M
            self.idx = block_idx.x * grid_dim.y + block_idx.y
        else:
            self.idx = block_idx.x + grid_dim.x * block_idx.y

    @always_inline
    fn get_current_work_info(self) -> WorkInfo:
        m, n = self._index_to_mn()
        is_valid = m < self.prob_shape[0] and n < self.prob_shape[1]

        # Only support K starting from 0 for now.
        return WorkInfo(m, n, 0, self.prob_shape[2] // tile_shape[2], is_valid)

    @always_inline
    fn advance(mut self):
        self.idx += Self.num_grids

    @always_inline
    fn fetch_next_work(mut self) -> WorkInfo:
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

        by, bx = divmod(UInt(Int(self.idx)), UInt(Int(logical_grid_dim[0])))
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
            UInt(Int(wave_m + m_in_wave * tile_shape[0])),
            UInt(Int(wave_n + n_in_wave * tile_shape[1])),
        )

    @always_inline
    fn num_output_tiles(self) -> UInt:
        return ceildiv(self.prob_shape[0], Self.wave_shape[0]) * ceildiv(
            self.prob_shape[1], Self.wave_shape[1]
        )


fn _check_cluster(cluster_dims: IndexList[3], raster_dim: UInt32) -> Bool:
    """Check if block cluster is along the raster dimention."""

    @parameter
    for i in range(3):
        if cluster_dims[i] > 1 and i != Int(raster_dim):
            return False

    return True
