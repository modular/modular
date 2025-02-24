# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from math import ceildiv
from gpu.id import grid_dim, block_idx
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
    var is_valid: Bool

    @always_inline
    fn __bool__(self) -> Bool:
        return self.is_valid

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
            self.is_valid,
            ")",
        )


# ===----------------------------------------------------------------------=== #
# Output Tile Scheduler
# ===----------------------------------------------------------------------=== #


@register_passable("trivial")
struct TileScheduler[
    tile_shape: IndexList[3],
    grid_shape: IndexList[2],
    cluster: IndexList[3] = Index(1, 1, 1),
    raster_dim: Int = 1,
]:
    alias wave_shape = Index(
        tile_shape[0] * grid_shape[0], tile_shape[1] * grid_shape[1]
    )
    # This has to match the grid dimension for the kernel launch.
    alias num_grids: UInt = grid_shape[0] * grid_shape[1]
    var idx: UInt
    var prob_shape: IndexList[3]  # M x N x K
    var num_waves_m: UInt
    var num_waves_n: UInt

    @always_inline
    fn __init__(mut self, prob_shape: IndexList[3]):
        constrained[
            _check_cluster(cluster, raster_dim),
            "Only support block cluster in along raster dimention.",
        ]()

        @parameter
        if raster_dim == 0:  # rasterize along M
            self.idx = block_idx.x * grid_dim.y + block_idx.y
        else:
            self.idx = block_idx.x + grid_dim.x * block_idx.y

        self.prob_shape = prob_shape
        self.num_waves_m = ceildiv(self.prob_shape[0], Self.wave_shape[0])
        self.num_waves_n = ceildiv(self.prob_shape[1], Self.wave_shape[1])

    @always_inline
    fn work_info(self) -> WorkInfo:
        m, n = self._index_to_mn()
        is_valid = m < self.prob_shape[0] and n < self.prob_shape[1]

        # Only support K starting from 0 for now.
        return WorkInfo(m, n, 0, self.prob_shape[2] // tile_shape[2], is_valid)

    @always_inline
    fn advance(mut self):
        self.idx += Self.num_grids

    @always_inline
    fn _index_to_mn(self) -> Tuple[UInt, UInt]:
        # We consider a sweep on busy SMs a wave, not all SMs

        num_waves_executed, idx_in_wave = divmod(self.idx, Self.num_grids)
        num_waves_executed_m, num_waves_executed_n = divmod(
            num_waves_executed, UInt(self.num_waves_n)
        )

        # The wave maps to a BM x grid_shape[0] by BN x grid_shape[1]
        # submatrix in C.
        wave_m = num_waves_executed_m * Self.wave_shape[0]
        wave_n = num_waves_executed_n * Self.wave_shape[1]

        m_in_wave, n_in_wave = divmod(idx_in_wave, UInt(grid_shape[1]))

        return (
            wave_m + m_in_wave * tile_shape[0],
            wave_n + n_in_wave * tile_shape[1],
        )

    @always_inline
    fn num_output_tiles(self) -> UInt:
        return ceildiv(self.prob_shape[0], Self.wave_shape[0]) * ceildiv(
            self.prob_shape[1], Self.wave_shape[1]
        )


fn _check_cluster(cluster_dims: IndexList[3], raster_dim: Int) -> Bool:
    """Check if block cluster is along the raster dimention."""

    @parameter
    for i in range(3):
        if cluster_dims[i] > 1 and i != raster_dim:
            return False

    return True
