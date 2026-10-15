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

from std.sys import simd_width_of

from std.algorithm import vectorize

from max.algorithm import sync_parallelize
from layout import *
from layout._fillers import arange
from layout.tile_tensor import stack_allocation
from layout.tensor_engine import DefaultEngine
from std.sys import align_of


@fieldwise_init
struct Dim(ImplicitlyCopyable, RegisterPassable, Writable):
    var m: Int
    var n: Int
    var k: Int

    def subrange(self, sub_dim: Self) -> Self:
        return Self(
            self.m // sub_dim.m, self.n // sub_dim.n, self.k // sub_dim.k
        )

    def write_to(self, mut writer: Some[Writer]):
        """Writes a string representation of the dim.

        Args:
            writer: The writer to write to.
        """
        t"m: {self.m}, n: {self.n}, k: {self.k}".write_to(writer)


trait TiledOp:
    @staticmethod
    def op(
        dst: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
        lhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
        rhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    ):
        pass


# matrix multiply and accumulate
struct MMA(TiledOp):
    @staticmethod
    def op(
        dst: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
        lhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
        rhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    ):
        comptime dtype = dst.dtype

        comptime assert (
            dst.rank
            == dst.flat_rank
            == lhs.rank
            == lhs.flat_rank
            == rhs.rank
            == rhs.flat_rank
            == 2
        )
        comptime assert (
            dst.all_dims_known and lhs.all_dims_known and rhs.all_dims_known
        )
        comptime M = Int(type_of(dst).static_shape[0])
        comptime N = Int(type_of(dst).static_shape[1])
        comptime K = Int(type_of(lhs).static_shape[1])

        for m in range(M):
            for n in range(N):
                for k in range(K):
                    dst[m, n] += rebind[dst.ElementType](
                        lhs[m, k].cast[dtype]()
                    ) * rebind[dst.ElementType](rhs[n, k].cast[dtype]())


# matrix multiply and accumulate, vectorized and parallelized
struct MMA_Vec(TiledOp):
    @staticmethod
    def op(
        dst: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
        lhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
        rhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    ):
        comptime assert (
            dst.rank
            == dst.flat_rank
            == lhs.rank
            == lhs.flat_rank
            == rhs.rank
            == rhs.flat_rank
            == 2
        )
        comptime assert (
            dst.all_dims_known and lhs.all_dims_known and rhs.all_dims_known
        )
        comptime M = Int(type_of(dst).static_shape[0])
        comptime N = Int(type_of(dst).static_shape[1])
        comptime K = Int(type_of(lhs).static_shape[1])

        comptime width = simd_width_of[dst.dtype]() * 2

        for var m in range(M):
            for var n in range(N):

                def dot[width: Int](k: Int) {imm}:
                    dst.store[width=width, alignment=align_of[dst.dtype]()](
                        Coord(m, n),
                        rebind[SIMD[dst.dtype, width]](
                            dst.load[
                                width=width, alignment=align_of[dst.dtype]()
                            ](Coord(m, n))
                        )
                        + rebind[SIMD[dst.dtype, width]](
                            lhs[m, k].cast[dst.dtype]()
                        )
                        * rhs.load[
                            width=width, alignment=align_of[rhs.dtype]()
                        ](Coord(n, k)).cast[dst.dtype](),
                    )

                vectorize[width, size=K](dot)


def gemm_l2_cache[
    mma: TiledOp, L1: Dim, L2: Dim
](
    dst: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    lhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    rhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
) raises:
    comptime assert (
        dst.rank
        == dst.flat_rank
        == lhs.rank
        == lhs.flat_rank
        == rhs.rank
        == rhs.flat_rank
        == 2
    )
    comptime assert (
        dst.all_dims_known and lhs.all_dims_known and rhs.all_dims_known
    )
    comptime M = Int(type_of(dst).static_shape[0])
    comptime N = Int(type_of(dst).static_shape[1])
    comptime K = Int(type_of(lhs).static_shape[1])

    # Dimensions of the Operation
    comptime op_dim = Dim(M, N, K)

    # L1 and L2 tile ranges
    comptime l1_size = op_dim.subrange(L1)
    comptime l2_size = L1.subrange(L2)

    # Cache matrix to materialize L2 transposed tiles
    var l2_rhs_storage = List[Scalar[dst.dtype]](length=L2.n * L2.k, fill=0)
    var l2_rhs_cache = TileTensor(Span(l2_rhs_storage), col_major[L2.n, L2.k]())

    # First level of tiling (grid_blocks, L1 cache ..etc).
    for m_1 in range(l1_size.m):
        for n_1 in range(l1_size.n):
            var dst_l1_tile = dst.tile[L1.m, L1.n](m_1, n_1)

            for k_1 in range(l1_size.k):
                var lhs_l1_tile = lhs.tile[L1.m, L1.k](m_1, k_1)
                var rhs_l1_tile = rhs.tile[L1.k, L1.n](k_1, n_1)

                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(l2_size.m):
                    for n_2 in range(l2_size.n):
                        var dst_l2_tile = dst_l1_tile.tile[L2.m, L2.n](m_2, n_2)

                        for k_2 in range(l2_size.k):
                            var lhs_l2_tile = lhs_l1_tile.tile[L2.m, L2.k](
                                m_2, k_2
                            )
                            var rhs_l2_tile = rhs_l1_tile.tile[L2.k, L2.n](
                                k_2, n_2
                            )

                            # Materialize L2 rhs transposed tile
                            l2_rhs_cache.copy_from(rhs_l2_tile.transpose())

                            # Execute mma.op with the transposed RHS tile
                            mma.op(dst_l2_tile, lhs_l2_tile, l2_rhs_cache)
    _ = l2_rhs_storage^


def gemm_l1_cache[
    mma: TiledOp, L1: Dim, L2: Dim
](
    dst: TileTensor[mut=True, Engine=DefaultEngine[element_width=1], ...],
    lhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
    rhs: TileTensor[Engine=DefaultEngine[element_width=1], ...],
):
    comptime assert (
        dst.rank
        == dst.flat_rank
        == lhs.rank
        == lhs.flat_rank
        == rhs.rank
        == rhs.flat_rank
        == 2
    )
    comptime assert (
        dst.all_dims_known and lhs.all_dims_known and rhs.all_dims_known
    )
    comptime M = Int(type_of(dst).static_shape[0])
    comptime N = Int(type_of(dst).static_shape[1])
    comptime K = Int(type_of(lhs).static_shape[1])

    # Dimensions of the Operation
    comptime op_dim = Dim(M, N, K)

    # L1 and L2 tile ranges
    comptime l1_size = op_dim.subrange(L1)
    comptime l2_size = L1.subrange(L2)

    def process_raw(m_1: Int) {imm}:
        # Cache the current lhs tile and reuse it for all rhs tiles in the column
        var l1_lhs_cache = stack_allocation[dtype=dst.dtype](
            col_major[L1.m, L1.k]()
        )
        var l1_rhs_cache = stack_allocation[dtype=dst.dtype](
            col_major[L1.n, L1.k]()
        )

        for k_1 in range(l1_size.k):
            l1_lhs_cache.copy_from(lhs.tile[L1.m, L1.k](m_1, k_1))

            for n_1 in range(l1_size.n):
                var dst_l1_tile = dst.tile[L1.m, L1.n](m_1, n_1)

                # Materialize L1 rhs transposed tile
                l1_rhs_cache.copy_from(
                    rhs.tile[L1.k, L1.n](k_1, n_1).transpose()
                )

                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(l2_size.m):
                    for n_2 in range(l2_size.n):
                        var dst_l2_tile = dst_l1_tile.tile[L2.m, L2.n](m_2, n_2)

                        for k_2 in range(l2_size.k):
                            var lhs_l2_tile = l1_lhs_cache.tile[L2.m, L2.k](
                                m_2, k_2
                            )
                            # Transposed tile -> transposed indices
                            var rhs_l2_tile = l1_rhs_cache.tile[L2.n, L2.k](
                                n_2, k_2
                            )

                            # Execute mma.op with the transposed RHS tile
                            mma.op(dst_l2_tile, lhs_l2_tile, rhs_l2_tile)

    sync_parallelize(process_raw, l1_size.m)


def test_tiled_matmul[use_l1_cache: Bool]() raises:
    if use_l1_cache:
        print("=== test_tiled_matmul_l1_cache")
    else:
        print("=== test_tiled_matmul_l2_cache")

    var dst_storage = List[Float32](length=64, fill=0)
    var dst = TileTensor(Span(dst_storage), col_major[8, 8]())
    var rhs_storage = List[Float32](length=64, fill=0)
    var rhs = TileTensor(Span(rhs_storage), col_major[8, 8]())
    var lhs_storage = List[Float32](length=64, fill=0)
    var lhs = TileTensor(Span(lhs_storage), col_major[8, 8]())

    _ = dst.fill(0)
    arange(rhs)
    arange(lhs)

    if use_l1_cache:
        gemm_l1_cache[
            MMA_Vec,
            Dim(4, 4, 2),
            Dim(2, 2, 1),
        ](dst, lhs, rhs)
    else:
        gemm_l2_cache[
            MMA_Vec,
            Dim(4, 4, 2),
            Dim(2, 2, 1),
        ](dst, lhs, rhs)
    # Preserve the numerical FileCheck rows independently of tensor formatting.
    for m in range(8):
        for n in range(8):
            print(dst[m, n], end="   " if n != 7 else "\n")

    _ = rhs_storage^
    _ = lhs_storage^
    _ = dst_storage^


def main() raises:
    # CHECK: === test_tiled_matmul_l1_cache
    # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
    # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
    # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
    # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
    # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
    # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
    # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
    # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
    test_tiled_matmul[use_l1_cache=True]()

    # CHECK: === test_tiled_matmul_l2_cache
    # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
    # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
    # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
    # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
    # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
    # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
    # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
    # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
    test_tiled_matmul[use_l1_cache=False]()
