# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from kernel_utils.int_tuple import IntTuple
from kernel_utils.layout import Layout, LayoutList, composition
from kernel_utils.layout_tensor import LayoutTensor


fn fill_matrix(matrix: LayoutTensor[DType.float32]):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(M):
        for n in range(N):
            matrix[IntTuple(m, n)] = m * M + n


fn fill_matrix(matrix: LayoutTensor[DType.float32], val: Float32):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(M):
        for n in range(N):
            matrix[IntTuple(m, n)] = val


fn print_row_major_matrix[dtype: DType](matrix: LayoutTensor[dtype]):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(M):
        for n in range(N):
            print_no_newline(matrix[IntTuple(m, n)], "  ")
        print("")


@register_passable
struct Dim(Stringable):
    var m: Int
    var n: Int
    var k: Int

    fn __init__(m: Int, n: Int, k: Int) -> Self:
        return Self {m: m, n: n, k: k}

    fn subrange(self, sub_dim: Self) -> Self:
        return Self(
            self.m // sub_dim.m, self.n // sub_dim.n, self.k // sub_dim.k
        )

    fn __str__(self) -> String:
        return (
            "m: " + str(self.m) + ", n: " + str(self.n) + ", k: " + str(self.k)
        )


trait TiledOp:
    @staticmethod
    fn l1_dim() -> Dim:
        pass

    @staticmethod
    fn l2_dim() -> Dim:
        pass

    @staticmethod
    fn op[
        dtype: DType, rhs_transposed: Bool = False
    ](
        inout dst: LayoutTensor[dtype],
        lhs: LayoutTensor[dtype],
        rhs: LayoutTensor[dtype],
    ):
        pass


fn compatible[
    dtype: DType, rhs_transposed: Bool = False
](
    dst: LayoutTensor[dtype],
    lhs: LayoutTensor[dtype],
    rhs: LayoutTensor[dtype],
) -> Bool:
    if rhs_transposed:
        return (
            lhs.dim(0) == dst.dim(0)
            and rhs.dim(0) == dst.dim(1)
            and lhs.dim(1) == rhs.dim(1)
        )
    else:
        return (
            lhs.dim(0) == dst.dim(0)
            and rhs.dim(1) == dst.dim(1)
            and lhs.dim(1) == rhs.dim(0)
        )


# matrix multiply and accumlate
struct MMA(TiledOp):
    # FIXME: Non square tiles don't seem to work.
    @staticmethod
    fn l1_dim() -> Dim:
        return Dim(4, 4, 4)

    @staticmethod
    fn l2_dim() -> Dim:
        return Dim(2, 2, 2)

    @staticmethod
    fn op[
        dtype: DType, rhs_transposed: Bool = False
    ](
        inout dst: LayoutTensor[dtype],
        lhs: LayoutTensor[dtype],
        rhs: LayoutTensor[dtype],
    ):
        if not compatible(dst, lhs, rhs):
            trap("Incompatible matrices")

        let M = dst.dim(0)
        let N = dst.dim(1)
        let K = lhs.dim(1)
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    let rhs_t = rhs[IntTuple(n, k)] if rhs_transposed else rhs[
                        IntTuple(k, n)
                    ]
                    dst[IntTuple(m, n)] += lhs[IntTuple(m, k)] * rhs_t


struct Tiling(Stringable):
    var dst: LayoutList
    var lhs: LayoutList
    var rhs: LayoutList

    fn __init__(inout self, dim: Dim):
        self.dst = Self.tile_layout(dim.m, dim.n)
        self.lhs = Self.tile_layout(dim.m, dim.k)
        self.rhs = Self.tile_layout(dim.k, dim.n)

    @staticmethod
    fn tile_layout(n: Int, m: Int) -> LayoutList:
        return LayoutList(Layout(n, 1), Layout(m, 1))

    fn __str__(self) -> String:
        return (
            "dst: "
            + str(self.dst)
            + ", lhs: "
            + str(self.lhs)
            + ", rhs: "
            + str(self.rhs)
        )


fn gemm[
    dtype: DType,
    mma: TiledOp,
    rhs_transposed: Bool = False,
](dst: LayoutTensor[dtype], lhs: LayoutTensor[dtype], rhs: LayoutTensor[dtype]):
    if not compatible[dtype, False](dst, lhs, rhs):
        trap("Incompatible matrices")

    # Dimensions of the Operation
    let op_dim = Dim(dst.dim(0), dst.dim(1), lhs.dim(1))

    # L1 and L2 Tilings
    let l1_tiling = Tiling(mma.l1_dim())
    let l2_tiling = Tiling(mma.l2_dim())

    # L1 and L2 Tiile ranges
    let l1_dim = op_dim.subrange(mma.l1_dim())
    let l2_dim = mma.l1_dim().subrange(mma.l2_dim())

    # Cache matrix to materialize L1 tiles
    var l1_rhs_cache = LayoutTensor[dtype](
        mma.l1_dim().k, mma.l1_dim().n
    ) if rhs_transposed else LayoutTensor[dtype](mma.l1_dim().n, mma.l1_dim().k)

    # Transposed view on rhs
    let rhs_t = rhs.transpose() if rhs_transposed else rhs

    # First level of tiling (grid_blocks, L1 cache ..etc).
    for m_1 in range(l1_dim.m):
        for n_1 in range(l1_dim.n):
            var dst_l1_tile = dst.view(l1_tiling.dst, IntTuple(m_1, n_1))

            for k_1 in range(l1_dim.k):
                let lhs_l1_tile = lhs.view(l1_tiling.lhs, IntTuple(m_1, k_1))

                # Materialize L1 rhs (transposed) tile
                # rhs tile indices have to be swapped when transposed
                let rhs_l1_tile_idx = IntTuple(
                    n_1, k_1
                ) if rhs_transposed else IntTuple(k_1, n_1)
                rhs_t.view(
                    l1_tiling.rhs,
                    rhs_l1_tile_idx,
                ).copyTo(l1_rhs_cache)
                let rhs_l1_tile = l1_rhs_cache

                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(l2_dim.m):
                    for n_2 in range(l2_dim.n):
                        var dst_l2_tile = dst_l1_tile.view(
                            l2_tiling.dst, IntTuple(m_2, n_2)
                        )

                        for k_2 in range(l2_dim.k):
                            let lhs_l2_tile = lhs_l1_tile.view(
                                l2_tiling.lhs, IntTuple(m_2, k_2)
                            )

                            # rhs tile indices have to be swapped when transposed
                            let rhs_l2_tile_idx = IntTuple(
                                n_2, k_2
                            ) if rhs_transposed else IntTuple(k_2, n_2)
                            let rhs_l2_tile = rhs_l1_tile.view(
                                l2_tiling.rhs,
                                rhs_l2_tile_idx,
                            )

                            # Execute mma.op - rhs_l2_tile is already transposed
                            mma.op[dtype, rhs_transposed](
                                dst_l2_tile, lhs_l2_tile, rhs_l2_tile
                            )

    # TODO: Make tensor more ergonomic
    l1_rhs_cache.ptr.free()


fn test_tiled_matmul[rhs_transposed: Bool = False]():
    print("=== test_tiled_matmul")
    let dst = LayoutTensor[DType.float32](8, 8)
    let rhs = LayoutTensor[DType.float32](8, 8)
    let lhs = LayoutTensor[DType.float32](8, 8)
    fill_matrix(dst, 0)
    fill_matrix(rhs)
    fill_matrix(lhs)
    gemm[DType.float32, MMA, rhs_transposed](dst, lhs, rhs)
    print_row_major_matrix(dst)
    dst.ptr.free()
    lhs.ptr.free()
    rhs.ptr.free()


fn main():
    # CHECK: === test_tiled_matmul
    # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
    # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
    # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
    # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
    # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
    # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
    # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
    # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
    test_tiled_matmul()

    # CHECK: === test_tiled_matmul_transpose
    # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
    # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
    # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
    # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
    # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
    # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
    # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
    # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
    test_tiled_matmul[True]()
