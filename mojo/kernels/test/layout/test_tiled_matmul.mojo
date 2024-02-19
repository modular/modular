# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from algorithm import parallelize, vectorize
from kernel_utils.layout import Layout, LayoutList, composition
from kernel_utils.layout_tensor import LayoutTensor


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
    fn op[
        dtype: DType,
        M: Int,
        N: Int,
        K: Int,
    ](
        inout dst: LayoutTensor[dtype, M, N],
        lhs: LayoutTensor[dtype, M, K],
        rhs: LayoutTensor[dtype, N, K],
    ):
        pass


# matrix multiply and accumlate
struct MMA(TiledOp):
    @staticmethod
    fn op[
        dtype: DType,
        M: Int,
        N: Int,
        K: Int,
    ](
        inout dst: LayoutTensor[dtype, M, N],
        lhs: LayoutTensor[dtype, M, K],
        rhs: LayoutTensor[dtype, N, K],
    ):
        for m in range(M):
            for n in range(N):
                for k in range(K):
                    dst[m, n] += lhs[m, k] * rhs[n, k]


# matrix multiply and accumlate, vectorized and parallelized
struct MMA_VecPar(TiledOp):
    @staticmethod
    fn op[
        dtype: DType,
        M: Int,
        N: Int,
        K: Int,
    ](
        inout dst: LayoutTensor[dtype, M, N],
        lhs: LayoutTensor[dtype, M, K],
        rhs: LayoutTensor[dtype, N, K],
    ):
        alias width = simdwidthof[dtype]() * 2

        @parameter
        fn calc_row(m: Int):
            for n in range(N):

                @parameter
                fn dot[width: Int](k: Int):
                    dst.store[width](
                        m,
                        n,
                        dst.load[width](m, n)
                        + lhs[m, k] * rhs.load[width](n, k),
                    )

                vectorize[dot, width, size=K]()

        parallelize[calc_row](M, M)


fn gemm_l2_cache[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    mma: TiledOp,
    l1: Dim,
    l2: Dim,
](
    dst: LayoutTensor[dtype, M, N],
    lhs: LayoutTensor[dtype, M, K],
    rhs: LayoutTensor[dtype, K, N],
):
    # Dimensions of the Operation
    alias op_dim = Dim(M, N, K)

    # L1 and L2 Tiile ranges
    alias l1_size = op_dim.subrange(l1)
    alias l2_size = l1.subrange(l2)

    # Cache matrix to materialize L2 transposed tiles
    var l2_rhs_cache = LayoutTensor[dtype, l2.n, l2.k]()

    # First level of tiling (grid_blocks, L1 cache ..etc).
    for m_1 in range(l1_size.m):
        for n_1 in range(l1_size.n):
            var dst_l1_tile = dst.view[l1.m, l1.n](m_1, n_1)

            for k_1 in range(l1_size.k):
                var lhs_l1_tile = lhs.view[l1.m, l1.k](m_1, k_1)
                var rhs_l1_tile = rhs.view[l1.k, l1.n](k_1, n_1)

                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(l2_size.m):
                    for n_2 in range(l2_size.n):
                        var dst_l2_tile = dst_l1_tile.view[l2.m, l2.n](m_2, n_2)

                        for k_2 in range(l2_size.k):
                            var lhs_l2_tile = lhs_l1_tile.view[l2.m, l2.k](
                                m_2, k_2
                            )
                            var rhs_l2_tile = rhs_l1_tile.view[l2.k, l2.n](
                                k_2, n_2
                            )

                            # Materialize L2 rhs transposed tile
                            rhs_l2_tile.transpose().copyTo(l2_rhs_cache)

                            # Execute mma.op - rhs_l2_tile is already transposed
                            mma.op[
                                dtype,
                                l2.m,
                                l2.n,
                                l2.k,
                            ](dst_l2_tile, lhs_l2_tile, l2_rhs_cache)

    # TODO: Make tensor more ergonomic
    l2_rhs_cache.ptr.free()


fn gemm_l1_cache[
    dtype: DType,
    M: Int,
    N: Int,
    K: Int,
    mma: TiledOp,
    l1: Dim,
    l2: Dim,
](
    dst: LayoutTensor[dtype, M, N],
    lhs: LayoutTensor[dtype, M, K],
    rhs: LayoutTensor[dtype, K, N],
):
    # Dimensions of the Operation
    alias op_dim = Dim(M, N, K)

    # L1 and L2 Tiile ranges
    alias l1_size = op_dim.subrange(l1)
    alias l2_size = l1.subrange(l2)

    # Cache matrix to materialize L1 transposed tiles
    var l1_rhs_cache = LayoutTensor[dtype, l1.n, l1.k]()

    # First level of tiling (grid_blocks, L1 cache ..etc).
    for m_1 in range(l1_size.m):
        for n_1 in range(l1_size.n):
            var dst_l1_tile = dst.view[l1.m, l1.n](m_1, n_1)

            for k_1 in range(l1_size.k):
                var lhs_l1_tile = lhs.view[l1.m, l1.k](m_1, k_1)

                # Materialize L1 rhs transposed tile
                rhs.view[l1.k, l1.n](k_1, n_1).transpose().copyTo(l1_rhs_cache)

                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(l2_size.m):
                    for n_2 in range(l2_size.n):
                        var dst_l2_tile = dst_l1_tile.view[l2.m, l2.n](m_2, n_2)

                        for k_2 in range(l2_size.k):
                            var lhs_l2_tile = lhs_l1_tile.view[l2.m, l2.k](
                                m_2, k_2
                            )
                            # Transposed tile -> transposed indices
                            var rhs_l2_tile = l1_rhs_cache.view[l2.n, l2.k](
                                n_2, k_2
                            )

                            # Execute mma.op - rhs_l2_tile is already transposed
                            mma.op[
                                dtype,
                                l2.m,
                                l2.n,
                                l2.k,
                            ](dst_l2_tile, lhs_l2_tile, rhs_l2_tile)

    # TODO: Make tensor more ergonomic
    l1_rhs_cache.ptr.free()


fn test_tiled_matmul[use_l1_cache: Bool]():
    print("=== test_tiled_matmul")

    var dst = LayoutTensor[DType.float32, 8, 8]()
    var rhs = LayoutTensor[DType.float32, 8, 8]()
    var lhs = LayoutTensor[DType.float32, 8, 8]()

    dst.fill(0)
    rhs.linspace()
    lhs.linspace()

    if use_l1_cache:
        gemm_l1_cache[
            DType.float32,
            8,
            8,
            8,
            MMA_VecPar,
            Dim(4, 4, 2),
            Dim(2, 2, 1),
        ](dst, lhs, rhs)
    else:
        gemm_l2_cache[
            DType.float32,
            8,
            8,
            8,
            MMA_VecPar,
            Dim(4, 4, 2),
            Dim(2, 2, 1),
        ](dst, lhs, rhs)
    dst.print()

    dst.ptr.free()
    lhs.ptr.free()
    rhs.ptr.free()


fn main():
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
