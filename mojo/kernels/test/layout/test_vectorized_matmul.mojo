# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# RUN: %mojo-no-debug %s -t | FileCheck %s

from layout import IntTuple, Layout, LayoutTensor
from layout.fillers import arange


fn outer_product[
    TM: Int, TN: Int
](
    lhs: SIMD[DType.float32, TM],
    rhs: SIMD[DType.float32, TN],
) -> SIMD[
    DType.float32, TM * TN
]:
    var res = SIMD[DType.float32, TM * TN]()
    # Note: Outputs are columns.
    for i in range(TM):
        for j in range(TN):
            res[i * TN + j] = lhs[i] * rhs[j]
    return res


fn test_tiled_and_vectorized_matmul():
    print("== test_tiled_and_vectorized_matmul")
    alias M = 8
    alias N = 8
    alias K = 8

    alias BM = 4
    alias BN = 4
    alias BK = 4

    alias TM = 2
    alias TN = 2

    var tensor_a = LayoutTensor[
        DType.float32, Layout(IntTuple(M, K), IntTuple(K, 1))
    ].stack_allocation()
    var tensor_b = LayoutTensor[
        DType.float32, Layout(IntTuple(K, N), IntTuple(N, 1))
    ].stack_allocation()
    var tensor_c = LayoutTensor[
        DType.float32, Layout(IntTuple(M, N), IntTuple(N, 1))
    ].stack_allocation()
    arange(tensor_a)
    arange(tensor_b)
    _ = tensor_c.fill(0)

    for bm in range(M // BK):
        for bn in range(N // BN):
            var tile_c = tensor_c.tile[BM, BN](bm, bn)
            for bk in range(K // BK):
                var tile_a = tensor_a.tile[BM, BK](bm, bk)
                var tile_b = tensor_b.tile[BK, BN](bk, bn)

                var vec_c = tile_c.vectorize[TM, TN]()
                var vec_a = tile_a.vectorize[TM, 1]()
                var vec_b = tile_b.vectorize[1, TN]()

                for m_i in range(vec_c.shape[0]()):
                    for n_i in range(vec_c.shape[1]()):
                        for k_i in range(BK):
                            vec_c[m_i, n_i] += rebind[vec_c.element_type](
                                outer_product(vec_a[m_i, k_i], vec_b[k_i, n_i])
                            )

    print(tensor_c)


fn main():
    # CHECK-LABEL: test_tiled_and_vectorized_matmul
    # CHECK: 1120.0   1148.0   1176.0   1204.0   1232.0   1260.0   1288.0   1316.0
    # CHECK: 2912.0   3004.0   3096.0   3188.0   3280.0   3372.0   3464.0   3556.0
    # CHECK: 4704.0   4860.0   5016.0   5172.0   5328.0   5484.0   5640.0   5796.0
    # CHECK: 6496.0   6716.0   6936.0   7156.0   7376.0   7596.0   7816.0   8036.0
    # CHECK: 8288.0   8572.0   8856.0   9140.0   9424.0   9708.0   9992.0   10276.0
    # CHECK: 10080.0   10428.0   10776.0   11124.0   11472.0   11820.0   12168.0   12516.0
    # CHECK: 11872.0   12284.0   12696.0   13108.0   13520.0   13932.0   14344.0   14756.0
    # CHECK: 13664.0   14140.0   14616.0   15092.0   15568.0   16044.0   16520.0   16996.0
    test_tiled_and_vectorized_matmul()
