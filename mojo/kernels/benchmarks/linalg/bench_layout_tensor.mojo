# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #

from random import rand

import benchmark
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from algorithm import parallelize, sync_parallelize, vectorize
from memory import memset_zero
from python import Python

from kernel_utils.layout import Layout
from kernel_utils.layout_tensor import LayoutTensor
from kernel_utils.int_tuple import IntTuple


alias M = 512  # rows of A and C
alias N = 4096  # cols of B and C
alias K = 512  # cols of A and rows of B

alias dtype = DType.float32


struct Matrix[rows: Int, cols: Int]:
    var data: DTypePointer[dtype]

    # Initialize zeroeing all values
    fn __init__(inout self):
        self.data = DTypePointer[dtype].alloc(rows * cols)
        memset_zero(self.data, rows * cols)

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, data: DTypePointer[dtype]):
        self.data = data

    ## Initialize with random values
    @staticmethod
    fn rand() -> Self:
        var data = DTypePointer[dtype].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(data)

    fn __getitem__(self, y: Int, x: Int) -> SIMD[dtype, 1]:
        return self.load[1](y, x)

    fn __setitem__(inout self, y: Int, x: Int, val: SIMD[dtype, 1]):
        self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[dtype, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[dtype, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


fn matmul_naive(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Perform 2D tiling on the iteration space defined by end_x and end_y
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Unroll the vectorized loop by a constant factor
fn matmul_unrolled(inout C: Matrix, A: Matrix, B: Matrix):
    # simdwidth of = amount of `dtype` elements that fit into a single SIMD register
    # 2x multiplier will use multiple SIMD registers in parallel where possible
    alias nelts = simdwidthof[dtype]() * 2
    alias tile_n = 64  # N must be a multiple of this
    alias tile_k = 4  # K must be a multiple of this

    constrained[N % tile_n == 0, "N must be a multiple of tile_n"]()
    constrained[K % tile_k == 0, "K must be a multiple of tile_k"]()

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            @unroll(tile_y)
            for k in range(y, y + tile_y):
                var A_m_k_val = A[m, k]

                @parameter
                fn dot[nelts: Int](n: Int):
                    C.store(
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A_m_k_val * B.load[nelts](k, n + x),
                    )

                alias unroll_factor = tile_x // nelts
                vectorize[dot, nelts, tile_x, unroll_factor]()

        tile[calc_tile, tile_n, tile_k](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


struct TensorBuilder[
    M: Int,
    N: Int,
    dtype: DType,
    layout: Layout = Layout(IntTuple(M, N), IntTuple(N, 1)),
]:
    alias Type = LayoutTensor[layout, dtype]

    @staticmethod
    fn Wrap(ptr: DTypePointer[dtype]) -> Self.Type:
        return Self.Type(ptr)


fn matmul_tiled_l(inout C: Matrix, A: Matrix, B: Matrix):
    var dst = TensorBuilder[M, N, dtype].Wrap(C.data)
    var lhs = TensorBuilder[M, K, dtype].Wrap(A.data)
    var rhs = TensorBuilder[K, N, dtype].Wrap(B.data)

    alias block_m = 2
    alias block_n = 64
    alias block_k = 4

    constrained[M % block_m == 0, "N must be a multiple of block_m"]()
    constrained[N % block_n == 0, "N must be a multiple of block_n"]()
    constrained[K % block_k == 0, "K must be a multiple of block_k"]()

    alias vec_size = simdwidthof[dtype]() * 2

    @parameter
    fn calc_row(m_1: Int):
        for k_1 in range(K // block_k):
            for n_1 in range(N // block_n):
                var lhs_view = lhs.tile[block_m, block_k](m_1, k_1)
                var dst_view = dst.tile[block_m, block_n](m_1, n_1)
                var rhs_view = rhs.tile[block_k, block_n](k_1, n_1)

                @unroll
                for m_2 in range(block_m):

                    @unroll
                    for k_2 in range(block_k):
                        var lhs_val = lhs_view[m_2, k_2]

                        @parameter
                        fn dot[size: Int](n: Int):
                            dst_view.store[size](
                                m_2,
                                n,
                                dst_view.load[size](m_2, n)
                                + lhs_val * rhs_view.load[size](k_2, n),
                            )

                        alias unroll_factor = block_n // vec_size
                        vectorize[dot, vec_size, block_n, unroll_factor]()

    sync_parallelize[calc_row](M // block_m)


@always_inline
fn bench[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
]() raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    var secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()

    A.data.free()
    B.data.free()
    C.data.free()

    var gflops = ((2 * M * N * K) / secs) / 1e9

    var py = Python.import_module("builtins")
    _ = py.print(py.str("{:<13}{:>8.3f} GFLOPS").format(name, gflops))


@always_inline
fn test_matrix_equal[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](inout C: Matrix, A: Matrix, B: Matrix) raises -> Bool:
    """Runs a matmul function on A and B and tests the result for equality with
    C on every element.
    """
    var result = Matrix[M, N]()
    _ = func(result, A, B)
    for i in range(C.rows):
        for j in range(C.cols):
            if C[i, j] != result[i, j]:
                return False
    return True


fn test_all() raises:
    var A = Matrix[M, K].rand()
    var B = Matrix[K, N].rand()
    var C = Matrix[M, N]()

    matmul_naive(C, A, B)

    if not test_matrix_equal[matmul_tiled_l](C, A, B):
        raise Error(
            "Layout Tiled Parallel Vectorized output does not match naive"
            " implementation"
        )
    if not test_matrix_equal[matmul_unrolled](C, A, B):
        raise Error("Unroll output does not match naive implementation")

    A.data.free()
    B.data.free()
    C.data.free()


fn main() raises:
    test_all()
    print("CPU Results\n")

    bench[matmul_naive, "Naive:"]()
    bench[matmul_unrolled, "Unrolled:"]()
    bench[matmul_tiled_l, "LayoutTensor:"]()
