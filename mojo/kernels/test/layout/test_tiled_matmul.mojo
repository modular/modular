# ===----------------------------------------------------------------------=== #
#
# This file is Modular Inc proprietary.
#
# ===----------------------------------------------------------------------=== #
# REQUIRES: disabled
# RUN: %mojo %s | FileCheck %s

from kernel_utils.layout_tensor import LayoutTensor, tile
from kernel_utils.layout import Layout, LayoutList, composition
from kernel_utils.int_tuple import IntTuple


fn create_matrix[M: Int, N: Int]() -> LayoutTensor[DType.float32]:
    let ptr = DTypePointer[DType.float32].alloc(M * N)
    return LayoutTensor[DType.float32](
        Layout(IntTuple(M, N), IntTuple(N, 1)), ptr
    )


fn fill_matrix(matrix: LayoutTensor[DType.float32]):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(matrix.dim(0)):
        for n in range(matrix.dim(1)):
            matrix[IntTuple(m, n)] = m * M + n


fn fill_matrix(matrix: LayoutTensor[DType.float32], val: Float32):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(matrix.dim(0)):
        for n in range(matrix.dim(1)):
            matrix[IntTuple(m, n)] = val


fn print_raw_major_matrix[dtype: DType](matrix: LayoutTensor[dtype]):
    let M = matrix.dim(0)
    let N = matrix.dim(1)
    for m in range(M):
        for n in range(N):
            print_no_newline(matrix[IntTuple(m, n)], "  ")
        print("")


# matrix multiple and accumlate
fn mma(
    inout dst: LayoutTensor[DType.float32],
    lhs: LayoutTensor[DType.float32],
    rhs: LayoutTensor[DType.float32],
):
    let M = dst.dim(0)
    let N = dst.dim(1)
    let K = lhs.dim(1)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                dst[IntTuple(m, n)] += lhs[IntTuple(m, k)] * rhs[IntTuple(k, n)]


# matrix multiple transposed and accumlate, returns matmul(lhs, rhs^T)
fn mmta(
    inout dst: LayoutTensor[DType.float32],
    lhs: LayoutTensor[DType.float32],
    rhs: LayoutTensor[DType.float32],
):
    let M = dst.dim(0)
    let N = dst.dim(1)
    let K = lhs.dim(1)
    for m in range(M):
        for n in range(N):
            for k in range(K):
                dst[IntTuple(m, n)] += lhs[IntTuple(m, k)] * rhs[IntTuple(n, k)]


fn tiled_matmul(
    dst: LayoutTensor[DType.float32],
    lhs: LayoutTensor[DType.float32],
    rhs: LayoutTensor[DType.float32],
    transposed_mmta: Bool,
    tile_1_m_size: Int = 4,
    tile_1_n_size: Int = 4,
    tile_1_k_size: Int = 4,
    tile_2_m_size: Int = 2,
    tile_2_n_size: Int = 2,
    tile_2_k_size: Int = 2,
):
    let m = dst.dim(0)
    let n = dst.dim(1)
    let k = lhs.dim(1)

    # First level of tiling (grid_blocks, L1 cache ..etc).
    for m_1 in range(m // tile_1_m_size):
        for n_1 in range(n // tile_1_n_size):
            var dst_l1_tile = tile(
                dst,
                LayoutList(
                    Layout(tile_1_m_size, 1),
                    Layout(tile_1_n_size, 1),
                ),
                IntTuple(m_1, n_1),
            )
            for k_1 in range(k // tile_1_k_size):
                let lhs_l1_tile = tile(
                    lhs,
                    LayoutList(
                        Layout(tile_1_m_size, 1),
                        Layout(tile_1_k_size, 1),
                    ),
                    IntTuple(m_1, k_1),
                )
                let rhs_l1_tile = tile(
                    rhs,
                    LayoutList(
                        Layout(tile_1_k_size, 1),
                        Layout(tile_1_n_size, 1),
                    ),
                    IntTuple(k_1, n_1),
                )
                # Second level of tiling (instruction, vectorization..etc)
                for m_2 in range(tile_1_m_size // tile_2_m_size):
                    for n_2 in range(tile_1_n_size // tile_2_n_size):
                        var dst_l2_tile = tile(
                            dst_l1_tile,
                            LayoutList(
                                Layout(tile_2_m_size, 1),
                                Layout(tile_2_n_size, 1),
                            ),
                            IntTuple(m_2, n_2),
                        )

                        for k_2 in range(tile_1_k_size // tile_2_k_size):
                            let lhs_l2_tile = tile(
                                lhs_l1_tile,
                                LayoutList(
                                    Layout(tile_2_m_size, 1),
                                    Layout(tile_2_k_size, 1),
                                ),
                                IntTuple(m_2, k_2),
                            )
                            let rhs_l2_tile = tile(
                                rhs_l1_tile,
                                LayoutList(
                                    Layout(tile_2_k_size, 1),
                                    Layout(tile_2_n_size, 1),
                                ),
                                IntTuple(k_2, n_2),
                            )
                            if transposed_mmta:
                                # Apply transpose composition to access transpose of r.h.s
                                let rhs_l2_tile_transposed = composition(
                                    rhs_l2_tile.layout,
                                    Layout(IntTuple(2, 2), IntTuple(2, 1)),
                                )
                                let rhs_l2_tile_t = LayoutTensor(
                                    rhs_l2_tile_transposed, rhs_l2_tile.ptr
                                )
                                mmta(dst_l2_tile, lhs_l2_tile, rhs_l2_tile_t)
                            else:
                                mma(dst_l2_tile, lhs_l2_tile, rhs_l2_tile)


fn test_tiled_matmul():
    print("=== test_tiled_matmul")
    let dst = create_matrix[8, 8]()
    let rhs = create_matrix[8, 8]()
    let lhs = create_matrix[8, 8]()
    fill_matrix(dst, 0)
    fill_matrix(rhs)
    fill_matrix(lhs)
    tiled_matmul(dst, lhs, rhs, False)
    print_raw_major_matrix(dst)
    dst.ptr.free()
    lhs.ptr.free()
    rhs.ptr.free()


fn test_tiled_matmul_transpose():
    print("=== test_tiled_matmul_transpose")
    let dst = create_matrix[8, 8]()
    let rhs = create_matrix[8, 8]()
    let lhs = create_matrix[8, 8]()
    fill_matrix(dst, 0)
    fill_matrix(rhs)
    fill_matrix(lhs)
    tiled_matmul(dst, lhs, rhs, True)
    print_raw_major_matrix(dst)
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
    test_tiled_matmul_transpose()
